"""What a tool call is, for the approval gate: never allowed, harmless, or in need of review.

`assess` turns a call into a `Review`. The regexes and `shlex` checks here are hints and
seatbelts, not a sandbox: the floor catches accidents and naive injection, and the protected-path
scan catches the obvious spellings of secret and self-modifying actions. The real gate is review
by the judge or the owner, and reviewing is the default. An odd spelling can only move an action
from "owner" to "judge", never to "run unreviewed": that needs plain text that `shlex` fully
understands, made only of read-only programs.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from arti import config


@dataclass
class Review:
    text: str  # canonical JSON of tool, args and cwd: what the owner/judge sees and grants hash
    flags: list[str] = field(default_factory=list)  # why it needs review; empty = run
    owner_only: bool = False  # the judge may not decide; grants cover only this exact action
    keys: set[str] = field(default_factory=set)  # what grants cover ("shell:ls", "gui:<tool>")
    floor: str | None = None  # why it is never allowed


# --- protected locations -----------------------------------------------------------------------

# Paths below home whose contents are secret (keys, tokens, shell history) or run at login
# (shell and X session start-up files). Reading or writing them needs the owner. All of
# ~/.config, with no exceptions: it holds tokens (gh, gcloud), browser profiles, autostart
# entries and systemd user units, and which other parts are harmless is not known. ~/snap and
# ~/.var/app hold each snap's and Flatpak's own data (a snap Firefox keeps its profile there).
_HOME_SECRETS = """
    .ssh .gnupg .aws .azure .config .netrc .git-credentials .kube .docker/config.json .npmrc
    .pypirc .codex .zsh_history .bash_history .python_history
    .node_repl_history .zshrc .zshenv .zprofile .zlogin .bashrc .bash_profile .bash_login
    .bash_logout .profile .login .xprofile .xinitrc .xsession .xsessionrc .pam_environment
    .claude .claude.json .pgpass .my.cnf .cargo/credentials .cargo/credentials.toml
    .gem/credentials .vault-token .terraform.d .password-store .local/share/keyrings
    .local/share/kwalletd .mozilla snap .var/app
""".split()
# Below Arti's data directory: its secrets, settings, grants, logs and logins.
# A secret file name anywhere (compared case-folded):
_SECRET_FILE = re.compile(
    # .env and .env.local (not .env.example, .env.sample or .env.template); direnv's .envrc
    r"\.env(?!\.(?:example|sample|template)$)(?:\..*)?|\.envrc"
    # auth.json, credentials and .credentials.json
    r"|auth\.json|\.?credentials(?:\.\w+)?"
    # id_rsa and id_ed25519.pub
    r"|id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?"
    # any name with "secret" in it: secrets.json, client_secret.json, .secret
    r"|.*secret.*"
    # cloud service account keys: my-service-account.json, service_account-prod.json
    r"|.*service[-_]?account.*\.json"
    # private keys and certificates: server.pem, tls.key, cert.p12, identity.pfx
    r"|.*\.(?:pem|key|p12|pfx)"
)
# Writing here needs the owner: bin directories, and user units that start programs (systemd
# timers and services, D-Bus activation).
_OWNER_WRITE = """
    bin .local/bin /usr/local/bin /home/linuxbrew/.linuxbrew/bin
    .local/share/systemd .local/share/dbus-1
""".split()
# The data dir is Arti's own state (logins, the database, logs, what plugins keep), so it fails
# closed: everything in it needs the owner, except the folders tools work in and SOUL.md. A file a
# plugin adds there is protected without the plugin saying so, even when it is disabled; a plugin
# changes its own files through its own tools.
_DATA_READABLE = ("cache", "artifacts", "media", "soul.md")  # (case-folded)
_CONTEXT_READABLE = ("cache", "artifacts", "media")  # in a group's contexts/<scope>/
_DATA_ORDINARY = ("cache", "artifacts")  # writing here needs no review


def _fold(path: Path) -> str:
    """A path as compared here: resolved (symlinks followed) and case-folded, so a folder
    mounted from a case-insensitive disk cannot dodge a check by spelling."""
    return str(path.resolve()).casefold()


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


@functools.lru_cache(maxsize=4)
def _roots(home: str, data: str, code: str) -> dict:
    """The protected roots for this home, data dir and code dir, resolved once."""
    home_p, data_p = Path(home), Path(data)
    scratch = ("/tmp", tempfile.gettempdir())
    return {
        "secret": tuple(_fold(home_p / n) for n in _HOME_SECRETS),
        "owner": tuple(_fold(home_p / n) for n in _OWNER_WRITE) + (_fold(Path(code)),),
        "ordinary": tuple(_fold(Path(n)) for n in scratch)
        + tuple(_fold(data_p / n) for n in _DATA_ORDINARY),
        "data": _fold(data_p),
    }


def _current_roots() -> dict:
    return _roots(str(Path.home()), str(config.data_dir()), str(config.code_dir()))


class PathError(ValueError):
    """A path that cannot be read for review: `~name` of a user this computer does not have."""


def _expand(path: str | Path) -> Path:
    """Path(path).expanduser(), with a readable PathError instead of Python's bare
    RuntimeError for an unknown `~name` (what it would mean is not known, so it is reviewed)."""
    try:
        return Path(path).expanduser()
    except RuntimeError:
        user = str(path).split("/", 1)[0]
        raise PathError(f"{user} names the home folder of no user on this computer") from None


def _resolve(path: str | Path, cwd: Path | None = None) -> str:
    """The folded real path a word names, read from `cwd` when relative. PathError for an
    unknown `~name`."""
    text = str(path)
    for spelling in ("$HOME", "${HOME}"):
        if text == spelling or text.startswith(spelling + "/"):
            text = str(Path.home()) + text[len(spelling) :]
    p = _expand(text)
    return _fold(p if p.is_absolute() else (cwd or Path.cwd()) / p)


def _data_part(folded: str, data: str) -> list[str] | None:
    """`folded`'s parts below the data dir, or None outside it."""
    return folded[len(data) :].strip("/").split("/") if _under(folded, data) else None


def _private_data(folded: str, data: str) -> bool:
    """Whether reading `folded` needs the owner because it is Arti's own state (see
    _DATA_READABLE): anything in the data dir but the folders tools may read."""
    parts = _data_part(folded, data)
    if not parts or parts == [""]:
        return False  # outside, or the data dir itself (a listing)
    if parts[0] == "contexts":  # a group's folder: open only where its files are
        return len(parts) > 2 and parts[2] not in _CONTEXT_READABLE
    return parts[0] not in _DATA_READABLE


def _secret(folded: str) -> bool:
    roots = _current_roots()
    if any(_under(folded, root) for root in roots["secret"]):
        return True
    if _private_data(folded, roots["data"]):
        return True
    return bool(_SECRET_FILE.fullmatch(folded.rsplit("/", 1)[-1]))


def _owner_write(folded: str) -> bool:
    """Writing here needs the owner: secrets (shell start-up files and ~/.config among them),
    Arti's code and its data dir outside the folders tools work in (what a plugin keeps there
    changes only through its tools), _OWNER_WRITE (bin directories, user units) and git hooks or
    config."""
    roots = _current_roots()
    if _secret(folded) or any(_under(folded, root) for root in roots["owner"]):
        return True
    parts = _data_part(folded, roots["data"])
    if parts == [""]:
        return True  # the data dir itself: moving into it, or a command over all of it
    if parts:
        if parts[0] == "contexts":
            return len(parts) > 2 and parts[2] not in _DATA_ORDINARY
        return parts[0] not in _DATA_ORDINARY
    return "/.git/hooks/" in folded + "/" or folded.endswith("/.git/config")


def is_secret_path(path: str | Path) -> bool:
    """True when reading `path` needs the owner (credentials, Arti's config, logs, profiles)."""
    return _secret(_resolve(path))


# --- reading shell text ------------------------------------------------------------------------


def _segments(text: str) -> list[list[str]] | None:
    """The words of each command in `text`, split at | || && ; & and newlines; None when shlex
    cannot read it (an unclosed quote). `#` is a plain character here, so a comment can hide
    nothing: bash only starts one at the start of a word, and seeing too much is the safe side."""
    lexer = shlex.shlex(text.replace("\n", ";"), posix=True, punctuation_chars=";&|")
    lexer.whitespace_split, lexer.commenters = True, ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    segments, current = [], []
    for token in tokens:
        if token and set(token) <= set(";&|"):
            segments.append(current)
            current = []
        else:
            current.append(token)
    return segments + [current]


def _short_flags(args: list[str]) -> str:
    """The single-dash options run together, to look for a letter: `-r -fL` gives "-r-fL"."""
    return "".join(a for a in args if a.startswith("-") and not a.startswith("--"))


def _cd(dirs: set[Path], args: list[str]) -> set[Path]:
    """The dirs a later command may run in after `cd`: a failed cd (`;` goes on) stays put,
    and so does `cd ~nobody` (an unknown user's home), which always fails."""
    try:
        return dirs | {Path(_resolve(args[-1] if args else "~", d)) for d in dirs}
    except PathError:
        return dirs


# --- the floor ---------------------------------------------------------------------------------

_SYSTEM_DIRS = """
    etc usr var bin sbin opt boot lib lib32 lib64 libx32 root home srv snap proc sys dev run mnt
    media
""".split()
_SYSTEM = "(?:" + "|".join(_SYSTEM_DIRS) + ")"
# A whole-tree target as one word, optionally quoted or with /*: `/`, `~`, `"$HOME"/`, `${HOME}`,
# `/usr/*`, `/home/alex`. It must end the word (`/tmp/x` is not one).
_ROOT_WORD = (
    rf"""["']?(?:/|~/?|\$HOME/?|\$\{{HOME\}}/?|/{_SYSTEM}/?|/home/[^/\s"']+/?)\*?["']?"""
    r"""(?=$|[\s;&|)"'])"""
)
# Catastrophic commands, refused in every mode. Each pattern reads the raw text, so a command
# that shlex cannot parse is still caught; floor_reason below also checks parsed words. The
# table of what each rule must catch and must let through is tests/plugins/safety/test_risk.py.
_FLOOR = [
    # rm -rf ~, rm -v -fr "/usr", rm --recursive /
    (rf"\brm\s+(?:[^;&|\n]*\s)?-{{1,2}}[\w-]*[rR][^;&|\n]*?\s{_ROOT_WORD}", "delete everything"),
    # chmod -R 777 /, chown -R me ~
    (rf"\bch(?:mod|own)\s+(?:[^;&|\n]*\s)?-\w*R[^;&|\n]*?\s{_ROOT_WORD}", "change a whole tree"),
    # mkfs.ext4 /dev/sdb, wipefs -a /dev/sda, blkdiscard /dev/nvme0n1
    (r"\bmkfs\b|\bwipefs\b|\bblkdiscard\b", "format a disk"),
    # dd if=img of=/dev/sda, cat x > /dev/vda, > /dev/nvme0n1, shred /dev/sdb
    (
        r"\bdd\b[^;&|\n]*\bof=/dev/|\bshred\b[^;&|\n]*\s/dev/"
        r"|>\s*/dev/(?:[shv]d[a-z]|xvd[a-z]|nvme\d|mmcblk\d|md\d|dm-\d|loop\d|mapper/)",
        "write onto a raw disk",
    ),
    # :(){ :|:& };:  and the same with a named function, bomb(){ bomb|bomb& }
    (r"(\w+|:)\s*\(\)\s*\{[^}]*?(?:\1)\s*\|\s*(?:\1)\s*&", "fork bomb"),
    # kill -9 -1, kill -1
    (r"\bkill\s+(?:-\w+\s+)*-1\s*(?:$|[;&|])", "kill every process"),
    # shutdown -h now, sudo reboot, ls; /sbin/halt, init 0, telinit 6 (as a command, not inside a
    # word or argument)
    (
        r"(?:^|[;&|(`]|\b(?:sudo|doas|pkexec|run0|exec))\s*(?:/(?:usr/)?s?bin/)?"
        r"(?:(?:shutdown|reboot|halt|poweroff)\b|(?:tel)?init\s+[016sS](?=$|[\s;&|)]))",
        "shut down the computer",
    ),
    # systemctl poweroff, systemctl -i suspend, systemctl isolate rescue.target (no network)
    (
        r"\bsystemctl\s+(?:[^;&|\n]*\s)?(?:(?:reboot|poweroff|halt|kexec|soft-reboot|suspend"
        r"|hibernate|hybrid-sleep|suspend-then-hibernate|sleep|rescue|emergency)(?=$|[\s;&|)])"
        r"|(?:start|isolate)\s+(?:reboot|poweroff|halt|kexec|rescue|emergency)\.target)",
        "shut down the computer",
    ),
    # loginctl poweroff, loginctl terminate-user alex (ends every process of the user, Arti too)
    (
        r"\bloginctl\s+(?:[^;&|\n]*\s)?(?:reboot|poweroff|halt|suspend|hibernate|hybrid-sleep"
        r"|kexec|terminate-user|terminate-session|kill-user|kill-session)\b",
        "shut down the computer",
    ),
]
_FLOOR = [(re.compile(pattern), reason) for pattern, reason in _FLOOR]
_FIND_FILTERS = {"-name", "-iname", "-path", "-ipath", "-regex", "-iregex"}


def _whole_tree(word: str, cwd: Path) -> bool:
    """Whether a path word names /, a home directory or a top-level system directory.

    A glob or brace stands for its directory: `*` read in ~ is all of ~, so is `~/{.,}*`. The
    shell leaves `~nobody` (an unknown user) as it is: a name in cwd, not a whole tree."""
    try:
        folded = _resolve(re.split(r"[*?\[{]", word, maxsplit=1)[0] or ".", cwd)
    except PathError:
        return False
    tops = {_fold(Path("/" + name)) for name in _SYSTEM_DIRS} | {"/", _fold(Path.home())}
    return folded in tops or os.path.dirname(folded) == _fold(Path("/home"))


def floor_reason(text: str, cwd: Path) -> str | None:
    """Why `text` is never allowed (a catastrophic command), or None."""
    for pattern, reason in _FLOOR:
        if pattern.search(text):
            return reason
    dirs = {cwd}
    for words in _segments(text) or []:
        program, args = os.path.basename(words[0]) if words else "", words[1:]
        if program == "cd":
            dirs = _cd(dirs, args)
            continue
        flags = _short_flags(args)
        operands = [a for a in args if not a.startswith("-")]
        if program == "mv":
            operands = operands[:-1]  # moving something INTO home is fine
        elif program == "find":
            # find ROOTS -expression: only an unfiltered delete takes the whole tree.
            if _FIND_FILTERS & set(args) or not {"-delete", "rm"} & set(args):
                continue
            first_test = next((i for i, a in enumerate(args) if a[:1] in "-(!"), len(args))
            operands = args[:first_test] or ["."]
        elif not (
            (program == "rm" and ("r" in flags.lower() or "--recursive" in args))
            or (program in ("chmod", "chown") and "R" in flags)
        ):
            continue
        if any(_whole_tree(w, d) for w in operands for d in dirs):
            return "delete or change a whole tree"
    return None


# --- plain read-only shell ---------------------------------------------------------------------

# Shell syntax that can hide what runs or what is read: expansions ($HOME, `cmd`, $(cmd)),
# escapes, history (!), globs, braces, subshells, redirects and comments.
_META = set("$`\\!*?[]{}()<>#\n")
# The two harmless redirects ` 2>/dev/null` and ` 2>&1`, removed before the check above.
_STDERR = re.compile(r"\s2>(?:/dev/null|&1)(?=\s|$|[|;&])")
_READ_ONLY = set(
    """
    cat head tail wc ls pwd echo which stat du df basename dirname realpath readlink true uname
    whoami id grep find sort uniq jq diff cmp cd date hostname
    """.split()
)
_FIND_ACTIONS = set("-exec -execdir -ok -okdir -delete -fprint -fprint0 -fprintf -fls".split())
# The short option that makes a walk follow symlinks, which could lead into a secret location
# that is below none of the roots checked here. (diff -r follows them always.)
_FOLLOWS = {"grep": "S", "find": "L", "du": "L", "ls": "L"}


def _args_ok(program: str, args: list[str]) -> bool:
    """The one-line argument check of each read-only program: no option that writes or runs."""
    options = [a for a in args if a.startswith("-")]
    short = _short_flags(args)
    if any("-from" in a for a in options):  # --files0-from=LIST: reads paths not named here
        return False
    follows = program in _FOLLOWS and _FOLLOWS[program] in short
    if follows or "-follow" in args or "--deref" in "".join(args):
        return False
    if program == "grep" and _recursive_roots(program, args):
        return False  # it would read every file below, secret-named ones included
    if program == "diff":
        return "r" not in short and not any("recurs" in a for a in args)
    if program == "find":
        return not _FIND_ACTIONS & set(args)
    if program == "sort":
        return "o" not in short and not any(a.startswith(("--o", "--compress")) for a in options)
    if program == "uniq":
        return len(args) - len(options) <= 1
    if program == "date":
        return all(a.startswith("+") or a == "-u" for a in args)
    if program == "hostname":
        return not args
    if program == "cd":
        return len(args) <= 1 and args != ["-"]
    return True


def _recursive_roots(program: str, args: list[str]) -> list[str]:
    """The directories a recursive reader walks (none for other programs)."""
    short = _short_flags(args)
    # "recurs" also catches --recursive and grep's `-d recurse` / --directories=recurse.
    named = any("recurs" in a for a in args)
    recursive = (
        program in ("find", "du")
        or (program == "grep" and ("r" in short.lower() or named))
        or (program == "ls" and ("R" in short or named))
    )
    if not recursive:
        return []
    operands = [a for a in args if not a.startswith("-")]
    if program == "grep" and not {"-e", "--regexp"} & set(args):
        operands = operands[1:]  # the first one is the pattern
    return operands or ["."]


def _word_paths(word: str) -> list[str]:
    """The path-like parts of a word: itself, an option's value (--file=X, -fX)."""
    if not word.startswith("-"):
        return [word]
    if "=" in word:
        value = word.split("=", 1)[1]
        return [value] if value and not value.startswith("-") else []
    if word.startswith("--"):
        return []
    # A cluster of short options (-rfFILE): any letter may be the one that takes the rest.
    return [word[i:] for i in range(2, len(word))]


def plain_read_only(command: str, cwd: Path) -> bool:
    """True when `command` is plain text made only of read-only programs reading no secret."""
    command = _STDERR.sub("", command)
    if _META & set(command) or "&" in command.replace("&&", ""):
        return False
    if any(unicodedata.category(c)[0] in "CZ" and c != " " for c in command):
        return False
    segments = _segments(command)
    if not segments or not all(segments):
        return False
    dirs, secret_roots, data = {cwd}, _current_roots()["secret"], _current_roots()["data"]
    for words in segments:
        program, args = words[0], words[1:]
        if program not in _READ_ONLY or not _args_ok(program, args):
            return False
        if any(_secret(_fold(d)) for d in dirs):
            return False
        paths = [path for word in args for path in _word_paths(word)]
        if any(_secret(_resolve(path, d)) for path in paths for d in dirs):
            return False
        # A recursive reader must not walk into a secret location below where it starts.
        for start in _recursive_roots(program, args):
            folded = {_resolve(start, d) for d in dirs}
            if any(_under(root, f) for f in folded for root in secret_roots):
                return False
            if any(_under(data, f) for f in folded):  # it would walk into Arti's own state
                return False
        if program == "cd":
            dirs = _cd(dirs, args)
    return True


# --- protected mentions in free text -----------------------------------------------------------

# Where the raw-text scan splits words: whitespace, quotes and shell punctuation, so
# `cat<~/.ssh/id_rsa` and `x="$HOME/.aws"` still show their paths.
_SPLIT = re.compile(r"""[\s'"=:;|&<>(),`]+""")
# Mentions that need the owner wherever they appear: `.env`, `id_rsa`, reading the keyring
# (`secret-tool lookup`, `kwallet-query`), `gpg --export-secret-keys`, `keychain`, `crontab`,
# Arti's grants.json, config.toml, state.db, and `systemctl --user enable x` (it starts x at
# every login without naming a path).
_HINTS = re.compile(
    r"(?i)\.env\b|\bid_(?:rsa|ed25519|ecdsa|dsa)\b|\bsecret-tool\b|\bkwallet-query\b"
    r"|--export-secret|keychain|\bcrontab\b|\bgrants\.json\b|\bconfig\.toml\b|\bstate\.db\b"
    r"|\bsystemctl\s+(?:[^;&|\n]*\s)?(?:enable|link|edit|preset|add-wants|add-requires)\b"
)
# Running as administrator: `sudo ls`, `x; su root`, `doas`, polkit's `pkexec` and systemd's
# `run0`.
_ADMIN = re.compile(r"(?i)(?:^|[\s;&|(`])(?:sudo|su|doas|pkexec|run0)(?=\s|$)")


def _protected_mention(text: str, cwd: Path) -> bool:
    """Whether a shell command seems to touch a secret or owner-only path.

    Words are read both as raw text and as shlex reads them (so `~/.ss''h` is seen too), each
    from every directory a `cd` before it may have led to."""
    if _HINTS.search(text):
        return True
    dirs = {cwd}
    for words in (_segments(text) or []) + [_SPLIT.split(text)]:
        paths = [path for word in words for path in _word_paths(word) if path]
        if any(_owner_write(_resolve(path, d)) for path in paths for d in dirs):
            return True
        if words and words[0] == "cd":
            dirs = _cd(dirs, words[1:])
    return False


# --- assess ------------------------------------------------------------------------------------

# Words after which the next word is not the program that runs, programs that run any program
# they are given, and interpreters that run any script: a category grant named after them
# ("shell:bash", "shell:python3") would cover anything, so their grants are exact only.
_SHELL_WORDS = set(
    """
    if then else elif fi case esac for select while until do done function time ! [[ { } coproc
    """.split()
)
_RUNNERS = set(
    """
    exec command builtin eval source . env xargs nohup nice ionice chrt taskset flock timeout
    sudo doas pkexec run0 runuser setpriv chroot unshare nsenter strace
    sh bash zsh dash ksh fish csh tcsh swift awk gawk mawk nawk tclsh wish
    npx uvx pipx bunx deno bun uv watch script parallel stdbuf unbuffer
    bwrap firejail xdg-open gio systemctl systemd-run systemd-inhibit setsid at batch
    start-stop-daemon
    """.split()
)
_NO_KEY = _SHELL_WORDS | _RUNNERS
# Interpreters with a version in the name: python, python3, python3.11, pypy3, node22, perl5.
_INTERPRETER = re.compile(r"(?:python|pypy|node|nodejs|perl|ruby|php|lua|luajit|Rscript)[\d.]*")
# Task runners: they run whatever the project's files (Makefile, package.json, tox.ini, ...)
# say, so a grant for one task is no grant for the next. npm, pnpm and yarn run package
# scripts on almost every subcommand (`yarn build`, `npm install`'s lifecycle scripts).
_TASK_RUNNERS = set("make gmake just rake tox nox gradle mvn ant npm pnpm yarn pytest".split())
# Tools that run project code or a given command only through some subcommands (`cargo run`).
_RUN_TOOLS = set("cargo go poetry pipenv pdm hatch bundle composer dotnet".split())
_RUN_SUBCOMMANDS = set("run exec x test bench build generate run-script".split())
# Options that make a program run another command (`find -exec`, `git rebase --exec`, `rg --pre`).
_EXEC_OPTIONS = set("-exec -execdir -ok -okdir --exec --exec-batch --pre".split())
# git words that run a command or define one for later (`git -c alias.x='!cmd' x`,
# `git config alias.x ...`, `git bisect run`, `git submodule foreach`, `git rebase -x`).
_GIT_RUNS = set("-c config run foreach filter-branch difftool mergetool -x --exec".split())
_GIT_RUNS_PREFIXES = ("--config-env", "--exec=", "--upload-pack", "--receive-pack", "--extcmd")


def _no_key(name: str) -> bool:
    return name in _NO_KEY or bool(_INTERPRETER.fullmatch(name))


def _runs_other_code(program: str, args: list[str]) -> bool:
    """Whether a program's arguments make it run something besides itself: a command or script
    named in them, or a task the project defines."""
    if program in _TASK_RUNNERS or _EXEC_OPTIONS & set(args):
        return True
    subcommand = next((a for a in args if not a.startswith("-")), "")
    if program in _RUN_TOOLS and subcommand in _RUN_SUBCOMMANDS:
        return True
    if program == "git" and (
        _GIT_RUNS & set(args) or any(a.startswith(_GIT_RUNS_PREFIXES) for a in args)
    ):
        return True
    # A runner named anywhere later (`ssh host bash`, `--rsh=/bin/sh`): it may run what follows.
    for word in args:
        for name in (word, word.rsplit("=", 1)[-1]):
            name = name.rsplit("/", 1)[-1]
            if name != "." and (name in _RUNNERS or _INTERPRETER.fullmatch(name)):
                return True
    return False


def _shell_keys(command: str) -> set[str]:
    """Category keys of a command: one per program, only when every program word is plain and
    no program runs other code (a script, a command in its arguments, a project's task)."""
    command = _STDERR.sub("", command)
    # A substitution (`cmd`, $(cmd)) hides programs and a redirect (<, >) writes or reads
    # files: grant the exact text only.
    if re.search(r"`|\$\(|[<>]", command):
        return set()
    segments = [words for words in _segments(command) or [] if words]
    names = [words[0] for words in segments]
    # Only plain program names (ls, git, python3.11, g++), not paths or quoted words.
    if not names or any(_no_key(n) or not re.fullmatch(r"[\w.+-]+", n) for n in names):
        return set()
    if any(_runs_other_code(words[0], words[1:]) for words in segments):
        return set()
    return {"shell:" + name for name in names}


def session_cwd(state: dict) -> Path:
    """The session's working directory (a shell tool's `cd`), or home if unset or deleted.

    Relative paths of every tool resolve from here, and so do this review's, so the path
    reviewed is always the path used."""
    cwd = Path(state.get("cwd") or Path.home())
    return cwd if cwd.is_dir() else Path.home()


def resolve_path(path: str, state: dict) -> Path:
    """The real path a tool argument names: `~` expanded, relative to the session's cwd,
    symlinks followed (a tool acts on exactly the path reviewed). PathError
    (a ValueError) for an unknown `~name`, which review flags too."""
    return (session_cwd(state) / _expand(path)).resolve()


def readable_path(source: str, ctx) -> Path:
    """The real path of a local file a tool call `ctx` names to read (resolve_path), when it is
    not a protected location; ValueError (safe to show the model) when it is (the image tools'
    load_image)."""
    path = resolve_path(source, ctx.state)
    if is_secret_path(path):
        raise ValueError("that file is in a protected location")
    return path


def assess(kind: str, tool: str, args: dict, *, cwd: Path) -> Review:
    """Classify one tool call of review kind `kind` (shell, read, write, gui, prompt, url).

    `cwd` is `session_cwd(state)`: where a shell command runs and what relative file paths mean.
    `url` is a fetch whose URL could carry data out (`net.url_carries_data`): any grant covers
    only that exact call, since a category grant would let every later URL through."""
    unreadable = None  # a PathError: some path in the call cannot be read for review
    if kind == "shell" and args.get("workdir"):
        try:
            cwd = (cwd / _expand(args["workdir"])).resolve()
        except PathError as error:
            unreadable = error  # the command is still read from cwd below, for the floor
    text = json.dumps(
        {"tool": tool, "args": args, "cwd": str(cwd)}, sort_keys=True, ensure_ascii=False
    )
    review = Review(text=text)
    try:
        _classify(review, kind, tool, args, cwd)
    except PathError as error:  # the floor, if any, was set before anything could raise
        unreadable = error
    if unreadable:  # what the path means is unknown: the owner decides, for this call only
        review.flags, review.keys = [str(unreadable), *review.flags], set()
        review.owner_only = True
    return review


def _classify(review: Review, kind: str, tool: str, args: dict, cwd: Path) -> None:
    if kind == "shell":
        _assess_shell(review, args.get("command"), cwd)
    elif kind in ("read", "write"):
        _assess_file(review, kind, _resolve(args.get("path") or ".", cwd))
    elif kind == "gui":
        review.flags, review.keys = ["control an app"], {"gui:" + tool}  # per tool: a grant
        # for one app-controlling tool (the browser) must not cover another's
    elif kind == "prompt":
        review.flags, review.keys = ["writes text future prompts read"], {"prompt:" + tool}
    elif kind == "url":
        review.flags = ["send what could carry data out (a long or encoded URL, script or text)"]
    else:
        review.flags, review.owner_only = [f"unknown review kind {kind!r}"], True


def _assess_shell(review: Review, command: str | None, cwd: Path) -> None:
    if command is None:  # e.g. stopping a background job: the arguments say it all
        review.flags = ["manage a process"]
        return
    review.floor = floor_reason(command, cwd)
    if plain_read_only(command, cwd):
        return
    review.flags = ["shell command"]
    review.keys = _shell_keys(command)
    if _protected_mention(command, cwd) or _owner_write(_fold(cwd)):
        review.flags.append("touches a protected path")
        review.owner_only = True
    if _ADMIN.search(command):
        review.flags.append("runs as administrator")
        review.owner_only = True


def _assess_file(review: Review, kind: str, folded: str):
    """Reads need the owner for secrets; writes are free only in scratch and Arti's own work
    dirs, and need the owner for code, config and startup files."""
    if kind == "read":
        if _secret(folded):
            review.flags, review.owner_only = ["read a secret location"], True
    elif _owner_write(folded):
        review.flags = [
            "change Arti's code, config, grants or protected data, a secret or a startup file"
        ]
        review.owner_only = True
    elif not any(_under(folded, root) for root in _current_roots()["ordinary"]):
        review.flags = ["write a file"]
