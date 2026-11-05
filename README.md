# Arti

Arti is a small personal AI agent you talk to in a chat app. One machine (macOS or Linux) can
run Arti for several people. Each person gets their own agent, and each agent runs in its own
Linux sandbox. An agent can run shell commands, read and edit files, search and read the web,
see and make images, remember things, keep skills and run scheduled jobs.

**Status:** work in progress. No chat app is included yet. The host reaches people through a
**channel** package, and until one is added the host won't start (see [Channels](#channels)).
The agents, the sandboxes and the tests work without one.

Arti is Python 3.11 and uses only the standard library. Heavier features run in separate child
processes, some in their own virtual environment.

More: [how it works](docs/architecture.md), [the agent's settings](config.example.toml), [the
host's settings](host.example.toml), [writing a plugin or a channel](plugins/README.md). To read
the code, start with the glossary and the reading order at the top of docs/architecture.md.
Every module opens with what it does and how it fits.

## How it fits together

There are two programs:

- The **host** (`python -m host`, the `host/` package) runs directly on the machine. It owns
  the chat platform (a **channel**, a package in `channels/`), the list of **users** and one
  **sandbox** per user. It passes each message to the agent of the person who sent it, and
  sends what the agents answer.
- The **agent** (`python -m arti`, the `arti/` package with `plugins/`) runs inside a user's
  sandbox. It is that user's Arti: its own history, memory, skills, jobs, settings and model
  sign-in.

A sandbox is a rootless [Podman](https://podman.io) container of the sandbox image
(`sandbox/Containerfile`: Debian 13 "trixie", Python 3.11). Even a single user gets one. The
host and an agent talk only through files in the agent's data folder; there are no ports
between them.

```
chat app ──▶ channel ──▶ host (on the machine) ──▶ Alex's sandbox: Alex's agent
                                               └─▶ Sam's sandbox:  Sam's agent
```

Everyone talks to the same bot, called Arti. Each person has their own private Arti behind it,
and only that person can talk to it, in a one-on-one chat or in any group chat. A message from
anyone who is not a user is dropped.

## Requirements

- macOS or Linux, with Python 3.11 (the host needs nothing from pip).
- [Podman](https://podman.io/docs/installation). On macOS, Podman runs its containers in a
  small Linux VM of its own (`podman machine`). Keep the repository and Arti's data folder in
  your home folder there: Podman's VM shares it by default.
- A channel for your chat app (none is included yet; see [Channels](#channels)).
- For each user, one model provider (see [Providers](#providers)): a Claude subscription, a
  ChatGPT login, or an API key.

## Install

```sh
cd arti
python3.11 -m venv .venv          # the host needs nothing from pip
podman machine init               # macOS only, once
podman machine start              # macOS: before build after a reboot (the host starts it)
.venv/bin/python -m host build    # build the sandbox image (arti-sandbox)
```

The image holds the system and the child environments: ImageMagick (image conversion), git,
curl, procps, the web plugin's page and PDF parser (`/opt/arti/.venv-web`) and the official
Claude Code CLI (`claude`). Arti's code is not copied in: it is mounted read only, so after you
update the code you only restart the host. Run `build` again for a new image (a new Claude Code
CLI, say), then restart the host.

## Channels

A channel connects the host to one chat app: it brings messages in, names each sender by a
stable handle (their user id on that app), and sends the agents' answers. It is a package in
`channels/<name>/` that runs in the host, on the machine. `host.channel` in `host.toml` picks one,
and is left empty when there is only one.

No channel is included yet. [Writing a host channel](plugins/README.md#writing-a-host-channel)
describes what a channel does, and `host/channel.py` holds the interface. Until one is installed:

- `python -m host` stops at once with "no chat channel is installed".
- `python -m host user add` is refused too, because a user is added with their handle on the
  channel.
- `build`, `test`, `service`, `user list` and `user remove` work. `user exec` needs a running
  host, so it needs a channel too.

A channel's credentials go in the host's secrets file, `~/.arti/.env`, as `KEY=value` lines. Its
settings go in its own section of `host.toml`, named after it.

## The data folder

Everything Arti keeps on the machine is in `~/.arti` (or `$ARTI_HOME`):

```
host.toml           the host's settings (optional; see host.example.toml)
.env                the host's secrets: the channel's credentials
users.json          the users (only the host writes it)
routes.json         which chat each user's group conversations are in
homes/<user>/       each user's home folder: their sandbox's /home/arti
logs/               the host's logs (arti.log, hang.log, and daemon.*.log under the service)
arti.lock           one host at a time
```

The host's `.env` and `users.json` are never mounted into a sandbox, so no agent can read the
channel's secrets or change whom it answers.

## Add users

```sh
.venv/bin/python -m host user add alex --name Alex --handle U123
```

- `alex` is the user's id: lower-case letters, digits and `-`, starting with a letter. It names
  their folder (`homes/alex/`) and their sandbox (`arti-alex`).
- `--name` is how Arti calls them.
- `--handle` is their id on the channel, the stable id the chat app names them by (`U123` is
  only a placeholder).
- `--provider` is their model provider (see [Providers](#providers)); `claude-subscription` if
  you leave it out.

`add` also writes the agent's first settings file, `homes/alex/.arti/config.toml`, with the
user's name, the machine's time zone and the provider.

Other user commands:

```sh
.venv/bin/python -m host user list              # every user and their handles
.venv/bin/python -m host user remove alex       # forget a user, remove their sandbox; folder kept
.venv/bin/python -m host user exec alex         # a shell in Alex's sandbox (the host must run)
.venv/bin/python -m host user exec alex CMD...  # run one command there
```

## Sign each agent in to a model provider

Each user's agent signs in by itself, inside its sandbox. Start the host first (the sandboxes
run while it runs), then pick one:

- **Claude subscription** (`claude-subscription`): run the Claude Code CLI in the sandbox and
  sign in there: `.venv/bin/python -m host user exec alex claude`. Its login stays in Alex's
  home folder; Arti never reads it.
- **ChatGPT login** (`openai-codex`):
  `.venv/bin/python -m host user exec alex python -m arti login`.
- **API key** (`anthropic` or `openai`): put `ANTHROPIC_API_KEY=...` or `OPENAI_API_KEY=...` in
  that user's `homes/alex/.arti/.env`.

The provider is named in the user's `homes/alex/.arti/config.toml`, which `user add` wrote.
Four providers are loaded, so it must be set:

```toml
[agent]
provider = "claude-subscription"
user_name = "Alex"
timezone = "Europe/Berlin"
```

Restart the host after changing an agent's settings (or send `/restart` to Arti). Every key is
listed in [config.example.toml](config.example.toml).

## Run

```sh
.venv/bin/python -m host
```

This starts every user's sandbox afresh (current code, current image), then the channel. It
stops the sandboxes when it stops. While it runs, Podman restarts an agent that exits.

To keep the host running, make it a service:

```sh
.venv/bin/python -m host service
```

This writes a launchd agent (`ai.arti.host`) on macOS or a systemd user unit
(`arti-host.service`) on Linux. It starts the host at login and again 30 seconds after it stops.
Nothing is started for you: the command prints the commands that start it (on Linux also
`loginctl enable-linger`, so it runs while you are logged out). On macOS the host starts
Podman's VM itself when it is down.

`.venv/bin/python -m host --help` lists every host command. Logs: the host's in
`~/.arti/logs/`, each agent's in `homes/<user>/.arti/logs/arti.log`.

## Talking to Arti

- **One-on-one:** message Arti directly. This is your **home** chat with your Arti. Approval
  questions always come here.
- **In a group chat:** call Arti the way your channel allows (a mention, say). You get a
  conversation of your own with your Arti in that chat: its own history, `/stop` and `/new`.
  Your Arti sees only your messages to it, not the rest of the chat, and other people there
  cannot steer it. Others who are users talk to their own Arti in the same way; everyone's
  answers come from the same bot.
- Memory, skills and `SOUL.md` (Arti's persona, written on first use) are the same in all your
  conversations. A memory written in a group chat is tagged with the chat's name. Files from a
  group chat's conversation stay in that chat's own folder, and only those files can be sent
  there.

Chat commands, in any of your conversations (for that conversation only):

- `/stop` stops the current task; `/new` also starts a fresh conversation.
- `/status` shows the model, what Arti is doing and quota; `/help` lists the commands.

Only in your one-on-one chat:

- `/model [provider/]model` and `/effort low|medium|high|xhigh|max` change the model.
- `/approve [id] [once|session|always]` and `/deny [id]` answer an approval question (the id
  only when several are waiting; both come with the safety plugin).
- `/restart` restarts your Arti.

## Providers

Each provider comes from a plugin. Choose one with `[agent] provider` in the user's
config.toml, or with `/model`.

| Provider | Plugin | Sign-in | Default model |
|---|---|---|---|
| `claude-subscription` | `plugins/claude/` | the Claude Code CLI's own login, in the sandbox (Arti never reads it) | `claude-opus-5-5` |
| `anthropic` | `plugins/anthropic/` | `ANTHROPIC_API_KEY` in the user's `.env` | `claude-opus-5-5` |
| `openai` | `plugins/openai/` | `OPENAI_API_KEY` in the user's `.env` | `gpt-5.4` |
| `openai-codex` | `plugins/openai/` | `python -m arti login` in the sandbox (ChatGPT device login) | `gpt-5.4` |

## Tools

| Plugin | Tools |
|---|---|
| files, terminal | `read_file` `write_file` `patch` `search_files`; `terminal` `process` |
| web, browser | `web_search` `web_extract`; `browser` (headless Chromium actions) |
| vision, image | `vision_analyze`; `image_generate` (needs the ChatGPT login, `python -m arti login`, whatever the provider) |
| memory, skills, todo | `memory`; `skill_view` `skill_manage`; `todo` |
| cron, delegate, session_search, wait | `cronjob`; `delegate_task`; `session_search`; `wait` (come back to the conversation later) |
| review | no tool: once a conversation has at least 4 new messages from people and has been quiet for 30 minutes, a background run looks back at it and saves memories worth keeping (and, from your one-on-one, skills) |

The `browser` tool needs Chromium or Google Chrome on the sandbox's PATH (or
`browser.executable`). The image has none yet, so the tool is hidden there.

## Safety

There are two layers.

- **The sandbox.** Each agent runs in its own container, as your own user (`--userns
  keep-id`), with no Linux capabilities, no privilege gain, a read-only system, a scratch `/tmp`
  and limits on memory, CPU and processes. It can write only in that user's home folder
  (`homes/<user>/`, its `/home/arti`). It sees Arti's code read only, and nothing else of the
  machine. It can reach the network.
- **The review gate** (the safety plugin, `plugins/safety/`), inside each agent: the approval
  gate, redaction, network rules and the handling of child processes, with `/approve`, `/deny`
  and the `[approvals]` and `[network]` settings.

What the gate does:

- **Approvals:** shell commands, file writes, writes to text later prompts read, and URLs or page
  text that could carry data out are reviewed before they run; reading a file is checked, and
  reading a secret location needs you. Plainly read-only commands (`ls`, `cat`, `grep`, ...) run
  unreviewed. In `smart` mode (the default) a judge model approves routine actions and asks you
  when unsure; `manual` always asks you; `off` asks only for what is always yours to decide
  (secrets, changes to Arti itself, `sudo`, startup files, systemd units, ...). Catastrophic
  commands (`rm -rf ~`, disk erase, ...) are blocked in every mode. Scheduled jobs cannot ask, so
  by default they only run what is already allowed (`approvals.unattended = "smart"` lets the
  judge decide for them).
- **Secrets:** the values in the agent's `.env`, the tokens of its ChatGPT login and things that
  look like keys are redacted everywhere the agent writes (logs, history, messages; values of 8
  characters or more), and its child processes never get them. Reading credential stores
  (`~/.ssh`, `.env` files, ...) needs you. No secret file is ever sent. The host also masks its
  own secrets in everything it sends.
- **Arti's own data:** anything in the agent's data folder (config, database, logs, logins,
  browser profiles, jobs, memories, ...) needs you to read or write it, except reading `SOUL.md`
  and the folders tools work in: `cache/`, `artifacts/` and `media/` may be read, `cache/` and
  `artifacts/` written (the same in a group chat's `contexts/<scope>/`). Memory, skills and jobs
  change through their own tools.
- **Network:** web tools and the browser refuse private and local addresses unless
  `network.allow_private_urls` is set; cloud-metadata addresses are always refused.
- **Processes:** every child runs in its own process group and is killed on timeout, `/stop` and
  shutdown; leftovers of a crash are killed at the next start.

Without the safety plugin (listed in `[plugins] disabled`, or failing to load) an agent runs
unguarded: tool calls run with no review and nothing is redacted. So every plugin that can do
something risky requires the safety plugin, and is skipped (and logged) whenever it is not
loaded.

The host trusts no agent: an agent may only answer its own user's conversations, may only send
files inside its own home folder, and its requests have size limits.

## Tests

```sh
.venv/bin/python -m host test                  # the whole suite, inside the sandbox image
uvx ruff@0.14.0 check && uvx ruff@0.14.0 format --check
```

`python -m host test` runs the tests on the system the agents run on, with the code and tests
mounted read only. The tests are offline (any non-loopback connection fails them) and use a
temporary `ARTI_HOME`. The browser's tests against a real Chrome or Chromium
(`ARTI_TEST_CHROME=1`) don't run there yet: the image has none.

## Layout

```
host/            the host: its commands (__main__.py), message routing (host.py), users,
                 sandboxes, settings, the service, and the channel interface (channel.py)
channels/        the chat platforms the host can run, one package each (none included yet)
sandbox/         the sandbox image (Containerfile)
arti/            the agent's core: app, chat, turn, tools, plugin (the plugin API), link (the
                 files shared with the host), store, secret (.env), lock, the provider
                 interface (providers/), ... It imports no plugin and names none.
plugins/         everything else in the agent, one plugin each: safety/ (approvals, redaction,
                 network rules, child processes), the model providers (anthropic/, claude/,
                 openai/) and every tool (image/ is image_generate; web/ holds the child script
                 run in .venv-web); a plugin is <name>.py or a package <name>/
tests/           the offline test suite, laid out like the source
docs/            how it works (architecture.md)
```

Planned: a chat channel, Linux computer use and voice built for the sandbox, and Chromium in
the image for the browser tool.
