# How Arti works

Arti is two programs. The **host** (`host/`, `python -m host`) runs directly on the machine,
macOS or Linux. It owns the chat platform (a **channel** package in `channels/`; none is included
yet), the **users** (`users.json`) and one **sandbox** per user: a rootless Podman container
of the sandbox image (`sandbox/Containerfile`, Debian 13). Each sandbox runs that user's
**agent** (`arti/`, `python -m arti`, with `plugins/`). The host and an agent talk only through
files in the agent's data directory (`arti/link.py`); there are no ports between them.

The agent is one Python process with one asyncio event loop. Everything optional in it is a
**plugin** (`plugins/`): safety, the tools and the model providers. A **turn** runs the agent
loop (ask the model, run its tool calls, repeat). Tool handlers run in worker threads; heavy or
risky work (shell commands, Chromium, page parsing) runs in child processes. The agent runs only
on Debian, in its sandbox: it has no macOS code.

The path of one message, in three steps:

1. **In:** the channel hands the message to the host (`Host.receive`). The host finds the user
   by the sender's handle (`host/users.py`), names the conversation, and leaves the message as a
   file in that user's agent's `link/in/` (`host/host.py`). The agent's channel, the **link**
   (`arti/link.py`), reads it and hands it to `app.admit()`, which saves it in the inbox
   (SQLite).
2. **Think:** `chat.py` (one `Chat` per conversation) queues it and starts a turn; `turn.py` asks
   the model (a provider plugin's) and runs its tool calls through `tools.py`, where the gate
   (the safety plugin's `plugins/safety/approvals.py`) checks each call before the plugin runs
   it.
3. **Out:** `app.send()` splits the reply into text and file parts (`reply.py`: a
   `MEDIA:/path` line is a file), checks every file (`App.file_problem`), and hands the parts to
   the link, which writes a request into `link/out/`. The host takes it, checks it again, and has
   the channel send it (formatted and split for its platform). The host's answer
   ("sent", "failed" or "unknown") comes back in `link/done/`.

## Glossary

- **host**: the program on the machine (`host/`): the channel, the users, their sandboxes, and
  the routing between them. It trusts no agent.
- **channel**: two things with one name. On the host, a chat platform: a package in
  `channels/<name>/` with `SETTINGS` and `make(host, settings)`, implementing
  `host.channel.Channel` (`start`, `send(chat, parts)`, `home_chat`, `notes`). In the agent, the
  interface core talks to (`arti.channel.Channel`), which is always the link.
- **user**: a person the host runs an agent for (`host/users.py`): an id (`alex`, the folder and
  sandbox name), a name, and a **handle** per channel, the stable id the platform names them by
  (a user ID, say). Only the operator adds users (`python -m host user add`).
- **sandbox**: a user's container, `arti-<user>` (`host/sandbox.py`). Their home folder,
  `<data>/homes/<user>/`, is its `/home/arti`; the agent's data directory is `/home/arti/.arti`.
- **agent**: one user's Arti: the `arti` package and the plugins, running in their sandbox.
  One persona, Arti, for everyone, with a private agent per user.
- **link**: the files the host and an agent share in the agent's data directory
  (`arti/link.py`): `link/me.json`, `link/in/`, `link/out/`, `link/done/`. `Link` is the agent's
  side.
- **chat**: a chat on the platform, named by the channel's own string for it.
  Only the host and its channel know chats; the agent never sees one.
- **home**, **owner**: the owner's one-on-one conversation key and handle (`Channel.home`,
  `Channel.owner`; `App.home`, `App.owner`). For the link, home is `"home"` and the owner is
  the user's handle, from `link/me.json`.
- **key**: a conversation key: `"home"`, or `"<place>|<handle>"` for the user's conversation in a
  group chat; `""` in a headless run. Core works on keys (`Input.key`,
  `Conversation.key`, `ctx.key`, the store, the channel's methods) and never parses one: it asks
  the channel what a key means (`scope_of`, `person_of`). The host keeps which chat each key is
  (`routes.json`).
- **scope**, **place**: the place a key lives in (`Channel.scope_of`, `App.scope_for`): `"home"`,
  or `"chat-<16 hex>"` (from the place's id) for a group chat. `App.place_name(scope)` is its
  name for tags and questions.
- **person**: whose conversation a key is (`Channel.person_of`, `App.person_of`): `""` for the
  owner's one-on-one, the handle after the `|` for a place, or None for a key that is no
  conversation.
- **context folder**: a scope's folder for files: the data dir for home, `contexts/<scope>/` for
  a group chat (`App.context_dir`, `ctx.context_dir`).
- **conversation**: what a turn works on (`chat.Conversation`). A `Chat` is a conversation that
  lives in a chat (`app.conversation(key)`). A `Headless` one is a run with no chat (cron,
  delegation, review, `--once`), its history in memory only.
- **session**: a conversation's stored state (`store.current_session(key)`): route, history,
  summary, tools' state. `/new` starts a fresh one.
- **turn**: one run of the agent loop (`turn.run_turn`): ask the model, run its tool calls, repeat
  until it answers without tools.
- **input**: one inbound event (`store.Input`), kept in the inbox until it is in history: kind
  `message`, `notice` or `note`.
- **notice / note**: a notice starts a turn but needs no answer (a finished job, a wait;
  `App.notify`); a note is context for the next turn and starts none (an input of kind `note`,
  or a host note, `Conversation.note`).
- **steering**: a person's new input reaching a conversation while it works: the reply's tool
  calls that come after it are skipped so the model can reconsider (`Chat.person_inputs`,
  `turn.run_calls`), and an approval question ends when the owner writes back in words instead
  of answering (`Chat.messages_in`, `ToolContext.wait`).
- **plugin**: `plugins/<name>.py` or `plugins/<name>/`, with `register(api)` (`plugin.py`): it
  adds tools, prompt sections, services, hooks, the gate, the redactor, file checks, chat
  commands, its config section, providers and commands. It may require other plugins
  (`REQUIRES`) and import from them. **safety plugin**: `plugins/safety/`, the gate, redaction,
  the network rules and child processes; every plugin that can do something risky requires it.
  **provider**: a model route registered by name (`App.provider_specs`); **route**: a
  conversation's (provider, model, effort).
- **trusted**: `Conversation.trusted` (`ctx.trusted`): the owner's own conversation (`App.owns`:
  its person is `""` or the owner's handle; the owner's jobs too): every tool and private context.
  The host passes an agent only its own user's messages, so every chat conversation is trusted;
  the guest machinery (`guest=True` tools, no private context) remains for a conversation that
  is not.

## Reading order

The agent first, then the host.

1. `arti/turn.py`: the agent loop. Read this first; everything else serves it.
2. `arti/chat.py`: a conversation (your one-on-one, or yours in a group chat): its queue of
   inputs, its runner, `/stop` and `/new`.
3. `arti/app.py`: wiring: admitting inputs, sending, headless runs, startup, recovery, shutdown.
   `arti/channel.py` is the interface core talks to: sending parts, and what a key means;
   `arti/reply.py` turns a reply into those parts.
4. `arti/link.py`: the agent's channel, the files it shares with the host.
5. `arti/store.py`: the SQLite database: sessions, history, inbox, outbox and a small key-value
   table. `arti/secret.py` (`.env`, and the hook the redactor fills) and `arti/lock.py` (one Arti
   per data directory) are small.
6. `arti/plugin.py`: loading plugins and what they register; `arti/tools.py`: what a tool
   handler sees (`ToolContext`) and running one call.
7. `plugins/safety/`: whether a tool call may run (`approvals.py`, `risk.py`), redaction
   (`redact.py`; the core's side is `arti/secret.py`), URLs (`net.py`) and child processes
   (`proc.py`).
8. `arti/providers/`: the provider interface and its helpers (`__init__.py`, `http.py`).
9. `plugins/`: everything else, each plugin's `register()` first: the model providers
   (`plugins/anthropic/`: `messages.py`, `models.py`; `plugins/claude/`, claude-subscription:
   `claude.py`, then `relay.py` and `rewrite.py`; `plugins/openai/`) and the tools, one plugin
   each.
10. `host/`: the operator's commands (`__main__.py`), then `host.py` (routing messages in and
    requests out), `users.py` (`users.json` and the home folders), `sandbox.py` (the Podman
    command line), `config.py` (`host.toml`), `channel.py` (what a channel package implements)
    and `service.py` (launchd or systemd).
11. `sandbox/Containerfile`: what the image holds.

## The host

`python -m host` (`host/__main__.py`) loads `host.toml` (`host/config.py`), the host's `.env`
(the channel's secrets) and the channel (`host.channel` picks a package in `channels/`; `""`
picks the only one; with none installed it stops with a message saying so), takes the lock on
`<data>/arti.lock`, and starts the `Host` (`host/host.py`). The host:

- **prepares each user** (`Host.prepare`): makes their link folders and writes `link/me.json`:
  the owner's handle on this channel and the channel's notes for the system prompt.
- **starts each sandbox afresh** (`Sandboxes.start`): it removes any old container and runs a
  new one of the current image, with the current code. Podman restarts an agent that exits
  (`--restart unless-stopped`); the host stops every sandbox when it stops.
- **starts the channel**, which calls `host.receive(message)` for each message.
- **pumps the agents' requests** (`Host._pump`): every 0.1 s it looks in each user's
  `link/out/` and carries out their requests, oldest first, one user at a time.

A sandbox (`host/sandbox.py`) is run as the operator's own user (`--userns keep-id`: what it
writes in the home folder is theirs), with:

- the user's home folder at `/home/arti`, the one place it can write besides a scratch `/tmp`
  (a 512 MB tmpfs);
- Arti's code (`arti/`, `plugins/`) read only at `/opt/arti`, so a new version needs only a
  restart; the image holds the system and the child environments (`/opt/arti/.venv-web`, the
  `claude` CLI);
- no Linux capabilities (`--cap-drop ALL`), no privilege gain (`no-new-privileges`), a read-only
  root, and limits: `host.memory`, `host.cpus` and 512 processes.

The host's `.env`, `users.json` and `routes.json` are never mounted, so no agent can read the
channel's secrets or change whom it answers. On macOS, Podman runs the containers in its own
Linux VM (`podman machine`).

### Messages in

`Host.receive` finds the user whose handle on this channel is the sender's (`Users.by_handle`).
A message from anyone else is dropped. It names the conversation: `"home"` for a one-on-one
with the bot, or `"<place>|<handle>"` for the user's conversation in a group chat (the place
is the channel's stable id for it), and remembers which chat that key is in `routes.json`. It
moves any attachment into the user's `.arti/media/incoming/` and writes the message to
`link/in/` as one JSON file (an `Input`, plus the place's name). The files are named in arrival
order.

The agent's `Link` reads `link/in/` every 0.2 s, admits the inputs (the store drops an event it
already has, by its id) and only then deletes the files. So a crash on either side loses
nothing and runs nothing twice.

### Requests out

The agent asks with a file in `link/out/`: `{"id", "type": "send", "key", "parts"}` or
`{"id", "type": "typing", "key", "on"}`. The host takes one by renaming it to `.taken`, then
reads it. An agent is not trusted, so the host checks every request:

- at most 1 MB, and its id must be its file name;
- the key must be one of that user's own conversations (`"home"`, or a key in their routes):
  any other gets "failed";
- a file part must be a regular file inside that user's home folder (a link out of it is judged
  by where it points), or it becomes the text `[file not sent]`;
- text parts are redacted with the host's own secrets too.

Then the channel sends it, and the host writes the outcome to `link/done/<id>.json`: "sent",
"failed" (provably nothing sent) or "unknown". A send the host took but never finished (it
crashed) is answered "unknown" at the next start, never done twice. The agent's `Link.send`
waits for that answer; the App bounds the wait.

Every link file is written whole (a temporary name, then a rename), so a reader never sees half
of one.

## A message, end to end (in the agent)

1. The link hands the host's inputs to `app.admit`. The store records each input in the inbox
   before anything acts on it, and drops one whose event id it already has.
2. `app.admit` runs a slash command at once (marked done first, so a crash never re-runs it) and
   hands anything else to its conversation's `Chat`, which starts a turn if none is running.
   Arti never decides on the model's behalf what an event means: every event that reaches a
   conversation starts a turn, and the model replies or stays `[SILENT]`.
3. `turn.run_turn` moves the waiting inputs into history (audio is transcribed by the
   registered transcriber, if any, and images normalised on the way, `media.py`), builds the
   system prompt (`prompt.py`), compacts history if it is near the context window
   (`context.py`) and asks the provider. Compacting replaces old messages by a summary; outcomes
   still unknown and pending waits are kept word for word by the core, and plugins add what they
   must keep (`api.on_compact`: the todo list, the skills loaded).
4. The reply is stored in history **before** anything acts on it. Its text is sent (its
   `MEDIA:` lines as files, each one checked, below); its tool calls run one at a time
   (`tools.run_tool`: schema check, the gate, handler in a thread with a deadline that starts
   once the gate has passed, result redacted and capped). Each result is stored as soon as it is
   known.
5. Repeat until the model answers without tools. A final `[SILENT]` sends nothing; if the user's
   message got no answer at all, the model is reminded (twice at most).

Input that arrives while a turn runs joins it ("steering"): it is added after the current tool
results. If a message from the conversation's person arrived after the model was asked, that
reply's remaining tool calls are not run, so a correction ("no, stop, do X") takes effect before
anything else happens. Host notices (a job finished, a wait is over) skip nothing. No tool
blocks waiting for the user: `wait` schedules a notice (an inbox row with a `due` time, re-armed
at startup, cancelled by `/stop` and `/new`) and ends the turn.

## Conversations in a group chat

The core knows only conversation keys and the questions of `arti/channel.py`. In a group chat
each user who calls Arti has a conversation of their own with their own agent,
keyed `"<place>|<handle>"`. Everything in core works on keys: sessions and history, the inbox,
the outbox, `Chat` objects, waits, cron jobs. Core asks the link what a key means: its place
(`scope_of`: `"home"`, or `"chat-<16 hex>"` for a group chat), whose it is (`person_of`: `""`
for home, the handle after the `|`) and the place's name (`place_name`, kept in the store's kv
table as the host names it). Only the host turns a key back into the chat to send to.

An agent sees only its own user's messages to it, never the rest of a group chat, and only that
user can stop or steer it. Several users' conversations in one group chat run in separate
agents, side by side; everyone's answers come from the same bot.

A conversation is **trusted** when it is the owner's (`App.owns`). Trust is one flag,
`Conversation.trusted`, and everything that grants the owner's privileges checks it: the tools
offered (only `guest=True` tools otherwise), the approval gate, the prompt (no private memory,
no skills index or preloaded skills), memory reads and edits, `session_search`, vision's local
files and private-network URLs (web, vision, image, browser). Headless runs carry the trust of
what started them. Since the host passes an agent only its own user's messages, every chat
conversation an agent has today is trusted.

What still differs in a group chat is the place: its files (attachments, cache files, generated
images) live in its own folder, `contexts/<scope>/`, and a conversation there can send only files
from that folder. A memory written there is tagged with the place's name.

## Reliability rules

- **Persist before act.** An input is in the inbox before Arti acts on it; a reply is in history
  before its text is sent or its tools run; an outbox row exists before the link is called.
- **Exactly once in, at most once out.** An input reaches history and its inbox row is marked
  done in one transaction; a re-delivered input is dropped by its event id. A send whose outcome
  is unknown (crash, timeout, a send the host took but did not finish) is never repeated; the
  model is told it may or may not have been delivered.
- **Every tool call gets a result**, even when the turn is stopped, fails or Arti crashes
  (providers reject a history with an unanswered call). After a crash, the call that was running
  is marked "outcome unknown" and the conversation gets one notice, so Arti picks the task up
  again.
- **Every wait has a timeout**: model requests, tool calls, sends, child processes, shutdown. A
  hung tool thread is abandoned, never waited on forever.
- **A hang restarts the program** (`arti/watchdog.py`, used by both). A heartbeat on the event
  loop, fed only while the program still takes in messages (the agent: the link reads
  `link/in/`; the host: the channel is alive and the pump runs), resets a countdown kept by a
  separate thread. After `agent.hang_timeout` or `host.hang_timeout` (5 minutes) without one,
  every thread's stack goes to `logs/hang.log` and the program exits. Podman starts an agent
  again; the host's service (`python -m host service`: launchd on macOS, systemd on Linux,
  `host/service.py`) starts the host again. The rules above recover the rest. The clock stands
  still while the computer sleeps, so sleep is not a hang.
- **One program per data directory** (`arti/lock.py`: an exclusive lock on `arti.lock`): one
  host on the machine's data directory, one agent on each user's.
- **No leftover processes.** Every child runs in its own process group (the safety plugin's
  `proc.py`), recorded under `run/children/` while it lives; a timeout, `/stop` or shutdown kills
  the whole group, and the next start kills groups a crashed Arti left behind. Both are the
  safety plugin's start and stop hooks: every plugin that starts children requires it, so it
  loads before them, its start hook (the sweep) runs before theirs and its stop hook (kill every
  child) after theirs, as stop hooks run in reverse load order.
- **Bounded disk.** Images are stored as files in `media/`, never in the database; `cache/` and
  `media/` files (also each group chat's, under `contexts/`) older than
  `storage.cache_max_age_days` and inbox/outbox rows older than 7 days are deleted daily; link
  answers nobody waited for are deleted after an hour; conversations other than home idle for an
  hour are unloaded from memory. Large tool output is capped in history, with the full text in
  a cache file. Conversation history itself is kept: it is the archive `session_search` and the
  background reviewer read (text only, so it grows slowly).

## Safety

There are two layers: the sandbox (above), which limits what an agent can reach at all, and the
safety plugin inside each agent, which reviews what it does.

Safety is a plugin, `plugins/safety/`. The core offers it hooks and knows nothing of what it
does: `api.gate` (`tools.run_tool` calls the gate before every tool handler, and the tool's
deadline starts once it has passed), `api.redactor` (the core writes everything through it,
`arti/secret.py`), `api.file_check` (no secret file is sent), `api.chat_command` (`/approve`,
`/deny`), `api.settings` (`[approvals]`, `[network]`) and start and stop hooks (the process
sweep, also before a `--once` task, and kill-all). Without the plugin an agent runs unguarded:
every tool call runs unreviewed and nothing is redacted. But a plugin that requires it
(`REQUIRES = ("safety",)`) is skipped whenever it is disabled, missing or failed to load, and
every plugin with a reviewed tool must require it (tests/test_plugin.py checks), as must every
plugin that imports its modules: the risky tools, image generation and `claude-subscription`
(it runs the CLI as a child). The `anthropic` and `openai` plugins need nothing of it. So
without the safety plugin an agent still has the `anthropic`, `openai` and `openai-codex`
providers and the tools that need no review (`todo`, `wait`, `delegate_task`,
`session_search`) (tests/test_without_safety.py).

The safety plugin's `approvals.check` is the gate; no other plugin can skip it (there is one
gate at most, and only the core runs tools). Each tool declares a review kind (`shell`, `read`,
`write`, `gui`, `prompt`, `url` or none), or a function that picks one per call. `risk.assess`
turns a call into a `Review`:

- **Floor**: catastrophic commands (`rm -rf ~` or of a home in `/home` or a top-level system
  dir such as `/etc`, `/boot` or `/lib`, `mkfs`/`wipefs`, writing onto a raw disk, fork bombs,
  `kill -9 -1`, shutting down, rebooting or suspending through `shutdown`, `systemctl` or
  `loginctl`, ...) are blocked in every mode, plus your own `approvals.deny` patterns.
- **Owner-only**: reading secrets (credential stores, keyrings, `.env`/`.envrc` files, keys and
  certificates, files named like secrets or service accounts, all of `~/.config`), changing
  Arti's own code, startup files, bin directories, systemd user units, D-Bus services and
  autostart entries, `crontab`, reading the keyring (`secret-tool`, `kwallet-query`,
  `gpg --export-secret-keys`), `sudo`/`pkexec`/`run0`, and Arti's data dir, which fails closed
  (`risk._private_data`, `risk._owner_write`): everything in it (config, database, logs,
  logins, browser profiles, jobs, ...) needs the owner to read or write, except the folders
  tools work in and SOUL.md. Reading is open in `cache/`, `artifacts/` and `media/` and for
  `SOUL.md`, writing in `cache/` and `artifacts/`, and the same in a group chat's
  `contexts/<scope>/`. Memories, skills and jobs change only through their own tools. A file a
  plugin keeps there is protected without the plugin doing anything, even when it is disabled.
  The judge never decides these.
- **Plain read-only**: a shell command with no shell syntax at all (no `$`, globs, redirects,
  ...) whose programs are all on a short read-only list runs without review.
- **Data going out**: fetching a URL (web_extract, vision, image references, the browser) or
  giving a browser page a script or text normally needs no review, but one that could carry data
  out (over 300 characters, a part over 100, a long hex or base64 run, or more than 64
  characters in all of runs of 8+ letters and digits that hold both: `net.url_carries_data`,
  `net.text_carries_data`) is reviewed as `url`, so a file read earlier can't leave quietly. It
  is a heuristic, like the rest of the gate: it catches the obvious ways, not a determined
  encoder.
- Everything else is **flagged**: in `smart` mode the judge model (one cheap model call, the
  action shown as untrusted data) approves, denies or asks you; in `manual` mode you are asked.

A grant for a script runner or launcher (a shell, `python3`, `node`, `systemd-run`, `setsid`,
`bwrap`, ...) covers only that exact call, never "any later script", as does one for a call that
makes another program run code (`find -exec`, `git -c`/`git config`, a runner named later in the
command such as `ssh host bash`, or a task runner that runs what the project defines: `make`,
`npm`, `cargo run`, `tox`, ...); the floor and your `approvals.deny` patterns hold in every
conversation.

You answer with `/approve` or `/deny` in your one-on-one chat, where every question is asked; a
bare `/approve` answers the single waiting question, and a message in words instead counts as
no. In a conversation that is not the owner's, the gate would ask nothing: it runs a guest
tool's `prompt` writes unasked and refuses any other reviewed call. Scheduled jobs cannot ask
anyone, so a flagged call there needs a grant (or the judge, if `approvals.unattended =
"smart"`).

Memory, skills and SOUL.md are one set per agent. A memory written in a group chat is tagged
`[in <group>]` by the tool (a tag the model writes itself is dropped, and a replaced entry keeps
its tag), and the owner may mark an entry `[shared]`; the prompt tells the model untagged
entries come from the owner and tagged ones are hearsay.

The checks read command text without running it, so they are a review aid, not a sandbox: an
approved command runs with the agent's permissions, inside its sandbox. Odd spellings that fool
a pattern only move a call from "you decide" to "the judge decides", never to "runs
unreviewed".

The core keeps a small secret store (`arti/secret.py`): it loads `.env` (kept out of
`os.environ`, read by name with `get`, all of it with `env()`), and `remember` adds secrets that
arrive while it runs (the ChatGPT login's tokens, `plugins/openai/codex.py`; a channel's tokens,
in the host); everything it writes goes through the registered redactor. The safety
plugin's (`redact.py`) replaces every such value of 8 characters or more (also JSON-, URL- and
base64-encoded) and anything shaped like a key or token with `[REDACTED]` in history (including
the provider's stored replay copy of a reply, kept only when nothing in the message needed
redacting; a signed thinking block is the one exception, as changing it would break the reply),
logs, sent messages, judge prompts, tool results and the files that hold a long tool output in
full. Its `child_env` gives child processes an environment without `.env` values, secret-named
variables, model-provider settings or shell hooks. The host has no redactor plugin: it masks
the values of its own `.env` and remembered secrets as written.

## Providers

A plugin registers each provider by name (`api.provider`): its factory, its default, judge and
summary models, its context window and its `[providers.<id>]` options. `App.provider(name)`
builds it on first use: an object with
`await complete(messages, system=, tools=, model=, effort=, session_id=) -> Reply`. A provider
sends one request and returns one reply; it never runs tools or loops. The core keeps only the
interface and its helpers (`arti/providers/__init__.py`) and the one streaming HTTP client
(`arti/providers/http.py`: asyncio, verified TLS, no proxies or redirects). The four providers
come from three plugins:

- `plugins/anthropic/`: `anthropic`, the Messages API with an API key (`messages.py`: history
  encoding, stream assembly, rate-limit headers). `models.py` holds the Claude models: the
  defaults, the short names and which have a 1M window.
- `plugins/claude/`: `claude-subscription`; it requires `safety` and `anthropic` (the CLI is a
  child process; the wire format and the models are anthropic's). Each request starts the
  official `claude` CLI (`claude.py`; the sandbox image has it on PATH) with its own tools,
  settings and session saving off, replays the history to it, and judges the run; a watchdog
  fails a CLI that goes quiet (60 s before its request, 180 s after) as retryable. The CLI sees
  Arti's tools through an inert MCP server that only lists them (`mcp_stub.py`) and talks to a
  loopback relay in Arti (`relay.py`) that admits exactly one upstream request; `rewrite.py`
  rewrites that request: it restores Arti's own tool results and moves cache breakpoints onto
  content that repeats. Arti never reads the CLI's
  login; each user signs it in from their sandbox (`python -m host user exec ID claude`).
- `plugins/openai/`: `openai` (the Responses API, `responses.py`, with an API key) and
  `openai-codex` (the same API through a ChatGPT login, `codex.py`, which also holds the
  ChatGPT image client and the `login`/`logout`/`auth-status` commands).

Image generation is a tool, not a provider: `plugins/image/` (`image_generate`, its `[image]`
section) requires `safety` and `openai` and uses `codex.py`'s login and client.

## Tools and plugins

A plugin is `plugins/<name>.py`, or a package `plugins/<name>/` holding its own helper modules,
with a `register(api)` function (see `plugins/README.md`). Through `api` it adds tools, prompt
sections, services, start, stop, new-session and compaction hooks, a transcriber, its own config
section (`api.settings`), the gate (`api.gate`), the redactor (`api.redactor`), file checks
(`api.file_check`), chat commands (`api.chat_command`), model providers (`api.provider`) and
`python -m arti <name>` commands (`api.command`). The core never depends on a plugin, with no
exceptions: no module under `arti/` imports one, nor the host or a channel (tests/test_plugin.py
checks), and the core finds providers and commands only by name in what was registered. A plugin
depends on `arti.*`, reaching the core through `api` at registration and `ctx` (a
`ToolContext`) in handlers, and on the plugins it names in `REQUIRES`, which it may import from
(`from plugins.safety import proc`); on no other, and never on `host` or `channels`
(tests/test_plugin.py reads every plugin's imports). The host and the channels, in turn, use
only the core's small helpers (`arti.*`) and never import a plugin.

Startup: `config.load` validates the core sections (`agent`, `storage`, `plugins`,
`providers`) and keeps the others as written; the plugins are imported (to read `REQUIRES`) and
registered in dependency order, otherwise in name order (`plugin.load_order`), each
`api.settings` checking its section as it is declared. Disabled and broken ones are skipped
(names starting with `_` or `.` are never plugins), and so is a plugin whose requirement is not
loaded (transitively) or that is caught in a cycle of requirements. Then `config.check_plugins`
rejects any section nobody declared (while some plugin is disabled or broken, such a section is
ignored instead: it may be that plugin's), checks the providers named (`agent.provider` and the
plugins' `provider_keys`, such as `approvals.provider`) and their tables, and fills in the
default model. Only then does the App make its link, take the lock and start.

What several plugins share lives in a plugin they require, mostly the safety plugin
(`risk.resolve_path`, `net.private_ok`, `net.untrusted`, `proc.run`). Two small helpers are
copied instead, because the plugins that use them need nothing else of each other:
`replace_text` (the patch rule) in `plugins/files.py` and `plugins/skills.py`, and `load_image`
(a tool's image argument) in `plugins/vision.py` and `plugins/image/`; each pair is checked by
the same tests (tests/plugins/test_replace_text.py, tests/plugins/test_load_image.py).
`arti/media.py` keeps only what the core needs: an inbound input to content blocks, and
normalising images (their size read from their own bytes; converted, when needed, with
ImageMagick, which is in the sandbox image).

Sending is the core's, so every channel gets the same rules and none can forget one. Everything
Arti sends goes through `App.send`: replies, scheduled results (`App.deliver`), approval
questions. A reply's `MEDIA:/path` lines become file parts (`reply.py`), and every file passes
`App.file_problem`: it must be a readable file, not raw tool output, inside the conversation's
own place folder unless the conversation is home, and pass every plugin's `api.file_check` (the
safety plugin's refuses secret files). A file that fails is sent as a note saying why. The parts
then go to `Channel.send(key, parts)`: the link, and through it the host, which checks them once
more (above). What is left to the host's channel is its platform: formatting and splitting text,
and what to do with a file its platform can't carry.

Memory and skills grow two ways: during a conversation, when the model uses `memory` or
`skill_manage`, and afterwards, through `plugins/review.py` (`[review] enabled`). Once a
conversation has been quiet for 30 minutes after a few new messages, a background run on the
summary model reads what is new (from the archive, so summarising or `/new` loses nothing) and
saves what is worth keeping: memory (tagged with the group chat, for one) and, for the owner's
one-on-one only, skills. It runs with the trust of the conversation it reviews and reads only
what people and Arti said: no tool results or host notes, since it writes without asking (and
writes nothing else, and never marks memory shared); every other background run still needs a
grant for that.

## Data directories

On the machine, the host's (`~/.arti`, or `$ARTI_HOME`):

```
host.toml .env              the host's settings and secrets (the channel's credentials)
users.json                  the users and their handles (only the host writes it)
routes.json                 which chat each user's group conversation keys name
homes/<user>/               each user's home folder: their sandbox's /home/arti
logs/                       arti.log (rotating, redacted), hang.log, daemon.{out,err}.log
arti.lock                   one host at a time
```

In each sandbox, the agent's (`/home/arti/.arti`, that is `homes/<user>/.arti/` on the
machine):

```
config.toml .env            the agent's settings and secrets
link/                       the files shared with the host: me.json, in/, out/, done/
state.db                    SQLite: sessions, history, inbox, outbox, kv (the core's store)
grants.json                 "always" approvals (the safety plugin)
logs/arti.log               rotating, redacted (hang.log: stacks of a hang, watchdog.py)
SOUL.md memories/ skills/   persona, memory, skills: one set, used in every chat
browser_profiles/           the browser's signed-in profiles (browser-login)
contexts/<scope>/           one folder per group chat for its files (media/, cache/,
                            artifacts/, browser_profiles/)
cron/ scripts/              scheduled jobs and their output; their scripts
media/ cache/               inbound files (media/incoming/, from the host) and images;
                            disposable files (both pruned)
artifacts/images/           generated images (kept)
claude/ codex_auth.json     the claude CLI's working dir; the ChatGPT login
arti.lock run/children/     one agent at a time; live child processes (the safety plugin)
```

The rest of the user's home folder is the agent's own (`~/.claude`, the Claude Code CLI's
login, say). Everything in the agent's data directory needs the owner to read or write (the
safety plugin's rule), except reading `SOUL.md`, `cache/`, `artifacts/` and `media/`, and
writing `cache/` and `artifacts/`; the same folders of each `contexts/<scope>/`.
