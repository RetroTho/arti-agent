# Writing a plugin

Everything optional in a user's agent is a plugin: safety (the approval gate and redaction),
every tool, every model provider, their settings and their command-line commands. Plugins run in
the agent, inside the user's sandbox (Debian, the sandbox image), never on the host. The core
(`arti/`) never imports a plugin; it knows each one only by what it registered. The chat platform
is not a plugin: it is the host's, a channel package in `channels/` (see
[Writing a host channel](#writing-a-host-channel)).

A plugin is either one file, `plugins/<name>.py`, or a package, `plugins/<name>/` with an
`__init__.py`; either has a `register(api)` function. A package keeps its own helper modules
beside its `__init__.py` and imports them itself (`from . import devtools`); nothing else loads
them. Arti loads every plugin here at startup, in dependency order and otherwise in name order
(see [Requiring other plugins](#requiring-other-plugins-requires)), except those in
`[plugins] disabled` and names starting with `_` or `.`. A plugin that fails to import or register
is logged and skipped (its registrations are dropped, and its config section ignored); Arti still
starts. A plugin imports only `arti.*`, its own modules, and the plugins it requires.

```python
"""shout: reply in capitals."""


def shout(args, ctx):
    return args["text"].upper()


def register(api):
    schema = {"properties": {"text": {"type": "string"}}, "required": ["text"]}
    api.tool("shout", "Return the text in capitals.", schema, shout, review=None)
```

`api.app` is the App and `api.config` its settings (the parsed config.toml). The whole API, in
`arti/plugin.py`:

| | |
|---|---|
| `api.tool(name, description, schema, handler, *, review, ...)` | a tool the model can call |
| `api.prompt_section(fn)` | text for the system prompt |
| `api.service(name, fn)` | a function other parts call by name |
| `api.on_start(fn, once=False)`, `api.on_stop(fn)` | hooks at startup and shutdown |
| `api.on_new_session(fn)` | a hook when `/new` starts a conversation afresh |
| `api.on_compact(fn)` | text kept word for word when old history is summarised |
| `api.transcriber(fn)` | turns audio into text (at most one) |
| `api.gate(fn)` | reviews every tool call before it runs (at most one: the safety plugin's) |
| `api.redactor(text, obj)` | hides secrets in everything the core writes (at most one: the safety plugin's) |
| `api.file_check(fn)` | refuses files that must not be sent (the safety plugin's: secret files) |
| `api.chat_command(name, fn, help, *, home_only=True)` | a slash command, like `/approve` |
| `api.settings(section, defaults, choices=None, provider_keys=())` | the plugin's own config.toml section |
| `api.provider(name, factory, *, default_model, judge_model, summary_model, context_window, options)` | a model provider |
| `api.command(name, func, help)` | a `python -m arti <name>` command |

## `api.tool(name, description, schema, handler, *, review, ...)`

- `schema`: the arguments as JSON schema properties (`type: object` and
  `additionalProperties: false` are added). Calls are checked against it before anything runs.
- `handler(args, ctx)` runs in a thread of its own and may block. Return a string, a dict (sent as
  JSON), or a list of blocks (`{"type": "text", "text": ...}`, `{"type": "image", "path": ...}`).
  Raise `arti.tools.ToolError("why")` when nothing happened, `arti.tools.UnknownOutcome("why")`
  when something may have happened (a send, a purchase) so the model checks before retrying.
- `review`: what the gate (the safety plugin's approval gate) checks first. `"shell"` (args
  `command`, `workdir`), `"read"` or `"write"` (arg `path`), `"gui"` (controls apps), `"prompt"`
  (writes text later prompts read), `"url"` (sends something that could carry data out; see
  `plugins.safety.net.url_carries_data`), or `None`. It may also be a function `args -> kind`
  (e.g. no review for read-only actions). A plugin with any tool whose review is not `None`
  must require the safety plugin (`REQUIRES = ("safety",)`; tests/test_plugin.py checks), so
  that it never runs without the gate.
- Options: `timeout` (seconds, default 300; it starts once the gate has let the call through,
  so time spent waiting for your `/approve` does not count), `interactive=True` (needs a chat;
  hidden in cron and delegated runs), `guest=True` (also offered to a conversation that is not the owner's; see
  below), `available=lambda app: ...` (hide the tool when a dependency is missing, such as a
  program not in the sandbox image). A tool never waits for the user's answer: replies reach
  the model as inputs of their own.

### `guest=True`

A conversation is trusted when it is the owner's (`ctx.trusted`). The host passes each agent only
its own user's messages, so today every conversation with a chat is trusted and gets every
tool. The core still keeps a stricter mode for a conversation that is not the owner's: it gets
only the tools registered with `guest=True`, and never an approval question (the safety plugin's
gate runs a guest tool's calls of review kind `prompt` unasked and refuses any other reviewed
call). So mark a tool `guest` only when it is harmless for a stranger to use as is, and bound
whatever it can reach with `ctx.trusted` and `ctx.context_dir` (the memory tool, for example,
lets a guest read and change only that place's entries).

## What a handler gets: `ctx`

| | |
|---|---|
| `ctx.app` | the App: `config`, `data_dir`, `store`, `services`, `state`, `channel`, `tz`, `home`, `owner` |
| `ctx.key`, `ctx.scope` | the conversation key (`"home"`, or `"<place>|<handle>"` in a group chat; `""` in a background run) and its context (`"home"` or `"chat-..."`, a group chat) |
| `ctx.trusted` | the owner's own conversation (their chats, their jobs); False in anyone else's: show nothing private; check this, not the scope |
| `ctx.headless` | True in a run with no chat (cron, delegation, review): nothing can be sent |
| `ctx.conv` | the conversation itself (`chat.Conversation`), for what the fields here leave out |
| `ctx.context_dir` | this context's folder (your data dir, or `contexts/<scope>/`) |
| `ctx.cache_dir(name)` | a private cache folder of this context; put every file a call makes here or under `ctx.context_dir`, so a group chat never sees another's files |
| `ctx.state` | a dict saved with the session (cwd, grants, tools' own state); reset by `/new` |
| `ctx.cancel`, `ctx.check()` | set by `/stop`, a timeout or shutdown; `check()` raises if set |
| `ctx.wait(timeout, until=event)` | wait for an event, new input from a person (unless `on_message=False`), a cancel or the timeout |
| `ctx.mark_answered()` | count the latest message as answered without a text reply |
| `ctx.notify(text, at=None)` | start a later turn in this conversation (a background job finished), or at unix time `at` (kept across restarts; `/stop` and `/new` cancel it) |
| `ctx.call(coro, timeout=60)` | run a coroutine on the App's event loop and wait for it (a cancel stops waiting) |
| `ctx.run_headless(prompt, instructions="")` | run a sub-agent with (at most) this conversation's tools; `instructions` is extra system-prompt text |

A tool sends nothing itself: what the model writes is the reply, and a file goes out as a
`MEDIA:/absolute/path` line in it (the core checks every file). Run child processes with the
safety plugin's `proc.run(...)` (killed with their whole process
group on timeout or cancel, and at shutdown) and fetch URLs with its `net.request(...)` (refuses
local addresses): `from plugins.safety import net, proc`, with `REQUIRES = ("safety",)`.

## Requiring other plugins: `REQUIRES`

Plugins share code by requiring each other. A plugin names the plugins it needs in a
module-level tuple, and may then import from them (absolute imports, by the plugin's name):

```python
"""fetcher: a tool that downloads a page."""

from plugins.safety import net

REQUIRES = ("safety",)
```

- Arti imports every plugin first (that is how it reads `REQUIRES`; `register` has not run),
  then registers them so that each comes after every plugin it requires, and otherwise in name
  order. So the start hooks of a plugin run after those of the plugins it requires, and its stop
  hooks before theirs.
- A plugin whose requirement is disabled, missing or failed to load is skipped (logged), and so
  is every plugin that requires it, however indirectly. Plugins that require each other in a
  cycle are skipped too, with the plugins that require them. A skipped plugin's config section
  is ignored, like a disabled one's.
- A plugin's modules import only `arti.*`, their own plugin and the plugins in its `REQUIRES`
  (tests/test_plugin.py reads every plugin's source and checks), never `tests`, `host` or
  `channels`.
- Plugins run only on Debian, in the sandbox. A program a plugin needs must be in the sandbox
  image (`sandbox/Containerfile`); hide the tool with `available=` when it is missing.
- A small helper two plugins need, when neither needs anything else of the other, is copied
  into both rather than shared, and one test module checks both copies: `replace_text` in
  `files.py` and `skills.py` (tests/plugins/test_replace_text.py), `load_image` in `vision.py`
  and `image/` (tests/plugins/test_load_image.py).

### The safety plugin

`plugins/safety/` is the approval gate (`approvals.py`, with `risk.py` classifying calls),
redaction (`redact.py`), the network rules and the one HTTP request function (`net.py`) and the
handling of child processes (`proc.py`). It registers the gate, the redactor, `/approve` and
`/deny`, its `[approvals]` and `[network]` sections and two hooks: at start (also before a
`--once` task) it kills the process groups a crashed Arti left behind and starts recording
children; at shutdown it kills every child still running.

Without it Arti runs unguarded: no gate (every tool call runs unreviewed) and no redactor
(nothing is redacted). That is the reason for the rule above: every plugin that can do something
risky requires it, so that it is skipped rather than run unguarded. Every plugin that uses its
modules requires it anyway: the risky tools, `image` and `claude` (claude-subscription).
`anthropic` and `openai` do not, so without the safety plugin an agent still has their
providers and the tools that need no review (tests/test_without_safety.py).

## Prompt sections, services and hooks

- `api.prompt_section(fn)`: `fn(info) -> str | None` adds text to the system prompt (`info.app`,
  `info.scope`, `info.context_dir`, `info.key`, `info.headless`, `info.trusted`: leave anything
  private out when it is False). Keep it short and stable: it is sent with every request.
- `api.service(name, fn)`: a function other plugins call by name, `app.services[name](app, ...)`
  (e.g. `preload_skills`). Check that it exists: the plugin offering it may be disabled.
- `api.on_start(fn, once=False)`: `fn(app)` (plain or async) runs at startup, after recovery and
  before the channel starts, in load order. A `--once` task runs only the hooks registered with
  `once=True` (the safety plugin's sweep); a command runs none.
- `api.on_stop(fn)`: `fn(app)` runs at every shutdown (a `--once` task's too), in reverse load
  order: a plugin stops before the plugins it requires. Each gets 5 s.
- `api.gate(fn)`: `fn(ctx, tool, args)` reviews every tool call before its handler runs (in the
  call's worker thread, after the schema check): None lets the call run, a string refuses it and
  is the call's result. It may wait (for the owner's answer, say): the tool's own deadline starts
  once it has passed. At most one plugin registers one; with none, every call runs unreviewed.
- `api.redactor(text, obj)`: `text(s)` returns `s` with secrets hidden and `obj(value)` a hidden
  copy of a message or other JSON-like value; the core then writes everything through them
  (`arti/secret.py`): inputs, history, tool results, error messages, sends, the log and the
  command line's errors. At most one; with none, nothing is redacted.
- `api.file_check(fn)`: `fn(path)` returns why a file must not be sent, or None. Every file Arti
  sends passes every registered check (`App.file_problem`); it runs in a thread and must be
  quick. The safety plugin's refuses secret files.
- `api.chat_command(name, fn, help, *, home_only=True)`: `/name arg` runs `fn(app, inp, arg)` at
  once, even mid-turn (`inp` is the message, a `store.Input`; plain or async), and sends the
  reply it returns. With `home_only` (the default) it works only in the owner's one-on-one
  chat; otherwise in any conversation, for that conversation. `help` follows `/name ` in `/help`
  (`"[id] - answer an approval question"`). The name is one lower-case word; the core's
  commands (`/stop`, `/new`, `/status`, `/help`, `/model`, `/effort`, `/reasoning`, `/restart`)
  and another plugin's can't be taken.
- `api.on_new_session(fn)`: `fn(app, key)` runs in a thread when `/new` starts that conversation
  afresh (stop its background jobs, say); all hooks together get 15 s.
- `api.on_compact(fn)`: `fn(conv, span)` returns text to keep word for word under the summary when
  old history is summarised (the open todo list, the skills loaded), or None; it may also drop
  its own `conv.state` keys. Unknown outcomes and pending waits are kept by the core itself.
- `api.transcriber(fn)`: `fn(app, path, cancel)` returns an audio file's words, or None; the core
  uses it for audio that arrives. At most one plugin may register one (no stock plugin does).

## Settings: `api.settings(section, defaults, choices=None, provider_keys=())`

Declares the plugin's own config.toml section and returns it validated, with every default filled
in (also at `app.config[section]`), so `register` can use it at once:

```python
SETTINGS = {"user_chars": 1375, "notes_chars": 2200}

def register(api):
    settings = api.settings("memory", SETTINGS)
```

A default's type is the rule for its value, as for the core settings: a bool, a whole number, a
number (an integer is fine too), a string or a list of nonempty strings; numbers must be > 0.
`choices` maps a key to its allowed strings (`{"model": ("low", "high")}`). Anything else (a
check across keys, say) the plugin does itself and fails with
`arti.config.fail("<section>.<key>", "must be ...")`. A bad value stops startup with
`invalid configuration: <section>.<key> ...`, and so does a section no loaded plugin declared,
unless some plugin was skipped (disabled, failed, or missing a requirement): then such a section
is ignored, as it may be that plugin's. A section can be declared once, by one plugin. The core
sections (`agent`, `storage`, `plugins`, `providers`) are not a plugin's to declare.
`provider_keys` names keys whose value, unless `""`, must be a provider's name; they are checked
once every plugin is loaded (`approvals.provider`, say). A plugin can read the settings
of one it requires (`app.plugin_sections` says who declared what); prefer its own.

## Model providers: `api.provider(...)`

```python
api.provider(
    "anthropic",
    lambda app, options: AnthropicProvider(options),
    default_model="claude-opus-5-5",
    judge_model="claude-haiku-5-5",      # small and fast: the approval judge
    summary_model="claude-sonnet-5-5",   # summarises old history, reviews chats
    context_window=lambda model: 200_000,  # a conservative window for compaction
    options={"api_key_env": "", "base_url": "", "timeout": 0.0, "max_output_tokens": 0},
)
```

The name is what `agent.provider`, `approvals.provider`, `--provider` and `/model` accept.
`factory(app, options)` builds the client on first use: an `arti.providers.Provider`, whose
`await complete(messages, system=, tools=, model=, effort=, session_id=)` sends one request and
returns one `Reply` (it never runs tools or loops). `options` declares the keys of the optional
`[providers.<name>]` table with their defaults (here 0 is allowed and means "the default"); the
factory gets the table checked and filled in. A `base_url` must be https, or http on loopback,
whoever declares it. `arti.providers` and `arti.providers.http` hold what every provider shares:
`Reply`, `ProviderError` and `classify`, `api_key`, `deadline`, `number` and `retry_info`, the
replay carrier (`carrier`, `native_items`, `finish`) and the streaming HTTP client:
`http.stream(url, headers, payload, rate=None)`, where `rate(headers)` is the provider's own quota
parser, giving `{"status"?, "windows"?: {name: {"used", "resets_at"?}}}` for `/status`. With no
`agent.provider` set, the only provider loaded is used; with several, it must be set.

## The agent's channel

A plugin never talks to the chat platform. The agent's channel is always the link to the host
(`arti/link.py`), and every method takes a conversation key, which core never parses. What a
plugin may ask, through the App:

- `app.home` and `app.owner`: the owner's one-on-one key (`"home"`) and handle.
- `app.scope_for(key)`: the place a conversation lives in, `"home"` or `"chat-..."` (its files go
  in `contexts/<scope>/`); `app.person_of(key)`: whose it is (`""` for the owner's one-on-one);
  `app.owns(key)`: whether it is the owner's (trusted); `app.place_name(scope)`: a place's name.
- `await app.send(key, text)`: send text (its `MEDIA:` lines as files, each checked with
  `app.file_problem(key, path)`: readable, no raw tool output, a place's own files only, every
  `api.file_check`); `"sent"`, `"failed"` or `"unknown"`. `app.deliver(key, text)` also records
  it in the conversation (a scheduled result); `app.notify(key, text, at=None)` starts a turn
  there instead.

## Commands: `api.command(name, func, help)`

`python -m arti <name> [args...]` runs `func(app, args)` and exits with the number it returns;
`python -m arti --help` lists every command with its `help`. The name is lower-case words joined
by `-`. The config is loaded and checked and the plugins are registered, but the App is not
started: no lock, no store, no channel, no start hooks. A command that must not run beside a live
Arti takes the lock itself (`arti.lock.acquire(app.data_dir)`). Parse `args` with
`argparse.ArgumentParser(prog="python -m arti <name>")`. Commands run in the user's sandbox; the
operator runs one with `python -m host user exec ID python -m arti <name> [args...]`.

## Testing

`tests/support.py` builds a real App on a temporary data directory (`make_app`) with a scripted
model (`FakeProvider`, behind every provider name) and a recording channel. It has no gate and no
redactor unless the safety plugin's `register` is among its `plugins`. See `tests/plugins/`
for examples. Run the whole suite in the sandbox image with `python -m host test`.

## Writing a host channel

A chat platform is a package of the host, `channels/<name>/`, not a plugin. It runs on the
machine, in the host process, and imports only the core's small helpers (`arti.*`) and `host.*`,
never a plugin (tests/test_plugin.py checks). `host.channel` in `host.toml` picks it (`""`
picks the only one). None is included yet; see `host/channel.py` for the interface, and
`tests/host/test_host.py` for a channel at work (a fake one). The package has:

- `SETTINGS`: its section's defaults, read from `host.toml` under the channel's name and checked
  like any other section (a default's type is the rule).
- `make(host, settings) -> Channel`: builds the channel (a `host.channel.Channel`). Check the
  credentials it needs here (`arti.secret.get`, from the host's `.env`) and fail with
  `arti.config.fail(...)` when one is missing.

The `Channel` it returns:

- `name` (the channel's name, the key of users' handles) and `notes` (how to write for this
  platform, added to every agent's system prompt).
- `start(host)` and `stop()`; `alive()` (whether it still takes in messages: the host's hang
  watchdog asks).
- For each message, `await host.receive(Message(...))`: a unique and stable `event_id` (one
  delivered twice keeps it), the `chat` an answer goes to, the `sender`'s stable handle on the
  platform, and optionally `sender_name`, `text`, `files` (downloaded attachments; the host
  moves them into the user's home), `place` (a group chat's stable id; `""` for a one-on-one),
  `place_name` and `ts`. The host drops a message from anyone who is not a user.
- `home_chat(handle)`: the chat of a one-on-one with that person.
- `await send(chat, parts)`: deliver `arti.channel.Part`s in order: `Part("text", text)` as the
  model wrote it (format it for the platform, split long text) or `Part("file", path)` (a host
  path the host has already checked). Return `"sent"`, `"failed"` when provably nothing was
  sent, or `"unknown"`, and stop at the first part that failed or may have.
- `await typing(chat, on)`, if the platform has a typing indicator.

