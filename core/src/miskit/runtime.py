from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from miskit.channel import Channel
from miskit.compaction import Compactor
from miskit.config import Config
from miskit.cron import CronLog
from miskit.cron import CronService
from miskit.dream import Dream
from miskit.heartbeat import HEARTBEAT_JOB_ID, HEARTBEAT_QUIET_REPLY, HeartbeatLog, HeartbeatTasks
from miskit.history import History
from miskit.image import ImageStore
from miskit.memory import Memory
from miskit.message import Message
from miskit.provider import Model
from miskit.runner import Runner
from miskit.runner import DEFAULT_MAX_TOOL_ROUNDS
from miskit.tool import Tool
from miskit.truncation import TruncationStore


DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 1800


@dataclass
class Runtime:
    config: Config
    instance: Path
    runner: Runner
    cron: CronService
    image_store: ImageStore


def max_tool_rounds_from_config(context):
    """How many tool rounds to allow per user message; set context.maxToolRounds in config.json."""
    raw = context.get("maxToolRounds", DEFAULT_MAX_TOOL_ROUNDS)
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("context.maxToolRounds must be an integer") from error
    if value < 1:
        raise ValueError("context.maxToolRounds must be at least 1")
    return value


def build_runner(config, instance, services=None):
    instance = Path(instance).expanduser()
    memory = Memory(instance / "memory")
    history = History(instance / "history")
    image_store = ImageStore(instance / "images")
    workspace_config = config.section("workspace")
    workspace_path = workspace_config.get("path")
    workspace = Path(workspace_path).expanduser().resolve() if workspace_path else instance / "workspace"
    memory.setup()
    history.setup()
    image_store.setup()
    workspace.mkdir(parents=True, exist_ok=True)
    context_config = config.section("context")
    provider_config = dict(config.section("provider"))
    provider_config["maxTokens"] = context_config.get("outputTokens", 2000)

    truncation_store = TruncationStore(instance / "truncated")
    truncation_store.setup()

    services = dict(services or {})
    services.setdefault("memory", memory)
    services.setdefault("image_store", image_store)
    services.setdefault("context_tokens", context_config.get("tokens", 8000))
    services.setdefault("heartbeat_path", heartbeat_path(instance))
    services.setdefault("heartbeat_log", HeartbeatLog(heartbeat_log_path(instance)))
    services.setdefault("cron_log", CronLog(cron_log_path(instance)))
    services.setdefault("workspace", workspace)
    services.setdefault("truncation_store", truncation_store)
    services.setdefault(
        "restrict_to_workspace",
        config.section("workspace").get("restrictToWorkspace", True),
    )
    services.setdefault(
        "run_as_user",
        config.section("security").get("runAsUser"),
    )

    model = Model.load(provider_config)
    compactor = Compactor.from_config(context_config)
    tools = Tool.load_all(config.entries("tools"), services=services)
    dream = Dream(
        model,
        memory,
        instance / "history" / "archive",
        instance / "dream.json",
        max_chunk_chars=context_config.get("tokens", 8000) * 5 // 2,
    )
    return Runner(
        model,
        memory=memory,
        history=history,
        tools=tools,
        compactor=compactor,
        dream=dream,
        max_tool_rounds=max_tool_rounds_from_config(context_config),
        truncation_store=truncation_store,
    )


def build_channel(config, image_store=None):
    return Channel.load(config.section("channel"), image_store=image_store)


def channel_name(config):
    return str(config.section("channel").get("name", "")).strip()


def build_cron(instance):
    instance = Path(instance).expanduser()
    return CronService(instance / "cron" / "jobs.json")


def heartbeat_path(instance):
    return Path(instance).expanduser() / "cron" / "heartbeat.md"


def heartbeat_log_path(instance):
    return Path(instance).expanduser() / "cron" / "heartbeats.jsonl"


def cron_log_path(instance):
    return Path(instance).expanduser() / "cron" / "cron.jsonl"


def setup_heartbeat_cron(config, instance, cron):
    section = config.section("heartbeat")
    if not section.get("enabled", False):
        cron.remove_job(HEARTBEAT_JOB_ID)
        return None

    try:
        interval_seconds = int(section.get("intervalSeconds", DEFAULT_HEARTBEAT_INTERVAL_SECONDS))
    except (TypeError, ValueError) as error:
        raise ValueError("heartbeat.intervalSeconds must be a positive integer") from error

    if interval_seconds <= 0:
        raise ValueError("heartbeat.intervalSeconds must be a positive integer")

    heartbeat = HeartbeatTasks(heartbeat_path(instance))
    message = heartbeat.read().strip()

    if not message:
        cron.remove_job(HEARTBEAT_JOB_ID)
        return None

    return cron.add_recurring_job(
        name="heartbeat",
        every_seconds=interval_seconds,
        message=message,
        job_id=HEARTBEAT_JOB_ID,
    )


def connect_cron(cron, runner, write=print, channel=None, heartbeat_file=None, heartbeat_log=None, cron_log=None):
    async def on_cron_job(job):
        content = cron_job_prompt(job, heartbeat_file=heartbeat_file)
        if not content:
            return

        snapshot = len(runner.conversation.messages)
        reply = await runner.chat(content)

        if job.id != HEARTBEAT_JOB_ID:
            timestamp = datetime.now().astimezone().isoformat()
            turn_messages = runner.conversation.messages[snapshot:]
            runner.rollback_conversation(snapshot)

            if cron_log is not None:
                cron_log.log(timestamp, job.id, job.name, turn_messages)

            marker = Message(
                "user",
                f"[Cron {job.name} {timestamp}] "
                f"Use cron(action=\"history\") to review the full details.",
            )
            runner.conversation.add(marker)
            runner.conversation.add(reply)
            runner.log_conversation()

            if channel is None:
                write(f"Cron: {reply.content}")
            else:
                await channel.send(reply.content)
            return

        quiet = heartbeat_reply_is_quiet(reply.content)
        timestamp = datetime.now().astimezone().isoformat()
        turn_messages = runner.conversation.messages[snapshot:]

        if heartbeat_log is not None:
            heartbeat_log.log(timestamp, turn_messages, quiet)

        runner.rollback_conversation(snapshot)

        if not quiet:
            marker = Message(
                "user",
                f"[Heartbeat {timestamp}] You responded to the user during this heartbeat. "
                f"Use heartbeat(action=\"history\") to review the full details.",
            )
            runner.conversation.add(marker)
            runner.conversation.add(reply)
            runner.log_conversation()

            if channel is None:
                write(f"Cron: {reply.content}")
            else:
                await channel.send(reply.content)

    cron.on_job = on_cron_job


def cron_job_prompt(job, heartbeat_file=None):
    message = job.message
    if job.id == HEARTBEAT_JOB_ID and heartbeat_file is not None:
        message = heartbeat_file.read().strip()

    if not message:
        return ""

    lines = [
        "[Cron Job Triggered]",
        f"Name: {job.name}",
        f"Scheduled For: {job.at.isoformat()}",
    ]
    if job.every_seconds is not None:
        lines.append(f"Repeats Every: {job.every_seconds} seconds")
    lines.append(f"Message: {message}")
    return "\n".join(lines)


def heartbeat_reply_is_quiet(content):
    text = content.strip()
    return not text or text.upper() == HEARTBEAT_QUIET_REPLY


def build_runtime(instance=None, write=print):
    if instance is None:
        instance = Config.instance

    instance = Path(instance).expanduser()
    path = instance / "config.json"

    if not path.exists():
        Config.write_default(path)
        write(f"Created {path}. Add your API key, then run miskit again.")
        return None

    config = Config.load(path)
    cron = build_cron(instance)
    image_store = ImageStore(instance / "images")
    image_store.setup()
    setup_heartbeat_cron(config, instance, cron)
    runner = build_runner(config, instance, services={"cron": cron, "image_store": image_store})
    return Runtime(config, instance, runner, cron, image_store)
