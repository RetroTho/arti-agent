"""run_turn: the agent loop, the heart of Arti.

A turn answers the input waiting in a conversation. It repeats these steps until the model gives
a reply with no tool calls:

  1. Move the waiting input (messages, notices, notes) into history as one user message.
  2. Ask the model, with the system prompt, the history and the tools on offer.
  3. Store the reply in history *before* acting on it, then send its text.
  4. Run its tool calls one at a time; each result is stored as soon as it is known.

Input that arrives while a turn runs joins it at step 1 of the next round ("steering"). If
something from a person (a message, reaction, edit or vote) arrived after the model was asked, the
tool calls of that reply that have not run yet are skipped, so a correction takes effect before
anything else happens; host notices (a job finished, a wait ended) never skip calls. Every tool
call always gets a result, even when the turn is stopped or fails, because providers reject a
history with an unanswered call. A final reply of [SILENT] sends nothing.
"""

import asyncio
import logging

from arti import context, prompt, tools
from arti.providers import ProviderError

log = logging.getLogger(__name__)

SILENT_MARKERS = ("[SILENT]", "SILENT", "NO_REPLY")  # a whole reply that sends nothing (is_silent)
MAX_NUDGES = 2  # times the model is reminded to answer a message it left unanswered
RETRY_DELAYS = (2, 10, 30)  # seconds before each retry of a retryable error; then the turn fails
MAX_RETRY_AFTER = 60  # a provider asking us to wait longer than this is not retried
PROMPT_TIMEOUT = 30  # seconds to build the system prompt (plugins read files)
NUDGE = (
    "[Arti: you have not answered {name}'s latest message, and nothing was sent. Reply to it "
    "now; stay silent only if it truly needs no answer.]"
)


class TurnError(Exception):
    """A turn cannot go on; the message is shown to the user."""


class SendFailed(TurnError):
    """The channel could not deliver the reply's text."""


async def run_turn(conv) -> str:
    """Run one turn of `conv` and return the final reply text ("" if the turn failed)."""
    conv.begin_turn()
    try:
        return await _loop(conv)
    except asyncio.CancelledError:
        repair_open_calls(conv, "the task was stopped")
        raise
    except Exception as error:
        if not isinstance(error, (TurnError, ProviderError)):
            log.exception("turn failed in %s", conv.key or "a headless run")
        repair_open_calls(conv, str(error) if isinstance(error, SendFailed) else "the turn failed")
        if conv.headless:
            raise
        await _report_failure(conv, error)
        return ""
    finally:
        conv.end_turn()


async def _loop(conv) -> str:
    steps = conv.app.config["agent"]["max_iterations"]
    nudges = 0
    for _ in range(steps):
        await take_input(conv)
        reply = await ask_model(conv)
        conv.append(assistant_message(reply))

        text = reply.text.strip()
        if text and not conv.headless and not is_silent(text):
            await send_reply(conv, text)

        if reply.tool_calls:  # (providers drop the calls of a cut-off reply)
            await run_calls(conv, reply.tool_calls)
            continue
        if conv.unanswered and nudges < MAX_NUDGES and not conv.headless:
            nudges += 1
            conv.note(NUDGE.format(name=conv.person_name))
            continue
        return text
    raise TurnError(f"I stopped after {steps} steps.")


async def take_input(conv) -> None:
    """Append the waiting inputs and host notes as one user message (nothing if none)."""
    blocks, seqs = await conv.drain()
    if blocks:
        conv.append({"role": "user", "content": blocks}, done=seqs)


async def ask_model(conv):
    """One model request: compacting history first if it is near the context window, with a
    deadline. A retryable error (overloaded, a server error, a provider timeout) is retried after
    each of RETRY_DELAYS (or the provider's retry_after, if longer); "context too long" compacts
    history once and asks again."""
    provider, model, effort = conv.route
    offered = tools.tools_for(conv)
    conv.offered = {tool.name: tool for tool in offered}
    specs = tools.specs(offered)
    system = conv.system or await build_prompt(conv)
    if await context.maybe_compact(conv, system, specs):
        system = await build_prompt(conv)
    conv.typing(True)
    timeout = conv.app.config["agent"]["request_timeout"]
    delays, compacted = iter(RETRY_DELAYS), False
    while True:
        messages = await asyncio.to_thread(context.for_request, conv)
        try:
            async with asyncio.timeout(timeout):
                reply = await conv.app.provider(provider).complete(
                    messages,
                    system=system,
                    tools=specs,
                    model=model,
                    effort=effort,
                    session_id=conv.session_id,
                )
            conv.rate_limit = reply.rate_limit or conv.rate_limit
            return reply
        except TimeoutError:
            raise TurnError(f"The model did not answer within {timeout:.0f} seconds.") from None
        except ProviderError as error:
            if error.code == "context_length" and not compacted:
                compacted = True
                if not await context.maybe_compact(conv, system, specs, force=True):
                    raise
                system = await build_prompt(conv)
                continue
            asked = error.retry_after or 0  # the provider's own retry-after, if it gave one
            delay = next(delays, None) if error.retryable and asked <= MAX_RETRY_AFTER else None
            if delay is None:  # not retryable, or retried RETRY_DELAYS times already
                raise
            delay = max(delay, asked)
            log.warning("model request failed (%s); retrying in %.0f s", error.code, delay)
            await asyncio.sleep(delay)


async def build_prompt(conv) -> str:
    """Build the system prompt and keep it for the rest of the conversation (chat.py says why)."""
    conv.system = await asyncio.wait_for(asyncio.to_thread(prompt.build, conv), PROMPT_TIMEOUT)
    return conv.system


def assistant_message(reply) -> dict:
    """The reply as a history message. `native` is the provider's own replay data; the
    provider checks it still belongs to the route before using it."""
    message = {"role": "assistant", "content": reply.content}
    if reply.native:
        message["native"] = reply.native
    return message


async def send_reply(conv, text: str) -> None:
    """Send the reply's text; a failed send ends the turn, an unknown one is told to the model."""
    status = await conv.send(text)
    if status == "failed":
        raise SendFailed("your message failed to send")
    if status == "unknown":
        conv.note("[Arti: your last message may or may not have been delivered.]", unknown=True)


async def run_calls(conv, calls: list[dict]) -> None:
    """Run the reply's tool calls one at a time, storing each result as soon as it is known.
    Once a person has said something the model did not see, the remaining calls are skipped."""
    for call in calls:
        if conv.person_spoke_since_request():
            result = tools.error_result(
                call["id"], "[Arti: not run: something new arrived in the chat; reconsider]"
            )
        else:
            # Left set if the turn is cancelled, so repair knows which call was running.
            conv.running_call, conv.running_tool = call["id"], call["name"]
            result = await tools.run_tool(conv, call)
            conv.running_call = conv.running_tool = ""
        conv.append({"role": "user", "content": [result]})


def is_silent(text: str) -> bool:
    """True when the reply asks to send nothing. The prompt asks for exactly "[SILENT]", but
    models drift, so these all count as silence:

    - a whole reply (stripped) that is one of SILENT_MARKERS: "[SILENT]", "SILENT" or
      "NO_REPLY" (case matters: "silent" is a word, not a marker);
    - a reply that starts with "[SILENT]" (e.g. "[SILENT] nothing new today"); and
    - a reply that ends with "[SILENT]", even after other text ("ok [SILENT]"): the model
      thought aloud and then chose silence, so none of it is sent.

    The bare words count only as the whole reply, so "Keep it SILENT" is sent."""
    text = text.strip()
    return text in SILENT_MARKERS or text.startswith("[SILENT]") or text.endswith("[SILENT]")


def open_calls(messages: list[dict]) -> list[dict]:
    """The tool_use blocks of the last assistant message that have no tool_result yet."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            answered = {
                block["tool_use_id"]
                for later in messages[index + 1 :]
                for block in later["content"]
                if block["type"] == "tool_result"
            }
            return [
                block
                for block in messages[index]["content"]
                if block["type"] == "tool_use" and block["id"] not in answered
            ]
    return []


def repair_open_calls(conv, why: str) -> None:
    """Give every unanswered tool call a result: unknown for the one that was running."""
    results = [
        tools.unknown_result(call["id"], f"[Arti: {why} while this ran; its outcome is unknown]")
        if call["id"] == conv.running_call
        else tools.error_result(call["id"], f"[Arti: not run: {why}]")
        for call in open_calls(conv.messages)
    ]
    if results:
        conv.append({"role": "user", "content": results})


async def _report_failure(conv, error: Exception) -> None:
    """Tell the user briefly, and leave a host note in history (never a second assistant reply)."""
    text = error.message if isinstance(error, ProviderError) else str(error)
    text = text or type(error).__name__
    conv.last_error = text
    conv.note(f"[Arti: the previous turn failed: {text}]")
    if not isinstance(error, SendFailed):
        await conv.send(f"Sorry, I hit a problem and stopped: {text}")
