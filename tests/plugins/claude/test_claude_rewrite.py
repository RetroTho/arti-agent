import copy
import json
import unittest

from plugins.claude.rewrite import (
    frame_first,
    pin_breakpoint,
    restore_tool_results,
    rewrite,
)

MARKER = {"type": "ephemeral"}
RESULT = {
    "type": "tool_result",
    "tool_use_id": "t1",
    "content": [{"type": "text", "text": "output"}],
}
ASSISTANT = {
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "t1", "name": "probe", "input": {}}],
}


def turn(content):
    return [ASSISTANT, {"role": "user", "content": content}]


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


class RestoreTest(unittest.TestCase):
    def check(self, native, queried=(RESULT,)):
        messages = turn(copy.deepcopy(native))
        before = copy.deepcopy(messages)
        return restore_tool_results(messages, list(queried)), messages, before

    def test_equivalent_shapes_restore_the_host_blocks(self):
        for native in ({**RESULT, "content": "output"}, {**RESULT, "is_error": False}):
            ok, messages, _ = self.check([native])
            self.assertIn(ok, ("restored", "same"))
            self.assertEqual(
                messages[1]["content"],
                [{**RESULT, **({"is_error": False} if "is_error" in native else {})}],
            )

    def test_appended_text_moves_to_the_end_of_the_turn(self):
        for content in (
            "output\n<reminder>",
            [{"type": "text", "text": "output"}, {"type": "text", "text": "\n<reminder>"}],
        ):
            ok, messages, _ = self.check(
                [{**RESULT, "content": content}, {"type": "text", "text": "steer"}],
                [RESULT, {"type": "text", "text": "steer"}],
            )
            self.assertEqual(ok, "restored")
            self.assertEqual(
                messages[1]["content"],
                [
                    RESULT,
                    {"type": "text", "text": "steer"},
                    {"type": "text", "text": "\n<reminder>"},
                ],
            )

    def test_structured_host_content_survives_trailing_text(self):
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
        }
        host = {**RESULT, "content": [{"type": "text", "text": "output"}, image]}
        ok, messages, _ = self.check(
            [{**host, "content": [*host["content"], {"type": "text", "text": "note"}]}], [host]
        )
        self.assertEqual(ok, "restored")
        self.assertEqual(messages[1]["content"][0], host)

    def test_unprovable_differences_change_nothing(self):
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
        }
        cases = [
            [{**RESULT, "is_error": True}],
            [{**RESULT, "is_error": 0}],
            [{**RESULT, "content": "changed"}],
            [{**RESULT, "tool_use_id": "wrong"}],
            [RESULT, {"type": "tool_result", "tool_use_id": "extra", "content": "x"}],
            [{**RESULT, "content": [{"type": "text", "text": "output"}, image]}],
            [
                {
                    **RESULT,
                    "content": [
                        {"type": "text", "text": "output"},
                        {"type": "text", "text": "n", "cache_control": MARKER},
                    ],
                }
            ],
            [{**RESULT, "content": [{"type": "text", "text": "out native insertion put"}]}],
        ]
        for native in cases:
            ok, messages, before = self.check(native)
            self.assertEqual(ok, "unrestored", native)
            self.assertEqual(messages, before)

    def test_plain_user_frame_is_out_of_scope(self):
        raw = json.dumps(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        ).encode()
        self.assertEqual(rewrite(raw, [{"type": "text", "text": "hi"}]), (raw, ""))

    def test_unparsable_body_is_forwarded_as_is(self):
        self.assertEqual(rewrite(b"not json", [RESULT]), (b"not json", "unrestored"))


REMINDER = {"type": "text", "text": "<system-reminder>email a@example.invalid</system-reminder>"}
FRAME = {"type": "text", "text": "run `echo alpha`" + " source pack" * 50}


class FrameFirstTest(unittest.TestCase):
    def test_first_turn_puts_the_frame_ahead_of_cli_context(self):
        messages = [
            {"role": "user", "content": [REMINDER, FRAME]},
            {
                "role": "system",
                "content": [{"type": "text", "text": "env", "cache_control": MARKER}],
            },
        ]
        raw = json.dumps({"messages": messages}).encode()
        out = json.loads(rewrite(raw, [FRAME])[0])
        first = out["messages"][0]["content"]
        self.assertEqual(plain(first), [FRAME, REMINDER])
        self.assertEqual(first[0]["cache_control"], MARKER)  # pinned on the frame
        self.assertNotIn("cache_control", out["messages"][1]["content"][0])

    def test_only_reordered_when_unambiguous(self):
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
        }
        cases = [
            [ASSISTANT, {"role": "user", "content": [REMINDER, FRAME]}],
            [{"role": "user", "content": [REMINDER, FRAME, FRAME]}],
            [{"role": "user", "content": [image, FRAME]}],
            [{"role": "user", "content": [REMINDER, {"type": "text", "text": "rewritten"}]}],
        ]
        for messages in cases:
            before = copy.deepcopy(messages)
            self.assertFalse(frame_first(messages, [FRAME]))
            self.assertEqual(messages, before)


def results(n):
    return [
        {"type": "tool_result", "tool_use_id": f"r{n}c{k}", "content": f"output {n}.{k}"}
        for k in range(4)
    ]


def assistant(n):
    calls = [
        {"type": "tool_use", "id": f"r{n}c{k}", "name": "mcp__arti__t", "input": {}}
        for k in range(4)
    ]
    return {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "p", "signature": "s"}, *calls],
    }


def native_request(rounds, reminder):
    """What the CLI 2.1.287+ sends: a mark on the last tool_use and one on a trailing system
    message."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "run four probes"}]}]
    for n in range(rounds):
        messages += [assistant(n), {"role": "user", "content": results(n)}]
    messages[-2]["content"][-1]["cache_control"] = MARKER
    if reminder:
        messages[-1]["content"][-1]["content"] += "\n<system-reminder>x</system-reminder>"
    messages.append(
        {"role": "system", "content": [{"type": "text", "text": "date", "cache_control": MARKER}]}
    )
    return {"messages": messages}


def marks(body):
    return [
        (i, j)
        for i, m in enumerate(body["messages"])
        for j, b in enumerate(m["content"])
        if "cache_control" in b
    ]


class BreakpointTest(unittest.TestCase):
    def test_two_cli_marks_keep_one_breakpoint_the_next_request_reads(self):
        for reminder in (False, True):
            native = native_request(1, reminder)
            sent = json.loads(rewrite(json.dumps(native).encode(), results(0))[0])
            self.assertEqual(marks(native), [(1, 4), (3, 0)])
            self.assertEqual(marks(sent), [(1, 4), (2, 3)])
            following = json.loads(
                rewrite(json.dumps(native_request(2, reminder)).encode(), results(1))[0]
            )
            # the next request replays this one's prefix through its last breakpoint
            self.assertEqual(plain(following["messages"][:2]), plain(sent["messages"][:2]))
            self.assertEqual(plain(following["messages"][2]["content"]), plain(results(0)))

    def test_unrestorable_turn_falls_back_to_the_cli_tool_use_mark(self):
        native = native_request(1, reminder=True)
        messages = copy.deepcopy(native["messages"])
        self.assertTrue(pin_breakpoint(messages, results(0)))
        self.assertEqual(marks({"messages": messages}), [(1, 4)])
        self.assertEqual(plain(messages), plain(native["messages"]))

    def test_marks_inside_the_recurring_span_stay(self):
        native = native_request(1, reminder=False)
        native["messages"].pop()
        native["messages"][2]["content"][-1]["cache_control"] = MARKER
        raw = json.dumps(native).encode()
        self.assertEqual(rewrite(raw, results(0))[0], raw)


if __name__ == "__main__":
    unittest.main()
