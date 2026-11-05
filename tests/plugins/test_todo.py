import types
import unittest

from arti.tools import ToolError
from plugins import todo


class TodoTest(unittest.TestCase):
    def setUp(self):
        self.ctx = types.SimpleNamespace(state={})

    def call(self, **args):
        return todo.todo(args, self.ctx)

    def test_compaction_keeps_the_list(self):
        self.assertIsNone(todo.on_compact(self.ctx, []))
        self.call(todos=[{"id": "1", "content": "plan"}])
        self.assertEqual(
            todo.on_compact(self.ctx, []), "Current todo list:\nTodo (0/1 done):\n[ ] 1. plan"
        )

    def test_replace_merge_and_read(self):
        self.assertEqual(self.call(), "Todo list is empty.")
        self.call(todos=[{"id": "1", "content": "plan"}, {"id": "2", "content": "build"}])
        self.call(
            todos=[{"id": "1", "status": "completed"}, {"id": "3", "content": "ship"}], merge=True
        )
        out = self.call()
        self.assertIn("(1/3 done)", out)
        self.assertIn("[x] 1. plan", out)
        self.assertIn("[ ] 3. ship", out)
        self.call(todos=[{"id": "9", "content": "fresh"}])
        self.assertEqual([i["id"] for i in self.ctx.state["todo"]], ["9"])

    def test_errors_leave_list_alone(self):
        self.call(todos=[{"id": "1", "content": "a"}])
        with self.assertRaises(ToolError):
            self.call(todos=[{"id": "2"}], merge=True)  # a new item needs content
        with self.assertRaises(ToolError):
            self.call(todos=[{"id": "x", "content": "a"}, {"id": "x", "content": "b"}])
        with self.assertRaises(ToolError):
            self.call(todos=[{"id": str(n), "content": "a"} for n in range(51)])
        self.assertEqual(len(self.ctx.state["todo"]), 1)
