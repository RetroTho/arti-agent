import types
import unittest

from plugins import delegate


class DelegateTest(unittest.TestCase):
    def test_runs_each_task_as_a_restricted_leaf(self):
        calls = []

        def run_headless(prompt, **options):
            calls.append((prompt, options))
            if "boom" in prompt:
                raise RuntimeError("leaf crashed")
            return "summary of " + prompt.rsplit("GOAL:\n", 1)[1].split("\n")[0]

        app = types.SimpleNamespace(
            services={
                "preload_skills": lambda app, names: "".join(
                    f"# Skill: {n}\nDo {n}." for n in names
                )
            }
        )
        ctx = types.SimpleNamespace(run_headless=run_headless, app=app, trusted=True)
        tasks = [{"goal": "find A", "context": "ctx A", "skills": ["s1"]}, {"goal": "boom"}]
        out = delegate.delegate_task({"tasks": tasks}, ctx)
        self.assertIn("### Task 1: find A\nsummary of find A", out)
        self.assertIn("### Task 2: boom\nFAILED: RuntimeError: leaf crashed", out)
        prompts = {p: o for p, o in calls}
        leaf = next(o for p, o in prompts.items() if "find A" in p)
        self.assertEqual(leaf["instructions"], "# Skill: s1\nDo s1.")
        for name in ("delegate_task", "cronjob", "memory", "skill_manage"):
            self.assertIn(name, leaf["exclude"])
        self.assertIn("CONTEXT:\nctx A", next(p for p in prompts if "find A" in p))
