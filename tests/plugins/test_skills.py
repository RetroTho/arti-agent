import tempfile
import types
import unittest
from pathlib import Path

from arti.tools import ToolError
from plugins import skills

SKILL = '---\nname: {name}\ndescription: "Use when {name}."\n---\nDo the thing.\n'


class SkillsTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        app = types.SimpleNamespace(data_dir=self.dir)
        self.ctx = types.SimpleNamespace(app=app, context_dir=self.dir / "contexts" / "chat-1")

    def manage(self, action, name, **args):
        return skills.skill_manage({"action": action, "name": name, **args}, self.ctx)

    def view(self, name, **args):
        return skills.skill_view({"name": name, **args}, self.ctx)

    def test_frontmatter(self):
        meta, body = skills.parse_frontmatter(
            "---\nname: a\ndescription: 'x: y'\nmeta:\n  k: v\n---\nbody"
        )
        self.assertEqual(meta, {"name": "a", "description": "x: y", "meta": ""})
        self.assertEqual(body, "body")
        self.assertEqual(skills.parse_frontmatter("no frontmatter"), ({}, "no frontmatter"))

    def test_create_view_patch_files_delete(self):
        self.manage("create", "brief", content=SKILL.format(name="brief"), category="daily")
        self.assertTrue((self.dir / "skills" / "daily" / "brief" / "SKILL.md").is_file())
        self.manage("write_file", "brief", file_path="scripts/run.py", content="print(1)")
        text = self.view("brief")
        self.assertIn("Do the thing.", text)
        self.assertIn("scripts/run.py", text)
        self.assertEqual(self.view("brief", file_path="scripts/run.py"), "print(1)")
        self.manage("patch", "brief", old_string="the thing", new_string="it well")
        self.assertIn("Do it well.", self.view("brief"))
        self.manage("remove_file", "brief", file_path="scripts/run.py")
        self.assertNotIn("scripts/run.py", self.view("brief"))
        self.assertIn("Do it well", skills.load_skill(self.ctx.app, "brief"))
        preloaded = skills.preload(self.ctx.app, ["brief", "nope"])
        self.assertTrue(preloaded.startswith("# Skill: brief\n"))
        self.assertIn("(skill 'nope' not found)", preloaded)
        self.manage("delete", "brief")
        self.assertIsNone(skills.load_skill(self.ctx.app, "brief"))

    def test_compaction_keeps_the_names_of_skills_loaded(self):
        conv = types.SimpleNamespace(state={"skills_seen": ["old"]})
        view = {"type": "tool_use", "id": "1", "name": "skill_view", "input": {"name": "brief"}}
        other = {"type": "tool_use", "id": "2", "name": "todo", "input": {"name": "x"}}
        span = [{"role": "assistant", "content": [view, other]}]
        text = skills.on_compact(conv, span)
        self.assertEqual(
            text, "Skills loaded earlier (call skill_view again before following one): brief, old"
        )
        self.assertEqual(conv.state["skills_seen"], ["brief", "old"])  # kept for the next pass
        self.assertIsNone(skills.on_compact(types.SimpleNamespace(state={}), []))

    def test_validation(self):
        with self.assertRaises(ToolError):  # name mismatch
            self.manage("create", "one", content=SKILL.format(name="two"))
        with self.assertRaises(ToolError):  # empty body
            self.manage("create", "one", content="---\nname: one\ndescription: d\n---\n")
        with self.assertRaises(ToolError):
            self.manage(
                "create", "one", content=f"---\nname: one\ndescription: {'x' * 401}\n---\nb"
            )
        with self.assertRaises(ToolError):
            self.manage("create", "Bad Name", content=SKILL.format(name="Bad Name"))
        self.manage("create", "one", content=SKILL.format(name="one"))
        with self.assertRaises(ToolError):  # a patch may not break SKILL.md
            self.manage("patch", "one", old_string="name: one", new_string="name: other")
        with self.assertRaises(ToolError):
            self.manage("create", "one", content=SKILL.format(name="one"))

    def test_paths_are_confined(self):
        self.manage("create", "one", content=SKILL.format(name="one"))
        for bad in ("../x.md", "scripts/../../x", "other/x", "scripts", "/etc/passwd"):
            with self.assertRaises(ToolError, msg=bad):
                self.manage("write_file", "one", file_path=bad, content="x")
        outside = self.dir / "outside"
        outside.mkdir()
        (self.dir / "skills" / "one" / "assets").symlink_to(outside)
        with self.assertRaises(ToolError):
            self.manage("write_file", "one", file_path="assets/x", content="x")

    def test_index_skips_hidden_and_caches(self):
        self.manage("create", "one", content=SKILL.format(name="one"))
        archived = self.dir / "skills" / ".archive" / "old"
        archived.mkdir(parents=True)
        (archived / "SKILL.md").write_text(SKILL.format(name="old"))
        info = types.SimpleNamespace(app=self.ctx.app, trusted=True)
        section = skills.prompt_section(info)
        guest = types.SimpleNamespace(app=self.ctx.app, trusted=False)
        self.assertIsNone(skills.prompt_section(guest))  # skills are the owner's
        self.assertIn("- one: Use when one.", section)
        self.assertNotIn("old", section)
        self.manage("patch", "one", old_string="Use when one.", new_string="Use when two.")
        self.assertIn("Use when two.", skills.prompt_section(info))

    def test_a_stray_skills_root_file_is_not_a_skill(self):
        self.manage("create", "one", content=SKILL.format(name="one"))
        (self.dir / "skills" / "SKILL.md").write_text(SKILL.format(name="root"))
        with self.assertRaises(ToolError):  # deleting it would remove every skill
            self.manage("delete", "root")
        self.assertIn("Do the thing.", self.view("one"))
