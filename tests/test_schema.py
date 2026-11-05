import unittest

from arti import schema

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "pattern": "^[a-z]+$"},
        "count": {"type": "integer", "minimum": 0, "maximum": 5},
        "mode": {"type": "string", "enum": ["a", "b"]},
        "items": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


class SchemaTest(unittest.TestCase):
    def test_valid(self):
        schema.check(SCHEMA)
        schema.validate({"name": "ok", "count": 3, "mode": "a", "items": [{"id": "x"}]}, SCHEMA)

    def test_errors_name_the_path(self):
        cases = [
            ({}, "$.name: required"),
            ({"name": "ok", "extra": 1}, "$.extra: unexpected"),
            ({"name": "ok", "count": True}, "$.count: expected integer"),
            ({"name": "ok", "count": 9}, "$.count: must be <="),
            ({"name": "OK"}, "$.name: does not match"),
            ({"name": "ok", "mode": "c"}, "$.mode: must be one of"),
            ({"name": "ok", "items": [{}]}, "$.items[0].id: required"),
            ({"name": "ok", "items": [{"id": "1"}] * 3}, "$.items: allows at most 2"),
            ("text", "$: expected object"),
        ]
        for value, message in cases:
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                schema.validate(value, SCHEMA)
            self.assertIn(message, str(caught.exception))

    def test_check_rejects_unknown_keywords_types_and_patterns(self):
        for bad in (
            {"type": "object", "oneOf": []},
            {"type": "thing"},
            {"type": "string", "pattern": "("},
        ):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                schema.check(bad)


if __name__ == "__main__":
    unittest.main()
