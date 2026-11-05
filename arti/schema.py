"""A small JSON-schema validator for tool arguments.

Supports the subset tool schemas use: type, properties, required, additionalProperties, items,
enum, minimum/maximum, minLength/maxLength, minItems/maxItems and pattern. `check` rejects any
other keyword at registration, so a typo in a schema fails loudly instead of being ignored.
"""

import re

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}
# Every keyword a tool schema may use: the validated ones, and three that only document.
KEYWORDS = set(
    """
    type properties required additionalProperties items enum minimum maximum minLength
    maxLength minItems maxItems pattern description title default
    """.split()
)


def check(schema: dict) -> None:
    """Raise ValueError if `schema` uses a keyword, type or pattern this validator does not know."""
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    unknown = set(schema) - KEYWORDS
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")
    for name in _names(schema.get("type", [])):
        if name not in TYPES:
            raise ValueError(f"unknown schema type {name!r}")
    if "pattern" in schema:
        re.compile(schema["pattern"])
    subs = [
        *schema.get("properties", {}).values(),
        schema.get("items"),
        schema.get("additionalProperties"),
    ]
    for sub in subs:
        if isinstance(sub, dict):
            check(sub)


def _names(types) -> list[str]:
    return [types] if isinstance(types, str) else types


def _is(value, name: str) -> bool:
    # bool is a subclass of int in Python, but true is not a number in JSON.
    if isinstance(value, bool) and name in ("integer", "number"):
        return False
    return isinstance(value, TYPES[name])


def validate(value, schema: dict, path: str = "$") -> None:
    """Raise ValueError("<path>: <problem>") if `value` does not match `schema`."""
    if "type" in schema:
        names = _names(schema["type"])
        if not any(_is(value, name) for name in names):
            raise ValueError(f"{path}: expected {' or '.join(names)}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: must be <= {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path}: allows at most {schema['maxItems']} items")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(f"{path}.{name}: required")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                validate(item, properties[name], f"{path}.{name}")
            elif extra is False:
                raise ValueError(f"{path}.{name}: unexpected property")
            elif isinstance(extra, dict):
                validate(item, extra, f"{path}.{name}")
