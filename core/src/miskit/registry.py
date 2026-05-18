from importlib import import_module
from importlib.metadata import entry_points


def load_module(name, *, group, builtins_prefix, kind, allow_direct=False):
    name = str(name or "").strip()
    if not name:
        raise ValueError(f"{kind}.name is required")

    entry_point = _find_entry_point(group, name)
    if entry_point is not None:
        return entry_point.load()

    for module_name in _module_candidates(name, builtins_prefix, allow_direct):
        module = _try_import(module_name)
        if module is not None:
            return module

    raise ValueError(f"unknown {kind}: {name}")


def _find_entry_point(group, name):
    for ep in entry_points(group=group):
        if ep.name == name:
            return ep
    return None


def _module_candidates(name, builtins_prefix, allow_direct):
    if "." in name or allow_direct:
        yield name
    if "." not in name:
        yield f"{builtins_prefix}.{name}"


def _try_import(module_name):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            return None
        raise
