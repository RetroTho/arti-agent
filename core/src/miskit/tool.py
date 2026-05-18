from miskit.registry import load_module


class Tool:
    @classmethod
    def load(cls, config, services=None):
        name = config.get("name")
        if not name:
            raise ValueError("tool.name is required")
        module = load_module(
            name,
            group="miskit.tools",
            builtins_prefix="miskit.tools",
            kind="tool",
        )
        if not hasattr(module, "create_tool"):
            raise ValueError(f"tool must define create_tool: {name}")
        return module.create_tool(config, services=services)

    @classmethod
    def load_all(cls, configs, services=None):
        return [cls.load(config, services=services) for config in configs]

    def run(self, arguments):
        raise NotImplementedError("tools must define run")
