from miskit.registry import load_module


class Channel:
    @classmethod
    def load(cls, config, image_store=None):
        name = config.get("name")
        if not name:
            raise ValueError("channel.name is required")
        module = load_module(
            name,
            group="miskit.channels",
            builtins_prefix="miskit.channels",
            kind="channel",
        )
        if not hasattr(module, "create_channel"):
            raise ValueError(f"channel must define create_channel: {name}")
        return module.create_channel(config, image_store=image_store)

    async def run(self, runner):
        raise NotImplementedError("channels must define run")

    async def send(self, content):
        raise NotImplementedError("channels must define send")
