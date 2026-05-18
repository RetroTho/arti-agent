from miskit.registry import load_module


class Model:
    @classmethod
    def load(cls, config):
        name = config.get("name")
        if not name:
            raise ValueError("provider.name is required")
        provider = load_provider(name)
        return provider.create_model(config)

    async def complete(self, messages, tools=None):
        raise NotImplementedError("models must define complete")


class Provider:
    def create_model(self, config):
        raise NotImplementedError("providers must define create_model")


def load_provider(name):
    module = load_module(
        name,
        group="miskit.providers",
        builtins_prefix="miskit.providers",
        kind="provider",
        allow_direct=True,
    )
    provider = module
    if hasattr(module, "provider"):
        provider = module.provider
    if not hasattr(provider, "create_model"):
        raise ValueError(f"provider must define create_model: {name}")
    return provider
