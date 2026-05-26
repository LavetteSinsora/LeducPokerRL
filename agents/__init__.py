"""Public agent package surface for the reorganized PokerRL repo."""

from .base import BaseAgent

__all__ = ["BaseAgent", "AgentMetadata", "registry"]


class _RegistryProxy:
    """Delay importing the concrete registry until it is actually used."""

    def __getattr__(self, name):
        from .registry import registry as concrete_registry

        return getattr(concrete_registry, name)


registry = _RegistryProxy()


def __getattr__(name):
    if name == "AgentMetadata":
        from .registry import AgentMetadata

        return AgentMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
