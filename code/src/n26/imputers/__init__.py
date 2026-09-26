"""Imputer registry."""
from __future__ import annotations
from typing import Callable, Type

from n26.imputers.base import Imputer

_IMPUTERS: dict[str, Type[Imputer]] = {}


def register_imputer(name: str) -> Callable[[Type[Imputer]], Type[Imputer]]:
    """Decorator: register an Imputer subclass under `name`."""

    def _wrap(cls: Type[Imputer]) -> Type[Imputer]:
        if name in _IMPUTERS:
            raise ValueError(
                f"Imputer '{name}' already registered by "
                f"{_IMPUTERS[name].__qualname__}"
            )
        cls.name = name
        _IMPUTERS[name] = cls
        return cls

    return _wrap


def get_imputer(name: str, **kwargs) -> Imputer:
    """Instantiate the imputer registered under `name`."""
    if name not in _IMPUTERS:
        raise KeyError(f"No imputer named '{name}'. Known: {sorted(_IMPUTERS)}")
    return _IMPUTERS[name](**kwargs)


def list_imputers() -> list[str]:
    return sorted(_IMPUTERS)


__all__ = ["Imputer", "register_imputer", "get_imputer", "list_imputers"]


from n26.imputers import (  # noqa: E402, F401
    cfmi, diffputer, mean, mice, miri, missforest, native, tabcsdi, zeros,
)
