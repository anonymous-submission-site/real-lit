"""Dataset registry."""
from __future__ import annotations
from typing import Callable

from n26.data.base import Dataset

_LOADERS: dict[str, Callable[[], Dataset]] = {}


def register_dataset(name: str) -> Callable[[Callable[[], Dataset]], Callable[[], Dataset]]:
    """Decorator: register a no-arg loader function under `name`."""

    def _wrap(fn: Callable[[], Dataset]) -> Callable[[], Dataset]:
        if name in _LOADERS:
            raise ValueError(
                f"Dataset '{name}' already registered by {_LOADERS[name].__qualname__}"
            )
        _LOADERS[name] = fn
        return fn

    return _wrap


def load_dataset(name: str) -> Dataset:
    if name not in _LOADERS:
        raise KeyError(f"No dataset named '{name}'. Known: {sorted(_LOADERS)}")
    return _LOADERS[name]()


def list_datasets() -> list[str]:
    return sorted(_LOADERS)


__all__ = ["Dataset", "register_dataset", "load_dataset", "list_datasets"]


from n26.data import d1, d2, d3_g12, d3_g3, d4  # noqa: E402, F401
