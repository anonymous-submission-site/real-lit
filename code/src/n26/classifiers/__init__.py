"""Classifier registry."""
from __future__ import annotations
from typing import Callable, Type

from n26.classifiers.base import Classifier

_CLASSIFIERS: dict[str, Type[Classifier]] = {}


def register_classifier(name: str) -> Callable[[Type[Classifier]], Type[Classifier]]:
    def _wrap(cls: Type[Classifier]) -> Type[Classifier]:
        if name in _CLASSIFIERS:
            raise ValueError(
                f"Classifier '{name}' already registered by "
                f"{_CLASSIFIERS[name].__qualname__}"
            )
        cls.name = name
        _CLASSIFIERS[name] = cls
        return cls

    return _wrap


def get_classifier(name: str, **kwargs) -> Classifier:
    if name not in _CLASSIFIERS:
        raise KeyError(
            f"No classifier named '{name}'. Known: {sorted(_CLASSIFIERS)}"
        )
    return _CLASSIFIERS[name](**kwargs)


def list_classifiers() -> list[str]:
    return sorted(_CLASSIFIERS)


__all__ = ["Classifier", "register_classifier", "get_classifier", "list_classifiers"]


from n26.classifiers import (  # noqa: E402, F401
    maskmlp, sklearn_baseline, tabdpt, tabicl, tabpfn,
)
