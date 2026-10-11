"""Model registry. Composition roots pick models by name, never by isinstance."""

from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def get_model(name: str, **kwargs: Any) -> Any:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
