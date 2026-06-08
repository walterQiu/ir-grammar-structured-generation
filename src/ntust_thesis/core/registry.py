"""Component registry for datasets/models/metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

Factory = Callable[..., T]


class Registry[T]:
    """Simple named factory registry."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._store: dict[str, Factory[T]] = {}

    def register(self, key: str, factory: Factory[T]) -> None:
        """Register a factory with a unique key."""
        if key in self._store:
            msg = f"Duplicate registration: {key}"
            raise ValueError(msg)
        self._store[key] = factory

    def create(self, key: str, **kwargs: object) -> T:
        """Create object from a registered factory."""
        if key not in self._store:
            msg = f"Unknown registry key: {key}"
            raise KeyError(msg)
        return self._store[key](**kwargs)

    def keys(self) -> list[str]:
        """List available registry keys."""
        return sorted(self._store.keys())


DATASET_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
METRIC_REGISTRY = Registry()
