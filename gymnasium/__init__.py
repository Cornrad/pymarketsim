"""Lightweight subset of the :mod:`gymnasium` API used in tests."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from . import spaces


class Env:
    metadata = {}

    def __init__(self) -> None:
        self.observation_space: spaces.Space
        self.action_space: spaces.Space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> Tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: Any):  # pragma: no cover - interfaces mirror gymnasium
        raise NotImplementedError

    def render(self):
        return None

    def close(self) -> None:
        return None


__all__ = ["Env", "spaces"]
