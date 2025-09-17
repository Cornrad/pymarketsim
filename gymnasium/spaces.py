"""Minimal implementations of the :mod:`gymnasium.spaces` classes used by tests."""

from __future__ import annotations

import numpy as np


class Space:
    def sample(self):  # pragma: no cover - compatibility shim
        raise NotImplementedError


class Box(Space):
    def __init__(self, low, high, shape, dtype=float):
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape)
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


__all__ = ["Space", "Box"]
