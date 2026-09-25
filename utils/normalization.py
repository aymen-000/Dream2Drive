from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    eps: float = 1e-6

    @classmethod
    def fit(cls, data: np.ndarray, feature_names: List[str] | None = None) -> "Normalizer":
        """data: [N, D]"""
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std, feature_names=feature_names or [])

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
            feature_names=d.get("feature_names", []),
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
