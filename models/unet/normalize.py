"""
UrbanFlood360 MODIFICATION (not repository code).

Normalization fitted from TRAIN patches only, loaded/applied at data-loading
time. The stored raw patch .npy files are never modified. Two methods are
implemented per the project's normalization investigation; "mean_std" is the
active default (chosen because all six channels' train-only statistics show
mean approx equal to median with no extreme skew - see
reports/unet_repository_integration_report.md), with "percentile" available
as a documented alternative.
"""
import json

import numpy as np


class Normalizer:
    def __init__(self, stats: dict):
        self.stats = stats
        self.method = stats["method_active"]
        self.channels = stats["channels"]

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(json.load(f))

    def apply(self, img: np.ndarray) -> np.ndarray:
        """img: (H,W,C) float32, may contain NaN. Returns same shape, normalized."""
        out = img.copy()
        for c, name in enumerate(self.channels):
            s = self.stats["per_channel"][name]
            if self.method == "mean_std":
                out[..., c] = (img[..., c] - s["mean"]) / (s["std"] + 1e-8)
            elif self.method == "percentile":
                lo, hi = s["p2"], s["p98"]
                out[..., c] = np.clip((img[..., c] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
        return out
