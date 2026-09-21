"""Split-conformal calibration for OOD scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_scores(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    scores = np.asarray(value, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values")
    return scores


@dataclass(frozen=True)
class ConformalOODCalibrator:
    """Calibrate any score where larger values indicate stronger OOD evidence.

    Under exchangeability of the calibration examples and a new ID example,
    ``p_values`` are marginally super-uniform. Ties use the conservative
    ``>=`` rule, so no randomization is required.
    """

    calibration_scores: np.ndarray

    def __post_init__(self) -> None:
        scores = _as_scores(self.calibration_scores).copy()
        scores.setflags(write=False)
        object.__setattr__(self, "calibration_scores", scores)

    @property
    def size(self) -> int:
        return int(self.calibration_scores.size)

    def p_values(self, test_scores: Any) -> np.ndarray:
        """Return conformal p-values for the null hypothesis that inputs are ID."""
        test = _as_scores(test_scores)
        exceedances = (self.calibration_scores[:, None] >= test[None, :]).sum(axis=0)
        return (1.0 + exceedances) / (self.size + 1.0)

    def reject(self, test_scores: Any, *, alpha: float = 0.05) -> np.ndarray:
        """Flag samples with conformal p-value at most ``alpha`` as OOD."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        return self.p_values(test_scores) <= alpha

    def minimum_p_value(self) -> float:
        return 1.0 / (self.size + 1.0)

