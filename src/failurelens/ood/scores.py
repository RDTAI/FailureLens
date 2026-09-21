"""Post-hoc OOD scores computed from classifier logits.

Every function follows one convention: larger values mean "more OOD-like".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _as_logits(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    logits = np.asarray(value, dtype=float)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [samples, classes] with at least two classes")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain only finite values")
    return logits


def _probabilities(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = _as_logits(logits) / temperature
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def msp_score(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Return ``1 - max softmax probability`` for each sample."""
    probabilities = _probabilities(logits, temperature=temperature)
    return 1.0 - probabilities.max(axis=1)


def entropy_score(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Return predictive entropy in nats."""
    probabilities = _probabilities(logits, temperature=temperature)
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)


def probability_margin_score(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Return one minus the gap between the two largest class probabilities."""
    probabilities = _probabilities(logits, temperature=temperature)
    top_two = np.partition(probabilities, kth=-2, axis=1)[:, -2:]
    gap = top_two.max(axis=1) - top_two.min(axis=1)
    return 1.0 - gap


def energy_score(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Return classifier energy; larger (less negative) values are more OOD-like."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = _as_logits(logits) / temperature
    maximum = values.max(axis=1)
    logsumexp = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    return -temperature * logsumexp


SCORE_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "msp": msp_score,
    "entropy": entropy_score,
    "probability_margin": probability_margin_score,
    "energy": energy_score,
}


@dataclass(frozen=True)
class MahalanobisOODDetector:
    """Class-conditional Mahalanobis distance in a learned feature space."""

    class_labels: np.ndarray
    class_means: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(
        cls,
        features: Any,
        labels: Any,
        *,
        regularization: float = 1e-4,
    ) -> "MahalanobisOODDetector":
        feature_array = np.asarray(features, dtype=float)
        label_array = np.asarray(labels).reshape(-1)
        if feature_array.ndim != 2 or feature_array.shape[0] != label_array.size:
            raise ValueError("features must be [samples, dimensions] and match labels")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        class_labels = np.unique(label_array)
        if class_labels.size < 2:
            raise ValueError("at least two classes are required")
        means = np.stack([feature_array[label_array == label].mean(axis=0) for label in class_labels])
        centered = np.empty_like(feature_array)
        for label, mean in zip(class_labels, means):
            mask = label_array == label
            centered[mask] = feature_array[mask] - mean
        covariance = centered.T @ centered / max(1, feature_array.shape[0] - class_labels.size)
        covariance += regularization * np.eye(feature_array.shape[1])
        precision = np.linalg.pinv(covariance, hermitian=True)
        return cls(class_labels=class_labels, class_means=means, precision=precision)

    def score(self, features: Any) -> np.ndarray:
        feature_array = np.asarray(features, dtype=float)
        if feature_array.ndim != 2 or feature_array.shape[1] != self.class_means.shape[1]:
            raise ValueError("features have an incompatible shape")
        differences = feature_array[:, None, :] - self.class_means[None, :, :]
        distances = np.einsum("ncd,df,ncf->nc", differences, self.precision, differences)
        return distances.min(axis=1)
