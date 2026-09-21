"""Threshold-free and operating-point metrics for OOD detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from failurelens.metrics import binary_auroc


def _validate(labels: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    labels_np = np.asarray(labels, dtype=bool).reshape(-1)
    scores_np = np.asarray(scores, dtype=float).reshape(-1)
    if labels_np.size != scores_np.size:
        raise ValueError("labels and scores must have equal length")
    if labels_np.size == 0 or labels_np.all() or (~labels_np).all():
        raise ValueError("metrics require both ID and OOD examples")
    if not np.all(np.isfinite(scores_np)):
        raise ValueError("scores must contain only finite values")
    return labels_np, scores_np


def average_precision(labels: Any, scores: Any) -> float:
    """Average precision with OOD as the positive class."""
    labels_np, scores_np = _validate(labels, scores)
    order = np.argsort(-scores_np, kind="mergesort")
    ranked = labels_np[order].astype(float)
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, ranked.size + 1)
    return float(np.sum(precision * ranked) / ranked.sum())


def fpr_at_tpr(labels: Any, scores: Any, *, target_tpr: float = 0.95) -> float:
    """Minimum ID false-positive rate at or above the requested OOD TPR."""
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must lie in (0, 1]")
    labels_np, scores_np = _validate(labels, scores)
    thresholds = np.unique(scores_np)[::-1]
    positives = labels_np.sum()
    negatives = (~labels_np).sum()
    best = 1.0
    reached = False
    for threshold in thresholds:
        predicted = scores_np >= threshold
        tpr = np.sum(predicted & labels_np) / positives
        if tpr >= target_tpr:
            reached = True
            best = min(best, float(np.sum(predicted & ~labels_np) / negatives))
    return best if reached else 1.0


def ood_detection_metrics(labels: Any, scores: Any) -> dict[str, float]:
    """Report common OOD metrics using OOD as the positive class."""
    labels_np, scores_np = _validate(labels, scores)
    return {
        "auroc": binary_auroc(labels_np, scores_np),
        "aupr_ood": average_precision(labels_np, scores_np),
        "fpr_at_95_tpr": fpr_at_tpr(labels_np, scores_np, target_tpr=0.95),
    }

