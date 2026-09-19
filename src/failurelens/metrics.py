"""Small dependency-free metrics for synthetic label-noise experiments."""

from __future__ import annotations

import numpy as np


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC using average ranks, including tied scores."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("labels and scores must have equal length")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative examples")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    positive_rank_sum = ranks[labels].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def detection_metrics(
    known_problematic: np.ndarray,
    suspicion_scores: np.ndarray,
    *,
    k: int | None = None,
) -> dict[str, float]:
    """Return AUROC, precision@k, and recall@k for a known synthetic mask."""
    known = np.asarray(known_problematic, dtype=bool).reshape(-1)
    scores = np.asarray(suspicion_scores, dtype=float).reshape(-1)
    if known.size != scores.size:
        raise ValueError("known_problematic and suspicion_scores must have equal length")
    if k is None:
        k = max(1, int(known.sum()))
    if k <= 0 or k > known.size:
        raise ValueError("k must be between 1 and the dataset size")
    selected = np.argsort(-scores, kind="mergesort")[:k]
    true_positives = int(known[selected].sum())
    return {
        "auroc": binary_auroc(known, scores),
        "precision_at_k": true_positives / k,
        "recall_at_k": true_positives / max(1, int(known.sum())),
        "k": float(k),
    }

