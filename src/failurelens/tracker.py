"""Record per-example classification behavior over training epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy arrays or CPU/GPU tensors to NumPy without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return average tie-aware ranks in [0, 1], larger values ranking higher."""
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks / (values.size - 1)


@dataclass(frozen=True)
class DynamicsSummary:
    """Per-example statistics computed from a complete training run."""

    index: np.ndarray
    mean_confidence: np.ndarray
    variability: np.ndarray
    correctness: np.ndarray
    forgetting_events: np.ndarray
    learned_epoch: np.ndarray
    mean_margin: np.ndarray
    mean_loss: np.ndarray
    suspicion_score: np.ndarray
    region: np.ndarray

    def ranked_indices(self) -> np.ndarray:
        """Indices ordered from most to least suspicious."""
        return self.index[np.argsort(-self.suspicion_score, kind="mergesort")]

    def as_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for i in range(self.index.size):
            records.append(
                {
                    "index": int(self.index[i]),
                    "suspicion_score": float(self.suspicion_score[i]),
                    "mean_margin": float(self.mean_margin[i]),
                    "mean_loss": float(self.mean_loss[i]),
                    "mean_confidence": float(self.mean_confidence[i]),
                    "variability": float(self.variability[i]),
                    "correctness": float(self.correctness[i]),
                    "forgetting_events": int(self.forgetting_events[i]),
                    "learned_epoch": int(self.learned_epoch[i]),
                    "region": str(self.region[i]),
                }
            )
        return records


class TrainingDynamicsTracker:
    """Collect per-example training dynamics for a classification dataset.

    Call ``begin_epoch()``, one or more ``update_batch(...)`` calls, then
    ``end_epoch()``. Dataset indices must be stable across epochs.
    """

    def __init__(self, num_examples: int) -> None:
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        self.num_examples = int(num_examples)
        self._epochs: list[dict[str, np.ndarray]] = []
        self._current: dict[str, np.ndarray] | None = None

    @property
    def num_epochs(self) -> int:
        return len(self._epochs)

    def begin_epoch(self) -> None:
        if self._current is not None:
            raise RuntimeError("end the current epoch before beginning another")
        self._current = {
            "confidence": np.full(self.num_examples, np.nan),
            "margin": np.full(self.num_examples, np.nan),
            "loss": np.full(self.num_examples, np.nan),
            "correct": np.full(self.num_examples, np.nan),
        }

    def update_batch(
        self,
        indices: Any,
        logits: Any,
        labels: Any,
        losses: Any | None = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("call begin_epoch before update_batch")

        indices_np = _as_numpy(indices).astype(int).reshape(-1)
        logits_np = _as_numpy(logits).astype(float)
        labels_np = _as_numpy(labels).astype(int).reshape(-1)
        if logits_np.ndim != 2:
            raise ValueError("logits must have shape [batch, classes]")
        if len(indices_np) != len(labels_np) or logits_np.shape[0] != len(labels_np):
            raise ValueError("indices, logits, and labels must have matching batch sizes")
        if np.any(indices_np < 0) or np.any(indices_np >= self.num_examples):
            raise IndexError("sample index is outside the configured dataset")
        if np.any(labels_np < 0) or np.any(labels_np >= logits_np.shape[1]):
            raise ValueError("label is outside the logit class dimension")

        probabilities = _softmax(logits_np)
        rows = np.arange(len(labels_np))
        confidence = probabilities[rows, labels_np]
        correct = logits_np.argmax(axis=1) == labels_np

        masked = logits_np.copy()
        masked[rows, labels_np] = -np.inf
        margin = logits_np[rows, labels_np] - masked.max(axis=1)

        if losses is None:
            loss = -np.log(np.clip(confidence, 1e-12, 1.0))
        else:
            loss = _as_numpy(losses).astype(float).reshape(-1)
            if len(loss) != len(labels_np):
                raise ValueError("losses must contain one value per sample")

        if np.unique(indices_np).size != indices_np.size:
            raise ValueError("duplicate sample indices in one batch")
        already_seen = ~np.isnan(self._current["confidence"][indices_np])
        if np.any(already_seen):
            raise ValueError("a sample was recorded more than once in this epoch")

        self._current["confidence"][indices_np] = confidence
        self._current["margin"][indices_np] = margin
        self._current["loss"][indices_np] = loss
        self._current["correct"][indices_np] = correct.astype(float)

    def end_epoch(self, *, allow_partial: bool = False) -> None:
        if self._current is None:
            raise RuntimeError("call begin_epoch before end_epoch")
        missing = np.isnan(self._current["confidence"])
        if np.any(missing) and not allow_partial:
            raise ValueError(f"epoch is missing {int(missing.sum())} examples")
        self._epochs.append(self._current)
        self._current = None

    def summarize(self) -> DynamicsSummary:
        if self._current is not None:
            raise RuntimeError("finish the current epoch before summarizing")
        if not self._epochs:
            raise RuntimeError("no completed epochs to summarize")

        confidence = np.stack([epoch["confidence"] for epoch in self._epochs])
        margin = np.stack([epoch["margin"] for epoch in self._epochs])
        loss = np.stack([epoch["loss"] for epoch in self._epochs])
        correct = np.stack([epoch["correct"] for epoch in self._epochs])

        mean_confidence = np.nanmean(confidence, axis=0)
        variability = np.nanstd(confidence, axis=0)
        correctness = np.nanmean(correct, axis=0)
        mean_margin = np.nanmean(margin, axis=0)
        mean_loss = np.nanmean(loss, axis=0)

        transitions = (correct[:-1] == 1) & (correct[1:] == 0)
        forgetting_events = transitions.sum(axis=0).astype(int)
        learned_epoch = np.full(self.num_examples, -1, dtype=int)
        for sample in range(self.num_examples):
            learned = np.flatnonzero(correct[:, sample] == 1)
            if learned.size:
                learned_epoch[sample] = int(learned[0])

        # A transparent review-priority heuristic. Persistent disagreement with
        # the observed label carries most weight; forgetting is supplementary.
        suspicion_score = (
            0.40 * _percentile_rank(-mean_margin)
            + 0.25 * _percentile_rank(mean_loss)
            + 0.20 * _percentile_rank(-mean_confidence)
            + 0.15 * _percentile_rank(forgetting_events)
        )

        median_variability = float(np.nanmedian(variability))
        region = np.full(self.num_examples, "hard-to-learn", dtype=object)
        region[(mean_confidence >= 0.5) & (variability <= median_variability)] = "easy-to-learn"
        region[variability > median_variability] = "ambiguous"

        return DynamicsSummary(
            index=np.arange(self.num_examples),
            mean_confidence=mean_confidence,
            variability=variability,
            correctness=correctness,
            forgetting_events=forgetting_events,
            learned_epoch=learned_epoch,
            mean_margin=mean_margin,
            mean_loss=mean_loss,
            suspicion_score=suspicion_score,
            region=region,
        )
