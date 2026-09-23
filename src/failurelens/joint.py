"""Joint analysis of training dynamics and inference-time false alarms."""

from __future__ import annotations

from typing import Any

import numpy as np

from failurelens.metrics import binary_auroc


def _as_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty [samples, dimensions] array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def local_training_signal(
    train_features: Any,
    train_signal: Any,
    query_features: Any,
    *,
    k: int = 25,
) -> np.ndarray:
    """Interpolate a per-training-sample signal onto query samples with k-NN.

    Neighbours receive inverse-distance weights. Exact feature matches receive
    all weight, split equally if more than one training point matches.
    """
    train = _as_matrix(train_features, "train_features")
    query = _as_matrix(query_features, "query_features")
    signal = np.asarray(train_signal, dtype=float).reshape(-1)
    if train.shape[0] != signal.size:
        raise ValueError("train_signal must contain one value per training sample")
    if train.shape[1] != query.shape[1]:
        raise ValueError("train and query features must have the same dimension")
    if not np.all(np.isfinite(signal)):
        raise ValueError("train_signal must contain only finite values")
    if k <= 0 or k > train.shape[0]:
        raise ValueError("k must lie between 1 and the number of training samples")

    squared_distances = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=2)
    neighbours = np.argpartition(squared_distances, kth=k - 1, axis=1)[:, :k]
    local_distances = np.take_along_axis(squared_distances, neighbours, axis=1)
    local_signal = signal[neighbours]

    result = np.empty(query.shape[0], dtype=float)
    for row in range(query.shape[0]):
        exact = local_distances[row] <= 1e-20
        if np.any(exact):
            result[row] = float(local_signal[row, exact].mean())
        else:
            weights = 1.0 / np.sqrt(local_distances[row])
            result[row] = float(np.sum(weights * local_signal[row]) / weights.sum())
    return result


def _risk_contrast(
    signal: np.ndarray,
    alarms: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> tuple[float, float, float, float, int, int]:
    low = signal <= low_threshold
    high = signal >= high_threshold
    low_rate = float(alarms[low].mean())
    high_rate = float(alarms[high].mean())
    difference = high_rate - low_rate
    high_events = int(alarms[high].sum())
    low_events = int(alarms[low].sum())
    # Half-count correction keeps the ratio finite when one region has no alarms.
    ratio = ((high_events + 0.5) / (high.sum() + 1.0)) / (
        (low_events + 0.5) / (low.sum() + 1.0)
    )
    return low_rate, high_rate, difference, ratio, int(low.sum()), int(high.sum())


def false_alarm_association(
    local_signal: Any,
    false_alarms: Any,
    *,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
    bootstrap_samples: int = 2000,
    permutations: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Measure whether a local training signal is elevated at false alarms.

    Quantile thresholds are fixed on the observed test signal. Bootstrap
    intervals resample test examples, while the permutation p-value tests the
    null that alarm labels are unrelated to the fixed regional signal.
    """
    signal = np.asarray(local_signal, dtype=float).reshape(-1)
    alarms = np.asarray(false_alarms, dtype=bool).reshape(-1)
    if signal.size != alarms.size or signal.size < 4:
        raise ValueError("local_signal and false_alarms must match and contain at least 4 samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("local_signal must contain only finite values")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    if bootstrap_samples <= 0 or permutations <= 0:
        raise ValueError("bootstrap_samples and permutations must be positive")

    low_threshold, high_threshold = np.quantile(signal, [low_quantile, high_quantile])
    low_rate, high_rate, difference, ratio, low_size, high_size = _risk_contrast(
        signal, alarms, float(low_threshold), float(high_threshold)
    )
    alarm_auroc = (
        binary_auroc(alarms, signal) if np.any(alarms) and np.any(~alarms) else float("nan")
    )

    rng = np.random.default_rng(seed)
    low_alarms = alarms[signal <= low_threshold]
    high_alarms = alarms[signal >= high_threshold]
    bootstrap_differences = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sampled_low = low_alarms[rng.integers(0, low_alarms.size, size=low_alarms.size)]
        sampled_high = high_alarms[rng.integers(0, high_alarms.size, size=high_alarms.size)]
        bootstrap_differences[index] = sampled_high.mean() - sampled_low.mean()
    lower, upper = np.quantile(bootstrap_differences, [0.025, 0.975])

    extreme = 0
    for _ in range(permutations):
        permuted = rng.permutation(alarms)
        permuted_difference = _risk_contrast(
            signal, permuted, float(low_threshold), float(high_threshold)
        )[2]
        extreme += abs(permuted_difference) >= abs(difference)
    p_value = (extreme + 1.0) / (permutations + 1.0)

    return {
        "samples": float(signal.size),
        "overall_false_alarm_rate": float(alarms.mean()),
        "low_signal_threshold": float(low_threshold),
        "high_signal_threshold": float(high_threshold),
        "low_region_size": float(low_size),
        "high_region_size": float(high_size),
        "low_region_false_alarm_rate": low_rate,
        "high_region_false_alarm_rate": high_rate,
        "risk_difference": difference,
        "risk_difference_ci_low": float(lower),
        "risk_difference_ci_high": float(upper),
        "risk_ratio": ratio,
        "false_alarm_auroc": alarm_auroc,
        "permutation_p_value": float(p_value),
    }
