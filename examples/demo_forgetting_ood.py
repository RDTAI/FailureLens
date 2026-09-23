"""Test whether locally forgotten training regions have more ID OOD false alarms."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from failurelens import TrainingDynamicsTracker
from failurelens.joint import false_alarm_association, local_training_signal
from failurelens.ood import ConformalOODCalibrator, MahalanobisOODDetector
from failurelens.ood.scores import SCORE_FUNCTIONS


CENTERS = np.array([[-2.2, -1.1], [2.2, -1.0], [0.0, 2.2]], dtype=np.float32)


def sample_id(
    rng: np.random.Generator, *, samples_per_class: int, scale: float = 1.25
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for label, center in enumerate(CENTERS):
        features.append(rng.normal(center, scale, size=(samples_per_class, 2)))
        labels.append(np.full(samples_per_class, label))
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    order = rng.permutation(y.size)
    return x[order], y[order]


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, 3)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


def predict(model: TinyMLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x)
        embedding = model.forward_features(tensor)
        return model.classifier(embedding).numpy(), embedding.numpy()


def write_test_table(
    path: Path,
    x: np.ndarray,
    local_forgetting: np.ndarray,
    values: dict[str, np.ndarray],
    p_values: dict[str, np.ndarray],
    alarms: dict[str, np.ndarray],
) -> None:
    fields = ["sample_index", "x1", "x2", "local_forgetting_burden"]
    for name in values:
        fields.extend([f"{name}_score", f"{name}_p_value", f"{name}_false_alarm"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, point in enumerate(x):
            row: dict[str, object] = {
                "sample_index": index,
                "x1": float(point[0]),
                "x2": float(point[1]),
                "local_forgetting_burden": float(local_forgetting[index]),
            }
            for name in values:
                row[f"{name}_score"] = float(values[name][index])
                row[f"{name}_p_value"] = float(p_values[name][index])
                row[f"{name}_false_alarm"] = bool(alarms[name][index])
            writer.writerow(row)


def plot_result(
    path: Path,
    train_x: np.ndarray,
    forgetting_events: np.ndarray,
    test_x: np.ndarray,
    local_forgetting: np.ndarray,
    alarms: dict[str, np.ndarray],
    results: dict[str, dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    first = axes[0].scatter(
        train_x[:, 0], train_x[:, 1], c=forgetting_events, cmap="magma", s=14, alpha=0.7
    )
    axes[0].set_title("Training forgetting events")
    figure.colorbar(first, ax=axes[0], label="events")

    second = axes[1].scatter(
        test_x[:, 0], test_x[:, 1], c=local_forgetting, cmap="magma", s=14, alpha=0.7
    )
    msp_alarm = alarms["msp"]
    axes[1].scatter(
        test_x[msp_alarm, 0],
        test_x[msp_alarm, 1],
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.3,
        s=48,
        label="MSP false alarm",
    )
    axes[1].set_title("Local forgetting and MSP false alarms")
    axes[1].legend(fontsize=8)
    figure.colorbar(second, ax=axes[1], label="local forgetting burden")

    names = list(results)
    positions = np.arange(len(names))
    width = 0.36
    low = [results[name]["low_region_false_alarm_rate"] for name in names]
    high = [results[name]["high_region_false_alarm_rate"] for name in names]
    axes[2].bar(positions - width / 2, low, width, label="low-signal region")
    axes[2].bar(positions + width / 2, high, width, label="high-signal region")
    axes[2].set_xticks(positions, names, rotation=30, ha="right")
    axes[2].set_ylabel("ID false-alarm rate")
    axes[2].set_title("False alarms by forgetting region")
    axes[2].legend(fontsize=8)

    for axis in axes[:2]:
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.grid(alpha=0.15)
    figure.suptitle("FailureLens: do forgotten regions attract OOD false alarms?")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    seed: int,
    epochs: int,
    alpha: float,
    neighbours: int,
    resamples: int,
    output_dir: Path,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_x, train_y = sample_id(rng, samples_per_class=300)
    calibration_x, _ = sample_id(rng, samples_per_class=200)
    test_x, _ = sample_id(rng, samples_per_class=200)
    train_tensor = torch.from_numpy(train_x)
    label_tensor = torch.from_numpy(train_y)
    index_tensor = torch.arange(train_y.size)
    loader = DataLoader(
        TensorDataset(index_tensor, train_tensor, label_tensor),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    tracker = TrainingDynamicsTracker(train_y.size)
    for _ in range(epochs):
        model.train()
        for _, batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        tracker.begin_epoch()
        with torch.no_grad():
            logits = model(train_tensor)
            losses = nn.functional.cross_entropy(logits, label_tensor, reduction="none")
            tracker.update_batch(index_tensor, logits, label_tensor, losses)
        tracker.end_epoch()

    dynamics = tracker.summarize()
    local_forgetting = local_training_signal(
        train_x, dynamics.forgetting_events, test_x, k=neighbours
    )
    _, train_embedding = predict(model, train_x)
    calibration_logits, calibration_embedding = predict(model, calibration_x)
    test_logits, test_embedding = predict(model, test_x)

    score_values: dict[str, np.ndarray] = {}
    p_values: dict[str, np.ndarray] = {}
    alarms: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, float]] = {}
    for offset, (name, score_function) in enumerate(SCORE_FUNCTIONS.items()):
        calibrator = ConformalOODCalibrator(score_function(calibration_logits))
        score_values[name] = score_function(test_logits)
        p_values[name] = calibrator.p_values(score_values[name])
        alarms[name] = p_values[name] <= alpha
        results[name] = false_alarm_association(
            local_forgetting,
            alarms[name],
            bootstrap_samples=resamples,
            permutations=resamples,
            seed=seed + offset,
        )

    detector = MahalanobisOODDetector.fit(train_embedding, train_y, regularization=1e-3)
    calibrator = ConformalOODCalibrator(detector.score(calibration_embedding))
    score_values["mahalanobis"] = detector.score(test_embedding)
    p_values["mahalanobis"] = calibrator.p_values(score_values["mahalanobis"])
    alarms["mahalanobis"] = p_values["mahalanobis"] <= alpha
    results["mahalanobis"] = false_alarm_association(
        local_forgetting,
        alarms["mahalanobis"],
        bootstrap_samples=resamples,
        permutations=resamples,
        seed=seed + len(SCORE_FUNCTIONS),
    )

    report: dict[str, object] = {
        "seed": seed,
        "epochs": epochs,
        "alpha": alpha,
        "neighbours": neighbours,
        "train_samples": int(train_y.size),
        "calibration_samples": int(calibration_x.shape[0]),
        "test_samples": int(test_x.shape[0]),
        "training_samples_with_forgetting": int((dynamics.forgetting_events > 0).sum()),
        "mean_forgetting_events": float(dynamics.forgetting_events.mean()),
        "detectors": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forgetting_ood_association.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_test_table(
        output_dir / "id_test_false_alarms.csv",
        test_x,
        local_forgetting,
        score_values,
        p_values,
        alarms,
    )
    plot_result(
        output_dir / "forgetting_ood_map.png",
        train_x,
        dynamics.forgetting_events,
        test_x,
        local_forgetting,
        alarms,
        results,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--neighbours", type=int, default=30)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forgetting_ood"))
    args = parser.parse_args()
    report = run(
        args.seed,
        args.epochs,
        args.alpha,
        args.neighbours,
        args.resamples,
        args.output_dir,
    )
    compact = {
        name: {
            "overall_fpr": values["overall_false_alarm_rate"],
            "low_forgetting_fpr": values["low_region_false_alarm_rate"],
            "high_forgetting_fpr": values["high_region_false_alarm_rate"],
            "risk_difference": values["risk_difference"],
            "risk_difference_95ci": [
                values["risk_difference_ci_low"],
                values["risk_difference_ci_high"],
            ],
            "permutation_p": values["permutation_p_value"],
        }
        for name, values in report["detectors"].items()
    }
    print(json.dumps(compact, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
