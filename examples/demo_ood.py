"""Synthetic near-, far-, and high-confidence OOD benchmark."""

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

from failurelens.ood import (
    ConformalOODCalibrator,
    MahalanobisOODDetector,
    ood_detection_metrics,
)
from failurelens.ood.scores import SCORE_FUNCTIONS


CENTERS = np.array([[-2.2, -1.2], [2.2, -1.0], [0.0, 2.4]], dtype=np.float32)


def sample_mixture(
    rng: np.random.Generator,
    *,
    samples_per_class: int,
    centers: np.ndarray = CENTERS,
    scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for class_id, center in enumerate(centers):
        features.append(rng.normal(center, scale, size=(samples_per_class, 2)))
        labels.append(np.full(samples_per_class, class_id))
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def sample_far_ood(rng: np.random.Generator, size: int) -> np.ndarray:
    radii = rng.uniform(7.0, 10.0, size=size)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=size)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).astype(np.float32)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(48, 3)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


def predict_outputs(model: TinyMLP, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features)
        embeddings = model.forward_features(inputs)
        return model.classifier(embeddings).cpu().numpy(), embeddings.cpu().numpy()


def select_high_confidence_ood(
    rng: np.random.Generator, model: nn.Module, *, size: int
) -> np.ndarray:
    candidates = rng.uniform(-12.0, 12.0, size=(20_000, 2)).astype(np.float32)
    distance = np.linalg.norm(candidates[:, None, :] - CENTERS[None, :, :], axis=2).min(axis=1)
    candidates = candidates[distance >= 5.0]
    logits, _ = predict_outputs(model, candidates)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    selected = np.argsort(-confidence, kind="mergesort")[:size]
    return candidates[selected]


def _combined_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate(
        [np.zeros(len(id_scores), dtype=bool), np.ones(len(ood_scores), dtype=bool)]
    )
    return ood_detection_metrics(labels, np.concatenate([id_scores, ood_scores]))


def write_sample_table(
    path: Path,
    groups: dict[str, np.ndarray],
    score_values: dict[str, dict[str, np.ndarray]],
    calibrators: dict[str, ConformalOODCalibrator],
    *,
    alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "sample_index", "x1", "x2"]
    for score_name in score_values:
        fields.extend([score_name, f"{score_name}_p_value", f"{score_name}_reject"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_name, features in groups.items():
            p_values = {
                name: calibrators[name].p_values(score_values[name][group_name])
                for name in score_values
            }
            for index, point in enumerate(features):
                row: dict[str, object] = {
                    "group": group_name,
                    "sample_index": index,
                    "x1": float(point[0]),
                    "x2": float(point[1]),
                }
                for score_name in score_values:
                    p_value = float(p_values[score_name][index])
                    row[score_name] = float(score_values[score_name][group_name][index])
                    row[f"{score_name}_p_value"] = p_value
                    row[f"{score_name}_reject"] = p_value <= alpha
                writer.writerow(row)


def plot_scores(
    path: Path,
    groups: dict[str, np.ndarray],
    score_values: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    for axis, score_name in zip(axes, ["msp", "energy", "mahalanobis"]):
        for group_name, marker in [
            ("id_test", "o"),
            ("near_ood", "s"),
            ("far_ood", "^"),
            ("high_confidence_ood", "x"),
        ]:
            points = groups[group_name]
            values = score_values[score_name][group_name]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                c=values,
                cmap="viridis",
                marker=marker,
                s=13,
                alpha=0.58,
                label=group_name,
            )
        axis.set_title(f"{score_name.upper()} OOD score")
        axis.set_xlabel("x1")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("x2")
    axes[-1].legend(loc="upper right", fontsize=7)
    figure.suptitle("FailureLens v0.2: synthetic OOD stress test")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(seed: int, epochs: int, alpha: float, output_dir: Path) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_x, train_y = sample_mixture(rng, samples_per_class=300)
    calibration_x, _ = sample_mixture(rng, samples_per_class=150)
    id_test_x, _ = sample_mixture(rng, samples_per_class=150)
    near_centers = CENTERS + np.array([[0.8, 0.7], [-0.8, 0.7], [0.0, -0.9]])
    near_ood_x, _ = sample_mixture(
        rng, samples_per_class=150, centers=near_centers, scale=1.15
    )
    far_ood_x = sample_far_ood(rng, len(id_test_x))

    model = TinyMLP()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    high_confidence_ood_x = select_high_confidence_ood(rng, model, size=len(id_test_x))
    groups = {
        "id_test": id_test_x,
        "near_ood": near_ood_x,
        "far_ood": far_ood_x,
        "high_confidence_ood": high_confidence_ood_x,
    }
    _, train_embeddings = predict_outputs(model, train_x)
    calibration_logits, calibration_embeddings = predict_outputs(model, calibration_x)
    group_outputs = {name: predict_outputs(model, values) for name, values in groups.items()}
    group_logits = {name: outputs[0] for name, outputs in group_outputs.items()}
    group_embeddings = {name: outputs[1] for name, outputs in group_outputs.items()}

    score_values: dict[str, dict[str, np.ndarray]] = {}
    calibrators: dict[str, ConformalOODCalibrator] = {}
    results: dict[str, object] = {
        "seed": seed,
        "epochs": epochs,
        "alpha": alpha,
        "calibration_size": len(calibration_x),
        "scores": {},
    }
    for score_name, score_function in SCORE_FUNCTIONS.items():
        calibration_scores = score_function(calibration_logits)
        calibrator = ConformalOODCalibrator(calibration_scores)
        calibrators[score_name] = calibrator
        values = {name: score_function(logits) for name, logits in group_logits.items()}
        score_values[score_name] = values

        id_scores = values["id_test"]
        id_false_alarm = float(calibrator.reject(id_scores, alpha=alpha).mean())
        score_result: dict[str, object] = {
            "id_false_alarm_rate": id_false_alarm,
            "minimum_p_value": calibrator.minimum_p_value(),
        }
        pooled_ood = np.concatenate(
            [values["near_ood"], values["far_ood"], values["high_confidence_ood"]]
        )
        score_result["pooled"] = _combined_metrics(id_scores, pooled_ood)
        for group_name in ["near_ood", "far_ood", "high_confidence_ood"]:
            score_result[group_name] = {
                **_combined_metrics(id_scores, values[group_name]),
                "conformal_power": float(
                    calibrator.reject(values[group_name], alpha=alpha).mean()
                ),
            }
        results["scores"][score_name] = score_result

    mahalanobis = MahalanobisOODDetector.fit(train_embeddings, train_y, regularization=1e-3)
    mahalanobis_calibration = mahalanobis.score(calibration_embeddings)
    calibrator = ConformalOODCalibrator(mahalanobis_calibration)
    calibrators["mahalanobis"] = calibrator
    values = {name: mahalanobis.score(embedding) for name, embedding in group_embeddings.items()}
    score_values["mahalanobis"] = values
    id_scores = values["id_test"]
    mahalanobis_result: dict[str, object] = {
        "id_false_alarm_rate": float(calibrator.reject(id_scores, alpha=alpha).mean()),
        "minimum_p_value": calibrator.minimum_p_value(),
    }
    pooled_ood = np.concatenate(
        [values["near_ood"], values["far_ood"], values["high_confidence_ood"]]
    )
    mahalanobis_result["pooled"] = _combined_metrics(id_scores, pooled_ood)
    for group_name in ["near_ood", "far_ood", "high_confidence_ood"]:
        mahalanobis_result[group_name] = {
            **_combined_metrics(id_scores, values[group_name]),
            "conformal_power": float(calibrator.reject(values[group_name], alpha=alpha).mean()),
        }
    results["scores"]["mahalanobis"] = mahalanobis_result

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ood_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    write_sample_table(
        output_dir / "ood_sample_scores.csv",
        groups,
        score_values,
        calibrators,
        alpha=alpha,
    )
    plot_scores(output_dir / "ood_score_map.png", groups, score_values)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ood"))
    args = parser.parse_args()
    results = run(args.seed, args.epochs, args.alpha, args.output_dir)
    compact = {
        name: {
            "id_false_alarm_rate": values["id_false_alarm_rate"],
            "near_ood_auroc": values["near_ood"]["auroc"],
            "near_ood_power": values["near_ood"]["conformal_power"],
            "far_ood_auroc": values["far_ood"]["auroc"],
            "far_ood_power": values["far_ood"]["conformal_power"],
            "high_confidence_ood_auroc": values["high_confidence_ood"]["auroc"],
            "high_confidence_ood_power": values["high_confidence_ood"]["conformal_power"],
        }
        for name, values in results["scores"].items()
    }
    print(json.dumps(compact, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
