"""End-to-end demonstration on a classification dataset with injected label noise."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from failurelens import TrainingDynamicsTracker, detection_metrics
from failurelens.report import plot_data_map, write_csv


def make_dataset(
    *, seed: int, samples_per_class: int = 200, noise_rate: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.2, -1.2], [2.2, -1.0], [0.0, 2.4]], dtype=np.float32)
    features = []
    labels = []
    for class_id, center in enumerate(centers):
        features.append(rng.normal(center, 0.85, size=(samples_per_class, 2)))
        labels.append(np.full(samples_per_class, class_id))
    x = np.concatenate(features).astype(np.float32)
    clean_y = np.concatenate(labels).astype(np.int64)
    order = rng.permutation(len(clean_y))
    x, clean_y = x[order], clean_y[order]

    observed_y = clean_y.copy()
    num_noisy = int(round(noise_rate * len(clean_y)))
    noisy_indices = rng.choice(len(clean_y), size=num_noisy, replace=False)
    offsets = rng.integers(1, len(centers), size=num_noisy)
    observed_y[noisy_indices] = (clean_y[noisy_indices] + offsets) % len(centers)
    is_noisy = observed_y != clean_y
    return x, observed_y, clean_y, is_noisy


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def run(seed: int, epochs: int, noise_rate: float, output_dir: Path) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x, observed_y, _, is_noisy = make_dataset(seed=seed, noise_rate=noise_rate)
    features = torch.from_numpy(x)
    labels = torch.from_numpy(observed_y)
    indices = torch.arange(len(labels))
    loader = DataLoader(
        TensorDataset(indices, features, labels),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    tracker = TrainingDynamicsTracker(len(labels))

    for _ in range(epochs):
        model.train()
        for _, batch_x, batch_y in loader:
            optimizer.zero_grad()
            batch_logits = model(batch_x)
            loss = criterion(batch_logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        tracker.begin_epoch()
        with torch.no_grad():
            logits = model(features)
            per_sample_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
            tracker.update_batch(indices, logits, labels, per_sample_loss)
        tracker.end_epoch()

    summary = tracker.summarize()
    metrics = detection_metrics(is_noisy, summary.suspicion_score)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, output_dir / "sample_diagnostics.csv")
    plot_data_map(summary, output_dir / "training_dynamics_map.png", known_problematic=is_noisy)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--noise-rate", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    metrics = run(args.seed, args.epochs, args.noise_rate, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

