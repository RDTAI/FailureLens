"""Export diagnostic tables and data-map figures."""

from __future__ import annotations

import csv
from pathlib import Path

from .tracker import DynamicsSummary


def write_csv(summary: DynamicsSummary, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(summary.as_records(), key=lambda row: -float(row["suspicion_score"]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


def plot_data_map(
    summary: DynamicsSummary,
    path: str | Path,
    *,
    known_problematic=None,
) -> Path:
    """Plot confidence versus variability; matplotlib is imported lazily."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = summary.suspicion_score if known_problematic is None else np.asarray(known_problematic)
    label = "Suspicion score" if known_problematic is None else "Injected label noise"

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    points = axis.scatter(
        summary.mean_confidence,
        summary.variability,
        c=colors,
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )
    axis.set(xlabel="Mean assigned-label confidence", ylabel="Confidence variability")
    axis.set_title("FailureLens training-dynamics map")
    axis.grid(alpha=0.2)
    figure.colorbar(points, ax=axis, label=label)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
