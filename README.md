# FailureLens

**Training-dynamics and out-of-distribution diagnostics for reliable classification.**

FailureLens records how a classifier behaves on every training example across epochs. It turns those trajectories into an auditable table that helps answer:

- Which samples are repeatedly forgotten?
- Which labels receive persistently negative prediction margins?
- Which samples are consistently hard versus genuinely ambiguous?
- Which examples should a human inspect first?

The project is designed as a compact research and teaching tool for reliable AI. It does **not** automatically declare that a sample is wrong; it prioritizes candidates for human review.

## New in v0.2: synthetic OOD diagnostics

Version 0.2 adds a post-hoc OOD module with a consistent “larger means more OOD-like” interface:

- maximum softmax probability (MSP) baseline;
- predictive entropy;
- top-two probability margin;
- classifier energy;
- class-conditional Mahalanobis distance in the penultimate feature space;
- split-conformal OOD p-values calibrated only on held-out ID data;
- AUROC, AUPR-OOD, FPR@95TPR, ID false-alarm rate, and OOD detection power;
- separate near-OOD, far-OOD, and deliberately high-confidence OOD stress tests.

```bash
python examples/demo_ood.py
```

The experiment writes `ood_metrics.json`, a per-sample score table, and an MSP/energy/Mahalanobis score map to `outputs/ood/`. See [`docs/ood_protocol.md`](docs/ood_protocol.md) for the evaluation contract and validity assumptions.

One reference run (seed 7, 40 epochs) illustrates why multiple detectors are useful:

| Detector | Near-OOD AUROC | Far-OOD AUROC | High-confidence OOD AUROC | ID false-alarm rate at α=0.05 |
| --- | ---: | ---: | ---: | ---: |
| MSP | 0.699 | 0.163 | 0.002 | 0.047 |
| Energy | 0.675 | 0.060 | 0.002 | 0.036 |
| Mahalanobis | 0.723 | 1.000 | 1.000 | 0.031 |

The deliberately selected high-confidence OOD samples expose a known failure mode of softmax-derived scores: a classifier can be confidently wrong far from the training distribution. Feature-space distance catches this synthetic case, while near-OOD remains difficult. The table is a reproducible stress test, not a claim of universal method superiority.

## What it measures

| Signal | Interpretation |
| --- | --- |
| Forgetting events | Correct-to-incorrect transitions between epochs |
| Mean margin (AUM-style) | Assigned-label logit minus the largest alternative logit, averaged over epochs |
| Mean confidence | Average probability assigned to the observed label |
| Variability | Standard deviation of assigned-label confidence |
| Correctness | Fraction of epochs in which the observed label is predicted |
| Mean loss | Average per-sample cross-entropy |
| Suspicion score | Tie-aware weighted rank aggregation for prioritizing manual review |

The implementation reports the raw signals so that users are not forced to rely on the composite score. The `easy-to-learn`, `ambiguous`, and `hard-to-learn` regions are descriptive heuristics, not statistical guarantees.

## Quick start

```bash
git clone https://github.com/RDTAI/FailureLens.git
cd FailureLens
python -m pip install -e '.[demo]'
python examples/demo_synthetic.py
```

The demo creates a three-class dataset, injects 15% label noise, trains a small MLP, and writes:

```text
outputs/
├── metrics.json
├── sample_diagnostics.csv
└── training_dynamics_map.png
```

`sample_diagnostics.csv` is ordered by suspicion score and retains all component metrics for inspection.

## Use it in an existing PyTorch loop

The tracker is framework-light: its core depends only on NumPy and accepts NumPy arrays or PyTorch tensors.

```python
from failurelens import TrainingDynamicsTracker

tracker = TrainingDynamicsTracker(num_examples=len(train_dataset))

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)

    tracker.begin_epoch()
    model.eval()
    for indices, inputs, labels in audit_loader:  # audit_loader must not shuffle indices
        logits = model(inputs)
        tracker.update_batch(indices, logits, labels)
    tracker.end_epoch()

summary = tracker.summarize()
samples_to_review = summary.ranked_indices()[:100]
```

Stable dataset indices are essential. Evaluate all training examples under the same deterministic audit transform after each epoch; stochastic augmentation can otherwise inflate variability.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Responsible interpretation

A high score can indicate an incorrect label, ambiguity, minority-group difficulty, distribution shift, an unsuitable model, or aggressive augmentation. Removing examples solely because they are difficult can erase rare but important cases and worsen subgroup reliability—especially in medical datasets. Use the ranking to support review, not to replace it.

For medical imaging, keep patient-level splits, audit clinically meaningful subgroups separately, and never publish protected patient data or identifiers.

## Research foundations

- Toneva et al. (2019), [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159).
- Pleiss et al. (2020), [Identifying Mislabeled Data using the Area Under the Margin Ranking](https://arxiv.org/abs/2001.10528).
- Swayamdipta et al. (2020), [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746/).
- Hendrycks and Gimpel (2017), [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136).
- Liu et al. (2020), [Energy-based Out-of-distribution Detection](https://papers.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html).

FailureLens is an independent educational implementation inspired by these ideas. It does not reproduce every calibration or thresholding procedure in the original papers.

## Roadmap

- CIFAR-10 label-noise and OOD examples
- Multi-run stability and confidence intervals
- Class-conditional and subgroup diagnostics
- Medical-imaging dataset adapters
- Comparison with loss-only and uncertainty baselines

## License

MIT
