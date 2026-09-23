# Training Forgetting and OOD False Alarms

This experiment asks whether ID test samples located near frequently forgotten
training examples are more likely to be falsely flagged as OOD.

## Estimand

For every ID test example, FailureLens estimates a local forgetting burden by
inverse-distance-weighted k-nearest-neighbour interpolation of training-sample
forgetting counts. The primary contrast is:

```text
FPR(high local forgetting region) - FPR(low local forgetting region)
```

The default high and low regions are defined by the 75th and 25th percentiles.
Ties are retained, so a region can contain more than exactly 25% of test samples.

## Leakage-safe split

1. The training set fits the classifier and supplies forgetting counts.
2. A disjoint ID calibration set converts each OOD score into conformal p-values.
3. A disjoint ID test set estimates false-alarm associations.

No ID test labels or alarms are used to construct the local forgetting burden or
calibrate OOD thresholds.

## Association statistics

For each detector, the report includes:

- overall ID false-alarm rate;
- false-alarm rates in the low and high forgetting regions;
- absolute risk difference and a percentile bootstrap 95% interval;
- a half-count-corrected risk ratio, which remains finite when a region has zero alarms;
- AUROC for using local forgetting burden to rank false alarms;
- a two-sided permutation p-value for the high-minus-low risk difference.

The bootstrap resamples alarms independently within the fixed low and high
regions. The permutation test keeps regional scores fixed and shuffles alarm
labels. These procedures quantify sampling uncertainty conditional on this
trained model and dataset; they do not capture variability across training seeds.

## Interpretation limits

This is an association study, not a causal analysis. Output-based OOD scores and
training forgetting both respond to decision-boundary uncertainty, so their
association can arise from a shared cause. Results should be repeated across
seeds, architectures, neighbourhood sizes, score definitions, and datasets.

For medical imaging, construct neighbourhoods in a validated representation,
split by patient, and audit site/scanner/demographic composition. Clustered
bootstrap or hierarchical models are preferable when multiple images belong to
the same patient.

## Run

```bash
python examples/demo_forgetting_ood.py
```

The command produces a JSON statistical report, a per-test-sample CSV table, and
a three-panel diagnostic figure under `outputs/forgetting_ood/`.
