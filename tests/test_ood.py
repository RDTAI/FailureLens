import unittest

import numpy as np

from failurelens.ood import (
    ConformalOODCalibrator,
    MahalanobisOODDetector,
    energy_score,
    entropy_score,
    msp_score,
    ood_detection_metrics,
    probability_margin_score,
)


class ScoreTests(unittest.TestCase):
    def test_uncertain_logits_receive_larger_probability_scores(self):
        confident = np.array([[8.0, 0.0, -1.0]])
        uncertain = np.array([[0.0, 0.0, 0.0]])
        self.assertLess(msp_score(confident)[0], msp_score(uncertain)[0])
        self.assertLess(entropy_score(confident)[0], entropy_score(uncertain)[0])
        self.assertLess(
            probability_margin_score(confident)[0], probability_margin_score(uncertain)[0]
        )

    def test_energy_matches_logsumexp_definition(self):
        logits = np.array([[0.0, 0.0], [2.0, 1.0]])
        expected = -np.log(np.exp(logits).sum(axis=1))
        np.testing.assert_allclose(energy_score(logits), expected)

    def test_invalid_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            msp_score([[1.0, 0.0]], temperature=0)

    def test_mahalanobis_distance_detects_far_features(self):
        features = np.array([[-1.1, 0.0], [-0.9, 0.0], [0.9, 0.0], [1.1, 0.0]])
        labels = np.array([0, 0, 1, 1])
        detector = MahalanobisOODDetector.fit(features, labels, regularization=0.01)
        scores = detector.score(np.array([[-1.0, 0.0], [8.0, 8.0]]))
        self.assertLess(scores[0], scores[1])


class ConformalTests(unittest.TestCase):
    def test_p_values_use_conservative_upper_tail_rank(self):
        calibrator = ConformalOODCalibrator([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(calibrator.p_values([0.05, 0.25, 0.5]), [1.0, 0.6, 0.2])
        self.assertEqual(calibrator.minimum_p_value(), 0.2)

    def test_rejection_requires_valid_alpha(self):
        calibrator = ConformalOODCalibrator([0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "alpha"):
            calibrator.reject([0.3], alpha=1.0)


class OODMetricTests(unittest.TestCase):
    def test_perfect_separation(self):
        labels = [False, False, True, True]
        scores = [0.1, 0.2, 0.8, 0.9]
        result = ood_detection_metrics(labels, scores)
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["aupr_ood"], 1.0)
        self.assertEqual(result["fpr_at_95_tpr"], 0.0)


if __name__ == "__main__":
    unittest.main()
