import unittest

import numpy as np

from failurelens.metrics import binary_auroc, detection_metrics
from failurelens.tracker import TrainingDynamicsTracker


class TrackerTests(unittest.TestCase):
    def test_statistics_and_forgetting_events(self):
        tracker = TrainingDynamicsTracker(3)
        labels = np.array([0, 1, 0])
        epoch_logits = [
            np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]),
            np.array([[0.0, 3.0], [0.0, 3.0], [2.0, 0.0]]),
            np.array([[3.0, 0.0], [3.0, 0.0], [2.0, 0.0]]),
        ]
        for logits in epoch_logits:
            tracker.begin_epoch()
            tracker.update_batch(np.arange(3), logits, labels)
            tracker.end_epoch()

        summary = tracker.summarize()
        np.testing.assert_array_equal(summary.forgetting_events, [1, 1, 0])
        np.testing.assert_array_equal(summary.learned_epoch, [0, 0, 0])
        self.assertEqual(summary.ranked_indices().shape, (3,))
        self.assertTrue(np.all((summary.suspicion_score >= 0) & (summary.suspicion_score <= 1)))

    def test_partial_epoch_is_rejected_by_default(self):
        tracker = TrainingDynamicsTracker(2)
        tracker.begin_epoch()
        tracker.update_batch([0], [[2.0, 0.0]], [0])
        with self.assertRaisesRegex(ValueError, "missing 1"):
            tracker.end_epoch()

    def test_duplicate_indices_are_rejected(self):
        tracker = TrainingDynamicsTracker(2)
        tracker.begin_epoch()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tracker.update_batch([0, 0], [[2.0, 0.0], [0.0, 2.0]], [0, 1])


class MetricTests(unittest.TestCase):
    def test_perfect_auroc(self):
        labels = np.array([0, 0, 1, 1], dtype=bool)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(binary_auroc(labels, scores), 1.0)

    def test_detection_metrics(self):
        result = detection_metrics([False, True, False, True], [0.1, 0.9, 0.2, 0.8])
        self.assertEqual(result["precision_at_k"], 1.0)
        self.assertEqual(result["recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()

