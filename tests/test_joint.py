import unittest

import numpy as np

from failurelens.joint import false_alarm_association, local_training_signal


class LocalSignalTests(unittest.TestCase):
    def test_nearest_training_region_transfers_signal(self) -> None:
        train_x = np.array([[0.0], [1.0], [10.0], [11.0]])
        train_signal = np.array([0.0, 0.0, 5.0, 5.0])
        query_x = np.array([[0.2], [10.2]])
        result = local_training_signal(train_x, train_signal, query_x, k=2)
        np.testing.assert_allclose(result, [0.0, 5.0])

    def test_exact_match_is_stable(self) -> None:
        result = local_training_signal(
            np.array([[0.0], [0.0], [2.0]]),
            np.array([2.0, 4.0, 100.0]),
            np.array([[0.0]]),
            k=3,
        )
        self.assertAlmostEqual(float(result[0]), 3.0)


class AssociationTests(unittest.TestCase):
    def test_detects_higher_false_alarm_risk_in_high_signal_region(self) -> None:
        signal = np.arange(100, dtype=float)
        alarms = signal >= 75
        result = false_alarm_association(
            signal,
            alarms,
            bootstrap_samples=100,
            permutations=100,
            seed=4,
        )
        self.assertEqual(result["low_region_false_alarm_rate"], 0.0)
        self.assertEqual(result["high_region_false_alarm_rate"], 1.0)
        self.assertEqual(result["risk_difference"], 1.0)
        self.assertEqual(result["false_alarm_auroc"], 1.0)

    def test_rejects_invalid_quantiles(self) -> None:
        with self.assertRaises(ValueError):
            false_alarm_association(
                np.arange(10),
                np.zeros(10),
                low_quantile=0.8,
                high_quantile=0.2,
            )


if __name__ == "__main__":
    unittest.main()
