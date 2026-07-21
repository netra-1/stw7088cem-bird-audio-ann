from __future__ import annotations

import io
import math
import unittest

import numpy as np

from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.task1_final_metrics import (
    CLASS_COUNT,
    CLASS_ORDER,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    STABILITY_SEEDS,
    RecordingPrediction,
    evaluate_recording_predictions,
    session_cluster_bootstrap_seed37,
    summarize_stability,
)


def _prediction(
    recording_id: str,
    session_group: str,
    true_class_index: int,
    predicted_class_index: int,
) -> RecordingPrediction:
    logits = [-5.0] * CLASS_COUNT
    logits[predicted_class_index] = 5.0
    return RecordingPrediction(
        recording_id=recording_id,
        session_group=session_group,
        true_class_index=true_class_index,
        mean_logits=tuple(logits),
        predicted_class_index=predicted_class_index,
    )


def _bootstrap_fixture() -> tuple[RecordingPrediction, ...]:
    anchor = tuple(
        _prediction(
            f"anchor-{class_index:02d}",
            "session-a",
            class_index,
            class_index,
        )
        for class_index in range(CLASS_COUNT)
    )
    uneven = tuple(_prediction(f"uneven-{index:02d}", "session-b", 0, 1) for index in range(3))
    return (*anchor, *uneven)


class RecordingPredictionContractTests(unittest.TestCase):
    def test_locked_class_order_and_argmax_binding_are_exact(self) -> None:
        self.assertEqual(CLASS_ORDER, LOCKED_TASK1_CLASS_ORDER)
        self.assertEqual(len(CLASS_ORDER), 15)
        logits = [-2.0] * CLASS_COUNT
        logits[2] = 4.0
        logits[3] = 4.0
        prediction = RecordingPrediction(
            recording_id="XC1",
            session_group="session:one",
            true_class_index=2,
            mean_logits=tuple(logits),
            predicted_class_index=2,
        )
        self.assertEqual(prediction.predicted_class_index, 2)
        self.assertEqual(prediction.to_record()["predicted_class_name"], CLASS_ORDER[2])

        with self.assertRaisesRegex(ValueError, "argmax"):
            RecordingPrediction(
                recording_id="XC2",
                session_group="session:two",
                true_class_index=2,
                mean_logits=tuple(logits),
                predicted_class_index=3,
            )

    def test_contract_rejects_invalid_identity_index_and_logits(self) -> None:
        valid = {
            "recording_id": "XC1",
            "session_group": "session:one",
            "true_class_index": 0,
            "mean_logits": tuple([1.0, *([-1.0] * (CLASS_COUNT - 1))]),
            "predicted_class_index": 0,
        }
        mutations = (
            ("recording_id", ""),
            ("session_group", " session:one"),
            ("true_class_index", True),
            ("true_class_index", CLASS_COUNT),
            ("mean_logits", tuple([1.0] * (CLASS_COUNT - 1))),
            ("mean_logits", tuple([float("nan"), *([0.0] * (CLASS_COUNT - 1))])),
            ("mean_logits", tuple([np.float32(1.0), *([0.0] * (CLASS_COUNT - 1))])),
        )
        for field, value in mutations:
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                RecordingPrediction(**{**valid, field: value})

    def test_evaluation_rejects_duplicate_recording_identity(self) -> None:
        prediction = _prediction("XC1", "session:one", 0, 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_recording_predictions((prediction, prediction))


class ClassificationArithmeticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = (
            _prediction("R0A", "S0A", 0, 0),
            _prediction("R0B", "S0B", 0, 1),
            _prediction("R1", "S1", 1, 1),
            _prediction("R2", "S2", 2, 1),
        )

    def test_confusion_precision_recall_f1_and_macro_arithmetic(self) -> None:
        result = evaluate_recording_predictions(self.predictions)
        expected = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        expected[0, 0] = 1
        expected[0, 1] = 1
        expected[1, 1] = 1
        expected[2, 1] = 1
        np.testing.assert_array_equal(result.confusion_counts, expected)
        self.assertEqual(result.recording_count, 4)
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.per_class[0].support, 2)
        self.assertEqual(result.per_class[0].precision, 1.0)
        self.assertEqual(result.per_class[0].recall, 0.5)
        self.assertEqual(result.per_class[0].f1, 2.0 / 3.0)
        self.assertEqual(result.per_class[1].support, 1)
        self.assertEqual(result.per_class[1].precision, 1.0 / 3.0)
        self.assertEqual(result.per_class[1].recall, 1.0)
        self.assertEqual(result.per_class[1].f1, 0.5)
        self.assertAlmostEqual(result.macro_f1, 7.0 / 90.0)

    def test_zero_division_and_row_normalization_are_locked(self) -> None:
        result = evaluate_recording_predictions(self.predictions)
        np.testing.assert_array_equal(
            result.row_normalized_confusion[0, :3],
            np.asarray([0.5, 0.5, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            result.row_normalized_confusion[1, :3],
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            result.row_normalized_confusion[2, :3],
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            result.row_normalized_confusion[3],
            np.zeros(CLASS_COUNT, dtype=np.float64),
        )
        self.assertEqual(result.per_class[3].support, 0)
        self.assertEqual(result.per_class[3].precision, 0.0)
        self.assertEqual(result.per_class[3].recall, 0.0)
        self.assertEqual(result.per_class[3].f1, 0.0)
        self.assertEqual(result.to_record()["zero_division"], 0)

    def test_metric_arrays_are_defensive_and_read_only(self) -> None:
        result = evaluate_recording_predictions(self.predictions)
        self.assertFalse(result.confusion_counts.flags.writeable)
        self.assertFalse(result.row_normalized_confusion.flags.writeable)
        with self.assertRaises(ValueError):
            result.confusion_counts[0, 0] = 99
        with self.assertRaises(ValueError):
            result.row_normalized_confusion[0, 0] = 0.0


class StabilitySummaryTests(unittest.TestCase):
    def test_exact_seed_order_mean_and_sample_standard_deviation(self) -> None:
        seed_metrics = {
            13: {"accuracy": 0.50, "macro_f1": 0.20},
            37: {"accuracy": 0.75, "macro_f1": 0.40},
            71: {"accuracy": 1.00, "macro_f1": 0.80},
        }
        result = summarize_stability(seed_metrics)
        self.assertEqual(result.accuracy.seeds, STABILITY_SEEDS)
        self.assertEqual(result.accuracy.values, (0.50, 0.75, 1.00))
        self.assertEqual(result.accuracy.mean, 0.75)
        self.assertEqual(result.accuracy.sample_standard_deviation, 0.25)
        expected_macro = np.asarray([0.20, 0.40, 0.80], dtype=np.float64)
        self.assertEqual(result.macro_f1.mean, float(np.mean(expected_macro)))
        self.assertEqual(
            result.macro_f1.sample_standard_deviation,
            float(np.std(expected_macro, ddof=1)),
        )
        self.assertEqual(result.to_record()["accuracy"]["standard_deviation_ddof"], 1)

    def test_summary_requires_exact_seeds_and_metric_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            summarize_stability(
                {
                    13: {"accuracy": 0.5, "macro_f1": 0.5},
                    37: {"accuracy": 0.5, "macro_f1": 0.5},
                }
            )
        with self.assertRaisesRegex(ValueError, "exactly accuracy"):
            summarize_stability(
                {seed: {"accuracy": 0.5, "macro_f1": 0.5, "extra": 0.5} for seed in STABILITY_SEEDS}
            )


class SessionClusterBootstrapTests(unittest.TestCase):
    def test_uneven_sessions_are_resampled_whole_and_missing_classes_are_redrawn(self) -> None:
        result = session_cluster_bootstrap_seed37(
            _bootstrap_fixture(),
            task1_seed=37,
            replicate_count=2,
            maximum_attempts=5,
        )
        self.assertEqual(result.replicates.attempts, 3)
        self.assertEqual(result.replicates.rejected_attempts, 1)
        np.testing.assert_array_equal(
            result.replicates.recording_counts,
            np.asarray([30, 18], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result.replicates.accuracy,
            np.asarray([1.0, 15.0 / 18.0], dtype=np.float64),
        )

    def test_replicates_are_deterministic_and_input_order_invariant(self) -> None:
        predictions = _bootstrap_fixture()
        forward = session_cluster_bootstrap_seed37(
            predictions,
            task1_seed=37,
            replicate_count=40,
            maximum_attempts=100,
        )
        repeated = session_cluster_bootstrap_seed37(
            predictions,
            task1_seed=37,
            replicate_count=40,
            maximum_attempts=100,
        )
        reversed_result = session_cluster_bootstrap_seed37(
            tuple(reversed(predictions)),
            task1_seed=37,
            replicate_count=40,
            maximum_attempts=100,
        )
        for name in ("accuracy", "macro_f1", "per_class_f1", "recording_counts"):
            np.testing.assert_array_equal(
                getattr(forward.replicates, name),
                getattr(repeated.replicates, name),
            )
            np.testing.assert_array_equal(
                getattr(forward.replicates, name),
                getattr(reversed_result.replicates, name),
            )
        self.assertEqual(forward.replicates.attempts, repeated.replicates.attempts)
        self.assertEqual(forward.to_record(), reversed_result.to_record())

    def test_maximum_attempt_guard_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "maximum_attempts exhausted"):
            session_cluster_bootstrap_seed37(
                _bootstrap_fixture(),
                task1_seed=37,
                replicate_count=2,
                maximum_attempts=2,
            )

    def test_bootstrap_requires_seed37_locked_rng_and_all_classes(self) -> None:
        predictions = _bootstrap_fixture()
        with self.assertRaisesRegex(ValueError, "seed 37 only"):
            session_cluster_bootstrap_seed37(
                predictions,
                task1_seed=13,
                replicate_count=1,
            )
        with self.assertRaisesRegex(ValueError, "20260713"):
            session_cluster_bootstrap_seed37(
                predictions,
                task1_seed=37,
                replicate_count=1,
                bootstrap_seed=1,
            )
        with self.assertRaisesRegex(ValueError, "every locked true class"):
            session_cluster_bootstrap_seed37(
                predictions[:-4],
                task1_seed=37,
                replicate_count=1,
            )

    def test_default_2000_replicates_are_read_only_and_npz_safe(self) -> None:
        predictions = tuple(
            _prediction(f"recording-{index:02d}", "one-session", index, index)
            for index in range(CLASS_COUNT)
        )
        result = session_cluster_bootstrap_seed37(predictions, task1_seed=37)
        replicates = result.replicates
        self.assertEqual(replicates.replicate_count, DEFAULT_BOOTSTRAP_REPLICATES)
        self.assertEqual(replicates.bootstrap_seed, DEFAULT_BOOTSTRAP_SEED)
        self.assertEqual(replicates.attempts, DEFAULT_BOOTSTRAP_REPLICATES)
        self.assertEqual(replicates.accuracy.shape, (DEFAULT_BOOTSTRAP_REPLICATES,))
        self.assertEqual(
            replicates.per_class_f1.shape,
            (DEFAULT_BOOTSTRAP_REPLICATES, CLASS_COUNT),
        )
        self.assertFalse(replicates.accuracy.flags.writeable)
        self.assertFalse(replicates.macro_f1.flags.writeable)
        self.assertFalse(replicates.per_class_f1.flags.writeable)
        self.assertFalse(replicates.recording_counts.flags.writeable)
        self.assertEqual(result.accuracy_interval.lower, 1.0)
        self.assertEqual(result.accuracy_interval.upper, 1.0)
        self.assertEqual(result.macro_f1_interval.lower, 1.0)
        self.assertEqual(result.macro_f1_interval.upper, 1.0)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **replicates.to_npz_payload())
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(replicates.to_npz_payload()))
            self.assertTrue(all(archive[name].dtype != np.dtype(object) for name in archive.files))
            np.testing.assert_array_equal(archive["accuracy"], replicates.accuracy)

    def test_bootstrap_macro_is_exact_mean_of_per_class_f1(self) -> None:
        result = session_cluster_bootstrap_seed37(
            _bootstrap_fixture(),
            task1_seed=37,
            replicate_count=10,
            maximum_attempts=30,
        )
        np.testing.assert_array_equal(
            result.replicates.macro_f1,
            np.mean(result.replicates.per_class_f1, axis=1, dtype=np.float64),
        )
        self.assertTrue(math.isfinite(result.macro_f1_interval.lower))
        self.assertTrue(math.isfinite(result.macro_f1_interval.upper))


if __name__ == "__main__":
    unittest.main()
