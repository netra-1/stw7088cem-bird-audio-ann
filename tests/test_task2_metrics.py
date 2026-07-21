from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import bird_audio.task2_metrics as task2_metrics
from bird_audio.task2_metrics import (
    CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    METRIC_NAMES,
    STABILITY_SEEDS,
    BootstrapReplicates,
    MetricValues,
    ScoredRecording,
    evaluate_novelty_scores,
    session_cluster_bootstrap,
    summarize_across_seeds,
    tie_aware_auroc,
)


def _recording(
    recording_id: str,
    session_group: str,
    species: str,
    source: str,
    score: float,
) -> ScoredRecording:
    return ScoredRecording(
        recording_id=recording_id,
        session_group=session_group,
        species_scientific_name=species,
        source=source,
        score=score,
    )


def _uneven_records() -> tuple[ScoredRecording, ...]:
    return (
        _recording("K1", "known:one", "Corvus splendens", "known", 0.10),
        _recording("K2", "known:one", "Corvus macrorhynchos", "known", 0.20),
        _recording("K3", "known:two", "Acridotheres tristis", "known", 0.80),
        _recording("A1", "alpha:one", "Ceryle rudis", "unknown", 0.40),
        _recording("A2", "alpha:one", "Ceryle rudis", "unknown", 0.90),
        _recording("A3", "alpha:two", "Ceryle rudis", "unknown", 0.80),
        _recording("B1", "beta:one", "Psilopogon zeylanicus", "unknown", 0.20),
        _recording("B2", "beta:two", "Psilopogon zeylanicus", "unknown", 0.70),
        _recording("B3", "beta:two", "Psilopogon zeylanicus", "unknown", 0.70),
        _recording("B4", "beta:two", "Psilopogon zeylanicus", "unknown", 0.95),
    )


def _metric_array(known: np.ndarray, unknown: np.ndarray, threshold: float) -> np.ndarray:
    sensitivity = float(np.count_nonzero(unknown > threshold) / unknown.size)
    specificity = float(np.count_nonzero(known <= threshold) / known.size)
    return np.asarray(
        [
            tie_aware_auroc(known, unknown),
            sensitivity,
            specificity,
            0.5 * (sensitivity + specificity),
        ],
        dtype=np.float64,
    )


class ScoredRecordingTests(unittest.TestCase):
    def test_contract_canonicalizes_numpy_float64(self) -> None:
        value = _recording(
            "XC1",
            "session:1",
            "Corvus splendens",
            "known",
            np.float64(0.25),
        )
        self.assertIs(type(value.score), float)
        self.assertEqual(value.to_record()["score"], 0.25)

    def test_contract_rejects_invalid_identity_source_label_and_score(self) -> None:
        valid = {
            "recording_id": "XC1",
            "session_group": "session:1",
            "species_scientific_name": "Corvus splendens",
            "source": "known",
            "score": 0.25,
        }
        mutations = (
            ("recording_id", ""),
            ("session_group", " session:1"),
            ("species_scientific_name", "house crow"),
            ("species_scientific_name", "Corvus"),
            ("source", "validation"),
            ("score", np.float32(0.25)),
            ("score", 1),
            ("score", float("nan")),
            ("score", float("inf")),
        )
        for key, value in mutations:
            payload = {**valid, key: value}
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                ScoredRecording(**payload)


class PointMetricTests(unittest.TestCase):
    def test_auroc_gives_half_credit_to_ties(self) -> None:
        self.assertEqual(tie_aware_auroc([0.0, 1.0], [1.0, 2.0]), 0.875)
        self.assertEqual(tie_aware_auroc([1.0, 1.0], [1.0, 1.0]), 0.5)
        self.assertEqual(tie_aware_auroc([2.0], [1.0]), 0.0)
        self.assertEqual(tie_aware_auroc([1.0], [2.0]), 1.0)

    def test_auroc_rejects_empty_or_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            tie_aware_auroc([], [1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            tie_aware_auroc([0.0], [float("nan")])
        with self.assertRaises(TypeError):
            tie_aware_auroc([False], [1.0])

    def test_threshold_equality_is_classified_as_known(self) -> None:
        records = (
            _recording("K1", "KS1", "Corvus splendens", "known", 0.50),
            _recording("K2", "KS2", "Corvus splendens", "known", 0.90),
            _recording("U1", "US1", "Ceryle rudis", "unknown", 0.50),
            _recording("U2", "US2", "Ceryle rudis", "unknown", 1.00),
        )
        result = evaluate_novelty_scores(records, 0.50)
        self.assertEqual(result.pooled.values.auroc, 0.625)
        self.assertEqual(result.pooled.values.sensitivity, 0.5)
        self.assertEqual(result.pooled.values.specificity, 0.5)
        self.assertEqual(result.pooled.values.balanced_accuracy, 0.5)

    def test_species_are_sorted_and_macro_is_unweighted_arithmetic_mean(self) -> None:
        result = evaluate_novelty_scores(tuple(reversed(_uneven_records())), 0.70)
        self.assertEqual(
            result.species_scientific_names,
            ("Ceryle rudis", "Psilopogon zeylanicus"),
        )
        self.assertEqual(result.pooled.known_recording_count, 3)
        self.assertEqual(result.pooled.unknown_recording_count, 7)
        for name in METRIC_NAMES:
            expected = float(
                np.mean(
                    np.asarray(
                        [getattr(item.metrics.values, name) for item in result.per_species],
                        dtype=np.float64,
                    )
                )
            )
            self.assertEqual(getattr(result.macro, name), expected)
        self.assertNotEqual(
            result.macro.sensitivity,
            result.pooled.values.sensitivity,
            "Unequal species sizes should distinguish macro and pooled sensitivity",
        )

    def test_metric_values_reject_an_incorrect_balanced_accuracy(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be the mean"):
            MetricValues(
                auroc=0.5,
                sensitivity=0.4,
                specificity=0.8,
                balanced_accuracy=0.7,
            )


class SessionValidationTests(unittest.TestCase):
    def test_duplicate_recording_identity_is_rejected(self) -> None:
        records = list(_uneven_records())
        records[-1] = _recording(
            records[0].recording_id,
            "beta:two",
            "Psilopogon zeylanicus",
            "unknown",
            0.95,
        )
        with self.assertRaisesRegex(ValueError, "globally unique"):
            evaluate_novelty_scores(records, 0.7)

    def test_session_overlap_between_sources_is_rejected(self) -> None:
        records = list(_uneven_records())
        records[3] = _recording("A1", "known:one", "Ceryle rudis", "unknown", 0.4)
        with self.assertRaisesRegex(ValueError, "overlaps known and unknown"):
            session_cluster_bootstrap(records, 0.7, replicate_count=2)

    def test_unknown_session_spanning_species_is_rejected(self) -> None:
        records = list(_uneven_records())
        records[6] = _recording(
            "B1",
            "alpha:one",
            "Psilopogon zeylanicus",
            "unknown",
            0.2,
        )
        with self.assertRaisesRegex(ValueError, "overlaps scientific species"):
            session_cluster_bootstrap(records, 0.7, replicate_count=2)

    def test_known_and_unknown_species_label_overlap_is_rejected(self) -> None:
        records = list(_uneven_records())
        records[3] = _recording("A1", "alpha:one", "Corvus splendens", "unknown", 0.4)
        records[4] = _recording("A2", "alpha:one", "Corvus splendens", "unknown", 0.9)
        records[5] = _recording("A3", "alpha:two", "Corvus splendens", "unknown", 0.8)
        with self.assertRaisesRegex(ValueError, "species labels overlap"):
            evaluate_novelty_scores(records, 0.7)

    def test_record_level_arrays_and_missing_session_contract_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            session_cluster_bootstrap([0.1, 0.9], 0.5, replicate_count=2)
        with self.assertRaisesRegex(ValueError, "session_group"):
            _recording("K1", "", "Corvus splendens", "known", 0.1)


class ClusterBootstrapTests(unittest.TestCase):
    def test_uneven_session_members_are_resampled_together(self) -> None:
        result = session_cluster_bootstrap(_uneven_records(), 0.70, replicate_count=1)
        generator = np.random.Generator(np.random.PCG64(DEFAULT_BOOTSTRAP_SEED))
        known_groups = (
            np.asarray([0.10, 0.20], dtype=np.float64),
            np.asarray([0.80], dtype=np.float64),
        )
        unknown_groups = (
            np.asarray([0.40, 0.90], dtype=np.float64),
            np.asarray([0.80], dtype=np.float64),
            np.asarray([0.20], dtype=np.float64),
            np.asarray([0.70, 0.70, 0.95], dtype=np.float64),
        )
        known_indices = generator.integers(0, len(known_groups), size=len(known_groups))
        unknown_indices = generator.integers(0, len(unknown_groups), size=len(unknown_groups))
        expected_known = np.concatenate([known_groups[int(index)] for index in known_indices])
        expected_unknown = np.concatenate([unknown_groups[int(index)] for index in unknown_indices])
        np.testing.assert_array_equal(
            result.replicates.pooled[0],
            _metric_array(expected_known, expected_unknown, 0.70),
        )

    def test_determinism_and_input_order_invariance(self) -> None:
        records = _uneven_records()
        forward = session_cluster_bootstrap(records, 0.70, replicate_count=40)
        repeated = session_cluster_bootstrap(records, 0.70, replicate_count=40)
        reversed_result = session_cluster_bootstrap(
            tuple(reversed(records)),
            0.70,
            replicate_count=40,
        )
        for name in ("pooled", "per_species", "macro"):
            np.testing.assert_array_equal(
                getattr(forward.replicates, name),
                getattr(repeated.replicates, name),
            )
            np.testing.assert_array_equal(
                getattr(forward.replicates, name),
                getattr(reversed_result.replicates, name),
            )
        self.assertEqual(forward.to_record(), reversed_result.to_record())

    def test_macro_replicates_are_exact_species_row_means(self) -> None:
        result = session_cluster_bootstrap(_uneven_records(), 0.70, replicate_count=50)
        np.testing.assert_array_equal(
            result.replicates.macro,
            np.mean(result.replicates.per_species, axis=1, dtype=np.float64),
        )
        for metric_index, name in enumerate(METRIC_NAMES):
            expected = np.mean(result.replicates.per_species[:, :, metric_index], axis=1)
            np.testing.assert_array_equal(result.replicates.macro[:, metric_index], expected)
            point_expected = float(
                np.mean(
                    np.asarray(
                        [
                            getattr(item.metrics.values, name)
                            for item in result.point_estimates.per_species
                        ],
                        dtype=np.float64,
                    )
                )
            )
            self.assertEqual(getattr(result.point_estimates.macro, name), point_expected)

    def test_each_replicate_reuses_one_known_session_resample(self) -> None:
        replicate_count = 7
        species_count = 2
        original = task2_metrics._resample_session_scores
        with mock.patch.object(
            task2_metrics,
            "_resample_session_scores",
            wraps=original,
        ) as sampler:
            session_cluster_bootstrap(
                _uneven_records(),
                0.70,
                replicate_count=replicate_count,
            )
        expected_calls_per_replicate = 2 + species_count
        self.assertEqual(
            sampler.call_count,
            replicate_count * expected_calls_per_replicate,
        )

    def test_intervals_have_finite_ordered_95_percent_bounds(self) -> None:
        result = session_cluster_bootstrap(_uneven_records(), 0.70, replicate_count=80)
        self.assertEqual(result.to_record()["interval_method"], "percentile")
        interval_sets = [
            result.pooled_intervals,
            result.macro_intervals,
            *(item.intervals for item in result.per_species_intervals),
        ]
        for interval_set in interval_sets:
            for name in METRIC_NAMES:
                interval = getattr(interval_set, name)
                self.assertEqual(interval.confidence_level, CONFIDENCE_LEVEL)
                self.assertTrue(math.isfinite(interval.lower))
                self.assertTrue(math.isfinite(interval.upper))
                self.assertLessEqual(interval.lower, interval.upper)
                self.assertGreaterEqual(interval.lower, 0.0)
                self.assertLessEqual(interval.upper, 1.0)

    def test_default_produces_exactly_2000_valid_replicates(self) -> None:
        records = (
            _recording("K1", "KS1", "Corvus splendens", "known", 0.1),
            _recording("K2", "KS2", "Corvus splendens", "known", 0.2),
            _recording("U1", "US1", "Ceryle rudis", "unknown", 0.8),
            _recording("U2", "US2", "Ceryle rudis", "unknown", 0.9),
        )
        result = session_cluster_bootstrap(records, 0.5)
        self.assertEqual(result.replicates.replicate_count, DEFAULT_BOOTSTRAP_REPLICATES)
        self.assertEqual(result.replicates.seed, DEFAULT_BOOTSTRAP_SEED)
        self.assertEqual(result.replicates.pooled.shape, (2000, 4))
        self.assertTrue(np.all(np.isfinite(result.replicates.pooled)))

    def test_replicates_are_read_only_and_safe_for_compressed_archive(self) -> None:
        result = session_cluster_bootstrap(_uneven_records(), 0.70, replicate_count=10)
        with self.assertRaises(ValueError):
            result.replicates.pooled[0, 0] = 0.0
        payload = result.replicates.to_npz_payload()
        self.assertTrue(all(value.dtype != np.dtype(object) for value in payload.values()))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "replicates.npz"
            np.savez_compressed(destination, **payload)
            with np.load(destination, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["pooled"], result.replicates.pooled)
                np.testing.assert_array_equal(
                    archive["per_species"],
                    result.replicates.per_species,
                )
                np.testing.assert_array_equal(archive["macro"], result.replicates.macro)
                self.assertEqual(int(archive["seed"]), DEFAULT_BOOTSTRAP_SEED)

    def test_invalid_bootstrap_controls_and_array_contract_are_rejected(self) -> None:
        records = _uneven_records()
        with self.assertRaises(TypeError):
            session_cluster_bootstrap(records, 0.7, replicate_count=True)
        with self.assertRaises(ValueError):
            session_cluster_bootstrap(records, 0.7, replicate_count=0)
        with self.assertRaisesRegex(ValueError, "20260713"):
            session_cluster_bootstrap(records, 0.7, replicate_count=2, seed=1)
        with self.assertRaises(TypeError):
            BootstrapReplicates(
                pooled=np.zeros((1, 4), dtype=np.float32),
                per_species=np.zeros((1, 1, 4), dtype=np.float64),
                macro=np.zeros((1, 4), dtype=np.float64),
                species_scientific_names=("Ceryle rudis",),
                replicate_count=1,
            )


class SeedSummaryTests(unittest.TestCase):
    def test_exact_seeds_arithmetic_mean_and_sample_standard_deviation(self) -> None:
        result = summarize_across_seeds(
            {
                13: {"auroc": 0.2, "balanced_accuracy": 0.6},
                37: {"auroc": 0.4, "balanced_accuracy": 0.7},
                71: {"auroc": 0.6, "balanced_accuracy": 0.8},
            }
        )
        self.assertEqual(tuple(item.metric_name for item in result), ("auroc", "balanced_accuracy"))
        auroc = result[0]
        self.assertEqual(auroc.seeds, STABILITY_SEEDS)
        self.assertTrue(math.isclose(auroc.mean, 0.4, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(
            math.isclose(
                auroc.sample_standard_deviation,
                0.2,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertEqual(auroc.to_record()["standard_deviation_ddof"], 1)

    def test_seed_summary_rejects_wrong_seed_or_metric_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            summarize_across_seeds({13: {"auroc": 0.5}, 37: {"auroc": 0.6}})
        with self.assertRaisesRegex(ValueError, "identical metric names"):
            summarize_across_seeds(
                {
                    13: {"auroc": 0.5},
                    37: {"auroc": 0.6, "sensitivity": 0.7},
                    71: {"auroc": 0.7},
                }
            )
        with self.assertRaises(TypeError):
            summarize_across_seeds(
                {
                    13: {"auroc": np.float32(0.5)},
                    37: {"auroc": 0.6},
                    71: {"auroc": 0.7},
                }
            )


if __name__ == "__main__":
    unittest.main()
