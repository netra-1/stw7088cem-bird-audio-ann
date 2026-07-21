from __future__ import annotations

import json
import math
import unittest

import numpy as np

from bird_audio.task2_scoring import (
    KNOWN_TRAINING_ROLE,
    KNOWN_VALIDATION_ROLE,
    LATENT_SCORE_NAME,
    NEAREST_NEIGHBOURS,
    NOVELTY_DIRECTION,
    RECONSTRUCTION_SCORE_NAME,
    THRESHOLD_QUANTILE,
    THRESHOLD_QUANTILE_METHOD,
    ClipIdentity,
    LatentNoveltyScore,
    NoveltyDecision,
    NoveltyThreshold,
    ReconstructionThreshold,
    RecordingBatch,
    RecordingScore,
    aggregate_recordings,
    classify_latent_scores,
    classify_reconstruction_scores,
    clip_reconstruction_mse,
    fit_known_training_reference,
    fit_known_validation_latent_threshold,
    fit_known_validation_reconstruction_threshold,
    fit_known_validation_threshold,
    latent_knn_novelty_scores,
    score_recording_batch,
)


def _recording(
    recording_id: str,
    embedding: tuple[float, ...],
    reconstruction_mse: float = 0.0,
    *,
    clip_ids: tuple[str, ...] = ("clip-0",),
) -> RecordingScore:
    return RecordingScore(
        recording_id=recording_id,
        clip_ids=clip_ids,
        reconstruction_mse=reconstruction_mse,
        mean_latent_embedding=embedding,
    )


def _batch(
    source_role: str,
    entries: list[tuple[str, tuple[float, ...], float]],
) -> RecordingBatch:
    return RecordingBatch(
        source_role=source_role,
        recordings=tuple(
            sorted(
                (
                    _recording(recording_id, embedding, score)
                    for recording_id, embedding, score in entries
                ),
                key=lambda recording: recording.recording_id,
            )
        ),
    )


def _latent_score(
    recording_id: str,
    value: float,
    *,
    neighbour_recording_ids: tuple[str, ...] | None = None,
) -> LatentNoveltyScore:
    neighbours = neighbour_recording_ids or tuple(f"training-{index:02d}" for index in range(10))
    return LatentNoveltyScore(
        recording_id=recording_id,
        score=value,
        neighbour_recording_ids=neighbours,
        neighbour_distances=(value,) * 10,
    )


class Task2ScoringArithmeticTests(unittest.TestCase):
    def test_clip_reconstruction_mse_is_pixel_mean_float64(self) -> None:
        inputs = np.zeros((2, 1, 1, 2), dtype=np.float32)
        reconstructions = np.asarray(
            [
                [[[1.0, 2.0]]],
                [[[0.0, 2.0]]],
            ],
            dtype=np.float32,
        )

        result = clip_reconstruction_mse(inputs, reconstructions)

        np.testing.assert_array_equal(result, np.asarray([2.5, 2.0], dtype=np.float64))
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.shape, (2,))
        self.assertFalse(result.flags.writeable)

    def test_recording_grouping_uses_even_median_and_mean_latent(self) -> None:
        identities = (
            ClipIdentity("recording-b", "clip-2"),
            ClipIdentity("recording-a", "clip-2"),
            ClipIdentity("recording-b", "clip-1"),
            ClipIdentity("recording-a", "clip-1"),
        )
        clip_scores = np.asarray([4.0, 9.0, 0.0, 1.0], dtype=np.float32)
        latent = np.asarray(
            [
                [4.0, 6.0],
                [9.0, 1.0],
                [0.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=np.float32,
        )

        result = aggregate_recordings(
            identities,
            clip_scores,
            latent,
            source_role="evaluation",
        )

        self.assertEqual(result.recording_ids, ("recording-a", "recording-b"))
        self.assertEqual(result.recordings[0].clip_ids, ("clip-1", "clip-2"))
        self.assertEqual(result.recordings[0].clip_count, 2)
        self.assertEqual(result.recordings[0].reconstruction_mse, 5.0)
        self.assertEqual(result.recordings[0].mean_latent_embedding, (5.0, 2.0))
        self.assertEqual(result.recordings[1].reconstruction_mse, 2.0)
        self.assertEqual(result.recordings[1].mean_latent_embedding, (2.0, 4.0))

    def test_grouping_is_deterministic_under_bound_row_permutation(self) -> None:
        identities = np.asarray(
            [
                ClipIdentity("r-b", "c-2"),
                ClipIdentity("r-a", "c-1"),
                ClipIdentity("r-a", "c-2"),
                ClipIdentity("r-b", "c-1"),
            ],
            dtype=object,
        )
        scores = np.asarray([8.0, 2.0, 4.0, 6.0], dtype=np.float64)
        latent = np.asarray([[8.0], [2.0], [4.0], [6.0]], dtype=np.float64)
        reference = aggregate_recordings(
            tuple(identities),
            scores,
            latent,
            source_role="evaluation",
        )

        permutation = np.asarray([2, 0, 3, 1])
        permuted = aggregate_recordings(
            tuple(identities[permutation]),
            scores[permutation],
            latent[permutation],
            source_role="evaluation",
        )

        self.assertEqual(permuted, reference)

    def test_score_recording_batch_composes_clip_and_recording_scoring(self) -> None:
        identities = (
            ClipIdentity("r-a", "c-1"),
            ClipIdentity("r-a", "c-2"),
        )
        inputs = np.zeros((2, 1, 1, 1), dtype=np.float64)
        reconstructions = np.asarray([[[[1.0]]], [[[3.0]]]], dtype=np.float64)
        latent = np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float64)

        result = score_recording_batch(
            identities,
            inputs,
            reconstructions,
            latent,
            source_role="evaluation",
        )

        self.assertEqual(result.recordings[0].reconstruction_mse, 5.0)
        self.assertEqual(result.recordings[0].mean_latent_embedding, (2.0, 4.0))


class Task2LatentReferenceTests(unittest.TestCase):
    def test_fit_uses_float64_population_variance_and_zero_variance_scale_one(self) -> None:
        training = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (float(index), 7.0), 0.0) for index in range(10)],
        )

        reference = fit_known_training_reference(training)

        self.assertEqual(reference.fit_role, KNOWN_TRAINING_ROLE)
        self.assertEqual(reference.recording_ids, training.recording_ids)
        self.assertEqual(reference.coordinate_mean, (4.5, 7.0))
        self.assertEqual(reference.population_variance, (8.25, 0.0))
        self.assertEqual(reference.coordinate_scale, (math.sqrt(8.25), 1.0))
        self.assertTrue(all(row[1] == 0.0 for row in reference.standardized_embeddings))
        standardized_first = np.asarray(
            [row[0] for row in reference.standardized_embeddings],
            dtype=np.float64,
        )
        self.assertAlmostEqual(float(standardized_first.mean()), 0.0)
        self.assertAlmostEqual(float(np.mean(standardized_first**2)), 1.0)

    def test_brute_force_score_uses_exactly_ten_nearest_training_recordings(self) -> None:
        training = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (float(index),), 0.0) for index in range(12)],
        )
        reference = fit_known_training_reference(training)
        queries = _batch("evaluation", [("query", (20.0,), 0.0)])

        result = latent_knn_novelty_scores(reference, queries)[0]

        scale = math.sqrt(float(np.var(np.arange(12, dtype=np.float64), ddof=0)))
        self.assertEqual(
            result.neighbour_recording_ids, tuple(f"r-{i:02d}" for i in range(11, 1, -1))
        )
        self.assertEqual(len(result.neighbour_recording_ids), NEAREST_NEIGHBOURS)
        self.assertAlmostEqual(result.score, 13.5 / scale)
        np.testing.assert_allclose(
            result.neighbour_distances,
            np.arange(9.0, 19.0, dtype=np.float64) / scale,
        )
        self.assertEqual(result.direction, NOVELTY_DIRECTION)

    def test_distance_ties_resolve_by_recording_identity_deterministically(self) -> None:
        training = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (5.0, 5.0), 0.0) for index in reversed(range(12))],
        )
        reference = fit_known_training_reference(training)
        queries = _batch("evaluation", [("query", (5.0, 5.0), 0.0)])

        first = latent_knn_novelty_scores(reference, queries)
        second = latent_knn_novelty_scores(reference, queries)

        self.assertEqual(first, second)
        self.assertEqual(
            first[0].neighbour_recording_ids,
            tuple(f"r-{index:02d}" for index in range(10)),
        )
        self.assertEqual(first[0].neighbour_distances, (0.0,) * 10)
        self.assertEqual(first[0].score, 0.0)

    def test_fit_and_query_reject_leakage(self) -> None:
        evaluation = _batch(
            "evaluation",
            [(f"r-{index:02d}", (float(index),), 0.0) for index in range(10)],
        )
        with self.assertRaisesRegex(PermissionError, "known_training"):
            fit_known_training_reference(evaluation)

        training = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (float(index),), 0.0) for index in range(10)],
        )
        reference = fit_known_training_reference(training)
        overlapping_query = _batch("evaluation", [("r-05", (99.0,), 0.0)])
        with self.assertRaisesRegex(PermissionError, "overlap"):
            latent_knn_novelty_scores(reference, overlapping_query)

    def test_fit_requires_ten_recordings_and_query_dimensions_must_match(self) -> None:
        too_small = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (float(index),), 0.0) for index in range(9)],
        )
        with self.assertRaisesRegex(ValueError, "at least 10"):
            fit_known_training_reference(too_small)

        training = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (float(index),), 0.0) for index in range(10)],
        )
        reference = fit_known_training_reference(training)
        wrong_dimensions = _batch("evaluation", [("query", (1.0, 2.0), 0.0)])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            latent_knn_novelty_scores(reference, wrong_dimensions)


class Task2ThresholdTests(unittest.TestCase):
    def test_each_locked_score_fits_its_own_higher_quantile_threshold(self) -> None:
        validation = _batch(
            KNOWN_VALIDATION_ROLE,
            [(f"r-{index:02d}", (0.0,), float(index * 2)) for index in range(20)],
        )
        latent_scores = tuple(
            _latent_score(recording_id, float(index))
            for index, recording_id in enumerate(validation.recording_ids)
        )

        reconstruction = fit_known_validation_reconstruction_threshold(validation)
        compatibility = fit_known_validation_threshold(validation)
        latent = fit_known_validation_latent_threshold(validation, latent_scores)

        self.assertEqual(reconstruction, compatibility)
        self.assertEqual(reconstruction.score_name, RECONSTRUCTION_SCORE_NAME)
        self.assertEqual(reconstruction.value, 38.0)
        self.assertEqual(latent.score_name, LATENT_SCORE_NAME)
        self.assertEqual(latent.value, 19.0)
        self.assertEqual(latent.calibration_recording_ids, validation.recording_ids)
        self.assertEqual(latent.quantile, THRESHOLD_QUANTILE)
        self.assertEqual(latent.method, THRESHOLD_QUANTILE_METHOD)
        self.assertEqual(latent.classification_operator, ">")

    def test_strict_threshold_equality_is_known_for_both_scores(self) -> None:
        validation = _batch(
            KNOWN_VALIDATION_ROLE,
            [(f"v-{index:02d}", (0.0,), float(index)) for index in range(20)],
        )
        validation_latent = tuple(
            _latent_score(recording_id, float(index))
            for index, recording_id in enumerate(validation.recording_ids)
        )
        reconstruction_threshold = fit_known_validation_reconstruction_threshold(validation)
        latent_threshold = fit_known_validation_latent_threshold(validation, validation_latent)

        evaluation = _batch(
            "evaluation",
            [
                ("above", (0.0,), 20.0),
                ("below", (0.0,), 18.0),
                ("equal", (0.0,), 19.0),
            ],
        )
        decisions = {
            decision.recording_id: decision
            for decision in classify_reconstruction_scores(evaluation, reconstruction_threshold)
        }
        self.assertTrue(decisions["above"].is_novel)
        self.assertFalse(decisions["below"].is_novel)
        self.assertFalse(decisions["equal"].is_novel)
        self.assertEqual(decisions["equal"].direction, NOVELTY_DIRECTION)
        self.assertEqual(decisions["equal"].score_name, RECONSTRUCTION_SCORE_NAME)

        evaluation_latent = tuple(
            _latent_score(recording_id, value)
            for recording_id, value in (
                ("above", 20.0),
                ("below", 18.0),
                ("equal", 19.0),
            )
        )
        latent_decisions = {
            decision.recording_id: decision
            for decision in classify_latent_scores(
                evaluation,
                evaluation_latent,
                latent_threshold,
            )
        }
        self.assertTrue(latent_decisions["above"].is_novel)
        self.assertFalse(latent_decisions["below"].is_novel)
        self.assertFalse(latent_decisions["equal"].is_novel)
        self.assertEqual(latent_decisions["equal"].score_name, LATENT_SCORE_NAME)

    def test_both_thresholds_reject_nonvalidation_calibration(self) -> None:
        training = _batch(
            KNOWN_TRAINING_ROLE,
            [("r-00", (0.0,), 1.0)],
        )
        with self.assertRaisesRegex(PermissionError, "known_validation"):
            fit_known_validation_threshold(training)
        with self.assertRaisesRegex(PermissionError, "known_validation"):
            fit_known_validation_latent_threshold(training, (_latent_score("r-00", 1.0),))

    def test_latent_calibration_requires_exact_ordered_validation_identities(self) -> None:
        validation = _batch(
            KNOWN_VALIDATION_ROLE,
            [("a", (0.0,), 1.0), ("b", (0.0,), 2.0)],
        )
        ordered = (_latent_score("a", 1.0), _latent_score("b", 2.0))
        self.assertEqual(
            fit_known_validation_latent_threshold(validation, ordered).score_name,
            LATENT_SCORE_NAME,
        )
        with self.assertRaisesRegex(ValueError, "order.*exactly match"):
            fit_known_validation_latent_threshold(validation, tuple(reversed(ordered)))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            fit_known_validation_latent_threshold(validation, ordered[:1])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            fit_known_validation_latent_threshold(
                validation,
                (_latent_score("a", 1.0), _latent_score("c", 2.0)),
            )

    def test_latent_calibration_rejects_reference_query_identity_overlap(self) -> None:
        validation = _batch(KNOWN_VALIDATION_ROLE, [("a", (0.0,), 1.0)])
        overlapping_neighbours = (
            "a",
            *(f"training-{index:02d}" for index in range(9)),
        )
        latent_scores = (
            _latent_score(
                "a",
                1.0,
                neighbour_recording_ids=overlapping_neighbours,
            ),
        )
        with self.assertRaisesRegex(PermissionError, "overlap"):
            fit_known_validation_latent_threshold(validation, latent_scores)

    def test_classification_rejects_threshold_score_name_mismatch(self) -> None:
        validation = _batch(KNOWN_VALIDATION_ROLE, [("validation", (0.0,), 1.0)])
        validation_latent = (_latent_score("validation", 2.0),)
        reconstruction_threshold = fit_known_validation_reconstruction_threshold(validation)
        latent_threshold = fit_known_validation_latent_threshold(
            validation,
            validation_latent,
        )
        evaluation = _batch("evaluation", [("query", (0.0,), 2.0)])
        evaluation_latent = (_latent_score("query", 3.0),)

        with self.assertRaisesRegex(ValueError, "score_name"):
            classify_reconstruction_scores(evaluation, latent_threshold)
        with self.assertRaisesRegex(ValueError, "score_name"):
            classify_latent_scores(
                evaluation,
                evaluation_latent,
                reconstruction_threshold,
            )


class Task2ScoringValidationTests(unittest.TestCase):
    def test_array_contracts_reject_bad_types_shapes_and_values(self) -> None:
        valid = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaisesRegex(TypeError, "NumPy"):
            clip_reconstruction_mse([[0.0]], valid)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "floating"):
            clip_reconstruction_mse(np.zeros((1, 1), dtype=np.int64), valid)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            clip_reconstruction_mse(valid, np.zeros((1, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "clip axis"):
            clip_reconstruction_mse(np.asarray(1.0), np.asarray(1.0))
        with self.assertRaisesRegex(ValueError, "empty dimension"):
            clip_reconstruction_mse(
                np.empty((0, 1), dtype=np.float64),
                np.empty((0, 1), dtype=np.float64),
            )
        invalid = valid.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            clip_reconstruction_mse(valid, invalid)

    def test_metadata_and_aggregation_contracts_reject_malformed_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "trimmed"):
            ClipIdentity(" recording", "clip")
        with self.assertRaisesRegex(TypeError, "string"):
            ClipIdentity(1, "clip")  # type: ignore[arg-type]

        identities = (ClipIdentity("r", "c"),)
        scores = np.asarray([1.0], dtype=np.float64)
        latent = np.asarray([[1.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "do not match"):
            aggregate_recordings(
                identities,
                np.asarray([1.0, 2.0], dtype=np.float64),
                latent,
                source_role="evaluation",
            )
        with self.assertRaisesRegex(ValueError, "negative"):
            aggregate_recordings(
                identities,
                np.asarray([-1.0], dtype=np.float64),
                latent,
                source_role="evaluation",
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_recordings(
                (identities[0], identities[0]),
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([[1.0], [2.0]], dtype=np.float64),
                source_role="evaluation",
            )
        with self.assertRaisesRegex(ValueError, "rows do not match"):
            aggregate_recordings(
                identities,
                scores,
                np.asarray([[1.0], [2.0]], dtype=np.float64),
                source_role="evaluation",
            )

    def test_record_contracts_reject_identity_dimension_and_decision_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "lexical"):
            _recording("r", (0.0,), clip_ids=("c-2", "c-1"))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _recording("r", (0.0,), -1.0)
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            RecordingBatch(
                source_role="evaluation",
                recordings=(
                    _recording("a", (0.0,)),
                    _recording("b", (0.0, 1.0)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            RecordingBatch(
                source_role="evaluation",
                recordings=(_recording("a", (0.0,)), _recording("a", (1.0,))),
            )
        with self.assertRaisesRegex(ValueError, "strict"):
            NoveltyDecision(
                recording_id="r",
                score=1.0,
                threshold=1.0,
                is_novel=True,
            )

    def test_threshold_and_latent_score_dataclasses_reject_protocol_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "q=0.95"):
            ReconstructionThreshold(
                value=1.0,
                calibration_role=KNOWN_VALIDATION_ROLE,
                calibration_recording_ids=("r",),
                quantile=0.90,
            )
        with self.assertRaisesRegex(ValueError, "score_name"):
            NoveltyThreshold(
                value=1.0,
                calibration_role=KNOWN_VALIDATION_ROLE,
                calibration_recording_ids=("r",),
                score_name="unlocked_score",
            )
        with self.assertRaisesRegex(ValueError, "strict >"):
            NoveltyThreshold(
                value=1.0,
                calibration_role=KNOWN_VALIDATION_ROLE,
                calibration_recording_ids=("r",),
                classification_operator=">=",
            )
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            LatentNoveltyScore(
                recording_id="query",
                score=0.0,
                neighbour_recording_ids=("r",),
                neighbour_distances=(0.0,),
            )

    def test_outputs_are_json_serializable_records(self) -> None:
        training = _batch(
            KNOWN_TRAINING_ROLE,
            [(f"r-{index:02d}", (float(index),), 0.0) for index in range(10)],
        )
        reference = fit_known_training_reference(training)
        queries = _batch("evaluation", [("query", (20.0,), 2.0)])
        latent_score = latent_knn_novelty_scores(reference, queries)[0]
        validation = _batch(KNOWN_VALIDATION_ROLE, [("validation", (0.0,), 1.0)])
        threshold = fit_known_validation_threshold(validation)
        decision = classify_reconstruction_scores(queries, threshold)[0]
        latent_validation = (_latent_score("validation", 2.0),)
        latent_threshold = fit_known_validation_latent_threshold(
            validation,
            latent_validation,
        )
        latent_decision = classify_latent_scores(
            queries,
            latent_knn_novelty_scores(reference, queries),
            latent_threshold,
        )[0]

        payload = {
            "clip": ClipIdentity("recording", "clip").to_record(),
            "batch": training.to_record(),
            "reference": reference.to_record(),
            "latent_score": latent_score.to_record(),
            "threshold": threshold.to_record(),
            "latent_threshold": latent_threshold.to_record(),
            "decision": decision.to_record(),
            "latent_decision": latent_decision.to_record(),
        }
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.assertIn('"higher_is_more_novel"', serialized)
        self.assertIn(f'"{RECONSTRUCTION_SCORE_NAME}"', serialized)
        self.assertIn(f'"{LATENT_SCORE_NAME}"', serialized)


if __name__ == "__main__":
    unittest.main()
