from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from bird_audio import final_evaluation as evaluation
from bird_audio.final_evaluation_inference import FinalRecordingMetadata
from bird_audio.task1_final_metrics import RecordingPrediction
from bird_audio.task2_scoring import LatentReference, NoveltyThreshold, RecordingScore


class _Paths:
    def __init__(self, temporary: str) -> None:
        self.project = Path(temporary).resolve() / "project"
        self.root = self.project / "runs" / "final_evaluation"
        self.attempt = self.root / "attempt_v1"
        self.gate = self.root / "gate_v1" / "gate.json"
        self.gate_lock = self.root / "gate_v1" / "lock.json"
        self.attempt.mkdir(parents=True)
        self.gate.parent.mkdir(parents=True)
        self.gate.write_text('{"gate":true}\n', encoding="utf-8")
        self.gate_lock.write_text('{"lock":true}\n', encoding="utf-8")

    def patches(self) -> mock._patch:
        return mock.patch.multiple(
            evaluation,
            PROJECT_ROOT=self.project,
            FINAL_EVALUATION_ROOT=self.root,
            FINAL_EVALUATION_ATTEMPT_DIRECTORY=self.attempt,
            FINAL_EVALUATION_GATE_PATH=self.gate,
            FINAL_EVALUATION_GATE_LOCK_PATH=self.gate_lock,
            FINAL_EVALUATION_CLAIM_PATH=self.root / "final_evaluation_attempt_v1.json",
        )


def _prediction(recording_id: str, *, class_index: int = 0) -> RecordingPrediction:
    logits = [0.0] * 15
    logits[class_index] = 1.0
    return RecordingPrediction(
        recording_id=recording_id,
        session_group=f"session:{recording_id}",
        true_class_index=class_index,
        mean_logits=tuple(logits),
        predicted_class_index=class_index,
    )


def _known_metadata(
    recording_id: str,
    *,
    class_index: int = 0,
    session_group: str | None = None,
    clip_id: str | None = None,
) -> FinalRecordingMetadata:
    common, scientific = evaluation.KNOWN_COMMON_TO_SCIENTIFIC[class_index]
    return FinalRecordingMetadata(
        source_role=evaluation.FINAL_KNOWN_TEST_ROLE,
        recording_id=recording_id,
        session_group=session_group or f"session:{recording_id}",
        species_common_name=common,
        species_scientific_name=scientific,
        class_index=class_index,
        clip_ids=(clip_id or f"clip:{recording_id}",),
    )


def _run(seed: int = 13, *, task: str = "task1") -> dict[str, object]:
    source_name = (
        "source_fingerprint_sha256" if task == "task1" else "release_source_fingerprint_sha256"
    )
    return {
        "seed": seed,
        "run_id": f"{task}_seed_{seed}",
        "run_identity_sha256": str(seed % 10) * 64,
        source_name: "a" * 64,
        "best_checkpoint": {
            "path": f"/checkpoint/{task}_{seed}.pt",
            "sha256": "b" * 64,
            "size_bytes": 10,
        },
    }


def _model_metadata(run: dict[str, object], *, task: str) -> dict[str, object]:
    checkpoint_field = "checkpoint_sha256" if task == "task1" else "best_checkpoint_sha256"
    return {
        "run_id": run["run_id"],
        "run_identity_sha256": run["run_identity_sha256"],
        "seed": run["seed"],
        "scope": "production",
        "production_evidence": True,
        checkpoint_field: run["best_checkpoint"]["sha256"],
    }


def _reference(*, overlap: str | None = None) -> LatentReference:
    identities = [f"train{index:02d}" for index in range(10)]
    if overlap is not None:
        identities[0] = overlap
    identities.sort()
    return LatentReference(
        fit_role="known_training",
        recording_ids=tuple(identities),
        coordinate_mean=(0.0,),
        population_variance=(0.0,),
        coordinate_scale=(1.0,),
        standardized_embeddings=tuple((0.0,) for _ in identities),
        nearest_neighbours=10,
    )


def _threshold(score_name: str, value: float = 0.1) -> NoveltyThreshold:
    return NoveltyThreshold(
        score_name=score_name,
        value=value,
        calibration_role="known_validation",
        calibration_recording_ids=("validation01",),
    )


class SecurityAndOrderingTests(unittest.TestCase):
    def test_public_run_requires_nonempty_command_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "command provenance"):
            evaluation.run_final_evaluation()

    def test_preflight_failure_happens_before_claim_and_final_readers(self) -> None:
        gate = {
            "ready": True,
            "seed_order": [13, 37, 71],
            "task1": {},
            "task2": {},
            "shared_identity": {"source_fingerprint_sha256": "a" * 64},
        }
        claim = mock.Mock()
        known = mock.Mock()
        unknown = mock.Mock()
        with (
            mock.patch.object(evaluation, "_transaction_lock", return_value=nullcontext()),
            mock.patch.object(
                evaluation,
                "verify_final_evaluation_gate",
                return_value={"gate": gate},
            ),
            mock.patch.object(
                evaluation,
                "_gate_artifacts",
                return_value=(
                    {"path": "/gate", "sha256": "1" * 64, "size_bytes": 1},
                    {"path": "/lock", "sha256": "2" * 64, "size_bytes": 1},
                ),
            ),
            mock.patch.object(evaluation, "_assert_gate_current"),
            mock.patch.object(
                evaluation, "_prepare_production_runtime", return_value=torch.device("cpu")
            ),
            mock.patch.object(
                evaluation, "_preflight_models", side_effect=RuntimeError("preflight")
            ),
            mock.patch.object(evaluation, "claim_final_evaluation_attempt", claim),
            mock.patch.object(evaluation, "open_final_known_test_data", known),
            mock.patch.object(evaluation, "open_final_unknown_data", unknown),
            self.assertRaisesRegex(RuntimeError, "preflight"),
        ):
            evaluation.run_final_evaluation(command=("bird-audio", "final-evaluation"))
        claim.assert_not_called()
        known.assert_not_called()
        unknown.assert_not_called()

    def test_transaction_uses_exclusive_flock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            calls: list[int] = []
            with (
                paths.patches(),
                mock.patch.object(
                    evaluation.fcntl,
                    "flock",
                    side_effect=lambda _descriptor, operation: calls.append(operation),
                ),
                evaluation._transaction_lock(exclusive=True),
            ):
                pass
            self.assertEqual(calls, [evaluation.fcntl.LOCK_EX, evaluation.fcntl.LOCK_UN])

    def test_secure_writer_rejects_symlink_parent_and_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            with paths.patches():
                directory = evaluation._secure_ensure_directory(paths.attempt / "safe")
                destination = directory / "value.json"
                evaluation._write_json_create_only(destination, {"value": 1})
                with self.assertRaises(FileExistsError):
                    evaluation._write_json_create_only(destination, {"value": 1})
                outside = paths.project / "outside"
                outside.mkdir()
                (paths.attempt / "linked").symlink_to(outside, target_is_directory=True)
                with self.assertRaises(OSError):
                    evaluation._secure_ensure_directory(paths.attempt / "linked" / "child")
                self.assertFalse((outside / "child").exists())

    def test_source_drift_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            gate = {"shared_identity": {"source_fingerprint_sha256": "a" * 64}}
            with (
                paths.patches(),
                mock.patch.object(evaluation, "source_fingerprint", return_value="b" * 64),
                self.assertRaisesRegex(PermissionError, "source fingerprint"),
            ):
                gate_record = evaluation._external_artifact_record(paths.gate)
                evaluation._assert_gate_current(gate, gate_record["sha256"], full=False)

    def test_diagnostic_failure_does_not_mask_primary_failure(self) -> None:
        gate = {
            "ready": True,
            "seed_order": [13, 37, 71],
            "task1": {},
            "task2": {},
            "shared_identity": {"source_fingerprint_sha256": "a" * 64},
        }
        claim = {"claimed_at_utc": "2026-07-14T00:00:00+00:00"}
        claim_record = {"path": "/claim", "sha256": "3" * 64, "size_bytes": 1}
        gate_record = {"path": "/gate", "sha256": "1" * 64, "size_bytes": 1}
        gate_lock_record = {"path": "/lock", "sha256": "2" * 64, "size_bytes": 1}
        with (
            mock.patch.object(evaluation, "_transaction_lock", return_value=nullcontext()),
            mock.patch.object(
                evaluation,
                "verify_final_evaluation_gate",
                return_value={"gate": gate},
            ),
            mock.patch.object(
                evaluation,
                "_gate_artifacts",
                return_value=(gate_record, gate_lock_record),
            ),
            mock.patch.object(evaluation, "_assert_gate_current"),
            mock.patch.object(
                evaluation,
                "_prepare_production_runtime",
                return_value=torch.device("cpu"),
            ),
            mock.patch.object(evaluation, "_preflight_models"),
            mock.patch.object(
                evaluation,
                "_claim_after_gate",
                return_value=(object(), gate, claim, claim_record),
            ),
            mock.patch.object(
                evaluation,
                "_validate_attempt_entries",
                side_effect=RuntimeError("primary failure"),
            ),
            mock.patch.object(
                evaluation,
                "_write_failure_diagnostic",
                side_effect=OSError("diagnostic failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "primary failure") as raised,
        ):
            evaluation.run_final_evaluation(command=("bird-audio", "final-evaluation"))
        self.assertTrue(
            any("diagnostic publication" in note for note in raised.exception.__notes__)
        )


class ModelPreflightTests(unittest.TestCase):
    def test_all_six_models_synchronize_then_release_cache(self) -> None:
        task1_runs = tuple(_run(seed, task="task1") for seed in (13, 37, 71))
        task2_runs = tuple(_run(seed, task="task2") for seed in (13, 37, 71))
        events: list[str] = []

        def task1_loader(*_args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            run = next(
                item
                for item in task1_runs
                if item["seed"] == kwargs.get("device_seed", item["seed"])
            )
            checkpoint = kwargs["expected_run_identity_sha256"]
            run = next(item for item in task1_runs if item["run_identity_sha256"] == checkpoint)
            events.append(f"load1:{run['seed']}")
            return object(), _model_metadata(run, task="task1")

        def task2_loader(*_args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            checkpoint = kwargs["expected_run_identity_sha256"]
            run = next(item for item in task2_runs if item["run_identity_sha256"] == checkpoint)
            events.append(f"load2:{run['seed']}")
            return object(), _model_metadata(run, task="task2")

        with (
            mock.patch.object(
                evaluation,
                "_run_inventory",
                side_effect=lambda _gate, task: task1_runs if task == "task1" else task2_runs,
            ),
            mock.patch.object(evaluation, "load_locked_task1_best_model", side_effect=task1_loader),
            mock.patch.object(
                evaluation,
                "load_locked_task2_best_model_for_evaluation",
                side_effect=task2_loader,
            ),
            mock.patch.object(
                evaluation,
                "_synchronize_device",
                side_effect=lambda _device: events.append("sync"),
            ),
            mock.patch.object(
                evaluation,
                "_release_device_cache",
                side_effect=lambda _device: events.append("release"),
            ),
        ):
            evaluation._preflight_models({}, torch.device("mps"))
        self.assertEqual(
            events,
            [
                "load1:13",
                "sync",
                "release",
                "load1:37",
                "sync",
                "release",
                "load1:71",
                "sync",
                "release",
                "load2:13",
                "sync",
                "release",
                "load2:37",
                "sync",
                "release",
                "load2:71",
                "sync",
                "release",
            ],
        )


class ResumeAndShardTests(unittest.TestCase):
    def test_task1_resume_passes_only_verified_shard_ids_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            run = _run(task="task1")
            rows = tuple(
                {
                    "recording_id": recording_id,
                    "clip_id": f"clip:{recording_id}",
                    "session_group": f"session:{recording_id}",
                    "species_common_name": "Asian Koel",
                    "class_index": "0",
                }
                for recording_id in ("recording-a", "recording-b")
            )
            data = SimpleNamespace(
                recording_ids=("recording-a", "recording-b"),
                iter_metadata=lambda: iter(rows),
            )
            skipped: list[frozenset[str]] = []

            def iterator(*_args: object, **kwargs: object):
                skipped.append(kwargs["skip_recording_ids"])
                yield SimpleNamespace(
                    recording_ids=("recording-b",),
                    predictions=(_prediction("recording-b"),),
                )

            with (
                paths.patches(),
                mock.patch.object(evaluation, "TASK1_FINAL_RECORDINGS", 2),
                mock.patch.object(evaluation, "_assert_run_current"),
            ):
                stage = evaluation._ensure_stage_directory("task1_seed_13")
                shards = evaluation._secure_ensure_directory(stage / "shards")
                evaluation._write_task1_shard(
                    shards,
                    _prediction("recording-a"),
                    _known_metadata("recording-a"),
                    stage_id="task1_seed_13",
                    run=run,
                    gate_sha256="c" * 64,
                    claim_sha256="d" * 64,
                )
                with (
                    mock.patch.object(
                        evaluation,
                        "load_locked_task1_best_model",
                        return_value=(object(), _model_metadata(run, task="task1")),
                    ) as loader,
                    mock.patch.object(
                        evaluation,
                        "iter_task1_recording_batches",
                        side_effect=iterator,
                    ),
                ):
                    first = evaluation._run_task1_stage(
                        run=run,
                        data=data,
                        device=torch.device("cpu"),
                        gate_sha256="c" * 64,
                        claim_sha256="d" * 64,
                    )
                    second = evaluation._run_task1_stage(
                        run=run,
                        data=data,
                        device=torch.device("cpu"),
                        gate_sha256="c" * 64,
                        claim_sha256="d" * 64,
                    )
                self.assertEqual(skipped, [frozenset({"recording-a"})])
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(first["result"], second["result"])

    def test_tampered_or_extra_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            run = _run(task="task1")
            with (
                paths.patches(),
                mock.patch.object(evaluation, "TASK1_FINAL_RECORDINGS", 1),
                mock.patch.object(evaluation, "_assert_run_current"),
            ):
                stage = evaluation._ensure_stage_directory("task1_seed_13")
                shards = evaluation._secure_ensure_directory(stage / "shards")
                evaluation._write_task1_shard(
                    shards,
                    _prediction("recording-a"),
                    _known_metadata("recording-a"),
                    stage_id="task1_seed_13",
                    run=run,
                    gate_sha256="c" * 64,
                    claim_sha256="d" * 64,
                )
                extra = shards / ("0" * 64 + ".json")
                extra.write_bytes(evaluation._canonical_json_bytes({"unexpected": True}))
                with self.assertRaises(ValueError):
                    evaluation._read_task1_shards(
                        stage_id="task1_seed_13",
                        run=run,
                        gate_sha256="c" * 64,
                        claim_sha256="d" * 64,
                    )

    def test_wrong_partial_task1_metadata_fails_before_result_or_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            run = _run(task="task1")
            row = {
                "recording_id": "recording-a",
                "clip_id": "clip:recording-a",
                "session_group": "session:recording-a",
                "species_common_name": "Asian Koel",
                "class_index": "0",
            }
            data = SimpleNamespace(
                recording_ids=("recording-a",),
                iter_metadata=lambda: iter((row,)),
            )
            with (
                paths.patches(),
                mock.patch.object(evaluation, "TASK1_FINAL_RECORDINGS", 1),
                mock.patch.object(evaluation, "_assert_run_current"),
            ):
                stage = evaluation._ensure_stage_directory("task1_seed_13")
                shards = evaluation._secure_ensure_directory(stage / "shards")
                evaluation._write_task1_shard(
                    shards,
                    _prediction("recording-a"),
                    _known_metadata("recording-a", clip_id="wrong-clip"),
                    stage_id="task1_seed_13",
                    run=run,
                    gate_sha256="c" * 64,
                    claim_sha256="d" * 64,
                )
                with self.assertRaisesRegex(ValueError, "partial shard labels"):
                    evaluation._run_task1_stage(
                        run=run,
                        data=data,
                        device=torch.device("cpu"),
                        gate_sha256="c" * 64,
                        claim_sha256="d" * 64,
                    )
                self.assertFalse((stage / "result.json").exists())
                self.assertFalse((stage / "lock.json").exists())

    def test_wrong_partial_task2_metadata_fails_before_result_or_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            run = _run(task="task2")
            known_row = {
                "recording_id": "known-final",
                "clip_id": "known-clip",
                "session_group": "known-session",
                "species_common_name": "Asian Koel",
                "class_index": "0",
            }
            unknown_row = {
                "recording_id": "unknown-final",
                "clip_id": "unknown-clip",
                "session_group": "unknown-session",
                "species_common_name": "Brown-headed Barbet",
                "species_scientific_name": "Psilopogon zeylanicus",
            }
            known_data = SimpleNamespace(
                recording_ids=("known-final",),
                iter_metadata=lambda: iter((known_row,)),
            )
            unknown_data = SimpleNamespace(
                recording_ids=("unknown-final",),
                iter_metadata=lambda: iter((unknown_row,)),
            )
            wrong_metadata = _known_metadata(
                "known-final",
                session_group="wrong-session",
                clip_id="known-clip",
            )
            score = RecordingScore(
                recording_id="known-final",
                clip_ids=("known-clip",),
                reconstruction_mse=0.1,
                mean_latent_embedding=(0.0,),
            )
            with paths.patches(), mock.patch.object(evaluation, "_assert_run_current"):
                stage = evaluation._ensure_stage_directory("task2_seed_13")
                known_shards = evaluation._secure_ensure_directory(stage / "known_test_shards")
                evaluation._secure_ensure_directory(stage / "unknown_shards")
                evaluation._write_task2_shard(
                    known_shards,
                    score,
                    wrong_metadata,
                    stage_id="task2_seed_13",
                    run=run,
                    gate_sha256="c" * 64,
                    claim_sha256="d" * 64,
                )
                with self.assertRaisesRegex(ValueError, "partial shard metadata"):
                    evaluation._run_task2_stage(
                        run=run,
                        known_data=known_data,
                        unknown_data=unknown_data,
                        device=torch.device("cpu"),
                        gate_sha256="c" * 64,
                        claim_sha256="d" * 64,
                    )
                self.assertFalse((stage / "result.json").exists())
                self.assertFalse((stage / "lock.json").exists())


class Task2ProtocolTests(unittest.TestCase):
    def _items(self) -> tuple[dict[str, tuple[RecordingScore, FinalRecordingMetadata]], ...]:
        known_metadata = FinalRecordingMetadata(
            source_role=evaluation.FINAL_KNOWN_TEST_ROLE,
            recording_id="known-final",
            session_group="known-session",
            species_common_name="Asian Koel",
            species_scientific_name="Eudynamys scolopaceus",
            class_index=0,
            clip_ids=("known-clip",),
        )
        unknown_metadata = FinalRecordingMetadata(
            source_role=evaluation.FINAL_UNKNOWN_ROLE,
            recording_id="unknown-final",
            session_group="unknown-session",
            species_common_name="Brown-headed Barbet",
            species_scientific_name="Psilopogon zeylanicus",
            class_index=None,
            clip_ids=("unknown-clip",),
        )
        known_score = RecordingScore(
            recording_id="known-final",
            clip_ids=("known-clip",),
            reconstruction_mse=0.1,
            mean_latent_embedding=(0.0,),
        )
        unknown_score = RecordingScore(
            recording_id="unknown-final",
            clip_ids=("unknown-clip",),
            reconstruction_mse=0.2,
            mean_latent_embedding=(2.0,),
        )
        return (
            {"known-final": (known_score, known_metadata)},
            {"unknown-final": (unknown_score, unknown_metadata)},
        )

    def test_both_frozen_streams_and_strict_threshold_equality(self) -> None:
        known, unknown = self._items()
        run = _run(task="task2")
        evidence = (
            _reference(),
            {
                "reconstruction": _threshold(evaluation.RECONSTRUCTION_SCORE_NAME),
                "latent": _threshold(evaluation.LATENT_SCORE_NAME),
            },
            {"training_latent_reference": {}, "thresholds": {}},
        )
        with (
            mock.patch.object(evaluation, "TASK2_KNOWN_FINAL_RECORDINGS", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_FINAL_RECORDINGS", 1),
            mock.patch.object(evaluation, "TASK2_KNOWN_FINAL_CLIPS", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_FINAL_CLIPS", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_SPECIES", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_RECORDINGS_PER_SPECIES", 1),
            mock.patch.object(evaluation, "_task2_development_evidence", return_value=evidence),
        ):
            result, streams = evaluation._task2_stage_result(
                stage_id="task2_seed_13",
                run=run,
                known_items=known,
                unknown_items=unknown,
                gate_sha256="c" * 64,
                claim_sha256="d" * 64,
                completed_at_utc="2026-07-14T00:00:00+00:00",
            )
        self.assertEqual(set(streams), {"reconstruction", "latent"})
        self.assertEqual(set(result["score_streams"]), {"reconstruction", "latent"})
        self.assertEqual(result["score_streams"]["reconstruction"]["pooled"]["specificity"], 1.0)
        self.assertFalse(any(name.startswith("fit_") for name in vars(evaluation)))

    def test_final_identity_overlap_with_fit_or_calibration_is_rejected(self) -> None:
        known, unknown = self._items()
        run = _run(task="task2")
        evidence = (
            _reference(overlap="known-final"),
            {
                "reconstruction": _threshold(evaluation.RECONSTRUCTION_SCORE_NAME),
                "latent": _threshold(evaluation.LATENT_SCORE_NAME),
            },
            {"training_latent_reference": {}, "thresholds": {}},
        )
        with (
            mock.patch.object(evaluation, "TASK2_KNOWN_FINAL_RECORDINGS", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_FINAL_RECORDINGS", 1),
            mock.patch.object(evaluation, "TASK2_KNOWN_FINAL_CLIPS", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_FINAL_CLIPS", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_SPECIES", 1),
            mock.patch.object(evaluation, "TASK2_UNKNOWN_RECORDINGS_PER_SPECIES", 1),
            mock.patch.object(evaluation, "_task2_development_evidence", return_value=evidence),
            self.assertRaisesRegex(PermissionError, "overlap"),
        ):
            evaluation._task2_stage_result(
                stage_id="task2_seed_13",
                run=run,
                known_items=known,
                unknown_items=unknown,
                gate_sha256="c" * 64,
                claim_sha256="d" * 64,
                completed_at_utc="2026-07-14T00:00:00+00:00",
            )


class BootstrapAndVerificationTests(unittest.TestCase):
    def test_completed_verifier_finishes_with_full_gate_revalidation(self) -> None:
        gate = {"shared_identity": {"source_fingerprint_sha256": "a" * 64}}
        claim = {"claimed_at_utc": "2026-07-14T00:00:00+00:00"}
        gate_record = {"path": "/gate", "sha256": "1" * 64, "size_bytes": 1}
        gate_lock_record = {"path": "/lock", "sha256": "2" * 64, "size_bytes": 1}
        claim_record = {"path": "/claim", "sha256": "3" * 64, "size_bytes": 1}
        full_values: list[bool] = []
        with (
            mock.patch.object(
                evaluation,
                "_verify_existing_claim",
                return_value=(gate, claim, claim_record, gate_record),
            ),
            mock.patch.object(
                evaluation,
                "_gate_artifacts",
                return_value=(gate_record, gate_lock_record),
            ),
            mock.patch.object(
                evaluation,
                "_assert_gate_current",
                side_effect=lambda *_args, full: full_values.append(full),
            ),
            mock.patch.object(evaluation, "_validate_attempt_entries"),
            mock.patch.object(evaluation, "_authorization_from_existing", return_value=object()),
            mock.patch.object(evaluation, "open_final_known_test_data", return_value=object()),
            mock.patch.object(evaluation, "open_final_unknown_data", return_value=object()),
            mock.patch.object(evaluation, "_verify_reader_feature_bytes"),
            mock.patch.object(evaluation, "_run_inventory", return_value=()),
            mock.patch.object(
                evaluation,
                "_verify_summary",
                return_value={"stage_id": "summary"},
            ),
            mock.patch.object(
                evaluation,
                "_verify_final_artifacts",
                return_value={"complete": True},
            ),
        ):
            result = evaluation._verify_completed_evaluation_locked(
                verified_gate={"gate": gate},
                ffmpeg=None,
            )
        self.assertEqual(result, {"complete": True})
        self.assertTrue(full_values[-1])
        self.assertEqual(full_values.count(True), 1)

    def test_npz_rejects_object_arrays_and_roundtrip_has_no_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "object"):
            evaluation._validate_npz_payload({"unsafe": np.asarray([{"value": 1}], dtype=object)})
        with tempfile.TemporaryDirectory() as temporary:
            paths = _Paths(temporary)
            with paths.patches():
                archive = paths.attempt / "safe.npz"
                evaluation._write_or_verify_npz(
                    archive,
                    {
                        "values": np.asarray([0.1, 0.2], dtype=np.float64),
                        "names": np.asarray(["one", "two"], dtype=np.str_),
                    },
                )
                payload, _ = evaluation._read_npz(
                    archive,
                    expected_keys={"values", "names"},
                )
                self.assertTrue(all(not value.dtype.hasobject for value in payload.values()))
                buffer = io.BytesIO()
                np.savez(buffer, unsafe=np.asarray([object()], dtype=object))
                unsafe = paths.attempt / "unsafe.npz"
                evaluation._create_only_bytes(unsafe, buffer.getvalue())
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    evaluation._read_npz(unsafe, expected_keys={"unsafe"})

    def test_recursive_public_verify_never_calls_inference_or_model_loaders(self) -> None:
        expected = {"complete": True}
        with (
            mock.patch.object(evaluation, "_transaction_lock", return_value=nullcontext()),
            mock.patch.object(
                evaluation,
                "verify_final_evaluation_gate",
                return_value={"gate": {}},
            ),
            mock.patch.object(
                evaluation,
                "_verify_completed_evaluation_locked",
                return_value=expected,
            ),
            mock.patch.object(evaluation, "iter_task1_recording_batches") as task1_inference,
            mock.patch.object(evaluation, "iter_task2_recording_batches") as task2_inference,
            mock.patch.object(evaluation, "load_locked_task1_best_model") as task1_loader,
            mock.patch.object(
                evaluation,
                "load_locked_task2_best_model_for_evaluation",
            ) as task2_loader,
        ):
            self.assertEqual(evaluation.verify_final_evaluation(), expected)
        task1_inference.assert_not_called()
        task2_inference.assert_not_called()
        task1_loader.assert_not_called()
        task2_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
