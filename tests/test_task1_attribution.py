from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from bird_audio import task1_attribution as attribution


def _record(path: str, character: str = "a") -> dict[str, object]:
    return {"path": path, "sha256": character * 64, "size_bytes": 10}


def _context() -> dict[str, object]:
    checkpoint = _record("/checkpoint.pt", "c")
    return {
        "final_result": _record("/final-result.json", "1"),
        "final_lock": _record("/final-lock.json", "2"),
        "final_claim": _record("/final-claim.json", "a"),
        "gate": _record("/gate.json", "3"),
        "gate_lock": _record("/gate-lock.json", "4"),
        "source_fingerprint_sha256": "5" * 64,
        "claim_sha256": "6" * 64,
        "seed_37": {
            "run_id": "task1_seed_37",
            "run_identity_sha256": "7" * 64,
            "checkpoint": checkpoint,
            "cache_lock_sha256": attribution.KNOWN_CACHE_LOCK_SHA256,
            "source_fingerprint_sha256": "5" * 64,
            "final_stage_result": _record("/stage-result.json", "8"),
        },
        "run": {
            "seed": 37,
            "run_id": "task1_seed_37",
            "run_identity_sha256": "7" * 64,
            "best_checkpoint": checkpoint,
        },
        "gate_value": {"ready": True},
    }


def _candidate(recording_id: str, *, correct: bool) -> dict[str, object]:
    predicted = 0 if correct else 1
    logits = [0.0] * len(attribution.LOCKED_TASK1_CLASS_ORDER)
    logits[predicted] = 1.0
    return {
        "recording_id": recording_id,
        "session_group": f"session:{recording_id}",
        "clip_ids": [f"{recording_id}:clip:1", f"{recording_id}:clip:2"],
        "clip_count": 2,
        "true_class_index": 0,
        "true_class_name": attribution.LOCKED_TASK1_CLASS_ORDER[0],
        "predicted_class_index": predicted,
        "predicted_class_name": attribution.LOCKED_TASK1_CLASS_ORDER[predicted],
        "mean_logits": logits,
        "shard": _record(f"/{recording_id}.json", recording_id[-1].lower()),
    }


def _final_shard(recording_id: str, *, correct: bool) -> dict[str, object]:
    predicted = 0 if correct else 1
    logits = [0.0] * len(attribution.LOCKED_TASK1_CLASS_ORDER)
    logits[predicted] = 1.0
    clip_ids = [f"{recording_id}:clip:1", f"{recording_id}:clip:2"]
    return {
        "schema_version": "1.0",
        "stage_id": "task1_seed_37",
        "task": "task1_classification",
        "seed": 37,
        "recording_id": recording_id,
        "session_group": f"session:{recording_id}",
        "true_class_index": 0,
        "true_class_name": attribution.LOCKED_TASK1_CLASS_ORDER[0],
        "mean_logits": logits,
        "predicted_class_index": predicted,
        "predicted_class_name": attribution.LOCKED_TASK1_CLASS_ORDER[predicted],
        "metadata": {
            "recording_id": recording_id,
            "clip_ids": clip_ids,
            "clip_count": len(clip_ids),
        },
        "run_id": "task1_seed_37",
        "run_identity_sha256": "7" * 64,
        "checkpoint_sha256": "c" * 64,
        "gate_sha256": "3" * 64,
        "claim_sha256": "6" * 64,
        "cache_lock_sha256": attribution.KNOWN_CACHE_LOCK_SHA256,
        "source_role": attribution.FINAL_KNOWN_TEST_ROLE,
        "source_fingerprint_sha256": "5" * 64,
    }


def _selection() -> dict[str, object]:
    items: list[dict[str, object]] = []
    strata: dict[str, list[str]] = {}
    for stratum, correct in (("correct", True), ("error", False)):
        selected_ids = [f"{stratum}-{index}" for index in range(1, 4)]
        strata[stratum] = selected_ids
        for rank, recording_id in enumerate(selected_ids, start=1):
            base = _candidate(recording_id, correct=correct)
            items.append(
                {
                    "stratum": stratum,
                    "rank": rank,
                    "selection_key_sha256": attribution._selection_digest(stratum, recording_id),
                    "image_filename": attribution._image_filename(stratum, rank, recording_id),
                    **base,
                }
            )
    context = _context()
    return {
        "schema_version": attribution.ATTRIBUTION_SCHEMA_VERSION,
        "attribution_id": attribution.ATTRIBUTION_ID,
        "detail_seed": attribution.DETAIL_SEED,
        "selection_order": {
            "algorithm": "sha256",
            "domain": attribution.SELECTION_ORDER_DOMAIN,
            "within_stratum": True,
            "saliency_independent": True,
        },
        "selections_per_stratum": attribution.SELECTIONS_PER_STRATUM,
        "selection_count": 6,
        "strata": strata,
        "bindings": {
            key: value for key, value in context.items() if key not in {"run", "gate_value"}
        },
        "items": items,
    }


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1), nn.ReLU()),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, len(attribution.LOCKED_TASK1_CLASS_ORDER))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))


class SelectionTests(unittest.TestCase):
    def test_selection_is_stable_within_correctness_strata_and_saliency_independent(self) -> None:
        candidates = [
            *(_candidate(f"correct-{index}", correct=True) for index in range(1, 7)),
            *(_candidate(f"error-{index}", correct=False) for index in range(1, 7)),
        ]
        saliency = mock.Mock(side_effect=AssertionError("selection touched saliency"))
        with (
            mock.patch.object(attribution, "_seed37_items", return_value=tuple(candidates)),
            mock.patch.object(attribution, "_compute_gradcam", saliency),
        ):
            first = attribution._selection_value(_context())
        with mock.patch.object(
            attribution, "_seed37_items", return_value=tuple(reversed(candidates))
        ):
            second = attribution._selection_value(_context())
        self.assertEqual(first, second)
        self.assertEqual(first["selection_count"], 6)
        self.assertEqual(
            [item["stratum"] for item in first["items"]],
            ["correct", "correct", "correct", "error", "error", "error"],
        )
        saliency.assert_not_called()

    def test_selection_fails_closed_when_either_stratum_has_fewer_than_three(self) -> None:
        candidates = (
            *(_candidate(f"correct-{index}", correct=True) for index in range(3)),
            *(_candidate(f"error-{index}", correct=False) for index in range(2)),
        )
        with (
            mock.patch.object(attribution, "_seed37_items", return_value=candidates),
            self.assertRaisesRegex(ValueError, "at least three"),
        ):
            attribution._selection_value(_context())

    def test_build_publishes_selection_and_record_before_model_feature_or_saliency_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            events: list[str] = []
            selection = _selection()

            def publish(
                path: str | Path,
                _payload: bytes,
                **_kwargs: object,
            ) -> tuple[dict[str, object], bool]:
                name = Path(path).name
                events.append(name)
                return _record(name, str(len(events))), True

            def produce(
                _selection_value: dict[str, object],
                _context_value: dict[str, object],
                **_kwargs: object,
            ) -> tuple[dict[str, object], ...]:
                self.assertEqual(
                    events[:2],
                    [
                        attribution._SELECTION_FILENAME,
                        attribution._SELECTION_RECORD_FILENAME,
                    ],
                )
                events.append("model-feature-saliency")
                return tuple(
                    {
                        "recording_id": item["recording_id"],
                        "stratum": item["stratum"],
                        "rank": item["rank"],
                        "true_class_name": item["true_class_name"],
                        "predicted_class_name": item["predicted_class_name"],
                        "clip_ids": item["clip_ids"],
                        "clip_count": item["clip_count"],
                        "all_clips_included": True,
                        "target_class_index": item["predicted_class_index"],
                        "artifact": _record(item["image_filename"], "9"),
                    }
                    for item in selection["items"]
                )

            with (
                mock.patch.object(attribution, "ATTRIBUTION_ROOT", root),
                mock.patch.object(attribution, "verify_final_evaluation", return_value={}),
                mock.patch.object(attribution, "verify_final_evaluation_gate", return_value={}),
                mock.patch.object(attribution, "_verified_context", return_value=_context()),
                mock.patch.object(attribution, "_ensure_directory"),
                mock.patch.object(
                    attribution,
                    "_open_directory",
                    side_effect=lambda _path: os.open(root, os.O_RDONLY),
                ),
                mock.patch.object(attribution, "_directory_entries", return_value={}),
                mock.patch.object(attribution, "_validate_partial_inventory"),
                mock.patch.object(attribution, "_assert_context_current"),
                mock.patch.object(attribution, "_selection_value", return_value=selection),
                mock.patch.object(attribution, "_publish_or_verify", side_effect=publish),
                mock.patch.object(attribution, "_produce_images", side_effect=produce),
                mock.patch.object(
                    attribution,
                    "_verify_with_context",
                    return_value={"created": False},
                ),
            ):
                result = attribution.build_task1_attributions()
            self.assertTrue(result["created"])
            self.assertLess(
                events.index(attribution._SELECTION_RECORD_FILENAME),
                events.index("model-feature-saliency"),
            )


class SecurityBoundaryTests(unittest.TestCase):
    def test_final_attempt_records_resolve_beneath_the_attempt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            attempt = project / "runs" / "final_evaluation_v2" / "attempt_v2"
            stage = attempt / "task1_seed_37"
            stage.mkdir(parents=True)
            payload = b'{"complete":true}\n'
            (attempt / "result.json").write_bytes(payload)
            (project / "result.json").write_bytes(b"project-root-decoy")
            record = {
                "path": "result.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(
                    attribution,
                    "FINAL_EVALUATION_ATTEMPT_DIRECTORY",
                    attempt,
                ),
            ):
                self.assertEqual(
                    attribution._attempt_record_from_verified(
                        record,
                        "result.json",
                        "Final result",
                    ),
                    record,
                )
                with self.assertRaisesRegex(ValueError, "artifact record"):
                    attribution._attempt_record_from_verified(
                        {**record, "path": str(attempt / "result.json")},
                        "result.json",
                        "Final result",
                    )

    def test_locked_directory_descriptor_survives_swap_and_detects_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            root = project / "report_assets" / attribution.ATTRIBUTION_ID
            root.mkdir(parents=True)
            descriptor = os.open(root, os.O_RDONLY)
            detached = root.with_name("detached")
            try:
                root.rename(detached)
                root.mkdir()
                with (
                    mock.patch.object(attribution, "PROJECT_ROOT", project),
                    mock.patch.object(attribution, "ATTRIBUTION_ROOT", root),
                ):
                    attribution._create_only(
                        "held.txt",
                        b"held-directory",
                        directory_descriptor=descriptor,
                    )
                    self.assertEqual((detached / "held.txt").read_bytes(), b"held-directory")
                    self.assertEqual(tuple(root.iterdir()), ())
                    with self.assertRaisesRegex(PermissionError, "changed"):
                        attribution._assert_root_descriptor_current(descriptor)
            finally:
                os.close(descriptor)

    def test_snapshot_rejects_same_size_in_place_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            evidence = project / "evidence"
            evidence.mkdir(parents=True)
            path = evidence / "artifact.bin"
            path.write_bytes(b"original-bytes")
            original_read = os.pread
            changed = False

            def mutating_read(descriptor: int, count: int, offset: int) -> bytes:
                nonlocal changed
                chunk = original_read(descriptor, count, offset)
                if chunk and not changed:
                    changed = True
                    path.write_bytes(b"changed--bytes")
                return chunk

            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(attribution.os, "pread", side_effect=mutating_read),
                self.assertRaisesRegex(PermissionError, "changed"),
            ):
                attribution._snapshot(path)

    def test_context_recheck_rejects_claim_and_checkpoint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            attempt = project / "runs" / "final_evaluation_v2" / "attempt_v2"
            stage = attempt / "task1_seed_37"
            stage.mkdir(parents=True)
            external = project / "external"
            external.mkdir()

            def write(path: Path, payload: bytes) -> dict[str, object]:
                path.write_bytes(payload)
                return {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }

            final_result = write(attempt / "result.json", b"final-result")
            final_result["path"] = "result.json"
            final_lock = write(attempt / "lock.json", b"final-lock")
            final_lock["path"] = "lock.json"
            stage_result = write(stage / "result.json", b"stage-result")
            stage_result["path"] = "task1_seed_37/result.json"
            stage_lock = write(stage / "lock.json", b"stage-lock")
            stage_lock["path"] = "task1_seed_37/lock.json"
            gate = write(external / "gate.json", b"gate")
            gate_lock = write(external / "gate-lock.json", b"gate-lock")
            claim = write(external / "claim.json", b"claim")
            checkpoint = write(external / "checkpoint.pt", b"checkpoint")
            cache = write(external / "cache-lock.json", b"cache-lock")
            source = "5" * 64
            context = {
                "final_result": final_result,
                "final_lock": final_lock,
                "final_claim": claim,
                "gate": gate,
                "gate_lock": gate_lock,
                "source_fingerprint_sha256": source,
                "known_cache_lock": cache,
                "seed_37": {
                    "checkpoint": checkpoint,
                    "final_stage_result": stage_result,
                    "final_stage_lock": stage_lock,
                },
            }
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(
                    attribution,
                    "FINAL_EVALUATION_ATTEMPT_DIRECTORY",
                    attempt,
                ),
                mock.patch.object(attribution, "source_fingerprint", return_value=source),
                mock.patch.object(attribution, "_seed37_items", return_value=()),
            ):
                attribution._assert_context_current(context)
                (external / "claim.json").write_bytes(b"changed-claim")
                with self.assertRaisesRegex(PermissionError, "Final claim artifact changed"):
                    attribution._assert_context_current(context)
                (external / "claim.json").write_bytes(b"claim")
                (external / "checkpoint.pt").write_bytes(b"changed-checkpoint")
                with self.assertRaisesRegex(PermissionError, "checkpoint.*changed"):
                    attribution._assert_context_current(context)

    def test_project_root_rejects_an_intermediate_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            real_parent = base / "real-parent"
            project = real_parent / "project"
            project.mkdir(parents=True)
            alias = base / "alias"
            alias.symlink_to(real_parent, target_is_directory=True)
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", alias / "project"),
                self.assertRaisesRegex(PermissionError, "safely open"),
            ):
                attribution._open_project_root()

    def test_seed37_selection_reads_only_stage_lock_bound_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            attempt = project / "runs" / "final_evaluation_v2" / "attempt_v2"
            shards = attempt / "task1_seed_37" / "shards"
            shards.mkdir(parents=True)
            context = _context()
            records: list[dict[str, object]] = []
            recording_ids = ["recording-a", "recording-b"]
            for recording_id, correct in zip(recording_ids, (True, False), strict=True):
                value = _final_shard(recording_id, correct=correct)
                payload = attribution._json_bytes(value)
                filename = hashlib.sha256(recording_id.encode()).hexdigest() + ".json"
                (shards / filename).write_bytes(payload)
                records.append(
                    {
                        "path": f"task1_seed_37/shards/{filename}",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                )
            records.sort(key=lambda record: str(record["path"]))
            context["seed_37"]["final_shards"] = records
            context["seed_37"]["final_recording_ids"] = recording_ids
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(
                    attribution,
                    "FINAL_EVALUATION_ATTEMPT_DIRECTORY",
                    attempt,
                ),
                mock.patch.object(attribution, "KNOWN_TEST_RECORDINGS", 2),
            ):
                items = attribution._seed37_items(context)
                self.assertEqual(
                    tuple(sorted(item["recording_id"] for item in items)),
                    tuple(recording_ids),
                )
                target = shards / Path(records[0]["path"]).name
                changed = _final_shard("recording-a", correct=True)
                changed["session_group"] = "changed-session"
                target.write_bytes(attribution._json_bytes(changed))
                with self.assertRaisesRegex(PermissionError, "stage lock"):
                    attribution._seed37_items(context)


class GradCamAndRenderingTests(unittest.TestCase):
    def test_private_gradcam_math_shape_finite_and_exact_final_block_hook(self) -> None:
        torch.manual_seed(9)
        model = TinyClassifier()
        self.assertIs(attribution._final_convolutional_block(model), model.features[8])
        native = torch.rand(2, 1, 128, 372, dtype=torch.float32)
        maps, logits = attribution._compute_gradcam(
            model,
            native,
            target_class_index=0,
            device=torch.device("cpu"),
        )
        self.assertEqual(maps.shape, (2, 128, 372))
        self.assertEqual(logits.shape, (len(attribution.LOCKED_TASK1_CLASS_ORDER),))
        self.assertTrue(np.all(np.isfinite(maps)))
        self.assertTrue(np.all((maps >= 0.0) & (maps <= 1.0)))

    def test_final_target_rejects_a_nonconvolutional_ninth_feature(self) -> None:
        model = TinyClassifier()
        model.features[8] = nn.Identity()
        with self.assertRaisesRegex(ValueError, "final convolutional"):
            attribution._final_convolutional_block(model)

    def test_slaney_mel_ticks_use_internal_filter_centers(self) -> None:
        centers = attribution._mel_filter_center_mels()
        ticks = attribution._mel_tick_positions()
        self.assertEqual(centers.shape, (128,))
        self.assertTrue(np.all(np.diff(centers) > 0.0))
        self.assertEqual(ticks[0], 0.0)
        self.assertEqual(ticks[-1], 127.0)
        expected_one_khz = float(
            np.interp(
                attribution._slaney_hz_to_mel(1000.0),
                centers,
                np.arange(128, dtype=np.float64),
            )
        )
        self.assertAlmostEqual(ticks[1], expected_one_khz, places=12)

    def test_png_contains_every_clip_and_manifest_declares_axes_and_labels(self) -> None:
        selected = _selection()["items"][0]
        native = np.linspace(0.0, 1.0, num=2 * 128 * 372, dtype=np.float32).reshape(2, 1, 128, 372)
        maps = np.ones((2, 128, 372), dtype=np.float32) * 0.5
        payload = attribution._render_png(native, maps, selected)
        repeated = attribution._render_png(native, maps, selected)
        contract = attribution._render_contract()
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(payload, repeated)
        self.assertEqual(contract["frequency_span_khz"], [0.15, 14.0])
        self.assertEqual(contract["time_axis_seconds"], [0.0, 3.0])
        self.assertTrue(contract["all_selected_clips"])
        self.assertEqual(
            contract["labels"],
            ["true_class", "predicted_class", "correct_or_error_stratum"],
        )

    def test_runtime_rejects_fast_math_and_prefer_metal_overrides(self) -> None:
        for name in ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
            with (
                mock.patch.dict(os.environ, {name: "1"}, clear=False),
                mock.patch.object(
                    attribution.sys, "prefix", str(attribution.PROJECT_ROOT / ".venv")
                ),
                mock.patch.object(
                    attribution.sys,
                    "executable",
                    str(attribution.PROJECT_ROOT / ".venv" / "bin" / "python"),
                ),
                self.assertRaisesRegex(RuntimeError, name),
            ):
                attribution._prepare_runtime()


class PublicationVerifierTests(unittest.TestCase):
    def _publish_fixture(self, project: Path) -> tuple[dict[str, object], dict[str, object]]:
        root = attribution.ATTRIBUTION_ROOT
        attribution._ensure_directory(root)
        selection = _selection()
        selection_artifact = attribution._create_only(
            root / attribution._SELECTION_FILENAME,
            attribution._json_bytes(selection),
        )
        selection_record_artifact = attribution._create_only(
            root / attribution._SELECTION_RECORD_FILENAME,
            attribution._json_bytes(attribution._selection_record_value(selection_artifact)),
        )
        images: list[dict[str, object]] = []
        for item in selection["items"]:
            artifact = attribution._create_only(
                root / item["image_filename"],
                b"\x89PNG\r\n\x1a\nfixed-test-payload-" + item["recording_id"].encode(),
            )
            images.append(
                {
                    "recording_id": item["recording_id"],
                    "stratum": item["stratum"],
                    "rank": item["rank"],
                    "true_class_name": item["true_class_name"],
                    "predicted_class_name": item["predicted_class_name"],
                    "clip_ids": item["clip_ids"],
                    "clip_count": item["clip_count"],
                    "all_clips_included": True,
                    "target_class_index": item["predicted_class_index"],
                    "artifact": artifact,
                }
            )
        manifest = attribution._manifest_value(
            selection,
            selection_artifact,
            selection_record_artifact,
            images,
        )
        manifest_artifact = attribution._create_only(
            root / attribution._MANIFEST_FILENAME,
            attribution._json_bytes(manifest),
        )
        lock = attribution._lock_value(
            selection_artifact,
            selection_record_artifact,
            manifest_artifact,
            images,
        )
        attribution._create_only(
            root / attribution._LOCK_FILENAME,
            attribution._json_bytes(lock),
        )
        return selection, _context()

    def test_verifier_uses_no_model_reader_feature_or_saliency_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            project.mkdir()
            root = project / "report_assets" / attribution.ATTRIBUTION_ID
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(attribution, "ATTRIBUTION_ROOT", root),
                mock.patch.object(
                    attribution, "_selection_value", side_effect=lambda _value: _selection()
                ),
            ):
                self._publish_fixture(project)
                model = mock.Mock(side_effect=AssertionError("verifier loaded model"))
                reader = mock.Mock(side_effect=AssertionError("verifier opened features"))
                saliency = mock.Mock(side_effect=AssertionError("verifier computed saliency"))
                order: list[str] = []
                with (
                    mock.patch.object(
                        attribution,
                        "verify_final_evaluation",
                        side_effect=lambda: order.append("final") or {},
                    ),
                    mock.patch.object(
                        attribution,
                        "verify_final_evaluation_gate",
                        side_effect=lambda: order.append("gate") or {},
                    ),
                    mock.patch.object(attribution, "_verified_context", return_value=_context()),
                    mock.patch.object(attribution, "_assert_context_current"),
                    mock.patch.object(attribution, "load_locked_task1_best_model", model),
                    mock.patch.object(attribution, "open_final_known_test_data", reader),
                    mock.patch.object(attribution, "_compute_gradcam", saliency),
                ):
                    result = attribution.verify_task1_attributions()
                self.assertFalse(result["created"])
                self.assertEqual(order, ["final", "gate"])
                model.assert_not_called()
                reader.assert_not_called()
                saliency.assert_not_called()

    def test_manifest_tamper_and_extra_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            project.mkdir()
            root = project / "report_assets" / attribution.ATTRIBUTION_ID
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(attribution, "ATTRIBUTION_ROOT", root),
                mock.patch.object(
                    attribution, "_selection_value", side_effect=lambda _value: _selection()
                ),
            ):
                self._publish_fixture(project)
                manifest_path = root / attribution._MANIFEST_FILENAME
                original = manifest_path.read_bytes()
                manifest_path.write_bytes(original + b"tamper")
                with self.assertRaises((ValueError, PermissionError)):
                    attribution._verify_with_context(_context())
                manifest_path.write_bytes(original)
                (root / "unexpected.txt").write_text("extra", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unexpected|inventory"):
                    attribution._verify_with_context(_context())

    def test_symlink_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve() / "project"
            project.mkdir()
            root = project / "report_assets" / attribution.ATTRIBUTION_ID
            with (
                mock.patch.object(attribution, "PROJECT_ROOT", project),
                mock.patch.object(attribution, "ATTRIBUTION_ROOT", root),
                mock.patch.object(
                    attribution, "_selection_value", side_effect=lambda _value: _selection()
                ),
            ):
                self._publish_fixture(project)
                target = root / _selection()["items"][0]["image_filename"]
                payload = target.read_bytes()
                target.unlink()
                outside = project / "outside.png"
                outside.write_bytes(payload)
                target.symlink_to(outside)
                with self.assertRaises((ValueError, PermissionError)):
                    attribution._verify_with_context(_context())


if __name__ == "__main__":
    unittest.main()
