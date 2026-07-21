from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bird_audio.final_evaluation_data as final_data
from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.hashing import sha256_json


def _cache_record(
    name: str,
    sha256: str,
    version: str,
    content_sha256: str,
    requirements_sha256: str,
) -> dict[str, object]:
    return {
        "path": str(
            (final_data.PROJECT_ROOT / "data" / "processed" / name / "lock.json").resolve()
        ),
        "sha256": sha256,
        "size_bytes": 100,
        "cache_version": version,
        "cache_content_sha256": content_sha256,
        "requirements_lock_sha256": requirements_sha256,
    }


def _gate() -> dict[str, object]:
    known_content = "1" * 64
    unknown_content = "2" * 64
    requirements = "3" * 64
    shared = {
        "known_cache_lock_sha256": final_data.KNOWN_CACHE_LOCK_SHA256,
        "known_cache_content_sha256": known_content,
        "unknown_cache_lock_sha256": final_data.UNKNOWN_CACHE_LOCK_SHA256,
        "unknown_cache_content_sha256": unknown_content,
        "requirements_lock_sha256": requirements,
        "source_fingerprint_sha256": "4" * 64,
    }
    return {
        "gate_id": final_data.FINAL_EVALUATION_GATE_ID,
        "ready": True,
        "seed_order": [13, 37, 71],
        "cache_locks": {
            "known": _cache_record(
                "known_clips_v1",
                final_data.KNOWN_CACHE_LOCK_SHA256,
                "known_clips_v1",
                known_content,
                requirements,
            ),
            "unknown": _cache_record(
                "unknown_clips_v2",
                final_data.UNKNOWN_CACHE_LOCK_SHA256,
                "unknown_clips_v2",
                unknown_content,
                requirements,
            ),
        },
        "shared_identity": shared,
        "shared_identity_sha256": sha256_json(shared),
    }


class _TemporaryFinalPaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.final_root = root / "runs" / "final_evaluation_v2"
        self.gate_directory = self.final_root / "gate_v2"
        self.gate_path = self.gate_directory / "gate.json"
        self.gate_lock_path = self.gate_directory / "lock.json"
        self.claim_path = self.final_root / "final_evaluation_attempt_v2.json"
        self.attempt_directory = self.final_root / "attempt_v2"
        self.known_lock_path = root / "data" / "processed" / "known_clips_v1" / "lock.json"
        self.unknown_lock_path = root / "data" / "processed" / "unknown_clips_v2" / "lock.json"
        self.gate_directory.mkdir(parents=True)
        self.gate_path.write_text('{"sealed":true}\n', encoding="utf-8")
        self.gate_lock_path.write_text('{"locked":true}\n', encoding="utf-8")

    def patches(self) -> mock._patch:
        return mock.patch.multiple(
            final_data,
            PROJECT_ROOT=self.root,
            FINAL_EVALUATION_ROOT=self.final_root,
            FINAL_EVALUATION_GATE_PATH=self.gate_path,
            FINAL_EVALUATION_GATE_LOCK_PATH=self.gate_lock_path,
            FINAL_EVALUATION_CLAIM_PATH=self.claim_path,
            FINAL_EVALUATION_ATTEMPT_DIRECTORY=self.attempt_directory,
            KNOWN_CACHE_LOCK_PATH=self.known_lock_path,
            UNKNOWN_CACHE_LOCK_PATH=self.unknown_lock_path,
            KNOWN_CACHE_ROOT=self.known_lock_path.parent,
            UNKNOWN_CACHE_ROOT=self.unknown_lock_path.parent,
            require_safe_output=lambda path: Path(path),
        )


class _FakeUnknownSource:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.root = Path("/unknown")
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)


class FinalAttemptClaimTests(unittest.TestCase):
    def test_descriptor_snapshot_rejects_a_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "project"
            outside = root / "real_parent"
            root.mkdir()
            outside.mkdir()
            artifact = outside / "artifact.json"
            artifact.write_text('{"sealed":true}\n', encoding="utf-8")
            (root / "linked").symlink_to(outside, target_is_directory=True)
            with (
                mock.patch.object(final_data, "PROJECT_ROOT", root),
                self.assertRaisesRegex(PermissionError, "parent cannot be opened safely"),
            ):
                final_data._descriptor_snapshot(root / "linked" / artifact.name)

    def test_descriptor_snapshot_fails_closed_without_nofollow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = root / "artifact.json"
            artifact.write_text('{"sealed":true}\n', encoding="utf-8")
            with (
                mock.patch.object(final_data, "PROJECT_ROOT", root),
                mock.patch.object(final_data.os, "O_NOFOLLOW", 0),
                self.assertRaisesRegex(RuntimeError, "O_NOFOLLOW"),
            ):
                final_data._descriptor_snapshot(artifact)

    def test_create_only_writer_rejects_a_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "project"
            real_parent = root / "real_parent"
            root.mkdir()
            real_parent.mkdir()
            (root / "linked").symlink_to(real_parent, target_is_directory=True)
            destination = root / "linked" / "claim.json"
            with (
                mock.patch.object(final_data, "PROJECT_ROOT", root),
                mock.patch.object(final_data, "require_safe_output", side_effect=lambda path: path),
                self.assertRaisesRegex(PermissionError, "parent cannot be opened safely"),
            ):
                final_data._write_json_create_only(destination, {"sealed": True})
            self.assertFalse((real_parent / "claim.json").exists())

    def test_attempt_directory_rejects_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "project"
            real_attempt = root / "real_attempt"
            root.mkdir()
            real_attempt.mkdir()
            linked_attempt = root / "attempt_v2"
            linked_attempt.symlink_to(real_attempt, target_is_directory=True)
            with (
                mock.patch.object(final_data, "PROJECT_ROOT", root),
                mock.patch.object(
                    final_data,
                    "FINAL_EVALUATION_ATTEMPT_DIRECTORY",
                    linked_attempt,
                ),
                self.assertRaisesRegex(PermissionError, "attempt directory is invalid"),
            ):
                final_data._require_attempt_directory()

    def test_claim_timestamp_must_use_utc(self) -> None:
        gate_record = {"path": "/gate", "sha256": "1" * 64, "size_bytes": 1}
        gate_lock_record = {"path": "/lock", "sha256": "2" * 64, "size_bytes": 1}
        claim = {
            "schema_version": final_data.FINAL_EVALUATION_DATA_SCHEMA_VERSION,
            "attempt_id": final_data.FINAL_EVALUATION_ATTEMPT_ID,
            "claimed_at_utc": "2026-07-14T12:00:00+05:45",
            "gate_id": final_data.FINAL_EVALUATION_GATE_ID,
            "gate": gate_record,
            "gate_lock": gate_lock_record,
            "attempt_directory": final_data.FINAL_EVALUATION_ATTEMPT_DIRECTORY.relative_to(
                final_data.PROJECT_ROOT
            ).as_posix(),
            "stage_order": list(final_data.STAGE_ORDER),
            "single_attempt": True,
        }
        with self.assertRaisesRegex(ValueError, "must use UTC"):
            final_data._validate_claim(
                claim,
                gate_record=gate_record,
                gate_lock_record=gate_lock_record,
            )

    def test_claim_is_create_only_and_idempotent_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            with paths.patches():
                gate_record = final_data._artifact_record(paths.gate_path)
                lock_record = final_data._artifact_record(paths.gate_lock_path)
                verified = {
                    "gate": _gate(),
                    "gate_artifact": gate_record,
                    "lock_artifact": lock_record,
                    "created": False,
                }
                with mock.patch.object(
                    final_data,
                    "verify_final_evaluation_gate",
                    return_value=verified,
                ):
                    first = final_data.claim_final_evaluation_attempt()
                    second = final_data.claim_final_evaluation_attempt()
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(first["claim"], second["claim"])
            self.assertEqual(
                first["claim_artifact"]["sha256"],
                second["claim_artifact"]["sha256"],
            )
            self.assertTrue(paths.attempt_directory.is_dir())

    def test_tampered_existing_claim_is_rejected_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            with paths.patches():
                verified = {
                    "gate": _gate(),
                    "gate_artifact": final_data._artifact_record(paths.gate_path),
                    "lock_artifact": final_data._artifact_record(paths.gate_lock_path),
                    "created": False,
                }
                with mock.patch.object(
                    final_data,
                    "verify_final_evaluation_gate",
                    return_value=verified,
                ):
                    final_data.claim_final_evaluation_attempt()
                    claim = json.loads(paths.claim_path.read_text(encoding="utf-8"))
                    claim["single_attempt"] = False
                    payload = json.dumps(claim, indent=2, sort_keys=True) + "\n"
                    paths.claim_path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "claim binding"):
                        final_data.claim_final_evaluation_attempt()
            self.assertFalse(json.loads(paths.claim_path.read_text())["single_attempt"])

    def test_unclaimed_attempt_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            paths.attempt_directory.mkdir(parents=True)
            with paths.patches():
                verified = {
                    "gate": _gate(),
                    "gate_artifact": final_data._artifact_record(paths.gate_path),
                    "lock_artifact": final_data._artifact_record(paths.gate_lock_path),
                    "created": False,
                }
                with (
                    mock.patch.object(
                        final_data,
                        "verify_final_evaluation_gate",
                        return_value=verified,
                    ),
                    self.assertRaisesRegex(PermissionError, "without its claim"),
                ):
                    final_data.claim_final_evaluation_attempt()

    def test_existing_claim_with_deleted_attempt_is_not_recreated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            with paths.patches():
                verified = {
                    "gate": _gate(),
                    "gate_artifact": final_data._artifact_record(paths.gate_path),
                    "lock_artifact": final_data._artifact_record(paths.gate_lock_path),
                    "created": False,
                }
                with mock.patch.object(
                    final_data,
                    "verify_final_evaluation_gate",
                    return_value=verified,
                ):
                    final_data.claim_final_evaluation_attempt()
                    paths.attempt_directory.rmdir()
                    with self.assertRaisesRegex(
                        PermissionError,
                        "attempt directory is invalid",
                    ):
                        final_data.claim_final_evaluation_attempt()
            self.assertTrue(paths.claim_path.is_file())
            self.assertFalse(paths.attempt_directory.exists())

    def test_gate_cache_bindings_are_exact(self) -> None:
        gate = _gate()
        final_data._validate_gate_for_data(gate)
        gate["cache_locks"]["unknown"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "unknown cache binding"):
            final_data._validate_gate_for_data(gate)

    def test_stale_gate_verification_fails_before_v2_claim_and_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            with (
                paths.patches(),
                mock.patch.object(
                    final_data,
                    "verify_final_evaluation_gate",
                    side_effect=ValueError("implementation_sha256"),
                ),
                self.assertRaisesRegex(ValueError, "implementation_sha256"),
            ):
                final_data.claim_final_evaluation_attempt()
            self.assertFalse(paths.claim_path.exists())
            self.assertFalse(paths.attempt_directory.exists())

    def test_v1_sibling_is_preserved_and_does_not_block_v2_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            legacy_attempt = paths.root / "runs" / "final_evaluation" / "attempt_v1"
            legacy_attempt.mkdir(parents=True)
            legacy_marker = legacy_attempt / "failure.json"
            legacy_marker.write_bytes(b"legacy-v1-evidence")
            with paths.patches():
                verified = {
                    "gate": _gate(),
                    "gate_artifact": final_data._artifact_record(paths.gate_path),
                    "lock_artifact": final_data._artifact_record(paths.gate_lock_path),
                    "created": False,
                }
                with mock.patch.object(
                    final_data,
                    "verify_final_evaluation_gate",
                    return_value=verified,
                ):
                    claimed = final_data.claim_final_evaluation_attempt()
            self.assertTrue(claimed["created"])
            self.assertEqual(legacy_marker.read_bytes(), b"legacy-v1-evidence")
            self.assertTrue(paths.claim_path.is_file())

    def test_unexpected_v3_attempt_blocks_v2_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _TemporaryFinalPaths(Path(temporary).resolve())
            (paths.final_root / "attempt_v3").mkdir(parents=True)
            with paths.patches():
                verified = {
                    "gate": _gate(),
                    "gate_artifact": final_data._artifact_record(paths.gate_path),
                    "lock_artifact": final_data._artifact_record(paths.gate_lock_path),
                    "created": False,
                }
                with (
                    mock.patch.object(
                        final_data,
                        "verify_final_evaluation_gate",
                        return_value=verified,
                    ),
                    self.assertRaisesRegex(PermissionError, "Another final evaluation"),
                ):
                    final_data.claim_final_evaluation_attempt()
            self.assertFalse(paths.claim_path.exists())


class ReaderBoundaryTests(unittest.TestCase):
    def test_known_reader_rejects_before_opening_cache(self) -> None:
        with (
            mock.patch.object(final_data, "_load_known_cache_metadata") as loader,
            self.assertRaisesRegex(TypeError, "authorization"),
        ):
            final_data.open_final_known_test_data(object())
        loader.assert_not_called()

    def test_unknown_reader_rejects_before_opening_cache(self) -> None:
        with (
            mock.patch.object(final_data, "load_unknown_scoring_clip_cache") as loader,
            self.assertRaisesRegex(TypeError, "authorization"),
        ):
            final_data.open_final_unknown_data(object())
        loader.assert_not_called()

    def test_known_reader_requires_exact_locked_counts(self) -> None:
        rows: list[dict[str, str]] = []
        for recording_index in range(final_data.KNOWN_TEST_RECORDINGS):
            clip_count = 5 if recording_index < 85 else 4
            for _clip_index in range(clip_count):
                rows.append(
                    {
                        "recording_id": f"K{recording_index:03d}",
                        "energy_selected": "true",
                        "energy_clip_count": str(clip_count),
                    }
                )
        statistics = {
            "recordings": final_data.KNOWN_TEST_RECORDINGS,
            "energy_memberships": final_data.KNOWN_TEST_ENERGY_CLIPS,
        }
        config = {
            "known_species": [
                {"common_name": common_name} for common_name in LOCKED_TASK1_CLASS_ORDER
            ]
        }
        authorization = object()
        with (
            mock.patch.object(final_data, "_require_authorization_current"),
            mock.patch.object(
                final_data,
                "_load_known_cache_metadata",
                return_value=(
                    Path("/cache"),
                    {"artifacts": {"splits": {"test": {}}}},
                    {"splits": {"test": statistics}},
                    {"config": config},
                ),
            ),
            mock.patch.object(
                final_data,
                "_read_known_split_index",
                return_value=(rows, statistics),
            ) as read_index,
        ):
            reader = final_data.FinalKnownTestData(authorization)
        self.assertEqual(len(reader), final_data.KNOWN_TEST_ENERGY_CLIPS)
        self.assertEqual(reader.recording_count, final_data.KNOWN_TEST_RECORDINGS)
        self.assertTrue(read_index.call_args.kwargs["verify_feature_bytes"])

    def test_unknown_reader_requires_five_equal_species_sets(self) -> None:
        rows: list[dict[str, str]] = []
        for recording_index in range(final_data.UNKNOWN_RECORDINGS):
            species_index = recording_index // final_data.UNKNOWN_RECORDINGS_PER_SPECIES
            clip_count = 5 if recording_index < 43 else 4
            for _ in range(clip_count):
                rows.append(
                    {
                        "candidate_id": f"U{recording_index:03d}",
                        "species_scientific_name": f"Species name{species_index}",
                    }
                )
        source = _FakeUnknownSource(rows)
        with (
            mock.patch.object(final_data, "_require_authorization_current"),
            mock.patch.object(
                final_data,
                "load_unknown_scoring_clip_cache",
                return_value=source,
            ),
        ):
            reader = final_data.FinalUnknownData(object())
        self.assertEqual(len(reader), final_data.UNKNOWN_ENERGY_CLIPS)
        self.assertEqual(reader.recording_count, final_data.UNKNOWN_RECORDINGS)
        self.assertEqual(len(reader.species_scientific_names), final_data.UNKNOWN_SPECIES)


if __name__ == "__main__":
    unittest.main()
