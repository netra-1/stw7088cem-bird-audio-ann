from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bird_audio.clip_cache import (
    INDEX_FIELDS,
    _assert_locked_cache_config,
    _atomic_publish_directory_no_replace,
    build_known_clip_cache,
    load_development_clip_cache,
    verify_known_clip_cache,
)
from bird_audio.clip_selection import EnergyCandidate
from bird_audio.config import config_fingerprint, load_toml
from bird_audio.hashing import sha256_file, sha256_json
from bird_audio.paths import PROJECT_ROOT
from bird_audio.signal import CLIP_SAMPLES, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH
from bird_audio.splitting import SPLIT_FIELDS


class KnownClipCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=interim)
        self.root = Path(self.temporary.name)
        self.raw_root = self.root / "raw"
        self.raw_root.mkdir()
        self.manifest = self.root / "manifest.csv"
        self.split = self.root / "split.csv"
        self.split_summary = self.root / "split_summary.json"
        self.split_lock = self.root / "split_lock.json"
        self.review_lock = self.root / "review_lock.json"
        self.ffmpeg = self.root / "ffmpeg"
        self.ffmpeg.write_text("#!/bin/sh\necho 'ffmpeg version test-runtime'\n", encoding="utf-8")
        self.ffmpeg.chmod(0o700)
        self.config_path = PROJECT_ROOT / "configs" / "data.toml"
        self.config = load_toml(self.config_path)
        self.species = [entry["common_name"] for entry in self.config["known_species"][:3]]

        self.manifest_rows: list[dict[str, str]] = []
        self.split_rows: list[dict[str, str]] = []
        self.waveforms: dict[str, np.ndarray] = {}
        for index, split_name in enumerate(("train", "validation", "test"), start=1):
            recording_id = f"XC90000{index}"
            raw_path = self.raw_root / f"{recording_id}.mp3"
            raw_path.write_bytes(f"fixture-{recording_id}".encode())
            relative_path = raw_path.relative_to(PROJECT_ROOT).as_posix()
            digest = sha256_file(raw_path)
            self.manifest_rows.append(
                {
                    "recording_id": recording_id,
                    "relative_path": relative_path,
                    "sha256": digest,
                    "species_common_name": self.species[index - 1],
                    "session_group": f"session:{recording_id}",
                    "local_qc_status": "include",
                    "probe_ok": "true",
                    "full_decode_status": "ok",
                    "ffprobe_duration_seconds": "3.000000",
                }
            )
            self.split_rows.append(
                {
                    "recording_id": recording_id,
                    "relative_path": relative_path,
                    "sha256": digest,
                    "species_common_name": self.species[index - 1],
                    "session_group": f"session:{recording_id}",
                    "split": split_name,
                    "split_seed": str(self.config["split_seed"]),
                    "source_manifest_sha256": "",
                }
            )
            self.waveforms[recording_id] = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self._write_bound_artifacts()
        self.raw_root_patch = patch("bird_audio.clip_cache.RAW_DATA_ROOT", self.raw_root)
        self.raw_root_patch.start()
        self.addCleanup(self.raw_root_patch.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(
        self,
        path: Path,
        rows: list[dict[str, str]],
        fields: list[str],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_bound_artifacts(self) -> None:
        manifest_fields = [
            "recording_id",
            "relative_path",
            "sha256",
            "species_common_name",
            "session_group",
            "local_qc_status",
            "probe_ok",
            "full_decode_status",
            "ffprobe_duration_seconds",
        ]
        self._write_csv(self.manifest, self.manifest_rows, manifest_fields)
        manifest_sha256 = sha256_file(self.manifest)
        for row in self.split_rows:
            row["source_manifest_sha256"] = manifest_sha256
        self._write_csv(self.split, self.split_rows, SPLIT_FIELDS)
        self.split_summary.write_text("{}\n", encoding="utf-8")
        self.review_lock.write_text(
            json.dumps({"final_manifest_sha256": manifest_sha256}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_lock = {
            "source_manifest_sha256": manifest_sha256,
            "split_sha256": sha256_file(self.split),
            "summary_sha256": sha256_file(self.split_summary),
            "review_lock_sha256": sha256_file(self.review_lock),
            "config_sha256": config_fingerprint(self.config),
        }
        self.split_lock.write_text(json.dumps(split_lock, sort_keys=True) + "\n", encoding="utf-8")

    def _decode(self, path: Path, _ffmpeg: Path, **_kwargs) -> np.ndarray:
        return self.waveforms[path.stem].copy()

    def _cache_arguments(self) -> dict[str, object]:
        return {
            "ffmpeg": self.ffmpeg,
            "config_path": self.config_path,
            "manifest_path": self.manifest,
            "split_path": self.split,
            "split_summary_path": self.split_summary,
            "split_lock_path": self.split_lock,
            "review_lock_path": self.review_lock,
        }

    def _build(
        self,
        destination: Path,
        *,
        decode=None,
        progress_callback=None,
    ):
        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {"all": True}},
            ),
            patch(
                "bird_audio.clip_cache.decode_audio_ffmpeg",
                side_effect=decode or self._decode,
            ),
        ):
            return build_known_clip_cache(
                destination,
                progress_callback=progress_callback,
                **self._cache_arguments(),
            )

    def _interrupt_after_first_checkpoint(self, destination: Path) -> Path:
        calls = 0

        def interrupted_decode(path: Path, ffmpeg: Path, **kwargs) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("simulated interruption")
            return self._decode(path, ffmpeg, **kwargs)

        with self.assertRaisesRegex(KeyboardInterrupt, "simulated interruption"):
            self._build(destination, decode=interrupted_decode)
        self.assertFalse(destination.exists())
        working = destination.with_name(f".{destination.name}.working")
        self.assertTrue(working.is_dir())
        self.assertEqual(
            len([path for path in (working / "completed").iterdir() if path.is_dir()]),
            1,
        )
        return working

    def _rewrite_locked_index(
        self,
        cache_root: Path,
        split: str,
        rows: list[dict[str, str]],
    ) -> None:
        index_path = cache_root / split / "index.csv"
        self._write_csv(index_path, rows, INDEX_FIELDS)
        lock_path = cache_root / "lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        index_entry = lock["artifacts"]["splits"][split]["index"]
        index_entry["sha256"] = sha256_file(index_path)
        index_entry["rows"] = len(rows)
        summary = json.loads((cache_root / "summary.json").read_text(encoding="utf-8"))
        lock["cache_content_sha256"] = sha256_json(
            {
                "provenance": lock["provenance"],
                "artifacts": lock["artifacts"],
                "summary": summary,
            }
        )
        lock_path.write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_provenance_validation_failure_precedes_decode(self) -> None:
        destination = self.root / "provenance_v1"
        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": False, "checks": {"review_lock_valid": False}},
            ) as validate_mock,
            patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
            self.assertRaisesRegex(ValueError, "provenance validation failed"),
        ):
            build_known_clip_cache(
                destination,
                ffmpeg=self.ffmpeg,
                config_path=self.config_path,
                manifest_path=self.manifest,
                split_path=self.split,
                split_summary_path=self.split_summary,
                split_lock_path=self.split_lock,
                review_lock_path=self.review_lock,
            )
        validate_mock.assert_called_once()
        decode_mock.assert_not_called()
        self.assertFalse(destination.exists())

    def test_provenance_inputs_cannot_leave_project_root(self) -> None:
        destination = self.root / "outside_input_v1"
        with tempfile.TemporaryDirectory() as external_directory:
            external_config = Path(external_directory) / "data.toml"
            external_config.write_bytes(self.config_path.read_bytes())
            with (
                patch("bird_audio.clip_cache.validate_frozen_split") as validate_mock,
                patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
                self.assertRaisesRegex(ValueError, "leaves the project root"),
            ):
                build_known_clip_cache(
                    destination,
                    ffmpeg=self.ffmpeg,
                    config_path=external_config,
                )
        validate_mock.assert_not_called()
        decode_mock.assert_not_called()

    def test_path_and_hash_drift_are_rejected_before_ffmpeg(self) -> None:
        outside = self.root / "outside.mp3"
        outside.write_bytes(b"outside")
        for drift_type in ("path", "hash"):
            with self.subTest(drift_type=drift_type):
                original_manifest = [dict(row) for row in self.manifest_rows]
                original_split = [dict(row) for row in self.split_rows]
                if drift_type == "path":
                    relative = outside.relative_to(PROJECT_ROOT).as_posix()
                    self.manifest_rows[0]["relative_path"] = relative
                    self.split_rows[0]["relative_path"] = relative
                    digest = sha256_file(outside)
                    self.manifest_rows[0]["sha256"] = digest
                    self.split_rows[0]["sha256"] = digest
                else:
                    self.manifest_rows[0]["sha256"] = "0" * 64
                    self.split_rows[0]["sha256"] = "0" * 64
                self._write_bound_artifacts()
                destination = self.root / f"drift_{drift_type}_v1"
                with (
                    patch(
                        "bird_audio.clip_cache.validate_frozen_split",
                        return_value={"valid": True, "checks": {}},
                    ),
                    patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
                    self.assertRaises((ValueError, FileNotFoundError)),
                ):
                    build_known_clip_cache(
                        destination,
                        ffmpeg=self.ffmpeg,
                        config_path=self.config_path,
                        manifest_path=self.manifest,
                        split_path=self.split,
                        split_summary_path=self.split_summary,
                        split_lock_path=self.split_lock,
                        review_lock_path=self.review_lock,
                    )
                decode_mock.assert_not_called()
                self.assertFalse(destination.exists())
                self.manifest_rows = original_manifest
                self.split_rows = original_split
                self._write_bound_artifacts()

    def test_decoded_duration_ratio_rejection_precedes_feature_work(self) -> None:
        self.manifest_rows[0]["ffprobe_duration_seconds"] = "1.000000"
        self._write_bound_artifacts()
        destination = self.root / "duration_v1"
        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {}},
            ),
            patch("bird_audio.clip_cache.decode_audio_ffmpeg", side_effect=self._decode),
            patch("bird_audio.clip_cache.native_log_mel_spectrogram") as feature_mock,
            self.assertRaisesRegex(ValueError, "duration ratio outside"),
        ):
            build_known_clip_cache(
                destination,
                ffmpeg=self.ffmpeg,
                config_path=self.config_path,
                manifest_path=self.manifest,
                split_path=self.split,
                split_summary_path=self.split_summary,
                split_lock_path=self.split_lock,
                review_lock_path=self.review_lock,
            )
        feature_mock.assert_not_called()
        self.assertFalse(destination.exists())

    def test_cache_is_deterministic_shape_checked_and_physically_split(self) -> None:
        first_root, first_summary = self._build(self.root / "cache_first_v1")
        second_root, second_summary = self._build(self.root / "cache_second_v1")
        self.assertEqual(first_summary, second_summary)

        first_files = {
            path.relative_to(first_root).as_posix(): path.read_bytes()
            for path in first_root.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second_root).as_posix(): path.read_bytes()
            for path in second_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)
        for split_name in ("train", "validation", "test"):
            index_path = first_root / split_name / "index.csv"
            feature_paths = list((first_root / split_name / "features").glob("*.npy"))
            self.assertEqual(len(feature_paths), 1)
            with index_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(list(reader.fieldnames or []), INDEX_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["split"], split_name)
            tensor = np.load(feature_paths[0], allow_pickle=False)
            self.assertEqual(tensor.shape, (1, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH))
            self.assertEqual(tensor.dtype, np.float32)

        train_cache = load_development_clip_cache(
            first_root, "train", "uniform", ffmpeg=self.ffmpeg
        )
        feature, metadata = train_cache[0]
        self.assertEqual(feature.shape, (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH))
        self.assertEqual(feature.dtype, np.float32)
        self.assertEqual(metadata["split"], "train")

    def test_uniform_and_energy_union_is_deduplicated(self) -> None:
        self.manifest_rows = [self.manifest_rows[0]]
        self.split_rows = [self.split_rows[0]]
        recording_id = self.manifest_rows[0]["recording_id"]
        self.manifest_rows[0]["ffprobe_duration_seconds"] = "6.000000"
        self.waveforms[recording_id] = np.zeros(2 * CLIP_SAMPLES, dtype=np.float32)
        self._write_bound_artifacts()

        cache_root, _ = self._build(self.root / "deduplicated_v1")

        with (cache_root / "train" / "index.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["start_sample"]) for row in rows], [0, CLIP_SAMPLES])
        self.assertTrue(all(row["uniform_selected"] == "true" for row in rows))
        self.assertTrue(all(row["energy_selected"] == "true" for row in rows))
        tensor = np.load(cache_root / rows[0]["feature_file"], allow_pickle=False)
        self.assertEqual(tensor.shape, (2, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH))

    def test_existing_cache_is_never_overwritten(self) -> None:
        destination, _ = self._build(self.root / "immutable_v1")
        original_lock = (destination / "lock.json").read_bytes()
        with (
            patch("bird_audio.clip_cache.validate_frozen_split") as validate_mock,
            patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
            self.assertRaisesRegex(RuntimeError, "cannot be replaced"),
        ):
            build_known_clip_cache(destination, ffmpeg=self.ffmpeg)
        validate_mock.assert_not_called()
        decode_mock.assert_not_called()
        self.assertEqual((destination / "lock.json").read_bytes(), original_lock)

    def test_atomic_publication_never_replaces_a_racing_empty_directory(self) -> None:
        source = self.root / ".publication_source"
        destination = self.root / "publication_race_v1"
        source.mkdir()
        (source / "marker").write_text("source", encoding="utf-8")
        destination.mkdir()

        with self.assertRaises(FileExistsError):
            _atomic_publish_directory_no_replace(source, destination)

        self.assertTrue(source.is_dir())
        self.assertTrue((source / "marker").is_file())
        self.assertTrue(destination.is_dir())
        self.assertEqual(list(destination.iterdir()), [])

    def test_loader_rejects_cache_content_and_lock_structure_tampering(self) -> None:
        destination, _ = self._build(self.root / "tamper_v1")
        lock_path = destination / "lock.json"
        original = json.loads(lock_path.read_text(encoding="utf-8"))

        content_tamper = dict(original)
        content_tamper["cache_content_sha256"] = "0" * 64
        lock_path.write_text(json.dumps(content_tamper) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content hash"):
            load_development_clip_cache(destination, "train", "uniform", ffmpeg=self.ffmpeg)

        structure_tamper = dict(original)
        structure_tamper["unexpected"] = True
        lock_path.write_text(json.dumps(structure_tamper) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fields are not exact"):
            load_development_clip_cache(destination, "train", "uniform", ffmpeg=self.ffmpeg)

    def test_development_loader_rejects_test_before_filesystem_access(self) -> None:
        with self.assertRaisesRegex(PermissionError, "cannot open the final test"):
            load_development_clip_cache(
                self.root / "does_not_exist_v1", "test", "uniform", ffmpeg=self.ffmpeg
            )

    def test_interrupted_build_resumes_without_redecoding_completed_recording(self) -> None:
        destination = self.root / "resumed_v1"
        working = self._interrupt_after_first_checkpoint(destination)
        completed_recording = next((working / "completed").iterdir()).name

        resumed_decode_ids: list[str] = []
        progress_events: list[dict[str, object]] = []

        def resumed_decode(path: Path, ffmpeg: Path, **kwargs) -> np.ndarray:
            resumed_decode_ids.append(path.stem)
            return self._decode(path, ffmpeg, **kwargs)

        resumed_root, resumed_summary = self._build(
            destination,
            decode=resumed_decode,
            progress_callback=progress_events.append,
        )
        self.assertNotIn(completed_recording, resumed_decode_ids)
        self.assertEqual(len(resumed_decode_ids), len(self.split_rows) - 1)
        self.assertFalse(working.exists())
        self.assertTrue(
            any(
                event.get("event") == "recording_complete"
                and event.get("recording_id") == completed_recording
                and event.get("resumed") is True
                for event in progress_events
            )
        )
        self.assertEqual(progress_events[-1]["event"], "published")

        fresh_root, fresh_summary = self._build(self.root / "fresh_v1")
        self.assertEqual(resumed_summary, fresh_summary)
        resumed_files = {
            path.relative_to(resumed_root).as_posix(): path.read_bytes()
            for path in resumed_root.rglob("*")
            if path.is_file()
        }
        fresh_files = {
            path.relative_to(fresh_root).as_posix(): path.read_bytes()
            for path in fresh_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(resumed_files, fresh_files)

    def test_stale_resume_identity_is_rejected_before_decode(self) -> None:
        destination = self.root / "stale_resume_v1"
        working = self._interrupt_after_first_checkpoint(destination)
        state_path = working / "resume.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["build_identity_sha256"] = "0" * 64
        state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")

        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {"all": True}},
            ),
            patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
            self.assertRaisesRegex(ValueError, "Resume state does not match"),
        ):
            build_known_clip_cache(destination, **self._cache_arguments())
        decode_mock.assert_not_called()
        self.assertTrue(working.is_dir())

    def test_tampered_resume_tensor_is_rejected_before_decode(self) -> None:
        destination = self.root / "tampered_resume_v1"
        working = self._interrupt_after_first_checkpoint(destination)
        completed_directory = next((working / "completed").iterdir())
        feature_path = completed_directory / "feature.npy"
        feature_path.write_bytes(feature_path.read_bytes() + b"tamper")

        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {"all": True}},
            ),
            patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
            self.assertRaisesRegex(ValueError, "Feature hash drift"),
        ):
            build_known_clip_cache(destination, **self._cache_arguments())
        decode_mock.assert_not_called()
        self.assertTrue(working.is_dir())

    def test_disk_preflight_fails_before_decode_and_preserves_working_state(self) -> None:
        destination = self.root / "no_space_v1"
        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {"all": True}},
            ),
            patch("bird_audio.clip_cache.shutil.disk_usage", return_value=SimpleNamespace(free=0)),
            patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
            self.assertRaisesRegex(OSError, "Insufficient free space"),
        ):
            build_known_clip_cache(destination, **self._cache_arguments())
        decode_mock.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertTrue(destination.with_name(f".{destination.name}.working").is_dir())

    def test_final_raw_reverification_blocks_publication_after_late_drift(self) -> None:
        destination = self.root / "late_raw_drift_v1"
        final_raw = self.raw_root / f"{self.split_rows[-1]['recording_id']}.mp3"

        def mutate_after_last_recording(event: dict[str, object]) -> None:
            if event.get("event") == "recording_complete" and event.get(
                "recordings_completed"
            ) == len(self.split_rows):
                final_raw.write_bytes(b"late-drift")

        with self.assertRaisesRegex(ValueError, "Raw SHA-256 drift"):
            self._build(destination, progress_callback=mutate_after_last_recording)
        self.assertFalse(destination.exists())
        self.assertTrue(destination.with_name(f".{destination.name}.working").is_dir())

    def test_strategy_loaders_filter_counts_and_follow_strategy_rank(self) -> None:
        self.manifest_rows = [self.manifest_rows[0]]
        self.split_rows = [self.split_rows[0]]
        recording_id = self.manifest_rows[0]["recording_id"]
        self.manifest_rows[0]["ffprobe_duration_seconds"] = "9.000000"
        self.waveforms[recording_id] = np.zeros(3 * CLIP_SAMPLES, dtype=np.float32)
        self._write_bound_artifacts()
        energy_candidates = (
            EnergyCandidate(start_sample=2 * CLIP_SAMPLES, energy=2.0),
            EnergyCandidate(start_sample=0, energy=1.0),
        )
        with patch(
            "bird_audio.clip_cache.select_energy_candidates",
            return_value=energy_candidates,
        ):
            cache_root, summary = self._build(self.root / "strategies_v1")

        uniform = load_development_clip_cache(cache_root, "train", "uniform", ffmpeg=self.ffmpeg)
        energy = load_development_clip_cache(cache_root, "train", "energy", ffmpeg=self.ffmpeg)
        self.assertEqual(len(uniform), 3)
        self.assertEqual(len(energy), 2)
        self.assertEqual(
            [int(uniform[index][1]["start_sample"]) for index in range(len(uniform))],
            [0, CLIP_SAMPLES, 2 * CLIP_SAMPLES],
        )
        self.assertEqual(
            [int(energy[index][1]["start_sample"]) for index in range(len(energy))],
            [2 * CLIP_SAMPLES, 0],
        )
        self.assertEqual(summary["splits"]["train"]["clips"], 3)
        self.assertEqual(summary["splits"]["train"]["uniform_memberships"], 3)
        self.assertEqual(summary["splits"]["train"]["energy_memberships"], 2)

    def test_public_verifier_checks_all_splits_without_exposing_test_rows(self) -> None:
        destination, _ = self._build(self.root / "verified_v1")
        with patch(
            "bird_audio.clip_cache.validate_frozen_split",
            return_value={"valid": True, "checks": {"all": True}},
        ):
            result = verify_known_clip_cache(destination, ffmpeg=self.ffmpeg)
        self.assertEqual(
            set(result),
            {"valid", "cache_version", "lock_sha256", "recordings", "clips", "feature_files"},
        )
        self.assertEqual(result["recordings"], 3)
        self.assertEqual(result["feature_files"], 3)

        (destination / "test" / "features" / "unexpected.npy").write_bytes(b"unexpected")
        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {"all": True}},
            ),
            self.assertRaisesRegex(ValueError, "Physical feature-file set"),
        ):
            verify_known_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_relocked_index_corruption_is_rejected_by_invariant_validation(self) -> None:
        destination, _ = self._build(self.root / "index_corruption_v1")
        index_path = destination / "train" / "index.csv"
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["feature_row"] = "1"
        self._rewrite_locked_index(destination, "train", rows)

        with (
            patch(
                "bird_audio.clip_cache.validate_frozen_split",
                return_value={"valid": True, "checks": {"all": True}},
            ),
            self.assertRaisesRegex(ValueError, "per-recording index invariants"),
        ):
            verify_known_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_loader_rejects_stale_ffmpeg_runtime_provenance(self) -> None:
        destination, _ = self._build(self.root / "stale_runtime_v1")
        self.ffmpeg.write_text(
            "#!/bin/sh\necho 'ffmpeg version changed-runtime'\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Cache provenance is stale"):
            load_development_clip_cache(destination, "train", "uniform", ffmpeg=self.ffmpeg)

    def test_locked_config_contract_rejects_relevant_drift(self) -> None:
        for section, key, value in (
            ("clip_selection", "maximum_clips_per_recording", 4),
            ("spectrogram", "f_min_hz", 200),
            ("spectrogram", "center", True),
        ):
            with self.subTest(section=section, key=key):
                drifted = copy.deepcopy(self.config)
                drifted[section][key] = value
                with self.assertRaisesRegex(ValueError, "implemented cache contract"):
                    _assert_locked_cache_config(drifted)

    def test_cache_construction_requires_the_project_virtualenv(self) -> None:
        destination = self.root / "wrong_venv_v1"
        with (
            patch("bird_audio.clip_cache.sys.prefix", str(self.root / "another-venv")),
            patch("bird_audio.clip_cache.validate_frozen_split") as validate_mock,
            patch("bird_audio.clip_cache.decode_audio_ffmpeg") as decode_mock,
            self.assertRaisesRegex(RuntimeError, "project virtualenv"),
        ):
            build_known_clip_cache(destination, **self._cache_arguments())
        validate_mock.assert_not_called()
        decode_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
