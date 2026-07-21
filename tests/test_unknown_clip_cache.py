from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

import bird_audio.unknown_clip_cache as cache
from bird_audio.hashing import sha256_file, sha256_json
from bird_audio.paths import PROJECT_ROOT


def _write_private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


class UnknownSelectionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data")
        self.root = Path(self.temporary.name)
        self.raw_root = self.root / "unknown_raw"
        self.checkpoint_root = self.root / "checkpoints"
        self.raw_root.mkdir()
        self.checkpoint_root.mkdir()
        self.data_config = {
            "unknown_species": [
                {
                    "common_name": "Alpha Bird",
                    "scientific_name": "Alpha avis",
                    "difficulty_group": "family_matched",
                },
                {
                    "common_name": "Beta Bird",
                    "scientific_name": "Beta avis",
                    "difficulty_group": "other_family",
                },
            ],
            "fallback_unknown_species": [
                {
                    "common_name": "Gamma Bird",
                    "scientific_name": "Gamma avis",
                }
            ],
        }
        self.species_ids = {
            "Gamma avis": ["XC1", "XC2"],
            "Beta avis": ["XC3", "XC4"],
        }
        self.audit = {
            "gate": {
                "status": "ready_with_fallback",
                "fallback_active": True,
                "replacement": {
                    "replaced_scientific_name": "Alpha avis",
                    "replacement_scientific_name": "Gamma avis",
                },
            },
            "selection": {
                "selected_recordings": 4,
                "species_count": 2,
                "zero_candidate_overlap": True,
                "zero_session_overlap": True,
                "species": {},
            },
        }
        for species_index, (scientific_name, candidate_ids) in enumerate(self.species_ids.items()):
            assignments = []
            for selection_rank, candidate_id in enumerate(candidate_ids):
                session = f"session:{species_index}{selection_rank}"
                assignments.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_session_group": session,
                    }
                )
                self._write_source_and_checkpoint(candidate_id, scientific_name, session)
            self.audit["selection"]["species"][scientific_name] = {
                "selected_candidates": 2,
                "selected_candidate_ids": candidate_ids,
                "assignment": {"assignments": assignments},
            }
        unselected = self.raw_root / "Gamma_avis" / "XC999.audio"
        unselected.write_bytes(b"unselected")
        unselected.chmod(0o600)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_source_and_checkpoint(
        self, candidate_id: str, scientific_name: str, session_group: str
    ) -> None:
        source = self.raw_root / scientific_name.replace(" ", "_") / f"{candidate_id}.audio"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"audio-{candidate_id}".encode())
        source.chmod(0o600)
        relative_path = source.relative_to(PROJECT_ROOT).as_posix()
        checkpoint = {
            "schema_version": "1.0",
            "preflight_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "candidate_id": candidate_id,
            "scientific_name": scientific_name,
            "queue_rank": 1,
            "session_group": session_group,
            "disposition": "eligible",
            "reasons": [],
            "download_receipt": {},
            "audio_qc": {
                "candidate_id": candidate_id,
                "scientific_name": scientific_name,
                "session_group": session_group,
                "relative_path": relative_path,
                "sha256": sha256_file(source),
                "file_size_bytes": source.stat().st_size,
                "header_detection_status": "recognized",
                "probe_status": "ok",
                "full_decode_status": "ok",
                "decoded_duration_seconds": 3.0,
                "disposition": "eligible",
                "reasons": [],
            },
        }
        _write_private_json(self.checkpoint_root / f"{candidate_id}.json", checkpoint)

    @contextmanager
    def small_protocol(self):
        with patch.multiple(
            cache,
            TARGET_SPECIES=2,
            TARGET_RECORDINGS_PER_SPECIES=2,
            TARGET_RECORDINGS=4,
        ):
            yield

    def selected(self) -> tuple[cache._SelectedRecording, ...]:
        with self.small_protocol():
            return cache._selected_records(
                self.data_config,
                self.audit,
                self.checkpoint_root,
                self.raw_root,
            )


class UnknownSelectionTests(UnknownSelectionFixture):
    def test_implementation_fingerprint_uses_only_semantic_dependencies(self) -> None:
        observed: list[str] = []

        def fake_bound_artifact(
            path: Path,
            _label: str,
            *,
            private: bool,
        ) -> cache._BoundArtifact:
            self.assertFalse(private)
            observed.append(path.name)
            return cache._BoundArtifact(
                path=path,
                sha256=f"{len(observed):064x}",
                size_bytes=1,
                private=False,
            )

        with patch.object(cache, "_bound_artifact", side_effect=fake_bound_artifact):
            fingerprint = cache._implementation_fingerprint()

        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(observed, list(cache._IMPLEMENTATION_FILES))
        self.assertIn("unknown_clip_cache.py", observed)
        self.assertNotIn("config.py", observed)
        self.assertNotIn("task1_training.py", observed)
        self.assertNotIn("task2_training.py", observed)
        self.assertNotIn("cli.py", observed)

    def test_exact_effective_fallback_set_excludes_unselected_raw_audio(self) -> None:
        selected = self.selected()
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [row.scientific_name for row in selected[::2]],
            ["Gamma avis", "Beta avis"],
        )
        self.assertNotIn("Alpha avis", {row.scientific_name for row in selected})
        self.assertNotIn("XC999", {row.candidate_id for row in selected})

    def test_audit_selected_set_tamper_is_rejected(self) -> None:
        self.audit["selection"]["species"]["Gamma avis"]["selected_candidate_ids"][1] = "XC1"
        with self.small_protocol(), self.assertRaisesRegex(ValueError, "count is invalid"):
            cache._selected_records(
                self.data_config,
                self.audit,
                self.checkpoint_root,
                self.raw_root,
            )

    def test_selected_checkpoint_tamper_is_rejected(self) -> None:
        checkpoint_path = self.checkpoint_root / "XC1.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["disposition"] = "audio_qc_excluded"
        _write_private_json(checkpoint_path, checkpoint)
        with self.small_protocol(), self.assertRaisesRegex(ValueError, "not eligible"):
            cache._selected_records(
                self.data_config,
                self.audit,
                self.checkpoint_root,
                self.raw_root,
            )

    def test_selected_raw_tamper_is_rejected(self) -> None:
        raw = self.raw_root / "Gamma_avis" / "XC1.audio"
        raw.write_bytes(b"changed")
        raw.chmod(0o600)
        with self.small_protocol(), self.assertRaisesRegex(ValueError, "raw audio"):
            cache._selected_records(
                self.data_config,
                self.audit,
                self.checkpoint_root,
                self.raw_root,
            )

    def test_selected_raw_symlink_is_rejected(self) -> None:
        raw = self.raw_root / "Gamma_avis" / "XC1.audio"
        moved = raw.with_suffix(".original")
        raw.rename(moved)
        raw.symlink_to(moved.name)
        with self.small_protocol(), self.assertRaisesRegex(ValueError, "symbolic link"):
            cache._selected_records(
                self.data_config,
                self.audit,
                self.checkpoint_root,
                self.raw_root,
            )

    def test_descriptor_decode_remains_bound_to_the_open_inode(self) -> None:
        source = self.root / "descriptor.audio"
        source.write_bytes(b"descriptor-source")
        source.chmod(0o600)
        executable = self.root / "descriptor_ffmpeg"
        executable.write_text(
            "#!/bin/sh\n"
            f"exec '{sys.executable}' -c \"import struct,sys; "
            "source=sys.argv[sys.argv.index('-i')+1]; "
            "handle=open(source,'rb'); size=len(handle.read()); handle.close(); "
            'sys.stdout.buffer.write(struct.pack(\'<f\',float(size)))" "$@"\n',
            encoding="utf-8",
        )
        executable.chmod(0o700)
        digest = sha256_file(source)
        with cache._verified_raw_descriptor(source, digest, source.stat().st_size) as descriptor:
            waveform = cache._decode_audio_descriptor(
                descriptor,
                source,
                executable,
            )
        self.assertEqual(waveform.tolist(), [float(len(b"descriptor-source"))])

    def test_raw_path_swap_is_rejected_after_descriptor_use(self) -> None:
        source = self.root / "swap.audio"
        source.write_bytes(b"sealed")
        source.chmod(0o600)
        digest = sha256_file(source)
        moved = source.with_suffix(".original")
        with (
            self.assertRaisesRegex(ValueError, "no longer names"),
            cache._verified_raw_descriptor(source, digest, source.stat().st_size),
        ):
            source.rename(moved)
            source.write_bytes(b"replacement")
            source.chmod(0o600)

    def test_json_path_swap_during_parse_is_rejected(self) -> None:
        path = self.root / "snapshot.json"
        _write_private_json(path, {"value": "sealed"})
        original_loads = json.loads

        def swap_after_snapshot(payload):
            result = original_loads(payload)
            replacement = self.root / "replacement.json"
            _write_private_json(replacement, {"value": "changed"})
            os.replace(replacement, path)
            return result

        with (
            patch.object(cache.json, "loads", side_effect=swap_after_snapshot),
            self.assertRaisesRegex(ValueError, "changed while its JSON snapshot was parsed"),
        ):
            cache._read_json_snapshot(path, "fixture JSON", private=True)

    def test_csv_path_swap_during_parse_is_rejected(self) -> None:
        path = self.root / "snapshot.csv"
        path.write_text("value\nsealed\n", encoding="utf-8")
        path.chmod(0o600)
        original_reader = csv.DictReader

        def swap_after_snapshot(*args, **kwargs):
            reader = original_reader(*args, **kwargs)
            replacement = self.root / "replacement.csv"
            replacement.write_text("value\nchanged\n", encoding="utf-8")
            replacement.chmod(0o600)
            os.replace(replacement, path)
            return reader

        with (
            patch.object(cache.csv, "DictReader", side_effect=swap_after_snapshot),
            self.assertRaisesRegex(ValueError, "changed while its CSV snapshot was parsed"),
        ):
            cache._read_csv_snapshot(path, "fixture CSV")

    def test_npy_path_swap_during_parse_is_rejected(self) -> None:
        path = self.root / "snapshot.npy"
        tensor = np.zeros((1, *cache.NATIVE_FEATURE_SHAPE), dtype=np.float32)
        cache._write_npy(path, tensor)
        digest = sha256_file(path)
        original_load = np.load

        def swap_after_snapshot(*args, **kwargs):
            loaded = original_load(*args, **kwargs)
            replacement = self.root / "replacement.npy"
            cache._write_npy(replacement, np.ones_like(tensor))
            os.replace(replacement, path)
            return loaded

        with (
            patch.object(cache.np, "load", side_effect=swap_after_snapshot),
            self.assertRaisesRegex(ValueError, "changed while its NPY snapshot was parsed"),
        ):
            cache._read_verified_feature_tensor(path, digest)

    def test_audit_swap_around_verifier_is_rejected(self) -> None:
        config_path = self.root / "unknown.toml"
        config_path.write_text("fixture = true\n", encoding="utf-8")
        config, config_sha256, _ = cache._read_toml_snapshot(config_path, "fixture config")
        audit_path = self.root / "bound_audit.json"
        lock_path = self.root / "bound_lock.json"
        _write_private_json(audit_path, {"ready_for_unknown_scoring": True})
        _write_private_json(lock_path, {"ready_for_unknown_scoring": True})
        original_audit_sha256 = sha256_file(audit_path)

        def swap_audit(_config_path):
            _write_private_json(audit_path, {"ready_for_unknown_scoring": False})
            return {"audit_sha256": original_audit_sha256}

        with (
            patch.object(cache, "verify_unknown_audio_audit", side_effect=swap_audit),
            self.assertRaisesRegex(RuntimeError, "changed around their verifier"),
        ):
            cache._verified_audit_snapshots(
                config_path,
                config,
                config_sha256,
                audit_path,
                lock_path,
            )


class UnknownCacheBuildTests(UnknownSelectionFixture):
    def setUp(self) -> None:
        super().setUp()
        self.selected_rows = self.selected()
        self.data_config_file = self.root / "data_config.toml"
        self.unknown_config_file = self.root / "unknown_audio.toml"
        self.audit_file = self.root / "audit.json"
        self.audit_lock_file = self.root / "audit_lock.json"
        self.requirements_file = self.root / "requirements.lock"
        self.data_config_file.write_text("fixture = true\n", encoding="utf-8")
        self.unknown_config_file.write_text("fixture = true\n", encoding="utf-8")
        self.requirements_file.write_text("fixture==1\n", encoding="utf-8")
        _write_private_json(self.audit_file, self.audit)
        audit_lock = {
            "checkpoint_set_sha256": "a" * 64,
            "raw_file_set_sha256": "b" * 64,
        }
        _write_private_json(self.audit_lock_file, audit_lock)
        selected_checkpoints = [
            {
                "path": row.checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": row.checkpoint_sha256,
            }
            for row in sorted(self.selected_rows, key=lambda item: item.candidate_id)
        ]
        selected_raw = [
            {
                "path": row.relative_path,
                "sha256": row.source_sha256,
                "file_size_bytes": row.source_file_size_bytes,
            }
            for row in sorted(self.selected_rows, key=lambda item: item.relative_path)
        ]
        artifact_hashes = {
            self.data_config_file: sha256_file(self.data_config_file),
            self.unknown_config_file: sha256_file(self.unknown_config_file),
            self.audit_file: sha256_file(self.audit_file),
            self.audit_lock_file: sha256_file(self.audit_lock_file),
            self.requirements_file: sha256_file(self.requirements_file),
            **{row.checkpoint_path: row.checkpoint_sha256 for row in self.selected_rows},
        }
        checkpoint_paths = tuple(sorted(self.checkpoint_root.iterdir(), key=lambda item: item.name))
        raw_paths = tuple(
            sorted(
                (path for path in self.raw_root.rglob("*") if path.is_file()),
                key=lambda item: item.as_posix(),
            )
        )
        raw_directory_paths = tuple(
            sorted(
                (path for path in self.raw_root.rglob("*") if path.is_dir()),
                key=lambda item: item.as_posix(),
            )
        )
        private_paths = {
            self.audit_file,
            self.audit_lock_file,
            *checkpoint_paths,
            *raw_paths,
        }
        all_paths = {
            self.data_config_file,
            self.unknown_config_file,
            self.audit_file,
            self.audit_lock_file,
            self.requirements_file,
            *checkpoint_paths,
            *raw_paths,
        }
        artifact_bindings = tuple(
            cache._BoundArtifact(
                path=path,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                private=path in private_paths,
            )
            for path in sorted(all_paths, key=lambda item: item.as_posix())
        )
        artifact_hashes = {row.path: row.sha256 for row in artifact_bindings}
        self.inputs = cache._ValidatedInputs(
            data_config={
                "quality_control": {
                    "minimum_decoded_to_ffprobe_duration_ratio": 0.98,
                    "maximum_decoded_to_ffprobe_duration_ratio": 1.02,
                }
            },
            data_config_file=self.data_config_file,
            unknown_audio_config={},
            unknown_audio_config_file=self.unknown_config_file,
            audit=self.audit,
            audit_file=self.audit_file,
            audit_lock=audit_lock,
            audit_lock_file=self.audit_lock_file,
            checkpoint_root=self.checkpoint_root,
            requirements_lock_file=self.requirements_file,
            raw_root=self.raw_root,
            selected=self.selected_rows,
            artifact_hashes=artifact_hashes,
            artifact_bindings=artifact_bindings,
            checkpoint_file_paths=checkpoint_paths,
            raw_file_paths=raw_paths,
            raw_directory_paths=raw_directory_paths,
            data_config_file_sha256=sha256_file(self.data_config_file),
            data_config_sha256="1" * 64,
            unknown_audio_config_sha256=sha256_file(self.unknown_config_file),
            audit_sha256=sha256_file(self.audit_file),
            audit_lock_sha256=sha256_file(self.audit_lock_file),
            requirements_lock_sha256=sha256_file(self.requirements_file),
            selected_checkpoint_set_sha256=sha256_json(selected_checkpoints),
            selected_raw_file_set_sha256=sha256_json(selected_raw),
        )
        self.ffmpeg = self.root / "ffmpeg"
        self.ffmpeg.write_text("fixture\n", encoding="utf-8")
        self.ffmpeg.chmod(0o700)
        self.runtime = {
            "python_version": "fixture",
            "python_implementation": "fixture",
            "platform_system": "fixture",
            "platform_machine": "fixture",
            "numpy_version": "fixture",
            "librosa_version": "fixture",
            "ffmpeg_version_output": "fixture",
        }

    @contextmanager
    def build_environment(self, decode=None):
        waveform = np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)
        decode_function = decode or (lambda *_args, **_kwargs: waveform.copy())
        with ExitStack() as stack:
            stack.enter_context(self.small_protocol())
            stack.enter_context(
                patch.object(cache, "_load_validated_inputs", return_value=self.inputs)
            )
            stack.enter_context(
                patch.object(cache, "_runtime_provenance", return_value=self.runtime)
            )
            stack.enter_context(
                patch.object(cache, "_decode_audio_descriptor", side_effect=decode_function)
            )
            yield

    def cache_destination(self, name: str) -> Path:
        return self.root / name / cache.CACHE_VERSION

    def build(self, name: str) -> Path:
        destination = self.cache_destination(name)
        cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
        return destination

    def test_v2_identity_keeps_v1_acquisition_inputs(self) -> None:
        self.assertEqual(cache.CACHE_VERSION, "unknown_clips_v2")
        self.assertEqual(
            cache.DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
            "data/processed/unknown_clips_v2",
        )
        self.assertEqual(cache.DEFAULT_AUDIT, "data/unknown/audio/unknown_audio_audit_v1.json")
        self.assertEqual(
            cache.DEFAULT_AUDIT_LOCK,
            "data/unknown/audio/unknown_audio_audit_v1_lock.json",
        )
        self.assertEqual(
            cache.DEFAULT_CHECKPOINT_ROOT,
            "data/unknown/interim/audio_acquisition_v1/checkpoints",
        )

    def test_build_and_verify_reject_legacy_root_without_touching_it(self) -> None:
        legacy = self.root / "preserved" / "unknown_clips_v1"
        legacy.mkdir(parents=True)
        sentinel = legacy / "sentinel.txt"
        sentinel.write_text("preserve-v1\n", encoding="utf-8")

        with (
            patch.object(cache, "_load_validated_inputs") as load_inputs,
            self.assertRaisesRegex(ValueError, "basename must be exactly unknown_clips_v2"),
        ):
            cache.build_unknown_clip_cache(legacy, ffmpeg=self.ffmpeg)
        load_inputs.assert_not_called()

        with self.assertRaisesRegex(ValueError, "cache root is invalid"):
            cache.verify_unknown_clip_cache(legacy, ffmpeg=self.ffmpeg)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve-v1\n")
        self.assertEqual({child.name for child in legacy.iterdir()}, {"sentinel.txt"})

    def test_build_rejects_v2_nested_inside_preserved_v1_before_writing(self) -> None:
        preserved = self.root / "protected" / "unknown_clips_v1"
        preserved.mkdir(parents=True)
        sentinel = preserved / "sentinel.txt"
        sentinel.write_text("preserve-v1\n", encoding="utf-8")
        destination = preserved / cache.CACHE_VERSION

        with (
            patch.object(cache, "PRESERVED_V1_ROOTS", (preserved,)),
            patch.object(cache, "_load_validated_inputs") as load_inputs,
            self.assertRaisesRegex(ValueError, "protected v1 evidence"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

        load_inputs.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve-v1\n")
        self.assertEqual({child.name for child in preserved.iterdir()}, {"sentinel.txt"})

    def test_all_manifest_protected_v1_roots_reject_nested_v2_output(self) -> None:
        expected_labels = (
            "runs/final_evaluation",
            "runs/task1",
            "runs/task2",
            "data/processed/known_clips_v1",
            "data/processed/unknown_clips_v1",
            "report_assets/provenance",
            "data/unknown/audio",
            "data/unknown/interim/audio_acquisition_v1/checkpoints",
        )
        self.assertEqual(
            tuple(root.relative_to(PROJECT_ROOT).as_posix() for root in cache.PRESERVED_V1_ROOTS),
            expected_labels,
        )

        for index, label in enumerate(expected_labels):
            with self.subTest(root=label):
                preserved = self.root / "protected_roots" / f"root_{index}"
                preserved.mkdir(parents=True)
                sentinel = preserved / "sentinel.txt"
                sentinel.write_text(label + "\n", encoding="utf-8")
                destination = preserved / cache.CACHE_VERSION
                with (
                    patch.object(cache, "PRESERVED_V1_ROOTS", (preserved,)),
                    patch.object(cache, "_load_validated_inputs") as load_inputs,
                    self.assertRaisesRegex(ValueError, "protected v1 evidence"),
                ):
                    cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
                load_inputs.assert_not_called()
                self.assertFalse(destination.exists())
                self.assertEqual(sentinel.read_text(encoding="utf-8"), label + "\n")

    def test_build_rejects_v2_output_that_contains_a_protected_v1_root(self) -> None:
        destination = self.root / "ancestor_overlap" / cache.CACHE_VERSION
        protected = destination / "protected_v1_child"
        protected.mkdir(parents=True)
        sentinel = protected / "sentinel.txt"
        sentinel.write_text("preserve-v1\n", encoding="utf-8")

        with (
            patch.object(cache, "PRESERVED_V1_ROOTS", (protected,)),
            patch.object(cache, "_load_validated_inputs") as load_inputs,
            self.assertRaisesRegex(ValueError, "protected v1 evidence"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

        load_inputs.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve-v1\n")
        self.assertEqual({child.name for child in destination.iterdir()}, {protected.name})

    def test_build_rejects_output_equal_to_a_protected_v1_root(self) -> None:
        destination = self.root / "equal_overlap" / cache.CACHE_VERSION
        destination.mkdir(parents=True)
        sentinel = destination / "sentinel.txt"
        sentinel.write_text("preserve-v1\n", encoding="utf-8")

        with (
            patch.object(cache, "PRESERVED_V1_ROOTS", (destination,)),
            patch.object(cache, "_load_validated_inputs") as load_inputs,
            self.assertRaisesRegex(ValueError, "protected v1 evidence"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

        load_inputs.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve-v1\n")
        self.assertEqual({child.name for child in destination.iterdir()}, {sentinel.name})

    def test_deterministic_publication_energy_loader_and_scoring_boundary(self) -> None:
        with self.build_environment():
            first = self.build("first")
            second = self.build("second")
            self.assertEqual(sha256_file(first / "lock.json"), sha256_file(second / "lock.json"))
            verified = cache.verify_unknown_clip_cache(first, ffmpeg=self.ffmpeg)
            self.assertTrue(verified["valid"])
            self.assertEqual(verified["cache_version"], "unknown_clips_v2")
            self.assertTrue(verified["scoring_only"])
            self.assertEqual(verified["recordings"], 4)
            self.assertEqual(first.name, cache.CACHE_VERSION)
            self.assertEqual(
                json.loads((first / "lock.json").read_text(encoding="utf-8"))["cache_version"],
                cache.CACHE_VERSION,
            )
            self.assertEqual(
                json.loads((first / "summary.json").read_text(encoding="utf-8"))["cache_version"],
                cache.CACHE_VERSION,
            )
            scoring = cache.load_unknown_scoring_clip_cache(first, ffmpeg=self.ffmpeg)
            feature, metadata = scoring[0]
            self.assertEqual(feature.shape, cache.NATIVE_FEATURE_SHAPE)
            self.assertEqual(metadata["selection_strategy"], "energy")
            self.assertEqual(metadata["data_boundary"], "unknown_scoring_only")
            self.assertNotIn("split", metadata)
            index_ids = {row["candidate_id"] for row in scoring.rows}
            self.assertEqual(index_ids, {"XC1", "XC2", "XC3", "XC4"})
            self.assertNotIn("XC999", index_ids)

    def test_interrupted_build_revalidates_completed_record_before_resume(self) -> None:
        destination = self.cache_destination("resumed")

        def interrupted(_descriptor, source_path, *_args, **_kwargs):
            if Path(source_path).stem == "XC2":
                raise RuntimeError("fixture interruption")
            return np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)

        with (
            self.build_environment(interrupted),
            self.assertRaisesRegex(RuntimeError, "fixture interruption"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
        completed = destination.with_name(f".{destination.name}.working") / "completed"
        self.assertTrue((completed / "XC1" / "checkpoint.json").is_file())

        decoded: list[str] = []

        def resumed(_descriptor, source_path, *_args, **_kwargs):
            decoded.append(Path(source_path).stem)
            return np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)

        with self.build_environment(resumed):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
        self.assertEqual(decoded, ["XC1", "XC2", "XC3", "XC4"])

    def test_feature_corruption_is_rejected(self) -> None:
        with self.build_environment():
            destination = self.build("corrupt")
            feature = destination / "scoring" / "features" / "XC1.npy"
            feature.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "feature hash drift"):
                cache.verify_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_unselected_raw_drift_is_rejected_by_full_input_binding(self) -> None:
        unselected = self.raw_root / "Gamma_avis" / "XC999.audio"
        unselected.write_bytes(b"changed-unselected")
        unselected.chmod(0o600)
        with self.assertRaisesRegex(RuntimeError, "Input changed"):
            cache._require_inputs_unchanged(self.inputs)

    def test_extra_raw_directory_is_rejected_by_physical_tree_binding(self) -> None:
        (self.raw_root / "unexpected").mkdir()
        with self.assertRaisesRegex(RuntimeError, "full input file sets changed"):
            cache._require_inputs_unchanged(self.inputs)

    def test_energy_start_outside_candidate_grid_is_rejected(self) -> None:
        record = self.selected_rows[0]
        waveform = np.zeros(cache.CLIP_SAMPLES + cache.ENERGY_CANDIDATE_HOP_SAMPLES)
        with patch.object(
            cache,
            "_decode_audio_descriptor",
            return_value=waveform.astype(np.float32),
        ):
            rows, _, _, _ = cache._derive_recording(
                record,
                self.raw_root,
                self.ffmpeg,
                0.1,
                2.0,
            )
        row = {field: str(rows[0][field]) for field in cache.INDEX_FIELDS}
        row["start_sample"] = "1"
        row["clip_id"] = f"{record.candidate_id}:000000000001"
        with self.assertRaisesRegex(ValueError, "locked energy grid"):
            cache._validate_index_row(row, record)

    def test_resume_energy_tamper_is_recomputed_and_rejected(self) -> None:
        destination = self.cache_destination("tampered_resume")

        def interrupted(_descriptor, source_path, *_args, **_kwargs):
            if Path(source_path).stem == "XC2":
                raise RuntimeError("fixture interruption")
            return np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)

        with (
            self.build_environment(interrupted),
            self.assertRaisesRegex(RuntimeError, "fixture interruption"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
        checkpoint_path = (
            destination.with_name(f".{destination.name}.working")
            / "completed"
            / "XC1"
            / "checkpoint.json"
        )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["index_rows"][0]["energy_value"] = "1"
        _write_private_json(checkpoint_path, checkpoint)
        with (
            self.build_environment(),
            self.assertRaisesRegex(ValueError, "differs from sealed audio"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_nonprivate_resume_checkpoint_is_rejected(self) -> None:
        destination = self.cache_destination("private_resume")

        def interrupted(_descriptor, source_path, *_args, **_kwargs):
            if Path(source_path).stem == "XC2":
                raise RuntimeError("fixture interruption")
            return np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)

        with (
            self.build_environment(interrupted),
            self.assertRaisesRegex(RuntimeError, "fixture interruption"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
        checkpoint = (
            destination.with_name(f".{destination.name}.working")
            / "completed"
            / "XC1"
            / "checkpoint.json"
        )
        checkpoint.chmod(0o644)
        with (
            self.build_environment(),
            self.assertRaisesRegex(ValueError, "private mode 0600"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_stale_implementation_invalidates_resume_identity(self) -> None:
        destination = self.cache_destination("stale_implementation")

        def interrupted(_descriptor, source_path, *_args, **_kwargs):
            if Path(source_path).stem == "XC2":
                raise RuntimeError("fixture interruption")
            return np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)

        with (
            self.build_environment(interrupted),
            self.assertRaisesRegex(RuntimeError, "fixture interruption"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)
        with (
            self.build_environment(),
            patch.object(cache, "_implementation_fingerprint", return_value="f" * 64),
            self.assertRaisesRegex(ValueError, "resume state differs"),
        ):
            cache.build_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_existing_valid_destination_returns_idempotently(self) -> None:
        decoded: list[str] = []

        def decode(_descriptor, source_path, *_args, **_kwargs):
            decoded.append(Path(source_path).stem)
            return np.zeros(cache.CLIP_SAMPLES, dtype=np.float32)

        with self.build_environment(decode):
            destination = self.build("idempotent")
            first_lock = sha256_file(destination / "lock.json")
            decoded.clear()
            returned, summary = cache.build_unknown_clip_cache(
                destination,
                ffmpeg=self.ffmpeg,
            )
        self.assertEqual(returned, destination)
        self.assertEqual(summary["totals"]["recordings"], 4)
        self.assertEqual(sha256_file(destination / "lock.json"), first_lock)
        self.assertEqual(decoded, ["XC1", "XC2", "XC3", "XC4"])

    def test_expected_lock_mismatch_is_rejected(self) -> None:
        with self.build_environment():
            destination = self.build("expected_lock")
            with self.assertRaisesRegex(ValueError, "expected SHA-256"):
                cache.verify_unknown_clip_cache(
                    destination,
                    ffmpeg=self.ffmpeg,
                    expected_lock_sha256="0" * 64,
                )

    def test_extra_feature_file_is_rejected(self) -> None:
        with self.build_environment():
            destination = self.build("extra_feature")
            extra = destination / "scoring" / "features" / "extra.npy"
            extra.write_bytes(b"unexpected")
            extra.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "physical feature file set"):
                cache.verify_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_feature_hardlink_alias_is_rejected(self) -> None:
        with self.build_environment():
            destination = self.build("hardlink_feature")
            feature = destination / "scoring" / "features" / "XC1.npy"
            os.link(feature, self.root / "feature_alias.npy")
            with self.assertRaisesRegex(ValueError, "single-link regular file"):
                cache.verify_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_symlinked_cache_root_alias_is_rejected(self) -> None:
        with self.build_environment():
            destination = self.build("canonical_root")
            alias = self.root / "alias_root" / cache.CACHE_VERSION
            alias.parent.mkdir()
            alias.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "cache root is invalid"):
                cache.verify_unknown_clip_cache(alias, ffmpeg=self.ffmpeg)

    def test_late_index_symlink_replacement_is_rejected(self) -> None:
        with self.build_environment():
            destination = self.build("index_symlink")
            index = destination / "scoring" / "index.csv"
            moved = self.root / "moved_index.csv"
            index.rename(moved)
            index.symlink_to(moved)
            with self.assertRaisesRegex(ValueError, "scoring partition|symbolic link"):
                cache.verify_unknown_clip_cache(destination, ffmpeg=self.ffmpeg)

    def test_advisory_lock_reuses_a_stale_lock_file(self) -> None:
        destination = self.cache_destination("advisory")
        destination.parent.mkdir()
        lock_path = destination.with_name(f".{destination.name}.build.lock")
        _write_private_json(lock_path, {"stale": True})
        with (
            cache._unknown_cache_build_lock(destination),
            self.assertRaisesRegex(RuntimeError, "already active"),
            cache._unknown_cache_build_lock(destination),
        ):
            self.fail("second lock unexpectedly acquired")
        with cache._unknown_cache_build_lock(destination):
            self.assertTrue(lock_path.is_file())

    def test_advisory_lock_is_released_by_hard_process_exit(self) -> None:
        destination = self.cache_destination("hard_exit")
        destination.parent.mkdir()
        script = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "from bird_audio.unknown_clip_cache import _unknown_cache_build_lock\n"
            "with _unknown_cache_build_lock(Path(sys.argv[1])):\n"
            "    os._exit(0)\n"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(destination)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        with cache._unknown_cache_build_lock(destination):
            self.assertTrue(destination.with_name(f".{destination.name}.build.lock").is_file())

    def test_create_only_directory_publish_does_not_replace_destination(self) -> None:
        source = self.root / "source"
        destination = self.root / "destination"
        source.mkdir()
        destination.mkdir()
        (source / "source.txt").write_text("source", encoding="utf-8")
        (destination / "destination.txt").write_text("destination", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            cache._atomic_publish_directory_no_replace(source, destination)
        self.assertTrue((source / "source.txt").is_file())
        self.assertEqual(
            (destination / "destination.txt").read_text(encoding="utf-8"),
            "destination",
        )


if __name__ == "__main__":
    unittest.main()
