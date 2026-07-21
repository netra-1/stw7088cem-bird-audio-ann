from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from bird_audio.hashing import sha256_file
from bird_audio.paths import PROJECT_ROOT
from bird_audio.splitting import (
    _verify_raw_bindings,
    allocate_grouped_split,
    freeze_grouped_split,
    integer_targets,
    validate_frozen_split,
)


class GroupedSplitTests(unittest.TestCase):
    def test_raw_binding_verifier_checks_safe_path_and_current_bytes(self) -> None:
        manifest = PROJECT_ROOT / "data" / "manifests" / "local_recordings.csv"
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertTrue(_verify_raw_bindings([row])["valid"])

        tampered = dict(row)
        tampered["sha256"] = "0" * 64
        result = _verify_raw_bindings([tampered])
        self.assertFalse(result["valid"])
        self.assertIn("raw_sha256_mismatch", result["failures"][0])

        unsafe = dict(row)
        unsafe["relative_path"] = "data/manifests/local_recordings.csv"
        result = _verify_raw_bindings([unsafe])
        self.assertFalse(result["valid"])
        self.assertIn("unsafe_raw_path", result["failures"][0])

    def test_integer_targets_preserve_total(self) -> None:
        targets = integer_targets(
            10,
            {"train": 0.70, "validation": 0.15, "test": 0.15},
        )
        self.assertEqual(targets, {"train": 7, "validation": 2, "test": 1})

    def test_all_recordings_in_session_receive_same_split(self) -> None:
        rows = [
            {
                "recording_id": f"XC{species_index}{recording_index}",
                "species_common_name": species,
                "session_group": f"session:{recording_index}",
            }
            for species_index, species in enumerate(("A", "B", "C"), start=1)
            for recording_index in range(10)
        ]
        fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
        assignment, diagnostics = allocate_grouped_split(rows, fractions, seed=20260713)
        self.assertEqual(len(assignment), 10)
        self.assertEqual(sum(diagnostics["global_achieved"].values()), 30)
        for species in ("A", "B", "C"):
            achieved = diagnostics["achieved"][species]
            self.assertEqual(sum(achieved.values()), 10)
            self.assertGreater(achieved["train"], 0)
            self.assertGreater(achieved["validation"], 0)
            self.assertGreater(achieved["test"], 0)

    def test_allocation_is_deterministic(self) -> None:
        rows = [
            {
                "recording_id": f"XC{index}",
                "species_common_name": "A" if index % 2 else "B",
                "session_group": f"session:{index // 2}",
            }
            for index in range(40)
        ]
        fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
        first, _ = allocate_grouped_split(rows, fractions, seed=13)
        second, _ = allocate_grouped_split(rows, fractions, seed=13)
        self.assertEqual(first, second)
        self.assertEqual(Counter(first.values())["train"] > 0, True)

    def test_relocation_is_rechecked_after_an_accepted_swap(self) -> None:
        rows = []
        for group_index, size in enumerate((6, 1, 4, 1, 5)):
            rows.extend(
                {
                    "recording_id": f"XC{group_index}{item}",
                    "species_common_name": "A",
                    "session_group": f"session:{group_index}",
                }
                for item in range(size)
            )
        _, diagnostics = allocate_grouped_split(
            rows,
            {"train": 0.70, "validation": 0.15, "test": 0.15},
            seed=20260713,
        )
        self.assertEqual(diagnostics["achieved"]["A"]["test"], 2)
        self.assertTrue(diagnostics["locally_optimal_for_single_relocations_and_pair_swaps"])

    def test_infeasible_class_coverage_is_rejected(self) -> None:
        rows = [
            {
                "recording_id": f"XC{index}",
                "species_common_name": "A",
                "session_group": "session:only",
            }
            for index in range(10)
        ]
        with self.assertRaises(RuntimeError):
            allocate_grouped_split(
                rows,
                {"train": 0.70, "validation": 0.15, "test": 0.15},
                seed=13,
            )

    def test_split_lock_binds_every_manifest_field_and_artifact(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            split = root / "split.csv"
            summary = root / "summary.json"
            lock = root / "lock.json"
            review_lock = root / "review_lock.json"
            fields = [
                "recording_id",
                "relative_path",
                "sha256",
                "species_common_name",
                "session_group",
                "local_qc_status",
                "metadata_status",
                "session_review_flag",
            ]
            rows = [
                {
                    "recording_id": f"XC{species_index:02d}{session_index:02d}",
                    "relative_path": f"dataset/species_{species_index}/{session_index}.mp3",
                    "sha256": f"{species_index:02x}{session_index:02x}".ljust(64, "0"),
                    "species_common_name": f"Species {species_index:02d}",
                    "session_group": f"session:{session_index:02d}",
                    "local_qc_status": "include",
                    "metadata_status": "ok",
                    "session_review_flag": "false",
                }
                for species_index in range(15)
                for session_index in range(10)
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            review_lock.write_text("{}\n", encoding="utf-8")
            review_record = {
                "ready_for_split": True,
                "final_manifest_sha256": sha256_file(manifest),
            }

            with (
                patch(
                    "bird_audio.splitting.verify_review_lock",
                    return_value=review_record,
                ),
                patch(
                    "bird_audio.splitting._verify_raw_bindings",
                    return_value={
                        "valid": True,
                        "recordings_checked": len(rows),
                        "failures": [],
                    },
                ),
            ):
                freeze_grouped_split(
                    "configs/data.toml",
                    manifest,
                    split,
                    summary,
                    lock,
                    review_lock_path=review_lock,
                )
                valid = validate_frozen_split(
                    manifest,
                    split,
                    lock,
                    config_path="configs/data.toml",
                    summary_path=summary,
                    review_lock_path=review_lock,
                )
            self.assertTrue(valid["valid"])

            with split.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                split_rows = list(reader)
                split_fields = list(reader.fieldnames or [])
            split_rows[0]["relative_path"] = "dataset/tampered.mp3"
            with split.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=split_fields)
                writer.writeheader()
                writer.writerows(split_rows)
            with (
                patch(
                    "bird_audio.splitting.verify_review_lock",
                    return_value=review_record,
                ),
                patch(
                    "bird_audio.splitting._verify_raw_bindings",
                    return_value={
                        "valid": True,
                        "recordings_checked": len(rows),
                        "failures": [],
                    },
                ),
            ):
                invalid = validate_frozen_split(
                    manifest,
                    split,
                    lock,
                    config_path="configs/data.toml",
                    summary_path=summary,
                    review_lock_path=review_lock,
                )
            self.assertFalse(invalid["valid"])
            self.assertFalse(invalid["checks"]["row_bindings_match_manifest"])

    def test_unresolved_session_review_flag_blocks_freeze(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            review_lock = root / "review_lock.json"
            fields = [
                "recording_id",
                "relative_path",
                "sha256",
                "species_common_name",
                "session_group",
                "local_qc_status",
                "metadata_status",
                "session_review_flag",
            ]
            row = {
                "recording_id": "XC123",
                "relative_path": "dataset/species/123.mp3",
                "sha256": "a" * 64,
                "species_common_name": "Species",
                "session_group": "session:123",
                "local_qc_status": "include",
                "metadata_status": "ok",
                "session_review_flag": "true",
            }
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            review_lock.write_text("{}\n", encoding="utf-8")
            with (
                patch(
                    "bird_audio.splitting.verify_review_lock",
                    return_value={
                        "ready_for_split": True,
                        "final_manifest_sha256": sha256_file(manifest),
                    },
                ),
                patch(
                    "bird_audio.splitting._verify_raw_bindings",
                    return_value={
                        "valid": True,
                        "recordings_checked": 1,
                        "failures": [],
                    },
                ),
                self.assertRaisesRegex(ValueError, "session-review flags"),
            ):
                freeze_grouped_split(
                    "configs/data.toml",
                    manifest,
                    root / "split.csv",
                    root / "summary.json",
                    root / "lock.json",
                    review_lock_path=review_lock,
                )


if __name__ == "__main__":
    unittest.main()
