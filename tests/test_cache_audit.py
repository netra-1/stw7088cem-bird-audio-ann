from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bird_audio.cache_audit import (
    CacheAuditError,
    _require_cross_split_disjointness,
    audit_known_clip_cache,
)
from bird_audio.clip_cache import INDEX_FIELDS
from bird_audio.hashing import sha256_file
from bird_audio.paths import PROJECT_ROOT
from bird_audio.splitting import SPLIT_NAMES


class IndependentCacheAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=interim)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "known_fixture_v1"
        self.cache.mkdir()
        self.split = self.root / "split.csv"
        self.lock = self.cache / "lock.json"
        self.split_rows: list[dict[str, str]] = []
        self.index_rows: dict[str, list[dict[str, str]]] = {}
        for number, split in enumerate(SPLIT_NAMES, start=1):
            recording_id = f"XC99000{number}"
            source_sha256 = str(number) * 64
            split_row = {
                "recording_id": recording_id,
                "relative_path": f"dataset/species/{recording_id}.mp3",
                "sha256": source_sha256,
                "species_common_name": f"Species {number}",
                "session_group": f"session:{number}",
                "split": split,
            }
            self.split_rows.append(split_row)
            self.index_rows[split] = [self._index_row(split_row)]
        self._write_split()
        self._refresh_lock()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _index_row(self, source: dict[str, str]) -> dict[str, str]:
        row = {field: "" for field in INDEX_FIELDS}
        row.update(
            {
                "schema_version": "1.0",
                "clip_id": f"{source['recording_id']}:000000000000",
                "recording_id": source["recording_id"],
                "relative_path": source["relative_path"],
                "source_sha256": source["sha256"],
                "species_common_name": source["species_common_name"],
                "class_index": "0",
                "session_group": source["session_group"],
                "split": source["split"],
                "start_sample": "0",
                "feature_row": "0",
                "cached_clip_count": "1",
                "uniform_clip_count": "1",
                "energy_clip_count": "1",
                "uniform_selected": "true",
                "uniform_rank": "0",
                "energy_selected": "true",
                "energy_rank": "0",
                "energy_value": "1.0",
            }
        )
        return row

    def _write_csv(
        self,
        path: Path,
        rows: list[dict[str, str]],
        fields: list[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_split(self) -> None:
        fields = [
            "recording_id",
            "relative_path",
            "sha256",
            "species_common_name",
            "session_group",
            "split",
        ]
        self._write_csv(self.split, self.split_rows, fields)

    def _refresh_lock(self) -> None:
        split_artifacts: dict[str, object] = {}
        for split in SPLIT_NAMES:
            index_path = self.cache / split / "index.csv"
            self._write_csv(index_path, self.index_rows[split], INDEX_FIELDS)
            split_artifacts[split] = {
                "index": {
                    "path": f"{split}/index.csv",
                    "sha256": sha256_file(index_path),
                    "rows": len(self.index_rows[split]),
                }
            }
        payload = {
            "provenance": {
                "input_paths": {
                    "split": self.split.relative_to(PROJECT_ROOT).as_posix(),
                },
                "split_sha256": sha256_file(self.split),
            },
            "artifacts": {"splits": split_artifacts},
        }
        self.lock.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _public_result(self, *_args, **_kwargs) -> dict[str, object]:
        return {
            "valid": True,
            "cache_version": "known_clips_v1",
            "lock_sha256": sha256_file(self.lock),
            "recordings": 3,
            "clips": sum(len(rows) for rows in self.index_rows.values()),
            "feature_files": 3,
        }

    def _audit(self) -> dict[str, object]:
        with patch(
            "bird_audio.cache_audit.verify_known_clip_cache",
            side_effect=self._public_result,
        ) as public_verify:
            result = audit_known_clip_cache(self.cache, ffmpeg=self.root / "ffmpeg")
        public_verify.assert_called_once_with(
            self.cache,
            ffmpeg=self.root / "ffmpeg",
            expected_lock_sha256=None,
        )
        return result

    def _add_energy_row(
        self,
        split: str,
        *,
        start_sample: int,
        rank: int,
        energy: float,
    ) -> None:
        first = self.index_rows[split][0]
        first["cached_clip_count"] = "2"
        first["energy_clip_count"] = "2"
        second = dict(first)
        second.update(
            {
                "clip_id": f"{first['recording_id']}:{start_sample:012d}",
                "start_sample": str(start_sample),
                "feature_row": "1",
                "uniform_selected": "false",
                "uniform_rank": "",
                "energy_rank": str(rank),
                "energy_value": str(energy),
            }
        )
        self.index_rows[split].append(second)

    def test_valid_cache_runs_public_verification_first_and_reports_compact_counts(self) -> None:
        result = self._audit()
        self.assertTrue(result["valid"])
        self.assertTrue(result["source_bindings_exact"])
        self.assertTrue(result["energy_selection_invariants_valid"])
        self.assertEqual(result["recordings"], 3)
        self.assertEqual(result["clips"], 3)

    def test_relocked_row_binding_drift_is_rejected(self) -> None:
        self.index_rows["train"][0]["session_group"] = "session:changed"
        self._refresh_lock()
        with self.assertRaisesRegex(CacheAuditError, "frozen split binding"):
            self._audit()

    def test_exact_per_split_recording_set_is_required(self) -> None:
        self.index_rows["test"] = []
        self._refresh_lock()
        with self.assertRaisesRegex(CacheAuditError, "recording set differs"):
            self._audit()

    def test_cross_split_recording_hash_and_session_sets_are_checked(self) -> None:
        self.split_rows[2]["session_group"] = self.split_rows[0]["session_group"]
        self.index_rows["test"][0]["session_group"] = self.split_rows[0]["session_group"]
        self._write_split()
        self._refresh_lock()
        with self.assertRaisesRegex(CacheAuditError, "overlap on session_group"):
            self._audit()

    def test_each_cross_split_identity_field_is_enforced(self) -> None:
        for field in ("recording_id", "source_sha256", "session_group"):
            with self.subTest(field=field):
                identities = {
                    split: {
                        "recording_id": {f"{split}:recording"},
                        "source_sha256": {f"{split}:sha256"},
                        "session_group": {f"{split}:session"},
                    }
                    for split in SPLIT_NAMES
                }
                identities["validation"][field] = set(identities["train"][field])
                with self.assertRaisesRegex(CacheAuditError, f"overlap on {field}"):
                    _require_cross_split_disjointness(identities)

    def test_energy_starts_must_obey_minimum_separation(self) -> None:
        self.index_rows["train"][0]["energy_value"] = "2.0"
        self._add_energy_row("train", start_sample=48_000, rank=1, energy=1.0)
        self._refresh_lock()
        with self.assertRaisesRegex(CacheAuditError, "minimum separation"):
            self._audit()

    def test_energy_ranks_follow_descending_energy_and_earlier_ties(self) -> None:
        cases = (
            (0, 1.0, 96_000, 2.0),
            (96_000, 1.0, 0, 1.0),
        )
        for first_start, first_energy, second_start, second_energy in cases:
            with self.subTest(
                first_start=first_start,
                first_energy=first_energy,
                second_start=second_start,
                second_energy=second_energy,
            ):
                original = self._index_row(self.split_rows[0])
                original.update(
                    {
                        "start_sample": str(first_start),
                        "clip_id": f"{original['recording_id']}:{first_start:012d}",
                        "energy_value": str(first_energy),
                    }
                )
                self.index_rows["train"] = [original]
                self._add_energy_row(
                    "train",
                    start_sample=second_start,
                    rank=1,
                    energy=second_energy,
                )
                self._refresh_lock()
                with self.assertRaisesRegex(CacheAuditError, "deterministic ordering"):
                    self._audit()


if __name__ == "__main__":
    unittest.main()
