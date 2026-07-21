from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bird_audio.hashing import sha256_file
from bird_audio.io_utils import require_unchanged
from bird_audio.manifest import (
    _annotate_duplicate_groups,
    apply_qc_reason,
    summarize_local_manifest,
)


class ManifestQcTests(unittest.TestCase):
    def test_manual_review_never_downgrades_exclusion(self) -> None:
        row = {
            "local_qc_status": "exclude",
            "exclusion_reasons": "source_sample_rate_below_32000_hz",
        }
        apply_qc_reason(row, "decode_warning_manual_review", "manual_review")
        self.assertEqual(row["local_qc_status"], "exclude")
        self.assertEqual(
            row["exclusion_reasons"],
            "decode_warning_manual_review;source_sample_rate_below_32000_hz",
        )

    def test_duplicate_canonicalization_excludes_only_later_recording(self) -> None:
        rows = [
            {
                "sha256": "a" * 64,
                "xc_id": "200",
                "recording_id": "XC200",
                "species_common_name": "Species",
                "local_qc_status": "pending_metadata",
                "exclusion_reasons": "",
            },
            {
                "sha256": "a" * 64,
                "xc_id": "100",
                "recording_id": "XC100",
                "species_common_name": "Species",
                "local_qc_status": "pending_metadata",
                "exclusion_reasons": "",
            },
        ]
        _annotate_duplicate_groups(rows)
        by_id = {row["recording_id"]: row for row in rows}
        self.assertEqual(by_id["XC100"]["duplicate_canonical_recording_id"], "XC100")
        self.assertEqual(by_id["XC100"]["local_qc_status"], "pending_metadata")
        self.assertEqual(by_id["XC200"]["local_qc_status"], "exclude")
        self.assertIn("exact_duplicate_noncanonical", by_id["XC200"]["exclusion_reasons"])

    def test_stale_input_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            path.write_text("before\n", encoding="utf-8")
            digest = sha256_file(path)
            path.write_text("after\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                require_unchanged(path, digest)

    def test_raw_data_fingerprint_is_independent_of_manifest_row_order(self) -> None:
        first = {
            "relative_path": "dataset/Species/200.mp3",
            "sha256": "b" * 64,
            "species_folder": "Species",
            "species_common_name": "Species",
        }
        second = {
            "relative_path": "dataset/Species/100.mp3",
            "sha256": "a" * 64,
            "species_folder": "Species",
            "species_common_name": "Species",
        }
        forward = summarize_local_manifest([first, second], "0" * 64, 32000)
        reverse = summarize_local_manifest([second, first], "0" * 64, 32000)
        self.assertEqual(
            forward["raw_data_fingerprint"],
            reverse["raw_data_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
