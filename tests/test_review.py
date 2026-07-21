from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bird_audio.hashing import sha256_file
from bird_audio.paths import PROJECT_ROOT
from bird_audio.review import (
    REVIEW_DECISION_FIELDS,
    REVIEW_ITEM_CONTEXT_FIELDS,
    apply_manual_review,
    prepare_manual_review,
    verify_review_lock,
)


def review_row(recording_id: str, **changes: str) -> dict[str, str]:
    xc_id = recording_id.removeprefix("XC")
    row = {
        "recording_id": recording_id,
        "xc_id": xc_id,
        "xc_url": f"https://xeno-canto.org/{xc_id}",
        "relative_path": f"dataset/species/{xc_id}.mp3",
        "sha256": (xc_id * 64)[:64],
        "species_folder": "Species",
        "species_common_name": "Known Bird",
        "scientific_name": "Avis cognita",
        "primary_label": "Known Bird",
        "api_scientific_name": "Avis cognita",
        "api_group": "birds",
        "secondary_labels": "[]",
        "target_secondary_labels": "[]",
        "recordist": "Recorder",
        "country": "Nepal",
        "locality": "Kathmandu",
        "latitude": "27.7000",
        "longitude": "85.3000",
        "recorded_date": "2025-04-10",
        "recorded_time": "06:30",
        "quality": "A",
        "remarks": "",
        "licence": "https://creativecommons.org/licenses/by/4.0/",
        "licence_status": "recorded",
        "attribution": f"{recording_id}, Known Bird, Recorder",
        "metadata_status": "ok",
        "metadata_error": "",
        "identity_validation_status": "exact_match",
        "licence_validation_status": "recognized_cc",
        "probe_ok": "true",
        "full_decode_status": "ok",
        "local_qc_status": "include",
        "exclusion_reasons": "",
        "session_group": f"session:{xc_id}",
        "session_review_flag": "false",
        "session_review_reason": "",
    }
    row.update(changes)
    return row


class ManualReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=interim)
        self.root = Path(self.temporary.name)
        self.source = self.root / "enriched.csv"
        self.enrichment_lock = self.root / "enrichment_lock.json"
        self.items = self.root / "review_items.csv"
        self.decisions = self.root / "review_decisions.csv"
        self.preparation = self.root / "review_preparation.json"
        self.final = self.root / "recordings.csv"
        self.resolution = self.root / "review_resolution.json"
        self.review_lock = self.root / "review_lock.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_source(self, rows: list[dict[str, str]]) -> None:
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_ITEM_CONTEXT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.enrichment_lock.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "ready_for_manual_review": True,
                    "enriched_manifest_sha256": sha256_file(self.source),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _fake_verify(lock_path: str | Path, expected_path: str | Path) -> dict:
        lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        if lock["enriched_manifest_sha256"] != sha256_file(Path(expected_path)):
            raise ValueError("enrichment binding mismatch")
        return lock

    def _prepare(self) -> dict:
        _, preparation = prepare_manual_review(
            self.source,
            self.enrichment_lock,
            self.items,
            self.decisions,
            self.preparation,
        )
        return preparation

    def _complete_decisions(
        self,
        decision: str,
        reason: str,
        confirmed_session_group: str = "",
    ) -> None:
        with self.decisions.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update(
                {
                    "decision": decision,
                    "decision_reason": reason,
                    "confirmed_session_group": confirmed_session_group,
                }
            )
        with self.decisions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_DECISION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _apply(self) -> tuple[Path, Path, dict]:
        return apply_manual_review(
            self.source,
            self.enrichment_lock,
            self.items,
            self.decisions,
            self.preparation,
            self.final,
            self.resolution,
            self.review_lock,
        )

    def test_prepare_apply_and_verify_hash_bound_review(self) -> None:
        rows = [
            review_row("XC100"),
            review_row(
                "XC200",
                local_qc_status="manual_review",
                exclusion_reasons="session_date_missing_manual_review",
                session_review_flag="true",
                session_review_reason="session_date_missing",
            ),
            review_row(
                "XC300",
                local_qc_status="exclude",
                exclusion_reasons="target_species_in_secondary_labels",
                target_secondary_labels='["Known Bird"]',
                session_review_flag="true",
                session_review_reason="session_recordist_missing",
            ),
        ]
        self._write_source(rows)
        with patch(
            "bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify
        ) as verify_mock:
            preparation = self._prepare()
            self.assertEqual(preparation["review_items"], 1)
            self.assertEqual(preparation["not_required"], 2)
            self.assertEqual(preparation["source_manifest_sha256"], sha256_file(self.source))
            self.assertEqual(preparation["review_items_sha256"], sha256_file(self.items))

            self._complete_decisions(
                "include",
                "Recording date and session membership checked",
                "session:200",
            )
            final, lock_path, resolution = self._apply()
            self.assertEqual(final, self.final)
            self.assertEqual(lock_path, self.review_lock)
            self.assertTrue(resolution["ready_for_split"])
            self.assertEqual(resolution["decision_counts"], {"include": 1})

            with final.open("r", encoding="utf-8", newline="") as handle:
                final_rows = {row["recording_id"]: row for row in csv.DictReader(handle)}
            reviewed = final_rows["XC200"]
            self.assertEqual(reviewed["review_status"], "resolved_include")
            self.assertEqual(reviewed["local_qc_status"], "include")
            self.assertEqual(reviewed["session_review_flag"], "false")
            self.assertEqual(reviewed["session_review_reason"], "")
            self.assertEqual(reviewed["exclusion_reasons"], "")
            self.assertEqual(
                reviewed["review_original_exclusion_reasons"],
                "session_date_missing_manual_review",
            )
            self.assertEqual(final_rows["XC100"]["review_status"], "not_required")
            self.assertEqual(final_rows["XC300"]["session_review_flag"], "true")

            verified = verify_review_lock(self.review_lock, self.final)
            self.assertTrue(verified["ready_for_split"])
            self.assertEqual(verified["final_manifest_sha256"], sha256_file(self.final))
            for artifact in verified["artifacts"].values():
                self.assertFalse(Path(artifact["path"]).is_absolute())
            self.assertGreaterEqual(verify_mock.call_count, 3)

    def test_incomplete_decision_is_rejected_before_outputs(self) -> None:
        self._write_source(
            [
                review_row("XC100"),
                review_row(
                    "XC200",
                    local_qc_status="manual_review",
                    exclusion_reasons="session_date_missing_manual_review",
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            with self.assertRaisesRegex(ValueError, "Decision must be include or exclude"):
                self._apply()
        self.assertFalse(self.final.exists())
        self.assertFalse(self.review_lock.exists())

    def test_non_session_reason_cannot_be_overridden_by_include(self) -> None:
        self._write_source(
            [
                review_row("XC100"),
                review_row(
                    "XC200",
                    local_qc_status="manual_review",
                    exclusion_reasons="licence_missing_manual_review",
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            self._complete_decisions("include", "Requested inclusion")
            with self.assertRaisesRegex(ValueError, "not eligible for inclusion"):
                self._apply()
        self.assertFalse(self.final.exists())

    def test_exclude_decision_preserves_original_reasons(self) -> None:
        self._write_source(
            [
                review_row("XC100"),
                review_row(
                    "XC200",
                    metadata_status="error",
                    metadata_error="request failed",
                    identity_validation_status="mismatch",
                    licence="",
                    licence_validation_status="missing",
                    local_qc_status="manual_review",
                    exclusion_reasons="metadata_fetch_failed",
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            self._complete_decisions("exclude", "Metadata could not be validated")
            _, _, resolution = self._apply()
            self.assertTrue(resolution["ready_for_split"])

        with self.final.open("r", encoding="utf-8", newline="") as handle:
            final_rows = {row["recording_id"]: row for row in csv.DictReader(handle)}
        excluded = final_rows["XC200"]
        self.assertEqual(excluded["review_status"], "resolved_exclude")
        self.assertEqual(
            excluded["exclusion_reasons"],
            "metadata_fetch_failed;manual_review_decision_exclude",
        )
        self.assertEqual(excluded["review_original_exclusion_reasons"], "metadata_fetch_failed")

    def test_item_tampering_breaks_preparation_binding(self) -> None:
        self._write_source(
            [
                review_row("XC100"),
                review_row(
                    "XC200",
                    local_qc_status="manual_review",
                    exclusion_reasons="session_date_missing_manual_review",
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            with self.items.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                item_rows = list(reader)
                fields = list(reader.fieldnames or [])
            item_rows[0]["locality"] = "Tampered"
            with self.items.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(item_rows)
            self._complete_decisions("exclude", "Cannot trust changed item")
            with self.assertRaisesRegex(ValueError, "preparation binding mismatch"):
                self._apply()

    def test_prepare_refuses_an_enrichment_lock_for_another_manifest(self) -> None:
        self._write_source([review_row("XC100")])
        lock = json.loads(self.enrichment_lock.read_text(encoding="utf-8"))
        lock["enriched_manifest_sha256"] = "0" * 64
        self.enrichment_lock.write_text(json.dumps(lock), encoding="utf-8")
        with (
            patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify),
            self.assertRaisesRegex(ValueError, "enrichment binding mismatch"),
        ):
            self._prepare()
        self.assertFalse(self.items.exists())

    def test_zero_item_review_still_creates_and_verifies_final_lock(self) -> None:
        self._write_source(
            [
                review_row("XC100"),
                review_row(
                    "XC200",
                    local_qc_status="exclude",
                    exclusion_reasons="target_species_in_secondary_labels",
                    target_secondary_labels='["Known Bird"]',
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            preparation = self._prepare()
            self.assertEqual(preparation["review_items"], 0)
            _, _, resolution = self._apply()
            self.assertTrue(resolution["ready_for_split"])
            self.assertEqual(resolution["decision_counts"], {})
            self.assertTrue(verify_review_lock(self.review_lock, self.final)["ready_for_split"])

    def test_cross_species_session_reassignment_is_rejected(self) -> None:
        self._write_source(
            [
                review_row(
                    "XC200",
                    local_qc_status="manual_review",
                    exclusion_reasons="session_date_missing_manual_review",
                    session_review_flag="true",
                    session_review_reason="session_date_missing",
                ),
                review_row(
                    "XC300",
                    species_folder="Other",
                    species_common_name="Other Bird",
                    scientific_name="Avis altera",
                    primary_label="Other Bird",
                    api_scientific_name="Avis altera",
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            self._complete_decisions(
                "include",
                "Attempted cross-species reassignment",
                "session:300",
            )
            with self.assertRaisesRegex(ValueError, "must confirm its original group"):
                self._apply()

    def test_original_multi_recording_session_cannot_be_fractured(self) -> None:
        self._write_source(
            [
                review_row(
                    "XC200",
                    session_group="session:shared",
                    local_qc_status="manual_review",
                    exclusion_reasons="session_date_missing_manual_review",
                    session_review_flag="true",
                    session_review_reason="session_date_missing",
                ),
                review_row("XC201", session_group="session:shared"),
                review_row("XC300"),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            self._complete_decisions(
                "include",
                "Attempted session split",
                "session:300",
            )
            with self.assertRaisesRegex(ValueError, "must confirm its original group"):
                self._apply()

    def test_decision_context_cannot_be_edited(self) -> None:
        self._write_source(
            [
                review_row("XC100"),
                review_row(
                    "XC200",
                    local_qc_status="manual_review",
                    exclusion_reasons="session_date_missing_manual_review",
                ),
            ]
        )
        with patch("bird_audio.review.verify_enrichment_lock", side_effect=self._fake_verify):
            self._prepare()
            with self.decisions.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["species_common_name"] = "Tampered Bird"
            rows[0]["decision"] = "exclude"
            rows[0]["decision_reason"] = "Context was changed"
            with self.decisions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_DECISION_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Immutable decision context changed"):
                self._apply()


if __name__ == "__main__":
    unittest.main()
