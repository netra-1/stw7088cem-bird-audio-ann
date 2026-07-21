from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from bird_audio.hashing import sha256_file, sha256_json
from bird_audio.metadata import API_VERSION, DEFAULT_ENDPOINT
from bird_audio.metadata_artifacts import (
    create_enrichment_lock,
    seal_metadata_cache,
    verify_enrichment_lock,
    verify_metadata_cache_lock,
)
from bird_audio.paths import PROJECT_ROOT


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MetadataArtifactTests(unittest.TestCase):
    def _fixture(self, root: Path, status: str = "ok") -> tuple[Path, Path]:
        local = root / "local.csv"
        _write_csv(local, [{"recording_id": "XC123", "xc_id": "123"}])
        local_sha256 = sha256_file(local)
        entry = {
            "status": status,
            "fetched_at_utc": "2026-07-13T00:00:00+00:00",
            "recording": {"id": "123"} if status == "ok" else {},
            "error": ""
            if status == "ok"
            else "recording unavailable"
            if status == "unavailable"
            else "temporary failure",
        }
        cache = root / "working.json"
        cache.write_text(
            json.dumps(
                {
                    "schema_version": "1.1",
                    "api_version": API_VERSION,
                    "endpoint": DEFAULT_ENDPOINT,
                    "query_form": "nr:<xc_id>",
                    "source_manifest_sha256": local_sha256,
                    "source_recording_ids_sha256": sha256_json(["123"]),
                    "records": {"123": entry},
                }
            ),
            encoding="utf-8",
        )
        return local, cache

    def test_complete_cache_is_sealed_and_tamper_detected(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            local, working = self._fixture(root)
            sealed = root / "sealed.json"
            lock = root / "cache_lock.json"
            seal_metadata_cache(local, working, sealed, lock)
            self.assertTrue(verify_metadata_cache_lock(lock)["ready_for_enrichment"])

            payload = json.loads(sealed.read_text(encoding="utf-8"))
            payload["records"]["123"]["recording"]["id"] = "999"
            sealed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_metadata_cache_lock(lock)

    def test_incomplete_cache_cannot_be_sealed(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            local, working = self._fixture(root, status="error")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                seal_metadata_cache(
                    local,
                    working,
                    root / "sealed.json",
                    root / "cache_lock.json",
                )

    def test_terminal_unavailable_cache_can_be_sealed_and_working_cache_can_change(
        self,
    ) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            local, working = self._fixture(root, status="unavailable")
            sealed = root / "sealed.json"
            lock = root / "cache_lock.json"
            seal_metadata_cache(local, working, sealed, lock)
            self.assertTrue(verify_metadata_cache_lock(lock)["ready_for_enrichment"])

            working.write_text('{"staging":"changed after sealing"}\n', encoding="utf-8")
            self.assertTrue(verify_metadata_cache_lock(lock)["ready_for_enrichment"])

    def test_enrichment_lock_binds_every_artifact(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            local, working = self._fixture(root)
            sealed = root / "sealed.json"
            cache_lock = root / "cache_lock.json"
            seal_metadata_cache(local, working, sealed, cache_lock)

            enriched = root / "enriched.csv"
            licences = root / "licences.csv"
            summary = root / "summary.json"
            enrichment_lock = root / "enrichment_lock.json"
            _write_csv(
                enriched,
                [
                    {
                        "recording_id": "XC123",
                        "metadata_status": "ok",
                        "local_qc_status": "include",
                    }
                ],
            )
            _write_csv(licences, [{"recording_id": "XC123", "licence": "recognized"}])
            summary.write_text(
                json.dumps(
                    {
                        "source_local_manifest_sha256": sha256_file(local),
                        "source_metadata_cache_sha256": sha256_file(sealed),
                        "enriched_manifest_sha256": sha256_file(enriched),
                        "recordings": 1,
                        "ready_for_manual_review": True,
                    }
                ),
                encoding="utf-8",
            )
            create_enrichment_lock(
                "configs/data.toml",
                local,
                sealed,
                cache_lock,
                enriched,
                licences,
                summary,
                enrichment_lock,
            )
            self.assertTrue(
                verify_enrichment_lock(enrichment_lock, enriched)["ready_for_manual_review"]
            )

            licences.write_text("recording_id,licence\nXC123,tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_enrichment_lock(enrichment_lock, enriched)

    def test_enrichment_lock_accepts_terminal_unavailable_exclusion(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=interim) as temporary:
            root = Path(temporary)
            local, working = self._fixture(root, status="unavailable")
            sealed = root / "sealed.json"
            cache_lock = root / "cache_lock.json"
            seal_metadata_cache(local, working, sealed, cache_lock)

            enriched = root / "enriched.csv"
            licences = root / "licences.csv"
            summary = root / "summary.json"
            enrichment_lock = root / "enrichment_lock.json"
            _write_csv(
                enriched,
                [
                    {
                        "recording_id": "XC123",
                        "metadata_status": "unavailable",
                        "local_qc_status": "exclude",
                    }
                ],
            )
            _write_csv(licences, [{"recording_id": "XC123", "licence": ""}])
            summary.write_text(
                json.dumps(
                    {
                        "source_local_manifest_sha256": sha256_file(local),
                        "source_metadata_cache_sha256": sha256_file(sealed),
                        "enriched_manifest_sha256": sha256_file(enriched),
                        "recordings": 1,
                        "ready_for_manual_review": True,
                    }
                ),
                encoding="utf-8",
            )
            create_enrichment_lock(
                "configs/data.toml",
                local,
                sealed,
                cache_lock,
                enriched,
                licences,
                summary,
                enrichment_lock,
            )
            self.assertTrue(
                verify_enrichment_lock(enrichment_lock, enriched)["ready_for_manual_review"]
            )


if __name__ == "__main__":
    unittest.main()
