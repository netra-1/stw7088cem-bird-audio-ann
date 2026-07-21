from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from bird_audio.hashing import sha256_json
from bird_audio.metadata import (
    API_VERSION,
    DEFAULT_ENDPOINT,
    XenoCantoApiError,
    XenoCantoClient,
    XenoCantoFatalApiError,
    XenoCantoRecordUnavailableError,
    _apply_metadata,
    _configured_target_labels,
    _is_recognized_cc_licence_uri,
    _labels_overlap,
    _load_cache,
    _same_binomial_identity,
    assign_session_groups,
    fetch_metadata_cache,
    load_metadata_cache_snapshot,
)
from bird_audio.paths import PROJECT_ROOT


def base_row() -> dict[str, str]:
    return {
        "recording_id": "XC123",
        "xc_id": "123",
        "species_common_name": "Asian Koel",
        "scientific_name": "Eudynamys scolopaceus",
        "local_qc_status": "pending_metadata",
        "exclusion_reasons": "",
        "session_group": "",
        "session_review_flag": "false",
        "session_review_reason": "",
    }


def cache_entry(
    secondary: list[str] | None = None,
    licence: str = "//creativecommons.org/licenses/by-nc-sa/4.0/",
) -> dict:
    return {
        "status": "ok",
        "fetched_at_utc": "2026-07-13T00:00:00+00:00",
        "recording": {
            "id": "123",
            "gen": "Eudynamys",
            "sp": "scolopaceus",
            "grp": "birds",
            "en": "Asian Koel",
            "rec": "Recorder",
            "cnt": "Nepal",
            "loc": "Location",
            "lat": "27.700",
            "lng": "85.300",
            "date": "2024-04-02",
            "time": "06:30",
            "q": "A",
            "type": ["song"],
            "also": secondary or [],
            "lic": licence,
            "file": "//audio.example/123.mp3",
            "rmk": "",
        },
    }


def http_error(status: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(
        "https://xeno-canto.org/api/3/recordings?key=must-not-leak",
        status,
        "request failed",
        headers,
        None,
    )


class MetadataTests(unittest.TestCase):
    def test_fetch_pacing_and_checkpoint_constraints_are_locked(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1.0"):
            fetch_metadata_cache(
                "does-not-matter.csv",
                "data/interim/does-not-matter.json",
                api_key="secret",
                request_interval_seconds=0.5,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint_every"):
            fetch_metadata_cache(
                "does-not-matter.csv",
                "data/interim/does-not-matter.json",
                api_key="secret",
                checkpoint_every=0,
            )

    @patch.object(XenoCantoClient, "_request")
    def test_client_selects_exact_recording(self, request_mock) -> None:
        request_mock.return_value = {"recordings": [{"id": "123"}]}
        client = XenoCantoClient("secret", maximum_retries=0)
        self.assertEqual(client.fetch_recording("123")["id"], "123")

    @patch.object(XenoCantoClient, "_request")
    def test_client_accepts_exact_v3_count_and_preserves_required_fields(
        self,
        request_mock,
    ) -> None:
        request_mock.return_value = {
            "numRecordings": "1",
            "nr": "1",
            "recordings": [
                {
                    "nr": "123",
                    "lon": "85.25",
                    "file-name": "XC123.mp3",
                    "length": "42.5",
                    "sex": "male",
                    "stage": "adult",
                    "unapproved": "discarded",
                }
            ],
        }
        recording = XenoCantoClient("secret", maximum_retries=0).fetch_recording("123")
        self.assertEqual(recording["lon"], "85.25")
        self.assertEqual(recording["file-name"], "XC123.mp3")
        self.assertEqual(recording["length"], "42.5")
        self.assertEqual(recording["sex"], "male")
        self.assertEqual(recording["stage"], "adult")
        self.assertNotIn("unapproved", recording)

    @patch.object(XenoCantoClient, "_request")
    def test_client_stops_immediately_on_nonretryable_http_statuses(
        self,
        request_mock,
    ) -> None:
        with patch("bird_audio.metadata.time.sleep") as sleep_mock:
            for status in (400, 401, 403, 404):
                with self.subTest(status=status):
                    request_mock.reset_mock()
                    request_mock.side_effect = http_error(status)
                    client = XenoCantoClient("secret", maximum_retries=3)
                    with self.assertRaises(XenoCantoApiError) as context:
                        client.fetch_recording("123")
                    self.assertEqual(request_mock.call_count, 1)
                    self.assertIn(f"status={status}", str(context.exception))
                    self.assertNotIn("must-not-leak", str(context.exception))
            sleep_mock.assert_not_called()

    @patch.object(XenoCantoClient, "_request")
    def test_client_retries_429_and_bounds_retry_after(self, request_mock) -> None:
        request_mock.side_effect = [
            http_error(429, retry_after="120"),
            {"numRecordings": "1", "recordings": [{"id": "123"}]},
        ]
        with patch("bird_audio.metadata.time.sleep") as sleep_mock:
            recording = XenoCantoClient("secret", maximum_retries=1).fetch_recording("123")
        self.assertEqual(recording["id"], "123")
        sleep_mock.assert_called_once_with(60.0)

    @patch.object(XenoCantoClient, "_request")
    def test_client_retries_server_and_transport_errors(self, request_mock) -> None:
        for transient in (http_error(503), urllib.error.URLError(TimeoutError())):
            with self.subTest(transient=type(transient).__name__):
                request_mock.reset_mock()
                request_mock.side_effect = [
                    transient,
                    {"recordings": [{"id": "123"}]},
                ]
                with patch("bird_audio.metadata.time.sleep") as sleep_mock:
                    recording = XenoCantoClient(
                        "secret",
                        maximum_retries=1,
                    ).fetch_recording("123")
                self.assertEqual(recording["id"], "123")
                sleep_mock.assert_called_once_with(1)

    @patch.object(XenoCantoClient, "_request")
    def test_client_classifies_missing_record_as_terminal_unavailable(
        self,
        request_mock,
    ) -> None:
        for payload_or_error in (
            {"numRecordings": "0", "recordings": []},
            http_error(404),
        ):
            with self.subTest(case=type(payload_or_error).__name__):
                request_mock.reset_mock()
                request_mock.side_effect = (
                    payload_or_error if isinstance(payload_or_error, BaseException) else None
                )
                request_mock.return_value = (
                    payload_or_error if isinstance(payload_or_error, dict) else None
                )
                with self.assertRaises(XenoCantoRecordUnavailableError):
                    XenoCantoClient("secret", maximum_retries=2).fetch_recording("123")
                self.assertEqual(request_mock.call_count, 1)

    @patch.object(XenoCantoClient, "_request")
    def test_client_rejects_top_level_errors_and_non_single_results(
        self,
        request_mock,
    ) -> None:
        invalid_payloads = (
            {"error": "invalid query", "recordings": []},
            {"numRecordings": "2", "recordings": [{"id": "123"}]},
            {"nr": "0", "recordings": []},
            {"recordings": [{"id": "123"}, {"id": "999"}]},
            {"recordings": [{"id": "999"}]},
        )
        with patch("bird_audio.metadata.time.sleep") as sleep_mock:
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    request_mock.reset_mock()
                    request_mock.return_value = payload
                    with self.assertRaises(XenoCantoApiError):
                        XenoCantoClient("secret", maximum_retries=2).fetch_recording("123")
                    self.assertEqual(request_mock.call_count, 1)
            sleep_mock.assert_not_called()

    @patch.object(XenoCantoClient, "_request")
    def test_client_marks_top_level_api_error_as_fatal(self, request_mock) -> None:
        request_mock.return_value = {"error": "authentication failed", "recordings": []}
        with self.assertRaises(XenoCantoFatalApiError):
            XenoCantoClient("secret", maximum_retries=3).fetch_recording("123")
        self.assertEqual(request_mock.call_count, 1)

    @patch.object(XenoCantoClient, "_request")
    def test_client_error_never_exposes_key(self, request_mock) -> None:
        request_mock.side_effect = urllib.error.URLError("secret-key-in-url")
        client = XenoCantoClient("secret-key-in-url", maximum_retries=0)
        with self.assertRaises(XenoCantoApiError) as context:
            client.fetch_recording("123")
        self.assertNotIn("secret-key-in-url", str(context.exception))

    def test_valid_metadata_promotes_pending_row_to_include(self) -> None:
        row = base_row()
        targets = {"asian koel", "eudynamys scolopaceus", "common myna"}
        _apply_metadata(row, cache_entry(), targets)
        assign_session_groups([row])
        self.assertEqual(row["metadata_status"], "ok")
        self.assertEqual(row["local_qc_status"], "include")
        self.assertEqual(row["identity_validation_status"], "exact_match")
        self.assertEqual(row["licence_validation_status"], "recognized_cc")
        self.assertEqual(
            row["attribution"],
            "Recorder, XC123. Accessible at www.xeno-canto.org/123.",
        )
        self.assertTrue(row["session_group"].startswith("session:"))

    def test_transient_metadata_failure_blocks_splitting_without_exclusion(self) -> None:
        row = base_row()
        _apply_metadata(row, {"status": "error", "error": "network"}, {"asian koel"})
        self.assertEqual(row["metadata_status"], "error")
        self.assertEqual(row["local_qc_status"], "manual_review")
        self.assertIn("metadata_fetch_failed", row["exclusion_reasons"])

    def test_terminal_unavailable_metadata_is_conservatively_excluded(self) -> None:
        row = base_row()
        _apply_metadata(
            row,
            {
                "status": "unavailable",
                "error": "XC123: no matching recording",
                "recording": {},
            },
            {"asian koel"},
        )
        self.assertEqual(row["metadata_status"], "unavailable")
        self.assertEqual(row["local_qc_status"], "exclude")
        self.assertIn("metadata_record_unavailable", row["exclusion_reasons"])

    def test_target_secondary_label_is_excluded(self) -> None:
        row = base_row()
        targets = {"asian koel", "eudynamys scolopaceus", "common myna"}
        _apply_metadata(row, cache_entry(["Common Myna"]), targets)
        self.assertEqual(row["local_qc_status"], "exclude")
        self.assertIn("target_species_in_secondary_labels", row["exclusion_reasons"])

    def test_configured_target_labels_include_unknown_and_fallback_species(self) -> None:
        labels = _configured_target_labels(
            {
                "known_species": [
                    {
                        "common_name": "Asian Koel",
                        "scientific_name": "Eudynamys scolopaceus",
                    }
                ],
                "unknown_species": [
                    {"common_name": "House Crow", "scientific_name": "Corvus splendens"}
                ],
                "fallback_unknown_species": [
                    {
                        "common_name": "Oriental Turtle Dove",
                        "scientific_name": "Streptopelia orientalis",
                    }
                ],
            }
        )
        self.assertIn("asian koel", labels)
        self.assertIn("corvus splendens", labels)
        self.assertIn("oriental turtle dove", labels)

    def test_composite_self_secondary_identity_is_not_excluded(self) -> None:
        row = base_row()
        targets = {"asian koel", "eudynamys scolopaceus", "common myna"}
        _apply_metadata(
            row,
            cache_entry(["Asian Koel (Eudynamys scolopaceus)"]),
            targets,
        )
        self.assertEqual(row["local_qc_status"], "include")
        self.assertEqual(row["target_secondary_labels"], "[]")

    def test_composite_secondary_with_different_target_is_excluded(self) -> None:
        row = base_row()
        targets = {"asian koel", "eudynamys scolopaceus", "common myna"}
        _apply_metadata(
            row,
            cache_entry(["Asian Koel (Eudynamys scolopaceus), Common Myna"]),
            targets,
        )
        self.assertEqual(row["local_qc_status"], "exclude")

    def test_missing_licence_requires_review(self) -> None:
        row = base_row()
        _apply_metadata(row, cache_entry(licence=""), {"asian koel"})
        self.assertEqual(row["local_qc_status"], "manual_review")
        self.assertEqual(row["licence_validation_status"], "missing")
        self.assertIn("licence_missing_manual_review", row["exclusion_reasons"])

    def test_unrecognized_licence_uri_requires_review(self) -> None:
        row = base_row()
        _apply_metadata(row, cache_entry(licence="https://example.com/by/4.0"), {"asian koel"})
        self.assertEqual(row["local_qc_status"], "manual_review")
        self.assertEqual(row["licence_validation_status"], "unrecognized_uri")
        self.assertIn("licence_unrecognized_manual_review", row["exclusion_reasons"])

    def test_recognized_cc_licence_uri_validation_is_host_and_path_exact(self) -> None:
        self.assertTrue(
            _is_recognized_cc_licence_uri("//creativecommons.org/licenses/by-nc-sa/4.0/")
        )
        self.assertTrue(
            _is_recognized_cc_licence_uri("https://creativecommons.org/publicdomain/zero/1.0/")
        )
        self.assertFalse(
            _is_recognized_cc_licence_uri(
                "https://creativecommons.org.example.com/licenses/by/4.0/"
            )
        )
        self.assertFalse(
            _is_recognized_cc_licence_uri("https://creativecommons.org/licenses/not-a-license/4.0/")
        )
        self.assertFalse(
            _is_recognized_cc_licence_uri("https://creativecommons.org/licenses/by/99.0/")
        )
        self.assertFalse(
            _is_recognized_cc_licence_uri(
                "https://creativecommons.org/licenses/by/4.0/not-a-real-license/"
            )
        )

    def test_binomial_identity_is_exact_and_subspecies_tolerant(self) -> None:
        self.assertTrue(
            _same_binomial_identity(
                "Eudynamys scolopaceus",
                "Eudynamys scolopaceus chinensis",
            )
        )
        self.assertFalse(
            _same_binomial_identity(
                "Eudynamys scolopaceus",
                "Eudynamys scolopaceusx",
            )
        )
        self.assertFalse(_same_binomial_identity("Eudynamys", "Eudynamys scolopaceus"))

    def test_secondary_matching_is_exact_not_substring_based(self) -> None:
        self.assertFalse(_labels_overlap("Common Cuckooshrike", {"common cuckoo"}))
        self.assertTrue(
            _labels_overlap(
                "Common Myna (Acridotheres tristis)",
                {"acridotheres tristis"},
            )
        )

    def test_v3_lon_is_preferred_with_legacy_lng_fallback(self) -> None:
        row = base_row()
        entry = cache_entry()
        entry["recording"]["lon"] = "84.100"
        _apply_metadata(row, entry, {"asian koel"})
        self.assertEqual(row["longitude"], "84.100")

        fallback_row = base_row()
        fallback_entry = cache_entry()
        _apply_metadata(fallback_row, fallback_entry, {"asian koel"})
        self.assertEqual(fallback_row["longitude"], "85.300")

    def test_same_individual_reference_links_across_metadata_buckets(self) -> None:
        left = base_row()
        right = {**base_row(), "recording_id": "XC124", "xc_id": "124"}
        _apply_metadata(left, cache_entry(), {"asian koel"})
        right_entry = cache_entry()
        right_entry["recording"]["id"] = "124"
        right_entry["recording"]["rec"] = "Different recorder"
        right_entry["recording"]["date"] = "2023-05-01"
        _apply_metadata(right, right_entry, {"asian koel"})
        left["remarks"] = "same individual as XC124"
        assign_session_groups([left, right])
        self.assertEqual(left["session_group"], right["session_group"])
        self.assertEqual(left["session_review_flag"], "false")

    def test_unresolved_same_individual_and_missing_date_require_review(self) -> None:
        row = base_row()
        entry = cache_entry()
        entry["recording"]["date"] = ""
        entry["recording"]["rmk"] = "same individual as XC999"
        _apply_metadata(row, entry, {"asian koel"})
        assign_session_groups([row])
        self.assertEqual(row["session_review_flag"], "true")
        self.assertEqual(row["local_qc_status"], "manual_review")
        self.assertIn("session_date_missing", row["session_review_reason"])
        self.assertIn("session_same_individual_unresolved", row["session_review_reason"])

    def test_invalid_date_and_coordinates_require_session_review(self) -> None:
        row = base_row()
        entry = cache_entry()
        entry["recording"]["date"] = "2024-02-31"
        entry["recording"]["lat"] = "91.0"
        _apply_metadata(row, entry, {"asian koel"})
        assign_session_groups([row])
        self.assertEqual(row["local_qc_status"], "manual_review")
        self.assertIn("session_date_invalid", row["session_review_reason"])
        self.assertIn("session_coordinates_invalid", row["session_review_reason"])

    def test_cross_species_same_individual_reference_is_not_auto_linked(self) -> None:
        left = base_row()
        right = {
            **base_row(),
            "recording_id": "XC124",
            "xc_id": "124",
            "species_common_name": "Common Myna",
            "scientific_name": "Acridotheres tristis",
        }
        left_entry = cache_entry()
        left_entry["recording"]["rmk"] = "same individual as XC124"
        right_entry = cache_entry()
        right_entry["recording"].update(
            {
                "id": "124",
                "gen": "Acridotheres",
                "sp": "tristis",
                "en": "Common Myna",
                "rec": "Different recorder",
                "date": "2023-05-01",
                "loc": "Different location",
                "lat": "26.0",
                "lon": "84.0",
            }
        )
        _apply_metadata(left, left_entry, {"asian koel", "common myna"})
        _apply_metadata(right, right_entry, {"asian koel", "common myna"})
        assign_session_groups([left, right])
        self.assertNotEqual(left["session_group"], right["session_group"])
        self.assertEqual(left["local_qc_status"], "manual_review")
        self.assertIn(
            "metadata_cross_species_same_individual_manual_review",
            left["exclusion_reasons"],
        )

    def test_missing_or_non_bird_api_group_requires_review(self) -> None:
        for group in ("", "grasshoppers"):
            with self.subTest(group=group):
                row = base_row()
                entry = cache_entry()
                entry["recording"]["grp"] = group
                _apply_metadata(row, entry, {"asian koel"})
                self.assertEqual(row["local_qc_status"], "manual_review")
                self.assertEqual(
                    row["identity_validation_status"],
                    "group_mismatch_or_missing",
                )
                self.assertIn(
                    "metadata_group_mismatch_manual_review",
                    row["exclusion_reasons"],
                )

    def test_one_sided_missing_location_joins_the_recordist_date_bucket(self) -> None:
        left = base_row()
        right = {**base_row(), "recording_id": "XC124", "xc_id": "124"}
        _apply_metadata(left, cache_entry(), {"asian koel"})
        right_entry = cache_entry()
        right_entry["recording"]["id"] = "124"
        right_entry["recording"]["loc"] = ""
        right_entry["recording"]["lat"] = ""
        right_entry["recording"]["lng"] = ""
        _apply_metadata(right, right_entry, {"asian koel"})
        assign_session_groups([left, right])
        self.assertEqual(left["session_group"], right["session_group"])

    def test_non_latin_recordist_and_locality_are_preserved_for_sessions(self) -> None:
        left = base_row()
        right = {**base_row(), "recording_id": "XC124", "xc_id": "124"}
        left_entry = cache_entry()
        left_entry["recording"].update(
            {
                "rec": "小菜鸟",
                "loc": "浙江省海盐县北团村",
                "lat": "30.4555",
                "lng": "120.9103",
            }
        )
        right_entry = cache_entry()
        right_entry["recording"].update(
            {
                "id": "124",
                "rec": "小菜鸟",
                "loc": "浙江省海盐县北团村",
                "lat": "30.4742",
                "lng": "120.9006",
            }
        )
        _apply_metadata(left, left_entry, {"asian koel"})
        _apply_metadata(right, right_entry, {"asian koel"})
        assign_session_groups([left, right])
        self.assertEqual(left["session_group"], right["session_group"])
        self.assertEqual(left["session_review_flag"], "false")
        self.assertEqual(right["session_review_flag"], "false")

    def test_distinct_non_latin_recordists_remain_separate(self) -> None:
        left = base_row()
        right = {**base_row(), "recording_id": "XC124", "xc_id": "124"}
        left_entry = cache_entry()
        left_entry["recording"].update({"rec": "小菜鸟", "loc": "同一地点"})
        right_entry = cache_entry()
        right_entry["recording"].update({"id": "124", "rec": "另一位录音者", "loc": "同一地点"})
        _apply_metadata(left, left_entry, {"asian koel"})
        _apply_metadata(right, right_entry, {"asian koel"})
        assign_session_groups([left, right])
        self.assertNotEqual(left["session_group"], right["session_group"])
        self.assertEqual(left["session_review_flag"], "false")
        self.assertEqual(right["session_review_flag"], "false")

    def test_cache_rejects_api_and_recording_identity_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.json"
            payload = {
                "schema_version": "1.1",
                "api_version": API_VERSION,
                "endpoint": DEFAULT_ENDPOINT,
                "query_form": "nr:<xc_id>",
                "records": {
                    "123": {
                        "status": "ok",
                        "recording": {"id": "999"},
                    }
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_cache(path)

            payload["records"]["123"]["recording"]["id"] = "123"
            payload["api_version"] = "wrong"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_cache(path)

    def test_cache_snapshot_hashes_the_exact_parsed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.json"
            payload = {
                "schema_version": "1.1",
                "api_version": API_VERSION,
                "endpoint": DEFAULT_ENDPOINT,
                "query_form": "nr:<xc_id>",
                "records": {},
            }
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            path.write_bytes(raw)
            loaded, digest = load_metadata_cache_snapshot(path)
            self.assertEqual(loaded, payload)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    @patch.object(XenoCantoClient, "_request")
    def test_fetch_refuses_to_reopen_a_sealed_cache(self, request_mock) -> None:
        data_root = PROJECT_ROOT / "data"
        with tempfile.TemporaryDirectory(dir=data_root) as temporary:
            directory = Path(temporary)
            manifest = directory / "local.csv"
            cache = directory / "sealed.json"
            manifest.write_text("xc_id\n123\n", encoding="utf-8")
            cache.write_text(
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "api_version": API_VERSION,
                        "endpoint": DEFAULT_ENDPOINT,
                        "query_form": "nr:<xc_id>",
                        "sealed": True,
                        "records": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sealed metadata cache"):
                fetch_metadata_cache(manifest, cache, api_key="secret")
        request_mock.assert_not_called()

    @patch.object(XenoCantoClient, "_request")
    def test_nonempty_cache_cannot_rebind_to_changed_exact_manifest(
        self,
        request_mock,
    ) -> None:
        data_root = PROJECT_ROOT / "data"
        with tempfile.TemporaryDirectory(dir=data_root) as temporary:
            directory = Path(temporary)
            manifest = directory / "local.csv"
            cache = directory / "cache.json"
            manifest.write_text("xc_id\n123\n", encoding="utf-8")
            cache.write_text(
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "api_version": API_VERSION,
                        "endpoint": DEFAULT_ENDPOINT,
                        "query_form": "nr:<xc_id>",
                        "source_manifest_sha256": "0" * 64,
                        "source_recording_ids_sha256": sha256_json(["123"]),
                        "records": {
                            "123": {
                                "status": "ok",
                                "recording": {"id": "123"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different exact local manifest"):
                fetch_metadata_cache(manifest, cache, api_key="secret", maximum_retries=0)
        request_mock.assert_not_called()

    @patch.object(XenoCantoClient, "_request")
    def test_cache_fetch_aborts_after_first_fatal_api_response(
        self,
        request_mock,
    ) -> None:
        request_mock.side_effect = http_error(401)
        data_root = PROJECT_ROOT / "data"
        with tempfile.TemporaryDirectory(dir=data_root) as temporary:
            directory = Path(temporary)
            manifest = directory / "local.csv"
            cache = directory / "cache.json"
            manifest.write_text("xc_id\n123\n124\n", encoding="utf-8")
            with self.assertRaises(XenoCantoFatalApiError):
                fetch_metadata_cache(
                    manifest,
                    cache,
                    api_key="secret",
                    request_interval_seconds=1.0,
                    maximum_retries=3,
                )
        self.assertEqual(request_mock.call_count, 1)

    @patch.object(XenoCantoClient, "_request")
    def test_terminal_unavailable_cache_entry_is_complete_and_not_retried(
        self,
        request_mock,
    ) -> None:
        request_mock.side_effect = http_error(404)
        data_root = PROJECT_ROOT / "data"
        with tempfile.TemporaryDirectory(dir=data_root) as temporary:
            directory = Path(temporary)
            manifest = directory / "local.csv"
            cache = directory / "cache.json"
            manifest.write_text("xc_id\n123\n", encoding="utf-8")
            _, first = fetch_metadata_cache(
                manifest,
                cache,
                api_key="secret",
                maximum_retries=0,
            )
            self.assertEqual(first["records"]["123"]["status"], "unavailable")
            self.assertEqual(request_mock.call_count, 1)

            request_mock.reset_mock()
            _, second = fetch_metadata_cache(
                manifest,
                cache,
                api_key="secret",
                maximum_retries=0,
            )
            self.assertEqual(second["records"]["123"]["status"], "unavailable")
            request_mock.assert_not_called()

    @patch.object(XenoCantoClient, "_request")
    def test_client_refuses_to_persist_a_reflected_secret(self, request_mock) -> None:
        request_mock.return_value = {"recordings": [{"id": "123", "rmk": "secret-value"}]}
        client = XenoCantoClient("secret-value", maximum_retries=0)
        with self.assertRaises(XenoCantoApiError) as context:
            client.fetch_recording("123")
        self.assertNotIn("secret-value", str(context.exception))


if __name__ == "__main__":
    unittest.main()
