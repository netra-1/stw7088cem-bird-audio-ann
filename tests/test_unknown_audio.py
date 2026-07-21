from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from bird_audio.audio import AudioProbe, AudioToolError, FullDecodeResult
from bird_audio.io_utils import read_csv
from bird_audio.paths import PROJECT_ROOT
from bird_audio.unknown_audio import (
    UnknownAudioError,
    UnknownAudioRetryableError,
    UnknownAudioTerminalUnavailableError,
    _canonical_json_bytes,
    _locked_output_path,
    _pending_qc_path,
    _recover_orphan_audit,
    _require_no_pending_or_staging_artifacts,
    _run_unknown_audio_acquisition,
    _species_audit,
    _staging_audio_path,
    audit_unknown_audio_file,
    build_unknown_audio_preflight,
    evaluate_fallback_gate,
    load_unknown_audio_config,
    run_unknown_audio_acquisition,
    select_final_unknown_recordings,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "unknown_audio.toml"
PLAN_PATH = PROJECT_ROOT / "data" / "unknown" / "planning" / "unknown_candidate_plan_v1.json"
KNOWN_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "recordings.csv"


def _real_inputs() -> tuple[dict, list[dict[str, str]], dict]:
    config = load_unknown_audio_config(CONFIG_PATH)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    retained = [row for row in read_csv(KNOWN_MANIFEST) if row.get("local_qc_status") == "include"]
    return plan, retained, config


def _species_result(
    scientific_name: str,
    eligible: int,
    *,
    role: str = "primary",
    state: str = "inventory_exhausted",
    unresolved: int = 0,
) -> dict:
    return {
        "role": role,
        "scientific_name": scientific_name,
        "eligible_recordings": eligible,
        "unresolved_retryable": unresolved,
        "completion_state": state,
    }


class UnknownAudioPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan, cls.retained, cls.config = _real_inputs()

    def test_real_preflight_is_read_only_and_reports_locked_capacity(self) -> None:
        before = {
            path: path.exists()
            for path in (
                PROJECT_ROOT / self.config["outputs"]["working_directory"],
                PROJECT_ROOT / self.config["outputs"]["audit"],
                PROJECT_ROOT / self.config["outputs"]["audit_lock"],
            )
        }
        result = build_unknown_audio_preflight(
            self.plan,
            self.retained,
            self.config,
            plan_sha256="a" * 64,
            plan_lock_sha256="b" * 64,
            known_manifest_sha256="c" * 64,
            config_sha256="d" * 64,
            available_disk_bytes=10**12,
        )
        self.assertEqual(result["network_requests"], 0)
        self.assertEqual(result["audio_downloads"], 0)
        self.assertFalse(result["fallback_active"])
        self.assertEqual(len(result["candidates"]), 905)
        self.assertEqual(result["plan_lock_sha256"], "b" * 64)
        counts = {
            row["scientific_name"]: (
                row["canonical_sessions_before_audio_qc"],
                row["canonical_session_margin_for_final_target"],
            )
            for row in result["species"]
        }
        self.assertEqual(counts["Psilopogon zeylanicus"], (42, 2))
        self.assertEqual(counts["Acridotheres fuscus"], (13, -27))
        self.assertEqual(counts["Ceryle rudis"], (100, 60))
        self.assertEqual(counts["Corvus splendens"], (120, 80))
        self.assertEqual(counts["Ortygornis pondicerianus"], (99, 59))
        self.assertEqual(counts["Streptopelia orientalis"], (66, 26))
        after = {path: path.exists() for path in before}
        self.assertEqual(after, before)

    def test_api_rate_and_duration_are_advisory_only(self) -> None:
        reference = build_unknown_audio_preflight(
            self.plan, self.retained, self.config, available_disk_bytes=10**12
        )
        chosen = next(
            row
            for row in reference["candidates"]
            if row["disposition"] == "canonical_pending_audio_qc"
        )
        changed = copy.deepcopy(self.plan)
        raw = next(
            candidate
            for queue in changed["candidate_queues"]
            for candidate in queue["candidates"]
            if candidate["candidate_id"] == chosen["candidate_id"]
        )
        raw["metadata"]["smp"] = "8000"
        raw["metadata"]["length"] = "not-known"
        result = build_unknown_audio_preflight(
            changed, self.retained, self.config, available_disk_bytes=10**12
        )
        row = next(
            item for item in result["candidates"] if item["candidate_id"] == chosen["candidate_id"]
        )
        self.assertEqual(row["disposition"], "canonical_pending_audio_qc")
        self.assertEqual(row["declared_sample_rate_hz"], 8000)
        self.assertIsNone(row["estimated_duration_seconds"])

    def test_forbidden_outcome_field_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        changed["candidate_queues"][0]["candidates"][0]["metadata"]["model_score"] = 0.9
        with self.assertRaisesRegex(UnknownAudioError, "forbidden outcome fields"):
            build_unknown_audio_preflight(
                changed, self.retained, self.config, available_disk_bytes=10**12
            )

    def test_config_rejects_download_policy_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["download"]["maximum_retries"] = 4
        with self.assertRaisesRegex(UnknownAudioError, "download policy"):
            build_unknown_audio_preflight(
                self.plan, self.retained, changed, available_disk_bytes=10**12
            )

    def test_config_path_outside_project_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary) / "unknown_audio.toml"
            outside.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(UnknownAudioError, "inside the project"):
                load_unknown_audio_config(outside)

    def test_every_noncanonical_points_to_earlier_fixed_canonical(self) -> None:
        result = build_unknown_audio_preflight(
            self.plan, self.retained, self.config, available_disk_bytes=10**12
        )
        by_id = {row["candidate_id"]: row for row in result["candidates"]}
        noncanonical = [
            row for row in result["candidates"] if row["disposition"] == "session_noncanonical"
        ]
        self.assertTrue(noncanonical)
        for row in noncanonical:
            canonical = by_id[row["canonical_candidate_id"]]
            self.assertEqual(canonical["disposition"], "canonical_pending_audio_qc")
            self.assertEqual(canonical["session_group"], row["session_group"])
            self.assertLess(canonical["queue_rank"], row["queue_rank"])


class UnknownAudioFallbackTests(unittest.TestCase):
    def _passing_primaries(self) -> list[dict]:
        return [
            _species_result("Psilopogon zeylanicus", 42),
            _species_result("Acridotheres fuscus", 13),
            _species_result("Ceryle rudis", 80, state="pool_satisfied"),
            _species_result("Corvus splendens", 80, state="pool_satisfied"),
            _species_result("Ortygornis pondicerianus", 80, state="pool_satisfied"),
        ]

    def test_jungle_must_be_terminally_exhausted_before_fallback(self) -> None:
        rows = self._passing_primaries()
        rows[1]["completion_state"] = "blocked_retryable"
        rows[1]["unresolved_retryable"] = 1
        result = evaluate_fallback_gate(rows)
        self.assertFalse(result["fallback_active"])
        self.assertEqual(result["status"], "blocked_retryable_or_incomplete_primary_audit")

    def test_exactly_one_failed_primary_activates_fallback(self) -> None:
        first = evaluate_fallback_gate(self._passing_primaries())
        self.assertEqual(first["status"], "fallback_audit_required")
        self.assertTrue(first["fallback_active"])
        rows = [
            *self._passing_primaries(),
            _species_result("Streptopelia orientalis", 66, role="fallback"),
        ]
        final = evaluate_fallback_gate(rows)
        self.assertEqual(final["status"], "ready_with_fallback")
        self.assertEqual(
            final["replacement"],
            {
                "replaced_scientific_name": "Acridotheres fuscus",
                "replacement_scientific_name": "Streptopelia orientalis",
            },
        )

    def test_two_failed_primaries_require_protocol_decision(self) -> None:
        rows = self._passing_primaries()
        rows[0]["eligible_recordings"] = 39
        result = evaluate_fallback_gate(rows)
        self.assertEqual(result["status"], "protocol_decision_required")
        self.assertEqual(result["reason"], "more_than_one_primary_below_40")

    def test_fallback_below_40_requires_protocol_decision(self) -> None:
        rows = [
            *self._passing_primaries(),
            _species_result("Streptopelia orientalis", 39, role="fallback"),
        ]
        result = evaluate_fallback_gate(rows)
        self.assertEqual(result["status"], "protocol_decision_required")
        self.assertEqual(result["reason"], "fallback_below_40")

    def test_boolean_retryable_count_is_rejected(self) -> None:
        rows = self._passing_primaries()
        rows[0]["unresolved_retryable"] = True
        with self.assertRaisesRegex(UnknownAudioError, "retryable count"):
            evaluate_fallback_gate(rows)


class UnknownAudioQCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_unknown_audio_config(CONFIG_PATH)

    def setUp(self) -> None:
        root = PROJECT_ROOT / "data" / "unknown" / "interim"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="unknown-qc-test-", dir=root)
        self.path = Path(self.temporary.name) / "XC1.audio"
        self.path.write_bytes(b"test audio bytes")
        self.candidate = {
            "candidate_id": "XC1",
            "scientific_name": "Testus birdus",
            "session_group": "session:test",
            "quality": "A",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def probe(rate: int = 48000, channels: int = 1, duration: float = 10.0) -> AudioProbe:
        return AudioProbe(
            probe_ok=True,
            format_name="mp3",
            codec_name="mp3",
            source_sample_rate_hz=rate,
            channels=channels,
            ffprobe_duration_seconds=duration,
        )

    def audit(self, probe: AudioProbe, decode: FullDecodeResult) -> dict:
        return audit_unknown_audio_file(
            self.path,
            self.candidate,
            self.config,
            ffprobe=Path("ffprobe"),
            ffmpeg=Path("ffmpeg"),
            detect_header_fn=lambda _path: "mp3_id3",
            probe_fn=lambda _path, _tool: probe,
            full_decode_fn=lambda _path, _tool: decode,
        )

    def test_qc_accepts_inclusive_duration_ratio_boundaries(self) -> None:
        for duration in (9.8, 10.2):
            with self.subTest(duration=duration):
                result = self.audit(self.probe(), FullDecodeResult(duration, ""))
                self.assertEqual(result["disposition"], "eligible")
                self.assertEqual(result["header_detection_status"], "recognized")
                self.assertEqual(result["probe_status"], "ok")
                self.assertEqual(result["full_decode_status"], "ok")
                self.assertIn("assignment_descriptor", result)

    def test_qc_excludes_low_rate_and_full_decode_warning(self) -> None:
        low_rate = self.audit(self.probe(rate=16000), FullDecodeResult(10.0, ""))
        self.assertEqual(low_rate["disposition"], "audio_qc_excluded")
        self.assertEqual(low_rate["full_decode_status"], "not_run")
        self.assertIn("source_sample_rate_below_32000_hz", low_rate["reasons"])
        warning = self.audit(self.probe(), FullDecodeResult(10.0, "corrupt frame"))
        self.assertEqual(warning["disposition"], "audio_qc_excluded")
        self.assertEqual(warning["full_decode_status"], "warning")
        self.assertIn("full_decode_warning", warning["reasons"])

    def test_qc_normalizes_nonfinite_decoded_duration(self) -> None:
        result = self.audit(self.probe(), FullDecodeResult(float("nan"), ""))
        self.assertEqual(result["disposition"], "audio_qc_excluded")
        self.assertEqual(result["decoded_duration_seconds"], 0.0)
        self.assertEqual(result["decoded_duration_ratio"], 0.0)
        self.assertEqual(result["full_decode_status"], "invalid_duration")
        self.assertIn("non_positive_decoded_duration", result["reasons"])

    def test_content_probe_failure_has_fixed_normalized_terminal_state(self) -> None:
        result = self.audit(
            AudioProbe(probe_ok=False, probe_error="No audio stream found"),
            FullDecodeResult(10.0, ""),
        )
        self.assertEqual(result["probe_status"], "content_failure")
        self.assertEqual(result["full_decode_status"], "not_run")
        self.assertEqual(result["format_name"], "")
        self.assertEqual(result["source_sample_rate_hz"], 0)
        self.assertEqual(result["reasons"], ["ffprobe_content_failure"])

    def test_tool_invocation_and_unknown_injected_failures_are_retryable(self) -> None:
        cases = (
            {
                "detect_header_fn": lambda _path: (_ for _ in ()).throw(OSError("read")),
                "probe_fn": lambda _path, _tool: self.probe(),
                "full_decode_fn": lambda _path, _tool: FullDecodeResult(10.0, ""),
            },
            {
                "detect_header_fn": lambda _path: "mp3_id3",
                "probe_fn": lambda _path, _tool: AudioProbe(
                    probe_ok=False, probe_error="ffprobe invocation timed out"
                ),
                "full_decode_fn": lambda _path, _tool: FullDecodeResult(10.0, ""),
            },
            {
                "detect_header_fn": lambda _path: "mp3_id3",
                "probe_fn": lambda _path, _tool: self.probe(),
                "full_decode_fn": lambda _path, _tool: (_ for _ in ()).throw(
                    AudioToolError("Full decode invocation failed")
                ),
            },
            {
                "detect_header_fn": lambda _path: "mp3_id3",
                "probe_fn": lambda _path, _tool: (_ for _ in ()).throw(
                    RuntimeError("unknown injected failure")
                ),
                "full_decode_fn": lambda _path, _tool: FullDecodeResult(10.0, ""),
            },
        )
        for functions in cases:
            with (
                self.subTest(functions=functions),
                self.assertRaises(UnknownAudioRetryableError),
            ):
                audit_unknown_audio_file(
                    self.path,
                    self.candidate,
                    self.config,
                    ffprobe=Path("ffprobe"),
                    ffmpeg=Path("ffmpeg"),
                    **functions,
                )


class UnknownAudioCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_unknown_audio_config(CONFIG_PATH)

    def setUp(self) -> None:
        interim = PROJECT_ROOT / "data" / "unknown" / "interim"
        raw = PROJECT_ROOT / "data" / "unknown" / "raw" / "audio_v1"
        interim.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        self.working_temp = tempfile.TemporaryDirectory(prefix="audit-test-", dir=interim)
        self.working = Path(self.working_temp.name)
        self.raw = raw
        self.species_directory = raw / "Testus_unittestus"
        shutil.rmtree(self.species_directory, ignore_errors=True)
        self.candidate = {
            "candidate_id": "XC999999991",
            "queue_rank": 1,
            "role": "primary",
            "scientific_name": "Testus unittestus",
            "session_group": "session:test",
            "download_url": "https://xeno-canto.org/999999991/download",
            "quality": "A",
            "disposition": "canonical_pending_audio_qc",
            "reasons": [],
        }
        self.preflight = {
            "preflight_sha256": "a" * 64,
            "plan_sha256": "b" * 64,
        }

    def tearDown(self) -> None:
        self.working_temp.cleanup()
        shutil.rmtree(self.species_directory, ignore_errors=True)

    @staticmethod
    def _publish_fake_audio(destination: Path, value: bytes = b"downloaded bytes") -> None:
        destination.write_bytes(value)
        destination.chmod(0o600)

    def _run(
        self,
        client: object,
        progress_callback=None,
        *,
        probe_fn=None,
        full_decode_fn=None,
    ) -> dict:
        return _species_audit(
            [self.candidate],
            self.preflight,
            self.config,
            client,
            self.working,
            self.raw,
            ffprobe=Path("ffprobe"),
            ffmpeg=Path("ffmpeg"),
            known_hashes=set(),
            observed_unknown_hashes={},
            detect_header_fn=lambda _path: "mp3_id3",
            probe_fn=probe_fn or (lambda _path, _tool: UnknownAudioQCTests.probe()),
            full_decode_fn=full_decode_fn or (lambda _path, _tool: FullDecodeResult(10.0, "")),
            progress_callback=progress_callback,
        )

    def test_terminal_checkpoint_is_reused_without_second_download(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {
                    "candidate_id": candidate_id,
                    "source_url": source_url,
                    "bytes_written": destination.stat().st_size,
                }

        client = Client()
        first = self._run(client)
        second = self._run(client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(first["eligible_recordings"], 1)
        self.assertEqual(second["eligible_recordings"], 1)
        self.assertEqual(first["completion_state"], "inventory_exhausted")

    def test_retryable_failure_blocks_and_creates_no_terminal_checkpoint(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                raise UnknownAudioRetryableError("retry later")

        result = self._run(Client())
        self.assertEqual(result["completion_state"], "blocked_retryable")
        self.assertEqual(result["unresolved_retryable"], 1)
        self.assertFalse((self.working / "checkpoints" / "XC999999991.json").exists())

    def test_terminal_unavailable_prunes_empty_staging_artifacts(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                raise UnknownAudioTerminalUnavailableError("gone")

        result = self._run(Client())
        self.assertEqual(result["completion_state"], "inventory_exhausted")
        self.assertFalse((self.working / "staging").exists())
        self.assertFalse((self.working / "pending_qc").exists())
        _require_no_pending_or_staging_artifacts(self.working)

    def test_unbound_staging_file_is_never_adopted(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                raise AssertionError("network must not run")

        stage = _staging_audio_path(self.working, self.candidate)
        stage.parent.mkdir(parents=True, exist_ok=True)
        self._publish_fake_audio(stage)
        client = Client()
        with self.assertRaisesRegex(UnknownAudioError, "unbound audio"):
            self._run(client)
        self.assertEqual(client.calls, 0)
        self.assertTrue(stage.is_file())

    def test_resumed_checkpoint_detects_downloaded_audio_tampering(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        self._run(Client())
        downloaded = self.species_directory / "XC999999991.audio"
        downloaded.write_bytes(b"tampered")
        with self.assertRaisesRegex(UnknownAudioError, "audio binding"):
            self._run(Client())

    def test_progress_callback_receives_terminal_and_species_events(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        events: list[dict] = []
        self._run(Client(), events.append)
        self.assertEqual(
            [event["event"] for event in events],
            [
                "candidate_terminal",
                "species_complete",
            ],
        )
        self.assertEqual(events[0]["disposition"], "eligible")
        self.assertEqual(events[1]["completion_state"], "inventory_exhausted")

    def test_receipt_candidate_mismatch_is_rejected(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": "XC999", "source_url": source_url}

        with self.assertRaisesRegex(UnknownAudioError, "receipt candidate binding"):
            self._run(Client())

    def test_resumed_checkpoint_rejects_non_private_raw_mode(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        self._run(Client())
        downloaded = self.species_directory / "XC999999991.audio"
        downloaded.chmod(0o644)
        with self.assertRaisesRegex(UnknownAudioError, "private-file state"):
            self._run(Client())

    def test_resumed_checkpoint_rejects_hardlinked_raw_audio(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        self._run(Client())
        downloaded = self.species_directory / "XC999999991.audio"
        os.link(downloaded, self.species_directory / "hardlink-test.audio")
        with self.assertRaisesRegex(UnknownAudioError, "private-file state"):
            self._run(Client())

    def test_resumed_checkpoint_recomputes_assignment_descriptor(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        self._run(Client())
        checkpoint = self.working / "checkpoints" / "XC999999991.json"
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        value["audio_qc"]["assignment_descriptor"]["duration_bucket"] = "below_3"
        checkpoint.write_bytes(_canonical_json_bytes(value))
        with self.assertRaisesRegex(UnknownAudioError, "descriptor drifted"):
            self._run(Client())

    def test_resumed_checkpoint_rejects_malformed_receipt_hash(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        self._run(Client())
        checkpoint = self.working / "checkpoints" / "XC999999991.json"
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        value["download_receipt"]["receipt_sha256"] = "not-a-sha256"
        checkpoint.write_bytes(_canonical_json_bytes(value))
        with self.assertRaisesRegex(UnknownAudioError, "receipt hash"):
            self._run(Client())

    def test_resumed_checkpoint_detects_sanitized_receipt_tampering(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {
                    "candidate_id": candidate_id,
                    "source_url": source_url,
                    "attempts": 1,
                }

        self._run(Client())
        checkpoint = self.working / "checkpoints" / "XC999999991.json"
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        value["download_receipt"]["attempts"] = 2
        checkpoint.write_bytes(_canonical_json_bytes(value))
        with self.assertRaisesRegex(UnknownAudioError, "receipt hash binding"):
            self._run(Client())

    def test_retryable_qc_resumes_pending_audio_without_redownload(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        client = Client()
        blocked = self._run(
            client,
            probe_fn=lambda _path, _tool: AudioProbe(
                probe_ok=False, probe_error="ffprobe invocation timed out"
            ),
        )
        pending = _pending_qc_path(self.working, self.candidate["candidate_id"])
        raw = self.species_directory / "XC999999991.audio"
        self.assertEqual(blocked["completion_state"], "blocked_retryable")
        self.assertEqual(client.calls, 1)
        self.assertTrue(pending.is_file())
        self.assertTrue(raw.is_file())
        self.assertFalse((self.working / "checkpoints" / "XC999999991.json").exists())

        complete = self._run(client)
        self.assertEqual(complete["completion_state"], "inventory_exhausted")
        self.assertEqual(client.calls, 1)
        self.assertFalse(pending.exists())
        self.assertTrue((self.working / "checkpoints" / "XC999999991.json").is_file())

    def test_keyboard_interrupt_preserves_pending_audio_for_resume(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        client = Client()
        with self.assertRaises(KeyboardInterrupt):
            self._run(
                client,
                full_decode_fn=lambda _path, _tool: (_ for _ in ()).throw(KeyboardInterrupt()),
            )
        pending = _pending_qc_path(self.working, self.candidate["candidate_id"])
        self.assertTrue(pending.is_file())
        self.assertTrue((self.species_directory / "XC999999991.audio").is_file())
        self._run(client)
        self.assertEqual(client.calls, 1)
        self.assertFalse(pending.exists())

    def test_stage_only_and_two_link_crash_states_resume_without_download(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        for state in ("stage_only", "two_link"):
            with self.subTest(state=state):
                client = Client()
                self._run(
                    client,
                    probe_fn=lambda _path, _tool: AudioProbe(
                        probe_ok=False, probe_error="ffprobe invocation failed"
                    ),
                )
                raw = self.species_directory / "XC999999991.audio"
                stage = _staging_audio_path(self.working, self.candidate)
                stage.parent.mkdir(parents=True, exist_ok=True)
                os.link(raw, stage)
                if state == "stage_only":
                    raw.unlink()
                self._run(client)
                self.assertEqual(client.calls, 1)
                self.assertTrue(raw.is_file())
                self.assertFalse(stage.exists())
                shutil.rmtree(self.species_directory, ignore_errors=True)
                shutil.rmtree(self.working / "checkpoints", ignore_errors=True)

    def test_checkpoint_commit_with_pending_record_recovers_cleanup(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        client = Client()
        with (
            patch(
                "bird_audio.unknown_audio._remove_matching_pending_qc_record",
                side_effect=KeyboardInterrupt(),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            self._run(client)
        pending = _pending_qc_path(self.working, self.candidate["candidate_id"])
        checkpoint = self.working / "checkpoints" / "XC999999991.json"
        self.assertTrue(pending.is_file())
        self.assertTrue(checkpoint.is_file())
        self._run(client)
        self.assertEqual(client.calls, 1)
        self.assertFalse(pending.exists())

    def test_pending_inode_replacement_and_third_link_fail_closed(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        for replacement in ("different_inode", "third_link"):
            with self.subTest(replacement=replacement):
                self._run(
                    Client(),
                    probe_fn=lambda _path, _tool: AudioProbe(
                        probe_ok=False, probe_error="ffprobe invocation timed out"
                    ),
                )
                raw = self.species_directory / "XC999999991.audio"
                if replacement == "different_inode":
                    payload = raw.read_bytes()
                    raw.unlink()
                    self._publish_fake_audio(raw, payload)
                else:
                    os.link(raw, self.species_directory / "extra-one.audio")
                    os.link(raw, self.species_directory / "extra-two.audio")
                with self.assertRaises(UnknownAudioError):
                    self._run(Client())
                shutil.rmtree(self.species_directory, ignore_errors=True)
                shutil.rmtree(self.working / "pending_qc", ignore_errors=True)

    def test_coordinated_qc_forgery_and_status_drift_are_rejected(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        for mutation in ("coordinated", "status_only"):
            with self.subTest(mutation=mutation):
                self._run(Client())
                checkpoint = self.working / "checkpoints" / "XC999999991.json"
                value = json.loads(checkpoint.read_text(encoding="utf-8"))
                if mutation == "coordinated":
                    value["disposition"] = "audio_qc_excluded"
                    value["reasons"] = ["ffprobe_content_failure"]
                    value["audio_qc"]["disposition"] = "audio_qc_excluded"
                    value["audio_qc"]["reasons"] = ["ffprobe_content_failure"]
                    value["audio_qc"]["probe_status"] = "content_failure"
                    value["audio_qc"].pop("assignment_descriptor")
                else:
                    value["audio_qc"]["full_decode_status"] = "warning"
                checkpoint.write_bytes(_canonical_json_bytes(value))
                with self.assertRaisesRegex(UnknownAudioError, "QC derivation"):
                    self._run(Client())
                shutil.rmtree(self.species_directory, ignore_errors=True)
                shutil.rmtree(self.working / "checkpoints", ignore_errors=True)

    def test_checkpoint_mode_link_and_canonical_bytes_are_enforced(self) -> None:
        class Client:
            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        for mutation in ("mode", "hardlink", "whitespace"):
            with self.subTest(mutation=mutation):
                self._run(Client())
                checkpoint = self.working / "checkpoints" / "XC999999991.json"
                if mutation == "mode":
                    checkpoint.chmod(0o644)
                elif mutation == "hardlink":
                    os.link(checkpoint, self.working / "checkpoint-hardlink.json")
                else:
                    value = json.loads(checkpoint.read_text(encoding="utf-8"))
                    checkpoint.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(UnknownAudioError):
                    self._run(Client())
                (self.working / "checkpoint-hardlink.json").unlink(missing_ok=True)
                shutil.rmtree(self.species_directory, ignore_errors=True)
                shutil.rmtree(self.working / "checkpoints", ignore_errors=True)

    def test_broken_checkpoint_and_pending_symlinks_are_never_followed(self) -> None:
        class Client:
            calls = 0

            def download(self, candidate_id: str, source_url: str, destination: Path) -> dict:
                self.calls += 1
                UnknownAudioCheckpointTests._publish_fake_audio(destination)
                return {"candidate_id": candidate_id, "source_url": source_url}

        client = Client()
        for kind in ("checkpoint", "pending"):
            with self.subTest(kind=kind):
                if kind == "checkpoint":
                    path = self.working / "checkpoints" / "XC999999991.json"
                else:
                    path = _pending_qc_path(self.working, self.candidate["candidate_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                target = self.working / f"broken-{kind}-target.json"
                path.symlink_to(target)
                with self.assertRaises(UnknownAudioError):
                    self._run(client)
                self.assertTrue(path.is_symlink())
                self.assertFalse(target.exists())
                self.assertEqual(client.calls, 0)
                path.unlink()


class UnknownAudioPublicBoundaryTests(unittest.TestCase):
    def test_supplied_preflight_must_match_regenerated_record(self) -> None:
        config_sha = __import__("hashlib").sha256(CONFIG_PATH.read_bytes()).hexdigest()
        regenerated = {
            "config_sha256": config_sha,
            "disk": {"available_bytes": 10**12, "estimated_space_sufficient": True},
        }
        supplied = copy.deepcopy(regenerated)
        supplied["candidates"] = [{"candidate_id": "XC1"}]
        with (
            patch("bird_audio.unknown_audio.preflight_unknown_audio", return_value=regenerated),
            self.assertRaisesRegex(UnknownAudioError, "does not match regenerated"),
        ):
            run_unknown_audio_acquisition(
                object(),
                preflight=supplied,
                ffprobe="ffprobe",
                ffmpeg="ffmpeg",
            )

    def test_acquisition_requires_positive_disk_preflight(self) -> None:
        config_sha = __import__("hashlib").sha256(CONFIG_PATH.read_bytes()).hexdigest()
        regenerated = {
            "config_sha256": config_sha,
            "disk": {"available_bytes": None, "estimated_space_sufficient": None},
        }
        root = PROJECT_ROOT / "data" / "unknown" / "interim"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="disk-preflight-test-", dir=root) as temporary:
            isolated = Path(temporary)
            outputs = {
                "working directory": isolated / "working",
                "raw directory": isolated / "raw",
                "audit": isolated / "audit.json",
                "audit lock": isolated / "audit_lock.json",
            }

            def isolated_output(path: str | Path, label: str) -> Path:
                return outputs.get(label, _locked_output_path(path, label))

            with (
                patch(
                    "bird_audio.unknown_audio.preflight_unknown_audio",
                    return_value=regenerated,
                ),
                patch(
                    "bird_audio.unknown_audio._locked_output_path",
                    side_effect=isolated_output,
                ),
                self.assertRaisesRegex(UnknownAudioError, "disk preflight"),
            ):
                run_unknown_audio_acquisition(object(), ffprobe="ffprobe", ffmpeg="ffmpeg")

    def test_locked_output_rejects_symlink_component_before_use(self) -> None:
        root = PROJECT_ROOT / "data" / "unknown" / "interim"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="output-link-test-", dir=root) as temporary:
            directory = Path(temporary)
            target = directory / "target"
            target.mkdir()
            link = directory / "link"
            link.symlink_to(target, target_is_directory=True)
            relative = (link / "artifact.json").relative_to(PROJECT_ROOT).as_posix()
            with self.assertRaisesRegex(UnknownAudioError, "symbolic link"):
                _locked_output_path(relative, "test output")

    def test_publication_rejects_any_pending_or_staging_root(self) -> None:
        root = PROJECT_ROOT / "data" / "unknown" / "interim"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pending-publish-test-", dir=root) as temporary:
            working = Path(temporary)
            (working / "pending_qc").mkdir()
            with self.assertRaisesRegex(UnknownAudioError, "pending_qc"):
                _require_no_pending_or_staging_artifacts(working)

    def test_global_acquisition_stops_before_later_species_after_retryable(self) -> None:
        config = load_unknown_audio_config(CONFIG_PATH)
        config_sha = __import__("hashlib").sha256(CONFIG_PATH.read_bytes()).hexdigest()
        names = [f"Species {index}" for index in range(5)]
        preflight = {
            "config_sha256": config_sha,
            "disk": {"available_bytes": 10**12, "estimated_space_sufficient": True},
            "known_manifest_sha256": "k" * 64,
            "candidates": [
                {"candidate_id": f"XC{index + 1}", "scientific_name": name}
                for index, name in enumerate(names)
            ],
        }
        plan = {
            "candidate_queues": [{"scientific_name": name, "role": "primary"} for name in names]
        }
        retryable = {
            "role": "primary",
            "scientific_name": names[0],
            "inventory_recordings": 1,
            "terminal_recordings": 0,
            "eligible_recordings": 0,
            "unresolved_retryable": 1,
            "unresolved_candidate_ids": ["XC1"],
            "completion_state": "blocked_retryable",
            "dispositions": {"unresolved_retryable": 1},
            "eligible_descriptors": [],
        }
        root = PROJECT_ROOT / "data" / "unknown" / "interim"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="global-stop-test-", dir=root) as temporary:
            directory = Path(temporary)
            paths = {
                "working directory": directory / "working",
                "raw directory": directory / "raw",
                "audit": directory / "audit.json",
                "audit lock": directory / "audit-lock.json",
            }
            with (
                patch(
                    "bird_audio.unknown_audio.load_unknown_audio_config",
                    return_value=config,
                ),
                patch(
                    "bird_audio.unknown_audio.preflight_unknown_audio",
                    return_value=preflight,
                ),
                patch("bird_audio.unknown_audio._load_plan_for_preflight", return_value=plan),
                patch(
                    "bird_audio.unknown_audio.read_csv_snapshot",
                    return_value=([], "k" * 64),
                ),
                patch(
                    "bird_audio.unknown_audio._locked_output_path",
                    side_effect=lambda _value, context: paths[context],
                ),
                patch("bird_audio.unknown_audio._validate_injected_client_policy"),
                patch("bird_audio.unknown_audio.project_lock", return_value=nullcontext()),
                patch("bird_audio.unknown_audio._prepare_acquisition_roots"),
                patch(
                    "bird_audio.unknown_audio._species_audit", return_value=retryable
                ) as species_audit,
                patch("bird_audio.unknown_audio._checkpoint_set", return_value=[]),
            ):
                result = _run_unknown_audio_acquisition(
                    object(), ffprobe="ffprobe", ffmpeg="ffmpeg"
                )
        self.assertEqual(result["status"], "blocked_retryable_or_incomplete_primary_audit")
        self.assertEqual(species_audit.call_count, 1)


class UnknownAudioOrphanRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        root = PROJECT_ROOT / "data" / "unknown" / "interim"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="orphan-audit-test-", dir=root)
        self.directory = Path(self.temporary.name)
        self.audit_path = self.directory / "audit.json"
        self.lock_path = self.directory / "audit_lock.json"
        self.expected_audit = {"schema_version": "1.0", "ready": True}
        self.expected_lock = {"schema_version": "1.0", "audit_sha256": "a" * 64}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _recover(self) -> None:
        with (
            patch(
                "bird_audio.unknown_audio._publication_records",
                return_value=([], {}, {}, self.expected_audit, self.expected_lock),
            ),
            patch("bird_audio.unknown_audio._require_publication_inputs_unchanged"),
        ):
            _recover_orphan_audit(
                config_file=CONFIG_PATH,
                config={},
                config_sha256="b" * 64,
                preflight={},
                plan={},
                known_path=KNOWN_MANIFEST,
                known_sha256="c" * 64,
                known_hashes=set(),
                working_directory=self.directory,
                raw_directory=self.directory,
                audit_path=self.audit_path,
                audit_lock_path=self.lock_path,
            )

    def test_exact_orphan_audit_recovers_only_the_lock(self) -> None:
        original = _canonical_json_bytes(self.expected_audit)
        self.audit_path.write_bytes(original)
        self.audit_path.chmod(0o600)
        self._recover()
        self.assertEqual(self.audit_path.read_bytes(), original)
        self.assertEqual(json.loads(self.lock_path.read_text(encoding="utf-8")), self.expected_lock)

    def test_noncanonical_orphan_is_preserved_and_not_locked(self) -> None:
        original = json.dumps(self.expected_audit).encode("utf-8")
        self.audit_path.write_bytes(original)
        self.audit_path.chmod(0o600)
        with self.assertRaisesRegex(UnknownAudioError, "canonical|exact reproducible publication"):
            self._recover()
        self.assertEqual(self.audit_path.read_bytes(), original)
        self.assertFalse(self.lock_path.exists())


class UnknownAudioSelectionTests(unittest.TestCase):
    def test_exact_five_by_40_hungarian_selection(self) -> None:
        reference = {
            "reference_slots": [
                {
                    "slot_id": f"slot_{index:02d}",
                    "source_recording_id": f"KNOWN{index}",
                    "source_sha256": f"{index:064x}",
                    "source_session_group": f"known-session-{index}",
                    "container": "mp3",
                    "source_rate_bucket": "48000",
                    "channels": "mono",
                    "quality": "A",
                    "duration_bucket": "10_to_below_30",
                    "duration_seconds": "10.000000",
                }
                for index in range(40)
            ]
        }
        candidates: dict[str, list[dict]] = {}
        for species_index in range(5):
            scientific_name = f"Species {species_index}"
            candidates[scientific_name] = [
                {
                    "candidate_id": f"XC{species_index + 1}{index + 1000}",
                    "session_group": f"session-{species_index}-{index}",
                    "container": "mp3",
                    "source_rate_bucket": "48000",
                    "channels": "mono",
                    "quality": "A",
                    "duration_bucket": "10_to_below_30",
                    "duration_seconds": "10.000000",
                }
                for index in range(40)
            ]
        result = select_final_unknown_recordings(reference, candidates)
        self.assertEqual(result["species_count"], 5)
        self.assertEqual(result["selected_recordings"], 200)
        self.assertTrue(result["zero_session_overlap"])

    def test_selection_rejects_forbidden_outcome_field(self) -> None:
        reference = {"reference_slots": []}
        candidates = {f"Species {index}": [] for index in range(5)}
        candidates["Species 0"] = [{"model_score": 1.0}]
        with self.assertRaisesRegex(UnknownAudioError, "forbidden fields"):
            select_final_unknown_recordings(reference, candidates)


if __name__ == "__main__":
    unittest.main()
