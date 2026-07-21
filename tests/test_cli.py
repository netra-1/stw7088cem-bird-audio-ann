from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bird_audio.cli import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_UNKNOWN_ACQUISITION_CONFIG,
    DEFAULT_UNKNOWN_AUDIO_CONFIG,
    DEFAULT_UNKNOWN_CANDIDATE_PLAN,
    DEFAULT_UNKNOWN_CANDIDATE_PLAN_LOCK,
    DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
    DEFAULT_UNKNOWN_METADATA_LOCK,
    DEFAULT_UNKNOWN_SEALED_METADATA,
    DEFAULT_UNKNOWN_SELECTION_CONFIG,
    DEFAULT_UNKNOWN_WORKING_METADATA,
    TASK1_KNOWN_CACHE_LOCK_SHA256,
    TASK1_KNOWN_CACHE_ROOT,
    TASK1_RUN_ROOT,
    TASK2_KNOWN_CACHE_LOCK_SHA256,
    TASK2_KNOWN_CACHE_ROOT,
    TASK2_RUN_ROOT,
    build_parser,
)
from bird_audio.metadata import XenoCantoApiError
from bird_audio.unknown_acquisition import (
    UnknownAcquisitionConfigError,
    UnknownAcquisitionCredentialError,
    UnknownMetadataCacheError,
)
from bird_audio.unknown_audio import UnknownAudioError


class UnknownMetadataCliTests(unittest.TestCase):
    def test_unknown_metadata_commands_use_separate_metadata_only_defaults(self) -> None:
        parser = build_parser()

        discover = parser.parse_args(["discover-unknown-metadata"])
        self.assertEqual(discover.config, DEFAULT_UNKNOWN_ACQUISITION_CONFIG)
        self.assertEqual(discover.working_cache, DEFAULT_UNKNOWN_WORKING_METADATA)
        self.assertIn("data/unknown/metadata", discover.working_cache)

        seal = parser.parse_args(["seal-unknown-metadata"])
        self.assertEqual(seal.output, DEFAULT_UNKNOWN_SEALED_METADATA)
        self.assertEqual(seal.lock, DEFAULT_UNKNOWN_METADATA_LOCK)

        validate = parser.parse_args(["validate-unknown-metadata"])
        self.assertEqual(validate.cache, DEFAULT_UNKNOWN_SEALED_METADATA)
        self.assertEqual(validate.lock, DEFAULT_UNKNOWN_METADATA_LOCK)

    @patch("bird_audio.cli.fetch_unknown_metadata_cache")
    def test_discovery_output_is_a_small_status_summary_not_recording_metadata(self, fetch) -> None:
        fetch.return_value = (
            Path("/project/data/unknown/metadata/working.json"),
            {
                "complete": True,
                "species": {
                    "Ceryle rudis": {
                        "active": True,
                        "role": "primary",
                        "snapshot": {"num_recordings": 123},
                        "pages": {"1": {"recordings": [{"rmk": "not printed"}]}},
                    }
                },
            },
        )
        args = build_parser().parse_args(["discover-unknown-metadata"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        printed = output.getvalue()
        payload = json.loads(printed)
        self.assertEqual(exit_status, 0)
        self.assertTrue(payload["metadata_only"])
        self.assertEqual(payload["species"]["Ceryle rudis"]["recordings_reported"], 123)
        self.assertNotIn("not printed", printed)
        progress_callback = fetch.call_args.kwargs["progress_callback"]
        progress_output = io.StringIO()
        with redirect_stdout(progress_output):
            progress_callback(
                {
                    "phase": "fetch",
                    "scientific_name": "Ceryle rudis",
                    "page": 2,
                    "total_pages": 4,
                }
            )
        self.assertEqual(
            progress_output.getvalue(),
            "Fetched metadata: Ceryle rudis page 2/4\n",
        )

    def test_discovery_expected_failures_print_one_error_line_without_traceback(self) -> None:
        failures = (
            XenoCantoApiError(
                "Acridotheres fuscus reports fewer than the locked 80 candidate recordings"
            ),
            UnknownAcquisitionConfigError("unknown acquisition config is invalid"),
            UnknownAcquisitionCredentialError(
                "a non-empty XENO_CANTO_API_KEY environment variable is required"
            ),
            UnknownMetadataCacheError("unknown metadata cache is invalid"),
        )
        for failure in failures:
            args = build_parser().parse_args(["discover-unknown-metadata"])
            output = io.StringIO()
            errors = io.StringIO()

            with (
                self.subTest(error_type=type(failure).__name__),
                patch(
                    "bird_audio.cli.fetch_unknown_metadata_cache",
                    side_effect=failure,
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                exit_status = args.handler(args)

            self.assertEqual(exit_status, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(errors.getvalue(), f"ERROR: {failure}\n")
            self.assertNotIn("Traceback", errors.getvalue())

    @patch("bird_audio.cli.fetch_unknown_metadata_cache")
    def test_discovery_redacts_raw_and_encoded_key_variants_from_domain_errors(self, fetch) -> None:
        secret = "raw secret+/value"
        variants = (
            secret,
            "raw%20secret%2B%2Fvalue",
            "raw+secret%2B%2Fvalue",
            "raw%20secret%2b%2fvalue",
            "raw%2520secret%252B%252Fvalue",
        )
        args = build_parser().parse_args(["discover-unknown-metadata"])
        for variant in variants:
            fetch.side_effect = UnknownMetadataCacheError(
                f"invalid field {variant} at https://example.invalid/?key={variant}"
            )
            output = io.StringIO()
            errors = io.StringIO()
            with (
                self.subTest(variant=variant),
                patch.dict(os.environ, {"XENO_CANTO_API_KEY": secret}),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                exit_status = args.handler(args)

            self.assertEqual(exit_status, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertTrue(errors.getvalue().startswith("ERROR: "))
            self.assertNotIn("https://example.invalid", errors.getvalue())
            for secret_variant in variants:
                self.assertNotIn(secret_variant, errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())

    @patch(
        "bird_audio.cli.fetch_unknown_metadata_cache",
        side_effect=KeyError("programming fault"),
    )
    def test_discovery_does_not_hide_unexpected_programming_errors(self, _fetch) -> None:
        args = build_parser().parse_args(["discover-unknown-metadata"])

        with self.assertRaisesRegex(KeyError, "programming fault"):
            args.handler(args)

    @patch(
        "bird_audio.cli.fetch_unknown_metadata_cache",
        side_effect=ValueError("unexpected value fault"),
    )
    def test_discovery_does_not_hide_unexpected_value_errors(self, _fetch) -> None:
        args = build_parser().parse_args(["discover-unknown-metadata"])

        with self.assertRaisesRegex(ValueError, "unexpected value fault"):
            args.handler(args)


class UnknownPlanningCliTests(unittest.TestCase):
    def test_planning_commands_use_versioned_metadata_only_defaults(self) -> None:
        parser = build_parser()

        build = parser.parse_args(["build-unknown-candidate-plan"])
        self.assertEqual(build.config, DEFAULT_UNKNOWN_SELECTION_CONFIG)
        self.assertEqual(build.metadata, DEFAULT_UNKNOWN_SEALED_METADATA)
        self.assertEqual(build.metadata_lock, DEFAULT_UNKNOWN_METADATA_LOCK)
        self.assertEqual(build.output, DEFAULT_UNKNOWN_CANDIDATE_PLAN)
        self.assertEqual(build.lock, DEFAULT_UNKNOWN_CANDIDATE_PLAN_LOCK)
        self.assertIn("data/unknown/planning", build.output)

        validate = parser.parse_args(["validate-unknown-candidate-plan"])
        self.assertEqual(validate.plan, DEFAULT_UNKNOWN_CANDIDATE_PLAN)
        self.assertEqual(validate.lock, DEFAULT_UNKNOWN_CANDIDATE_PLAN_LOCK)

    @patch("bird_audio.cli.build_unknown_candidate_plan")
    def test_build_planning_command_binds_every_input_and_prints_summary(self, build) -> None:
        build.return_value = (
            Path("/project/data/unknown/planning/plan.json"),
            Path("/project/data/unknown/planning/plan_lock.json"),
            {
                "valid": True,
                "ready_for_candidate_qc": True,
                "candidate_recordings_total": 480,
                "reference_slots": 40,
                "plan_sha256": "a" * 64,
            },
        )
        args = build_parser().parse_args(
            [
                "build-unknown-candidate-plan",
                "--config",
                "selection.toml",
                "--metadata",
                "unknown.json",
                "--metadata-lock",
                "unknown_lock.json",
                "--manifest",
                "known.csv",
                "--review-lock",
                "review.json",
                "--split",
                "split.csv",
                "--split-summary",
                "split_summary.json",
                "--split-lock",
                "split_lock.json",
                "--output",
                "plan.json",
                "--lock",
                "plan_lock.json",
            ]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        build.assert_called_once_with(
            config_path="selection.toml",
            unknown_metadata_path="unknown.json",
            unknown_metadata_lock_path="unknown_lock.json",
            manifest_path="known.csv",
            review_lock_path="review.json",
            split_path="split.csv",
            split_summary_path="split_summary.json",
            split_lock_path="split_lock.json",
            output_path="plan.json",
            lock_path="plan_lock.json",
        )
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["ready_for_candidate_qc"])
        self.assertEqual(payload["candidate_recordings_total"], 480)
        self.assertEqual(payload["reference_slots"], 40)
        self.assertNotIn("candidate_queues", payload)

    @patch("bird_audio.cli.validate_unknown_candidate_plan")
    def test_validate_planning_command_checks_expected_plan(self, validate) -> None:
        validate.return_value = {
            "valid": True,
            "ready_for_candidate_qc": True,
            "candidate_recordings_total": 480,
            "reference_slots": 40,
            "plan_sha256": "b" * 64,
        }
        args = build_parser().parse_args(
            [
                "validate-unknown-candidate-plan",
                "--plan",
                "plan.json",
                "--lock",
                "plan_lock.json",
            ]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        validate.assert_called_once_with("plan_lock.json", "plan.json")
        self.assertTrue(json.loads(output.getvalue())["valid"])


class UnknownAudioPreflightCliTests(unittest.TestCase):
    @staticmethod
    def _result(*, sufficient: bool) -> dict[str, object]:
        return {
            "candidate_pool_target_recordings_per_species": 80,
            "target_recordings_per_species": 40,
            "network_requests": 0,
            "audio_downloads": 0,
            "fallback_active": False,
            "estimated_active_download_bytes": 300_000_000,
            "estimated_fallback_contingency_bytes": 21_600_000,
            "estimated_download_bytes_with_fallback_contingency": 321_600_000,
            "estimated_required_disk_bytes": 1_700_000_000,
            "preflight_sha256": "a" * 64,
            "disk": {
                "available_bytes": 90_000_000_000,
                "estimated_required_bytes": 1_700_000_000,
                "estimated_space_sufficient": sufficient,
            },
            "species": [
                {
                    "role": "primary",
                    "active": True,
                    "scientific_name": "Psilopogon zeylanicus",
                    "inventory_recordings": 95,
                    "canonical_sessions_before_audio_qc": 42,
                    "estimated_download_duration_seconds": 1200,
                    "estimated_download_bytes": 28_800_000,
                    "fallback_status": "not_applicable",
                    "private_candidate_detail": "must not print",
                },
                {
                    "role": "primary",
                    "active": True,
                    "scientific_name": "Acridotheres fuscus",
                    "inventory_recordings": 39,
                    "canonical_sessions_before_audio_qc": 13,
                    "estimated_download_duration_seconds": 200,
                    "estimated_download_bytes": 4_800_000,
                    "fallback_status": "not_applicable",
                },
                {
                    "role": "fallback",
                    "active": False,
                    "scientific_name": "Streptopelia orientalis",
                    "inventory_recordings": 129,
                    "canonical_sessions_before_audio_qc": 66,
                    "estimated_download_duration_seconds": 900,
                    "estimated_download_bytes": 21_600_000,
                    "fallback_status": "inactive_until_protocol_gate",
                },
            ],
            "candidates": [
                {
                    "candidate_id": "XC123",
                    "download_url": "https://example.invalid/private-audio",
                    "session_group": "private-session",
                    "remarks": "private remarks",
                }
            ],
        }

    def test_parser_exposes_only_the_locked_config_override(self) -> None:
        args = build_parser().parse_args(["preflight-unknown-audio"])

        self.assertEqual(args.config, DEFAULT_UNKNOWN_AUDIO_CONFIG)
        for unsafe in (
            "plan",
            "lock",
            "output",
            "disk",
            "species",
            "limit",
            "workers",
            "overwrite",
            "ffmpeg",
            "ffprobe",
            "api_key",
        ):
            self.assertFalse(hasattr(args, unsafe), unsafe)

    @patch("bird_audio.cli.resolve_tool")
    @patch("bird_audio.cli.preflight_unknown_audio")
    def test_command_prints_only_the_compact_allowlisted_summary(
        self, preflight, resolve_tool_mock
    ) -> None:
        preflight.return_value = self._result(sufficient=True)
        args = build_parser().parse_args(
            ["preflight-unknown-audio", "--config", "configs/custom.toml"]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        preflight.assert_called_once_with("configs/custom.toml")
        resolve_tool_mock.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(
            set(payload),
            {
                "mode",
                "network_requests",
                "audio_downloads",
                "fallback_active",
                "candidate_pool_target_recordings_per_species",
                "target_recordings_per_species",
                "candidate_recordings_total",
                "active_canonical_sessions_before_audio_qc",
                "estimated_active_download_bytes",
                "estimated_fallback_contingency_bytes",
                "estimated_download_bytes_with_fallback_contingency",
                "estimated_required_disk_bytes",
                "available_disk_bytes",
                "estimated_space_sufficient",
                "preflight_sha256",
                "species",
            },
        )
        self.assertEqual(payload["mode"], "read_only_preflight")
        self.assertEqual(payload["candidate_recordings_total"], 263)
        self.assertEqual(payload["active_canonical_sessions_before_audio_qc"], 55)
        self.assertEqual(payload["estimated_fallback_contingency_bytes"], 21_600_000)
        self.assertEqual(
            payload["estimated_download_bytes_with_fallback_contingency"],
            321_600_000,
        )
        barbet = payload["species"]["Psilopogon zeylanicus"]
        jungle = payload["species"]["Acridotheres fuscus"]
        self.assertEqual(barbet["final_target_margin_before_audio_qc"], 2)
        self.assertEqual(barbet["candidate_pool_shortfall_before_audio_qc"], 38)
        self.assertEqual(jungle["final_target_margin_before_audio_qc"], -27)
        self.assertEqual(jungle["candidate_pool_shortfall_before_audio_qc"], 67)
        printed = output.getvalue()
        for private_text in (
            "must not print",
            "XC123",
            "https://example.invalid/private-audio",
            "private-session",
            "private remarks",
            "private_candidate_detail",
            "candidates",
        ):
            self.assertNotIn(private_text, printed)

    @patch("bird_audio.cli.preflight_unknown_audio")
    def test_insufficient_estimated_space_returns_one_with_the_summary(self, preflight) -> None:
        preflight.return_value = self._result(sufficient=False)
        args = build_parser().parse_args(["preflight-unknown-audio"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 1)
        self.assertFalse(json.loads(output.getvalue())["estimated_space_sufficient"])

    @patch(
        "bird_audio.cli.preflight_unknown_audio",
        side_effect=UnknownAudioError("unknown audio config is invalid"),
    )
    def test_expected_error_is_one_stderr_line_without_traceback(self, _preflight) -> None:
        args = build_parser().parse_args(["preflight-unknown-audio"])
        output = io.StringIO()
        errors = io.StringIO()

        with redirect_stdout(output), redirect_stderr(errors):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(errors.getvalue(), "ERROR: unknown audio config is invalid\n")
        self.assertNotIn("Traceback", errors.getvalue())

    def test_unexpected_programming_errors_are_not_hidden(self) -> None:
        args = build_parser().parse_args(["preflight-unknown-audio"])
        failures = (KeyError("programming fault"), ValueError("unexpected value fault"))
        for failure in failures:
            with (
                self.subTest(failure=type(failure).__name__),
                patch("bird_audio.cli.preflight_unknown_audio", side_effect=failure),
                self.assertRaises(type(failure)),
            ):
                args.handler(args)

    def test_signal_smoke_has_a_provenance_output_default(self) -> None:
        args = build_parser().parse_args(["signal-smoke", "dataset/example.mp3"])
        self.assertEqual(
            args.output,
            "report_assets/provenance_v2/signal_smoke_v2.json",
        )

    def test_environment_and_mps_smoke_use_v2_provenance_defaults(self) -> None:
        parser = build_parser()
        environment = parser.parse_args(["environment"])
        smoke = parser.parse_args(["mps-smoke"])
        self.assertEqual(
            environment.output,
            "report_assets/provenance_v2/environment_v2.json",
        )
        self.assertEqual(
            smoke.output,
            "report_assets/provenance_v2/mps_smoke_v2.json",
        )
        self.assertEqual(
            smoke.checkpoint,
            "report_assets/provenance_v2/mps_smoke_checkpoint_v2.pt",
        )

    def test_known_clip_cache_commands_use_the_versioned_processed_root(self) -> None:
        parser = build_parser()
        build = parser.parse_args(["build-known-clip-cache"])
        verify = parser.parse_args(["verify-known-clip-cache"])
        audit = parser.parse_args(["audit-known-clip-cache"])
        self.assertEqual(build.cache_root, DEFAULT_CACHE_ROOT)
        self.assertEqual(verify.cache_root, DEFAULT_CACHE_ROOT)
        self.assertEqual(audit.cache_root, DEFAULT_CACHE_ROOT)
        self.assertIn("data/processed/known_clips_v1", build.cache_root)

    def test_unknown_clip_cache_commands_use_a_separate_versioned_root(self) -> None:
        parser = build_parser()
        build = parser.parse_args(["build-unknown-clip-cache"])
        verify = parser.parse_args(["verify-unknown-clip-cache"])
        self.assertEqual(build.cache_root, DEFAULT_UNKNOWN_CLIP_CACHE_ROOT)
        self.assertEqual(verify.cache_root, DEFAULT_UNKNOWN_CLIP_CACHE_ROOT)
        self.assertIn("data/processed/unknown_clips_v2", build.cache_root)
        self.assertNotEqual(build.cache_root, DEFAULT_CACHE_ROOT)

    @patch("bird_audio.cli.build_known_clip_cache")
    def test_cache_build_progress_is_throttled_and_final_summary_is_compact(
        self, build_cache
    ) -> None:
        def run_build(_root, **kwargs):
            progress = kwargs["progress_callback"]
            progress(
                {
                    "event": "preflight",
                    "recordings_total": 20,
                    "recordings_completed": 5,
                    "recordings_remaining": 15,
                    "required_free_bytes": 2 * 1024**3,
                    "available_free_bytes": 10 * 1024**3,
                }
            )
            for completed in (6, 10, 20):
                progress(
                    {
                        "event": "recording_complete",
                        "recording_id": f"XC{completed}",
                        "recordings_completed": completed,
                        "recordings_total": 20,
                        "resumed": False,
                    }
                )
            progress(
                {
                    "event": "published",
                    "recordings": 20,
                    "clips": 55,
                    "destination": "/project/cache",
                }
            )
            return Path("/project/cache"), {
                "totals": {"recordings": 20, "clips": 55, "feature_bytes": 1234}
            }

        build_cache.side_effect = run_build
        args = build_parser().parse_args(["build-known-clip-cache"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        printed = output.getvalue()
        self.assertEqual(exit_status, 0)
        self.assertIn("Cache preflight: 5/20 complete", printed)
        self.assertNotIn("Cached recordings: 6/20", printed)
        self.assertIn("Cached recordings: 10/20", printed)
        self.assertIn("Cached recordings: 20/20", printed)
        self.assertIn("Published cache: 20 recordings, 55 unique features", printed)

    @patch("bird_audio.cli.build_unknown_clip_cache")
    def test_unknown_cache_build_reports_scoring_only_progress(self, build_cache) -> None:
        def run_build(_root, **kwargs):
            progress = kwargs["progress_callback"]
            progress(
                {
                    "event": "preflight",
                    "recordings_total": 200,
                    "recordings_completed": 0,
                    "recordings_remaining": 200,
                    "required_free_bytes": 1024**3,
                    "available_free_bytes": 10 * 1024**3,
                }
            )
            progress(
                {
                    "event": "recording_complete",
                    "candidate_id": "XC1",
                    "recordings_completed": 1,
                    "recordings_total": 200,
                    "resumed": False,
                }
            )
            progress(
                {
                    "event": "published",
                    "recordings": 200,
                    "clips": 800,
                    "destination": "/project/unknown-cache",
                }
            )
            return Path("/project/unknown-cache"), {
                "totals": {
                    "species": 5,
                    "recordings": 200,
                    "clips": 800,
                    "feature_bytes": 1234,
                }
            }

        build_cache.side_effect = run_build
        args = build_parser().parse_args(["build-unknown-clip-cache"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        printed = output.getvalue()
        self.assertEqual(exit_status, 0)
        self.assertIn("Unknown cache preflight: 0/200 complete", printed)
        self.assertIn("Cached unknown recordings: 1/200", printed)
        self.assertIn("Published unknown cache: 200 recordings, 800 energy features", printed)


class UnknownAudioAcquisitionCliTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "download": {
                "allowed_initial_hosts": ["xeno-canto.org"],
                "allowed_redirect_hosts": ["xeno-canto.org"],
                "maximum_redirects": 3,
                "request_interval_seconds": 1.0,
                "timeout_seconds": 60.0,
                "total_timeout_seconds": 900.0,
                "maximum_retries": 3,
                "maximum_retry_after_seconds": 60.0,
                "chunk_size_bytes": 1_048_576,
                "maximum_file_size_bytes": 536_870_912,
                "user_agent": "STW7088CEM-bird-audio-coursework/0.1",
                "proxy_policy": "disabled",
                "cookie_policy": "disabled",
            }
        }

    @staticmethod
    def _verified_result(*, valid: bool = True) -> dict[str, object]:
        return {
            "valid": valid,
            "ready_for_unknown_scoring": True,
            "selected_recordings": 200,
            "species": 5,
            "fallback_active": True,
            "audit": "data/unknown/audio/unknown_audio_audit_v1.json",
            "audit_sha256": "b" * 64,
            "checkpoint_count": 905,
            "private_verification_detail": "must not print",
        }

    def test_parsers_expose_only_the_locked_operational_options(self) -> None:
        parser = build_parser()
        acquire = parser.parse_args(["acquire-unknown-audio"])
        verify = parser.parse_args(["verify-unknown-audio-audit"])

        self.assertEqual(acquire.config, DEFAULT_UNKNOWN_AUDIO_CONFIG)
        self.assertIsNone(acquire.ffmpeg)
        self.assertIsNone(acquire.ffprobe)
        self.assertEqual(verify.config, DEFAULT_UNKNOWN_AUDIO_CONFIG)
        self.assertFalse(hasattr(verify, "ffmpeg"))
        self.assertFalse(hasattr(verify, "ffprobe"))
        for unsafe in (
            "api_key",
            "plan",
            "lock",
            "output",
            "overwrite",
            "workers",
            "species",
            "limit",
        ):
            self.assertFalse(hasattr(acquire, unsafe), unsafe)
            self.assertFalse(hasattr(verify, unsafe), unsafe)

    @patch("bird_audio.cli.run_unknown_audio_acquisition")
    @patch("bird_audio.cli.resolve_tool")
    @patch("bird_audio.cli.SecureXenoCantoAudioClient")
    @patch("bird_audio.cli.load_unknown_audio_config")
    def test_acquire_constructs_exact_policy_and_prints_compact_verified_summary(
        self,
        load_config,
        client_class,
        resolve_tool_mock,
        acquire,
    ) -> None:
        config = self._config()
        load_config.return_value = config
        resolve_tool_mock.side_effect = [Path("/tools/ffmpeg"), Path("/tools/ffprobe")]
        acquire.return_value = self._verified_result()
        args = build_parser().parse_args(
            [
                "acquire-unknown-audio",
                "--config",
                "configs/custom.toml",
                "--ffmpeg",
                "/tools/ffmpeg",
                "--ffprobe",
                "/tools/ffprobe",
            ]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        load_config.assert_called_once_with("configs/custom.toml")
        policy = client_class.call_args.args[0]
        download = config["download"]
        self.assertEqual(policy.allowed_hosts, tuple(download["allowed_initial_hosts"]))
        self.assertEqual(policy.maximum_redirects, download["maximum_redirects"])
        self.assertEqual(
            policy.request_interval_seconds,
            download["request_interval_seconds"],
        )
        self.assertEqual(policy.timeout_seconds, download["timeout_seconds"])
        self.assertEqual(policy.total_timeout_seconds, download["total_timeout_seconds"])
        self.assertEqual(policy.maximum_retries, download["maximum_retries"])
        self.assertEqual(policy.chunk_size_bytes, download["chunk_size_bytes"])
        self.assertEqual(
            policy.maximum_file_bytes,
            download["maximum_file_size_bytes"],
        )
        self.assertEqual(
            policy.maximum_retry_after_seconds,
            download["maximum_retry_after_seconds"],
        )
        self.assertEqual(policy.user_agent, download["user_agent"])
        self.assertEqual(
            resolve_tool_mock.call_args_list,
            [
                unittest.mock.call("ffmpeg", "/tools/ffmpeg"),
                unittest.mock.call("ffprobe", "/tools/ffprobe"),
            ],
        )
        positional, keywords = acquire.call_args
        self.assertEqual(positional, (client_class.return_value,))
        self.assertEqual(keywords["config_path"], "configs/custom.toml")
        self.assertEqual(keywords["ffmpeg"], Path("/tools/ffmpeg"))
        self.assertEqual(keywords["ffprobe"], Path("/tools/ffprobe"))
        self.assertTrue(callable(keywords["progress_callback"]))
        payload = json.loads(output.getvalue())
        self.assertEqual(
            set(payload),
            {
                "complete",
                "status",
                "ready_for_unknown_scoring",
                "selected_recordings",
                "species_count",
                "fallback_active",
                "checkpoint_count",
                "audit",
                "audit_sha256",
            },
        )
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["status"], "complete")
        self.assertNotIn("private_verification_detail", output.getvalue())

    @patch("bird_audio.cli.run_unknown_audio_acquisition")
    @patch("bird_audio.cli.resolve_tool")
    @patch("bird_audio.cli.SecureXenoCantoAudioClient")
    @patch("bird_audio.cli.load_unknown_audio_config")
    def test_incomplete_acquisition_is_allowlisted_and_returns_one(
        self,
        load_config,
        _client_class,
        resolve_tool_mock,
        acquire,
    ) -> None:
        load_config.return_value = self._config()
        resolve_tool_mock.side_effect = [Path("/tools/ffmpeg"), Path("/tools/ffprobe")]
        acquire.return_value = {
            "complete": False,
            "status": "blocked_retryable_or_incomplete_primary_audit",
            "gate": {
                "fallback_active": False,
                "failed_primary_species": [],
                "blocked_species": ["Ceryle rudis"],
                "replacement": None,
                "private_gate_detail": "must not print",
            },
            "species": [
                {
                    "scientific_name": "Ceryle rudis",
                    "role": "primary",
                    "inventory_recordings": 211,
                    "terminal_recordings": 19,
                    "eligible_recordings": 12,
                    "unresolved_retryable": 1,
                    "completion_state": "blocked_retryable",
                    "dispositions": {
                        "eligible": 12,
                        "audio_qc_excluded": 7,
                        "private_disposition": 99,
                    },
                    "unresolved_candidate_ids": ["XC-PRIVATE"],
                    "eligible_descriptors": [{"download_url": "private-url"}],
                }
            ],
            "checkpoint_count": 19,
        }
        args = build_parser().parse_args(["acquire-unknown-audio"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["unresolved_retryable_total"], 1)
        self.assertIsNone(payload["reason"])
        self.assertEqual(
            payload["species"]["Ceryle rudis"]["dispositions"],
            {"audio_qc_excluded": 7, "eligible": 12},
        )
        for private_text in (
            "must not print",
            "private_gate_detail",
            "private_disposition",
            "XC-PRIVATE",
            "private-url",
            "unresolved_candidate_ids",
            "eligible_descriptors",
        ):
            self.assertNotIn(private_text, output.getvalue())

    @patch("bird_audio.cli.run_unknown_audio_acquisition")
    @patch("bird_audio.cli.resolve_tool")
    @patch("bird_audio.cli.SecureXenoCantoAudioClient")
    @patch("bird_audio.cli.load_unknown_audio_config")
    def test_protocol_decision_prints_the_locked_reason(
        self,
        load_config,
        _client_class,
        resolve_tool_mock,
        acquire,
    ) -> None:
        load_config.return_value = self._config()
        resolve_tool_mock.side_effect = [Path("/tools/ffmpeg"), Path("/tools/ffprobe")]
        acquire.return_value = {
            "complete": False,
            "status": "protocol_decision_required",
            "gate": {
                "fallback_active": True,
                "reason": "fallback_below_40",
                "failed_primary_species": ["Acridotheres fuscus"],
                "blocked_species": [],
                "replacement": None,
            },
            "species": [],
            "checkpoint_count": 905,
        }
        args = build_parser().parse_args(["acquire-unknown-audio"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "protocol_decision_required")
        self.assertEqual(payload["reason"], "fallback_below_40")

    @patch("bird_audio.cli.run_unknown_audio_acquisition")
    @patch("bird_audio.cli.resolve_tool")
    @patch("bird_audio.cli.SecureXenoCantoAudioClient")
    @patch("bird_audio.cli.load_unknown_audio_config")
    def test_progress_is_throttled_and_omits_candidate_details(
        self,
        load_config,
        _client_class,
        resolve_tool_mock,
        acquire,
    ) -> None:
        load_config.return_value = self._config()
        resolve_tool_mock.side_effect = [Path("/tools/ffmpeg"), Path("/tools/ffprobe")]
        acquire.return_value = self._verified_result()
        args = build_parser().parse_args(["acquire-unknown-audio"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(args.handler(args), 0)
        progress = acquire.call_args.kwargs["progress_callback"]
        output = io.StringIO()

        with redirect_stdout(output):
            for terminal in (1, 2, 10):
                progress(
                    {
                        "event": "candidate_terminal",
                        "scientific_name": "Ceryle rudis",
                        "terminal_recordings": terminal,
                        "inventory_recordings": 211,
                        "eligible_recordings": terminal - 1,
                        "candidate_id": "XC-PRIVATE",
                        "queue_rank": terminal,
                        "disposition": "private-disposition",
                    }
                )
            progress(
                {
                    "event": "species_complete",
                    "scientific_name": "Ceryle rudis",
                    "terminal_recordings": 211,
                    "inventory_recordings": 211,
                    "eligible_recordings": 80,
                    "completion_state": "pool_satisfied",
                }
            )

        printed = output.getvalue()
        self.assertIn("1/211 terminal", printed)
        self.assertNotIn("2/211 terminal", printed)
        self.assertIn("10/211 terminal", printed)
        self.assertIn("211/211 terminal", printed)
        self.assertNotIn("XC-PRIVATE", printed)
        self.assertNotIn("private-disposition", printed)

    def test_expected_acquisition_errors_are_one_safe_line(self) -> None:
        args = build_parser().parse_args(["acquire-unknown-audio"])
        failures = (
            ("run", UnknownAudioError("locked input changed\nresume safely")),
            ("tool", FileNotFoundError("ffmpeg was not found\ninstall FFmpeg")),
        )
        for source, failure in failures:
            output = io.StringIO()
            errors = io.StringIO()
            with (
                self.subTest(source=source),
                patch("bird_audio.cli.load_unknown_audio_config", return_value=self._config()),
                patch("bird_audio.cli.SecureXenoCantoAudioClient"),
                patch(
                    "bird_audio.cli.resolve_tool",
                    side_effect=(
                        failure
                        if source == "tool"
                        else [Path("/tools/ffmpeg"), Path("/tools/ffprobe")]
                    ),
                ),
                patch(
                    "bird_audio.cli.run_unknown_audio_acquisition",
                    side_effect=failure if source == "run" else None,
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                exit_status = args.handler(args)

            self.assertEqual(exit_status, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(len(errors.getvalue().splitlines()), 1)
            self.assertTrue(errors.getvalue().startswith("ERROR: "))
            self.assertNotIn("Traceback", errors.getvalue())

    def test_acquisition_programming_faults_propagate(self) -> None:
        args = build_parser().parse_args(["acquire-unknown-audio"])
        for failure in (KeyError("programming fault"), ValueError("value fault")):
            with (
                self.subTest(failure=type(failure).__name__),
                patch("bird_audio.cli.load_unknown_audio_config", return_value=self._config()),
                patch("bird_audio.cli.SecureXenoCantoAudioClient"),
                patch(
                    "bird_audio.cli.resolve_tool",
                    side_effect=[Path("/tools/ffmpeg"), Path("/tools/ffprobe")],
                ),
                patch("bird_audio.cli.run_unknown_audio_acquisition", side_effect=failure),
                self.assertRaises(type(failure)),
            ):
                args.handler(args)


class UnknownAudioVerificationCliTests(unittest.TestCase):
    @staticmethod
    def _result(*, valid: bool) -> dict[str, object]:
        return UnknownAudioAcquisitionCliTests._verified_result(valid=valid)

    @patch("bird_audio.cli.verify_unknown_audio_audit")
    def test_verify_prints_only_compact_allowlisted_summary(self, verify) -> None:
        verify.return_value = self._result(valid=True)
        args = build_parser().parse_args(
            ["verify-unknown-audio-audit", "--config", "configs/custom.toml"]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        verify.assert_called_once_with("configs/custom.toml")
        payload = json.loads(output.getvalue())
        self.assertEqual(
            set(payload),
            {
                "valid",
                "ready_for_unknown_scoring",
                "selected_recordings",
                "species_count",
                "fallback_active",
                "checkpoint_count",
                "audit",
                "audit_sha256",
            },
        )
        self.assertNotIn("private_verification_detail", output.getvalue())

    @patch("bird_audio.cli.verify_unknown_audio_audit")
    def test_invalid_verification_returns_one(self, verify) -> None:
        verify.return_value = self._result(valid=False)
        args = build_parser().parse_args(["verify-unknown-audio-audit"])
        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)
        self.assertEqual(exit_status, 1)

    def test_expected_verification_errors_are_one_safe_line(self) -> None:
        args = build_parser().parse_args(["verify-unknown-audio-audit"])
        failures = (
            UnknownAudioError("audit binding failed\nrun acquisition again"),
            FileNotFoundError("audit file was not found\nrun acquisition first"),
        )
        for failure in failures:
            output = io.StringIO()
            errors = io.StringIO()
            with (
                self.subTest(failure=type(failure).__name__),
                patch("bird_audio.cli.verify_unknown_audio_audit", side_effect=failure),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                exit_status = args.handler(args)

            self.assertEqual(exit_status, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(len(errors.getvalue().splitlines()), 1)
            self.assertTrue(errors.getvalue().startswith("ERROR: "))
            self.assertNotIn("Traceback", errors.getvalue())

    def test_verification_programming_faults_propagate(self) -> None:
        args = build_parser().parse_args(["verify-unknown-audio-audit"])
        for failure in (KeyError("programming fault"), ValueError("value fault")):
            with (
                self.subTest(failure=type(failure).__name__),
                patch("bird_audio.cli.verify_unknown_audio_audit", side_effect=failure),
                self.assertRaises(type(failure)),
            ):
                args.handler(args)


class TrainingCliTests(unittest.TestCase):
    def _assert_parse_error(self, argv: list[str]) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_training_command_surfaces_are_narrow(self) -> None:
        parser = build_parser()

        preflight = parser.parse_args(["preflight-task1-weights"])
        self.assertFalse(preflight.populate)
        populated = parser.parse_args(["preflight-task1-weights", "--populate"])
        self.assertTrue(populated.populate)

        task1_benchmark = parser.parse_args(["benchmark-task1", "--ffmpeg", "/tools/ffmpeg"])
        self.assertEqual(task1_benchmark.ffmpeg, "/tools/ffmpeg")
        self.assertFalse(hasattr(task1_benchmark, "seed"))

        task2_benchmark = parser.parse_args(["benchmark-task2", "--ffmpeg", "/tools/ffmpeg"])
        self.assertEqual(task2_benchmark.ffmpeg, "/tools/ffmpeg")
        self.assertFalse(hasattr(task2_benchmark, "seed"))

        for command in ("train-task1", "train-task2"):
            for seed in (13, 37, 71):
                with self.subTest(command=command, seed=seed):
                    parsed = parser.parse_args([command, "--seed", str(seed)])
                    self.assertEqual(parsed.seed, seed)
                    self.assertIsNone(parsed.ffmpeg)
                    self.assertIsNone(parsed.resume_checkpoint)
                    self.assertIsNone(parsed.resume_checkpoint_sha256)

    def test_benchmarks_and_training_reject_forbidden_options(self) -> None:
        valued_forbidden = (
            "--cache-root",
            "--config",
            "--output-root",
            "--run-id",
            "--device",
            "--strategy",
            "--threshold",
            "--model",
            "--test-injection",
            "--weights",
        )
        bases = (
            ["benchmark-task1"],
            ["benchmark-task2"],
            ["train-task1", "--seed", "13"],
            ["train-task2", "--seed", "13"],
        )
        for base in bases:
            for option in valued_forbidden:
                with self.subTest(command=base[0], option=option):
                    self._assert_parse_error([*base, option, "forbidden"])
            with self.subTest(command=base[0], option="--populate"):
                self._assert_parse_error([*base, "--populate"])

        for command in ("benchmark-task1", "benchmark-task2"):
            with self.subTest(command=command, option="--seed"):
                self._assert_parse_error([command, "--seed", "37"])

        self._assert_parse_error(["preflight-task1-weights", "--ffmpeg", "/tools/ffmpeg"])
        self._assert_parse_error(["preflight-task1-weights", "--pop"])

    def test_training_seed_is_required_and_locked(self) -> None:
        for command in ("train-task1", "train-task2"):
            with self.subTest(command=command, seed="missing"):
                self._assert_parse_error([command])
            for seed in (0, 12, 14, 36, 72):
                with self.subTest(command=command, seed=seed):
                    self._assert_parse_error([command, "--seed", str(seed)])

    def test_resume_options_must_be_supplied_together(self) -> None:
        for command in ("train-task1", "train-task2"):
            base = [command, "--seed", "13"]
            with self.subTest(command=command, missing="sha256"):
                self._assert_parse_error([*base, "--resume-checkpoint", "recovery.pt"])
            with self.subTest(command=command, missing="checkpoint"):
                self._assert_parse_error([*base, "--resume-checkpoint-sha256", "a" * 64])

    @patch("bird_audio.cli.preflight_efficientnet_weights")
    def test_weight_preflight_prints_only_safe_artifact_metadata(self, preflight) -> None:
        preflight.return_value = SimpleNamespace(
            identifier="EfficientNet_B0_Weights.IMAGENET1K_V1",
            path=Path("/cache/weights.pt"),
            sha256="a" * 64,
            size_bytes=12345,
        )
        args = build_parser().parse_args(["preflight-task1-weights", "--populate"])
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        preflight.assert_called_once_with(populate=True)
        payload = json.loads(output.getvalue())
        self.assertEqual(
            set(payload),
            {"identifier", "path", "populated", "sha256", "size_bytes", "verified"},
        )
        self.assertEqual(payload["path"], "/cache/weights.pt")
        self.assertTrue(payload["populated"])
        self.assertTrue(payload["verified"])

    @patch("bird_audio.cli.benchmark_task1_full_epoch")
    def test_task1_benchmark_binds_production_inputs_and_exact_argv(self, benchmark) -> None:
        benchmark.return_value = {
            "benchmark_only": True,
            "result_artifact": {"path": Path("/project/task1_benchmark.json")},
        }
        actual_argv = [
            "/project/.venv/bin/bird-audio",
            "benchmark-task1",
            "--ffmpeg",
            "/tools/ffmpeg",
        ]
        output = io.StringIO()

        with patch.object(sys, "argv", actual_argv):
            args = build_parser().parse_args()
            with redirect_stdout(output):
                exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        benchmark.assert_called_once_with(
            seed=13,
            cache_root=TASK1_KNOWN_CACHE_ROOT,
            ffmpeg="/tools/ffmpeg",
            expected_lock_sha256=TASK1_KNOWN_CACHE_LOCK_SHA256,
            command=tuple(actual_argv),
        )
        self.assertEqual(
            json.loads(output.getvalue())["result_artifact"]["path"],
            "/project/task1_benchmark.json",
        )

    @patch("bird_audio.cli.run_task1_development")
    def test_task1_training_binds_production_inputs(self, run) -> None:
        run.return_value = {
            "complete": True,
            "run_directory": Path("/project/runs/task1_v2/run"),
        }
        argv = ["train-task1", "--seed", "37", "--ffmpeg", "/tools/ffmpeg"]
        args = build_parser().parse_args(argv)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        run.assert_called_once_with(
            seed=37,
            cache_root=TASK1_KNOWN_CACHE_ROOT,
            ffmpeg="/tools/ffmpeg",
            expected_lock_sha256=TASK1_KNOWN_CACHE_LOCK_SHA256,
            output_root=TASK1_RUN_ROOT,
            command=tuple(argv),
            resume_checkpoint=None,
            resume_checkpoint_sha256=None,
        )
        self.assertEqual(
            json.loads(output.getvalue())["run_directory"],
            "/project/runs/task1_v2/run",
        )

    @patch("bird_audio.cli.run_task1_development")
    def test_task1_training_passes_paired_resume_inputs(self, run) -> None:
        run.return_value = {"complete": True}
        argv = [
            "train-task1",
            "--seed",
            "71",
            "--resume-checkpoint",
            "runs/task1_v2/run/recovery/recovery_epoch_0002.pt",
            "--resume-checkpoint-sha256",
            "b" * 64,
        ]
        args = build_parser().parse_args(argv)

        with redirect_stdout(io.StringIO()):
            args.handler(args)

        self.assertEqual(
            run.call_args.kwargs["resume_checkpoint"],
            "runs/task1_v2/run/recovery/recovery_epoch_0002.pt",
        )
        self.assertEqual(run.call_args.kwargs["resume_checkpoint_sha256"], "b" * 64)
        self.assertEqual(run.call_args.kwargs["command"], tuple(argv))

    @patch("bird_audio.cli.benchmark_task2_full_epoch")
    def test_task2_benchmark_binds_production_inputs(self, benchmark) -> None:
        benchmark.return_value = {"benchmark_only": True}
        argv = ["benchmark-task2", "--ffmpeg", "/tools/ffmpeg"]
        args = build_parser().parse_args(argv)

        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        benchmark.assert_called_once_with(
            seed=13,
            cache_root=TASK2_KNOWN_CACHE_ROOT,
            ffmpeg="/tools/ffmpeg",
            expected_lock_sha256=TASK2_KNOWN_CACHE_LOCK_SHA256,
            command=tuple(argv),
        )

    @patch("bird_audio.cli.run_task2_development")
    def test_task2_training_binds_production_and_resume_inputs(self, run) -> None:
        run.return_value = {"complete": True}
        argv = [
            "train-task2",
            "--seed",
            "13",
            "--resume-checkpoint",
            "runs/task2_v2/run/recovery/recovery_epoch_0003.pt",
            "--resume-checkpoint-sha256",
            "c" * 64,
        ]
        args = build_parser().parse_args(argv)

        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)

        self.assertEqual(exit_status, 0)
        run.assert_called_once_with(
            seed=13,
            cache_root=TASK2_KNOWN_CACHE_ROOT,
            ffmpeg=None,
            expected_lock_sha256=TASK2_KNOWN_CACHE_LOCK_SHA256,
            output_root=TASK2_RUN_ROOT,
            command=tuple(argv),
            resume_checkpoint="runs/task2_v2/run/recovery/recovery_epoch_0003.pt",
            resume_checkpoint_sha256="c" * 64,
        )

    def test_training_command_programming_faults_propagate(self) -> None:
        scenarios = (
            ("preflight-task1-weights", "bird_audio.cli.preflight_efficientnet_weights"),
            ("benchmark-task1", "bird_audio.cli.benchmark_task1_full_epoch"),
            ("train-task1 --seed 13", "bird_audio.cli.run_task1_development"),
            ("benchmark-task2", "bird_audio.cli.benchmark_task2_full_epoch"),
            ("train-task2 --seed 13", "bird_audio.cli.run_task2_development"),
        )
        for command, target in scenarios:
            args = build_parser().parse_args(command.split())
            with (
                self.subTest(command=command),
                patch(target, side_effect=KeyError("programming fault")),
                self.assertRaisesRegex(KeyError, "programming fault"),
            ):
                args.handler(args)


class RecoveryCliTests(unittest.TestCase):
    def test_recovery_command_surfaces_are_narrow(self) -> None:
        parser = build_parser()
        verify_v1 = parser.parse_args(["verify-v1-recovery-manifest"])
        seal_v2 = parser.parse_args(["seal-v2-cache-equivalence", "--ffmpeg", "/tools/ffmpeg"])
        verify_v2 = parser.parse_args(
            [
                "verify-v2-cache-equivalence",
                "--ffmpeg",
                "/tools/ffmpeg",
                "--full-rederivation",
            ]
        )

        self.assertFalse(hasattr(verify_v1, "ffmpeg"))
        self.assertEqual(seal_v2.ffmpeg, "/tools/ffmpeg")
        self.assertEqual(verify_v2.ffmpeg, "/tools/ffmpeg")
        self.assertTrue(verify_v2.full_rederivation)

    @patch("bird_audio.cli.verify_v1_recovery_manifest")
    def test_v1_recovery_manifest_dispatch(self, verify) -> None:
        verify.return_value = {"valid": True}
        args = build_parser().parse_args(["verify-v1-recovery-manifest"])
        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)
        self.assertEqual(exit_status, 0)
        verify.assert_called_once_with()

    @patch("bird_audio.cli.seal_unknown_cache_v2_equivalence")
    def test_v2_equivalence_seal_dispatch(self, seal) -> None:
        seal.return_value = {"valid": True, "created": True}
        args = build_parser().parse_args(["seal-v2-cache-equivalence", "--ffmpeg", "/tools/ffmpeg"])
        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)
        self.assertEqual(exit_status, 0)
        seal.assert_called_once_with(ffmpeg="/tools/ffmpeg")

    @patch("bird_audio.cli.verify_unknown_cache_v2_equivalence_certificate")
    def test_v2_equivalence_verify_dispatch(self, verify) -> None:
        verify.return_value = {"valid": True, "created": False}
        args = build_parser().parse_args(
            [
                "verify-v2-cache-equivalence",
                "--ffmpeg",
                "/tools/ffmpeg",
                "--full-rederivation",
            ]
        )
        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)
        self.assertEqual(exit_status, 0)
        verify.assert_called_once_with(
            ffmpeg="/tools/ffmpeg",
            full_rederivation=True,
        )


class FinalEvaluationCliTests(unittest.TestCase):
    def _assert_parse_error(self, argv: list[str]) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_final_evaluation_command_surfaces_are_narrow(self) -> None:
        parser = build_parser()
        seal = parser.parse_args(["seal-final-evaluation-gate"])
        run = parser.parse_args(["run-final-evaluation", "--ffmpeg", "/tools/ffmpeg"])
        verify = parser.parse_args(["verify-final-evaluation"])
        self.assertFalse(hasattr(seal, "ffmpeg"))
        self.assertEqual(run.ffmpeg, "/tools/ffmpeg")
        self.assertFalse(hasattr(verify, "ffmpeg"))

        forbidden = (
            "--attempt-id",
            "--output-root",
            "--seed",
            "--resume",
            "--data-root",
            "--model",
            "--checkpoint",
            "--threshold",
            "--refit",
            "--bootstrap-seed",
            "--bootstrap-replicates",
            "--test-injection",
        )
        for command in (
            "seal-final-evaluation-gate",
            "run-final-evaluation",
            "verify-final-evaluation",
        ):
            for option in forbidden:
                with self.subTest(command=command, option=option):
                    self._assert_parse_error([command, option, "forbidden"])
        self._assert_parse_error(["seal-final-evaluation-gate", "--ffmpeg", "/tools/ffmpeg"])
        self._assert_parse_error(["verify-final-evaluation", "--ffmpeg", "/tools/ffmpeg"])

    @patch("bird_audio.cli.seal_final_evaluation_gate")
    def test_seal_final_gate_dispatch_has_no_mutable_inputs(self, seal) -> None:
        seal.return_value = {"created": True, "gate": {"ready": True}}
        args = build_parser().parse_args(["seal-final-evaluation-gate"])
        output = io.StringIO()
        with redirect_stdout(output):
            exit_status = args.handler(args)
        self.assertEqual(exit_status, 0)
        seal.assert_called_once_with()
        self.assertTrue(json.loads(output.getvalue())["gate"]["ready"])

    @patch("bird_audio.cli.run_final_evaluation")
    def test_run_final_evaluation_binds_exact_argv(self, run) -> None:
        run.return_value = {"complete": True, "attempt_id": "final_evaluation_attempt_v2"}
        argv = [
            "/project/.venv/bin/bird-audio",
            "run-final-evaluation",
            "--ffmpeg",
            "/tools/ffmpeg",
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", argv):
            args = build_parser().parse_args()
            with redirect_stdout(output):
                exit_status = args.handler(args)
        self.assertEqual(exit_status, 0)
        run.assert_called_once_with(
            ffmpeg="/tools/ffmpeg",
            command=tuple(argv),
        )
        self.assertTrue(json.loads(output.getvalue())["complete"])

    @patch("bird_audio.cli.verify_final_evaluation")
    def test_verify_final_evaluation_dispatch_has_no_overrides(self, verify) -> None:
        verify.return_value = {"complete": True}
        args = build_parser().parse_args(["verify-final-evaluation"])
        with redirect_stdout(io.StringIO()):
            exit_status = args.handler(args)
        self.assertEqual(exit_status, 0)
        verify.assert_called_once_with()

    def test_final_evaluation_programming_faults_propagate(self) -> None:
        scenarios = (
            (
                "seal-final-evaluation-gate",
                "bird_audio.cli.seal_final_evaluation_gate",
            ),
            ("run-final-evaluation", "bird_audio.cli.run_final_evaluation"),
            ("verify-final-evaluation", "bird_audio.cli.verify_final_evaluation"),
        )
        for command, target in scenarios:
            args = build_parser().parse_args([command])
            with (
                self.subTest(command=command),
                patch(target, side_effect=KeyError("programming fault")),
                self.assertRaisesRegex(KeyError, "programming fault"),
            ):
                args.handler(args)


class FinalEvidenceCliTests(unittest.TestCase):
    COMMANDS = (
        "build-final-report-assets",
        "verify-final-report-assets",
        "build-task1-attributions",
        "verify-task1-attributions",
    )

    def _assert_parse_error(self, argv: list[str]) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_final_evidence_command_surfaces_have_no_overrides(self) -> None:
        parser = build_parser()
        for command in self.COMMANDS:
            args = parser.parse_args([command])
            self.assertEqual(args.command, command)
            self.assertFalse(hasattr(args, "ffmpeg"))
            for option in (
                "--output-root",
                "--seed",
                "--model",
                "--checkpoint",
                "--recording-id",
                "--correct-count",
                "--error-count",
                "--target-layer",
                "--test-injection",
            ):
                with self.subTest(command=command, option=option):
                    self._assert_parse_error([command, option, "forbidden"])

    def test_final_report_asset_dispatch_has_no_mutable_inputs(self) -> None:
        scenarios = (
            (
                "build-final-report-assets",
                "bird_audio.cli.build_final_report_assets",
                {"created": True, "assets": []},
            ),
            (
                "verify-final-report-assets",
                "bird_audio.cli.verify_final_report_assets",
                {"created": False, "assets": []},
            ),
            (
                "build-task1-attributions",
                "bird_audio.cli.build_task1_attributions",
                {"created": True, "manifest": {"image_count": 6}},
            ),
            (
                "verify-task1-attributions",
                "bird_audio.cli.verify_task1_attributions",
                {"created": False, "manifest": {"image_count": 6}},
            ),
        )
        for command, target, result in scenarios:
            with self.subTest(command=command), patch(target, return_value=result) as operation:
                args = build_parser().parse_args([command])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_status = args.handler(args)
                self.assertEqual(exit_status, 0)
                operation.assert_called_once_with()
                self.assertEqual(json.loads(output.getvalue()), result)

    def test_final_evidence_programming_faults_propagate(self) -> None:
        scenarios = (
            ("build-final-report-assets", "bird_audio.cli.build_final_report_assets"),
            ("verify-final-report-assets", "bird_audio.cli.verify_final_report_assets"),
            ("build-task1-attributions", "bird_audio.cli.build_task1_attributions"),
            ("verify-task1-attributions", "bird_audio.cli.verify_task1_attributions"),
        )
        for command, target in scenarios:
            args = build_parser().parse_args([command])
            with (
                self.subTest(command=command),
                patch(target, side_effect=KeyError("programming fault")),
                self.assertRaisesRegex(KeyError, "programming fault"),
            ):
                args.handler(args)


if __name__ == "__main__":
    unittest.main()
