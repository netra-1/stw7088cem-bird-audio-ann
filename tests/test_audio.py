from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from bird_audio.audio import (
    AudioToolError,
    decode_smoke_test,
    detect_header,
    probe_audio,
    tool_version,
    verify_full_decode,
)


class DetectHeaderTests(unittest.TestCase):
    def _write_header(self, payload: bytes) -> Path:
        with tempfile.NamedTemporaryFile(delete=False) as temporary:
            temporary.write(payload)
            path = Path(temporary.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_detects_riff_wave_with_mp3_irrelevant_suffix(self) -> None:
        path = self._write_header(b"RIFF\x08\x00\x00\x00WAVEfmt ")
        self.assertEqual(detect_header(path), "riff_wave")

    def test_detects_id3_mp3(self) -> None:
        path = self._write_header(b"ID3\x04\x00\x00\x00\x00\x00\x00")
        self.assertEqual(detect_header(path), "mp3_id3")

    def test_detects_mpeg_frame_sync(self) -> None:
        path = self._write_header(bytes([0xFF, 0xFB, 0x90, 0x64]))
        self.assertEqual(detect_header(path), "mpeg_audio")

    @patch("bird_audio.audio.subprocess.run")
    def test_probe_preserves_both_bit_depth_fields(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"streams":[{"codec_name":"pcm_s16le","codec_long_name":"PCM",'
                '"sample_rate":"44100","channels":2,"sample_fmt":"s16",'
                '"bits_per_sample":16,"bits_per_raw_sample":"0","duration":"2.0"}],'
                '"format":{"format_name":"wav","duration":"2.0","bit_rate":"1411200"}}'
            ),
            stderr="",
        )
        result = probe_audio(Path("fixture.mp3"), Path("ffprobe"))
        self.assertTrue(result.probe_ok)
        self.assertEqual(result.bits_per_sample, 16)
        self.assertEqual(result.bits_per_raw_sample, 0)
        self.assertEqual(result.ffprobe_duration_seconds, 2.0)

    @patch("bird_audio.audio.subprocess.run")
    def test_full_decode_extracts_duration_and_normalizes_addresses(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout="out_time_us=3000000\nprogress=end\n",
            stderr="[decoder @ 0x12abc]  Invalid packet\n",
        )
        result = verify_full_decode(Path("fixture.mp3"), Path("ffmpeg"))
        self.assertEqual(result.decoded_duration_seconds, 3.0)
        self.assertEqual(result.diagnostic, "[decoder @ 0xADDR] Invalid packet")


class AudioSubprocessHardeningTests(unittest.TestCase):
    secret = "private key+/value"
    source = Path("private/audio/fixture.mp3")

    def _assert_safe_invocation(
        self,
        run_mock,
        *,
        protocols: str | None,
        nostdin: bool,
    ) -> list[str]:
        command = run_mock.call_args.args[0]
        options = run_mock.call_args.kwargs
        self.assertIsInstance(command, list)
        if nostdin:
            self.assertIn("-nostdin", command)
        else:
            self.assertNotIn("-nostdin", command)
        self.assertNotIn("shell", options)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertNotIn("XENO_CANTO_API_KEY", options["env"])
        self.assertEqual(options["env"]["AUDIO_TEST_SENTINEL"], "preserved")
        if protocols is None:
            self.assertNotIn("-protocol_whitelist", command)
        else:
            index = command.index("-protocol_whitelist")
            self.assertEqual(command[index + 1], protocols)
        return command

    @patch("bird_audio.audio.subprocess.run")
    def test_tool_version_scrubs_environment_and_blocks_stdin(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[], returncode=0, stdout="ffmpeg version test\n", stderr=""
        )
        with patch.dict(
            os.environ,
            {
                "XENO_CANTO_API_KEY": self.secret,
                "AUDIO_TEST_SENTINEL": "preserved",
            },
        ):
            result = tool_version(Path("ffmpeg"))

        self.assertEqual(result, "ffmpeg version test")
        command = self._assert_safe_invocation(run_mock, protocols=None, nostdin=True)
        self.assertEqual(command, ["ffmpeg", "-nostdin", "-version"])

    @patch("bird_audio.audio.subprocess.run")
    def test_ffprobe_version_avoids_unsupported_nostdin_option(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[], returncode=0, stdout="ffprobe version test\n", stderr=""
        )
        with patch.dict(
            os.environ,
            {
                "XENO_CANTO_API_KEY": self.secret,
                "AUDIO_TEST_SENTINEL": "preserved",
            },
        ):
            result = tool_version(Path("ffprobe"))

        self.assertEqual(result, "ffprobe version test")
        command = self._assert_safe_invocation(run_mock, protocols=None, nostdin=False)
        self.assertEqual(command, ["ffprobe", "-version"])

    @patch("bird_audio.audio.subprocess.run")
    def test_probe_uses_only_local_file_protocol_and_scrubs_environment(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"streams":[{"codec_name":"mp3","sample_rate":"44100",'
                '"channels":2,"duration":"1.0"}],'
                '"format":{"format_name":"mp3","duration":"1.0"}}'
            ),
            stderr="",
        )
        with patch.dict(
            os.environ,
            {
                "XENO_CANTO_API_KEY": self.secret,
                "AUDIO_TEST_SENTINEL": "preserved",
            },
        ):
            result = probe_audio(self.source, Path("ffprobe"))

        self.assertTrue(result.probe_ok)
        command = self._assert_safe_invocation(run_mock, protocols="file", nostdin=False)
        self.assertEqual(command[-1], str(self.source))

    @patch("bird_audio.audio.subprocess.run")
    def test_smoke_decode_uses_only_local_and_pipe_protocols(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=struct.pack("<ff", -0.25, 0.5),
            stderr=b"",
        )
        with patch.dict(
            os.environ,
            {
                "XENO_CANTO_API_KEY": self.secret,
                "AUDIO_TEST_SENTINEL": "preserved",
            },
        ):
            result = decode_smoke_test(self.source, Path("ffmpeg"))

        self.assertEqual(result.sample_count, 2)
        command = self._assert_safe_invocation(run_mock, protocols="file,pipe", nostdin=True)
        self.assertEqual(command[command.index("-i") + 1], str(self.source))

    @patch("bird_audio.audio.subprocess.run")
    def test_full_decode_uses_only_local_and_pipe_protocols(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout="out_time_us=1000000\nprogress=end\n",
            stderr="",
        )
        with patch.dict(
            os.environ,
            {
                "XENO_CANTO_API_KEY": self.secret,
                "AUDIO_TEST_SENTINEL": "preserved",
            },
        ):
            result = verify_full_decode(self.source, Path("ffmpeg"))

        self.assertEqual(result.decoded_duration_seconds, 1.0)
        command = self._assert_safe_invocation(run_mock, protocols="file,pipe", nostdin=True)
        self.assertEqual(command[command.index("-i") + 1], str(self.source))

    def test_tool_version_normalizes_timeout_oserror_and_nonzero_exit(self) -> None:
        failures = (
            subprocess.TimeoutExpired(cmd=[str(self.source), self.secret], timeout=15),
            OSError(f"cannot execute {self.source} with {self.secret}"),
        )
        with patch.dict(os.environ, {"XENO_CANTO_API_KEY": self.secret}):
            for failure in failures:
                with (
                    self.subTest(failure=type(failure).__name__),
                    patch("bird_audio.audio.subprocess.run", side_effect=failure),
                ):
                    self.assertEqual(tool_version(Path("ffmpeg")), "unavailable")

            completed = CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr=f"failed at {self.source} with {self.secret}",
            )
            with patch("bird_audio.audio.subprocess.run", return_value=completed):
                self.assertEqual(tool_version(Path("ffmpeg")), "unavailable")

    def test_probe_normalizes_timeout_and_oserror_without_sensitive_text(self) -> None:
        failures = (
            (
                subprocess.TimeoutExpired(cmd=[str(self.source), self.secret], timeout=30),
                "ffprobe invocation timed out",
            ),
            (
                OSError(f"cannot probe {self.source} with {self.secret}"),
                "ffprobe invocation failed",
            ),
        )
        with patch.dict(os.environ, {"XENO_CANTO_API_KEY": self.secret}):
            for failure, expected in failures:
                with (
                    self.subTest(failure=type(failure).__name__),
                    patch("bird_audio.audio.subprocess.run", side_effect=failure),
                ):
                    result = probe_audio(self.source, Path("ffprobe"))
                self.assertFalse(result.probe_ok)
                self.assertEqual(result.probe_error, expected)
                self.assertNotIn(self.secret, result.probe_error)
                self.assertNotIn(str(self.source), result.probe_error)

    def test_decoders_normalize_timeout_and_oserror_without_sensitive_text(self) -> None:
        cases = (
            (
                decode_smoke_test,
                subprocess.TimeoutExpired(cmd=[str(self.source), self.secret], timeout=30),
                "FFmpeg decode timed out",
            ),
            (
                decode_smoke_test,
                OSError(f"cannot decode {self.source} with {self.secret}"),
                "FFmpeg decode invocation failed",
            ),
            (
                verify_full_decode,
                subprocess.TimeoutExpired(cmd=[str(self.source), self.secret], timeout=3600),
                "Full decode timed out",
            ),
            (
                verify_full_decode,
                OSError(f"cannot decode {self.source} with {self.secret}"),
                "Full decode invocation failed",
            ),
        )
        with patch.dict(os.environ, {"XENO_CANTO_API_KEY": self.secret}):
            for function, failure, expected in cases:
                with (
                    self.subTest(function=function.__name__, failure=type(failure).__name__),
                    patch("bird_audio.audio.subprocess.run", side_effect=failure),
                    self.assertRaisesRegex(AudioToolError, f"^{expected}$") as caught,
                ):
                    function(self.source, Path("ffmpeg"))
                message = str(caught.exception)
                self.assertNotIn(self.secret, message)
                self.assertNotIn(str(self.source), message)
                self.assertIsNone(caught.exception.__cause__)
                self.assertTrue(caught.exception.__suppress_context__)

    def test_nonzero_diagnostics_redact_secret_and_input_path(self) -> None:
        diagnostic = f"decoder failed for {self.source} with {self.secret} at 0x123abc"
        with patch.dict(os.environ, {"XENO_CANTO_API_KEY": self.secret}):
            probe_completed = CompletedProcess(args=[], returncode=1, stdout="", stderr=diagnostic)
            with patch("bird_audio.audio.subprocess.run", return_value=probe_completed):
                probe = probe_audio(self.source, Path("ffprobe"))
            self.assertFalse(probe.probe_ok)
            self.assertIn("[INPUT]", probe.probe_error)
            self.assertIn("[REDACTED]", probe.probe_error)
            self.assertIn("0xADDR", probe.probe_error)

            smoke_completed = CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=diagnostic.encode()
            )
            with (
                patch("bird_audio.audio.subprocess.run", return_value=smoke_completed),
                self.assertRaises(AudioToolError) as smoke_caught,
            ):
                decode_smoke_test(self.source, Path("ffmpeg"))

            full_completed = CompletedProcess(args=[], returncode=1, stdout="", stderr=diagnostic)
            with (
                patch("bird_audio.audio.subprocess.run", return_value=full_completed),
                self.assertRaises(AudioToolError) as full_caught,
            ):
                verify_full_decode(self.source, Path("ffmpeg"))

        for message in (
            probe.probe_error,
            str(smoke_caught.exception),
            str(full_caught.exception),
        ):
            self.assertNotIn(self.secret, message)
            self.assertNotIn(str(self.source), message)
            self.assertIn("[INPUT]", message)
            self.assertIn("[REDACTED]", message)

    def test_nonzero_probe_diagnostic_redacts_encoded_secret_variants(self) -> None:
        quoted = urllib.parse.quote(self.secret, safe="")
        variants = (
            quoted,
            urllib.parse.quote_plus(self.secret, safe=""),
            urllib.parse.quote(quoted, safe=""),
            quoted.replace("%2B", "%2b").replace("%2F", "%2f"),
        )
        with patch.dict(os.environ, {"XENO_CANTO_API_KEY": self.secret}):
            for variant in variants:
                completed = CompletedProcess(
                    args=[], returncode=1, stdout="", stderr=f"failure token={variant}"
                )
                with (
                    self.subTest(variant=variant),
                    patch("bird_audio.audio.subprocess.run", return_value=completed),
                ):
                    result = probe_audio(self.source, Path("ffprobe"))
                self.assertFalse(result.probe_ok)
                self.assertNotIn(variant, result.probe_error)
                self.assertIn("[REDACTED]", result.probe_error)

    @patch("bird_audio.audio.subprocess.run")
    def test_invalid_probe_json_does_not_echo_payload(self, run_mock) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"not-json {self.secret} {self.source}",
            stderr="",
        )
        with patch.dict(os.environ, {"XENO_CANTO_API_KEY": self.secret}):
            result = probe_audio(self.source, Path("ffprobe"))
        self.assertEqual(result.probe_error, "Invalid ffprobe JSON")
        self.assertNotIn(self.secret, result.probe_error)
        self.assertNotIn(str(self.source), result.probe_error)


if __name__ == "__main__":
    unittest.main()
