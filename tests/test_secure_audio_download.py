from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import unittest
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import bird_audio.secure_audio_download as secure_download
from bird_audio.secure_audio_download import (
    AudioDownloadCommandError,
    AudioDownloadSecurityError,
    DownloadPolicy,
    RetryableAudioDownloadError,
    SecureXenoCantoAudioClient,
    TerminalAudioUnavailableError,
)


def _headers(**values: str) -> Message:
    headers = Message()
    for name, value in values.items():
        headers.add_header(name.replace("_", "-"), value)
    return headers


class _FakeResponse:
    def __init__(
        self,
        status: int,
        body: bytes = b"",
        headers: Message | None = None,
        *,
        chunks: list[bytes | BaseException | object] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or Message()
        self._body = io.BytesIO(body)
        self._chunks = list(chunks) if chunks is not None else None
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self._chunks is None:
            return self._body.read(size)
        if not self._chunks:
            return b""
        value = self._chunks.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[return-value]

    def close(self) -> None:
        self.closed = True


class _AdvancingResponse(_FakeResponse):
    def __init__(self, clock: _Clock, seconds_per_read: float, body: bytes) -> None:
        super().__init__(200, body)
        self._clock = clock
        self._seconds_per_read = seconds_per_read

    def read(self, size: int = -1) -> bytes:
        value = super().read(size)
        self._clock.now += self._seconds_per_read
        return value


class _HardLinkingResponse(_FakeResponse):
    def __init__(self, body: bytes, destination: Path, alias: Path) -> None:
        super().__init__(200, body)
        self._destination = destination
        self._alias = alias

    def read(self, size: int = -1) -> bytes:
        if not self._alias.exists():
            os.link(self._destination, self._alias)
        return super().read(size)


class _FakeOpener:
    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []

    def open(self, request: urllib.request.Request, timeout: float) -> _FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self.actions:
            raise AssertionError("unexpected request")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if not isinstance(action, _FakeResponse):
            raise TypeError("invalid fake action")
        return action


class _AdvancingOpener(_FakeOpener):
    def __init__(self, actions: list[object], clock: _Clock, seconds_per_open: float) -> None:
        super().__init__(actions)
        self._clock = clock
        self._seconds_per_open = seconds_per_open

    def open(self, request: urllib.request.Request, timeout: float) -> _FakeResponse:
        response = super().open(request, timeout)
        self._clock.now += self._seconds_per_open
        return response


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SecureAudioDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate_id = "XC123"
        self.source_url = "https://xeno-canto.org/123/download"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _client(
        self,
        actions: list[object],
        policy: DownloadPolicy | None = None,
    ) -> tuple[SecureXenoCantoAudioClient, _FakeOpener, _Clock]:
        opener = _FakeOpener(actions)
        clock = _Clock()
        client = SecureXenoCantoAudioClient(
            policy or DownloadPolicy(),
            opener=opener,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        return client, opener, clock

    def test_success_is_private_hashed_and_uses_only_fixed_headers(self) -> None:
        payload = b"ID3\x04\x00\x00safe-audio"
        response = _FakeResponse(
            200,
            payload,
            _headers(Content_Length=str(len(payload)), Content_Type="audio/mpeg"),
        )
        client, opener, _ = self._client([response])
        destination = self.root / "XC123.download"

        receipt = client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        self.assertEqual(receipt.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt.bytes_written, len(payload))
        self.assertEqual(receipt.content_length, len(payload))
        self.assertEqual(receipt.content_type, "audio/mpeg")
        self.assertEqual(receipt.attempts, 1)
        request = opener.requests[0]
        request_headers = {name.casefold(): value for name, value in request.header_items()}
        self.assertEqual(request_headers["accept-encoding"], "identity")
        self.assertEqual(request_headers["connection"], "close")
        self.assertEqual(
            request_headers["user-agent"],
            "STW7088CEM-bird-audio-coursework/0.1",
        )
        self.assertNotIn("authorization", request_headers)
        self.assertNotIn("cookie", request_headers)
        self.assertNotIn("range", request_headers)
        self.assertNotIn("xeno_canto_api_key", request_headers)
        self.assertTrue(response.closed)

    def test_success_synchronizes_the_file_and_parent_directory(self) -> None:
        payload = b"ID3durable-audio"
        client, _, _ = self._client([_FakeResponse(200, payload)])
        destination = self.root / "durable"
        original_fsync = os.fsync
        synchronized_types: list[str] = []

        def observe_fsync(descriptor: int) -> None:
            mode = os.fstat(descriptor).st_mode
            synchronized_types.append("directory" if stat.S_ISDIR(mode) else "file")
            original_fsync(descriptor)

        with patch.object(secure_download.os, "fsync", side_effect=observe_fsync):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(synchronized_types, ["file", "directory"])

    def test_parent_directory_sync_failure_removes_the_owned_destination(self) -> None:
        payload = b"ID3durable-audio"
        client, _, _ = self._client([_FakeResponse(200, payload)])
        destination = self.root / "directory-sync-failure"
        original_fsync = os.fsync

        def fail_directory_fsync(descriptor: int) -> None:
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("simulated directory fsync failure")
            original_fsync(descriptor)

        with (
            patch.object(secure_download.os, "fsync", side_effect=fail_directory_fsync),
            self.assertRaisesRegex(AudioDownloadSecurityError, "synchronized"),
        ):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertFalse(destination.exists())

    def test_environment_api_key_cannot_enter_url_or_headers(self) -> None:
        payload = b"ID3audio"
        client, opener, _ = self._client([_FakeResponse(200, payload)])
        destination = self.root / "no-key"
        original = os.environ.get("XENO_CANTO_API_KEY")
        os.environ["XENO_CANTO_API_KEY"] = "environment-secret"
        try:
            client.download(self.candidate_id, self.source_url, destination)
        finally:
            if original is None:
                os.environ.pop("XENO_CANTO_API_KEY", None)
            else:
                os.environ["XENO_CANTO_API_KEY"] = original
        request = opener.requests[0]
        serialized = request.full_url + repr(request.header_items())
        self.assertNotIn("environment-secret", serialized)
        self.assertNotIn("Range", dict(request.header_items()))

    def test_policy_rejects_unlocked_or_unsafe_values(self) -> None:
        invalid = (
            {"allowed_hosts": ()},
            {"allowed_hosts": ("xeno-canto.org", "xeno-canto.org")},
            {"allowed_hosts": ("xeno-canto.org", "Example.com")},
            {"allowed_hosts": ("example.com",)},
            {"maximum_redirects": True},
            {"maximum_redirects": 6},
            {"request_interval_seconds": 0.5},
            {"timeout_seconds": float("inf")},
            {"total_timeout_seconds": 30.0},
            {"maximum_retries": -1},
            {"chunk_size_bytes": 4096},
            {"maximum_file_bytes": 1024},
            {"maximum_retry_after_seconds": 61.0},
            {"user_agent": "different"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DownloadPolicy(**kwargs)

    def test_candidate_and_source_url_are_exactly_bound(self) -> None:
        invalid = (
            ("XC0", self.source_url),
            ("XC0123", self.source_url),
            ("xc123", self.source_url),
            ("XC\uff11\uff12\uff13", self.source_url),
            ("XC124", self.source_url),
            (self.candidate_id, "http://xeno-canto.org/123/download"),
            (self.candidate_id, "https://xeno-canto.org:443/123/download"),
            (self.candidate_id, "https://xeno-canto.org/123/download?key=secret"),
            (self.candidate_id, "https://xeno-canto.org/123/download#fragment"),
            (self.candidate_id, "https://xeno-canto.org/%31%32%33/download"),
            (self.candidate_id, "https://xeno-canto.org.evil/123/download"),
            (self.candidate_id, "https://xeno-canto.org@evil.test/123/download"),
        )
        for index, (candidate_id, source_url) in enumerate(invalid):
            client, opener, _ = self._client([])
            with self.subTest(candidate_id=candidate_id, source_url=source_url):
                with self.assertRaises(AudioDownloadSecurityError):
                    client.download(candidate_id, source_url, self.root / f"invalid-{index}")
                self.assertFalse(opener.requests)

    def test_manual_same_host_redirect_is_revalidated(self) -> None:
        redirect = _FakeResponse(302, headers=_headers(Location="/sounds/XC123.mp3"))
        payload = b"ID3audio"
        final = _FakeResponse(200, payload, _headers(Content_Length=str(len(payload))))
        client, opener, clock = self._client([redirect, final])
        destination = self.root / "redirected"

        receipt = client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(receipt.redirect_count, 1)
        self.assertEqual(receipt.final_url, "https://xeno-canto.org/sounds/XC123.mp3")
        self.assertEqual(
            [request.full_url for request in opener.requests],
            [
                self.source_url,
                "https://xeno-canto.org/sounds/XC123.mp3",
            ],
        )
        self.assertIn(1.0, clock.sleeps)
        self.assertTrue(redirect.closed)

    def test_redirect_targets_fail_closed(self) -> None:
        invalid_locations = (
            "http://xeno-canto.org/sounds/XC123.mp3",
            "https://evil.test/XC123.mp3",
            "https://xeno-canto.org.evil/XC123.mp3",
            "https://user@xeno-canto.org/XC123.mp3",
            "https://xeno-canto.org:443/XC123.mp3",
            "https://xeno-canto.org/XC123.mp3?token=value",
            "https://xeno-canto.org/XC123.mp3#fragment",
            "/sounds/XC%20123.mp3",
            "//evil.test/XC123.mp3",
        )
        for index, location in enumerate(invalid_locations):
            response = _FakeResponse(302, headers=_headers(Location=location))
            client, opener, _ = self._client([response])
            destination = self.root / f"redirect-invalid-{index}"
            with self.subTest(location=location), self.assertRaises(AudioDownloadSecurityError):
                client.download(self.candidate_id, self.source_url, destination)
            self.assertEqual(len(opener.requests), 1)
            self.assertFalse(destination.exists())

    def test_redirect_loop_and_limit_are_rejected(self) -> None:
        loop = _FakeResponse(302, headers=_headers(Location=self.source_url))
        client, _, _ = self._client([loop])
        with self.assertRaisesRegex(AudioDownloadSecurityError, "loop"):
            client.download(self.candidate_id, self.source_url, self.root / "loop")

        policy = DownloadPolicy(maximum_redirects=1)
        first = _FakeResponse(302, headers=_headers(Location="/one"))
        second = _FakeResponse(302, headers=_headers(Location="/two"))
        client, _, _ = self._client([first, second], policy)
        with self.assertRaisesRegex(AudioDownloadSecurityError, "redirect limit"):
            client.download(self.candidate_id, self.source_url, self.root / "limit")

    def test_terminal_and_command_http_statuses_are_distinct(self) -> None:
        for status in (404, 410):
            client, _, _ = self._client([_FakeResponse(status)])
            with self.subTest(status=status), self.assertRaises(TerminalAudioUnavailableError):
                client.download(self.candidate_id, self.source_url, self.root / f"status-{status}")
        for status in (401, 403):
            client, _, _ = self._client([_FakeResponse(status)])
            with self.subTest(status=status), self.assertRaises(AudioDownloadCommandError):
                client.download(self.candidate_id, self.source_url, self.root / f"status-{status}")
        client, _, _ = self._client([_FakeResponse(400)])
        with self.assertRaises(AudioDownloadSecurityError):
            client.download(self.candidate_id, self.source_url, self.root / "status-400")

    def test_transient_status_retries_and_reports_attempt_count(self) -> None:
        policy = DownloadPolicy(maximum_retries=1)
        retry = _FakeResponse(503)
        payload = b"RIFF\x04\x00\x00\x00WAVE"
        success = _FakeResponse(200, payload, _headers(Content_Length=str(len(payload))))
        client, opener, clock = self._client([retry, success], policy)

        receipt = client.download(self.candidate_id, self.source_url, self.root / "retried")

        self.assertEqual(receipt.attempts, 2)
        self.assertEqual(len(opener.requests), 2)
        self.assertIn(1.0, clock.sleeps)

    def test_retry_after_is_bounded(self) -> None:
        policy = DownloadPolicy(maximum_retries=1)
        retry = _FakeResponse(429, headers=_headers(Retry_After="1000"))
        payload = b"ID3audio"
        success = _FakeResponse(200, payload, _headers(Content_Length=str(len(payload))))
        client, _, clock = self._client([retry, success], policy)

        client.download(self.candidate_id, self.source_url, self.root / "retry-after")

        self.assertIn(60.0, clock.sleeps)

    def test_total_deadline_includes_retry_backoff_and_all_attempts(self) -> None:
        policy = DownloadPolicy(
            timeout_seconds=1.0,
            total_timeout_seconds=1.0,
            maximum_retries=3,
        )
        retry = _FakeResponse(503)
        client, opener, clock = self._client([retry], policy)

        with self.assertRaisesRegex(RetryableAudioDownloadError, "total download deadline"):
            client.download(self.candidate_id, self.source_url, self.root / "deadline-retry")

        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(clock.sleeps, [])

    def test_total_deadline_spans_redirects_and_streaming(self) -> None:
        policy = DownloadPolicy(
            timeout_seconds=1.0,
            total_timeout_seconds=1.0,
            maximum_retries=0,
        )
        redirect = _FakeResponse(302, headers=_headers(Location="/next"))
        client, opener, _ = self._client([redirect], policy)
        with self.assertRaises(RetryableAudioDownloadError):
            client.download(self.candidate_id, self.source_url, self.root / "deadline-redirect")
        self.assertEqual(len(opener.requests), 1)

        clock = _Clock()
        response = _AdvancingResponse(clock, 2.0, b"ID3audio")
        stream_opener = _FakeOpener([response])
        stream_client = SecureXenoCantoAudioClient(
            policy,
            opener=stream_opener,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        destination = self.root / "deadline-stream"
        with self.assertRaises(RetryableAudioDownloadError):
            stream_client.download(self.candidate_id, self.source_url, destination)
        self.assertFalse(destination.exists())

    def test_eof_read_that_crosses_deadline_is_retryable_and_cleans_file(self) -> None:
        policy = DownloadPolicy(
            timeout_seconds=1.0,
            total_timeout_seconds=1.0,
            maximum_retries=0,
        )
        clock = _Clock()
        response = _AdvancingResponse(clock, 0.6, b"ID3audio")
        opener = _FakeOpener([response])
        client = SecureXenoCantoAudioClient(
            policy,
            opener=opener,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        destination = self.root / "deadline-eof"

        with self.assertRaisesRegex(RetryableAudioDownloadError, "total download deadline"):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertFalse(destination.exists())
        self.assertTrue(response.closed)

    def test_response_returned_after_deadline_is_not_classified(self) -> None:
        policy = DownloadPolicy(
            timeout_seconds=1.0,
            total_timeout_seconds=1.0,
            maximum_retries=0,
        )
        for index, status in enumerate((200, 404)):
            with self.subTest(status=status):
                clock = _Clock()
                response = _FakeResponse(status, b"ID3audio")
                opener = _AdvancingOpener([response], clock, 2.0)
                client = SecureXenoCantoAudioClient(
                    policy,
                    opener=opener,
                    sleep=clock.sleep,
                    monotonic=clock.monotonic,
                )
                destination = self.root / f"deadline-open-{index}"

                with self.assertRaisesRegex(
                    RetryableAudioDownloadError,
                    "total download deadline",
                ):
                    client.download(self.candidate_id, self.source_url, destination)

                self.assertFalse(destination.exists())
                self.assertTrue(response.closed)

    def test_transport_error_is_secret_safe_and_unresolved(self) -> None:
        secret = "private-key-value"
        policy = DownloadPolicy(maximum_retries=0)
        client, _, _ = self._client([urllib.error.URLError(secret)], policy)
        destination = self.root / "transport"

        with self.assertRaises(RetryableAudioDownloadError) as raised:
            client.download(self.candidate_id, self.source_url, destination)

        self.assertNotIn(secret, str(raised.exception))
        self.assertFalse(destination.exists())

    def test_partial_attempt_is_removed_before_clean_retry(self) -> None:
        policy = DownloadPolicy(maximum_retries=1)
        short = _FakeResponse(200, b"ID3", _headers(Content_Length="8"))
        payload = b"ID3audio"
        complete = _FakeResponse(200, payload, _headers(Content_Length="8"))
        client, _, _ = self._client([short, complete], policy)
        destination = self.root / "clean-retry"

        receipt = client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(receipt.attempts, 2)
        self.assertEqual(destination.read_bytes(), payload)

    def test_failed_attempt_never_deletes_an_unowned_replacement(self) -> None:
        policy = DownloadPolicy(maximum_retries=1)
        response = _FakeResponse(200, b"ID3", _headers(Content_Length="5"))
        client, _, _ = self._client([response], policy)
        destination = self.root / "replacement-race"
        original_cleanup = secure_download._remove_created_destination
        replacement = b"owner replacement"

        def replace_after_owned_cleanup(
            path: Path,
            expected_stat: os.stat_result | None = None,
        ) -> None:
            original_cleanup(path, expected_stat)
            path.write_bytes(replacement)

        with (
            patch.object(
                secure_download,
                "_remove_created_destination",
                side_effect=replace_after_owned_cleanup,
            ),
            self.assertRaisesRegex(
                AudioDownloadSecurityError,
                "appeared after failed attempt",
            ),
        ):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(destination.read_bytes(), replacement)

    def test_fchmod_failure_never_deletes_an_unowned_replacement(self) -> None:
        payload = b"ID3audio"
        client, _, _ = self._client([_FakeResponse(200, payload)])
        destination = self.root / "fchmod-replacement-race"
        replacement = b"owner replacement"

        def replace_before_failure(_descriptor: int, _mode: int) -> None:
            destination.unlink()
            destination.write_bytes(replacement)
            raise OSError("simulated fchmod failure")

        with (
            patch.object(secure_download.os, "fchmod", side_effect=replace_before_failure),
            self.assertRaisesRegex(
                AudioDownloadSecurityError,
                "appeared after failed attempt",
            ),
        ):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(destination.read_bytes(), replacement)

    def test_hard_link_added_during_streaming_is_rejected(self) -> None:
        payload = b"ID3audio"
        destination = self.root / "hard-link-source"
        alias = self.root / "hard-link-alias"
        client, _, _ = self._client([_HardLinkingResponse(payload, destination, alias)])

        with self.assertRaisesRegex(
            AudioDownloadSecurityError,
            "download destination changed during streaming",
        ):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertTrue(alias.exists())
        self.assertFalse(destination.exists())

    def test_incomplete_body_remains_retryable_not_terminal(self) -> None:
        policy = DownloadPolicy(maximum_retries=0)
        response = _FakeResponse(200, b"ID3", _headers(Content_Length="8"))
        client, _, _ = self._client([response], policy)
        destination = self.root / "incomplete"

        with self.assertRaises(RetryableAudioDownloadError):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertFalse(destination.exists())

    def test_content_length_is_optional(self) -> None:
        payload = b"ID3audio-without-length"
        client, _, _ = self._client([_FakeResponse(200, payload)])

        receipt = client.download(self.candidate_id, self.source_url, self.root / "no-length")

        self.assertIsNone(receipt.content_length)
        self.assertEqual(receipt.bytes_written, len(payload))

    def test_locked_source_headers_are_accepted(self) -> None:
        payloads = (
            b"ID3audio",
            bytes((0xFF, 0xFB, 0x90, 0x64)),
            b"RIFF\x04\x00\x00\x00WAVE",
            b"RF64\x04\x00\x00\x00WAVE",
        )
        for index, payload in enumerate(payloads):
            client, _, _ = self._client([_FakeResponse(200, payload)])
            destination = self.root / f"header-{index}"
            with self.subTest(payload=payload):
                receipt = client.download(self.candidate_id, self.source_url, destination)
                self.assertEqual(receipt.bytes_written, len(payload))

    def test_response_protocol_violations_are_rejected_and_cleaned(self) -> None:
        duplicate = _headers(Content_Length="3")
        duplicate.add_header("Content-Length", "3")
        cases = (
            _FakeResponse(206, b"abc"),
            _FakeResponse(200, b"abc", duplicate),
            _FakeResponse(200, b"abc", _headers(Content_Length=" 3")),
            _FakeResponse(200, b"", _headers(Content_Length="0")),
            _FakeResponse(
                200,
                b"",
                _headers(Content_Length=str(512 * 1024 * 1024 + 1)),
            ),
            _FakeResponse(200, b"abc", _headers(Content_Range="bytes 0-2/3")),
            _FakeResponse(200, b"abc", _headers(Content_Encoding="gzip")),
            _FakeResponse(200, b"abc", _headers(Transfer_Encoding="gzip")),
            _FakeResponse(
                200,
                b"abc",
                _headers(Transfer_Encoding="chunked", Content_Length="3"),
            ),
            _FakeResponse(200, b"<html>error</html>", _headers(Content_Type="text/html")),
            _FakeResponse(200, b"<html>error</html>"),
            _FakeResponse(200, b""),
            _FakeResponse(200, chunks=["not bytes"]),
        )
        for index, response in enumerate(cases):
            client, _, _ = self._client([response])
            destination = self.root / f"protocol-{index}"
            with self.subTest(index=index), self.assertRaises(AudioDownloadSecurityError):
                client.download(self.candidate_id, self.source_url, destination)
            self.assertFalse(destination.exists())

    def test_streaming_limit_is_enforced_independently_of_content_length(self) -> None:
        policy = DownloadPolicy()
        object.__setattr__(policy, "maximum_file_bytes", 4)
        response = _FakeResponse(200, chunks=[b"123", b"45"])
        client, _, _ = self._client([response], policy)
        destination = self.root / "stream-limit"

        with self.assertRaisesRegex(AudioDownloadSecurityError, "file-size limit"):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertFalse(destination.exists())

    def test_existing_destination_is_never_replaced_or_deleted(self) -> None:
        destination = self.root / "existing"
        destination.write_bytes(b"owner-data")
        client, opener, _ = self._client([])

        with self.assertRaisesRegex(AudioDownloadSecurityError, "already exists"):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertEqual(destination.read_bytes(), b"owner-data")
        self.assertFalse(opener.requests)

    def test_symlink_destination_is_never_followed_or_deleted(self) -> None:
        target = self.root / "target"
        target.write_bytes(b"owner-data")
        destination = self.root / "link"
        destination.symlink_to(target)
        client, opener, _ = self._client([])

        with self.assertRaises(AudioDownloadSecurityError):
            client.download(self.candidate_id, self.source_url, destination)

        self.assertTrue(destination.is_symlink())
        self.assertEqual(target.read_bytes(), b"owner-data")
        self.assertFalse(opener.requests)

    def test_missing_or_symlinked_destination_parent_is_rejected(self) -> None:
        client, opener, _ = self._client([])
        with self.assertRaises(AudioDownloadSecurityError):
            client.download(
                self.candidate_id,
                self.source_url,
                self.root / "missing" / "audio",
            )
        real = self.root / "real"
        real.mkdir()
        linked = self.root / "linked"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaises(AudioDownloadSecurityError):
            client.download(self.candidate_id, self.source_url, linked / "audio")
        self.assertFalse(opener.requests)

    def test_http_error_objects_are_classified_without_reading_error_body(self) -> None:
        secret = b"private error response"
        error = urllib.error.HTTPError(
            self.source_url,
            404,
            "not found",
            Message(),
            io.BytesIO(secret),
        )
        client, _, _ = self._client([error])

        with self.assertRaises(TerminalAudioUnavailableError) as raised:
            client.download(self.candidate_id, self.source_url, self.root / "http-error")

        self.assertNotIn(secret.decode(), str(raised.exception))

    def test_default_opener_has_empty_proxy_map_and_no_cookie_handler(self) -> None:
        client = SecureXenoCantoAudioClient()
        handlers = client._opener.handlers  # type: ignore[attr-defined]
        proxy_handlers = [
            handler for handler in handlers if isinstance(handler, urllib.request.ProxyHandler)
        ]
        cookie_handlers = [
            handler
            for handler in handlers
            if isinstance(handler, urllib.request.HTTPCookieProcessor)
        ]
        self.assertTrue(all(handler.proxies == {} for handler in proxy_handlers))
        self.assertEqual(cookie_handlers, [])


if __name__ == "__main__":
    unittest.main()
