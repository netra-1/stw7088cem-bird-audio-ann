from __future__ import annotations

import email.utils
import hashlib
import math
import os
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol

DEFAULT_ALLOWED_HOSTS = ("xeno-canto.org",)
DEFAULT_USER_AGENT = "STW7088CEM-bird-audio-coursework/0.1"
DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024
DEFAULT_MAXIMUM_FILE_BYTES = 512 * 1024 * 1024
DEFAULT_MAXIMUM_RETRY_AFTER_SECONDS = 60.0

_CANDIDATE_ID = re.compile(r"^XC([1-9][0-9]*)$")
_HOST = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
_CONTENT_LENGTH = re.compile(r"^[0-9]+$")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_RETRYABLE_STATUSES = frozenset({408, 429})
_OBVIOUS_NON_AUDIO_TYPES = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/problem+json",
        "application/xhtml+xml",
        "application/xml",
        "text/html",
        "text/javascript",
        "text/json",
        "text/plain",
        "text/xml",
    }
)
_FIXED_HEADERS = {
    "Accept": (
        "audio/mpeg, audio/mp3, audio/wav, audio/x-wav, audio/vnd.wave, application/octet-stream"
    ),
    "Accept-Encoding": "identity",
    "Connection": "close",
}


class SecureAudioDownloadError(RuntimeError):
    """Base class for safe audio download failures."""


class TerminalAudioUnavailableError(SecureAudioDownloadError):
    """The source recording is terminally absent and may be audited as unavailable."""


class RetryableAudioDownloadError(SecureAudioDownloadError):
    """The download remains unresolved after bounded transient retries."""


class AudioDownloadSecurityError(SecureAudioDownloadError):
    """The request or response violated the locked security protocol."""


class AudioDownloadCommandError(AudioDownloadSecurityError):
    """The remote service rejected the command in a way that must stop the run."""


@dataclass(frozen=True)
class DownloadPolicy:
    """Strict transport policy for public Xeno-canto audio files."""

    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    maximum_redirects: int = 3
    request_interval_seconds: float = 1.0
    timeout_seconds: float = 60.0
    total_timeout_seconds: float = 900.0
    maximum_retries: int = 3
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES
    maximum_file_bytes: int = DEFAULT_MAXIMUM_FILE_BYTES
    maximum_retry_after_seconds: float = DEFAULT_MAXIMUM_RETRY_AFTER_SECONDS
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_hosts, tuple) or not self.allowed_hosts:
            raise ValueError("allowed_hosts must be a non-empty tuple")
        if len(set(self.allowed_hosts)) != len(self.allowed_hosts):
            raise ValueError("allowed_hosts must not contain duplicates")
        for host in self.allowed_hosts:
            if (
                not isinstance(host, str)
                or not _HOST.fullmatch(host)
                or host != host.casefold()
                or host.startswith(".")
                or host.endswith(".")
            ):
                raise ValueError("allowed_hosts must contain exact lowercase DNS hosts")
        if "xeno-canto.org" not in self.allowed_hosts:
            raise ValueError("allowed_hosts must include xeno-canto.org")
        _bounded_integer(self.maximum_redirects, "maximum_redirects", minimum=0, maximum=5)
        _positive_finite(
            self.request_interval_seconds,
            "request_interval_seconds",
            minimum=1.0,
            maximum=60.0,
        )
        _positive_finite(
            self.timeout_seconds,
            "timeout_seconds",
            minimum=1.0,
            maximum=300.0,
        )
        _positive_finite(
            self.total_timeout_seconds,
            "total_timeout_seconds",
            minimum=self.timeout_seconds,
            maximum=3600.0,
        )
        _bounded_integer(self.maximum_retries, "maximum_retries", minimum=0, maximum=10)
        if self.chunk_size_bytes != DEFAULT_CHUNK_SIZE_BYTES:
            raise ValueError("chunk_size_bytes must be exactly 1048576")
        if self.maximum_file_bytes != DEFAULT_MAXIMUM_FILE_BYTES:
            raise ValueError("maximum_file_bytes must be exactly 536870912")
        _positive_finite(
            self.maximum_retry_after_seconds,
            "maximum_retry_after_seconds",
            minimum=1.0,
            maximum=DEFAULT_MAXIMUM_RETRY_AFTER_SECONDS,
        )
        if self.user_agent != DEFAULT_USER_AGENT:
            raise ValueError("user_agent is not the locked downloader identity")


@dataclass(frozen=True)
class DownloadReceipt:
    candidate_id: str
    source_url: str
    final_url: str
    destination: str
    sha256: str
    bytes_written: int
    content_length: int | None
    content_type: str
    redirect_count: int
    attempts: int


class _Response(Protocol):
    status: int
    headers: Any

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class _Opener(Protocol):
    def open(self, request: urllib.request.Request, timeout: float) -> _Response: ...


class _NoAutomaticRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        _request: urllib.request.Request,
        _file_pointer: BinaryIO,
        _code: int,
        _message: str,
        _headers: Any,
        _new_url: str,
    ) -> urllib.request.Request | None:
        return None


@dataclass(frozen=True)
class _RetrySignal(Exception):
    delay_seconds: float | None = None


def _bounded_integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _positive_finite(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return number


def _default_opener() -> _Opener:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoAutomaticRedirects(),
    )


def _header_values(headers: Any, name: str) -> list[str]:
    if headers is None:
        return []
    getter = getattr(headers, "get_all", None)
    if callable(getter):
        values = getter(name)
        if values is not None:
            return [str(value) for value in values]
    if isinstance(headers, Mapping):
        values: list[str] = []
        for key, value in headers.items():
            if str(key).casefold() != name.casefold():
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                values.extend(str(item) for item in value)
            else:
                values.append(str(value))
        return values
    value = getattr(headers, "get", lambda _name: None)(name)
    return [] if value is None else [str(value)]


def _single_header(headers: Any, name: str) -> str | None:
    values = _header_values(headers, name)
    if len(values) > 1:
        raise AudioDownloadSecurityError(f"response has multiple {name} headers")
    return values[0] if values else None


def _strict_content_length(headers: Any, maximum_file_bytes: int) -> int | None:
    value = _single_header(headers, "Content-Length")
    if value is None:
        return None
    if value != value.strip() or not _CONTENT_LENGTH.fullmatch(value):
        raise AudioDownloadSecurityError("response Content-Length is invalid")
    length = int(value)
    if length <= 0:
        raise AudioDownloadSecurityError("response Content-Length must be positive")
    if length > maximum_file_bytes:
        raise AudioDownloadSecurityError("response Content-Length exceeds the file-size limit")
    return length


def _validated_content_type(headers: Any) -> str:
    value = _single_header(headers, "Content-Type")
    if value is None:
        return ""
    content_type = value.split(";", 1)[0].strip().casefold()
    if not content_type or any(ord(character) < 32 for character in content_type):
        raise AudioDownloadSecurityError("response Content-Type is invalid")
    if content_type.startswith("text/") or content_type in _OBVIOUS_NON_AUDIO_TYPES:
        raise AudioDownloadSecurityError("response Content-Type is not an audio download")
    if "html" in content_type or "json" in content_type or "javascript" in content_type:
        raise AudioDownloadSecurityError("response Content-Type is not an audio download")
    return content_type


def _body_looks_like_text_document(value: bytes) -> bool:
    prefix = value[:512].lstrip().lower()
    return prefix.startswith(
        (
            b"<!doctype html",
            b"<html",
            b"<?xml",
            b"<script",
            b'{"error"',
            b'{"message"',
            b'[{"',
        )
    )


def _accepted_audio_header(value: bytes) -> bool:
    if value.startswith(b"ID3"):
        return True
    if len(value) >= 2 and value[0] == 0xFF and value[1] & 0xE0 == 0xE0:
        return True
    return len(value) >= 12 and value[:4] in (b"RIFF", b"RF64") and value[8:12] == b"WAVE"


def _validate_transfer_headers(headers: Any) -> None:
    if _header_values(headers, "Content-Range"):
        raise AudioDownloadSecurityError("partial-content responses are not permitted")
    encoding = _single_header(headers, "Content-Encoding")
    if encoding is not None and encoding.strip().casefold() != "identity":
        raise AudioDownloadSecurityError("compressed transport encoding is not permitted")
    transfer_encoding = _single_header(headers, "Transfer-Encoding")
    content_length = _header_values(headers, "Content-Length")
    if transfer_encoding is not None:
        if transfer_encoding.strip().casefold() != "chunked":
            raise AudioDownloadSecurityError("response Transfer-Encoding is invalid")
        if content_length:
            raise AudioDownloadSecurityError(
                "response cannot combine Transfer-Encoding with Content-Length"
            )


def _safe_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if callable(getcode) else None
    if isinstance(status, bool) or not isinstance(status, int):
        raise AudioDownloadSecurityError("response status is invalid")
    return status


def _validate_redirect_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    if not isinstance(url, str) or not url or url != url.strip():
        raise AudioDownloadSecurityError("redirect target is invalid")
    if "%" in url or "\\" in url or any(ord(character) < 32 for character in url):
        raise AudioDownloadSecurityError("redirect target is invalid")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError:
        raise AudioDownloadSecurityError("redirect target is invalid") from None
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.netloc != parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise AudioDownloadSecurityError("redirect target is outside the approved HTTPS hosts")
    return url


def _source_identity(candidate_id: str, source_url: str) -> tuple[str, str]:
    if not isinstance(candidate_id, str):
        raise AudioDownloadSecurityError("candidate ID is invalid")
    match = _CANDIDATE_ID.fullmatch(candidate_id)
    if match is None:
        raise AudioDownloadSecurityError("candidate ID is invalid")
    expected_url = f"https://xeno-canto.org/{match.group(1)}/download"
    if not isinstance(source_url, str) or source_url != expected_url:
        raise AudioDownloadSecurityError("source URL does not match the canonical candidate URL")
    return candidate_id, expected_url


def _retry_after(headers: Any, maximum_seconds: float) -> float | None:
    value = _single_header(headers, "Retry-After")
    if value is None:
        return None
    text = value.strip()
    try:
        delay = float(text)
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(text)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(delay):
        return None
    return min(max(delay, 0.0), maximum_seconds)


def _remove_created_destination(path: Path, expected_stat: os.stat_result | None = None) -> None:
    if expected_stat is None:
        raise AudioDownloadSecurityError(
            "download destination ownership could not be established for cleanup"
        )
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise AudioDownloadSecurityError("download destination cleanup failed") from None
    if observed.st_dev != expected_stat.st_dev or observed.st_ino != expected_stat.st_ino:
        raise AudioDownloadSecurityError("destination changed during failed download")
    try:
        path.unlink()
    except OSError:
        raise AudioDownloadSecurityError("download destination cleanup failed") from None


class SecureXenoCantoAudioClient:
    """Download public Xeno-canto audio under a bounded, fail-closed policy."""

    def __init__(
        self,
        policy: DownloadPolicy | None = None,
        *,
        opener: _Opener | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or DownloadPolicy()
        if not isinstance(self.policy, DownloadPolicy):
            raise TypeError("policy must be a DownloadPolicy")
        if opener is not None and not hasattr(opener, "open"):
            raise TypeError("opener must provide an open method")
        if not callable(sleep) or not callable(monotonic):
            raise TypeError("sleep and monotonic must be callable")
        self._opener = opener or _default_opener()
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_started: float | None = None

    def _remaining_seconds(self, started_at: float) -> float:
        return self.policy.total_timeout_seconds - (self._monotonic() - started_at)

    def _pace(self, started_at: float) -> None:
        now = self._monotonic()
        if self._last_request_started is not None:
            remaining = self.policy.request_interval_seconds - (now - self._last_request_started)
            if remaining > 0:
                if remaining >= self._remaining_seconds(started_at):
                    raise _RetrySignal()
                self._sleep(remaining)
                self._check_deadline(started_at)
        self._last_request_started = self._monotonic()

    def _open(self, url: str, started_at: float) -> _Response:
        self._pace(started_at)
        remaining = self._remaining_seconds(started_at)
        if remaining <= 0:
            raise _RetrySignal()
        headers = {**_FIXED_HEADERS, "User-Agent": self.policy.user_agent}
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            response = self._opener.open(
                request,
                timeout=min(self.policy.timeout_seconds, remaining),
            )
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            raise _RetrySignal() from None
        except Exception:
            raise AudioDownloadSecurityError("HTTP opener failed unexpectedly") from None
        try:
            self._check_deadline(started_at)
        except _RetrySignal:
            with suppress(Exception):
                response.close()
            raise
        return response

    def _check_deadline(self, started_at: float) -> None:
        if self._remaining_seconds(started_at) <= 0:
            raise _RetrySignal()

    def _stream_response(
        self,
        response: _Response,
        destination: Path,
        started_at: float,
        content_length: int | None,
    ) -> tuple[str, int]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(destination, flags, 0o600)
        except FileExistsError:
            raise AudioDownloadSecurityError("download destination already exists") from None
        except OSError:
            raise AudioDownloadSecurityError("download destination could not be created") from None

        descriptor_stat: os.stat_result | None = None
        digest = hashlib.sha256()
        bytes_written = 0
        header_prefix = bytearray()
        header_accepted = False
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
                raise AudioDownloadSecurityError("download destination is not a private file")
            os.fchmod(descriptor, 0o600)
            private_stat = os.fstat(descriptor)
            if (
                descriptor_stat.st_dev != private_stat.st_dev
                or descriptor_stat.st_ino != private_stat.st_ino
                or not stat.S_ISREG(private_stat.st_mode)
                or private_stat.st_nlink != 1
                or stat.S_IMODE(private_stat.st_mode) != 0o600
            ):
                raise AudioDownloadSecurityError("download destination is not a private file")
            descriptor_stat = private_stat
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                while True:
                    self._check_deadline(started_at)
                    try:
                        chunk = response.read(self.policy.chunk_size_bytes)
                    except (TimeoutError, ConnectionError, OSError):
                        raise _RetrySignal() from None
                    except Exception:
                        raise _RetrySignal() from None
                    self._check_deadline(started_at)
                    if not isinstance(chunk, bytes):
                        raise AudioDownloadSecurityError("response body yielded non-byte data")
                    if not chunk:
                        break
                    if bytes_written == 0 and _body_looks_like_text_document(chunk):
                        raise AudioDownloadSecurityError(
                            "response body is an obvious text document"
                        )
                    if not header_accepted:
                        needed = 16 - len(header_prefix)
                        if needed > 0:
                            header_prefix.extend(chunk[:needed])
                        header_accepted = _accepted_audio_header(header_prefix)
                        if len(header_prefix) >= 16 and not header_accepted:
                            raise AudioDownloadSecurityError(
                                "response body has no accepted audio header"
                            )
                    bytes_written += len(chunk)
                    if bytes_written > self.policy.maximum_file_bytes:
                        raise AudioDownloadSecurityError(
                            "response body exceeds the file-size limit"
                        )
                    if content_length is not None and bytes_written > content_length:
                        raise AudioDownloadSecurityError(
                            "response body exceeds its declared Content-Length"
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                if bytes_written == 0:
                    raise AudioDownloadSecurityError("response body is empty")
                if not header_accepted:
                    raise AudioDownloadSecurityError("response body has no accepted audio header")
                if content_length is not None and bytes_written != content_length:
                    raise _RetrySignal()
                self._check_deadline(started_at)
                handle.flush()
                os.fsync(handle.fileno())
                final_stat = os.fstat(handle.fileno())
                if (
                    descriptor_stat.st_dev != final_stat.st_dev
                    or descriptor_stat.st_ino != final_stat.st_ino
                    or final_stat.st_size != bytes_written
                    or stat.S_IMODE(final_stat.st_mode) != 0o600
                    or not stat.S_ISREG(final_stat.st_mode)
                    or final_stat.st_nlink != 1
                ):
                    raise AudioDownloadSecurityError(
                        "download destination changed during streaming"
                    )
            observed = destination.lstat()
            if (
                descriptor_stat.st_dev != observed.st_dev
                or descriptor_stat.st_ino != observed.st_ino
                or observed.st_size != bytes_written
                or stat.S_IMODE(observed.st_mode) != 0o600
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
            ):
                raise AudioDownloadSecurityError("download destination changed after streaming")
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            try:
                directory_descriptor = os.open(destination.parent, directory_flags)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                raise AudioDownloadSecurityError(
                    "download destination directory could not be synchronized"
                ) from None
            durable = destination.lstat()
            if (
                descriptor_stat.st_dev != durable.st_dev
                or descriptor_stat.st_ino != durable.st_ino
                or durable.st_size != bytes_written
                or stat.S_IMODE(durable.st_mode) != 0o600
                or not stat.S_ISREG(durable.st_mode)
                or durable.st_nlink != 1
            ):
                raise AudioDownloadSecurityError(
                    "download destination changed during directory synchronization"
                )
            self._check_deadline(started_at)
            return digest.hexdigest(), bytes_written
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            _remove_created_destination(destination, descriptor_stat)
            raise

    def _attempt(
        self,
        candidate_id: str,
        source_url: str,
        destination: Path,
        started_at: float,
    ) -> DownloadReceipt:
        current_url = source_url
        visited = {source_url}
        redirects = 0
        while True:
            self._check_deadline(started_at)
            response = self._open(current_url, started_at)
            try:
                status = _safe_status(response)
                headers = getattr(response, "headers", None)
                if status in _REDIRECT_STATUSES:
                    if redirects >= self.policy.maximum_redirects:
                        raise AudioDownloadSecurityError("response exceeded the redirect limit")
                    location = _single_header(headers, "Location")
                    if location is None or not location or location != location.strip():
                        raise AudioDownloadSecurityError("redirect response has no valid Location")
                    target = urllib.parse.urljoin(current_url, location)
                    target = _validate_redirect_url(target, self.policy.allowed_hosts)
                    if target in visited:
                        raise AudioDownloadSecurityError("response contains a redirect loop")
                    visited.add(target)
                    redirects += 1
                    current_url = target
                    continue
                if status in _RETRYABLE_STATUSES or 500 <= status <= 599:
                    raise _RetrySignal(
                        _retry_after(headers, self.policy.maximum_retry_after_seconds)
                    )
                if status in {404, 410}:
                    raise TerminalAudioUnavailableError(
                        f"{candidate_id} is unavailable with HTTP status {status}"
                    )
                if status in {401, 403}:
                    raise AudioDownloadCommandError(
                        f"audio service rejected the command with HTTP status {status}"
                    )
                if status != 200:
                    raise AudioDownloadSecurityError(
                        f"response HTTP status {status} is not permitted"
                    )
                _validate_transfer_headers(headers)
                content_length = _strict_content_length(headers, self.policy.maximum_file_bytes)
                content_type = _validated_content_type(headers)
                sha256, bytes_written = self._stream_response(
                    response,
                    destination,
                    started_at,
                    content_length,
                )
                return DownloadReceipt(
                    candidate_id=candidate_id,
                    source_url=source_url,
                    final_url=current_url,
                    destination=str(destination),
                    sha256=sha256,
                    bytes_written=bytes_written,
                    content_length=content_length,
                    content_type=content_type,
                    redirect_count=redirects,
                    attempts=1,
                )
            finally:
                with suppress(Exception):
                    response.close()

    def download(
        self,
        candidate_id: str,
        source_url: str,
        destination: str | Path,
    ) -> DownloadReceipt:
        """Download one canonical recording into a new private file."""
        canonical_id, canonical_url = _source_identity(candidate_id, source_url)
        try:
            destination_path = Path(destination).expanduser()
            if "\x00" in os.fspath(destination_path):
                raise ValueError
            destination_exists = os.path.lexists(destination_path)
        except (TypeError, ValueError, OSError):
            raise AudioDownloadSecurityError("download destination is invalid") from None
        if destination_exists:
            raise AudioDownloadSecurityError("download destination already exists")
        try:
            parent = destination_path.parent.resolve(strict=True)
        except (FileNotFoundError, OSError):
            raise AudioDownloadSecurityError("download destination parent is invalid") from None
        if not parent.is_dir() or destination_path.parent.is_symlink():
            raise AudioDownloadSecurityError("download destination parent is invalid")
        destination_path = parent / destination_path.name
        if not destination_path.name or destination_path.name in {".", ".."}:
            raise AudioDownloadSecurityError("download destination filename is invalid")

        started_at = self._monotonic()
        for attempt in range(1, self.policy.maximum_retries + 2):
            try:
                self._check_deadline(started_at)
            except _RetrySignal:
                raise RetryableAudioDownloadError(
                    f"{canonical_id} exceeded the total download deadline"
                ) from None
            if os.path.lexists(destination_path):
                raise AudioDownloadSecurityError("download destination appeared before request")
            try:
                receipt = self._attempt(
                    canonical_id,
                    canonical_url,
                    destination_path,
                    started_at,
                )
                return replace(receipt, attempts=attempt)
            except _RetrySignal as signal:
                if os.path.lexists(destination_path):
                    raise AudioDownloadSecurityError(
                        "download destination appeared after failed attempt"
                    ) from None
                if self._remaining_seconds(started_at) <= 0:
                    raise RetryableAudioDownloadError(
                        f"{canonical_id} exceeded the total download deadline"
                    ) from None
                if attempt > self.policy.maximum_retries:
                    raise RetryableAudioDownloadError(
                        f"{canonical_id} remains unresolved after {attempt} attempts"
                    ) from None
                delay = signal.delay_seconds
                if delay is None:
                    delay = min(float(2 ** (attempt - 1)), 16.0)
                if delay >= self._remaining_seconds(started_at):
                    raise RetryableAudioDownloadError(
                        f"{canonical_id} exceeded the total download deadline"
                    ) from None
                self._sleep(delay)
                try:
                    self._check_deadline(started_at)
                except _RetrySignal:
                    raise RetryableAudioDownloadError(
                        f"{canonical_id} exceeded the total download deadline"
                    ) from None
            except (TerminalAudioUnavailableError, AudioDownloadSecurityError):
                if os.path.lexists(destination_path):
                    raise AudioDownloadSecurityError(
                        "download destination appeared after failed attempt"
                    ) from None
                raise
            except Exception:
                if os.path.lexists(destination_path):
                    raise AudioDownloadSecurityError(
                        "download destination appeared after failed attempt"
                    ) from None
                raise AudioDownloadSecurityError("download failed unexpectedly") from None
