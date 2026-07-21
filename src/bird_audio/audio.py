from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from array import array
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class AudioToolError(RuntimeError):
    pass


_AUDIO_API_KEY_ENVIRONMENT = "XENO_CANTO_API_KEY"
_LOCAL_INPUT_PROTOCOLS = "file"
_LOCAL_DECODE_PROTOCOLS = "file,pipe"


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop(_AUDIO_API_KEY_ENVIRONMENT, None)
    return environment


def _tool_supports_nostdin(tool: Path) -> bool:
    return tool.name.casefold().startswith("ffmpeg")


def _secret_variants(secret: str) -> set[str]:
    if not secret:
        return set()
    quoted_variants = {
        urllib.parse.quote(secret, safe=""),
        urllib.parse.quote_plus(secret, safe=""),
    }
    encoded_variants = {
        value
        for quoted in quoted_variants
        for value in (quoted, urllib.parse.quote(quoted, safe=""))
    }
    variants = {
        secret,
        *encoded_variants,
    }
    variants.update(
        re.sub(r"%[0-9A-F]{2}", lambda match: match.group(0).lower(), value)
        for value in encoded_variants
    )
    return variants


def _redact_secret(value: str) -> str:
    secret = os.environ.get(_AUDIO_API_KEY_ENVIRONMENT, "")
    redacted = value
    for variant in sorted(_secret_variants(secret), key=len, reverse=True):
        redacted = redacted.replace(variant, "[REDACTED]")
    return redacted


def _safe_audio_diagnostic(value: str, path: Path, maximum_length: int = 1000) -> str:
    normalized = normalize_ffmpeg_diagnostic(value)
    path_variants: set[str] = {str(path), path.as_posix(), path.name}
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        resolved = path.expanduser().absolute()
    path_variants.update({str(resolved), resolved.as_posix()})
    with suppress(ValueError):
        path_variants.add(resolved.as_uri())
    path_variants.discard("")
    encoded = {
        variant
        for raw in path_variants
        if raw
        for variant in (
            urllib.parse.quote(raw, safe=""),
            urllib.parse.quote_plus(raw, safe=""),
        )
    }
    for variant in sorted(path_variants | encoded, key=len, reverse=True):
        normalized = normalized.replace(variant, "[INPUT]")
    return normalized[:maximum_length]


@dataclass(frozen=True)
class AudioProbe:
    probe_ok: bool
    format_name: str = ""
    codec_name: str = ""
    codec_long_name: str = ""
    source_sample_rate_hz: int = 0
    channels: int = 0
    channel_layout: str = ""
    sample_format: str = ""
    bits_per_sample: int = 0
    bits_per_raw_sample: int = 0
    ffprobe_duration_seconds: float = 0.0
    bit_rate_bps: int = 0
    probe_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodeSmokeResult:
    sample_count: int
    sample_rate_hz: int
    channels: int
    duration_seconds: float
    finite: bool
    minimum: float
    maximum: float
    rms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullDecodeResult:
    decoded_duration_seconds: float
    diagnostic: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_tool(name: str, explicit: str | Path | None = None) -> Path:
    """Resolve FFmpeg tooling from an argument, environment variable, or PATH."""
    environment_name = f"BIRD_AUDIO_{name.upper()}"
    candidate = str(explicit) if explicit else os.environ.get(environment_name, "")
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"Configured {name} is not executable: {path}")

    discovered = shutil.which(name)
    if discovered:
        return Path(discovered).resolve()

    raise FileNotFoundError(f"{name} was not found. Install FFmpeg or set {environment_name}.")


def tool_version(tool: Path) -> str:
    command = [str(tool)]
    if _tool_supports_nostdin(tool):
        command.append("-nostdin")
    command.append("-version")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    output = completed.stdout.strip() or completed.stderr.strip()
    first_line = output.splitlines()[0] if output else ""
    return _redact_secret(first_line)[:500] if first_line else "unavailable"


def detect_header(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(16)

    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "riff_wave"
    if header.startswith(b"RF64") and header[8:12] == b"WAVE":
        return "rf64_wave"
    if header.startswith(b"ID3"):
        return "mp3_id3"
    if len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0:
        return "mpeg_audio"
    if header.startswith(b"fLaC"):
        return "flac"
    if header.startswith(b"OggS"):
        return "ogg"
    return "unknown"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def normalize_ffmpeg_diagnostic(value: str) -> str:
    """Remove process-specific addresses and normalize whitespace for stable hashes."""
    normalized = _redact_secret(value)
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def probe_audio(path: Path, ffprobe: Path, timeout_seconds: int = 30) -> AudioProbe:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-protocol_whitelist",
        _LOCAL_INPUT_PROTOCOLS,
        "-select_streams",
        "a:0",
        "-show_entries",
        (
            "format=format_name,duration,bit_rate:"
            "stream=codec_name,codec_long_name,sample_rate,channels,channel_layout,"
            "sample_fmt,bits_per_sample,bits_per_raw_sample,duration,bit_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            env=_subprocess_environment(),
        )
    except subprocess.TimeoutExpired:
        return AudioProbe(probe_ok=False, probe_error="ffprobe invocation timed out")
    except OSError:
        return AudioProbe(probe_ok=False, probe_error="ffprobe invocation failed")

    if completed.returncode != 0:
        error = _safe_audio_diagnostic(completed.stderr, path)
        if not error:
            error = f"ffprobe exited with status {completed.returncode}"
        return AudioProbe(probe_ok=False, probe_error=error)

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return AudioProbe(probe_ok=False, probe_error="Invalid ffprobe JSON")

    streams = payload.get("streams") or []
    if not streams:
        return AudioProbe(probe_ok=False, probe_error="No audio stream found")

    stream = streams[0]
    audio_format = payload.get("format") or {}
    duration = _as_float(stream.get("duration")) or _as_float(audio_format.get("duration"))
    bit_rate = _as_int(stream.get("bit_rate")) or _as_int(audio_format.get("bit_rate"))
    return AudioProbe(
        probe_ok=True,
        format_name=str(audio_format.get("format_name") or ""),
        codec_name=str(stream.get("codec_name") or ""),
        codec_long_name=str(stream.get("codec_long_name") or ""),
        source_sample_rate_hz=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        channel_layout=str(stream.get("channel_layout") or ""),
        sample_format=str(stream.get("sample_fmt") or ""),
        bits_per_sample=_as_int(stream.get("bits_per_sample")),
        bits_per_raw_sample=_as_int(stream.get("bits_per_raw_sample")),
        ffprobe_duration_seconds=duration,
        bit_rate_bps=bit_rate,
    )


def decode_smoke_test(
    path: Path,
    ffmpeg: Path,
    seconds: float = 3.0,
    sample_rate_hz: int = 32000,
) -> DecodeSmokeResult:
    """Decode a short canonical mono float32 segment and inspect its samples."""
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        _LOCAL_DECODE_PROTOCOLS,
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=max(30, int(seconds * 10)),
            stdin=subprocess.DEVNULL,
            env=_subprocess_environment(),
        )
    except subprocess.TimeoutExpired:
        raise AudioToolError("FFmpeg decode timed out") from None
    except OSError:
        raise AudioToolError("FFmpeg decode invocation failed") from None
    if completed.returncode != 0:
        error = _safe_audio_diagnostic(completed.stderr.decode("utf-8", errors="replace"), path)
        if not error:
            error = f"FFmpeg exited with status {completed.returncode}"
        raise AudioToolError(f"FFmpeg decode failed: {error}")
    if len(completed.stdout) % 4:
        raise AudioToolError("Unexpected float32 byte count")

    samples = array("f")
    samples.frombytes(completed.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise AudioToolError("FFmpeg produced no audio samples")

    finite = all(math.isfinite(value) for value in samples)
    if not finite:
        raise AudioToolError("Non-finite decoded samples")
    sum_squares = math.fsum(float(value) * float(value) for value in samples)
    return DecodeSmokeResult(
        sample_count=len(samples),
        sample_rate_hz=sample_rate_hz,
        channels=1,
        duration_seconds=len(samples) / sample_rate_hz,
        finite=True,
        minimum=min(samples),
        maximum=max(samples),
        rms=math.sqrt(sum_squares / len(samples)),
    )


def verify_full_decode(
    path: Path,
    ffmpeg: Path,
    timeout_seconds: int = 3600,
) -> FullDecodeResult:
    """Decode a complete file to a null sink and return duration plus diagnostics."""
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        _LOCAL_DECODE_PROTOCOLS,
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
        "-progress",
        "pipe:1",
        "-nostats",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            env=_subprocess_environment(),
        )
    except subprocess.TimeoutExpired:
        raise AudioToolError("Full decode timed out") from None
    except OSError:
        raise AudioToolError("Full decode invocation failed") from None
    diagnostics = _safe_audio_diagnostic(completed.stderr, path)
    if completed.returncode != 0:
        if not diagnostics:
            diagnostics = f"FFmpeg exited with status {completed.returncode}"
        raise AudioToolError(f"Full decode failed: {diagnostics}")
    progress: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    decoded_duration = _as_float(progress.get("out_time_us")) / 1_000_000
    return FullDecodeResult(
        decoded_duration_seconds=decoded_duration,
        diagnostic=diagnostics,
    )
