from __future__ import annotations

import csv
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from bird_audio.audio import AudioToolError, normalize_ffmpeg_diagnostic, resolve_tool
from bird_audio.clip_selection import (
    ENERGY_CANDIDATE_HOP_SAMPLES,
    MAXIMUM_CLIPS_PER_RECORDING,
    MINIMUM_SELECTED_START_SEPARATION_SAMPLES,
    energy_candidate_starts,
    select_energy_candidates,
)
from bird_audio.config import validate_config
from bird_audio.hashing import sha256_json
from bird_audio.io_utils import atomic_write_csv, atomic_write_json
from bird_audio.paths import (
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    is_relative_to,
    require_safe_output,
    resolve_project_path,
)
from bird_audio.signal import (
    CLIP_DURATION_SECONDS,
    CLIP_SAMPLES,
    F_MAX_HZ,
    F_MIN_HZ,
    HOP_LENGTH,
    MAXIMUM_DB,
    MINIMUM_DB,
    N_FFT,
    N_MELS,
    NATIVE_MEL_HEIGHT,
    NATIVE_MEL_WIDTH,
    POWER_TO_DB_AMIN,
    TARGET_SAMPLE_RATE_HZ,
    WIN_LENGTH,
    iter_extracted_clips,
    native_log_mel_spectrogram,
)
from bird_audio.unknown_audio import (
    load_unknown_audio_config,
    verify_unknown_audio_audit,
)

CACHE_SCHEMA_VERSION = "1.0"
CACHE_VERSION = "unknown_clips_v2"
RESUME_SCHEMA_VERSION = "1.0"
DEFAULT_UNKNOWN_CLIP_CACHE_ROOT = "data/processed/unknown_clips_v2"
PRESERVED_UNKNOWN_CLIP_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "unknown_clips_v1"
PRESERVED_V1_ROOTS = (
    PROJECT_ROOT / "runs" / "final_evaluation",
    PROJECT_ROOT / "runs" / "task1",
    PROJECT_ROOT / "runs" / "task2",
    PROJECT_ROOT / "data" / "processed" / "known_clips_v1",
    PRESERVED_UNKNOWN_CLIP_CACHE_ROOT,
    PROJECT_ROOT / "report_assets" / "provenance",
    PROJECT_ROOT / "data" / "unknown" / "audio",
    PROJECT_ROOT / "data" / "unknown" / "interim" / "audio_acquisition_v1" / "checkpoints",
)
DEFAULT_AUDIT = "data/unknown/audio/unknown_audio_audit_v1.json"
DEFAULT_AUDIT_LOCK = "data/unknown/audio/unknown_audio_audit_v1_lock.json"
DEFAULT_CHECKPOINT_ROOT = "data/unknown/interim/audio_acquisition_v1/checkpoints"
DEFAULT_DATA_CONFIG = "configs/data.toml"
DEFAULT_UNKNOWN_AUDIO_CONFIG = "configs/unknown_audio.toml"
SCORING_PARTITION = "scoring"
NATIVE_FEATURE_SHAPE = (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH)
TARGET_SPECIES = 5
TARGET_RECORDINGS_PER_SPECIES = 40
TARGET_RECORDINGS = TARGET_SPECIES * TARGET_RECORDINGS_PER_SPECIES

# Keep the published cache bound to the project modules whose executable semantics
# affect sealed-audio verification or numerical feature derivation. The immutable
# data configuration has its own canonical content hash and is validated again on
# every load, so the shared config dispatcher is intentionally excluded. Unrelated
# model, training, reporting, CLI, and task-validation edits must not invalidate an
# unchanged cache.
_IMPLEMENTATION_FILES = (
    "audio.py",
    "clip_selection.py",
    "hashing.py",
    "io_utils.py",
    "locking.py",
    "manifest.py",
    "metadata.py",
    "metadata_artifacts.py",
    "paths.py",
    "review.py",
    "secure_audio_download.py",
    "signal.py",
    "unknown_acquisition.py",
    "unknown_audio.py",
    "unknown_clip_cache.py",
    "unknown_planning.py",
)

INDEX_FIELDS = [
    "schema_version",
    "clip_id",
    "candidate_id",
    "species_scientific_name",
    "species_common_name",
    "species_index",
    "difficulty_group",
    "selection_rank",
    "session_group",
    "relative_path",
    "source_sha256",
    "source_file_size_bytes",
    "feature_file",
    "feature_file_sha256",
    "feature_row",
    "energy_clip_count",
    "energy_rank",
    "energy_value",
    "start_sample",
    "valid_samples",
    "valid_audio_fraction",
    "left_padding_samples",
    "right_padding_samples",
    "decoded_samples",
    "decoded_duration_seconds",
    "audit_decoded_duration_seconds",
    "decoded_to_audit_duration_ratio",
]

_XC_ID = re.compile(r"XC[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_LOCK_FIELDS = {
    "schema_version",
    "cache_version",
    "provenance",
    "artifacts",
    "cache_content_sha256",
}
_PROVENANCE_FIELDS = {
    "data_config_file_sha256",
    "data_config_sha256",
    "unknown_audio_config_sha256",
    "audit_sha256",
    "audit_lock_sha256",
    "audit_checkpoint_set_sha256",
    "audit_raw_file_set_sha256",
    "selected_checkpoint_set_sha256",
    "selected_raw_file_set_sha256",
    "ffmpeg_executable_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "runtime",
    "input_paths",
}
_INPUT_PATH_FIELDS = {
    "data_config",
    "unknown_audio_config",
    "audit",
    "audit_lock",
    "checkpoint_root",
    "requirements_lock",
}
_RUNTIME_FIELDS = {
    "python_version",
    "python_implementation",
    "platform_system",
    "platform_machine",
    "numpy_version",
    "librosa_version",
    "ffmpeg_version_output",
}


@dataclass(frozen=True)
class _SelectedRecording:
    candidate_id: str
    scientific_name: str
    common_name: str
    species_index: int
    difficulty_group: str
    selection_rank: int
    session_group: str
    relative_path: str
    source_sha256: str
    source_file_size_bytes: int
    audit_decoded_duration_seconds: float
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class _BoundArtifact:
    path: Path
    sha256: str
    size_bytes: int
    private: bool


@dataclass(frozen=True)
class _ValidatedInputs:
    data_config: dict[str, Any]
    data_config_file: Path
    unknown_audio_config: dict[str, Any]
    unknown_audio_config_file: Path
    audit: dict[str, Any]
    audit_file: Path
    audit_lock: dict[str, Any]
    audit_lock_file: Path
    checkpoint_root: Path
    requirements_lock_file: Path
    raw_root: Path
    selected: tuple[_SelectedRecording, ...]
    artifact_hashes: dict[Path, str]
    artifact_bindings: tuple[_BoundArtifact, ...]
    checkpoint_file_paths: tuple[Path, ...]
    raw_file_paths: tuple[Path, ...]
    raw_directory_paths: tuple[Path, ...]
    data_config_file_sha256: str
    data_config_sha256: str
    unknown_audio_config_sha256: str
    audit_sha256: str
    audit_lock_sha256: str
    requirements_lock_sha256: str
    selected_checkpoint_set_sha256: str
    selected_raw_file_set_sha256: str


def _require_project_venv() -> None:
    expected = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(
            f"Unknown cache construction must run inside the project virtualenv: {expected}"
        )


def _project_label(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _resolve_input_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    lexical = Path(os.path.abspath(candidate))
    resolved = candidate.resolve(strict=True)
    if not is_relative_to(resolved, PROJECT_ROOT) or is_relative_to(resolved, RAW_DATA_ROOT):
        raise ValueError(f"{label} must remain inside the project and outside raw known data")
    if resolved != lexical:
        raise ValueError(f"{label} traverses a symbolic link")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular project file: {resolved}")
    return resolved


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_no_follow(path: Path, flags: int = os.O_RDONLY, mode: int = 0o600) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("Unknown cache descriptor reads require O_NOFOLLOW")
    return os.open(path, flags | no_follow | getattr(os, "O_CLOEXEC", 0), mode)


def _read_bound_file(
    path: Path,
    label: str,
    *,
    private: bool = False,
    collect_bytes: bool,
) -> tuple[bytes | None, str, int]:
    descriptor = _open_no_follow(path)
    payload = bytearray() if collect_bytes else None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if private and stat.S_IMODE(before.st_mode) != 0o600:
            raise ValueError(f"{label} must use private mode 0600")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
        after = os.fstat(descriptor)
        try:
            path_stat = path.lstat()
        except OSError as exc:
            raise ValueError(f"{label} path changed while it was being read") from exc
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(path_stat)
            or stat.S_ISLNK(path_stat.st_mode)
        ):
            raise ValueError(f"{label} changed while it was being read")
        return bytes(payload) if payload is not None else None, digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def _bound_artifact(path: Path, label: str, *, private: bool) -> _BoundArtifact:
    _, digest, size = _read_bound_file(
        path,
        label,
        private=private,
        collect_bytes=False,
    )
    return _BoundArtifact(path=path, sha256=digest, size_bytes=size, private=private)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _read_json_snapshot(
    path: Path,
    label: str,
    *,
    private: bool = False,
) -> tuple[dict[str, Any], str]:
    payload, digest, _ = _read_bound_file(
        path,
        label,
        private=private,
        collect_bytes=True,
    )
    if payload is None:
        raise RuntimeError(f"{label} snapshot was not collected")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if payload != _canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical JSON encoding")
    current = _bound_artifact(path, label, private=private)
    if current.sha256 != digest or current.size_bytes != len(payload):
        raise ValueError(f"{label} changed while its JSON snapshot was parsed")
    return value, digest


def _read_toml_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    payload, digest, size = _read_bound_file(
        path,
        label,
        collect_bytes=True,
    )
    if payload is None:
        raise RuntimeError(f"{label} snapshot was not collected")
    try:
        value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 TOML") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a TOML table")
    current = _bound_artifact(path, label, private=False)
    if current.sha256 != digest or current.size_bytes != size:
        raise ValueError(f"{label} changed while its TOML snapshot was parsed")
    return value, digest, size


def _read_csv_snapshot(
    path: Path,
    label: str,
) -> tuple[list[str], list[dict[str, str]], str]:
    payload, digest, _ = _read_bound_file(
        path,
        label,
        private=True,
        collect_bytes=True,
    )
    if payload is None:
        raise RuntimeError(f"{label} snapshot was not collected")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    current = _bound_artifact(path, label, private=True)
    if current.sha256 != digest or current.size_bytes != len(payload):
        raise ValueError(f"{label} changed while its CSV snapshot was parsed")
    return fieldnames, rows, digest


def _assert_locked_signal_config(config: Mapping[str, Any]) -> None:
    clip = config.get("clip_selection")
    spectrogram = config.get("spectrogram")
    if not isinstance(clip, Mapping) or not isinstance(spectrogram, Mapping):
        raise ValueError("Data configuration lacks signal preprocessing sections")
    expected = {
        "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "target_channels": 1,
        "audio_dtype": "float32",
        "clip_duration_seconds": CLIP_DURATION_SECONDS,
        "maximum_clips_per_recording": MAXIMUM_CLIPS_PER_RECORDING,
        "energy_candidate_hop_seconds": ENERGY_CANDIDATE_HOP_SAMPLES / TARGET_SAMPLE_RATE_HZ,
        "include_end_aligned_candidate": True,
        "energy_measure": "mean_unlogged_stft_power_150_to_14000_hz",
        "tie_break": "earlier_start",
        "minimum_selected_start_separation_seconds": (
            MINIMUM_SELECTED_START_SEPARATION_SAMPLES / TARGET_SAMPLE_RATE_HZ
        ),
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "window": "hann",
        "center": False,
        "n_mels": N_MELS,
        "f_min_hz": F_MIN_HZ,
        "f_max_hz": F_MAX_HZ,
        "power": 2.0,
        "mel_scale": "slaney",
        "htk": False,
        "mel_normalization": "slaney",
        "power_to_db_reference": "per_clip_max",
        "power_to_db_amin": POWER_TO_DB_AMIN,
        "minimum_db": MINIMUM_DB,
        "maximum_db": MAXIMUM_DB,
        "expected_native_height": NATIVE_MEL_HEIGHT,
        "expected_native_width": NATIVE_MEL_WIDTH,
    }
    observed = {
        "target_sample_rate_hz": config.get("target_sample_rate_hz"),
        "target_channels": config.get("target_channels"),
        "audio_dtype": config.get("audio_dtype"),
        "clip_duration_seconds": config.get("clip_duration_seconds"),
        **{key: clip.get(key) for key in expected if key in clip},
        **{key: spectrogram.get(key) for key in expected if key in spectrogram},
    }
    mismatches = [
        key for key, expected_value in expected.items() if observed.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Data configuration differs from the unknown cache contract: {mismatches}"
        )


def _runtime_provenance(ffmpeg: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("XENO_CANTO_API_KEY", None)
    completed = subprocess.run(
        [str(ffmpeg), "-version"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError("Unable to capture FFmpeg runtime identity")
    output = "\n".join(line.rstrip() for line in completed.stdout.splitlines()).strip()
    if not output:
        raise RuntimeError("FFmpeg returned an empty runtime identity")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "librosa_version": librosa.__version__,
        "ffmpeg_version_output": output,
    }


def _implementation_fingerprint() -> str:
    implementation_root = resolve_project_path("src/bird_audio")
    paths = tuple(implementation_root / name for name in _IMPLEMENTATION_FILES)
    if (
        len(paths) != len(set(paths))
        or Path(__file__).resolve() not in {path.resolve() for path in paths}
        or any(not path.is_file() for path in paths)
    ):
        raise RuntimeError("Unknown cache implementation file set is incomplete")
    digest = hashlib.sha256()
    for path in paths:
        artifact = _bound_artifact(path, f"implementation file {path.name}", private=False)
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(artifact.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_private_bound_file(
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if _SHA256.fullmatch(expected_sha256) is None or expected_size <= 0:
        raise ValueError("Unknown raw audio binding is invalid")
    artifact = _bound_artifact(path, "unknown raw audio", private=True)
    if artifact.size_bytes != expected_size or artifact.sha256 != expected_sha256:
        raise ValueError("Unknown raw audio changed or failed its SHA-256 binding")


def _hash_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validate_raw_descriptor(
    descriptor: int,
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> tuple[int, int, int, int, int]:
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != expected_size
        or _hash_descriptor(descriptor) != expected_sha256
    ):
        raise ValueError("Unknown raw descriptor failed its private content binding")
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or _file_identity(path_stat) != _file_identity(observed):
        raise ValueError("Unknown raw path no longer names the verified descriptor")
    return _file_identity(observed)


@contextmanager
def _verified_raw_descriptor(
    path: Path,
    expected_sha256: str,
    expected_size: int,
):
    descriptor = _open_no_follow(path)
    try:
        initial_identity = _validate_raw_descriptor(
            descriptor,
            path,
            expected_sha256,
            expected_size,
        )
        yield descriptor
        final_identity = _validate_raw_descriptor(
            descriptor,
            path,
            expected_sha256,
            expected_size,
        )
        if final_identity != initial_identity:
            raise ValueError("Unknown raw descriptor changed during FFmpeg decoding")
    finally:
        os.close(descriptor)


def _decode_audio_descriptor(
    descriptor: int,
    source_path: Path,
    ffmpeg: Path,
    *,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    timeout_seconds: int = 3_600,
) -> np.ndarray:
    """Decode one inherited, already verified regular-file descriptor."""
    descriptor_path = f"/dev/fd/{descriptor}"
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        descriptor_path,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
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
    environment = os.environ.copy()
    environment.pop("XENO_CANTO_API_KEY", None)
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            pass_fds=(descriptor,),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioToolError(
            f"FFmpeg descriptor decode failed for {source_path.name}: {type(exc).__name__}"
        ) from exc
    if completed.returncode != 0:
        diagnostic = normalize_ffmpeg_diagnostic(completed.stderr.decode("utf-8", errors="replace"))
        raise AudioToolError(
            f"FFmpeg descriptor decode failed for {source_path.name}: {diagnostic}"
        )
    if len(completed.stdout) % np.dtype("<f4").itemsize:
        raise AudioToolError(f"Unexpected float32 byte count for {source_path.name}")
    waveform = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if waveform.size == 0 or not bool(np.all(np.isfinite(waveform))):
        raise AudioToolError(f"FFmpeg produced invalid audio for {source_path.name}")
    return waveform


def _resolve_and_verify_raw_file(
    raw_root: Path,
    candidate_id: str,
    scientific_name: str,
    relative_path: str,
    source_sha256: str,
    source_file_size_bytes: int,
) -> Path:
    expected_relative = (
        raw_root.relative_to(PROJECT_ROOT).as_posix()
        + f"/{scientific_name.replace(' ', '_')}/{candidate_id}.audio"
    )
    if relative_path != expected_relative or Path(relative_path).is_absolute():
        raise ValueError(f"Unknown raw path is not canonical: {candidate_id}")
    lexical = PROJECT_ROOT / relative_path
    resolved = resolve_project_path(relative_path)
    if (
        resolved != lexical
        or not is_relative_to(resolved, raw_root)
        or raw_root.resolve() != raw_root
        or lexical.parent.resolve() != lexical.parent
    ):
        raise ValueError(f"Unknown raw path traverses a symbolic link: {candidate_id}")
    _validate_private_bound_file(lexical, source_sha256, source_file_size_bytes)
    return lexical


def _strict_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _strict_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _effective_species_order(data_config: Mapping[str, Any], audit: Mapping[str, Any]) -> list[str]:
    primary = [str(row["scientific_name"]) for row in data_config["unknown_species"]]
    gate = audit.get("gate")
    if not isinstance(gate, Mapping) or gate.get("status") not in {
        "ready_without_fallback",
        "ready_with_fallback",
    }:
        raise ValueError("Unknown audit gate is not ready for scoring")
    replacement = gate.get("replacement")
    if gate.get("fallback_active") is True:
        if not isinstance(replacement, Mapping):
            raise ValueError("Fallback audit lacks its exact species replacement")
        replaced = str(replacement.get("replaced_scientific_name") or "")
        replacement_name = str(replacement.get("replacement_scientific_name") or "")
        if replaced not in primary or not replacement_name:
            raise ValueError("Fallback replacement is not valid")
        primary[primary.index(replaced)] = replacement_name
    elif replacement is not None:
        raise ValueError("Non-fallback audit unexpectedly contains a replacement")
    if len(primary) != TARGET_SPECIES or len(set(primary)) != TARGET_SPECIES:
        raise ValueError("Effective unknown species set is not exact")
    return primary


def _checkpoint_record(
    checkpoint_root: Path,
    candidate_id: str,
    scientific_name: str,
    common_name: str,
    species_index: int,
    difficulty_group: str,
    selection_rank: int,
    raw_root: Path,
) -> _SelectedRecording:
    checkpoint_path = checkpoint_root / f"{candidate_id}.json"
    if checkpoint_path.parent != checkpoint_root:
        raise ValueError("Selected checkpoint path is unsafe")
    checkpoint, checkpoint_sha256 = _read_json_snapshot(
        checkpoint_path, f"checkpoint {candidate_id}", private=True
    )
    required_checkpoint_fields = {
        "schema_version",
        "preflight_sha256",
        "plan_sha256",
        "candidate_id",
        "scientific_name",
        "queue_rank",
        "session_group",
        "disposition",
        "reasons",
        "download_receipt",
        "audio_qc",
    }
    if (
        set(checkpoint) != required_checkpoint_fields
        or checkpoint.get("schema_version") != "1.0"
        or checkpoint.get("candidate_id") != candidate_id
        or checkpoint.get("scientific_name") != scientific_name
        or checkpoint.get("disposition") != "eligible"
        or checkpoint.get("reasons") != []
    ):
        raise ValueError(f"Selected terminal checkpoint is not eligible: {candidate_id}")
    session_group = checkpoint.get("session_group")
    qc = checkpoint.get("audio_qc")
    if not isinstance(session_group, str) or not session_group.startswith("session:"):
        raise ValueError(f"Selected checkpoint session is invalid: {candidate_id}")
    if not isinstance(qc, Mapping):
        raise ValueError(f"Selected checkpoint QC is missing: {candidate_id}")
    if (
        qc.get("candidate_id") != candidate_id
        or qc.get("scientific_name") != scientific_name
        or qc.get("session_group") != session_group
        or qc.get("disposition") != "eligible"
        or qc.get("reasons") != []
        or qc.get("probe_status") != "ok"
        or qc.get("full_decode_status") != "ok"
        or qc.get("header_detection_status") != "recognized"
    ):
        raise ValueError(f"Selected checkpoint QC binding is invalid: {candidate_id}")
    relative_path = qc.get("relative_path")
    source_sha256 = qc.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(source_sha256, str):
        raise ValueError(f"Selected checkpoint raw binding is invalid: {candidate_id}")
    source_file_size_bytes = _strict_positive_int(
        qc.get("file_size_bytes"), f"{candidate_id} file size"
    )
    decoded_duration = _strict_positive_float(
        qc.get("decoded_duration_seconds"), f"{candidate_id} decoded duration"
    )
    _resolve_and_verify_raw_file(
        raw_root,
        candidate_id,
        scientific_name,
        relative_path,
        source_sha256,
        source_file_size_bytes,
    )
    return _SelectedRecording(
        candidate_id=candidate_id,
        scientific_name=scientific_name,
        common_name=common_name,
        species_index=species_index,
        difficulty_group=difficulty_group,
        selection_rank=selection_rank,
        session_group=session_group,
        relative_path=relative_path,
        source_sha256=source_sha256,
        source_file_size_bytes=source_file_size_bytes,
        audit_decoded_duration_seconds=decoded_duration,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )


def _selected_records(
    data_config: Mapping[str, Any],
    audit: Mapping[str, Any],
    checkpoint_root: Path,
    raw_root: Path,
) -> tuple[_SelectedRecording, ...]:
    selection = audit.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Unknown audit selection is missing")
    if (
        selection.get("selected_recordings") != TARGET_RECORDINGS
        or selection.get("species_count") != TARGET_SPECIES
        or selection.get("zero_candidate_overlap") is not True
        or selection.get("zero_session_overlap") is not True
    ):
        raise ValueError("Unknown audit selection totals are not exact")
    species_selection = selection.get("species")
    if not isinstance(species_selection, Mapping):
        raise ValueError("Unknown audit species selection is invalid")

    all_species: dict[str, tuple[str, str]] = {}
    for row in data_config["unknown_species"]:
        all_species[str(row["scientific_name"])] = (
            str(row["common_name"]),
            str(row["difficulty_group"]),
        )
    for row in data_config["fallback_unknown_species"]:
        all_species[str(row["scientific_name"])] = (str(row["common_name"]), "fallback")
    effective = _effective_species_order(data_config, audit)
    if set(species_selection) != set(effective):
        raise ValueError("Selected species differ from the effective fallback gate")

    selected: list[_SelectedRecording] = []
    candidate_ids: set[str] = set()
    sessions: set[str] = set()
    for species_index, scientific_name in enumerate(effective):
        common_name, difficulty_group = all_species[scientific_name]
        species_entry = species_selection[scientific_name]
        if not isinstance(species_entry, Mapping):
            raise ValueError(f"Unknown selection entry is invalid: {scientific_name}")
        ids = species_entry.get("selected_candidate_ids")
        assignment = species_entry.get("assignment")
        if (
            species_entry.get("selected_candidates") != TARGET_RECORDINGS_PER_SPECIES
            or not isinstance(ids, list)
            or len(ids) != TARGET_RECORDINGS_PER_SPECIES
            or len(set(ids)) != TARGET_RECORDINGS_PER_SPECIES
            or not isinstance(assignment, Mapping)
            or not isinstance(assignment.get("assignments"), list)
        ):
            raise ValueError(f"Unknown selection count is invalid: {scientific_name}")
        assignment_ids = [row.get("candidate_id") for row in assignment["assignments"]]
        if assignment_ids != ids:
            raise ValueError(
                f"Unknown selected ID order is not assignment-bound: {scientific_name}"
            )
        for selection_rank, candidate_id_value in enumerate(ids):
            if (
                not isinstance(candidate_id_value, str)
                or _XC_ID.fullmatch(candidate_id_value) is None
            ):
                raise ValueError("Unknown selection contains an unsafe candidate ID")
            if candidate_id_value in candidate_ids:
                raise ValueError("Unknown selected candidate IDs overlap")
            record = _checkpoint_record(
                checkpoint_root,
                candidate_id_value,
                scientific_name,
                common_name,
                species_index,
                difficulty_group,
                selection_rank,
                raw_root,
            )
            assignment_row = assignment["assignments"][selection_rank]
            if assignment_row.get("candidate_session_group") != record.session_group:
                raise ValueError("Unknown assignment and checkpoint sessions differ")
            if record.session_group in sessions:
                raise ValueError("Unknown selected sessions overlap")
            candidate_ids.add(candidate_id_value)
            sessions.add(record.session_group)
            selected.append(record)
    if len(selected) != TARGET_RECORDINGS:
        raise ValueError("Unknown selected recording total is not exact")
    return tuple(selected)


def _enumerate_checkpoint_artifacts(
    checkpoint_root: Path,
) -> tuple[list[dict[str, str]], tuple[_BoundArtifact, ...], tuple[Path, ...]]:
    children = sorted(checkpoint_root.iterdir(), key=lambda item: item.name)
    if any(
        _XC_ID.fullmatch(child.stem) is None
        or child.suffix != ".json"
        or child.is_symlink()
        or not child.is_file()
        for child in children
    ):
        raise ValueError("Unknown checkpoint physical file set is invalid")
    records: list[dict[str, str]] = []
    artifacts: list[_BoundArtifact] = []
    for path in children:
        _, digest = _read_json_snapshot(path, f"checkpoint set {path.stem}", private=True)
        size = path.lstat().st_size
        records.append({"path": _project_label(path), "sha256": digest})
        artifacts.append(_BoundArtifact(path=path, sha256=digest, size_bytes=size, private=True))
    return records, tuple(artifacts), tuple(children)


def _enumerate_raw_artifacts(
    raw_root: Path,
) -> tuple[
    list[dict[str, Any]],
    tuple[_BoundArtifact, ...],
    tuple[Path, ...],
    tuple[Path, ...],
]:
    records: list[dict[str, Any]] = []
    artifacts: list[_BoundArtifact] = []
    paths: list[Path] = []
    directories: list[Path] = []
    for path in sorted(raw_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("Unknown raw physical tree contains a symbolic link")
        if path.is_dir():
            if path.resolve() != path:
                raise ValueError("Unknown raw directory is noncanonical")
            directories.append(path)
            continue
        if not path.is_file():
            raise ValueError("Unknown raw physical tree contains a non-file artifact")
        artifact = _bound_artifact(path, f"unknown raw set {path.name}", private=True)
        records.append(
            {
                "path": _project_label(path),
                "sha256": artifact.sha256,
                "file_size_bytes": artifact.size_bytes,
            }
        )
        artifacts.append(artifact)
        paths.append(path)
    return records, tuple(artifacts), tuple(paths), tuple(directories)


def _require_declared_full_sets(
    audit: Mapping[str, Any],
    audit_lock: Mapping[str, Any],
    checkpoint_root: Path,
    raw_root: Path,
) -> tuple[
    tuple[_BoundArtifact, ...],
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
]:
    checkpoints, checkpoint_artifacts, checkpoint_paths = _enumerate_checkpoint_artifacts(
        checkpoint_root
    )
    raw_files, raw_artifacts, raw_paths, raw_directories = _enumerate_raw_artifacts(raw_root)
    checkpoint_sha256 = sha256_json(checkpoints)
    raw_sha256 = sha256_json(raw_files)
    if (
        audit.get("checkpoint_count") != len(checkpoints)
        or audit_lock.get("checkpoint_count") != len(checkpoints)
        or audit.get("checkpoint_set_sha256") != checkpoint_sha256
        or audit_lock.get("checkpoint_set_sha256") != checkpoint_sha256
        or audit.get("raw_file_count") != len(raw_files)
        or audit_lock.get("raw_file_count") != len(raw_files)
        or audit.get("raw_file_set_sha256") != raw_sha256
        or audit_lock.get("raw_file_set_sha256") != raw_sha256
    ):
        raise ValueError("Unknown audit-declared full file sets are not exact")
    return (
        (*checkpoint_artifacts, *raw_artifacts),
        checkpoint_paths,
        raw_paths,
        raw_directories,
    )


def _verified_audit_snapshots(
    unknown_config_file: Path,
    expected_unknown_config: Mapping[str, Any],
    expected_unknown_config_sha256: str,
    audit_file: Path,
    audit_lock_file: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    audit, audit_sha256 = _read_json_snapshot(audit_file, "unknown audio audit", private=True)
    audit_lock, audit_lock_sha256 = _read_json_snapshot(
        audit_lock_file, "unknown audio audit lock", private=True
    )
    verification = verify_unknown_audio_audit(unknown_config_file)
    audit_after, audit_sha256_after = _read_json_snapshot(
        audit_file, "unknown audio audit", private=True
    )
    audit_lock_after, audit_lock_sha256_after = _read_json_snapshot(
        audit_lock_file, "unknown audio audit lock", private=True
    )
    unknown_config_after, unknown_config_sha256_after, _ = _read_toml_snapshot(
        unknown_config_file, "unknown audio config"
    )
    if (
        audit_after != audit
        or audit_sha256_after != audit_sha256
        or audit_lock_after != audit_lock
        or audit_lock_sha256_after != audit_lock_sha256
        or unknown_config_after != expected_unknown_config
        or unknown_config_sha256_after != expected_unknown_config_sha256
        or verification.get("audit_sha256") != audit_sha256
    ):
        raise RuntimeError("Unknown audit snapshots changed around their verifier")
    return audit, audit_sha256, audit_lock, audit_lock_sha256, verification


def _load_validated_inputs(
    *,
    audit_path: str | Path,
    audit_lock_path: str | Path,
    checkpoint_root: str | Path,
    config_path: str | Path,
    unknown_audio_config_path: str | Path,
) -> _ValidatedInputs:
    data_config_file = _resolve_input_file(config_path, "data config")
    data_config, data_config_file_sha256, data_config_size = _read_toml_snapshot(
        data_config_file, "data config"
    )
    if "extends" in data_config:
        raise ValueError("Unknown cache data config cannot use inheritance")
    validate_config(data_config)
    data_config["_config_path"] = str(data_config_file)
    _assert_locked_signal_config(data_config)

    unknown_config_file = _resolve_input_file(unknown_audio_config_path, "unknown audio config")
    unknown_config, unknown_config_sha256, unknown_config_size = _read_toml_snapshot(
        unknown_config_file, "unknown audio config"
    )
    validated_unknown_config = load_unknown_audio_config(unknown_config_file)
    if validated_unknown_config != unknown_config:
        raise RuntimeError("Unknown audio config changed between snapshot and validation")

    audit_file = _resolve_input_file(audit_path, "unknown audio audit")
    audit_lock_file = _resolve_input_file(audit_lock_path, "unknown audio audit lock")
    expected_audit = resolve_project_path(unknown_config["outputs"]["audit"])
    expected_audit_lock = resolve_project_path(unknown_config["outputs"]["audit_lock"])
    if audit_file != expected_audit or audit_lock_file != expected_audit_lock:
        raise ValueError("Unknown cache audit paths differ from the locked acquisition config")

    working_root = resolve_project_path(unknown_config["outputs"]["working_directory"])
    resolved_checkpoint_root = resolve_project_path(checkpoint_root)
    if resolved_checkpoint_root != working_root / "checkpoints":
        raise ValueError("Unknown checkpoint root differs from the locked acquisition config")
    if (
        resolved_checkpoint_root.is_symlink()
        or not resolved_checkpoint_root.is_dir()
        or resolved_checkpoint_root.resolve() != resolved_checkpoint_root
    ):
        raise ValueError("Unknown checkpoint root is unsafe")
    raw_root = resolve_project_path(unknown_config["outputs"]["raw_directory"])
    expected_raw_root = PROJECT_ROOT / "data" / "unknown" / "raw" / "audio_v1"
    if raw_root != expected_raw_root or raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("Unknown raw audio root is not the locked private root")

    (
        audit,
        audit_sha256,
        audit_lock,
        audit_lock_sha256,
        verification,
    ) = _verified_audit_snapshots(
        unknown_config_file,
        unknown_config,
        unknown_config_sha256,
        audit_file,
        audit_lock_file,
    )
    if (
        verification.get("valid") is not True
        or verification.get("ready_for_unknown_scoring") is not True
        or verification.get("selected_recordings") != TARGET_RECORDINGS
        or verification.get("species") != TARGET_SPECIES
    ):
        raise ValueError("Unknown audio audit did not verify for scoring")
    if (
        audit_lock.get("audit_sha256") != audit_sha256
        or audit_lock.get("config_sha256") != unknown_config_sha256
        or audit_lock.get("selected_recordings") != TARGET_RECORDINGS
        or audit_lock.get("ready_for_unknown_scoring") is not True
        or audit.get("ready_for_unknown_scoring") is not True
    ):
        raise ValueError("Unknown audio audit lock binding is invalid")

    requirements_lock_file = _resolve_input_file("requirements.lock", "requirements lock")
    requirements_artifact = _bound_artifact(
        requirements_lock_file, "requirements lock", private=False
    )
    requirements_lock_sha256 = requirements_artifact.sha256
    (
        full_artifacts,
        checkpoint_file_paths,
        raw_file_paths,
        raw_directory_paths,
    ) = _require_declared_full_sets(
        audit,
        audit_lock,
        resolved_checkpoint_root,
        raw_root,
    )
    selected = _selected_records(data_config, audit, resolved_checkpoint_root, raw_root)
    selected_checkpoints = [
        {"path": _project_label(row.checkpoint_path), "sha256": row.checkpoint_sha256}
        for row in sorted(selected, key=lambda item: item.candidate_id)
    ]
    selected_raw = [
        {
            "path": row.relative_path,
            "sha256": row.source_sha256,
            "file_size_bytes": row.source_file_size_bytes,
        }
        for row in sorted(selected, key=lambda item: item.relative_path)
    ]
    data_config_artifact = _bound_artifact(data_config_file, "data config", private=False)
    unknown_config_artifact = _bound_artifact(
        unknown_config_file, "unknown audio config", private=False
    )
    audit_artifact = _bound_artifact(audit_file, "unknown audio audit", private=True)
    audit_lock_artifact = _bound_artifact(audit_lock_file, "unknown audio audit lock", private=True)
    if (
        (data_config_artifact.sha256, data_config_artifact.size_bytes)
        != (data_config_file_sha256, data_config_size)
        or (unknown_config_artifact.sha256, unknown_config_artifact.size_bytes)
        != (unknown_config_sha256, unknown_config_size)
        or audit_artifact.sha256 != audit_sha256
        or audit_lock_artifact.sha256 != audit_lock_sha256
    ):
        raise RuntimeError("Unknown cache fixed inputs changed during validation")
    fixed_artifacts = (
        data_config_artifact,
        unknown_config_artifact,
        audit_artifact,
        audit_lock_artifact,
        requirements_artifact,
    )
    artifact_bindings = (*fixed_artifacts, *full_artifacts)
    artifact_hashes = {row.path: row.sha256 for row in artifact_bindings}
    return _ValidatedInputs(
        data_config=data_config,
        data_config_file=data_config_file,
        unknown_audio_config=unknown_config,
        unknown_audio_config_file=unknown_config_file,
        audit=audit,
        audit_file=audit_file,
        audit_lock=audit_lock,
        audit_lock_file=audit_lock_file,
        checkpoint_root=resolved_checkpoint_root,
        requirements_lock_file=requirements_lock_file,
        raw_root=raw_root,
        selected=selected,
        artifact_hashes=artifact_hashes,
        artifact_bindings=artifact_bindings,
        checkpoint_file_paths=checkpoint_file_paths,
        raw_file_paths=raw_file_paths,
        raw_directory_paths=raw_directory_paths,
        data_config_file_sha256=data_config_file_sha256,
        data_config_sha256=sha256_json(data_config),
        unknown_audio_config_sha256=unknown_config_sha256,
        audit_sha256=audit_sha256,
        audit_lock_sha256=audit_lock_sha256,
        requirements_lock_sha256=requirements_lock_sha256,
        selected_checkpoint_set_sha256=sha256_json(selected_checkpoints),
        selected_raw_file_set_sha256=sha256_json(selected_raw),
    )


def _require_inputs_unchanged(inputs: _ValidatedInputs) -> None:
    _, checkpoint_artifacts, checkpoint_paths = _enumerate_checkpoint_artifacts(
        inputs.checkpoint_root
    )
    _, raw_artifacts, raw_paths, raw_directories = _enumerate_raw_artifacts(inputs.raw_root)
    if (
        checkpoint_paths != inputs.checkpoint_file_paths
        or raw_paths != inputs.raw_file_paths
        or raw_directories != inputs.raw_directory_paths
    ):
        raise RuntimeError("Unknown full input file sets changed during cache construction")
    current_full = {row.path: row for row in (*checkpoint_artifacts, *raw_artifacts)}
    expected_full_paths = set(inputs.checkpoint_file_paths) | set(inputs.raw_file_paths)
    for expected in inputs.artifact_bindings:
        if expected.path in expected_full_paths:
            current = current_full[expected.path]
        else:
            current = _bound_artifact(
                expected.path,
                f"bound input {expected.path.name}",
                private=expected.private,
            )
        if current != expected:
            raise RuntimeError(f"Input changed during unknown cache construction: {expected.path}")
    for row in inputs.selected:
        _resolve_and_verify_raw_file(
            inputs.raw_root,
            row.candidate_id,
            row.scientific_name,
            row.relative_path,
            row.source_sha256,
            row.source_file_size_bytes,
        )


def _cache_provenance(
    inputs: _ValidatedInputs,
    ffmpeg_sha256: str,
    implementation_sha256: str,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "data_config_file_sha256": inputs.data_config_file_sha256,
        "data_config_sha256": inputs.data_config_sha256,
        "unknown_audio_config_sha256": inputs.unknown_audio_config_sha256,
        "audit_sha256": inputs.audit_sha256,
        "audit_lock_sha256": inputs.audit_lock_sha256,
        "audit_checkpoint_set_sha256": inputs.audit_lock["checkpoint_set_sha256"],
        "audit_raw_file_set_sha256": inputs.audit_lock["raw_file_set_sha256"],
        "selected_checkpoint_set_sha256": inputs.selected_checkpoint_set_sha256,
        "selected_raw_file_set_sha256": inputs.selected_raw_file_set_sha256,
        "ffmpeg_executable_sha256": ffmpeg_sha256,
        "implementation_sha256": implementation_sha256,
        "requirements_lock_sha256": inputs.requirements_lock_sha256,
        "runtime": dict(runtime),
        "input_paths": {
            "data_config": _project_label(inputs.data_config_file),
            "unknown_audio_config": _project_label(inputs.unknown_audio_config_file),
            "audit": _project_label(inputs.audit_file),
            "audit_lock": _project_label(inputs.audit_lock_file),
            "checkpoint_root": _project_label(inputs.checkpoint_root),
            "requirements_lock": _project_label(inputs.requirements_lock_file),
        },
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity(path: Path) -> tuple[int, int]:
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError(f"Unknown cache directory identity is invalid: {path}")
    return observed.st_dev, observed.st_ino


def _atomic_write_json_durable(path: Path, value: Any, *, private: bool = False) -> None:
    atomic_write_json(path, value)
    if private:
        path.chmod(0o600)
    _fsync_directory(path.parent)


def _atomic_write_csv_durable(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    atomic_write_csv(path, rows, fieldnames)
    _fsync_directory(path.parent)


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _copy_file_create_only(source: Path, destination: Path, expected_sha256: str) -> None:
    source_descriptor = _open_no_follow(source)
    destination_descriptor = -1
    digest = hashlib.sha256()
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_nlink != 1:
            raise ValueError("Unknown resume feature source is not a single-link regular file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while chunk := os.read(source_descriptor, 1024 * 1024):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        source_after = os.fstat(source_descriptor)
        if (
            _file_identity(source_before) != _file_identity(source_after)
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError("Unknown resume feature changed during publication copy")
    except BaseException:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
            destination_descriptor = -1
        destination.unlink(missing_ok=True)
        if destination.parent.is_dir():
            _fsync_directory(destination.parent)
        raise
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    _fsync_directory(destination.parent)


def _serialized_npy_identity(value: np.ndarray) -> tuple[str, int]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    payload = buffer.getvalue()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _derive_recording(
    record: _SelectedRecording,
    raw_root: Path,
    ffmpeg: Path,
    minimum_duration_ratio: float,
    maximum_duration_ratio: float,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, str], dict[str, int]]:
    raw_path = _resolve_and_verify_raw_file(
        raw_root,
        record.candidate_id,
        record.scientific_name,
        record.relative_path,
        record.source_sha256,
        record.source_file_size_bytes,
    )
    with _verified_raw_descriptor(
        raw_path,
        record.source_sha256,
        record.source_file_size_bytes,
    ) as descriptor:
        waveform = _decode_audio_descriptor(
            descriptor,
            raw_path,
            ffmpeg,
            sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        )
    decoded_samples = int(waveform.size)
    decoded_duration = decoded_samples / TARGET_SAMPLE_RATE_HZ
    ratio = decoded_duration / record.audit_decoded_duration_seconds
    if not minimum_duration_ratio <= ratio <= maximum_duration_ratio:
        raise ValueError(
            f"Decoded duration ratio outside accepted bounds for {record.candidate_id}: {ratio:.9f}"
        )
    candidates = select_energy_candidates(waveform)
    if not 1 <= len(candidates) <= MAXIMUM_CLIPS_PER_RECORDING:
        raise RuntimeError(f"Energy selector returned an invalid count for {record.candidate_id}")

    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    logical_feature_path = Path(SCORING_PARTITION) / "features" / f"{record.candidate_id}.npy"
    extracted = iter_extracted_clips(
        waveform,
        (candidate.start_sample for candidate in candidates),
        clip_samples=CLIP_SAMPLES,
    )
    for energy_rank, (candidate, clip) in enumerate(zip(candidates, extracted, strict=True)):
        feature = native_log_mel_spectrogram(clip.samples)
        if (
            feature.shape != NATIVE_FEATURE_SHAPE
            or feature.dtype != np.float32
            or not bool(np.all(np.isfinite(feature)))
            or float(feature.min()) < 0
            or float(feature.max()) > 1
        ):
            raise RuntimeError(f"Native feature contract failed for {record.candidate_id}")
        features.append(feature)
        rows.append(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "clip_id": f"{record.candidate_id}:{clip.start_sample:012d}",
                "candidate_id": record.candidate_id,
                "species_scientific_name": record.scientific_name,
                "species_common_name": record.common_name,
                "species_index": record.species_index,
                "difficulty_group": record.difficulty_group,
                "selection_rank": record.selection_rank,
                "session_group": record.session_group,
                "relative_path": record.relative_path,
                "source_sha256": record.source_sha256,
                "source_file_size_bytes": record.source_file_size_bytes,
                "feature_file": logical_feature_path.as_posix(),
                "feature_file_sha256": "",
                "feature_row": energy_rank,
                "energy_clip_count": len(candidates),
                "energy_rank": energy_rank,
                "energy_value": f"{candidate.energy:.17g}",
                "start_sample": clip.start_sample,
                "valid_samples": clip.valid_samples,
                "valid_audio_fraction": f"{clip.valid_audio_fraction:.9f}",
                "left_padding_samples": clip.left_padding_samples,
                "right_padding_samples": clip.right_padding_samples,
                "decoded_samples": decoded_samples,
                "decoded_duration_seconds": f"{decoded_duration:.9f}",
                "audit_decoded_duration_seconds": (f"{record.audit_decoded_duration_seconds:.9f}"),
                "decoded_to_audit_duration_ratio": f"{ratio:.9f}",
            }
        )
    tensor = np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32)
    feature_sha256, feature_bytes = _serialized_npy_identity(tensor)
    for row in rows:
        row["feature_file_sha256"] = feature_sha256
    feature_record = {"path": logical_feature_path.as_posix(), "sha256": feature_sha256}
    statistics = {
        "clips": len(rows),
        "feature_bytes": feature_bytes,
    }
    return rows, tensor, feature_record, statistics


def _process_recording(
    record: _SelectedRecording,
    raw_root: Path,
    ffmpeg: Path,
    minimum_duration_ratio: float,
    maximum_duration_ratio: float,
    feature_output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    rows, tensor, feature_record, statistics = _derive_recording(
        record,
        raw_root,
        ffmpeg,
        minimum_duration_ratio,
        maximum_duration_ratio,
    )
    _write_npy(feature_output_path, tensor)
    feature_artifact = _bound_artifact(
        feature_output_path,
        f"unknown feature {record.candidate_id}",
        private=True,
    )
    if (
        feature_artifact.sha256 != feature_record["sha256"]
        or feature_artifact.size_bytes != statistics["feature_bytes"]
    ):
        raise RuntimeError(f"Serialized feature identity drifted for {record.candidate_id}")
    return rows, feature_record, statistics


def _atomic_publish_directory_no_replace(source: Path, destination: Path) -> None:
    if source.parent.resolve() != destination.parent.resolve():
        raise ValueError("Atomic cache publication requires one parent filesystem")
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    ctypes.set_errno(0)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename_exclusive = library.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename_exclusive = library.renameat2
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise RuntimeError("This platform has no supported atomic no-replace directory rename")
    if result == 0:
        _fsync_directory(destination.parent)
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Cache destination appeared before atomic publication",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


@contextmanager
def _unknown_cache_build_lock(destination: Path):
    lock_path = destination.with_name(f".{destination.name}.build.lock")
    descriptor = _open_no_follow(
        lock_path,
        os.O_RDWR | os.O_CREAT,
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ValueError("Unknown cache advisory lock is not a single-link regular file")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Unknown cache build is already active: {destination}") from exc
        payload = _canonical_json_bytes(
            {
                "cache_version": CACHE_VERSION,
                "destination": _project_label(destination),
                "pid": os.getpid(),
            }
        )
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        _fsync_directory(lock_path.parent)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _resume_identity(
    provenance: Mapping[str, Any], selected: Sequence[_SelectedRecording]
) -> tuple[str, list[str]]:
    recording_order = [row.candidate_id for row in selected]
    identity = sha256_json(
        {
            "schema_version": RESUME_SCHEMA_VERSION,
            "cache_version": CACHE_VERSION,
            "provenance": provenance,
            "index_fields": INDEX_FIELDS,
            "recordings": [
                {
                    "candidate_id": row.candidate_id,
                    "scientific_name": row.scientific_name,
                    "session_group": row.session_group,
                    "source_sha256": row.source_sha256,
                    "selection_rank": row.selection_rank,
                    "audit_decoded_duration_seconds": row.audit_decoded_duration_seconds,
                }
                for row in selected
            ],
        }
    )
    return identity, recording_order


def _prepare_working_directory(
    working_root: Path,
    build_identity_sha256: str,
    recording_order: Sequence[str],
) -> Path:
    expected_state = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "build_identity_sha256": build_identity_sha256,
        "recordings": len(recording_order),
        "recording_order_sha256": sha256_json(list(recording_order)),
    }
    if not working_root.exists():
        initial = Path(
            tempfile.mkdtemp(
                prefix=f".{working_root.name}.",
                suffix=".initializing",
                dir=working_root.parent,
            )
        )
        try:
            initial.chmod(0o700)
            (initial / "completed").mkdir(mode=0o700)
            _fsync_directory(initial)
            _atomic_write_json_durable(initial / "resume.json", expected_state, private=True)
            _atomic_publish_directory_no_replace(initial, working_root)
        except BaseException:
            shutil.rmtree(initial, ignore_errors=True)
            raise
    if (
        not working_root.is_dir()
        or working_root.is_symlink()
        or working_root.resolve() != working_root
        or stat.S_IMODE(working_root.stat().st_mode) != 0o700
    ):
        raise ValueError("Unknown cache resume root is unsafe")
    for child in list(working_root.iterdir()):
        if child.name.startswith(".") and child.name.endswith(".partial"):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
                _fsync_directory(working_root)
            else:
                raise ValueError("Unsafe partial unknown cache artifact")
    if {child.name for child in working_root.iterdir()} != {"completed", "resume.json"}:
        raise ValueError("Unknown cache resume root contains unexpected artifacts")
    completed = working_root / "completed"
    state, _ = _read_json_snapshot(
        working_root / "resume.json", "unknown cache resume state", private=True
    )
    if (
        state != expected_state
        or completed.is_symlink()
        or not completed.is_dir()
        or stat.S_IMODE(completed.stat().st_mode) != 0o700
    ):
        raise ValueError("Unknown cache resume state differs from the current build")
    for child in list(completed.iterdir()):
        if child.name.startswith(".") and child.name.endswith(".partial"):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
                _fsync_directory(completed)
            else:
                raise ValueError("Unsafe partial unknown recording checkpoint")
    return completed


def _parse_int(row: Mapping[str, str], field: str, context: str) -> int:
    try:
        return int(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"Invalid integer {field} in {context}") from exc


def _parse_float(row: Mapping[str, str], field: str, context: str) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"Invalid float {field} in {context}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {field} in {context}")
    return value


def _validate_index_row(
    row: Mapping[str, str], record: _SelectedRecording
) -> dict[str, int | float]:
    context = record.candidate_id
    expected_feature = f"{SCORING_PARTITION}/features/{record.candidate_id}.npy"
    if set(row) != set(INDEX_FIELDS):
        raise ValueError(f"Unknown cache index schema is invalid: {context}")
    expected_strings = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "candidate_id": record.candidate_id,
        "species_scientific_name": record.scientific_name,
        "species_common_name": record.common_name,
        "difficulty_group": record.difficulty_group,
        "session_group": record.session_group,
        "relative_path": record.relative_path,
        "source_sha256": record.source_sha256,
        "feature_file": expected_feature,
    }
    if any(row.get(field) != value for field, value in expected_strings.items()):
        raise ValueError(f"Unknown cache row source binding is invalid: {context}")
    feature_sha256 = row.get("feature_file_sha256", "")
    if _SHA256.fullmatch(feature_sha256) is None:
        raise ValueError(f"Unknown cache feature SHA-256 is invalid: {context}")

    integer_fields = (
        "species_index",
        "selection_rank",
        "source_file_size_bytes",
        "feature_row",
        "energy_clip_count",
        "energy_rank",
        "start_sample",
        "valid_samples",
        "left_padding_samples",
        "right_padding_samples",
        "decoded_samples",
    )
    values: dict[str, int | float] = {
        field: _parse_int(row, field, context) for field in integer_fields
    }
    if (
        values["species_index"] != record.species_index
        or values["selection_rank"] != record.selection_rank
        or values["source_file_size_bytes"] != record.source_file_size_bytes
    ):
        raise ValueError(f"Unknown cache selected-recording binding drifted: {context}")
    energy = _parse_float(row, "energy_value", context)
    valid_fraction = _parse_float(row, "valid_audio_fraction", context)
    decoded_duration = _parse_float(row, "decoded_duration_seconds", context)
    audit_duration = _parse_float(row, "audit_decoded_duration_seconds", context)
    duration_ratio = _parse_float(row, "decoded_to_audit_duration_ratio", context)
    values.update(
        {
            "energy_value": energy,
            "valid_audio_fraction": valid_fraction,
            "decoded_duration_seconds": decoded_duration,
            "audit_decoded_duration_seconds": audit_duration,
            "decoded_to_audit_duration_ratio": duration_ratio,
        }
    )

    start = int(values["start_sample"])
    valid = int(values["valid_samples"])
    left = int(values["left_padding_samples"])
    right = int(values["right_padding_samples"])
    decoded = int(values["decoded_samples"])
    clip_count = int(values["energy_clip_count"])
    rank = int(values["energy_rank"])
    feature_row = int(values["feature_row"])
    if (
        decoded <= 0
        or start < 0
        or min(valid, left, right, feature_row, rank) < 0
        or valid + left + right != CLIP_SAMPLES
        or not 1 <= clip_count <= MAXIMUM_CLIPS_PER_RECORDING
        or rank >= clip_count
        or feature_row != rank
        or energy < 0
        or abs(valid_fraction - valid / CLIP_SAMPLES) > 1e-9
        or abs(decoded_duration - decoded / TARGET_SAMPLE_RATE_HZ) > 1e-9
        or abs(audit_duration - record.audit_decoded_duration_seconds) > 1e-9
        or audit_duration <= 0
        or abs(duration_ratio - decoded_duration / audit_duration) > 2e-9
    ):
        raise ValueError(f"Unknown cache clip arithmetic is invalid: {context}:{rank}")
    if decoded < CLIP_SAMPLES:
        missing = CLIP_SAMPLES - decoded
        expected_padding = (0, decoded, missing // 2, missing - missing // 2)
        if (start, valid, left, right) != expected_padding:
            raise ValueError(f"Unknown cache short clip padding is invalid: {context}")
    elif start > decoded - CLIP_SAMPLES or valid != CLIP_SAMPLES or left or right:
        raise ValueError(f"Unknown cache clip bounds are invalid: {context}:{rank}")
    if start not in set(energy_candidate_starts(decoded)):
        raise ValueError(f"Unknown cache clip start is outside the locked energy grid: {context}")
    if row.get("clip_id") != f"{record.candidate_id}:{start:012d}":
        raise ValueError(f"Unknown cache clip ID is invalid: {context}:{rank}")
    return values


def _read_verified_feature_tensor(path: Path, expected_sha256: str) -> tuple[np.ndarray, int]:
    payload, digest, size = _read_bound_file(
        path,
        f"unknown feature {path.name}",
        private=True,
        collect_bytes=True,
    )
    if payload is None or digest != expected_sha256:
        raise ValueError(f"Unknown feature hash drift: {path}")
    tensor = np.load(io.BytesIO(payload), allow_pickle=False)
    current = _bound_artifact(path, f"unknown feature {path.name}", private=True)
    if current.sha256 != digest or current.size_bytes != size:
        raise ValueError(f"Unknown feature changed while its NPY snapshot was parsed: {path}")
    if (
        tensor.dtype != np.float32
        or tensor.ndim != 4
        or tuple(tensor.shape[1:]) != NATIVE_FEATURE_SHAPE
        or tensor.shape[0] < 1
        or tensor.shape[0] > MAXIMUM_CLIPS_PER_RECORDING
        or not bool(np.all(np.isfinite(tensor)))
        or float(tensor.min()) < 0
        or float(tensor.max()) > 1
    ):
        raise ValueError(f"Unknown feature tensor contract is invalid: {path}")
    return np.ascontiguousarray(tensor, dtype=np.float32), size


def _checkpoint_statistics(rows: Sequence[Mapping[str, Any]], feature_bytes: int) -> dict[str, int]:
    return {"clips": len(rows), "feature_bytes": feature_bytes}


def _validate_recording_checkpoint(
    directory: Path,
    record: _SelectedRecording,
    build_identity_sha256: str,
    *,
    raw_root: Path | None = None,
    ffmpeg: Path | None = None,
    minimum_duration_ratio: float | None = None,
    maximum_duration_ratio: float | None = None,
    recompute: bool = False,
) -> tuple[dict[str, Any], Path]:
    checkpoint_path = directory / "checkpoint.json"
    feature_path = directory / "feature.npy"
    if (
        directory.name != record.candidate_id
        or directory.is_symlink()
        or not directory.is_dir()
        or stat.S_IMODE(directory.stat().st_mode) != 0o700
        or checkpoint_path.is_symlink()
        or feature_path.is_symlink()
        or not checkpoint_path.is_file()
        or not feature_path.is_file()
        or {child.name for child in directory.iterdir()} != {"checkpoint.json", "feature.npy"}
    ):
        raise ValueError(f"Unknown cache resume checkpoint is unsafe: {record.candidate_id}")
    checkpoint, _ = _read_json_snapshot(
        checkpoint_path,
        f"unknown cache checkpoint {record.candidate_id}",
        private=True,
    )
    expected_fields = {
        "schema_version",
        "cache_version",
        "build_identity_sha256",
        "candidate_id",
        "scientific_name",
        "index_rows",
        "feature_record",
        "statistics",
    }
    if (
        set(checkpoint) != expected_fields
        or checkpoint.get("schema_version") != RESUME_SCHEMA_VERSION
        or checkpoint.get("cache_version") != CACHE_VERSION
        or checkpoint.get("build_identity_sha256") != build_identity_sha256
        or checkpoint.get("candidate_id") != record.candidate_id
        or checkpoint.get("scientific_name") != record.scientific_name
        or not isinstance(checkpoint.get("index_rows"), list)
        or not isinstance(checkpoint.get("feature_record"), dict)
        or not isinstance(checkpoint.get("statistics"), dict)
    ):
        raise ValueError(f"Unknown cache checkpoint binding is invalid: {record.candidate_id}")
    raw_rows = checkpoint["index_rows"]
    if not raw_rows:
        raise ValueError(f"Unknown cache checkpoint has no rows: {record.candidate_id}")
    rows: list[dict[str, str]] = []
    values: list[dict[str, int | float]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or set(raw_row) != set(INDEX_FIELDS):
            raise ValueError(f"Unknown resume index schema is invalid: {record.candidate_id}")
        row = {field: str(raw_row[field]) for field in INDEX_FIELDS}
        rows.append(row)
        values.append(_validate_index_row(row, record))
    clip_count = len(rows)
    if (
        any(int(value["energy_clip_count"]) != clip_count for value in values)
        or [int(value["energy_rank"]) for value in values] != list(range(clip_count))
        or [int(value["feature_row"]) for value in values] != list(range(clip_count))
        or len({int(value["start_sample"]) for value in values}) != clip_count
    ):
        raise ValueError(f"Unknown resume energy ranks are invalid: {record.candidate_id}")
    ranked = [(-float(value["energy_value"]), int(value["start_sample"])) for value in values]
    starts = [int(value["start_sample"]) for value in values]
    if ranked != sorted(ranked) or any(
        abs(left - right) < MINIMUM_SELECTED_START_SEPARATION_SAMPLES
        for index, left in enumerate(starts)
        for right in starts[index + 1 :]
    ):
        raise ValueError(f"Unknown resume energy selection is invalid: {record.candidate_id}")
    expected_feature = f"{SCORING_PARTITION}/features/{record.candidate_id}.npy"
    feature_record = checkpoint["feature_record"]
    if (
        set(feature_record) != {"path", "sha256"}
        or feature_record.get("path") != expected_feature
        or _SHA256.fullmatch(str(feature_record.get("sha256") or "")) is None
        or any(row["feature_file"] != expected_feature for row in rows)
        or any(row["feature_file_sha256"] != feature_record["sha256"] for row in rows)
    ):
        raise ValueError(f"Unknown resume feature binding is invalid: {record.candidate_id}")
    tensor, feature_bytes = _read_verified_feature_tensor(feature_path, feature_record["sha256"])
    if tensor.shape[0] != clip_count:
        raise ValueError(f"Unknown resume feature row count is invalid: {record.candidate_id}")
    if checkpoint["statistics"] != _checkpoint_statistics(raw_rows, feature_bytes):
        raise ValueError(f"Unknown resume statistics are invalid: {record.candidate_id}")
    if recompute:
        if (
            raw_root is None
            or ffmpeg is None
            or minimum_duration_ratio is None
            or maximum_duration_ratio is None
        ):
            raise RuntimeError("Unknown resume recomputation lacks its sealed runtime inputs")
        expected_rows, expected_tensor, expected_feature, expected_statistics = _derive_recording(
            record,
            raw_root,
            ffmpeg,
            minimum_duration_ratio,
            maximum_duration_ratio,
        )
        if (
            raw_rows != expected_rows
            or feature_record != expected_feature
            or checkpoint["statistics"] != expected_statistics
            or tensor.shape != expected_tensor.shape
            or not np.array_equal(tensor, expected_tensor)
        ):
            raise ValueError(
                f"Unknown resume derivation differs from sealed audio: {record.candidate_id}"
            )
    return checkpoint, feature_path


def _disk_preflight(
    parent: Path,
    remaining_recordings: int,
    publication_recordings: int,
) -> dict[str, int]:
    bytes_per_clip = int(np.prod(NATIVE_FEATURE_SHAPE)) * np.dtype(np.float32).itemsize
    maximum_feature_bytes = (
        (remaining_recordings + publication_recordings)
        * MAXIMUM_CLIPS_PER_RECORDING
        * (bytes_per_clip + 512)
    )
    required_free_bytes = int(maximum_feature_bytes * 1.25) + 64 * 1024 * 1024
    available_free_bytes = shutil.disk_usage(parent).free
    if available_free_bytes < required_free_bytes:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient free space for unknown cache: need {required_free_bytes}, "
            f"available {available_free_bytes}",
            str(parent),
        )
    return {
        "required_free_bytes": required_free_bytes,
        "available_free_bytes": available_free_bytes,
    }


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    payload: Mapping[str, Any],
) -> None:
    if callback is not None:
        callback(dict(payload))


def _build_into_staging(
    staging_root: Path,
    inputs: _ValidatedInputs,
    provenance: Mapping[str, Any],
    recording_results: Sequence[tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    index_rows: list[dict[str, Any]] = []
    feature_records: list[dict[str, str]] = []
    feature_bytes = 0
    species_statistics: dict[str, dict[str, Any]] = {
        row.scientific_name: {
            "common_name": row.common_name,
            "species_index": row.species_index,
            "recordings": 0,
            "clips": 0,
        }
        for row in inputs.selected
    }
    for checkpoint, source_feature in recording_results:
        feature_record = dict(checkpoint["feature_record"])
        target_feature = staging_root / feature_record["path"]
        target_feature.parent.mkdir(parents=True, exist_ok=True)
        _copy_file_create_only(source_feature, target_feature, feature_record["sha256"])
        rows = list(checkpoint["index_rows"])
        index_rows.extend(rows)
        feature_records.append(feature_record)
        statistics = checkpoint["statistics"]
        feature_bytes += int(statistics["feature_bytes"])
        species_name = str(checkpoint["scientific_name"])
        species_statistics[species_name]["recordings"] += 1
        species_statistics[species_name]["clips"] += int(statistics["clips"])

    scoring_root = staging_root / SCORING_PARTITION
    (scoring_root / "features").mkdir(parents=True, exist_ok=True)
    _fsync_directory(scoring_root)
    index_path = scoring_root / "index.csv"
    _atomic_write_csv_durable(index_path, index_rows, INDEX_FIELDS)
    feature_records.sort(key=lambda item: item["path"])
    summary = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "partition": "scoring_only",
        "selection_strategy": "energy",
        "feature_dtype": "float32",
        "feature_shape": list(NATIVE_FEATURE_SHAPE),
        "recording_tensor_shape": ["energy_clips", *NATIVE_FEATURE_SHAPE],
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "clip_samples": CLIP_SAMPLES,
        "species": species_statistics,
        "totals": {
            "species": len(species_statistics),
            "recordings": len(recording_results),
            "clips": len(index_rows),
            "feature_files": len(feature_records),
            "feature_bytes": feature_bytes,
        },
    }
    summary_path = staging_root / "summary.json"
    _atomic_write_json_durable(summary_path, summary, private=True)
    summary_artifact = _bound_artifact(summary_path, "unknown cache summary", private=True)
    index_artifact = _bound_artifact(index_path, "unknown cache index", private=True)
    artifacts = {
        "summary": {"path": "summary.json", "sha256": summary_artifact.sha256},
        "index": {
            "path": f"{SCORING_PARTITION}/index.csv",
            "sha256": index_artifact.sha256,
            "rows": len(index_rows),
        },
        "features": {
            "directory": f"{SCORING_PARTITION}/features",
            "files": len(feature_records),
            "feature_set_sha256": sha256_json(feature_records),
        },
    }
    lock = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "provenance": dict(provenance),
        "artifacts": artifacts,
        "cache_content_sha256": sha256_json(
            {"provenance": provenance, "artifacts": artifacts, "summary": summary}
        ),
    }
    _require_inputs_unchanged(inputs)
    _atomic_write_json_durable(staging_root / "lock.json", lock, private=True)
    _require_inputs_unchanged(inputs)
    return summary


def _resolve_cache_artifact(root: Path, relative_path: str, expected: str) -> Path:
    if relative_path != expected or Path(relative_path).is_absolute():
        raise ValueError(f"Unknown cache artifact path is invalid: {relative_path}")
    lexical = root / relative_path
    resolved = resolve_project_path(lexical)
    if resolved != lexical or not is_relative_to(resolved, root):
        raise ValueError(f"Unknown cache artifact traverses a symbolic link: {relative_path}")
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"Unknown cache artifact is missing: {relative_path}")
    return resolved


def _read_index(
    root: Path,
    artifacts: Mapping[str, Any],
    inputs: _ValidatedInputs,
    *,
    verify_feature_bytes: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    _ = verify_feature_bytes
    index_entry = artifacts["index"]
    index_path = _resolve_cache_artifact(
        root,
        str(index_entry.get("path") or ""),
        f"{SCORING_PARTITION}/index.csv",
    )
    fieldnames, rows, index_sha256 = _read_csv_snapshot(index_path, "unknown cache index")
    if index_sha256 != index_entry.get("sha256"):
        raise ValueError("Unknown cache index hash does not match its lock")
    if fieldnames != INDEX_FIELDS:
        raise ValueError("Unknown cache index field order is invalid")
    if len(rows) != index_entry.get("rows"):
        raise ValueError("Unknown cache index row count is invalid")

    selected_by_id = {row.candidate_id: row for row in inputs.selected}
    quality_control = inputs.data_config["quality_control"]
    minimum_duration_ratio = float(quality_control["minimum_decoded_to_ffprobe_duration_ratio"])
    maximum_duration_ratio = float(quality_control["maximum_decoded_to_ffprobe_duration_ratio"])
    groups: dict[str, list[tuple[dict[str, str], dict[str, int | float]]]] = {}
    clip_ids: set[str] = set()
    order: list[tuple[int, int, int]] = []
    for row in rows:
        record = selected_by_id.get(row.get("candidate_id", ""))
        if record is None:
            raise ValueError("Unknown cache index contains an unselected recording")
        values = _validate_index_row(row, record)
        if (
            not minimum_duration_ratio
            <= float(values["decoded_to_audit_duration_ratio"])
            <= maximum_duration_ratio
        ):
            raise ValueError("Unknown cache decoded duration is outside its locked QC bounds")
        if row["clip_id"] in clip_ids:
            raise ValueError("Unknown cache contains a duplicate clip ID")
        clip_ids.add(row["clip_id"])
        groups.setdefault(record.candidate_id, []).append((row, values))
        order.append(
            (
                int(values["species_index"]),
                int(values["selection_rank"]),
                int(values["energy_rank"]),
            )
        )
    if order != sorted(order) or set(groups) != set(selected_by_id):
        raise ValueError("Unknown cache index order or selected recording set is invalid")

    feature_records: list[dict[str, str]] = []
    feature_bytes = 0
    species_statistics: dict[str, dict[str, Any]] = {
        row.scientific_name: {
            "common_name": row.common_name,
            "species_index": row.species_index,
            "recordings": 0,
            "clips": 0,
        }
        for row in inputs.selected
    }
    for record in inputs.selected:
        group = groups[record.candidate_id]
        first_row = group[0][0]
        values = [item[1] for item in group]
        clip_count = len(group)
        constant_fields = {
            "species_scientific_name",
            "species_common_name",
            "species_index",
            "difficulty_group",
            "selection_rank",
            "session_group",
            "relative_path",
            "source_sha256",
            "source_file_size_bytes",
            "feature_file",
            "feature_file_sha256",
            "energy_clip_count",
            "decoded_samples",
            "decoded_duration_seconds",
            "audit_decoded_duration_seconds",
            "decoded_to_audit_duration_ratio",
        }
        starts = [int(value["start_sample"]) for value in values]
        ranked = [
            (-float(value["energy_value"]), start)
            for value, start in zip(values, starts, strict=True)
        ]
        if (
            any(int(value["energy_clip_count"]) != clip_count for value in values)
            or [int(value["energy_rank"]) for value in values] != list(range(clip_count))
            or [int(value["feature_row"]) for value in values] != list(range(clip_count))
            or len(set(starts)) != clip_count
            or ranked != sorted(ranked)
            or any(
                abs(left - right) < MINIMUM_SELECTED_START_SEPARATION_SAMPLES
                for index, left in enumerate(starts)
                for right in starts[index + 1 :]
            )
            or any(
                any(row[field] != first_row[field] for field in constant_fields) for row, _ in group
            )
        ):
            raise ValueError(
                f"Unknown cache per-recording invariants failed: {record.candidate_id}"
            )
        feature_path = _resolve_cache_artifact(
            root, first_row["feature_file"], first_row["feature_file"]
        )
        tensor, current_bytes = _read_verified_feature_tensor(
            feature_path, first_row["feature_file_sha256"]
        )
        tensor_rows = tensor.shape[0]
        if tensor_rows != clip_count:
            raise ValueError(f"Unknown feature row count is invalid: {feature_path}")
        feature_records.append(
            {"path": first_row["feature_file"], "sha256": first_row["feature_file_sha256"]}
        )
        feature_bytes += current_bytes
        species = species_statistics[record.scientific_name]
        species["recordings"] += 1
        species["clips"] += clip_count

    feature_records.sort(key=lambda item: item["path"])
    features_entry = artifacts["features"]
    if (
        set(features_entry) != {"directory", "files", "feature_set_sha256"}
        or features_entry.get("directory") != f"{SCORING_PARTITION}/features"
        or features_entry.get("files") != len(feature_records)
        or features_entry.get("feature_set_sha256") != sha256_json(feature_records)
    ):
        raise ValueError("Unknown feature set binding is invalid")
    feature_directory = root / SCORING_PARTITION / "features"
    if (
        not feature_directory.is_dir()
        or feature_directory.is_symlink()
        or feature_directory.resolve() != feature_directory
    ):
        raise ValueError("Unknown feature directory is invalid")
    children = list(feature_directory.iterdir())
    actual_files = {
        child.relative_to(root).as_posix()
        for child in children
        if stat.S_ISREG(child.lstat().st_mode)
    }
    expected_files = {row["path"] for row in feature_records}
    if actual_files != expected_files or any(
        child.is_symlink()
        or not stat.S_ISREG(child.lstat().st_mode)
        or child.lstat().st_nlink != 1
        or stat.S_IMODE(child.lstat().st_mode) != 0o600
        for child in children
    ):
        raise ValueError("Unknown physical feature file set is not exact")
    statistics = {
        "species": species_statistics,
        "totals": {
            "species": len(species_statistics),
            "recordings": len(groups),
            "clips": len(rows),
            "feature_files": len(feature_records),
            "feature_bytes": feature_bytes,
        },
    }
    return rows, statistics


def _verify_energy_derivations(
    root: Path,
    rows: Sequence[Mapping[str, str]],
    inputs: _ValidatedInputs,
    ffmpeg: Path,
) -> None:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    quality_control = inputs.data_config["quality_control"]
    minimum_ratio = float(quality_control["minimum_decoded_to_ffprobe_duration_ratio"])
    maximum_ratio = float(quality_control["maximum_decoded_to_ffprobe_duration_ratio"])
    for record in inputs.selected:
        expected_rows, expected_tensor, feature_record, _ = _derive_recording(
            record,
            inputs.raw_root,
            ffmpeg,
            minimum_ratio,
            maximum_ratio,
        )
        expected_csv_rows = [
            {field: str(row[field]) for field in INDEX_FIELDS} for row in expected_rows
        ]
        observed_rows = [dict(row) for row in grouped[record.candidate_id]]
        feature_path = _resolve_cache_artifact(
            root,
            feature_record["path"],
            feature_record["path"],
        )
        observed_tensor, _ = _read_verified_feature_tensor(
            feature_path,
            feature_record["sha256"],
        )
        if (
            observed_rows != expected_csv_rows
            or observed_tensor.shape != expected_tensor.shape
            or not np.array_equal(observed_tensor, expected_tensor)
        ):
            raise ValueError(
                f"Unknown cache feature derivation is not reproducible: {record.candidate_id}"
            )


def _validate_lock_structure(lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(lock) != _LOCK_FIELDS:
        raise ValueError("Unknown cache lock fields are not exact")
    if (
        lock.get("schema_version") != CACHE_SCHEMA_VERSION
        or lock.get("cache_version") != CACHE_VERSION
    ):
        raise ValueError("Unknown cache lock schema or version is unsupported")
    provenance = lock.get("provenance")
    artifacts = lock.get("artifacts")
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("Unknown cache provenance fields are not exact")
    if (
        not isinstance(provenance.get("input_paths"), dict)
        or set(provenance["input_paths"]) != _INPUT_PATH_FIELDS
        or not isinstance(provenance.get("runtime"), dict)
        or set(provenance["runtime"]) != _RUNTIME_FIELDS
    ):
        raise ValueError("Unknown cache provenance structure is invalid")
    hash_fields = _PROVENANCE_FIELDS - {"runtime", "input_paths"}
    if any(_SHA256.fullmatch(str(provenance.get(field) or "")) is None for field in hash_fields):
        raise ValueError("Unknown cache provenance contains a malformed SHA-256")
    if not isinstance(artifacts, dict) or set(artifacts) != {"summary", "index", "features"}:
        raise ValueError("Unknown cache artifact fields are not exact")
    summary = artifacts["summary"]
    index = artifacts["index"]
    features = artifacts["features"]
    if (
        not isinstance(summary, dict)
        or set(summary) != {"path", "sha256"}
        or summary.get("path") != "summary.json"
        or _SHA256.fullmatch(str(summary.get("sha256") or "")) is None
        or not isinstance(index, dict)
        or set(index) != {"path", "sha256", "rows"}
        or index.get("path") != f"{SCORING_PARTITION}/index.csv"
        or _SHA256.fullmatch(str(index.get("sha256") or "")) is None
        or not isinstance(index.get("rows"), int)
        or index["rows"] < TARGET_RECORDINGS
        or not isinstance(features, dict)
        or set(features) != {"directory", "files", "feature_set_sha256"}
        or features.get("directory") != f"{SCORING_PARTITION}/features"
        or features.get("files") != TARGET_RECORDINGS
        or _SHA256.fullmatch(str(features.get("feature_set_sha256") or "")) is None
    ):
        raise ValueError("Unknown cache artifact bindings are invalid")
    return provenance, artifacts


def _validate_summary(summary: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "cache_version",
        "partition",
        "selection_strategy",
        "feature_dtype",
        "feature_shape",
        "recording_tensor_shape",
        "sample_rate_hz",
        "clip_samples",
        "species",
        "totals",
    }
    if (
        set(summary) != expected_keys
        or summary.get("schema_version") != CACHE_SCHEMA_VERSION
        or summary.get("cache_version") != CACHE_VERSION
        or summary.get("partition") != "scoring_only"
        or summary.get("selection_strategy") != "energy"
        or summary.get("feature_dtype") != "float32"
        or summary.get("feature_shape") != list(NATIVE_FEATURE_SHAPE)
        or summary.get("recording_tensor_shape") != ["energy_clips", *NATIVE_FEATURE_SHAPE]
        or summary.get("sample_rate_hz") != TARGET_SAMPLE_RATE_HZ
        or summary.get("clip_samples") != CLIP_SAMPLES
        or not isinstance(summary.get("species"), dict)
        or not isinstance(summary.get("totals"), dict)
    ):
        raise ValueError("Unknown cache summary signal contract is invalid")


def _validate_publishing_tree(
    root: Path,
    expected_summary: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    inputs: _ValidatedInputs,
) -> None:
    if {child.name for child in root.iterdir()} != {
        SCORING_PARTITION,
        "summary.json",
        "lock.json",
    }:
        raise ValueError("Unknown cache publication root contains unexpected artifacts")
    scoring_children = {child.name for child in (root / SCORING_PARTITION).iterdir()}
    if scoring_children != {"index.csv", "features"}:
        raise ValueError("Unknown cache scoring partition is not exact")
    lock, _ = _read_json_snapshot(root / "lock.json", "unknown cache publishing lock", private=True)
    provenance, artifacts = _validate_lock_structure(lock)
    if provenance != expected_provenance:
        raise ValueError("Unknown publishing provenance differs from the current inputs")
    summary_path = _resolve_cache_artifact(root, "summary.json", "summary.json")
    summary, summary_sha256 = _read_json_snapshot(
        summary_path, "unknown cache publishing summary", private=True
    )
    if summary_sha256 != artifacts["summary"]["sha256"]:
        raise ValueError("Unknown publishing summary hash does not match its lock")
    _validate_summary(summary)
    if summary != expected_summary:
        raise ValueError("Unknown publishing summary differs from the constructed summary")
    _, statistics = _read_index(root, artifacts, inputs, verify_feature_bytes=True)
    if summary["species"] != statistics["species"] or summary["totals"] != statistics["totals"]:
        raise ValueError("Unknown publishing summary does not match its artifacts")
    expected_content_sha256 = sha256_json(
        {"provenance": provenance, "artifacts": artifacts, "summary": summary}
    )
    if lock.get("cache_content_sha256") != expected_content_sha256:
        raise ValueError("Unknown publishing content hash is invalid")


def build_unknown_clip_cache(
    cache_root: str | Path = DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
    *,
    ffmpeg: str | Path | None = None,
    audit_path: str | Path = DEFAULT_AUDIT,
    audit_lock_path: str | Path = DEFAULT_AUDIT_LOCK,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    config_path: str | Path = DEFAULT_DATA_CONFIG,
    unknown_audio_config_path: str | Path = DEFAULT_UNKNOWN_AUDIO_CONFIG,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and publish the energy-only cache for sealed unknown scoring audio."""
    _require_project_venv()
    requested = Path(cache_root).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    lexical_destination = Path(os.path.abspath(requested))
    destination = require_safe_output(cache_root)
    if destination != lexical_destination:
        raise ValueError("Unknown cache output path traverses a symbolic link")
    overlapping_v1_root = next(
        (
            root
            for root in PRESERVED_V1_ROOTS
            if is_relative_to(destination, root) or is_relative_to(root, destination)
        ),
        None,
    )
    if overlapping_v1_root is not None:
        raise ValueError(
            f"Unknown cache output overlaps protected v1 evidence: {overlapping_v1_root}"
        )
    if destination.name != CACHE_VERSION:
        raise ValueError(f"Unknown cache root basename must be exactly {CACHE_VERSION}")
    if is_relative_to(destination, RAW_DATA_ROOT):
        raise ValueError("Unknown cache cannot be written inside immutable known audio")
    if destination.exists():
        verification = verify_unknown_clip_cache(destination, ffmpeg=ffmpeg)
        if verification.get("valid") is not True:
            raise RuntimeError(f"Existing unknown clip cache is not valid: {destination}")
        summary, _ = _read_json_snapshot(
            destination / "summary.json", "unknown cache summary", private=True
        )
        return destination, summary
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent:
        raise ValueError("Unknown cache parent traverses a symbolic link")
    _fsync_directory(destination.parent)

    with _unknown_cache_build_lock(destination):
        if destination.exists():
            verification = verify_unknown_clip_cache(destination, ffmpeg=ffmpeg)
            if verification.get("valid") is not True:
                raise RuntimeError(f"Existing unknown clip cache is not valid: {destination}")
            summary, _ = _read_json_snapshot(
                destination / "summary.json", "unknown cache summary", private=True
            )
            return destination, summary
        inputs = _load_validated_inputs(
            audit_path=audit_path,
            audit_lock_path=audit_lock_path,
            checkpoint_root=checkpoint_root,
            config_path=config_path,
            unknown_audio_config_path=unknown_audio_config_path,
        )
        ffmpeg_path = resolve_tool("ffmpeg", ffmpeg)
        ffmpeg_artifact = _bound_artifact(ffmpeg_path, "FFmpeg executable", private=False)
        ffmpeg_sha256 = ffmpeg_artifact.sha256
        implementation_sha256 = _implementation_fingerprint()
        runtime = _runtime_provenance(ffmpeg_path)
        provenance = _cache_provenance(inputs, ffmpeg_sha256, implementation_sha256, runtime)
        build_identity_sha256, recording_order = _resume_identity(provenance, inputs.selected)
        selected_by_id = {row.candidate_id: row for row in inputs.selected}
        working_root = destination.with_name(f".{destination.name}.working")
        completed_root = _prepare_working_directory(
            working_root, build_identity_sha256, recording_order
        )
        quality_control = inputs.data_config["quality_control"]
        minimum_ratio = float(quality_control["minimum_decoded_to_ffprobe_duration_ratio"])
        maximum_ratio = float(quality_control["maximum_decoded_to_ffprobe_duration_ratio"])
        completed_results: dict[str, tuple[dict[str, Any], Path]] = {}
        for child in sorted(completed_root.iterdir(), key=lambda item: item.name):
            record = selected_by_id.get(child.name)
            if record is None:
                raise ValueError(f"Unknown resume checkpoint is unselected: {child.name}")
            completed_results[child.name] = _validate_recording_checkpoint(
                child,
                record,
                build_identity_sha256,
                raw_root=inputs.raw_root,
                ffmpeg=ffmpeg_path,
                minimum_duration_ratio=minimum_ratio,
                maximum_duration_ratio=maximum_ratio,
                recompute=True,
            )
        disk = _disk_preflight(
            destination.parent,
            len(inputs.selected) - len(completed_results),
            len(inputs.selected),
        )
        _emit_progress(
            progress_callback,
            {
                "event": "preflight",
                "recordings_total": len(inputs.selected),
                "recordings_completed": len(completed_results),
                "recordings_remaining": len(inputs.selected) - len(completed_results),
                **disk,
            },
        )

        for record in inputs.selected:
            if record.candidate_id in completed_results:
                _emit_progress(
                    progress_callback,
                    {
                        "event": "recording_complete",
                        "candidate_id": record.candidate_id,
                        "recordings_completed": len(completed_results),
                        "recordings_total": len(inputs.selected),
                        "resumed": True,
                    },
                )
                continue
            partial = Path(
                tempfile.mkdtemp(
                    prefix=f".{record.candidate_id}.",
                    suffix=".partial",
                    dir=completed_root,
                )
            )
            partial.chmod(0o700)
            _fsync_directory(completed_root)
            try:
                rows, feature_record, statistics = _process_recording(
                    record,
                    inputs.raw_root,
                    ffmpeg_path,
                    minimum_ratio,
                    maximum_ratio,
                    partial / "feature.npy",
                )
                checkpoint = {
                    "schema_version": RESUME_SCHEMA_VERSION,
                    "cache_version": CACHE_VERSION,
                    "build_identity_sha256": build_identity_sha256,
                    "candidate_id": record.candidate_id,
                    "scientific_name": record.scientific_name,
                    "index_rows": rows,
                    "feature_record": feature_record,
                    "statistics": statistics,
                }
                _atomic_write_json_durable(partial / "checkpoint.json", checkpoint, private=True)
                completed_directory = completed_root / record.candidate_id
                _atomic_publish_directory_no_replace(partial, completed_directory)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                _fsync_directory(completed_root)
                raise
            completed_results[record.candidate_id] = _validate_recording_checkpoint(
                completed_directory,
                record,
                build_identity_sha256,
                recompute=False,
            )
            _emit_progress(
                progress_callback,
                {
                    "event": "recording_complete",
                    "candidate_id": record.candidate_id,
                    "recordings_completed": len(completed_results),
                    "recordings_total": len(inputs.selected),
                    "resumed": False,
                },
            )

        recording_results = [completed_results[candidate_id] for candidate_id in recording_order]
        staging_root = destination.with_name(f".{destination.name}.publishing")
        if staging_root.exists():
            if staging_root.is_symlink() or not staging_root.is_dir():
                raise ValueError("Interrupted unknown publication path is unsafe")
            shutil.rmtree(staging_root)
            _fsync_directory(staging_root.parent)
        staging_root.mkdir(mode=0o700)
        _fsync_directory(staging_root.parent)
        try:
            summary = _build_into_staging(staging_root, inputs, provenance, recording_results)
            staging_identity = _directory_identity(staging_root)
            _validate_publishing_tree(staging_root, summary, provenance, inputs)
            if _bound_artifact(ffmpeg_path, "FFmpeg executable", private=False) != ffmpeg_artifact:
                raise RuntimeError("FFmpeg changed during unknown cache construction")
            if _implementation_fingerprint() != implementation_sha256:
                raise RuntimeError("Unknown signal implementation changed during construction")
            if _runtime_provenance(ffmpeg_path) != runtime:
                raise RuntimeError("Unknown cache numerical runtime changed during construction")
            _require_inputs_unchanged(inputs)
            if destination.exists():
                raise RuntimeError("Unknown cache destination appeared before publication")
            if _directory_identity(staging_root) != staging_identity:
                raise RuntimeError("Unknown publishing directory changed before publication")
            _atomic_publish_directory_no_replace(staging_root, destination)
            if _directory_identity(destination) != staging_identity:
                raise RuntimeError("Unknown published directory identity is not the staged inode")
            _validate_publishing_tree(destination, summary, provenance, inputs)
            if (
                _bound_artifact(ffmpeg_path, "FFmpeg executable", private=False) != ffmpeg_artifact
                or _implementation_fingerprint() != implementation_sha256
                or _runtime_provenance(ffmpeg_path) != runtime
            ):
                raise RuntimeError("Unknown cache runtime changed across final publication")
            _require_inputs_unchanged(inputs)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            _fsync_directory(staging_root.parent)
            raise
        shutil.rmtree(working_root)
        _fsync_directory(working_root.parent)
        _emit_progress(
            progress_callback,
            {
                "event": "published",
                "recordings": summary["totals"]["recordings"],
                "clips": summary["totals"]["clips"],
                "destination": str(destination),
            },
        )
    return destination, summary


def _resolve_cache_root(cache_root: str | Path) -> Path:
    requested = Path(cache_root).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    lexical = Path(os.path.abspath(requested))
    root = resolve_project_path(cache_root)
    if (
        root != lexical
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve() != root
        or not is_relative_to(root, PROJECT_ROOT)
        or is_relative_to(root, RAW_DATA_ROOT)
        or root.name != CACHE_VERSION
    ):
        raise ValueError(f"Unknown cache root is invalid: {root}")
    return root


def _validate_current_provenance(
    provenance: Mapping[str, Any],
    ffmpeg: str | Path | None,
) -> tuple[_ValidatedInputs, Path]:
    paths = provenance.get("input_paths")
    runtime = provenance.get("runtime")
    if not isinstance(paths, Mapping) or set(paths) != _INPUT_PATH_FIELDS:
        raise ValueError("Unknown cache input paths are not exact")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("Unknown cache runtime fields are not exact")
    for name, value in paths.items():
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError(f"Unknown cache input path is invalid: {name}")
        resolved = resolve_project_path(value)
        if not is_relative_to(resolved, PROJECT_ROOT) or _project_label(resolved) != value:
            raise ValueError(f"Unknown cache input path is noncanonical: {name}")
    inputs = _load_validated_inputs(
        audit_path=paths["audit"],
        audit_lock_path=paths["audit_lock"],
        checkpoint_root=paths["checkpoint_root"],
        config_path=paths["data_config"],
        unknown_audio_config_path=paths["unknown_audio_config"],
    )
    if _project_label(inputs.requirements_lock_file) != paths["requirements_lock"]:
        raise ValueError("Unknown cache requirements path is noncanonical")
    ffmpeg_path = resolve_tool("ffmpeg", ffmpeg)
    expected = _cache_provenance(
        inputs,
        _bound_artifact(ffmpeg_path, "FFmpeg executable", private=False).sha256,
        _implementation_fingerprint(),
        _runtime_provenance(ffmpeg_path),
    )
    if dict(provenance) != expected:
        mismatches = sorted(
            key for key in _PROVENANCE_FIELDS if provenance.get(key) != expected.get(key)
        )
        raise ValueError(f"Unknown cache provenance is stale: {mismatches}")
    return inputs, ffmpeg_path


def _load_cache_metadata(
    cache_root: str | Path,
    *,
    ffmpeg: str | Path | None,
    expected_lock_sha256: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any], _ValidatedInputs]:
    _require_project_venv()
    root = _resolve_cache_root(cache_root)
    if {child.name for child in root.iterdir()} != {
        SCORING_PARTITION,
        "summary.json",
        "lock.json",
    }:
        raise ValueError("Unknown cache root contains unexpected artifacts")
    scoring_root = root / SCORING_PARTITION
    if (
        scoring_root.is_symlink()
        or not scoring_root.is_dir()
        or scoring_root.resolve() != scoring_root
        or {child.name for child in scoring_root.iterdir()} != {"index.csv", "features"}
    ):
        raise ValueError("Unknown cache scoring partition is invalid")
    lock_path = root / "lock.json"
    lock, lock_sha256 = _read_json_snapshot(lock_path, "unknown cache lock", private=True)
    if expected_lock_sha256 is not None:
        if _SHA256.fullmatch(expected_lock_sha256) is None:
            raise ValueError("Expected unknown cache lock SHA-256 is malformed")
        if lock_sha256 != expected_lock_sha256:
            raise ValueError("Unknown cache lock does not match the expected SHA-256")
    provenance, artifacts = _validate_lock_structure(lock)
    inputs, _ = _validate_current_provenance(provenance, ffmpeg)
    summary_path = _resolve_cache_artifact(root, "summary.json", "summary.json")
    summary, summary_sha256 = _read_json_snapshot(
        summary_path, "unknown cache summary", private=True
    )
    if summary_sha256 != artifacts["summary"]["sha256"]:
        raise ValueError("Unknown cache summary hash does not match its lock")
    _validate_summary(summary)
    expected_content_sha256 = sha256_json(
        {"provenance": provenance, "artifacts": artifacts, "summary": summary}
    )
    if lock.get("cache_content_sha256") != expected_content_sha256:
        raise ValueError("Unknown cache content hash does not match its locked artifacts")
    _require_inputs_unchanged(inputs)
    return root, lock, summary, inputs


def verify_unknown_clip_cache(
    cache_root: str | Path = DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
    *,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every selected source, index row, and unknown scoring feature."""
    root, lock, summary, inputs = _load_cache_metadata(
        cache_root,
        ffmpeg=ffmpeg,
        expected_lock_sha256=expected_lock_sha256,
    )
    _require_inputs_unchanged(inputs)
    rows, statistics = _read_index(root, lock["artifacts"], inputs, verify_feature_bytes=True)
    ffmpeg_path = resolve_tool("ffmpeg", ffmpeg)
    _verify_energy_derivations(root, rows, inputs, ffmpeg_path)
    _require_inputs_unchanged(inputs)
    if summary["species"] != statistics["species"] or summary["totals"] != statistics["totals"]:
        raise ValueError("Unknown cache summary does not match verified artifacts")
    totals = statistics["totals"]
    lock_sha256 = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
    return {
        "valid": True,
        "cache_version": CACHE_VERSION,
        "scoring_only": True,
        "selection_strategy": "energy",
        "lock_sha256": lock_sha256,
        "species": totals["species"],
        "recordings": totals["recordings"],
        "clips": totals["clips"],
        "feature_files": totals["feature_files"],
    }


class UnknownScoringClipCache(Sequence[tuple[np.ndarray, dict[str, str]]]):
    """Read-only energy features for sealed unknown scoring records."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        ffmpeg: str | Path | None = None,
        expected_lock_sha256: str | None = None,
    ) -> None:
        root, lock, summary, inputs = _load_cache_metadata(
            cache_root,
            ffmpeg=ffmpeg,
            expected_lock_sha256=expected_lock_sha256,
        )
        rows, statistics = _read_index(root, lock["artifacts"], inputs, verify_feature_bytes=True)
        _require_inputs_unchanged(inputs)
        if summary["species"] != statistics["species"] or summary["totals"] != statistics["totals"]:
            raise ValueError("Unknown cache summary does not match scoring artifacts")
        self.root = root
        self.rows = tuple(rows)
        self.lock_sha256 = hashlib.sha256(_canonical_json_bytes(lock)).hexdigest()
        self.scoring_only = True
        self.selection_strategy = "energy"
        self._loaded_feature_path: Path | None = None
        self._loaded_feature_tensor: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]:
        row = self.rows[index]
        feature_path = _resolve_cache_artifact(self.root, row["feature_file"], row["feature_file"])
        if feature_path != self._loaded_feature_path:
            self._loaded_feature_tensor, _ = _read_verified_feature_tensor(
                feature_path, row["feature_file_sha256"]
            )
            self._loaded_feature_path = feature_path
        if self._loaded_feature_tensor is None:
            raise RuntimeError("Unknown scoring feature tensor was not loaded")
        feature = self._loaded_feature_tensor[int(row["feature_row"])].copy()
        metadata = dict(row)
        metadata["selection_strategy"] = "energy"
        metadata["data_boundary"] = "unknown_scoring_only"
        return feature, metadata


def load_unknown_scoring_clip_cache(
    cache_root: str | Path = DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
    *,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
) -> UnknownScoringClipCache:
    """Open the sealed unknown cache through its scoring-only boundary."""
    return UnknownScoringClipCache(
        cache_root,
        ffmpeg=ffmpeg,
        expected_lock_sha256=expected_lock_sha256,
    )
