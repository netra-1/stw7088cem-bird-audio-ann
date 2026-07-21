from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from bird_audio.audio import resolve_tool
from bird_audio.clip_selection import (
    ENERGY_CANDIDATE_HOP_SAMPLES,
    MAXIMUM_CLIPS_PER_RECORDING,
    MINIMUM_SELECTED_START_SEPARATION_SAMPLES,
    select_energy_candidates,
    uniform_clip_starts,
)
from bird_audio.config import config_fingerprint, load_toml
from bird_audio.hashing import fingerprint_files, sha256_file, sha256_json
from bird_audio.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_snapshot,
    require_unchanged,
)
from bird_audio.locking import project_lock
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
    decode_audio_ffmpeg,
    iter_extracted_clips,
    native_log_mel_spectrogram,
)
from bird_audio.splitting import SPLIT_NAMES, validate_frozen_split

CACHE_SCHEMA_VERSION = "1.0"
CACHE_VERSION = "known_clips_v1"
RESUME_SCHEMA_VERSION = "1.0"
DEFAULT_CACHE_ROOT = "data/processed/known_clips_v1"
DEFAULT_REVIEW_LOCK = "data/manifests/review_v1_lock.json"
DEVELOPMENT_SPLITS = ("train", "validation")
NATIVE_FEATURE_SHAPE = (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH)

INDEX_FIELDS = [
    "schema_version",
    "clip_id",
    "recording_id",
    "relative_path",
    "source_sha256",
    "species_common_name",
    "class_index",
    "session_group",
    "split",
    "feature_file",
    "feature_file_sha256",
    "feature_row",
    "cached_clip_count",
    "uniform_clip_count",
    "energy_clip_count",
    "start_sample",
    "valid_samples",
    "valid_audio_fraction",
    "left_padding_samples",
    "right_padding_samples",
    "uniform_selected",
    "uniform_rank",
    "energy_selected",
    "energy_rank",
    "energy_value",
    "decoded_samples",
    "decoded_duration_seconds",
    "manifest_probe_duration_seconds",
    "decoded_to_probe_duration_ratio",
]

_SAFE_RECORDING_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VERSIONED_ROOT = re.compile(r".+_v[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_LOCK_FIELDS = {
    "schema_version",
    "cache_version",
    "provenance",
    "artifacts",
    "cache_content_sha256",
}
_PROVENANCE_FIELDS = {
    "config_file_sha256",
    "config_sha256",
    "final_manifest_sha256",
    "review_lock_sha256",
    "split_sha256",
    "split_summary_sha256",
    "split_lock_sha256",
    "ffmpeg_executable_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "runtime",
    "input_paths",
}
_INPUT_PATH_FIELDS = {
    "config",
    "final_manifest",
    "review_lock",
    "split",
    "split_summary",
    "split_lock",
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
class _ValidatedInputs:
    config: dict[str, Any]
    config_file: Path
    manifest_file: Path
    split_file: Path
    split_summary_file: Path
    split_lock_file: Path
    review_lock_file: Path
    requirements_lock_file: Path
    manifest_rows: tuple[dict[str, str], ...]
    split_rows: tuple[dict[str, str], ...]
    artifact_hashes: dict[Path, str]
    config_sha256: str
    manifest_sha256: str
    split_sha256: str
    split_summary_sha256: str
    split_lock_sha256: str
    review_lock_sha256: str
    requirements_lock_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_input(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    if not is_relative_to(resolved, PROJECT_ROOT):
        raise ValueError(f"Cache provenance input leaves the project root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Required cache input is not a regular file: {resolved}")
    return resolved


def _require_artifacts_unchanged(artifacts: dict[Path, str]) -> None:
    for path, digest in artifacts.items():
        require_unchanged(path, digest)


def _require_project_venv() -> None:
    expected = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(f"Cache construction must run inside the project virtualenv: {expected}")


def _assert_locked_cache_config(config: dict[str, Any]) -> None:
    clip = config["clip_selection"]
    spectrogram = config["spectrogram"]
    expected = {
        "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "target_channels": 1,
        "audio_dtype": "float32",
        "clip_duration_seconds": CLIP_DURATION_SECONDS,
        "maximum_clips_per_recording": MAXIMUM_CLIPS_PER_RECORDING,
        "uniform_clip_count_formula": "min(5,max(1,floor(duration_seconds/3)))",
        "uniform_start_rule": "linspace_0_to_duration_minus_3",
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
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise ValueError(
            f"Data configuration differs from the implemented cache contract: {mismatches}"
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
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to capture FFmpeg runtime identity: {completed.stderr.strip()}")
    version_output = "\n".join(line.rstrip() for line in completed.stdout.splitlines()).strip()
    if not version_output:
        raise RuntimeError("FFmpeg returned an empty runtime identity")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "librosa_version": librosa.__version__,
        "ffmpeg_version_output": version_output,
    }


def _project_label(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _load_validated_inputs(
    config_path: str | Path,
    manifest_path: str | Path | None,
    split_path: str | Path | None,
    split_summary_path: str | Path | None,
    split_lock_path: str | Path | None,
    review_lock_path: str | Path,
) -> _ValidatedInputs:
    config_file = _resolve_input(config_path)
    config_file_sha256 = sha256_file(config_file)
    config = load_toml(config_file)
    require_unchanged(config_file, config_file_sha256)
    _assert_locked_cache_config(config)

    manifest_file = _resolve_input(manifest_path or str(config["enriched_manifest"]))
    split_file = _resolve_input(split_path or str(config["split_manifest"]))
    split_summary_file = _resolve_input(split_summary_path or str(config["split_summary"]))
    split_lock_file = _resolve_input(split_lock_path or str(config["split_lock"]))
    review_lock_file = _resolve_input(review_lock_path)
    requirements_lock_file = _resolve_input("requirements.lock")
    artifact_hashes = {
        path: sha256_file(path)
        for path in (
            config_file,
            manifest_file,
            split_file,
            split_summary_file,
            split_lock_file,
            review_lock_file,
            requirements_lock_file,
        )
    }

    validation = validate_frozen_split(
        manifest_file,
        split_file,
        split_lock_file,
        config_path=config_file,
        summary_path=split_summary_file,
        review_lock_path=review_lock_file,
    )
    _require_artifacts_unchanged(artifact_hashes)
    if validation.get("valid") is not True:
        failed = sorted(
            name for name, passed in dict(validation.get("checks") or {}).items() if not passed
        )
        raise ValueError(f"Frozen split provenance validation failed: {failed}")

    manifest_rows, manifest_sha256 = read_csv_snapshot(manifest_file)
    split_rows, split_sha256 = read_csv_snapshot(split_file)
    if manifest_sha256 != artifact_hashes[manifest_file]:
        raise RuntimeError("Final reviewed manifest changed after provenance validation")
    if split_sha256 != artifact_hashes[split_file]:
        raise RuntimeError("Frozen split changed after provenance validation")

    split_lock = _read_json(split_lock_file)
    review_lock = _read_json(review_lock_file)
    expected_bindings = {
        "source_manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "summary_sha256": artifact_hashes[split_summary_file],
        "review_lock_sha256": artifact_hashes[review_lock_file],
        "config_sha256": config_fingerprint(config),
    }
    mismatches = [
        key for key, expected in expected_bindings.items() if split_lock.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"Split lock changed or lost required bindings: {mismatches}")
    if review_lock.get("final_manifest_sha256") != manifest_sha256:
        raise ValueError("Review lock does not bind the exact final manifest")

    return _ValidatedInputs(
        config=config,
        config_file=config_file,
        manifest_file=manifest_file,
        split_file=split_file,
        split_summary_file=split_summary_file,
        split_lock_file=split_lock_file,
        review_lock_file=review_lock_file,
        requirements_lock_file=requirements_lock_file,
        manifest_rows=tuple(manifest_rows),
        split_rows=tuple(split_rows),
        artifact_hashes=artifact_hashes,
        config_sha256=config_fingerprint(config),
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        split_summary_sha256=artifact_hashes[split_summary_file],
        split_lock_sha256=artifact_hashes[split_lock_file],
        review_lock_sha256=artifact_hashes[review_lock_file],
        requirements_lock_sha256=artifact_hashes[requirements_lock_file],
    )


def _validate_manifest_and_split_rows(
    inputs: _ValidatedInputs,
) -> tuple[list[tuple[dict[str, str], dict[str, str]]], dict[str, int]]:
    included: dict[str, dict[str, str]] = {}
    for row in inputs.manifest_rows:
        if row.get("local_qc_status") != "include":
            continue
        recording_id = row.get("recording_id", "")
        if not recording_id or recording_id in included:
            raise ValueError(
                f"Included manifest has a duplicate or empty recording ID: {recording_id}"
            )
        if row.get("probe_ok") != "true" or row.get("full_decode_status") != "ok":
            raise ValueError(f"Included recording lacks accepted decode evidence: {recording_id}")
        try:
            probe_duration = float(row.get("ffprobe_duration_seconds") or 0)
        except ValueError as exc:
            raise ValueError(f"Invalid probe duration for {recording_id}") from exc
        if not np.isfinite(probe_duration) or probe_duration <= 0:
            raise ValueError(f"Invalid probe duration for {recording_id}")
        included[recording_id] = row

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    seen: set[str] = set()
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    for split_row in inputs.split_rows:
        recording_id = split_row.get("recording_id", "")
        if recording_id in seen:
            raise ValueError(f"Split contains duplicate recording ID: {recording_id}")
        seen.add(recording_id)
        manifest_row = included.get(recording_id)
        if manifest_row is None:
            raise ValueError(
                f"Split recording is not included in the final manifest: {recording_id}"
            )
        split_name = split_row.get("split", "")
        if split_name not in split_order:
            raise ValueError(f"Invalid split for {recording_id}: {split_name}")
        for field in ("relative_path", "sha256", "species_common_name", "session_group"):
            if split_row.get(field) != manifest_row.get(field):
                raise ValueError(f"Split-to-manifest binding drift: {recording_id}:{field}")
        if split_row.get("source_manifest_sha256") != inputs.manifest_sha256:
            raise ValueError(f"Split source-manifest binding drift: {recording_id}")
        pairs.append((split_row, manifest_row))
    if seen != set(included):
        raise ValueError("Frozen split recording set differs from included final-manifest rows")

    class_names = [str(entry["common_name"]) for entry in inputs.config["known_species"]]
    class_indices = {name: index for index, name in enumerate(class_names)}
    if len(class_indices) != len(class_names):
        raise ValueError("Known-species class order is not unique")
    if any(row[0]["species_common_name"] not in class_indices for row in pairs):
        raise ValueError("Frozen split contains a species outside the locked class order")

    pairs.sort(
        key=lambda pair: (
            split_order[pair[0]["split"]],
            class_indices[pair[0]["species_common_name"]],
            pair[0]["recording_id"],
        )
    )
    return pairs, class_indices


def _resolve_and_verify_raw_file(row: dict[str, str]) -> Path:
    recording_id = row.get("recording_id", "unknown")
    relative_path = row.get("relative_path", "")
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"Unsafe raw path for {recording_id}")
    try:
        path = resolve_project_path(relative_path)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Raw path cannot be resolved for {recording_id}") from exc
    if not is_relative_to(path, RAW_DATA_ROOT):
        raise ValueError(f"Raw path leaves the immutable dataset root for {recording_id}")
    if path.relative_to(PROJECT_ROOT).as_posix() != relative_path:
        raise ValueError(f"Raw path is noncanonical or drifted for {recording_id}")
    if not path.is_file():
        raise FileNotFoundError(f"Raw audio is not an existing regular file for {recording_id}")
    expected_sha256 = row.get("sha256", "")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError(f"Raw SHA-256 drift for {recording_id}")
    return path


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _process_recording(
    staging_root: Path,
    split_row: dict[str, str],
    manifest_row: dict[str, str],
    class_index: int,
    ffmpeg: Path,
    minimum_duration_ratio: float,
    maximum_duration_ratio: float,
    feature_output_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    recording_id = split_row["recording_id"]
    if _SAFE_RECORDING_ID.fullmatch(recording_id) is None:
        raise ValueError(f"Recording ID is unsafe for a feature filename: {recording_id}")
    raw_path = _resolve_and_verify_raw_file(split_row)
    waveform = decode_audio_ffmpeg(raw_path, ffmpeg, sample_rate_hz=TARGET_SAMPLE_RATE_HZ)
    verified_after_decode = _resolve_and_verify_raw_file(split_row)
    if verified_after_decode != raw_path:
        raise RuntimeError(f"Raw audio changed during FFmpeg decode: {recording_id}")

    probe_duration = float(manifest_row["ffprobe_duration_seconds"])
    decoded_samples = int(waveform.size)
    decoded_duration = decoded_samples / TARGET_SAMPLE_RATE_HZ
    duration_ratio = decoded_duration / probe_duration
    if not minimum_duration_ratio <= duration_ratio <= maximum_duration_ratio:
        raise ValueError(
            f"Decoded duration ratio outside accepted bounds for {recording_id}: "
            f"{duration_ratio:.9f} not in [{minimum_duration_ratio}, {maximum_duration_ratio}]"
        )

    uniform_starts = uniform_clip_starts(decoded_samples)
    energy_candidates = select_energy_candidates(waveform)
    energy_starts = tuple(candidate.start_sample for candidate in energy_candidates)
    uniform_ranks = {start: rank for rank, start in enumerate(uniform_starts)}
    energy_ranks = {start: rank for rank, start in enumerate(energy_starts)}
    energy_values = {candidate.start_sample: candidate.energy for candidate in energy_candidates}
    unique_starts = tuple(sorted(set(uniform_starts) | set(energy_starts)))

    features: list[np.ndarray] = []
    clip_metadata: list[tuple[int, int, float, int, int]] = []
    for clip in iter_extracted_clips(waveform, unique_starts, clip_samples=CLIP_SAMPLES):
        feature = native_log_mel_spectrogram(clip.samples)
        if feature.shape != NATIVE_FEATURE_SHAPE or feature.dtype != np.float32:
            raise RuntimeError(f"Native feature contract failed for {recording_id}")
        if (
            not bool(np.all(np.isfinite(feature)))
            or float(feature.min()) < 0
            or float(feature.max()) > 1
        ):
            raise RuntimeError(f"Native feature values are invalid for {recording_id}")
        features.append(feature)
        clip_metadata.append(
            (
                clip.start_sample,
                clip.valid_samples,
                clip.valid_audio_fraction,
                clip.left_padding_samples,
                clip.right_padding_samples,
            )
        )

    feature_tensor = np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32)
    split_name = split_row["split"]
    logical_feature_path = Path(split_name) / "features" / f"{recording_id}.npy"
    physical_feature_path = feature_output_path or staging_root / logical_feature_path
    _write_npy(physical_feature_path, feature_tensor)
    feature_sha256 = sha256_file(physical_feature_path)
    feature_bytes = physical_feature_path.stat().st_size

    index_rows: list[dict[str, Any]] = []
    for feature_row, metadata in enumerate(clip_metadata):
        start, valid_samples, valid_fraction, left_padding, right_padding = metadata
        uniform_selected = start in uniform_ranks
        energy_selected = start in energy_ranks
        index_rows.append(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "clip_id": f"{recording_id}:{start:012d}",
                "recording_id": recording_id,
                "relative_path": split_row["relative_path"],
                "source_sha256": split_row["sha256"],
                "species_common_name": split_row["species_common_name"],
                "class_index": class_index,
                "session_group": split_row["session_group"],
                "split": split_name,
                "feature_file": logical_feature_path.as_posix(),
                "feature_file_sha256": feature_sha256,
                "feature_row": feature_row,
                "cached_clip_count": len(unique_starts),
                "uniform_clip_count": len(uniform_starts),
                "energy_clip_count": len(energy_starts),
                "start_sample": start,
                "valid_samples": valid_samples,
                "valid_audio_fraction": f"{valid_fraction:.9f}",
                "left_padding_samples": left_padding,
                "right_padding_samples": right_padding,
                "uniform_selected": str(uniform_selected).lower(),
                "uniform_rank": uniform_ranks[start] if uniform_selected else "",
                "energy_selected": str(energy_selected).lower(),
                "energy_rank": energy_ranks[start] if energy_selected else "",
                "energy_value": f"{energy_values[start]:.17g}" if energy_selected else "",
                "decoded_samples": decoded_samples,
                "decoded_duration_seconds": f"{decoded_duration:.9f}",
                "manifest_probe_duration_seconds": f"{probe_duration:.9f}",
                "decoded_to_probe_duration_ratio": f"{duration_ratio:.9f}",
            }
        )
    feature_record = {
        "path": logical_feature_path.as_posix(),
        "sha256": feature_sha256,
    }
    statistics = {
        "clips": len(unique_starts),
        "uniform_memberships": len(uniform_starts),
        "energy_memberships": len(energy_starts),
        "shared_memberships": len(set(uniform_starts) & set(energy_starts)),
        "feature_bytes": feature_bytes,
    }
    return index_rows, feature_record, statistics


def _implementation_fingerprint() -> str:
    paths = [
        Path(__file__).resolve(),
        resolve_project_path("src/bird_audio/signal.py"),
        resolve_project_path("src/bird_audio/clip_selection.py"),
    ]
    return fingerprint_files(paths, PROJECT_ROOT)


def _atomic_publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""
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
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Cache destination appeared before atomic publication",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _cache_provenance(
    inputs: _ValidatedInputs,
    ffmpeg_sha256: str,
    implementation_sha256: str,
    runtime: dict[str, str],
) -> dict[str, Any]:
    return {
        "config_file_sha256": inputs.artifact_hashes[inputs.config_file],
        "config_sha256": inputs.config_sha256,
        "final_manifest_sha256": inputs.manifest_sha256,
        "review_lock_sha256": inputs.review_lock_sha256,
        "split_sha256": inputs.split_sha256,
        "split_summary_sha256": inputs.split_summary_sha256,
        "split_lock_sha256": inputs.split_lock_sha256,
        "ffmpeg_executable_sha256": ffmpeg_sha256,
        "implementation_sha256": implementation_sha256,
        "requirements_lock_sha256": inputs.requirements_lock_sha256,
        "runtime": runtime,
        "input_paths": {
            "config": _project_label(inputs.config_file),
            "final_manifest": _project_label(inputs.manifest_file),
            "review_lock": _project_label(inputs.review_lock_file),
            "split": _project_label(inputs.split_file),
            "split_summary": _project_label(inputs.split_summary_file),
            "split_lock": _project_label(inputs.split_lock_file),
            "requirements_lock": _project_label(inputs.requirements_lock_file),
        },
    }


def _build_into_staging(
    staging_root: Path,
    inputs: _ValidatedInputs,
    ffmpeg: Path,
    ffmpeg_sha256: str,
    implementation_sha256: str,
    runtime: dict[str, str],
    recording_results: Sequence[tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    split_index_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_feature_records: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    split_statistics: dict[str, dict[str, int]] = {
        name: {
            "recordings": 0,
            "clips": 0,
            "uniform_memberships": 0,
            "energy_memberships": 0,
            "shared_memberships": 0,
            "feature_bytes": 0,
        }
        for name in SPLIT_NAMES
    }
    for checkpoint, source_feature in recording_results:
        split_name = str(checkpoint["split"])
        rows = list(checkpoint["index_rows"])
        feature_record = dict(checkpoint["feature_record"])
        statistics = dict(checkpoint["statistics"])
        target_feature = staging_root / feature_record["path"]
        target_feature.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_feature, target_feature)
        split_index_rows[split_name].extend(rows)
        split_feature_records[split_name].append(feature_record)
        split_statistics[split_name]["recordings"] += 1
        for key, value in statistics.items():
            split_statistics[split_name][key] += int(value)

    split_artifacts: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        (staging_root / split_name / "features").mkdir(parents=True, exist_ok=True)
        index_path = staging_root / split_name / "index.csv"
        atomic_write_csv(index_path, split_index_rows[split_name], INDEX_FIELDS)
        feature_records = sorted(split_feature_records[split_name], key=lambda item: item["path"])
        split_artifacts[split_name] = {
            "index": {
                "path": f"{split_name}/index.csv",
                "sha256": sha256_file(index_path),
                "rows": len(split_index_rows[split_name]),
            },
            "features": {
                "directory": f"{split_name}/features",
                "files": len(feature_records),
                "feature_set_sha256": sha256_json(feature_records),
            },
        }

    total_statistics = {
        key: sum(split_statistics[name][key] for name in SPLIT_NAMES)
        for key in next(iter(split_statistics.values()))
    }
    summary = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "feature_dtype": "float32",
        "feature_shape": list(NATIVE_FEATURE_SHAPE),
        "recording_tensor_shape": ["selected_clips", *NATIVE_FEATURE_SHAPE],
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "clip_samples": CLIP_SAMPLES,
        "selection_strategies": ["uniform", "energy"],
        "splits": split_statistics,
        "totals": total_statistics,
    }
    summary_path = staging_root / "summary.json"
    atomic_write_json(summary_path, summary)
    summary_sha256 = sha256_file(summary_path)

    provenance = _cache_provenance(inputs, ffmpeg_sha256, implementation_sha256, runtime)
    artifacts = {
        "summary": {"path": "summary.json", "sha256": summary_sha256},
        "splits": split_artifacts,
    }
    lock = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "provenance": provenance,
        "artifacts": artifacts,
        "cache_content_sha256": sha256_json(
            {"provenance": provenance, "artifacts": artifacts, "summary": summary}
        ),
    }
    _require_artifacts_unchanged(inputs.artifact_hashes)
    atomic_write_json(staging_root / "lock.json", lock)
    _require_artifacts_unchanged(inputs.artifact_hashes)
    return summary


def _reverify_all_raw_sources(split_rows: Sequence[dict[str, str]]) -> None:
    for row in split_rows:
        _resolve_and_verify_raw_file(row)


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(dict(payload))


def _resume_identity(
    provenance: dict[str, Any],
    pairs: Sequence[tuple[dict[str, str], dict[str, str]]],
) -> tuple[str, list[str]]:
    recording_order = [split_row["recording_id"] for split_row, _ in pairs]
    identity = sha256_json(
        {
            "schema_version": RESUME_SCHEMA_VERSION,
            "cache_version": CACHE_VERSION,
            "provenance": provenance,
            "index_fields": INDEX_FIELDS,
            "recordings": [
                {
                    "recording_id": split_row["recording_id"],
                    "sha256": split_row["sha256"],
                    "split": split_row["split"],
                    "probe_duration": manifest_row["ffprobe_duration_seconds"],
                }
                for split_row, manifest_row in pairs
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
                prefix=f".{working_root.name}.", suffix=".initializing", dir=working_root.parent
            )
        )
        try:
            (initial / "completed").mkdir()
            atomic_write_json(initial / "resume.json", expected_state)
            _atomic_publish_directory_no_replace(initial, working_root)
        except BaseException:
            shutil.rmtree(initial, ignore_errors=True)
            raise
    if (
        not working_root.is_dir()
        or working_root.is_symlink()
        or working_root.resolve() != working_root
        or working_root.parent.resolve() != working_root.parent
    ):
        raise ValueError(f"Resume working path is invalid: {working_root}")

    for child in list(working_root.iterdir()):
        if child.name.startswith(".") and child.name.endswith(".partial"):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                raise ValueError(f"Unsafe partial resume artifact: {child}")
    if {child.name for child in working_root.iterdir()} != {"completed", "resume.json"}:
        raise ValueError("Resume working directory contains unexpected artifacts")
    completed = working_root / "completed"
    state_path = working_root / "resume.json"
    if (
        not completed.is_dir()
        or completed.is_symlink()
        or completed.resolve() != completed
        or not state_path.is_file()
        or state_path.is_symlink()
    ):
        raise ValueError("Resume working directory is incomplete")
    for child in list(completed.iterdir()):
        if child.name.startswith(".") and child.name.endswith(".partial"):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                raise ValueError(f"Unsafe partial recording checkpoint: {child}")
    state = _read_json(state_path)
    if set(state) != set(expected_state) or state != expected_state:
        raise ValueError("Resume state does not match the current cache build identity")
    return completed


def _checkpoint_statistics(rows: Sequence[dict[str, Any]], feature_bytes: int) -> dict[str, int]:
    return {
        "clips": len(rows),
        "uniform_memberships": sum(row["uniform_selected"] == "true" for row in rows),
        "energy_memberships": sum(row["energy_selected"] == "true" for row in rows),
        "shared_memberships": sum(
            row["uniform_selected"] == row["energy_selected"] == "true" for row in rows
        ),
        "feature_bytes": feature_bytes,
    }


def _validate_recording_checkpoint(
    directory: Path,
    split_row: dict[str, str],
    class_indices: dict[str, int],
    build_identity_sha256: str,
) -> tuple[dict[str, Any], Path]:
    recording_id = split_row["recording_id"]
    checkpoint_path = directory / "checkpoint.json"
    feature_path = directory / "feature.npy"
    if (
        directory.name != recording_id
        or directory.is_symlink()
        or not directory.is_dir()
        or checkpoint_path.is_symlink()
        or feature_path.is_symlink()
        or not checkpoint_path.is_file()
        or not feature_path.is_file()
    ):
        raise ValueError(f"Resume checkpoint directory is unsafe: {directory}")
    if {child.name for child in directory.iterdir()} != {"checkpoint.json", "feature.npy"}:
        raise ValueError(f"Resume checkpoint files are incomplete: {recording_id}")
    checkpoint = _read_json(checkpoint_path)
    expected_fields = {
        "schema_version",
        "cache_version",
        "build_identity_sha256",
        "recording_id",
        "split",
        "index_rows",
        "feature_record",
        "statistics",
    }
    if (
        set(checkpoint) != expected_fields
        or checkpoint.get("schema_version") != RESUME_SCHEMA_VERSION
        or checkpoint.get("cache_version") != CACHE_VERSION
        or checkpoint.get("build_identity_sha256") != build_identity_sha256
        or checkpoint.get("recording_id") != recording_id
        or checkpoint.get("split") != split_row["split"]
        or not isinstance(checkpoint.get("index_rows"), list)
        or not isinstance(checkpoint.get("feature_record"), dict)
        or not isinstance(checkpoint.get("statistics"), dict)
    ):
        raise ValueError(f"Resume checkpoint binding is invalid: {recording_id}")
    rows = checkpoint["index_rows"]
    if not rows:
        raise ValueError(f"Resume checkpoint has no feature rows: {recording_id}")
    string_rows: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(INDEX_FIELDS):
            raise ValueError(f"Resume index schema is invalid: {recording_id}")
        string_row = {field: str(row[field]) for field in INDEX_FIELDS}
        if (
            string_row["recording_id"] != recording_id
            or string_row["split"] != split_row["split"]
            or string_row["relative_path"] != split_row["relative_path"]
            or string_row["source_sha256"] != split_row["sha256"]
            or string_row["species_common_name"] != split_row["species_common_name"]
            or string_row["session_group"] != split_row["session_group"]
        ):
            raise ValueError(f"Resume index source binding is invalid: {recording_id}")
        _validate_index_row(string_row, split_row["split"], class_indices)
        string_rows.append(string_row)
    cached_count = int(string_rows[0]["cached_clip_count"])
    uniform_count = int(string_rows[0]["uniform_clip_count"])
    energy_count = int(string_rows[0]["energy_clip_count"])
    uniform_by_rank = [
        int(row["start_sample"])
        for row in sorted(
            (row for row in string_rows if row["uniform_selected"] == "true"),
            key=lambda row: int(row["uniform_rank"]),
        )
    ]
    expected_uniform = list(uniform_clip_starts(int(string_rows[0]["decoded_samples"])))
    if (
        len(string_rows) != cached_count
        or any(int(row["cached_clip_count"]) != cached_count for row in string_rows)
        or any(int(row["uniform_clip_count"]) != uniform_count for row in string_rows)
        or any(int(row["energy_clip_count"]) != energy_count for row in string_rows)
        or [int(row["feature_row"]) for row in string_rows] != list(range(cached_count))
        or [int(row["start_sample"]) for row in string_rows]
        != sorted(int(row["start_sample"]) for row in string_rows)
        or any(
            row[field] != string_rows[0][field]
            for row in string_rows
            for field in (
                "relative_path",
                "source_sha256",
                "species_common_name",
                "class_index",
                "session_group",
                "feature_file",
                "feature_file_sha256",
                "decoded_samples",
                "decoded_duration_seconds",
                "manifest_probe_duration_seconds",
                "decoded_to_probe_duration_ratio",
            )
        )
        or sorted(
            int(row["uniform_rank"]) for row in string_rows if row["uniform_selected"] == "true"
        )
        != list(range(uniform_count))
        or sorted(
            int(row["energy_rank"]) for row in string_rows if row["energy_selected"] == "true"
        )
        != list(range(energy_count))
        or uniform_by_rank != expected_uniform
    ):
        raise ValueError(f"Resume strategy counts are invalid: {recording_id}")
    feature_record = checkpoint["feature_record"]
    expected_feature_path = f"{split_row['split']}/features/{recording_id}.npy"
    if (
        set(feature_record) != {"path", "sha256"}
        or feature_record.get("path") != expected_feature_path
        or _SHA256.fullmatch(str(feature_record.get("sha256") or "")) is None
        or any(row["feature_file"] != expected_feature_path for row in string_rows)
        or any(row["feature_file_sha256"] != feature_record["sha256"] for row in string_rows)
    ):
        raise ValueError(f"Resume feature binding is invalid: {recording_id}")
    tensor, feature_bytes = _read_verified_feature_tensor(feature_path, feature_record["sha256"])
    if tensor.shape[0] != cached_count:
        raise ValueError(f"Resume tensor row count is invalid: {recording_id}")
    expected_statistics = _checkpoint_statistics(rows, feature_bytes)
    if checkpoint["statistics"] != expected_statistics:
        raise ValueError(f"Resume checkpoint statistics are invalid: {recording_id}")
    return checkpoint, feature_path


def _disk_preflight(parent: Path, remaining_recordings: int) -> dict[str, int]:
    bytes_per_clip = int(np.prod(NATIVE_FEATURE_SHAPE)) * np.dtype(np.float32).itemsize
    maximum_feature_bytes = (
        remaining_recordings * 2 * MAXIMUM_CLIPS_PER_RECORDING * (bytes_per_clip + 512)
    )
    required_free_bytes = int(maximum_feature_bytes * 1.25) + 64 * 1024 * 1024
    available_free_bytes = shutil.disk_usage(parent).free
    if available_free_bytes < required_free_bytes:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient free space for cache: need {required_free_bytes}, "
            f"available {available_free_bytes}",
            str(parent),
        )
    return {
        "required_free_bytes": required_free_bytes,
        "available_free_bytes": available_free_bytes,
    }


def build_known_clip_cache(
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    *,
    ffmpeg: str | Path | None = None,
    config_path: str | Path = "configs/data.toml",
    manifest_path: str | Path | None = None,
    split_path: str | Path | None = None,
    split_summary_path: str | Path | None = None,
    split_lock_path: str | Path | None = None,
    review_lock_path: str | Path = DEFAULT_REVIEW_LOCK,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resume, validate, and atomically publish the locked known-species feature cache."""
    destination = require_safe_output(cache_root)
    if _VERSIONED_ROOT.fullmatch(destination.name) is None:
        raise ValueError("Cache root name must end with a positive version suffix such as _v1")
    if destination.exists():
        raise RuntimeError(f"Known clip cache already exists and cannot be replaced: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with project_lock("known_clip_cache_build"):
        if destination.exists():
            raise RuntimeError(
                f"Known clip cache already exists and cannot be replaced: {destination}"
            )
        _require_project_venv()
        inputs = _load_validated_inputs(
            config_path,
            manifest_path,
            split_path,
            split_summary_path,
            split_lock_path,
            review_lock_path,
        )
        ffmpeg_path = resolve_tool("ffmpeg", ffmpeg)
        ffmpeg_sha256 = sha256_file(ffmpeg_path)
        implementation_sha256 = _implementation_fingerprint()
        runtime = _runtime_provenance(ffmpeg_path)
        provenance = _cache_provenance(
            inputs,
            ffmpeg_sha256,
            implementation_sha256,
            runtime,
        )
        pairs, class_indices = _validate_manifest_and_split_rows(inputs)
        build_identity_sha256, recording_order = _resume_identity(provenance, pairs)
        expected_pairs = {
            split_row["recording_id"]: (split_row, manifest_row)
            for split_row, manifest_row in pairs
        }
        working_root = destination.with_name(f".{destination.name}.working")
        completed_root = _prepare_working_directory(
            working_root, build_identity_sha256, recording_order
        )

        completed_results: dict[str, tuple[dict[str, Any], Path]] = {}
        for child in sorted(completed_root.iterdir(), key=lambda path: path.name):
            if child.name not in expected_pairs:
                raise ValueError(
                    f"Resume checkpoint is not part of the current split: {child.name}"
                )
            split_row, _ = expected_pairs[child.name]
            completed_results[child.name] = _validate_recording_checkpoint(
                child,
                split_row,
                class_indices,
                build_identity_sha256,
            )
        disk = _disk_preflight(destination.parent, len(pairs) - len(completed_results))
        _emit_progress(
            progress_callback,
            {
                "event": "preflight",
                "recordings_total": len(pairs),
                "recordings_completed": len(completed_results),
                "recordings_remaining": len(pairs) - len(completed_results),
                **disk,
            },
        )

        quality_control = inputs.config["quality_control"]
        minimum_ratio = float(quality_control["minimum_decoded_to_ffprobe_duration_ratio"])
        maximum_ratio = float(quality_control["maximum_decoded_to_ffprobe_duration_ratio"])
        for split_row, manifest_row in pairs:
            recording_id = split_row["recording_id"]
            if recording_id in completed_results:
                _emit_progress(
                    progress_callback,
                    {
                        "event": "recording_complete",
                        "recording_id": recording_id,
                        "recordings_completed": len(completed_results),
                        "recordings_total": len(pairs),
                        "resumed": True,
                    },
                )
                continue
            partial = Path(
                tempfile.mkdtemp(prefix=f".{recording_id}.", suffix=".partial", dir=completed_root)
            )
            try:
                rows, feature_record, statistics = _process_recording(
                    working_root,
                    split_row,
                    manifest_row,
                    class_indices[split_row["species_common_name"]],
                    ffmpeg_path,
                    minimum_ratio,
                    maximum_ratio,
                    feature_output_path=partial / "feature.npy",
                )
                checkpoint = {
                    "schema_version": RESUME_SCHEMA_VERSION,
                    "cache_version": CACHE_VERSION,
                    "build_identity_sha256": build_identity_sha256,
                    "recording_id": recording_id,
                    "split": split_row["split"],
                    "index_rows": rows,
                    "feature_record": feature_record,
                    "statistics": statistics,
                }
                atomic_write_json(partial / "checkpoint.json", checkpoint)
                completed_directory = completed_root / recording_id
                _atomic_publish_directory_no_replace(partial, completed_directory)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
            completed_results[recording_id] = _validate_recording_checkpoint(
                completed_directory,
                split_row,
                class_indices,
                build_identity_sha256,
            )
            _emit_progress(
                progress_callback,
                {
                    "event": "recording_complete",
                    "recording_id": recording_id,
                    "recordings_completed": len(completed_results),
                    "recordings_total": len(pairs),
                    "resumed": False,
                },
            )

        recording_results = [completed_results[recording_id] for recording_id in recording_order]
        staging_root = destination.with_name(f".{destination.name}.publishing")
        if staging_root.exists():
            if not staging_root.is_dir() or staging_root.is_symlink():
                raise ValueError(f"Unsafe interrupted publication path: {staging_root}")
            shutil.rmtree(staging_root)
        staging_root.mkdir()
        try:
            summary = _build_into_staging(
                staging_root,
                inputs,
                ffmpeg_path,
                ffmpeg_sha256,
                implementation_sha256,
                runtime,
                recording_results,
            )
            _validate_publishing_tree(staging_root, summary, class_indices)
            if sha256_file(ffmpeg_path) != ffmpeg_sha256:
                raise RuntimeError("FFmpeg executable changed during cache construction")
            if _implementation_fingerprint() != implementation_sha256:
                raise RuntimeError(
                    "Signal preprocessing implementation changed during cache construction"
                )
            if _runtime_provenance(ffmpeg_path) != runtime:
                raise RuntimeError("Numerical or FFmpeg runtime changed during cache construction")
            _reverify_all_raw_sources(inputs.split_rows)
            _require_artifacts_unchanged(inputs.artifact_hashes)
            if destination.exists():
                raise RuntimeError(
                    f"Known clip cache appeared during construction and will not be replaced: {destination}"
                )
            _atomic_publish_directory_no_replace(staging_root, destination)
        except BaseException:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        shutil.rmtree(working_root)
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


def _resolve_cache_artifact(root: Path, relative_path: str, expected: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute() or relative_path != expected:
        raise ValueError(f"Cache lock artifact path is invalid: {relative_path}")
    resolved = (root / relative_path).resolve()
    if not is_relative_to(resolved, root) or resolved.relative_to(root).as_posix() != relative_path:
        raise ValueError(f"Cache artifact path leaves its locked root: {relative_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Locked cache artifact is missing: {relative_path}")
    return resolved


def _resolve_cache_root(cache_root: str | Path) -> Path:
    root = resolve_project_path(cache_root)
    if (
        not root.is_dir()
        or not is_relative_to(root, PROJECT_ROOT)
        or is_relative_to(root, RAW_DATA_ROOT)
        or _VERSIONED_ROOT.fullmatch(root.name) is None
    ):
        raise ValueError(f"Cache root is invalid: {root}")
    return root


def _validate_current_provenance(
    provenance: dict[str, Any], ffmpeg: str | Path | None
) -> dict[str, Any]:
    paths = provenance.get("input_paths")
    runtime = provenance.get("runtime")
    if not isinstance(paths, dict) or set(paths) != _INPUT_PATH_FIELDS:
        raise ValueError("Cache provenance input paths are not exact")
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("Cache runtime provenance is not exact")
    scalar_hash_fields = _PROVENANCE_FIELDS - {"runtime", "input_paths"}
    if any(
        _SHA256.fullmatch(str(provenance.get(field) or "")) is None for field in scalar_hash_fields
    ):
        raise ValueError("Cache provenance contains an invalid SHA-256 value")

    resolved = {name: _resolve_input(str(value)) for name, value in paths.items()}
    if any(_project_label(resolved[name]) != paths[name] for name in paths):
        raise ValueError("Cache provenance contains a noncanonical input path")
    current_hashes = {
        "config_file_sha256": sha256_file(resolved["config"]),
        "final_manifest_sha256": sha256_file(resolved["final_manifest"]),
        "review_lock_sha256": sha256_file(resolved["review_lock"]),
        "split_sha256": sha256_file(resolved["split"]),
        "split_summary_sha256": sha256_file(resolved["split_summary"]),
        "split_lock_sha256": sha256_file(resolved["split_lock"]),
        "requirements_lock_sha256": sha256_file(resolved["requirements_lock"]),
        "implementation_sha256": _implementation_fingerprint(),
    }
    mismatches = [
        field for field, digest in current_hashes.items() if provenance.get(field) != digest
    ]
    config = load_toml(resolved["config"])
    _assert_locked_cache_config(config)
    if provenance.get("config_sha256") != config_fingerprint(config):
        mismatches.append("config_sha256")
    ffmpeg_path = resolve_tool("ffmpeg", ffmpeg)
    if provenance.get("ffmpeg_executable_sha256") != sha256_file(ffmpeg_path):
        mismatches.append("ffmpeg_executable_sha256")
    if provenance.get("runtime") != _runtime_provenance(ffmpeg_path):
        mismatches.append("runtime")
    if mismatches:
        raise ValueError(f"Cache provenance is stale: {sorted(set(mismatches))}")
    return {"config": config, "paths": resolved, "ffmpeg": ffmpeg_path}


def _load_cache_metadata(
    cache_root: str | Path,
    *,
    ffmpeg: str | Path | None,
    expected_lock_sha256: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_project_venv()
    root = _resolve_cache_root(cache_root)
    lock_path = root / "lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Cache lock is missing: {lock_path}")
    lock_sha256 = sha256_file(lock_path)
    if expected_lock_sha256 is not None:
        if _SHA256.fullmatch(expected_lock_sha256) is None:
            raise ValueError("Expected cache-lock SHA-256 is malformed")
        if lock_sha256 != expected_lock_sha256:
            raise ValueError("Cache lock does not match the expected SHA-256")
    lock = _read_json(lock_path)
    if set(lock) != _LOCK_FIELDS:
        raise ValueError("Cache lock fields are not exact")
    if (
        lock.get("schema_version") != CACHE_SCHEMA_VERSION
        or lock.get("cache_version") != CACHE_VERSION
    ):
        raise ValueError("Cache lock schema or version is unsupported")
    provenance = lock.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("Cache lock provenance fields are not exact")
    current = _validate_current_provenance(provenance, ffmpeg)
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"summary", "splits"}:
        raise ValueError("Cache lock artifact fields are not exact")
    summary_entry = artifacts.get("summary")
    split_entries = artifacts.get("splits")
    if (
        not isinstance(summary_entry, dict)
        or set(summary_entry) != {"path", "sha256"}
        or _SHA256.fullmatch(str(summary_entry.get("sha256") or "")) is None
        or not isinstance(split_entries, dict)
        or set(split_entries) != set(SPLIT_NAMES)
    ):
        raise ValueError("Cache lock artifact bindings are incomplete")
    for split_name, split_entry in split_entries.items():
        if (
            not isinstance(split_entry, dict)
            or set(split_entry) != {"index", "features"}
            or not isinstance(split_entry.get("index"), dict)
            or set(split_entry["index"]) != {"path", "sha256", "rows"}
            or split_entry["index"].get("path") != f"{split_name}/index.csv"
            or _SHA256.fullmatch(str(split_entry["index"].get("sha256") or "")) is None
            or not isinstance(split_entry["index"].get("rows"), int)
            or split_entry["index"]["rows"] < 0
            or not isinstance(split_entry.get("features"), dict)
            or set(split_entry["features"]) != {"directory", "files", "feature_set_sha256"}
            or split_entry["features"].get("directory") != f"{split_name}/features"
            or not isinstance(split_entry["features"].get("files"), int)
            or split_entry["features"]["files"] < 0
            or _SHA256.fullmatch(str(split_entry["features"].get("feature_set_sha256") or ""))
            is None
        ):
            raise ValueError(f"Cache lock split binding is invalid: {split_name}")

    summary_path = _resolve_cache_artifact(
        root, str(summary_entry.get("path") or ""), "summary.json"
    )
    if sha256_file(summary_path) != summary_entry.get("sha256"):
        raise ValueError("Cache summary hash does not match its lock")
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != CACHE_SCHEMA_VERSION
        or summary.get("cache_version") != CACHE_VERSION
        or summary.get("feature_dtype") != "float32"
        or summary.get("feature_shape") != list(NATIVE_FEATURE_SHAPE)
        or summary.get("recording_tensor_shape") != ["selected_clips", *NATIVE_FEATURE_SHAPE]
        or summary.get("sample_rate_hz") != TARGET_SAMPLE_RATE_HZ
        or summary.get("clip_samples") != CLIP_SAMPLES
        or summary.get("selection_strategies") != ["uniform", "energy"]
        or set(summary)
        != {
            "schema_version",
            "cache_version",
            "feature_dtype",
            "feature_shape",
            "recording_tensor_shape",
            "sample_rate_hz",
            "clip_samples",
            "selection_strategies",
            "splits",
            "totals",
        }
    ):
        raise ValueError("Cache summary signal contract is invalid")
    expected_content_sha256 = sha256_json(
        {"provenance": provenance, "artifacts": artifacts, "summary": summary}
    )
    if lock.get("cache_content_sha256") != expected_content_sha256:
        raise ValueError("Cache content hash does not match the locked artifacts")
    return root, lock, summary, current


def _parse_int(row: dict[str, str], field: str, context: str) -> int:
    try:
        return int(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"Invalid integer {field} in {context}") from exc


def _parse_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"Invalid float {field} in {context}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {field} in {context}")
    return value


def _read_verified_feature_tensor(path: Path, expected_sha256: str) -> tuple[np.ndarray, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        initial_stat = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Feature hash drift: {path}")
        handle.seek(0)
        tensor = np.load(handle, allow_pickle=False)
        final_stat = os.fstat(handle.fileno())
    initial_identity = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
        initial_stat.st_ctime_ns,
    )
    final_identity = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        final_stat.st_ctime_ns,
    )
    if initial_identity != final_identity:
        raise ValueError(f"Feature changed while it was being verified: {path}")
    if (
        tensor.dtype != np.float32
        or tensor.ndim != 4
        or tuple(tensor.shape[1:]) != NATIVE_FEATURE_SHAPE
        or not bool(np.all(np.isfinite(tensor)))
        or float(tensor.min()) < 0
        or float(tensor.max()) > 1
    ):
        raise ValueError(f"Feature tensor contract is invalid: {path}")
    return np.ascontiguousarray(tensor, dtype=np.float32), initial_stat.st_size


def _load_verified_feature_tensor(path: Path, expected_sha256: str) -> np.ndarray:
    tensor, _ = _read_verified_feature_tensor(path, expected_sha256)
    return tensor


def _validate_index_row(
    row: dict[str, str], split: str, class_indices: dict[str, int]
) -> dict[str, Any]:
    recording_id = row.get("recording_id", "")
    context = f"{split}:{recording_id}"
    species = row.get("species_common_name", "")
    if (
        row.get("schema_version") != CACHE_SCHEMA_VERSION
        or row.get("split") != split
        or _SAFE_RECORDING_ID.fullmatch(recording_id) is None
        or species not in class_indices
        or _parse_int(row, "class_index", context) != class_indices.get(species, -1)
        or not row.get("session_group")
        or _SHA256.fullmatch(row.get("source_sha256", "")) is None
        or _SHA256.fullmatch(row.get("feature_file_sha256", "")) is None
    ):
        raise ValueError(f"Cache index identity binding is invalid: {context}")
    expected_feature = f"{split}/features/{recording_id}.npy"
    if row.get("feature_file") != expected_feature:
        raise ValueError(f"Cache feature path is invalid: {context}")
    raw_relative = row.get("relative_path", "")
    if not raw_relative or Path(raw_relative).is_absolute():
        raise ValueError(f"Cache raw path is invalid: {context}")
    raw_resolved = resolve_project_path(raw_relative)
    if (
        not is_relative_to(raw_resolved, RAW_DATA_ROOT)
        or raw_resolved.relative_to(PROJECT_ROOT).as_posix() != raw_relative
    ):
        raise ValueError(f"Cache raw path leaves the dataset: {context}")

    values = {
        field: _parse_int(row, field, context)
        for field in (
            "start_sample",
            "valid_samples",
            "left_padding_samples",
            "right_padding_samples",
            "decoded_samples",
            "feature_row",
            "cached_clip_count",
            "uniform_clip_count",
            "energy_clip_count",
        )
    }
    start = values["start_sample"]
    valid = values["valid_samples"]
    left = values["left_padding_samples"]
    right = values["right_padding_samples"]
    decoded = values["decoded_samples"]
    valid_fraction = _parse_float(row, "valid_audio_fraction", context)
    decoded_duration = _parse_float(row, "decoded_duration_seconds", context)
    probe_duration = _parse_float(row, "manifest_probe_duration_seconds", context)
    duration_ratio = _parse_float(row, "decoded_to_probe_duration_ratio", context)
    uniform_selected = row.get("uniform_selected")
    energy_selected = row.get("energy_selected")
    if (
        start < 0
        or decoded <= 0
        or min(valid, left, right, values["feature_row"]) < 0
        or valid + left + right != CLIP_SAMPLES
        or not 1 <= values["cached_clip_count"] <= 2 * MAXIMUM_CLIPS_PER_RECORDING
        or not 1 <= values["uniform_clip_count"] <= MAXIMUM_CLIPS_PER_RECORDING
        or not 1 <= values["energy_clip_count"] <= MAXIMUM_CLIPS_PER_RECORDING
        or uniform_selected not in {"true", "false"}
        or energy_selected not in {"true", "false"}
        or uniform_selected == energy_selected == "false"
        or abs(valid_fraction - valid / CLIP_SAMPLES) > 1e-9
        or abs(decoded_duration - decoded / TARGET_SAMPLE_RATE_HZ) > 1e-9
        or probe_duration <= 0
        or abs(duration_ratio - decoded_duration / probe_duration) > 2e-9
    ):
        raise ValueError(f"Cache clip arithmetic is invalid: {context}:{start}")
    if decoded < CLIP_SAMPLES:
        missing = CLIP_SAMPLES - decoded
        if (start, valid, left, right) != (0, decoded, missing // 2, missing - missing // 2):
            raise ValueError(f"Cache short-clip padding is invalid: {context}")
    elif start > decoded - CLIP_SAMPLES or valid != CLIP_SAMPLES or left != 0 or right != 0:
        raise ValueError(f"Cache long-clip bounds are invalid: {context}:{start}")
    if row.get("clip_id") != f"{recording_id}:{start:012d}":
        raise ValueError(f"Cache clip ID is invalid: {context}:{start}")

    for strategy, selected in (("uniform", uniform_selected), ("energy", energy_selected)):
        rank_field = f"{strategy}_rank"
        if selected == "true":
            rank = _parse_int(row, rank_field, context)
            if rank < 0:
                raise ValueError(f"Cache {strategy} rank is invalid: {context}")
            values[rank_field] = rank
        elif row.get(rank_field):
            raise ValueError(f"Unselected cache row has a {strategy} rank: {context}")
        else:
            values[rank_field] = None
    if energy_selected == "true":
        if _parse_float(row, "energy_value", context) < 0:
            raise ValueError(f"Cache energy is negative: {context}")
    elif row.get("energy_value"):
        raise ValueError(f"Unselected cache row has an energy value: {context}")
    return values


def _read_split_index(
    root: Path,
    split: str,
    split_entry: dict[str, Any],
    class_indices: dict[str, int],
    *,
    verify_feature_bytes: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    index_entry = split_entry["index"]
    index_path = _resolve_cache_artifact(
        root, str(index_entry.get("path") or ""), f"{split}/index.csv"
    )
    if sha256_file(index_path) != index_entry.get("sha256"):
        raise ValueError(f"Cache index hash does not match its lock for {split}")
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != INDEX_FIELDS:
            raise ValueError(f"Cache index schema is invalid for {split}")
        rows = list(reader)
    if len(rows) != index_entry.get("rows"):
        raise ValueError(f"Cache index count is invalid for {split}")

    clip_ids: set[str] = set()
    groups: dict[str, list[tuple[dict[str, str], dict[str, Any]]]] = {}
    order_keys: list[tuple[int, str, int]] = []
    for row in rows:
        values = _validate_index_row(row, split, class_indices)
        if row["clip_id"] in clip_ids:
            raise ValueError(f"Cache clip ID is duplicated: {row['clip_id']}")
        clip_ids.add(row["clip_id"])
        groups.setdefault(row["recording_id"], []).append((row, values))
        order_keys.append(
            (
                class_indices[row["species_common_name"]],
                row["recording_id"],
                values["start_sample"],
            )
        )
    if order_keys != sorted(order_keys):
        raise ValueError(f"Cache index order is not deterministic for {split}")

    feature_records: list[dict[str, str]] = []
    feature_bytes = 0
    for recording_id, group in sorted(groups.items()):
        first_row, first_values = group[0]
        cached_count = first_values["cached_clip_count"]
        uniform_count = first_values["uniform_clip_count"]
        energy_count = first_values["energy_clip_count"]
        constant_fields = {
            "relative_path",
            "source_sha256",
            "species_common_name",
            "class_index",
            "session_group",
            "feature_file",
            "feature_file_sha256",
            "cached_clip_count",
            "uniform_clip_count",
            "energy_clip_count",
            "decoded_samples",
            "decoded_duration_seconds",
            "manifest_probe_duration_seconds",
            "decoded_to_probe_duration_ratio",
        }
        uniform_by_rank = [
            values["start_sample"]
            for _, values in sorted(
                ((row, values) for row, values in group if values["uniform_rank"] is not None),
                key=lambda item: item[1]["uniform_rank"],
            )
        ]
        expected_uniform = list(uniform_clip_starts(first_values["decoded_samples"]))
        if (
            len(group) != cached_count
            or any(values["cached_clip_count"] != cached_count for _, values in group)
            or any(values["uniform_clip_count"] != uniform_count for _, values in group)
            or any(values["energy_clip_count"] != energy_count for _, values in group)
            or [values["feature_row"] for _, values in group] != list(range(cached_count))
            or [values["start_sample"] for _, values in group]
            != sorted(values["start_sample"] for _, values in group)
            or len({row["start_sample"] for row, _ in group}) != cached_count
            or any(
                any(row[field] != first_row[field] for field in constant_fields) for row, _ in group
            )
            or sorted(
                values["uniform_rank"] for _, values in group if values["uniform_rank"] is not None
            )
            != list(range(uniform_count))
            or sorted(
                values["energy_rank"] for _, values in group if values["energy_rank"] is not None
            )
            != list(range(energy_count))
            or uniform_by_rank != expected_uniform
        ):
            raise ValueError(f"Cache per-recording index invariants failed: {split}:{recording_id}")
        feature_path = _resolve_cache_artifact(
            root, first_row["feature_file"], first_row["feature_file"]
        )
        if verify_feature_bytes:
            tensor, current_feature_bytes = _read_verified_feature_tensor(
                feature_path, first_row["feature_file_sha256"]
            )
            tensor_rows = tensor.shape[0]
        else:
            tensor = np.load(feature_path, allow_pickle=False, mmap_mode="r")
            try:
                if (
                    tensor.dtype != np.float32
                    or tensor.ndim != 4
                    or tuple(tensor.shape[1:]) != NATIVE_FEATURE_SHAPE
                ):
                    raise ValueError(f"Feature tensor contract is invalid: {feature_path}")
                tensor_rows = tensor.shape[0]
            finally:
                memory_map = getattr(tensor, "_mmap", None)
                if memory_map is not None:
                    memory_map.close()
            current_feature_bytes = feature_path.stat().st_size
        if tensor_rows != cached_count:
            raise ValueError(f"Feature tensor row count is invalid: {feature_path}")
        feature_records.append(
            {"path": first_row["feature_file"], "sha256": first_row["feature_file_sha256"]}
        )
        feature_bytes += current_feature_bytes

    features_entry = split_entry["features"]
    if (
        set(features_entry) != {"directory", "files", "feature_set_sha256"}
        or features_entry.get("directory") != f"{split}/features"
        or features_entry.get("files") != len(feature_records)
        or features_entry.get("feature_set_sha256") != sha256_json(feature_records)
    ):
        raise ValueError(f"Feature-set binding is invalid for {split}")
    feature_directory = (root / split / "features").resolve()
    if (
        not feature_directory.is_dir()
        or not is_relative_to(feature_directory, root)
        or feature_directory.relative_to(root).as_posix() != f"{split}/features"
    ):
        raise ValueError(f"Feature directory is invalid for {split}")
    children = list(feature_directory.iterdir())
    physical_files = {path.relative_to(root).as_posix() for path in children if path.is_file()}
    expected_files = {record["path"] for record in feature_records}
    if physical_files != expected_files or any(not path.is_file() for path in children):
        raise ValueError(f"Physical feature-file set is invalid for {split}")

    statistics = {
        "recordings": len(groups),
        "clips": len(rows),
        "uniform_memberships": sum(row["uniform_selected"] == "true" for row in rows),
        "energy_memberships": sum(row["energy_selected"] == "true" for row in rows),
        "shared_memberships": sum(
            row["uniform_selected"] == row["energy_selected"] == "true" for row in rows
        ),
        "feature_bytes": feature_bytes,
    }
    return rows, statistics


def _validate_publishing_tree(
    root: Path,
    expected_summary: dict[str, Any],
    class_indices: dict[str, int],
) -> None:
    """Validate the complete hidden publication tree before its final rename."""
    lock = _read_json(root / "lock.json")
    if set(lock) != _LOCK_FIELDS:
        raise ValueError("Publishing lock fields are not exact")
    if (
        lock.get("schema_version") != CACHE_SCHEMA_VERSION
        or lock.get("cache_version") != CACHE_VERSION
    ):
        raise ValueError("Publishing lock schema or version is unsupported")
    provenance = lock.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("Publishing provenance fields are not exact")
    if (
        not isinstance(provenance.get("input_paths"), dict)
        or set(provenance["input_paths"]) != _INPUT_PATH_FIELDS
        or not isinstance(provenance.get("runtime"), dict)
        or set(provenance["runtime"]) != _RUNTIME_FIELDS
    ):
        raise ValueError("Publishing provenance structure is invalid")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"summary", "splits"}:
        raise ValueError("Publishing artifact fields are not exact")
    summary_entry = artifacts.get("summary")
    split_entries = artifacts.get("splits")
    if (
        not isinstance(summary_entry, dict)
        or set(summary_entry) != {"path", "sha256"}
        or summary_entry.get("path") != "summary.json"
        or _SHA256.fullmatch(str(summary_entry.get("sha256") or "")) is None
        or not isinstance(split_entries, dict)
        or set(split_entries) != set(SPLIT_NAMES)
    ):
        raise ValueError("Publishing artifact bindings are incomplete")
    summary_path = _resolve_cache_artifact(root, "summary.json", "summary.json")
    if sha256_file(summary_path) != summary_entry["sha256"]:
        raise ValueError("Publishing summary hash does not match its lock")
    summary = _read_json(summary_path)
    if summary != expected_summary:
        raise ValueError("Publishing summary differs from the constructed summary")

    statistics: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        split_entry = split_entries[split]
        if (
            not isinstance(split_entry, dict)
            or set(split_entry) != {"index", "features"}
            or not isinstance(split_entry.get("index"), dict)
            or set(split_entry["index"]) != {"path", "sha256", "rows"}
            or split_entry["index"].get("path") != f"{split}/index.csv"
            or _SHA256.fullmatch(str(split_entry["index"].get("sha256") or "")) is None
            or not isinstance(split_entry["index"].get("rows"), int)
            or split_entry["index"]["rows"] < 0
            or not isinstance(split_entry.get("features"), dict)
            or set(split_entry["features"]) != {"directory", "files", "feature_set_sha256"}
            or split_entry["features"].get("directory") != f"{split}/features"
            or not isinstance(split_entry["features"].get("files"), int)
            or split_entry["features"]["files"] < 0
            or _SHA256.fullmatch(str(split_entry["features"].get("feature_set_sha256") or ""))
            is None
        ):
            raise ValueError(f"Publishing split binding is invalid: {split}")
        _, statistics[split] = _read_split_index(
            root,
            split,
            split_entry,
            class_indices,
            verify_feature_bytes=True,
        )
    if summary.get("splits") != statistics:
        raise ValueError("Publishing summary split counts do not match its artifacts")
    totals = {
        key: sum(statistics[split][key] for split in SPLIT_NAMES)
        for key in next(iter(statistics.values()))
    }
    if summary.get("totals") != totals:
        raise ValueError("Publishing summary totals do not match its artifacts")
    expected_content_sha256 = sha256_json(
        {"provenance": provenance, "artifacts": artifacts, "summary": summary}
    )
    if lock.get("cache_content_sha256") != expected_content_sha256:
        raise ValueError("Publishing content hash does not match its locked artifacts")


def verify_known_clip_cache(
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    *,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every cache artifact without returning test examples or index rows."""
    root, lock, summary, current = _load_cache_metadata(
        cache_root,
        ffmpeg=ffmpeg,
        expected_lock_sha256=expected_lock_sha256,
    )
    paths = current["paths"]
    inputs = _load_validated_inputs(
        paths["config"],
        paths["final_manifest"],
        paths["split"],
        paths["split_summary"],
        paths["split_lock"],
        paths["review_lock"],
    )
    _reverify_all_raw_sources(inputs.split_rows)
    class_indices = {
        str(entry["common_name"]): index
        for index, entry in enumerate(current["config"]["known_species"])
    }
    statistics: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        _, statistics[split] = _read_split_index(
            root,
            split,
            lock["artifacts"]["splits"][split],
            class_indices,
            verify_feature_bytes=True,
        )
    if summary.get("splits") != statistics:
        raise ValueError("Cache summary split counts do not match verified artifacts")
    totals = {
        key: sum(statistics[split][key] for split in SPLIT_NAMES)
        for key in next(iter(statistics.values()))
    }
    if summary.get("totals") != totals:
        raise ValueError("Cache summary totals do not match verified artifacts")
    return {
        "valid": True,
        "cache_version": CACHE_VERSION,
        "lock_sha256": sha256_file(root / "lock.json"),
        "recordings": totals["recordings"],
        "clips": totals["clips"],
        "feature_files": totals["recordings"],
    }


class DevelopmentClipCache(Sequence[tuple[np.ndarray, dict[str, str]]]):
    """Read-only strategy-filtered train or validation access to native features."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        strategy: str,
        *,
        ffmpeg: str | Path | None = None,
        expected_lock_sha256: str | None = None,
    ) -> None:
        if split == "test":
            raise PermissionError("The development cache API cannot open the final test split")
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError(f"Development split must be one of {DEVELOPMENT_SPLITS}")
        if strategy not in {"uniform", "energy"}:
            raise ValueError("Development cache strategy must be uniform or energy")
        root, lock, summary, current = _load_cache_metadata(
            cache_root,
            ffmpeg=ffmpeg,
            expected_lock_sha256=expected_lock_sha256,
        )
        class_indices = {
            str(entry["common_name"]): index
            for index, entry in enumerate(current["config"]["known_species"])
        }
        all_rows, statistics = _read_split_index(
            root,
            split,
            lock["artifacts"]["splits"][split],
            class_indices,
            verify_feature_bytes=False,
        )
        if summary.get("splits", {}).get(split) != statistics:
            raise ValueError(f"Cache summary counts do not match the {split} artifacts")
        self.root = root
        self.split = split
        self.strategy = strategy
        strategy_rows = [row for row in all_rows if row[f"{strategy}_selected"] == "true"]
        strategy_rows.sort(
            key=lambda row: (
                int(row["class_index"]),
                row["recording_id"],
                int(row[f"{strategy}_rank"]),
            )
        )
        self.rows = tuple(strategy_rows)
        self.lock_sha256 = sha256_file(root / "lock.json")
        self._loaded_feature_path: Path | None = None
        self._loaded_feature_tensor: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]:
        row = self.rows[index]
        feature_path = _resolve_cache_artifact(self.root, row["feature_file"], row["feature_file"])
        if feature_path != self._loaded_feature_path:
            self._loaded_feature_tensor = _load_verified_feature_tensor(
                feature_path, row["feature_file_sha256"]
            )
            self._loaded_feature_path = feature_path
        if self._loaded_feature_tensor is None:
            raise RuntimeError("Development feature tensor was not loaded")
        feature_row = int(row["feature_row"])
        feature = self._loaded_feature_tensor[feature_row].copy()
        metadata = dict(row)
        metadata["selection_strategy"] = self.strategy
        metadata["strategy_clip_count"] = row[f"{self.strategy}_clip_count"]
        return feature, metadata


def load_development_clip_cache(
    cache_root: str | Path,
    split: str,
    strategy: str,
    *,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
) -> DevelopmentClipCache:
    """Open only one strategy's train or validation clips through the development API."""
    return DevelopmentClipCache(
        cache_root,
        split,
        strategy,
        ffmpeg=ffmpeg,
        expected_lock_sha256=expected_lock_sha256,
    )
