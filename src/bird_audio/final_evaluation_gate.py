from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import secrets
import stat
import sys
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.config import config_fingerprint, public_config
from bird_audio.hashing import sha256_json
from bird_audio.paths import PROJECT_ROOT
from bird_audio.provenance import source_fingerprint
from bird_audio.task1_training import (
    CHECKPOINT_SCHEMA_VERSION as TASK1_CHECKPOINT_SCHEMA_VERSION,
)
from bird_audio.task1_training import DEFAULT_RUN_ROOT as TASK1_DEFAULT_RUN_ROOT
from bird_audio.task1_training import FINAL_CONFIG_PATH as TASK1_CONFIG_PATH
from bird_audio.task1_training import (
    KNOWN_CACHE_LOCK_SHA256 as TASK1_KNOWN_CACHE_LOCK_SHA256,
)
from bird_audio.task1_training import PRODUCTION_SCOPE as TASK1_PRODUCTION_SCOPE
from bird_audio.task1_training import RUN_SCHEMA_VERSION as TASK1_RUN_SCHEMA_VERSION
from bird_audio.task1_training import (
    load_final_task1_config,
    load_task1_checkpoint,
    verify_locked_task1_best_checkpoint_model_state,
    verify_task1_development_run,
)
from bird_audio.task2_scoring import (
    KNOWN_TRAINING_ROLE,
    KNOWN_VALIDATION_ROLE,
    LATENT_SCORE_NAME,
    NOVELTY_DIRECTION,
    RECONSTRUCTION_SCORE_NAME,
)
from bird_audio.task2_training import DEFAULT_RUN_ROOT as TASK2_DEFAULT_RUN_ROOT
from bird_audio.task2_training import (
    DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
    EXPECTED_PARAMETER_COUNT,
    load_locked_task2_config,
    load_task2_checkpoint,
    verify_task2_development_run,
)
from bird_audio.task2_training import (
    KNOWN_CACHE_LOCK_SHA256 as TASK2_KNOWN_CACHE_LOCK_SHA256,
)
from bird_audio.task2_training import LOCKED_CONFIG_PATH as TASK2_CONFIG_PATH
from bird_audio.task2_training import PRODUCTION_SCOPE as TASK2_PRODUCTION_SCOPE
from bird_audio.task2_training import RUN_SCHEMA_VERSION as TASK2_RUN_SCHEMA_VERSION
from bird_audio.unknown_clip_cache import load_unknown_scoring_clip_cache

FINAL_EVALUATION_GATE_SCHEMA_VERSION = "1.0"
FINAL_EVALUATION_GATE_ID = "final_evaluation_gate_v2"
FINAL_EVALUATION_GATE_DIRECTORY = PROJECT_ROOT / "runs" / "final_evaluation_v2" / "gate_v2"
FINAL_EVALUATION_GATE_PATH = FINAL_EVALUATION_GATE_DIRECTORY / "gate.json"
FINAL_EVALUATION_GATE_LOCK_PATH = FINAL_EVALUATION_GATE_DIRECTORY / "lock.json"

TASK1_RUN_ROOT = TASK1_DEFAULT_RUN_ROOT
TASK2_RUN_ROOT = TASK2_DEFAULT_RUN_ROOT
KNOWN_CACHE_LOCK_PATH = PROJECT_ROOT / "data" / "processed" / "known_clips_v1" / "lock.json"
UNKNOWN_CACHE_LOCK_PATH = PROJECT_ROOT / "data" / "processed" / "unknown_clips_v2" / "lock.json"
EXPECTED_KNOWN_CACHE_LOCK_SHA256 = TASK1_KNOWN_CACHE_LOCK_SHA256
EXPECTED_UNKNOWN_CACHE_LOCK_SHA256 = (
    "222ca630ce28ea05998c74592ad6c47795cde75176db1fcce6930dcbf49fe91b"
)

SEED_ORDER = (13, 37, 71)
PRODUCTION_DATA_COUNTS = {
    "train_clips": 5_319,
    "train_recordings": 1_254,
    "validation_clips": 1_138,
    "validation_recordings": 271,
}
_SHA256_LENGTH = 64
_ALLOWED_RUN_ROOT_FILES = {".gitkeep"}

_TASK1_RUN_IDENTITY_FIELDS = {
    "schema_version",
    "run_id",
    "task",
    "seed",
    "config_sha256",
    "cache_lock_sha256",
    "weight_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "scope",
    "production_evidence",
}
_TASK1_RESULT_FIELDS = {
    "schema_version",
    "complete",
    "run_id",
    "run_directory",
    "run_identity_sha256",
    "config_sha256",
    "cache_lock_sha256",
    "weight_sha256",
    "source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "scope",
    "production_evidence",
    "resumed",
    "resume_checkpoint",
    "epochs_completed",
    "early_stopped",
    "best_epoch",
    "best_validation_macro_f1",
    "best_validation_loss",
    "best_checkpoint",
    "latest_recovery_checkpoint",
    "artifacts",
}
_TASK1_ARTIFACT_FIELDS = {
    "resolved_config",
    "run_identity",
    "provenance",
    "epoch_history",
    "best_validation_predictions",
    "best_checkpoint",
    "latest_recovery",
}
_TASK1_COMPLETION_FIELDS = {
    "schema_version",
    "run_identity_sha256",
    "source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "scope",
    "production_evidence",
    "result",
}
_TASK1_PROVENANCE_FIELDS = {
    "schema_version",
    "created_at_utc",
    "run_identity_sha256",
    "command",
    "config_path",
    "config_file_sha256",
    "config_sha256",
    "cache_root",
    "cache_lock_sha256",
    "weight_path",
    "weight_sha256",
    "weight_size_bytes",
    "source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_path",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "numerical_runtime",
    "scope",
    "production_evidence",
    "environment",
    "parameter_counts",
    "optimizer_groups",
    "initial_artifacts",
}

_TASK2_RUN_IDENTITY_FIELDS = {
    "schema_version",
    "run_id",
    "task",
    "seed",
    "config_sha256",
    "config_file_sha256",
    "cache_lock_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime",
    "numerical_runtime_sha256",
    "model_contract",
    "model_contract_sha256",
    "optimizer_contract",
    "final_evaluation_contract",
    "scope",
    "production_evidence",
    "limits",
    "data",
}
_TASK2_RESULT_FIELDS = {
    "schema_version",
    "complete",
    "started_at_utc",
    "completed_at_utc",
    "run_id",
    "run_directory",
    "run_identity_sha256",
    "config_sha256",
    "config_file_sha256",
    "cache_lock_sha256",
    "release_source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "model_contract_sha256",
    "scope",
    "production_evidence",
    "resumed",
    "resume_checkpoint",
    "epochs_completed",
    "early_stopped",
    "best_epoch",
    "best_validation_loss",
    "best_checkpoint",
    "latest_recovery_checkpoint",
    "development_bundle",
    "artifacts",
}
_TASK2_ARTIFACT_FIELDS = {
    "resolved_config",
    "run_identity",
    "provenance",
    "epoch_history",
    "best_checkpoint",
    "latest_recovery",
    "development",
    "development_bundle",
}
_TASK2_DEVELOPMENT_ARTIFACT_FIELDS = {
    "known_training_scores",
    "known_validation_scores",
    "training_latent_reference",
    "thresholds",
}
_TASK2_PROVENANCE_FIELDS = {
    "schema_version",
    "started_at_utc",
    "run_identity_sha256",
    "command",
    "config_path",
    "config_file_sha256",
    "config_sha256",
    "cache_root",
    "cache_lock_sha256",
    "release_source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime",
    "numerical_runtime_sha256",
    "model_contract",
    "model_contract_sha256",
    "optimizer_contract",
    "final_evaluation_contract",
    "scope",
    "production_evidence",
    "initial_artifacts",
}
_TASK2_COMPLETION_FIELDS = {
    "schema_version",
    "run_identity_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "scope",
    "production_evidence",
    "result",
    "development_bundle",
}
_TASK2_DEVELOPMENT_BINDING_FIELDS = {
    "run_identity_sha256",
    "config_sha256",
    "config_file_sha256",
    "cache_lock_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "model_contract_sha256",
    "scope",
    "production_evidence",
    "seed",
    "best_checkpoint_sha256",
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} is not a lowercase SHA-256 value")
    return value


def _require_exact_mapping(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} schema is invalid")
    return value


def _require_finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _require_utc_timestamp(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC")
    return value


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _is_within(path: Path, boundary: Path) -> bool:
    return path == boundary or boundary in path.parents


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Gate filesystem access requires a nonzero O_NOFOLLOW")
    if not isinstance(directory, int) or directory == 0:
        raise RuntimeError("Gate filesystem access requires a nonzero O_DIRECTORY")
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | directory


def _open_absolute_directory_no_follow(path: Path) -> int:
    candidate = _absolute(path)
    flags = _directory_open_flags()
    descriptor = os.open("/", flags)
    try:
        for part in candidate.parts[1:]:
            if part in {"", ".", ".."}:
                raise ValueError("Gate directory component is invalid")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Gate directory component changed type")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_parent(path: Path, boundary: Path) -> tuple[int, str]:
    candidate = _absolute(path)
    root = _absolute(boundary)
    if candidate == root or not _is_within(candidate, root):
        raise ValueError(f"Gate artifact leaves its boundary: {candidate}")
    parts = candidate.relative_to(root).parts
    if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
        raise ValueError("Gate artifact path is invalid")
    descriptor = _open_absolute_directory_no_follow(root)
    flags = _directory_open_flags()
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Gate artifact parent changed type")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _secure_ensure_directory(path: Path, boundary: Path) -> Path:
    candidate = _absolute(path)
    root = _absolute(boundary)
    if candidate == root:
        descriptor = _open_absolute_directory_no_follow(root)
        os.close(descriptor)
        return candidate
    if not _is_within(candidate, root):
        raise ValueError("Gate directory leaves its boundary")
    parts = candidate.relative_to(root).parts
    descriptor = _open_absolute_directory_no_follow(root)
    flags = _directory_open_flags()
    try:
        for part in parts:
            if part in {"", ".", ".."} or "/" in part:
                raise ValueError("Gate directory path is invalid")
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Gate directory path changed type")
            os.close(descriptor)
            descriptor = next_descriptor
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return candidate


def _json_bytes(value: Any) -> bytes:
    try:
        payload = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Gate JSON value is not finite and serializable") from exc
    return payload


def _require_directory(path: Path, name: str) -> None:
    try:
        descriptor = _open_absolute_directory_no_follow(_absolute(path))
    except OSError as exc:
        raise ValueError(f"{name} directory is unavailable: {path}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{name} is not a direct directory: {path}")
    finally:
        os.close(descriptor)


def _require_parent_directories(path: Path, boundary: Path) -> None:
    try:
        descriptor, _ = _open_relative_parent(path, boundary)
    except OSError as exc:
        raise ValueError(f"Artifact parent cannot be opened safely: {path}") from exc
    os.close(descriptor)


def _snapshot_file_descriptor(descriptor: int, path: Path) -> tuple[bytes, str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Gate artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise RuntimeError(f"Gate artifact ended while being read: {path}")
        chunks.append(chunk)
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if offset != before.st_size or before_identity != after_identity:
        raise RuntimeError(f"Gate artifact changed while being read: {path}")
    return b"".join(chunks), digest.hexdigest(), before.st_size


def _descriptor_snapshot(path: Path, *, boundary: Path = PROJECT_ROOT) -> tuple[bytes, str, int]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Gate artifact reads require O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    parent_descriptor: int | None = None
    try:
        parent_descriptor, filename = _open_relative_parent(path, boundary)
        descriptor = os.open(filename, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError(f"Gate artifact cannot be opened: {path}") from exc
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    try:
        return _snapshot_file_descriptor(descriptor, path)
    finally:
        os.close(descriptor)


def _artifact_record(
    value: object,
    expected_path: Path,
    name: str,
    *,
    boundary: Path,
) -> dict[str, Any]:
    record = _require_exact_mapping(value, {"path", "sha256", "size_bytes"}, name)
    expected = _absolute(expected_path)
    if type(record["path"]) is not str or _absolute(record["path"]) != expected:
        raise ValueError(f"{name} path is not canonical")
    expected_sha256 = _require_sha256(record["sha256"], f"{name} SHA-256")
    expected_size = record["size_bytes"]
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError(f"{name} size is invalid")
    _require_parent_directories(expected, boundary)
    _, observed_sha256, observed_size = _descriptor_snapshot(expected, boundary=boundary)
    if observed_sha256 != expected_sha256 or observed_size != expected_size:
        raise ValueError(f"{name} descriptor does not match its record")
    return {"path": str(expected), "sha256": expected_sha256, "size_bytes": expected_size}


def _read_canonical_json(
    path: Path,
    *,
    expected_record: object | None = None,
    name: str,
    boundary: Path,
) -> tuple[Any, dict[str, Any]]:
    expected = _absolute(path)
    if expected_record is not None:
        record = _artifact_record(expected_record, expected, name, boundary=boundary)
    else:
        _require_parent_directories(expected, boundary)
        _, sha256, size_bytes = _descriptor_snapshot(expected, boundary=boundary)
        record = {"path": str(expected), "sha256": sha256, "size_bytes": size_bytes}
    payload, sha256, size_bytes = _descriptor_snapshot(expected, boundary=boundary)
    if sha256 != record["sha256"] or size_bytes != record["size_bytes"]:
        raise RuntimeError(f"{name} changed between descriptor reads")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if _json_bytes(value) != payload:
        raise ValueError(f"{name} is not canonical JSON")
    return value, record


def _verify_binary_record(
    value: object,
    expected_path: Path,
    name: str,
    *,
    boundary: Path,
) -> dict[str, Any]:
    return _artifact_record(value, expected_path, name, boundary=boundary)


def _canonical_config_record(
    path: Path,
    config: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    expected = _absolute(path)
    _require_parent_directories(expected, PROJECT_ROOT)
    payload, sha256, size_bytes = _descriptor_snapshot(expected)
    try:
        snapshot_config = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{name} configuration is invalid") from exc
    if snapshot_config != public_config(dict(config)):
        raise RuntimeError(f"{name} configuration changed while it was loaded")
    return {
        "path": str(expected),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "config_sha256": config_fingerprint(config),
        "resolved": public_config(config),
    }


def _canonical_configs() -> dict[str, Any]:
    task1_config = load_final_task1_config()
    task2_config = load_locked_task2_config()
    return {
        "task1": _canonical_config_record(TASK1_CONFIG_PATH, task1_config, "Task 1"),
        "task2": _canonical_config_record(TASK2_CONFIG_PATH, task2_config, "Task 2"),
    }


def _cache_lock_record(
    path: Path,
    *,
    expected_sha256: str,
    expected_version: str,
    name: str,
) -> dict[str, Any]:
    expected = _absolute(path)
    _require_parent_directories(expected, PROJECT_ROOT)
    payload, sha256, size_bytes = _descriptor_snapshot(expected)
    if sha256 != _require_sha256(expected_sha256, f"Expected {name} lock SHA-256"):
        raise ValueError(f"{name} lock differs from the published lock")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} lock is not valid JSON") from exc
    if _json_bytes(value) != payload:
        raise ValueError(f"{name} lock is not canonical JSON")
    lock = _require_exact_mapping(
        value,
        {"schema_version", "cache_version", "cache_content_sha256", "provenance", "artifacts"},
        f"{name} lock",
    )
    if lock["schema_version"] != "1.0" or lock["cache_version"] != expected_version:
        raise ValueError(f"{name} lock identity is invalid")
    content_sha256 = _require_sha256(lock["cache_content_sha256"], f"{name} cache content SHA-256")
    provenance = lock["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{name} lock provenance is invalid")
    requirements_sha256 = _require_sha256(
        provenance.get("requirements_lock_sha256"),
        f"{name} requirements lock SHA-256",
    )
    if not isinstance(lock["artifacts"], Mapping) or not lock["artifacts"]:
        raise ValueError(f"{name} cache artifact metadata is invalid")
    return {
        "path": str(expected),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "cache_version": expected_version,
        "cache_content_sha256": content_sha256,
        "requirements_lock_sha256": requirements_sha256,
    }


def _cache_locks() -> dict[str, Any]:
    if TASK1_KNOWN_CACHE_LOCK_SHA256 != TASK2_KNOWN_CACHE_LOCK_SHA256:
        raise RuntimeError("Task cache lock constants do not agree")
    if EXPECTED_KNOWN_CACHE_LOCK_SHA256 != TASK1_KNOWN_CACHE_LOCK_SHA256:
        raise RuntimeError("Final gate known-cache constant does not agree with the tasks")
    known = _cache_lock_record(
        KNOWN_CACHE_LOCK_PATH,
        expected_sha256=EXPECTED_KNOWN_CACHE_LOCK_SHA256,
        expected_version="known_clips_v1",
        name="Known cache",
    )
    unknown = _cache_lock_record(
        UNKNOWN_CACHE_LOCK_PATH,
        expected_sha256=EXPECTED_UNKNOWN_CACHE_LOCK_SHA256,
        expected_version="unknown_clips_v2",
        name="Unknown cache",
    )
    unknown_cache = load_unknown_scoring_clip_cache(
        UNKNOWN_CACHE_LOCK_PATH.parent,
        expected_lock_sha256=EXPECTED_UNKNOWN_CACHE_LOCK_SHA256,
    )
    if (
        _absolute(unknown_cache.root) != _absolute(UNKNOWN_CACHE_LOCK_PATH.parent)
        or unknown_cache.lock_sha256 != EXPECTED_UNKNOWN_CACHE_LOCK_SHA256
    ):
        raise ValueError("Unknown scoring cache binding is invalid")
    return {"known": known, "unknown": unknown}


def _recovery_artifact_record(
    value: object,
    expected_path: Path,
    name: str,
) -> dict[str, Any]:
    record = _require_exact_mapping(value, {"path", "sha256", "size_bytes"}, name)
    expected = _absolute(expected_path)
    project = _absolute(PROJECT_ROOT)
    if not _is_within(expected, project) or expected == project:
        raise ValueError(f"{name} leaves the project")
    expected_label = expected.relative_to(project).as_posix()
    if record["path"] != expected_label:
        raise ValueError(f"{name} path is not canonical")
    expected_sha256 = _require_sha256(record["sha256"], f"{name} SHA-256")
    expected_size = record["size_bytes"]
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError(f"{name} size is invalid")
    _, observed_sha256, observed_size = _descriptor_snapshot(expected, boundary=project)
    if observed_sha256 != expected_sha256 or observed_size != expected_size:
        raise ValueError(f"{name} descriptor does not match its artifact")
    return {
        "path": str(expected),
        "sha256": expected_sha256,
        "size_bytes": expected_size,
    }


def _recovery_evidence(
    current_source_fingerprint_sha256: str,
    expected_v2_cache_lock_sha256: str,
) -> dict[str, Any]:
    from bird_audio.recovery_v2 import (
        EQUIVALENCE_ID,
        EQUIVALENCE_LOCK_PATH,
        EQUIVALENCE_PATH,
        RECOVERY_LOCK_PATH,
        RECOVERY_MANIFEST_ID,
        RECOVERY_MANIFEST_PATH,
        verify_unknown_cache_v2_equivalence_certificate,
    )

    current_source = _require_sha256(
        current_source_fingerprint_sha256,
        "Current recovery source fingerprint",
    )
    expected_v2_lock = _require_sha256(
        expected_v2_cache_lock_sha256,
        "Expected recovery v2 cache lock SHA-256",
    )
    verified = verify_unknown_cache_v2_equivalence_certificate(
        full_rederivation=False,
    )
    if not isinstance(verified, Mapping) or verified.get("valid") is not True:
        raise ValueError("V2 cache equivalence verification is invalid")
    certificate = verified.get("equivalence")
    if not isinstance(certificate, Mapping):
        raise ValueError("V2 cache equivalence certificate is invalid")
    equivalence = certificate.get("equivalence")
    if not isinstance(equivalence, Mapping):
        raise ValueError("V2 cache scientific equivalence is invalid")
    if (
        certificate.get("schema_version") != "1.0"
        or certificate.get("equivalence_id") != EQUIVALENCE_ID
        or certificate.get("source_fingerprint_sha256") != current_source
        or certificate.get("complete") is not True
        or equivalence.get("valid") is not True
        or equivalence.get("full_rederivation") is not True
        or equivalence.get("scientific_artifacts_identical") is not True
        or equivalence.get("file_inodes_disjoint") is not True
        or equivalence.get("v2_cache_lock_sha256") != expected_v2_lock
    ):
        raise ValueError("V2 cache equivalence is not the locked scientific identity")
    _require_utc_timestamp(
        certificate.get("certified_at_utc"),
        "V2 cache equivalence certification time",
    )
    equivalence_record = _recovery_artifact_record(
        verified.get("equivalence_artifact"),
        EQUIVALENCE_PATH,
        "V2 cache equivalence certificate",
    )
    equivalence_lock_record = _recovery_artifact_record(
        verified.get("lock_artifact"),
        EQUIVALENCE_LOCK_PATH,
        "V2 cache equivalence lock",
    )
    recovery_manifest_sha256 = _require_sha256(
        equivalence.get("v1_recovery_manifest_sha256"),
        "V1 recovery manifest SHA-256",
    )
    recovery_lock_sha256 = _require_sha256(
        equivalence.get("v1_recovery_lock_sha256"),
        "V1 recovery lock SHA-256",
    )
    recovery_manifest_record = _recovery_artifact_record(
        {
            "path": RECOVERY_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": recovery_manifest_sha256,
            "size_bytes": RECOVERY_MANIFEST_PATH.stat().st_size,
        },
        RECOVERY_MANIFEST_PATH,
        "V1 recovery manifest",
    )
    recovery_lock_record = _recovery_artifact_record(
        {
            "path": RECOVERY_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": recovery_lock_sha256,
            "size_bytes": RECOVERY_LOCK_PATH.stat().st_size,
        },
        RECOVERY_LOCK_PATH,
        "V1 recovery lock",
    )
    return {
        "source_fingerprint_sha256": current_source,
        "v1_recovery": {
            "manifest_id": RECOVERY_MANIFEST_ID,
            "manifest": recovery_manifest_record,
            "lock": recovery_lock_record,
        },
        "v2_cache_equivalence": {
            "equivalence_id": EQUIVALENCE_ID,
            "certificate": equivalence_record,
            "lock": equivalence_lock_record,
            "full_rederivation": True,
            "scientific_artifacts_identical": True,
            "v2_cache_lock_sha256": expected_v2_lock,
        },
    }


def _completed_run_directories(root: Path, task_name: str) -> tuple[Path, ...]:
    expected_root = _absolute(root)
    completed: list[Path] = []
    try:
        root_descriptor = _open_absolute_directory_no_follow(expected_root)
    except OSError as exc:
        raise ValueError(f"{task_name} run root cannot be opened safely") from exc
    directory_flags = _directory_open_flags()
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        with os.scandir(root_descriptor) as entries:
            for entry in entries:
                path = expected_root / entry.name
                if entry.name in _ALLOWED_RUN_ROOT_FILES:
                    try:
                        marker = os.open(entry.name, file_flags, dir_fd=root_descriptor)
                    except OSError as exc:
                        raise ValueError(f"{task_name} run-root marker is invalid") from exc
                    try:
                        if not stat.S_ISREG(os.fstat(marker).st_mode):
                            raise ValueError(f"{task_name} run-root marker is invalid")
                    finally:
                        os.close(marker)
                    continue
                try:
                    run_descriptor = os.open(
                        entry.name,
                        directory_flags,
                        dir_fd=root_descriptor,
                    )
                except OSError as exc:
                    entry_kind = (
                        "symbolic link"
                        if entry.is_symlink()
                        else "unexpected file"
                        if entry.is_file(follow_symlinks=False)
                        else "unsafe entry"
                    )
                    raise ValueError(
                        f"{task_name} run root contains a {entry_kind}: {entry.name}"
                    ) from exc
                try:
                    if not stat.S_ISDIR(os.fstat(run_descriptor).st_mode):
                        raise ValueError(
                            f"{task_name} run root contains an unexpected file: {entry.name}"
                        )
                    for required_name in ("result.json", "result.lock.json"):
                        try:
                            required = os.open(
                                required_name,
                                file_flags,
                                dir_fd=run_descriptor,
                            )
                        except OSError as exc:
                            raise ValueError(f"{task_name} run is partial: {entry.name}") from exc
                        try:
                            if not stat.S_ISREG(os.fstat(required).st_mode):
                                raise ValueError(f"{task_name} run is partial: {entry.name}")
                        finally:
                            os.close(required)
                finally:
                    os.close(run_descriptor)
                completed.append(path)
    finally:
        os.close(root_descriptor)
    if len(completed) != len(SEED_ORDER):
        raise ValueError(f"{task_name} requires exactly three complete runs")
    return tuple(sorted(completed, key=lambda path: path.name))


def _validate_resolved_config(
    value: object,
    *,
    task: str,
    config_record: Mapping[str, Any],
    relative_path: str,
) -> None:
    resolved = _require_exact_mapping(
        value,
        {"config_path", "config_file_sha256", "config_sha256", "resolved"},
        f"{task} resolved configuration",
    )
    expected = {
        "config_path": relative_path,
        "config_file_sha256": config_record["sha256"],
        "config_sha256": config_record["config_sha256"],
        "resolved": config_record["resolved"],
    }
    if dict(resolved) != expected:
        raise ValueError(f"{task} resolved configuration changed")


def _checkpoint_common_matches(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    name: str,
) -> None:
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"{name} changed bound field: {key}")


def _record_inside_directory(value: object, directory: Path, name: str) -> Path:
    record = _require_exact_mapping(value, {"path", "sha256", "size_bytes"}, name)
    if type(record["path"]) is not str:
        raise ValueError(f"{name} path is invalid")
    path = _absolute(record["path"])
    expected_directory = _absolute(directory)
    if path.parent != expected_directory:
        raise ValueError(f"{name} path is outside its canonical directory")
    return path


def _validate_resume_state(result: Mapping[str, Any], run_directory: Path, task: str) -> None:
    if type(result["resumed"]) is not bool:
        raise ValueError(f"{task} resumed flag is invalid")
    resume = result["resume_checkpoint"]
    if result["resumed"] is not (resume is not None):
        raise ValueError(f"{task} resume record is inconsistent")
    if resume is not None:
        _record_inside_directory(resume, run_directory / "recovery", f"{task} resume checkpoint")


def _validate_result_scalars(result: Mapping[str, Any], task: str) -> None:
    if (
        result["complete"] is not True
        or type(result["epochs_completed"]) is not int
        or result["epochs_completed"] <= 0
        or type(result["early_stopped"]) is not bool
        or type(result["best_epoch"]) is not int
        or not 1 <= result["best_epoch"] <= result["epochs_completed"]
    ):
        raise ValueError(f"{task} completion state is invalid")
    _require_finite_float(
        result["best_validation_loss"], f"{task} validation loss", nonnegative=True
    )


def _task1_prediction_records(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    state = checkpoint.get("predictions")
    if not isinstance(state, Mapping):
        raise ValueError("Task 1 checkpoint predictions are unavailable")
    try:
        recording_ids = tuple(state["recording_ids"])
        session_groups = tuple(state["session_groups"])
        true_labels = state["true_labels"].detach().cpu().tolist()
        mean_logits = state["mean_logits"].detach().cpu().tolist()
        predicted_labels = state["predicted_labels"].detach().cpu().tolist()
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Task 1 checkpoint predictions cannot be read") from exc
    if not (
        len(recording_ids)
        == len(session_groups)
        == len(true_labels)
        == len(mean_logits)
        == len(predicted_labels)
    ):
        raise ValueError("Task 1 checkpoint prediction lengths differ")
    return [
        {
            "recording_id": recording_id,
            "session_group": session_groups[index],
            "true_class_index": int(true_labels[index]),
            "predicted_class_index": int(predicted_labels[index]),
            "mean_logits": [float(value) for value in mean_logits[index]],
        }
        for index, recording_id in enumerate(recording_ids)
    ]


def _validate_task1_provenance(
    value: object,
    *,
    result: Mapping[str, Any],
    identity: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    config_record: Mapping[str, Any],
) -> None:
    provenance = _require_exact_mapping(value, _TASK1_PROVENANCE_FIELDS, "Task 1 provenance")
    bindings = {
        "schema_version": TASK1_RUN_SCHEMA_VERSION,
        "run_identity_sha256": result["run_identity_sha256"],
        "config_path": "configs/task1/final.toml",
        "config_file_sha256": config_record["sha256"],
        "config_sha256": result["config_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "weight_sha256": result["weight_sha256"],
        "source_fingerprint_sha256": result["source_fingerprint_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": TASK1_PRODUCTION_SCOPE,
        "production_evidence": True,
    }
    for key, expected in bindings.items():
        if provenance[key] != expected:
            raise ValueError(f"Task 1 provenance changed bound field: {key}")
    if (
        not isinstance(provenance["command"], list)
        or not isinstance(provenance["numerical_runtime"], Mapping)
        or sha256_json(provenance["numerical_runtime"]) != result["numerical_runtime_sha256"]
        or provenance["initial_artifacts"]
        != {
            "resolved_config": artifacts["resolved_config"],
            "run_identity": artifacts["run_identity"],
        }
        or not isinstance(provenance["environment"], Mapping)
        or provenance["environment"].get("device") != "mps"
        or provenance["environment"].get("mps_built") is not True
        or provenance["environment"].get("mps_available") is not True
        or provenance["environment"].get("deterministic_algorithms") is not True
    ):
        raise ValueError("Task 1 provenance is not production evidence")
    if identity["numerical_runtime_sha256"] != sha256_json(provenance["numerical_runtime"]):
        raise ValueError("Task 1 numerical runtime binding is invalid")


def _load_task1_checkpoint_record(
    record_value: object,
    expected_path: Path,
    *,
    run_directory: Path,
    run_identity_sha256: str,
    expected_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _verify_binary_record(
        record_value,
        expected_path,
        f"Task 1 {expected_type} checkpoint",
        boundary=run_directory,
    )
    checkpoint = load_task1_checkpoint(
        expected_path,
        expected_sha256=record["sha256"],
        expected_run_identity_sha256=run_identity_sha256,
        expected_type=expected_type,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Task 1 checkpoint loader returned an invalid value")
    return checkpoint, record


def _inspect_task1_run(
    run_directory: Path,
    *,
    config_record: Mapping[str, Any],
) -> dict[str, Any]:
    task = "Task 1"
    completion, completion_record = _read_canonical_json(
        run_directory / "result.lock.json",
        name="Task 1 completion lock",
        boundary=run_directory,
    )
    completion = _require_exact_mapping(completion, _TASK1_COMPLETION_FIELDS, task)
    if completion["schema_version"] != TASK1_RUN_SCHEMA_VERSION:
        raise ValueError("Task 1 completion lock version is unsupported")
    recursive_verification = verify_task1_development_run(
        run_directory / "result.lock.json",
        expected_sha256=completion_record["sha256"],
        require_production=True,
    )
    recursive_verification = _require_exact_mapping(
        recursive_verification,
        {
            "valid",
            "complete",
            "run_id",
            "seed",
            "scope",
            "production_evidence",
            "completion_lock_sha256",
            "run_identity_sha256",
            "best_checkpoint_sha256",
            "validation_recordings",
            "validation_classes",
            "macro_f1_rederived",
            "selection_rederived",
            "resume_prefix_verified",
        },
        "Task 1 recursive verification",
    )
    if (
        recursive_verification["valid"] is not True
        or recursive_verification["complete"] is not True
        or recursive_verification["scope"] != TASK1_PRODUCTION_SCOPE
        or recursive_verification["production_evidence"] is not True
        or recursive_verification["completion_lock_sha256"] != completion_record["sha256"]
        or recursive_verification["validation_recordings"]
        != PRODUCTION_DATA_COUNTS["validation_recordings"]
        or recursive_verification["validation_classes"] != 15
        or recursive_verification["macro_f1_rederived"] is not True
        or recursive_verification["selection_rederived"] is not True
        or recursive_verification["resume_prefix_verified"] is not True
    ):
        raise ValueError("Task 1 recursive verification is not production evidence")
    result, result_record = _read_canonical_json(
        run_directory / "result.json",
        expected_record=completion["result"],
        name="Task 1 result",
        boundary=run_directory,
    )
    result = _require_exact_mapping(result, _TASK1_RESULT_FIELDS, "Task 1 result")
    _validate_result_scalars(result, task)
    _validate_resume_state(result, run_directory, task)
    if (
        result["schema_version"] != TASK1_RUN_SCHEMA_VERSION
        or result["run_id"] != run_directory.name
        or result["run_directory"] != str(_absolute(run_directory))
        or result["scope"] != TASK1_PRODUCTION_SCOPE
        or result["production_evidence"] is not True
    ):
        raise ValueError("Task 1 result is not canonical production evidence")
    seed = result.get("seed")
    artifacts = _require_exact_mapping(
        result["artifacts"], _TASK1_ARTIFACT_FIELDS, "Task 1 artifacts"
    )
    identity, identity_record = _read_canonical_json(
        run_directory / "run_identity.json",
        expected_record=artifacts["run_identity"],
        name="Task 1 run identity",
        boundary=run_directory,
    )
    identity = _require_exact_mapping(identity, _TASK1_RUN_IDENTITY_FIELDS, "Task 1 identity")
    seed = identity["seed"]
    if type(seed) is not int or seed not in SEED_ORDER:
        raise ValueError("Task 1 seed is invalid")
    if sha256_json(identity) != result["run_identity_sha256"]:
        raise ValueError("Task 1 run identity SHA-256 is invalid")
    identity_bindings = {
        "schema_version": TASK1_RUN_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "task": "task1_classification",
        "config_sha256": result["config_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "weight_sha256": result["weight_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": TASK1_PRODUCTION_SCOPE,
        "production_evidence": True,
    }
    for key, expected in identity_bindings.items():
        if identity[key] != expected:
            raise ValueError(f"Task 1 identity changed bound field: {key}")
    completion_bindings = {
        "run_identity_sha256": result["run_identity_sha256"],
        "source_fingerprint_sha256": result["source_fingerprint_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": TASK1_PRODUCTION_SCOPE,
        "production_evidence": True,
    }
    for key, expected in completion_bindings.items():
        if completion[key] != expected:
            raise ValueError(f"Task 1 completion lock changed bound field: {key}")
    for key in (
        "run_identity_sha256",
        "config_sha256",
        "cache_lock_sha256",
        "weight_sha256",
        "source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
    ):
        _require_sha256(result[key], f"Task 1 {key}")
    if result["config_sha256"] != config_record["config_sha256"]:
        raise ValueError("Task 1 result uses another configuration")

    resolved_config, _ = _read_canonical_json(
        run_directory / "resolved_config.json",
        expected_record=artifacts["resolved_config"],
        name="Task 1 resolved configuration",
        boundary=run_directory,
    )
    _validate_resolved_config(
        resolved_config,
        task=task,
        config_record=config_record,
        relative_path="configs/task1/final.toml",
    )
    provenance, _ = _read_canonical_json(
        run_directory / "provenance.json",
        expected_record=artifacts["provenance"],
        name="Task 1 provenance",
        boundary=run_directory,
    )
    _validate_task1_provenance(
        provenance,
        result=result,
        identity=identity,
        artifacts=artifacts,
        config_record=config_record,
    )

    history, _ = _read_canonical_json(
        run_directory / "epoch_history.json",
        expected_record=artifacts["epoch_history"],
        name="Task 1 epoch history",
        boundary=run_directory,
    )
    if not isinstance(history, list) or len(history) != result["epochs_completed"]:
        raise ValueError("Task 1 epoch history length is invalid")
    predictions, _ = _read_canonical_json(
        run_directory / "best_validation_predictions.json",
        expected_record=artifacts["best_validation_predictions"],
        name="Task 1 validation predictions",
        boundary=run_directory,
    )
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Task 1 validation predictions are empty")

    best_path = run_directory / "best_candidates" / f"best_epoch_{result['best_epoch']:04d}.pt"
    best_checkpoint, best_record = _load_task1_checkpoint_record(
        result["best_checkpoint"],
        best_path,
        run_directory=run_directory,
        run_identity_sha256=result["run_identity_sha256"],
        expected_type="best",
    )
    if best_record != artifacts["best_checkpoint"]:
        raise ValueError("Task 1 best checkpoint records differ")
    model_state_verification = verify_locked_task1_best_checkpoint_model_state(
        best_path,
        expected_sha256=best_record["sha256"],
        expected_run_identity_sha256=result["run_identity_sha256"],
    )
    model_state_verification = _require_exact_mapping(
        model_state_verification,
        {
            "valid",
            "schema_version",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "run_id",
            "run_identity_sha256",
            "config_sha256",
            "cache_lock_sha256",
            "weight_sha256",
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
            "scope",
            "production_evidence",
            "seed",
            "epoch",
            "score",
            "model_contract",
        },
        "Task 1 model-state verification",
    )
    expected_model_contract = {
        "architecture": "efficientnet_b0",
        "model_type": "bird_audio.models.EfficientNetB0Classifier",
        "class_count": 15,
        "dropout": 0.2,
        "classifier_in_features": 1_280,
        "trainable_feature_indices": [6, 7, 8],
        "frozen_feature_indices": [0, 1, 2, 3, 4, 5],
        "parameter_counts": {"total": 4_026_763, "trainable": 3_174_955},
        "state_tensor_count": 360,
    }
    if (
        model_state_verification["valid"] is not True
        or model_state_verification["schema_version"] != TASK1_CHECKPOINT_SCHEMA_VERSION
        or model_state_verification["checkpoint_path"] != str(best_path)
        or model_state_verification["checkpoint_sha256"] != best_record["sha256"]
        or model_state_verification["checkpoint_size_bytes"] != best_record["size_bytes"]
        or model_state_verification["run_id"] != result["run_id"]
        or model_state_verification["run_identity_sha256"] != result["run_identity_sha256"]
        or model_state_verification["config_sha256"] != result["config_sha256"]
        or model_state_verification["cache_lock_sha256"] != result["cache_lock_sha256"]
        or model_state_verification["weight_sha256"] != result["weight_sha256"]
        or model_state_verification["implementation_sha256"] != result["implementation_sha256"]
        or model_state_verification["requirements_lock_sha256"]
        != result["requirements_lock_sha256"]
        or model_state_verification["numerical_runtime_sha256"]
        != result["numerical_runtime_sha256"]
        or model_state_verification["scope"] != TASK1_PRODUCTION_SCOPE
        or model_state_verification["production_evidence"] is not True
        or model_state_verification["seed"] != seed
        or model_state_verification["epoch"] != result["best_epoch"]
        or model_state_verification["model_contract"] != expected_model_contract
    ):
        raise ValueError("Task 1 model-state verification differs from the completed run")
    checkpoint_common = {
        "schema_version": TASK1_CHECKPOINT_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "weight_sha256": result["weight_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": TASK1_PRODUCTION_SCOPE,
        "production_evidence": True,
        "seed": seed,
    }
    _checkpoint_common_matches(best_checkpoint, checkpoint_common, "Task 1 best checkpoint")
    score = best_checkpoint.get("score")
    if (
        best_checkpoint.get("epoch") != result["best_epoch"]
        or not isinstance(score, Mapping)
        or score.get("epoch") != result["best_epoch"]
        or score.get("macro_f1") != result["best_validation_macro_f1"]
        or score.get("validation_loss") != result["best_validation_loss"]
        or model_state_verification["score"] != dict(score)
    ):
        raise ValueError("Task 1 best checkpoint score differs from the result")
    _require_finite_float(
        result["best_validation_macro_f1"], "Task 1 validation macro F1", nonnegative=True
    )
    if _task1_prediction_records(best_checkpoint) != predictions:
        raise ValueError("Task 1 prediction artifact differs from the best checkpoint")

    latest_path = run_directory / "recovery" / f"recovery_epoch_{result['epochs_completed']:04d}.pt"
    latest_checkpoint, latest_record = _load_task1_checkpoint_record(
        result["latest_recovery_checkpoint"],
        latest_path,
        run_directory=run_directory,
        run_identity_sha256=result["run_identity_sha256"],
        expected_type="recovery",
    )
    if latest_record != artifacts["latest_recovery"]:
        raise ValueError("Task 1 latest recovery records differ")
    _checkpoint_common_matches(latest_checkpoint, checkpoint_common, "Task 1 recovery checkpoint")
    if latest_checkpoint.get("completed_epoch") != result["epochs_completed"]:
        raise ValueError("Task 1 latest recovery epoch differs from the result")
    if latest_checkpoint.get("history") != history:
        raise ValueError("Task 1 history artifact differs from the latest recovery")
    if result["resume_checkpoint"] is not None:
        resume_path = _record_inside_directory(
            result["resume_checkpoint"], run_directory / "recovery", "Task 1 resume checkpoint"
        )
        _load_task1_checkpoint_record(
            result["resume_checkpoint"],
            resume_path,
            run_directory=run_directory,
            run_identity_sha256=result["run_identity_sha256"],
            expected_type="recovery",
        )
    if (
        recursive_verification["run_id"] != result["run_id"]
        or recursive_verification["seed"] != seed
        or recursive_verification["run_identity_sha256"] != result["run_identity_sha256"]
        or recursive_verification["best_checkpoint_sha256"] != best_record["sha256"]
    ):
        raise ValueError("Task 1 recursive verification differs from normalized evidence")

    return {
        "run_id": result["run_id"],
        "seed": seed,
        "run_directory": str(_absolute(run_directory)),
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "weight_sha256": result["weight_sha256"],
        "source_fingerprint_sha256": result["source_fingerprint_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": result["scope"],
        "production_evidence": result["production_evidence"],
        "best_epoch": result["best_epoch"],
        "best_validation_macro_f1": result["best_validation_macro_f1"],
        "best_validation_loss": result["best_validation_loss"],
        "run_identity": identity_record,
        "best_checkpoint": best_record,
        "latest_recovery_checkpoint": latest_record,
        "result": result_record,
        "completion_lock": completion_record,
    }


def _validate_task2_production_identity(identity: Mapping[str, Any]) -> None:
    if identity["scope"] != TASK2_PRODUCTION_SCOPE or identity["production_evidence"] is not True:
        raise ValueError("Task 2 identity is not production evidence")
    runtime_fields = {
        "schema_version",
        "python_executable",
        "python_prefix",
        "python_implementation",
        "python_version",
        "platform_system",
        "platform_release",
        "platform_machine",
        "macos_version",
        "hardware_model",
        "processor_brand",
        "torch_version",
        "torch_build_config",
        "torch_num_threads",
        "torch_num_interop_threads",
        "torchvision_version",
        "numpy_version",
        "device",
        "mps_built",
        "mps_available",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "mps_fallback",
        "mps_fast_math",
        "mps_prefer_metal",
        "float32_matmul_precision",
        "default_dtype",
        "training_dtype",
        "numerical_environment",
    }
    runtime = _require_exact_mapping(
        identity["numerical_runtime"], runtime_fields, "Task 2 numerical runtime"
    )
    if sha256_json(runtime) != identity["numerical_runtime_sha256"]:
        raise ValueError("Task 2 numerical runtime identity binding is invalid")
    numerical_environment = _require_exact_mapping(
        runtime["numerical_environment"],
        {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "PYTORCH_MPS_FAST_MATH",
            "PYTORCH_MPS_PREFER_METAL",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        },
        "Task 2 numerical environment",
    )
    if (
        runtime["schema_version"] != "1.0"
        or runtime.get("device") != "mps"
        or runtime.get("mps_built") is not True
        or runtime.get("mps_available") is not True
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_warn_only") is not False
        or runtime.get("default_dtype") != "torch.float32"
        or runtime.get("training_dtype") != "torch.float32"
        or runtime.get("mps_fallback") not in {"disabled", "0", "false"}
        or numerical_environment["PYTORCH_ENABLE_MPS_FALLBACK"].strip().casefold()
        not in {"unset", "0", "false"}
        or any(
            type(runtime[name]) is not str or not runtime[name]
            for name in runtime_fields
            - {
                "mps_built",
                "mps_available",
                "deterministic_algorithms",
                "deterministic_warn_only",
                "torch_num_threads",
                "torch_num_interop_threads",
                "numerical_environment",
            }
        )
        or any(type(value) is not str or not value for value in numerical_environment.values())
        or type(runtime["torch_num_threads"]) is not int
        or runtime["torch_num_threads"] <= 0
        or type(runtime["torch_num_interop_threads"]) is not int
        or runtime["torch_num_interop_threads"] <= 0
        or type(runtime.get("python_executable")) is not str
        or Path(runtime["python_executable"]).resolve()
        != (PROJECT_ROOT / ".venv" / "bin" / "python").resolve()
        or type(runtime.get("python_prefix")) is not str
        or Path(runtime["python_prefix"]).resolve() != (PROJECT_ROOT / ".venv").resolve()
    ):
        raise ValueError("Task 2 numerical runtime is not production MPS evidence")
    if identity["limits"] != {
        "maximum_epochs": 100,
        "batch_size": 64,
        "patience": 10,
    }:
        raise ValueError("Task 2 run limits are not the locked production limits")
    if identity["data"] != {**PRODUCTION_DATA_COUNTS, "selection_strategy": "energy"}:
        raise ValueError("Task 2 data identity is not the production development scope")
    model = _require_exact_mapping(
        identity["model_contract"],
        {
            "architecture",
            "model_type",
            "input_shape",
            "latent_dimensions",
            "parameter_counts",
            "state",
        },
        "Task 2 model contract",
    )
    if sha256_json(model) != identity["model_contract_sha256"]:
        raise ValueError("Task 2 model contract binding is invalid")
    if (
        model.get("architecture") != "skip_free_undercomplete_convolutional_autoencoder"
        or model.get("model_type") != "bird_audio.models.ConvolutionalAutoencoder"
        or model.get("input_shape") != [1, 224, 224]
        or model.get("latent_dimensions") != 64
        or model.get("parameter_counts")
        != {"total": EXPECTED_PARAMETER_COUNT, "trainable": EXPECTED_PARAMETER_COUNT}
        or not isinstance(model.get("state"), list)
        or not model["state"]
    ):
        raise ValueError("Task 2 model contract is not the locked production model")
    if identity["optimizer_contract"] != {
        "type": "torch.optim.AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }:
        raise ValueError("Task 2 optimizer contract changed")
    if identity["final_evaluation_contract"] != {
        "primary_score": "median_clip_reconstruction_mse",
        "secondary_readout": "recording_mean_latent_knn_distance",
        "score_direction": "higher_is_more_novel",
        "nearest_neighbours": 10,
        "threshold_quantile": 0.95,
        "threshold_quantile_method": "higher",
        "threshold_scope": "per_seed_known_validation",
        "threshold_operator": ">",
        "bootstrap_seed": 20260713,
        "bootstrap_replicates": 2000,
        "bootstrap_interval_method": "percentile",
        "bootstrap_confidence_level": 0.95,
        "bootstrap_resampling_unit": "session_cluster",
        "detailed_figure_seed": 37,
    }:
        raise ValueError("Task 2 final evaluation contract changed")


def _validate_task2_provenance(
    value: object,
    *,
    result: Mapping[str, Any],
    identity: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    config_record: Mapping[str, Any],
) -> None:
    provenance = _require_exact_mapping(value, _TASK2_PROVENANCE_FIELDS, "Task 2 provenance")
    bindings = {
        "schema_version": TASK2_RUN_SCHEMA_VERSION,
        "started_at_utc": result["started_at_utc"],
        "run_identity_sha256": result["run_identity_sha256"],
        "config_path": "configs/task2/autoencoder.toml",
        "config_file_sha256": config_record["sha256"],
        "config_sha256": result["config_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "release_source_fingerprint_sha256": result["release_source_fingerprint_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime": identity["numerical_runtime"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "model_contract": identity["model_contract"],
        "model_contract_sha256": result["model_contract_sha256"],
        "optimizer_contract": identity["optimizer_contract"],
        "final_evaluation_contract": identity["final_evaluation_contract"],
        "scope": TASK2_PRODUCTION_SCOPE,
        "production_evidence": True,
        "initial_artifacts": {
            "resolved_config": artifacts["resolved_config"],
            "run_identity": artifacts["run_identity"],
        },
    }
    for key, expected in bindings.items():
        if provenance[key] != expected:
            raise ValueError(f"Task 2 provenance changed bound field: {key}")
    _require_utc_timestamp(provenance["started_at_utc"], "Task 2 provenance start time")
    if (
        not isinstance(provenance["command"], list)
        or any(type(part) is not str for part in provenance["command"])
        or type(provenance["cache_root"]) is not str
        or Path(provenance["cache_root"]).resolve() != _absolute(KNOWN_CACHE_LOCK_PATH).parent
    ):
        raise ValueError("Task 2 provenance command or cache path is invalid")


def _validate_scored_split(
    value: object,
    *,
    role: str,
    run_identity_sha256: str,
    best_checkpoint_sha256: str,
    expected_clip_count: int,
    expected_recording_count: int,
    include_latent_scores: bool,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    fields = {
        "schema_version",
        "run_identity_sha256",
        "best_checkpoint_sha256",
        "source_role",
        "clip_count",
        "recording_count",
        "recordings",
        "clips",
    }
    if include_latent_scores:
        fields.add("latent_novelty_scores")
    scored = _require_exact_mapping(value, fields, f"Task 2 {role} scores")
    if (
        scored["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or scored["run_identity_sha256"] != run_identity_sha256
        or scored["best_checkpoint_sha256"] != best_checkpoint_sha256
        or scored["source_role"] != role
        or scored["clip_count"] != expected_clip_count
        or scored["recording_count"] != expected_recording_count
        or not isinstance(scored["recordings"], list)
        or len(scored["recordings"]) != expected_recording_count
        or not isinstance(scored["clips"], list)
        or len(scored["clips"]) != expected_clip_count
    ):
        raise ValueError(f"Task 2 {role} score artifact identity is invalid")
    recording_ids: list[str] = []
    reconstruction_scores: list[float] = []
    for record in scored["recordings"]:
        row = _require_exact_mapping(
            record,
            {
                "recording_id",
                "clip_ids",
                "clip_count",
                "reconstruction_mse",
                "mean_latent_embedding",
                "session_group",
            },
            f"Task 2 {role} recording",
        )
        recording_id = row["recording_id"]
        if (
            type(recording_id) is not str
            or not recording_id
            or not isinstance(row["clip_ids"], list)
            or row["clip_count"] != len(row["clip_ids"])
            or not isinstance(row["mean_latent_embedding"], list)
            or len(row["mean_latent_embedding"]) != 64
            or type(row["session_group"]) is not str
            or not row["session_group"]
        ):
            raise ValueError(f"Task 2 {role} recording value is invalid")
        recording_ids.append(recording_id)
        reconstruction_scores.append(
            _require_finite_float(
                row["reconstruction_mse"],
                f"Task 2 {role} reconstruction score",
                nonnegative=True,
            )
        )
        for coordinate in row["mean_latent_embedding"]:
            _require_finite_float(coordinate, f"Task 2 {role} latent coordinate")
    if recording_ids != sorted(recording_ids) or len(set(recording_ids)) != len(recording_ids):
        raise ValueError(f"Task 2 {role} recording order is invalid")

    recording_set = set(recording_ids)
    clip_pairs: set[tuple[str, str]] = set()
    for record in scored["clips"]:
        row = _require_exact_mapping(
            record,
            {
                "recording_id",
                "clip_id",
                "session_group",
                "reconstruction_mse",
                "latent_embedding",
            },
            f"Task 2 {role} clip",
        )
        pair = (row["recording_id"], row["clip_id"])
        if (
            type(pair[0]) is not str
            or type(pair[1]) is not str
            or not pair[0]
            or not pair[1]
            or pair[0] not in recording_set
            or pair in clip_pairs
            or type(row["session_group"]) is not str
            or not row["session_group"]
            or not isinstance(row["latent_embedding"], list)
            or len(row["latent_embedding"]) != 64
        ):
            raise ValueError(f"Task 2 {role} clip value is invalid")
        clip_pairs.add(pair)
        _require_finite_float(
            row["reconstruction_mse"], f"Task 2 {role} clip score", nonnegative=True
        )
        for coordinate in row["latent_embedding"]:
            _require_finite_float(coordinate, f"Task 2 {role} clip latent coordinate")

    latent_values: list[float] = []
    if include_latent_scores:
        latent_scores = scored["latent_novelty_scores"]
        if not isinstance(latent_scores, list) or len(latent_scores) != expected_recording_count:
            raise ValueError("Task 2 validation latent score count is invalid")
        latent_ids: list[str] = []
        for record in latent_scores:
            row = _require_exact_mapping(
                record,
                {
                    "recording_id",
                    "score",
                    "direction",
                    "neighbour_recording_ids",
                    "neighbour_distances",
                },
                "Task 2 validation latent score",
            )
            if (
                type(row["recording_id"]) is not str
                or row["direction"] != NOVELTY_DIRECTION
                or not isinstance(row["neighbour_recording_ids"], list)
                or len(row["neighbour_recording_ids"]) != 10
                or len(set(row["neighbour_recording_ids"])) != 10
                or not isinstance(row["neighbour_distances"], list)
                or len(row["neighbour_distances"]) != 10
            ):
                raise ValueError("Task 2 validation latent score is invalid")
            distances = tuple(
                _require_finite_float(
                    distance, "Task 2 latent neighbour distance", nonnegative=True
                )
                for distance in row["neighbour_distances"]
            )
            score = _require_finite_float(
                row["score"], "Task 2 validation latent score", nonnegative=True
            )
            if distances != tuple(sorted(distances)) or not math.isclose(
                score, math.fsum(distances) / 10
            ):
                raise ValueError("Task 2 validation latent score arithmetic is invalid")
            latent_ids.append(row["recording_id"])
            latent_values.append(score)
        if latent_ids != recording_ids:
            raise ValueError("Task 2 validation latent score identities differ")
    return tuple(recording_ids), tuple(reconstruction_scores), tuple(latent_values)


def _validate_latent_reference(
    value: object,
    *,
    run_identity_sha256: str,
    best_checkpoint_sha256: str,
    training_ids: tuple[str, ...],
) -> None:
    artifact = _require_exact_mapping(
        value,
        {"schema_version", "run_identity_sha256", "best_checkpoint_sha256", "reference"},
        "Task 2 latent reference artifact",
    )
    if (
        artifact["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or artifact["run_identity_sha256"] != run_identity_sha256
        or artifact["best_checkpoint_sha256"] != best_checkpoint_sha256
    ):
        raise ValueError("Task 2 latent reference binding is invalid")
    reference = _require_exact_mapping(
        artifact["reference"],
        {
            "fit_role",
            "recording_ids",
            "recording_count",
            "coordinate_mean",
            "population_variance",
            "coordinate_scale",
            "standardized_embeddings",
            "nearest_neighbours",
        },
        "Task 2 latent reference",
    )
    if (
        reference["fit_role"] != KNOWN_TRAINING_ROLE
        or reference["recording_ids"] != list(training_ids)
        or reference["recording_count"] != len(training_ids)
        or reference["nearest_neighbours"] != 10
        or not isinstance(reference["coordinate_mean"], list)
        or len(reference["coordinate_mean"]) != 64
        or not isinstance(reference["population_variance"], list)
        or len(reference["population_variance"]) != 64
        or not isinstance(reference["coordinate_scale"], list)
        or len(reference["coordinate_scale"]) != 64
        or not isinstance(reference["standardized_embeddings"], list)
        or len(reference["standardized_embeddings"]) != len(training_ids)
    ):
        raise ValueError("Task 2 latent reference shape is invalid")
    for mean, variance, scale in zip(
        reference["coordinate_mean"],
        reference["population_variance"],
        reference["coordinate_scale"],
        strict=True,
    ):
        _require_finite_float(mean, "Task 2 latent mean")
        variance_value = _require_finite_float(
            variance, "Task 2 latent population variance", nonnegative=True
        )
        scale_value = _require_finite_float(scale, "Task 2 latent scale", nonnegative=True)
        expected_scale = 1.0 if variance_value == 0.0 else math.sqrt(variance_value)
        if scale_value != expected_scale:
            raise ValueError("Task 2 latent scale differs from its variance")
    for row in reference["standardized_embeddings"]:
        if not isinstance(row, list) or len(row) != 64:
            raise ValueError("Task 2 standardized latent embedding shape is invalid")
        for coordinate in row:
            _require_finite_float(coordinate, "Task 2 standardized latent coordinate")


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Task 2 threshold source is empty")
    return ordered[math.ceil((len(ordered) - 1) * quantile)]


def _validate_thresholds(
    value: object,
    *,
    run_identity_sha256: str,
    best_checkpoint_sha256: str,
    validation_ids: tuple[str, ...],
    reconstruction_scores: tuple[float, ...],
    latent_scores: tuple[float, ...],
) -> None:
    artifact = _require_exact_mapping(
        value,
        {
            "schema_version",
            "run_identity_sha256",
            "best_checkpoint_sha256",
            "reconstruction",
            "latent",
        },
        "Task 2 thresholds artifact",
    )
    if (
        artifact["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or artifact["run_identity_sha256"] != run_identity_sha256
        or artifact["best_checkpoint_sha256"] != best_checkpoint_sha256
    ):
        raise ValueError("Task 2 thresholds binding is invalid")
    for key, score_name, scores in (
        ("reconstruction", RECONSTRUCTION_SCORE_NAME, reconstruction_scores),
        ("latent", LATENT_SCORE_NAME, latent_scores),
    ):
        threshold = _require_exact_mapping(
            artifact[key],
            {
                "score_name",
                "value",
                "calibration_role",
                "calibration_recording_ids",
                "quantile",
                "method",
                "direction",
                "classification_operator",
            },
            f"Task 2 {key} threshold",
        )
        if (
            threshold["score_name"] != score_name
            or threshold["calibration_role"] != KNOWN_VALIDATION_ROLE
            or threshold["calibration_recording_ids"] != list(validation_ids)
            or threshold["quantile"] != 0.95
            or threshold["method"] != "higher"
            or threshold["direction"] != NOVELTY_DIRECTION
            or threshold["classification_operator"] != ">"
            or threshold["value"] != _higher_quantile(scores, 0.95)
        ):
            raise ValueError(f"Task 2 {key} threshold is invalid")


def _load_task2_checkpoint_record(
    record_value: object,
    expected_path: Path,
    *,
    run_directory: Path,
    run_identity_sha256: str,
    expected_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _verify_binary_record(
        record_value,
        expected_path,
        f"Task 2 {expected_type} checkpoint",
        boundary=run_directory,
    )
    checkpoint = load_task2_checkpoint(
        expected_path,
        expected_sha256=record["sha256"],
        expected_run_identity_sha256=run_identity_sha256,
        expected_type=expected_type,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Task 2 checkpoint loader returned an invalid value")
    return checkpoint, record


def _inspect_task2_development_bundle(
    *,
    run_directory: Path,
    expected_binding: Mapping[str, Any],
    final_evaluation_contract: Mapping[str, Any],
    best_checkpoint_record: Mapping[str, Any],
    development_records: object,
    bundle_record_value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    development = _require_exact_mapping(
        development_records,
        _TASK2_DEVELOPMENT_ARTIFACT_FIELDS,
        "Task 2 development artifacts",
    )
    development_directory = run_directory / "development"
    bundle, bundle_record = _read_canonical_json(
        development_directory / "development_bundle.lock.json",
        expected_record=bundle_record_value,
        name="Task 2 development bundle",
        boundary=run_directory,
    )
    bundle = _require_exact_mapping(
        bundle,
        {
            "schema_version",
            "complete",
            *_TASK2_DEVELOPMENT_BINDING_FIELDS,
            "best_checkpoint",
            "artifacts",
            "fit_roles",
            "threshold_operator",
            "final_evaluation_contract",
        },
        "Task 2 development bundle",
    )
    if (
        bundle["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or bundle["complete"] is not True
        or bundle["best_checkpoint"] != best_checkpoint_record
        or bundle["artifacts"] != development
        or bundle["fit_roles"]
        != {
            "latent_reference": KNOWN_TRAINING_ROLE,
            "reconstruction_threshold": KNOWN_VALIDATION_ROLE,
            "latent_threshold": KNOWN_VALIDATION_ROLE,
        }
        or bundle["threshold_operator"] != ">"
        or bundle["final_evaluation_contract"] != final_evaluation_contract
        or any(bundle[key] != value for key, value in expected_binding.items())
    ):
        raise ValueError("Task 2 development bundle identity is invalid")

    children = (
        (
            "known_training_scores",
            "known_training_recording_scores.json",
            "Task 2 known-training scores",
        ),
        (
            "known_validation_scores",
            "known_validation_recording_scores.json",
            "Task 2 known-validation scores",
        ),
        (
            "training_latent_reference",
            "known_training_latent_reference.json",
            "Task 2 training latent reference",
        ),
        ("thresholds", "known_validation_thresholds.json", "Task 2 validation thresholds"),
    )
    for key, filename, name in children:
        child, _ = _read_canonical_json(
            development_directory / filename,
            expected_record=development[key],
            name=name,
            boundary=run_directory,
        )
        if (
            not isinstance(child, Mapping)
            or child.get("schema_version") != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
            or any(child.get(field) != value for field, value in expected_binding.items())
        ):
            raise ValueError(f"{name} binding is invalid")
    return dict(development), bundle_record


def _inspect_task2_run(
    run_directory: Path,
    *,
    config_record: Mapping[str, Any],
) -> dict[str, Any]:
    task = "Task 2"
    completion, completion_record = _read_canonical_json(
        run_directory / "result.lock.json",
        name="Task 2 completion lock",
        boundary=run_directory,
    )
    completion = _require_exact_mapping(completion, _TASK2_COMPLETION_FIELDS, task)
    if completion["schema_version"] != TASK2_RUN_SCHEMA_VERSION:
        raise ValueError("Task 2 completion lock version is unsupported")
    recursive_verification = verify_task2_development_run(
        run_directory / "result.lock.json",
        expected_sha256=completion_record["sha256"],
        require_production=True,
    )
    recursive_verification = _require_exact_mapping(
        recursive_verification,
        {
            "valid",
            "complete",
            "run_id",
            "seed",
            "scope",
            "production_evidence",
            "completion_lock_sha256",
            "run_identity_sha256",
            "best_checkpoint_sha256",
            "development_bundle_sha256",
            "training_recordings",
            "training_clips",
            "validation_recordings",
            "validation_clips",
            "thresholds_rederived",
        },
        "Task 2 recursive verification",
    )
    if (
        recursive_verification["valid"] is not True
        or recursive_verification["complete"] is not True
        or recursive_verification["scope"] != TASK2_PRODUCTION_SCOPE
        or recursive_verification["production_evidence"] is not True
        or recursive_verification["thresholds_rederived"] is not True
        or recursive_verification["completion_lock_sha256"] != completion_record["sha256"]
        or recursive_verification["training_recordings"]
        != PRODUCTION_DATA_COUNTS["train_recordings"]
        or recursive_verification["training_clips"] != PRODUCTION_DATA_COUNTS["train_clips"]
        or recursive_verification["validation_recordings"]
        != PRODUCTION_DATA_COUNTS["validation_recordings"]
        or recursive_verification["validation_clips"] != PRODUCTION_DATA_COUNTS["validation_clips"]
    ):
        raise ValueError("Task 2 recursive verification is not production evidence")
    result, result_record = _read_canonical_json(
        run_directory / "result.json",
        expected_record=completion["result"],
        name="Task 2 result",
        boundary=run_directory,
    )
    result = _require_exact_mapping(result, _TASK2_RESULT_FIELDS, "Task 2 result")
    _validate_result_scalars(result, task)
    _validate_resume_state(result, run_directory, task)
    if (
        result["schema_version"] != TASK2_RUN_SCHEMA_VERSION
        or result["run_id"] != run_directory.name
        or result["run_directory"] != str(_absolute(run_directory))
        or result["scope"] != TASK2_PRODUCTION_SCOPE
        or result["production_evidence"] is not True
    ):
        raise ValueError("Task 2 result is not canonical production evidence")
    started_at = _require_utc_timestamp(result["started_at_utc"], "Task 2 start time")
    completed_at = _require_utc_timestamp(result["completed_at_utc"], "Task 2 completion time")
    if datetime.fromisoformat(completed_at) < datetime.fromisoformat(started_at):
        raise ValueError("Task 2 completion time precedes its start")
    artifacts = _require_exact_mapping(
        result["artifacts"], _TASK2_ARTIFACT_FIELDS, "Task 2 artifacts"
    )
    identity, identity_record = _read_canonical_json(
        run_directory / "run_identity.json",
        expected_record=artifacts["run_identity"],
        name="Task 2 run identity",
        boundary=run_directory,
    )
    identity = _require_exact_mapping(identity, _TASK2_RUN_IDENTITY_FIELDS, "Task 2 identity")
    seed = identity["seed"]
    if type(seed) is not int or seed not in SEED_ORDER:
        raise ValueError("Task 2 seed is invalid")
    if sha256_json(identity) != result["run_identity_sha256"]:
        raise ValueError("Task 2 run identity SHA-256 is invalid")
    identity_bindings = {
        "schema_version": TASK2_RUN_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "task": "task2_novelty_detection_development",
        "config_sha256": result["config_sha256"],
        "config_file_sha256": result["config_file_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "model_contract_sha256": result["model_contract_sha256"],
        "scope": TASK2_PRODUCTION_SCOPE,
        "production_evidence": True,
    }
    for key, expected in identity_bindings.items():
        if identity[key] != expected:
            raise ValueError(f"Task 2 identity changed bound field: {key}")
    _validate_task2_production_identity(identity)
    completion_bindings = {
        "run_identity_sha256": result["run_identity_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": TASK2_PRODUCTION_SCOPE,
        "production_evidence": True,
    }
    for key, expected in completion_bindings.items():
        if completion[key] != expected:
            raise ValueError(f"Task 2 completion lock changed bound field: {key}")
    for key in (
        "run_identity_sha256",
        "config_sha256",
        "config_file_sha256",
        "cache_lock_sha256",
        "release_source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "model_contract_sha256",
    ):
        _require_sha256(result[key], f"Task 2 {key}")
    if (
        result["config_sha256"] != config_record["config_sha256"]
        or result["config_file_sha256"] != config_record["sha256"]
    ):
        raise ValueError("Task 2 result uses another configuration")

    resolved_config, _ = _read_canonical_json(
        run_directory / "resolved_config.json",
        expected_record=artifacts["resolved_config"],
        name="Task 2 resolved configuration",
        boundary=run_directory,
    )
    _validate_resolved_config(
        resolved_config,
        task=task,
        config_record=config_record,
        relative_path="configs/task2/autoencoder.toml",
    )
    provenance, _ = _read_canonical_json(
        run_directory / "provenance.json",
        expected_record=artifacts["provenance"],
        name="Task 2 provenance",
        boundary=run_directory,
    )
    _validate_task2_provenance(
        provenance,
        result=result,
        identity=identity,
        artifacts=artifacts,
        config_record=config_record,
    )
    history, _ = _read_canonical_json(
        run_directory / "epoch_history.json",
        expected_record=artifacts["epoch_history"],
        name="Task 2 epoch history",
        boundary=run_directory,
    )
    if not isinstance(history, list) or len(history) != result["epochs_completed"]:
        raise ValueError("Task 2 epoch history length is invalid")

    best_path = run_directory / "best_candidates" / f"best_epoch_{result['best_epoch']:04d}.pt"
    best_record = _verify_binary_record(
        result["best_checkpoint"],
        best_path,
        "Task 2 best checkpoint",
        boundary=run_directory,
    )
    if best_record != artifacts["best_checkpoint"]:
        raise ValueError("Task 2 best checkpoint records differ")

    latest_path = run_directory / "recovery" / f"recovery_epoch_{result['epochs_completed']:04d}.pt"
    latest_record = _verify_binary_record(
        result["latest_recovery_checkpoint"],
        latest_path,
        "Task 2 latest recovery checkpoint",
        boundary=run_directory,
    )
    if latest_record != artifacts["latest_recovery"]:
        raise ValueError("Task 2 latest recovery records differ")

    development_binding = {
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "config_file_sha256": result["config_file_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "model_contract_sha256": result["model_contract_sha256"],
        "scope": TASK2_PRODUCTION_SCOPE,
        "production_evidence": True,
        "seed": seed,
        "best_checkpoint_sha256": best_record["sha256"],
    }

    development, development_bundle_record = _inspect_task2_development_bundle(
        run_directory=run_directory,
        expected_binding=development_binding,
        final_evaluation_contract=identity["final_evaluation_contract"],
        best_checkpoint_record=best_record,
        development_records=artifacts["development"],
        bundle_record_value=result["development_bundle"],
    )
    if (
        development != artifacts["development"]
        or development_bundle_record != artifacts["development_bundle"]
        or development_bundle_record != completion["development_bundle"]
    ):
        raise ValueError("Task 2 development bundle records differ")
    if (
        recursive_verification["run_id"] != result["run_id"]
        or recursive_verification["seed"] != seed
        or recursive_verification["run_identity_sha256"] != result["run_identity_sha256"]
        or recursive_verification["best_checkpoint_sha256"] != best_record["sha256"]
        or recursive_verification["development_bundle_sha256"]
        != development_bundle_record["sha256"]
    ):
        raise ValueError("Task 2 recursive verification differs from normalized evidence")

    return {
        "run_id": result["run_id"],
        "seed": seed,
        "run_directory": str(_absolute(run_directory)),
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "config_file_sha256": result["config_file_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "release_source_fingerprint_sha256": result["release_source_fingerprint_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "model_contract_sha256": result["model_contract_sha256"],
        "numerical_runtime": identity["numerical_runtime"],
        "model_contract": identity["model_contract"],
        "optimizer_contract": identity["optimizer_contract"],
        "final_evaluation_contract": identity["final_evaluation_contract"],
        "scope": TASK2_PRODUCTION_SCOPE,
        "production_evidence": True,
        "started_at_utc": result["started_at_utc"],
        "completed_at_utc": result["completed_at_utc"],
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result["best_validation_loss"],
        "run_identity": identity_record,
        "best_checkpoint": best_record,
        "latest_recovery_checkpoint": latest_record,
        "development_artifacts": development,
        "development_bundle": development_bundle_record,
        "result": result_record,
        "completion_lock": completion_record,
    }


def _uniform_value(runs: Sequence[Mapping[str, Any]], key: str, task: str) -> Any:
    values = [run[key] for run in runs]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{task} runs have mixed {key}")
    return values[0]


def _validate_seed_inventory(runs: Sequence[Mapping[str, Any]], task: str) -> None:
    seeds = tuple(run["seed"] for run in runs)
    run_ids = tuple(run["run_id"] for run in runs)
    if seeds != SEED_ORDER or len(set(run_ids)) != len(SEED_ORDER):
        raise ValueError(f"{task} run inventory must contain exactly seeds 13, 37, and 71")


def _task1_section(
    directories: Sequence[Path],
    config_record: Mapping[str, Any],
    known_cache: Mapping[str, Any],
) -> dict[str, Any]:
    runs = sorted(
        (_inspect_task1_run(path, config_record=config_record) for path in directories),
        key=lambda run: run["seed"],
    )
    _validate_seed_inventory(runs, "Task 1")
    identity = {
        key: _uniform_value(runs, key, "Task 1")
        for key in (
            "config_sha256",
            "cache_lock_sha256",
            "weight_sha256",
            "source_fingerprint_sha256",
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
            "scope",
            "production_evidence",
        )
    }
    if (
        identity["config_sha256"] != config_record["config_sha256"]
        or identity["cache_lock_sha256"] != known_cache["sha256"]
        or identity["scope"] != TASK1_PRODUCTION_SCOPE
        or identity["production_evidence"] is not True
    ):
        raise ValueError("Task 1 shared identity is not production evidence")
    return {
        "task": "task1_classification",
        "run_schema_version": TASK1_RUN_SCHEMA_VERSION,
        "run_count": len(runs),
        "seeds": list(SEED_ORDER),
        "identity": identity,
        "runs": runs,
    }


def _task2_section(
    directories: Sequence[Path],
    config_record: Mapping[str, Any],
    known_cache: Mapping[str, Any],
) -> dict[str, Any]:
    runs = sorted(
        (_inspect_task2_run(path, config_record=config_record) for path in directories),
        key=lambda run: run["seed"],
    )
    _validate_seed_inventory(runs, "Task 2")
    identity = {
        key: _uniform_value(runs, key, "Task 2")
        for key in (
            "config_sha256",
            "config_file_sha256",
            "cache_lock_sha256",
            "release_source_fingerprint_sha256",
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
            "model_contract_sha256",
            "numerical_runtime",
            "model_contract",
            "optimizer_contract",
            "final_evaluation_contract",
            "scope",
            "production_evidence",
        )
    }
    if (
        identity["config_sha256"] != config_record["config_sha256"]
        or identity["config_file_sha256"] != config_record["sha256"]
        or identity["cache_lock_sha256"] != known_cache["sha256"]
        or identity["scope"] != TASK2_PRODUCTION_SCOPE
        or identity["production_evidence"] is not True
    ):
        raise ValueError("Task 2 shared identity is not production evidence")
    return {
        "task": "task2_novelty_detection_development",
        "run_schema_version": TASK2_RUN_SCHEMA_VERSION,
        "run_count": len(runs),
        "seeds": list(SEED_ORDER),
        "identity": identity,
        "runs": runs,
    }


def _validated_sealed_at(value: object) -> str:
    if type(value) is not str:
        raise ValueError("Final gate seal time is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Final gate seal time is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("Final gate seal time must use UTC")
    return value


def _build_gate_value(*, sealed_at_utc: str) -> dict[str, Any]:
    sealed_at = _validated_sealed_at(sealed_at_utc)
    current_source_fingerprint_sha256 = source_fingerprint()
    _require_sha256(current_source_fingerprint_sha256, "current source fingerprint")
    configs = _canonical_configs()
    cache_locks = _cache_locks()
    recovery_evidence = _recovery_evidence(
        current_source_fingerprint_sha256,
        cache_locks["unknown"]["sha256"],
    )
    task1_directories = _completed_run_directories(TASK1_RUN_ROOT, "Task 1")
    task2_directories = _completed_run_directories(TASK2_RUN_ROOT, "Task 2")
    task1 = _task1_section(task1_directories, configs["task1"], cache_locks["known"])
    task2 = _task2_section(task2_directories, configs["task2"], cache_locks["known"])
    if (
        task1["identity"]["source_fingerprint_sha256"]
        != task2["identity"]["release_source_fingerprint_sha256"]
        or task1["identity"]["source_fingerprint_sha256"] != current_source_fingerprint_sha256
        or task1["identity"]["requirements_lock_sha256"]
        != task2["identity"]["requirements_lock_sha256"]
    ):
        raise ValueError("Task, current source, or dependency identities do not agree")
    shared_identity = {
        "known_cache_lock_sha256": cache_locks["known"]["sha256"],
        "known_cache_content_sha256": cache_locks["known"]["cache_content_sha256"],
        "unknown_cache_lock_sha256": cache_locks["unknown"]["sha256"],
        "unknown_cache_content_sha256": cache_locks["unknown"]["cache_content_sha256"],
        "requirements_lock_sha256": task1["identity"]["requirements_lock_sha256"],
        "source_fingerprint_sha256": task1["identity"]["source_fingerprint_sha256"],
    }
    if (
        cache_locks["known"]["requirements_lock_sha256"]
        != shared_identity["requirements_lock_sha256"]
        or cache_locks["unknown"]["requirements_lock_sha256"]
        != shared_identity["requirements_lock_sha256"]
    ):
        raise ValueError("Cache and task dependency identities do not agree")
    if source_fingerprint() != current_source_fingerprint_sha256:
        raise RuntimeError("Current source changed while the final gate was built")
    inventory = {"task1": task1["runs"], "task2": task2["runs"]}
    return {
        "schema_version": FINAL_EVALUATION_GATE_SCHEMA_VERSION,
        "gate_id": FINAL_EVALUATION_GATE_ID,
        "sealed_at_utc": sealed_at,
        "seed_order": list(SEED_ORDER),
        "canonical_configs": configs,
        "cache_locks": cache_locks,
        "recovery_evidence": recovery_evidence,
        "shared_identity": shared_identity,
        "shared_identity_sha256": sha256_json(shared_identity),
        "task1": task1,
        "task2": task2,
        "inventory_sha256": sha256_json(inventory),
        "ready": True,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory_no_follow(_absolute(path))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_only_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if not payload:
        raise ValueError("Final gate artifact payload cannot be empty")
    destination = _absolute(path)
    parent_descriptor, destination_name = _open_relative_parent(destination, PROJECT_ROOT)
    temporary_name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        os.close(parent_descriptor)
        raise RuntimeError("Final gate publication requires O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | no_follow
    temporary_descriptor: int | None = None
    try:
        temporary_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(temporary_descriptor, view[written:])
            if count <= 0:
                raise RuntimeError("Final gate publication write made no progress")
            written += count
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        os.link(
            temporary_name,
            destination_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
        read_descriptor = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow,
            dir_fd=parent_descriptor,
        )
        try:
            observed, sha256, size_bytes = _snapshot_file_descriptor(
                read_descriptor,
                destination,
            )
        finally:
            os.close(read_descriptor)
        if observed != payload:
            raise RuntimeError("Final gate artifact changed during publication")
        return {"path": str(destination), "sha256": sha256, "size_bytes": size_bytes}
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _publish_directory_no_replace(
    source_name: str,
    destination_name: str,
    parent_descriptor: int,
) -> None:
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        renameatx_np = libc.renameatx_np
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        status = renameatx_np(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            4,
        )
        if status == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination_name)
        raise OSError(error, os.strerror(error), destination_name)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        status = renameat2(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
        if status == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination_name)
        raise OSError(error, os.strerror(error), destination_name)
    raise RuntimeError("Final gate publication requires atomic no-replace rename support")


def _remove_staging_directory(parent_descriptor: int, staging_name: str) -> None:
    try:
        staging_descriptor = os.open(
            staging_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except FileNotFoundError:
        return
    try:
        with os.scandir(staging_descriptor) as entries:
            names = [entry.name for entry in entries]
        for name in names:
            try:
                os.unlink(name, dir_fd=staging_descriptor)
            except IsADirectoryError as exc:
                raise RuntimeError(
                    "Final gate staging directory contains an unexpected child"
                ) from exc
        os.fsync(staging_descriptor)
    finally:
        os.close(staging_descriptor)
    os.rmdir(staging_name, dir_fd=parent_descriptor)


def _publish_gate_directory(gate_value: Mapping[str, Any]) -> None:
    destination = _absolute(FINAL_EVALUATION_GATE_DIRECTORY)
    parent = destination.parent
    _secure_ensure_directory(parent, PROJECT_ROOT)
    parent_descriptor = _open_absolute_directory_no_follow(parent)
    staging_name = f".gate_v2.staging.{secrets.token_hex(16)}"
    os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)
    staging = parent / staging_name
    published = False
    try:
        gate_payload = _json_bytes(gate_value)
        gate_sha256 = hashlib.sha256(gate_payload).hexdigest()
        _atomic_create_only_bytes(staging / "gate.json", gate_payload)
        lock_value = {
            "schema_version": FINAL_EVALUATION_GATE_SCHEMA_VERSION,
            "gate_id": FINAL_EVALUATION_GATE_ID,
            "gate": {
                "path": "gate.json",
                "sha256": gate_sha256,
                "size_bytes": len(gate_payload),
            },
        }
        _atomic_create_only_bytes(staging / "lock.json", _json_bytes(lock_value))
        _fsync_directory(staging)
        _publish_directory_no_replace(staging_name, destination.name, parent_descriptor)
        published = True
        os.fsync(parent_descriptor)
    finally:
        if not published:
            _remove_staging_directory(parent_descriptor, staging_name)
        os.close(parent_descriptor)


def _validate_gate_shape(gate: object) -> Mapping[str, Any]:
    value = _require_exact_mapping(
        gate,
        {
            "schema_version",
            "gate_id",
            "sealed_at_utc",
            "seed_order",
            "canonical_configs",
            "cache_locks",
            "recovery_evidence",
            "shared_identity",
            "shared_identity_sha256",
            "task1",
            "task2",
            "inventory_sha256",
            "ready",
        },
        "Final evaluation gate",
    )
    if (
        value["schema_version"] != FINAL_EVALUATION_GATE_SCHEMA_VERSION
        or value["gate_id"] != FINAL_EVALUATION_GATE_ID
        or value["seed_order"] != list(SEED_ORDER)
        or value["ready"] is not True
    ):
        raise ValueError("Final evaluation gate identity is invalid")
    _validated_sealed_at(value["sealed_at_utc"])
    _require_sha256(value["shared_identity_sha256"], "Final gate shared identity SHA-256")
    _require_sha256(value["inventory_sha256"], "Final gate inventory SHA-256")
    return value


def _verify_existing_gate() -> dict[str, Any]:
    directory = _absolute(FINAL_EVALUATION_GATE_DIRECTORY)
    try:
        directory_descriptor = _open_absolute_directory_no_follow(directory)
    except OSError as exc:
        raise ValueError("Final evaluation gate cannot be opened safely") from exc
    try:
        with os.scandir(directory_descriptor) as entries:
            observed_entries = {
                entry.name: (
                    "file"
                    if entry.is_file(follow_symlinks=False) and not entry.is_symlink()
                    else "unsafe"
                )
                for entry in entries
            }
        if observed_entries != {"gate.json": "file", "lock.json": "file"}:
            raise ValueError("Final evaluation gate directory contains unexpected entries")
    finally:
        os.close(directory_descriptor)
    gate, gate_record = _read_canonical_json(
        directory / "gate.json",
        name="Final evaluation gate",
        boundary=directory,
    )
    lock, lock_record = _read_canonical_json(
        directory / "lock.json",
        name="Final evaluation gate lock",
        boundary=directory,
    )
    lock = _require_exact_mapping(
        lock,
        {"schema_version", "gate_id", "gate"},
        "Final evaluation gate lock",
    )
    gate_reference = _require_exact_mapping(
        lock["gate"], {"path", "sha256", "size_bytes"}, "Final gate reference"
    )
    if (
        lock["schema_version"] != FINAL_EVALUATION_GATE_SCHEMA_VERSION
        or lock["gate_id"] != FINAL_EVALUATION_GATE_ID
        or gate_reference
        != {
            "path": "gate.json",
            "sha256": gate_record["sha256"],
            "size_bytes": gate_record["size_bytes"],
        }
    ):
        raise ValueError("Final evaluation gate lock is invalid")
    gate = _validate_gate_shape(gate)
    expected = _build_gate_value(sealed_at_utc=gate["sealed_at_utc"])
    if dict(gate) != expected:
        raise ValueError("Sealed final evaluation gate differs from current development evidence")
    return {
        "gate": dict(gate),
        "gate_artifact": gate_record,
        "lock_artifact": lock_record,
        "created": False,
    }


def verify_final_evaluation_gate() -> dict[str, Any]:
    """Verify the immutable final evaluation gate and all bound development evidence."""

    return _verify_existing_gate()


def seal_final_evaluation_gate() -> dict[str, Any]:
    """Create the gate once, or verify the exact existing seal without rewriting it."""

    directory = _absolute(FINAL_EVALUATION_GATE_DIRECTORY)
    parent = directory.parent
    _secure_ensure_directory(parent, PROJECT_ROOT)
    descriptor = _open_absolute_directory_no_follow(parent)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            existing = os.stat(directory.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISDIR(existing.st_mode):
                raise ValueError("Final evaluation gate path is not a direct directory")
            return _verify_existing_gate()
        gate = _build_gate_value(sealed_at_utc=datetime.now(UTC).isoformat())
        _publish_gate_directory(gate)
        verified = _verify_existing_gate()
        return {**verified, "created": True}
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
