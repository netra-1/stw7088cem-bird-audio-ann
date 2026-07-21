from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bird_audio import final_evaluation_data as final_data
from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.final_evaluation_data import (
    FINAL_EVALUATION_ATTEMPT_ID,
    STAGE_ORDER,
    FinalEvaluationAuthorization,
    claim_final_evaluation_attempt,
    open_final_known_test_data,
    open_final_unknown_data,
)
from bird_audio.final_evaluation_gate import verify_final_evaluation_gate
from bird_audio.final_evaluation_inference import (
    FINAL_KNOWN_TEST_ROLE,
    FINAL_UNKNOWN_ROLE,
    KNOWN_COMMON_TO_SCIENTIFIC,
    FinalRecordingMetadata,
    iter_task1_recording_batches,
    iter_task2_recording_batches,
)
from bird_audio.paths import PROJECT_ROOT
from bird_audio.provenance import source_fingerprint
from bird_audio.task1_final_metrics import (
    BOOTSTRAP_METRIC_NAMES as TASK1_BOOTSTRAP_METRIC_NAMES,
)
from bird_audio.task1_final_metrics import (
    BootstrapReplicates as Task1BootstrapReplicates,
)
from bird_audio.task1_final_metrics import BootstrapResult as Task1BootstrapResult
from bird_audio.task1_final_metrics import PercentileInterval as Task1PercentileInterval
from bird_audio.task1_final_metrics import (
    RecordingPrediction,
    evaluate_recording_predictions,
    session_cluster_bootstrap_seed37,
    summarize_stability,
)
from bird_audio.task1_training import load_locked_task1_best_model
from bird_audio.task2_metrics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    KNOWN_SOURCE,
    UNKNOWN_SOURCE,
    MetricIntervals,
    ScoredRecording,
    SessionBootstrapResult,
    SpeciesMetricIntervals,
    evaluate_novelty_scores,
    session_cluster_bootstrap,
    summarize_across_seeds,
)
from bird_audio.task2_metrics import (
    METRIC_NAMES as TASK2_METRIC_NAMES,
)
from bird_audio.task2_metrics import (
    BootstrapReplicates as Task2BootstrapReplicates,
)
from bird_audio.task2_metrics import (
    PercentileInterval as Task2PercentileInterval,
)
from bird_audio.task2_scoring import (
    LATENT_SCORE_NAME,
    RECONSTRUCTION_SCORE_NAME,
    LatentReference,
    NoveltyThreshold,
    RecordingBatch,
    RecordingScore,
    latent_knn_novelty_scores,
)
from bird_audio.task2_training import load_locked_task2_best_model_for_evaluation

FINAL_EVALUATION_SCHEMA_VERSION = "1.0"
FINAL_EVALUATION_ATTEMPT_DIRECTORY = final_data.FINAL_EVALUATION_ATTEMPT_DIRECTORY
FINAL_EVALUATION_CLAIM_PATH = final_data.FINAL_EVALUATION_CLAIM_PATH
FINAL_EVALUATION_GATE_PATH = final_data.FINAL_EVALUATION_GATE_PATH
FINAL_EVALUATION_GATE_LOCK_PATH = final_data.FINAL_EVALUATION_GATE_LOCK_PATH
SEED_ORDER = (13, 37, 71)
DETAIL_SEED = 37
BOOTSTRAP_REPLICATES = DEFAULT_BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED
TASK1_BOOTSTRAP_FILENAME = "task1_seed_37_bootstrap.npz"
TASK2_RECONSTRUCTION_BOOTSTRAP_FILENAME = "task2_seed_37_reconstruction_bootstrap.npz"
TASK2_LATENT_BOOTSTRAP_FILENAME = "task2_seed_37_latent_bootstrap.npz"
FAILURE_DIRECTORY_NAME = "failures"
FINAL_EVALUATION_ROOT = final_data.FINAL_EVALUATION_ROOT
TASK1_FINAL_RECORDINGS = final_data.KNOWN_TEST_RECORDINGS
TASK2_KNOWN_FINAL_RECORDINGS = final_data.KNOWN_TEST_RECORDINGS
TASK2_UNKNOWN_FINAL_RECORDINGS = final_data.UNKNOWN_RECORDINGS
TASK2_UNKNOWN_SPECIES = final_data.UNKNOWN_SPECIES
TASK2_UNKNOWN_RECORDINGS_PER_SPECIES = final_data.UNKNOWN_RECORDINGS_PER_SPECIES
TASK2_KNOWN_FINAL_CLIPS = final_data.KNOWN_TEST_ENERGY_CLIPS
TASK2_UNKNOWN_FINAL_CLIPS = final_data.UNKNOWN_ENERGY_CLIPS

_SHA256_LENGTH = 64
_TASK1_SHARD_FIELDS = {
    "schema_version",
    "stage_id",
    "task",
    "seed",
    "recording_id",
    "session_group",
    "true_class_index",
    "true_class_name",
    "mean_logits",
    "predicted_class_index",
    "predicted_class_name",
    "metadata",
    "run_id",
    "run_identity_sha256",
    "checkpoint_sha256",
    "gate_sha256",
    "claim_sha256",
    "cache_lock_sha256",
    "source_role",
    "source_fingerprint_sha256",
}
_TASK2_SHARD_FIELDS = {
    "schema_version",
    "stage_id",
    "task",
    "seed",
    "source_role",
    "recording_id",
    "reconstruction_mse",
    "mean_latent_embedding",
    "metadata",
    "run_id",
    "run_identity_sha256",
    "checkpoint_sha256",
    "gate_sha256",
    "claim_sha256",
    "cache_lock_sha256",
    "source_fingerprint_sha256",
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


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Final evaluation JSON value is not serializable") from exc


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _is_within(path: Path, boundary: Path) -> bool:
    return path == boundary or boundary in path.parents


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Final evaluation requires a nonzero O_NOFOLLOW")
    if not isinstance(directory, int) or directory == 0:
        raise RuntimeError("Final evaluation requires a nonzero O_DIRECTORY")
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | directory


def _open_absolute_directory_no_follow(path: Path) -> int:
    candidate = _absolute(path)
    if not candidate.is_absolute():
        raise ValueError("Final evaluation directory path must be absolute")
    flags = _directory_open_flags()
    descriptor = os.open("/", flags)
    try:
        for part in candidate.parts[1:]:
            if part in {"", ".", ".."}:
                raise ValueError("Final evaluation directory component is invalid")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final evaluation directory component changed type")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_file_beneath(path: Path, boundary: Path) -> int:
    candidate = _absolute(path)
    resolved_boundary = _absolute(boundary)
    if candidate == resolved_boundary or not _is_within(candidate, resolved_boundary):
        raise ValueError(f"Final evaluation artifact leaves its boundary: {candidate}")
    parts = candidate.relative_to(resolved_boundary).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Final evaluation artifact path is invalid")
    descriptor = _open_absolute_directory_no_follow(resolved_boundary)
    directory_flags = _directory_open_flags()
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final evaluation artifact parent changed type")
            os.close(descriptor)
            descriptor = next_descriptor
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(no_follow, int) or no_follow == 0:
            raise RuntimeError("Final evaluation file reads require O_NOFOLLOW")
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow,
            dir_fd=descriptor,
        )
        return file_descriptor
    finally:
        os.close(descriptor)


def _descriptor_snapshot(
    path: str | Path, *, boundary: Path | None = None
) -> tuple[bytes, str, int]:
    candidate = _absolute(path)
    resolved_boundary = _absolute(PROJECT_ROOT if boundary is None else boundary)
    try:
        descriptor = _open_file_beneath(candidate, resolved_boundary)
    except OSError as exc:
        raise ValueError(f"Final evaluation artifact cannot be opened: {candidate}") from exc
    try:
        return _snapshot_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _read_json(
    path: Path, *, boundary: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, size_bytes = _descriptor_snapshot(path, boundary=boundary)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Final evaluation JSON is invalid: {path}") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != payload:
        raise ValueError(f"Final evaluation JSON is not canonical: {path}")
    return value, _artifact_record_from_snapshot(path, digest, size_bytes)


def _attempt_directory() -> Path:
    return _absolute(FINAL_EVALUATION_ATTEMPT_DIRECTORY)


def _artifact_record_from_snapshot(path: Path, digest: str, size_bytes: int) -> dict[str, Any]:
    candidate = _absolute(path)
    attempt = _attempt_directory()
    if _is_within(candidate, attempt):
        record_path = candidate.relative_to(attempt).as_posix()
    else:
        record_path = str(candidate)
    return {"path": record_path, "sha256": digest, "size_bytes": size_bytes}


def _artifact_record(path: Path, *, boundary: Path | None = None) -> dict[str, Any]:
    _, digest, size_bytes = _descriptor_snapshot(path, boundary=boundary)
    return _artifact_record_from_snapshot(path, digest, size_bytes)


def _external_artifact_record(path: Path) -> dict[str, Any]:
    _, digest, size_bytes = _descriptor_snapshot(path, boundary=_absolute(PROJECT_ROOT))
    return {"path": str(_absolute(path)), "sha256": digest, "size_bytes": size_bytes}


def _relative_attempt_parts(path: Path) -> tuple[str, ...]:
    candidate = _absolute(path)
    attempt = _attempt_directory()
    if candidate == attempt or not _is_within(candidate, attempt):
        raise ValueError("Final evaluation output leaves the fixed attempt")
    parts = candidate.relative_to(attempt).parts
    if not parts or any(part in {"", ".", ".."} or "/" in part for part in parts):
        raise ValueError("Final evaluation output path is invalid")
    return parts


def _open_attempt_root() -> int:
    try:
        descriptor = _open_absolute_directory_no_follow(_attempt_directory())
    except OSError as exc:
        raise ValueError("Final evaluation attempt directory cannot be opened safely") from exc
    mode = os.fstat(descriptor).st_mode
    if not stat.S_ISDIR(mode):
        os.close(descriptor)
        raise ValueError("Final evaluation attempt descriptor is not a directory")
    return descriptor


def _open_relative_parent(path: Path) -> tuple[int, str]:
    parts = _relative_attempt_parts(path)
    descriptor = _open_attempt_root()
    flags = _directory_open_flags()
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final evaluation output parent is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _secure_ensure_directory(path: Path) -> Path:
    candidate = _absolute(path)
    parts = _relative_attempt_parts(candidate)
    descriptor = _open_attempt_root()
    flags = _directory_open_flags()
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final evaluation directory path changed")
            os.close(descriptor)
            descriptor = next_descriptor
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return candidate


def _snapshot_descriptor(descriptor: int) -> tuple[bytes, str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Final evaluation publication is not a regular file")
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise RuntimeError("Final evaluation publication ended while read")
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
    if before_identity != after_identity or offset != before.st_size:
        raise RuntimeError("Final evaluation publication changed while verified")
    return b"".join(chunks), digest.hexdigest(), before.st_size


def _create_only_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if not payload:
        raise ValueError("Final evaluation artifact payload cannot be empty")
    destination = _absolute(path)
    parent_descriptor, destination_name = _open_relative_parent(destination)
    temporary_name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Final evaluation publication requires O_NOFOLLOW")
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
                raise RuntimeError("Final evaluation publication write made no progress")
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
            observed, digest, size_bytes = _snapshot_descriptor(read_descriptor)
        finally:
            os.close(read_descriptor)
        if observed != payload:
            raise RuntimeError("Final evaluation artifact changed during publication")
        return _artifact_record_from_snapshot(destination, digest, size_bytes)
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _create_only_bytes(path, _canonical_json_bytes(dict(value)))


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if os.path.lexists(path):
        observed, record = _read_json(path, boundary=_attempt_directory())
        if observed != dict(value):
            raise ValueError(f"Existing final evaluation JSON differs: {path}")
        return record
    try:
        return _write_json_create_only(path, value)
    except FileExistsError:
        observed, record = _read_json(path, boundary=_attempt_directory())
        if observed != dict(value):
            raise ValueError(f"Concurrent final evaluation JSON differs: {path}") from None
        return record


def _require_real_directory(path: Path, name: str) -> None:
    try:
        descriptor = _open_absolute_directory_no_follow(_absolute(path))
    except OSError as exc:
        raise ValueError(f"{name} does not exist") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{name} is not a real directory")
    finally:
        os.close(descriptor)


def _directory_entries(path: Path, name: str) -> dict[str, str]:
    try:
        descriptor = _open_absolute_directory_no_follow(_absolute(path))
    except OSError as exc:
        raise ValueError(f"{name} cannot be opened safely") from exc
    result: dict[str, str] = {}
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError(f"{name} contains a symlink: {entry.name}")
                if entry.is_file(follow_symlinks=False):
                    kind = "file"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "directory"
                else:
                    raise ValueError(f"{name} contains a nonregular entry: {entry.name}")
                result[entry.name] = kind
    finally:
        os.close(descriptor)
    return result


def _validate_artifact_reference(
    value: object,
    expected_path: Path,
    *,
    internal: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Final evaluation artifact reference schema is invalid")
    expected = (
        _artifact_record(expected_path, boundary=_attempt_directory())
        if internal
        else _external_artifact_record(expected_path)
    )
    if value != expected:
        raise ValueError("Final evaluation artifact reference changed")
    return expected


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_utc_timestamp(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} timestamp must use UTC")
    return value


def _completion_time(path: Path, field: str = "completed_at_utc") -> str:
    if not os.path.lexists(path):
        return _utc_now()
    value, _ = _read_json(path, boundary=_attempt_directory())
    return _require_utc_timestamp(value.get(field), field)


def _shard_filename(recording_id: str) -> str:
    if type(recording_id) is not str or not recording_id:
        raise ValueError("Recording identity is invalid")
    return hashlib.sha256(recording_id.encode("utf-8")).hexdigest() + ".json"


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be a sequence of strings")
    resolved = tuple(command)
    if any(
        type(item) is not str
        or not item
        or item != item.strip()
        or any(ord(character) < 32 for character in item)
        for item in resolved
    ):
        raise ValueError("command contains an invalid argument")
    return resolved


def _prepare_production_runtime() -> torch.device:
    expected_venv = _absolute(PROJECT_ROOT / ".venv")
    if _absolute(sys.prefix) != expected_venv or not _is_within(
        _absolute(sys.executable), expected_venv
    ):
        raise RuntimeError("Final evaluation must run inside the project .venv")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
    if fallback not in {"", "0", "false"}:
        raise RuntimeError("Final evaluation requires MPS fallback to be disabled")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Final evaluation requires an available MPS device")
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.get_default_dtype() != torch.float32
    ):
        raise RuntimeError("Final evaluation deterministic float32 runtime was not established")
    return torch.device("mps")


def _synchronize_device(device: torch.device) -> None:
    if isinstance(device, torch.device) and device.type == "mps":
        torch.mps.synchronize()


def _release_device_cache(device: torch.device) -> None:
    if isinstance(device, torch.device) and device.type == "mps":
        torch.mps.empty_cache()


@contextmanager
def _transaction_lock(*, exclusive: bool) -> Any:
    root = _absolute(FINAL_EVALUATION_ROOT)
    descriptor = _open_absolute_directory_no_follow(root)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("Final evaluation root descriptor is not a directory")
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _run_source_fingerprint(run: Mapping[str, Any]) -> str:
    value = run.get("source_fingerprint_sha256")
    if value is None:
        value = run.get("release_source_fingerprint_sha256")
    return _require_sha256(value, "Final evaluation run source fingerprint")


def _assert_gate_current(
    gate: Mapping[str, Any],
    gate_sha256: str,
    *,
    full: bool,
) -> None:
    expected_source = _require_sha256(
        gate.get("shared_identity", {}).get("source_fingerprint_sha256")
        if isinstance(gate.get("shared_identity"), dict)
        else None,
        "Final evaluation gate source fingerprint",
    )
    if source_fingerprint() != expected_source:
        raise PermissionError("Current source fingerprint differs from the sealed gate")
    observed_gate = _external_artifact_record(_absolute(FINAL_EVALUATION_GATE_PATH))
    if observed_gate["sha256"] != gate_sha256:
        raise PermissionError("Sealed final evaluation gate changed")
    if full:
        verified = verify_final_evaluation_gate()
        observed_gate_record, _ = _gate_artifacts(verified)
        if verified.get("gate") != dict(gate) or observed_gate_record["sha256"] != gate_sha256:
            raise PermissionError("Current final evaluation evidence differs from the sealed gate")


def _assert_run_current(run: Mapping[str, Any], gate_sha256: str) -> None:
    if source_fingerprint() != _run_source_fingerprint(run):
        raise PermissionError("Current source fingerprint differs from the evaluation run")
    if _external_artifact_record(_absolute(FINAL_EVALUATION_GATE_PATH))["sha256"] != gate_sha256:
        raise PermissionError("Sealed final evaluation gate changed before publication")


def _preflight_models(
    gate: Mapping[str, Any],
    device: torch.device,
) -> None:
    for run in _run_inventory(gate, "task1"):
        model, metadata = load_locked_task1_best_model(
            run["best_checkpoint"]["path"],
            checkpoint_sha256=run["best_checkpoint"]["sha256"],
            expected_run_identity_sha256=run["run_identity_sha256"],
            device=device,
        )
        try:
            _synchronize_device(device)
            _validate_model_metadata(metadata, run, task="task1")
        finally:
            del model
            _release_device_cache(device)
    for run in _run_inventory(gate, "task2"):
        model, metadata = load_locked_task2_best_model_for_evaluation(
            run["best_checkpoint"]["path"],
            expected_sha256=run["best_checkpoint"]["sha256"],
            expected_run_identity_sha256=run["run_identity_sha256"],
            device=device,
        )
        try:
            _synchronize_device(device)
            _validate_model_metadata(metadata, run, task="task2")
        finally:
            del model
            _release_device_cache(device)


def _gate_artifacts(verified: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_record = _external_artifact_record(_absolute(FINAL_EVALUATION_GATE_PATH))
    lock_record = _external_artifact_record(_absolute(FINAL_EVALUATION_GATE_LOCK_PATH))
    if verified.get("gate_artifact") != gate_record or verified.get("lock_artifact") != lock_record:
        raise ValueError("Final evaluation gate artifact bindings are inconsistent")
    return gate_record, lock_record


def _validate_gate_value(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Final evaluation gate value is invalid")
    if (
        value.get("ready") is not True
        or value.get("seed_order") != list(SEED_ORDER)
        or not isinstance(value.get("task1"), dict)
        or not isinstance(value.get("task2"), dict)
    ):
        raise ValueError("Final evaluation gate is not ready for the fixed seed inventory")
    return value


def _run_inventory(gate: Mapping[str, Any], task: str) -> tuple[dict[str, Any], ...]:
    section = gate.get(task)
    if not isinstance(section, dict) or section.get("seeds") != list(SEED_ORDER):
        raise ValueError(f"Final evaluation {task} gate section is invalid")
    raw_runs = section.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != len(SEED_ORDER):
        raise ValueError(f"Final evaluation {task} run inventory is invalid")
    runs: list[dict[str, Any]] = []
    for seed, raw_run in zip(SEED_ORDER, raw_runs, strict=True):
        if not isinstance(raw_run, dict) or raw_run.get("seed") != seed:
            raise ValueError(f"Final evaluation {task} run seed inventory changed")
        if (
            type(raw_run.get("run_id")) is not str
            or not raw_run["run_id"]
            or not _is_sha256(raw_run.get("run_identity_sha256"))
        ):
            raise ValueError(f"Final evaluation {task} run identity is invalid")
        checkpoint = raw_run.get("best_checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != {"path", "sha256", "size_bytes"}
            or type(checkpoint.get("path")) is not str
        ):
            raise ValueError(f"Final evaluation {task} best checkpoint binding is invalid")
        _validate_artifact_reference(
            checkpoint,
            _absolute(checkpoint["path"]),
            internal=False,
        )
        runs.append(raw_run)
    if tuple(run["seed"] for run in runs) != SEED_ORDER:
        raise ValueError(f"Final evaluation {task} run ordering changed")
    return tuple(runs)


def _claim_fields() -> set[str]:
    return {
        "schema_version",
        "attempt_id",
        "claimed_at_utc",
        "gate_id",
        "gate",
        "gate_lock",
        "attempt_directory",
        "stage_order",
        "single_attempt",
    }


def _validate_claim_value(
    claim: object,
    claim_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
    gate_lock_record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(claim, dict) or set(claim) != _claim_fields():
        raise ValueError("Final evaluation claim schema is invalid")
    expected_relative = _attempt_directory().relative_to(_absolute(PROJECT_ROOT)).as_posix()
    if (
        claim.get("schema_version") != final_data.FINAL_EVALUATION_DATA_SCHEMA_VERSION
        or claim.get("attempt_id") != FINAL_EVALUATION_ATTEMPT_ID
        or claim.get("gate") != gate_record
        or claim.get("gate_lock") != gate_lock_record
        or claim.get("attempt_directory") != expected_relative
        or claim.get("stage_order") != list(STAGE_ORDER)
        or claim.get("single_attempt") is not True
        or not _is_sha256(claim_record.get("sha256"))
    ):
        raise ValueError("Final evaluation claim binding is invalid")
    _require_utc_timestamp(claim.get("claimed_at_utc"), "claim")
    return claim


def _verify_existing_claim(
    verified_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate = _validate_gate_value(verified_gate.get("gate"))
    gate_record, gate_lock_record = _gate_artifacts(verified_gate)
    claim, claim_record = _read_json(
        _absolute(FINAL_EVALUATION_CLAIM_PATH),
        boundary=_absolute(PROJECT_ROOT),
    )
    _validate_claim_value(claim, claim_record, gate_record, gate_lock_record)
    return gate, claim, claim_record, gate_record


def _claim_after_gate(
    verified_gate: Mapping[str, Any],
) -> tuple[
    FinalEvaluationAuthorization,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    gate = _validate_gate_value(verified_gate.get("gate"))
    gate_record, gate_lock_record = _gate_artifacts(verified_gate)
    claimed = claim_final_evaluation_attempt()
    if not isinstance(claimed, dict):
        raise ValueError("Final evaluation claim return value is invalid")
    authorization = claimed.get("authorization")
    claim = claimed.get("claim")
    claim_record = claimed.get("claim_artifact")
    if not isinstance(authorization, FinalEvaluationAuthorization):
        raise TypeError("Final evaluation claim did not return an authorization")
    if claimed.get("gate") != gate or not isinstance(claim_record, dict):
        raise ValueError("Final evaluation claim differs from the verified gate")
    _validate_claim_value(claim, claim_record, gate_record, gate_lock_record)
    if (
        authorization.gate_sha256 != gate_record["sha256"]
        or authorization.claim_sha256 != claim_record["sha256"]
        or _absolute(authorization.attempt_directory) != _attempt_directory()
    ):
        raise ValueError("Final evaluation authorization binding is invalid")
    _require_real_directory(_attempt_directory(), "Final evaluation attempt")
    return authorization, gate, claim, claim_record


def _validate_attempt_entries(*, complete: bool) -> None:
    entries = _directory_entries(_attempt_directory(), "Final evaluation attempt")
    allowed = set(STAGE_ORDER) | {FAILURE_DIRECTORY_NAME, "result.json", "lock.json"}
    if not set(entries).issubset(allowed):
        raise ValueError("Final evaluation attempt contains unexpected entries")
    for name, kind in entries.items():
        expected_kind = "file" if name in {"result.json", "lock.json"} else "directory"
        if kind != expected_kind:
            raise ValueError(f"Final evaluation attempt entry type changed: {name}")
    if complete:
        required = set(STAGE_ORDER) | {"result.json", "lock.json"}
        if not required.issubset(entries):
            raise ValueError("Final evaluation attempt is missing completed evidence")


def _validate_model_metadata(
    metadata: object,
    run: Mapping[str, Any],
    *,
    task: str,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError(f"Final evaluation {task} model metadata is invalid")
    checkpoint_sha256 = run["best_checkpoint"]["sha256"]
    expected = {
        "run_id": run["run_id"],
        "run_identity_sha256": run["run_identity_sha256"],
        "seed": run["seed"],
        "scope": "production",
        "production_evidence": True,
    }
    checkpoint_field = "checkpoint_sha256" if task == "task1" else "best_checkpoint_sha256"
    expected[checkpoint_field] = checkpoint_sha256
    mismatches = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatches:
        raise ValueError(f"Final evaluation {task} model identity differs: {mismatches}")


def _stage_directory(stage_id: str) -> Path:
    if stage_id not in STAGE_ORDER:
        raise ValueError("Final evaluation stage identity is invalid")
    return _attempt_directory() / stage_id


def _ensure_stage_directory(stage_id: str) -> Path:
    directory = _stage_directory(stage_id)
    _secure_ensure_directory(directory)
    _require_real_directory(directory, f"Final evaluation stage {stage_id}")
    return directory


def _validate_stage_entries(stage_id: str, *, task: str, complete: bool) -> None:
    directory = _stage_directory(stage_id)
    entries = _directory_entries(directory, f"Final evaluation stage {stage_id}")
    shard_directories = {"shards"} if task == "task1" else {"known_test_shards", "unknown_shards"}
    allowed = shard_directories | {"result.json", "lock.json"}
    if not set(entries).issubset(allowed):
        raise ValueError(f"Final evaluation stage {stage_id} contains unexpected entries")
    for name, kind in entries.items():
        expected_kind = "directory" if name in shard_directories else "file"
        if kind != expected_kind:
            raise ValueError(f"Final evaluation stage {stage_id} entry type changed")
    if complete and set(entries) != allowed:
        raise ValueError(f"Final evaluation stage {stage_id} is incomplete")


def _task1_shard_value(
    prediction: RecordingPrediction,
    metadata: FinalRecordingMetadata,
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    if (
        metadata.source_role != FINAL_KNOWN_TEST_ROLE
        or metadata.recording_id != prediction.recording_id
        or metadata.session_group != prediction.session_group
        or metadata.class_index != prediction.true_class_index
    ):
        raise ValueError("Task 1 prediction and immutable metadata differ")
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "stage_id": stage_id,
        "task": "task1_classification",
        "seed": run["seed"],
        **prediction.to_record(),
        "metadata": metadata.to_record(),
        "run_id": run["run_id"],
        "run_identity_sha256": run["run_identity_sha256"],
        "checkpoint_sha256": run["best_checkpoint"]["sha256"],
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "cache_lock_sha256": final_data.KNOWN_CACHE_LOCK_SHA256,
        "source_role": FINAL_KNOWN_TEST_ROLE,
        "source_fingerprint_sha256": _run_source_fingerprint(run),
    }


def _parse_task1_shard(
    value: object,
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> tuple[RecordingPrediction, FinalRecordingMetadata]:
    if not isinstance(value, dict) or set(value) != _TASK1_SHARD_FIELDS:
        raise ValueError("Task 1 final shard schema is invalid")
    try:
        prediction = RecordingPrediction(
            recording_id=value["recording_id"],
            session_group=value["session_group"],
            true_class_index=value["true_class_index"],
            mean_logits=tuple(value["mean_logits"]),
            predicted_class_index=value["predicted_class_index"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task 1 final shard prediction is invalid") from exc
    metadata = _metadata_from_record(value.get("metadata"))
    if value != _task1_shard_value(
        prediction,
        metadata,
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    ):
        raise ValueError("Task 1 final shard identity or labels changed")
    return prediction, metadata


def _metadata_from_record(value: object) -> FinalRecordingMetadata:
    expected_fields = {
        "source_role",
        "recording_id",
        "session_group",
        "species_common_name",
        "species_scientific_name",
        "class_index",
        "clip_ids",
        "clip_count",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Task 2 final shard metadata schema is invalid")
    try:
        metadata = FinalRecordingMetadata(
            source_role=value["source_role"],
            recording_id=value["recording_id"],
            session_group=value["session_group"],
            species_common_name=value["species_common_name"],
            species_scientific_name=value["species_scientific_name"],
            class_index=value["class_index"],
            clip_ids=tuple(value["clip_ids"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task 2 final shard metadata is invalid") from exc
    if metadata.to_record() != value:
        raise ValueError("Task 2 final shard metadata changed")
    return metadata


def _task2_shard_value(
    score: RecordingScore,
    metadata: FinalRecordingMetadata,
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    if score.recording_id != metadata.recording_id or score.clip_ids != metadata.clip_ids:
        raise ValueError("Task 2 final score and metadata identities differ")
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "stage_id": stage_id,
        "task": "task2_novelty_detection",
        "seed": run["seed"],
        "source_role": metadata.source_role,
        "recording_id": score.recording_id,
        "reconstruction_mse": float(score.reconstruction_mse),
        "mean_latent_embedding": [float(value) for value in score.mean_latent_embedding],
        "metadata": metadata.to_record(),
        "run_id": run["run_id"],
        "run_identity_sha256": run["run_identity_sha256"],
        "checkpoint_sha256": run["best_checkpoint"]["sha256"],
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "cache_lock_sha256": (
            final_data.KNOWN_CACHE_LOCK_SHA256
            if metadata.source_role == FINAL_KNOWN_TEST_ROLE
            else final_data.UNKNOWN_CACHE_LOCK_SHA256
        ),
        "source_fingerprint_sha256": _run_source_fingerprint(run),
    }


def _parse_task2_shard(
    value: object,
    *,
    source_role: str,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> tuple[RecordingScore, FinalRecordingMetadata]:
    if not isinstance(value, dict) or set(value) != _TASK2_SHARD_FIELDS:
        raise ValueError("Task 2 final shard schema is invalid")
    metadata = _metadata_from_record(value.get("metadata"))
    if metadata.source_role != source_role:
        raise ValueError("Task 2 final shard source role changed")
    if type(value.get("reconstruction_mse")) is not float or not math.isfinite(
        value["reconstruction_mse"]
    ):
        raise ValueError("Task 2 final shard reconstruction score is invalid")
    latent = value.get("mean_latent_embedding")
    if (
        not isinstance(latent, list)
        or not latent
        or any(type(item) is not float or not math.isfinite(item) for item in latent)
    ):
        raise ValueError("Task 2 final shard latent embedding is invalid")
    try:
        score = RecordingScore(
            recording_id=value["recording_id"],
            clip_ids=metadata.clip_ids,
            reconstruction_mse=value["reconstruction_mse"],
            mean_latent_embedding=tuple(latent),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task 2 final shard score is invalid") from exc
    if value != _task2_shard_value(
        score,
        metadata,
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    ):
        raise ValueError("Task 2 final shard identity binding changed")
    return score, metadata


def _read_shard_directory(
    directory: Path,
    *,
    parser: Any,
) -> tuple[dict[str, tuple[Any, ...]], tuple[dict[str, Any], ...]]:
    entries = _directory_entries(directory, f"Final evaluation shard directory {directory.name}")
    if any(kind != "file" for kind in entries.values()):
        raise ValueError("Final evaluation shard directory contains another directory")
    parsed: dict[str, tuple[Any, ...]] = {}
    records: list[dict[str, Any]] = []
    for filename in sorted(entries):
        if len(filename) != _SHA256_LENGTH + len(".json") or not filename.endswith(".json"):
            raise ValueError("Final evaluation shard filename is invalid")
        if not _is_sha256(filename[:_SHA256_LENGTH]):
            raise ValueError("Final evaluation shard filename digest is invalid")
        value, record = _read_json(directory / filename, boundary=_attempt_directory())
        item = parser(value)
        recording_id = item[0].recording_id
        if filename != _shard_filename(recording_id):
            raise ValueError("Final evaluation shard filename does not bind its recording")
        if recording_id in parsed:
            raise ValueError("Final evaluation shard recording identity is duplicated")
        parsed[recording_id] = tuple(item)
        records.append(record)
    return parsed, tuple(records)


def _write_task1_shard(
    directory: Path,
    prediction: RecordingPrediction,
    metadata: FinalRecordingMetadata,
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    value = _task1_shard_value(
        prediction,
        metadata,
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    return _write_json_create_only(directory / _shard_filename(prediction.recording_id), value)


def _write_task2_shard(
    directory: Path,
    score: RecordingScore,
    metadata: FinalRecordingMetadata,
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    value = _task2_shard_value(
        score,
        metadata,
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    return _write_json_create_only(directory / _shard_filename(score.recording_id), value)


def _read_bound_development_json(record: object) -> dict[str, Any]:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256", "size_bytes"}
        or type(record.get("path")) is not str
    ):
        raise ValueError("Task 2 development artifact record is invalid")
    path = _absolute(record["path"])
    _validate_artifact_reference(record, path, internal=False)
    value, observed = _read_json(path, boundary=_absolute(PROJECT_ROOT))
    if observed != record:
        raise ValueError("Task 2 development artifact changed")
    return value


def _latent_reference_from_record(value: object) -> LatentReference:
    if not isinstance(value, dict):
        raise ValueError("Task 2 latent reference record is invalid")
    try:
        reference = LatentReference(
            fit_role=value["fit_role"],
            recording_ids=tuple(value["recording_ids"]),
            coordinate_mean=tuple(value["coordinate_mean"]),
            population_variance=tuple(value["population_variance"]),
            coordinate_scale=tuple(value["coordinate_scale"]),
            standardized_embeddings=tuple(tuple(row) for row in value["standardized_embeddings"]),
            nearest_neighbours=value["nearest_neighbours"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task 2 latent reference record is invalid") from exc
    if reference.to_record() != value:
        raise ValueError("Task 2 latent reference record changed")
    return reference


def _threshold_from_record(value: object) -> NoveltyThreshold:
    if not isinstance(value, dict):
        raise ValueError("Task 2 novelty threshold record is invalid")
    try:
        threshold = NoveltyThreshold(
            score_name=value["score_name"],
            value=value["value"],
            calibration_role=value["calibration_role"],
            calibration_recording_ids=tuple(value["calibration_recording_ids"]),
            quantile=value["quantile"],
            method=value["method"],
            direction=value["direction"],
            classification_operator=value["classification_operator"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task 2 novelty threshold record is invalid") from exc
    if threshold.to_record() != value:
        raise ValueError("Task 2 novelty threshold record changed")
    return threshold


def _task2_development_evidence(
    run: Mapping[str, Any],
) -> tuple[LatentReference, dict[str, NoveltyThreshold], dict[str, Any]]:
    artifacts = run.get("development_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Task 2 gate lacks development artifacts")
    reference_record = artifacts.get("training_latent_reference")
    thresholds_record = artifacts.get("thresholds")
    reference_artifact = _read_bound_development_json(reference_record)
    thresholds_artifact = _read_bound_development_json(thresholds_record)
    checkpoint_sha256 = run["best_checkpoint"]["sha256"]
    for name, artifact in (
        ("latent reference", reference_artifact),
        ("thresholds", thresholds_artifact),
    ):
        if (
            artifact.get("run_identity_sha256") != run["run_identity_sha256"]
            or artifact.get("best_checkpoint_sha256") != checkpoint_sha256
            or artifact.get("seed") != run["seed"]
        ):
            raise ValueError(f"Task 2 {name} development binding changed")
    reference = _latent_reference_from_record(reference_artifact.get("reference"))
    reconstruction = _threshold_from_record(thresholds_artifact.get("reconstruction"))
    latent = _threshold_from_record(thresholds_artifact.get("latent"))
    if (
        reconstruction.score_name != RECONSTRUCTION_SCORE_NAME
        or latent.score_name != LATENT_SCORE_NAME
    ):
        raise ValueError("Task 2 development threshold score names changed")
    bindings = {
        "training_latent_reference": dict(reference_record),
        "thresholds": dict(thresholds_record),
    }
    return reference, {"reconstruction": reconstruction, "latent": latent}, bindings


def _recording_batch(
    source_role: str,
    items: Mapping[str, tuple[RecordingScore, FinalRecordingMetadata]],
) -> tuple[RecordingBatch, tuple[FinalRecordingMetadata, ...]]:
    ordered = tuple(items[recording_id] for recording_id in sorted(items))
    batch = RecordingBatch(source_role=source_role, recordings=tuple(item[0] for item in ordered))
    metadata = tuple(item[1] for item in ordered)
    if batch.recording_ids != tuple(item.recording_id for item in metadata):
        raise ValueError("Task 2 shard score and metadata ordering differs")
    return batch, metadata


def _scored_recordings(
    batch: RecordingBatch,
    metadata: Sequence[FinalRecordingMetadata],
    *,
    scores: Mapping[str, float],
) -> tuple[ScoredRecording, ...]:
    by_id = {item.recording_id: item for item in metadata}
    source = KNOWN_SOURCE if batch.source_role == FINAL_KNOWN_TEST_ROLE else UNKNOWN_SOURCE
    if not set(batch.recording_ids).issubset(scores) or set(by_id) != set(batch.recording_ids):
        raise ValueError("Task 2 final score identities are incomplete")
    return tuple(
        ScoredRecording(
            recording_id=recording_id,
            session_group=by_id[recording_id].session_group,
            species_scientific_name=by_id[recording_id].species_scientific_name,
            source=source,
            score=np.float64(scores[recording_id]),
        )
        for recording_id in sorted(batch.recording_ids)
    )


def _task2_score_streams(
    known_batch: RecordingBatch,
    known_metadata: Sequence[FinalRecordingMetadata],
    unknown_batch: RecordingBatch,
    unknown_metadata: Sequence[FinalRecordingMetadata],
    reference: LatentReference,
) -> dict[str, tuple[ScoredRecording, ...]]:
    reconstruction: dict[str, float] = {
        item.recording_id: item.reconstruction_mse
        for item in (*known_batch.recordings, *unknown_batch.recordings)
    }
    known_latent = latent_knn_novelty_scores(reference, known_batch)
    unknown_latent = latent_knn_novelty_scores(reference, unknown_batch)
    latent = {item.recording_id: item.score for item in (*known_latent, *unknown_latent)}
    return {
        "reconstruction": (
            *_scored_recordings(known_batch, known_metadata, scores=reconstruction),
            *_scored_recordings(unknown_batch, unknown_metadata, scores=reconstruction),
        ),
        "latent": (
            *_scored_recordings(known_batch, known_metadata, scores=latent),
            *_scored_recordings(unknown_batch, unknown_metadata, scores=latent),
        ),
    }


def _task1_stage_result(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    predictions: Sequence[RecordingPrediction],
    gate_sha256: str,
    claim_sha256: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    ordered = tuple(sorted(predictions, key=lambda item: item.recording_id))
    if len(ordered) != TASK1_FINAL_RECORDINGS:
        raise ValueError("Task 1 final stage recording count differs from the fixed protocol")
    metrics = evaluate_recording_predictions(ordered)
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "complete": True,
        "stage_id": stage_id,
        "task": "task1_classification",
        "seed": run["seed"],
        "completed_at_utc": _require_utc_timestamp(completed_at_utc, stage_id),
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "run_id": run["run_id"],
        "run_identity_sha256": run["run_identity_sha256"],
        "checkpoint": dict(run["best_checkpoint"]),
        "cache_lock_sha256": final_data.KNOWN_CACHE_LOCK_SHA256,
        "source_role": FINAL_KNOWN_TEST_ROLE,
        "source_fingerprint_sha256": _run_source_fingerprint(run),
        "class_order": list(LOCKED_TASK1_CLASS_ORDER),
        "recording_count": len(ordered),
        "recording_ids": [item.recording_id for item in ordered],
        "metrics": metrics.to_record(),
    }


def _task1_stage_lock(
    *,
    stage_id: str,
    result_record: Mapping[str, Any],
    shard_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "stage_id": stage_id,
        "result": dict(result_record),
        "shards": [dict(record) for record in shard_records],
    }


def _read_task1_shards(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> tuple[
    dict[str, RecordingPrediction],
    dict[str, FinalRecordingMetadata],
    tuple[dict[str, Any], ...],
]:
    directory = _stage_directory(stage_id) / "shards"
    parsed, records = _read_shard_directory(
        directory,
        parser=lambda value: _parse_task1_shard(
            value,
            stage_id=stage_id,
            run=run,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        ),
    )
    return (
        {key: value[0] for key, value in parsed.items()},
        {key: value[1] for key, value in parsed.items()},
        records,
    )


def _finalize_task1_stage(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    _validate_stage_entries(stage_id, task="task1", complete=False)
    predictions, _, shard_records = _read_task1_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    result_path = _stage_directory(stage_id) / "result.json"
    completed_at = _completion_time(result_path)
    result = _task1_stage_result(
        stage_id=stage_id,
        run=run,
        predictions=tuple(predictions.values()),
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        completed_at_utc=completed_at,
    )
    if result["recording_count"] <= 0:
        raise ValueError("Task 1 final stage contains no recording shards")
    _assert_run_current(run, gate_sha256)
    result_record = _write_or_verify_json(result_path, result)
    lock = _task1_stage_lock(
        stage_id=stage_id,
        result_record=result_record,
        shard_records=shard_records,
    )
    _assert_run_current(run, gate_sha256)
    lock_record = _write_or_verify_json(_stage_directory(stage_id) / "lock.json", lock)
    return _verify_task1_stage(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        expected_result=result,
        expected_lock=lock,
        lock_record=lock_record,
    )


def _verify_task1_stage(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
    expected_result: Mapping[str, Any] | None = None,
    expected_lock: Mapping[str, Any] | None = None,
    lock_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_stage_entries(stage_id, task="task1", complete=True)
    predictions, shard_metadata, shard_records = _read_task1_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    result_path = _stage_directory(stage_id) / "result.json"
    observed_result, result_record = _read_json(result_path, boundary=_attempt_directory())
    result = _task1_stage_result(
        stage_id=stage_id,
        run=run,
        predictions=tuple(predictions.values()),
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        completed_at_utc=observed_result.get("completed_at_utc"),
    )
    if observed_result != result or (
        expected_result is not None and dict(expected_result) != result
    ):
        raise ValueError(f"Task 1 final stage result changed: {stage_id}")
    expected_ids = tuple(result["recording_ids"])
    if set(expected_ids) != set(predictions) or len(expected_ids) != len(predictions):
        raise ValueError("Task 1 final stage shard inventory is incomplete")
    lock_path = _stage_directory(stage_id) / "lock.json"
    observed_lock, observed_lock_record = _read_json(lock_path, boundary=_attempt_directory())
    lock = _task1_stage_lock(
        stage_id=stage_id,
        result_record=result_record,
        shard_records=shard_records,
    )
    if observed_lock != lock or (expected_lock is not None and dict(expected_lock) != lock):
        raise ValueError(f"Task 1 final stage lock changed: {stage_id}")
    if lock_record is not None and dict(lock_record) != observed_lock_record:
        raise ValueError("Task 1 final stage lock publication changed")
    return {
        "stage_id": stage_id,
        "seed": run["seed"],
        "result": result,
        "result_artifact": result_record,
        "lock_artifact": observed_lock_record,
        "predictions": tuple(predictions[recording_id] for recording_id in sorted(predictions)),
        "metadata": shard_metadata,
    }


def _run_task1_stage(
    *,
    run: Mapping[str, Any],
    data: Any,
    device: torch.device,
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    stage_id = f"task1_seed_{run['seed']}"
    directory = _ensure_stage_directory(stage_id)
    lock_path = directory / "lock.json"
    result_path = directory / "result.json"
    if os.path.lexists(lock_path):
        if not os.path.lexists(result_path):
            raise ValueError("Task 1 final stage lock exists without its result")
        return _verify_task1_stage(
            stage_id=stage_id,
            run=run,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        )
    shard_directory = directory / "shards"
    _secure_ensure_directory(shard_directory)
    _validate_stage_entries(stage_id, task="task1", complete=False)
    existing, existing_metadata, _ = _read_task1_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    expected_ids = tuple(data.recording_ids)
    if len(set(expected_ids)) != len(expected_ids) or not set(existing).issubset(expected_ids):
        raise ValueError("Task 1 final stage has extra or duplicate recording shards")
    expected_metadata = _reader_metadata(data, source_role=FINAL_KNOWN_TEST_ROLE)
    if any(
        existing_metadata[recording_id].to_record() != expected_metadata[recording_id].to_record()
        for recording_id in existing
    ):
        raise ValueError("Task 1 partial shard labels differ from gated reader metadata")
    if os.path.lexists(result_path):
        if set(existing) != set(expected_ids):
            raise ValueError("Task 1 final stage result exists before all shards")
        return _finalize_task1_stage(
            stage_id=stage_id,
            run=run,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        )
    if set(existing) != set(expected_ids):
        torch.manual_seed(run["seed"])
        model, metadata = load_locked_task1_best_model(
            run["best_checkpoint"]["path"],
            checkpoint_sha256=run["best_checkpoint"]["sha256"],
            expected_run_identity_sha256=run["run_identity_sha256"],
            device=device,
        )
        try:
            _synchronize_device(device)
            _validate_model_metadata(metadata, run, task="task1")
            seen = set(existing)
            for batch in iter_task1_recording_batches(
                model,
                data,
                device=device,
                skip_recording_ids=frozenset(existing),
            ):
                batch_ids = set(batch.recording_ids)
                if (
                    len(batch_ids) != len(batch.recording_ids)
                    or batch_ids.intersection(seen)
                    or not batch_ids.issubset(expected_ids)
                ):
                    raise ValueError("Task 1 inference yielded invalid recording identities")
                _synchronize_device(device)
                _assert_run_current(run, gate_sha256)
                for prediction in batch.predictions:
                    _write_task1_shard(
                        shard_directory,
                        prediction,
                        expected_metadata[prediction.recording_id],
                        stage_id=stage_id,
                        run=run,
                        gate_sha256=gate_sha256,
                        claim_sha256=claim_sha256,
                    )
                seen.update(batch_ids)
        finally:
            _synchronize_device(device)
            del model
            _release_device_cache(device)
    completed, completed_metadata, _ = _read_task1_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    if set(completed) != set(expected_ids) or len(completed) != len(expected_ids):
        raise RuntimeError("Task 1 final inference did not produce every recording shard")
    if any(
        completed_metadata[recording_id].to_record() != expected_metadata[recording_id].to_record()
        for recording_id in completed
    ):
        raise ValueError("Task 1 final shard metadata differs from its gated reader")
    return _finalize_task1_stage(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )


def _task2_stage_result(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    known_items: Mapping[str, tuple[RecordingScore, FinalRecordingMetadata]],
    unknown_items: Mapping[str, tuple[RecordingScore, FinalRecordingMetadata]],
    gate_sha256: str,
    claim_sha256: str,
    completed_at_utc: str,
) -> tuple[dict[str, Any], dict[str, tuple[ScoredRecording, ...]]]:
    if (
        len(known_items) != TASK2_KNOWN_FINAL_RECORDINGS
        or len(unknown_items) != TASK2_UNKNOWN_FINAL_RECORDINGS
        or set(known_items).intersection(unknown_items)
    ):
        raise ValueError("Task 2 final stage recording counts differ from the fixed protocol")
    if (
        sum(len(item[1].clip_ids) for item in known_items.values()) != TASK2_KNOWN_FINAL_CLIPS
        or sum(len(item[1].clip_ids) for item in unknown_items.values())
        != TASK2_UNKNOWN_FINAL_CLIPS
    ):
        raise ValueError("Task 2 final stage clip counts differ from the fixed protocol")
    unknown_species: dict[str, int] = {}
    for _, metadata in unknown_items.values():
        unknown_species[metadata.species_scientific_name] = (
            unknown_species.get(metadata.species_scientific_name, 0) + 1
        )
    if len(unknown_species) != TASK2_UNKNOWN_SPECIES or set(unknown_species.values()) != {
        TASK2_UNKNOWN_RECORDINGS_PER_SPECIES
    }:
        raise ValueError("Task 2 final unknown species counts differ from the fixed protocol")
    known_batch, known_metadata = _recording_batch(FINAL_KNOWN_TEST_ROLE, known_items)
    unknown_batch, unknown_metadata = _recording_batch(FINAL_UNKNOWN_ROLE, unknown_items)
    reference, thresholds, development_bindings = _task2_development_evidence(run)
    final_ids = set(known_items) | set(unknown_items)
    calibration_ids = {
        recording_id
        for threshold in thresholds.values()
        for recording_id in threshold.calibration_recording_ids
    }
    if final_ids.intersection(reference.recording_ids) or final_ids.intersection(calibration_ids):
        raise PermissionError("Task 2 final identities overlap development fit identities")
    streams = _task2_score_streams(
        known_batch,
        known_metadata,
        unknown_batch,
        unknown_metadata,
        reference,
    )
    evaluations = {
        name: evaluate_novelty_scores(streams[name], thresholds[name].value).to_record()
        for name in ("reconstruction", "latent")
    }
    result = {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "complete": True,
        "stage_id": stage_id,
        "task": "task2_novelty_detection",
        "seed": run["seed"],
        "completed_at_utc": _require_utc_timestamp(completed_at_utc, stage_id),
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "run_id": run["run_id"],
        "run_identity_sha256": run["run_identity_sha256"],
        "checkpoint": dict(run["best_checkpoint"]),
        "cache_locks": {
            FINAL_KNOWN_TEST_ROLE: final_data.KNOWN_CACHE_LOCK_SHA256,
            FINAL_UNKNOWN_ROLE: final_data.UNKNOWN_CACHE_LOCK_SHA256,
        },
        "source_roles": [FINAL_KNOWN_TEST_ROLE, FINAL_UNKNOWN_ROLE],
        "source_fingerprint_sha256": _run_source_fingerprint(run),
        "development_bindings": development_bindings,
        "thresholds": {name: thresholds[name].to_record() for name in ("reconstruction", "latent")},
        "known_test_recording_count": len(known_items),
        "known_test_recording_ids": sorted(known_items),
        "unknown_recording_count": len(unknown_items),
        "unknown_recording_ids": sorted(unknown_items),
        "score_streams": evaluations,
    }
    return result, streams


def _task2_stage_lock(
    *,
    stage_id: str,
    result_record: Mapping[str, Any],
    known_shards: Sequence[Mapping[str, Any]],
    unknown_shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "stage_id": stage_id,
        "result": dict(result_record),
        "known_test_shards": [dict(record) for record in known_shards],
        "unknown_shards": [dict(record) for record in unknown_shards],
    }


def _read_task2_role_shards(
    *,
    stage_id: str,
    source_role: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> tuple[
    dict[str, tuple[RecordingScore, FinalRecordingMetadata]],
    tuple[dict[str, Any], ...],
]:
    name = "known_test_shards" if source_role == FINAL_KNOWN_TEST_ROLE else "unknown_shards"
    parsed, records = _read_shard_directory(
        _stage_directory(stage_id) / name,
        parser=lambda value: _parse_task2_shard(
            value,
            source_role=source_role,
            stage_id=stage_id,
            run=run,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        ),
    )
    return {key: (value[0], value[1]) for key, value in parsed.items()}, records


def _read_task2_shards(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> tuple[
    dict[str, tuple[RecordingScore, FinalRecordingMetadata]],
    dict[str, tuple[RecordingScore, FinalRecordingMetadata]],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    known, known_records = _read_task2_role_shards(
        stage_id=stage_id,
        source_role=FINAL_KNOWN_TEST_ROLE,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    unknown, unknown_records = _read_task2_role_shards(
        stage_id=stage_id,
        source_role=FINAL_UNKNOWN_ROLE,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    if set(known).intersection(unknown):
        raise ValueError("Task 2 known and unknown shard identities overlap")
    return known, unknown, known_records, unknown_records


def _finalize_task2_stage(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    _validate_stage_entries(stage_id, task="task2", complete=False)
    known, unknown, known_records, unknown_records = _read_task2_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    if not known or not unknown:
        raise ValueError("Task 2 final stage requires known and unknown shards")
    result_path = _stage_directory(stage_id) / "result.json"
    result, _ = _task2_stage_result(
        stage_id=stage_id,
        run=run,
        known_items=known,
        unknown_items=unknown,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        completed_at_utc=_completion_time(result_path),
    )
    _assert_run_current(run, gate_sha256)
    result_record = _write_or_verify_json(result_path, result)
    lock = _task2_stage_lock(
        stage_id=stage_id,
        result_record=result_record,
        known_shards=known_records,
        unknown_shards=unknown_records,
    )
    _assert_run_current(run, gate_sha256)
    lock_record = _write_or_verify_json(_stage_directory(stage_id) / "lock.json", lock)
    return _verify_task2_stage(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        expected_result=result,
        expected_lock=lock,
        lock_record=lock_record,
    )


def _verify_task2_stage(
    *,
    stage_id: str,
    run: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
    expected_result: Mapping[str, Any] | None = None,
    expected_lock: Mapping[str, Any] | None = None,
    lock_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_stage_entries(stage_id, task="task2", complete=True)
    known, unknown, known_records, unknown_records = _read_task2_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    result_path = _stage_directory(stage_id) / "result.json"
    observed_result, result_record = _read_json(result_path, boundary=_attempt_directory())
    result, streams = _task2_stage_result(
        stage_id=stage_id,
        run=run,
        known_items=known,
        unknown_items=unknown,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        completed_at_utc=observed_result.get("completed_at_utc"),
    )
    if observed_result != result or (
        expected_result is not None and dict(expected_result) != result
    ):
        raise ValueError(f"Task 2 final stage result changed: {stage_id}")
    if set(result["known_test_recording_ids"]) != set(known) or set(
        result["unknown_recording_ids"]
    ) != set(unknown):
        raise ValueError("Task 2 final stage shard inventory is incomplete")
    lock_path = _stage_directory(stage_id) / "lock.json"
    observed_lock, observed_lock_record = _read_json(lock_path, boundary=_attempt_directory())
    lock = _task2_stage_lock(
        stage_id=stage_id,
        result_record=result_record,
        known_shards=known_records,
        unknown_shards=unknown_records,
    )
    if observed_lock != lock or (expected_lock is not None and dict(expected_lock) != lock):
        raise ValueError(f"Task 2 final stage lock changed: {stage_id}")
    if lock_record is not None and dict(lock_record) != observed_lock_record:
        raise ValueError("Task 2 final stage lock publication changed")
    return {
        "stage_id": stage_id,
        "seed": run["seed"],
        "result": result,
        "result_artifact": result_record,
        "lock_artifact": observed_lock_record,
        "score_streams": streams,
        "known_items": known,
        "unknown_items": unknown,
    }


def _run_task2_stage(
    *,
    run: Mapping[str, Any],
    known_data: Any,
    unknown_data: Any,
    device: torch.device,
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    stage_id = f"task2_seed_{run['seed']}"
    directory = _ensure_stage_directory(stage_id)
    lock_path = directory / "lock.json"
    result_path = directory / "result.json"
    if os.path.lexists(lock_path):
        if not os.path.lexists(result_path):
            raise ValueError("Task 2 final stage lock exists without its result")
        return _verify_task2_stage(
            stage_id=stage_id,
            run=run,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        )
    known_directory = directory / "known_test_shards"
    unknown_directory = directory / "unknown_shards"
    _secure_ensure_directory(known_directory)
    _secure_ensure_directory(unknown_directory)
    _validate_stage_entries(stage_id, task="task2", complete=False)
    known, unknown, _, _ = _read_task2_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    expected_known = tuple(known_data.recording_ids)
    expected_unknown = tuple(unknown_data.recording_ids)
    if (
        len(set(expected_known)) != len(expected_known)
        or len(set(expected_unknown)) != len(expected_unknown)
        or set(expected_known).intersection(expected_unknown)
        or not set(known).issubset(expected_known)
        or not set(unknown).issubset(expected_unknown)
    ):
        raise ValueError("Task 2 final stage has extra or duplicate recording shards")
    expected_known_metadata = _reader_metadata(
        known_data,
        source_role=FINAL_KNOWN_TEST_ROLE,
    )
    expected_unknown_metadata = _reader_metadata(
        unknown_data,
        source_role=FINAL_UNKNOWN_ROLE,
    )
    if any(
        item[1].to_record() != expected_known_metadata[recording_id].to_record()
        for recording_id, item in known.items()
    ) or any(
        item[1].to_record() != expected_unknown_metadata[recording_id].to_record()
        for recording_id, item in unknown.items()
    ):
        raise ValueError("Task 2 partial shard metadata differs from its gated reader")
    if os.path.lexists(result_path):
        if set(known) != set(expected_known) or set(unknown) != set(expected_unknown):
            raise ValueError("Task 2 final stage result exists before all shards")
        return _finalize_task2_stage(
            stage_id=stage_id,
            run=run,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        )
    if set(known) != set(expected_known) or set(unknown) != set(expected_unknown):
        torch.manual_seed(run["seed"])
        model, metadata = load_locked_task2_best_model_for_evaluation(
            run["best_checkpoint"]["path"],
            expected_sha256=run["best_checkpoint"]["sha256"],
            expected_run_identity_sha256=run["run_identity_sha256"],
            device=device,
        )
        try:
            _synchronize_device(device)
            _validate_model_metadata(metadata, run, task="task2")
            roles = (
                (
                    known_data,
                    FINAL_KNOWN_TEST_ROLE,
                    known_directory,
                    frozenset(known),
                    set(expected_known),
                ),
                (
                    unknown_data,
                    FINAL_UNKNOWN_ROLE,
                    unknown_directory,
                    frozenset(unknown),
                    set(expected_unknown),
                ),
            )
            for data, source_role, shard_directory, skipped, expected in roles:
                if set(skipped) == expected:
                    continue
                seen = set(skipped)
                for batch in iter_task2_recording_batches(
                    model,
                    data,
                    source_role=source_role,
                    device=device,
                    skip_recording_ids=skipped,
                ):
                    batch_ids = set(batch.recording_ids)
                    if (
                        len(batch_ids) != len(batch.recording_ids)
                        or batch_ids.intersection(seen)
                        or not batch_ids.issubset(expected)
                    ):
                        raise ValueError("Task 2 inference yielded invalid recording identities")
                    _synchronize_device(device)
                    _assert_run_current(run, gate_sha256)
                    for score, recording_metadata in zip(
                        batch.scores.recordings,
                        batch.metadata,
                        strict=True,
                    ):
                        _write_task2_shard(
                            shard_directory,
                            score,
                            recording_metadata,
                            stage_id=stage_id,
                            run=run,
                            gate_sha256=gate_sha256,
                            claim_sha256=claim_sha256,
                        )
                    seen.update(batch_ids)
        finally:
            _synchronize_device(device)
            del model
            _release_device_cache(device)
    complete_known, complete_unknown, _, _ = _read_task2_shards(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )
    if (
        set(complete_known) != set(expected_known)
        or len(complete_known) != len(expected_known)
        or set(complete_unknown) != set(expected_unknown)
        or len(complete_unknown) != len(expected_unknown)
    ):
        raise RuntimeError("Task 2 final inference did not produce every recording shard")
    return _finalize_task2_stage(
        stage_id=stage_id,
        run=run,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
    )


def _validate_npz_payload(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Final evaluation NPZ payload is invalid")
    resolved: dict[str, np.ndarray] = {}
    for name, value in payload.items():
        if type(name) is not str or not name or not isinstance(value, np.ndarray):
            raise ValueError("Final evaluation NPZ payload entry is invalid")
        if value.dtype.hasobject:
            raise ValueError("Final evaluation NPZ cannot contain object arrays")
        array = np.array(value, copy=True, order="C")
        if np.issubdtype(array.dtype, np.floating) and not bool(np.all(np.isfinite(array))):
            raise ValueError("Final evaluation NPZ contains nonfinite values")
        resolved[name] = array
    return resolved


def _read_npz(
    path: Path,
    *,
    expected_keys: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload, digest, size_bytes = _descriptor_snapshot(path, boundary=_attempt_directory())
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)) or set(archive.files) != expected_keys:
                raise ValueError("Final evaluation NPZ member inventory is invalid")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Final evaluation NPZ is unsafe or invalid: {path}") from exc
    resolved = _validate_npz_payload(arrays)
    return resolved, _artifact_record_from_snapshot(path, digest, size_bytes)


def _npz_bytes(payload: Mapping[str, np.ndarray]) -> bytes:
    resolved = _validate_npz_payload(payload)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **resolved)
    value = buffer.getvalue()
    if not value:
        raise RuntimeError("Final evaluation NPZ serialization returned no bytes")
    return value


def _write_or_verify_npz(
    path: Path,
    payload: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    expected = _validate_npz_payload(payload)
    if os.path.lexists(path):
        observed, record = _read_npz(path, expected_keys=set(expected))
        if any(
            observed[name].dtype != expected[name].dtype
            or not np.array_equal(observed[name], expected[name])
            for name in expected
        ):
            raise ValueError(f"Existing final evaluation NPZ differs: {path}")
        return record
    try:
        record = _create_only_bytes(path, _npz_bytes(expected))
    except FileExistsError:
        observed, record = _read_npz(path, expected_keys=set(expected))
        if any(
            observed[name].dtype != expected[name].dtype
            or not np.array_equal(observed[name], expected[name])
            for name in expected
        ):
            raise ValueError(f"Concurrent final evaluation NPZ differs: {path}") from None
        return record
    observed, observed_record = _read_npz(path, expected_keys=set(expected))
    if observed_record != record or any(
        observed[name].dtype != expected[name].dtype
        or not np.array_equal(observed[name], expected[name])
        for name in expected
    ):
        raise RuntimeError("Final evaluation NPZ failed publication verification")
    return record


def _scalar_int(array: np.ndarray, name: str) -> int:
    if array.dtype != np.dtype(np.int64) or array.shape != ():
        raise ValueError(f"Final evaluation NPZ {name} scalar is invalid")
    return int(array.item())


def _task1_bootstrap_result_from_payload(
    payload: Mapping[str, np.ndarray],
) -> Task1BootstrapResult:
    required = {
        "task1_seed",
        "bootstrap_seed",
        "replicate_count",
        "attempts",
        "maximum_attempts",
        "metric_names",
        "class_order",
        "accuracy",
        "macro_f1",
        "per_class_f1",
        "recording_counts",
    }
    if set(payload) != required:
        raise ValueError("Task 1 bootstrap NPZ member inventory changed")
    replicate_count = _scalar_int(payload["replicate_count"], "replicate_count")
    if (
        _scalar_int(payload["task1_seed"], "task1_seed") != DETAIL_SEED
        or _scalar_int(payload["bootstrap_seed"], "bootstrap_seed") != BOOTSTRAP_SEED
        or replicate_count != BOOTSTRAP_REPLICATES
        or payload["metric_names"].dtype.kind != "U"
        or tuple(payload["metric_names"].tolist()) != TASK1_BOOTSTRAP_METRIC_NAMES
        or payload["class_order"].dtype.kind != "U"
        or tuple(payload["class_order"].tolist()) != tuple(LOCKED_TASK1_CLASS_ORDER)
    ):
        raise ValueError("Task 1 bootstrap NPZ fixed protocol changed")
    replicates = Task1BootstrapReplicates(
        accuracy=payload["accuracy"],
        macro_f1=payload["macro_f1"],
        per_class_f1=payload["per_class_f1"],
        recording_counts=payload["recording_counts"],
        replicate_count=replicate_count,
        attempts=_scalar_int(payload["attempts"], "attempts"),
        maximum_attempts=_scalar_int(payload["maximum_attempts"], "maximum_attempts"),
        task1_seed=DETAIL_SEED,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    def interval(values: np.ndarray) -> Task1PercentileInterval:
        bounds = np.percentile(values, [2.5, 97.5], method="linear")
        return Task1PercentileInterval(lower=float(bounds[0]), upper=float(bounds[1]))

    return Task1BootstrapResult(
        replicates=replicates,
        accuracy_interval=interval(replicates.accuracy),
        macro_f1_interval=interval(replicates.macro_f1),
        per_class_f1_intervals=tuple(
            interval(replicates.per_class_f1[:, index])
            for index in range(len(LOCKED_TASK1_CLASS_ORDER))
        ),
    )


def _metric_intervals(values: np.ndarray) -> MetricIntervals:
    bounds = np.percentile(values, [2.5, 97.5], axis=0, method="linear")
    if bounds.shape != (2, len(TASK2_METRIC_NAMES)) or not bool(np.all(np.isfinite(bounds))):
        raise ValueError("Task 2 bootstrap interval calculation failed")
    return MetricIntervals(
        **{
            name: Task2PercentileInterval(
                lower=float(bounds[0, index]),
                upper=float(bounds[1, index]),
            )
            for index, name in enumerate(TASK2_METRIC_NAMES)
        }
    )


def _task2_bootstrap_result_from_payload(
    payload: Mapping[str, np.ndarray],
    *,
    point_estimates: Any,
) -> SessionBootstrapResult:
    required = {
        "seed",
        "replicate_count",
        "metric_names",
        "species_scientific_names",
        "pooled",
        "per_species",
        "macro",
    }
    if set(payload) != required:
        raise ValueError("Task 2 bootstrap NPZ member inventory changed")
    replicate_count = _scalar_int(payload["replicate_count"], "replicate_count")
    if (
        _scalar_int(payload["seed"], "seed") != BOOTSTRAP_SEED
        or replicate_count != BOOTSTRAP_REPLICATES
        or payload["metric_names"].dtype.kind != "U"
        or tuple(payload["metric_names"].tolist()) != TASK2_METRIC_NAMES
        or payload["species_scientific_names"].dtype.kind != "U"
    ):
        raise ValueError("Task 2 bootstrap NPZ fixed protocol changed")
    species = tuple(payload["species_scientific_names"].tolist())
    replicates = Task2BootstrapReplicates(
        pooled=payload["pooled"],
        per_species=payload["per_species"],
        macro=payload["macro"],
        species_scientific_names=species,
        replicate_count=replicate_count,
        seed=BOOTSTRAP_SEED,
    )
    return SessionBootstrapResult(
        point_estimates=point_estimates,
        pooled_intervals=_metric_intervals(replicates.pooled),
        per_species_intervals=tuple(
            SpeciesMetricIntervals(
                species_scientific_name=name,
                intervals=_metric_intervals(replicates.per_species[:, index, :]),
            )
            for index, name in enumerate(species)
        ),
        macro_intervals=_metric_intervals(replicates.macro),
        replicates=replicates,
    )


def _task1_npz_keys() -> set[str]:
    return {
        "task1_seed",
        "bootstrap_seed",
        "replicate_count",
        "attempts",
        "maximum_attempts",
        "metric_names",
        "class_order",
        "accuracy",
        "macro_f1",
        "per_class_f1",
        "recording_counts",
    }


def _task2_npz_keys() -> set[str]:
    return {
        "seed",
        "replicate_count",
        "metric_names",
        "species_scientific_names",
        "pooled",
        "per_species",
        "macro",
    }


def _obtain_task1_bootstrap(
    path: Path,
    predictions: Sequence[RecordingPrediction],
    *,
    create: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if os.path.lexists(path):
        payload, record = _read_npz(path, expected_keys=_task1_npz_keys())
        bootstrap = session_cluster_bootstrap_seed37(
            predictions,
            task1_seed=DETAIL_SEED,
            replicate_count=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        expected = bootstrap.replicates.to_npz_payload()
        if any(
            payload[name].dtype != expected[name].dtype
            or not np.array_equal(payload[name], expected[name])
            for name in expected
        ):
            raise ValueError("Task 1 bootstrap archive is not bound to current seed 37 input")
        observed = _task1_bootstrap_result_from_payload(payload)
        if observed.to_record() != bootstrap.to_record():
            raise ValueError("Task 1 bootstrap intervals differ from current seed 37 input")
        return observed.to_record(), record
    if not create:
        raise ValueError("Task 1 bootstrap archive is missing")
    bootstrap = session_cluster_bootstrap_seed37(
        predictions,
        task1_seed=DETAIL_SEED,
        replicate_count=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    record = _write_or_verify_npz(path, bootstrap.replicates.to_npz_payload())
    payload, observed_record = _read_npz(path, expected_keys=_task1_npz_keys())
    observed = _task1_bootstrap_result_from_payload(payload)
    if observed_record != record or observed.to_record() != bootstrap.to_record():
        raise RuntimeError("Task 1 bootstrap archive differs from its intervals")
    return observed.to_record(), record


def _obtain_task2_bootstrap(
    path: Path,
    recordings: Sequence[ScoredRecording],
    threshold: float,
    *,
    create: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    point_estimates = evaluate_novelty_scores(recordings, threshold)
    if os.path.lexists(path):
        payload, record = _read_npz(path, expected_keys=_task2_npz_keys())
        bootstrap = session_cluster_bootstrap(
            recordings,
            threshold,
            replicate_count=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        expected = bootstrap.replicates.to_npz_payload()
        if any(
            payload[name].dtype != expected[name].dtype
            or not np.array_equal(payload[name], expected[name])
            for name in expected
        ):
            raise ValueError("Task 2 bootstrap archive is not bound to current seed 37 input")
        observed = _task2_bootstrap_result_from_payload(
            payload,
            point_estimates=point_estimates,
        )
        if observed.to_record() != bootstrap.to_record():
            raise ValueError("Task 2 bootstrap intervals differ from current seed 37 input")
        return observed.to_record(), record
    if not create:
        raise ValueError("Task 2 bootstrap archive is missing")
    bootstrap = session_cluster_bootstrap(
        recordings,
        threshold,
        replicate_count=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    record = _write_or_verify_npz(path, bootstrap.replicates.to_npz_payload())
    payload, observed_record = _read_npz(path, expected_keys=_task2_npz_keys())
    observed = _task2_bootstrap_result_from_payload(
        payload,
        point_estimates=point_estimates,
    )
    if observed_record != record or observed.to_record() != bootstrap.to_record():
        raise RuntimeError("Task 2 bootstrap archive differs from its intervals")
    return observed.to_record(), record


def _task2_metric_values(record: Mapping[str, Any], *, location: str) -> dict[str, float]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Task 2 {location} metric record is invalid")
    values: dict[str, float] = {}
    for name in TASK2_METRIC_NAMES:
        value = record.get(name)
        if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Task 2 {location} metric value is invalid")
        values[name] = value
    return values


def _task2_stability_record(
    stages: Mapping[int, Mapping[str, Any]],
    *,
    stream_name: str,
) -> dict[str, Any]:
    evaluations = {
        seed: stages[seed]["result"]["score_streams"][stream_name] for seed in SEED_ORDER
    }
    pooled = {
        seed: _task2_metric_values(evaluations[seed]["pooled"], location="pooled")
        for seed in SEED_ORDER
    }
    macro = {
        seed: _task2_metric_values(evaluations[seed]["macro"], location="macro")
        for seed in SEED_ORDER
    }
    species_names: tuple[str, ...] | None = None
    by_seed_species: dict[int, dict[str, Mapping[str, Any]]] = {}
    for seed in SEED_ORDER:
        per_species = evaluations[seed].get("per_species")
        if not isinstance(per_species, list):
            raise ValueError("Task 2 per-species metric inventory is invalid")
        names = tuple(item.get("species_scientific_name") for item in per_species)
        if (
            any(type(name) is not str for name in names)
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
        ):
            raise ValueError("Task 2 per-species metric ordering changed")
        if species_names is None:
            species_names = names
        elif names != species_names:
            raise ValueError("Task 2 per-species inventory differs across seeds")
        by_seed_species[seed] = {item["species_scientific_name"]: item for item in per_species}
    if species_names is None or len(species_names) != TASK2_UNKNOWN_SPECIES:
        raise ValueError("Task 2 per-species metric inventory is incomplete")

    def summaries(values: Mapping[int, Mapping[str, float]]) -> list[dict[str, Any]]:
        return [item.to_record() for item in summarize_across_seeds(values)]

    return {
        "seed_order": list(SEED_ORDER),
        "pooled": summaries(pooled),
        "per_species": [
            {
                "species_scientific_name": species,
                "metrics": summaries(
                    {
                        seed: _task2_metric_values(
                            by_seed_species[seed][species],
                            location=f"species {species}",
                        )
                        for seed in SEED_ORDER
                    }
                ),
            }
            for species in species_names
        ],
        "macro": summaries(macro),
    }


def _summary_directory() -> Path:
    return _stage_directory("summary")


def _validate_summary_entries(*, complete: bool) -> None:
    entries = _directory_entries(_summary_directory(), "Final evaluation summary")
    allowed = {
        TASK1_BOOTSTRAP_FILENAME,
        TASK2_RECONSTRUCTION_BOOTSTRAP_FILENAME,
        TASK2_LATENT_BOOTSTRAP_FILENAME,
        "result.json",
        "lock.json",
    }
    if not set(entries).issubset(allowed) or any(kind != "file" for kind in entries.values()):
        raise ValueError("Final evaluation summary contains unexpected entries")
    if complete and set(entries) != allowed:
        raise ValueError("Final evaluation summary evidence is incomplete")


def _summary_result_value(
    *,
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    bootstrap_records: Mapping[str, Mapping[str, Any]],
    bootstrap_intervals: Mapping[str, Mapping[str, Any]],
    gate_sha256: str,
    claim_sha256: str,
    source_fingerprint_sha256: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    task1_seed_metrics = {
        seed: {
            "accuracy": task1_stages[seed]["result"]["metrics"]["accuracy"],
            "macro_f1": task1_stages[seed]["result"]["metrics"]["macro_f1"],
        }
        for seed in SEED_ORDER
    }
    task1_stability = summarize_stability(task1_seed_metrics).to_record()
    stage_locks = {
        stage["stage_id"]: dict(stage["lock_artifact"])
        for stage in (*task1_stages.values(), *task2_stages.values())
    }
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "complete": True,
        "stage_id": "summary",
        "completed_at_utc": _require_utc_timestamp(completed_at_utc, "summary"),
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "seed_order": list(SEED_ORDER),
        "detail_seed": DETAIL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "stage_locks": stage_locks,
        "task1": {
            "stability": task1_stability,
            "seed_37_metrics": task1_stages[DETAIL_SEED]["result"]["metrics"],
            "seed_37_bootstrap": dict(bootstrap_intervals["task1"]),
            "bootstrap_archive": dict(bootstrap_records["task1"]),
        },
        "task2": {
            "reconstruction": {
                "score_name": RECONSTRUCTION_SCORE_NAME,
                "stability": _task2_stability_record(
                    task2_stages,
                    stream_name="reconstruction",
                ),
                "seed_37_point_estimates": task2_stages[DETAIL_SEED]["result"]["score_streams"][
                    "reconstruction"
                ],
                "seed_37_bootstrap": dict(bootstrap_intervals["task2_reconstruction"]),
                "bootstrap_archive": dict(bootstrap_records["task2_reconstruction"]),
            },
            "latent": {
                "score_name": LATENT_SCORE_NAME,
                "stability": _task2_stability_record(task2_stages, stream_name="latent"),
                "seed_37_point_estimates": task2_stages[DETAIL_SEED]["result"]["score_streams"][
                    "latent"
                ],
                "seed_37_bootstrap": dict(bootstrap_intervals["task2_latent"]),
                "bootstrap_archive": dict(bootstrap_records["task2_latent"]),
            },
        },
    }


def _summary_lock_value(
    *,
    result_record: Mapping[str, Any],
    stage_locks: Mapping[str, Mapping[str, Any]],
    bootstrap_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "stage_id": "summary",
        "result": dict(result_record),
        "stage_locks": {name: dict(record) for name, record in stage_locks.items()},
        "bootstrap_archives": {name: dict(record) for name, record in bootstrap_records.items()},
    }


def _summary_bootstrap_evidence(
    *,
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    create: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    directory = _summary_directory()
    task1_intervals, task1_record = _obtain_task1_bootstrap(
        directory / TASK1_BOOTSTRAP_FILENAME,
        task1_stages[DETAIL_SEED]["predictions"],
        create=create,
    )
    reconstruction_stage = task2_stages[DETAIL_SEED]
    reconstruction_threshold = reconstruction_stage["result"]["thresholds"]["reconstruction"][
        "value"
    ]
    reconstruction_intervals, reconstruction_record = _obtain_task2_bootstrap(
        directory / TASK2_RECONSTRUCTION_BOOTSTRAP_FILENAME,
        reconstruction_stage["score_streams"]["reconstruction"],
        reconstruction_threshold,
        create=create,
    )
    latent_threshold = reconstruction_stage["result"]["thresholds"]["latent"]["value"]
    latent_intervals, latent_record = _obtain_task2_bootstrap(
        directory / TASK2_LATENT_BOOTSTRAP_FILENAME,
        reconstruction_stage["score_streams"]["latent"],
        latent_threshold,
        create=create,
    )
    if (
        task1_intervals.get("task1_seed") != DETAIL_SEED
        or reconstruction_intervals.get("point_estimates")
        != reconstruction_stage["result"]["score_streams"]["reconstruction"]
        or latent_intervals.get("point_estimates")
        != reconstruction_stage["result"]["score_streams"]["latent"]
    ):
        raise ValueError("Final evaluation bootstrap point evidence changed")
    return (
        {
            "task1": task1_record,
            "task2_reconstruction": reconstruction_record,
            "task2_latent": latent_record,
        },
        {
            "task1": task1_intervals,
            "task2_reconstruction": reconstruction_intervals,
            "task2_latent": latent_intervals,
        },
    )


def _verify_summary(
    *,
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    gate: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
    expected_result: Mapping[str, Any] | None = None,
    expected_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_summary_entries(complete=True)
    bootstrap_records, bootstrap_intervals = _summary_bootstrap_evidence(
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        create=False,
    )
    result_path = _summary_directory() / "result.json"
    observed_result, result_record = _read_json(result_path, boundary=_attempt_directory())
    source_sha256 = _require_sha256(
        gate["shared_identity"]["source_fingerprint_sha256"],
        "Final evaluation summary source fingerprint",
    )
    result = _summary_result_value(
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        bootstrap_records=bootstrap_records,
        bootstrap_intervals=bootstrap_intervals,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        source_fingerprint_sha256=source_sha256,
        completed_at_utc=observed_result.get("completed_at_utc"),
    )
    if observed_result != result or (
        expected_result is not None and dict(expected_result) != result
    ):
        raise ValueError("Final evaluation summary result changed")
    stage_locks = {
        stage["stage_id"]: stage["lock_artifact"]
        for stage in (*task1_stages.values(), *task2_stages.values())
    }
    lock = _summary_lock_value(
        result_record=result_record,
        stage_locks=stage_locks,
        bootstrap_records=bootstrap_records,
    )
    observed_lock, lock_record = _read_json(
        _summary_directory() / "lock.json",
        boundary=_attempt_directory(),
    )
    if observed_lock != lock or (expected_lock is not None and dict(expected_lock) != lock):
        raise ValueError("Final evaluation summary lock changed")
    return {
        "stage_id": "summary",
        "result": result,
        "result_artifact": result_record,
        "lock_artifact": lock_record,
        "bootstrap_archives": bootstrap_records,
    }


def _run_summary(
    *,
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    gate: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> dict[str, Any]:
    _secure_ensure_directory(_summary_directory())
    _validate_summary_entries(complete=False)
    result_path = _summary_directory() / "result.json"
    lock_path = _summary_directory() / "lock.json"
    if os.path.lexists(lock_path):
        if not os.path.lexists(result_path):
            raise ValueError("Final evaluation summary lock exists without its result")
        return _verify_summary(
            task1_stages=task1_stages,
            task2_stages=task2_stages,
            gate=gate,
            gate_sha256=gate_sha256,
            claim_sha256=claim_sha256,
        )
    _assert_gate_current(gate, gate_sha256, full=False)
    bootstrap_records, bootstrap_intervals = _summary_bootstrap_evidence(
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        create=True,
    )
    source_sha256 = _require_sha256(
        gate["shared_identity"]["source_fingerprint_sha256"],
        "Final evaluation summary source fingerprint",
    )
    result = _summary_result_value(
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        bootstrap_records=bootstrap_records,
        bootstrap_intervals=bootstrap_intervals,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        source_fingerprint_sha256=source_sha256,
        completed_at_utc=_completion_time(result_path),
    )
    _assert_gate_current(gate, gate_sha256, full=False)
    result_record = _write_or_verify_json(result_path, result)
    stage_locks = {
        stage["stage_id"]: stage["lock_artifact"]
        for stage in (*task1_stages.values(), *task2_stages.values())
    }
    lock = _summary_lock_value(
        result_record=result_record,
        stage_locks=stage_locks,
        bootstrap_records=bootstrap_records,
    )
    _assert_gate_current(gate, gate_sha256, full=False)
    _write_or_verify_json(lock_path, lock)
    return _verify_summary(
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        gate=gate,
        gate_sha256=gate_sha256,
        claim_sha256=claim_sha256,
        expected_result=result,
        expected_lock=lock,
    )


def _reader_metadata(
    data: Any,
    *,
    source_role: str,
) -> dict[str, FinalRecordingMetadata]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in data.iter_metadata():
        if not isinstance(row, Mapping) or type(row.get("recording_id")) is not str:
            raise ValueError("Final gated reader metadata row is invalid")
        grouped.setdefault(row["recording_id"], []).append(row)
    if set(grouped) != set(data.recording_ids) or len(grouped) != len(data.recording_ids):
        raise ValueError("Final gated reader metadata identities are incomplete")
    known_scientific = dict(KNOWN_COMMON_TO_SCIENTIFIC)
    result: dict[str, FinalRecordingMetadata] = {}
    for recording_id, rows in grouped.items():
        sessions = {row.get("session_group") for row in rows}
        common_names = {row.get("species_common_name") for row in rows}
        clip_ids = tuple(sorted(str(row.get("clip_id")) for row in rows))
        if len(sessions) != 1 or len(common_names) != 1 or None in sessions | common_names:
            raise ValueError("Final gated reader recording metadata changed within a recording")
        common_name = next(iter(common_names))
        class_index: int | None = None
        if source_role == FINAL_KNOWN_TEST_ROLE:
            class_values = {row.get("class_index") for row in rows}
            if len(class_values) != 1:
                raise ValueError("Final known reader class changes within a recording")
            try:
                class_index = int(next(iter(class_values)))
            except (TypeError, ValueError) as exc:
                raise ValueError("Final known reader class index is invalid") from exc
            scientific_name = known_scientific.get(common_name)
        else:
            scientific_values = {row.get("species_scientific_name") for row in rows}
            if len(scientific_values) != 1:
                raise ValueError("Final unknown reader species changes within a recording")
            scientific_name = next(iter(scientific_values))
        result[recording_id] = FinalRecordingMetadata(
            source_role=source_role,
            recording_id=recording_id,
            session_group=next(iter(sessions)),
            species_common_name=common_name,
            species_scientific_name=scientific_name,
            class_index=class_index,
            clip_ids=clip_ids,
        )
    return result


def _verify_reader_feature_bytes(data: Any) -> None:
    observed_clips = 0
    for recording_id in data.recording_ids:
        features, rows = data.get_recording(recording_id)
        if (
            not isinstance(features, np.ndarray)
            or features.dtype != np.float32
            or features.ndim != 4
            or type(rows) is not tuple
            or features.shape[0] != len(rows)
            or len(rows) <= 0
            or not bool(np.all(np.isfinite(features)))
        ):
            raise ValueError("Final gated reader feature bytes are invalid")
        observed_clips += len(rows)
        del features
    if observed_clips != len(data):
        raise ValueError("Final gated reader feature inventory is incomplete")


def _validate_task1_reader_evidence(stage: Mapping[str, Any], known_data: Any) -> None:
    reader_ids = tuple(known_data.recording_ids)
    result_ids = tuple(stage["result"]["recording_ids"])
    if (
        len(known_data) != final_data.KNOWN_TEST_ENERGY_CLIPS
        or len(reader_ids) != TASK1_FINAL_RECORDINGS
        or len(set(reader_ids)) != len(reader_ids)
        or len(result_ids) != len(reader_ids)
        or set(result_ids) != set(reader_ids)
    ):
        raise ValueError("Task 1 completed shards differ from the gated reader inventory")
    expected_metadata = _reader_metadata(known_data, source_role=FINAL_KNOWN_TEST_ROLE)
    predictions = {item.recording_id: item for item in stage["predictions"]}
    observed_metadata = stage.get("metadata")
    if (
        not isinstance(observed_metadata, Mapping)
        or set(predictions) != set(expected_metadata)
        or set(observed_metadata) != set(expected_metadata)
        or any(
            observed_metadata[recording_id].to_record() != metadata.to_record()
            for recording_id, metadata in expected_metadata.items()
        )
    ):
        raise ValueError("Task 1 completed shard labels differ from gated reader metadata")


def _validate_task2_reader_evidence(
    stage: Mapping[str, Any],
    known_data: Any,
    unknown_data: Any,
) -> None:
    known_ids = tuple(known_data.recording_ids)
    unknown_ids = tuple(unknown_data.recording_ids)
    result_known = tuple(stage["result"]["known_test_recording_ids"])
    result_unknown = tuple(stage["result"]["unknown_recording_ids"])
    if (
        len(known_data) != TASK2_KNOWN_FINAL_CLIPS
        or len(unknown_data) != TASK2_UNKNOWN_FINAL_CLIPS
        or len(known_ids) != TASK2_KNOWN_FINAL_RECORDINGS
        or len(unknown_ids) != TASK2_UNKNOWN_FINAL_RECORDINGS
        or len(set(known_ids)) != len(known_ids)
        or len(set(unknown_ids)) != len(unknown_ids)
        or set(known_ids).intersection(unknown_ids)
        or len(result_known) != len(known_ids)
        or set(result_known) != set(known_ids)
        or len(result_unknown) != len(unknown_ids)
        or set(result_unknown) != set(unknown_ids)
    ):
        raise ValueError("Task 2 completed shards differ from the gated reader inventories")
    expected_known = _reader_metadata(known_data, source_role=FINAL_KNOWN_TEST_ROLE)
    expected_unknown = _reader_metadata(unknown_data, source_role=FINAL_UNKNOWN_ROLE)
    observed_known = stage.get("known_items")
    observed_unknown = stage.get("unknown_items")
    if not isinstance(observed_known, Mapping) or not isinstance(observed_unknown, Mapping):
        raise ValueError("Task 2 completed stage lacks verified shard metadata")
    if set(observed_known) != set(expected_known) or set(observed_unknown) != set(expected_unknown):
        raise ValueError("Task 2 completed shard metadata inventory changed")
    for observed, expected in (
        (observed_known, expected_known),
        (observed_unknown, expected_unknown),
    ):
        if any(
            observed[recording_id][1].to_record() != metadata.to_record()
            for recording_id, metadata in expected.items()
        ):
            raise ValueError("Task 2 completed shard metadata differs from its gated reader")


def _failure_artifacts(
    *,
    gate_sha256: str,
    claim_sha256: str,
    source_fingerprint_sha256: str,
) -> tuple[dict[str, Any], ...]:
    directory = _attempt_directory() / FAILURE_DIRECTORY_NAME
    if not os.path.lexists(directory):
        return ()
    entries = _directory_entries(directory, "Final evaluation failure diagnostics")
    records: list[dict[str, Any]] = []
    fields = {
        "schema_version",
        "attempt_id",
        "failed_at_utc",
        "exception_type",
        "exception_message",
        "command",
        "gate_sha256",
        "claim_sha256",
        "source_fingerprint_sha256",
        "completed_evidence_unchanged",
    }
    for name in sorted(entries):
        if entries[name] != "file" or not name.startswith("failure_") or not name.endswith(".json"):
            raise ValueError("Final evaluation failure directory contains an unexpected entry")
        value, record = _read_json(directory / name, boundary=_attempt_directory())
        if (
            set(value) != fields
            or value.get("schema_version") != FINAL_EVALUATION_SCHEMA_VERSION
            or value.get("attempt_id") != FINAL_EVALUATION_ATTEMPT_ID
            or type(value.get("exception_type")) is not str
            or not value["exception_type"]
            or type(value.get("exception_message")) is not str
            or value.get("completed_evidence_unchanged") is not True
            or value.get("gate_sha256") != gate_sha256
            or value.get("claim_sha256") != claim_sha256
            or value.get("source_fingerprint_sha256") != source_fingerprint_sha256
        ):
            raise ValueError("Final evaluation failure diagnostic is invalid")
        _require_utc_timestamp(value.get("failed_at_utc"), "failure")
        _validate_command(value.get("command"))
        records.append(record)
    return tuple(records)


def _write_failure_diagnostic(
    exc: BaseException,
    *,
    command: Sequence[str],
    gate: Mapping[str, Any],
    gate_sha256: str,
    claim_sha256: str,
) -> None:
    if os.path.lexists(_attempt_directory() / "lock.json"):
        return
    directory = _secure_ensure_directory(_attempt_directory() / FAILURE_DIRECTORY_NAME)
    failed_at = _utc_now()
    value = {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "attempt_id": FINAL_EVALUATION_ATTEMPT_ID,
        "failed_at_utc": failed_at,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "command": list(command),
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "source_fingerprint_sha256": gate["shared_identity"]["source_fingerprint_sha256"],
        "completed_evidence_unchanged": True,
    }
    safe_time = failed_at.replace(":", "").replace("+", "_").replace(".", "_")
    path = directory / f"failure_{safe_time}_{secrets.token_hex(8)}.json"
    _write_json_create_only(path, value)


def _final_result_value(
    *,
    claim: Mapping[str, Any],
    claim_sha256: str,
    gate: Mapping[str, Any],
    gate_sha256: str,
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
    command: Sequence[str],
    completed_at_utc: str,
) -> dict[str, Any]:
    stage_results = {
        stage["stage_id"]: dict(stage["result_artifact"])
        for stage in (*task1_stages.values(), *task2_stages.values(), summary)
    }
    if tuple(stage_results) != STAGE_ORDER:
        raise ValueError("Final evaluation stage result ordering changed")
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "attempt_id": FINAL_EVALUATION_ATTEMPT_ID,
        "complete": True,
        "claimed_at_utc": claim["claimed_at_utc"],
        "completed_at_utc": _require_utc_timestamp(completed_at_utc, "final evaluation"),
        "gate_sha256": gate_sha256,
        "claim_sha256": claim_sha256,
        "source_fingerprint_sha256": gate["shared_identity"]["source_fingerprint_sha256"],
        "seed_order": list(SEED_ORDER),
        "stage_order": list(STAGE_ORDER),
        "command": list(command),
        "cache_locks": gate["cache_locks"],
        "stage_results": stage_results,
        "task1_summary": summary["result"]["task1"],
        "task2_summary": summary["result"]["task2"],
    }


def _final_lock_value(
    *,
    gate_record: Mapping[str, Any],
    gate_lock_record: Mapping[str, Any],
    claim_record: Mapping[str, Any],
    result_record: Mapping[str, Any],
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_locks = {
        stage["stage_id"]: dict(stage["lock_artifact"])
        for stage in (*task1_stages.values(), *task2_stages.values(), summary)
    }
    return {
        "schema_version": FINAL_EVALUATION_SCHEMA_VERSION,
        "attempt_id": FINAL_EVALUATION_ATTEMPT_ID,
        "gate": dict(gate_record),
        "gate_lock": dict(gate_lock_record),
        "claim": dict(claim_record),
        "result": dict(result_record),
        "stage_locks": stage_locks,
        "failure_diagnostics": [dict(record) for record in failures],
    }


def _verify_final_artifacts(
    *,
    claim: Mapping[str, Any],
    claim_record: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_record: Mapping[str, Any],
    gate_lock_record: Mapping[str, Any],
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
    expected_result: Mapping[str, Any] | None = None,
    expected_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_attempt_entries(complete=True)
    result_path = _attempt_directory() / "result.json"
    observed_result, result_record = _read_json(result_path, boundary=_attempt_directory())
    command = _validate_command(observed_result.get("command"))
    result = _final_result_value(
        claim=claim,
        claim_sha256=claim_record["sha256"],
        gate=gate,
        gate_sha256=gate_record["sha256"],
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        summary=summary,
        command=command,
        completed_at_utc=observed_result.get("completed_at_utc"),
    )
    if observed_result != result or (
        expected_result is not None and dict(expected_result) != result
    ):
        raise ValueError("Final evaluation result changed")
    failures = _failure_artifacts(
        gate_sha256=gate_record["sha256"],
        claim_sha256=claim_record["sha256"],
        source_fingerprint_sha256=gate["shared_identity"]["source_fingerprint_sha256"],
    )
    lock = _final_lock_value(
        gate_record=gate_record,
        gate_lock_record=gate_lock_record,
        claim_record=claim_record,
        result_record=result_record,
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        summary=summary,
        failures=failures,
    )
    observed_lock, lock_record = _read_json(
        _attempt_directory() / "lock.json",
        boundary=_attempt_directory(),
    )
    if observed_lock != lock or (expected_lock is not None and dict(expected_lock) != lock):
        raise ValueError("Final evaluation completion lock changed")
    return {
        **result,
        "result_artifact": result_record,
        "completion_lock_artifact": lock_record,
    }


def _run_final_publication(
    *,
    claim: Mapping[str, Any],
    claim_record: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_record: Mapping[str, Any],
    gate_lock_record: Mapping[str, Any],
    task1_stages: Mapping[int, Mapping[str, Any]],
    task2_stages: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    result_path = _attempt_directory() / "result.json"
    lock_path = _attempt_directory() / "lock.json"
    if os.path.lexists(lock_path):
        if not os.path.lexists(result_path):
            raise ValueError("Final evaluation lock exists without its result")
        return _verify_final_artifacts(
            claim=claim,
            claim_record=claim_record,
            gate=gate,
            gate_record=gate_record,
            gate_lock_record=gate_lock_record,
            task1_stages=task1_stages,
            task2_stages=task2_stages,
            summary=summary,
        )
    resolved_command = tuple(command)
    if os.path.lexists(result_path):
        existing, _ = _read_json(result_path, boundary=_attempt_directory())
        resolved_command = _validate_command(existing.get("command"))
    result = _final_result_value(
        claim=claim,
        claim_sha256=claim_record["sha256"],
        gate=gate,
        gate_sha256=gate_record["sha256"],
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        summary=summary,
        command=resolved_command,
        completed_at_utc=_completion_time(result_path),
    )
    _assert_gate_current(gate, gate_record["sha256"], full=False)
    result_record = _write_or_verify_json(result_path, result)
    failures = _failure_artifacts(
        gate_sha256=gate_record["sha256"],
        claim_sha256=claim_record["sha256"],
        source_fingerprint_sha256=gate["shared_identity"]["source_fingerprint_sha256"],
    )
    lock = _final_lock_value(
        gate_record=gate_record,
        gate_lock_record=gate_lock_record,
        claim_record=claim_record,
        result_record=result_record,
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        summary=summary,
        failures=failures,
    )
    _assert_gate_current(gate, gate_record["sha256"], full=True)
    _write_or_verify_json(lock_path, lock)
    _assert_gate_current(gate, gate_record["sha256"], full=True)
    return _verify_final_artifacts(
        claim=claim,
        claim_record=claim_record,
        gate=gate,
        gate_record=gate_record,
        gate_lock_record=gate_lock_record,
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        summary=summary,
        expected_result=result,
        expected_lock=lock,
    )


def _authorization_from_existing(
    gate_record: Mapping[str, Any],
    gate_lock_record: Mapping[str, Any],
    claim_record: Mapping[str, Any],
) -> FinalEvaluationAuthorization:
    return FinalEvaluationAuthorization(
        gate_sha256=gate_record["sha256"],
        gate_lock_sha256=gate_lock_record["sha256"],
        claim_sha256=claim_record["sha256"],
        attempt_directory=_attempt_directory(),
    )


def _verify_completed_evaluation_locked(
    *,
    verified_gate: Mapping[str, Any],
    ffmpeg: str | Path | None,
) -> dict[str, Any]:
    gate, claim, claim_record, gate_record = _verify_existing_claim(verified_gate)
    _, gate_lock_record = _gate_artifacts(verified_gate)
    _assert_gate_current(gate, gate_record["sha256"], full=False)
    _validate_attempt_entries(complete=True)
    authorization = _authorization_from_existing(gate_record, gate_lock_record, claim_record)
    known_data = open_final_known_test_data(authorization, ffmpeg=ffmpeg)
    unknown_data = open_final_unknown_data(authorization, ffmpeg=ffmpeg)
    _verify_reader_feature_bytes(known_data)
    _verify_reader_feature_bytes(unknown_data)
    task1_runs = _run_inventory(gate, "task1")
    task2_runs = _run_inventory(gate, "task2")
    task1_stages: dict[int, dict[str, Any]] = {}
    task2_stages: dict[int, dict[str, Any]] = {}
    for run in task1_runs:
        _assert_gate_current(gate, gate_record["sha256"], full=False)
        stage = _verify_task1_stage(
            stage_id=f"task1_seed_{run['seed']}",
            run=run,
            gate_sha256=gate_record["sha256"],
            claim_sha256=claim_record["sha256"],
        )
        _validate_task1_reader_evidence(stage, known_data)
        task1_stages[run["seed"]] = stage
    for run in task2_runs:
        _assert_gate_current(gate, gate_record["sha256"], full=False)
        stage = _verify_task2_stage(
            stage_id=f"task2_seed_{run['seed']}",
            run=run,
            gate_sha256=gate_record["sha256"],
            claim_sha256=claim_record["sha256"],
        )
        _validate_task2_reader_evidence(stage, known_data, unknown_data)
        task2_stages[run["seed"]] = stage
    _assert_gate_current(gate, gate_record["sha256"], full=False)
    summary = _verify_summary(
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        gate=gate,
        gate_sha256=gate_record["sha256"],
        claim_sha256=claim_record["sha256"],
    )
    _assert_gate_current(gate, gate_record["sha256"], full=False)
    result = _verify_final_artifacts(
        claim=claim,
        claim_record=claim_record,
        gate=gate,
        gate_record=gate_record,
        gate_lock_record=gate_lock_record,
        task1_stages=task1_stages,
        task2_stages=task2_stages,
        summary=summary,
    )
    _assert_gate_current(gate, gate_record["sha256"], full=True)
    return result


def run_final_evaluation(
    *,
    ffmpeg: str | Path | None = None,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    """Run or resume the single sealed final evaluation attempt."""

    resolved_command = _validate_command(command)
    if not resolved_command:
        raise ValueError("Final evaluation command provenance cannot be empty")
    claimed = False
    gate: dict[str, Any] | None = None
    gate_record: dict[str, Any] | None = None
    claim_record: dict[str, Any] | None = None
    with _transaction_lock(exclusive=True):
        verified_gate = verify_final_evaluation_gate()
        gate = _validate_gate_value(verified_gate.get("gate"))
        gate_record, gate_lock_record = _gate_artifacts(verified_gate)
        _assert_gate_current(gate, gate_record["sha256"], full=False)
        device = _prepare_production_runtime()
        _preflight_models(gate, device)
        refreshed_gate = verify_final_evaluation_gate()
        refreshed_record, refreshed_lock_record = _gate_artifacts(refreshed_gate)
        if (
            refreshed_gate.get("gate") != gate
            or refreshed_record != gate_record
            or refreshed_lock_record != gate_lock_record
        ):
            raise PermissionError("Final evaluation gate changed during model preflight")
        _assert_gate_current(gate, gate_record["sha256"], full=False)
        authorization, gate, claim, claim_record = _claim_after_gate(refreshed_gate)
        claimed = True
        try:
            _validate_attempt_entries(complete=False)
            known_data = open_final_known_test_data(authorization, ffmpeg=ffmpeg)
            unknown_data = open_final_unknown_data(authorization, ffmpeg=ffmpeg)
            task1_runs = _run_inventory(gate, "task1")
            task2_runs = _run_inventory(gate, "task2")
            task1_stages: dict[int, dict[str, Any]] = {}
            task2_stages: dict[int, dict[str, Any]] = {}
            for run in task1_runs:
                _assert_gate_current(gate, gate_record["sha256"], full=False)
                stage = _run_task1_stage(
                    run=run,
                    data=known_data,
                    device=device,
                    gate_sha256=gate_record["sha256"],
                    claim_sha256=claim_record["sha256"],
                )
                _validate_task1_reader_evidence(stage, known_data)
                task1_stages[run["seed"]] = stage
            for run in task2_runs:
                _assert_gate_current(gate, gate_record["sha256"], full=False)
                stage = _run_task2_stage(
                    run=run,
                    known_data=known_data,
                    unknown_data=unknown_data,
                    device=device,
                    gate_sha256=gate_record["sha256"],
                    claim_sha256=claim_record["sha256"],
                )
                _validate_task2_reader_evidence(stage, known_data, unknown_data)
                task2_stages[run["seed"]] = stage
            _assert_gate_current(gate, gate_record["sha256"], full=False)
            summary = _run_summary(
                task1_stages=task1_stages,
                task2_stages=task2_stages,
                gate=gate,
                gate_sha256=gate_record["sha256"],
                claim_sha256=claim_record["sha256"],
            )
            _assert_gate_current(gate, gate_record["sha256"], full=False)
            return _run_final_publication(
                claim=claim,
                claim_record=claim_record,
                gate=gate,
                gate_record=gate_record,
                gate_lock_record=gate_lock_record,
                task1_stages=task1_stages,
                task2_stages=task2_stages,
                summary=summary,
                command=resolved_command,
            )
        except BaseException as exc:
            if (
                claimed
                and gate is not None
                and gate_record is not None
                and claim_record is not None
            ):
                try:
                    _write_failure_diagnostic(
                        exc,
                        command=resolved_command,
                        gate=gate,
                        gate_sha256=gate_record["sha256"],
                        claim_sha256=claim_record["sha256"],
                    )
                except Exception as diagnostic_exc:
                    exc.add_note(
                        "Final evaluation diagnostic publication also failed: "
                        f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                    )
            raise


def verify_final_evaluation() -> dict[str, Any]:
    """Recursively verify completed final evidence without model inference or refitting."""

    with _transaction_lock(exclusive=False):
        verified_gate = verify_final_evaluation_gate()
        return _verify_completed_evaluation_locked(
            verified_gate=verified_gate,
            ffmpeg=None,
        )
