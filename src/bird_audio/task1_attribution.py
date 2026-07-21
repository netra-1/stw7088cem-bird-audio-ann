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
from contextlib import suppress
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional

from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.final_evaluation import verify_final_evaluation
from bird_audio.final_evaluation_data import (
    FINAL_EVALUATION_ATTEMPT_DIRECTORY,
    KNOWN_CACHE_LOCK_SHA256,
    KNOWN_TEST_RECORDINGS,
    FinalEvaluationAuthorization,
    claim_final_evaluation_attempt,
    open_final_known_test_data,
)
from bird_audio.final_evaluation_gate import verify_final_evaluation_gate
from bird_audio.final_evaluation_inference import FINAL_KNOWN_TEST_ROLE
from bird_audio.paths import PROJECT_ROOT
from bird_audio.provenance import source_fingerprint
from bird_audio.task1_training import load_locked_task1_best_model
from bird_audio.training_batching import to_efficientnet_batch

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

ATTRIBUTION_SCHEMA_VERSION = "1.0"
ATTRIBUTION_ID = "task1_attribution_v2"
ATTRIBUTION_ROOT = PROJECT_ROOT / "report_assets" / ATTRIBUTION_ID
DETAIL_SEED = 37
SELECTIONS_PER_STRATUM = 3
SELECTION_ORDER_DOMAIN = "task1-attribution-v1:stable-sha256-order"
FINAL_CONVOLUTIONAL_TARGET = "features[8]"
TIME_SECONDS = (0.0, 3.0)
MEL_FREQUENCY_TICKS_KHZ = (0.15, 1.0, 2.0, 4.0, 8.0, 14.0)
FIGURE_DPI = 120
OVERLAY_ALPHA = 0.45

_SHA256_LENGTH = 64
_SELECTION_FILENAME = "selection.json"
_SELECTION_RECORD_FILENAME = "selection.record.json"
_MANIFEST_FILENAME = "manifest.json"
_LOCK_FILENAME = "lock.json"
_FIXED_FILES = {
    _SELECTION_FILENAME,
    _SELECTION_RECORD_FILENAME,
    _MANIFEST_FILENAME,
    _LOCK_FILENAME,
}
_SHARD_FIELDS = {
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


def _json_bytes(value: Any) -> bytes:
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
        raise ValueError("Task 1 attribution JSON is not canonicalizable") from exc


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _project_relative(path: str | Path) -> tuple[Path, tuple[str, ...]]:
    candidate = _absolute(path)
    project = _absolute(PROJECT_ROOT)
    try:
        relative = candidate.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"Task 1 attribution path leaves the project: {candidate}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Task 1 attribution path is not canonical: {candidate}")
    return candidate, relative.parts


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Task 1 attribution requires O_NOFOLLOW")
    if not isinstance(directory, int) or directory == 0:
        raise RuntimeError("Task 1 attribution requires O_DIRECTORY")
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | directory


def _open_project_root() -> int:
    candidate = _absolute(PROJECT_ROOT)
    descriptor = os.open("/", _directory_flags())
    try:
        for part in candidate.parts[1:]:
            if part in {"", ".", ".."}:
                raise PermissionError("Task 1 attribution project path is invalid")
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise PermissionError("Task 1 attribution project path changed type")
            os.close(descriptor)
            descriptor = child
    except (OSError, PermissionError) as exc:
        os.close(descriptor)
        raise PermissionError("Task 1 attribution cannot safely open the project root") from exc
    return descriptor


def _open_directory(path: str | Path) -> int:
    candidate, parts = _project_relative(path)
    descriptor = _open_project_root()
    try:
        for part in parts:
            try:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise PermissionError(
                    f"Task 1 attribution cannot safely open directory: {candidate}"
                ) from exc
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise PermissionError("Task 1 attribution directory changed type")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _ensure_directory(path: str | Path) -> Path:
    candidate, parts = _project_relative(path)
    descriptor = _open_project_root()
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise PermissionError(
                    f"Task 1 attribution directory is unsafe: {candidate}"
                ) from exc
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise PermissionError("Task 1 attribution directory changed type")
            os.close(descriptor)
            descriptor = child
        return candidate
    finally:
        os.close(descriptor)


def _open_parent(path: str | Path) -> tuple[int, str, Path]:
    candidate, _ = _project_relative(path)
    return _open_directory(candidate.parent), candidate.name, candidate


def _internal_filename(path: str | Path) -> str:
    value = os.fspath(path)
    if type(value) is not str or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("Task 1 attribution internal filename is invalid")
    return value


def _snapshot(
    path: str | Path,
    *,
    internal: bool = False,
    directory_descriptor: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if directory_descriptor is None:
        parent, name, candidate = _open_parent(path)
        record_path = candidate.name if internal else str(candidate)
    else:
        if not internal:
            raise ValueError("Descriptor-relative attribution reads must be internal")
        name = _internal_filename(path)
        parent = os.dup(directory_descriptor)
        candidate = ATTRIBUTION_ROOT / name
        record_path = name
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow,
                dir_fd=parent,
            )
        except OSError as exc:
            raise PermissionError(f"Task 1 attribution cannot safely read: {candidate}") from exc
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or (internal and status.st_nlink != 1):
                raise PermissionError("Task 1 attribution artifact is not a private regular file")
            chunks: list[bytes] = []
            offset = 0
            while offset < status.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1024 * 1024, status.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise PermissionError("Task 1 attribution artifact ended while being read")
                chunks.append(chunk)
                offset += len(chunk)
            payload = b"".join(chunks)
            final_status = os.fstat(descriptor)
            try:
                leaf_status = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as exc:
                raise PermissionError(
                    "Task 1 attribution artifact name changed while being read"
                ) from exc
            if (
                not stat.S_ISREG(final_status.st_mode)
                or final_status.st_dev != status.st_dev
                or final_status.st_ino != status.st_ino
                or final_status.st_size != len(payload)
                or final_status.st_nlink != status.st_nlink
                or (internal and final_status.st_nlink != 1)
                or final_status.st_mtime_ns != status.st_mtime_ns
                or final_status.st_ctime_ns != status.st_ctime_ns
                or not stat.S_ISREG(leaf_status.st_mode)
                or leaf_status.st_dev != final_status.st_dev
                or leaf_status.st_ino != final_status.st_ino
                or leaf_status.st_size != final_status.st_size
                or leaf_status.st_nlink != final_status.st_nlink
                or leaf_status.st_mtime_ns != final_status.st_mtime_ns
                or leaf_status.st_ctime_ns != final_status.st_ctime_ns
            ):
                raise PermissionError("Task 1 attribution artifact changed while being read")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    return payload, {
        "path": record_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _artifact_record(
    path: str | Path,
    *,
    internal: bool = False,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    return _snapshot(
        path,
        internal=internal,
        directory_descriptor=directory_descriptor,
    )[1]


def _read_json(
    path: str | Path,
    *,
    internal: bool = False,
    directory_descriptor: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _snapshot(
        path,
        internal=internal,
        directory_descriptor=directory_descriptor,
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Task 1 attribution JSON is invalid") from exc
    if not isinstance(value, dict) or _json_bytes(value) != payload:
        raise ValueError("Task 1 attribution JSON is not canonical")
    return value, record


def _create_only(
    path: str | Path,
    payload: bytes,
    *,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    if not payload:
        raise ValueError("Task 1 attribution cannot publish an empty artifact")
    if directory_descriptor is None:
        parent, name, candidate = _open_parent(path)
    else:
        name = _internal_filename(path)
        parent = os.dup(directory_descriptor)
        candidate = ATTRIBUTION_ROOT / name
    temporary = f".{name}.{secrets.token_hex(16)}.tmp"
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | no_follow
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise RuntimeError("Task 1 attribution publication made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
        os.fsync(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent)
        os.close(parent)
    observed, record = _snapshot(
        name if directory_descriptor is not None else candidate,
        internal=True,
        directory_descriptor=directory_descriptor,
    )
    if observed != payload:
        raise RuntimeError("Task 1 attribution artifact changed during publication")
    return record


def _publish_or_verify(
    path: str | Path,
    payload: bytes,
    *,
    directory_descriptor: int | None = None,
) -> tuple[dict[str, Any], bool]:
    name = _internal_filename(path) if directory_descriptor is not None else Path(path).name
    if directory_descriptor is None:
        exists = os.path.lexists(path)
    else:
        try:
            os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            exists = True
        except FileNotFoundError:
            exists = False
    if exists:
        observed, record = _snapshot(
            name if directory_descriptor is not None else path,
            internal=True,
            directory_descriptor=directory_descriptor,
        )
        if observed != payload:
            raise ValueError(f"Existing Task 1 attribution artifact changed: {name}")
        return record, False
    try:
        return (
            _create_only(
                name if directory_descriptor is not None else path,
                payload,
                directory_descriptor=directory_descriptor,
            ),
            True,
        )
    except FileExistsError:
        observed, record = _snapshot(
            name if directory_descriptor is not None else path,
            internal=True,
            directory_descriptor=directory_descriptor,
        )
        if observed != payload:
            raise ValueError(f"Racing Task 1 attribution artifact changed: {name}") from None
        return record, False


def _directory_entries(directory_descriptor: int | None = None) -> dict[str, str]:
    descriptor = (
        _open_directory(ATTRIBUTION_ROOT)
        if directory_descriptor is None
        else os.dup(directory_descriptor)
    )
    try:
        result: dict[str, str] = {}
        for name in os.listdir(descriptor):
            if name in {".", ".."}:
                raise ValueError("Task 1 attribution directory contains an invalid name")
            status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(status.st_mode):
                kind = "file"
            elif stat.S_ISDIR(status.st_mode):
                kind = "directory"
            elif stat.S_ISLNK(status.st_mode):
                kind = "symlink"
            else:
                kind = "other"
            result[name] = kind
        return result
    finally:
        os.close(descriptor)


def _assert_directory_descriptor_current(
    path: Path,
    directory_descriptor: int,
    name: str,
) -> None:
    current = _open_directory(path)
    try:
        held_status = os.fstat(directory_descriptor)
        current_status = os.fstat(current)
        if (
            held_status.st_dev != current_status.st_dev
            or held_status.st_ino != current_status.st_ino
            or not stat.S_ISDIR(held_status.st_mode)
        ):
            raise PermissionError(f"{name} changed during operation")
    finally:
        os.close(current)


def _assert_root_descriptor_current(directory_descriptor: int) -> None:
    _assert_directory_descriptor_current(
        ATTRIBUTION_ROOT,
        directory_descriptor,
        "Task 1 attribution output directory",
    )


def _validate_partial_inventory(
    selection: Mapping[str, Any] | None = None,
    *,
    directory_descriptor: int | None = None,
) -> None:
    entries = _directory_entries(directory_descriptor)
    if any(kind != "file" for kind in entries.values()):
        raise ValueError("Task 1 attribution directory contains a non-file entry")
    allowed = set(_FIXED_FILES)
    if selection is not None:
        allowed.update(item["image_filename"] for item in selection["items"])
    if not set(entries).issubset(allowed):
        raise ValueError("Task 1 attribution directory contains an unexpected artifact")
    if _SELECTION_FILENAME not in entries and entries:
        raise ValueError("Task 1 attribution evidence exists before its selection")
    if _SELECTION_RECORD_FILENAME not in entries and any(
        name not in {_SELECTION_FILENAME, _SELECTION_RECORD_FILENAME} for name in entries
    ):
        raise ValueError("Task 1 attribution evidence exists before the selection record")
    if _LOCK_FILENAME in entries and set(entries) != allowed:
        raise ValueError("Task 1 attribution lock exists with an incomplete inventory")


def _record_from_verified(value: object, name: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or type(value.get("path")) is not str
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"{name} artifact record is invalid")
    record = dict(value)
    if _artifact_record(record["path"]) != record:
        raise PermissionError(f"{name} artifact changed")
    return record


def _validated_attempt_record(
    value: object,
    expected_relative_path: str,
    name: str,
) -> tuple[dict[str, Any], Path]:
    expected = Path(expected_relative_path)
    if (
        expected.is_absolute()
        or expected.as_posix() != expected_relative_path
        or not expected.parts
        or any(part in {"", ".", ".."} for part in expected.parts)
    ):
        raise ValueError("Task 1 attribution expected attempt path is invalid")
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or value.get("path") != expected_relative_path
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"{name} artifact record is invalid")
    return dict(value), expected


def _attempt_snapshot_from_verified(
    value: object,
    expected_relative_path: str,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    record, expected = _validated_attempt_record(value, expected_relative_path, name)
    payload, observed = _snapshot(FINAL_EVALUATION_ATTEMPT_DIRECTORY / expected)
    if observed["sha256"] != record["sha256"] or observed["size_bytes"] != record["size_bytes"]:
        raise PermissionError(f"{name} artifact changed")
    return payload, record


def _attempt_record_from_verified(
    value: object,
    expected_relative_path: str,
    name: str,
) -> dict[str, Any]:
    _, record = _attempt_snapshot_from_verified(value, expected_relative_path, name)
    return record


def _attempt_json_from_verified(
    value: object,
    expected_relative_path: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _attempt_snapshot_from_verified(value, expected_relative_path, name)
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not canonical JSON") from exc
    if not isinstance(parsed, dict) or _json_bytes(parsed) != payload:
        raise ValueError(f"{name} is not canonical JSON")
    return parsed, record


def _verified_context(final: Mapping[str, Any], gate_result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(final, Mapping) or not isinstance(gate_result, Mapping):
        raise ValueError("Task 1 attribution requires verified final evidence")
    gate = gate_result.get("gate")
    if not isinstance(gate, Mapping) or gate.get("ready") is not True:
        raise ValueError("Task 1 attribution gate is not ready")
    final_result = _attempt_record_from_verified(
        final.get("result_artifact"), "result.json", "Final result"
    )
    final_lock_value, final_lock = _attempt_json_from_verified(
        final.get("completion_lock_artifact"), "lock.json", "Final completion lock"
    )
    gate_artifact = _record_from_verified(gate_result.get("gate_artifact"), "Gate")
    gate_lock = _record_from_verified(gate_result.get("lock_artifact"), "Gate lock")
    claim_sha256 = _require_sha256(final.get("claim_sha256"), "Final claim")
    final_claim = _record_from_verified(final_lock_value.get("claim"), "Final claim")
    if (
        final.get("complete") is not True
        or final.get("gate_sha256") != gate_artifact["sha256"]
        or final.get("source_fingerprint_sha256")
        != gate.get("shared_identity", {}).get("source_fingerprint_sha256")
        or set(final_lock_value)
        != {
            "schema_version",
            "attempt_id",
            "gate",
            "gate_lock",
            "claim",
            "result",
            "stage_locks",
            "failure_diagnostics",
        }
        or final_lock_value.get("result") != final_result
        or final_lock_value.get("gate") != gate_artifact
        or final_lock_value.get("gate_lock") != gate_lock
        or final_claim["sha256"] != claim_sha256
    ):
        raise ValueError("Task 1 attribution final and gate bindings differ")
    current_source = source_fingerprint()
    _require_sha256(current_source, "Current source fingerprint")
    if current_source != gate["shared_identity"]["source_fingerprint_sha256"]:
        raise PermissionError("Current source fingerprint differs from the sealed final evidence")
    runs = gate.get("task1", {}).get("runs")
    if not isinstance(runs, list):
        raise ValueError("Task 1 gate run inventory is invalid")
    selected_runs = [
        run for run in runs if isinstance(run, Mapping) and run.get("seed") == DETAIL_SEED
    ]
    if len(selected_runs) != 1:
        raise ValueError("Task 1 gate lacks the unique seed 37 run")
    run = dict(selected_runs[0])
    checkpoint = run.get("best_checkpoint")
    if (
        type(run.get("run_id")) is not str
        or not _is_sha256(run.get("run_identity_sha256"))
        or not isinstance(checkpoint, Mapping)
        or not _is_sha256(checkpoint.get("sha256"))
        or type(checkpoint.get("path")) is not str
    ):
        raise ValueError("Task 1 seed 37 run identity is invalid")
    checkpoint_record = _record_from_verified(checkpoint, "Task 1 seed 37 checkpoint")
    cache = gate.get("cache_locks", {}).get("known")
    if (
        not isinstance(cache, Mapping)
        or cache.get("sha256") != KNOWN_CACHE_LOCK_SHA256
        or type(cache.get("path")) is not str
    ):
        raise ValueError("Task 1 seed 37 cache lock binding is invalid")
    cache_file_record = {
        "path": cache["path"],
        "sha256": cache["sha256"],
        "size_bytes": cache.get("size_bytes"),
    }
    _record_from_verified(cache_file_record, "Known cache lock")
    if run.get("cache_lock_sha256") != cache["sha256"]:
        raise ValueError("Task 1 seed 37 run uses another cache lock")
    if (
        not _is_sha256(run.get("source_fingerprint_sha256"))
        or run["source_fingerprint_sha256"] != current_source
    ):
        raise ValueError("Task 1 seed 37 run source fingerprint changed")
    stage_result = final.get("stage_results", {}).get("task1_seed_37")
    if not isinstance(stage_result, Mapping):
        raise ValueError("Task 1 seed 37 final stage result is missing")
    stage_result_value, stage_result_record = _attempt_json_from_verified(
        stage_result,
        "task1_seed_37/result.json",
        "Task 1 seed 37 stage result",
    )
    if (
        stage_result_value.get("stage_id") != "task1_seed_37"
        or stage_result_value.get("seed") != DETAIL_SEED
        or stage_result_value.get("recording_count") != KNOWN_TEST_RECORDINGS
        or not isinstance(stage_result_value.get("recording_ids"), list)
        or len(stage_result_value["recording_ids"]) != KNOWN_TEST_RECORDINGS
        or any(type(value) is not str or not value for value in stage_result_value["recording_ids"])
        or len(set(stage_result_value["recording_ids"])) != KNOWN_TEST_RECORDINGS
    ):
        raise ValueError("Task 1 seed 37 final stage result inventory changed")
    stage_locks = final_lock_value.get("stage_locks")
    if not isinstance(stage_locks, Mapping):
        raise ValueError("Final completion lock lacks stage lock bindings")
    stage_lock_value, stage_lock_record = _attempt_json_from_verified(
        stage_locks.get("task1_seed_37"),
        "task1_seed_37/lock.json",
        "Task 1 seed 37 stage lock",
    )
    if (
        set(stage_lock_value) != {"schema_version", "stage_id", "result", "shards"}
        or stage_lock_value.get("stage_id") != "task1_seed_37"
        or stage_lock_value.get("result") != stage_result_record
        or not isinstance(stage_lock_value.get("shards"), list)
        or len(stage_lock_value["shards"]) != KNOWN_TEST_RECORDINGS
    ):
        raise ValueError("Task 1 seed 37 stage lock inventory changed")
    shard_records: list[dict[str, Any]] = []
    shard_paths: list[str] = []
    for value in stage_lock_value["shards"]:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256", "size_bytes"}
            or type(value.get("path")) is not str
            or Path(value["path"]).parts != ("task1_seed_37", "shards", Path(value["path"]).name)
            or len(Path(value["path"]).name) != 69
            or not Path(value["path"]).name.endswith(".json")
            or not _is_sha256(Path(value["path"]).name[:-5])
            or not _is_sha256(value.get("sha256"))
            or type(value.get("size_bytes")) is not int
            or value["size_bytes"] <= 0
        ):
            raise ValueError("Task 1 seed 37 shard artifact record is invalid")
        shard_paths.append(value["path"])
        shard_records.append(dict(value))
    if shard_paths != sorted(shard_paths) or len(set(shard_paths)) != KNOWN_TEST_RECORDINGS:
        raise ValueError("Task 1 seed 37 shard artifact inventory is not canonical")
    return {
        "final_result": final_result,
        "final_lock": final_lock,
        "final_claim": final_claim,
        "gate": gate_artifact,
        "gate_lock": gate_lock,
        "source_fingerprint_sha256": current_source,
        "claim_sha256": claim_sha256,
        "known_cache_lock": dict(cache),
        "seed_37": {
            "run_id": run["run_id"],
            "run_identity_sha256": run["run_identity_sha256"],
            "checkpoint": checkpoint_record,
            "cache_lock_sha256": cache["sha256"],
            "source_fingerprint_sha256": run.get("source_fingerprint_sha256"),
            "final_stage_result": stage_result_record,
            "final_stage_lock": stage_lock_record,
            "final_shards": shard_records,
            "final_recording_ids": list(stage_result_value["recording_ids"]),
        },
        "run": run,
        "gate_value": dict(gate),
    }


def _selection_digest(stratum: str, recording_id: str) -> str:
    value = f"{SELECTION_ORDER_DOMAIN}\0{stratum}\0{recording_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _image_filename(stratum: str, rank: int, recording_id: str) -> str:
    identity = hashlib.sha256(recording_id.encode()).hexdigest()[:12]
    return f"{stratum}_{rank:02d}_{identity}.png"


def _shard_path(recording_id: str) -> Path:
    filename = hashlib.sha256(recording_id.encode()).hexdigest() + ".json"
    return FINAL_EVALUATION_ATTEMPT_DIRECTORY / "task1_seed_37" / "shards" / filename


def _parse_shard(
    value: object, record: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SHARD_FIELDS:
        raise ValueError("Task 1 attribution encountered an invalid seed 37 shard schema")
    true_index = value.get("true_class_index")
    predicted_index = value.get("predicted_class_index")
    logits = value.get("mean_logits")
    metadata = value.get("metadata")
    if (
        type(true_index) is not int
        or type(predicted_index) is not int
        or not 0 <= true_index < len(LOCKED_TASK1_CLASS_ORDER)
        or not 0 <= predicted_index < len(LOCKED_TASK1_CLASS_ORDER)
        or not isinstance(logits, list)
        or len(logits) != len(LOCKED_TASK1_CLASS_ORDER)
        or any(type(item) is not float or not math.isfinite(item) for item in logits)
        or not isinstance(metadata, dict)
    ):
        raise ValueError("Task 1 attribution encountered invalid shard predictions")
    clip_ids = metadata.get("clip_ids")
    recording_id = value.get("recording_id")
    if (
        type(recording_id) is not str
        or not recording_id
        or not isinstance(clip_ids, list)
        or not clip_ids
        or any(type(item) is not str or not item for item in clip_ids)
        or len(set(clip_ids)) != len(clip_ids)
        or metadata.get("recording_id") != recording_id
        or metadata.get("clip_count") != len(clip_ids)
        or value.get("seed") != DETAIL_SEED
        or value.get("stage_id") != "task1_seed_37"
        or value.get("run_id") != context["seed_37"]["run_id"]
        or value.get("run_identity_sha256") != context["seed_37"]["run_identity_sha256"]
        or value.get("checkpoint_sha256") != context["seed_37"]["checkpoint"]["sha256"]
        or value.get("cache_lock_sha256") != context["seed_37"]["cache_lock_sha256"]
        or value.get("gate_sha256") != context["gate"]["sha256"]
        or value.get("claim_sha256") != context["claim_sha256"]
        or value.get("source_fingerprint_sha256") != context["seed_37"]["source_fingerprint_sha256"]
        or value.get("task") != "task1_classification"
        or value.get("source_role") != FINAL_KNOWN_TEST_ROLE
        or value.get("true_class_name") != LOCKED_TASK1_CLASS_ORDER[true_index]
        or value.get("predicted_class_name") != LOCKED_TASK1_CLASS_ORDER[predicted_index]
    ):
        raise ValueError("Task 1 attribution seed 37 shard binding changed")
    if int(np.argmax(np.asarray(logits, dtype=np.float64))) != predicted_index:
        raise ValueError("Task 1 attribution shard predicted class differs from its logits")
    return {
        "recording_id": recording_id,
        "session_group": value["session_group"],
        "clip_ids": sorted(clip_ids),
        "clip_count": len(clip_ids),
        "true_class_index": true_index,
        "true_class_name": value["true_class_name"],
        "predicted_class_index": predicted_index,
        "predicted_class_name": value["predicted_class_name"],
        "mean_logits": logits,
        "shard": dict(record),
    }


def _seed37_items(context: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    directory = FINAL_EVALUATION_ATTEMPT_DIRECTORY / "task1_seed_37" / "shards"
    seed_context = context.get("seed_37")
    if not isinstance(seed_context, Mapping):
        raise ValueError("Task 1 attribution lacks the seed 37 context")
    locked_records = seed_context.get("final_shards")
    if not isinstance(locked_records, list) or len(locked_records) != KNOWN_TEST_RECORDINGS:
        raise ValueError("Task 1 attribution lacks the locked shard inventory")
    records_by_name = {
        Path(record["path"]).name: record
        for record in locked_records
        if isinstance(record, Mapping) and type(record.get("path")) is str
    }
    if len(records_by_name) != KNOWN_TEST_RECORDINGS:
        raise ValueError("Task 1 attribution locked shard filenames are invalid")
    descriptor = _open_directory(directory)
    try:
        names = sorted(os.listdir(descriptor))
        if names != sorted(records_by_name):
            raise ValueError("Task 1 seed 37 shard inventory differs from its stage lock")
        items: list[dict[str, Any]] = []
        for name in names:
            if len(name) != 69 or not name.endswith(".json") or not _is_sha256(name[:-5]):
                raise ValueError("Task 1 seed 37 shard inventory contains an invalid filename")
            value, observed = _read_json(
                name,
                internal=True,
                directory_descriptor=descriptor,
            )
            record = records_by_name[name]
            if (
                observed["sha256"] != record["sha256"]
                or observed["size_bytes"] != record["size_bytes"]
            ):
                raise PermissionError("Task 1 seed 37 shard differs from its stage lock")
            item = _parse_shard(value, record, context)
            expected_relative = _shard_path(item["recording_id"]).relative_to(
                FINAL_EVALUATION_ATTEMPT_DIRECTORY
            )
            if record["path"] != expected_relative.as_posix():
                raise ValueError("Task 1 attribution shard filename changed")
            items.append(item)
        _assert_directory_descriptor_current(
            directory,
            descriptor,
            "Task 1 seed 37 shard directory",
        )
    finally:
        os.close(descriptor)
    identities = [item["recording_id"] for item in items]
    expected_identities = seed_context.get("final_recording_ids")
    if (
        len(set(identities)) != KNOWN_TEST_RECORDINGS
        or not isinstance(expected_identities, list)
        or sorted(identities) != sorted(expected_identities)
    ):
        raise ValueError("Task 1 seed 37 shard recording identities changed")
    return tuple(items)


def _assert_context_current(context: Mapping[str, Any]) -> None:
    expected_source = context.get("source_fingerprint_sha256")
    if not _is_sha256(expected_source) or source_fingerprint() != expected_source:
        raise PermissionError("Task 1 attribution source changed during operation")

    attempt_records = (
        (context.get("final_result"), "result.json", "Final result"),
        (context.get("final_lock"), "lock.json", "Final completion lock"),
    )
    seed_context = context.get("seed_37")
    if not isinstance(seed_context, Mapping):
        raise ValueError("Task 1 attribution lacks the seed 37 context")
    attempt_records += (
        (
            seed_context.get("final_stage_result"),
            "task1_seed_37/result.json",
            "Task 1 seed 37 stage result",
        ),
        (
            seed_context.get("final_stage_lock"),
            "task1_seed_37/lock.json",
            "Task 1 seed 37 stage lock",
        ),
    )
    for record, relative_path, name in attempt_records:
        _attempt_record_from_verified(record, relative_path, name)

    for key, name in (
        ("final_claim", "Final claim"),
        ("gate", "Gate"),
        ("gate_lock", "Gate lock"),
    ):
        _record_from_verified(context.get(key), name)
    _record_from_verified(seed_context.get("checkpoint"), "Task 1 seed 37 checkpoint")

    cache = context.get("known_cache_lock")
    if not isinstance(cache, Mapping):
        raise ValueError("Task 1 attribution known cache lock binding is invalid")
    cache_record = {
        "path": cache.get("path"),
        "sha256": cache.get("sha256"),
        "size_bytes": cache.get("size_bytes"),
    }
    _record_from_verified(cache_record, "Known cache lock")
    _seed37_items(context)

    if source_fingerprint() != expected_source:
        raise PermissionError("Task 1 attribution source changed during operation")


def _selection_value(context: Mapping[str, Any]) -> dict[str, Any]:
    by_stratum: dict[str, list[dict[str, Any]]] = {"correct": [], "error": []}
    for item in _seed37_items(context):
        stratum = (
            "correct" if item["true_class_index"] == item["predicted_class_index"] else "error"
        )
        by_stratum[stratum].append(item)
    if any(len(items) < SELECTIONS_PER_STRATUM for items in by_stratum.values()):
        raise ValueError(
            "Task 1 attribution requires at least three correct and three error recordings"
        )
    selected: list[dict[str, Any]] = []
    strata: dict[str, list[str]] = {}
    for stratum in ("correct", "error"):
        ranked = sorted(
            by_stratum[stratum],
            key=lambda item: (
                _selection_digest(stratum, item["recording_id"]),
                item["recording_id"],
            ),
        )[:SELECTIONS_PER_STRATUM]
        strata[stratum] = [item["recording_id"] for item in ranked]
        for rank, base in enumerate(ranked, start=1):
            selected.append(
                {
                    "stratum": stratum,
                    "rank": rank,
                    "selection_key_sha256": _selection_digest(stratum, base["recording_id"]),
                    "image_filename": _image_filename(stratum, rank, base["recording_id"]),
                    **base,
                }
            )
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "attribution_id": ATTRIBUTION_ID,
        "detail_seed": DETAIL_SEED,
        "selection_order": {
            "algorithm": "sha256",
            "domain": SELECTION_ORDER_DOMAIN,
            "within_stratum": True,
            "saliency_independent": True,
        },
        "selections_per_stratum": SELECTIONS_PER_STRATUM,
        "selection_count": 2 * SELECTIONS_PER_STRATUM,
        "strata": strata,
        "bindings": {
            key: value for key, value in context.items() if key not in {"run", "gate_value"}
        },
        "items": selected,
    }


def _selection_record_value(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "attribution_id": ATTRIBUTION_ID,
        "selection": dict(record),
    }


def _prepare_runtime() -> torch.device:
    project_venv = _absolute(PROJECT_ROOT / ".venv")
    executable = _absolute(sys.executable)
    if _absolute(sys.prefix) != project_venv or project_venv not in executable.parents:
        raise RuntimeError("Task 1 attribution must run inside the project .venv")
    environment_controls = {
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        "PYTORCH_MPS_FAST_MATH": os.environ.get("PYTORCH_MPS_FAST_MATH", ""),
        "PYTORCH_MPS_PREFER_METAL": os.environ.get("PYTORCH_MPS_PREFER_METAL", ""),
    }
    enabled = [
        name
        for name, value in environment_controls.items()
        if value.strip().lower() not in {"", "0", "false"}
    ]
    if enabled:
        raise RuntimeError(
            f"Task 1 attribution requires disabled MPS numerical overrides: {enabled}"
        )
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Task 1 attribution requires an available MPS device")
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.get_default_dtype() != torch.float32
    ):
        raise RuntimeError("Task 1 attribution deterministic float32 runtime was not established")
    return torch.device("mps")


def _final_convolutional_block(model: nn.Module) -> nn.Module:
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Sequential) or len(features) != 9:
        raise ValueError("Task 1 attribution model lacks the locked EfficientNet feature layout")
    target = features[8]
    if not isinstance(target, nn.Module) or not any(
        isinstance(module, nn.Conv2d) for module in target.modules()
    ):
        raise ValueError("Task 1 attribution final convolutional block changed")
    return target


def _compute_gradcam(
    model: nn.Module,
    native: torch.Tensor,
    *,
    target_class_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(native, torch.Tensor)
        or native.dtype != torch.float32
        or native.device.type != "cpu"
        or native.ndim != 4
        or tuple(native.shape[1:]) != (1, 128, 372)
        or native.shape[0] <= 0
        or not bool(torch.isfinite(native).all().item())
    ):
        raise ValueError("Task 1 attribution native recording tensor is invalid")
    if type(target_class_index) is not int or not 0 <= target_class_index < len(
        LOCKED_TASK1_CLASS_ORDER
    ):
        raise ValueError("Task 1 attribution target class is invalid")
    target_module = _final_convolutional_block(model)
    activation: list[torch.Tensor] = []
    gradient: list[torch.Tensor] = []
    tensor_hook: list[Any] = []

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Task 1 attribution final block output is invalid")
        activation.append(output)
        tensor_hook.append(output.register_hook(lambda value: gradient.append(value)))

    forward_hook = target_module.register_forward_hook(capture)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        inputs_cpu = to_efficientnet_batch(native)
        if (
            inputs_cpu.dtype != torch.float32
            or tuple(inputs_cpu.shape) != (native.shape[0], 3, 224, 224)
            or not bool(torch.isfinite(inputs_cpu).all().item())
        ):
            raise RuntimeError("Task 1 attribution preprocessing returned an invalid tensor")
        inputs = inputs_cpu.to(device=device, dtype=torch.float32)
        with torch.enable_grad():
            logits = model(inputs)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.dtype != torch.float32
                or logits.device.type != device.type
                or tuple(logits.shape) != (native.shape[0], len(LOCKED_TASK1_CLASS_ORDER))
            ):
                raise ValueError("Task 1 attribution model logits are invalid")
            score = logits[:, target_class_index].mean()
            if not bool(torch.isfinite(score).detach().to("cpu").item()):
                raise ValueError("Task 1 attribution target logit is nonfinite")
            score.backward()
        if len(activation) != 1 or len(gradient) != 1:
            raise RuntimeError(
                "Task 1 attribution hooks did not capture one forward and backward pass"
            )
        activations = activation[0]
        gradients = gradient[0]
        if activations.shape != gradients.shape or tuple(activations.shape[:1]) != (
            native.shape[0],
        ):
            raise ValueError("Task 1 attribution activation and gradient shapes differ")
        finite = (
            torch.stack((torch.isfinite(activations).all(), torch.isfinite(gradients).all()))
            .detach()
            .to("cpu")
        )
        if not bool(finite.all().item()):
            raise ValueError("Task 1 attribution hooks captured nonfinite values")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        maxima = maps.amax(dim=(2, 3), keepdim=True)
        maps = torch.where(maxima > 0, maps / maxima.clamp_min(torch.finfo(maps.dtype).tiny), maps)
        maps = functional.interpolate(
            maps,
            size=(native.shape[2], native.shape[3]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        if device.type == "mps":
            torch.mps.synchronize()
        maps_cpu = maps.detach().to(device="cpu", dtype=torch.float32, copy=True).numpy()
        mean_logits = logits.detach().mean(dim=0).to(device="cpu", dtype=torch.float32).numpy()
        if (
            maps_cpu.shape != (native.shape[0], 128, 372)
            or mean_logits.shape != (len(LOCKED_TASK1_CLASS_ORDER),)
            or not bool(np.all(np.isfinite(maps_cpu)))
            or not bool(np.all(np.isfinite(mean_logits)))
            or bool(np.any(maps_cpu < 0.0))
            or bool(np.any(maps_cpu > 1.0 + 1e-6))
        ):
            raise ValueError("Task 1 attribution Grad-CAM output is invalid")
        return maps_cpu, mean_logits
    finally:
        forward_hook.remove()
        for hook in tensor_hook:
            hook.remove()
        model.zero_grad(set_to_none=True)


def _slaney_hz_to_mel(hz: float) -> float:
    if not math.isfinite(hz) or hz < 0.0:
        raise ValueError("Slaney Mel conversion requires a nonnegative finite frequency")
    frequency_step = 200.0 / 3.0
    if hz < 1000.0:
        return hz / frequency_step
    minimum_log_mel = 1000.0 / frequency_step
    logarithmic_step = math.log(6.4) / 27.0
    return minimum_log_mel + math.log(hz / 1000.0) / logarithmic_step


def _mel_filter_center_mels() -> np.ndarray:
    lower = _slaney_hz_to_mel(150.0)
    upper = _slaney_hz_to_mel(14_000.0)
    return np.linspace(lower, upper, num=128 + 2, dtype=np.float64)[1:-1]


def _mel_tick_positions() -> tuple[float, ...]:
    centers = _mel_filter_center_mels()
    positions = np.arange(128, dtype=np.float64)
    return tuple(
        float(
            np.interp(
                _slaney_hz_to_mel(value * 1000.0),
                centers,
                positions,
                left=0.0,
                right=127.0,
            )
        )
        for value in MEL_FREQUENCY_TICKS_KHZ
    )


def _render_png(native: np.ndarray, maps: np.ndarray, item: Mapping[str, Any]) -> bytes:
    clip_ids = item["clip_ids"]
    if (
        not isinstance(native, np.ndarray)
        or native.dtype != np.float32
        or native.shape != (len(clip_ids), 1, 128, 372)
        or maps.shape != (len(clip_ids), 128, 372)
        or not bool(np.all(np.isfinite(native)))
        or not bool(np.all(np.isfinite(maps)))
    ):
        raise ValueError("Task 1 attribution render arrays are invalid")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    ):
        figure, axes = plt.subplots(
            len(clip_ids),
            2,
            figsize=(12.0, 2.35 * len(clip_ids)),
            dpi=FIGURE_DPI,
            squeeze=False,
        )
        try:
            extent = (TIME_SECONDS[0], TIME_SECONDS[1], 0.0, 127.0)
            ticks = _mel_tick_positions()
            tick_labels = [f"{value:g}" for value in MEL_FREQUENCY_TICKS_KHZ]
            for row, clip_id in enumerate(clip_ids):
                spectrogram = native[row, 0]
                for column, title in ((0, "Log-Mel spectrogram"), (1, "Grad-CAM overlay")):
                    axis = axes[row, column]
                    axis.imshow(
                        spectrogram,
                        origin="lower",
                        aspect="auto",
                        extent=extent,
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                    )
                    if column == 1:
                        axis.imshow(
                            maps[row],
                            origin="lower",
                            aspect="auto",
                            extent=extent,
                            cmap="magma",
                            vmin=0.0,
                            vmax=1.0,
                            alpha=OVERLAY_ALPHA,
                            interpolation="bilinear",
                        )
                    axis.set_xlim(*TIME_SECONDS)
                    axis.set_ylim(0.0, 127.0)
                    axis.set_xticks((0.0, 1.0, 2.0, 3.0))
                    axis.set_yticks(ticks, tick_labels)
                    axis.set_xlabel("Time (s)")
                    axis.set_ylabel("Mel frequency (kHz)")
                    axis.set_title(f"Clip {row + 1}: {clip_id} | {title}")
            figure.suptitle(
                f"{item['stratum'].upper()} | true: {item['true_class_name']} | "
                f"predicted: {item['predicted_class_name']}",
                fontsize=11,
                y=0.995,
            )
            figure.subplots_adjust(left=0.075, right=0.985, bottom=0.055, top=0.94, hspace=0.62)
            buffer = io.BytesIO()
            figure.savefig(
                buffer,
                format="png",
                dpi=FIGURE_DPI,
                facecolor="white",
                metadata={"Software": "bird_audio.task1_attribution"},
            )
            payload = buffer.getvalue()
        finally:
            plt.close(figure)
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Task 1 attribution renderer did not produce PNG bytes")
    return payload


def _render_contract() -> dict[str, Any]:
    return {
        "backend": "matplotlib_agg",
        "dpi": FIGURE_DPI,
        "panels_per_clip": ["grayscale_log_mel", "grad_cam_overlay"],
        "all_selected_clips": True,
        "time_axis_seconds": list(TIME_SECONDS),
        "time_ticks_seconds": [0.0, 1.0, 2.0, 3.0],
        "frequency_axis": "Mel frequency (kHz)",
        "frequency_ticks_khz": list(MEL_FREQUENCY_TICKS_KHZ),
        "frequency_span_khz": [0.15, 14.0],
        "overlay_colormap": "magma",
        "overlay_alpha": OVERLAY_ALPHA,
        "labels": ["true_class", "predicted_class", "correct_or_error_stratum"],
    }


def _method_contract() -> dict[str, Any]:
    return {
        "method": "grad_cam",
        "model_seed": DETAIL_SEED,
        "target_module": FINAL_CONVOLUTIONAL_TARGET,
        "target": "stored_predicted_recording_class",
        "recording_logit": "mean_of_all_selected_clip_logits",
        "channel_weights": "spatial_mean_of_target_gradients",
        "activation": "relu",
        "normalization": "independent_zero_to_one_per_clip",
        "preprocessing": "locked_to_efficientnet_batch",
    }


def _reader_recording(
    data: Any, item: Mapping[str, Any]
) -> tuple[np.ndarray, tuple[dict[str, str], ...]]:
    features, rows = data.get_recording(item["recording_id"])
    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.float32
        or features.shape != (item["clip_count"], 1, 128, 372)
        or not isinstance(rows, tuple)
        or len(rows) != item["clip_count"]
        or not bool(np.all(np.isfinite(features)))
    ):
        raise ValueError("Task 1 attribution reader returned invalid recording features")
    try:
        order = sorted(range(len(rows)), key=lambda index: rows[index]["clip_id"])
        clip_ids = [rows[index]["clip_id"] for index in order]
    except (KeyError, TypeError) as exc:
        raise ValueError("Task 1 attribution reader clip metadata is invalid") from exc
    if clip_ids != item["clip_ids"] or len(set(clip_ids)) != len(clip_ids):
        raise ValueError(
            "Task 1 attribution reader did not return every selected clip exactly once"
        )
    ordered_features = np.ascontiguousarray(features[order], dtype=np.float32)
    ordered_rows = tuple(rows[index] for index in order)
    return ordered_features, ordered_rows


def _validate_model_metadata(metadata: object, context: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("Task 1 attribution model metadata is invalid")
    expected = {
        "seed": DETAIL_SEED,
        "run_id": context["seed_37"]["run_id"],
        "run_identity_sha256": context["seed_37"]["run_identity_sha256"],
        "checkpoint_sha256": context["seed_37"]["checkpoint"]["sha256"],
        "cache_lock_sha256": context["seed_37"]["cache_lock_sha256"],
        "scope": "production",
        "production_evidence": True,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"Task 1 attribution model identity differs: {mismatches}")


def _produce_images(
    selection: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    directory_descriptor: int | None = None,
) -> tuple[dict[str, Any], ...]:
    claimed = claim_final_evaluation_attempt()
    if not isinstance(claimed, Mapping):
        raise ValueError("Task 1 attribution final claim return value is invalid")
    authorization = claimed.get("authorization")
    claim_artifact = claimed.get("claim_artifact")
    if not isinstance(authorization, FinalEvaluationAuthorization):
        raise TypeError("Task 1 attribution requires the final evaluation authorization")
    if (
        claimed.get("created") is not False
        or claimed.get("gate") != context["gate_value"]
        or not isinstance(claim_artifact, Mapping)
        or claim_artifact.get("sha256") != context["claim_sha256"]
    ):
        raise PermissionError("Task 1 attribution claim differs from the verified final evidence")
    device = _prepare_runtime()
    run = context["run"]
    model: nn.Module | None = None
    records: list[dict[str, Any]] = []
    try:
        model, metadata = load_locked_task1_best_model(
            run["best_checkpoint"]["path"],
            checkpoint_sha256=run["best_checkpoint"]["sha256"],
            expected_run_identity_sha256=run["run_identity_sha256"],
            device=device,
        )
        _validate_model_metadata(metadata, context)
        data = open_final_known_test_data(authorization)
        if data.lock_sha256 != context["seed_37"]["cache_lock_sha256"]:
            raise PermissionError("Task 1 attribution reader uses another cache lock")
        for item in selection["items"]:
            features, rows = _reader_recording(data, item)
            native = torch.from_numpy(features.copy()).to(dtype=torch.float32)
            maps, mean_logits = _compute_gradcam(
                model,
                native,
                target_class_index=item["predicted_class_index"],
                device=device,
            )
            if int(np.argmax(mean_logits)) != item["predicted_class_index"]:
                raise ValueError(
                    "Task 1 attribution model prediction differs from the locked shard"
                )
            payload = _render_png(features, maps, item)
            artifact, _ = _publish_or_verify(
                item["image_filename"]
                if directory_descriptor is not None
                else ATTRIBUTION_ROOT / item["image_filename"],
                payload,
                directory_descriptor=directory_descriptor,
            )
            records.append(
                {
                    "recording_id": item["recording_id"],
                    "stratum": item["stratum"],
                    "rank": item["rank"],
                    "true_class_name": item["true_class_name"],
                    "predicted_class_name": item["predicted_class_name"],
                    "clip_ids": list(item["clip_ids"]),
                    "clip_count": item["clip_count"],
                    "all_clips_included": len(rows) == item["clip_count"],
                    "target_class_index": item["predicted_class_index"],
                    "artifact": artifact,
                }
            )
            del native, maps, mean_logits, features
            if device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
        return tuple(records)
    finally:
        if model is not None:
            del model
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()


def _manifest_value(
    selection: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    selection_record_record: Mapping[str, Any],
    images: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(images) != 2 * SELECTIONS_PER_STRATUM:
        raise ValueError("Task 1 attribution image inventory is incomplete")
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "attribution_id": ATTRIBUTION_ID,
        "complete": True,
        "selection": dict(selection_record),
        "selection_record": dict(selection_record_record),
        "bindings": selection["bindings"],
        "method": _method_contract(),
        "rendering": _render_contract(),
        "image_count": len(images),
        "images": [dict(item) for item in images],
    }


def _lock_value(
    selection_record: Mapping[str, Any],
    selection_record_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    images: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "attribution_id": ATTRIBUTION_ID,
        "selection": dict(selection_record),
        "selection_record": dict(selection_record_record),
        "manifest": dict(manifest_record),
        "images": [dict(item["artifact"]) for item in images],
    }


def _validate_selection_shape(value: object) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "attribution_id",
        "detail_seed",
        "selection_order",
        "selections_per_stratum",
        "selection_count",
        "strata",
        "bindings",
        "items",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema_version") != ATTRIBUTION_SCHEMA_VERSION
        or value.get("attribution_id") != ATTRIBUTION_ID
        or value.get("detail_seed") != DETAIL_SEED
        or value.get("selections_per_stratum") != SELECTIONS_PER_STRATUM
        or value.get("selection_count") != 2 * SELECTIONS_PER_STRATUM
        or not isinstance(value.get("items"), list)
        or len(value["items"]) != 2 * SELECTIONS_PER_STRATUM
    ):
        raise ValueError("Task 1 attribution selection schema is invalid")
    return value


def _verify_with_context(
    context: Mapping[str, Any],
    *,
    directory_descriptor: int | None = None,
) -> dict[str, Any]:
    selection, selection_record = _read_json(
        _SELECTION_FILENAME
        if directory_descriptor is not None
        else ATTRIBUTION_ROOT / _SELECTION_FILENAME,
        internal=True,
        directory_descriptor=directory_descriptor,
    )
    selection = _validate_selection_shape(selection)
    expected_selection = _selection_value(context)
    if selection != expected_selection:
        raise ValueError("Task 1 attribution selection changed")
    record_value, record_artifact = _read_json(
        _SELECTION_RECORD_FILENAME
        if directory_descriptor is not None
        else ATTRIBUTION_ROOT / _SELECTION_RECORD_FILENAME,
        internal=True,
        directory_descriptor=directory_descriptor,
    )
    if record_value != _selection_record_value(selection_record):
        raise ValueError("Task 1 attribution selection record changed")
    _validate_partial_inventory(selection, directory_descriptor=directory_descriptor)
    manifest, manifest_record = _read_json(
        _MANIFEST_FILENAME
        if directory_descriptor is not None
        else ATTRIBUTION_ROOT / _MANIFEST_FILENAME,
        internal=True,
        directory_descriptor=directory_descriptor,
    )
    images = manifest.get("images") if isinstance(manifest, dict) else None
    if not isinstance(images, list):
        raise ValueError("Task 1 attribution manifest image inventory is invalid")
    observed_images: list[dict[str, Any]] = []
    for item, selected in zip(images, selection["items"], strict=True):
        if not isinstance(item, dict):
            raise ValueError("Task 1 attribution manifest image entry is invalid")
        artifact = _artifact_record(
            selected["image_filename"]
            if directory_descriptor is not None
            else ATTRIBUTION_ROOT / selected["image_filename"],
            internal=True,
            directory_descriptor=directory_descriptor,
        )
        expected = {
            "recording_id": selected["recording_id"],
            "stratum": selected["stratum"],
            "rank": selected["rank"],
            "true_class_name": selected["true_class_name"],
            "predicted_class_name": selected["predicted_class_name"],
            "clip_ids": list(selected["clip_ids"]),
            "clip_count": selected["clip_count"],
            "all_clips_included": True,
            "target_class_index": selected["predicted_class_index"],
            "artifact": artifact,
        }
        if item != expected:
            raise ValueError("Task 1 attribution manifest image binding changed")
        observed_images.append(expected)
    expected_manifest = _manifest_value(
        selection,
        selection_record,
        record_artifact,
        observed_images,
    )
    if manifest != expected_manifest:
        raise ValueError("Task 1 attribution manifest changed")
    lock, lock_record = _read_json(
        _LOCK_FILENAME if directory_descriptor is not None else ATTRIBUTION_ROOT / _LOCK_FILENAME,
        internal=True,
        directory_descriptor=directory_descriptor,
    )
    expected_lock = _lock_value(
        selection_record,
        record_artifact,
        manifest_record,
        observed_images,
    )
    if lock != expected_lock:
        raise ValueError("Task 1 attribution lock changed")
    expected_inventory = _FIXED_FILES | {item["image_filename"] for item in selection["items"]}
    entries = _directory_entries(directory_descriptor)
    if set(entries) != expected_inventory or any(kind != "file" for kind in entries.values()):
        raise ValueError("Task 1 attribution final inventory changed")
    return {
        "selection": selection,
        "selection_artifact": selection_record,
        "selection_record_artifact": record_artifact,
        "manifest": manifest,
        "manifest_artifact": manifest_record,
        "lock": lock,
        "lock_artifact": lock_record,
        "created": False,
    }


def build_task1_attributions() -> dict[str, Any]:
    """Build the fixed seed 37 Grad-CAM evidence, or verify its exact existing lock."""

    final = verify_final_evaluation()
    gate = verify_final_evaluation_gate()
    context = _verified_context(final, gate)
    _ensure_directory(ATTRIBUTION_ROOT)
    descriptor = _open_directory(ATTRIBUTION_ROOT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _assert_root_descriptor_current(descriptor)
        entries = _directory_entries(descriptor)
        if _SELECTION_FILENAME in entries:
            existing_selection, _ = _read_json(
                _SELECTION_FILENAME,
                internal=True,
                directory_descriptor=descriptor,
            )
            existing_selection = _validate_selection_shape(existing_selection)
            if existing_selection != _selection_value(context):
                raise ValueError("Existing Task 1 attribution selection changed")
            _validate_partial_inventory(
                existing_selection,
                directory_descriptor=descriptor,
            )
        else:
            _validate_partial_inventory(directory_descriptor=descriptor)
        if _LOCK_FILENAME in entries:
            verified = _verify_with_context(
                context,
                directory_descriptor=descriptor,
            )
            _assert_context_current(context)
            _assert_root_descriptor_current(descriptor)
            return verified
        selection = _selection_value(context)
        selection_record, selection_created = _publish_or_verify(
            _SELECTION_FILENAME,
            _json_bytes(selection),
            directory_descriptor=descriptor,
        )
        selection_record_value = _selection_record_value(selection_record)
        selection_record_record, selection_record_created = _publish_or_verify(
            _SELECTION_RECORD_FILENAME,
            _json_bytes(selection_record_value),
            directory_descriptor=descriptor,
        )
        _validate_partial_inventory(
            selection,
            directory_descriptor=descriptor,
        )
        images = _produce_images(
            selection,
            context,
            directory_descriptor=descriptor,
        )
        _assert_context_current(context)
        manifest = _manifest_value(
            selection,
            selection_record,
            selection_record_record,
            images,
        )
        manifest_record, manifest_created = _publish_or_verify(
            _MANIFEST_FILENAME,
            _json_bytes(manifest),
            directory_descriptor=descriptor,
        )
        lock = _lock_value(
            selection_record,
            selection_record_record,
            manifest_record,
            images,
        )
        _assert_context_current(context)
        _, lock_created = _publish_or_verify(
            _LOCK_FILENAME,
            _json_bytes(lock),
            directory_descriptor=descriptor,
        )
        verified = _verify_with_context(
            context,
            directory_descriptor=descriptor,
        )
        _assert_context_current(context)
        _assert_root_descriptor_current(descriptor)
        return {
            **verified,
            "created": any(
                (selection_created, selection_record_created, manifest_created, lock_created)
            ),
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def verify_task1_attributions() -> dict[str, Any]:
    """Verify attribution evidence without model inference or saliency recomputation."""

    final = verify_final_evaluation()
    gate = verify_final_evaluation_gate()
    context = _verified_context(final, gate)
    descriptor = _open_directory(ATTRIBUTION_ROOT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        _assert_root_descriptor_current(descriptor)
        verified = _verify_with_context(
            context,
            directory_descriptor=descriptor,
        )
        _assert_context_current(context)
        _assert_root_descriptor_current(descriptor)
        return verified
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
