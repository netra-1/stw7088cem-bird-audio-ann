from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.final_evaluation import (
    FINAL_EVALUATION_ATTEMPT_DIRECTORY,
    verify_final_evaluation,
)
from bird_audio.final_evaluation_gate import (
    FINAL_EVALUATION_GATE_LOCK_PATH,
    FINAL_EVALUATION_GATE_PATH,
    verify_final_evaluation_gate,
)
from bird_audio.paths import PROJECT_ROOT
from bird_audio.provenance import source_fingerprint
from bird_audio.task2_scoring import LATENT_SCORE_NAME, RECONSTRUCTION_SCORE_NAME

FINAL_REPORT_ASSETS_SCHEMA_VERSION = "1.0"
FINAL_REPORT_ASSET_SET_ID = "final_report_assets_v2"
FINAL_REPORT_ASSET_ROOT = PROJECT_ROOT / "report_assets" / "final_v2"
FINAL_REPORT_MANIFEST_PATH = FINAL_REPORT_ASSET_ROOT / "manifest.json"
FINAL_REPORT_LOCK_PATH = FINAL_REPORT_ASSET_ROOT / "lock.json"
FINAL_EVALUATION_RESULT_PATH = FINAL_EVALUATION_ATTEMPT_DIRECTORY / "result.json"
FINAL_EVALUATION_LOCK_PATH = FINAL_EVALUATION_ATTEMPT_DIRECTORY / "lock.json"

SEED_ORDER = (13, 37, 71)
DETAIL_SEED = 37
TASK2_METRIC_ORDER = ("auroc", "sensitivity", "specificity", "balanced_accuracy")
_SHA256_LENGTH = 64
_PNG_DPI = 160
_PNG_SOFTWARE = "bird_audio_final_report_assets"

_ASSET_MEDIA_TYPES = {
    "task1_confusion_counts.csv": "text/csv; charset=utf-8",
    "task1_confusion_heatmap.png": "image/png",
    "task1_confusion_row_normalized.csv": "text/csv; charset=utf-8",
    "task1_seed37_per_class.csv": "text/csv; charset=utf-8",
    "task1_seed_metrics.csv": "text/csv; charset=utf-8",
    "task1_seed_stability.png": "image/png",
    "task1_stability.csv": "text/csv; charset=utf-8",
    "task1_training_history.csv": "text/csv; charset=utf-8",
    "task1_training_history.png": "image/png",
    "task2_reconstruction_seed_stability.png": "image/png",
    "task2_reconstruction_species_auroc_intervals.png": "image/png",
    "task2_seed37_metrics_intervals.csv": "text/csv; charset=utf-8",
    "task2_seed_metrics.csv": "text/csv; charset=utf-8",
    "task2_stability.csv": "text/csv; charset=utf-8",
    "task2_training_history.csv": "text/csv; charset=utf-8",
    "task2_training_history.png": "image/png",
}
_COMPLETE_ENTRIES = frozenset({*_ASSET_MEDIA_TYPES, "manifest.json", "lock.json"})

_RC_PARAMS: dict[str, Any] = {
    "axes.edgecolor": "#334155",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "axes.labelcolor": "#0f172a",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlecolor": "#0f172a",
    "axes.titlesize": 12,
    "figure.facecolor": "#ffffff",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "grid.alpha": 0.22,
    "grid.color": "#94a3b8",
    "legend.frameon": False,
    "savefig.facecolor": "#ffffff",
    "savefig.transparent": False,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
}


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _is_within(path: Path, boundary: Path) -> bool:
    return path == boundary or boundary in path.parents


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


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not a mapping")
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} is not a sequence")
    return value


def _require_finite(value: object, name: str, *, unit: bool = False) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} is not a float")
    resolved = float(value)
    if not math.isfinite(resolved) or (unit and not 0.0 <= resolved <= 1.0):
        raise ValueError(f"{name} is outside its valid range")
    return resolved


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
        raise ValueError("Final report metadata is not finite canonical JSON") from exc


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Final report assets require O_NOFOLLOW")
    if not isinstance(directory, int) or directory == 0:
        raise RuntimeError("Final report assets require O_DIRECTORY")
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow | directory


def _open_absolute_directory_no_follow(path: Path) -> int:
    candidate = _absolute(path)
    flags = _directory_open_flags()
    descriptor = os.open("/", flags)
    try:
        for part in candidate.parts[1:]:
            if part in {"", ".", ".."}:
                raise ValueError("Final report directory component is invalid")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final report directory component changed type")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _secure_ensure_directory(path: Path, boundary: Path) -> Path:
    candidate = _absolute(path)
    resolved_boundary = _absolute(boundary)
    if not _is_within(candidate, resolved_boundary):
        raise ValueError("Final report directory leaves the project root")
    descriptor = _open_absolute_directory_no_follow(resolved_boundary)
    try:
        for part in candidate.relative_to(resolved_boundary).parts:
            if part in {"", ".", ".."}:
                raise ValueError("Final report directory component is invalid")
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final report directory component is unsafe")
            os.close(descriptor)
            descriptor = next_descriptor
        return candidate
    finally:
        os.close(descriptor)


def _open_file_beneath(path: Path, boundary: Path) -> int:
    candidate = _absolute(path)
    resolved_boundary = _absolute(boundary)
    if candidate == resolved_boundary or not _is_within(candidate, resolved_boundary):
        raise ValueError("Final report artifact leaves its boundary")
    parts = candidate.relative_to(resolved_boundary).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Final report artifact path is invalid")
    descriptor = _open_absolute_directory_no_follow(resolved_boundary)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise ValueError("Final report artifact parent is unsafe")
            os.close(descriptor)
            descriptor = next_descriptor
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if not isinstance(no_follow, int) or no_follow == 0:
            raise RuntimeError("Final report file reads require O_NOFOLLOW")
        return os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow,
            dir_fd=descriptor,
        )
    finally:
        os.close(descriptor)


def _snapshot(path: Path, *, boundary: Path) -> tuple[bytes, str, int]:
    descriptor = _open_file_beneath(path, boundary)
    try:
        metadata_before = os.fstat(descriptor)
        if not stat.S_ISREG(metadata_before.st_mode):
            raise ValueError("Final report artifact is not a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        metadata_after = os.fstat(descriptor)
        if (
            metadata_before.st_dev != metadata_after.st_dev
            or metadata_before.st_ino != metadata_after.st_ino
            or metadata_before.st_size != metadata_after.st_size
            or size != metadata_after.st_size
        ):
            raise ValueError("Final report artifact changed while it was read")
        return b"".join(chunks), digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _external_record(value: object, expected_path: Path, name: str) -> dict[str, Any]:
    record = _require_mapping(value, name)
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{name} descriptor fields are invalid")
    candidate = _absolute(str(record["path"]))
    expected = _absolute(expected_path)
    if candidate != expected:
        raise ValueError(f"{name} path is not canonical")
    sha256 = _require_sha256(record["sha256"], f"{name} SHA-256")
    size = record["size_bytes"]
    if type(size) is not int or size <= 0:
        raise ValueError(f"{name} size is invalid")
    _, observed_sha256, observed_size = _snapshot(candidate, boundary=_absolute(PROJECT_ROOT))
    normalized = {"path": str(expected), "sha256": sha256, "size_bytes": size}
    if observed_sha256 != sha256 or observed_size != size:
        raise ValueError(f"{name} differs from its verified descriptor")
    return normalized


def _final_attempt_record(value: object, expected_path: Path, name: str) -> dict[str, Any]:
    record = _require_mapping(value, name)
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{name} descriptor fields are invalid")
    expected = _absolute(expected_path)
    attempt = _absolute(FINAL_EVALUATION_ATTEMPT_DIRECTORY)
    if not _is_within(expected, attempt) or expected == attempt:
        raise ValueError(f"{name} leaves the final evaluation attempt")
    expected_record_path = expected.relative_to(attempt).as_posix()
    if record["path"] != expected_record_path:
        raise ValueError(f"{name} path is not canonical")
    sha256 = _require_sha256(record["sha256"], f"{name} SHA-256")
    size = record["size_bytes"]
    if type(size) is not int or size <= 0:
        raise ValueError(f"{name} size is invalid")
    _, observed_sha256, observed_size = _snapshot(expected, boundary=attempt)
    if observed_sha256 != sha256 or observed_size != size:
        raise ValueError(f"{name} differs from its verified descriptor")
    return {"path": expected_record_path, "sha256": sha256, "size_bytes": size}


def _read_bound_json(
    record_value: object,
    expected_path: Path,
    *,
    boundary: Path,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    record = _external_record(record_value, expected_path, name)
    payload, observed_sha256, observed_size = _snapshot(expected_path, boundary=boundary)
    if observed_sha256 != record["sha256"] or observed_size != record["size_bytes"]:
        raise ValueError(f"{name} changed after descriptor verification")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not UTF-8 JSON") from exc
    return value, record


def _asset_record(name: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "media_type": _ASSET_MEDIA_TYPES[name],
    }


def _directory_entries() -> dict[str, str]:
    descriptor = _open_absolute_directory_no_follow(_absolute(FINAL_REPORT_ASSET_ROOT))
    try:
        with os.scandir(descriptor) as entries:
            return {
                entry.name: (
                    "file"
                    if entry.is_file(follow_symlinks=False) and not entry.is_symlink()
                    else "unsafe"
                )
                for entry in entries
            }
    finally:
        os.close(descriptor)


def _validate_entries(*, complete: bool) -> None:
    entries = _directory_entries()
    if not set(entries).issubset(_COMPLETE_ENTRIES) or any(
        kind != "file" for kind in entries.values()
    ):
        raise ValueError("Final report asset directory contains unexpected entries")
    if complete and set(entries) != _COMPLETE_ENTRIES:
        raise ValueError("Final report asset set is incomplete")


def _write_create_only(path: Path, payload: bytes) -> None:
    candidate = _absolute(path)
    root = _absolute(FINAL_REPORT_ASSET_ROOT)
    if candidate.parent != root or candidate.name not in _COMPLETE_ENTRIES:
        raise ValueError("Final report output path is not canonical")
    parent_descriptor = _open_absolute_directory_no_follow(root)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        os.close(parent_descriptor)
        raise RuntimeError("Final report writes require O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | no_follow
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate.name, flags, 0o644, dir_fd=parent_descriptor)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Final report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _write_or_verify(path: Path, payload: bytes) -> dict[str, Any]:
    if os.path.lexists(path):
        observed, sha256, size = _snapshot(path, boundary=_absolute(FINAL_REPORT_ASSET_ROOT))
        if observed != payload:
            raise ValueError(f"Final report artifact changed: {path.name}")
    else:
        _write_create_only(path, payload)
        observed, sha256, size = _snapshot(path, boundary=_absolute(FINAL_REPORT_ASSET_ROOT))
        if observed != payload:
            raise RuntimeError(f"Final report artifact publication failed: {path.name}")
    return {
        "path": path.name,
        "sha256": sha256,
        "size_bytes": size,
        **(
            {"media_type": _ASSET_MEDIA_TYPES[path.name]} if path.name in _ASSET_MEDIA_TYPES else {}
        ),
    }


def _float_text(value: object, name: str, *, unit: bool = False) -> str:
    return format(_require_finite(value, name, unit=unit), ".17g")


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _png_bytes(fig: plt.Figure) -> bytes:
    output = io.BytesIO()
    fig.savefig(
        output,
        format="png",
        dpi=_PNG_DPI,
        metadata={"Software": _PNG_SOFTWARE},
    )
    return output.getvalue()


def _seed_summary(value: object, metric_name: str, location: str) -> dict[str, Any]:
    summary = _require_mapping(value, location)
    if summary.get("metric_name") != metric_name or summary.get("seeds") != list(SEED_ORDER):
        raise ValueError(f"{location} seed identity is invalid")
    values = _require_sequence(summary.get("values"), f"{location} values")
    if len(values) != len(SEED_ORDER):
        raise ValueError(f"{location} does not contain three seed values")
    normalized_values = [
        _require_finite(item, f"{location} seed value", unit=True) for item in values
    ]
    mean = _require_finite(summary.get("mean"), f"{location} mean", unit=True)
    sd = _require_finite(
        summary.get("sample_standard_deviation"),
        f"{location} sample standard deviation",
    )
    if sd < 0.0 or summary.get("standard_deviation_ddof") != 1:
        raise ValueError(f"{location} stability definition is invalid")
    return {
        "metric_name": metric_name,
        "seeds": list(SEED_ORDER),
        "values": normalized_values,
        "mean": mean,
        "sample_standard_deviation": sd,
        "standard_deviation_ddof": 1,
    }


def _interval(value: object, location: str) -> dict[str, float]:
    record = _require_mapping(value, location)
    lower = _require_finite(record.get("lower"), f"{location} lower", unit=True)
    upper = _require_finite(record.get("upper"), f"{location} upper", unit=True)
    if lower > upper or record.get("confidence_level") != 0.95:
        raise ValueError(f"{location} interval is invalid")
    return {"lower": lower, "upper": upper, "confidence_level": 0.95}


def _verified_inputs() -> dict[str, Any]:
    final = _require_mapping(verify_final_evaluation(), "Final evaluation verification")
    gate_verification = _require_mapping(verify_final_evaluation_gate(), "Final gate verification")
    gate = _require_mapping(gate_verification.get("gate"), "Final gate")
    current_source = _require_sha256(source_fingerprint(), "Current source fingerprint")
    shared_identity = _require_mapping(gate.get("shared_identity"), "Gate shared identity")
    gate_source = _require_sha256(
        shared_identity.get("source_fingerprint_sha256"), "Gate source fingerprint"
    )
    final_source = _require_sha256(
        final.get("source_fingerprint_sha256"), "Final evaluation source fingerprint"
    )
    if current_source != gate_source or final_source != gate_source:
        raise PermissionError("Final report source fingerprint is not gate-bound")
    if final.get("seed_order") != list(SEED_ORDER):
        raise ValueError("Final evaluation seed order is invalid")

    final_result = _final_attempt_record(
        final.get("result_artifact"),
        FINAL_EVALUATION_RESULT_PATH,
        "Final evaluation result",
    )
    final_lock = _final_attempt_record(
        final.get("completion_lock_artifact"),
        FINAL_EVALUATION_LOCK_PATH,
        "Final evaluation completion lock",
    )
    gate_record = _external_record(
        gate_verification.get("gate_artifact"),
        FINAL_EVALUATION_GATE_PATH,
        "Final evaluation gate",
    )
    gate_lock = _external_record(
        gate_verification.get("lock_artifact"),
        FINAL_EVALUATION_GATE_LOCK_PATH,
        "Final evaluation gate lock",
    )
    if final.get("gate_sha256") != gate_record["sha256"]:
        raise ValueError("Final evaluation is bound to another gate")
    return {
        "final": dict(final),
        "gate": dict(gate),
        "source_fingerprint_sha256": current_source,
        "final_evaluation": {"result": final_result, "completion_lock": final_lock},
        "final_gate": {"gate": gate_record, "lock": gate_lock},
    }


def _assert_inputs_current(inputs: Mapping[str, Any]) -> None:
    if source_fingerprint() != inputs["source_fingerprint_sha256"]:
        raise PermissionError("Source fingerprint changed during final report publication")
    final_records = (
        (inputs["final_evaluation"]["result"], FINAL_EVALUATION_RESULT_PATH, "result"),
        (
            inputs["final_evaluation"]["completion_lock"],
            FINAL_EVALUATION_LOCK_PATH,
            "completion lock",
        ),
    )
    for record, path, name in final_records:
        observed = _final_attempt_record(record, path, f"Current final {name}")
        if observed != record:
            raise ValueError(f"Current final {name} descriptor changed")
    gate_records = (
        (inputs["final_gate"]["gate"], FINAL_EVALUATION_GATE_PATH, "gate"),
        (inputs["final_gate"]["lock"], FINAL_EVALUATION_GATE_LOCK_PATH, "gate lock"),
    )
    for record, path, name in gate_records:
        observed = _external_record(record, path, f"Current final {name}")
        if observed != record:
            raise ValueError(f"Current final {name} descriptor changed")


def _assert_history_sources_current(history_sources: Sequence[Mapping[str, Any]]) -> None:
    if len(history_sources) != 2 * len(SEED_ORDER):
        raise ValueError("Final report history source inventory is incomplete")
    expected_identities = tuple((task, seed) for task in ("task1", "task2") for seed in SEED_ORDER)
    observed_identities = tuple(
        (source.get("task"), source.get("seed")) for source in history_sources
    )
    if observed_identities != expected_identities:
        raise ValueError("Final report history source order is invalid")
    for source in history_sources:
        task = source.get("task")
        seed = source.get("seed")
        if task not in {"task1", "task2"} or seed not in SEED_ORDER:
            raise ValueError("Final report history source identity is invalid")
        for key in ("result", "epoch_history"):
            record = _require_mapping(
                source.get(key), f"Final report {task} seed {seed} {key} source"
            )
            path = record.get("path")
            if type(path) is not str:
                raise ValueError("Final report history source path is invalid")
            observed = _external_record(
                record,
                _absolute(path),
                f"Current {task} seed {seed} {key} source",
            )
            if observed != record:
                raise ValueError(f"Current {task} seed {seed} {key} descriptor changed")


def _task1_evidence(final: Mapping[str, Any]) -> dict[str, Any]:
    task = _require_mapping(final.get("task1_summary"), "Task 1 final summary")
    stability = _require_mapping(task.get("stability"), "Task 1 stability")
    if stability.get("seeds") != list(SEED_ORDER):
        raise ValueError("Task 1 stability seed order is invalid")
    summaries = {
        name: _seed_summary(stability.get(name), name, f"Task 1 {name} stability")
        for name in ("accuracy", "macro_f1")
    }
    detail = _require_mapping(task.get("seed_37_metrics"), "Task 1 seed 37 metrics")
    if detail.get("class_order") != list(LOCKED_TASK1_CLASS_ORDER):
        raise ValueError("Task 1 class order differs from the locked order")
    per_class = _require_sequence(detail.get("per_class"), "Task 1 per-class metrics")
    if len(per_class) != len(LOCKED_TASK1_CLASS_ORDER):
        raise ValueError("Task 1 per-class metrics are incomplete")
    normalized_per_class: list[dict[str, Any]] = []
    for index, (name, row_value) in enumerate(
        zip(LOCKED_TASK1_CLASS_ORDER, per_class, strict=True)
    ):
        row = _require_mapping(row_value, f"Task 1 class {index}")
        if row.get("class_index") != index or row.get("class_name") != name:
            raise ValueError("Task 1 per-class order is invalid")
        support = row.get("support")
        if type(support) is not int or support < 0:
            raise ValueError("Task 1 class support is invalid")
        normalized_per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "support": support,
                **{
                    metric: _require_finite(
                        row.get(metric), f"Task 1 class {index} {metric}", unit=True
                    )
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
    count = len(LOCKED_TASK1_CLASS_ORDER)
    confusion = _require_sequence(detail.get("confusion_counts"), "Task 1 confusion counts")
    normalized = _require_sequence(
        detail.get("row_normalized_confusion"), "Task 1 normalized confusion"
    )
    if len(confusion) != count or len(normalized) != count:
        raise ValueError("Task 1 confusion matrix dimensions are invalid")
    confusion_rows: list[list[int]] = []
    normalized_rows: list[list[float]] = []
    for row_index in range(count):
        count_row = _require_sequence(confusion[row_index], "Task 1 confusion row")
        normalized_row = _require_sequence(normalized[row_index], "Task 1 normalized row")
        if len(count_row) != count or len(normalized_row) != count:
            raise ValueError("Task 1 confusion matrix is not square")
        if any(type(item) is not int or item < 0 for item in count_row):
            raise ValueError("Task 1 confusion counts are invalid")
        confusion_rows.append(list(count_row))
        normalized_rows.append(
            [
                _require_finite(item, "Task 1 normalized confusion value", unit=True)
                for item in normalized_row
            ]
        )
    bootstrap = _require_mapping(task.get("seed_37_bootstrap"), "Task 1 bootstrap")
    if bootstrap.get("task1_seed") != DETAIL_SEED:
        raise ValueError("Task 1 bootstrap detail seed is invalid")
    class_intervals = _require_sequence(
        bootstrap.get("per_class_f1_intervals"), "Task 1 class intervals"
    )
    if len(class_intervals) != count:
        raise ValueError("Task 1 class intervals are incomplete")
    normalized_class_intervals: list[dict[str, Any]] = []
    for index, (name, value) in enumerate(
        zip(LOCKED_TASK1_CLASS_ORDER, class_intervals, strict=True)
    ):
        row = _require_mapping(value, f"Task 1 class {index} interval")
        if row.get("class_index") != index or row.get("class_name") != name:
            raise ValueError("Task 1 class interval order is invalid")
        normalized_class_intervals.append(_interval(row, f"Task 1 class {index} F1"))
    return {
        "summaries": summaries,
        "detail": {
            "recording_count": detail.get("recording_count"),
            "accuracy": _require_finite(detail.get("accuracy"), "Task 1 accuracy", unit=True),
            "macro_f1": _require_finite(detail.get("macro_f1"), "Task 1 macro F1", unit=True),
            "per_class": normalized_per_class,
            "confusion_counts": confusion_rows,
            "row_normalized_confusion": normalized_rows,
        },
        "bootstrap": {
            "accuracy_interval": _interval(bootstrap.get("accuracy_interval"), "Task 1 accuracy"),
            "macro_f1_interval": _interval(bootstrap.get("macro_f1_interval"), "Task 1 macro F1"),
            "per_class_f1_intervals": normalized_class_intervals,
        },
    }


def _task2_scope_summaries(value: object, *, location: str) -> dict[str, dict[str, Any]]:
    rows = _require_sequence(value, location)
    if len(rows) != len(TASK2_METRIC_ORDER):
        raise ValueError(f"{location} metric inventory is incomplete")
    result: dict[str, dict[str, Any]] = {}
    for metric, row in zip(TASK2_METRIC_ORDER, rows, strict=True):
        normalized = _seed_summary(row, metric, f"{location} {metric}")
        result[metric] = normalized
    return result


def _task2_point(value: object, location: str) -> dict[str, Any]:
    row = _require_mapping(value, location)
    return {
        metric: _require_finite(row.get(metric), f"{location} {metric}", unit=True)
        for metric in TASK2_METRIC_ORDER
    }


def _task2_binary_point(value: object, location: str) -> dict[str, Any]:
    row = _require_mapping(value, location)
    known_count = row.get("known_recording_count")
    unknown_count = row.get("unknown_recording_count")
    if any(type(count) is not int or count <= 0 for count in (known_count, unknown_count)):
        raise ValueError(f"{location} recording counts are invalid")
    return {
        **_task2_point(row, location),
        "known_recording_count": known_count,
        "unknown_recording_count": unknown_count,
    }


def _task2_stream(value: object, *, stream_name: str) -> dict[str, Any]:
    stream = _require_mapping(value, f"Task 2 {stream_name} summary")
    stability = _require_mapping(stream.get("stability"), f"Task 2 {stream_name} stability")
    if stability.get("seed_order") != list(SEED_ORDER):
        raise ValueError(f"Task 2 {stream_name} seed order is invalid")
    pooled_stability = _task2_scope_summaries(
        stability.get("pooled"), location=f"Task 2 {stream_name} pooled stability"
    )
    macro_stability = _task2_scope_summaries(
        stability.get("macro"), location=f"Task 2 {stream_name} macro stability"
    )
    species_stability_values = _require_sequence(
        stability.get("per_species"), f"Task 2 {stream_name} species stability"
    )
    species_stability: list[dict[str, Any]] = []
    species_names: list[str] = []
    for value_item in species_stability_values:
        item = _require_mapping(value_item, "Task 2 species stability item")
        species = item.get("species_scientific_name")
        if type(species) is not str or not species:
            raise ValueError("Task 2 species name is invalid")
        species_names.append(species)
        species_stability.append(
            {
                "species_scientific_name": species,
                "metrics": _task2_scope_summaries(
                    item.get("metrics"),
                    location=f"Task 2 {stream_name} {species} stability",
                ),
            }
        )
    if species_names != sorted(species_names) or len(set(species_names)) != len(species_names):
        raise ValueError("Task 2 species stability ordering is invalid")

    point = _require_mapping(
        stream.get("seed_37_point_estimates"), f"Task 2 {stream_name} point estimates"
    )
    threshold = _require_finite(point.get("threshold"), f"Task 2 {stream_name} threshold")
    if threshold < 0.0:
        raise ValueError(f"Task 2 {stream_name} threshold is negative")
    pooled_point = _task2_binary_point(point.get("pooled"), f"Task 2 {stream_name} pooled")
    macro_point = _task2_point(point.get("macro"), f"Task 2 {stream_name} macro")
    species_points_values = _require_sequence(
        point.get("per_species"), f"Task 2 {stream_name} species point estimates"
    )
    species_points: list[dict[str, Any]] = []
    for item_value in species_points_values:
        item = _require_mapping(item_value, "Task 2 species point estimate")
        species = item.get("species_scientific_name")
        if type(species) is not str or not species:
            raise ValueError("Task 2 point estimate species is invalid")
        known_count = item.get("known_recording_count")
        unknown_count = item.get("unknown_recording_count")
        if any(type(count) is not int or count <= 0 for count in (known_count, unknown_count)):
            raise ValueError("Task 2 point estimate counts are invalid")
        species_points.append(
            {
                "species_scientific_name": species,
                "known_recording_count": known_count,
                "unknown_recording_count": unknown_count,
                "metrics": _task2_point(item, f"Task 2 {stream_name} {species}"),
            }
        )
    if [item["species_scientific_name"] for item in species_points] != species_names:
        raise ValueError("Task 2 point estimate species differ from stability evidence")

    bootstrap = _require_mapping(stream.get("seed_37_bootstrap"), f"Task 2 {stream_name} bootstrap")
    pooled_intervals = {
        metric: _interval(
            _require_mapping(bootstrap.get("pooled_intervals"), "Task 2 pooled intervals").get(
                metric
            ),
            f"Task 2 {stream_name} pooled {metric}",
        )
        for metric in TASK2_METRIC_ORDER
    }
    macro_intervals = {
        metric: _interval(
            _require_mapping(bootstrap.get("macro_intervals"), "Task 2 macro intervals").get(
                metric
            ),
            f"Task 2 {stream_name} macro {metric}",
        )
        for metric in TASK2_METRIC_ORDER
    }
    interval_values = _require_sequence(
        bootstrap.get("per_species_intervals"), f"Task 2 {stream_name} species intervals"
    )
    species_intervals: list[dict[str, Any]] = []
    for item_value in interval_values:
        item = _require_mapping(item_value, "Task 2 species interval")
        species = item.get("species_scientific_name")
        species_intervals.append(
            {
                "species_scientific_name": species,
                "metrics": {
                    metric: _interval(item.get(metric), f"Task 2 {stream_name} {species} {metric}")
                    for metric in TASK2_METRIC_ORDER
                },
            }
        )
    if [item["species_scientific_name"] for item in species_intervals] != species_names:
        raise ValueError("Task 2 bootstrap species differ from point evidence")
    return {
        "score_name": stream.get("score_name"),
        "stability": {
            "pooled": pooled_stability,
            "macro": macro_stability,
            "per_species": species_stability,
        },
        "point": {
            "threshold": threshold,
            "pooled": pooled_point,
            "macro": macro_point,
            "per_species": species_points,
        },
        "intervals": {
            "pooled": pooled_intervals,
            "macro": macro_intervals,
            "per_species": species_intervals,
        },
    }


def _task2_evidence(final: Mapping[str, Any]) -> dict[str, Any]:
    task = _require_mapping(final.get("task2_summary"), "Task 2 final summary")
    result = {
        stream: _task2_stream(task.get(stream), stream_name=stream)
        for stream in ("reconstruction", "latent")
    }
    expected_names = {
        "reconstruction": RECONSTRUCTION_SCORE_NAME,
        "latent": LATENT_SCORE_NAME,
    }
    if any(result[name]["score_name"] != expected for name, expected in expected_names.items()):
        raise ValueError("Task 2 report score names differ from the locked score definitions")
    return result


def _history_evidence(gate: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"task1": [], "task2": [], "sources": []}
    for task_name in ("task1", "task2"):
        section = _require_mapping(gate.get(task_name), f"Gate {task_name} section")
        runs = _require_sequence(section.get("runs"), f"Gate {task_name} runs")
        if len(runs) != len(SEED_ORDER):
            raise ValueError(f"Gate {task_name} run inventory is incomplete")
        for seed, run_value in zip(SEED_ORDER, runs, strict=True):
            run = _require_mapping(run_value, f"Gate {task_name} seed {seed}")
            if run.get("seed") != seed:
                raise ValueError(f"Gate {task_name} seed order is invalid")
            run_directory_value = run.get("run_directory")
            if type(run_directory_value) is not str:
                raise ValueError(f"Gate {task_name} run directory is invalid")
            run_directory = _absolute(run_directory_value)
            if not _is_within(run_directory, _absolute(PROJECT_ROOT)):
                raise ValueError(f"Gate {task_name} run leaves the project")
            run_result, run_result_record = _read_bound_json(
                run.get("result"),
                run_directory / "result.json",
                boundary=_absolute(PROJECT_ROOT),
                name=f"Gate {task_name} seed {seed} result",
            )
            run_result_mapping = _require_mapping(
                run_result, f"Gate {task_name} seed {seed} result value"
            )
            artifacts = _require_mapping(
                run_result_mapping.get("artifacts"),
                f"Gate {task_name} seed {seed} artifacts",
            )
            history, history_record = _read_bound_json(
                artifacts.get("epoch_history"),
                run_directory / "epoch_history.json",
                boundary=_absolute(PROJECT_ROOT),
                name=f"Gate {task_name} seed {seed} epoch history",
            )
            rows = _require_sequence(history, f"Gate {task_name} seed {seed} history")
            if not rows:
                raise ValueError(f"Gate {task_name} seed {seed} history is empty")
            normalized_rows: list[dict[str, Any]] = []
            for expected_epoch, row_value in enumerate(rows, start=1):
                row = _require_mapping(
                    row_value, f"Gate {task_name} seed {seed} epoch {expected_epoch}"
                )
                if (
                    row.get("epoch") != expected_epoch
                    or type(row.get("checkpoint_improved")) is not bool
                ):
                    raise ValueError(f"Gate {task_name} history order is invalid")
                train = _require_mapping(row.get("train"), f"{task_name} training row")
                validation = _require_mapping(row.get("validation"), f"{task_name} validation row")
                if task_name == "task1":
                    normalized_rows.append(
                        {
                            "seed": seed,
                            "epoch": expected_epoch,
                            "elapsed_seconds": _require_finite(
                                row.get("elapsed_seconds"), "Task 1 epoch time"
                            ),
                            "training_loss": _require_finite(
                                train.get("clip_loss"), "Task 1 training loss"
                            ),
                            "validation_loss": _require_finite(
                                validation.get("clip_loss"), "Task 1 validation loss"
                            ),
                            "validation_accuracy": _require_finite(
                                validation.get("accuracy"),
                                "Task 1 validation accuracy",
                                unit=True,
                            ),
                            "validation_macro_f1": _require_finite(
                                validation.get("macro_f1"),
                                "Task 1 validation macro F1",
                                unit=True,
                            ),
                            "checkpoint_improved": row["checkpoint_improved"],
                        }
                    )
                else:
                    normalized_rows.append(
                        {
                            "seed": seed,
                            "epoch": expected_epoch,
                            "training_loss": _require_finite(
                                train.get("loss"), "Task 2 training loss"
                            ),
                            "validation_loss": _require_finite(
                                validation.get("loss"), "Task 2 validation loss"
                            ),
                            "checkpoint_improved": row["checkpoint_improved"],
                        }
                    )
            result[task_name].extend(normalized_rows)
            result["sources"].append(
                {
                    "task": task_name,
                    "seed": seed,
                    "result": run_result_record,
                    "epoch_history": history_record,
                }
            )
    return result


def _task1_payloads(evidence: Mapping[str, Any]) -> dict[str, bytes]:
    summaries = evidence["summaries"]
    detail = evidence["detail"]
    bootstrap = evidence["bootstrap"]
    seed_rows = []
    for index, seed in enumerate(SEED_ORDER):
        seed_rows.append(
            [
                seed,
                _float_text(summaries["accuracy"]["values"][index], "Task 1 accuracy"),
                _float_text(summaries["macro_f1"]["values"][index], "Task 1 macro F1"),
            ]
        )
    payloads = {"task1_seed_metrics.csv": _csv_bytes(("seed", "accuracy", "macro_f1"), seed_rows)}
    stability_rows = []
    for metric in ("accuracy", "macro_f1"):
        summary = summaries[metric]
        interval = bootstrap[f"{metric}_interval"]
        stability_rows.append(
            [
                metric,
                *(_float_text(value, f"Task 1 {metric} seed value") for value in summary["values"]),
                _float_text(summary["mean"], f"Task 1 {metric} mean"),
                _float_text(summary["sample_standard_deviation"], f"Task 1 {metric} SD"),
                1,
                _float_text(interval["lower"], f"Task 1 {metric} lower"),
                _float_text(interval["upper"], f"Task 1 {metric} upper"),
                _float_text(interval["confidence_level"], "Task 1 confidence"),
            ]
        )
    payloads["task1_stability.csv"] = _csv_bytes(
        (
            "metric",
            "seed_13",
            "seed_37",
            "seed_71",
            "mean",
            "sample_standard_deviation",
            "standard_deviation_ddof",
            "seed_37_interval_lower",
            "seed_37_interval_upper",
            "confidence_level",
        ),
        stability_rows,
    )
    per_class_rows = []
    for metric_row, interval in zip(
        detail["per_class"], bootstrap["per_class_f1_intervals"], strict=True
    ):
        per_class_rows.append(
            [
                metric_row["class_index"],
                metric_row["class_name"],
                metric_row["support"],
                _float_text(metric_row["precision"], "Task 1 precision"),
                _float_text(metric_row["recall"], "Task 1 recall"),
                _float_text(metric_row["f1"], "Task 1 F1"),
                _float_text(interval["lower"], "Task 1 class F1 lower"),
                _float_text(interval["upper"], "Task 1 class F1 upper"),
                _float_text(interval["confidence_level"], "Task 1 class confidence"),
            ]
        )
    payloads["task1_seed37_per_class.csv"] = _csv_bytes(
        (
            "class_index",
            "class_name",
            "support",
            "precision",
            "recall",
            "f1",
            "f1_interval_lower",
            "f1_interval_upper",
            "confidence_level",
        ),
        per_class_rows,
    )
    matrix_header = ("true_class", *LOCKED_TASK1_CLASS_ORDER)
    payloads["task1_confusion_counts.csv"] = _csv_bytes(
        matrix_header,
        [
            [name, *row]
            for name, row in zip(LOCKED_TASK1_CLASS_ORDER, detail["confusion_counts"], strict=True)
        ],
    )
    payloads["task1_confusion_row_normalized.csv"] = _csv_bytes(
        matrix_header,
        [
            [
                name,
                *(_float_text(value, "Task 1 normalized confusion", unit=True) for value in row),
            ]
            for name, row in zip(
                LOCKED_TASK1_CLASS_ORDER,
                detail["row_normalized_confusion"],
                strict=True,
            )
        ],
    )
    with matplotlib.rc_context(_RC_PARAMS):
        fig, axis = plt.subplots(figsize=(12.5, 10.5), constrained_layout=False)
        try:
            image = axis.imshow(
                np.asarray(detail["row_normalized_confusion"], dtype=np.float64),
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            axis.grid(False)
            axis.set_title("Task 1 seed 37 row-normalized confusion")
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            positions = np.arange(len(LOCKED_TASK1_CLASS_ORDER))
            axis.set_xticks(positions, labels=LOCKED_TASK1_CLASS_ORDER, rotation=55, ha="right")
            axis.set_yticks(positions, labels=LOCKED_TASK1_CLASS_ORDER)
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Row proportion")
            fig.subplots_adjust(left=0.24, bottom=0.25, right=0.9, top=0.94)
            payloads["task1_confusion_heatmap.png"] = _png_bytes(fig)
        finally:
            plt.close(fig)
        fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=False)
        try:
            axis.plot(
                SEED_ORDER,
                summaries["accuracy"]["values"],
                color="#2563eb",
                marker="o",
                linewidth=2.0,
                label="Accuracy",
            )
            axis.plot(
                SEED_ORDER,
                summaries["macro_f1"]["values"],
                color="#16a34a",
                marker="s",
                linewidth=2.0,
                label="Macro F1",
            )
            axis.set_title("Task 1 performance across fixed seeds")
            axis.set_xlabel("Seed")
            axis.set_ylabel("Metric value")
            axis.set_xticks(SEED_ORDER)
            axis.set_ylim(0.0, 1.0)
            axis.legend(loc="best")
            fig.subplots_adjust(left=0.11, bottom=0.14, right=0.97, top=0.9)
            payloads["task1_seed_stability.png"] = _png_bytes(fig)
        finally:
            plt.close(fig)
    return payloads


def _task2_payloads(evidence: Mapping[str, Any]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    seed_rows: list[list[object]] = []
    stability_rows: list[list[object]] = []
    for stream_name in ("reconstruction", "latent"):
        stream = evidence[stream_name]
        for scope in ("pooled", "macro"):
            summaries = stream["stability"][scope]
            for seed_index, seed in enumerate(SEED_ORDER):
                seed_rows.append(
                    [
                        stream_name,
                        seed,
                        scope,
                        *(
                            _float_text(
                                summaries[metric]["values"][seed_index],
                                f"Task 2 {stream_name} {scope} {metric}",
                            )
                            for metric in TASK2_METRIC_ORDER
                        ),
                    ]
                )
            for metric in TASK2_METRIC_ORDER:
                summary = summaries[metric]
                stability_rows.append(
                    [
                        stream_name,
                        scope,
                        "",
                        metric,
                        *(
                            _float_text(value, f"Task 2 {metric} seed value")
                            for value in summary["values"]
                        ),
                        _float_text(summary["mean"], f"Task 2 {metric} mean"),
                        _float_text(summary["sample_standard_deviation"], f"Task 2 {metric} SD"),
                        1,
                    ]
                )
        for species_item in stream["stability"]["per_species"]:
            species = species_item["species_scientific_name"]
            for metric in TASK2_METRIC_ORDER:
                summary = species_item["metrics"][metric]
                stability_rows.append(
                    [
                        stream_name,
                        "per_species",
                        species,
                        metric,
                        *(
                            _float_text(value, f"Task 2 {species} {metric} seed value")
                            for value in summary["values"]
                        ),
                        _float_text(summary["mean"], f"Task 2 {species} {metric} mean"),
                        _float_text(
                            summary["sample_standard_deviation"],
                            f"Task 2 {species} {metric} SD",
                        ),
                        1,
                    ]
                )
    payloads["task2_seed_metrics.csv"] = _csv_bytes(
        ("score_stream", "seed", "scope", *TASK2_METRIC_ORDER), seed_rows
    )
    payloads["task2_stability.csv"] = _csv_bytes(
        (
            "score_stream",
            "scope",
            "species_scientific_name",
            "metric",
            "seed_13",
            "seed_37",
            "seed_71",
            "mean",
            "sample_standard_deviation",
            "standard_deviation_ddof",
        ),
        stability_rows,
    )
    interval_rows: list[list[object]] = []
    for stream_name in ("reconstruction", "latent"):
        stream = evidence[stream_name]
        for scope in ("pooled", "macro"):
            for metric in TASK2_METRIC_ORDER:
                point = stream["point"][scope][metric]
                interval = stream["intervals"][scope][metric]
                interval_rows.append(
                    [
                        stream_name,
                        _float_text(stream["point"]["threshold"], "Task 2 threshold"),
                        scope,
                        "",
                        stream["point"][scope].get("known_recording_count", ""),
                        stream["point"][scope].get("unknown_recording_count", ""),
                        metric,
                        _float_text(point, "Task 2 point estimate"),
                        _float_text(interval["lower"], "Task 2 interval lower"),
                        _float_text(interval["upper"], "Task 2 interval upper"),
                        _float_text(interval["confidence_level"], "Task 2 confidence"),
                    ]
                )
        for point_item, interval_item in zip(
            stream["point"]["per_species"],
            stream["intervals"]["per_species"],
            strict=True,
        ):
            species = point_item["species_scientific_name"]
            for metric in TASK2_METRIC_ORDER:
                interval = interval_item["metrics"][metric]
                interval_rows.append(
                    [
                        stream_name,
                        _float_text(stream["point"]["threshold"], "Task 2 threshold"),
                        "per_species",
                        species,
                        point_item["known_recording_count"],
                        point_item["unknown_recording_count"],
                        metric,
                        _float_text(point_item["metrics"][metric], "Task 2 species metric"),
                        _float_text(interval["lower"], "Task 2 species interval lower"),
                        _float_text(interval["upper"], "Task 2 species interval upper"),
                        _float_text(interval["confidence_level"], "Task 2 confidence"),
                    ]
                )
    payloads["task2_seed37_metrics_intervals.csv"] = _csv_bytes(
        (
            "score_stream",
            "threshold",
            "scope",
            "species_scientific_name",
            "known_recording_count",
            "unknown_recording_count",
            "metric",
            "point_estimate",
            "interval_lower",
            "interval_upper",
            "confidence_level",
        ),
        interval_rows,
    )

    reconstruction = evidence["reconstruction"]
    species_points = reconstruction["point"]["per_species"]
    species_intervals = reconstruction["intervals"]["per_species"]
    with matplotlib.rc_context(_RC_PARAMS):
        fig, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=False)
        try:
            species_names = [item["species_scientific_name"] for item in species_points]
            point_values = np.asarray(
                [item["metrics"]["auroc"] for item in species_points], dtype=np.float64
            )
            lowers = np.asarray(
                [item["metrics"]["auroc"]["lower"] for item in species_intervals],
                dtype=np.float64,
            )
            uppers = np.asarray(
                [item["metrics"]["auroc"]["upper"] for item in species_intervals],
                dtype=np.float64,
            )
            positions = np.arange(len(species_names))
            axis.hlines(positions, lowers, uppers, color="#60a5fa", linewidth=2.0)
            axis.vlines(lowers, positions - 0.08, positions + 0.08, color="#60a5fa")
            axis.vlines(uppers, positions - 0.08, positions + 0.08, color="#60a5fa")
            axis.scatter(point_values, positions, color="#1d4ed8", s=32, zorder=3)
            axis.set_yticks(positions, labels=species_names)
            axis.invert_yaxis()
            axis.set_xlim(0.0, 1.0)
            axis.set_xlabel("AUROC")
            axis.set_ylabel("Unknown species")
            axis.set_title("Task 2 reconstruction score, seed 37 AUROC intervals")
            fig.subplots_adjust(left=0.3, bottom=0.14, right=0.97, top=0.9)
            payloads["task2_reconstruction_species_auroc_intervals.png"] = _png_bytes(fig)
        finally:
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=False)
        try:
            for axis, metric in zip(axes.flat, TASK2_METRIC_ORDER, strict=True):
                axis.plot(
                    SEED_ORDER,
                    reconstruction["stability"]["pooled"][metric]["values"],
                    color="#7c3aed",
                    marker="o",
                    linewidth=2.0,
                    label="Pooled",
                )
                axis.plot(
                    SEED_ORDER,
                    reconstruction["stability"]["macro"][metric]["values"],
                    color="#ea580c",
                    marker="s",
                    linewidth=2.0,
                    label="Macro",
                )
                axis.set_title(metric.replace("_", " ").title())
                axis.set_xlabel("Seed")
                axis.set_ylabel("Metric value")
                axis.set_xticks(SEED_ORDER)
                axis.set_ylim(0.0, 1.0)
                axis.legend(loc="best")
            fig.suptitle("Task 2 reconstruction score across fixed seeds", fontsize=13)
            fig.subplots_adjust(left=0.09, bottom=0.09, right=0.98, top=0.9, hspace=0.38)
            payloads["task2_reconstruction_seed_stability.png"] = _png_bytes(fig)
        finally:
            plt.close(fig)
    return payloads


def _history_payloads(history: Mapping[str, Any]) -> dict[str, bytes]:
    task1_rows = history["task1"]
    task2_rows = history["task2"]
    payloads = {
        "task1_training_history.csv": _csv_bytes(
            (
                "seed",
                "epoch",
                "elapsed_seconds",
                "training_loss",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_f1",
                "checkpoint_improved",
            ),
            [
                [
                    row["seed"],
                    row["epoch"],
                    _float_text(row["elapsed_seconds"], "Task 1 elapsed seconds"),
                    _float_text(row["training_loss"], "Task 1 training loss"),
                    _float_text(row["validation_loss"], "Task 1 validation loss"),
                    _float_text(
                        row["validation_accuracy"], "Task 1 validation accuracy", unit=True
                    ),
                    _float_text(
                        row["validation_macro_f1"], "Task 1 validation macro F1", unit=True
                    ),
                    str(row["checkpoint_improved"]).lower(),
                ]
                for row in task1_rows
            ],
        ),
        "task2_training_history.csv": _csv_bytes(
            (
                "seed",
                "epoch",
                "training_loss",
                "validation_loss",
                "checkpoint_improved",
            ),
            [
                [
                    row["seed"],
                    row["epoch"],
                    _float_text(row["training_loss"], "Task 2 training loss"),
                    _float_text(row["validation_loss"], "Task 2 validation loss"),
                    str(row["checkpoint_improved"]).lower(),
                ]
                for row in task2_rows
            ],
        ),
    }
    with matplotlib.rc_context(_RC_PARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=False)
        try:
            for seed, color in zip(SEED_ORDER, ("#2563eb", "#16a34a", "#dc2626"), strict=True):
                rows = [row for row in task1_rows if row["seed"] == seed]
                epochs = [row["epoch"] for row in rows]
                axes[0].plot(
                    epochs,
                    [row["validation_macro_f1"] for row in rows],
                    color=color,
                    marker="o",
                    linewidth=1.7,
                    label=f"Seed {seed}",
                )
                axes[1].plot(
                    epochs,
                    [row["training_loss"] for row in rows],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Seed {seed} training",
                )
                axes[1].plot(
                    epochs,
                    [row["validation_loss"] for row in rows],
                    color=color,
                    linewidth=1.8,
                    label=f"Seed {seed} validation",
                )
            axes[0].set_title("Validation macro F1")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Macro F1")
            axes[0].set_ylim(0.0, 1.0)
            axes[0].legend(loc="best")
            axes[1].set_title("Training history loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend(loc="best", fontsize=7)
            fig.suptitle("Task 1 gate-bound training history", fontsize=13)
            fig.subplots_adjust(left=0.08, bottom=0.14, right=0.98, top=0.86, wspace=0.28)
            payloads["task1_training_history.png"] = _png_bytes(fig)
        finally:
            plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=False)
        try:
            for axis, seed, color in zip(
                axes, SEED_ORDER, ("#2563eb", "#16a34a", "#dc2626"), strict=True
            ):
                rows = [row for row in task2_rows if row["seed"] == seed]
                epochs = [row["epoch"] for row in rows]
                axis.plot(
                    epochs,
                    [row["training_loss"] for row in rows],
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                    label="Training",
                )
                axis.plot(
                    epochs,
                    [row["validation_loss"] for row in rows],
                    color=color,
                    linewidth=1.9,
                    label="Validation",
                )
                axis.set_title(f"Seed {seed}")
                axis.set_xlabel("Epoch")
                axis.set_ylabel("Reconstruction loss")
                axis.legend(loc="best")
            fig.suptitle("Task 2 gate-bound training history", fontsize=13)
            fig.subplots_adjust(left=0.07, bottom=0.14, right=0.98, top=0.84, wspace=0.3)
            payloads["task2_training_history.png"] = _png_bytes(fig)
        finally:
            plt.close(fig)
    return payloads


def _asset_payloads(inputs: Mapping[str, Any]) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    task1 = _task1_evidence(inputs["final"])
    task2 = _task2_evidence(inputs["final"])
    history = _history_evidence(inputs["gate"])
    payloads = {
        **_task1_payloads(task1),
        **_task2_payloads(task2),
        **_history_payloads(history),
    }
    if set(payloads) != set(_ASSET_MEDIA_TYPES):
        raise RuntimeError("Final report asset generator returned an invalid inventory")
    for name, payload in payloads.items():
        if not payload:
            raise RuntimeError(f"Final report asset is empty: {name}")
        if name.endswith(".png") and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"Final report PNG signature is invalid: {name}")
    return payloads, history["sources"]


def _manifest_value(
    inputs: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    history_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    assets = [_asset_record(name, payloads[name]) for name in sorted(payloads)]
    return {
        "schema_version": FINAL_REPORT_ASSETS_SCHEMA_VERSION,
        "asset_set_id": FINAL_REPORT_ASSET_SET_ID,
        "complete": True,
        "source_fingerprint_sha256": inputs["source_fingerprint_sha256"],
        "seed_order": list(SEED_ORDER),
        "detail_seed": DETAIL_SEED,
        "final_evaluation": inputs["final_evaluation"],
        "final_evaluation_gate": inputs["final_gate"],
        "training_history_sources": [dict(item) for item in history_sources],
        "asset_count": len(assets),
        "assets": assets,
    }


def _lock_value(manifest_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FINAL_REPORT_ASSETS_SCHEMA_VERSION,
        "asset_set_id": FINAL_REPORT_ASSET_SET_ID,
        "manifest": dict(manifest_record),
    }


def _verify_locked_assets(
    *,
    inputs: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    history_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _assert_inputs_current(inputs)
    _assert_history_sources_current(history_sources)
    _validate_entries(complete=True)
    expected_records: list[dict[str, Any]] = []
    for name in sorted(payloads):
        payload, sha256, size = _snapshot(
            _absolute(FINAL_REPORT_ASSET_ROOT) / name,
            boundary=_absolute(FINAL_REPORT_ASSET_ROOT),
        )
        if payload != payloads[name]:
            raise ValueError(f"Final report evidence changed: {name}")
        expected_records.append(
            {
                "path": name,
                "sha256": sha256,
                "size_bytes": size,
                "media_type": _ASSET_MEDIA_TYPES[name],
            }
        )
    manifest_payload, manifest_sha256, manifest_size = _snapshot(
        _absolute(FINAL_REPORT_MANIFEST_PATH), boundary=_absolute(FINAL_REPORT_ASSET_ROOT)
    )
    try:
        observed_manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Final report manifest is not UTF-8 JSON") from exc
    expected_manifest = _manifest_value(inputs, payloads, history_sources)
    if (
        observed_manifest != expected_manifest
        or manifest_payload != _canonical_json_bytes(expected_manifest)
        or expected_manifest["assets"] != expected_records
    ):
        raise ValueError("Final report manifest differs from verified evidence")
    manifest_record = {
        "path": "manifest.json",
        "sha256": manifest_sha256,
        "size_bytes": manifest_size,
    }
    lock_payload, lock_sha256, lock_size = _snapshot(
        _absolute(FINAL_REPORT_LOCK_PATH), boundary=_absolute(FINAL_REPORT_ASSET_ROOT)
    )
    try:
        observed_lock = json.loads(lock_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Final report lock is not UTF-8 JSON") from exc
    expected_lock = _lock_value(manifest_record)
    if observed_lock != expected_lock or lock_payload != _canonical_json_bytes(expected_lock):
        raise ValueError("Final report lock differs from its manifest")
    _assert_history_sources_current(history_sources)
    _assert_inputs_current(inputs)
    return {
        "manifest": expected_manifest,
        "manifest_artifact": manifest_record,
        "lock_artifact": {
            "path": "lock.json",
            "sha256": lock_sha256,
            "size_bytes": lock_size,
        },
        "assets": expected_records,
        "created": False,
    }


def _transaction_descriptor(*, exclusive: bool, create_parent: bool) -> int:
    parent = _absolute(FINAL_REPORT_ASSET_ROOT).parent
    if not _is_within(parent, _absolute(PROJECT_ROOT)):
        raise ValueError("Final report transaction directory leaves the project")
    if create_parent:
        parent = _secure_ensure_directory(parent, PROJECT_ROOT)
    descriptor = _open_absolute_directory_no_follow(parent)
    fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    return descriptor


def verify_final_report_assets() -> dict[str, Any]:
    """Recursively verify the fixed final report evidence set without model execution."""

    descriptor = _transaction_descriptor(exclusive=False, create_parent=False)
    try:
        inputs = _verified_inputs()
        payloads, history_sources = _asset_payloads(inputs)
        return _verify_locked_assets(
            inputs=inputs,
            payloads=payloads,
            history_sources=history_sources,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def build_final_report_assets() -> dict[str, Any]:
    """Build or verify the one fixed, gate-bound final report evidence set."""

    descriptor = _transaction_descriptor(exclusive=True, create_parent=True)
    try:
        root = _secure_ensure_directory(_absolute(FINAL_REPORT_ASSET_ROOT), PROJECT_ROOT)
        _validate_entries(complete=False)
        inputs = _verified_inputs()
        payloads, history_sources = _asset_payloads(inputs)
        lock_exists = os.path.lexists(_absolute(FINAL_REPORT_LOCK_PATH))
        manifest_exists = os.path.lexists(_absolute(FINAL_REPORT_MANIFEST_PATH))
        if lock_exists:
            if not manifest_exists:
                raise ValueError("Final report lock exists without its manifest")
            return _verify_locked_assets(
                inputs=inputs,
                payloads=payloads,
                history_sources=history_sources,
            )
        for name in sorted(payloads):
            _write_or_verify(root / name, payloads[name])
        _assert_history_sources_current(history_sources)
        _assert_inputs_current(inputs)
        manifest = _manifest_value(inputs, payloads, history_sources)
        manifest_record = _write_or_verify(
            _absolute(FINAL_REPORT_MANIFEST_PATH), _canonical_json_bytes(manifest)
        )
        _assert_history_sources_current(history_sources)
        _assert_inputs_current(inputs)
        _write_or_verify(
            _absolute(FINAL_REPORT_LOCK_PATH),
            _canonical_json_bytes(_lock_value(manifest_record)),
        )
        verified = _verify_locked_assets(
            inputs=inputs,
            payloads=payloads,
            history_sources=history_sources,
        )
        return {**verified, "created": True}
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
