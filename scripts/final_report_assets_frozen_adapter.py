from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from bird_audio import final_evaluation_gate as gate_runtime
from bird_audio import final_report_assets as native
from bird_audio.paths import PROJECT_ROOT
from bird_audio.provenance import source_fingerprint

RECOVERY_SCHEMA_VERSION = "1.0"
RECOVERY_ID = "final_report_metric_order_adapter_v1"
RECOVERY_ROOT = PROJECT_ROOT / "evidence" / "recovery" / RECOVERY_ID
RECOVERY_MANIFEST_PATH = RECOVERY_ROOT / "manifest.json"
RECOVERY_LOCK_PATH = RECOVERY_ROOT / "lock.json"

OUTPUT_ASSET_SET_ID = "final_report_assets_v2_metric_order_recovery_v1"
OUTPUT_ROOT = PROJECT_ROOT / "report_assets" / "final_v2_metric_order_recovery_v1"
OUTPUT_MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
OUTPUT_LOCK_PATH = OUTPUT_ROOT / "lock.json"

ADAPTER_PATH = PROJECT_ROOT / "scripts" / "final_report_assets_frozen_adapter.py"
RENDERER_PATH = PROJECT_ROOT / "src" / "bird_audio" / "final_report_assets.py"
PRODUCER_PATH = PROJECT_ROOT / "src" / "bird_audio" / "task2_metrics.py"
GATE_PATH = PROJECT_ROOT / "runs" / "final_evaluation_v2" / "gate_v2" / "gate.json"
GATE_LOCK_PATH = PROJECT_ROOT / "runs" / "final_evaluation_v2" / "gate_v2" / "lock.json"
FINAL_RESULT_PATH = PROJECT_ROOT / "runs" / "final_evaluation_v2" / "attempt_v2" / "result.json"
FINAL_LOCK_PATH = PROJECT_ROOT / "runs" / "final_evaluation_v2" / "attempt_v2" / "lock.json"
SUMMARY_RESULT_PATH = (
    PROJECT_ROOT / "runs" / "final_evaluation_v2" / "attempt_v2" / "summary" / "result.json"
)
SUMMARY_LOCK_PATH = (
    PROJECT_ROOT / "runs" / "final_evaluation_v2" / "attempt_v2" / "summary" / "lock.json"
)

FROZEN_SOURCE_FINGERPRINT = "180d979be32e4e44d6879e6e5cfe34dd348a1010e6303714ccf52ab0340c260c"
PRESENTATION_METRIC_ORDER = (
    "auroc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
)
PRODUCER_METRIC_ORDER = (
    "auroc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)
SEED_ORDER = (13, 37, 71)

_ORIGINAL_ASSET_SET_ID = "final_report_assets_v2"
_ORIGINAL_OUTPUT_ROOT = PROJECT_ROOT / "report_assets" / "final_v2"
_SHA256_LENGTH = 64
_EVIDENCE_FILENAMES = frozenset({"manifest.json", "lock.json"})

_PINNED_SHA256 = {
    "renderer": "d81882356bdddaf7c489295f883eaa46f756f9b374b908632bca6467ca50edfb",
    "producer": "c54c5b6bed04e908fc43f9cd173a30e7df02cbaae267feb24632ab595b05d2cd",
    "gate": "c7a2175186750d6a8d3687f0a8e5c5ce9b79c3c1f8db99f9fb6545e23505d46b",
    "gate_lock": "a5da29ead082abc8c94986fcf84880b4eac63758d34449882b6234c000fd82e8",
    "final_result": "48d3f1b16b3dd81e48c4ebb4f15551876d55232fb8e7949460d24a7617b63cd5",
    "final_lock": "3042fc95a8389e82437ab9f35489f4536a0de2ec97c012de190073e8c6aac65a",
    "summary_result": "f5c8203b43205c415503ffb049e0a151b50df216d101f1ac95d4048265104951",
    "summary_lock": "f3302206ec737d669a965fe0e4b0912c4616f12e6db5027f9f023e75120d5240",
}

_PINNED_PATHS = {
    "renderer": RENDERER_PATH,
    "producer": PRODUCER_PATH,
    "gate": GATE_PATH,
    "gate_lock": GATE_LOCK_PATH,
    "final_result": FINAL_RESULT_PATH,
    "final_lock": FINAL_LOCK_PATH,
    "summary_result": SUMMARY_RESULT_PATH,
    "summary_lock": SUMMARY_LOCK_PATH,
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
        raise ValueError("Recovery evidence is not finite canonical JSON") from exc


def _project_relative(path: Path) -> str:
    candidate = _absolute(path)
    root = _absolute(PROJECT_ROOT)
    if candidate == root or not _is_within(candidate, root):
        raise ValueError("Recovery artifact leaves the project root")
    return candidate.relative_to(root).as_posix()


def _artifact_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    candidate = _absolute(path)
    payload, observed_sha256, size_bytes = native._snapshot(
        candidate,
        boundary=_absolute(PROJECT_ROOT),
    )
    if not payload or size_bytes != len(payload):
        raise ValueError(f"Recovery artifact is empty: {_project_relative(candidate)}")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(f"Frozen artifact changed: {_project_relative(candidate)}")
    return {
        "path": _project_relative(candidate),
        "sha256": observed_sha256,
        "size_bytes": size_bytes,
    }


def _require_adapter_current(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise PermissionError("Recovery adapter record is invalid")
    expected_path = _project_relative(ADAPTER_PATH)
    if (
        value.get("path") != expected_path
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
    ):
        raise PermissionError("Recovery adapter identity is invalid")
    current = _artifact_record(ADAPTER_PATH)
    if dict(value) != current:
        raise PermissionError("Recovery adapter changed during report recovery")
    return current


def _read_canonical_json(path: Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _absolute(path)
    payload, observed_sha256, size_bytes = native._snapshot(
        candidate,
        boundary=_absolute(PROJECT_ROOT),
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not UTF-8 JSON") from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise ValueError(f"{name} is not canonical JSON")
    return value, {
        "path": _project_relative(candidate),
        "sha256": observed_sha256,
        "size_bytes": size_bytes,
    }


def _normalize_task2_scope_summaries(
    value: object,
    *,
    location: str,
) -> dict[str, dict[str, Any]]:
    rows = native._require_sequence(value, location)
    if len(rows) != len(PRESENTATION_METRIC_ORDER):
        raise ValueError(f"{location} metric inventory is incomplete")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, row_value in enumerate(rows):
        row = native._require_mapping(row_value, f"{location} row {index}")
        metric_name = row.get("metric_name")
        if type(metric_name) is not str or metric_name not in PRESENTATION_METRIC_ORDER:
            raise ValueError(f"{location} contains an unknown metric identity")
        if metric_name in by_name:
            raise ValueError(f"{location} contains a duplicate metric identity")
        seeds = row.get("seeds")
        if (
            type(seeds) is not list
            or len(seeds) != len(SEED_ORDER)
            or any(type(seed) is not int for seed in seeds)
            or tuple(seeds) != SEED_ORDER
        ):
            raise ValueError(f"{location} contains an invalid seed identity")
        standard_deviation_ddof = row.get("standard_deviation_ddof")
        if type(standard_deviation_ddof) is not int or standard_deviation_ddof != 1:
            raise ValueError(f"{location} contains an invalid stability definition")
        by_name[metric_name] = row
    if set(by_name) != set(PRESENTATION_METRIC_ORDER):
        raise ValueError(f"{location} metric inventory is incomplete")
    return {
        metric_name: native._seed_summary(
            by_name[metric_name],
            metric_name,
            f"{location} {metric_name}",
        )
        for metric_name in PRESENTATION_METRIC_ORDER
    }


def _test_row(metric_name: str, offset: int) -> dict[str, Any]:
    values = [0.20 + offset * 0.10, 0.21 + offset * 0.10, 0.22 + offset * 0.10]
    return {
        "metric_name": metric_name,
        "seeds": list(SEED_ORDER),
        "values": values,
        "mean": sum(values) / len(values),
        "sample_standard_deviation": 0.01,
        "standard_deviation_ddof": 1,
    }


def _expect_value_error(callback: Any, name: str) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(f"Self-test did not reject {name}")


def _expect_permission_error(callback: Any, name: str) -> None:
    try:
        callback()
    except PermissionError:
        return
    raise AssertionError(f"Self-test did not reject {name}")


def run_self_tests() -> dict[str, Any]:
    rows = [_test_row(name, index) for index, name in enumerate(PRODUCER_METRIC_ORDER)]
    original_payload = _canonical_json_bytes(rows)
    normalized = _normalize_task2_scope_summaries(rows, location="self-test")
    if tuple(normalized) != PRESENTATION_METRIC_ORDER:
        raise AssertionError("Self-test presentation order is invalid")
    if _canonical_json_bytes(rows) != original_payload:
        raise AssertionError("Self-test input rows were mutated")
    expected_first_values = {row["metric_name"]: row["values"] for row in rows}
    if any(
        normalized[name]["values"] != expected_first_values[name]
        for name in PRESENTATION_METRIC_ORDER
    ):
        raise AssertionError("Self-test changed a metric value association")

    reversed_rows = list(reversed(rows))
    reversed_normalized = _normalize_task2_scope_summaries(
        reversed_rows,
        location="self-test reversed",
    )
    if reversed_normalized != normalized:
        raise AssertionError("Self-test depends on input row order")

    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(rows[:-1], location="self-test missing"),
        "a missing metric",
    )
    duplicate_rows = [*rows[:-1], rows[0]]
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            duplicate_rows,
            location="self-test duplicate",
        ),
        "a duplicate metric",
    )
    unknown_rows = [dict(row) for row in rows]
    unknown_rows[-1]["metric_name"] = "unknown_metric"
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            unknown_rows,
            location="self-test unknown",
        ),
        "an unknown metric",
    )
    bad_seed_rows = [dict(row) for row in rows]
    bad_seed_rows[0] = {**bad_seed_rows[0], "seeds": [13, 37, 72]}
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            bad_seed_rows,
            location="self-test bad seeds",
        ),
        "bad seeds",
    )
    float_seed_rows = [dict(row) for row in rows]
    float_seed_rows[0] = {**float_seed_rows[0], "seeds": [13.0, 37, 71]}
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            float_seed_rows,
            location="self-test float seed",
        ),
        "a float seed",
    )
    boolean_seed_rows = [dict(row) for row in rows]
    boolean_seed_rows[0] = {**boolean_seed_rows[0], "seeds": [13, 37, True]}
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            boolean_seed_rows,
            location="self-test boolean seed",
        ),
        "a boolean seed",
    )
    float_ddof_rows = [dict(row) for row in rows]
    float_ddof_rows[0] = {**float_ddof_rows[0], "standard_deviation_ddof": 1.0}
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            float_ddof_rows,
            location="self-test float ddof",
        ),
        "a float stability ddof",
    )
    boolean_ddof_rows = [dict(row) for row in rows]
    boolean_ddof_rows[0] = {**boolean_ddof_rows[0], "standard_deviation_ddof": True}
    _expect_value_error(
        lambda: _normalize_task2_scope_summaries(
            boolean_ddof_rows,
            location="self-test boolean ddof",
        ),
        "a boolean stability ddof",
    )

    original_state = _native_adapter_state()
    with _native_adapter():
        if (
            native.FINAL_REPORT_ASSET_SET_ID != OUTPUT_ASSET_SET_ID
            or _absolute(native.FINAL_REPORT_ASSET_ROOT) != _absolute(OUTPUT_ROOT)
            or _absolute(native.FINAL_REPORT_MANIFEST_PATH) != _absolute(OUTPUT_MANIFEST_PATH)
            or _absolute(native.FINAL_REPORT_LOCK_PATH) != _absolute(OUTPUT_LOCK_PATH)
            or native._task2_scope_summaries is not _normalize_task2_scope_summaries
        ):
            raise AssertionError("Self-test adapter context was not applied")
    if _native_adapter_state() != original_state:
        raise AssertionError("Self-test adapter context did not restore after success")

    try:
        with _native_adapter():
            raise RuntimeError("self-test sentinel")
    except RuntimeError as exc:
        if str(exc) != "self-test sentinel":
            raise
    if _native_adapter_state() != original_state:
        raise AssertionError("Self-test adapter context did not restore after an exception")

    adapter_record = _artifact_record(ADAPTER_PATH)
    if _require_adapter_current(adapter_record) != adapter_record:
        raise AssertionError("Self-test adapter identity changed without mutation")
    invalid_adapter_record = {**adapter_record, "sha256": "0" * _SHA256_LENGTH}
    _expect_permission_error(
        lambda: _require_adapter_current(invalid_adapter_record),
        "a changed adapter identity",
    )
    return {
        "passed": True,
        "tests": 15,
        "presentation_metric_order": list(PRESENTATION_METRIC_ORDER),
        "producer_metric_order": list(PRODUCER_METRIC_ORDER),
    }


def _frozen_records() -> dict[str, dict[str, Any]]:
    return {
        name: _artifact_record(path, expected_sha256=_PINNED_SHA256[name])
        for name, path in _PINNED_PATHS.items()
    }


def _validate_metric_order_incident(final_result: Mapping[str, Any]) -> None:
    task2 = final_result.get("task2_summary")
    if not isinstance(task2, Mapping):
        raise ValueError("Frozen final result lacks Task 2 summary evidence")
    for stream_name in ("reconstruction", "latent"):
        stream = task2.get(stream_name)
        if not isinstance(stream, Mapping):
            raise ValueError(f"Frozen final result lacks Task 2 {stream_name} evidence")
        stability = stream.get("stability")
        if not isinstance(stability, Mapping) or stability.get("seed_order") != list(SEED_ORDER):
            raise ValueError(f"Frozen Task 2 {stream_name} stability identity is invalid")
        scopes: list[tuple[str, object]] = [
            ("pooled", stability.get("pooled")),
            ("macro", stability.get("macro")),
        ]
        per_species = stability.get("per_species")
        if isinstance(per_species, (str, bytes)) or not isinstance(per_species, Sequence):
            raise ValueError(f"Frozen Task 2 {stream_name} species evidence is invalid")
        for item in per_species:
            if (
                not isinstance(item, Mapping)
                or type(item.get("species_scientific_name")) is not str
            ):
                raise ValueError(f"Frozen Task 2 {stream_name} species identity is invalid")
            scopes.append((str(item["species_scientific_name"]), item.get("metrics")))
        for scope_name, row_values in scopes:
            rows = native._require_sequence(
                row_values,
                f"Frozen Task 2 {stream_name} {scope_name}",
            )
            observed_order = tuple(
                native._require_mapping(row, "Frozen stability row").get("metric_name")
                for row in rows
            )
            if observed_order != PRODUCER_METRIC_ORDER:
                raise ValueError("Frozen Task 2 producer metric order differs from the incident")
            _normalize_task2_scope_summaries(
                rows,
                location=f"Frozen Task 2 {stream_name} {scope_name}",
            )


def _validate_frozen_identity() -> dict[str, Any]:
    if _absolute(__file__) != _absolute(ADAPTER_PATH):
        raise ValueError("Recovery adapter is not running from its canonical path")
    if not _is_sha256(FROZEN_SOURCE_FINGERPRINT) or any(
        not _is_sha256(value) for value in _PINNED_SHA256.values()
    ):
        raise ValueError("Recovery adapter contains an invalid pinned SHA-256 value")
    if source_fingerprint() != FROZEN_SOURCE_FINGERPRINT:
        raise PermissionError("Current source fingerprint differs from the frozen evaluation")
    if tuple(native.TASK2_METRIC_ORDER) != PRESENTATION_METRIC_ORDER:
        raise ValueError("Native presentation metric order changed")
    if native.FINAL_REPORT_ASSET_SET_ID != _ORIGINAL_ASSET_SET_ID:
        raise ValueError("Native report asset identity changed before adaptation")
    if _absolute(native.FINAL_REPORT_ASSET_ROOT) != _absolute(_ORIGINAL_OUTPUT_ROOT):
        raise ValueError("Native report output root changed before adaptation")
    if OUTPUT_ROOT.parent != PROJECT_ROOT / "report_assets":
        raise ValueError("Recovery report output root is not canonical")
    if RECOVERY_ROOT.parent != PROJECT_ROOT / "evidence" / "recovery":
        raise ValueError("Recovery evidence root is not canonical")

    records = _frozen_records()
    gate, _ = _read_canonical_json(GATE_PATH, "Frozen final gate")
    final_result, _ = _read_canonical_json(FINAL_RESULT_PATH, "Frozen final result")
    summary_result, _ = _read_canonical_json(SUMMARY_RESULT_PATH, "Frozen final summary")
    gate_source = (
        gate.get("shared_identity", {}).get("source_fingerprint_sha256")
        if isinstance(gate.get("shared_identity"), Mapping)
        else None
    )
    if gate.get("ready") is not True or gate_source != FROZEN_SOURCE_FINGERPRINT:
        raise ValueError("Frozen final gate identity is invalid")
    if (
        final_result.get("complete") is not True
        or final_result.get("source_fingerprint_sha256") != FROZEN_SOURCE_FINGERPRINT
        or final_result.get("gate_sha256") != _PINNED_SHA256["gate"]
        or final_result.get("seed_order") != list(SEED_ORDER)
    ):
        raise ValueError("Frozen final result identity is invalid")
    if (
        summary_result.get("complete") is not True
        or summary_result.get("source_fingerprint_sha256") != FROZEN_SOURCE_FINGERPRINT
        or summary_result.get("task2") != final_result.get("task2_summary")
    ):
        raise ValueError("Frozen final summary identity is invalid")
    _validate_metric_order_incident(final_result)
    return {
        "source_fingerprint_sha256": FROZEN_SOURCE_FINGERPRINT,
        "frozen_artifacts": records,
        "adapter": _artifact_record(ADAPTER_PATH),
    }


def _native_adapter_state() -> tuple[Any, ...]:
    return (
        native.FINAL_REPORT_ASSET_SET_ID,
        native.FINAL_REPORT_ASSET_ROOT,
        native.FINAL_REPORT_MANIFEST_PATH,
        native.FINAL_REPORT_LOCK_PATH,
        native._task2_scope_summaries,
    )


@contextmanager
def _native_adapter() -> Iterator[None]:
    original_state = _native_adapter_state()
    expected_original = (
        _ORIGINAL_ASSET_SET_ID,
        _ORIGINAL_OUTPUT_ROOT,
        _ORIGINAL_OUTPUT_ROOT / "manifest.json",
        _ORIGINAL_OUTPUT_ROOT / "lock.json",
    )
    if original_state[:4] != expected_original:
        raise RuntimeError("Native report namespace changed before adaptation")
    try:
        native.FINAL_REPORT_ASSET_SET_ID = OUTPUT_ASSET_SET_ID
        native.FINAL_REPORT_ASSET_ROOT = OUTPUT_ROOT
        native.FINAL_REPORT_MANIFEST_PATH = OUTPUT_MANIFEST_PATH
        native.FINAL_REPORT_LOCK_PATH = OUTPUT_LOCK_PATH
        native._task2_scope_summaries = _normalize_task2_scope_summaries
        yield
    finally:
        (
            native.FINAL_REPORT_ASSET_SET_ID,
            native.FINAL_REPORT_ASSET_ROOT,
            native.FINAL_REPORT_MANIFEST_PATH,
            native.FINAL_REPORT_LOCK_PATH,
            native._task2_scope_summaries,
        ) = original_state


def _evidence_entries(descriptor: int) -> dict[str, str]:
    with os.scandir(descriptor) as entries:
        return {
            entry.name: (
                "file"
                if entry.is_file(follow_symlinks=False) and not entry.is_symlink()
                else "unsafe"
            )
            for entry in entries
        }


@contextmanager
def _evidence_transaction(*, exclusive: bool, create: bool) -> Iterator[int]:
    if create:
        native._secure_ensure_directory(RECOVERY_ROOT, PROJECT_ROOT)
    descriptor = native._open_absolute_directory_no_follow(RECOVERY_ROOT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_or_verify_evidence(path: Path, payload: bytes) -> dict[str, Any]:
    candidate = _absolute(path)
    if candidate.parent != _absolute(RECOVERY_ROOT) or candidate.name not in _EVIDENCE_FILENAMES:
        raise ValueError("Recovery evidence output path is not canonical")
    if os.path.lexists(candidate):
        observed, observed_sha256, observed_size = native._snapshot(
            candidate,
            boundary=_absolute(RECOVERY_ROOT),
        )
        if observed != payload:
            raise ValueError(f"Recovery evidence changed: {candidate.name}")
    else:
        with suppress(FileExistsError):
            gate_runtime._atomic_create_only_bytes(candidate, payload)
        observed, observed_sha256, observed_size = native._snapshot(
            candidate,
            boundary=_absolute(RECOVERY_ROOT),
        )
        if observed != payload:
            raise RuntimeError(f"Recovery evidence publication failed: {candidate.name}")
    return {
        "path": candidate.name,
        "sha256": observed_sha256,
        "size_bytes": observed_size,
    }


def _recovery_manifest_value(adapter_record: Mapping[str, Any]) -> dict[str, Any]:
    bound_adapter = _require_adapter_current(adapter_record)
    frozen_records = _frozen_records()
    output_manifest = _artifact_record(OUTPUT_MANIFEST_PATH)
    output_lock = _artifact_record(OUTPUT_LOCK_PATH)
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "recovery_id": RECOVERY_ID,
        "complete": True,
        "incident": {
            "boundary": "post_evaluation_report_rendering",
            "exception_type": "ValueError",
            "exception_message": (
                "Task 2 reconstruction pooled stability sensitivity seed identity is invalid"
            ),
            "cause": (
                "The frozen producer emits metric-name records in lexical order while the "
                "native report renderer interpreted the sequence in presentation order."
            ),
            "failed_native_output_root": "report_assets/final_v2",
            "producer_metric_order": list(PRODUCER_METRIC_ORDER),
            "presentation_metric_order": list(PRESENTATION_METRIC_ORDER),
            "normalization_identity_field": "metric_name",
        },
        "no_scientific_mutation": {
            "source_fingerprint_sha256": FROZEN_SOURCE_FINGERPRINT,
            "models_retrained": False,
            "model_inference_repeated": False,
            "predictions_changed": False,
            "metrics_changed": False,
            "sealed_gate_changed": False,
            "sealed_final_evaluation_changed": False,
            "numeric_values_transformed": False,
            "scope": "report_presentation_only",
        },
        "adapter": bound_adapter,
        "original_sources": {
            "renderer": frozen_records["renderer"],
            "metric_producer": frozen_records["producer"],
        },
        "sealed_inputs": {
            "gate": frozen_records["gate"],
            "gate_lock": frozen_records["gate_lock"],
            "final_result": frozen_records["final_result"],
            "final_lock": frozen_records["final_lock"],
            "summary_result": frozen_records["summary_result"],
            "summary_lock": frozen_records["summary_lock"],
        },
        "report_output": {
            "asset_set_id": OUTPUT_ASSET_SET_ID,
            "root": _project_relative(OUTPUT_ROOT),
            "asset_count": len(native._ASSET_MEDIA_TYPES),
            "manifest": output_manifest,
            "lock": output_lock,
        },
    }


def _recovery_lock_value(manifest_record: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {"path", "sha256", "size_bytes"}
    if set(manifest_record) != expected_fields or manifest_record.get("path") != "manifest.json":
        raise ValueError("Recovery manifest record is invalid")
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "recovery_id": RECOVERY_ID,
        "manifest": dict(manifest_record),
    }


def _seal_recovery_evidence(adapter_record: Mapping[str, Any]) -> None:
    _require_adapter_current(adapter_record)
    with _evidence_transaction(exclusive=True, create=True) as descriptor:
        entries = _evidence_entries(descriptor)
        if not set(entries).issubset(_EVIDENCE_FILENAMES) or any(
            kind != "file" for kind in entries.values()
        ):
            raise ValueError("Recovery evidence directory contains unexpected entries")
        if "lock.json" in entries and "manifest.json" not in entries:
            raise ValueError("Recovery lock exists without its manifest")
        _require_adapter_current(adapter_record)
        manifest = _recovery_manifest_value(adapter_record)
        _require_adapter_current(adapter_record)
        manifest_record = _write_or_verify_evidence(
            RECOVERY_MANIFEST_PATH,
            _canonical_json_bytes(manifest),
        )
        _require_adapter_current(adapter_record)
        _write_or_verify_evidence(
            RECOVERY_LOCK_PATH,
            _canonical_json_bytes(_recovery_lock_value(manifest_record)),
        )
        _require_adapter_current(adapter_record)


def _verify_recovery_evidence(adapter_record: Mapping[str, Any]) -> dict[str, Any]:
    expected_adapter = _require_adapter_current(adapter_record)
    with _evidence_transaction(exclusive=False, create=False) as descriptor:
        entries = _evidence_entries(descriptor)
        if entries != {"manifest.json": "file", "lock.json": "file"}:
            raise ValueError("Recovery evidence bundle is incomplete")
        observed_manifest, observed_manifest_record = _read_canonical_json(
            RECOVERY_MANIFEST_PATH,
            "Recovery manifest",
        )
        sealed_adapter = _require_adapter_current(observed_manifest.get("adapter"))
        if sealed_adapter != expected_adapter:
            raise PermissionError("Recovery manifest binds another adapter identity")
        expected_manifest = _recovery_manifest_value(expected_adapter)
        if observed_manifest != expected_manifest:
            raise ValueError("Recovery manifest differs from current bound evidence")
        local_manifest_record = {
            "path": "manifest.json",
            "sha256": observed_manifest_record["sha256"],
            "size_bytes": observed_manifest_record["size_bytes"],
        }
        observed_lock, observed_lock_record = _read_canonical_json(
            RECOVERY_LOCK_PATH,
            "Recovery lock",
        )
        expected_lock = _recovery_lock_value(local_manifest_record)
        if observed_lock != expected_lock:
            raise ValueError("Recovery lock does not bind its manifest")
        _require_adapter_current(expected_adapter)
        return {
            "manifest": observed_manifest_record,
            "lock": observed_lock_record,
            "complete": True,
        }


def _assert_frozen_unchanged(before: Mapping[str, Any]) -> None:
    after = _validate_frozen_identity()
    if dict(before) != after:
        raise PermissionError("Frozen scientific evidence changed during report recovery")


def _compact_result(
    action: str,
    self_tests: Mapping[str, Any],
    native_result: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    created: bool,
) -> dict[str, Any]:
    assets = native_result.get("assets")
    if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
        raise ValueError("Native report verification returned an invalid asset inventory")
    return {
        "action": action,
        "complete": True,
        "created": created,
        "source_fingerprint_sha256": FROZEN_SOURCE_FINGERPRINT,
        "asset_set_id": OUTPUT_ASSET_SET_ID,
        "output_root": str(OUTPUT_ROOT),
        "asset_count": len(assets),
        "report_manifest": native_result.get("manifest_artifact"),
        "report_lock": native_result.get("lock_artifact"),
        "recovery_manifest": recovery.get("manifest"),
        "recovery_lock": recovery.get("lock"),
        "self_tests": dict(self_tests),
    }


def build() -> dict[str, Any]:
    self_tests = run_self_tests()
    frozen_before = _validate_frozen_identity()
    adapter_record = native._require_mapping(
        frozen_before.get("adapter"),
        "Captured recovery adapter",
    )
    with _native_adapter():
        built = native.build_final_report_assets()
        verified = native.verify_final_report_assets()
    _assert_frozen_unchanged(frozen_before)
    _require_adapter_current(adapter_record)
    _seal_recovery_evidence(adapter_record)
    _require_adapter_current(adapter_record)
    recovery = _verify_recovery_evidence(adapter_record)
    _require_adapter_current(adapter_record)
    _assert_frozen_unchanged(frozen_before)
    return _compact_result(
        "build",
        self_tests,
        verified,
        recovery,
        created=built.get("created") is True,
    )


def verify() -> dict[str, Any]:
    self_tests = run_self_tests()
    frozen_before = _validate_frozen_identity()
    adapter_record = native._require_mapping(
        frozen_before.get("adapter"),
        "Captured recovery adapter",
    )
    with _native_adapter():
        verified = native.verify_final_report_assets()
    _require_adapter_current(adapter_record)
    recovery = _verify_recovery_evidence(adapter_record)
    _require_adapter_current(adapter_record)
    _assert_frozen_unchanged(frozen_before)
    return _compact_result(
        "verify",
        self_tests,
        verified,
        recovery,
        created=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify frozen final report assets with metric-name normalization."
    )
    parser.add_argument("action", choices=("build", "verify", "self-test"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.action == "build":
        result = build()
    elif args.action == "verify":
        result = verify()
    else:
        result = {"action": "self-test", **run_self_tests()}
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
