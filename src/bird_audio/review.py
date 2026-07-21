from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.hashing import sha256_bytes, sha256_file, sha256_json
from bird_audio.io_utils import atomic_write_csv, atomic_write_json, require_unchanged
from bird_audio.locking import project_lock
from bird_audio.manifest import HARD_EXCLUSION_REASONS
from bird_audio.metadata_artifacts import verify_enrichment_lock
from bird_audio.paths import (
    PROJECT_ROOT,
    is_relative_to,
    require_safe_output,
    resolve_project_path,
)

PREPARATION_SCHEMA_VERSION = "1.0"
RESOLUTION_SCHEMA_VERSION = "1.0"
REVIEW_LOCK_SCHEMA_VERSION = "1.0"

REVIEW_ITEM_CONTEXT_FIELDS = [
    "recording_id",
    "xc_id",
    "xc_url",
    "relative_path",
    "sha256",
    "species_folder",
    "species_common_name",
    "scientific_name",
    "primary_label",
    "api_scientific_name",
    "api_group",
    "secondary_labels",
    "target_secondary_labels",
    "recordist",
    "country",
    "locality",
    "latitude",
    "longitude",
    "recorded_date",
    "recorded_time",
    "quality",
    "remarks",
    "licence",
    "licence_status",
    "attribution",
    "metadata_status",
    "metadata_error",
    "identity_validation_status",
    "licence_validation_status",
    "probe_ok",
    "full_decode_status",
    "local_qc_status",
    "exclusion_reasons",
    "session_group",
    "session_review_flag",
    "session_review_reason",
]

REVIEW_ITEM_FIELDS = [
    "source_manifest_sha256",
    "review_item_id",
    "item_context_sha256",
    *REVIEW_ITEM_CONTEXT_FIELDS,
]

REVIEW_DECISION_CONTEXT_FIELDS = [
    "review_item_id",
    "recording_id",
    "xc_url",
    "species_common_name",
    "scientific_name",
    "metadata_status",
    "identity_validation_status",
    "licence_validation_status",
    "exclusion_reasons",
    "session_group",
    "session_review_reason",
]

REVIEW_DECISION_FIELDS = [
    *REVIEW_DECISION_CONTEXT_FIELDS,
    "decision",
    "decision_reason",
    "confirmed_session_group",
]

REVIEW_PROVENANCE_FIELDS = [
    "review_status",
    "review_item_id",
    "review_item_context_sha256",
    "review_decision",
    "review_decision_reason",
    "review_confirmed_session_group",
    "reviewed_at_utc",
    "review_source_manifest_sha256",
    "review_enrichment_lock_sha256",
    "review_preparation_sha256",
    "review_items_sha256",
    "review_decisions_sha256",
    "review_original_local_qc_status",
    "review_original_exclusion_reasons",
    "review_original_session_group",
    "review_original_session_review_flag",
    "review_original_session_review_reason",
]

REQUIRED_SOURCE_FIELDS = frozenset(REVIEW_ITEM_CONTEXT_FIELDS)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _project_label(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_csv_snapshot(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError(f"CSV has duplicate field names: {path}")
    rows = list(reader)
    return rows, fieldnames, digest


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, sha256_bytes(payload)


def _require_distinct_paths(paths: list[Path], inputs: list[Path] | None = None) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError("Review output paths must be distinct")
    if inputs and set(paths) & set(inputs):
        raise ValueError("A review output path cannot replace an input artifact")


def _refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Review outputs already exist: " + ", ".join(existing))


def _require_exact_fields(actual: list[str], expected: list[str], path: Path) -> None:
    if actual != expected:
        raise ValueError(
            f"Unexpected CSV schema for {path}. Expected {expected}, received {actual}"
        )


def _validate_source_rows(
    rows: list[dict[str, str]], fieldnames: list[str], source: Path
) -> dict[str, dict[str, str]]:
    if not rows:
        raise ValueError(f"Source manifest has no recordings: {source}")
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(fieldnames))
    if missing:
        raise ValueError(f"Source manifest is missing required review fields: {missing}")
    collisions = sorted(set(fieldnames) & set(REVIEW_PROVENANCE_FIELDS))
    if collisions:
        raise ValueError(f"Source manifest already contains review provenance fields: {collisions}")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        recording_id = row.get("recording_id", "").strip()
        if not recording_id:
            raise ValueError("Source manifest contains an empty recording_id")
        if recording_id in by_id:
            raise ValueError(f"Duplicate source recording_id: {recording_id}")
        by_id[recording_id] = row
    return by_id


def _validate_enrichment_lock(lock: dict[str, Any], source_sha256: str) -> None:
    if lock.get("enriched_manifest_sha256") != source_sha256:
        raise ValueError("Enrichment lock does not bind the exact enriched manifest")
    if lock.get("ready_for_manual_review") is not True:
        raise ValueError("Enrichment lock is not ready for manual review")


def _verify_enrichment_snapshot(
    lock_path: Path,
    source: Path,
    lock_record: dict[str, Any],
    source_sha256: str,
) -> None:
    verified = verify_enrichment_lock(lock_path, source)
    if verified != lock_record:
        raise RuntimeError("Enrichment lock changed during verification")
    _validate_enrichment_lock(lock_record, source_sha256)


def _is_review_item(row: dict[str, str]) -> bool:
    return row.get("local_qc_status") != "exclude" and (
        row.get("local_qc_status") == "manual_review" or row.get("session_review_flag") == "true"
    )


def _context_payload(row: dict[str, str], source_sha256: str) -> dict[str, str]:
    return {
        "source_manifest_sha256": source_sha256,
        **{field: row.get(field, "") for field in REVIEW_ITEM_CONTEXT_FIELDS},
    }


def _item_identity(row: dict[str, str], source_sha256: str) -> tuple[str, str, dict[str, str]]:
    context = _context_payload(row, source_sha256)
    context_sha256 = sha256_json(context)
    review_item_id = "review:" + sha256_json(
        {
            "source_manifest_sha256": source_sha256,
            "recording_id": row["recording_id"],
            "item_context_sha256": context_sha256,
        }
    )
    return review_item_id, context_sha256, context


def _make_item(row: dict[str, str], source_sha256: str) -> dict[str, str]:
    review_item_id, context_sha256, context = _item_identity(row, source_sha256)
    return {
        "source_manifest_sha256": source_sha256,
        "review_item_id": review_item_id,
        "item_context_sha256": context_sha256,
        **{field: context[field] for field in REVIEW_ITEM_CONTEXT_FIELDS},
    }


def prepare_manual_review(
    enriched_manifest_path: str | Path,
    enrichment_lock_path: str | Path,
    items_path: str | Path,
    decisions_path: str | Path,
    preparation_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    source = resolve_project_path(enriched_manifest_path)
    enrichment_lock = resolve_project_path(enrichment_lock_path)
    items_destination = require_safe_output(items_path)
    decisions_destination = require_safe_output(decisions_path)
    preparation_destination = require_safe_output(preparation_path)
    outputs = [items_destination, decisions_destination, preparation_destination]
    inputs = [source, enrichment_lock]
    _require_distinct_paths(outputs, inputs)
    _refuse_existing(outputs)

    with project_lock("manual_review_prepare"):
        _refuse_existing(outputs)
        rows, fieldnames, source_sha256 = _read_csv_snapshot(source)
        _validate_source_rows(rows, fieldnames, source)
        enrichment_record, enrichment_lock_sha256 = _read_json_snapshot(enrichment_lock)
        _verify_enrichment_snapshot(enrichment_lock, source, enrichment_record, source_sha256)

        items = [_make_item(row, source_sha256) for row in rows if _is_review_item(row)]
        items.sort(key=lambda item: (item["recording_id"], item["review_item_id"]))
        item_ids = [item["review_item_id"] for item in items]
        if len(item_ids) != len(set(item_ids)):
            raise RuntimeError("Review item ID collision")
        decisions = [
            {
                **{field: item.get(field, "") for field in REVIEW_DECISION_CONTEXT_FIELDS},
                "decision": "",
                "decision_reason": "",
                "confirmed_session_group": "",
            }
            for item in items
        ]

        require_unchanged(source, source_sha256)
        require_unchanged(enrichment_lock, enrichment_lock_sha256)
        atomic_write_csv(items_destination, items, REVIEW_ITEM_FIELDS)
        items_sha256 = sha256_file(items_destination)
        atomic_write_csv(decisions_destination, decisions, REVIEW_DECISION_FIELDS)
        decision_template_sha256 = sha256_file(decisions_destination)
        require_unchanged(source, source_sha256)
        require_unchanged(enrichment_lock, enrichment_lock_sha256)

        status_counts = Counter(row.get("local_qc_status", "") for row in rows)
        preparation = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "prepared_at_utc": _utc_now(),
            "source_manifest_path": _project_label(source),
            "source_manifest_sha256": source_sha256,
            "enrichment_lock_path": _project_label(enrichment_lock),
            "enrichment_lock_sha256": enrichment_lock_sha256,
            "review_items_path": _project_label(items_destination),
            "review_items_sha256": items_sha256,
            "decision_template_path": _project_label(decisions_destination),
            "decision_template_sha256": decision_template_sha256,
            "source_recordings": len(rows),
            "review_items": len(items),
            "not_required": len(rows) - len(items),
            "manual_review_status_rows": status_counts.get("manual_review", 0),
            "session_review_flag_rows": sum(
                row.get("session_review_flag") == "true" and row.get("local_qc_status") != "exclude"
                for row in rows
            ),
            "review_item_ids": sorted(item_ids),
            "review_item_set_sha256": sha256_json(sorted(item_ids)),
        }
        atomic_write_json(preparation_destination, preparation)
    return preparation_destination, preparation


def _validate_preparation(
    preparation: dict[str, Any],
    source: Path,
    source_sha256: str,
    enrichment_lock: Path,
    enrichment_lock_sha256: str,
    items: Path,
    items_sha256: str,
) -> None:
    if preparation.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ValueError("Unsupported manual-review preparation schema")
    expected = {
        "source_manifest_path": _project_label(source),
        "source_manifest_sha256": source_sha256,
        "enrichment_lock_path": _project_label(enrichment_lock),
        "enrichment_lock_sha256": enrichment_lock_sha256,
        "review_items_path": _project_label(items),
        "review_items_sha256": items_sha256,
    }
    mismatches = [key for key, value in expected.items() if preparation.get(key) != value]
    if mismatches:
        raise ValueError(f"Manual-review preparation binding mismatch: {mismatches}")


def _validate_items(
    item_rows: list[dict[str, str]],
    source_rows: dict[str, dict[str, str]],
    source_sha256: str,
    preparation: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    items_by_id: dict[str, dict[str, str]] = {}
    recording_by_item: dict[str, str] = {}
    for item in item_rows:
        item_id = item.get("review_item_id", "")
        if not item_id or item_id in items_by_id:
            raise ValueError(f"Duplicate or empty review item ID: {item_id!r}")
        recording_id = item.get("recording_id", "")
        source_row = source_rows.get(recording_id)
        if source_row is None or not _is_review_item(source_row):
            raise ValueError(f"Review item is not required by the source manifest: {recording_id}")
        expected_id, expected_context_sha256, expected_context = _item_identity(
            source_row, source_sha256
        )
        item_context = {
            "source_manifest_sha256": item.get("source_manifest_sha256", ""),
            **{field: item.get(field, "") for field in REVIEW_ITEM_CONTEXT_FIELDS},
        }
        if item_context != expected_context:
            raise ValueError(f"Immutable review context mismatch for {recording_id}")
        if sha256_json(item_context) != item.get("item_context_sha256"):
            raise ValueError(f"Review item context hash mismatch for {recording_id}")
        if item.get("item_context_sha256") != expected_context_sha256 or item_id != expected_id:
            raise ValueError(f"Review item identity mismatch for {recording_id}")
        items_by_id[item_id] = item
        recording_by_item[item_id] = recording_id

    actual_ids = sorted(items_by_id)
    source_required_ids = sorted(
        _item_identity(row, source_sha256)[0]
        for row in source_rows.values()
        if _is_review_item(row)
    )
    if actual_ids != source_required_ids:
        raise ValueError("Review items do not exactly cover the source review queue")
    if preparation.get("review_item_ids") != source_required_ids:
        raise ValueError("Preparation review item list mismatch")
    if preparation.get("review_item_set_sha256") != sha256_json(source_required_ids):
        raise ValueError("Preparation review item set hash mismatch")
    if preparation.get("review_items") != len(actual_ids):
        raise ValueError("Preparation review item count mismatch")
    if preparation.get("source_recordings") != len(source_rows):
        raise ValueError("Preparation source recording count mismatch")
    return items_by_id, recording_by_item


def _split_reasons(value: str) -> set[str]:
    return {reason.strip() for reason in value.split(";") if reason.strip()}


def _has_target_secondary(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return True
    if parsed is None or parsed == "":
        return False
    if isinstance(parsed, list):
        return any(str(item).strip() for item in parsed)
    return True


def _include_errors(row: dict[str, str], check_active_reasons: bool = True) -> list[str]:
    errors: list[str] = []
    if row.get("metadata_status") != "ok":
        errors.append("metadata_status_not_ok")
    if row.get("identity_validation_status") != "exact_match":
        errors.append("identity_not_exact_match")
    if (
        row.get("licence_validation_status") != "recognized_cc"
        or not row.get("licence", "").strip()
    ):
        errors.append("licence_not_recognized_cc")
    if row.get("local_qc_status") == "exclude":
        errors.append("hard_excluded_status")
    reasons = _split_reasons(row.get("exclusion_reasons", ""))
    if reasons & HARD_EXCLUSION_REASONS:
        errors.append("hard_exclusion_reason")
    if _has_target_secondary(row.get("target_secondary_labels", "")):
        errors.append("target_secondary_label")
    if check_active_reasons and any(not reason.startswith("session_") for reason in reasons):
        errors.append("non_session_review_reason")
    session_reasons = _split_reasons(row.get("session_review_reason", ""))
    if any(not reason.startswith("session_") for reason in session_reasons):
        errors.append("invalid_session_review_reason")
    return errors


def _validate_decisions(
    decision_rows: list[dict[str, str]],
    items_by_id: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    item_ids = set(items_by_id)
    decisions: dict[str, dict[str, str]] = {}
    for raw in decision_rows:
        item_id = raw.get("review_item_id", "").strip()
        if not item_id or item_id in decisions:
            raise ValueError(f"Duplicate or empty decision review_item_id: {item_id!r}")
        item = items_by_id.get(item_id)
        if item is not None:
            context_mismatches = [
                field
                for field in REVIEW_DECISION_CONTEXT_FIELDS
                if raw.get(field, "") != item.get(field, "")
            ]
            if context_mismatches:
                raise ValueError(
                    f"Immutable decision context changed for {item_id}: {context_mismatches}"
                )
        decision = raw.get("decision", "").strip().casefold()
        reason = raw.get("decision_reason", "").strip()
        confirmed_group = raw.get("confirmed_session_group", "").strip()
        if decision not in {"include", "exclude"}:
            raise ValueError(f"Decision must be include or exclude for {item_id}")
        if not reason:
            raise ValueError(f"A decision_reason is required for {item_id}")
        decisions[item_id] = {
            "review_item_id": item_id,
            "decision": decision,
            "decision_reason": reason,
            "confirmed_session_group": confirmed_group,
        }
    if set(decisions) != item_ids:
        missing = sorted(item_ids - set(decisions))
        unexpected = sorted(set(decisions) - item_ids)
        raise ValueError(
            f"Decision ID set does not match review items. Missing={missing}, unexpected={unexpected}"
        )
    return decisions


def _base_review_provenance(
    row: dict[str, str],
    reviewed_at_utc: str,
    source_sha256: str,
    enrichment_lock_sha256: str,
    preparation_sha256: str,
    items_sha256: str,
    decisions_sha256: str,
) -> dict[str, str]:
    return {
        "review_status": "not_required",
        "review_item_id": "",
        "review_item_context_sha256": "",
        "review_decision": "",
        "review_decision_reason": "",
        "review_confirmed_session_group": "",
        "reviewed_at_utc": reviewed_at_utc,
        "review_source_manifest_sha256": source_sha256,
        "review_enrichment_lock_sha256": enrichment_lock_sha256,
        "review_preparation_sha256": preparation_sha256,
        "review_items_sha256": items_sha256,
        "review_decisions_sha256": decisions_sha256,
        "review_original_local_qc_status": row.get("local_qc_status", ""),
        "review_original_exclusion_reasons": row.get("exclusion_reasons", ""),
        "review_original_session_group": row.get("session_group", ""),
        "review_original_session_review_flag": row.get("session_review_flag", ""),
        "review_original_session_review_reason": row.get("session_review_reason", ""),
    }


def _assert_final_ready(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, list[str]]]:
    unresolved = [
        row["recording_id"]
        for row in rows
        if row.get("local_qc_status") not in {"include", "exclude"}
        or (row.get("session_review_flag") == "true" and row.get("local_qc_status") != "exclude")
    ]
    invalid_included: dict[str, list[str]] = {}
    included_rows = [row for row in rows if row.get("local_qc_status") == "include"]
    for row in rows:
        if row.get("local_qc_status") != "include":
            continue
        errors = _include_errors(row)
        if not row.get("session_group", "").strip():
            errors.append("missing_session_group")
        if errors:
            invalid_included[row["recording_id"]] = errors

    by_original_group: dict[str, list[dict[str, str]]] = {}
    for row in included_rows:
        original_group = row.get("review_original_session_group", "").strip()
        if original_group:
            by_original_group.setdefault(original_group, []).append(row)
    for group_rows in by_original_group.values():
        if len({row.get("session_group", "") for row in group_rows}) > 1:
            for row in group_rows:
                invalid_included.setdefault(row["recording_id"], []).append(
                    "original_session_group_fractured"
                )

    for row in included_rows:
        confirmed_group = row.get("review_confirmed_session_group", "").strip()
        original_group = row.get("review_original_session_group", "").strip()
        if confirmed_group and confirmed_group != original_group:
            invalid_included.setdefault(row["recording_id"], []).append(
                "confirmed_session_differs_from_original"
            )
    return unresolved, invalid_included


def apply_manual_review(
    enriched_manifest_path: str | Path,
    enrichment_lock_path: str | Path,
    items_path: str | Path,
    decisions_path: str | Path,
    preparation_path: str | Path,
    final_manifest_path: str | Path,
    resolution_path: str | Path,
    lock_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    source = resolve_project_path(enriched_manifest_path)
    enrichment_lock = resolve_project_path(enrichment_lock_path)
    items = resolve_project_path(items_path)
    decisions = resolve_project_path(decisions_path)
    preparation_file = resolve_project_path(preparation_path)
    final_destination = require_safe_output(final_manifest_path)
    resolution_destination = require_safe_output(resolution_path)
    lock_destination = require_safe_output(lock_path)
    inputs = [source, enrichment_lock, items, decisions, preparation_file]
    outputs = [final_destination, resolution_destination, lock_destination]
    _require_distinct_paths(outputs, inputs)
    _refuse_existing(outputs)

    with project_lock("manual_review_apply"):
        _refuse_existing(outputs)
        source_rows, source_fields, source_sha256 = _read_csv_snapshot(source)
        source_by_id = _validate_source_rows(source_rows, source_fields, source)
        enrichment_record, enrichment_lock_sha256 = _read_json_snapshot(enrichment_lock)
        _verify_enrichment_snapshot(enrichment_lock, source, enrichment_record, source_sha256)
        item_rows, item_fields, items_sha256 = _read_csv_snapshot(items)
        _require_exact_fields(item_fields, REVIEW_ITEM_FIELDS, items)
        decision_rows, decision_fields, decisions_sha256 = _read_csv_snapshot(decisions)
        _require_exact_fields(decision_fields, REVIEW_DECISION_FIELDS, decisions)
        preparation, preparation_sha256 = _read_json_snapshot(preparation_file)
        _validate_preparation(
            preparation,
            source,
            source_sha256,
            enrichment_lock,
            enrichment_lock_sha256,
            items,
            items_sha256,
        )
        items_by_id, recording_by_item = _validate_items(
            item_rows, source_by_id, source_sha256, preparation
        )
        decision_by_id = _validate_decisions(decision_rows, items_by_id)
        item_by_recording = {
            recording_id: item_id for item_id, recording_id in recording_by_item.items()
        }
        reviewed_at_utc = _utc_now()
        final_rows: list[dict[str, str]] = []

        for source_row in source_rows:
            row = dict(source_row)
            row.update(
                _base_review_provenance(
                    source_row,
                    reviewed_at_utc,
                    source_sha256,
                    enrichment_lock_sha256,
                    preparation_sha256,
                    items_sha256,
                    decisions_sha256,
                )
            )
            item_id = item_by_recording.get(source_row["recording_id"])
            if item_id is None:
                final_rows.append(row)
                continue

            item = items_by_id[item_id]
            decision = decision_by_id[item_id]
            selected = decision["decision"]
            confirmed_group = decision["confirmed_session_group"]
            if selected == "include":
                errors = _include_errors(source_row)
                if errors:
                    raise ValueError(
                        f"Review item {item_id} is not eligible for inclusion: {errors}"
                    )
                if source_row.get("session_review_flag") == "true":
                    if not confirmed_group or confirmed_group != source_row.get(
                        "session_group", ""
                    ):
                        raise ValueError(
                            f"Flagged session review {item_id} must confirm its original group"
                        )
                    row["session_group"] = confirmed_group
                elif confirmed_group and confirmed_group != source_row.get("session_group", ""):
                    raise ValueError(
                        f"Unflagged review item {item_id} cannot change its session group"
                    )
                row["local_qc_status"] = "include"
                row["exclusion_reasons"] = ""
            else:
                row["local_qc_status"] = "exclude"
                original_reasons = source_row.get("exclusion_reasons", "").strip(";")
                row["exclusion_reasons"] = ";".join(
                    value for value in (original_reasons, "manual_review_decision_exclude") if value
                )
            row["session_review_flag"] = "false"
            row["session_review_reason"] = ""
            row.update(
                {
                    "review_status": f"resolved_{selected}",
                    "review_item_id": item_id,
                    "review_item_context_sha256": item["item_context_sha256"],
                    "review_decision": selected,
                    "review_decision_reason": decision["decision_reason"],
                    "review_confirmed_session_group": confirmed_group,
                }
            )
            final_rows.append(row)

        unresolved, invalid_included = _assert_final_ready(final_rows)
        included_count = sum(row.get("local_qc_status") == "include" for row in final_rows)
        excluded_count = sum(row.get("local_qc_status") == "exclude" for row in final_rows)
        ready_for_split = not unresolved and not invalid_included and included_count > 0
        if not ready_for_split:
            raise ValueError(
                "Reviewed manifest is not ready for splitting. "
                f"Unresolved={unresolved[:20]}, invalid_included={invalid_included}"
            )

        input_hashes = {
            source: source_sha256,
            enrichment_lock: enrichment_lock_sha256,
            items: items_sha256,
            decisions: decisions_sha256,
            preparation_file: preparation_sha256,
        }
        for path, digest in input_hashes.items():
            require_unchanged(path, digest)

        final_fields = [*source_fields, *REVIEW_PROVENANCE_FIELDS]
        atomic_write_csv(final_destination, final_rows, final_fields)
        final_sha256 = sha256_file(final_destination)
        for path, digest in input_hashes.items():
            require_unchanged(path, digest)

        decision_counts = Counter(decision["decision"] for decision in decision_by_id.values())
        resolution = {
            "schema_version": RESOLUTION_SCHEMA_VERSION,
            "resolved_at_utc": reviewed_at_utc,
            "source_manifest_sha256": source_sha256,
            "enrichment_lock_sha256": enrichment_lock_sha256,
            "preparation_sha256": preparation_sha256,
            "review_items_sha256": items_sha256,
            "review_decisions_sha256": decisions_sha256,
            "final_manifest_sha256": final_sha256,
            "recordings": len(final_rows),
            "review_items": len(items_by_id),
            "decision_counts": dict(sorted(decision_counts.items())),
            "included": included_count,
            "excluded": excluded_count,
            "unresolved_recordings": unresolved,
            "invalid_included_recordings": invalid_included,
            "all_review_items_resolved": len(decision_by_id) == len(items_by_id),
            "all_included_rows_valid": not invalid_included,
            "ready_for_split": ready_for_split,
        }
        atomic_write_json(resolution_destination, resolution)
        resolution_sha256 = sha256_file(resolution_destination)
        for path, digest in input_hashes.items():
            require_unchanged(path, digest)

        lock_record = {
            "schema_version": REVIEW_LOCK_SCHEMA_VERSION,
            "locked_at_utc": _utc_now(),
            "ready_for_split": True,
            "source_manifest_sha256": source_sha256,
            "enrichment_lock_sha256": enrichment_lock_sha256,
            "preparation_sha256": preparation_sha256,
            "review_items_sha256": items_sha256,
            "review_decisions_sha256": decisions_sha256,
            "final_manifest_sha256": final_sha256,
            "resolution_sha256": resolution_sha256,
            "source_recordings": len(source_rows),
            "review_items": len(items_by_id),
            "review_item_set_sha256": sha256_json(sorted(items_by_id)),
            "final_recording_set_sha256": sha256_json(sorted(source_by_id)),
            "included": included_count,
            "excluded": excluded_count,
            "artifacts": {
                "source_manifest": {
                    "path": _project_label(source),
                    "sha256": source_sha256,
                },
                "enrichment_lock": {
                    "path": _project_label(enrichment_lock),
                    "sha256": enrichment_lock_sha256,
                },
                "preparation": {
                    "path": _project_label(preparation_file),
                    "sha256": preparation_sha256,
                },
                "review_items": {
                    "path": _project_label(items),
                    "sha256": items_sha256,
                },
                "review_decisions": {
                    "path": _project_label(decisions),
                    "sha256": decisions_sha256,
                },
                "final_manifest": {
                    "path": _project_label(final_destination),
                    "sha256": final_sha256,
                },
                "resolution": {
                    "path": _project_label(resolution_destination),
                    "sha256": resolution_sha256,
                },
            },
        }
        atomic_write_json(lock_destination, lock_record)
    return final_destination, lock_destination, resolution


REVIEW_ARTIFACT_HASH_FIELDS = {
    "source_manifest": "source_manifest_sha256",
    "enrichment_lock": "enrichment_lock_sha256",
    "preparation": "preparation_sha256",
    "review_items": "review_items_sha256",
    "review_decisions": "review_decisions_sha256",
    "final_manifest": "final_manifest_sha256",
    "resolution": "resolution_sha256",
}


def _resolve_review_artifacts(lock: dict[str, Any]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Review lock has no artifact table")
    if not set(REVIEW_ARTIFACT_HASH_FIELDS).issubset(artifacts):
        raise ValueError("Review lock is missing required artifacts")
    paths: dict[str, Path] = {}
    for name, hash_field in REVIEW_ARTIFACT_HASH_FIELDS.items():
        entry = artifacts[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Review lock artifact entry is invalid: {name}")
        path_value = str(entry.get("path") or "")
        if not path_value or Path(path_value).is_absolute():
            raise ValueError(f"Review lock artifact path is invalid: {name}")
        path = resolve_project_path(path_value)
        if not is_relative_to(path, PROJECT_ROOT):
            raise ValueError(f"Review lock artifact leaves the project: {name}")
        if not path.is_file():
            raise ValueError(f"Review lock artifact does not exist: {name}")
        digest = sha256_file(path)
        if digest != entry.get("sha256") or digest != lock.get(hash_field):
            raise ValueError(f"Review lock artifact hash mismatch: {name}")
        paths[name] = path
    return paths


def _verify_final_rows(
    final_rows: list[dict[str, str]],
    source_rows: dict[str, dict[str, str]],
    items_by_id: dict[str, dict[str, str]],
    recording_by_item: dict[str, str],
    decisions_by_id: dict[str, dict[str, str]],
    lock: dict[str, Any],
) -> None:
    final_by_id: dict[str, dict[str, str]] = {}
    for row in final_rows:
        recording_id = row.get("recording_id", "")
        if not recording_id or recording_id in final_by_id:
            raise ValueError(f"Final manifest has duplicate or empty recording ID: {recording_id}")
        final_by_id[recording_id] = row
    if set(final_by_id) != set(source_rows):
        raise ValueError("Final manifest recording set differs from the enriched manifest")

    item_by_recording = {
        recording_id: item_id for item_id, recording_id in recording_by_item.items()
    }
    allowed_changes = {
        "local_qc_status",
        "exclusion_reasons",
        "session_group",
        "session_review_flag",
        "session_review_reason",
    }
    binding_fields = {
        "review_source_manifest_sha256": lock["source_manifest_sha256"],
        "review_enrichment_lock_sha256": lock["enrichment_lock_sha256"],
        "review_preparation_sha256": lock["preparation_sha256"],
        "review_items_sha256": lock["review_items_sha256"],
        "review_decisions_sha256": lock["review_decisions_sha256"],
    }
    for recording_id, source_row in source_rows.items():
        final_row = final_by_id[recording_id]
        for field, value in source_row.items():
            if field not in allowed_changes and final_row.get(field, "") != value:
                raise ValueError(
                    f"Final manifest changed immutable source field: {recording_id}:{field}"
                )
        for field, value in binding_fields.items():
            if final_row.get(field) != value:
                raise ValueError(f"Final manifest review binding mismatch: {recording_id}:{field}")
        original_bindings = {
            "review_original_local_qc_status": source_row.get("local_qc_status", ""),
            "review_original_exclusion_reasons": source_row.get("exclusion_reasons", ""),
            "review_original_session_group": source_row.get("session_group", ""),
            "review_original_session_review_flag": source_row.get("session_review_flag", ""),
            "review_original_session_review_reason": source_row.get("session_review_reason", ""),
        }
        if any(final_row.get(field) != value for field, value in original_bindings.items()):
            raise ValueError(f"Final manifest lost original review context: {recording_id}")

        item_id = item_by_recording.get(recording_id)
        if item_id is None:
            if final_row.get("review_status") != "not_required":
                raise ValueError(f"Unexpected review decision for {recording_id}")
            if any(final_row.get(field, "") != value for field, value in source_row.items()):
                raise ValueError(f"Non-review row changed during adjudication: {recording_id}")
            continue

        item = items_by_id[item_id]
        decision = decisions_by_id[item_id]
        selected = decision["decision"]
        expected = {
            "review_status": f"resolved_{selected}",
            "review_item_id": item_id,
            "review_item_context_sha256": item["item_context_sha256"],
            "review_decision": selected,
            "review_decision_reason": decision["decision_reason"],
            "review_confirmed_session_group": decision["confirmed_session_group"],
            "session_review_flag": "false",
            "session_review_reason": "",
        }
        if any(final_row.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Final review decision binding mismatch: {recording_id}")
        if selected == "include":
            if final_row.get("local_qc_status") != "include" or final_row.get("exclusion_reasons"):
                raise ValueError(f"Included review row was not cleared correctly: {recording_id}")
        elif final_row.get("local_qc_status") != "exclude" or (
            "manual_review_decision_exclude"
            not in _split_reasons(final_row.get("exclusion_reasons", ""))
        ):
            raise ValueError(f"Excluded review row lacks its decision reason: {recording_id}")


def verify_review_lock(
    lock_path: str | Path,
    expected_manifest_path: str | Path,
) -> dict[str, Any]:
    lock_file = resolve_project_path(lock_path)
    expected_manifest = resolve_project_path(expected_manifest_path)
    lock, _ = _read_json_snapshot(lock_file)
    if lock.get("schema_version") != REVIEW_LOCK_SCHEMA_VERSION:
        raise ValueError("Review lock schema is not supported")
    if lock.get("ready_for_split") is not True:
        raise ValueError("Review lock is not ready for splitting")
    paths = _resolve_review_artifacts(lock)
    if paths["final_manifest"] != expected_manifest:
        raise ValueError("Review lock points to a different final manifest")

    source_rows, source_fields, source_sha256 = _read_csv_snapshot(paths["source_manifest"])
    source_by_id = _validate_source_rows(source_rows, source_fields, paths["source_manifest"])
    enrichment_record, enrichment_lock_sha256 = _read_json_snapshot(paths["enrichment_lock"])
    _verify_enrichment_snapshot(
        paths["enrichment_lock"],
        paths["source_manifest"],
        enrichment_record,
        source_sha256,
    )
    item_rows, item_fields, items_sha256 = _read_csv_snapshot(paths["review_items"])
    _require_exact_fields(item_fields, REVIEW_ITEM_FIELDS, paths["review_items"])
    decision_rows, decision_fields, decisions_sha256 = _read_csv_snapshot(paths["review_decisions"])
    _require_exact_fields(decision_fields, REVIEW_DECISION_FIELDS, paths["review_decisions"])
    preparation, preparation_sha256 = _read_json_snapshot(paths["preparation"])
    _validate_preparation(
        preparation,
        paths["source_manifest"],
        source_sha256,
        paths["enrichment_lock"],
        enrichment_lock_sha256,
        paths["review_items"],
        items_sha256,
    )
    items_by_id, recording_by_item = _validate_items(
        item_rows, source_by_id, source_sha256, preparation
    )
    decisions_by_id = _validate_decisions(decision_rows, items_by_id)

    final_rows, final_fields, final_sha256 = _read_csv_snapshot(paths["final_manifest"])
    _require_exact_fields(
        final_fields,
        [*source_fields, *REVIEW_PROVENANCE_FIELDS],
        paths["final_manifest"],
    )
    _verify_final_rows(
        final_rows,
        source_by_id,
        items_by_id,
        recording_by_item,
        decisions_by_id,
        lock,
    )
    unresolved, invalid_included = _assert_final_ready(final_rows)
    if unresolved or invalid_included:
        raise ValueError("Review lock final manifest contains unresolved or invalid rows")

    resolution, resolution_sha256 = _read_json_snapshot(paths["resolution"])
    expected_resolution_hashes = {
        "source_manifest_sha256": source_sha256,
        "enrichment_lock_sha256": enrichment_lock_sha256,
        "preparation_sha256": preparation_sha256,
        "review_items_sha256": items_sha256,
        "review_decisions_sha256": decisions_sha256,
        "final_manifest_sha256": final_sha256,
    }
    if resolution.get("schema_version") != RESOLUTION_SCHEMA_VERSION or any(
        resolution.get(field) != value for field, value in expected_resolution_hashes.items()
    ):
        raise ValueError("Review resolution binding is invalid")
    if (
        resolution.get("ready_for_split") is not True
        or resolution.get("unresolved_recordings") != []
        or resolution.get("invalid_included_recordings") != {}
    ):
        raise ValueError("Review resolution is not complete")

    included = sum(row.get("local_qc_status") == "include" for row in final_rows)
    excluded = sum(row.get("local_qc_status") == "exclude" for row in final_rows)
    expected_lock_values = {
        "source_manifest_sha256": source_sha256,
        "enrichment_lock_sha256": enrichment_lock_sha256,
        "preparation_sha256": preparation_sha256,
        "review_items_sha256": items_sha256,
        "review_decisions_sha256": decisions_sha256,
        "final_manifest_sha256": final_sha256,
        "resolution_sha256": resolution_sha256,
        "source_recordings": len(source_rows),
        "review_items": len(items_by_id),
        "review_item_set_sha256": sha256_json(sorted(items_by_id)),
        "final_recording_set_sha256": sha256_json(sorted(source_by_id)),
        "included": included,
        "excluded": excluded,
    }
    mismatches = [
        field for field, value in expected_lock_values.items() if lock.get(field) != value
    ]
    if mismatches:
        raise ValueError(f"Review lock binding mismatch: {mismatches}")
    return lock
