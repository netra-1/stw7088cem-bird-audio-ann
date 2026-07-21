from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any

from bird_audio.hashing import sha256_bytes, sha256_file, sha256_json
from bird_audio.io_utils import read_csv_snapshot, require_unchanged
from bird_audio.locking import project_lock
from bird_audio.metadata import PERSISTED_RECORDING_FIELDS
from bird_audio.paths import (
    PROJECT_ROOT,
    is_relative_to,
    require_safe_output,
    resolve_project_path,
)
from bird_audio.review import verify_review_lock
from bird_audio.unknown_acquisition import (
    LOCKED_UNKNOWN_SPECIES,
    verify_unknown_metadata_lock,
)

UNKNOWN_SELECTION_CONFIG_SCHEMA_VERSION = "1.0"
UNKNOWN_CANDIDATE_PLAN_SCHEMA_VERSION = "1.0"
UNKNOWN_CANDIDATE_PLAN_LOCK_SCHEMA_VERSION = "1.0"
SELECTION_SEED = 20260713
CANDIDATE_POOL_TARGET = 80
TARGET_RECORDINGS = 40

DEFAULT_CONFIG = "configs/unknown_selection.toml"
DEFAULT_UNKNOWN_METADATA = "data/unknown/metadata/unknown_metadata_v1.json"
DEFAULT_UNKNOWN_METADATA_LOCK = "data/unknown/metadata/unknown_metadata_v1_lock.json"
DEFAULT_KNOWN_MANIFEST = "data/manifests/recordings.csv"
DEFAULT_REVIEW_LOCK = "data/manifests/review_v1_lock.json"
DEFAULT_SPLIT = "data/splits/split_v1.csv"
DEFAULT_SPLIT_SUMMARY = "data/splits/split_v1_summary.json"
DEFAULT_SPLIT_LOCK = "data/splits/split_v1_lock.json"
DEFAULT_PLAN = "data/unknown/planning/unknown_candidate_plan_v1.json"
DEFAULT_PLAN_LOCK = "data/unknown/planning/unknown_candidate_plan_v1_lock.json"

STRATUM_FIELDS = (
    "container",
    "source_rate_bucket",
    "channels",
    "quality",
    "duration_bucket",
)
STRATUM_VALUES = {
    "container": ("mp3", "riff_wave", "other"),
    "source_rate_bucket": (
        "32000",
        "44100",
        "48000",
        "above_48000",
        "other_eligible",
    ),
    "channels": ("mono", "stereo", "other"),
    "quality": ("A", "B"),
    "duration_bucket": (
        "below_3",
        "3_to_below_10",
        "10_to_below_30",
        "30_to_below_60",
        "at_least_60",
    ),
}
KNOWN_SOURCE_FIELDS = (
    "recording_id",
    "sha256",
    "session_group",
    "header_type",
    "source_sample_rate_hz",
    "channels",
    "quality",
    "canonical_duration_seconds",
)
KNOWN_SPLIT_FIELDS = (
    "recording_id",
    "relative_path",
    "sha256",
    "species_common_name",
    "session_group",
    "split",
    "split_seed",
    "source_manifest_sha256",
)
ASSIGNMENT_SLOT_FIELDS = (
    "slot_id",
    *STRATUM_FIELDS,
    "duration_seconds",
)
ASSIGNMENT_CANDIDATE_FIELDS = (
    "candidate_id",
    "session_group",
    *STRATUM_FIELDS,
    "duration_seconds",
)
FORBIDDEN_OUTCOME_FIELDS = (
    "audio_energy",
    "listening_preference",
    "model_output",
    "model_score",
    "embedding",
    "threshold_result",
    "test_outcome",
)
DURATION_LOG_SCALE = 1_000_000_000_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_CONFIG_KEYS = {
    "schema_version",
    "selection_seed",
    "candidate_pool_target_recordings_per_species",
    "target_recordings_per_species",
    "candidate_order",
    "fallback_policy",
    "reference_allocation",
    "strata",
    "matching",
    "descriptor_policy",
}
_REFERENCE_CONFIG = {
    "source": "frozen_known_test_only",
    "method": "joint_stratum_largest_remainder",
    "remainder_tie_break": "ascending_sha256_json_seed_stratum",
    "within_stratum_order": ("ascending_sha256_json_seed_stratum_recording_id_sha256"),
}
_MATCHING_CONFIG = {
    "algorithm": "deterministic_rectangular_hungarian_minimum_sum",
    "categorical_fields": list(STRATUM_FIELDS),
    "categorical_cost": "mismatch_count",
    "duration_cost": "absolute_log1p_seconds",
    "duration_log_scale": DURATION_LOG_SCALE,
    "tie_break": "sha256_json_seed_slot_id_candidate_id",
    "objective_priority": ("total_categorical_then_total_duration_then_total_pair_hash"),
}
_DESCRIPTOR_CONFIG = {
    "known_source_fields": list(KNOWN_SOURCE_FIELDS),
    "assignment_slot_fields": list(ASSIGNMENT_SLOT_FIELDS),
    "assignment_candidate_fields": list(ASSIGNMENT_CANDIDATE_FIELDS),
    "forbidden_outcome_fields": list(FORBIDDEN_OUTCOME_FIELDS),
}
_EXPECTED_FALLBACK_POLICY = (
    "inactive_until_one_primary_complete_terminal_inventory_has_fewer_than_40_eligible_sessions"
)
_PLAN_KEYS = {
    "schema_version",
    "created_at_utc",
    "protocol",
    "source_bindings",
    "candidate_queues",
    "known_test_reference",
}
_LOCK_KEYS = {
    "schema_version",
    "locked_at_utc",
    "ready_for_candidate_qc",
    "selection_seed",
    "candidate_pool_target_recordings_per_species",
    "target_recordings_per_species",
    "primary_species_count",
    "inactive_fallback_count",
    "candidate_recordings_total",
    "candidate_set_sha256",
    "reference_slots",
    "reference_slot_set_sha256",
    "plan_sha256",
    "artifacts",
}


class UnknownPlanningError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_utc_timestamp(value: Any, context: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise UnknownPlanningError(f"{context} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise UnknownPlanningError(f"{context} is not a valid UTC timestamp")
    return text


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise UnknownPlanningError(
            f"{context} fields are invalid; missing={missing}, unexpected={unexpected}"
        )


def _project_path(path: str | Path, context: str) -> Path:
    resolved = resolve_project_path(path)
    if not is_relative_to(resolved, PROJECT_ROOT):
        raise UnknownPlanningError(f"{context} must be inside the project")
    if not resolved.is_file():
        raise UnknownPlanningError(f"{context} does not exist: {resolved}")
    return resolved


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    if not is_relative_to(resolved, PROJECT_ROOT):
        raise UnknownPlanningError("artifact path leaves the project")
    return resolved.relative_to(PROJECT_ROOT).as_posix()


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnknownPlanningError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise UnknownPlanningError(f"JSON artifact is not an object: {path}")
    return value, sha256_bytes(payload)


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _create_json_exclusive(path: str | Path, value: Any) -> Path:
    destination = require_safe_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=False,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _load_config_snapshot(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    config_path = _project_path(path, "unknown selection config")
    payload = config_path.read_bytes()
    try:
        config = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise UnknownPlanningError("unknown selection config is not valid UTF-8 TOML") from exc
    if not isinstance(config, dict):
        raise UnknownPlanningError("unknown selection config is not a TOML table")
    _validate_config(config)
    return config_path, config, sha256_bytes(payload)


def _validate_config(config: dict[str, Any]) -> None:
    _require_exact_keys(config, _CONFIG_KEYS, "unknown selection config")
    scalar_expectations = {
        "schema_version": UNKNOWN_SELECTION_CONFIG_SCHEMA_VERSION,
        "selection_seed": SELECTION_SEED,
        "candidate_pool_target_recordings_per_species": CANDIDATE_POOL_TARGET,
        "target_recordings_per_species": TARGET_RECORDINGS,
        "candidate_order": "ascending_sha256_json_seed_scientific_name_recording_id",
        "fallback_policy": _EXPECTED_FALLBACK_POLICY,
    }
    if any(config.get(key) != expected for key, expected in scalar_expectations.items()):
        raise UnknownPlanningError("unknown selection scalar protocol is not locked")
    if config.get("reference_allocation") != _REFERENCE_CONFIG:
        raise UnknownPlanningError("unknown reference-allocation protocol is not locked")
    expected_strata = {
        "field_order": list(STRATUM_FIELDS),
        **{field: list(values) for field, values in STRATUM_VALUES.items()},
    }
    if config.get("strata") != expected_strata:
        raise UnknownPlanningError("unknown selection strata are not locked")
    if config.get("matching") != _MATCHING_CONFIG:
        raise UnknownPlanningError("unknown matching protocol is not locked")
    if config.get("descriptor_policy") != _DESCRIPTOR_CONFIG:
        raise UnknownPlanningError("unknown descriptor policy is not locked")


def load_unknown_selection_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the exact metadata-only unknown-selection protocol."""
    _, config, _ = _load_config_snapshot(path)
    return copy.deepcopy(config)


def _strict_recording_id(recording: Mapping[str, Any], context: str) -> str:
    identifiers: list[str] = []
    for field in ("id", "nr"):
        value = recording.get(field)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            raise UnknownPlanningError(f"{context} has an invalid recording ID")
        text = str(value).removeprefix("XC")
        if not re.fullmatch(r"[1-9][0-9]*", text):
            raise UnknownPlanningError(f"{context} has an invalid recording ID")
        identifiers.append(text)
    if not identifiers or len(set(identifiers)) != 1:
        raise UnknownPlanningError(f"{context} has a missing or conflicting recording ID")
    return identifiers[0]


def build_candidate_queues(sealed_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build full recording queues without using API response position or audio evidence."""
    species_entries = sealed_metadata.get("species")
    if not isinstance(species_entries, Mapping):
        raise UnknownPlanningError("sealed unknown metadata has no species table")
    expected_names = [identity[3] for identity in LOCKED_UNKNOWN_SPECIES]
    if set(species_entries) != set(expected_names):
        raise UnknownPlanningError("sealed unknown metadata species set is not locked")

    queues: list[dict[str, Any]] = []
    all_candidate_ids: set[str] = set()
    for role, active, common_name, scientific_name, difficulty_group in LOCKED_UNKNOWN_SPECIES:
        raw_entry = species_entries[scientific_name]
        if not isinstance(raw_entry, Mapping):
            raise UnknownPlanningError(f"metadata entry is invalid: {scientific_name}")
        identity = {
            "role": role,
            "active": active,
            "common_name": common_name,
            "scientific_name": scientific_name,
            "difficulty_group": difficulty_group,
        }
        if any(raw_entry.get(key) != value for key, value in identity.items()):
            raise UnknownPlanningError(f"metadata identity is invalid: {scientific_name}")
        pages = raw_entry.get("pages")
        snapshot = raw_entry.get("snapshot")
        if not isinstance(pages, Mapping) or not isinstance(snapshot, Mapping):
            raise UnknownPlanningError(f"metadata inventory is incomplete: {scientific_name}")
        recordings: list[dict[str, Any]] = []
        species_ids: set[str] = set()
        genus, specific_epithet = scientific_name.split()
        for page in pages.values():
            if not isinstance(page, Mapping) or not isinstance(page.get("recordings"), list):
                raise UnknownPlanningError(f"metadata page is invalid: {scientific_name}")
            for raw_recording in page["recordings"]:
                if not isinstance(raw_recording, Mapping):
                    raise UnknownPlanningError(f"metadata recording is invalid: {scientific_name}")
                unexpected = set(raw_recording) - PERSISTED_RECORDING_FIELDS
                if unexpected:
                    raise UnknownPlanningError(
                        f"metadata recording has unapproved fields: {sorted(unexpected)}"
                    )
                if raw_recording.get("gen") != genus or raw_recording.get("sp") != specific_epithet:
                    raise UnknownPlanningError(
                        f"metadata recording identity is invalid: {scientific_name}"
                    )
                group_values = [
                    str(raw_recording[field]).strip().casefold()
                    for field in ("grp", "group")
                    if raw_recording.get(field) not in (None, "")
                ]
                if not group_values or any(value != "birds" for value in group_values):
                    raise UnknownPlanningError(
                        f"metadata recording group is invalid: {scientific_name}"
                    )
                numeric_id = _strict_recording_id(
                    raw_recording, f"{scientific_name} metadata recording"
                )
                candidate_id = f"XC{numeric_id}"
                if candidate_id in species_ids or candidate_id in all_candidate_ids:
                    raise UnknownPlanningError(f"duplicate unknown candidate ID: {candidate_id}")
                species_ids.add(candidate_id)
                all_candidate_ids.add(candidate_id)
                order_sha256 = sha256_json(
                    {
                        "seed": SELECTION_SEED,
                        "scientific_name": scientific_name,
                        "recording_id": candidate_id,
                    }
                )
                recordings.append(
                    {
                        "candidate_id": candidate_id,
                        "order_sha256": order_sha256,
                        "metadata": copy.deepcopy(dict(raw_recording)),
                    }
                )
        expected_count = snapshot.get("num_recordings")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != len(recordings)
        ):
            raise UnknownPlanningError(f"metadata inventory count is invalid: {scientific_name}")
        recordings.sort(
            key=lambda item: (
                item["order_sha256"],
                int(item["candidate_id"].removeprefix("XC")),
            )
        )
        inventory_recordings = len(recordings)
        inventory_shortfall = max(0, CANDIDATE_POOL_TARGET - inventory_recordings)
        candidates = [
            {**candidate, "queue_rank": index}
            for index, candidate in enumerate(recordings, start=1)
        ]
        queues.append(
            {
                **identity,
                "activation_status": (
                    "active_primary_queue" if active else "inactive_fallback_until_protocol_gate"
                ),
                "candidate_pool_target_recordings": CANDIDATE_POOL_TARGET,
                "candidate_pool_inventory_status": (
                    "complete_inventory_below_target"
                    if inventory_shortfall
                    else "inventory_at_or_above_target"
                ),
                "candidate_pool_inventory_shortfall_recordings": inventory_shortfall,
                "target_recordings": TARGET_RECORDINGS,
                "inventory_recordings": inventory_recordings,
                "queue_sha256": sha256_json(candidates),
                "candidates": candidates,
            }
        )
    return queues


def _strict_positive_decimal(value: Any, context: str) -> Decimal:
    if isinstance(value, bool):
        raise UnknownPlanningError(f"{context} must be a positive finite number")
    try:
        number = Decimal(str(value))
    except Exception as exc:
        raise UnknownPlanningError(f"{context} must be a positive finite number") from exc
    if not number.is_finite() or number <= 0:
        raise UnknownPlanningError(f"{context} must be a positive finite number")
    return number


def _container_bucket(header_type: str) -> str:
    if header_type in {"mp3_id3", "mpeg_audio"}:
        return "mp3"
    if header_type in {"riff_wave", "rf64_wave"}:
        return "riff_wave"
    return "other"


def _source_rate_bucket(value: Any) -> str:
    if isinstance(value, bool):
        raise UnknownPlanningError("source sample rate must be an eligible integer")
    try:
        rate = int(str(value))
    except ValueError as exc:
        raise UnknownPlanningError("source sample rate must be an eligible integer") from exc
    if str(rate) != str(value).strip() or rate < 32000:
        raise UnknownPlanningError("source sample rate must be an eligible integer")
    if rate in {32000, 44100, 48000}:
        return str(rate)
    if rate > 48000:
        return "above_48000"
    return "other_eligible"


def _channel_bucket(value: Any) -> str:
    if isinstance(value, bool):
        raise UnknownPlanningError("channel count must be a positive integer")
    try:
        channels = int(str(value))
    except ValueError as exc:
        raise UnknownPlanningError("channel count must be a positive integer") from exc
    if str(channels) != str(value).strip() or channels < 1:
        raise UnknownPlanningError("channel count must be a positive integer")
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return "other"


def _duration_bucket(duration: Decimal) -> str:
    if duration < 3:
        return "below_3"
    if duration < 10:
        return "3_to_below_10"
    if duration < 30:
        return "10_to_below_30"
    if duration < 60:
        return "30_to_below_60"
    return "at_least_60"


def _known_descriptor(row: Mapping[str, str]) -> dict[str, Any]:
    outcome_fields = set(row).intersection(FORBIDDEN_OUTCOME_FIELDS)
    if outcome_fields:
        raise UnknownPlanningError(
            f"known test row contains outcome-dependent fields: {sorted(outcome_fields)}"
        )
    missing = [field for field in KNOWN_SOURCE_FIELDS if not str(row.get(field, "")).strip()]
    if missing:
        raise UnknownPlanningError(f"known test source descriptor is missing: {missing}")
    sha256 = row["sha256"]
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise UnknownPlanningError("known test source SHA-256 is invalid")
    duration = _strict_positive_decimal(
        row["canonical_duration_seconds"], "known canonical duration"
    )
    quality = row["quality"]
    if quality not in STRATUM_VALUES["quality"]:
        raise UnknownPlanningError("known test quality must be A or B")
    return {
        "recording_id": row["recording_id"],
        "sha256": sha256,
        "session_group": row["session_group"],
        "container": _container_bucket(row["header_type"]),
        "source_rate_bucket": _source_rate_bucket(row["source_sample_rate_hz"]),
        "channels": _channel_bucket(row["channels"]),
        "quality": quality,
        "duration_bucket": _duration_bucket(duration),
        "duration_seconds": format(duration, "f"),
    }


def _stratum_tuple(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(descriptor.get(field) or "") for field in STRATUM_FIELDS)
    for field, value in zip(STRATUM_FIELDS, values, strict=True):
        if value not in STRATUM_VALUES[field]:
            raise UnknownPlanningError(f"descriptor has invalid {field}: {value!r}")
    return values


def _stratum_object(values: Sequence[str]) -> dict[str, str]:
    return dict(zip(STRATUM_FIELDS, values, strict=True))


def allocate_reference_slots(
    known_test_descriptors: Sequence[Mapping[str, Any]],
    target: int = TARGET_RECORDINGS,
    seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    """Allocate exact known-test joint-stratum slots by largest remainder."""
    if target != TARGET_RECORDINGS or seed != SELECTION_SEED:
        raise UnknownPlanningError("reference allocation target and seed are locked")
    if len(known_test_descriptors) < target:
        raise UnknownPlanningError("known test has fewer recordings than reference slots")
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for raw in known_test_descriptors:
        descriptor = dict(raw)
        expected = {
            "recording_id",
            "sha256",
            "session_group",
            *STRATUM_FIELDS,
            "duration_seconds",
        }
        _require_exact_keys(descriptor, expected, "known test descriptor")
        recording_id = str(descriptor["recording_id"])
        if not recording_id or recording_id in seen_ids:
            raise UnknownPlanningError("known test recording IDs must be unique and nonempty")
        seen_ids.add(recording_id)
        _strict_positive_decimal(descriptor["duration_seconds"], "known test duration")
        by_stratum[_stratum_tuple(descriptor)].append(descriptor)

    population = len(known_test_descriptors)
    floors: dict[tuple[str, ...], int] = {}
    remainders: dict[tuple[str, ...], int] = {}
    for stratum, rows in by_stratum.items():
        numerator = len(rows) * target
        floors[stratum], remainders[stratum] = divmod(numerator, population)
    remaining = target - sum(floors.values())
    remainder_order = sorted(
        by_stratum,
        key=lambda stratum: (
            -remainders[stratum],
            sha256_json({"seed": seed, "stratum": _stratum_object(stratum)}),
            stratum,
        ),
    )
    allocations = dict(floors)
    for stratum in remainder_order[:remaining]:
        allocations[stratum] += 1
    if sum(allocations.values()) != target:
        raise UnknownPlanningError("largest-remainder allocation did not produce 40 slots")

    selected: list[dict[str, Any]] = []
    allocation_records: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda row: (
                sha256_json(
                    {
                        "seed": seed,
                        "stratum": _stratum_object(stratum),
                        "recording_id": row["recording_id"],
                        "sha256": row["sha256"],
                    }
                ),
                row["recording_id"],
            ),
        )
        count = allocations[stratum]
        if count > len(ordered):
            raise UnknownPlanningError("stratum allocation exceeds its source population")
        selected_rows = ordered[:count]
        selected.extend(selected_rows)
        allocation_records.append(
            {
                "stratum": _stratum_object(stratum),
                "population": len(ordered),
                "quota_numerator": len(ordered) * target,
                "quota_denominator": population,
                "floor_slots": floors[stratum],
                "remainder_numerator": remainders[stratum],
                "allocated_slots": count,
                "selected_recording_ids": [row["recording_id"] for row in selected_rows],
            }
        )

    selected.sort(
        key=lambda row: (
            sha256_json(
                {
                    "seed": seed,
                    "reference_slot": row["recording_id"],
                    "sha256": row["sha256"],
                }
            ),
            row["recording_id"],
        )
    )
    slots = [
        {
            "slot_id": f"known_test_reference_{index:02d}",
            "source_recording_id": row["recording_id"],
            "source_sha256": row["sha256"],
            "source_session_group": row["session_group"],
            **{field: row[field] for field in STRATUM_FIELDS},
            "duration_seconds": row["duration_seconds"],
        }
        for index, row in enumerate(selected, start=1)
    ]
    return {
        "source": "frozen_known_test_only",
        "source_recordings": population,
        "target_slots": target,
        "allocation_method": "joint_stratum_largest_remainder",
        "stratum_fields": list(STRATUM_FIELDS),
        "allocations": allocation_records,
        "reference_slots": slots,
        "reference_slot_set_sha256": sha256_json(slots),
    }


def assignment_slots_from_reference(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Remove known-test identity bindings before later candidate matching."""
    slots = reference.get("reference_slots")
    if not isinstance(slots, list):
        raise UnknownPlanningError("reference slot table is invalid")
    return [
        {
            "slot_id": slot["slot_id"],
            **{field: slot[field] for field in STRATUM_FIELDS},
            "duration_seconds": slot["duration_seconds"],
        }
        for slot in slots
    ]


def _validate_assignment_descriptor(
    descriptor: Mapping[str, Any], expected_fields: tuple[str, ...], context: str
) -> dict[str, Any]:
    value = dict(descriptor)
    _require_exact_keys(value, set(expected_fields), context)
    forbidden = set(value).intersection(FORBIDDEN_OUTCOME_FIELDS)
    if forbidden:
        raise UnknownPlanningError(
            f"{context} contains outcome-dependent fields: {sorted(forbidden)}"
        )
    identifier_field = "slot_id" if context == "assignment slot" else "candidate_id"
    if not str(value.get(identifier_field) or "").strip():
        raise UnknownPlanningError(f"{context} has an empty identifier")
    if context == "assignment candidate" and not str(value.get("session_group") or "").strip():
        raise UnknownPlanningError("assignment candidate has no session group")
    duration = _strict_positive_decimal(value["duration_seconds"], f"{context} duration")
    if value.get("duration_bucket") != _duration_bucket(duration):
        raise UnknownPlanningError(f"{context} duration bucket is inconsistent")
    _stratum_tuple(value)
    value["duration_seconds"] = format(duration, "f")
    return value


def _duration_distance_units(left: Any, right: Any) -> int:
    left_duration = _strict_positive_decimal(left, "slot duration")
    right_duration = _strict_positive_decimal(right, "candidate duration")
    with localcontext() as context:
        context.prec = 60
        distance = abs((left_duration + 1).ln() - (right_duration + 1).ln())
        return int((distance * DURATION_LOG_SCALE).to_integral_value(rounding=ROUND_HALF_EVEN))


def _hungarian(costs: Sequence[Sequence[int]]) -> list[int]:
    rows = len(costs)
    columns = len(costs[0]) if rows else 0
    if rows == 0 or columns < rows or any(len(row) != columns for row in costs):
        raise UnknownPlanningError("Hungarian cost matrix must have 1 <= rows <= columns")
    u = [0] * (rows + 1)
    v = [0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        p[0] = row_index
        minimum: list[int | None] = [None] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            source_row = p[column0]
            delta: int | None = None
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[source_row - 1][column - 1] - u[source_row] - v[column]
                if minimum[column] is None or current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                candidate_delta = minimum[column]
                if candidate_delta is not None and (
                    delta is None
                    or candidate_delta < delta
                    or (candidate_delta == delta and column < column1)
                ):
                    delta = candidate_delta
                    column1 = column
            if delta is None:
                raise UnknownPlanningError("Hungarian assignment could not advance")
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                elif minimum[column] is not None:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise UnknownPlanningError("Hungarian assignment is incomplete")
    return assignment


def assign_candidates_to_slots(
    slots: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    """Assign QC-qualified session representatives with the locked cost priority."""
    if seed != SELECTION_SEED:
        raise UnknownPlanningError("candidate assignment seed is locked")
    validated_slots = sorted(
        (
            _validate_assignment_descriptor(slot, ASSIGNMENT_SLOT_FIELDS, "assignment slot")
            for slot in slots
        ),
        key=lambda item: item["slot_id"],
    )
    validated_candidates = sorted(
        (
            _validate_assignment_descriptor(
                candidate, ASSIGNMENT_CANDIDATE_FIELDS, "assignment candidate"
            )
            for candidate in candidates
        ),
        key=lambda item: item["candidate_id"],
    )
    slot_ids = [item["slot_id"] for item in validated_slots]
    candidate_ids = [item["candidate_id"] for item in validated_candidates]
    sessions = [item["session_group"] for item in validated_candidates]
    if len(slot_ids) != len(set(slot_ids)):
        raise UnknownPlanningError("assignment slot IDs must be unique")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise UnknownPlanningError("assignment candidate IDs must be unique")
    if len(sessions) != len(set(sessions)):
        raise UnknownPlanningError("assignment candidates must be unique session representatives")
    if len(validated_slots) != TARGET_RECORDINGS:
        raise UnknownPlanningError("candidate assignment requires exactly 40 reference slots")
    if len(validated_candidates) < len(validated_slots):
        raise UnknownPlanningError("candidate assignment has fewer candidates than slots")

    components: list[list[tuple[int, int, int]]] = []
    maximum_duration = 0
    maximum_tie = (1 << 256) - 1
    for slot in validated_slots:
        component_row: list[tuple[int, int, int]] = []
        for candidate in validated_candidates:
            mismatch = sum(slot[field] != candidate[field] for field in STRATUM_FIELDS)
            duration_units = _duration_distance_units(
                slot["duration_seconds"], candidate["duration_seconds"]
            )
            maximum_duration = max(maximum_duration, duration_units)
            tie_value = int(
                sha256_json(
                    {
                        "seed": seed,
                        "slot_id": slot["slot_id"],
                        "candidate_id": candidate["candidate_id"],
                    }
                ),
                16,
            )
            component_row.append((mismatch, duration_units, tie_value))
        components.append(component_row)

    assignment_size = len(validated_slots)
    tie_base = assignment_size * maximum_tie + 1
    duration_total_bound = assignment_size * maximum_duration
    categorical_base = duration_total_bound * tie_base + assignment_size * maximum_tie + 1
    costs = [
        [mismatch * categorical_base + duration * tie_base + tie for mismatch, duration, tie in row]
        for row in components
    ]
    selected_columns = _hungarian(costs)
    assignments: list[dict[str, Any]] = []
    total_mismatches = 0
    total_duration_units = 0
    total_pair_hash = 0
    for row_index, column_index in enumerate(selected_columns):
        mismatch, duration_units, tie_value = components[row_index][column_index]
        slot = validated_slots[row_index]
        candidate = validated_candidates[column_index]
        total_mismatches += mismatch
        total_duration_units += duration_units
        total_pair_hash += tie_value
        assignments.append(
            {
                "slot_id": slot["slot_id"],
                "candidate_id": candidate["candidate_id"],
                "candidate_session_group": candidate["session_group"],
                "categorical_mismatches": mismatch,
                "duration_distance_units": duration_units,
                "pair_tie_sha256": f"{tie_value:064x}",
            }
        )
    result = {
        "algorithm": _MATCHING_CONFIG["algorithm"],
        "selection_seed": seed,
        "objective_priority": _MATCHING_CONFIG["objective_priority"],
        "assignments": assignments,
        "total_categorical_mismatches": total_mismatches,
        "total_duration_distance_units": total_duration_units,
        "total_pair_hash_integer": str(total_pair_hash),
    }
    result["assignment_sha256"] = sha256_json(result)
    return result


def _verify_known_inputs(
    manifest_path: Path,
    review_lock_path: Path,
    split_path: Path,
    split_summary_path: Path,
    split_lock_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    manifest_rows, manifest_sha256 = read_csv_snapshot(manifest_path)
    review_lock_sha256 = sha256_file(review_lock_path)
    try:
        review_record = verify_review_lock(review_lock_path, manifest_path)
    except ValueError as exc:
        raise UnknownPlanningError(f"known review lock is invalid: {exc}") from exc
    require_unchanged(review_lock_path, review_lock_sha256)
    if (
        review_record.get("ready_for_split") is not True
        or review_record.get("final_manifest_sha256") != manifest_sha256
    ):
        raise UnknownPlanningError("known review lock does not bind the final manifest")

    split_rows, split_sha256 = read_csv_snapshot(split_path)
    if not split_rows or any(set(row) != set(KNOWN_SPLIT_FIELDS) for row in split_rows):
        raise UnknownPlanningError("known split descriptor fields are invalid")
    summary, summary_sha256 = _json_snapshot(split_summary_path)
    split_lock, split_lock_sha256 = _json_snapshot(split_lock_path)
    if split_lock.get("schema_version") != "1.2":
        raise UnknownPlanningError("known split lock schema is invalid")
    expected_split_lock = {
        "source_manifest_sha256": manifest_sha256,
        "review_lock_sha256": review_lock_sha256,
        "split_sha256": split_sha256,
        "summary_sha256": summary_sha256,
        "split_seed": SELECTION_SEED,
        "recordings": len(split_rows),
        "recording_set_sha256": sha256_json(
            sorted(row.get("recording_id", "") for row in split_rows)
        ),
    }
    if any(split_lock.get(key) != value for key, value in expected_split_lock.items()):
        raise UnknownPlanningError("known split lock binding is invalid")
    expected_summary = {
        "schema_version": "1.2",
        "source_manifest_sha256": manifest_sha256,
        "review_lock_sha256": review_lock_sha256,
        "split_sha256": split_sha256,
        "split_seed": SELECTION_SEED,
        "recordings": len(split_rows),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise UnknownPlanningError("known split summary binding is invalid")

    manifest_by_id: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        recording_id = row.get("recording_id", "")
        if not recording_id or recording_id in manifest_by_id:
            raise UnknownPlanningError("known manifest recording IDs are invalid")
        manifest_by_id[recording_id] = row
    included_ids = {
        recording_id
        for recording_id, row in manifest_by_id.items()
        if row.get("local_qc_status") == "include"
    }
    split_ids: set[str] = set()
    test_descriptors: list[dict[str, Any]] = []
    for split_row in split_rows:
        recording_id = split_row.get("recording_id", "")
        if not recording_id or recording_id in split_ids or recording_id not in manifest_by_id:
            raise UnknownPlanningError("known split recording IDs are invalid")
        split_ids.add(recording_id)
        if split_row.get("split") not in {"train", "validation", "test"}:
            raise UnknownPlanningError("known split contains an invalid split name")
        if split_row.get("split_seed") != str(SELECTION_SEED):
            raise UnknownPlanningError("known split row seed is invalid")
        if split_row.get("source_manifest_sha256") != manifest_sha256:
            raise UnknownPlanningError("known split row manifest binding is invalid")
        manifest_row = manifest_by_id[recording_id]
        for field in ("relative_path", "sha256", "species_common_name", "session_group"):
            if split_row.get(field) != manifest_row.get(field):
                raise UnknownPlanningError(f"known split row {field} binding is invalid")
        if split_row["split"] == "test":
            test_descriptors.append(_known_descriptor(manifest_row))
    if split_ids != included_ids:
        raise UnknownPlanningError("known split is not the exact included manifest set")
    if len(test_descriptors) < TARGET_RECORDINGS:
        raise UnknownPlanningError("known test split cannot supply 40 reference slots")
    hashes = {
        "known_manifest_sha256": manifest_sha256,
        "review_lock_sha256": review_lock_sha256,
        "split_sha256": split_sha256,
        "split_summary_sha256": summary_sha256,
        "split_lock_sha256": split_lock_sha256,
    }
    return hashes, test_descriptors


def _artifact(path: Path, sha256: str | None = None) -> dict[str, str]:
    return {
        "path": _project_relative(path),
        "sha256": sha256 or sha256_file(path),
    }


def _deterministic_sections(
    config_path: Path,
    config_sha256: str,
    unknown_metadata_path: Path,
    unknown_metadata_lock_path: Path,
    manifest_path: Path,
    review_lock_path: Path,
    split_path: Path,
    split_summary_path: Path,
    split_lock_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[Path, str]]:
    unknown_metadata_lock_sha256 = sha256_file(unknown_metadata_lock_path)
    try:
        metadata_lock = verify_unknown_metadata_lock(
            unknown_metadata_lock_path, unknown_metadata_path
        )
    except ValueError as exc:
        raise UnknownPlanningError(f"sealed unknown metadata lock is invalid: {exc}") from exc
    require_unchanged(unknown_metadata_lock_path, unknown_metadata_lock_sha256)
    sealed_metadata, unknown_metadata_sha256 = _json_snapshot(unknown_metadata_path)
    if (
        metadata_lock.get("ready_for_candidate_planning") is not True
        or metadata_lock.get("sealed_cache_sha256") != unknown_metadata_sha256
        or metadata_lock.get("candidate_pool_target_recordings_per_species")
        != CANDIDATE_POOL_TARGET
        or metadata_lock.get("target_recordings_per_species") != TARGET_RECORDINGS
        or metadata_lock.get("primary_species_count") != 5
        or metadata_lock.get("inactive_fallback_count") != 1
    ):
        raise UnknownPlanningError("sealed unknown metadata lock protocol is invalid")

    known_hashes, test_descriptors = _verify_known_inputs(
        manifest_path,
        review_lock_path,
        split_path,
        split_summary_path,
        split_lock_path,
    )
    candidate_queues = build_candidate_queues(sealed_metadata)
    known_reference = allocate_reference_slots(test_descriptors)
    source_bindings = {
        "selection_config": _artifact(config_path, config_sha256),
        "unknown_metadata": _artifact(unknown_metadata_path, unknown_metadata_sha256),
        "unknown_metadata_lock": _artifact(
            unknown_metadata_lock_path, unknown_metadata_lock_sha256
        ),
        "known_manifest": _artifact(manifest_path, known_hashes["known_manifest_sha256"]),
        "review_lock": _artifact(review_lock_path, known_hashes["review_lock_sha256"]),
        "split": _artifact(split_path, known_hashes["split_sha256"]),
        "split_summary": _artifact(split_summary_path, known_hashes["split_summary_sha256"]),
        "split_lock": _artifact(split_lock_path, known_hashes["split_lock_sha256"]),
    }
    input_hashes = {
        config_path: config_sha256,
        unknown_metadata_path: unknown_metadata_sha256,
        unknown_metadata_lock_path: unknown_metadata_lock_sha256,
        manifest_path: known_hashes["known_manifest_sha256"],
        review_lock_path: known_hashes["review_lock_sha256"],
        split_path: known_hashes["split_sha256"],
        split_summary_path: known_hashes["split_summary_sha256"],
        split_lock_path: known_hashes["split_lock_sha256"],
    }
    return source_bindings, candidate_queues, known_reference, input_hashes


def _protocol_record() -> dict[str, Any]:
    return {
        "selection_seed": SELECTION_SEED,
        "candidate_pool_target_recordings_per_species": CANDIDATE_POOL_TARGET,
        "target_recordings_per_species": TARGET_RECORDINGS,
        "candidate_order": "ascending_sha256_json_seed_scientific_name_recording_id",
        "fallback_policy": _EXPECTED_FALLBACK_POLICY,
        "reference_allocation": copy.deepcopy(_REFERENCE_CONFIG),
        "strata": {
            "field_order": list(STRATUM_FIELDS),
            **{field: list(values) for field, values in STRATUM_VALUES.items()},
        },
        "matching": copy.deepcopy(_MATCHING_CONFIG),
        "descriptor_policy": copy.deepcopy(_DESCRIPTOR_CONFIG),
    }


def _lock_record(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    queues = plan["candidate_queues"]
    reference = plan["known_test_reference"]
    candidate_ids = sorted(
        candidate["candidate_id"] for queue in queues for candidate in queue["candidates"]
    )
    return {
        "schema_version": UNKNOWN_CANDIDATE_PLAN_LOCK_SCHEMA_VERSION,
        "locked_at_utc": _utc_now(),
        "ready_for_candidate_qc": True,
        "selection_seed": SELECTION_SEED,
        "candidate_pool_target_recordings_per_species": CANDIDATE_POOL_TARGET,
        "target_recordings_per_species": TARGET_RECORDINGS,
        "primary_species_count": sum(queue["role"] == "primary" for queue in queues),
        "inactive_fallback_count": sum(
            queue["role"] == "fallback" and not queue["active"] for queue in queues
        ),
        "candidate_recordings_total": len(candidate_ids),
        "candidate_set_sha256": sha256_json(candidate_ids),
        "reference_slots": reference["target_slots"],
        "reference_slot_set_sha256": reference["reference_slot_set_sha256"],
        "plan_sha256": sha256_file(plan_path),
        "artifacts": {
            **copy.deepcopy(plan["source_bindings"]),
            "plan": _artifact(plan_path),
        },
    }


def build_unknown_candidate_plan(
    config_path: str | Path = DEFAULT_CONFIG,
    unknown_metadata_path: str | Path = DEFAULT_UNKNOWN_METADATA,
    unknown_metadata_lock_path: str | Path = DEFAULT_UNKNOWN_METADATA_LOCK,
    manifest_path: str | Path = DEFAULT_KNOWN_MANIFEST,
    review_lock_path: str | Path = DEFAULT_REVIEW_LOCK,
    split_path: str | Path = DEFAULT_SPLIT,
    split_summary_path: str | Path = DEFAULT_SPLIT_SUMMARY,
    split_lock_path: str | Path = DEFAULT_SPLIT_LOCK,
    output_path: str | Path = DEFAULT_PLAN,
    lock_path: str | Path = DEFAULT_PLAN_LOCK,
) -> tuple[Path, Path, dict[str, Any]]:
    """Create the immutable metadata-only queue and known-reference plan."""
    destination = require_safe_output(output_path)
    lock_destination = require_safe_output(lock_path)
    if destination == lock_destination:
        raise UnknownPlanningError("candidate plan and lock paths must be distinct")
    if lock_destination.exists():
        if not destination.exists():
            raise FileExistsError("candidate plan lock exists without its plan")
        result = validate_unknown_candidate_plan(lock_destination, destination)
        existing_lock, _ = _json_snapshot(lock_destination)
        existing_artifacts = _resolve_lock_artifacts(existing_lock)
        requested_inputs = {
            "selection_config": _project_path(config_path, "unknown selection config"),
            "unknown_metadata": _project_path(unknown_metadata_path, "sealed unknown metadata"),
            "unknown_metadata_lock": _project_path(
                unknown_metadata_lock_path, "sealed unknown metadata lock"
            ),
            "known_manifest": _project_path(manifest_path, "known manifest"),
            "review_lock": _project_path(review_lock_path, "known review lock"),
            "split": _project_path(split_path, "known split"),
            "split_summary": _project_path(split_summary_path, "known split summary"),
            "split_lock": _project_path(split_lock_path, "known split lock"),
        }
        rebound = [
            name
            for name, requested_path in requested_inputs.items()
            if existing_artifacts[name] != requested_path
        ]
        if rebound:
            raise UnknownPlanningError(
                f"existing candidate plan cannot be rebound to different inputs: {rebound}"
            )
        return destination, lock_destination, result
    if destination.exists():
        raise FileExistsError("unlocked candidate plan already exists and will not be overwritten")

    with project_lock("unknown_candidate_planning"):
        if destination.exists() or lock_destination.exists():
            raise FileExistsError("candidate planning outputs appeared during planning")
        config_file, _, config_sha256 = _load_config_snapshot(config_path)
        unknown_metadata_file = _project_path(unknown_metadata_path, "sealed unknown metadata")
        unknown_metadata_lock_file = _project_path(
            unknown_metadata_lock_path, "sealed unknown metadata lock"
        )
        manifest_file = _project_path(manifest_path, "known manifest")
        review_lock_file = _project_path(review_lock_path, "known review lock")
        split_file = _project_path(split_path, "known split")
        split_summary_file = _project_path(split_summary_path, "known split summary")
        split_lock_file = _project_path(split_lock_path, "known split lock")
        source_bindings, queues, known_reference, input_hashes = _deterministic_sections(
            config_file,
            config_sha256,
            unknown_metadata_file,
            unknown_metadata_lock_file,
            manifest_file,
            review_lock_file,
            split_file,
            split_summary_file,
            split_lock_file,
        )
        plan = {
            "schema_version": UNKNOWN_CANDIDATE_PLAN_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "protocol": _protocol_record(),
            "source_bindings": source_bindings,
            "candidate_queues": queues,
            "known_test_reference": known_reference,
        }
        for path, digest in input_hashes.items():
            require_unchanged(path, digest)
        _create_json_exclusive(destination, plan)
        for path, digest in input_hashes.items():
            require_unchanged(path, digest)
        lock = _lock_record(destination, plan)
        _create_json_exclusive(lock_destination, lock)
        for path, digest in input_hashes.items():
            require_unchanged(path, digest)
    return (
        destination,
        lock_destination,
        {
            "valid": True,
            "ready_for_candidate_qc": True,
            "candidate_recordings_total": lock["candidate_recordings_total"],
            "reference_slots": lock["reference_slots"],
            "plan_sha256": lock["plan_sha256"],
        },
    )


def _resolve_lock_artifacts(lock: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    required = {
        "selection_config",
        "unknown_metadata",
        "unknown_metadata_lock",
        "known_manifest",
        "review_lock",
        "split",
        "split_summary",
        "split_lock",
        "plan",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise UnknownPlanningError("candidate plan lock artifact table is invalid")
    resolved: dict[str, Path] = {}
    for name in sorted(required):
        entry = artifacts[name]
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise UnknownPlanningError(f"candidate plan lock artifact is invalid: {name}")
        relative = str(entry.get("path") or "")
        digest = str(entry.get("sha256") or "")
        if not relative or Path(relative).is_absolute() or not _SHA256_PATTERN.fullmatch(digest):
            raise UnknownPlanningError(f"candidate plan lock artifact is invalid: {name}")
        path = resolve_project_path(relative)
        if not is_relative_to(path, PROJECT_ROOT):
            raise UnknownPlanningError(f"candidate plan artifact leaves project: {name}")
        if not path.is_file() or sha256_file(path) != digest:
            raise UnknownPlanningError(f"candidate plan artifact hash mismatch: {name}")
        resolved[name] = path
    return resolved


def validate_unknown_candidate_plan(
    lock_path: str | Path = DEFAULT_PLAN_LOCK,
    expected_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild deterministic sections and verify every bound planning artifact."""
    lock_file = _project_path(lock_path, "candidate plan lock")
    lock, _ = _json_snapshot(lock_file)
    _require_exact_keys(lock, _LOCK_KEYS, "candidate plan lock")
    if (
        lock.get("schema_version") != UNKNOWN_CANDIDATE_PLAN_LOCK_SCHEMA_VERSION
        or lock.get("ready_for_candidate_qc") is not True
        or lock.get("selection_seed") != SELECTION_SEED
        or lock.get("candidate_pool_target_recordings_per_species") != CANDIDATE_POOL_TARGET
        or lock.get("target_recordings_per_species") != TARGET_RECORDINGS
        or lock.get("primary_species_count") != 5
        or lock.get("inactive_fallback_count") != 1
    ):
        raise UnknownPlanningError("candidate plan lock protocol is invalid")
    _require_utc_timestamp(lock.get("locked_at_utc"), "candidate plan lock timestamp")
    artifacts = _resolve_lock_artifacts(lock)
    if expected_plan_path is not None:
        expected = _project_path(expected_plan_path, "expected candidate plan")
        if artifacts["plan"] != expected:
            raise UnknownPlanningError("candidate plan lock points to a different plan")
    plan, plan_sha256 = _json_snapshot(artifacts["plan"])
    _require_exact_keys(plan, _PLAN_KEYS, "candidate plan")
    if (
        plan.get("schema_version") != UNKNOWN_CANDIDATE_PLAN_SCHEMA_VERSION
        or plan.get("protocol") != _protocol_record()
        or lock.get("plan_sha256") != plan_sha256
    ):
        raise UnknownPlanningError("candidate plan protocol or hash is invalid")
    _require_utc_timestamp(plan.get("created_at_utc"), "candidate plan timestamp")
    config_path, _, config_sha256 = _load_config_snapshot(artifacts["selection_config"])
    source_bindings, queues, known_reference, input_hashes = _deterministic_sections(
        config_path,
        config_sha256,
        artifacts["unknown_metadata"],
        artifacts["unknown_metadata_lock"],
        artifacts["known_manifest"],
        artifacts["review_lock"],
        artifacts["split"],
        artifacts["split_summary"],
        artifacts["split_lock"],
    )
    if plan.get("source_bindings") != source_bindings:
        raise UnknownPlanningError("candidate plan source bindings are invalid")
    if plan.get("candidate_queues") != queues:
        raise UnknownPlanningError("candidate plan queues are not reproducible")
    if plan.get("known_test_reference") != known_reference:
        raise UnknownPlanningError("candidate plan reference slots are not reproducible")
    candidate_ids = sorted(
        candidate["candidate_id"] for queue in queues for candidate in queue["candidates"]
    )
    expected_lock_summary = {
        "candidate_recordings_total": len(candidate_ids),
        "candidate_set_sha256": sha256_json(candidate_ids),
        "reference_slots": TARGET_RECORDINGS,
        "reference_slot_set_sha256": known_reference["reference_slot_set_sha256"],
    }
    if any(lock.get(key) != value for key, value in expected_lock_summary.items()):
        raise UnknownPlanningError("candidate plan lock summary is invalid")
    for path, digest in input_hashes.items():
        require_unchanged(path, digest)
    return {
        "valid": True,
        "ready_for_candidate_qc": True,
        "candidate_recordings_total": len(candidate_ids),
        "reference_slots": TARGET_RECORDINGS,
        "plan_sha256": plan_sha256,
    }
