from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from bird_audio.audio import (
    AudioProbe,
    AudioToolError,
    FullDecodeResult,
    detect_header,
    probe_audio,
    verify_full_decode,
)
from bird_audio.hashing import sha256_bytes, sha256_file, sha256_json
from bird_audio.io_utils import read_csv_snapshot, require_unchanged
from bird_audio.locking import project_lock
from bird_audio.metadata import (
    _is_recognized_cc_licence_uri,
    _normalize_text,
    _secondary_matches_different_target,
    assign_session_groups,
)
from bird_audio.paths import PROJECT_ROOT, is_relative_to, require_safe_output, resolve_project_path
from bird_audio.secure_audio_download import (
    RetryableAudioDownloadError,
    TerminalAudioUnavailableError,
)
from bird_audio.unknown_planning import (
    CANDIDATE_POOL_TARGET,
    FORBIDDEN_OUTCOME_FIELDS,
    SELECTION_SEED,
    TARGET_RECORDINGS,
    assign_candidates_to_slots,
    assignment_slots_from_reference,
    validate_unknown_candidate_plan,
)

UNKNOWN_AUDIO_CONFIG_SCHEMA_VERSION = "1.0"
UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION = "1.0"
DEFAULT_CONFIG = "configs/unknown_audio.toml"
DEFAULT_PLAN = "data/unknown/planning/unknown_candidate_plan_v1.json"
DEFAULT_PLAN_LOCK = "data/unknown/planning/unknown_candidate_plan_v1_lock.json"

TERMINAL_DISPOSITIONS = frozenset(
    {
        "metadata_excluded",
        "session_noncanonical",
        "download_unavailable_terminal",
        "audio_qc_excluded",
        "eligible",
        "not_evaluated_pool_target_reached",
    }
)
FORBIDDEN_FIELDS = frozenset(FORBIDDEN_OUTCOME_FIELDS)
_XC_ID = re.compile(r"^XC([1-9][0-9]*)$")
_API_DURATION = re.compile(r"^([0-9]+):([0-5][0-9])$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_HEADER_DETECTION_STATUSES = frozenset({"recognized", "unsupported"})
_PROBE_STATUSES = frozenset({"ok", "content_failure"})
_FULL_DECODE_STATUSES = frozenset(
    {"ok", "warning", "invalid_duration", "content_failure", "not_run"}
)
_DUPLICATE_QC_REASONS = frozenset(
    {
        "exact_duplicate_of_retained_known",
        "exact_duplicate_of_earlier_unknown_candidate",
    }
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "selection_seed",
    "candidate_pool_target_recordings_per_species",
    "target_recordings_per_species",
    "expected_retained_known_recordings",
    "session_scope",
    "cross_unknown_species_session_policy",
    "session_review_policy",
    "canonical_policy",
    "fallback_policy",
    "inputs",
    "metadata",
    "session",
    "audio_qc",
    "estimation",
    "download",
    "outputs",
}
_TABLE_KEYS = {
    "inputs": {"candidate_plan", "candidate_plan_lock", "known_manifest"},
    "metadata": {
        "accepted_quality",
        "required_group",
        "recognized_licence_policy",
        "secondary_label_policy",
        "api_sample_rate_policy",
        "api_duration_policy",
    },
    "session": {
        "coordinate_radius_km",
        "missing_date_policy",
        "missing_recordist_policy",
        "invalid_coordinates_policy",
        "same_individual_reference_policy",
    },
    "audio_qc": {
        "minimum_source_sample_rate_hz",
        "accepted_channels",
        "accepted_header_types",
        "minimum_decoded_to_ffprobe_duration_ratio",
        "maximum_decoded_to_ffprobe_duration_ratio",
        "full_decode_warning_policy",
        "canonical_duration_source",
    },
    "estimation": {
        "assumed_source_bitrate_kbps",
        "working_space_multiplier",
        "minimum_free_space_reserve_bytes",
    },
    "download": {
        "allowed_initial_hosts",
        "allowed_redirect_hosts",
        "maximum_redirects",
        "request_interval_seconds",
        "timeout_seconds",
        "total_timeout_seconds",
        "maximum_retries",
        "maximum_retry_after_seconds",
        "chunk_size_bytes",
        "maximum_file_size_bytes",
        "user_agent",
        "proxy_policy",
        "cookie_policy",
    },
    "outputs": {"working_directory", "raw_directory", "audit", "audit_lock"},
}


class UnknownAudioError(ValueError):
    pass


class UnknownAudioRetryableError(RuntimeError):
    """A nonterminal acquisition failure that must block deterministic progress."""


class UnknownAudioTerminalUnavailableError(RuntimeError):
    """A permanent recording absence that can count toward inventory exhaustion."""


class DownloadClient(Protocol):
    def download(
        self, candidate_id: str, source_url: str, destination: Path
    ) -> Mapping[str, Any] | Any: ...


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is not None:
        callback(copy.deepcopy(event))


def _exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnknownAudioError(f"{context} must be a table")
    actual = set(value)
    if actual != expected:
        raise UnknownAudioError(
            f"{context} keys are not locked: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _strict_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UnknownAudioError(f"{context} must be an integer at least {minimum}")
    return value


def _strict_number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnknownAudioError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise UnknownAudioError(f"{context} must be a finite positive number")
    return result


def _string_list(value: Any, context: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise UnknownAudioError(f"{context} must be a nonempty string list")
    if len(value) != len(set(value)):
        raise UnknownAudioError(f"{context} contains duplicates")
    return list(value)


def _project_input(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise UnknownAudioError(f"{context} must be a project-relative path")
    result = resolve_project_path(value)
    if not is_relative_to(result, PROJECT_ROOT):
        raise UnknownAudioError(f"{context} leaves the project")
    return result


def _locked_output_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise UnknownAudioError(f"{context} must be a project-relative path")
    lexical = PROJECT_ROOT / value
    resolved = require_safe_output(value)
    if resolved != lexical:
        raise UnknownAudioError(f"{context} traverses a symbolic link")
    current = PROJECT_ROOT
    for component in Path(value).parts:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise UnknownAudioError(f"{context} contains a symbolic-link component")
    return lexical


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(config, _TOP_LEVEL_KEYS, "unknown audio config")
    for table, keys in _TABLE_KEYS.items():
        _exact_keys(config.get(table), keys, f"unknown audio config [{table}]")
    expected_scalars = {
        "schema_version": UNKNOWN_AUDIO_CONFIG_SCHEMA_VERSION,
        "selection_seed": SELECTION_SEED,
        "candidate_pool_target_recordings_per_species": CANDIDATE_POOL_TARGET,
        "target_recordings_per_species": TARGET_RECORDINGS,
        "expected_retained_known_recordings": 1792,
        "session_scope": "joint_retained_known_and_all_unknown",
        "cross_unknown_species_session_policy": "exclude_entire_component",
        "session_review_policy": "exclude",
        "canonical_policy": (
            "earliest_queue_rank_metadata_eligible_before_audio_qc_no_replacement"
        ),
        "fallback_policy": ("activate_only_after_exactly_one_primary_terminally_has_fewer_than_40"),
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise UnknownAudioError(f"unknown audio config {key} is not locked")

    inputs = config["inputs"]
    if inputs != {
        "candidate_plan": DEFAULT_PLAN,
        "candidate_plan_lock": DEFAULT_PLAN_LOCK,
        "known_manifest": "data/manifests/recordings.csv",
    }:
        raise UnknownAudioError("unknown audio input paths are not locked")
    for key in _TABLE_KEYS["inputs"]:
        _project_input(inputs[key], f"inputs.{key}")
    metadata = config["metadata"]
    if metadata != {
        "accepted_quality": ["A", "B"],
        "required_group": "birds",
        "recognized_licence_policy": "canonical_creative_commons_uri",
        "secondary_label_policy": "exclude_every_other_configured_study_species",
        "api_sample_rate_policy": "advisory_only",
        "api_duration_policy": "advisory_only",
    }:
        raise UnknownAudioError("unknown audio metadata policy is not locked")
    session = config["session"]
    if session != {
        "coordinate_radius_km": 1.0,
        "missing_date_policy": "exclude_after_conservative_grouping",
        "missing_recordist_policy": "exclude_after_conservative_grouping",
        "invalid_coordinates_policy": "exclude_after_conservative_grouping",
        "same_individual_reference_policy": "global_link_or_exclude_unresolved",
    }:
        raise UnknownAudioError("unknown audio session policy is not locked")

    qc = config["audio_qc"]
    if (
        _strict_int(qc["minimum_source_sample_rate_hz"], "minimum sample rate", minimum=1) != 32000
        or _string_list(qc["accepted_header_types"], "accepted header types")
        != ["mp3_id3", "mpeg_audio", "riff_wave", "rf64_wave"]
        or qc["accepted_channels"] != [1, 2]
        or _strict_number(qc["minimum_decoded_to_ffprobe_duration_ratio"], "minimum ratio") != 0.98
        or _strict_number(qc["maximum_decoded_to_ffprobe_duration_ratio"], "maximum ratio") != 1.02
        or qc["full_decode_warning_policy"] != "exclude"
        or qc["canonical_duration_source"] != "full_decode"
    ):
        raise UnknownAudioError("unknown audio QC policy is not locked")

    estimation = config["estimation"]
    _strict_int(estimation["assumed_source_bitrate_kbps"], "estimated bitrate", minimum=1)
    _strict_number(estimation["working_space_multiplier"], "space multiplier", positive=True)
    _strict_int(estimation["minimum_free_space_reserve_bytes"], "free-space reserve")
    if estimation != {
        "assumed_source_bitrate_kbps": 192,
        "working_space_multiplier": 2.0,
        "minimum_free_space_reserve_bytes": 1_073_741_824,
    }:
        raise UnknownAudioError("unknown audio estimation policy is not locked")
    download = config["download"]
    _string_list(download["allowed_initial_hosts"], "allowed initial hosts")
    _string_list(download["allowed_redirect_hosts"], "allowed redirect hosts")
    for key in (
        "maximum_redirects",
        "maximum_retries",
        "chunk_size_bytes",
        "maximum_file_size_bytes",
    ):
        _strict_int(download[key], f"download.{key}", minimum=1)
    _strict_number(download["request_interval_seconds"], "request interval", positive=True)
    _strict_number(download["timeout_seconds"], "download timeout", positive=True)
    _strict_number(download["total_timeout_seconds"], "total download timeout", positive=True)
    _strict_number(download["maximum_retry_after_seconds"], "maximum retry-after", positive=True)
    if not isinstance(download["user_agent"], str) or not download["user_agent"].strip():
        raise UnknownAudioError("download user agent is invalid")
    if download["proxy_policy"] != "disabled" or download["cookie_policy"] != "disabled":
        raise UnknownAudioError("download proxy and cookie policies must be disabled")
    if set(download["allowed_initial_hosts"]) - set(download["allowed_redirect_hosts"]):
        raise UnknownAudioError("initial hosts must also be allowed redirect hosts")
    expected_download = {
        "allowed_initial_hosts": ["xeno-canto.org"],
        "allowed_redirect_hosts": ["xeno-canto.org"],
        "maximum_redirects": 3,
        "request_interval_seconds": 1.0,
        "timeout_seconds": 60.0,
        "total_timeout_seconds": 900.0,
        "maximum_retries": 3,
        "maximum_retry_after_seconds": 60.0,
        "chunk_size_bytes": 1_048_576,
        "maximum_file_size_bytes": 536_870_912,
        "user_agent": "STW7088CEM-bird-audio-coursework/0.1",
        "proxy_policy": "disabled",
        "cookie_policy": "disabled",
    }
    if download != expected_download:
        raise UnknownAudioError("unknown audio download policy is not locked")

    outputs = config["outputs"]
    if outputs != {
        "working_directory": "data/unknown/interim/audio_acquisition_v1",
        "raw_directory": "data/unknown/raw/audio_v1",
        "audit": "data/unknown/audio/unknown_audio_audit_v1.json",
        "audit_lock": "data/unknown/audio/unknown_audio_audit_v1_lock.json",
    }:
        raise UnknownAudioError("unknown audio output paths are not locked")
    resolved_outputs = [
        _locked_output_path(outputs[key], f"unknown audio output {key}")
        for key in _TABLE_KEYS["outputs"]
    ]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise UnknownAudioError("unknown audio output paths must be distinct")


def load_unknown_audio_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    if not is_relative_to(config_path, PROJECT_ROOT):
        raise UnknownAudioError("unknown audio config must remain inside the project")
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UnknownAudioError(
            f"unable to load unknown audio config: {type(exc).__name__}"
        ) from exc
    _validate_config(config)
    return config


def _parse_api_duration(value: Any) -> int | None:
    match = _API_DURATION.fullmatch(str(value or "").strip())
    if not match:
        return None
    seconds = int(match.group(1)) * 60 + int(match.group(2))
    return seconds if seconds > 0 else None


def _parse_advisory_rate(value: Any) -> int | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"[1-9][0-9]*", text):
        return None
    return int(text)


def _forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_FIELDS:
                found.append(path)
            found.extend(_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def _all_study_labels(
    queues: Sequence[Mapping[str, Any]], retained_known_rows: Sequence[Mapping[str, str]]
) -> set[str]:
    labels = {
        _normalize_text(value)
        for queue in queues
        for value in (queue.get("common_name"), queue.get("scientific_name"))
    }
    labels.update(
        _normalize_text(value)
        for row in retained_known_rows
        for value in (row.get("species_common_name"), row.get("scientific_name"))
    )
    labels.discard("")
    return labels


def _session_row(
    candidate_id: str,
    common_name: str,
    scientific_name: str,
    metadata: Mapping[str, Any],
    origin: str,
) -> dict[str, str]:
    longitude = metadata.get("lon")
    if longitude in (None, ""):
        longitude = metadata.get("lng")
    return {
        "metadata_status": "ok",
        "recording_id": candidate_id,
        "xc_id": candidate_id.removeprefix("XC"),
        "species_common_name": common_name,
        "scientific_name": scientific_name,
        "recordist": str(metadata.get("rec") or ""),
        "recorded_date": str(metadata.get("date") or ""),
        "locality": str(metadata.get("loc") or ""),
        "latitude": str(metadata.get("lat") or ""),
        "longitude": str(longitude or ""),
        "remarks": str(metadata.get("rmk") or ""),
        "local_qc_status": "include",
        "exclusion_reasons": "",
        "audit_origin": origin,
    }


def _known_session_row(row: Mapping[str, str]) -> dict[str, str]:
    result = dict(row)
    result["audit_origin"] = "known"
    result["metadata_status"] = "ok"
    result["local_qc_status"] = "include"
    result["exclusion_reasons"] = ""
    return result


def build_unknown_audio_preflight(
    plan: Mapping[str, Any],
    retained_known_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    *,
    plan_sha256: str = "",
    plan_lock_sha256: str = "",
    known_manifest_sha256: str = "",
    config_sha256: str = "",
    available_disk_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, read-only candidate and session disposition table."""
    _validate_config(config)
    forbidden = _forbidden_keys(plan)
    if forbidden:
        raise UnknownAudioError(f"candidate plan contains forbidden outcome fields: {forbidden}")
    queues = plan.get("candidate_queues")
    if not isinstance(queues, list) or len(queues) != 6:
        raise UnknownAudioError("candidate plan must contain six locked queues")
    if len(retained_known_rows) != config["expected_retained_known_recordings"]:
        raise UnknownAudioError("retained known recording count does not match the locked value")
    if any(row.get("local_qc_status") != "include" for row in retained_known_rows):
        raise UnknownAudioError("known session isolation received a non-retained row")

    study_labels = _all_study_labels(queues, retained_known_rows)
    candidates: list[dict[str, Any]] = []
    session_rows = [_known_session_row(row) for row in retained_known_rows]
    seen_ids: set[str] = set()
    for queue_index, queue in enumerate(queues):
        raw_candidates = queue.get("candidates")
        if not isinstance(raw_candidates, list):
            raise UnknownAudioError("candidate queue is invalid")
        scientific_name = str(queue.get("scientific_name") or "")
        common_name = str(queue.get("common_name") or "")
        try:
            genus, species = scientific_name.split()
        except ValueError as exc:
            raise UnknownAudioError("candidate queue scientific name is invalid") from exc
        for expected_rank, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, Mapping):
                raise UnknownAudioError("candidate record is invalid")
            candidate_id = str(raw_candidate.get("candidate_id") or "")
            match = _XC_ID.fullmatch(candidate_id)
            if not match or candidate_id in seen_ids:
                raise UnknownAudioError("candidate IDs must be unique canonical XC IDs")
            seen_ids.add(candidate_id)
            if raw_candidate.get("queue_rank") != expected_rank:
                raise UnknownAudioError(f"candidate queue rank is invalid: {candidate_id}")
            metadata = raw_candidate.get("metadata")
            if not isinstance(metadata, Mapping):
                raise UnknownAudioError(f"candidate metadata is invalid: {candidate_id}")
            reasons: list[str] = []
            if metadata.get("gen") != genus or metadata.get("sp") != species:
                reasons.append("metadata_identity_mismatch")
            groups = [
                str(metadata[field]).strip().casefold()
                for field in ("grp", "group")
                if metadata.get(field) not in (None, "")
            ]
            if not groups or any(group != config["metadata"]["required_group"] for group in groups):
                reasons.append("metadata_group_mismatch")
            expected_url = f"https://xeno-canto.org/{match.group(1)}/download"
            if metadata.get("file") != expected_url:
                reasons.append("download_url_mismatch")
            if not _is_recognized_cc_licence_uri(metadata.get("lic")):
                reasons.append("licence_missing_or_unrecognized")
            quality = str(metadata.get("q") or "")
            if quality not in config["metadata"]["accepted_quality"]:
                reasons.append("quality_not_A_or_B")
            own_labels = {_normalize_text(common_name), _normalize_text(scientific_name)}
            also = metadata.get("also") or []
            also_values = also if isinstance(also, list) else [also]
            target_secondary = sorted(
                str(value)
                for value in also_values
                if _secondary_matches_different_target(str(value), study_labels, own_labels)
            )
            if target_secondary:
                reasons.append("configured_study_species_in_secondary_labels")
            rate = _parse_advisory_rate(metadata.get("smp"))
            duration = _parse_api_duration(metadata.get("length"))
            candidate = {
                "candidate_id": candidate_id,
                "queue_index": queue_index,
                "queue_rank": expected_rank,
                "order_sha256": str(raw_candidate.get("order_sha256") or ""),
                "role": str(queue.get("role") or ""),
                "active": bool(queue.get("active")),
                "common_name": common_name,
                "scientific_name": scientific_name,
                "difficulty_group": str(queue.get("difficulty_group") or ""),
                "download_url": expected_url,
                "quality": quality,
                "declared_sample_rate_hz": rate,
                "estimated_duration_seconds": duration,
                "target_secondary_labels": target_secondary,
                "metadata_reasons": sorted(set(reasons)),
                "session_group": "",
                "session_review_reasons": [],
                "disposition": "metadata_excluded" if reasons else "pending_session",
                "reasons": sorted(set(reasons)),
                "canonical_candidate_id": "",
            }
            candidates.append(candidate)
            session_rows.append(
                _session_row(candidate_id, common_name, scientific_name, metadata, "unknown")
            )

    assign_session_groups(
        session_rows,
        coordinate_radius_km=float(config["session"]["coordinate_radius_km"]),
    )
    session_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in session_rows:
        session_members[row["session_group"]].append(row)
    session_by_id = {
        row["recording_id"]: row for row in session_rows if row["audit_origin"] == "unknown"
    }
    for candidate in candidates:
        row = session_by_id[candidate["candidate_id"]]
        group = row["session_group"]
        candidate["session_group"] = group
        review_reasons = sorted(filter(None, row.get("session_review_reason", "").split(";")))
        candidate["session_review_reasons"] = review_reasons
        members = session_members[group]
        if review_reasons:
            candidate["reasons"].append("unresolved_session_review")
        if any(member["audit_origin"] == "known" for member in members):
            candidate["reasons"].append("shared_session_with_retained_known")
        unknown_species = {
            member["scientific_name"] for member in members if member["audit_origin"] == "unknown"
        }
        if len(unknown_species) > 1:
            candidate["reasons"].append("cross_unknown_species_session")
        candidate["reasons"] = sorted(set(candidate["reasons"]))
        if candidate["reasons"]:
            candidate["disposition"] = "metadata_excluded"

    eligible_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["disposition"] == "pending_session":
            eligible_by_session[candidate["session_group"]].append(candidate)
    for members in eligible_by_session.values():
        ordered = sorted(members, key=lambda row: (row["queue_rank"], row["candidate_id"]))
        canonical = ordered[0]
        canonical["disposition"] = "canonical_pending_audio_qc"
        canonical["canonical_candidate_id"] = canonical["candidate_id"]
        for candidate in ordered[1:]:
            candidate["disposition"] = "session_noncanonical"
            candidate["reasons"] = ["later_metadata_eligible_member_of_same_session"]
            candidate["canonical_candidate_id"] = canonical["candidate_id"]

    candidates.sort(key=lambda row: (row["queue_index"], row["queue_rank"]))
    species_records: list[dict[str, Any]] = []
    assumed_bits_per_second = config["estimation"]["assumed_source_bitrate_kbps"] * 1000
    for queue in queues:
        scientific_name = queue["scientific_name"]
        rows = [row for row in candidates if row["scientific_name"] == scientific_name]
        canonical = [row for row in rows if row["disposition"] == "canonical_pending_audio_qc"]
        estimated_seconds = sum(row["estimated_duration_seconds"] or 0 for row in canonical)
        estimated_bytes = math.ceil(estimated_seconds * assumed_bits_per_second / 8)
        dispositions = Counter(row["disposition"] for row in rows)
        species_records.append(
            {
                "role": queue["role"],
                "active": queue["active"],
                "common_name": queue["common_name"],
                "scientific_name": scientific_name,
                "inventory_recordings": len(rows),
                "candidate_pool_target_recordings": CANDIDATE_POOL_TARGET,
                "target_recordings": TARGET_RECORDINGS,
                "canonical_sessions_before_audio_qc": len(canonical),
                "raw_inventory_shortfall_for_final_target": max(0, TARGET_RECORDINGS - len(rows)),
                "canonical_session_shortfall_for_final_target": max(
                    0, TARGET_RECORDINGS - len(canonical)
                ),
                "canonical_session_margin_for_final_target": len(canonical) - TARGET_RECORDINGS,
                "dispositions": dict(sorted(dispositions.items())),
                "estimated_download_duration_seconds": estimated_seconds,
                "estimated_download_bytes": estimated_bytes,
                "fallback_status": (
                    "inactive_until_protocol_gate"
                    if queue["role"] == "fallback"
                    else "not_applicable"
                ),
            }
        )
    active_estimated_bytes = sum(
        row["estimated_download_bytes"] for row in species_records if row["active"]
    )
    fallback_contingency_bytes = sum(
        row["estimated_download_bytes"] for row in species_records if row["role"] == "fallback"
    )
    estimated_bytes_with_contingency = active_estimated_bytes + fallback_contingency_bytes
    required_bytes = (
        math.ceil(
            estimated_bytes_with_contingency * config["estimation"]["working_space_multiplier"]
        )
        + config["estimation"]["minimum_free_space_reserve_bytes"]
    )
    deterministic = {
        "schema_version": UNKNOWN_AUDIO_CONFIG_SCHEMA_VERSION,
        "selection_seed": SELECTION_SEED,
        "plan_sha256": plan_sha256,
        "plan_lock_sha256": plan_lock_sha256,
        "known_manifest_sha256": known_manifest_sha256,
        "config_sha256": config_sha256,
        "download_policy_sha256": sha256_json(config["download"]),
        "candidate_pool_target_recordings_per_species": CANDIDATE_POOL_TARGET,
        "target_recordings_per_species": TARGET_RECORDINGS,
        "network_requests": 0,
        "audio_downloads": 0,
        "fallback_active": False,
        "species": species_records,
        "candidates": candidates,
        "estimated_active_download_bytes": active_estimated_bytes,
        "estimated_fallback_contingency_bytes": fallback_contingency_bytes,
        "estimated_download_bytes_with_fallback_contingency": estimated_bytes_with_contingency,
        "estimated_required_disk_bytes": required_bytes,
    }
    result = copy.deepcopy(deterministic)
    result["preflight_sha256"] = sha256_json(deterministic)
    result["disk"] = {
        "available_bytes": available_disk_bytes,
        "estimated_required_bytes": required_bytes,
        "estimated_space_sufficient": (
            None if available_disk_bytes is None else available_disk_bytes >= required_bytes
        ),
    }
    return result


def preflight_unknown_audio(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    available_disk_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate all bindings and perform a read-only, no-network preflight."""
    try:
        config_file = resolve_project_path(config_path)
        if not is_relative_to(config_file, PROJECT_ROOT):
            raise UnknownAudioError("unknown audio config must remain inside the project")
        config_bytes = config_file.read_bytes()
        config = tomllib.loads(config_bytes.decode("utf-8"))
        _validate_config(config)
        inputs = config["inputs"]
        plan_path = _project_input(inputs["candidate_plan"], "candidate plan")
        lock_path = _project_input(inputs["candidate_plan_lock"], "candidate plan lock")
        known_path = _project_input(inputs["known_manifest"], "known manifest")
        plan_lock_sha256 = sha256_file(lock_path)
        validated = validate_unknown_candidate_plan(lock_path, plan_path)
        plan_bytes = plan_path.read_bytes()
        plan_sha256 = sha256_bytes(plan_bytes)
        if validated.get("plan_sha256") != plan_sha256:
            raise UnknownAudioError("validated candidate plan hash changed")
        plan = json.loads(plan_bytes.decode("utf-8"))
        binding = plan.get("source_bindings", {}).get("known_manifest", {})
        known_rows, known_sha256 = read_csv_snapshot(known_path)
        if binding != {
            "path": inputs["known_manifest"],
            "sha256": known_sha256,
        }:
            raise UnknownAudioError("known manifest does not match the candidate plan binding")
        retained = [row for row in known_rows if row.get("local_qc_status") == "include"]
        if available_disk_bytes is None:
            available_disk_bytes = shutil.disk_usage(PROJECT_ROOT).free
        result = build_unknown_audio_preflight(
            plan,
            retained,
            config,
            plan_sha256=plan_sha256,
            plan_lock_sha256=plan_lock_sha256,
            known_manifest_sha256=known_sha256,
            config_sha256=sha256_bytes(config_bytes),
            available_disk_bytes=available_disk_bytes,
        )
        require_unchanged(config_file, sha256_bytes(config_bytes))
        require_unchanged(lock_path, plan_lock_sha256)
        require_unchanged(plan_path, plan_sha256)
        require_unchanged(known_path, known_sha256)
        return result
    except UnknownAudioError:
        raise
    except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
        raise UnknownAudioError(
            f"unknown audio preflight input validation failed: {type(exc).__name__}"
        ) from exc


def evaluate_fallback_gate(species_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the locked one-primary fallback gate to terminal species results."""
    primaries = [dict(row) for row in species_results if row.get("role") == "primary"]
    fallbacks = [dict(row) for row in species_results if row.get("role") == "fallback"]
    if len(primaries) != 5 or len(fallbacks) > 1:
        raise UnknownAudioError("fallback gate requires five primaries and at most one fallback")
    for row in primaries + fallbacks:
        if not isinstance(row.get("eligible_recordings"), int) or isinstance(
            row.get("eligible_recordings"), bool
        ):
            raise UnknownAudioError("species eligible count is invalid")
        if row["eligible_recordings"] < 0:
            raise UnknownAudioError("species eligible count cannot be negative")
        if (
            isinstance(row.get("unresolved_retryable"), bool)
            or not isinstance(row.get("unresolved_retryable"), int)
            or row["unresolved_retryable"] < 0
        ):
            raise UnknownAudioError("species retryable count is invalid")
    unresolved_primaries = [
        row["scientific_name"]
        for row in primaries
        if row["unresolved_retryable"]
        or row.get("completion_state") not in {"pool_satisfied", "inventory_exhausted"}
    ]
    jungle = next(
        (row for row in primaries if row.get("scientific_name") == "Acridotheres fuscus"),
        None,
    )
    if jungle is None:
        raise UnknownAudioError("Jungle Myna primary result is missing")
    if jungle.get("completion_state") != "inventory_exhausted":
        unresolved_primaries.append("Acridotheres fuscus")
    unresolved_primaries = sorted(set(unresolved_primaries))
    if unresolved_primaries:
        return {
            "status": "blocked_retryable_or_incomplete_primary_audit",
            "fallback_active": False,
            "failed_primary_species": [],
            "blocked_species": unresolved_primaries,
            "replacement": None,
        }
    failed = sorted(
        row["scientific_name"]
        for row in primaries
        if row["eligible_recordings"] < TARGET_RECORDINGS
    )
    if len(failed) > 1:
        return {
            "status": "protocol_decision_required",
            "reason": "more_than_one_primary_below_40",
            "fallback_active": False,
            "failed_primary_species": failed,
            "blocked_species": [],
            "replacement": None,
        }
    if not failed:
        return {
            "status": "ready_without_fallback",
            "fallback_active": False,
            "failed_primary_species": [],
            "blocked_species": [],
            "replacement": None,
        }
    if not fallbacks:
        return {
            "status": "fallback_audit_required",
            "fallback_active": True,
            "failed_primary_species": failed,
            "blocked_species": [],
            "replacement": None,
        }
    fallback = fallbacks[0]
    if fallback["unresolved_retryable"]:
        return {
            "status": "blocked_retryable_fallback_audit",
            "fallback_active": True,
            "failed_primary_species": failed,
            "blocked_species": [fallback["scientific_name"]],
            "replacement": None,
        }
    if fallback.get("completion_state") not in {"pool_satisfied", "inventory_exhausted"}:
        return {
            "status": "fallback_audit_required",
            "fallback_active": True,
            "failed_primary_species": failed,
            "blocked_species": [fallback["scientific_name"]],
            "replacement": None,
        }
    if fallback["eligible_recordings"] < TARGET_RECORDINGS:
        return {
            "status": "protocol_decision_required",
            "reason": "fallback_below_40",
            "fallback_active": True,
            "failed_primary_species": failed,
            "blocked_species": [],
            "replacement": None,
        }
    return {
        "status": "ready_with_fallback",
        "fallback_active": True,
        "failed_primary_species": failed,
        "blocked_species": [],
        "replacement": {
            "replaced_scientific_name": failed[0],
            "replacement_scientific_name": fallback["scientific_name"],
        },
    }


def _object_record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise UnknownAudioError("injected audio operation returned an invalid result")


def _duration_bucket(duration: float) -> str:
    if duration < 3:
        return "below_3"
    if duration < 10:
        return "3_to_below_10"
    if duration < 30:
        return "10_to_below_30"
    if duration < 60:
        return "30_to_below_60"
    return "at_least_60"


def _source_rate_bucket(rate: int) -> str:
    if rate in {32000, 44100, 48000}:
        return str(rate)
    if rate > 48000:
        return "above_48000"
    return "other_eligible"


def _container_bucket(header_type: str) -> str:
    if header_type in {"mp3_id3", "mpeg_audio"}:
        return "mp3"
    if header_type in {"riff_wave", "rf64_wave"}:
        return "riff_wave"
    raise UnknownAudioError("eligible audio has an unsupported container")


def _expected_assignment_descriptor(
    candidate: Mapping[str, Any], qc: Mapping[str, Any]
) -> dict[str, str]:
    try:
        rate = qc["source_sample_rate_hz"]
        channels = qc["channels"]
        decoded_duration = qc["decoded_duration_seconds"]
        header_type = qc["header_type"]
    except KeyError as exc:
        raise UnknownAudioError("QC record lacks an assignment input") from exc
    if isinstance(rate, bool) or not isinstance(rate, int):
        raise UnknownAudioError("QC source sample rate is invalid")
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise UnknownAudioError("QC channel count is invalid")
    if isinstance(decoded_duration, bool) or not isinstance(decoded_duration, (int, float)):
        raise UnknownAudioError("QC decoded duration is invalid")
    duration = float(decoded_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise UnknownAudioError("eligible QC decoded duration is invalid")
    if not isinstance(header_type, str):
        raise UnknownAudioError("QC detected header is invalid")
    quality = candidate.get("quality")
    if quality not in {"A", "B"}:
        raise UnknownAudioError("eligible candidate quality is invalid")
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "session_group": str(candidate["session_group"]),
        "container": _container_bucket(header_type),
        "source_rate_bucket": _source_rate_bucket(rate),
        "channels": "mono" if channels == 1 else "stereo",
        "quality": quality,
        "duration_bucket": _duration_bucket(duration),
        "duration_seconds": f"{duration:.6f}",
    }


def _expected_raw_relative_path(candidate: Mapping[str, Any]) -> str:
    scientific_name = candidate.get("scientific_name")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(scientific_name, str) or not scientific_name.strip():
        raise UnknownAudioError("candidate scientific name is invalid")
    if not isinstance(candidate_id, str) or not _XC_ID.fullmatch(candidate_id):
        raise UnknownAudioError("candidate ID is invalid")
    species_component = scientific_name.replace(" ", "_")
    if (
        species_component in {".", ".."}
        or Path(species_component).name != species_component
        or "\x00" in species_component
    ):
        raise UnknownAudioError("candidate scientific name is not path safe")
    return f"data/unknown/raw/audio_v1/{species_component}/{candidate_id}.audio"


def _validate_private_raw_file(relative: str, expected_sha256: str, expected_size: Any) -> Path:
    if not _SHA256.fullmatch(expected_sha256):
        raise UnknownAudioError("raw audio SHA-256 is invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise UnknownAudioError("raw audio file size is invalid")
    lexical = PROJECT_ROOT / relative
    resolved = resolve_project_path(relative)
    if resolved != lexical or not is_relative_to(resolved, PROJECT_ROOT):
        raise UnknownAudioError("raw audio path traverses a symbolic link")
    _validate_bound_audio_file(
        lexical,
        expected_sha256,
        expected_size,
        allowed_link_counts=frozenset({1}),
    )
    return lexical


def _private_audio_identity(
    path: Path, *, allowed_link_counts: frozenset[int] = frozenset({1})
) -> tuple[int, int, int, int, int, int]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(descriptor)
    except OSError as exc:
        raise UnknownAudioError("downloaded audio is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink not in allowed_link_counts
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise UnknownAudioError("downloaded audio private-file state is invalid")
    identity = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_mode,
        observed.st_nlink,
    )
    try:
        final = path.lstat()
    except OSError as exc:
        raise UnknownAudioError("downloaded audio disappeared during verification") from exc
    final_identity = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_mode,
        final.st_nlink,
    )
    if identity != final_identity:
        raise UnknownAudioError("downloaded audio path changed during verification")
    return identity


def _derive_objective_qc_reasons(qc: Mapping[str, Any]) -> list[str]:
    header_status = qc.get("header_detection_status")
    probe_status = qc.get("probe_status")
    decode_status = qc.get("full_decode_status")
    if header_status not in _HEADER_DETECTION_STATUSES:
        raise UnknownAudioError("QC header detection status is invalid")
    if probe_status not in _PROBE_STATUSES:
        raise UnknownAudioError("QC probe status is invalid")
    if decode_status not in _FULL_DECODE_STATUSES:
        raise UnknownAudioError("QC full decode status is invalid")

    text_fields = ("header_type", "format_name", "codec_name", "full_decode_diagnostic")
    if any(not isinstance(qc.get(key), str) for key in text_fields):
        raise UnknownAudioError("QC stage text is invalid")
    header_type = qc["header_type"]
    format_name = qc["format_name"]
    codec_name = qc["codec_name"]
    diagnostic = qc["full_decode_diagnostic"]
    expected_header_status = (
        "recognized"
        if header_type in {"mp3_id3", "mpeg_audio", "riff_wave", "rf64_wave"}
        else "unsupported"
    )
    if header_status != expected_header_status:
        raise UnknownAudioError("QC header detection status is inconsistent")

    rate = qc.get("source_sample_rate_hz")
    channels = qc.get("channels")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int)
        or rate < 0
        or isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels < 0
    ):
        raise UnknownAudioError("QC probe scalar is invalid")
    numeric: dict[str, float] = {}
    for key in (
        "ffprobe_duration_seconds",
        "decoded_duration_seconds",
        "decoded_duration_ratio",
    ):
        value = qc.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnknownAudioError("QC duration scalar is invalid")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise UnknownAudioError("QC duration scalar is invalid")
        numeric[key] = number
    ffprobe_duration = numeric["ffprobe_duration_seconds"]
    decoded_duration = numeric["decoded_duration_seconds"]
    ratio = numeric["decoded_duration_ratio"]
    expected_ratio = decoded_duration / ffprobe_duration if ffprobe_duration > 0 else 0.0
    if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise UnknownAudioError("QC decoded duration ratio is inconsistent")

    reasons: set[str] = set()
    if header_status == "unsupported":
        reasons.add("unsupported_detected_header")
    if probe_status == "content_failure":
        if format_name or codec_name or rate != 0 or channels != 0 or ffprobe_duration != 0:
            raise UnknownAudioError("failed QC probe fields are not normalized")
        reasons.add("ffprobe_content_failure")
    else:
        if rate < 32000:
            reasons.add("source_sample_rate_below_32000_hz")
        if channels not in {1, 2}:
            reasons.add("unsupported_channel_count")
        if ffprobe_duration <= 0:
            reasons.add("non_positive_ffprobe_duration")

    if reasons:
        if decode_status != "not_run" or decoded_duration != 0 or ratio != 0 or diagnostic:
            raise UnknownAudioError("not-run QC decode fields are not normalized")
        return sorted(reasons)

    if decode_status == "not_run":
        raise UnknownAudioError("eligible predecode QC cannot skip full decode")
    if decode_status == "content_failure":
        if decoded_duration != 0 or ratio != 0 or diagnostic:
            raise UnknownAudioError("failed QC decode fields are not normalized")
        reasons.add("full_decode_content_failure")
    elif decode_status == "invalid_duration":
        if decoded_duration != 0 or ratio != 0 or diagnostic:
            raise UnknownAudioError("invalid-duration QC fields are not normalized")
        reasons.add("non_positive_decoded_duration")
    elif decode_status in {"ok", "warning"}:
        if decoded_duration <= 0:
            raise UnknownAudioError("completed QC decode duration is invalid")
        if (decode_status == "warning") != bool(diagnostic):
            raise UnknownAudioError("QC decode warning status is inconsistent")
        if diagnostic:
            reasons.add("full_decode_warning")
        if not 0.98 <= ratio <= 1.02:
            reasons.add("decoded_to_ffprobe_duration_ratio_outside_bounds")
    return sorted(reasons)


def audit_unknown_audio_file(
    path: str | Path,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    ffprobe: str | Path,
    ffmpeg: str | Path,
    detect_header_fn: Callable[[Path], str] = detect_header,
    probe_fn: Callable[[Path, Path], AudioProbe | Mapping[str, Any]] = probe_audio,
    full_decode_fn: Callable[
        [Path, Path], FullDecodeResult | Mapping[str, Any]
    ] = verify_full_decode,
) -> dict[str, Any]:
    """Apply objective local-file QC and return one terminal QC record."""
    _validate_config(config)
    source = resolve_project_path(path)
    if not source.is_file() or not is_relative_to(source, PROJECT_ROOT):
        raise UnknownAudioError("unknown audio QC source must be a regular project file")
    before = source.stat()
    digest = sha256_file(source)
    try:
        header_type = detect_header_fn(source)
    except Exception as exc:
        raise UnknownAudioRetryableError("header detection is temporarily unresolved") from exc
    if not isinstance(header_type, str):
        raise UnknownAudioRetryableError("header detection returned an invalid result")
    header_status = (
        "recognized"
        if header_type in config["audio_qc"]["accepted_header_types"]
        else "unsupported"
    )
    try:
        probe = _object_record(probe_fn(source, Path(ffprobe)))
    except Exception as exc:
        raise UnknownAudioRetryableError("ffprobe execution is temporarily unresolved") from exc
    probe_ok = probe.get("probe_ok")
    if not isinstance(probe_ok, bool):
        raise UnknownAudioRetryableError("ffprobe returned an invalid status")
    probe_error = probe.get("probe_error")
    if not isinstance(probe_error, str):
        raise UnknownAudioRetryableError("ffprobe returned an invalid diagnostic status")
    if probe_error in {"ffprobe invocation timed out", "ffprobe invocation failed"}:
        raise UnknownAudioRetryableError("ffprobe invocation is temporarily unresolved")
    if probe_ok:
        probe_status = "ok"
        format_name = probe.get("format_name")
        codec_name = probe.get("codec_name")
        rate = probe.get("source_sample_rate_hz")
        channels = probe.get("channels")
        duration = probe.get("ffprobe_duration_seconds")
        if (
            not isinstance(format_name, str)
            or not isinstance(codec_name, str)
            or isinstance(rate, bool)
            or not isinstance(rate, int)
            or rate < 0
            or isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels < 0
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise UnknownAudioRetryableError("ffprobe returned invalid fields")
        probed_duration = float(duration)
    else:
        probe_status = "content_failure"
        format_name = ""
        codec_name = ""
        rate = 0
        channels = 0
        probed_duration = 0.0

    predecode_blocked = (
        header_status == "unsupported"
        or probe_status == "content_failure"
        or rate < config["audio_qc"]["minimum_source_sample_rate_hz"]
        or channels not in config["audio_qc"]["accepted_channels"]
        or probed_duration <= 0
    )
    decoded_duration = 0.0
    decode_diagnostic = ""
    decode_status = "not_run"
    if not predecode_blocked:
        try:
            decoded = _object_record(full_decode_fn(source, Path(ffmpeg)))
        except AudioToolError as exc:
            if str(exc) in {"Full decode timed out", "Full decode invocation failed"}:
                raise UnknownAudioRetryableError(
                    "full decode invocation is temporarily unresolved"
                ) from exc
            decode_status = "content_failure"
        except Exception as exc:
            raise UnknownAudioRetryableError("full decode is temporarily unresolved") from exc
        else:
            raw_duration = decoded.get("decoded_duration_seconds")
            raw_diagnostic = decoded.get("diagnostic")
            if (
                isinstance(raw_duration, bool)
                or not isinstance(raw_duration, (int, float))
                or not isinstance(raw_diagnostic, str)
            ):
                raise UnknownAudioRetryableError("full decode returned invalid fields")
            decoded_duration = float(raw_duration)
            decode_diagnostic = raw_diagnostic[:2000]
            if not math.isfinite(decoded_duration) or decoded_duration <= 0:
                decoded_duration = 0.0
                decode_diagnostic = ""
                decode_status = "invalid_duration"
            else:
                decode_status = "warning" if decode_diagnostic else "ok"
    ratio = decoded_duration / probed_duration if probed_duration > 0 else 0.0
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(source) != digest:
        raise UnknownAudioError("downloaded audio changed during QC")
    result: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "scientific_name": candidate["scientific_name"],
        "session_group": candidate["session_group"],
        "relative_path": source.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": digest,
        "file_size_bytes": before.st_size,
        "header_detection_status": header_status,
        "header_type": header_type,
        "probe_status": probe_status,
        "format_name": format_name,
        "codec_name": codec_name,
        "source_sample_rate_hz": rate,
        "channels": channels,
        "ffprobe_duration_seconds": probed_duration,
        "full_decode_status": decode_status,
        "decoded_duration_seconds": decoded_duration,
        "decoded_duration_ratio": ratio,
        "full_decode_diagnostic": decode_diagnostic,
    }
    reasons = _derive_objective_qc_reasons(result)
    result["disposition"] = "audio_qc_excluded" if reasons else "eligible"
    result["reasons"] = reasons
    if not reasons:
        result["assignment_descriptor"] = _expected_assignment_descriptor(candidate, result)
    return result


def select_final_unknown_recordings(
    known_test_reference: Mapping[str, Any],
    candidates_by_species: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Run the locked Hungarian assignment for every final unknown species."""
    if len(candidates_by_species) != 5:
        raise UnknownAudioError("final selection requires exactly five unknown species")
    slots = assignment_slots_from_reference(known_test_reference)
    expected_fields = {
        "candidate_id",
        "session_group",
        "container",
        "source_rate_bucket",
        "channels",
        "quality",
        "duration_bucket",
        "duration_seconds",
    }
    selections: dict[str, Any] = {}
    selected_ids: set[str] = set()
    selected_sessions: set[str] = set()
    for scientific_name in sorted(candidates_by_species):
        rows = [dict(row) for row in candidates_by_species[scientific_name]]
        for row in rows:
            forbidden = set(row).intersection(FORBIDDEN_FIELDS)
            if forbidden:
                raise UnknownAudioError(
                    f"assignment candidate contains forbidden fields: {sorted(forbidden)}"
                )
            if set(row) != expected_fields:
                raise UnknownAudioError("assignment candidate descriptor fields are not exact")
        assignment = assign_candidates_to_slots(slots, rows)
        ids = [row["candidate_id"] for row in assignment["assignments"]]
        sessions = [row["candidate_session_group"] for row in assignment["assignments"]]
        if len(ids) != TARGET_RECORDINGS or len(set(ids)) != TARGET_RECORDINGS:
            raise UnknownAudioError("Hungarian assignment did not select 40 unique candidates")
        if len(sessions) != TARGET_RECORDINGS or len(set(sessions)) != TARGET_RECORDINGS:
            raise UnknownAudioError("Hungarian assignment did not select 40 unique sessions")
        if selected_ids.intersection(ids) or selected_sessions.intersection(sessions):
            raise UnknownAudioError("final species selections overlap by candidate or session")
        selected_ids.update(ids)
        selected_sessions.update(sessions)
        selections[scientific_name] = {
            "eligible_candidates": len(rows),
            "selected_candidates": len(ids),
            "selected_candidate_ids": ids,
            "assignment": assignment,
        }
    result = {
        "species": selections,
        "species_count": len(selections),
        "selected_recordings": len(selected_ids),
        "zero_candidate_overlap": True,
        "zero_session_overlap": True,
    }
    result["selection_sha256"] = sha256_json(result)
    return result


def _canonical_json_bytes(value: Any) -> bytes:
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


def _create_json_exclusive(path: str | Path, value: Any) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.absolute()
    parent_probe = candidate.parent / f".{candidate.name}.parent-safety"
    if require_safe_output(parent_probe) != parent_probe:
        raise UnknownAudioError("JSON artifact parent traverses a symbolic link")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.parent.resolve() != candidate.parent:
        raise UnknownAudioError("JSON artifact parent is not the locked lexical directory")
    destination = candidate.parent / candidate.name
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnknownAudioError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise UnknownAudioError(f"JSON artifact is not an object: {path.name}")
    return value


def _private_canonical_json_object(path: Path, artifact: str) -> dict[str, Any]:
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise UnknownAudioError(
                f"{artifact} is not a private single-link regular file: {path.name}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise UnknownAudioError(f"{artifact} is unavailable: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
        before.st_nlink,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_nlink,
    )
    try:
        final = path.lstat()
    except OSError as exc:
        raise UnknownAudioError(f"{artifact} disappeared while reading: {path.name}") from exc
    final_identity = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_mode,
        final.st_nlink,
    )
    if before_identity != after_identity or after_identity != final_identity:
        raise UnknownAudioError(f"{artifact} changed while reading: {path.name}")
    try:
        checkpoint = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnknownAudioError(f"{artifact} JSON is invalid: {path.name}") from exc
    if not isinstance(checkpoint, dict):
        raise UnknownAudioError(f"{artifact} is not an object: {path.name}")
    if payload != _canonical_json_bytes(checkpoint):
        raise UnknownAudioError(f"{artifact} bytes are not canonical: {path.name}")
    return checkpoint


def _private_canonical_checkpoint_object(path: Path) -> dict[str, Any]:
    return _private_canonical_json_object(path, "terminal checkpoint")


def _safe_download_receipt(
    value: Any, candidate: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    raw = _object_record(value)
    forbidden_fragments = ("key", "token", "authorization", "cookie", "password", "secret")
    if any(fragment in str(key).casefold() for key in raw for fragment in forbidden_fragments):
        raise UnknownAudioError("download receipt contains a forbidden sensitive field")
    expected_candidate = candidate["candidate_id"]
    expected_source = candidate["download_url"]
    if raw.get("candidate_id") not in (None, expected_candidate):
        raise UnknownAudioError("download receipt candidate binding is invalid")
    if raw.get("source_url") not in (None, expected_source):
        raise UnknownAudioError("download receipt source binding is invalid")
    if raw.get("destination") not in (None, str(destination)):
        raise UnknownAudioError("download receipt destination binding is invalid")
    actual_sha256 = sha256_file(destination)
    actual_bytes = destination.stat().st_size
    if raw.get("sha256") not in (None, actual_sha256):
        raise UnknownAudioError("download receipt hash binding is invalid")
    if raw.get("bytes_written") not in (None, actual_bytes):
        raise UnknownAudioError("download receipt byte-count binding is invalid")
    if raw.get("file_size_bytes") not in (None, actual_bytes):
        raise UnknownAudioError("download receipt file-size binding is invalid")
    allowed = {
        "candidate_id",
        "bytes_written",
        "sha256",
        "attempts",
        "redirect_count",
        "content_length",
        "content_type",
    }
    result: dict[str, Any] = {}
    for key in sorted(set(raw).intersection(allowed)):
        item = raw[key]
        if item is None or isinstance(item, (str, int, float, bool)):
            result[key] = item
    result.update(
        {
            "candidate_id": expected_candidate,
            "bytes_written": actual_bytes,
            "sha256": actual_sha256,
        }
    )
    result["receipt_sha256"] = sha256_json(result)
    return result


def _validate_sanitized_receipt(
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_sha256: str,
    expected_size: Any,
    context: str,
) -> None:
    allowed = {
        "candidate_id",
        "bytes_written",
        "sha256",
        "receipt_sha256",
        "attempts",
        "redirect_count",
        "content_length",
        "content_type",
    }
    if (
        not {"candidate_id", "bytes_written", "sha256", "receipt_sha256"}.issubset(receipt)
        or set(receipt) - allowed
        or receipt.get("candidate_id") != candidate["candidate_id"]
        or receipt.get("sha256") != expected_sha256
        or receipt.get("bytes_written") != expected_size
    ):
        raise UnknownAudioError(f"{context} receipt binding is invalid")
    receipt_sha256 = receipt.get("receipt_sha256")
    if not isinstance(receipt_sha256, str) or not _SHA256.fullmatch(receipt_sha256):
        raise UnknownAudioError(f"{context} receipt hash is invalid")
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    if sha256_json(payload) != receipt_sha256:
        raise UnknownAudioError(f"{context} receipt hash binding is invalid")
    receipt_sha = receipt.get("sha256")
    receipt_bytes = receipt.get("bytes_written")
    if (
        not isinstance(receipt_sha, str)
        or not _SHA256.fullmatch(receipt_sha)
        or isinstance(receipt_bytes, bool)
        or not isinstance(receipt_bytes, int)
        or receipt_bytes <= 0
    ):
        raise UnknownAudioError(f"{context} receipt content is invalid")
    attempts = receipt.get("attempts")
    if "attempts" in receipt and (
        isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 4
    ):
        raise UnknownAudioError(f"{context} receipt attempts are invalid")
    redirects = receipt.get("redirect_count")
    if "redirect_count" in receipt and (
        isinstance(redirects, bool) or not isinstance(redirects, int) or not 0 <= redirects <= 3
    ):
        raise UnknownAudioError(f"{context} receipt redirect count is invalid")
    content_length = receipt.get("content_length")
    if (
        "content_length" in receipt
        and content_length is not None
        and (
            isinstance(content_length, bool)
            or not isinstance(content_length, int)
            or content_length <= 0
            or content_length != receipt_bytes
        )
    ):
        raise UnknownAudioError(f"{context} receipt content length is invalid")
    content_type = receipt.get("content_type")
    if (
        "content_type" in receipt
        and content_type is not None
        and (
            not isinstance(content_type, str)
            or len(content_type) > 200
            or any(ord(character) < 32 or ord(character) > 126 for character in content_type)
        )
    ):
        raise UnknownAudioError(f"{context} receipt content_type is invalid")


def _validate_injected_client_policy(client: DownloadClient, config: Mapping[str, Any]) -> None:
    policy = getattr(client, "policy", None)
    if policy is None:
        raise UnknownAudioError("injected download client has no bound policy")
    raw = _object_record(policy)
    expected = {
        "allowed_hosts": tuple(config["download"]["allowed_initial_hosts"]),
        "maximum_redirects": config["download"]["maximum_redirects"],
        "request_interval_seconds": config["download"]["request_interval_seconds"],
        "timeout_seconds": config["download"]["timeout_seconds"],
        "total_timeout_seconds": config["download"]["total_timeout_seconds"],
        "maximum_retries": config["download"]["maximum_retries"],
        "chunk_size_bytes": config["download"]["chunk_size_bytes"],
        "maximum_file_bytes": config["download"]["maximum_file_size_bytes"],
        "maximum_retry_after_seconds": config["download"]["maximum_retry_after_seconds"],
        "user_agent": config["download"]["user_agent"],
    }
    if raw != expected:
        raise UnknownAudioError("injected secure download client policy does not match config")


def _checkpoint_path(working_directory: Path, candidate_id: str) -> Path:
    if not _XC_ID.fullmatch(candidate_id):
        raise UnknownAudioError("checkpoint candidate ID is invalid")
    return working_directory / "checkpoints" / f"{candidate_id}.json"


def _pending_qc_path(working_directory: Path, candidate_id: str) -> Path:
    if not _XC_ID.fullmatch(candidate_id):
        raise UnknownAudioError("pending-QC candidate ID is invalid")
    return working_directory / "pending_qc" / f"{candidate_id}.json"


def _staging_audio_path(working_directory: Path, candidate: Mapping[str, Any]) -> Path:
    raw_relative = _expected_raw_relative_path(candidate)
    species_component = Path(raw_relative).parent.name
    return working_directory / "staging" / species_component / f"{candidate['candidate_id']}.audio"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_bound_audio_file(
    path: Path,
    expected_sha256: str,
    expected_size: Any,
    *,
    allowed_link_counts: frozenset[int],
    fsync_file: bool = False,
) -> tuple[int, int, int, int, int, int]:
    if not _SHA256.fullmatch(expected_sha256):
        raise UnknownAudioError("bound audio SHA-256 is invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise UnknownAudioError("bound audio size is invalid")
    if path.parent.resolve() != path.parent or not is_relative_to(path, PROJECT_ROOT):
        raise UnknownAudioError("bound audio path traverses a symbolic link")
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before_stat = os.fstat(descriptor)
        before = (
            before_stat.st_dev,
            before_stat.st_ino,
            before_stat.st_size,
            before_stat.st_mtime_ns,
            before_stat.st_mode,
            before_stat.st_nlink,
        )
        if (
            not stat.S_ISREG(before_stat.st_mode)
            or before_stat.st_nlink not in allowed_link_counts
            or stat.S_IMODE(before_stat.st_mode) != 0o600
            or before_stat.st_size != expected_size
        ):
            raise UnknownAudioError("bound audio private-file state is invalid")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise UnknownAudioError("bound audio content does not match its receipt")
        if fsync_file:
            os.fsync(descriptor)
        after_stat = os.fstat(descriptor)
        after = (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
            after_stat.st_mtime_ns,
            after_stat.st_mode,
            after_stat.st_nlink,
        )
    except OSError as exc:
        raise UnknownAudioError("bound audio is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        final_stat = path.lstat()
    except OSError as exc:
        raise UnknownAudioError("bound audio disappeared during verification") from exc
    final = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        final_stat.st_mode,
        final_stat.st_nlink,
    )
    if before != after or after != final:
        raise UnknownAudioError("bound audio changed during verification")
    if fsync_file:
        _fsync_directory(path.parent)
    return after


def _pending_qc_record(
    working_directory: Path,
    preflight: Mapping[str, Any],
    candidate: Mapping[str, Any],
    receipt: Mapping[str, Any],
    staging_identity: tuple[int, int, int, int, int, int],
) -> dict[str, Any]:
    staging = _staging_audio_path(working_directory, candidate)
    raw_relative = _expected_raw_relative_path(candidate)
    return {
        "schema_version": UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION,
        "state": "downloaded_pending_qc",
        "preflight_sha256": preflight["preflight_sha256"],
        "plan_sha256": preflight["plan_sha256"],
        "candidate_id": candidate["candidate_id"],
        "scientific_name": candidate["scientific_name"],
        "queue_rank": candidate["queue_rank"],
        "session_group": candidate["session_group"],
        "staging_device": staging_identity[0],
        "staging_inode": staging_identity[1],
        "staging_relative_path": staging.relative_to(PROJECT_ROOT).as_posix(),
        "raw_relative_path": raw_relative,
        "download_receipt": dict(receipt),
    }


def _validate_pending_qc_record(
    path: Path,
    working_directory: Path,
    preflight: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    record = _private_canonical_json_object(path, "downloaded-pending-QC record")
    expected_keys = {
        "schema_version",
        "state",
        "preflight_sha256",
        "plan_sha256",
        "candidate_id",
        "scientific_name",
        "queue_rank",
        "session_group",
        "staging_device",
        "staging_inode",
        "staging_relative_path",
        "raw_relative_path",
        "download_receipt",
    }
    expected_staging = (
        _staging_audio_path(working_directory, candidate).relative_to(PROJECT_ROOT).as_posix()
    )
    expected_raw = _expected_raw_relative_path(candidate)
    if (
        set(record) != expected_keys
        or _forbidden_keys(record)
        or record.get("schema_version") != UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION
        or record.get("state") != "downloaded_pending_qc"
        or record.get("preflight_sha256") != preflight.get("preflight_sha256")
        or record.get("plan_sha256") != preflight.get("plan_sha256")
        or record.get("candidate_id") != candidate.get("candidate_id")
        or record.get("scientific_name") != candidate.get("scientific_name")
        or record.get("queue_rank") != candidate.get("queue_rank")
        or record.get("session_group") != candidate.get("session_group")
        or isinstance(record.get("staging_device"), bool)
        or not isinstance(record.get("staging_device"), int)
        or record["staging_device"] < 0
        or isinstance(record.get("staging_inode"), bool)
        or not isinstance(record.get("staging_inode"), int)
        or record["staging_inode"] <= 0
        or record.get("staging_relative_path") != expected_staging
        or record.get("raw_relative_path") != expected_raw
    ):
        raise UnknownAudioError(f"downloaded-pending-QC binding is invalid: {path.name}")
    receipt = record.get("download_receipt")
    if not isinstance(receipt, Mapping):
        raise UnknownAudioError(f"downloaded-pending-QC receipt is invalid: {path.name}")
    _validate_sanitized_receipt(
        receipt,
        candidate,
        str(receipt.get("sha256") or ""),
        receipt.get("bytes_written"),
        "downloaded-pending-QC",
    )
    return record


def _prune_empty_staging_directories(working_directory: Path, staging_path: Path) -> None:
    for directory in (staging_path.parent, working_directory / "staging"):
        try:
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            break
        _fsync_directory(directory.parent)


def _promote_pending_qc_audio(
    working_directory: Path,
    raw_directory: Path,
    preflight: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    pending_path = _pending_qc_path(working_directory, candidate["candidate_id"])
    record = _validate_pending_qc_record(pending_path, working_directory, preflight, candidate)
    receipt = dict(record["download_receipt"])
    expected_sha256 = receipt["sha256"]
    expected_size = receipt["bytes_written"]
    expected_owner = (record["staging_device"], record["staging_inode"])
    staging_path = PROJECT_ROOT / record["staging_relative_path"]
    raw_path = PROJECT_ROOT / record["raw_relative_path"]
    expected_raw_root = PROJECT_ROOT / "data" / "unknown" / "raw" / "audio_v1"
    if raw_directory != expected_raw_root or not is_relative_to(raw_path, raw_directory):
        raise UnknownAudioError("pending-QC raw destination is outside the locked root")
    staging_exists = os.path.lexists(staging_path)
    raw_exists = os.path.lexists(raw_path)
    if not staging_exists and not raw_exists:
        raise UnknownAudioError("pending-QC record has no bound audio file")
    if staging_exists and raw_exists:
        staging_identity = _validate_bound_audio_file(
            staging_path,
            expected_sha256,
            expected_size,
            allowed_link_counts=frozenset({2}),
        )
        raw_identity = _validate_bound_audio_file(
            raw_path,
            expected_sha256,
            expected_size,
            allowed_link_counts=frozenset({2}),
        )
        if staging_identity[:2] != raw_identity[:2]:
            raise UnknownAudioError("pending-QC promotion paths are different inodes")
        if staging_identity[:2] != expected_owner:
            raise UnknownAudioError("pending-QC promotion inode is not the sealed download")
    elif staging_exists:
        staging_identity = _validate_bound_audio_file(
            staging_path,
            expected_sha256,
            expected_size,
            allowed_link_counts=frozenset({1}),
        )
        if staging_identity[:2] != expected_owner:
            raise UnknownAudioError("pending-QC staging inode is not the sealed download")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.parent.resolve() != raw_path.parent or os.path.lexists(raw_path):
            raise UnknownAudioError("pending-QC raw destination is not create-only")
        try:
            os.link(staging_path, raw_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise UnknownAudioError("pending-QC raw destination already exists") from exc
        except OSError as exc:
            raise UnknownAudioError("pending-QC raw hard-link promotion failed") from exc
        _fsync_directory(raw_path.parent)
        staging_identity = _validate_bound_audio_file(
            staging_path,
            expected_sha256,
            expected_size,
            allowed_link_counts=frozenset({2}),
        )
        raw_identity = _validate_bound_audio_file(
            raw_path,
            expected_sha256,
            expected_size,
            allowed_link_counts=frozenset({2}),
        )
        if staging_identity[:2] != raw_identity[:2]:
            raise UnknownAudioError("pending-QC promotion did not preserve the inode")
        if raw_identity[:2] != expected_owner:
            raise UnknownAudioError("pending-QC raw inode is not the sealed download")
    else:
        raw_identity = _validate_bound_audio_file(
            raw_path,
            expected_sha256,
            expected_size,
            allowed_link_counts=frozenset({1}),
        )
        if raw_identity[:2] != expected_owner:
            raise UnknownAudioError("pending-QC raw inode is not the sealed download")
        _prune_empty_staging_directories(working_directory, staging_path)
        return raw_path, receipt
    _fsync_directory(raw_path.parent)
    staging_identity = _validate_bound_audio_file(
        staging_path,
        expected_sha256,
        expected_size,
        allowed_link_counts=frozenset({2}),
    )
    if staging_identity[:2] != expected_owner:
        raise UnknownAudioError("pending-QC staging inode changed before unlink")
    try:
        staging_path.unlink()
        _fsync_directory(staging_path.parent)
    except OSError as exc:
        raise UnknownAudioError("pending-QC staging unlink failed") from exc
    _prune_empty_staging_directories(working_directory, staging_path)
    final_raw_identity = _validate_bound_audio_file(
        raw_path,
        expected_sha256,
        expected_size,
        allowed_link_counts=frozenset({1}),
    )
    if final_raw_identity[:2] != expected_owner:
        raise UnknownAudioError("pending-QC raw inode changed after promotion")
    return raw_path, receipt


def _remove_matching_pending_qc_record(
    working_directory: Path,
    preflight: Mapping[str, Any],
    candidate: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    pending_path = _pending_qc_path(working_directory, candidate["candidate_id"])
    staging_path = _staging_audio_path(working_directory, candidate)
    if os.path.lexists(staging_path):
        raise UnknownAudioError("terminal checkpoint retains a staging audio artifact")
    if not os.path.lexists(pending_path):
        return
    record = _validate_pending_qc_record(pending_path, working_directory, preflight, candidate)
    pending_identity = _private_audio_identity(pending_path)
    if checkpoint.get("disposition") not in {"eligible", "audio_qc_excluded"}:
        raise UnknownAudioError("non-audio checkpoint has a pending-QC record")
    qc = checkpoint.get("audio_qc")
    if not isinstance(qc, Mapping) or (
        record["raw_relative_path"] != qc.get("relative_path")
        or record["download_receipt"] != checkpoint.get("download_receipt")
    ):
        raise UnknownAudioError("terminal checkpoint does not match pending-QC state")
    receipt = record["download_receipt"]
    raw_identity = _validate_bound_audio_file(
        PROJECT_ROOT / record["raw_relative_path"],
        receipt["sha256"],
        receipt["bytes_written"],
        allowed_link_counts=frozenset({1}),
    )
    if raw_identity[:2] != (record["staging_device"], record["staging_inode"]):
        raise UnknownAudioError("terminal raw audio is not the sealed pending-QC inode")
    if (
        _validate_pending_qc_record(pending_path, working_directory, preflight, candidate) != record
        or _private_audio_identity(pending_path) != pending_identity
    ):
        raise UnknownAudioError("pending-QC record changed before cleanup")
    try:
        pending_path.unlink()
        _fsync_directory(pending_path.parent)
    except OSError as exc:
        raise UnknownAudioError("pending-QC cleanup failed") from exc
    try:
        pending_path.parent.rmdir()
    except OSError:
        return
    _fsync_directory(pending_path.parent.parent)


def _require_no_pending_or_staging_artifacts(working_directory: Path) -> None:
    for name in ("pending_qc", "staging"):
        path = working_directory / name
        if os.path.lexists(path):
            raise UnknownAudioError(f"completed acquisition retains {name} artifacts")


def _prepare_acquisition_roots(working_directory: Path, raw_directory: Path) -> None:
    for root, context in (
        (working_directory, "working directory"),
        (raw_directory, "raw directory"),
    ):
        if os.path.lexists(root) and (root.is_symlink() or not root.is_dir()):
            raise UnknownAudioError(f"{context} is not a locked directory")
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != root:
            raise UnknownAudioError(f"{context} traverses a symbolic link")
        observed = root.lstat()
        if not stat.S_ISDIR(observed.st_mode):
            raise UnknownAudioError(f"{context} is not a directory")
        _fsync_directory(root)
        _fsync_directory(root.parent)
    if working_directory.stat().st_dev != raw_directory.stat().st_dev:
        raise UnknownAudioError("working and raw roots must share one filesystem")


def _terminal_checkpoint(
    preflight: Mapping[str, Any],
    candidate: Mapping[str, Any],
    disposition: str,
    reasons: Sequence[str],
    *,
    receipt: Mapping[str, Any] | None = None,
    qc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if disposition not in TERMINAL_DISPOSITIONS:
        raise UnknownAudioError("refusing to checkpoint a nonterminal disposition")
    return {
        "schema_version": UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION,
        "preflight_sha256": preflight["preflight_sha256"],
        "plan_sha256": preflight["plan_sha256"],
        "candidate_id": candidate["candidate_id"],
        "scientific_name": candidate["scientific_name"],
        "queue_rank": candidate["queue_rank"],
        "session_group": candidate["session_group"],
        "disposition": disposition,
        "reasons": sorted(set(reasons)),
        "download_receipt": dict(receipt or {}),
        "audio_qc": dict(qc or {}),
    }


def _validate_checkpoint(
    path: Path, preflight: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint = _private_canonical_checkpoint_object(path)
    expected_keys = {
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
    if set(checkpoint) != expected_keys or _forbidden_keys(checkpoint):
        raise UnknownAudioError(f"terminal checkpoint schema is invalid: {path.name}")
    if (
        checkpoint.get("schema_version") != UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION
        or checkpoint.get("preflight_sha256") != preflight.get("preflight_sha256")
        or checkpoint.get("plan_sha256") != preflight.get("plan_sha256")
        or checkpoint.get("candidate_id") != candidate.get("candidate_id")
        or checkpoint.get("scientific_name") != candidate.get("scientific_name")
        or checkpoint.get("queue_rank") != candidate.get("queue_rank")
        or checkpoint.get("session_group") != candidate.get("session_group")
        or checkpoint.get("disposition") not in TERMINAL_DISPOSITIONS
    ):
        raise UnknownAudioError(f"terminal checkpoint binding is invalid: {path.name}")
    reasons = checkpoint.get("reasons")
    receipt = checkpoint.get("download_receipt")
    qc = checkpoint.get("audio_qc")
    if (
        not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or not isinstance(receipt, Mapping)
        or not isinstance(qc, Mapping)
    ):
        raise UnknownAudioError(f"terminal checkpoint content is invalid: {path.name}")
    disposition = checkpoint["disposition"]
    preflight_disposition = candidate.get("disposition")
    allowed_by_preflight = {
        "metadata_excluded": {"metadata_excluded"},
        "session_noncanonical": {"session_noncanonical"},
        "canonical_pending_audio_qc": {
            "download_unavailable_terminal",
            "audio_qc_excluded",
            "eligible",
            "not_evaluated_pool_target_reached",
        },
    }
    if disposition not in allowed_by_preflight.get(str(preflight_disposition), set()):
        raise UnknownAudioError(f"checkpoint disposition is impossible: {path.name}")
    if disposition in {"metadata_excluded", "session_noncanonical"}:
        if reasons != candidate.get("reasons") or receipt or qc:
            raise UnknownAudioError(f"preflight checkpoint content drifted: {path.name}")
    elif disposition == "download_unavailable_terminal":
        if reasons != ["source_audio_terminally_unavailable"] or receipt or qc:
            raise UnknownAudioError(f"terminal download checkpoint is invalid: {path.name}")
    elif disposition == "not_evaluated_pool_target_reached":
        if reasons != ["candidate_pool_target_reached"] or receipt or qc:
            raise UnknownAudioError(f"pool-stop checkpoint is invalid: {path.name}")
    else:
        qc_base_keys = {
            "candidate_id",
            "scientific_name",
            "session_group",
            "relative_path",
            "sha256",
            "file_size_bytes",
            "header_detection_status",
            "header_type",
            "probe_status",
            "format_name",
            "codec_name",
            "source_sample_rate_hz",
            "channels",
            "ffprobe_duration_seconds",
            "full_decode_status",
            "decoded_duration_seconds",
            "decoded_duration_ratio",
            "full_decode_diagnostic",
            "disposition",
            "reasons",
        }
        expected_qc_keys = qc_base_keys | (
            {"assignment_descriptor"} if disposition == "eligible" else set()
        )
        if set(qc) != expected_qc_keys or qc.get("disposition") != disposition:
            raise UnknownAudioError(f"checkpoint QC schema is invalid: {path.name}")
        if (
            qc.get("candidate_id") != candidate["candidate_id"]
            or qc.get("scientific_name") != candidate["scientific_name"]
            or qc.get("session_group") != candidate["session_group"]
            or qc.get("reasons") != reasons
        ):
            raise UnknownAudioError(f"checkpoint QC binding is invalid: {path.name}")
        try:
            _validate_sanitized_receipt(
                receipt,
                candidate,
                str(qc.get("sha256") or ""),
                qc.get("file_size_bytes"),
                "checkpoint",
            )
        except UnknownAudioError as exc:
            raise UnknownAudioError(f"{exc}: {path.name}") from exc
    if checkpoint["disposition"] == "eligible" and not isinstance(qc, Mapping):
        raise UnknownAudioError("eligible checkpoint has no QC record")
    if isinstance(qc, Mapping) and qc:
        relative = str(qc.get("relative_path") or "")
        digest = str(qc.get("sha256") or "")
        expected_relative = _expected_raw_relative_path(candidate)
        if relative != expected_relative:
            raise UnknownAudioError(f"checkpoint raw audio path is invalid: {path.name}")
        try:
            _validate_private_raw_file(relative, digest, qc.get("file_size_bytes"))
        except UnknownAudioError as exc:
            raise UnknownAudioError(f"checkpoint audio binding failed: {path.name}: {exc}") from exc
        try:
            objective_reasons = set(_derive_objective_qc_reasons(qc))
        except UnknownAudioError as exc:
            raise UnknownAudioError(f"checkpoint QC derivation failed: {path.name}: {exc}") from exc
        duplicate_reasons = set(reasons).intersection(_DUPLICATE_QC_REASONS)
        if set(reasons) != objective_reasons | duplicate_reasons:
            raise UnknownAudioError(f"checkpoint QC reason set is not exact: {path.name}")
        expected_disposition = (
            "audio_qc_excluded" if objective_reasons or duplicate_reasons else "eligible"
        )
        if disposition != expected_disposition:
            raise UnknownAudioError(f"checkpoint QC disposition is not derived: {path.name}")
    if disposition == "eligible":
        descriptor = qc.get("assignment_descriptor")
        expected_descriptor_keys = {
            "candidate_id",
            "session_group",
            "container",
            "source_rate_bucket",
            "channels",
            "quality",
            "duration_bucket",
            "duration_seconds",
        }
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != expected_descriptor_keys
            or reasons
        ):
            raise UnknownAudioError(f"eligible checkpoint descriptor is invalid: {path.name}")
        if dict(descriptor) != _expected_assignment_descriptor(candidate, qc):
            raise UnknownAudioError(f"eligible checkpoint descriptor drifted: {path.name}")
    return checkpoint


def _write_or_reuse_checkpoint(
    working_directory: Path,
    preflight: Mapping[str, Any],
    candidate: Mapping[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    path = _checkpoint_path(working_directory, candidate["candidate_id"])
    if os.path.lexists(path):
        existing = _validate_checkpoint(path, preflight, candidate)
        if existing != checkpoint:
            raise UnknownAudioError(f"terminal checkpoint cannot be rebound: {path.name}")
        return existing
    _create_json_exclusive(path, checkpoint)
    return _validate_checkpoint(path, preflight, candidate)


def _exception_kind(exc: BaseException) -> str:
    if isinstance(exc, (UnknownAudioTerminalUnavailableError, TerminalAudioUnavailableError)):
        return "terminal_unavailable"
    if isinstance(exc, (UnknownAudioRetryableError, RetryableAudioDownloadError)):
        return "retryable"
    return "fatal"


def _species_audit(
    candidates: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    client: DownloadClient,
    working_directory: Path,
    raw_directory: Path,
    *,
    ffprobe: Path,
    ffmpeg: Path,
    known_hashes: set[str],
    observed_unknown_hashes: dict[str, str],
    detect_header_fn: Callable[[Path], str],
    probe_fn: Callable[[Path, Path], AudioProbe | Mapping[str, Any]],
    full_decode_fn: Callable[[Path, Path], FullDecodeResult | Mapping[str, Any]],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise UnknownAudioError("species audit received no candidate inventory")
    scientific_name = str(candidates[0]["scientific_name"])
    role = str(candidates[0]["role"])
    eligible: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    unresolved: list[str] = []
    terminal_count = 0

    def emit(event: str, candidate: Mapping[str, Any] | None = None, disposition: str = "") -> None:
        record: dict[str, Any] = {
            "event": event,
            "scientific_name": scientific_name,
            "eligible_recordings": len(eligible),
            "terminal_recordings": terminal_count,
            "inventory_recordings": len(candidates),
        }
        if candidate is not None:
            record.update(
                {
                    "candidate_id": candidate["candidate_id"],
                    "queue_rank": candidate["queue_rank"],
                }
            )
        if disposition:
            record["completion_state" if event == "species_complete" else "disposition"] = (
                disposition
            )
        _emit_progress(progress_callback, record)

    for candidate in candidates:
        if candidate["scientific_name"] != scientific_name:
            raise UnknownAudioError("species audit candidate set is mixed")
        preflight_disposition = candidate["disposition"]
        checkpoint_path = _checkpoint_path(working_directory, candidate["candidate_id"])
        if os.path.lexists(checkpoint_path):
            checkpoint = _validate_checkpoint(checkpoint_path, preflight, candidate)
            if (
                checkpoint["disposition"] == "not_evaluated_pool_target_reached"
                and len(eligible) < CANDIDATE_POOL_TARGET
            ):
                raise UnknownAudioError("pool-stop checkpoint appears before 80 eligible records")
            if (
                preflight_disposition == "canonical_pending_audio_qc"
                and len(eligible) >= CANDIDATE_POOL_TARGET
                and checkpoint["disposition"] != "not_evaluated_pool_target_reached"
            ):
                raise UnknownAudioError("evaluated canonical checkpoint appears after pool target")
            disposition_counts[checkpoint["disposition"]] += 1
            terminal_count += 1
            if checkpoint["disposition"] == "eligible":
                descriptor = checkpoint["audio_qc"].get("assignment_descriptor")
                if not isinstance(descriptor, Mapping):
                    raise UnknownAudioError("eligible checkpoint descriptor is invalid")
                eligible.append(dict(descriptor))
            checkpoint_digest = str(checkpoint.get("audio_qc", {}).get("sha256") or "")
            if _SHA256.fullmatch(checkpoint_digest):
                duplicate_known = checkpoint_digest in known_hashes
                duplicate_unknown = checkpoint_digest in observed_unknown_hashes
                checkpoint_reasons = set(checkpoint["reasons"])
                if checkpoint["disposition"] == "eligible" and (
                    duplicate_known or duplicate_unknown
                ):
                    raise UnknownAudioError("resumed eligible checkpoint contains duplicate audio")
                expected_duplicate_reasons: set[str] = set()
                if duplicate_known:
                    expected_duplicate_reasons.add("exact_duplicate_of_retained_known")
                elif duplicate_unknown:
                    expected_duplicate_reasons.add("exact_duplicate_of_earlier_unknown_candidate")
                if (
                    checkpoint_reasons.intersection(_DUPLICATE_QC_REASONS)
                    != expected_duplicate_reasons
                ):
                    raise UnknownAudioError("resumed duplicate reason set is not exact")
                observed_unknown_hashes.setdefault(checkpoint_digest, candidate["candidate_id"])
            _remove_matching_pending_qc_record(working_directory, preflight, candidate, checkpoint)
            emit("candidate_terminal", candidate, checkpoint["disposition"])
            continue

        if preflight_disposition in {"metadata_excluded", "session_noncanonical"}:
            checkpoint = _terminal_checkpoint(
                preflight,
                candidate,
                preflight_disposition,
                candidate["reasons"],
            )
            _write_or_reuse_checkpoint(working_directory, preflight, candidate, checkpoint)
            disposition_counts[preflight_disposition] += 1
            terminal_count += 1
            emit("candidate_terminal", candidate, preflight_disposition)
            continue
        if preflight_disposition != "canonical_pending_audio_qc":
            raise UnknownAudioError("candidate has an invalid preflight disposition")
        if len(eligible) >= CANDIDATE_POOL_TARGET:
            checkpoint = _terminal_checkpoint(
                preflight,
                candidate,
                "not_evaluated_pool_target_reached",
                ["candidate_pool_target_reached"],
            )
            checkpoint = _write_or_reuse_checkpoint(
                working_directory, preflight, candidate, checkpoint
            )
            disposition_counts[checkpoint["disposition"]] += 1
            terminal_count += 1
            emit("candidate_terminal", candidate, checkpoint["disposition"])
            continue

        pending_path = _pending_qc_path(working_directory, candidate["candidate_id"])
        staging_path = _staging_audio_path(working_directory, candidate)
        raw_path = PROJECT_ROOT / _expected_raw_relative_path(candidate)
        if os.path.lexists(pending_path):
            destination, receipt = _promote_pending_qc_audio(
                working_directory, raw_directory, preflight, candidate
            )
        else:
            if os.path.lexists(staging_path) or os.path.lexists(raw_path):
                raise UnknownAudioError(
                    f"unbound audio exists without pending-QC state: {candidate['candidate_id']}"
                )
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            if staging_path.parent.resolve() != staging_path.parent:
                raise UnknownAudioError("staging audio parent traverses a symbolic link")
            try:
                receipt_object = client.download(
                    candidate["candidate_id"], candidate["download_url"], staging_path
                )
            except Exception as exc:
                if os.path.lexists(staging_path):
                    raise UnknownAudioError(
                        "secure downloader left an unbound staging artifact"
                    ) from exc
                _prune_empty_staging_directories(working_directory, staging_path)
                kind = _exception_kind(exc)
                if kind == "retryable":
                    unresolved.append(candidate["candidate_id"])
                    disposition_counts["unresolved_retryable"] += 1
                    emit("retryable_block", candidate, "unresolved_retryable")
                    break
                if kind == "terminal_unavailable":
                    checkpoint = _terminal_checkpoint(
                        preflight,
                        candidate,
                        "download_unavailable_terminal",
                        ["source_audio_terminally_unavailable"],
                    )
                    _write_or_reuse_checkpoint(working_directory, preflight, candidate, checkpoint)
                    disposition_counts["download_unavailable_terminal"] += 1
                    terminal_count += 1
                    emit("candidate_terminal", candidate, "download_unavailable_terminal")
                    continue
                raise UnknownAudioError(
                    f"fatal secure download failure for {candidate['candidate_id']}: "
                    f"{type(exc).__name__}"
                ) from exc
            initial_staging_identity = _private_audio_identity(staging_path)
            receipt = _safe_download_receipt(receipt_object, candidate, staging_path)
            _validate_sanitized_receipt(
                receipt,
                candidate,
                receipt["sha256"],
                receipt["bytes_written"],
                "fresh download",
            )
            staging_identity = _validate_bound_audio_file(
                staging_path,
                receipt["sha256"],
                receipt["bytes_written"],
                allowed_link_counts=frozenset({1}),
                fsync_file=True,
            )
            if staging_identity != initial_staging_identity:
                raise UnknownAudioError("staging audio changed before pending-QC sealing")
            pending = _pending_qc_record(
                working_directory,
                preflight,
                candidate,
                receipt,
                staging_identity,
            )
            _create_json_exclusive(pending_path, pending)
            _validate_pending_qc_record(pending_path, working_directory, preflight, candidate)
            destination, receipt = _promote_pending_qc_audio(
                working_directory, raw_directory, preflight, candidate
            )
        try:
            qc = audit_unknown_audio_file(
                destination,
                candidate,
                config,
                ffprobe=ffprobe,
                ffmpeg=ffmpeg,
                detect_header_fn=detect_header_fn,
                probe_fn=probe_fn,
                full_decode_fn=full_decode_fn,
            )
        except UnknownAudioRetryableError:
            unresolved.append(candidate["candidate_id"])
            disposition_counts["unresolved_retryable"] += 1
            emit("retryable_block", candidate, "unresolved_retryable")
            break
        digest = qc["sha256"]
        if digest in known_hashes:
            qc["disposition"] = "audio_qc_excluded"
            qc["reasons"] = sorted(set(qc["reasons"] + ["exact_duplicate_of_retained_known"]))
            qc.pop("assignment_descriptor", None)
        elif digest in observed_unknown_hashes:
            qc["disposition"] = "audio_qc_excluded"
            qc["reasons"] = sorted(
                set(qc["reasons"] + ["exact_duplicate_of_earlier_unknown_candidate"])
            )
            qc.pop("assignment_descriptor", None)
        observed_unknown_hashes.setdefault(digest, candidate["candidate_id"])
        disposition = qc["disposition"]
        checkpoint = _terminal_checkpoint(
            preflight,
            candidate,
            disposition,
            qc["reasons"],
            receipt=receipt,
            qc=qc,
        )
        checkpoint = _write_or_reuse_checkpoint(working_directory, preflight, candidate, checkpoint)
        _remove_matching_pending_qc_record(working_directory, preflight, candidate, checkpoint)
        disposition_counts[disposition] += 1
        terminal_count += 1
        if disposition == "eligible":
            eligible.append(dict(qc["assignment_descriptor"]))
        emit("candidate_terminal", candidate, disposition)

    if unresolved:
        completion_state = "blocked_retryable"
    elif len(eligible) >= CANDIDATE_POOL_TARGET:
        completion_state = "pool_satisfied"
    elif terminal_count == len(candidates):
        completion_state = "inventory_exhausted"
    else:
        raise UnknownAudioError("species audit stopped without a locked terminal state")
    result = {
        "role": role,
        "scientific_name": scientific_name,
        "inventory_recordings": len(candidates),
        "terminal_recordings": terminal_count,
        "eligible_recordings": len(eligible),
        "unresolved_retryable": len(unresolved),
        "unresolved_candidate_ids": unresolved,
        "completion_state": completion_state,
        "dispositions": dict(sorted(disposition_counts.items())),
        "eligible_descriptors": eligible,
    }
    emit("species_complete", disposition=completion_state)
    return result


def _checkpoint_set(working_directory: Path) -> list[dict[str, str]]:
    checkpoint_directory = working_directory / "checkpoints"
    if not checkpoint_directory.exists():
        return []
    unexpected = [
        path.name
        for path in checkpoint_directory.iterdir()
        if not re.fullmatch(r"XC[1-9][0-9]*\.json", path.name)
        or path.is_symlink()
        or not path.is_file()
    ]
    if unexpected:
        raise UnknownAudioError(f"unexpected checkpoint artifacts: {sorted(unexpected)}")
    result: list[dict[str, str]] = []
    for path in sorted(checkpoint_directory.glob("XC*.json"), key=lambda item: item.name):
        if path.is_file():
            _private_canonical_checkpoint_object(path)
            result.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    return result


def _raw_file_set(raw_directory: Path) -> list[dict[str, Any]]:
    if not raw_directory.exists():
        return []
    expected_root = PROJECT_ROOT / "data" / "unknown" / "raw" / "audio_v1"
    if raw_directory != expected_root or raw_directory.is_symlink() or not raw_directory.is_dir():
        raise UnknownAudioError("unknown raw audio root is invalid")
    result: list[dict[str, Any]] = []
    for path in sorted(raw_directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise UnknownAudioError("unknown raw audio artifact cannot be a symbolic link")
        if path.is_file():
            observed = path.lstat()
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                raise UnknownAudioError(
                    "unknown raw audio artifact is not a private single-link regular file"
                )
            result.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                    "file_size_bytes": path.stat().st_size,
                }
            )
    return result


def _load_plan_for_preflight(
    config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    plan_path = _project_input(config["inputs"]["candidate_plan"], "candidate plan")
    if sha256_file(plan_path) != preflight["plan_sha256"]:
        raise UnknownAudioError("candidate plan changed after preflight")
    return _json_object(plan_path)


def _compact_species_results(
    species_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "eligible_descriptors"}
        for row in species_results
    ]


def _publication_records(
    *,
    config_file: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    preflight: Mapping[str, Any],
    plan: Mapping[str, Any],
    known_manifest_sha256: str,
    known_hashes: set[str],
    working_directory: Path,
    raw_directory: Path,
    audit_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    _require_no_pending_or_staging_artifacts(working_directory)
    species_results, gate, selection = _rederive_completed_audit(
        preflight,
        plan,
        working_directory,
        raw_directory,
        known_hashes,
    )
    checkpoints = _checkpoint_set(working_directory)
    raw_files = _raw_file_set(raw_directory)
    audit = {
        "schema_version": UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "download_policy_sha256": sha256_json(config["download"]),
        "plan_sha256": preflight["plan_sha256"],
        "plan_lock_sha256": preflight["plan_lock_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "known_manifest_sha256": known_manifest_sha256,
        "gate": gate,
        "species": _compact_species_results(species_results),
        "selection": selection,
        "checkpoint_count": len(checkpoints),
        "checkpoint_set_sha256": sha256_json(checkpoints),
        "raw_file_count": len(raw_files),
        "raw_file_set_sha256": sha256_json(raw_files),
        "ready_for_unknown_scoring": True,
    }
    lock = {
        "schema_version": UNKNOWN_AUDIO_AUDIT_SCHEMA_VERSION,
        "audit_path": audit_path.relative_to(PROJECT_ROOT).as_posix(),
        "audit_sha256": sha256_bytes(_canonical_json_bytes(audit)),
        "config_path": config_file.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": config_sha256,
        "plan_sha256": preflight["plan_sha256"],
        "plan_lock_sha256": preflight["plan_lock_sha256"],
        "download_policy_sha256": sha256_json(config["download"]),
        "checkpoint_count": len(checkpoints),
        "checkpoint_set_sha256": sha256_json(checkpoints),
        "raw_file_count": len(raw_files),
        "raw_file_set_sha256": sha256_json(raw_files),
        "selected_recordings": selection["selected_recordings"],
        "ready_for_unknown_scoring": True,
    }
    return species_results, gate, selection, audit, lock


def _require_publication_inputs_unchanged(
    config_file: Path,
    config_sha256: str,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    known_path: Path,
    known_sha256: str,
) -> None:
    require_unchanged(config_file, config_sha256)
    require_unchanged(known_path, known_sha256)
    plan_path = _project_input(config["inputs"]["candidate_plan"], "candidate plan")
    plan_lock_path = _project_input(config["inputs"]["candidate_plan_lock"], "candidate plan lock")
    require_unchanged(plan_path, preflight["plan_sha256"])
    require_unchanged(plan_lock_path, preflight["plan_lock_sha256"])


def _recover_orphan_audit(
    *,
    config_file: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    preflight: Mapping[str, Any],
    plan: Mapping[str, Any],
    known_path: Path,
    known_sha256: str,
    known_hashes: set[str],
    working_directory: Path,
    raw_directory: Path,
    audit_path: Path,
    audit_lock_path: Path,
) -> None:
    if not audit_path.exists() or audit_lock_path.exists():
        raise UnknownAudioError("orphan audit recovery state is invalid")
    _, _, _, expected_audit, expected_lock = _publication_records(
        config_file=config_file,
        config=config,
        config_sha256=config_sha256,
        preflight=preflight,
        plan=plan,
        known_manifest_sha256=known_sha256,
        known_hashes=known_hashes,
        working_directory=working_directory,
        raw_directory=raw_directory,
        audit_path=audit_path,
    )
    observed_audit = _private_canonical_json_object(audit_path, "unknown audio audit")
    if observed_audit != expected_audit or audit_path.read_bytes() != _canonical_json_bytes(
        expected_audit
    ):
        raise UnknownAudioError(
            "orphan unknown audio audit is not the exact reproducible publication"
        )
    _require_publication_inputs_unchanged(
        config_file,
        config_sha256,
        config,
        preflight,
        known_path,
        known_sha256,
    )
    _create_json_exclusive(audit_lock_path, expected_lock)


def _run_unknown_audio_acquisition(
    client: DownloadClient,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    preflight: Mapping[str, Any] | None = None,
    ffprobe: str | Path,
    ffmpeg: str | Path,
    detect_header_fn: Callable[[Path], str] = detect_header,
    probe_fn: Callable[[Path, Path], AudioProbe | Mapping[str, Any]] = probe_audio,
    full_decode_fn: Callable[
        [Path, Path], FullDecodeResult | Mapping[str, Any]
    ] = verify_full_decode,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Acquire fixed canonical candidates with resumable terminal checkpoints."""
    config_file = resolve_project_path(config_path)
    if not is_relative_to(config_file, PROJECT_ROOT):
        raise UnknownAudioError("unknown audio config must remain inside the project")
    config_sha256 = sha256_file(config_file)
    config = load_unknown_audio_config(config_file)
    supplied_preflight = dict(preflight) if preflight is not None else None
    regenerated = preflight_unknown_audio(config_file)
    if supplied_preflight is not None:
        supplied_deterministic = {
            key: value for key, value in supplied_preflight.items() if key != "disk"
        }
        regenerated_deterministic = {
            key: value for key, value in regenerated.items() if key != "disk"
        }
        if supplied_deterministic != regenerated_deterministic:
            raise UnknownAudioError("supplied preflight does not match regenerated locked inputs")
    preflight_record = regenerated
    if preflight_record.get("config_sha256") != config_sha256:
        raise UnknownAudioError("preflight is bound to a different unknown audio config")
    working_directory = _locked_output_path(
        config["outputs"]["working_directory"], "working directory"
    )
    raw_directory = _locked_output_path(config["outputs"]["raw_directory"], "raw directory")
    audit_path = _locked_output_path(config["outputs"]["audit"], "audit")
    audit_lock_path = _locked_output_path(config["outputs"]["audit_lock"], "audit lock")
    if audit_lock_path.exists():
        return verify_unknown_audio_audit(config_file)
    if (
        not audit_path.exists()
        and preflight_record.get("disk", {}).get("estimated_space_sufficient") is not True
    ):
        raise UnknownAudioError("unknown audio disk preflight did not pass")
    plan = _load_plan_for_preflight(config, preflight_record)

    known_path = _project_input(config["inputs"]["known_manifest"], "known manifest")
    known_rows, known_sha256 = read_csv_snapshot(known_path)
    if known_sha256 != preflight_record["known_manifest_sha256"]:
        raise UnknownAudioError("known manifest changed after preflight")
    known_hashes = {
        row["sha256"]
        for row in known_rows
        if row.get("local_qc_status") == "include" and _SHA256.fullmatch(row.get("sha256", ""))
    }
    candidates = preflight_record.get("candidates")
    if not isinstance(candidates, list):
        raise UnknownAudioError("preflight candidate table is invalid")
    if audit_path.exists():
        with project_lock("unknown_audio_acquisition"):
            if audit_lock_path.exists():
                return verify_unknown_audio_audit(config_file)
            _recover_orphan_audit(
                config_file=config_file,
                config=config,
                config_sha256=config_sha256,
                preflight=preflight_record,
                plan=plan,
                known_path=known_path,
                known_sha256=known_sha256,
                known_hashes=known_hashes,
                working_directory=working_directory,
                raw_directory=raw_directory,
                audit_path=audit_path,
                audit_lock_path=audit_lock_path,
            )
        return verify_unknown_audio_audit(config_file)
    _validate_injected_client_policy(client, config)
    observed_hashes: dict[str, str] = {}
    species_results: list[dict[str, Any]] = []
    with project_lock("unknown_audio_acquisition"):
        if audit_lock_path.exists():
            return verify_unknown_audio_audit(config_file)
        if audit_path.exists():
            _recover_orphan_audit(
                config_file=config_file,
                config=config,
                config_sha256=config_sha256,
                preflight=preflight_record,
                plan=plan,
                known_path=known_path,
                known_sha256=known_sha256,
                known_hashes=known_hashes,
                working_directory=working_directory,
                raw_directory=raw_directory,
                audit_path=audit_path,
                audit_lock_path=audit_lock_path,
            )
            return verify_unknown_audio_audit(config_file)
        _prepare_acquisition_roots(working_directory, raw_directory)
        primary_names = [
            queue["scientific_name"]
            for queue in plan["candidate_queues"]
            if queue["role"] == "primary"
        ]
        for scientific_name in primary_names:
            rows = [row for row in candidates if row["scientific_name"] == scientific_name]
            species_result = _species_audit(
                rows,
                preflight_record,
                config,
                client,
                working_directory,
                raw_directory,
                ffprobe=Path(ffprobe),
                ffmpeg=Path(ffmpeg),
                known_hashes=known_hashes,
                observed_unknown_hashes=observed_hashes,
                detect_header_fn=detect_header_fn,
                probe_fn=probe_fn,
                full_decode_fn=full_decode_fn,
                progress_callback=progress_callback,
            )
            species_results.append(species_result)
            if species_result["unresolved_retryable"]:
                gate = {
                    "status": "blocked_retryable_or_incomplete_primary_audit",
                    "fallback_active": False,
                    "failed_primary_species": [],
                    "blocked_species": [scientific_name],
                    "replacement": None,
                }
                return {
                    "complete": False,
                    "status": gate["status"],
                    "gate": gate,
                    "species": _compact_species_results(species_results),
                    "checkpoint_count": len(_checkpoint_set(working_directory)),
                }
        gate = evaluate_fallback_gate(species_results)
        if gate["status"] == "fallback_audit_required":
            fallback_name = next(
                queue["scientific_name"]
                for queue in plan["candidate_queues"]
                if queue["role"] == "fallback"
            )
            rows = [row for row in candidates if row["scientific_name"] == fallback_name]
            fallback_result = _species_audit(
                rows,
                preflight_record,
                config,
                client,
                working_directory,
                raw_directory,
                ffprobe=Path(ffprobe),
                ffmpeg=Path(ffmpeg),
                known_hashes=known_hashes,
                observed_unknown_hashes=observed_hashes,
                detect_header_fn=detect_header_fn,
                probe_fn=probe_fn,
                full_decode_fn=full_decode_fn,
                progress_callback=progress_callback,
            )
            species_results.append(fallback_result)
            gate = evaluate_fallback_gate(species_results)
            if fallback_result["unresolved_retryable"]:
                return {
                    "complete": False,
                    "status": gate["status"],
                    "gate": gate,
                    "species": _compact_species_results(species_results),
                    "checkpoint_count": len(_checkpoint_set(working_directory)),
                }
        if gate["status"] not in {"ready_without_fallback", "ready_with_fallback"}:
            return {
                "complete": False,
                "status": gate["status"],
                "gate": gate,
                "species": [
                    {key: value for key, value in row.items() if key != "eligible_descriptors"}
                    for row in species_results
                ],
                "checkpoint_count": len(_checkpoint_set(working_directory)),
            }

        failed = set(gate["failed_primary_species"])
        final_rows = [
            row
            for row in species_results
            if row["eligible_recordings"] >= TARGET_RECORDINGS
            and (row["role"] == "primary" and row["scientific_name"] not in failed)
        ]
        if gate["fallback_active"]:
            final_rows.extend(row for row in species_results if row["role"] == "fallback")
        candidates_by_species = {
            row["scientific_name"]: row["eligible_descriptors"] for row in final_rows
        }
        selection = select_final_unknown_recordings(
            plan["known_test_reference"], candidates_by_species
        )
        (
            rederived_species,
            rederived_gate,
            rederived_selection,
            audit,
            lock,
        ) = _publication_records(
            config_file=config_file,
            config=config,
            config_sha256=config_sha256,
            preflight=preflight_record,
            plan=plan,
            known_manifest_sha256=known_sha256,
            known_hashes=known_hashes,
            working_directory=working_directory,
            raw_directory=raw_directory,
            audit_path=audit_path,
        )
        compact_species = _compact_species_results(species_results)
        rederived_compact_species = _compact_species_results(rederived_species)
        if (
            compact_species != rederived_compact_species
            or gate != rederived_gate
            or selection != rederived_selection
        ):
            raise UnknownAudioError("completed acquisition is not reproducible before publication")
        _require_publication_inputs_unchanged(
            config_file,
            config_sha256,
            config,
            preflight_record,
            known_path,
            known_sha256,
        )
        _create_json_exclusive(audit_path, audit)
        if sha256_file(audit_path) != lock["audit_sha256"]:
            raise UnknownAudioError("published unknown audio audit bytes are not canonical")
        _create_json_exclusive(audit_lock_path, lock)
    return verify_unknown_audio_audit(config_file)


def run_unknown_audio_acquisition(
    client: DownloadClient,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    preflight: Mapping[str, Any] | None = None,
    ffprobe: str | Path,
    ffmpeg: str | Path,
    detect_header_fn: Callable[[Path], str] = detect_header,
    probe_fn: Callable[[Path, Path], AudioProbe | Mapping[str, Any]] = probe_audio,
    full_decode_fn: Callable[
        [Path, Path], FullDecodeResult | Mapping[str, Any]
    ] = verify_full_decode,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Acquire fixed candidates and normalize operational failures at the boundary."""
    try:
        return _run_unknown_audio_acquisition(
            client,
            config_path=config_path,
            preflight=preflight,
            ffprobe=ffprobe,
            ffmpeg=ffmpeg,
            detect_header_fn=detect_header_fn,
            probe_fn=probe_fn,
            full_decode_fn=full_decode_fn,
            progress_callback=progress_callback,
        )
    except UnknownAudioError:
        raise
    except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
        raise UnknownAudioError(f"unknown audio acquisition failed: {type(exc).__name__}") from exc


def _rederive_completed_audit(
    preflight: Mapping[str, Any],
    plan: Mapping[str, Any],
    working_directory: Path,
    raw_directory: Path,
    known_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates = preflight.get("candidates")
    if not isinstance(candidates, list):
        raise UnknownAudioError("verified preflight candidate table is invalid")
    by_species = {
        queue["scientific_name"]: [
            row for row in candidates if row["scientific_name"] == queue["scientific_name"]
        ]
        for queue in plan["candidate_queues"]
    }
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    checkpoint_files = {
        path.stem: path
        for path in (working_directory / "checkpoints").glob("XC*.json")
        if path.is_file() and not path.is_symlink()
    }
    observed_hashes: dict[str, str] = {}
    expected_raw_paths: set[str] = set()

    def summarize(queue: Mapping[str, Any]) -> dict[str, Any]:
        rows = by_species[queue["scientific_name"]]
        eligible: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for candidate in rows:
            path = checkpoint_files.get(candidate["candidate_id"])
            if path is None:
                raise UnknownAudioError(
                    f"completed species lacks checkpoint: {candidate['candidate_id']}"
                )
            checkpoint = _validate_checkpoint(path, preflight, candidate)
            disposition = checkpoint["disposition"]
            if (
                disposition == "not_evaluated_pool_target_reached"
                and len(eligible) < CANDIDATE_POOL_TARGET
            ):
                raise UnknownAudioError("pool-stop checkpoint appears before 80 eligible records")
            if disposition == "eligible":
                if len(eligible) >= CANDIDATE_POOL_TARGET:
                    raise UnknownAudioError("eligible checkpoint appears after the pool target")
                eligible.append(dict(checkpoint["audio_qc"]["assignment_descriptor"]))
            elif (
                candidate["disposition"] == "canonical_pending_audio_qc"
                and len(eligible) >= CANDIDATE_POOL_TARGET
                and disposition != "not_evaluated_pool_target_reached"
            ):
                raise UnknownAudioError("evaluated canonical checkpoint appears after pool target")
            qc = checkpoint["audio_qc"]
            if qc:
                digest = qc["sha256"]
                expected_raw_paths.add(qc["relative_path"])
                duplicate_known = digest in known_hashes
                duplicate_unknown = digest in observed_hashes
                reasons = set(checkpoint["reasons"])
                if disposition == "eligible" and (duplicate_known or duplicate_unknown):
                    raise UnknownAudioError("eligible checkpoint contains duplicate audio")
                expected_duplicate_reasons: set[str] = set()
                if duplicate_known:
                    expected_duplicate_reasons.add("exact_duplicate_of_retained_known")
                elif duplicate_unknown:
                    expected_duplicate_reasons.add("exact_duplicate_of_earlier_unknown_candidate")
                if reasons.intersection(_DUPLICATE_QC_REASONS) != expected_duplicate_reasons:
                    raise UnknownAudioError(
                        "checkpoint exact-duplicate reasons are not reproducible"
                    )
                observed_hashes.setdefault(digest, candidate["candidate_id"])
            counts[disposition] += 1
        completion = (
            "pool_satisfied" if len(eligible) >= CANDIDATE_POOL_TARGET else "inventory_exhausted"
        )
        return {
            "role": queue["role"],
            "scientific_name": queue["scientific_name"],
            "inventory_recordings": len(rows),
            "terminal_recordings": len(rows),
            "eligible_recordings": len(eligible),
            "unresolved_retryable": 0,
            "unresolved_candidate_ids": [],
            "completion_state": completion,
            "dispositions": dict(sorted(counts.items())),
            "eligible_descriptors": eligible,
        }

    primary_queues = [queue for queue in plan["candidate_queues"] if queue["role"] == "primary"]
    species_results = [summarize(queue) for queue in primary_queues]
    gate = evaluate_fallback_gate(species_results)
    fallback_queue = next(
        queue for queue in plan["candidate_queues"] if queue["role"] == "fallback"
    )
    fallback_ids = {row["candidate_id"] for row in by_species[fallback_queue["scientific_name"]]}
    if gate["status"] == "fallback_audit_required":
        species_results.append(summarize(fallback_queue))
        gate = evaluate_fallback_gate(species_results)
    elif fallback_ids.intersection(checkpoint_files):
        raise UnknownAudioError("fallback checkpoints exist before fallback activation")
    if gate["status"] not in {"ready_without_fallback", "ready_with_fallback"}:
        raise UnknownAudioError("published audit does not satisfy the locked fallback gate")
    processed_ids = {
        row["candidate_id"]
        for result in species_results
        for row in by_species[result["scientific_name"]]
    }
    if set(checkpoint_files) != processed_ids:
        raise UnknownAudioError("checkpoint candidate set is not exact")
    if set(candidate_by_id) != {
        row["candidate_id"] for rows in by_species.values() for row in rows
    }:
        raise UnknownAudioError("preflight candidate IDs are not unique")

    actual_raw_paths = {row["path"] for row in _raw_file_set(raw_directory)}
    if actual_raw_paths != expected_raw_paths:
        raise UnknownAudioError("unknown raw audio file set does not match QC checkpoints")

    failed = set(gate["failed_primary_species"])
    final_rows = [
        row
        for row in species_results
        if row["role"] == "primary"
        and row["scientific_name"] not in failed
        and row["eligible_recordings"] >= TARGET_RECORDINGS
    ]
    if gate["fallback_active"]:
        final_rows.extend(row for row in species_results if row["role"] == "fallback")
    selection = select_final_unknown_recordings(
        plan["known_test_reference"],
        {row["scientific_name"]: row["eligible_descriptors"] for row in final_rows},
    )
    return species_results, gate, selection


def verify_unknown_audio_audit(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Rebuild and verify the audit, lock, checkpoints, audio, gate, and selection."""
    try:
        config_file = resolve_project_path(config_path)
        if not is_relative_to(config_file, PROJECT_ROOT):
            raise UnknownAudioError("unknown audio config must remain inside the project")
        config = load_unknown_audio_config(config_file)
        audit_path = _locked_output_path(config["outputs"]["audit"], "audit")
        lock_path = _locked_output_path(config["outputs"]["audit_lock"], "audit lock")
        audit = _private_canonical_json_object(audit_path, "unknown audio audit")
        lock = _private_canonical_json_object(lock_path, "unknown audio audit lock")
        preflight = preflight_unknown_audio(config_file)
        plan = _load_plan_for_preflight(config, preflight)
        known_path = _project_input(config["inputs"]["known_manifest"], "known manifest")
        known_rows, known_sha256 = read_csv_snapshot(known_path)
        known_hashes = {
            row["sha256"]
            for row in known_rows
            if row.get("local_qc_status") == "include" and _SHA256.fullmatch(row.get("sha256", ""))
        }
        working_directory = _locked_output_path(
            config["outputs"]["working_directory"], "working directory"
        )
        raw_directory = _locked_output_path(config["outputs"]["raw_directory"], "raw directory")
        (
            _species_results,
            gate,
            _selection,
            expected_audit,
            expected_lock,
        ) = _publication_records(
            config_file=config_file,
            config=config,
            config_sha256=sha256_file(config_file),
            preflight=preflight,
            plan=plan,
            known_manifest_sha256=known_sha256,
            known_hashes=known_hashes,
            working_directory=working_directory,
            raw_directory=raw_directory,
            audit_path=audit_path,
        )
        if audit != expected_audit or audit_path.read_bytes() != _canonical_json_bytes(
            expected_audit
        ):
            raise UnknownAudioError("unknown audio audit is not reproducible")
        if lock != expected_lock or lock_path.read_bytes() != _canonical_json_bytes(expected_lock):
            raise UnknownAudioError("unknown audio audit lock is not reproducible")
        return {
            "valid": True,
            "ready_for_unknown_scoring": True,
            "selected_recordings": expected_lock["selected_recordings"],
            "species": expected_audit["selection"]["species_count"],
            "fallback_active": bool(gate["fallback_active"]),
            "audit": audit_path.as_posix(),
            "audit_sha256": expected_lock["audit_sha256"],
            "checkpoint_count": expected_lock["checkpoint_count"],
        }
    except UnknownAudioError:
        raise
    except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
        raise UnknownAudioError(
            f"unknown audio audit verification failed: {type(exc).__name__}"
        ) from exc
