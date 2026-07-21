from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from bird_audio.audio import decode_smoke_test, probe_audio, resolve_tool, tool_version
from bird_audio.cache_audit import audit_known_clip_cache
from bird_audio.clip_cache import (
    DEFAULT_CACHE_ROOT,
    build_known_clip_cache,
    verify_known_clip_cache,
)
from bird_audio.config import (
    config_fingerprint,
    load_toml,
    public_config,
    validate_project_config_set,
)
from bird_audio.environment_tools import run_mps_smoke_test, write_dependency_lock
from bird_audio.final_evaluation import run_final_evaluation, verify_final_evaluation
from bird_audio.final_evaluation_gate import seal_final_evaluation_gate
from bird_audio.final_report_assets import build_final_report_assets, verify_final_report_assets
from bird_audio.manifest import audit_full_decodes, build_local_manifest, validate_local_manifest
from bird_audio.metadata import (
    DEFAULT_ENDPOINT,
    XenoCantoApiError,
    api_key_from_environment,
    enrich_manifest_from_cache,
    fetch_metadata_cache,
)
from bird_audio.metadata_artifacts import (
    create_enrichment_lock,
    seal_metadata_cache,
    verify_metadata_cache_lock,
)
from bird_audio.paths import PROJECT_ROOT, resolve_project_path
from bird_audio.provenance import (
    DEFAULT_ENVIRONMENT_V2_PATH,
    DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH,
    DEFAULT_MPS_SMOKE_V2_PATH,
    DEFAULT_SIGNAL_SMOKE_V2_PATH,
    save_environment,
)
from bird_audio.recovery_v2 import (
    seal_unknown_cache_v2_equivalence,
    verify_unknown_cache_v2_equivalence_certificate,
    verify_v1_recovery_manifest,
)
from bird_audio.review import apply_manual_review, prepare_manual_review, verify_review_lock
from bird_audio.run_identity import make_run_id
from bird_audio.secure_audio_download import DownloadPolicy, SecureXenoCantoAudioClient
from bird_audio.signal_smoke import run_signal_smoke_test
from bird_audio.splitting import freeze_grouped_split, validate_frozen_split
from bird_audio.task1_attribution import build_task1_attributions, verify_task1_attributions
from bird_audio.task1_training import (
    DEFAULT_CACHE_ROOT as TASK1_KNOWN_CACHE_ROOT,
)
from bird_audio.task1_training import (
    DEFAULT_RUN_ROOT as TASK1_RUN_ROOT,
)
from bird_audio.task1_training import (
    KNOWN_CACHE_LOCK_SHA256 as TASK1_KNOWN_CACHE_LOCK_SHA256,
)
from bird_audio.task1_training import (
    benchmark_task1_full_epoch,
    preflight_efficientnet_weights,
    run_task1_development,
)
from bird_audio.task2_training import (
    DEFAULT_CACHE_ROOT as TASK2_KNOWN_CACHE_ROOT,
)
from bird_audio.task2_training import (
    DEFAULT_RUN_ROOT as TASK2_RUN_ROOT,
)
from bird_audio.task2_training import (
    KNOWN_CACHE_LOCK_SHA256 as TASK2_KNOWN_CACHE_LOCK_SHA256,
)
from bird_audio.task2_training import (
    benchmark_task2_full_epoch,
    run_task2_development,
)
from bird_audio.unknown_acquisition import (
    UnknownAcquisitionConfigError,
    UnknownAcquisitionCredentialError,
    UnknownMetadataCacheError,
    fetch_unknown_metadata_cache,
    format_unknown_acquisition_error,
    seal_unknown_metadata_cache,
    verify_unknown_metadata_lock,
)
from bird_audio.unknown_audio import (
    DEFAULT_CONFIG as DEFAULT_UNKNOWN_AUDIO_CONFIG,
)
from bird_audio.unknown_audio import (
    UnknownAudioError,
    load_unknown_audio_config,
    preflight_unknown_audio,
    run_unknown_audio_acquisition,
    verify_unknown_audio_audit,
)
from bird_audio.unknown_clip_cache import (
    DEFAULT_AUDIT as DEFAULT_UNKNOWN_AUDIO_AUDIT,
)
from bird_audio.unknown_clip_cache import (
    DEFAULT_AUDIT_LOCK as DEFAULT_UNKNOWN_AUDIO_AUDIT_LOCK,
)
from bird_audio.unknown_clip_cache import (
    DEFAULT_CHECKPOINT_ROOT as DEFAULT_UNKNOWN_AUDIO_CHECKPOINT_ROOT,
)
from bird_audio.unknown_clip_cache import (
    DEFAULT_UNKNOWN_CLIP_CACHE_ROOT,
    build_unknown_clip_cache,
    verify_unknown_clip_cache,
)
from bird_audio.unknown_planning import (
    DEFAULT_CONFIG as DEFAULT_UNKNOWN_SELECTION_CONFIG,
)
from bird_audio.unknown_planning import (
    DEFAULT_PLAN as DEFAULT_UNKNOWN_CANDIDATE_PLAN,
)
from bird_audio.unknown_planning import (
    DEFAULT_PLAN_LOCK as DEFAULT_UNKNOWN_CANDIDATE_PLAN_LOCK,
)
from bird_audio.unknown_planning import (
    build_unknown_candidate_plan,
    validate_unknown_candidate_plan,
)

DEFAULT_DATA_CONFIG = "configs/data.toml"
DEFAULT_LOCAL_MANIFEST = "data/manifests/local_recordings.csv"
DEFAULT_LOCAL_SUMMARY = "data/manifests/local_manifest_summary.json"
DEFAULT_WORKING_METADATA_CACHE = "data/manifests/xc_metadata_working.json"
DEFAULT_SEALED_METADATA_CACHE = "data/manifests/xc_metadata_v1.json"
DEFAULT_METADATA_CACHE_LOCK = "data/manifests/xc_metadata_v1_lock.json"
DEFAULT_ENRICHED_BASE_MANIFEST = "data/manifests/recordings_enriched_v1.csv"
DEFAULT_LICENCE_MANIFEST = "data/manifests/licences_v1.csv"
DEFAULT_METADATA_SUMMARY = "data/manifests/metadata_enrichment_v1_summary.json"
DEFAULT_ENRICHMENT_LOCK = "data/manifests/metadata_enrichment_v1_lock.json"
DEFAULT_REVIEW_ITEMS = "data/manifests/manual_review_items_v1.csv"
DEFAULT_REVIEW_DECISIONS = "data/manifests/manual_review_decisions_v1.csv"
DEFAULT_REVIEW_PREPARATION = "data/manifests/manual_review_preparation_v1.json"
DEFAULT_FINAL_MANIFEST = "data/manifests/recordings.csv"
DEFAULT_REVIEW_RESOLUTION = "data/manifests/manual_review_resolution_v1.json"
DEFAULT_REVIEW_LOCK = "data/manifests/review_v1_lock.json"
DEFAULT_SPLIT = "data/splits/split_v1.csv"
DEFAULT_SPLIT_SUMMARY = "data/splits/split_v1_summary.json"
DEFAULT_SPLIT_LOCK = "data/splits/split_v1_lock.json"
DEFAULT_UNKNOWN_ACQUISITION_CONFIG = "configs/unknown_acquisition.toml"
DEFAULT_UNKNOWN_WORKING_METADATA = "data/unknown/metadata/unknown_metadata_working.json"
DEFAULT_UNKNOWN_SEALED_METADATA = "data/unknown/metadata/unknown_metadata_v1.json"
DEFAULT_UNKNOWN_METADATA_LOCK = "data/unknown/metadata/unknown_metadata_v1_lock.json"


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_print(value: object) -> None:
    print(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=_json_default,
        )
    )


def _environment_command(args: argparse.Namespace) -> int:
    destination, record = save_environment(
        args.output,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(f"Environment record saved: {destination}")
    _json_print(record)
    if not record["python"]["inside_project_venv"]:
        print("ERROR: command is not running from ANN_Project/.venv")
        return 2
    checks = {
        "pip_check": bool(record["pip_check"]["passed"]),
        "mps_built": bool(record["accelerator"]["mps"]["is_built"]),
        "mps_available": bool(record["accelerator"]["mps"]["is_available"]),
        "ffmpeg_available": not bool(record["tools"]["ffmpeg"]["error"]),
        "ffprobe_available": not bool(record["tools"]["ffprobe"]["error"]),
        "dependency_lock_verified": bool(
            record["verified_artifacts"]["dependency_lock"]["verified"]
        ),
        "mps_smoke_verified": bool(record["verified_artifacts"]["mps_smoke"]["verified"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"ERROR: environment verification failed: {', '.join(failed)}")
        return 2
    print("Environment verification passed")
    return 0


def _build_manifest_command(args: argparse.Namespace) -> int:
    ffprobe = resolve_tool("ffprobe", args.ffprobe)
    print(f"Using {tool_version(ffprobe)}")
    build_local_manifest(
        config_path=args.config,
        output_path=args.output,
        ffprobe=ffprobe,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    return 0


def _validate_manifest_command(args: argparse.Namespace) -> int:
    summary, issues = validate_local_manifest(
        config_path=args.config,
        manifest_path=args.manifest,
        summary_path=args.summary,
        verify_hashes=args.verify_hashes,
    )
    _json_print(summary)
    return 1 if any(issue["severity"] == "error" for issue in issues) else 0


def _audit_decode_command(args: argparse.Namespace) -> int:
    ffmpeg = resolve_tool("ffmpeg", args.ffmpeg)
    config = load_toml(args.config)
    quality_control = config["quality_control"]
    print(f"Using {tool_version(ffmpeg)}")
    audit_full_decodes(
        manifest_path=args.manifest,
        output_path=args.output,
        ffmpeg=ffmpeg,
        workers=args.workers,
        overwrite=args.overwrite,
        minimum_duration_ratio=float(quality_control["minimum_decoded_to_ffprobe_duration_ratio"]),
        maximum_duration_ratio=float(quality_control["maximum_decoded_to_ffprobe_duration_ratio"]),
        exclude_warnings=quality_control["full_decode_warning_policy"] == "exclude",
    )
    return 0


def _probe_smoke_command(args: argparse.Namespace) -> int:
    ffprobe = resolve_tool("ffprobe", args.ffprobe)
    ffmpeg = resolve_tool("ffmpeg", args.ffmpeg)
    results: list[dict[str, object]] = []
    for raw_path in args.paths:
        path = resolve_project_path(raw_path)
        probe = probe_audio(path, ffprobe)
        decoded = decode_smoke_test(
            path,
            ffmpeg,
            seconds=args.seconds,
            sample_rate_hz=args.sample_rate,
        )
        results.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "probe": probe.to_dict(),
                "canonical_decode": decoded.to_dict(),
            }
        )
    _json_print(
        {
            "ffprobe": {"path": str(ffprobe), "version": tool_version(ffprobe)},
            "ffmpeg": {"path": str(ffmpeg), "version": tool_version(ffmpeg)},
            "results": results,
        }
    )
    return 0


def _signal_smoke_command(args: argparse.Namespace) -> int:
    ffmpeg = resolve_tool("ffmpeg", args.ffmpeg)
    destination, result = run_signal_smoke_test(
        args.paths,
        args.output,
        ffmpeg=ffmpeg,
    )
    _json_print(
        {
            "result": str(destination),
            "passed": result["passed"],
            "recording_count": len(result["recordings"]),
            "feature_count": sum(
                recording["unique_feature_count"] for recording in result["recordings"]
            ),
        }
    )
    return 0 if result["passed"] else 1


def _build_known_clip_cache_command(args: argparse.Namespace) -> int:
    def show_progress(event: dict[str, object]) -> None:
        event_name = str(event["event"])
        if event_name == "preflight":
            required_gib = int(event["required_free_bytes"]) / (1024**3)
            available_gib = int(event["available_free_bytes"]) / (1024**3)
            print(
                "Cache preflight: "
                f"{event['recordings_completed']}/{event['recordings_total']} complete, "
                f"{required_gib:.2f} GiB required, {available_gib:.2f} GiB available",
                flush=True,
            )
            return
        if event_name == "recording_complete":
            if bool(event["resumed"]):
                return
            completed = int(event["recordings_completed"])
            total = int(event["recordings_total"])
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"Cached recordings: {completed}/{total}", flush=True)
            return
        if event_name == "published":
            print(
                f"Published cache: {event['recordings']} recordings, "
                f"{event['clips']} unique features",
                flush=True,
            )

    destination, summary = build_known_clip_cache(
        args.cache_root,
        ffmpeg=args.ffmpeg,
        config_path=args.config,
        manifest_path=args.manifest,
        split_path=args.split,
        split_summary_path=args.split_summary,
        split_lock_path=args.split_lock,
        review_lock_path=args.review_lock,
        progress_callback=show_progress,
    )
    _json_print(
        {
            "cache_root": str(destination),
            "recordings": summary["totals"]["recordings"],
            "clips": summary["totals"]["clips"],
            "feature_bytes": summary["totals"]["feature_bytes"],
        }
    )
    return 0


def _verify_known_clip_cache_command(args: argparse.Namespace) -> int:
    result = verify_known_clip_cache(
        args.cache_root,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=args.expected_lock_sha256,
    )
    _json_print(result)
    return 0 if result["valid"] else 1


def _audit_known_clip_cache_command(args: argparse.Namespace) -> int:
    result = audit_known_clip_cache(
        args.cache_root,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=args.expected_lock_sha256,
    )
    _json_print(result)
    return 0 if result["valid"] else 1


def _build_unknown_clip_cache_command(args: argparse.Namespace) -> int:
    def show_progress(event: dict[str, object]) -> None:
        event_name = str(event["event"])
        if event_name == "preflight":
            required_gib = int(event["required_free_bytes"]) / (1024**3)
            available_gib = int(event["available_free_bytes"]) / (1024**3)
            print(
                "Unknown cache preflight: "
                f"{event['recordings_completed']}/{event['recordings_total']} complete, "
                f"{required_gib:.2f} GiB required, {available_gib:.2f} GiB available",
                flush=True,
            )
            return
        if event_name == "recording_complete":
            if bool(event["resumed"]):
                return
            completed = int(event["recordings_completed"])
            total = int(event["recordings_total"])
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"Cached unknown recordings: {completed}/{total}", flush=True)
            return
        if event_name == "published":
            print(
                f"Published unknown cache: {event['recordings']} recordings, "
                f"{event['clips']} energy features",
                flush=True,
            )

    destination, summary = build_unknown_clip_cache(
        args.cache_root,
        ffmpeg=args.ffmpeg,
        audit_path=args.audit,
        audit_lock_path=args.audit_lock,
        checkpoint_root=args.checkpoint_root,
        config_path=args.config,
        unknown_audio_config_path=args.unknown_audio_config,
        progress_callback=show_progress,
    )
    _json_print(
        {
            "cache_root": str(destination),
            "recordings": summary["totals"]["recordings"],
            "clips": summary["totals"]["clips"],
            "feature_bytes": summary["totals"]["feature_bytes"],
            "species": summary["totals"]["species"],
        }
    )
    return 0


def _verify_unknown_clip_cache_command(args: argparse.Namespace) -> int:
    result = verify_unknown_clip_cache(
        args.cache_root,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=args.expected_lock_sha256,
    )
    _json_print(result)
    return 0 if result["valid"] else 1


def _show_config_command(args: argparse.Namespace) -> int:
    config = load_toml(args.path)
    _json_print(
        {
            "path": config["_config_path"],
            "sha256": config_fingerprint(config),
            "resolved": public_config(config),
        }
    )
    return 0


def _fetch_metadata_command(args: argparse.Namespace) -> int:
    api_key = api_key_from_environment(args.api_key_environment)
    destination, cache = fetch_metadata_cache(
        local_manifest_path=args.manifest,
        cache_path=args.cache,
        api_key=api_key,
        endpoint=args.endpoint,
        request_interval_seconds=args.request_interval,
        checkpoint_every=args.checkpoint_every,
        maximum_retries=args.maximum_retries,
        timeout_seconds=args.timeout,
    )
    status_counts: dict[str, int] = {}
    for entry in cache["records"].values():
        status = str(entry.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    _json_print({"cache": str(destination), "statuses": status_counts})
    return 1 if status_counts.get("error", 0) else 0


def _seal_metadata_command(args: argparse.Namespace) -> int:
    destination, lock_path, record = seal_metadata_cache(
        local_manifest_path=args.manifest,
        working_cache_path=args.working_cache,
        output_path=args.output,
        lock_path=args.lock,
    )
    _json_print(
        {
            "sealed_cache": str(destination),
            "lock": str(lock_path),
            **record,
        }
    )
    return 0


def _discover_unknown_metadata_command(args: argparse.Namespace) -> int:
    phase_labels = {
        "fetch": "Fetched",
        "resume_revalidation": "Revalidated cached",
        "completion_revalidation": "Revalidated final",
    }

    def show_progress(event: dict[str, object]) -> None:
        phase = str(event["phase"])
        label = phase_labels.get(phase, "Processed")
        print(
            f"{label} metadata: {event['scientific_name']} "
            f"page {event['page']}/{event['total_pages']}",
            flush=True,
        )

    try:
        destination, cache = fetch_unknown_metadata_cache(
            args.config,
            args.working_cache,
            progress_callback=show_progress,
        )
    except (
        UnknownAcquisitionConfigError,
        UnknownAcquisitionCredentialError,
        UnknownMetadataCacheError,
        XenoCantoApiError,
    ) as exc:
        print(f"ERROR: {format_unknown_acquisition_error(exc)}", file=sys.stderr)
        return 1
    species = {
        scientific_name: {
            "active": entry["active"],
            "role": entry["role"],
            "pages_fetched": len(entry["pages"]),
            "recordings_reported": (
                entry["snapshot"]["num_recordings"] if entry["snapshot"] is not None else 0
            ),
        }
        for scientific_name, entry in cache["species"].items()
    }
    _json_print(
        {
            "working_cache": str(destination),
            "complete": cache["complete"],
            "metadata_only": True,
            "species": species,
        }
    )
    return 0 if cache["complete"] else 1


def _seal_unknown_metadata_command(args: argparse.Namespace) -> int:
    destination, lock_path, lock = seal_unknown_metadata_cache(
        args.config,
        args.working_cache,
        args.output,
        args.lock,
    )
    _json_print(
        {
            "sealed_cache": str(destination),
            "lock": str(lock_path),
            "ready_for_candidate_planning": lock["ready_for_candidate_planning"],
            "species_count": lock["species_count"],
            "recordings_total": lock["recordings_total"],
            "sealed_cache_sha256": lock["sealed_cache_sha256"],
        }
    )
    return 0


def _validate_unknown_metadata_command(args: argparse.Namespace) -> int:
    result = verify_unknown_metadata_lock(args.lock, args.cache)
    _json_print(
        {
            "valid": True,
            "ready_for_candidate_planning": result["ready_for_candidate_planning"],
            "species_count": result["species_count"],
            "recordings_total": result["recordings_total"],
            "sealed_cache_sha256": result["sealed_cache_sha256"],
        }
    )
    return 0


def _build_unknown_candidate_plan_command(args: argparse.Namespace) -> int:
    destination, lock_path, result = build_unknown_candidate_plan(
        config_path=args.config,
        unknown_metadata_path=args.metadata,
        unknown_metadata_lock_path=args.metadata_lock,
        manifest_path=args.manifest,
        review_lock_path=args.review_lock,
        split_path=args.split,
        split_summary_path=args.split_summary,
        split_lock_path=args.split_lock,
        output_path=args.output,
        lock_path=args.lock,
    )
    _json_print(
        {
            "plan": str(destination),
            "lock": str(lock_path),
            **result,
        }
    )
    return 0 if result["valid"] else 1


def _validate_unknown_candidate_plan_command(args: argparse.Namespace) -> int:
    result = validate_unknown_candidate_plan(args.lock, args.plan)
    _json_print(result)
    return 0 if result["valid"] else 1


def _preflight_unknown_audio_command(args: argparse.Namespace) -> int:
    try:
        result = preflight_unknown_audio(args.config)
    except UnknownAudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pool_target = int(result["candidate_pool_target_recordings_per_species"])
    final_target = int(result["target_recordings_per_species"])
    species = {
        str(row["scientific_name"]): {
            "role": row["role"],
            "active": row["active"],
            "inventory_recordings": row["inventory_recordings"],
            "canonical_sessions_before_audio_qc": row["canonical_sessions_before_audio_qc"],
            "final_target_margin_before_audio_qc": (
                int(row["canonical_sessions_before_audio_qc"]) - final_target
            ),
            "candidate_pool_shortfall_before_audio_qc": max(
                0,
                pool_target - int(row["canonical_sessions_before_audio_qc"]),
            ),
            "estimated_download_duration_seconds": row["estimated_download_duration_seconds"],
            "estimated_download_bytes": row["estimated_download_bytes"],
            "fallback_status": row["fallback_status"],
        }
        for row in result["species"]
    }
    disk = result["disk"]
    payload = {
        "mode": "read_only_preflight",
        "network_requests": result["network_requests"],
        "audio_downloads": result["audio_downloads"],
        "fallback_active": result["fallback_active"],
        "candidate_pool_target_recordings_per_species": pool_target,
        "target_recordings_per_species": final_target,
        "candidate_recordings_total": sum(
            int(row["inventory_recordings"]) for row in result["species"]
        ),
        "active_canonical_sessions_before_audio_qc": sum(
            int(row["canonical_sessions_before_audio_qc"])
            for row in result["species"]
            if bool(row["active"])
        ),
        "estimated_active_download_bytes": result["estimated_active_download_bytes"],
        "estimated_fallback_contingency_bytes": result["estimated_fallback_contingency_bytes"],
        "estimated_download_bytes_with_fallback_contingency": result[
            "estimated_download_bytes_with_fallback_contingency"
        ],
        "estimated_required_disk_bytes": result["estimated_required_disk_bytes"],
        "available_disk_bytes": disk["available_bytes"],
        "estimated_space_sufficient": disk["estimated_space_sufficient"],
        "preflight_sha256": result["preflight_sha256"],
        "species": species,
    }
    _json_print(payload)
    return 0 if disk["estimated_space_sufficient"] is True else 1


_UNKNOWN_AUDIO_DISPOSITIONS = (
    "metadata_excluded",
    "session_noncanonical",
    "download_unavailable_terminal",
    "audio_qc_excluded",
    "eligible",
    "not_evaluated_pool_target_reached",
)


def _unknown_audio_error(exc: UnknownAudioError | FileNotFoundError) -> int:
    message = " ".join(str(exc).split()) or type(exc).__name__
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def _unknown_audio_download_policy(config: Mapping[str, object]) -> DownloadPolicy:
    download = config["download"]
    if not isinstance(download, Mapping):
        raise TypeError("unknown audio download config must be a mapping")
    return DownloadPolicy(
        allowed_hosts=tuple(download["allowed_initial_hosts"]),
        maximum_redirects=download["maximum_redirects"],
        request_interval_seconds=download["request_interval_seconds"],
        timeout_seconds=download["timeout_seconds"],
        total_timeout_seconds=download["total_timeout_seconds"],
        maximum_retries=download["maximum_retries"],
        chunk_size_bytes=download["chunk_size_bytes"],
        maximum_file_bytes=download["maximum_file_size_bytes"],
        maximum_retry_after_seconds=download["maximum_retry_after_seconds"],
        user_agent=download["user_agent"],
    )


def _compact_audit_path(value: object) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _compact_verified_unknown_audio(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "valid": result["valid"],
        "ready_for_unknown_scoring": result["ready_for_unknown_scoring"],
        "selected_recordings": result["selected_recordings"],
        "species_count": result["species"],
        "fallback_active": result["fallback_active"],
        "checkpoint_count": result["checkpoint_count"],
        "audit": _compact_audit_path(result["audit"]),
        "audit_sha256": result["audit_sha256"],
    }


def _compact_unknown_audio_species(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    species: dict[str, dict[str, object]] = {}
    for row in rows:
        dispositions = row["dispositions"]
        if not isinstance(dispositions, Mapping):
            raise TypeError("unknown audio dispositions must be a mapping")
        species[str(row["scientific_name"])] = {
            "role": row["role"],
            "inventory_recordings": row["inventory_recordings"],
            "terminal_recordings": row["terminal_recordings"],
            "eligible_recordings": row["eligible_recordings"],
            "unresolved_retryable": row["unresolved_retryable"],
            "completion_state": row["completion_state"],
            "dispositions": {
                name: dispositions[name]
                for name in _UNKNOWN_AUDIO_DISPOSITIONS
                if name in dispositions
            },
        }
    return species


def _compact_unknown_audio_replacement(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("unknown audio replacement must be a mapping")
    return {
        "replaced_scientific_name": value["replaced_scientific_name"],
        "replacement_scientific_name": value["replacement_scientific_name"],
    }


def _acquire_unknown_audio_command(args: argparse.Namespace) -> int:
    def show_progress(event: dict[str, object]) -> None:
        event_name = str(event["event"])
        if event_name == "candidate_terminal":
            terminal = int(event["terminal_recordings"])
            inventory = int(event["inventory_recordings"])
            if terminal != 1 and terminal % 10 != 0 and terminal != inventory:
                return
            print(
                f"Unknown audio {event['scientific_name']}: "
                f"{terminal}/{inventory} terminal, "
                f"{event['eligible_recordings']} eligible",
                flush=True,
            )
            return
        if event_name == "retryable_block":
            print(
                f"Unknown audio {event['scientific_name']}: "
                "paused with one retryable recording unresolved",
                flush=True,
            )
            return
        if event_name == "species_complete":
            state = str(event["completion_state"])
            if state not in {"pool_satisfied", "inventory_exhausted", "blocked_retryable"}:
                state = "unknown"
            print(
                f"Unknown audio {event['scientific_name']} complete: "
                f"{event['terminal_recordings']}/{event['inventory_recordings']} terminal, "
                f"{event['eligible_recordings']} eligible ({state})",
                flush=True,
            )

    try:
        config = load_unknown_audio_config(args.config)
        policy = _unknown_audio_download_policy(config)
        client = SecureXenoCantoAudioClient(policy)
        ffmpeg = resolve_tool("ffmpeg", args.ffmpeg)
        ffprobe = resolve_tool("ffprobe", args.ffprobe)
        result = run_unknown_audio_acquisition(
            client,
            config_path=args.config,
            ffprobe=ffprobe,
            ffmpeg=ffmpeg,
            progress_callback=show_progress,
        )
    except (UnknownAudioError, FileNotFoundError) as exc:
        return _unknown_audio_error(exc)

    if "valid" in result:
        verified = _compact_verified_unknown_audio(result)
        complete = result["valid"] is True and result["ready_for_unknown_scoring"] is True
        payload = {
            "complete": complete,
            "status": "complete" if complete else "verification_failed",
            **{key: value for key, value in verified.items() if key != "valid"},
        }
        _json_print(payload)
        return 0 if complete else 1

    gate = result["gate"]
    if not isinstance(gate, Mapping):
        raise TypeError("unknown audio gate must be a mapping")
    reason = gate.get("reason")
    if reason not in {None, "more_than_one_primary_below_40", "fallback_below_40"}:
        raise TypeError("unknown audio gate reason is invalid")
    raw_species = result["species"]
    if not isinstance(raw_species, Sequence) or isinstance(raw_species, (str, bytes)):
        raise TypeError("unknown audio species result must be a sequence")
    species = _compact_unknown_audio_species(raw_species)
    payload = {
        "complete": result["complete"],
        "status": result["status"],
        "ready_for_unknown_scoring": False,
        "checkpoint_count": result["checkpoint_count"],
        "selected_recordings": 0,
        "species_count": len(species),
        "fallback_active": gate["fallback_active"],
        "reason": reason,
        "failed_primary_species": gate["failed_primary_species"],
        "blocked_species": gate["blocked_species"],
        "replacement": _compact_unknown_audio_replacement(gate["replacement"]),
        "unresolved_retryable_total": sum(
            int(row["unresolved_retryable"]) for row in species.values()
        ),
        "species": species,
    }
    _json_print(payload)
    return 1


def _verify_unknown_audio_audit_command(args: argparse.Namespace) -> int:
    try:
        result = verify_unknown_audio_audit(args.config)
    except (UnknownAudioError, FileNotFoundError) as exc:
        return _unknown_audio_error(exc)
    payload = _compact_verified_unknown_audio(result)
    _json_print(payload)
    return 0 if result["valid"] is True else 1


def _enrich_metadata_command(args: argparse.Namespace) -> int:
    verify_metadata_cache_lock(args.cache_lock, args.cache)
    destination, summary = enrich_manifest_from_cache(
        config_path=args.config,
        local_manifest_path=args.local_manifest,
        cache_path=args.cache,
        output_path=args.output,
        licence_path=args.licences,
        summary_path=args.summary,
        overwrite=False,
    )
    lock_path, lock = create_enrichment_lock(
        config_path=args.config,
        local_manifest_path=args.local_manifest,
        sealed_cache_path=args.cache,
        cache_lock_path=args.cache_lock,
        enriched_manifest_path=args.output,
        licence_manifest_path=args.licences,
        summary_path=args.summary,
        lock_path=args.lock,
    )
    _json_print(
        {
            "manifest": str(destination),
            "enrichment_lock": str(lock_path),
            "metadata_cache_lock_sha256": lock["metadata_cache_lock_sha256"],
            **summary,
        }
    )
    return 0 if summary["ready_for_manual_review"] else 1


def _prepare_review_command(args: argparse.Namespace) -> int:
    preparation_path, preparation = prepare_manual_review(
        enriched_manifest_path=args.manifest,
        enrichment_lock_path=args.enrichment_lock,
        items_path=args.items,
        decisions_path=args.decisions,
        preparation_path=args.preparation,
    )
    _json_print(
        {
            "preparation": str(preparation_path),
            "items": args.items,
            "decisions_to_edit": args.decisions,
            **preparation,
        }
    )
    return 0


def _apply_review_command(args: argparse.Namespace) -> int:
    manifest_path, review_lock_path, resolution = apply_manual_review(
        enriched_manifest_path=args.manifest,
        enrichment_lock_path=args.enrichment_lock,
        items_path=args.items,
        decisions_path=args.decisions,
        preparation_path=args.preparation,
        final_manifest_path=args.output,
        resolution_path=args.resolution,
        lock_path=args.lock,
    )
    _json_print(
        {
            "manifest": str(manifest_path),
            "review_lock": str(review_lock_path),
            **resolution,
        }
    )
    return 0


def _validate_review_command(args: argparse.Namespace) -> int:
    result = verify_review_lock(args.lock, args.manifest)
    _json_print({"valid": True, **result})
    return 0


def _validate_configs_command(args: argparse.Namespace) -> int:
    counts = validate_project_config_set(args.data_config)
    _json_print({"valid": True, **counts})
    return 0


def _run_id_command(args: argparse.Namespace) -> int:
    print(
        make_run_id(
            task=args.task,
            rung=args.rung,
            seed=args.seed,
            config_hash=args.config_hash,
            data_hash=args.data_hash,
        )
    )
    return 0


def _lock_environment_command(args: argparse.Namespace) -> int:
    destination = write_dependency_lock(args.output)
    print(f"Dependency lock saved: {destination}")
    return 0


def _mps_smoke_command(args: argparse.Namespace) -> int:
    destination, result = run_mps_smoke_test(args.output, args.checkpoint)
    _json_print({"result": str(destination), **result})
    return 0 if result["passed"] else 1


def _freeze_split_command(args: argparse.Namespace) -> int:
    destination, diagnostics = freeze_grouped_split(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        summary_path=args.summary,
        lock_path=args.lock,
        review_lock_path=args.review_lock,
    )
    _json_print({"split": str(destination), **diagnostics})
    return 0


def _validate_split_command(args: argparse.Namespace) -> int:
    result = validate_frozen_split(
        args.manifest,
        args.split,
        args.lock,
        config_path=args.config,
        summary_path=args.summary,
        review_lock_path=args.review_lock,
    )
    _json_print(result)
    return 0 if result["valid"] else 1


def _command_argv(args: argparse.Namespace) -> tuple[str, ...]:
    value = getattr(args, "_command_argv", None)
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(part, str) for part in value)
    ):
        raise RuntimeError("The exact command argument vector is unavailable")
    return value


def _resume_arguments(args: argparse.Namespace) -> tuple[str | None, str | None]:
    checkpoint = args.resume_checkpoint
    checkpoint_sha256 = args.resume_checkpoint_sha256
    if (checkpoint is None) != (checkpoint_sha256 is None):
        raise ValueError("Resume checkpoint and SHA-256 must be supplied together")
    return checkpoint, checkpoint_sha256


def _preflight_task1_weights_command(args: argparse.Namespace) -> int:
    artifact = preflight_efficientnet_weights(populate=args.populate)
    _json_print(
        {
            "identifier": artifact.identifier,
            "path": str(artifact.path),
            "populated": args.populate,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "verified": True,
        }
    )
    return 0


def _benchmark_task1_command(args: argparse.Namespace) -> int:
    result = benchmark_task1_full_epoch(
        seed=13,
        cache_root=TASK1_KNOWN_CACHE_ROOT,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=TASK1_KNOWN_CACHE_LOCK_SHA256,
        command=_command_argv(args),
    )
    _json_print(result)
    return 0


def _train_task1_command(args: argparse.Namespace) -> int:
    resume_checkpoint, resume_checkpoint_sha256 = _resume_arguments(args)
    result = run_task1_development(
        seed=args.seed,
        cache_root=TASK1_KNOWN_CACHE_ROOT,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=TASK1_KNOWN_CACHE_LOCK_SHA256,
        output_root=TASK1_RUN_ROOT,
        command=_command_argv(args),
        resume_checkpoint=resume_checkpoint,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
    )
    _json_print(result)
    return 0


def _benchmark_task2_command(args: argparse.Namespace) -> int:
    result = benchmark_task2_full_epoch(
        seed=13,
        cache_root=TASK2_KNOWN_CACHE_ROOT,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=TASK2_KNOWN_CACHE_LOCK_SHA256,
        command=_command_argv(args),
    )
    _json_print(result)
    return 0


def _train_task2_command(args: argparse.Namespace) -> int:
    resume_checkpoint, resume_checkpoint_sha256 = _resume_arguments(args)
    result = run_task2_development(
        seed=args.seed,
        cache_root=TASK2_KNOWN_CACHE_ROOT,
        ffmpeg=args.ffmpeg,
        expected_lock_sha256=TASK2_KNOWN_CACHE_LOCK_SHA256,
        output_root=TASK2_RUN_ROOT,
        command=_command_argv(args),
        resume_checkpoint=resume_checkpoint,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
    )
    _json_print(result)
    return 0


def _seal_final_evaluation_gate_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(seal_final_evaluation_gate())
    return 0


def _run_final_evaluation_command(args: argparse.Namespace) -> int:
    result = run_final_evaluation(
        ffmpeg=args.ffmpeg,
        command=_command_argv(args),
    )
    _json_print(result)
    return 0


def _verify_final_evaluation_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(verify_final_evaluation())
    return 0


def _verify_v1_recovery_manifest_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(verify_v1_recovery_manifest())
    return 0


def _seal_v2_cache_equivalence_command(args: argparse.Namespace) -> int:
    _json_print(seal_unknown_cache_v2_equivalence(ffmpeg=args.ffmpeg))
    return 0


def _verify_v2_cache_equivalence_command(args: argparse.Namespace) -> int:
    _json_print(
        verify_unknown_cache_v2_equivalence_certificate(
            ffmpeg=args.ffmpeg,
            full_rederivation=args.full_rederivation,
        )
    )
    return 0


def _build_final_report_assets_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(build_final_report_assets())
    return 0


def _verify_final_report_assets_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(verify_final_report_assets())
    return 0


def _build_task1_attributions_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(build_task1_attributions())
    return 0


def _verify_task1_attributions_command(args: argparse.Namespace) -> int:
    _ = args
    _json_print(verify_task1_attributions())
    return 0


class _ProductionArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        command_argv = tuple(sys.argv if args is None else args)
        parsed = super().parse_args(None if args is None else command_argv, namespace)
        if parsed.command in {"train-task1", "train-task2"} and (
            (parsed.resume_checkpoint is None) != (parsed.resume_checkpoint_sha256 is None)
        ):
            self.error("resume checkpoint and SHA-256 must be supplied together")
        parsed._command_argv = command_argv
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _ProductionArgumentParser(
        prog="bird-audio",
        description="Reproducible bird-audio coursework commands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("environment", help="Capture environment provenance")
    environment.add_argument(
        "--output",
        default=str(DEFAULT_ENVIRONMENT_V2_PATH.relative_to(PROJECT_ROOT)),
    )
    environment.add_argument("--ffmpeg")
    environment.add_argument("--ffprobe")
    environment.set_defaults(handler=_environment_command)

    build_manifest = subparsers.add_parser(
        "build-local-manifest",
        help="Hash and probe every immutable raw recording",
    )
    build_manifest.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    build_manifest.add_argument("--output", default=DEFAULT_LOCAL_MANIFEST)
    build_manifest.add_argument("--ffprobe")
    build_manifest.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    build_manifest.add_argument("--overwrite", action="store_true")
    build_manifest.set_defaults(handler=_build_manifest_command)

    validate_manifest = subparsers.add_parser(
        "validate-local-manifest",
        help="Validate structure and summarize the local manifest",
    )
    validate_manifest.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    validate_manifest.add_argument("--manifest", default=DEFAULT_LOCAL_MANIFEST)
    validate_manifest.add_argument("--summary", default=DEFAULT_LOCAL_SUMMARY)
    validate_manifest.add_argument("--verify-hashes", action="store_true")
    validate_manifest.set_defaults(handler=_validate_manifest_command)

    audit_decode = subparsers.add_parser(
        "audit-full-decode",
        help="Decode every raw recording and persist failures or warnings",
    )
    audit_decode.add_argument("--manifest", default=DEFAULT_LOCAL_MANIFEST)
    audit_decode.add_argument("--output", default=DEFAULT_LOCAL_MANIFEST)
    audit_decode.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    audit_decode.add_argument("--ffmpeg")
    audit_decode.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    audit_decode.add_argument("--overwrite", action="store_true")
    audit_decode.set_defaults(handler=_audit_decode_command)

    probe_smoke = subparsers.add_parser(
        "probe-smoke",
        help="Verify canonical decoding on selected raw files",
    )
    probe_smoke.add_argument("paths", nargs="+")
    probe_smoke.add_argument("--ffmpeg")
    probe_smoke.add_argument("--ffprobe")
    probe_smoke.add_argument("--seconds", type=float, default=3.0)
    probe_smoke.add_argument("--sample-rate", type=int, default=32000)
    probe_smoke.set_defaults(handler=_probe_smoke_command)

    signal_smoke = subparsers.add_parser(
        "signal-smoke",
        help="Run the locked signal transform on selected immutable raw recordings",
    )
    signal_smoke.add_argument("paths", nargs="+")
    signal_smoke.add_argument("--ffmpeg")
    signal_smoke.add_argument(
        "--output",
        default=str(DEFAULT_SIGNAL_SMOKE_V2_PATH.relative_to(PROJECT_ROOT)),
    )
    signal_smoke.set_defaults(handler=_signal_smoke_command)

    build_clip_cache = subparsers.add_parser(
        "build-known-clip-cache",
        help="Resume and publish the locked known-species native feature cache",
    )
    build_clip_cache.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    build_clip_cache.add_argument("--ffmpeg")
    build_clip_cache.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    build_clip_cache.add_argument("--manifest", default=DEFAULT_FINAL_MANIFEST)
    build_clip_cache.add_argument("--review-lock", default=DEFAULT_REVIEW_LOCK)
    build_clip_cache.add_argument("--split", default=DEFAULT_SPLIT)
    build_clip_cache.add_argument("--split-summary", default=DEFAULT_SPLIT_SUMMARY)
    build_clip_cache.add_argument("--split-lock", default=DEFAULT_SPLIT_LOCK)
    build_clip_cache.set_defaults(handler=_build_known_clip_cache_command)

    verify_clip_cache = subparsers.add_parser(
        "verify-known-clip-cache",
        help="Verify every cache artifact without exposing final-test examples",
    )
    verify_clip_cache.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    verify_clip_cache.add_argument("--ffmpeg")
    verify_clip_cache.add_argument("--expected-lock-sha256")
    verify_clip_cache.set_defaults(handler=_verify_known_clip_cache_command)

    audit_clip_cache = subparsers.add_parser(
        "audit-known-clip-cache",
        help="Independently bind the known cache to the frozen split and selection rules",
    )
    audit_clip_cache.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    audit_clip_cache.add_argument("--ffmpeg")
    audit_clip_cache.add_argument("--expected-lock-sha256")
    audit_clip_cache.set_defaults(handler=_audit_known_clip_cache_command)

    build_unknown_cache = subparsers.add_parser(
        "build-unknown-clip-cache",
        help="Build the energy-only scoring cache from the sealed unknown selection",
    )
    build_unknown_cache.add_argument("--cache-root", default=DEFAULT_UNKNOWN_CLIP_CACHE_ROOT)
    build_unknown_cache.add_argument("--ffmpeg")
    build_unknown_cache.add_argument("--audit", default=DEFAULT_UNKNOWN_AUDIO_AUDIT)
    build_unknown_cache.add_argument("--audit-lock", default=DEFAULT_UNKNOWN_AUDIO_AUDIT_LOCK)
    build_unknown_cache.add_argument(
        "--checkpoint-root", default=DEFAULT_UNKNOWN_AUDIO_CHECKPOINT_ROOT
    )
    build_unknown_cache.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    build_unknown_cache.add_argument("--unknown-audio-config", default=DEFAULT_UNKNOWN_AUDIO_CONFIG)
    build_unknown_cache.set_defaults(handler=_build_unknown_clip_cache_command)

    verify_unknown_cache = subparsers.add_parser(
        "verify-unknown-clip-cache",
        help="Verify the scoring-only unknown cache and all sealed source bindings",
    )
    verify_unknown_cache.add_argument("--cache-root", default=DEFAULT_UNKNOWN_CLIP_CACHE_ROOT)
    verify_unknown_cache.add_argument("--ffmpeg")
    verify_unknown_cache.add_argument("--expected-lock-sha256")
    verify_unknown_cache.set_defaults(handler=_verify_unknown_clip_cache_command)

    show_config = subparsers.add_parser("show-config", help="Resolve and hash a TOML config")
    show_config.add_argument("path", type=Path)
    show_config.set_defaults(handler=_show_config_command)

    validate_configs = subparsers.add_parser(
        "validate-configs",
        help="Validate every data and experiment configuration",
    )
    validate_configs.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    validate_configs.set_defaults(handler=_validate_configs_command)

    fetch_metadata = subparsers.add_parser(
        "fetch-metadata",
        help="Resume Xeno-canto metadata retrieval without storing the API key",
    )
    fetch_metadata.add_argument("--manifest", default=DEFAULT_LOCAL_MANIFEST)
    fetch_metadata.add_argument("--cache", default=DEFAULT_WORKING_METADATA_CACHE)
    fetch_metadata.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    fetch_metadata.add_argument("--api-key-environment", default="XENO_CANTO_API_KEY")
    fetch_metadata.add_argument("--request-interval", type=float, default=1.0)
    fetch_metadata.add_argument("--checkpoint-every", type=int, default=20)
    fetch_metadata.add_argument("--maximum-retries", type=int, default=5)
    fetch_metadata.add_argument("--timeout", type=float, default=30)
    fetch_metadata.set_defaults(handler=_fetch_metadata_command)

    seal_metadata = subparsers.add_parser(
        "seal-metadata-cache",
        help="Seal a complete resumable metadata cache and lock every source hash",
    )
    seal_metadata.add_argument("--manifest", default=DEFAULT_LOCAL_MANIFEST)
    seal_metadata.add_argument("--working-cache", default=DEFAULT_WORKING_METADATA_CACHE)
    seal_metadata.add_argument("--output", default=DEFAULT_SEALED_METADATA_CACHE)
    seal_metadata.add_argument("--lock", default=DEFAULT_METADATA_CACHE_LOCK)
    seal_metadata.set_defaults(handler=_seal_metadata_command)

    discover_unknown_metadata = subparsers.add_parser(
        "discover-unknown-metadata",
        help="Fetch and resume metadata-only discovery for the locked unknown species",
    )
    discover_unknown_metadata.add_argument("--config", default=DEFAULT_UNKNOWN_ACQUISITION_CONFIG)
    discover_unknown_metadata.add_argument(
        "--working-cache", default=DEFAULT_UNKNOWN_WORKING_METADATA
    )
    discover_unknown_metadata.set_defaults(handler=_discover_unknown_metadata_command)

    seal_unknown_metadata = subparsers.add_parser(
        "seal-unknown-metadata",
        help="Seal the complete unknown-species metadata snapshot and provenance lock",
    )
    seal_unknown_metadata.add_argument("--config", default=DEFAULT_UNKNOWN_ACQUISITION_CONFIG)
    seal_unknown_metadata.add_argument("--working-cache", default=DEFAULT_UNKNOWN_WORKING_METADATA)
    seal_unknown_metadata.add_argument("--output", default=DEFAULT_UNKNOWN_SEALED_METADATA)
    seal_unknown_metadata.add_argument("--lock", default=DEFAULT_UNKNOWN_METADATA_LOCK)
    seal_unknown_metadata.set_defaults(handler=_seal_unknown_metadata_command)

    validate_unknown_metadata = subparsers.add_parser(
        "validate-unknown-metadata",
        help="Verify the sealed unknown-species metadata snapshot and provenance lock",
    )
    validate_unknown_metadata.add_argument("--cache", default=DEFAULT_UNKNOWN_SEALED_METADATA)
    validate_unknown_metadata.add_argument("--lock", default=DEFAULT_UNKNOWN_METADATA_LOCK)
    validate_unknown_metadata.set_defaults(handler=_validate_unknown_metadata_command)

    build_unknown_plan = subparsers.add_parser(
        "build-unknown-candidate-plan",
        help="Build and lock the metadata-only unknown candidate and reference plan",
    )
    build_unknown_plan.add_argument("--config", default=DEFAULT_UNKNOWN_SELECTION_CONFIG)
    build_unknown_plan.add_argument("--metadata", default=DEFAULT_UNKNOWN_SEALED_METADATA)
    build_unknown_plan.add_argument("--metadata-lock", default=DEFAULT_UNKNOWN_METADATA_LOCK)
    build_unknown_plan.add_argument("--manifest", default=DEFAULT_FINAL_MANIFEST)
    build_unknown_plan.add_argument("--review-lock", default=DEFAULT_REVIEW_LOCK)
    build_unknown_plan.add_argument("--split", default=DEFAULT_SPLIT)
    build_unknown_plan.add_argument("--split-summary", default=DEFAULT_SPLIT_SUMMARY)
    build_unknown_plan.add_argument("--split-lock", default=DEFAULT_SPLIT_LOCK)
    build_unknown_plan.add_argument("--output", default=DEFAULT_UNKNOWN_CANDIDATE_PLAN)
    build_unknown_plan.add_argument("--lock", default=DEFAULT_UNKNOWN_CANDIDATE_PLAN_LOCK)
    build_unknown_plan.set_defaults(handler=_build_unknown_candidate_plan_command)

    validate_unknown_plan = subparsers.add_parser(
        "validate-unknown-candidate-plan",
        help="Verify the locked unknown candidate and known-reference plan",
    )
    validate_unknown_plan.add_argument("--plan", default=DEFAULT_UNKNOWN_CANDIDATE_PLAN)
    validate_unknown_plan.add_argument("--lock", default=DEFAULT_UNKNOWN_CANDIDATE_PLAN_LOCK)
    validate_unknown_plan.set_defaults(handler=_validate_unknown_candidate_plan_command)

    preflight_unknown = subparsers.add_parser(
        "preflight-unknown-audio",
        help="Run a read-only unknown-audio preflight with no network or audio download",
    )
    preflight_unknown.add_argument("--config", default=DEFAULT_UNKNOWN_AUDIO_CONFIG)
    preflight_unknown.set_defaults(handler=_preflight_unknown_audio_command)

    acquire_unknown = subparsers.add_parser(
        "acquire-unknown-audio",
        help="Resume locked unknown-audio acquisition and publish the verified audit",
    )
    acquire_unknown.add_argument("--config", default=DEFAULT_UNKNOWN_AUDIO_CONFIG)
    acquire_unknown.add_argument("--ffmpeg")
    acquire_unknown.add_argument("--ffprobe")
    acquire_unknown.set_defaults(handler=_acquire_unknown_audio_command)

    verify_unknown_audio = subparsers.add_parser(
        "verify-unknown-audio-audit",
        help="Rebuild and verify the unknown-audio audit from locked artifacts",
    )
    verify_unknown_audio.add_argument("--config", default=DEFAULT_UNKNOWN_AUDIO_CONFIG)
    verify_unknown_audio.set_defaults(handler=_verify_unknown_audio_audit_command)

    enrich_metadata = subparsers.add_parser(
        "enrich-metadata",
        help="Join cached metadata and apply deterministic cleaning rules",
    )
    enrich_metadata.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    enrich_metadata.add_argument("--local-manifest", default=DEFAULT_LOCAL_MANIFEST)
    enrich_metadata.add_argument("--cache", default=DEFAULT_SEALED_METADATA_CACHE)
    enrich_metadata.add_argument("--cache-lock", default=DEFAULT_METADATA_CACHE_LOCK)
    enrich_metadata.add_argument("--output", default=DEFAULT_ENRICHED_BASE_MANIFEST)
    enrich_metadata.add_argument("--licences", default=DEFAULT_LICENCE_MANIFEST)
    enrich_metadata.add_argument("--summary", default=DEFAULT_METADATA_SUMMARY)
    enrich_metadata.add_argument("--lock", default=DEFAULT_ENRICHMENT_LOCK)
    enrich_metadata.set_defaults(handler=_enrich_metadata_command)

    prepare_review = subparsers.add_parser(
        "prepare-manual-review",
        help="Create immutable review items and a separate editable decision file",
    )
    prepare_review.add_argument("--manifest", default=DEFAULT_ENRICHED_BASE_MANIFEST)
    prepare_review.add_argument("--enrichment-lock", default=DEFAULT_ENRICHMENT_LOCK)
    prepare_review.add_argument("--items", default=DEFAULT_REVIEW_ITEMS)
    prepare_review.add_argument("--decisions", default=DEFAULT_REVIEW_DECISIONS)
    prepare_review.add_argument("--preparation", default=DEFAULT_REVIEW_PREPARATION)
    prepare_review.set_defaults(handler=_prepare_review_command)

    apply_review = subparsers.add_parser(
        "apply-manual-review",
        help="Validate every decision and lock the final recording manifest",
    )
    apply_review.add_argument("--manifest", default=DEFAULT_ENRICHED_BASE_MANIFEST)
    apply_review.add_argument("--enrichment-lock", default=DEFAULT_ENRICHMENT_LOCK)
    apply_review.add_argument("--items", default=DEFAULT_REVIEW_ITEMS)
    apply_review.add_argument("--decisions", default=DEFAULT_REVIEW_DECISIONS)
    apply_review.add_argument("--preparation", default=DEFAULT_REVIEW_PREPARATION)
    apply_review.add_argument("--output", default=DEFAULT_FINAL_MANIFEST)
    apply_review.add_argument("--resolution", default=DEFAULT_REVIEW_RESOLUTION)
    apply_review.add_argument("--lock", default=DEFAULT_REVIEW_LOCK)
    apply_review.set_defaults(handler=_apply_review_command)

    validate_review = subparsers.add_parser(
        "validate-manual-review",
        help="Verify the complete review provenance chain and final manifest",
    )
    validate_review.add_argument("--manifest", default=DEFAULT_FINAL_MANIFEST)
    validate_review.add_argument("--lock", default=DEFAULT_REVIEW_LOCK)
    validate_review.set_defaults(handler=_validate_review_command)

    run_id = subparsers.add_parser("run-id", help="Create a reproducible run identifier")
    run_id.add_argument("--task", required=True)
    run_id.add_argument("--rung", required=True)
    run_id.add_argument("--seed", required=True, type=int)
    run_id.add_argument("--config-hash", required=True)
    run_id.add_argument("--data-hash", required=True)
    run_id.set_defaults(handler=_run_id_command)

    lock_environment = subparsers.add_parser(
        "lock-environment",
        help="Freeze every installed external Python dependency",
    )
    lock_environment.add_argument("--output", default="requirements.lock")
    lock_environment.set_defaults(handler=_lock_environment_command)

    mps_smoke = subparsers.add_parser(
        "mps-smoke",
        help="Run a deterministic MPS forward, backward, update, and checkpoint test",
    )
    mps_smoke.add_argument(
        "--output",
        default=str(DEFAULT_MPS_SMOKE_V2_PATH.relative_to(PROJECT_ROOT)),
    )
    mps_smoke.add_argument(
        "--checkpoint",
        default=str(DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH.relative_to(PROJECT_ROOT)),
    )
    mps_smoke.set_defaults(handler=_mps_smoke_command)

    freeze_split = subparsers.add_parser(
        "freeze-split",
        help="Create and permanently lock the grouped recording split",
    )
    freeze_split.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    freeze_split.add_argument("--manifest", default=DEFAULT_FINAL_MANIFEST)
    freeze_split.add_argument("--review-lock", default=DEFAULT_REVIEW_LOCK)
    freeze_split.add_argument("--output", default=DEFAULT_SPLIT)
    freeze_split.add_argument("--summary", default=DEFAULT_SPLIT_SUMMARY)
    freeze_split.add_argument("--lock", default=DEFAULT_SPLIT_LOCK)
    freeze_split.set_defaults(handler=_freeze_split_command)

    validate_split = subparsers.add_parser(
        "validate-split",
        help="Verify the locked split and every overlap invariant",
    )
    validate_split.add_argument("--manifest", default=DEFAULT_FINAL_MANIFEST)
    validate_split.add_argument("--review-lock", default=DEFAULT_REVIEW_LOCK)
    validate_split.add_argument("--split", default=DEFAULT_SPLIT)
    validate_split.add_argument("--lock", default=DEFAULT_SPLIT_LOCK)
    validate_split.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    validate_split.add_argument("--summary", default=DEFAULT_SPLIT_SUMMARY)
    validate_split.set_defaults(handler=_validate_split_command)

    preflight_task1_weights = subparsers.add_parser(
        "preflight-task1-weights",
        help="Verify or explicitly populate the locked Task 1 weight artifact",
    )
    preflight_task1_weights.add_argument("--populate", action="store_true")
    preflight_task1_weights.set_defaults(handler=_preflight_task1_weights_command)

    benchmark_task1 = subparsers.add_parser(
        "benchmark-task1",
        help="Measure one locked Task 1 development epoch on MPS",
    )
    benchmark_task1.add_argument("--ffmpeg")
    benchmark_task1.set_defaults(handler=_benchmark_task1_command)

    train_task1 = subparsers.add_parser(
        "train-task1",
        help="Run one locked Task 1 development seed on MPS",
    )
    train_task1.add_argument("--seed", type=int, choices=(13, 37, 71), required=True)
    train_task1.add_argument("--ffmpeg")
    train_task1.add_argument("--resume-checkpoint")
    train_task1.add_argument("--resume-checkpoint-sha256")
    train_task1.set_defaults(handler=_train_task1_command)

    benchmark_task2 = subparsers.add_parser(
        "benchmark-task2",
        help="Measure one locked Task 2 development epoch on MPS",
    )
    benchmark_task2.add_argument("--ffmpeg")
    benchmark_task2.set_defaults(handler=_benchmark_task2_command)

    train_task2 = subparsers.add_parser(
        "train-task2",
        help="Run one locked Task 2 development seed on MPS",
    )
    train_task2.add_argument("--seed", type=int, choices=(13, 37, 71), required=True)
    train_task2.add_argument("--ffmpeg")
    train_task2.add_argument("--resume-checkpoint")
    train_task2.add_argument("--resume-checkpoint-sha256")
    train_task2.set_defaults(handler=_train_task2_command)

    seal_final_gate = subparsers.add_parser(
        "seal-final-evaluation-gate",
        help="Seal the exact six completed development runs for final evaluation",
    )
    seal_final_gate.set_defaults(handler=_seal_final_evaluation_gate_command)

    run_final = subparsers.add_parser(
        "run-final-evaluation",
        help="Run or resume the one sealed final evaluation attempt",
    )
    run_final.add_argument("--ffmpeg")
    run_final.set_defaults(handler=_run_final_evaluation_command)

    verify_final = subparsers.add_parser(
        "verify-final-evaluation",
        help="Recursively verify the completed final evaluation without inference",
    )
    verify_final.set_defaults(handler=_verify_final_evaluation_command)

    verify_v1_recovery = subparsers.add_parser(
        "verify-v1-recovery-manifest",
        help="Verify the preserved v1 source, caches, runs, gate, claim, and failure record",
    )
    verify_v1_recovery.set_defaults(handler=_verify_v1_recovery_manifest_command)

    seal_v2_equivalence = subparsers.add_parser(
        "seal-v2-cache-equivalence",
        help="Rederive and seal the scientific equivalence of unknown cache v1 and v2",
    )
    seal_v2_equivalence.add_argument("--ffmpeg")
    seal_v2_equivalence.set_defaults(handler=_seal_v2_cache_equivalence_command)

    verify_v2_equivalence = subparsers.add_parser(
        "verify-v2-cache-equivalence",
        help="Verify the sealed v1 to v2 unknown-cache equivalence certificate",
    )
    verify_v2_equivalence.add_argument("--ffmpeg")
    verify_v2_equivalence.add_argument("--full-rederivation", action="store_true")
    verify_v2_equivalence.set_defaults(handler=_verify_v2_cache_equivalence_command)

    build_report_assets = subparsers.add_parser(
        "build-final-report-assets",
        help="Build the fixed tables and figures from verified final evidence",
    )
    build_report_assets.set_defaults(handler=_build_final_report_assets_command)

    verify_report_assets = subparsers.add_parser(
        "verify-final-report-assets",
        help="Recursively verify the fixed final report evidence set",
    )
    verify_report_assets.set_defaults(handler=_verify_final_report_assets_command)

    build_attributions = subparsers.add_parser(
        "build-task1-attributions",
        help="Build the fixed seed 37 Task 1 Grad-CAM evidence",
    )
    build_attributions.set_defaults(handler=_build_task1_attributions_command)

    verify_attributions = subparsers.add_parser(
        "verify-task1-attributions",
        help="Verify the fixed Task 1 attribution evidence without inference",
    )
    verify_attributions.set_defaults(handler=_verify_task1_attributions_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.handler(args))
