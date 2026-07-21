from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.config import config_fingerprint, load_toml
from bird_audio.hashing import sha256_bytes, sha256_file, sha256_json
from bird_audio.io_utils import (
    atomic_write_json,
    read_csv_snapshot,
    require_unchanged,
)
from bird_audio.locking import project_lock
from bird_audio.metadata import (
    API_VERSION,
    DEFAULT_ENDPOINT,
    PERSISTED_RECORDING_FIELDS,
)
from bird_audio.paths import (
    PROJECT_ROOT,
    is_relative_to,
    require_safe_output,
    resolve_project_path,
)

METADATA_CACHE_LOCK_SCHEMA_VERSION = "1.0"
ENRICHMENT_LOCK_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value, digest


def _recording_id(recording: dict[str, Any]) -> str:
    return str(recording.get("id") or recording.get("nr") or "").removeprefix("XC")


def _validate_complete_cache(
    cache: dict[str, Any],
    local_rows: list[dict[str, str]],
    local_manifest_sha256: str,
) -> None:
    if cache.get("schema_version") != "1.1":
        raise ValueError("Metadata cache schema is not supported")
    if cache.get("api_version") != API_VERSION:
        raise ValueError("Metadata cache API version is not supported")
    if cache.get("endpoint") != DEFAULT_ENDPOINT:
        raise ValueError("Metadata cache endpoint is not approved")
    if cache.get("query_form") != "nr:<xc_id>":
        raise ValueError("Metadata cache query form is not approved")
    if cache.get("source_manifest_sha256") != local_manifest_sha256:
        raise ValueError("Metadata cache is not bound to the exact local manifest")

    expected_ids = sorted({row["xc_id"] for row in local_rows}, key=int)
    if cache.get("source_recording_ids_sha256") != sha256_json(expected_ids):
        raise ValueError("Metadata cache recording-set hash is invalid")
    records = cache.get("records")
    if not isinstance(records, dict) or set(records) != set(expected_ids):
        raise ValueError("Metadata cache must contain every expected recording exactly once")
    for xc_id in expected_ids:
        entry = records[xc_id]
        if not isinstance(entry, dict) or entry.get("status") not in {
            "ok",
            "unavailable",
        }:
            raise ValueError(f"Metadata cache is incomplete at XC{xc_id}")
        recording = entry.get("recording")
        if entry.get("status") == "unavailable":
            if recording not in ({}, None) or not str(entry.get("error") or "").strip():
                raise ValueError(f"Unavailable cache entry is invalid at XC{xc_id}")
            continue
        if not isinstance(recording, dict) or _recording_id(recording) != xc_id:
            raise ValueError(f"Metadata cache identity mismatch at XC{xc_id}")
        if not set(recording).issubset(PERSISTED_RECORDING_FIELDS):
            raise ValueError(f"Metadata cache has unapproved fields at XC{xc_id}")


def seal_metadata_cache(
    local_manifest_path: str | Path,
    working_cache_path: str | Path,
    output_path: str | Path,
    lock_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Seal a complete resumable cache into an immutable, hash-bound artifact."""
    local_manifest = resolve_project_path(local_manifest_path)
    working_cache = resolve_project_path(working_cache_path)
    destination = require_safe_output(output_path)
    lock_destination = require_safe_output(lock_path)
    for path in (destination, lock_destination):
        if path.exists():
            raise FileExistsError(f"Sealed metadata output already exists: {path}")

    with project_lock("metadata_cache"):
        local_rows, local_manifest_sha256 = read_csv_snapshot(local_manifest)
        cache, working_cache_sha256 = _read_json_snapshot(working_cache)
        _validate_complete_cache(cache, local_rows, local_manifest_sha256)

        sealed_cache = json.loads(json.dumps(cache, ensure_ascii=True))
        sealed_cache.update(
            {
                "sealed": True,
                "sealed_at_utc": _utc_now(),
                "source_working_cache_sha256": working_cache_sha256,
            }
        )
        require_unchanged(local_manifest, local_manifest_sha256)
        require_unchanged(working_cache, working_cache_sha256)
        destination = atomic_write_json(destination, sealed_cache)
        sealed_cache_sha256 = sha256_file(destination)
        lock = {
            "schema_version": METADATA_CACHE_LOCK_SCHEMA_VERSION,
            "locked_at_utc": _utc_now(),
            "ready_for_enrichment": True,
            "api_version": API_VERSION,
            "endpoint": DEFAULT_ENDPOINT,
            "recordings": len(local_rows),
            "recording_set_sha256": sha256_json(sorted(row["recording_id"] for row in local_rows)),
            "source_local_manifest_sha256": local_manifest_sha256,
            "source_working_cache_sha256": working_cache_sha256,
            "sealed_cache_sha256": sealed_cache_sha256,
            "artifacts": {
                "local_manifest": {
                    "path": _project_relative(local_manifest),
                    "sha256": local_manifest_sha256,
                },
                "working_cache": {
                    "path": _project_relative(working_cache),
                    "sha256": working_cache_sha256,
                },
                "sealed_cache": {
                    "path": _project_relative(destination),
                    "sha256": sealed_cache_sha256,
                },
            },
        }
        require_unchanged(local_manifest, local_manifest_sha256)
        require_unchanged(working_cache, working_cache_sha256)
        lock_destination = atomic_write_json(lock_destination, lock)
    return destination, lock_destination, lock


def _verify_artifact_table(lock: dict[str, Any], required: set[str]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("Artifact lock is missing required entries")
    resolved: dict[str, Path] = {}
    for name in required:
        entry = artifacts[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Artifact lock entry is invalid: {name}")
        path_value = str(entry.get("path") or "")
        if not path_value or Path(path_value).is_absolute():
            raise ValueError(f"Artifact lock path is invalid: {name}")
        path = resolve_project_path(path_value)
        if not is_relative_to(path, PROJECT_ROOT):
            raise ValueError(f"Artifact lock path leaves the project: {name}")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Artifact lock hash check failed: {name}")
        resolved[name] = path
    return resolved


def verify_metadata_cache_lock(
    lock_path: str | Path,
    expected_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    lock_file = resolve_project_path(lock_path)
    lock, _ = _read_json_snapshot(lock_file)
    if lock.get("schema_version") != METADATA_CACHE_LOCK_SCHEMA_VERSION:
        raise ValueError("Metadata cache lock schema is not supported")
    if lock.get("api_version") != API_VERSION or lock.get("endpoint") != DEFAULT_ENDPOINT:
        raise ValueError("Metadata cache lock API contract is invalid")
    if lock.get("ready_for_enrichment") is not True:
        raise ValueError("Metadata cache lock is not ready for enrichment")
    paths = _verify_artifact_table(lock, {"local_manifest", "sealed_cache"})
    if expected_cache_path is not None and paths["sealed_cache"] != resolve_project_path(
        expected_cache_path
    ):
        raise ValueError("Metadata cache lock points to a different sealed cache")
    local_rows, local_sha256 = read_csv_snapshot(paths["local_manifest"])
    cache, cache_sha256 = _read_json_snapshot(paths["sealed_cache"])
    _validate_complete_cache(cache, local_rows, local_sha256)
    if not cache.get("sealed"):
        raise ValueError("Metadata cache payload is not sealed")
    if cache.get("source_working_cache_sha256") != lock.get("source_working_cache_sha256"):
        raise ValueError("Metadata cache lock has an inconsistent working-cache hash")
    if lock.get("sealed_cache_sha256") != cache_sha256:
        raise ValueError("Metadata cache lock has an inconsistent cache hash")
    if lock.get("source_local_manifest_sha256") != local_sha256:
        raise ValueError("Metadata cache lock has an inconsistent manifest hash")
    return lock


def create_enrichment_lock(
    config_path: str | Path,
    local_manifest_path: str | Path,
    sealed_cache_path: str | Path,
    cache_lock_path: str | Path,
    enriched_manifest_path: str | Path,
    licence_manifest_path: str | Path,
    summary_path: str | Path,
    lock_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Bind every enrichment input and output after enrichment succeeds."""
    config_file = resolve_project_path(config_path)
    local_manifest = resolve_project_path(local_manifest_path)
    sealed_cache = resolve_project_path(sealed_cache_path)
    cache_lock_file = resolve_project_path(cache_lock_path)
    enriched_manifest = resolve_project_path(enriched_manifest_path)
    licence_manifest = resolve_project_path(licence_manifest_path)
    summary_file = resolve_project_path(summary_path)
    lock_destination = require_safe_output(lock_path)
    if lock_destination.exists():
        raise FileExistsError(f"Enrichment lock already exists: {lock_destination}")

    with project_lock("metadata_enrichment_lock"):
        cache_lock = verify_metadata_cache_lock(cache_lock_file, sealed_cache)
        if (
            resolve_project_path(cache_lock["artifacts"]["local_manifest"]["path"])
            != local_manifest
        ):
            raise ValueError("Metadata cache lock points to a different local manifest")
        config = load_toml(config_file)
        config_sha256 = config_fingerprint(config)
        local_manifest_sha256 = sha256_file(local_manifest)
        sealed_cache_sha256 = sha256_file(sealed_cache)
        enriched_rows, enriched_manifest_sha256 = read_csv_snapshot(enriched_manifest)
        licence_rows, licence_manifest_sha256 = read_csv_snapshot(licence_manifest)
        summary, summary_sha256 = _read_json_snapshot(summary_file)
        cache_lock_sha256 = sha256_file(cache_lock_file)

        if summary.get("source_local_manifest_sha256") != local_manifest_sha256:
            raise ValueError("Enrichment summary local-manifest hash is invalid")
        if summary.get("source_metadata_cache_sha256") != sealed_cache_sha256:
            raise ValueError("Enrichment summary metadata-cache hash is invalid")
        if summary.get("enriched_manifest_sha256") != enriched_manifest_sha256:
            raise ValueError("Enrichment summary manifest hash is invalid")
        if summary.get("ready_for_manual_review") is not True:
            raise ValueError("Enrichment did not retrieve complete metadata")
        if int(summary.get("recordings") or -1) != len(enriched_rows):
            raise ValueError("Enrichment summary recording count is invalid")
        if len(licence_rows) != len(enriched_rows):
            raise ValueError("Licence manifest must contain one row per recording")
        metadata_counts = Counter(row.get("metadata_status") for row in enriched_rows)
        if metadata_counts.get("ok", 0) + metadata_counts.get("unavailable", 0) != len(
            enriched_rows
        ) or metadata_counts.get("error", 0):
            raise ValueError("Every enriched row must have a terminal metadata status")

        lock = {
            "schema_version": ENRICHMENT_LOCK_SCHEMA_VERSION,
            "locked_at_utc": _utc_now(),
            "ready_for_manual_review": True,
            "recordings": len(enriched_rows),
            "config_sha256": config_sha256,
            "source_local_manifest_sha256": local_manifest_sha256,
            "source_metadata_cache_sha256": sealed_cache_sha256,
            "metadata_cache_lock_sha256": cache_lock_sha256,
            "enriched_manifest_sha256": enriched_manifest_sha256,
            "licence_manifest_sha256": licence_manifest_sha256,
            "summary_sha256": summary_sha256,
            "artifacts": {
                "config": {
                    "path": _project_relative(config_file),
                    "sha256": sha256_file(config_file),
                },
                "local_manifest": {
                    "path": _project_relative(local_manifest),
                    "sha256": local_manifest_sha256,
                },
                "sealed_cache": {
                    "path": _project_relative(sealed_cache),
                    "sha256": sealed_cache_sha256,
                },
                "metadata_cache_lock": {
                    "path": _project_relative(cache_lock_file),
                    "sha256": cache_lock_sha256,
                },
                "enriched_manifest": {
                    "path": _project_relative(enriched_manifest),
                    "sha256": enriched_manifest_sha256,
                },
                "licence_manifest": {
                    "path": _project_relative(licence_manifest),
                    "sha256": licence_manifest_sha256,
                },
                "summary": {
                    "path": _project_relative(summary_file),
                    "sha256": summary_sha256,
                },
            },
        }
        lock_destination = atomic_write_json(lock_destination, lock)
    return lock_destination, lock


def verify_enrichment_lock(
    lock_path: str | Path,
    expected_enriched_path: str | Path | None = None,
) -> dict[str, Any]:
    lock_file = resolve_project_path(lock_path)
    lock, _ = _read_json_snapshot(lock_file)
    if lock.get("schema_version") != ENRICHMENT_LOCK_SCHEMA_VERSION:
        raise ValueError("Enrichment lock schema is not supported")
    if lock.get("ready_for_manual_review") is not True:
        raise ValueError("Enrichment lock is not ready for manual review")
    paths = _verify_artifact_table(
        lock,
        {
            "config",
            "local_manifest",
            "sealed_cache",
            "metadata_cache_lock",
            "enriched_manifest",
            "licence_manifest",
            "summary",
        },
    )
    if expected_enriched_path is not None and paths["enriched_manifest"] != resolve_project_path(
        expected_enriched_path
    ):
        raise ValueError("Enrichment lock points to a different enriched manifest")
    if lock.get("enriched_manifest_sha256") != sha256_file(paths["enriched_manifest"]):
        raise ValueError("Enrichment lock has an inconsistent manifest hash")
    if lock.get("metadata_cache_lock_sha256") != sha256_file(paths["metadata_cache_lock"]):
        raise ValueError("Enrichment lock has an inconsistent cache-lock hash")
    verify_metadata_cache_lock(paths["metadata_cache_lock"], paths["sealed_cache"])
    config = load_toml(paths["config"])
    if lock.get("config_sha256") != config_fingerprint(config):
        raise ValueError("Enrichment lock has an inconsistent config hash")
    return lock
