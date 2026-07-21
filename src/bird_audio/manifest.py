from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bird_audio.audio import AudioProbe, detect_header, probe_audio, verify_full_decode
from bird_audio.config import load_toml
from bird_audio.hashing import sha256_file, sha256_json
from bird_audio.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_snapshot,
    require_unchanged,
)
from bird_audio.locking import project_lock
from bird_audio.paths import PROJECT_ROOT, is_relative_to, require_safe_output, resolve_project_path

LOCAL_MANIFEST_FIELDS = [
    "schema_version",
    "recording_id",
    "xc_id",
    "xc_url",
    "species_folder",
    "species_common_name",
    "scientific_name",
    "relative_path",
    "source_extension",
    "file_size_bytes",
    "source_mtime_ns",
    "source_inode",
    "sha256",
    "header_type",
    "format_name",
    "codec_name",
    "codec_long_name",
    "source_sample_rate_hz",
    "channels",
    "channel_layout",
    "sample_format",
    "bits_per_sample",
    "bits_per_raw_sample",
    "ffprobe_duration_seconds",
    "bit_rate_bps",
    "probe_ok",
    "probe_error",
    "duplicate_group",
    "duplicate_group_size",
    "duplicate_canonical_recording_id",
    "full_decode_status",
    "decoded_duration_seconds",
    "decoded_duration_ratio",
    "canonical_duration_seconds",
    "full_decode_diagnostic",
    "metadata_status",
    "metadata_fetched_at_utc",
    "metadata_query_version",
    "metadata_error",
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
    "vocalisation_type",
    "remarks",
    "recording_device",
    "microphone",
    "recording_method",
    "playback_used",
    "automatic_recording",
    "uploaded_date",
    "api_recording_url",
    "api_sample_rate_hz",
    "licence",
    "licence_status",
    "attribution",
    "source_audio_url",
    "session_group",
    "session_review_flag",
    "session_review_reason",
    "split",
    "local_qc_status",
    "exclusion_reasons",
]

QC_PRECEDENCE = {"include": 0, "pending_metadata": 1, "manual_review": 2, "exclude": 3}
HARD_EXCLUSION_REASONS = {
    "probe_failed",
    "local_scan_failed",
    "non_positive_duration",
    "raw_file_changed_during_scan",
    "full_decode_failed",
    "exact_duplicate_noncanonical",
    "cross_label_exact_duplicate",
    "full_decode_warning",
    "target_species_in_secondary_labels",
}


@dataclass(frozen=True)
class ScanJob:
    path: Path
    species_folder: str
    common_name: str
    scientific_name: str


def _species_lookup(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    entries = config.get("known_species") or []
    lookup: dict[str, dict[str, str]] = {}
    for entry in entries:
        folder = str(entry["folder"])
        if folder in lookup:
            raise ValueError(f"Duplicate species folder in config: {folder}")
        lookup[folder] = {
            "common_name": str(entry["common_name"]),
            "scientific_name": str(entry["scientific_name"]),
        }
    if not lookup:
        raise ValueError("No known species are configured")
    return lookup


def discover_scan_jobs(config: dict[str, Any]) -> list[ScanJob]:
    raw_root = resolve_project_path(str(config["raw_audio_dir"]))
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw audio directory does not exist: {raw_root}")

    species = _species_lookup(config)
    actual_directories = {path.name for path in raw_root.iterdir() if path.is_dir()}
    missing = sorted(set(species) - actual_directories)
    unexpected = sorted(actual_directories - set(species))
    if missing or unexpected:
        raise ValueError(f"Species directory mismatch. Missing={missing}, unexpected={unexpected}")

    jobs: list[ScanJob] = []
    for folder, identity in sorted(species.items()):
        directory = raw_root / folder
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == ".DS_Store":
                continue
            jobs.append(
                ScanJob(
                    path=path,
                    species_folder=folder,
                    common_name=identity["common_name"],
                    scientific_name=identity["scientific_name"],
                )
            )
    return jobs


def _format_float(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "0.000000"


def _base_row(job: ScanJob, schema_version: str, source_stat: Any) -> dict[str, Any]:
    xc_id = job.path.stem
    relative_path = job.path.resolve().relative_to(PROJECT_ROOT).as_posix()
    return {
        "schema_version": schema_version,
        "recording_id": f"XC{xc_id}",
        "xc_id": xc_id,
        "xc_url": f"https://xeno-canto.org/{xc_id}",
        "species_folder": job.species_folder,
        "species_common_name": job.common_name,
        "scientific_name": job.scientific_name,
        "relative_path": relative_path,
        "source_extension": job.path.suffix.lower(),
        "file_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_inode": source_stat.st_ino,
        "sha256": "",
        "header_type": "",
        "format_name": "",
        "codec_name": "",
        "codec_long_name": "",
        "source_sample_rate_hz": 0,
        "channels": 0,
        "channel_layout": "",
        "sample_format": "",
        "bits_per_sample": 0,
        "bits_per_raw_sample": 0,
        "ffprobe_duration_seconds": "0.000000",
        "bit_rate_bps": 0,
        "probe_ok": "false",
        "probe_error": "",
        "duplicate_group": "",
        "duplicate_group_size": 1,
        "duplicate_canonical_recording_id": "",
        "full_decode_status": "pending",
        "decoded_duration_seconds": "",
        "decoded_duration_ratio": "",
        "canonical_duration_seconds": "",
        "full_decode_diagnostic": "",
        "metadata_status": "pending",
        "metadata_fetched_at_utc": "",
        "metadata_query_version": "",
        "metadata_error": "",
        "primary_label": "",
        "api_scientific_name": "",
        "api_group": "",
        "secondary_labels": "",
        "target_secondary_labels": "",
        "recordist": "",
        "country": "",
        "locality": "",
        "latitude": "",
        "longitude": "",
        "recorded_date": "",
        "recorded_time": "",
        "quality": "",
        "vocalisation_type": "",
        "remarks": "",
        "recording_device": "",
        "microphone": "",
        "recording_method": "",
        "playback_used": "",
        "automatic_recording": "",
        "uploaded_date": "",
        "api_recording_url": "",
        "api_sample_rate_hz": "",
        "licence": "",
        "licence_status": "pending",
        "attribution": "",
        "source_audio_url": "",
        "session_group": "",
        "session_review_flag": "false",
        "session_review_reason": "",
        "split": "",
        "local_qc_status": "pending_metadata",
        "exclusion_reasons": "",
    }


def _reason_is_hard_exclusion(reason: str) -> bool:
    return reason in HARD_EXCLUSION_REASONS or reason.startswith("source_sample_rate_below_")


def apply_qc_reason(row: dict[str, Any], reason: str, target_status: str) -> None:
    """Add a QC reason without allowing a lower-severity status to win."""
    if target_status not in QC_PRECEDENCE:
        raise ValueError(f"Unknown QC status: {target_status}")
    reasons = set(filter(None, str(row.get("exclusion_reasons", "")).split(";")))
    reasons.add(reason)
    row["exclusion_reasons"] = ";".join(sorted(reasons))
    current = str(row.get("local_qc_status") or "pending_metadata")
    if QC_PRECEDENCE[target_status] > QC_PRECEDENCE.get(current, -1):
        row["local_qc_status"] = target_status


def _scan_one(
    job: ScanJob,
    ffprobe: Path,
    schema_version: str,
    minimum_sample_rate_hz: int,
) -> dict[str, Any]:
    source_stat = job.path.stat()
    row = _base_row(job, schema_version, source_stat)
    try:
        row["header_type"] = detect_header(job.path)
        row["sha256"] = sha256_file(job.path)
        probe: AudioProbe = probe_audio(job.path, ffprobe)
        row.update(probe.to_dict())
        row["probe_ok"] = str(probe.probe_ok).lower()
        row["ffprobe_duration_seconds"] = _format_float(probe.ffprobe_duration_seconds)
        if not probe.probe_ok:
            apply_qc_reason(row, "probe_failed", "exclude")
        if probe.probe_ok and probe.source_sample_rate_hz < minimum_sample_rate_hz:
            apply_qc_reason(
                row,
                f"source_sample_rate_below_{minimum_sample_rate_hz}_hz",
                "exclude",
            )
        if probe.probe_ok and probe.ffprobe_duration_seconds <= 0:
            apply_qc_reason(row, "non_positive_duration", "exclude")
        final_stat = job.path.stat()
        if (
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ino,
        ) != (
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ino,
        ):
            apply_qc_reason(row, "raw_file_changed_during_scan", "exclude")
    except Exception as exc:  # Preserve a row so that failures remain auditable.
        row["probe_ok"] = "false"
        row["probe_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        apply_qc_reason(row, "local_scan_failed", "exclude")
    return row


def _annotate_duplicate_groups(rows: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("sha256") or "")
        if digest:
            by_hash[digest].append(row)
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        labels = {str(row["species_common_name"]) for row in group}
        group_id = f"sha256:{digest[:16]}"
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["xc_id"]) if str(row["xc_id"]).isdigit() else math.inf,
                str(row["recording_id"]),
            ),
        )
        canonical = ordered[0]["recording_id"]
        for row in group:
            row["duplicate_group"] = group_id
            row["duplicate_group_size"] = len(group)
            row["duplicate_canonical_recording_id"] = canonical
        if len(labels) > 1:
            for row in group:
                apply_qc_reason(row, "cross_label_exact_duplicate", "exclude")
            continue
        for row in ordered[1:]:
            apply_qc_reason(row, "exact_duplicate_noncanonical", "exclude")


def build_local_manifest(
    config_path: str | Path,
    output_path: str | Path,
    ffprobe: Path,
    workers: int = 8,
    overwrite: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    config = load_toml(config_path)
    destination = require_safe_output(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to replace it: {destination}")
    with project_lock("local_manifest"):
        jobs = discover_scan_jobs(config)
        schema_version = str(config.get("schema_version", "1.0"))
        minimum_rate = int(config["minimum_source_sample_rate_hz"])
        print(f"Scanning {len(jobs)} raw recordings with {max(1, workers)} workers")

        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_scan_one, job, ffprobe, schema_version, minimum_rate): job
                for job in jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index % 100 == 0 or index == len(futures):
                    print(f"Scanned {index}/{len(futures)}")

        rows.sort(key=lambda row: (str(row["species_folder"]), str(row["xc_id"])))
        _annotate_duplicate_groups(rows)
        destination = atomic_write_csv(destination, rows, LOCAL_MANIFEST_FIELDS)
    print(f"Local manifest saved: {destination}")
    return destination, rows


def _decode_one(
    row: dict[str, str],
    ffmpeg: Path,
    minimum_duration_ratio: float,
    maximum_duration_ratio: float,
) -> tuple[str, str, str, float, float]:
    relative_path = row["relative_path"]
    path = resolve_project_path(relative_path)
    try:
        result = verify_full_decode(path, ffmpeg)
    except Exception as exc:
        return relative_path, "failed", f"{type(exc).__name__}: {exc}"[:2000], 0.0, 0.0
    expected_duration = float(row.get("ffprobe_duration_seconds") or 0)
    ratio = result.decoded_duration_seconds / expected_duration if expected_duration > 0 else 0.0
    duration_warning = ratio < minimum_duration_ratio or ratio > maximum_duration_ratio
    if result.diagnostic or duration_warning:
        diagnostic = result.diagnostic
        if duration_warning:
            suffix = f"decoded_duration_ratio={ratio:.6f}"
            diagnostic = f"{diagnostic} | {suffix}" if diagnostic else suffix
        return (
            relative_path,
            "warning",
            diagnostic[:2000],
            result.decoded_duration_seconds,
            ratio,
        )
    return relative_path, "ok", "", result.decoded_duration_seconds, ratio


def audit_full_decodes(
    manifest_path: str | Path,
    output_path: str | Path,
    ffmpeg: Path,
    workers: int = 4,
    overwrite: bool = False,
    minimum_duration_ratio: float = 0.98,
    maximum_duration_ratio: float = 1.02,
    exclude_warnings: bool = True,
) -> tuple[Path, list[dict[str, str]]]:
    """Decode every recording to a null sink and persist an auditable status."""
    resolved_manifest = resolve_project_path(manifest_path)
    destination = require_safe_output(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {destination}")
    with project_lock("local_manifest"):
        rows, input_sha256 = read_csv_snapshot(resolved_manifest)
        by_path = {row["relative_path"]: row for row in rows}
        candidates = [row for row in rows if row.get("probe_ok") == "true"]
        print(f"Full-decoding {len(candidates)} recordings with {max(1, workers)} workers")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    _decode_one,
                    row,
                    ffmpeg,
                    minimum_duration_ratio,
                    maximum_duration_ratio,
                ): row
                for row in candidates
            }
            for index, future in enumerate(as_completed(futures), start=1):
                relative_path, status, diagnostic, decoded_duration, duration_ratio = (
                    future.result()
                )
                row = by_path[relative_path]
                row["full_decode_status"] = status
                row["decoded_duration_seconds"] = _format_float(decoded_duration)
                row["decoded_duration_ratio"] = _format_float(duration_ratio)
                row["canonical_duration_seconds"] = _format_float(decoded_duration)
                row["full_decode_diagnostic"] = diagnostic
                if status == "failed":
                    apply_qc_reason(row, "full_decode_failed", "exclude")
                elif status == "warning":
                    if exclude_warnings:
                        apply_qc_reason(row, "full_decode_warning", "exclude")
                    else:
                        apply_qc_reason(row, "decode_warning_manual_review", "manual_review")
                if index % 100 == 0 or index == len(futures):
                    print(f"Decoded {index}/{len(futures)}")

        require_unchanged(resolved_manifest, input_sha256)
        rows.sort(key=lambda row: (row["species_folder"], row["xc_id"]))
        destination = atomic_write_csv(destination, rows, LOCAL_MANIFEST_FIELDS)
    counts = Counter(row["full_decode_status"] for row in rows)
    print(f"Full-decode audit saved: {destination}")
    print(f"Full-decode statuses: {dict(sorted(counts.items()))}")
    return destination, rows


def _counter(rows: Iterable[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "")) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _header_format_consistent(header: str, format_name: str) -> bool:
    formats = set(format_name.split(","))
    if header in {"riff_wave", "rf64_wave"}:
        return "wav" in formats
    if header in {"mp3_id3", "mpeg_audio"}:
        return "mp3" in formats
    return True


def summarize_local_manifest(
    rows: list[dict[str, str]],
    manifest_sha256: str,
    minimum_sample_rate_hz: int,
) -> dict[str, Any]:
    durations: list[float] = []
    duration_sources: Counter[str] = Counter()
    for row in rows:
        if row.get("probe_ok") != "true":
            continue
        canonical = float(row.get("canonical_duration_seconds") or 0)
        probed = float(row.get("ffprobe_duration_seconds") or 0)
        duration = canonical or probed
        if duration > 0:
            durations.append(duration)
            duration_sources["canonical_decode" if canonical > 0 else "ffprobe_fallback"] += 1
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group = row.get("duplicate_group", "")
        if group:
            duplicate_groups[group].append(row["relative_path"])

    rate_counts = Counter(
        int(row.get("source_sample_rate_hz") or 0) for row in rows if row.get("probe_ok") == "true"
    )
    header_mismatches = [
        row["relative_path"]
        for row in rows
        if not _header_format_consistent(row.get("header_type", ""), row.get("format_name", ""))
    ]
    raw_fingerprint = sha256_json(
        sorted(
            ({"relative_path": row["relative_path"], "sha256": row["sha256"]} for row in rows),
            key=lambda item: item["relative_path"],
        )
    )

    duration_summary: dict[str, float] = {}
    if durations:
        duration_summary = {
            "minimum_seconds": min(durations),
            "median_seconds": statistics.median(durations),
            "maximum_seconds": max(durations),
            "total_hours": math.fsum(durations) / 3600,
            "sources": dict(sorted(duration_sources.items())),
        }

    return {
        "schema_version": "1.0",
        "recordings": len(rows),
        "species": len({row["species_folder"] for row in rows}),
        "species_counts": _counter(rows, "species_common_name"),
        "header_types": _counter(rows, "header_type"),
        "format_names": _counter(rows, "format_name"),
        "codecs": _counter(rows, "codec_name"),
        "sample_rates_hz": {str(key): value for key, value in sorted(rate_counts.items())},
        "channels": _counter(rows, "channels"),
        "bits_per_sample": _counter(rows, "bits_per_sample"),
        "duration": duration_summary,
        "probe_failures": sum(row.get("probe_ok") != "true" for row in rows),
        "full_decode_statuses": _counter(rows, "full_decode_status"),
        "full_decode_review_files": [
            {
                "relative_path": row["relative_path"],
                "status": row.get("full_decode_status", ""),
                "diagnostic": row.get("full_decode_diagnostic", ""),
            }
            for row in rows
            if row.get("full_decode_status") in {"warning", "failed"}
        ],
        "below_minimum_sample_rate": sum(
            row.get("probe_ok") == "true"
            and int(row.get("source_sample_rate_hz") or 0) < minimum_sample_rate_hz
            for row in rows
        ),
        "minimum_sample_rate_hz": minimum_sample_rate_hz,
        "local_qc_statuses": _counter(rows, "local_qc_status"),
        "duplicate_groups": dict(sorted(duplicate_groups.items())),
        "duplicate_group_count": len(duplicate_groups),
        "recordings_in_duplicate_groups": sum(len(group) for group in duplicate_groups.values()),
        "duplicate_excess_count": sum(len(group) - 1 for group in duplicate_groups.values()),
        "header_format_mismatches": header_mismatches,
        "manifest_sha256": manifest_sha256,
        "raw_data_fingerprint": raw_fingerprint,
    }


def _issue(issues: list[dict[str, str]], code: str, detail: str, severity: str = "error") -> None:
    issues.append({"severity": severity, "code": code, "detail": detail})


def _as_number(value: str, kind: type[int] | type[float]) -> int | float | None:
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def _validate_duplicate_annotations(
    rows: list[dict[str, str]],
    issues: list[dict[str, str]],
) -> None:
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            by_hash[row["sha256"]].append(row)
    for digest, group in by_hash.items():
        if len(group) == 1:
            row = group[0]
            if row.get("duplicate_group") or row.get("duplicate_canonical_recording_id"):
                _issue(issues, "unexpected_duplicate_annotation", row["relative_path"])
            continue
        labels = {row["species_common_name"] for row in group}
        if len(labels) != 1:
            _issue(issues, "cross_label_exact_duplicate", digest)
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["xc_id"]) if row["xc_id"].isdigit() else math.inf,
                row["recording_id"],
            ),
        )
        canonical = ordered[0]["recording_id"]
        expected_group = f"sha256:{digest[:16]}"
        for row in group:
            if row.get("duplicate_group") != expected_group:
                _issue(issues, "duplicate_group_mismatch", row["relative_path"])
            if _as_number(row.get("duplicate_group_size", ""), int) != len(group):
                _issue(issues, "duplicate_group_size_mismatch", row["relative_path"])
            if row.get("duplicate_canonical_recording_id") != canonical:
                _issue(issues, "duplicate_canonical_mismatch", row["relative_path"])
        for row in ordered[1:]:
            reasons = set(filter(None, row.get("exclusion_reasons", "").split(";")))
            if (
                "exact_duplicate_noncanonical" not in reasons
                or row.get("local_qc_status") != "exclude"
            ):
                _issue(issues, "duplicate_noncanonical_not_excluded", row["relative_path"])


def validate_local_manifest(
    config_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    verify_hashes: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    config = load_toml(config_path)
    resolved_manifest = resolve_project_path(manifest_path)
    with project_lock("local_manifest"):
        rows, manifest_sha256 = read_csv_snapshot(resolved_manifest)
        jobs = discover_scan_jobs(config)
        issues: list[dict[str, str]] = []

        missing_columns = sorted(set(LOCAL_MANIFEST_FIELDS) - set(rows[0] if rows else []))
        if missing_columns:
            _issue(issues, "missing_columns", ",".join(missing_columns))

        expected_by_path = {
            job.path.resolve().relative_to(PROJECT_ROOT).as_posix(): job for job in jobs
        }
        manifest_paths = {row.get("relative_path", "") for row in rows}
        expected_paths = set(expected_by_path)
        if manifest_paths != expected_paths:
            missing = sorted(expected_paths - manifest_paths)
            unexpected = sorted(manifest_paths - expected_paths)
            _issue(
                issues,
                "raw_path_set_mismatch",
                f"missing={missing[:20]}, unexpected={unexpected[:20]}",
            )

        for field in ("recording_id", "relative_path"):
            values = [row.get(field, "") for row in rows]
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            if duplicates:
                _issue(issues, f"duplicate_{field}", ",".join(duplicates[:20]))

        configured_species = _species_lookup(config)
        manifest_species = {row.get("species_folder", "") for row in rows}
        if set(configured_species) != manifest_species:
            _issue(
                issues,
                "species_set_mismatch",
                f"configured={sorted(configured_species)}, manifest={sorted(manifest_species)}",
            )

        raw_root = resolve_project_path(str(config["raw_audio_dir"]))
        minimum_rate = int(config["minimum_source_sample_rate_hz"])
        allowed_qc = set(QC_PRECEDENCE)
        allowed_decode = {"pending", "ok", "warning", "failed"}
        for row in rows:
            relative = row.get("relative_path", "")
            expected = expected_by_path.get(relative)
            path = resolve_project_path(relative)
            if not is_relative_to(path, raw_root):
                _issue(issues, "raw_path_outside_root", relative)
                continue
            if not path.is_file():
                _issue(issues, "missing_raw_file", relative)
                continue
            if expected is None:
                continue

            expected_xc_id = expected.path.stem
            relationships = {
                "xc_id": expected_xc_id,
                "recording_id": f"XC{expected_xc_id}",
                "xc_url": f"https://xeno-canto.org/{expected_xc_id}",
                "species_folder": expected.species_folder,
                "species_common_name": expected.common_name,
                "scientific_name": expected.scientific_name,
                "source_extension": expected.path.suffix.lower(),
            }
            for field, expected_value in relationships.items():
                if row.get(field) != expected_value:
                    _issue(issues, f"{field}_relationship_mismatch", relative)

            for field in (
                "file_size_bytes",
                "source_mtime_ns",
                "source_inode",
                "source_sample_rate_hz",
                "channels",
                "bits_per_sample",
                "bits_per_raw_sample",
                "bit_rate_bps",
                "duplicate_group_size",
            ):
                if _as_number(row.get(field, ""), int) is None:
                    _issue(issues, f"invalid_integer_{field}", relative)
            for field in (
                "ffprobe_duration_seconds",
                "decoded_duration_seconds",
                "decoded_duration_ratio",
                "canonical_duration_seconds",
            ):
                value = row.get(field, "")
                if value and _as_number(value, float) is None:
                    _issue(issues, f"invalid_float_{field}", relative)

            if row.get("probe_ok") not in {"true", "false"}:
                _issue(issues, "invalid_probe_ok", relative)
            if row.get("local_qc_status") not in allowed_qc:
                _issue(issues, "invalid_local_qc_status", relative)
            if row.get("full_decode_status") not in allowed_decode:
                _issue(issues, "invalid_full_decode_status", relative)
            if not _header_format_consistent(
                row.get("header_type", ""), row.get("format_name", "")
            ):
                _issue(issues, "header_format_mismatch", relative)

            parsed_source_rate = _as_number(row.get("source_sample_rate_hz", ""), int)
            source_rate = int(parsed_source_rate) if parsed_source_rate is not None else 0
            reasons = set(filter(None, row.get("exclusion_reasons", "").split(";")))
            low_rate_reason = f"source_sample_rate_below_{minimum_rate}_hz"
            if (
                source_rate < minimum_rate
                and row.get("probe_ok") == "true"
                and (low_rate_reason not in reasons or row.get("local_qc_status") != "exclude")
            ):
                _issue(issues, "low_rate_not_excluded", relative)
            if (
                any(_reason_is_hard_exclusion(reason) for reason in reasons)
                and row.get("local_qc_status") != "exclude"
            ):
                _issue(issues, "hard_exclusion_status_precedence", relative)
            if row.get("full_decode_status") == "warning" and row.get("local_qc_status") not in {
                "manual_review",
                "exclude",
            }:
                _issue(issues, "decode_warning_not_flagged", relative)
            if row.get("full_decode_status") in {"ok", "warning"}:
                decoded_value = _as_number(row.get("decoded_duration_seconds", ""), float)
                canonical_value = _as_number(row.get("canonical_duration_seconds", ""), float)
                decoded = float(decoded_value) if decoded_value is not None else 0.0
                canonical = float(canonical_value) if canonical_value is not None else 0.0
                if decoded <= 0 or not math.isclose(decoded, canonical, rel_tol=0, abs_tol=1e-6):
                    _issue(issues, "canonical_duration_mismatch", relative)

            current_stat = path.stat()
            recorded_size = _as_number(row.get("file_size_bytes", ""), int)
            if recorded_size != current_stat.st_size:
                _issue(issues, "file_size_mismatch", relative)
            if verify_hashes and sha256_file(path) != row.get("sha256"):
                _issue(issues, "hash_mismatch", relative)

        _validate_duplicate_annotations(rows, issues)
        audit_complete = all(
            row.get("full_decode_status") in {"ok", "warning", "failed"} for row in rows
        )
        eligible = [row for row in rows if row.get("local_qc_status") == "include"]
        manual_reviews_resolved = not any(
            row.get("local_qc_status") == "manual_review" for row in rows
        )
        metadata_complete = bool(eligible) and all(
            row.get("metadata_status") == "ok" for row in eligible
        )
        split_complete = False
        split_validation: dict[str, Any] = {"valid": False, "status": "not_frozen"}
        split_paths = (
            resolve_project_path(config["split_manifest"]),
            resolve_project_path(config["split_lock"]),
            resolve_project_path(config["split_summary"]),
        )
        if eligible and all(path.is_file() for path in split_paths):
            try:
                from bird_audio.splitting import validate_frozen_split

                split_validation = validate_frozen_split(
                    resolved_manifest,
                    split_paths[0],
                    split_paths[1],
                    config_path=config_path,
                    summary_path=split_paths[2],
                )
                split_complete = bool(split_validation["valid"])
                if not split_complete:
                    _issue(issues, "locked_split_invalid", str(split_validation["checks"]))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                split_validation = {
                    "valid": False,
                    "status": "validation_error",
                    "error_type": type(exc).__name__,
                }
                _issue(issues, "locked_split_validation_error", type(exc).__name__)
        elif eligible and any(path.exists() for path in split_paths):
            _issue(issues, "partial_split_artifacts", "split, summary, and lock must all exist")
        structural_valid = not any(issue["severity"] == "error" for issue in issues)
        hashes_verified = verify_hashes and not any(
            issue["code"] == "hash_mismatch" for issue in issues
        )

        summary = summarize_local_manifest(rows, manifest_sha256, minimum_rate)
        summary["validation"] = {
            "structural_valid": structural_valid,
            "audit_complete": audit_complete,
            "hashes_verified": hashes_verified,
            "manual_reviews_resolved": manual_reviews_resolved,
            "metadata_complete": metadata_complete,
            "split_complete": split_complete,
            "split_validation": split_validation,
            "eligible_recordings": len(eligible),
            "training_ready": (
                structural_valid
                and audit_complete
                and hashes_verified
                and manual_reviews_resolved
                and metadata_complete
                and split_complete
            ),
            "issues": issues,
        }
        destination = atomic_write_json(summary_path, summary)
    print(f"Validation summary saved: {destination}")
    print(f"Structural validation passed: {summary['validation']['structural_valid']}")
    print(f"Training ready: {summary['validation']['training_ready']}")
    return summary, issues
