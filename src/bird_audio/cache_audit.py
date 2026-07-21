from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bird_audio.clip_cache import (
    DEFAULT_CACHE_ROOT,
    INDEX_FIELDS,
    verify_known_clip_cache,
)
from bird_audio.clip_selection import MINIMUM_SELECTED_START_SEPARATION_SAMPLES
from bird_audio.hashing import sha256_bytes, sha256_file
from bird_audio.io_utils import read_csv_snapshot
from bird_audio.paths import PROJECT_ROOT, is_relative_to, resolve_project_path
from bird_audio.splitting import SPLIT_NAMES


class CacheAuditError(ValueError):
    """Raised when independent cache-to-split validation fails."""


def _read_json_object_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise CacheAuditError(f"{label} must contain a JSON object")
    return value, sha256_bytes(payload)


def _locked_project_file(relative_path: object, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise CacheAuditError(f"Locked {label} path is invalid")
    path = resolve_project_path(relative_path)
    if (
        not is_relative_to(path, PROJECT_ROOT)
        or path.relative_to(PROJECT_ROOT).as_posix() != relative_path
        or not path.is_file()
    ):
        raise CacheAuditError(f"Locked {label} path is not a canonical project file")
    return path


def _expected_recordings_by_split(
    split_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Mapping[str, str]]]:
    expected: dict[str, dict[str, Mapping[str, str]]] = {split: {} for split in SPLIT_NAMES}
    global_recording_ids: set[str] = set()
    for row in split_rows:
        split = row.get("split", "")
        recording_id = row.get("recording_id", "")
        if split not in expected or not recording_id:
            raise CacheAuditError("Frozen split contains an invalid cache-binding row")
        if recording_id in global_recording_ids:
            raise CacheAuditError(f"Frozen split repeats recording ID {recording_id}")
        global_recording_ids.add(recording_id)
        expected[split][recording_id] = row
    return expected


def _read_locked_index(
    root: Path,
    split: str,
    entry: Mapping[str, Any],
) -> tuple[list[dict[str, str]], Path, str]:
    expected_relative = f"{split}/index.csv"
    if entry.get("path") != expected_relative:
        raise CacheAuditError(f"Locked {split} index path is invalid")
    path = (root / expected_relative).resolve()
    if (
        not is_relative_to(path, root)
        or path.relative_to(root).as_posix() != expected_relative
        or not path.is_file()
    ):
        raise CacheAuditError(f"Locked {split} index is not a canonical cache file")
    payload = path.read_bytes()
    snapshot_sha256 = sha256_bytes(payload)
    if snapshot_sha256 != entry.get("sha256"):
        raise CacheAuditError(f"Locked {split} index hash has drifted")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    if list(reader.fieldnames or []) != INDEX_FIELDS:
        raise CacheAuditError(f"Locked {split} index schema is invalid")
    rows = list(reader)
    if len(rows) != entry.get("rows"):
        raise CacheAuditError(f"Locked {split} index row count has drifted")
    return rows, path, snapshot_sha256


def _parse_energy_row(row: Mapping[str, str], context: str) -> tuple[int, int, float]:
    try:
        rank = int(row.get("energy_rank", ""))
        start = int(row.get("start_sample", ""))
        energy = float(row.get("energy_value", ""))
    except ValueError as exc:
        raise CacheAuditError(f"Cache energy fields are invalid: {context}") from exc
    if rank < 0 or start < 0 or not math.isfinite(energy) or energy < 0:
        raise CacheAuditError(f"Cache energy fields are invalid: {context}")
    return rank, start, energy


def _validate_energy_group(rows: Sequence[Mapping[str, str]], context: str) -> None:
    ranked = sorted(
        _parse_energy_row(row, context) for row in rows if row.get("energy_selected") == "true"
    )
    if not ranked or [rank for rank, _, _ in ranked] != list(range(len(ranked))):
        raise CacheAuditError(f"Cache energy ranks are not contiguous: {context}")
    starts = [start for _, start, _ in ranked]
    if len(starts) != len(set(starts)):
        raise CacheAuditError(f"Cache energy starts are duplicated: {context}")
    if any(
        abs(first - second) < MINIMUM_SELECTED_START_SEPARATION_SAMPLES
        for index, first in enumerate(starts)
        for second in starts[index + 1 :]
    ):
        raise CacheAuditError(f"Cache energy starts violate minimum separation: {context}")
    order_keys = [(-energy, start) for _, start, energy in ranked]
    if order_keys != sorted(order_keys):
        raise CacheAuditError(f"Cache energy ranks violate deterministic ordering: {context}")


def _audit_split_rows(
    split: str,
    rows: Sequence[dict[str, str]],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, set[str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        recording_id = row.get("recording_id", "")
        if not recording_id:
            raise CacheAuditError(f"Cache {split} index contains an empty recording ID")
        groups.setdefault(recording_id, []).append(row)
    if set(groups) != set(expected):
        missing = sorted(set(expected) - set(groups))
        unexpected = sorted(set(groups) - set(expected))
        raise CacheAuditError(
            f"Cache recording set differs from frozen {split}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    bindings = {
        "relative_path": "relative_path",
        "source_sha256": "sha256",
        "species_common_name": "species_common_name",
        "session_group": "session_group",
        "split": "split",
    }
    identities = {field: set() for field in ("recording_id", "source_sha256", "session_group")}
    for recording_id, group in sorted(groups.items()):
        expected_row = expected[recording_id]
        for row in group:
            for cache_field, source_field in bindings.items():
                if row.get(cache_field) != expected_row.get(source_field):
                    raise CacheAuditError(
                        f"Cache row differs from frozen split binding: {split}:{recording_id}"
                    )
        _validate_energy_group(group, f"{split}:{recording_id}")
        identities["recording_id"].add(recording_id)
        identities["source_sha256"].add(group[0]["source_sha256"])
        identities["session_group"].add(group[0]["session_group"])
    return identities


def _require_cross_split_disjointness(
    identities: Mapping[str, Mapping[str, set[str]]],
) -> None:
    for index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[index + 1 :]:
            for field in ("recording_id", "source_sha256", "session_group"):
                if identities[first][field].intersection(identities[second][field]):
                    raise CacheAuditError(f"Cache splits overlap on {field}: {first} and {second}")


def audit_known_clip_cache(
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    *,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Run public verification, then independently bind cache rows to the frozen split."""
    verified = verify_known_clip_cache(
        cache_root,
        ffmpeg=ffmpeg,
        expected_lock_sha256=expected_lock_sha256,
    )
    root = resolve_project_path(cache_root)
    lock_path = root / "lock.json"
    lock, lock_sha256 = _read_json_object_snapshot(lock_path, "cache lock")
    if lock_sha256 != verified.get("lock_sha256"):
        raise CacheAuditError("Cache lock changed after public verification")
    provenance = lock.get("provenance")
    artifacts = lock.get("artifacts")
    if not isinstance(provenance, dict) or not isinstance(artifacts, dict):
        raise CacheAuditError("Cache lock lacks provenance or artifact bindings")
    input_paths = provenance.get("input_paths")
    split_artifacts = artifacts.get("splits")
    if not isinstance(input_paths, dict) or not isinstance(split_artifacts, dict):
        raise CacheAuditError("Cache lock lacks split input or output bindings")

    split_path = _locked_project_file(input_paths.get("split"), "split")
    split_rows, split_sha256 = read_csv_snapshot(split_path)
    if split_sha256 != provenance.get("split_sha256"):
        raise CacheAuditError("Frozen split hash differs from cache provenance")
    expected = _expected_recordings_by_split(split_rows)

    identities: dict[str, dict[str, set[str]]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    index_snapshots: list[tuple[Path, str]] = []
    for split in SPLIT_NAMES:
        entry = split_artifacts.get(split)
        if not isinstance(entry, dict) or not isinstance(entry.get("index"), dict):
            raise CacheAuditError(f"Cache lock lacks the {split} index binding")
        rows, index_path, index_sha256 = _read_locked_index(root, split, entry["index"])
        index_snapshots.append((index_path, index_sha256))
        identities[split] = _audit_split_rows(split, rows, expected[split])
        split_counts[split] = {
            "recordings": len(identities[split]["recording_id"]),
            "clips": len(rows),
        }
    _require_cross_split_disjointness(identities)

    if sha256_file(lock_path) != lock_sha256:
        raise CacheAuditError("Cache lock changed during independent audit")
    if any(sha256_file(path) != digest for path, digest in index_snapshots):
        raise CacheAuditError("Cache index changed during independent audit")
    _, final_split_sha256 = read_csv_snapshot(split_path)
    if final_split_sha256 != split_sha256:
        raise CacheAuditError("Frozen split changed during independent audit")
    return {
        "valid": True,
        "cache_version": verified["cache_version"],
        "lock_sha256": lock_sha256,
        "recordings": sum(row["recordings"] for row in split_counts.values()),
        "clips": sum(row["clips"] for row in split_counts.values()),
        "splits": split_counts,
        "source_bindings_exact": True,
        "zero_recording_overlap": True,
        "zero_hash_overlap": True,
        "zero_session_overlap": True,
        "energy_selection_invariants_valid": True,
        "auditor_sha256": sha256_file(Path(__file__)),
    }
