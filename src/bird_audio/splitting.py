from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.config import config_fingerprint, load_toml
from bird_audio.hashing import sha256_file, sha256_json
from bird_audio.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    read_csv_snapshot,
    require_unchanged,
)
from bird_audio.locking import project_lock
from bird_audio.paths import (
    PROJECT_ROOT,
    RAW_DATA_ROOT,
    is_relative_to,
    require_safe_output,
    resolve_project_path,
)
from bird_audio.review import verify_review_lock

SPLIT_NAMES = ("train", "validation", "test")
ALLOCATOR_VERSION = "grouped_session_greedy_relocate_swap_v2"
SPLIT_FIELDS = [
    "recording_id",
    "relative_path",
    "sha256",
    "species_common_name",
    "session_group",
    "split",
    "split_seed",
    "source_manifest_sha256",
]


def _verify_raw_bindings(rows: list[dict[str, str]]) -> dict[str, Any]:
    failures: list[str] = []
    for row in rows:
        recording_id = row.get("recording_id", "unknown")
        relative_path = row.get("relative_path", "")
        path = resolve_project_path(relative_path)
        if (
            not relative_path
            or Path(relative_path).is_absolute()
            or not is_relative_to(path, RAW_DATA_ROOT)
        ):
            failures.append(f"{recording_id}:unsafe_raw_path")
            continue
        canonical_relative = path.relative_to(PROJECT_ROOT).as_posix()
        if canonical_relative != relative_path:
            failures.append(f"{recording_id}:noncanonical_raw_path")
            continue
        if not path.is_file():
            failures.append(f"{recording_id}:raw_file_missing")
            continue
        expected_sha256 = row.get("sha256", "")
        if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
            failures.append(f"{recording_id}:raw_sha256_mismatch")
    return {
        "valid": not failures,
        "recordings_checked": len(rows),
        "failures": failures,
    }


@dataclass(frozen=True)
class SessionGroup:
    group_id: str
    rows: tuple[dict[str, str], ...]
    species_counts: Counter[str]

    @property
    def size(self) -> int:
        return len(self.rows)


def integer_targets(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {split: total * fractions[split] for split in SPLIT_NAMES}
    targets = {split: math.floor(raw[split]) for split in SPLIT_NAMES}
    remainder = total - sum(targets.values())
    ranking = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(raw[split] - targets[split]), SPLIT_NAMES.index(split)),
    )
    for split in ranking[:remainder]:
        targets[split] += 1
    return targets


def _make_groups(rows: list[dict[str, str]]) -> list[SessionGroup]:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("session_group", "")
        if not group_id:
            raise ValueError(f"Included recording has no session group: {row.get('recording_id')}")
        by_group[group_id].append(row)
    return [
        SessionGroup(
            group_id=group_id,
            rows=tuple(group_rows),
            species_counts=Counter(row["species_common_name"] for row in group_rows),
        )
        for group_id, group_rows in sorted(by_group.items())
    ]


def _objective(
    achieved: dict[str, Counter[str]],
    targets: dict[str, dict[str, int]],
    global_targets: dict[str, int],
) -> float:
    score = 0.0
    for species, species_targets in targets.items():
        for split in SPLIT_NAMES:
            target = species_targets[split]
            actual = achieved[split][species]
            score += abs(actual - target) / max(1, target)
            if target > 0 and actual == 0:
                score += 5.0
    for split in SPLIT_NAMES:
        actual_total = sum(achieved[split].values())
        score += 0.2 * abs(actual_total - global_targets[split]) / max(1, global_targets[split])
    return score


def _apply_group(
    achieved: dict[str, Counter[str]],
    group: SessionGroup,
    split: str,
    sign: int,
) -> None:
    for species, count in group.species_counts.items():
        achieved[split][species] += sign * count


def allocate_grouped_split(
    rows: list[dict[str, str]],
    fractions: dict[str, float],
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups = _make_groups(rows)
    species_totals = Counter(row["species_common_name"] for row in rows)
    targets = {
        species: integer_targets(total, fractions)
        for species, total in sorted(species_totals.items())
    }
    global_targets = integer_targets(len(rows), fractions)
    achieved = {split: Counter() for split in SPLIT_NAMES}
    assignment: dict[str, str] = {}

    def rarity_score(group: SessionGroup) -> float:
        return sum(
            count / species_totals[species] for species, count in group.species_counts.items()
        )

    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -rarity_score(group),
            -group.size,
            sha256_json({"seed": seed, "group": group.group_id}),
            group.group_id,
        ),
    )
    for group in ordered_groups:
        options: list[tuple[float, int, str]] = []
        for split in SPLIT_NAMES:
            _apply_group(achieved, group, split, 1)
            score = _objective(achieved, targets, global_targets)
            _apply_group(achieved, group, split, -1)
            options.append((score, SPLIT_NAMES.index(split), split))
        selected = min(options)[2]
        assignment[group.group_id] = selected
        _apply_group(achieved, group, selected, 1)

    groups_by_id = {group.group_id: group for group in groups}
    ordered_ids = sorted(assignment)
    optimization_actions = 0
    maximum_actions = max(1000, len(groups) * 10)
    while optimization_actions < maximum_actions:
        current_score = _objective(achieved, targets, global_targets)
        relocation_applied = False
        for group_id in ordered_ids:
            group = groups_by_id[group_id]
            current_split = assignment[group_id]
            best = (current_score, SPLIT_NAMES.index(current_split), current_split)
            for candidate in SPLIT_NAMES:
                if candidate == current_split:
                    continue
                _apply_group(achieved, group, current_split, -1)
                _apply_group(achieved, group, candidate, 1)
                candidate_score = _objective(achieved, targets, global_targets)
                _apply_group(achieved, group, candidate, -1)
                _apply_group(achieved, group, current_split, 1)
                option = (candidate_score, SPLIT_NAMES.index(candidate), candidate)
                if option < best:
                    best = option
            if best[2] != current_split and best[0] < current_score - 1e-12:
                _apply_group(achieved, group, current_split, -1)
                _apply_group(achieved, group, best[2], 1)
                assignment[group_id] = best[2]
                optimization_actions += 1
                relocation_applied = True
                break
        if relocation_applied:
            continue

        swap_applied = False
        for left_index, left_id in enumerate(ordered_ids):
            left_split = assignment[left_id]
            left_group = groups_by_id[left_id]
            for right_id in ordered_ids[left_index + 1 :]:
                right_split = assignment[right_id]
                if left_split == right_split:
                    continue
                right_group = groups_by_id[right_id]
                _apply_group(achieved, left_group, left_split, -1)
                _apply_group(achieved, right_group, right_split, -1)
                _apply_group(achieved, left_group, right_split, 1)
                _apply_group(achieved, right_group, left_split, 1)
                candidate_score = _objective(achieved, targets, global_targets)
                if candidate_score < current_score - 1e-12:
                    assignment[left_id], assignment[right_id] = right_split, left_split
                    optimization_actions += 1
                    swap_applied = True
                    break
                _apply_group(achieved, left_group, right_split, -1)
                _apply_group(achieved, right_group, left_split, -1)
                _apply_group(achieved, left_group, left_split, 1)
                _apply_group(achieved, right_group, right_split, 1)
            if swap_applied:
                break
        if not swap_applied:
            break
    else:
        raise RuntimeError("Grouped split local search exceeded its deterministic action limit")

    achieved_table = {
        species: {split: achieved[split][species] for split in SPLIT_NAMES}
        for species in sorted(species_totals)
    }
    missing_coverage = [
        f"{species}:{split}"
        for species, species_targets in targets.items()
        for split in SPLIT_NAMES
        if species_targets[split] > 0 and achieved_table[species][split] == 0
    ]
    if missing_coverage:
        raise RuntimeError(
            "Grouped split could not preserve required class coverage: "
            + ", ".join(missing_coverage)
        )
    diagnostics = {
        "allocator_version": ALLOCATOR_VERSION,
        "objective": _objective(achieved, targets, global_targets),
        "objective_definition": (
            "sum target-normalized absolute per-species deviations plus a 5.0 "
            "missing-class penalty and a 0.2 weighted target-normalized global-size deviation"
        ),
        "optimization_actions": optimization_actions,
        "locally_optimal_for_single_relocations_and_pair_swaps": True,
        "targets": targets,
        "achieved": achieved_table,
        "deviations": {
            species: {
                split: achieved_table[species][split] - targets[species][split]
                for split in SPLIT_NAMES
            }
            for species in sorted(species_totals)
        },
        "global_targets": global_targets,
        "global_achieved": {split: sum(achieved[split].values()) for split in SPLIT_NAMES},
        "session_groups": len(groups),
        "session_groups_per_split": {
            split: Counter(assignment.values())[split] for split in SPLIT_NAMES
        },
        "class_coverage_complete": True,
    }
    return assignment, diagnostics


def freeze_grouped_split(
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    lock_path: str | Path,
    review_lock_path: str | Path = "data/manifests/review_v1_lock.json",
) -> tuple[Path, dict[str, Any]]:
    config = load_toml(config_path)
    config_sha256 = config_fingerprint(config)
    source = resolve_project_path(manifest_path)
    review_lock_file = resolve_project_path(review_lock_path)
    destination = require_safe_output(output_path)
    summary_destination = require_safe_output(summary_path)
    lock_destination = require_safe_output(lock_path)
    if lock_destination.exists():
        raise RuntimeError(f"Split is already locked and cannot be replaced: {lock_destination}")
    if destination.exists() or summary_destination.exists():
        raise FileExistsError(
            "Split outputs already exist; remove unfinished outputs before freezing"
        )

    with project_lock("split_freeze"):
        rows, manifest_sha256 = read_csv_snapshot(source)
        review_record = verify_review_lock(review_lock_file, source)
        review_lock_sha256 = sha256_file(review_lock_file)
        if review_record.get("final_manifest_sha256") != manifest_sha256:
            raise ValueError("Review lock does not bind the exact split-input manifest")
        unresolved = [
            row["recording_id"]
            for row in rows
            if row.get("local_qc_status") in {"pending_metadata", "manual_review"}
        ]
        if unresolved:
            raise ValueError(f"Cannot freeze split with {len(unresolved)} unresolved recordings")
        flagged_sessions = [
            row["recording_id"]
            for row in rows
            if row.get("session_review_flag") == "true" and row.get("local_qc_status") != "exclude"
        ]
        if flagged_sessions:
            raise ValueError(
                f"Cannot freeze split with {len(flagged_sessions)} unresolved session-review flags"
            )
        included = [row for row in rows if row.get("local_qc_status") == "include"]
        if not included:
            raise ValueError("No included recordings are available for splitting")
        if any(row.get("metadata_status") != "ok" for row in included):
            raise ValueError("Every included recording must have complete metadata")
        raw_bindings = _verify_raw_bindings(included)
        if not raw_bindings["valid"]:
            raise ValueError(
                "Included raw files do not match the reviewed manifest: "
                + ", ".join(raw_bindings["failures"][:20])
            )

        fractions = {
            "train": float(config["train_fraction"]),
            "validation": float(config["validation_fraction"]),
            "test": float(config["test_fraction"]),
        }
        seed = int(config["split_seed"])
        assignment, diagnostics = allocate_grouped_split(included, fractions, seed)
        split_rows = [
            {
                "recording_id": row["recording_id"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "species_common_name": row["species_common_name"],
                "session_group": row["session_group"],
                "split": assignment[row["session_group"]],
                "split_seed": seed,
                "source_manifest_sha256": manifest_sha256,
            }
            for row in included
        ]
        split_rows.sort(
            key=lambda row: (row["split"], row["species_common_name"], row["recording_id"])
        )
        require_unchanged(source, manifest_sha256)
        require_unchanged(review_lock_file, review_lock_sha256)
        destination = atomic_write_csv(destination, split_rows, SPLIT_FIELDS)
        split_sha256 = sha256_file(destination)
        diagnostics.update(
            {
                "schema_version": "1.2",
                "split_seed": seed,
                "source_manifest_sha256": manifest_sha256,
                "review_lock_sha256": review_lock_sha256,
                "split_sha256": split_sha256,
                "recordings": len(split_rows),
                "raw_files_verified": True,
                "raw_files_checked": raw_bindings["recordings_checked"],
                "zero_recording_overlap": len({row["recording_id"] for row in split_rows})
                == len(split_rows),
                "zero_hash_overlap": len({row["sha256"] for row in split_rows}) == len(split_rows),
                "zero_session_overlap": all(
                    len({row["split"] for row in split_rows if row["session_group"] == group}) == 1
                    for group in {row["session_group"] for row in split_rows}
                ),
            }
        )
        required_invariants = (
            diagnostics["zero_recording_overlap"],
            diagnostics["zero_hash_overlap"],
            diagnostics["zero_session_overlap"],
            diagnostics["class_coverage_complete"],
        )
        if not all(required_invariants):
            destination.unlink(missing_ok=True)
            raise RuntimeError("Split invariants failed; no summary or lock was created")
        require_unchanged(source, manifest_sha256)
        require_unchanged(review_lock_file, review_lock_sha256)
        atomic_write_json(summary_destination, diagnostics)
        summary_sha256 = sha256_file(summary_destination)
        lock_record = {
            "schema_version": "1.2",
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "source_manifest_sha256": manifest_sha256,
            "review_lock_sha256": review_lock_sha256,
            "split_sha256": split_sha256,
            "summary_sha256": summary_sha256,
            "config_sha256": config_sha256,
            "allocator_version": ALLOCATOR_VERSION,
            "split_seed": seed,
            "split_fractions": fractions,
            "recordings": len(split_rows),
            "recording_set_sha256": sha256_json(sorted(row["recording_id"] for row in split_rows)),
        }
        require_unchanged(source, manifest_sha256)
        require_unchanged(review_lock_file, review_lock_sha256)
        atomic_write_json(lock_destination, lock_record)
    return destination, diagnostics


def validate_frozen_split(
    manifest_path: str | Path,
    split_path: str | Path,
    lock_path: str | Path,
    config_path: str | Path = "configs/data.toml",
    summary_path: str | Path = "data/splits/split_v1_summary.json",
    review_lock_path: str | Path = "data/manifests/review_v1_lock.json",
) -> dict[str, Any]:
    config = load_toml(config_path)
    manifest = resolve_project_path(manifest_path)
    split_file = resolve_project_path(split_path)
    lock_file = resolve_project_path(lock_path)
    summary_file = resolve_project_path(summary_path)
    review_lock_file = resolve_project_path(review_lock_path)
    manifest_rows, manifest_sha256 = read_csv_snapshot(manifest)
    split_rows, split_sha256 = read_csv_snapshot(split_file)
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    review_record = verify_review_lock(review_lock_file, manifest)
    review_lock_sha256 = sha256_file(review_lock_file)
    included = [row for row in manifest_rows if row.get("local_qc_status") == "include"]
    raw_bindings = _verify_raw_bindings(included)
    included_ids = {row["recording_id"] for row in included}
    split_ids = {row["recording_id"] for row in split_rows}
    session_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        session_to_splits[row["session_group"]].add(row["split"])
        hash_to_splits[row["sha256"]].add(row["split"])
    included_by_id = {row["recording_id"]: row for row in included}
    row_bindings_match = all(
        row["recording_id"] in included_by_id
        and all(
            row.get(field) == included_by_id[row["recording_id"]].get(field)
            for field in (
                "relative_path",
                "sha256",
                "species_common_name",
                "session_group",
            )
        )
        for row in split_rows
    )
    species_split_counts = Counter((row["species_common_name"], row["split"]) for row in split_rows)
    species = sorted({row["species_common_name"] for row in included})
    class_coverage_complete = all(
        species_split_counts[(species_name, split)] > 0
        for species_name in species
        for split in SPLIT_NAMES
    )
    expected_fractions = {
        "train": float(config["train_fraction"]),
        "validation": float(config["validation_fraction"]),
        "test": float(config["test_fraction"]),
    }
    expected_seed = int(config["split_seed"])
    checks = {
        "lock_schema_valid": lock.get("schema_version") == "1.2",
        "manifest_matches_lock": lock.get("source_manifest_sha256") == manifest_sha256,
        "review_lock_valid": review_record.get("ready_for_split") is True,
        "review_lock_matches_manifest": review_record.get("final_manifest_sha256")
        == manifest_sha256,
        "review_lock_matches_split_lock": lock.get("review_lock_sha256") == review_lock_sha256,
        "split_matches_lock": lock.get("split_sha256") == split_sha256,
        "summary_matches_lock": summary_file.is_file()
        and lock.get("summary_sha256") == sha256_file(summary_file),
        "config_matches_lock": lock.get("config_sha256") == config_fingerprint(config),
        "allocator_matches_lock": lock.get("allocator_version") == ALLOCATOR_VERSION,
        "fractions_match_lock": lock.get("split_fractions") == expected_fractions,
        "seed_matches_lock": lock.get("split_seed") == expected_seed,
        "count_matches_lock": lock.get("recordings") == len(split_rows),
        "recording_set_matches_lock": lock.get("recording_set_sha256")
        == sha256_json(sorted(split_ids)),
        "recording_set_exact": included_ids == split_ids,
        "recording_ids_unique": len(split_rows) == len(split_ids),
        "row_bindings_match_manifest": row_bindings_match,
        "zero_session_overlap": all(len(values) == 1 for values in session_to_splits.values()),
        "zero_hash_overlap": all(len(values) == 1 for values in hash_to_splits.values()),
        "included_hashes_unique": len({row["sha256"] for row in split_rows}) == len(split_rows),
        "raw_files_match_manifest": raw_bindings["valid"],
        "raw_files_checked": raw_bindings["recordings_checked"] == len(included),
        "class_coverage_complete": class_coverage_complete,
        "session_reviews_resolved": not any(
            row.get("session_review_flag") == "true" and row.get("local_qc_status") != "exclude"
            for row in manifest_rows
        ),
        "split_values_valid": all(row.get("split") in SPLIT_NAMES for row in split_rows),
        "split_seed_consistent": all(
            row.get("split_seed") == str(expected_seed) for row in split_rows
        ),
        "source_hash_consistent": all(
            row.get("source_manifest_sha256") == manifest_sha256 for row in split_rows
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "recordings": len(split_rows),
        "raw_binding_failures": raw_bindings["failures"],
    }
