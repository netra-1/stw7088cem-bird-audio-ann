from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bird_audio.final_evaluation_gate import (
    _atomic_create_only_bytes,
    _json_bytes,
    _open_absolute_directory_no_follow,
    _publish_directory_no_replace,
    _secure_ensure_directory,
)
from bird_audio.hashing import sha256_json
from bird_audio.paths import PROJECT_ROOT
from bird_audio.provenance import source_fingerprint
from bird_audio.unknown_clip_cache import (
    load_unknown_scoring_clip_cache,
    verify_unknown_clip_cache,
)

RECOVERY_MANIFEST_ID = "final_evaluation_v1_preinference_failure_recovery_v1"
RECOVERY_BUNDLE = (
    PROJECT_ROOT / "evidence" / "recovery" / "final_evaluation_v1_preinference_failure_v1"
)
RECOVERY_MANIFEST_PATH = RECOVERY_BUNDLE / "manifest.json"
RECOVERY_LOCK_PATH = RECOVERY_BUNDLE / "lock.json"
RECOVERY_MANIFEST_SHA256 = "3429ea32f435c0f62393f2deb0a083333e8466de1f03214633b43206afb411ae"
RECOVERY_LOCK_SHA256 = "ac29540edca14ca6d8366cb4a175e8f34fad23693e8991fcb7f7a773e5d23e3d"
V1_SOURCE_FINGERPRINT_SHA256 = "74fa6bc91f37ca87f354761fadbf179807c1785fc1001ccd16f9c6a85c3d926b"

V1_UNKNOWN_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "unknown_clips_v1"
V2_UNKNOWN_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "unknown_clips_v2"
V1_UNKNOWN_CACHE_LOCK_SHA256 = "775b3b4980380d90e5ccf968040b577faec57ff58f36f4daac156384e5a7be1c"
V1_UNKNOWN_CACHE_CONTENT_SHA256 = "69fb250cf3f3270bf79441e7a915659c3bf0260098c8cfdb879b4ecf867e4254"
V1_UNKNOWN_INDEX_SHA256 = "85cc1a23cdb33d0ff7f8ed43a2b15e46b96d51b285c89fc17802721b4c74eb44"
V1_UNKNOWN_FEATURE_SET_SHA256 = "024f024c0d97970ca2ee189f8804bd5ab5facf2be74eb86611b7d6b5acfcefe0"
V1_UNKNOWN_RECORDINGS = 200
V1_UNKNOWN_CLIPS = 843
V1_UNKNOWN_FEATURE_BYTES = 160_586_752

EQUIVALENCE_ID = "unknown_cache_v1_to_v2_equivalence_v1"
EQUIVALENCE_ROOT = PROJECT_ROOT / "evidence" / "recovery" / "final_evaluation_v2_release_v1"
EQUIVALENCE_PATH = EQUIVALENCE_ROOT / "unknown_cache_equivalence.json"
EQUIVALENCE_LOCK_PATH = EQUIVALENCE_ROOT / "lock.json"

_SHA256_LENGTH = 64
_MANIFEST_FIELDS = {
    "schema_version",
    "manifest_id",
    "captured_at_utc",
    "failure_boundary",
    "source_snapshot",
    "unknown_cache_failure",
    "critical_artifacts",
    "run_bindings",
    "protected_trees",
    "inventory_sha256",
}
_PROTECTED_TREE_FIELDS = {
    "root",
    "directory_count",
    "file_count",
    "total_size_bytes",
    "entries",
    "tree_sha256",
}
_CACHE_LOCK_FIELDS = {
    "schema_version",
    "cache_version",
    "provenance",
    "artifacts",
    "cache_content_sha256",
}
_EQUIVALENCE_CERTIFICATE_FIELDS = {
    "schema_version",
    "equivalence_id",
    "certified_at_utc",
    "source_fingerprint_sha256",
    "complete",
    "equivalence",
}


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
        raise ValueError("Recovery JSON is not canonicalizable") from exc


def _read_canonical_json(path: Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{name} is not a regular file")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{name} changed while it was read")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} is not canonical JSON")
    return value, {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _file_record(path: Path) -> tuple[dict[str, Any], tuple[int, int]]:
    before_path = path.lstat()
    if not stat.S_ISREG(before_path.st_mode) or stat.S_ISLNK(before_path.st_mode):
        raise ValueError(f"Recovery artifact is not a regular file: {path}")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        raise RuntimeError("Recovery verification requires O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Recovery artifact changed type: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    identities = {
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        (
            before_path.st_dev,
            before_path.st_ino,
            before_path.st_size,
            before_path.st_mtime_ns,
        ),
        (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
        ),
    }
    if len(identities) != 1:
        raise RuntimeError(f"Recovery artifact changed while hashing: {path}")
    return (
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
        },
        (before.st_dev, before.st_ino),
    )


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Protected recovery root is invalid: {root}")
    paths = sorted((root, *root.rglob("*")), key=lambda path: path.as_posix())
    before_names = [path.relative_to(PROJECT_ROOT).as_posix() for path in paths]
    entries: list[dict[str, Any]] = []
    for path in paths:
        observed = path.lstat()
        label = path.relative_to(PROJECT_ROOT).as_posix()
        mode = f"{stat.S_IMODE(observed.st_mode):04o}"
        if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            entries.append({"kind": "directory", "mode": mode, "path": label})
        elif stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            record, _ = _file_record(path)
            entries.append({"kind": "file", "mode": mode, **record})
        else:
            raise ValueError(f"Protected recovery tree contains an unsafe entry: {path}")
    after_names = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted((root, *root.rglob("*")), key=lambda path: path.as_posix())
    ]
    if before_names != after_names:
        raise RuntimeError(f"Protected recovery tree changed while scanning: {root}")
    return entries


def _validate_artifact_records(records: object, name: str) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{name} is not a sequence")
    validated: list[dict[str, Any]] = []
    for item in records:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256", "size_bytes"}
            or type(item.get("path")) is not str
            or Path(item["path"]).is_absolute()
            or not _is_sha256(item.get("sha256"))
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] < 0
        ):
            raise ValueError(f"{name} contains an invalid record")
        validated.append(dict(item))
    return validated


def _source_fingerprint_from_records(records: Sequence[Mapping[str, Any]]) -> str:
    """Reconstruct fingerprint_files output from a historical file inventory."""
    labels = [str(record["path"]) for record in records]
    if len(labels) != len(set(labels)):
        raise ValueError("Source snapshot contains duplicate paths")
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        label = str(record["path"])
        relative = Path(label)
        if (
            not label
            or relative.is_absolute()
            or relative.as_posix() != label
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("Source snapshot contains an invalid path")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_canonical_utc_timestamp(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == timedelta(0)
        and parsed.isoformat() == value
    )


def verify_v1_recovery_manifest() -> dict[str, Any]:
    """Verify the create-only snapshot of every protected v1 artifact."""
    if (
        not RECOVERY_BUNDLE.is_dir()
        or RECOVERY_BUNDLE.is_symlink()
        or {path.name for path in RECOVERY_BUNDLE.iterdir()} != {"manifest.json", "lock.json"}
    ):
        raise ValueError("The v1 recovery bundle is incomplete")
    lock, lock_record = _read_canonical_json(RECOVERY_LOCK_PATH, "Recovery lock")
    manifest, manifest_record = _read_canonical_json(RECOVERY_MANIFEST_PATH, "Recovery manifest")
    if lock_record["sha256"] != RECOVERY_LOCK_SHA256:
        raise ValueError("The v1 recovery lock differs from its pinned SHA-256")
    if manifest_record["sha256"] != RECOVERY_MANIFEST_SHA256:
        raise ValueError("The v1 recovery manifest differs from its pinned SHA-256")
    if (
        set(lock) != {"schema_version", "manifest_id", "manifest"}
        or lock.get("schema_version") != "1.0"
    ):
        raise ValueError("The v1 recovery lock fields are invalid")
    expected_manifest_reference = {
        "path": "manifest.json",
        "sha256": manifest_record["sha256"],
        "size_bytes": manifest_record["size_bytes"],
    }
    if (
        lock.get("manifest_id") != RECOVERY_MANIFEST_ID
        or lock.get("manifest") != expected_manifest_reference
    ):
        raise ValueError("The v1 recovery lock does not bind its manifest")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema_version") != "1.0"
        or manifest.get("manifest_id") != RECOVERY_MANIFEST_ID
    ):
        raise ValueError("The v1 recovery manifest fields are invalid")
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise ValueError("The v1 source snapshot is invalid")
    source_files = _validate_artifact_records(source_snapshot.get("files"), "Source snapshot")
    reconstructed_source_fingerprint = _source_fingerprint_from_records(source_files)
    if (
        source_snapshot.get("source_fingerprint_sha256") != V1_SOURCE_FINGERPRINT_SHA256
        or reconstructed_source_fingerprint != V1_SOURCE_FINGERPRINT_SHA256
        or source_snapshot.get("source_fingerprint_algorithm")
        != "bird_audio.provenance.source_fingerprint"
        or source_snapshot.get("file_inventory_sha256") != sha256_json(source_files)
    ):
        raise ValueError("The v1 source snapshot identity is invalid")
    critical = _validate_artifact_records(manifest.get("critical_artifacts"), "Critical artifacts")
    for expected in critical:
        observed, _ = _file_record(PROJECT_ROOT / expected["path"])
        if observed != expected:
            raise ValueError(f"Protected v1 artifact changed: {expected['path']}")
    protected = manifest.get("protected_trees")
    if isinstance(protected, (str, bytes)) or not isinstance(protected, Sequence):
        raise ValueError("Protected v1 tree inventory is invalid")
    protected_summary: list[dict[str, Any]] = []
    for expected_tree in protected:
        if not isinstance(expected_tree, Mapping) or set(expected_tree) != _PROTECTED_TREE_FIELDS:
            raise ValueError("Protected v1 tree record is invalid")
        root_label = expected_tree.get("root")
        if type(root_label) is not str or Path(root_label).is_absolute():
            raise ValueError("Protected v1 tree root is invalid")
        entries = _tree_entries(PROJECT_ROOT / root_label)
        files = [entry for entry in entries if entry["kind"] == "file"]
        directories = [entry for entry in entries if entry["kind"] == "directory"]
        observed_tree = {
            "root": root_label,
            "directory_count": len(directories),
            "file_count": len(files),
            "total_size_bytes": sum(int(entry["size_bytes"]) for entry in files),
            "entries": entries,
            "tree_sha256": sha256_json(entries),
        }
        if dict(expected_tree) != observed_tree:
            raise ValueError(f"Protected v1 tree changed: {root_label}")
        protected_summary.append(
            {
                "root": root_label,
                "tree_sha256": observed_tree["tree_sha256"],
                "file_count": observed_tree["file_count"],
            }
        )
    unknown_failure = manifest.get("unknown_cache_failure")
    if (
        not isinstance(unknown_failure, Mapping)
        or unknown_failure.get("v1_cache_lock_sha256") != V1_UNKNOWN_CACHE_LOCK_SHA256
        or unknown_failure.get("v1_cache_content_sha256") != V1_UNKNOWN_CACHE_CONTENT_SHA256
        or unknown_failure.get("v1_index_sha256") != V1_UNKNOWN_INDEX_SHA256
        or unknown_failure.get("v1_feature_set_sha256") != V1_UNKNOWN_FEATURE_SET_SHA256
        or unknown_failure.get("mismatch_fields") != ["implementation_sha256"]
    ):
        raise ValueError("The v1 cache failure identity is invalid")
    expected_inventory_sha256 = sha256_json(
        {
            "source_files": source_files,
            "critical_artifacts": critical,
            "run_bindings": manifest["run_bindings"],
            "unknown_cache_failure": dict(unknown_failure),
            "protected_trees": list(protected),
        }
    )
    if manifest.get("inventory_sha256") != expected_inventory_sha256:
        raise ValueError("The v1 recovery inventory hash is invalid")
    return {
        "valid": True,
        "manifest_id": RECOVERY_MANIFEST_ID,
        "manifest": manifest_record,
        "lock": lock_record,
        "v1_source_fingerprint_sha256": V1_SOURCE_FINGERPRINT_SHA256,
        "protected_trees": protected_summary,
        "failure_boundary": manifest["failure_boundary"],
    }


def _cache_publication(root: Path, version: str) -> dict[str, Any]:
    lock, lock_record = _read_canonical_json(root / "lock.json", f"{version} cache lock")
    summary, summary_record = _read_canonical_json(
        root / "summary.json", f"{version} cache summary"
    )
    if set(lock) != _CACHE_LOCK_FIELDS or lock.get("cache_version") != version:
        raise ValueError(f"{version} cache lock identity is invalid")
    if summary.get("cache_version") != version:
        raise ValueError(f"{version} cache summary identity is invalid")
    index_path = root / "scoring" / "index.csv"
    index_record, index_identity = _file_record(index_path)
    features_root = root / "scoring" / "features"
    if not features_root.is_dir() or features_root.is_symlink():
        raise ValueError(f"{version} feature root is invalid")
    features: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for path in sorted(features_root.iterdir(), key=lambda item: item.name):
        record, identity = _file_record(path)
        if path.suffix != ".npy" or identity in identities:
            raise ValueError(f"{version} feature inventory is invalid")
        identities.add(identity)
        features.append(
            {
                "path": path.relative_to(features_root).as_posix(),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "mode": f"{stat.S_IMODE(path.lstat().st_mode):04o}",
            }
        )
    return {
        "root": root.relative_to(PROJECT_ROOT).as_posix(),
        "lock": lock,
        "lock_record": lock_record,
        "summary": summary,
        "summary_record": summary_record,
        "index_record": index_record,
        "index_identity": index_identity,
        "index_bytes": index_path.read_bytes(),
        "features": features,
        "feature_identities": identities,
    }


def verify_unknown_cache_v2_equivalence(
    *,
    ffmpeg: str | Path | None = None,
    full_rederivation: bool = True,
) -> dict[str, Any]:
    """Prove that v2 changes cache identity without changing scoring data."""
    recovery = verify_v1_recovery_manifest()
    if full_rederivation:
        verified_v2 = verify_unknown_clip_cache(V2_UNKNOWN_CACHE_ROOT, ffmpeg=ffmpeg)
    else:
        cache = load_unknown_scoring_clip_cache(V2_UNKNOWN_CACHE_ROOT, ffmpeg=ffmpeg)
        verified_v2 = {
            "valid": True,
            "lock_sha256": cache.lock_sha256,
            "recordings": len({row["candidate_id"] for row in cache.rows}),
            "clips": len(cache),
            "feature_files": len({row["feature_file"] for row in cache.rows}),
        }
    v1 = _cache_publication(V1_UNKNOWN_CACHE_ROOT, "unknown_clips_v1")
    v2 = _cache_publication(V2_UNKNOWN_CACHE_ROOT, "unknown_clips_v2")
    if v1["lock_record"]["sha256"] != V1_UNKNOWN_CACHE_LOCK_SHA256:
        raise ValueError("The preserved v1 unknown cache lock changed")
    if v1["lock"].get("cache_content_sha256") != V1_UNKNOWN_CACHE_CONTENT_SHA256:
        raise ValueError("The preserved v1 unknown cache content identity changed")
    if v1["index_record"]["sha256"] != V1_UNKNOWN_INDEX_SHA256:
        raise ValueError("The preserved v1 unknown index changed")
    if v1["index_bytes"] != v2["index_bytes"]:
        raise ValueError("The v2 unknown index differs from v1")
    if v1["index_identity"] == v2["index_identity"]:
        raise ValueError("The v1 and v2 caches share the index file inode")
    if v1["features"] != v2["features"]:
        raise ValueError("The v2 unknown feature files differ from v1")
    if v1["feature_identities"] & v2["feature_identities"]:
        raise ValueError("The v1 and v2 caches share file inodes")
    v1_feature_set = v1["lock"]["artifacts"]["features"].get("feature_set_sha256")
    v2_feature_set = v2["lock"]["artifacts"]["features"].get("feature_set_sha256")
    if v1_feature_set != V1_UNKNOWN_FEATURE_SET_SHA256 or v2_feature_set != v1_feature_set:
        raise ValueError("The v2 unknown feature-set identity differs from v1")
    normalized_v1_summary = dict(v1["summary"])
    normalized_v2_summary = dict(v2["summary"])
    normalized_v1_summary.pop("cache_version", None)
    normalized_v2_summary.pop("cache_version", None)
    if normalized_v1_summary != normalized_v2_summary:
        raise ValueError("The v2 unknown summary differs from v1")
    v1_provenance = dict(v1["lock"]["provenance"])
    v2_provenance = dict(v2["lock"]["provenance"])
    v1_implementation = v1_provenance.pop("implementation_sha256", None)
    v2_implementation = v2_provenance.pop("implementation_sha256", None)
    if (
        v1_provenance != v2_provenance
        or not _is_sha256(v1_implementation)
        or not _is_sha256(v2_implementation)
        or v1_implementation == v2_implementation
    ):
        raise ValueError("The v2 unknown cache provenance migration is invalid")
    expected_totals = {
        "recordings": V1_UNKNOWN_RECORDINGS,
        "clips": V1_UNKNOWN_CLIPS,
        "feature_files": V1_UNKNOWN_RECORDINGS,
    }
    if any(verified_v2.get(key) != value for key, value in expected_totals.items()):
        raise ValueError("The v2 verifier totals differ from the locked recovery counts")
    if normalized_v2_summary.get("totals", {}).get("feature_bytes") != V1_UNKNOWN_FEATURE_BYTES:
        raise ValueError("The v2 feature byte count differs from v1")
    return {
        "valid": True,
        "full_rederivation": full_rederivation,
        "v1_recovery_manifest_sha256": recovery["manifest"]["sha256"],
        "v1_recovery_lock_sha256": recovery["lock"]["sha256"],
        "v1_cache_lock_sha256": v1["lock_record"]["sha256"],
        "v2_cache_lock_sha256": v2["lock_record"]["sha256"],
        "v1_cache_content_sha256": v1["lock"]["cache_content_sha256"],
        "v2_cache_content_sha256": v2["lock"]["cache_content_sha256"],
        "index_sha256": v2["index_record"]["sha256"],
        "feature_set_sha256": v2_feature_set,
        "normalized_summary_sha256": sha256_json(normalized_v2_summary),
        "v1_implementation_sha256": v1_implementation,
        "v2_implementation_sha256": v2_implementation,
        "recordings": V1_UNKNOWN_RECORDINGS,
        "clips": V1_UNKNOWN_CLIPS,
        "feature_files": V1_UNKNOWN_RECORDINGS,
        "feature_bytes": V1_UNKNOWN_FEATURE_BYTES,
        "scientific_artifacts_identical": True,
        "file_inodes_disjoint": True,
    }


def _publish_equivalence_bundle(value: Mapping[str, Any]) -> None:
    if EQUIVALENCE_ROOT.exists() or EQUIVALENCE_ROOT.is_symlink():
        raise FileExistsError("The v2 cache equivalence bundle already exists")
    payload = _json_bytes(value)
    lock = {
        "schema_version": "1.0",
        "equivalence_id": EQUIVALENCE_ID,
        "equivalence": {
            "path": EQUIVALENCE_PATH.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
    }
    parent = EQUIVALENCE_ROOT.parent
    _secure_ensure_directory(parent, PROJECT_ROOT)
    parent_descriptor = _open_absolute_directory_no_follow(parent)
    staging_name = f".{EQUIVALENCE_ROOT.name}.staging.{secrets.token_hex(16)}"
    staging = parent / staging_name
    published = False
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        _atomic_create_only_bytes(staging / EQUIVALENCE_PATH.name, payload)
        _atomic_create_only_bytes(staging / EQUIVALENCE_LOCK_PATH.name, _json_bytes(lock))
        staging_descriptor = _open_absolute_directory_no_follow(staging)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        _publish_directory_no_replace(
            staging_name,
            EQUIVALENCE_ROOT.name,
            parent_descriptor,
        )
        published = True
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
        if not published and staging.exists():
            shutil.rmtree(staging)


def verify_unknown_cache_v2_equivalence_certificate(
    *,
    ffmpeg: str | Path | None = None,
    full_rederivation: bool = False,
) -> dict[str, Any]:
    if type(full_rederivation) is not bool:
        raise TypeError("full_rederivation must be a bool")
    if (
        not EQUIVALENCE_ROOT.is_dir()
        or EQUIVALENCE_ROOT.is_symlink()
        or {path.name for path in EQUIVALENCE_ROOT.iterdir()}
        != {EQUIVALENCE_PATH.name, EQUIVALENCE_LOCK_PATH.name}
    ):
        raise ValueError("The v2 cache equivalence bundle is incomplete")
    value, value_record = _read_canonical_json(EQUIVALENCE_PATH, "V2 cache equivalence certificate")
    lock, lock_record = _read_canonical_json(EQUIVALENCE_LOCK_PATH, "V2 cache equivalence lock")
    if (
        set(lock) != {"schema_version", "equivalence_id", "equivalence"}
        or lock.get("schema_version") != "1.0"
        or lock.get("equivalence_id") != EQUIVALENCE_ID
        or lock.get("equivalence")
        != {
            "path": EQUIVALENCE_PATH.name,
            "sha256": value_record["sha256"],
            "size_bytes": value_record["size_bytes"],
        }
    ):
        raise ValueError("The v2 cache equivalence lock is invalid")
    current_source_fingerprint = source_fingerprint()
    if (
        set(value) != _EQUIVALENCE_CERTIFICATE_FIELDS
        or value.get("schema_version") != "1.0"
        or value.get("equivalence_id") != EQUIVALENCE_ID
        or not _is_canonical_utc_timestamp(value.get("certified_at_utc"))
        or not _is_sha256(value.get("source_fingerprint_sha256"))
        or value.get("source_fingerprint_sha256") != current_source_fingerprint
        or value.get("complete") is not True
        or not isinstance(value.get("equivalence"), Mapping)
    ):
        raise ValueError("The v2 cache equivalence certificate identity is invalid")
    certified_equivalence = dict(value["equivalence"])
    if certified_equivalence.get("full_rederivation") is not True:
        raise ValueError("The v2 cache equivalence certificate lacks full rederivation")
    observed = verify_unknown_cache_v2_equivalence(
        ffmpeg=ffmpeg,
        full_rederivation=full_rederivation,
    )
    if source_fingerprint() != current_source_fingerprint:
        raise RuntimeError("Source changed while verifying the v2 equivalence certificate")
    if observed.get("full_rederivation") is not full_rederivation:
        raise ValueError("The v2 equivalence verifier reported an invalid derivation mode")
    certified_current = dict(certified_equivalence)
    observed_current = dict(observed)
    certified_current.pop("full_rederivation")
    observed_current.pop("full_rederivation")
    if certified_current != observed_current:
        raise ValueError("The v2 cache equivalence certificate differs from current evidence")
    return {
        "valid": True,
        "equivalence": dict(value),
        "equivalence_artifact": value_record,
        "lock_artifact": lock_record,
        "created": False,
    }


def seal_unknown_cache_v2_equivalence(
    *,
    ffmpeg: str | Path | None = None,
) -> dict[str, Any]:
    """Publish the create-only v1 to v2 scientific equivalence certificate."""
    if EQUIVALENCE_ROOT.exists() or EQUIVALENCE_ROOT.is_symlink():
        return verify_unknown_cache_v2_equivalence_certificate(
            ffmpeg=ffmpeg,
            full_rederivation=False,
        )
    source_before_rederivation = source_fingerprint()
    equivalence = verify_unknown_cache_v2_equivalence(
        ffmpeg=ffmpeg,
        full_rederivation=True,
    )
    source_after_rederivation = source_fingerprint()
    if source_after_rederivation != source_before_rederivation:
        raise RuntimeError("Source changed during full v2 cache equivalence rederivation")
    value = {
        "schema_version": "1.0",
        "equivalence_id": EQUIVALENCE_ID,
        "certified_at_utc": datetime.now(UTC).isoformat(),
        "source_fingerprint_sha256": source_after_rederivation,
        "complete": True,
        "equivalence": equivalence,
    }
    try:
        _publish_equivalence_bundle(value)
    except FileExistsError:
        return verify_unknown_cache_v2_equivalence_certificate(
            ffmpeg=ffmpeg,
            full_rederivation=False,
        )
    verified = verify_unknown_cache_v2_equivalence_certificate(
        ffmpeg=ffmpeg,
        full_rederivation=False,
    )
    return {**verified, "created": True}
