from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, overload

import numpy as np

from bird_audio.clip_cache import (
    _load_cache_metadata as _load_known_cache_metadata,
)
from bird_audio.clip_cache import (
    _load_verified_feature_tensor as _load_known_feature_tensor,
)
from bird_audio.clip_cache import (
    _read_split_index as _read_known_split_index,
)
from bird_audio.clip_cache import (
    _resolve_cache_artifact as _resolve_known_cache_artifact,
)
from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.final_evaluation_gate import (
    EXPECTED_KNOWN_CACHE_LOCK_SHA256,
    EXPECTED_UNKNOWN_CACHE_LOCK_SHA256,
    FINAL_EVALUATION_GATE_ID,
    FINAL_EVALUATION_GATE_LOCK_PATH,
    FINAL_EVALUATION_GATE_PATH,
    KNOWN_CACHE_LOCK_PATH,
    UNKNOWN_CACHE_LOCK_PATH,
    verify_final_evaluation_gate,
)
from bird_audio.hashing import sha256_json
from bird_audio.paths import PROJECT_ROOT, is_relative_to, require_safe_output
from bird_audio.unknown_clip_cache import (
    UnknownScoringClipCache,
    load_unknown_scoring_clip_cache,
)

FINAL_EVALUATION_DATA_SCHEMA_VERSION = "1.0"
FINAL_EVALUATION_ATTEMPT_ID = "final_evaluation_attempt_v2"
FINAL_EVALUATION_ROOT = PROJECT_ROOT / "runs" / "final_evaluation_v2"
FINAL_EVALUATION_CLAIM_PATH = FINAL_EVALUATION_ROOT / "final_evaluation_attempt_v2.json"
FINAL_EVALUATION_ATTEMPT_DIRECTORY = FINAL_EVALUATION_ROOT / "attempt_v2"
KNOWN_CACHE_ROOT = KNOWN_CACHE_LOCK_PATH.parent
UNKNOWN_CACHE_ROOT = UNKNOWN_CACHE_LOCK_PATH.parent
KNOWN_CACHE_LOCK_SHA256 = EXPECTED_KNOWN_CACHE_LOCK_SHA256
UNKNOWN_CACHE_LOCK_SHA256 = EXPECTED_UNKNOWN_CACHE_LOCK_SHA256
KNOWN_TEST_RECORDINGS = 267
KNOWN_TEST_ENERGY_CLIPS = 1153
UNKNOWN_RECORDINGS = 200
UNKNOWN_ENERGY_CLIPS = 843
UNKNOWN_SPECIES = 5
UNKNOWN_RECORDINGS_PER_SPECIES = 40
STAGE_ORDER = (
    "task1_seed_13",
    "task1_seed_37",
    "task1_seed_71",
    "task2_seed_13",
    "task2_seed_37",
    "task2_seed_71",
    "summary",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAIM_FIELDS = {
    "schema_version",
    "attempt_id",
    "claimed_at_utc",
    "gate_id",
    "gate",
    "gate_lock",
    "attempt_directory",
    "stage_order",
    "single_attempt",
}


def _json_bytes(value: Any) -> bytes:
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
        raise ValueError("Final evaluation JSON value is not serializable") from exc


def _secure_open_flags() -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if not isinstance(nofollow, int) or nofollow == 0:
        raise RuntimeError("Final evaluation requires O_NOFOLLOW support")
    if not isinstance(directory_flag, int) or directory_flag == 0:
        raise RuntimeError("Final evaluation requires O_DIRECTORY support")
    return nofollow, directory_flag


def _lexical_project_path(path: Path) -> tuple[Path, Path, Path]:
    candidate = Path(os.path.abspath(path.expanduser()))
    project_root = Path(os.path.abspath(PROJECT_ROOT))
    if not is_relative_to(candidate, project_root):
        raise ValueError(f"Final evaluation artifact leaves the project: {candidate}")
    relative = candidate.relative_to(project_root)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Final evaluation artifact path is not canonical: {candidate}")
    return candidate, project_root, relative


def _open_project_directory(directory: Path) -> int:
    candidate, project_root, relative = _lexical_project_path(directory)
    nofollow, directory_flag = _secure_open_flags()
    directory_flags = os.O_RDONLY | nofollow | directory_flag
    try:
        parent_descriptor = os.open(project_root, directory_flags)
    except OSError as exc:
        raise PermissionError("Final evaluation project root cannot be opened safely") from exc
    try:
        root_status = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(root_status.st_mode):
            raise PermissionError("Final evaluation project root is not a directory")
        for component in relative.parts:
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                raise PermissionError(
                    f"Final evaluation artifact parent cannot be opened safely: {candidate}"
                ) from exc
            child_status = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_status.st_mode):
                os.close(child_descriptor)
                raise PermissionError(
                    f"Final evaluation directory path is not a directory: {candidate}"
                )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor
        return parent_descriptor
    except BaseException:
        os.close(parent_descriptor)
        raise


def _descriptor_snapshot(path: Path) -> tuple[bytes, str, int]:
    candidate, _project_root, relative = _lexical_project_path(path)
    if not relative.parts:
        raise ValueError("Final evaluation artifact path names the project directory")
    nofollow, _directory_flag = _secure_open_flags()
    parent_descriptor = _open_project_directory(candidate.parent)
    try:
        try:
            descriptor = os.open(
                candidate.name,
                os.O_RDONLY | nofollow,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise FileNotFoundError(
                f"Final evaluation artifact cannot be opened safely: {candidate}"
            ) from exc
    finally:
        os.close(parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Final evaluation artifact is not a regular file: {candidate}")
        parts: list[bytes] = []
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            parts.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(f"Final evaluation artifact changed while read: {candidate}")
        payload = b"".join(parts)
        if len(payload) != before.st_size:
            raise ValueError(f"Final evaluation artifact size changed: {candidate}")
        return payload, digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def _artifact_record(path: Path) -> dict[str, Any]:
    _, digest, size_bytes = _descriptor_snapshot(path)
    return {
        "path": str(Path(os.path.abspath(path.expanduser()))),
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, size_bytes = _descriptor_snapshot(path)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Final evaluation JSON is invalid: {path}") from exc
    if not isinstance(value, dict) or _json_bytes(value) != payload:
        raise ValueError(f"Final evaluation JSON is not canonical: {path}")
    return value, {
        "path": str(Path(os.path.abspath(path.expanduser()))),
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = _open_project_directory(directory)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_create_only(path: Path, value: Any) -> dict[str, Any]:
    payload = _json_bytes(value)
    requested, _project_root, relative = _lexical_project_path(path)
    if not relative.parts:
        raise ValueError("Final evaluation output path names the project directory")
    destination = require_safe_output(requested)
    if Path(os.path.abspath(destination)) != requested:
        raise ValueError("Final evaluation output path is not direct and canonical")
    parent_descriptor = _open_project_directory(destination.parent)
    temporary_name = f".{destination.name}.{secrets.token_hex(16)}.tmp"
    nofollow, _directory_flag = _secure_open_flags()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary_name,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)
    observed, record = _read_json_snapshot(destination)
    if observed != value:
        raise RuntimeError("Final evaluation claim failed publication verification")
    return record


def _validate_artifact_record(value: object, expected_path: Path) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "size_bytes"}
        or value.get("path") != str(expected_path.resolve())
        or _SHA256.fullmatch(str(value.get("sha256") or "")) is None
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
    ):
        raise ValueError("Final evaluation claim artifact record is invalid")
    observed = _artifact_record(expected_path)
    if value != observed:
        raise ValueError("Final evaluation claim artifact binding changed")
    return observed


def _validate_gate_for_data(gate: object) -> dict[str, Any]:
    if not isinstance(gate, dict) or gate.get("ready") is not True:
        raise PermissionError("Final evaluation gate is not ready")
    if gate.get("gate_id") != FINAL_EVALUATION_GATE_ID:
        raise ValueError("Final evaluation gate ID is invalid")
    if gate.get("seed_order") != [13, 37, 71]:
        raise ValueError("Final evaluation gate seed order is invalid")
    cache_locks = gate.get("cache_locks")
    if not isinstance(cache_locks, dict) or set(cache_locks) != {"known", "unknown"}:
        raise ValueError("Final evaluation cache lock bindings are invalid")
    expected = {
        "known": (
            KNOWN_CACHE_LOCK_SHA256,
            "known_clips_v1",
            str(KNOWN_CACHE_LOCK_PATH.resolve()),
        ),
        "unknown": (
            UNKNOWN_CACHE_LOCK_SHA256,
            "unknown_clips_v2",
            str(UNKNOWN_CACHE_LOCK_PATH.resolve()),
        ),
    }
    for name, (expected_sha256, expected_version, expected_path) in expected.items():
        record = cache_locks.get(name)
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "path",
                "sha256",
                "size_bytes",
                "cache_version",
                "cache_content_sha256",
                "requirements_lock_sha256",
            }
            or record.get("path") != expected_path
            or record.get("sha256") != expected_sha256
            or record.get("cache_version") != expected_version
            or _SHA256.fullmatch(str(record.get("cache_content_sha256") or "")) is None
            or _SHA256.fullmatch(str(record.get("requirements_lock_sha256") or "")) is None
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError(f"Final evaluation {name} cache binding is invalid")
    shared = gate.get("shared_identity")
    if (
        not isinstance(shared, dict)
        or set(shared)
        != {
            "known_cache_lock_sha256",
            "known_cache_content_sha256",
            "unknown_cache_lock_sha256",
            "unknown_cache_content_sha256",
            "requirements_lock_sha256",
            "source_fingerprint_sha256",
        }
        or shared.get("known_cache_lock_sha256") != KNOWN_CACHE_LOCK_SHA256
        or shared.get("unknown_cache_lock_sha256") != UNKNOWN_CACHE_LOCK_SHA256
        or shared.get("known_cache_content_sha256") != cache_locks["known"]["cache_content_sha256"]
        or shared.get("unknown_cache_content_sha256")
        != cache_locks["unknown"]["cache_content_sha256"]
        or _SHA256.fullmatch(str(shared.get("requirements_lock_sha256") or "")) is None
        or _SHA256.fullmatch(str(shared.get("source_fingerprint_sha256") or "")) is None
        or sha256_json(shared) != gate.get("shared_identity_sha256")
    ):
        raise ValueError("Final evaluation shared identity is invalid")
    return gate


def _validate_claim(
    claim: object,
    *,
    gate_record: dict[str, Any],
    gate_lock_record: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(claim, dict) or set(claim) != _CLAIM_FIELDS:
        raise ValueError("Final evaluation attempt claim fields are not exact")
    if (
        claim.get("schema_version") != FINAL_EVALUATION_DATA_SCHEMA_VERSION
        or claim.get("attempt_id") != FINAL_EVALUATION_ATTEMPT_ID
        or claim.get("gate_id") != FINAL_EVALUATION_GATE_ID
        or claim.get("gate") != gate_record
        or claim.get("gate_lock") != gate_lock_record
        or claim.get("attempt_directory")
        != FINAL_EVALUATION_ATTEMPT_DIRECTORY.relative_to(PROJECT_ROOT).as_posix()
        or claim.get("stage_order") != list(STAGE_ORDER)
        or claim.get("single_attempt") is not True
    ):
        raise ValueError("Final evaluation attempt claim binding is invalid")
    timestamp = claim.get("claimed_at_utc")
    if type(timestamp) is not str:
        raise ValueError("Final evaluation claim timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Final evaluation claim timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Final evaluation claim timestamp must use UTC")
    return claim


@dataclass(frozen=True, slots=True)
class FinalEvaluationAuthorization:
    gate_sha256: str
    gate_lock_sha256: str
    claim_sha256: str
    attempt_directory: Path

    def __post_init__(self) -> None:
        if any(
            _SHA256.fullmatch(value) is None
            for value in (self.gate_sha256, self.gate_lock_sha256, self.claim_sha256)
        ):
            raise ValueError("Final evaluation authorization contains an invalid SHA-256")
        if self.attempt_directory != FINAL_EVALUATION_ATTEMPT_DIRECTORY:
            raise ValueError("Final evaluation authorization uses another attempt directory")


def _require_authorization_current(
    authorization: FinalEvaluationAuthorization,
) -> None:
    if not isinstance(authorization, FinalEvaluationAuthorization):
        raise TypeError("A final evaluation authorization is required")
    gate_record = _artifact_record(FINAL_EVALUATION_GATE_PATH)
    gate_lock_record = _artifact_record(FINAL_EVALUATION_GATE_LOCK_PATH)
    claim, claim_record = _read_json_snapshot(FINAL_EVALUATION_CLAIM_PATH)
    if (
        gate_record["sha256"] != authorization.gate_sha256
        or gate_lock_record["sha256"] != authorization.gate_lock_sha256
        or claim_record["sha256"] != authorization.claim_sha256
    ):
        raise PermissionError("Final evaluation authorization artifacts changed")
    _validate_claim(
        claim,
        gate_record=gate_record,
        gate_lock_record=gate_lock_record,
    )
    _require_attempt_directory()


def _require_attempt_directory() -> None:
    try:
        descriptor = _open_project_directory(FINAL_EVALUATION_ATTEMPT_DIRECTORY)
    except (OSError, RuntimeError, ValueError, PermissionError) as exc:
        raise PermissionError("Final evaluation attempt directory is invalid") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PermissionError("Final evaluation attempt directory is invalid")
    finally:
        os.close(descriptor)


def _ensure_attempt_directory() -> None:
    directory, _project_root, relative = _lexical_project_path(FINAL_EVALUATION_ATTEMPT_DIRECTORY)
    if not relative.parts:
        raise PermissionError("Final evaluation attempt directory is invalid")
    parent_descriptor = _open_project_directory(directory.parent)
    nofollow, directory_flag = _secure_open_flags()
    flags = os.O_RDONLY | nofollow | directory_flag
    try:
        with suppress(FileExistsError):
            os.mkdir(directory.name, mode=0o700, dir_fd=parent_descriptor)
        try:
            child_descriptor = os.open(
                directory.name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise PermissionError("Final evaluation attempt directory is invalid") from exc
        try:
            if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
                raise PermissionError("Final evaluation attempt directory is invalid")
        finally:
            os.close(child_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _competing_attempt_names() -> tuple[str, ...]:
    descriptor = _open_project_directory(FINAL_EVALUATION_ROOT)
    try:
        names = os.listdir(descriptor)
    finally:
        os.close(descriptor)
    expected = FINAL_EVALUATION_ATTEMPT_DIRECTORY.name
    return tuple(
        sorted(name for name in names if name.startswith("attempt_v") and name != expected)
    )


def claim_final_evaluation_attempt() -> dict[str, Any]:
    verified = verify_final_evaluation_gate()
    gate = _validate_gate_for_data(verified.get("gate"))
    gate_record = _artifact_record(FINAL_EVALUATION_GATE_PATH)
    gate_lock_record = _artifact_record(FINAL_EVALUATION_GATE_LOCK_PATH)
    if (
        verified.get("gate_artifact") != gate_record
        or verified.get("lock_artifact") != gate_lock_record
    ):
        raise ValueError("Final evaluation gate return records are inconsistent")

    competing = _competing_attempt_names()
    if competing:
        raise PermissionError("Another final evaluation attempt directory exists")

    created = False
    if FINAL_EVALUATION_CLAIM_PATH.exists() or FINAL_EVALUATION_CLAIM_PATH.is_symlink():
        claim, claim_record = _read_json_snapshot(FINAL_EVALUATION_CLAIM_PATH)
        _validate_claim(
            claim,
            gate_record=gate_record,
            gate_lock_record=gate_lock_record,
        )
    else:
        if FINAL_EVALUATION_ATTEMPT_DIRECTORY.exists():
            raise PermissionError("Final evaluation attempt directory exists without its claim")
        claim = {
            "schema_version": FINAL_EVALUATION_DATA_SCHEMA_VERSION,
            "attempt_id": FINAL_EVALUATION_ATTEMPT_ID,
            "claimed_at_utc": datetime.now(UTC).isoformat(),
            "gate_id": FINAL_EVALUATION_GATE_ID,
            "gate": gate_record,
            "gate_lock": gate_lock_record,
            "attempt_directory": FINAL_EVALUATION_ATTEMPT_DIRECTORY.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "stage_order": list(STAGE_ORDER),
            "single_attempt": True,
        }
        try:
            claim_record = _write_json_create_only(FINAL_EVALUATION_CLAIM_PATH, claim)
            created = True
        except FileExistsError:
            claim, claim_record = _read_json_snapshot(FINAL_EVALUATION_CLAIM_PATH)
            _validate_claim(
                claim,
                gate_record=gate_record,
                gate_lock_record=gate_lock_record,
            )
    if created:
        _ensure_attempt_directory()
    else:
        _require_attempt_directory()
    _require_attempt_directory()
    authorization = FinalEvaluationAuthorization(
        gate_sha256=gate_record["sha256"],
        gate_lock_sha256=gate_lock_record["sha256"],
        claim_sha256=claim_record["sha256"],
        attempt_directory=FINAL_EVALUATION_ATTEMPT_DIRECTORY,
    )
    _require_authorization_current(authorization)
    return {
        "authorization": authorization,
        "gate": gate,
        "claim": claim,
        "claim_artifact": claim_record,
        "created": created,
    }


class FinalKnownTestData(Sequence[tuple[np.ndarray, dict[str, str]]]):
    def __init__(
        self,
        authorization: FinalEvaluationAuthorization,
        *,
        ffmpeg: str | Path | None = None,
    ) -> None:
        _require_authorization_current(authorization)
        root, lock, summary, current = _load_known_cache_metadata(
            KNOWN_CACHE_ROOT,
            ffmpeg=ffmpeg,
            expected_lock_sha256=KNOWN_CACHE_LOCK_SHA256,
        )
        class_indices = {
            str(entry["common_name"]): index
            for index, entry in enumerate(current["config"]["known_species"])
        }
        if tuple(class_indices) != tuple(LOCKED_TASK1_CLASS_ORDER):
            raise ValueError("Final known-test class order changed")
        rows, statistics = _read_known_split_index(
            root,
            "test",
            lock["artifacts"]["splits"]["test"],
            class_indices,
            verify_feature_bytes=True,
        )
        if summary.get("splits", {}).get("test") != statistics:
            raise ValueError("Final known-test summary differs from its index")
        selected = tuple(row for row in rows if row["energy_selected"] == "true")
        recording_ids = tuple(dict.fromkeys(row["recording_id"] for row in selected))
        if (
            len(selected) != KNOWN_TEST_ENERGY_CLIPS
            or len(recording_ids) != KNOWN_TEST_RECORDINGS
            or statistics.get("recordings") != KNOWN_TEST_RECORDINGS
            or statistics.get("energy_memberships") != KNOWN_TEST_ENERGY_CLIPS
        ):
            raise ValueError("Final known-test counts differ from the locked protocol")
        self.authorization = authorization
        self.root = root
        self.split = "test"
        self.strategy = "energy"
        self.lock_sha256 = KNOWN_CACHE_LOCK_SHA256
        self.recording_count = len(recording_ids)
        self.recording_ids = recording_ids
        self._rows = selected
        self._recording_indices = self._build_recording_indices()
        self._loaded_feature_path: Path | None = None
        self._loaded_feature_tensor: np.ndarray | None = None

    def _build_recording_indices(self) -> dict[str, tuple[int, ...]]:
        positions: dict[str, list[int]] = {}
        for index, row in enumerate(self._rows):
            positions.setdefault(row["recording_id"], []).append(index)
        return {key: tuple(value) for key, value in positions.items()}

    @staticmethod
    def _metadata(row: dict[str, str]) -> dict[str, str]:
        return {
            **row,
            "selection_strategy": "energy",
            "strategy_clip_count": row["energy_clip_count"],
            "data_boundary": "gated_final_known_test",
        }

    def _load_feature(self, row: dict[str, str]) -> np.ndarray:
        _require_authorization_current(self.authorization)
        feature_path = _resolve_known_cache_artifact(
            self.root,
            row["feature_file"],
            row["feature_file"],
        )
        if feature_path != self._loaded_feature_path:
            self._loaded_feature_tensor = _load_known_feature_tensor(
                feature_path,
                row["feature_file_sha256"],
            )
            self._loaded_feature_path = feature_path
        if self._loaded_feature_tensor is None:
            raise RuntimeError("Final known-test feature tensor was not loaded")
        return self._loaded_feature_tensor[int(row["feature_row"])].copy()

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[np.ndarray, dict[str, str]]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[np.ndarray, dict[str, str]] | list[tuple[np.ndarray, dict[str, str]]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        row = self._rows[index]
        return self._load_feature(row), self._metadata(row)

    def metadata(self, index: int) -> dict[str, str]:
        _require_authorization_current(self.authorization)
        return self._metadata(self._rows[index])

    def iter_metadata(self) -> Iterator[dict[str, str]]:
        _require_authorization_current(self.authorization)
        for row in self._rows:
            yield self._metadata(row)

    def iter_recording_indices(self) -> Iterator[tuple[str, tuple[int, ...]]]:
        yield from self._recording_indices.items()

    def get_recording(
        self,
        recording_id: str,
    ) -> tuple[np.ndarray, tuple[dict[str, str], ...]]:
        try:
            indices = self._recording_indices[recording_id]
        except KeyError as exc:
            raise KeyError(f"Unknown final known-test recording ID: {recording_id}") from exc
        samples = [self[index] for index in indices]
        return (
            np.stack([sample[0] for sample in samples]).astype(np.float32, copy=False),
            tuple(sample[1] for sample in samples),
        )


class FinalUnknownData(Sequence[tuple[np.ndarray, dict[str, str]]]):
    def __init__(
        self,
        authorization: FinalEvaluationAuthorization,
        *,
        ffmpeg: str | Path | None = None,
    ) -> None:
        _require_authorization_current(authorization)
        source = load_unknown_scoring_clip_cache(
            UNKNOWN_CACHE_ROOT,
            ffmpeg=ffmpeg,
            expected_lock_sha256=UNKNOWN_CACHE_LOCK_SHA256,
        )
        recording_ids = tuple(dict.fromkeys(row["candidate_id"] for row in source.rows))
        species_by_recording: dict[str, str] = {}
        for row in source.rows:
            recording_id = row["candidate_id"]
            species = row["species_scientific_name"]
            existing = species_by_recording.setdefault(recording_id, species)
            if existing != species:
                raise ValueError("Final unknown recording changes scientific species")
        species_counts = Counter(species_by_recording.values())
        if (
            len(source) != UNKNOWN_ENERGY_CLIPS
            or len(recording_ids) != UNKNOWN_RECORDINGS
            or len(species_counts) != UNKNOWN_SPECIES
            or set(species_counts.values()) != {UNKNOWN_RECORDINGS_PER_SPECIES}
        ):
            raise ValueError("Final unknown counts differ from the locked protocol")
        self.authorization = authorization
        self.root = source.root
        self.split = "unknown"
        self.strategy = "energy"
        self.lock_sha256 = UNKNOWN_CACHE_LOCK_SHA256
        self.recording_count = len(recording_ids)
        self.recording_ids = recording_ids
        self.species_scientific_names = tuple(sorted(species_counts))
        self._source: UnknownScoringClipCache = source
        self._rows = tuple(source.rows)
        self._recording_indices = self._build_recording_indices()

    def _build_recording_indices(self) -> dict[str, tuple[int, ...]]:
        positions: dict[str, list[int]] = {}
        for index, row in enumerate(self._rows):
            positions.setdefault(row["candidate_id"], []).append(index)
        return {key: tuple(value) for key, value in positions.items()}

    @staticmethod
    def _metadata(row: dict[str, str]) -> dict[str, str]:
        return {
            **row,
            "recording_id": row["candidate_id"],
            "selection_strategy": "energy",
            "strategy_clip_count": row["energy_clip_count"],
            "data_boundary": "gated_final_unknown",
        }

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[np.ndarray, dict[str, str]]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[np.ndarray, dict[str, str]] | list[tuple[np.ndarray, dict[str, str]]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        _require_authorization_current(self.authorization)
        feature, _ = self._source[index]
        return feature, self._metadata(self._rows[index])

    def metadata(self, index: int) -> dict[str, str]:
        _require_authorization_current(self.authorization)
        return self._metadata(self._rows[index])

    def iter_metadata(self) -> Iterator[dict[str, str]]:
        _require_authorization_current(self.authorization)
        for row in self._rows:
            yield self._metadata(row)

    def iter_recording_indices(self) -> Iterator[tuple[str, tuple[int, ...]]]:
        yield from self._recording_indices.items()

    def get_recording(
        self,
        recording_id: str,
    ) -> tuple[np.ndarray, tuple[dict[str, str], ...]]:
        try:
            indices = self._recording_indices[recording_id]
        except KeyError as exc:
            raise KeyError(f"Unknown final unknown recording ID: {recording_id}") from exc
        samples = [self[index] for index in indices]
        return (
            np.stack([sample[0] for sample in samples]).astype(np.float32, copy=False),
            tuple(sample[1] for sample in samples),
        )


def open_final_known_test_data(
    authorization: FinalEvaluationAuthorization,
    *,
    ffmpeg: str | Path | None = None,
) -> FinalKnownTestData:
    return FinalKnownTestData(authorization, ffmpeg=ffmpeg)


def open_final_unknown_data(
    authorization: FinalEvaluationAuthorization,
    *,
    ffmpeg: str | Path | None = None,
) -> FinalUnknownData:
    return FinalUnknownData(authorization, ffmpeg=ffmpeg)
