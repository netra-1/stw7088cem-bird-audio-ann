from __future__ import annotations

import hashlib
import math
import mmap
import os
import re
import stat
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
from numpy.lib import format as npy_format

from bird_audio.clip_cache import (
    DEFAULT_CACHE_ROOT,
    NATIVE_FEATURE_SHAPE,
    load_development_clip_cache,
)
from bird_audio.paths import PROJECT_ROOT, is_relative_to, resolve_project_path

DEVELOPMENT_SPLITS = ("train", "validation")
SELECTION_STRATEGIES = ("uniform", "energy")
DEFAULT_MMAP_CAPACITY = 64
UNKNOWN_DATA_ROOT = PROJECT_ROOT / "data" / "unknown"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
_UNKNOWN_PROCESSED_CACHE_NAME = re.compile(r"unknown_clips_v[1-9][0-9]*")


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modification_time_ns: int
    change_time_ns: int


@dataclass(frozen=True)
class _VerifiedFeatureFile:
    path: Path
    sha256: str
    rows: int
    identity: _FileIdentity


@dataclass(frozen=True)
class _MappedFeatureFile:
    tensor: np.ndarray
    backing: mmap.mmap


def _identity(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        modification_time_ns=value.st_mtime_ns,
        change_time_ns=value.st_ctime_ns,
    )


def _path_identity(path: Path) -> _FileIdentity:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f"Verified feature path became a symbolic link: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Verified feature path is no longer a regular file: {path}")
    return _identity(value)


def _close_memory_map(value: _MappedFeatureFile) -> None:
    value.backing.close()


def _is_unknown_data_root(path: Path) -> bool:
    if is_relative_to(path, UNKNOWN_DATA_ROOT):
        return True
    if not is_relative_to(path, PROCESSED_DATA_ROOT):
        return False
    relative = path.relative_to(PROCESSED_DATA_ROOT)
    return bool(relative.parts and _UNKNOWN_PROCESSED_CACHE_NAME.fullmatch(relative.parts[0]))


def _npy_layout_from_descriptor(
    descriptor: int,
    path: Path,
) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Training feature uses an unsupported NPY version: {path}")
        data_offset = handle.tell()
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.hasobject:
        raise ValueError(f"Training feature cannot contain object data: {path}")
    return tuple(shape), bool(fortran_order), resolved_dtype, data_offset


def _open_verified_memory_map(record: _VerifiedFeatureFile) -> _MappedFeatureFile:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("Descriptor-bound training reads require O_NOFOLLOW")
    flags |= no_follow
    try:
        descriptor = os.open(record.path, flags)
    except OSError as exc:
        raise ValueError(
            f"Training feature cannot be opened without following links: {record.path}"
        ) from exc

    backing: mmap.mmap | None = None
    try:
        initial_stat = os.fstat(descriptor)
        initial_identity = _identity(initial_stat)
        if not stat.S_ISREG(initial_stat.st_mode) or initial_identity != record.identity:
            raise ValueError(f"Training feature changed before descriptor mapping: {record.path}")
        shape, fortran_order, dtype, data_offset = _npy_layout_from_descriptor(
            descriptor,
            record.path,
        )
        if (
            dtype != np.dtype(np.float32)
            or len(shape) != 4
            or tuple(shape[1:]) != NATIVE_FEATURE_SHAPE
            or shape[0] != record.rows
        ):
            raise ValueError(f"Training descriptor NPY contract is invalid: {record.path}")
        expected_size = data_offset + math.prod(shape) * dtype.itemsize
        if (
            expected_size != initial_identity.size
            or _identity(os.fstat(descriptor)) != record.identity
        ):
            raise ValueError(f"Training descriptor NPY contract is invalid: {record.path}")

        backing = mmap.mmap(descriptor, length=0, access=mmap.ACCESS_READ)
        tensor = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=backing,
            offset=data_offset,
            order="F" if fortran_order else "C",
        )
        if _identity(os.fstat(descriptor)) != record.identity:
            raise ValueError(f"Training feature changed while descriptor was mapped: {record.path}")
        if tensor.flags.writeable:
            raise ValueError(f"Training descriptor map is unexpectedly writeable: {record.path}")
    except BaseException:
        if backing is not None:
            backing.close()
        raise
    finally:
        os.close(descriptor)
    return _MappedFeatureFile(tensor=tensor, backing=backing)


def _resolve_feature_path(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"Training feature path is invalid: {relative_path}")
    path = (root / relative_path).resolve(strict=True)
    if not is_relative_to(path, root) or path.relative_to(root).as_posix() != relative_path:
        raise ValueError(f"Training feature path leaves its locked cache: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Training feature file is missing: {relative_path}")
    return path


def _verify_feature_file(
    path: Path,
    expected_sha256: str,
    expected_rows: int,
) -> _VerifiedFeatureFile:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        initial_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError(f"Training feature is not a regular file: {path}")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Training feature hash drift: {path}")
        handle.seek(0)
        tensor = np.load(handle, allow_pickle=False)
        final_stat = os.fstat(handle.fileno())

    initial_identity = _identity(initial_stat)
    if initial_identity != _identity(final_stat) or initial_identity != _path_identity(path):
        raise ValueError(f"Training feature changed during verification: {path}")
    if (
        tensor.dtype != np.float32
        or tensor.ndim != 4
        or tuple(tensor.shape[1:]) != NATIVE_FEATURE_SHAPE
        or tensor.shape[0] != expected_rows
        or not bool(np.all(np.isfinite(tensor)))
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise ValueError(f"Training feature tensor contract is invalid: {path}")
    return _VerifiedFeatureFile(
        path=path,
        sha256=expected_sha256,
        rows=expected_rows,
        identity=initial_identity,
    )


class DevelopmentTrainingData(Sequence[tuple[np.ndarray, dict[str, str]]]):
    """Verified read-only native features for training and validation."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        strategy: str,
        *,
        ffmpeg: str | Path | None = None,
        expected_lock_sha256: str | None = None,
        mmap_capacity: int = DEFAULT_MMAP_CAPACITY,
    ) -> None:
        if split == "test":
            raise PermissionError("Development training data cannot open the final test split")
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError(f"Development split must be one of {DEVELOPMENT_SPLITS}")
        if strategy not in SELECTION_STRATEGIES:
            raise ValueError(f"Selection strategy must be one of {SELECTION_STRATEGIES}")
        if isinstance(mmap_capacity, bool) or not isinstance(mmap_capacity, int):
            raise TypeError("mmap_capacity must be an integer")
        if mmap_capacity <= 0:
            raise ValueError("mmap_capacity must be positive")

        requested_root = resolve_project_path(cache_root)
        if _is_unknown_data_root(requested_root):
            raise PermissionError("Development training data accepts only the locked known cache")

        source = load_development_clip_cache(
            requested_root,
            split,
            strategy,
            ffmpeg=ffmpeg,
            expected_lock_sha256=expected_lock_sha256,
        )
        if _is_unknown_data_root(source.root):
            raise PermissionError("Development training data accepts only the locked known cache")

        self.root = source.root
        self.split = split
        self.strategy = strategy
        self.lock_sha256 = source.lock_sha256
        self.mmap_capacity = mmap_capacity
        self._rows = tuple(dict(row) for row in source.rows)
        self._recording_indices = self._build_recording_indices()
        self.recording_ids = tuple(self._recording_indices)
        self._verified_files = self._verify_files_once()
        self._memory_maps: OrderedDict[str, _MappedFeatureFile] = OrderedDict()
        self._closed = False

    def _build_recording_indices(self) -> dict[str, tuple[int, ...]]:
        positions: dict[str, list[int]] = {}
        previous_key: tuple[int, str, int] | None = None
        for index, row in enumerate(self._rows):
            key = (
                int(row["class_index"]),
                row["recording_id"],
                int(row[f"{self.strategy}_rank"]),
            )
            if previous_key is not None and key <= previous_key:
                raise ValueError("Development strategy rows are not in deterministic rank order")
            previous_key = key
            positions.setdefault(row["recording_id"], []).append(index)
        return {recording_id: tuple(indices) for recording_id, indices in positions.items()}

    def _verify_files_once(self) -> dict[str, _VerifiedFeatureFile]:
        expected: dict[str, tuple[str, int]] = {}
        for row in self._rows:
            relative_path = row["feature_file"]
            binding = (row["feature_file_sha256"], int(row["cached_clip_count"]))
            existing = expected.get(relative_path)
            if existing is not None and existing != binding:
                raise ValueError(f"Training feature binding is inconsistent: {relative_path}")
            expected[relative_path] = binding

        verified: dict[str, _VerifiedFeatureFile] = {}
        for relative_path in sorted(expected):
            expected_sha256, expected_rows = expected[relative_path]
            path = _resolve_feature_path(self.root, relative_path)
            verified[relative_path] = _verify_feature_file(
                path,
                expected_sha256,
                expected_rows,
            )
        return verified

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def recording_count(self) -> int:
        return len(self.recording_ids)

    @property
    def verified_feature_files(self) -> int:
        return len(self._verified_files)

    @property
    def open_recording_count(self) -> int:
        return len(self._memory_maps)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Development training data is closed")

    def _require_unchanged(self, relative_path: str) -> _VerifiedFeatureFile:
        record = self._verified_files[relative_path]
        if _path_identity(record.path) != record.identity:
            raise ValueError(f"Training feature changed after verification: {record.path}")
        return record

    def _memory_map(self, relative_path: str) -> np.ndarray:
        self._require_open()
        record = self._require_unchanged(relative_path)
        existing = self._memory_maps.get(relative_path)
        if existing is not None:
            self._memory_maps.move_to_end(relative_path)
            return existing.tensor

        mapped = _open_verified_memory_map(record)
        tensor = mapped.tensor
        try:
            if (
                tensor.dtype != np.float32
                or tensor.ndim != 4
                or tuple(tensor.shape[1:]) != NATIVE_FEATURE_SHAPE
                or tensor.shape[0] != record.rows
                or tensor.flags.writeable
                or _path_identity(record.path) != record.identity
            ):
                raise ValueError(f"Training memory map contract is invalid: {record.path}")
        except BaseException:
            _close_memory_map(mapped)
            raise

        self._memory_maps[relative_path] = mapped
        if len(self._memory_maps) > self.mmap_capacity:
            _, evicted = self._memory_maps.popitem(last=False)
            _close_memory_map(evicted)
        return tensor

    def _metadata(self, row: dict[str, str]) -> dict[str, str]:
        metadata = dict(row)
        metadata["selection_strategy"] = self.strategy
        metadata["strategy_clip_count"] = row[f"{self.strategy}_clip_count"]
        return metadata

    def metadata(self, index: int) -> dict[str, str]:
        """Return copied metadata without opening the recording feature file."""
        self._require_open()
        return self._metadata(self._rows[index])

    def iter_metadata(self) -> Iterator[dict[str, str]]:
        """Yield copied metadata in the exact deterministic sample order."""
        self._require_open()
        for row in self._rows:
            self._require_open()
            yield self._metadata(row)

    def _copy_rows(self, rows: Sequence[dict[str, str]]) -> np.ndarray:
        if not rows:
            raise ValueError("At least one training row is required")
        relative_path = rows[0]["feature_file"]
        if any(row["feature_file"] != relative_path for row in rows):
            raise ValueError("Recording rows reference more than one training feature file")
        tensor = self._memory_map(relative_path)
        record = self._require_unchanged(relative_path)
        feature_rows = [int(row["feature_row"]) for row in rows]
        copied = np.ascontiguousarray(tensor[feature_rows], dtype=np.float32)
        if _path_identity(record.path) != record.identity:
            raise ValueError(f"Training feature changed while being read: {record.path}")
        expected_shape = (len(rows), *NATIVE_FEATURE_SHAPE)
        if copied.shape != expected_shape:
            raise RuntimeError(f"Training copy has unexpected shape: {copied.shape}")
        return copied

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[np.ndarray, dict[str, str]]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[np.ndarray, dict[str, str]] | list[tuple[np.ndarray, dict[str, str]]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        row = self._rows[index]
        feature = self._copy_rows((row,))[0]
        return feature, self._metadata(row)

    def recording_indices(self, recording_id: str) -> tuple[int, ...]:
        try:
            return self._recording_indices[recording_id]
        except KeyError as exc:
            raise KeyError(f"Unknown development recording ID: {recording_id}") from exc

    def iter_recording_indices(self) -> Iterator[tuple[str, tuple[int, ...]]]:
        for recording_id in self.recording_ids:
            yield recording_id, self._recording_indices[recording_id]

    def get_recording(self, recording_id: str) -> tuple[np.ndarray, tuple[dict[str, str], ...]]:
        indices = self.recording_indices(recording_id)
        rows = tuple(self._rows[index] for index in indices)
        features = self._copy_rows(rows)
        metadata = tuple(self._metadata(row) for row in rows)
        return features, metadata

    def close(self) -> None:
        if self._closed:
            return
        while self._memory_maps:
            _, tensor = self._memory_maps.popitem(last=False)
            _close_memory_map(tensor)
        self._closed = True

    def __enter__(self) -> DevelopmentTrainingData:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def open_development_training_data(
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    *,
    split: str,
    strategy: str,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
    mmap_capacity: int = DEFAULT_MMAP_CAPACITY,
) -> DevelopmentTrainingData:
    """Open a verified known-species training or validation feature reader."""
    return DevelopmentTrainingData(
        cache_root,
        split,
        strategy,
        ffmpeg=ffmpeg,
        expected_lock_sha256=expected_lock_sha256,
        mmap_capacity=mmap_capacity,
    )
