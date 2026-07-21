from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torchvision.models import EfficientNet_B0_Weights

from bird_audio.config import (
    LOCKED_TASK1_CLASS_ORDER,
    config_fingerprint,
    load_toml,
    public_config,
)
from bird_audio.hashing import sha256_json
from bird_audio.models import (
    LOCKED_EFFICIENTNET_WEIGHTS,
    build_efficientnet_b0_classifier,
    parameter_counts,
)
from bird_audio.paths import PROJECT_ROOT, is_relative_to, require_safe_output, resolve_project_path
from bird_audio.provenance import PROVENANCE_V2_ROOT, source_fingerprint
from bird_audio.run_identity import make_run_id
from bird_audio.training_batching import (
    SPECAUGMENT_RANDOM_STREAM,
    RecordingBalancedEpochSampler,
    apply_locked_specaugment,
    collate_native_samples,
    make_epoch_cpu_generator,
    to_efficientnet_batch,
)
from bird_audio.training_data import DevelopmentTrainingData, open_development_training_data

FINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "task1" / "final.toml"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "known_clips_v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "task1_v2"
DEFAULT_BENCHMARK_RESULT_PATH = PROVENANCE_V2_ROOT / "task1_benchmark_v2.json"
DEFAULT_BENCHMARK_LOCK_PATH = PROVENANCE_V2_ROOT / "task1_benchmark_v2.lock.json"
WEIGHT_LOCK_PATH = PROJECT_ROOT / "data" / "manifests" / "efficientnet_b0_imagenet1k_v1.lock.json"
REQUIREMENTS_LOCK_PATH = PROJECT_ROOT / "requirements.lock"
KNOWN_CACHE_LOCK_SHA256 = "d2efbe39c56edc3044deda9692dddf9df02ecf07f0b65d4c9cb3eaa43aa52886"
WEIGHT_HASH_PREFIX = "7f5810bc"
CHECKPOINT_SCHEMA_VERSION = "1.1"
RUN_SCHEMA_VERSION = "1.1"
WEIGHT_LOCK_SCHEMA_VERSION = "1.0"
EXPECTED_TASK1_PARAMETER_COUNTS = {"total": 4_026_763, "trainable": 3_174_955}
EXPECTED_TASK1_VALIDATION_RECORDINGS = 271
EXPECTED_TASK1_VALIDATION_CLIPS = 1_138
EXPECTED_TASK1_TRAINING_CLIPS = 5_319
EXPECTED_TASK1_LIMITS = {"maximum_epochs": 30, "batch_size": 32, "patience": 5}
CONSERVATIVE_WALL_TIME_FACTOR = 1.25
PRODUCTION_SCOPE = "production"
ISOLATED_TEST_SCOPE = "isolated_test"
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_AMSGRAD = False
ADAMW_MAXIMIZE = False
ADAMW_FOREACH = False
ADAMW_CAPTURABLE = False
ADAMW_DIFFERENTIABLE = False
ADAMW_FUSED = False
TASK1_IMPLEMENTATION_FILES = (
    "src/bird_audio/audio.py",
    "src/bird_audio/clip_cache.py",
    "src/bird_audio/clip_selection.py",
    "src/bird_audio/config.py",
    "src/bird_audio/hashing.py",
    "src/bird_audio/io_utils.py",
    "src/bird_audio/locking.py",
    "src/bird_audio/manifest.py",
    "src/bird_audio/metadata.py",
    "src/bird_audio/metadata_artifacts.py",
    "src/bird_audio/models.py",
    "src/bird_audio/paths.py",
    "src/bird_audio/provenance.py",
    "src/bird_audio/review.py",
    "src/bird_audio/run_identity.py",
    "src/bird_audio/signal.py",
    "src/bird_audio/splitting.py",
    "src/bird_audio/task1_training.py",
    "src/bird_audio/training_batching.py",
    "src/bird_audio/training_data.py",
)
_SAFE_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CPU_DEVICE = torch.device("cpu")


class Task1Data(Protocol):
    root: Path
    split: str
    strategy: str
    lock_sha256: str
    recording_count: int

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]: ...

    def metadata(self, index: int) -> dict[str, str]: ...

    def iter_metadata(self) -> Iterator[dict[str, str]]: ...


@dataclass(frozen=True)
class WeightArtifact:
    path: Path
    sha256: str
    size_bytes: int
    identifier: str = LOCKED_EFFICIENTNET_WEIGHTS

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("Weight artifact path must be an absolute pathlib.Path")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("Weight artifact SHA-256 is invalid")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("Weight artifact size must be a positive integer")
        if self.identifier != LOCKED_EFFICIENTNET_WEIGHTS:
            raise ValueError("Weight artifact identifier is not the locked EfficientNet-B0 weight")


@dataclass(frozen=True)
class Task1ExecutionIdentity:
    implementation_sha256: str
    requirements_lock_sha256: str
    numerical_runtime: dict[str, Any]
    numerical_runtime_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.implementation_sha256,
            self.requirements_lock_sha256,
            self.numerical_runtime_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("Task 1 execution identity contains an invalid SHA-256")
        if sha256_json(self.numerical_runtime) != self.numerical_runtime_sha256:
            raise ValueError("Task 1 numerical runtime hash is inconsistent")


@dataclass(frozen=True)
class Task1TestInjection:
    """Explicit CPU-only dependency injection for isolated unit tests."""

    model_factory: Callable[[Mapping[str, Any]], nn.Module]
    weight_artifact: WeightArtifact
    device: torch.device = _CPU_DEVICE
    maximum_epochs: int | None = None
    batch_size: int | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if not callable(self.model_factory):
            raise TypeError("Task1TestInjection model factory must be callable")
        if not isinstance(self.device, torch.device):
            raise TypeError("Task1TestInjection device must be a torch.device")
        if self.device.type != "cpu":
            raise ValueError("Task1TestInjection permits only an explicit CPU device")
        for name, value in (
            ("maximum_epochs", self.maximum_epochs),
            ("batch_size", self.batch_size),
            ("early_stopping_patience", self.early_stopping_patience),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"Test injection {name} must be positive")


@dataclass(frozen=True)
class CheckpointScore:
    macro_f1: float
    validation_loss: float
    epoch: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.macro_f1)
            or not 0.0 <= self.macro_f1 <= 1.0
            or not math.isfinite(self.validation_loss)
            or self.validation_loss < 0.0
        ):
            raise ValueError("Checkpoint scores must be finite")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch <= 0:
            raise ValueError("Checkpoint epoch must be positive")


@dataclass(frozen=True)
class RecordingPredictions:
    recording_ids: tuple[str, ...]
    session_groups: tuple[str, ...]
    true_labels: torch.Tensor
    mean_logits: torch.Tensor
    predicted_labels: torch.Tensor


@dataclass(frozen=True)
class ValidationResult:
    clip_loss: float
    clip_count: int
    recording_count: int
    macro_f1: float
    accuracy: float
    predictions: RecordingPredictions


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("Early-stopping patience must be a positive integer")
        self.patience = patience
        self.best: CheckpointScore | None = None
        self.epochs_without_improvement = 0

    def update(self, score: CheckpointScore) -> tuple[bool, bool]:
        improved = is_better_checkpoint(score, self.best)
        if improved:
            self.best = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved, self.epochs_without_improvement >= self.patience


def _assert_final_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    augmentation = config["augmentation"]
    sampling = config["sampling"]
    loss = config["loss"]
    required = {
        "task": config.get("task") == "classification",
        "architecture": config.get("architecture") == "efficientnet_b0",
        "weights": config.get("pretrained_weights") == LOCKED_EFFICIENTNET_WEIGHTS,
        "weight_hash": config.get("pretrained_weight_cache_hash_required") is True,
        "classes": config.get("class_count") == 15
        and config.get("class_order") == list(LOCKED_TASK1_CLASS_ORDER),
        "dropout": config.get("dropout") == 0.2,
        "features": config.get("trainable_feature_indices") == [6, 7, 8],
        "trainable_from": config.get("trainable_backbone_from_block") == 6,
        "aggregation": config.get("aggregation") == "mean_logits",
        "primary_metric": config.get("primary_metric") == "recording_macro_f1",
        "zero_division": config.get("zero_division") == 0,
        "seeds": config.get("seeds") == [13, 37, 71],
        "rung": config.get("rung") == "final",
        "strategy": sampling.get("strategy") == "energy",
        "recording_weights": sampling.get("recording_balanced_weights") is True,
        "maximum_clips": sampling.get("maximum_clips_per_recording") == 5,
        "specaugment": augmentation.get("specaugment") is True,
        "frequency_mask": augmentation.get("frequency_mask_max_bins") == 16
        and augmentation.get("frequency_mask_probability") == 0.5,
        "time_mask": augmentation.get("time_mask_max_frames") == 40
        and augmentation.get("time_mask_probability") == 0.5,
        "fill": augmentation.get("fill_value") == 0.0,
        "loss": loss.get("name") == "cross_entropy" and loss.get("reduction") == "mean",
        "optimizer": training.get("optimizer") == "adamw",
        "scheduler": training.get("scheduler") == "none",
        "backbone_lr": training.get("backbone_learning_rate") == 0.00003,
        "head_lr": training.get("head_learning_rate") == 0.0003,
        "weight_decay": training.get("weight_decay") == 0.0001,
        "batch": training.get("batch_size") == 32,
        "maximum_epochs": training.get("maximum_epochs") == 30,
        "patience": training.get("early_stopping_patience") == 5,
        "dtype": training.get("dtype") == "float32" and training.get("mixed_precision") is False,
        "device": training.get("device_preference") == "mps"
        and training.get("allow_mps_fallback") is False,
        "deterministic": training.get("request_deterministic_algorithms") is True,
        "determinism_failure": training.get("determinism_failure_policy") == "fail_and_log",
        "seed_python": training.get("seed_python") is True,
        "seed_numpy": training.get("seed_numpy") is True,
        "seed_torch": training.get("seed_torch") is True,
        "seed_sampler": training.get("seed_sampler") is True,
        "workers": training.get("num_workers") == 0 and training.get("pin_memory") is False,
        "parameter_logging": training.get("log_parameter_counts") is True,
        "checkpoint": training.get("checkpoint_metric") == "validation_recording_macro_f1"
        and training.get("checkpoint_mode") == "max"
        and training.get("checkpoint_tie_break_1") == "lower_validation_loss"
        and training.get("checkpoint_tie_break_2") == "earlier_epoch",
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise ValueError(f"Final Task 1 configuration violates locked fields: {failed}")


def load_final_task1_config(path: str | Path = FINAL_CONFIG_PATH) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if resolved != FINAL_CONFIG_PATH.resolve():
        raise PermissionError("Task 1 training accepts only configs/task1/final.toml")
    config = load_toml(resolved)
    _assert_final_config(config)
    return config


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_regular_readonly(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Artifact cannot be opened as a regular file: {path}") from exc
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"Artifact is not a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _descriptor_snapshot(path: Path) -> tuple[bytes, str, int]:
    descriptor = _open_regular_readonly(path)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
            if not chunk:
                raise RuntimeError(f"Artifact ended while being read: {path}")
            chunks.append(chunk)
            digest.update(chunk)
            offset += len(chunk)
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
        if offset != before.st_size or identity_before != identity_after:
            raise RuntimeError(f"Artifact changed while being read: {path}")
        return b"".join(chunks), digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def _task1_implementation_record() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative_path in TASK1_IMPLEMENTATION_FILES:
        path = PROJECT_ROOT / relative_path
        _, sha256, size_bytes = _descriptor_snapshot(path)
        files.append(
            {
                "path": relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    return {"schema_version": "1.0", "files": files}


def _task1_implementation_fingerprint() -> str:
    return sha256_json(_task1_implementation_record())


def _requirements_lock_fingerprint() -> str:
    _, sha256, _ = _descriptor_snapshot(REQUIREMENTS_LOCK_PATH)
    return sha256


def _portable_hardware_value(name: str) -> str:
    if platform.system() != "Darwin":
        return "not_applicable"
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unavailable"


def _numerical_runtime_identity(device: torch.device) -> dict[str, Any]:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().casefold()
    return {
        "schema_version": "1.0",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "apple_hardware_model": _portable_hardware_value("hw.model"),
        "apple_processor_identifier": _portable_hardware_value("machdep.cpu.brand_string"),
        "torch_version": str(torch.__version__),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "numpy_version": np.__version__,
        "device": device.type,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "mps_fallback": fallback or "disabled",
        "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "").strip().casefold()
        or "disabled",
        "mps_prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL", "").strip().casefold()
        or "default",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "training_dtype": "torch.float32",
    }


def _capture_execution_identity(device: torch.device) -> Task1ExecutionIdentity:
    numerical_runtime = _numerical_runtime_identity(device)
    return Task1ExecutionIdentity(
        implementation_sha256=_task1_implementation_fingerprint(),
        requirements_lock_sha256=_requirements_lock_fingerprint(),
        numerical_runtime=numerical_runtime,
        numerical_runtime_sha256=sha256_json(numerical_runtime),
    )


def _require_execution_identity_unchanged(
    expected: Task1ExecutionIdentity,
    device: torch.device,
) -> None:
    observed = _capture_execution_identity(device)
    mismatches = [
        name
        for name in (
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
        )
        if getattr(observed, name) != getattr(expected, name)
    ]
    if mismatches:
        raise RuntimeError(f"Task 1 execution identity drifted: {mismatches}")


def _resolve_project_input_no_follow(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = Path(os.path.abspath(candidate))
    if not is_relative_to(candidate, PROJECT_ROOT):
        raise ValueError(f"Artifact input must stay inside the project: {candidate}")
    return candidate


def _atomic_create_only_bytes(path: str | Path, payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Atomic artifact payload must be nonempty bytes")
    destination = require_safe_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    observed, observed_sha256, observed_size = _descriptor_snapshot(destination)
    if observed != payload or observed_sha256 != expected_sha256:
        raise RuntimeError(f"Published artifact failed descriptor verification: {destination}")
    return {
        "path": str(destination),
        "sha256": observed_sha256,
        "size_bytes": observed_size,
    }


def _json_bytes(value: Any) -> bytes:
    try:
        payload = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("JSON artifact value is not finite and serializable") from exc
    return payload


def _json_values_exact(expected: Any, observed: Any) -> bool:
    if type(expected) is not type(observed):
        return False
    if isinstance(expected, dict):
        return set(expected) == set(observed) and all(
            _json_values_exact(expected[key], observed[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(expected) == len(observed) and all(
            _json_values_exact(left, right) for left, right in zip(expected, observed, strict=True)
        )
    return bool(expected == observed)


def _write_json_create_only(path: Path, value: Any) -> dict[str, Any]:
    return _atomic_create_only_bytes(path, _json_bytes(value))


def _write_or_verify_json(path: Path, value: Any) -> dict[str, Any]:
    try:
        return _write_json_create_only(path, value)
    except FileExistsError:
        observed, record = _read_json_snapshot(path)
        if not _json_values_exact(value, observed):
            raise ValueError(f"Existing JSON artifact differs from resumed state: {path}") from None
        return record


def _read_json_snapshot(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    resolved = _resolve_project_input_no_follow(path)
    payload, observed_sha256, size_bytes = _descriptor_snapshot(resolved)
    if expected_sha256 is not None:
        if _SHA256.fullmatch(expected_sha256) is None:
            raise ValueError("Expected JSON artifact SHA-256 is malformed")
        if observed_sha256 != expected_sha256:
            raise ValueError(f"JSON artifact SHA-256 does not match: {resolved}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact is invalid: {resolved}") from exc
    if _json_bytes(value) != payload:
        raise ValueError(f"JSON artifact is not in canonical form: {resolved}")
    return value, {
        "path": str(resolved),
        "sha256": observed_sha256,
        "size_bytes": size_bytes,
    }


def _artifact_path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_artifact_record(
    value: Any,
    *,
    expected_path: Path,
    context: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "size_bytes"}
        or value.get("path") != str(expected_path)
        or not isinstance(value.get("sha256"), str)
        or _SHA256.fullmatch(value["sha256"]) is None
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"{context} artifact record is invalid")
    _, observed_sha256, observed_size = _descriptor_snapshot(expected_path)
    if observed_sha256 != value["sha256"] or observed_size != value["size_bytes"]:
        raise ValueError(f"{context} artifact record does not match its file")
    return dict(value)


def _weight_cache_path() -> Path:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    if filename != "efficientnet_b0_rwightman-7f5810bc.pth":
        raise RuntimeError("Torchvision EfficientNet-B0 weight identity changed")
    return Path(torch.hub.get_dir()).resolve() / "checkpoints" / filename


def _load_weight_state_from_descriptor(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[dict[str, torch.Tensor], str, int]:
    payload, observed_sha256, observed_size = _descriptor_snapshot(path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError("EfficientNet-B0 weight SHA-256 changed")
    if expected_size is not None and observed_size != expected_size:
        raise RuntimeError("EfficientNet-B0 weight size changed")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()
        )
    ):
        raise ValueError("Cached EfficientNet-B0 weight state is invalid")
    return dict(state), observed_sha256, observed_size


def _weight_lock_value(weight: WeightArtifact) -> dict[str, Any]:
    return {
        "schema_version": WEIGHT_LOCK_SCHEMA_VERSION,
        "identifier": LOCKED_EFFICIENTNET_WEIGHTS,
        "filename": weight.path.name,
        "source_url": EfficientNet_B0_Weights.IMAGENET1K_V1.url,
        "sha256": weight.sha256,
        "size_bytes": weight.size_bytes,
    }


def _require_or_create_weight_lock(weight: WeightArtifact, *, create: bool) -> None:
    expected = _weight_lock_value(weight)
    if WEIGHT_LOCK_PATH.exists():
        observed, _ = _read_json_snapshot(WEIGHT_LOCK_PATH)
        if observed != expected:
            raise ValueError("Canonical EfficientNet-B0 weight lock does not match the cache")
        return
    if not create:
        raise FileNotFoundError(
            "Canonical EfficientNet-B0 weight lock is absent; run explicit weight population"
        )
    _write_json_create_only(WEIGHT_LOCK_PATH, expected)


def preflight_efficientnet_weights(*, populate: bool = False) -> WeightArtifact:
    """Populate explicitly or verify the canonical descriptor-bound weight artifact."""
    path = _weight_cache_path()
    if populate:
        state = EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(
            progress=True,
            check_hash=True,
        )
        del state
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "Verified EfficientNet-B0 weights are absent; run the explicit weight preflight "
            "with populate=True"
        )
    _, initial_sha256, size_bytes = _load_weight_state_from_descriptor(path)
    if not initial_sha256.startswith(WEIGHT_HASH_PREFIX):
        raise ValueError("Cached EfficientNet-B0 weights fail the official hash prefix")
    artifact = WeightArtifact(path, initial_sha256, size_bytes)
    _require_or_create_weight_lock(artifact, create=populate)
    return artifact


def _require_project_venv() -> None:
    expected = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(f"Task 1 training must run inside the project virtualenv: {expected}")


def _resolve_runtime(test_injection: Task1TestInjection | None) -> torch.device:
    _require_project_venv()
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().casefold()
    if fallback not in {"", "0", "false"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    if test_injection is not None:
        return test_injection.device
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Task 1 production training requires available Apple MPS")
    return torch.device("mps")


def seed_task1(seed: int, device: torch.device) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Task 1 seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _require_weight_unchanged(weight: WeightArtifact) -> None:
    _, observed_sha256, observed_size = _descriptor_snapshot(weight.path)
    if observed_size != weight.size_bytes or observed_sha256 != weight.sha256:
        raise RuntimeError("Task 1 weight artifact changed after preflight")


def _strict_load_verified_pretrained_state(model: nn.Module, weight: WeightArtifact) -> None:
    network = getattr(model, "network", None)
    if not isinstance(network, nn.Module):
        raise TypeError("Production Task 1 model does not expose its EfficientNet network")
    state, _, _ = _load_weight_state_from_descriptor(
        weight.path,
        expected_sha256=weight.sha256,
        expected_size=weight.size_bytes,
    )
    excluded = {"classifier.1.weight", "classifier.1.bias"}
    if not excluded.issubset(state):
        raise ValueError("Pretrained EfficientNet-B0 state lacks the replaced classifier tensors")
    head_weight = state["classifier.1.weight"]
    head_bias = state["classifier.1.bias"]
    if tuple(head_weight.shape) != (1000, 1280) or tuple(head_bias.shape) != (1000,):
        raise ValueError("Pretrained EfficientNet-B0 classifier tensors are invalid")
    backbone_state = {key: value for key, value in state.items() if key not in excluded}
    incompatible = network.load_state_dict(backbone_state, strict=False)
    if set(incompatible.missing_keys) != excluded or incompatible.unexpected_keys:
        raise RuntimeError("Pretrained EfficientNet-B0 state does not exactly match the backbone")


def _build_model(
    config: Mapping[str, Any],
    device: torch.device,
    weight: WeightArtifact,
    test_injection: Task1TestInjection | None,
) -> nn.Module:
    _require_weight_unchanged(weight)
    if test_injection is None:
        model = build_efficientnet_b0_classifier(
            class_count=int(config["class_count"]),
            dropout=float(config["dropout"]),
            weights_identifier=None,
            trainable_feature_indices=config["trainable_feature_indices"],
        )
        _strict_load_verified_pretrained_state(model, weight)
    else:
        model = test_injection.model_factory(config)
    _require_weight_unchanged(weight)
    return model.to(device=device, dtype=torch.float32)


def build_task1_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.AdamW:
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(features, nn.Module) or not isinstance(classifier, nn.Module):
        raise TypeError("Task 1 model must expose features and classifier modules")
    backbone = [parameter for parameter in features.parameters() if parameter.requires_grad]
    head = [parameter for parameter in classifier.parameters() if parameter.requires_grad]
    if not backbone or not head:
        raise ValueError("Task 1 optimizer requires nonempty backbone and head parameter groups")
    backbone_ids = {id(parameter) for parameter in backbone}
    head_ids = {id(parameter) for parameter in head}
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if backbone_ids.intersection(head_ids) or backbone_ids.union(head_ids) != trainable_ids:
        raise ValueError("Task 1 trainable parameters are not exactly partitioned")
    training = config["training"]
    return torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": float(training["backbone_learning_rate"]),
                "group_name": "backbone",
            },
            {
                "params": head,
                "lr": float(training["head_learning_rate"]),
                "group_name": "head",
            },
        ],
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=float(training["weight_decay"]),
        amsgrad=ADAMW_AMSGRAD,
        maximize=ADAMW_MAXIMIZE,
        foreach=ADAMW_FOREACH,
        capturable=ADAMW_CAPTURABLE,
        differentiable=ADAMW_DIFFERENTIABLE,
        fused=ADAMW_FUSED,
    )


def is_better_checkpoint(
    candidate: CheckpointScore,
    incumbent: CheckpointScore | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.macro_f1 != incumbent.macro_f1:
        return candidate.macro_f1 > incumbent.macro_f1
    if candidate.validation_loss != incumbent.validation_loss:
        return candidate.validation_loss < incumbent.validation_loss
    return candidate.epoch < incumbent.epoch


def aggregate_recording_logits(
    clip_logits: torch.Tensor,
    clip_labels: torch.Tensor,
    metadata: Sequence[Mapping[str, str]],
) -> RecordingPredictions:
    if (
        clip_logits.device.type != "cpu"
        or clip_logits.dtype != torch.float32
        or clip_logits.ndim != 2
        or clip_labels.device.type != "cpu"
        or clip_labels.dtype != torch.long
        or clip_labels.ndim != 1
        or clip_logits.shape[0] != clip_labels.shape[0]
        or clip_logits.shape[0] != len(metadata)
        or clip_logits.shape[0] == 0
    ):
        raise ValueError("Clip predictions violate the recording aggregation contract")
    if not bool(torch.isfinite(clip_logits).all()):
        raise ValueError("Clip logits contain non-finite values")

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for index, item in enumerate(metadata):
        recording_id = str(item.get("recording_id") or "")
        session_group = str(item.get("session_group") or "")
        if not recording_id or not session_group:
            raise ValueError("Validation metadata lacks recording identity")
        label = int(clip_labels[index])
        group = groups.setdefault(
            recording_id,
            {"session_group": session_group, "label": label, "indices": []},
        )
        if group["session_group"] != session_group or group["label"] != label:
            raise ValueError(f"Validation identity changes within recording: {recording_id}")
        group["indices"].append(index)

    recording_ids = tuple(groups)
    sessions = tuple(str(group["session_group"]) for group in groups.values())
    labels = torch.tensor([int(group["label"]) for group in groups.values()], dtype=torch.long)
    means = torch.stack(
        [clip_logits[group["indices"]].mean(dim=0) for group in groups.values()]
    ).contiguous()
    predicted = means.argmax(dim=1).to(dtype=torch.long)
    return RecordingPredictions(recording_ids, sessions, labels, means, predicted)


def fixed_class_metrics(
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    class_count: int,
) -> dict[str, Any]:
    if class_count <= 1:
        raise ValueError("class_count must exceed one")
    if (
        true_labels.dtype != torch.long
        or predicted_labels.dtype != torch.long
        or true_labels.device.type != "cpu"
        or predicted_labels.device.type != "cpu"
        or true_labels.ndim != 1
        or predicted_labels.shape != true_labels.shape
        or true_labels.numel() == 0
        or bool(torch.any(true_labels < 0))
        or bool(torch.any(true_labels >= class_count))
        or bool(torch.any(predicted_labels < 0))
        or bool(torch.any(predicted_labels >= class_count))
    ):
        raise ValueError("Recording labels violate the fixed-class metric contract")
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
    for truth, prediction in zip(true_labels.tolist(), predicted_labels.tolist(), strict=True):
        confusion[truth, prediction] += 1
    per_class_f1: list[float] = []
    for index in range(class_count):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": float((true_labels == predicted_labels).to(torch.float64).mean()),
        "macro_f1": float(sum(per_class_f1) / class_count),
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def _batch_positions(length: int, batch_size: int) -> Iterator[range]:
    for start in range(0, length, batch_size):
        yield range(start, min(start + batch_size, length))


def _validation_logits_to_cpu(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("Validation produced no logit batches")
    return torch.cat(tuple(parts), dim=0).to(device="cpu", dtype=torch.float32)


def _labels(metadata: Sequence[Mapping[str, str]], class_count: int) -> torch.Tensor:
    try:
        labels = torch.tensor([int(item["class_index"]) for item in metadata], dtype=torch.long)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Batch metadata contains an invalid class index") from exc
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= class_count)):
        raise ValueError("Batch class index is outside the fixed class order")
    return labels


def train_task1_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Task1Data,
    *,
    seed: int,
    epoch_index: int,
    batch_size: int,
    class_count: int,
    device: torch.device,
) -> dict[str, Any]:
    if data.split != "train" or data.strategy != "energy":
        raise PermissionError("Task 1 training accepts only energy-selected train data")
    sampler = RecordingBalancedEpochSampler(data, base_seed=seed)
    sampler.set_epoch(epoch_index)
    sampled_indices = list(sampler)
    augmentation_generator = make_epoch_cpu_generator(
        seed,
        epoch_index,
        SPECAUGMENT_RANDOM_STREAM,
    )
    model.train()
    loss_sum = 0.0
    completed = 0
    for positions in _batch_positions(len(sampled_indices), batch_size):
        indices = [sampled_indices[position] for position in positions]
        native = collate_native_samples([data[index] for index in indices])
        augmented = apply_locked_specaugment(native.tensor, generator=augmentation_generator)
        inputs = to_efficientnet_batch(augmented).to(device=device, dtype=torch.float32)
        targets = _labels(native.metadata, class_count).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if logits.dtype != torch.float32 or logits.shape != (len(indices), class_count):
            raise RuntimeError("Task 1 model returned invalid training logits")
        loss = functional.cross_entropy(logits, targets, reduction="mean")
        loss.backward()
        finite_parts = [torch.isfinite(loss.detach())]
        finite_parts.extend(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if len(finite_parts) <= 1:
            raise RuntimeError("Task 1 training produced no gradients")
        finite_reduction = torch.stack(finite_parts).all()
        batch_status = torch.stack((loss.detach(), finite_reduction.to(dtype=loss.dtype))).to(
            device="cpu"
        )
        if not bool(batch_status[1]):
            raise RuntimeError("Task 1 loss or gradient is non-finite")
        batch_loss = float(batch_status[0])
        optimizer.step()
        loss_sum += batch_loss * len(indices)
        completed += len(indices)
    if completed != len(data):
        raise RuntimeError("Task 1 epoch draw count is incomplete")
    return {
        "clip_loss": loss_sum / completed,
        "clips": completed,
        "batches": math.ceil(completed / batch_size),
        "sampler_seed": sampler.generator_seed,
    }


def validate_task1(
    model: nn.Module,
    data: Task1Data,
    *,
    batch_size: int,
    class_count: int,
    device: torch.device,
) -> ValidationResult:
    if data.split == "test":
        raise PermissionError("Task 1 development validation cannot open the final test split")
    if data.split != "validation" or data.strategy != "energy":
        raise PermissionError("Task 1 validation accepts only energy-selected validation data")
    model.eval()
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    metadata: list[dict[str, str]] = []
    with torch.no_grad():
        for positions in _batch_positions(len(data), batch_size):
            samples = [data[index] for index in positions]
            native = collate_native_samples(samples)
            inputs = to_efficientnet_batch(native.tensor).to(device=device, dtype=torch.float32)
            targets_cpu = _labels(native.metadata, class_count)
            logits = model(inputs)
            if logits.dtype != torch.float32 or logits.shape != (len(samples), class_count):
                raise RuntimeError("Task 1 model returned invalid validation logits")
            logits_parts.append(logits.detach())
            labels_parts.append(targets_cpu)
            metadata.extend(native.metadata)
    clip_logits = _validation_logits_to_cpu(logits_parts)
    clip_labels = torch.cat(labels_parts)
    clip_loss = float(functional.cross_entropy(clip_logits, clip_labels, reduction="mean"))
    predictions = aggregate_recording_logits(clip_logits, clip_labels, metadata)
    metrics = fixed_class_metrics(
        predictions.true_labels,
        predictions.predicted_labels,
        class_count=class_count,
    )
    return ValidationResult(
        clip_loss=clip_loss,
        clip_count=len(data),
        recording_count=len(predictions.recording_ids),
        macro_f1=float(metrics["macro_f1"]),
        accuracy=float(metrics["accuracy"]),
        predictions=predictions,
    )


def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _assert_round_trip(expected: Any, actual: Any, context: str = "checkpoint") -> None:
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(expected, actual):
            raise RuntimeError(f"Checkpoint tensor changed at {context}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise RuntimeError(f"Checkpoint dictionary changed at {context}")
        for key in expected:
            _assert_round_trip(expected[key], actual[key], f"{context}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise RuntimeError(f"Checkpoint sequence changed at {context}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_round_trip(left, right, f"{context}[{index}]")
        return
    if expected != actual:
        raise RuntimeError(f"Checkpoint scalar changed at {context}")


_CHECKPOINT_COMMON_FIELDS = {
    "schema_version",
    "checkpoint_type",
    "run_id",
    "run_identity_sha256",
    "config_sha256",
    "cache_lock_sha256",
    "weight_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "scope",
    "production_evidence",
    "seed",
}
_BEST_CHECKPOINT_FIELDS = _CHECKPOINT_COMMON_FIELDS | {
    "epoch",
    "score",
    "model",
    "optimizer",
    "predictions",
}
_RECOVERY_CHECKPOINT_FIELDS = _CHECKPOINT_COMMON_FIELDS | {
    "completed_epoch",
    "next_epoch_index",
    "stop_requested",
    "limits",
    "early_stopping",
    "best_candidate",
    "history",
    "model",
    "optimizer",
    "rng_state",
}
_COMPLETED_RUN_RESULT_FIELDS = {
    "schema_version",
    "complete",
    "run_id",
    "run_directory",
    "run_identity_sha256",
    "config_sha256",
    "cache_lock_sha256",
    "weight_sha256",
    "source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "scope",
    "production_evidence",
    "resumed",
    "resume_checkpoint",
    "epochs_completed",
    "early_stopped",
    "best_epoch",
    "best_validation_macro_f1",
    "best_validation_loss",
    "best_checkpoint",
    "latest_recovery_checkpoint",
    "artifacts",
}


def _validate_score(value: Any) -> CheckpointScore:
    if not isinstance(value, dict) or set(value) != {"macro_f1", "validation_loss", "epoch"}:
        raise ValueError("Checkpoint score schema is invalid")
    if (
        type(value["macro_f1"]) is not float
        or type(value["validation_loss"]) is not float
        or type(value["epoch"]) is not int
    ):
        raise ValueError("Checkpoint score types are invalid")
    return CheckpointScore(
        value["macro_f1"],
        value["validation_loss"],
        value["epoch"],
    )


def _prediction_checkpoint_state(predictions: RecordingPredictions) -> dict[str, Any]:
    return {
        "recording_ids": predictions.recording_ids,
        "session_groups": predictions.session_groups,
        "true_labels": predictions.true_labels.detach().cpu().clone(),
        "mean_logits": predictions.mean_logits.detach().cpu().clone(),
        "predicted_labels": predictions.predicted_labels.detach().cpu().clone(),
    }


def _predictions_from_checkpoint_state(value: Any) -> RecordingPredictions:
    required = {
        "recording_ids",
        "session_groups",
        "true_labels",
        "mean_logits",
        "predicted_labels",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Checkpoint prediction schema is invalid")
    recording_ids = value["recording_ids"]
    session_groups = value["session_groups"]
    true_labels = value["true_labels"]
    mean_logits = value["mean_logits"]
    predicted_labels = value["predicted_labels"]
    if (
        not isinstance(recording_ids, tuple)
        or not isinstance(session_groups, tuple)
        or not recording_ids
        or len(recording_ids) != len(session_groups)
        or len(set(recording_ids)) != len(recording_ids)
        or any(not isinstance(item, str) or not item for item in (*recording_ids, *session_groups))
        or not torch.is_tensor(true_labels)
        or true_labels.device.type != "cpu"
        or true_labels.dtype != torch.long
        or true_labels.shape != (len(recording_ids),)
        or not torch.is_tensor(mean_logits)
        or mean_logits.device.type != "cpu"
        or mean_logits.dtype != torch.float32
        or mean_logits.shape != (len(recording_ids), len(LOCKED_TASK1_CLASS_ORDER))
        or not bool(torch.isfinite(mean_logits).all())
        or not torch.is_tensor(predicted_labels)
        or predicted_labels.device.type != "cpu"
        or predicted_labels.dtype != torch.long
        or predicted_labels.shape != true_labels.shape
        or bool(torch.any(true_labels < 0))
        or bool(torch.any(true_labels >= len(LOCKED_TASK1_CLASS_ORDER)))
        or not torch.equal(predicted_labels, mean_logits.argmax(dim=1))
    ):
        raise ValueError("Checkpoint predictions violate the fixed class contract")
    return RecordingPredictions(
        recording_ids,
        session_groups,
        true_labels,
        mean_logits,
        predicted_labels,
    )


def _validate_checkpoint_common(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Task 1 checkpoint version is unsupported")
    run_id = checkpoint.get("run_id")
    if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Task 1 checkpoint run ID is invalid")
    for key in (
        "run_identity_sha256",
        "config_sha256",
        "cache_lock_sha256",
        "weight_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
    ):
        if not isinstance(checkpoint.get(key), str) or _SHA256.fullmatch(checkpoint[key]) is None:
            raise ValueError(f"Task 1 checkpoint {key} is invalid")
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {13, 37, 71}:
        raise ValueError("Task 1 checkpoint seed is invalid")
    scope = checkpoint.get("scope")
    production_evidence = checkpoint.get("production_evidence")
    if (
        scope not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or not isinstance(production_evidence, bool)
        or production_evidence is not (scope == PRODUCTION_SCOPE)
    ):
        raise ValueError("Task 1 checkpoint evidence scope is invalid")


def _validate_history(history: Any, completed_epoch: int) -> None:
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise ValueError("Recovery history length is invalid")
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, dict) or row.get("epoch") != expected_epoch:
            raise ValueError("Recovery history epoch order is invalid")
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Recovery history is not finite JSON") from exc


def _validate_checkpoint_state(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        raise ValueError("Task 1 checkpoint must be a dictionary")
    checkpoint_type = checkpoint.get("checkpoint_type")
    expected_fields = (
        _BEST_CHECKPOINT_FIELDS if checkpoint_type == "best" else _RECOVERY_CHECKPOINT_FIELDS
    )
    if checkpoint_type not in {"best", "recovery"} or set(checkpoint) != expected_fields:
        raise ValueError("Task 1 checkpoint schema is invalid")
    _validate_checkpoint_common(checkpoint)
    if not isinstance(checkpoint["model"], dict) or not checkpoint["model"]:
        raise ValueError("Task 1 checkpoint model state is invalid")
    if not isinstance(checkpoint["optimizer"], dict) or not checkpoint["optimizer"]:
        raise ValueError("Task 1 checkpoint optimizer state is invalid")
    if checkpoint_type == "best":
        epoch = checkpoint["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("Best checkpoint epoch is invalid")
        score = _validate_score(checkpoint["score"])
        if score.epoch != epoch:
            raise ValueError("Best checkpoint score epoch differs from its state")
        _predictions_from_checkpoint_state(checkpoint["predictions"])
        return checkpoint_type

    completed_epoch = checkpoint["completed_epoch"]
    next_epoch_index = checkpoint["next_epoch_index"]
    if (
        isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or completed_epoch <= 0
        or isinstance(next_epoch_index, bool)
        or not isinstance(next_epoch_index, int)
        or next_epoch_index != completed_epoch
        or not isinstance(checkpoint["stop_requested"], bool)
    ):
        raise ValueError("Recovery checkpoint epoch state is invalid")
    limits = checkpoint["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"maximum_epochs", "batch_size", "patience"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in limits.values()
        )
        or completed_epoch > limits["maximum_epochs"]
    ):
        raise ValueError("Recovery checkpoint limits are invalid")
    early = checkpoint["early_stopping"]
    if (
        not isinstance(early, dict)
        or set(early) != {"best", "epochs_without_improvement", "patience"}
        or early["patience"] != limits["patience"]
        or isinstance(early["epochs_without_improvement"], bool)
        or not isinstance(early["epochs_without_improvement"], int)
        or not 0 <= early["epochs_without_improvement"] <= completed_epoch
    ):
        raise ValueError("Recovery early-stopping state is invalid")
    best_score = _validate_score(early["best"])
    candidate = checkpoint["best_candidate"]
    expected_candidate_path = f"best_candidates/best_epoch_{best_score.epoch:04d}.pt"
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"path", "sha256", "epoch"}
        or candidate["path"] != expected_candidate_path
        or candidate["epoch"] != best_score.epoch
        or not isinstance(candidate["sha256"], str)
        or _SHA256.fullmatch(candidate["sha256"]) is None
        or best_score.epoch > completed_epoch
    ):
        raise ValueError("Recovery best-candidate binding is invalid")
    _validate_history(checkpoint["history"], completed_epoch)
    _validate_rng_state(checkpoint["rng_state"], expected_device=None)
    return checkpoint_type


def save_task1_checkpoint_create_only(path: str | Path, state: Mapping[str, Any]) -> dict[str, Any]:
    expected = _cpu_copy(dict(state))
    _validate_checkpoint_state(expected)
    buffer = io.BytesIO()
    torch.save(expected, buffer)
    payload = buffer.getvalue()
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    _validate_checkpoint_state(loaded)
    _assert_round_trip(expected, loaded)
    record = _atomic_create_only_bytes(path, payload)
    verified_payload, verified_sha256, _ = _descriptor_snapshot(Path(record["path"]))
    if verified_sha256 != record["sha256"]:
        raise RuntimeError("Task 1 checkpoint SHA-256 changed after publication")
    verified = torch.load(io.BytesIO(verified_payload), map_location="cpu", weights_only=True)
    _validate_checkpoint_state(verified)
    _assert_round_trip(expected, verified)
    return record


def load_task1_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_run_identity_sha256: str | None = None,
    expected_type: str | None = None,
) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Expected checkpoint SHA-256 is malformed")
    resolved = _resolve_project_input_no_follow(path)
    payload, observed_sha256, _ = _descriptor_snapshot(resolved)
    if observed_sha256 != expected_sha256:
        raise ValueError("Task 1 checkpoint SHA-256 does not match")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    checkpoint_type = _validate_checkpoint_state(checkpoint)
    if expected_type is not None and checkpoint_type != expected_type:
        raise ValueError("Task 1 checkpoint type does not match")
    if (
        expected_run_identity_sha256 is not None
        and checkpoint["run_identity_sha256"] != expected_run_identity_sha256
    ):
        raise ValueError("Task 1 checkpoint run identity does not match")
    return checkpoint


def _locked_task1_architecture_contract(
    model: nn.Module,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_type = f"{type(model).__module__}.{type(model).__qualname__}"
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    selected = tuple(getattr(model, "trainable_feature_indices", ()))
    frozen = tuple(getattr(model, "frozen_feature_indices", ()))
    if (
        model_type != "bird_audio.models.EfficientNetB0Classifier"
        or not isinstance(features, nn.Sequential)
        or len(features) != 9
        or not isinstance(classifier, nn.Sequential)
        or len(classifier) != 2
        or selected != (6, 7, 8)
        or frozen != (0, 1, 2, 3, 4, 5)
    ):
        raise ValueError("Task 1 model structure differs from the locked architecture")
    dropout = classifier[0]
    head = classifier[1]
    if (
        not isinstance(dropout, nn.Dropout)
        or dropout.p != float(config["dropout"])
        or dropout.inplace is not True
        or not isinstance(head, nn.Linear)
        or head.in_features != 1_280
        or head.out_features != len(LOCKED_TASK1_CLASS_ORDER)
        or head.bias is None
    ):
        raise ValueError("Task 1 classifier head differs from the locked contract")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Task 1 verification model must remain on CPU")
    if any(
        parameter.requires_grad for index in frozen for parameter in features[index].parameters()
    ):
        raise ValueError("Task 1 frozen feature parameters became trainable")
    if any(
        not parameter.requires_grad
        for index in selected
        for parameter in features[index].parameters()
    ) or any(not parameter.requires_grad for parameter in classifier.parameters()):
        raise ValueError("Task 1 selected feature or classifier parameters became frozen")
    counts = parameter_counts(model)
    if counts != EXPECTED_TASK1_PARAMETER_COUNTS:
        raise ValueError("Task 1 parameter counts differ from the locked contract")
    return {
        "architecture": "efficientnet_b0",
        "model_type": model_type,
        "class_count": len(LOCKED_TASK1_CLASS_ORDER),
        "dropout": float(dropout.p),
        "classifier_in_features": head.in_features,
        "trainable_feature_indices": list(selected),
        "frozen_feature_indices": list(frozen),
        "parameter_counts": counts,
        "state_tensor_count": len(model.state_dict()),
    }


def _strict_load_locked_task1_model_state(
    state: Any,
    config: Mapping[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    model = build_efficientnet_b0_classifier(
        class_count=int(config["class_count"]),
        dropout=float(config["dropout"]),
        weights_identifier=None,
        trainable_feature_indices=config["trainable_feature_indices"],
    )
    model_contract = _locked_task1_architecture_contract(model, config)
    expected_state = model.state_dict()
    if not isinstance(state, dict) or list(state) != list(expected_state):
        raise ValueError("Task 1 checkpoint model keys differ from the locked model")
    for key, expected_tensor in expected_state.items():
        observed_tensor = state[key]
        if (
            not torch.is_tensor(observed_tensor)
            or observed_tensor.device.type != "cpu"
            or observed_tensor.shape != expected_tensor.shape
            or observed_tensor.dtype != expected_tensor.dtype
            or not bool(torch.isfinite(observed_tensor).all())
        ):
            raise ValueError(f"Task 1 checkpoint model tensor is invalid: {key}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Task 1 checkpoint cannot strict-load into the locked model") from exc
    return model, model_contract


def _verify_locked_task1_recovery_state(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    model, _ = _strict_load_locked_task1_model_state(checkpoint["model"], config)
    optimizer = build_task1_optimizer(model, config)
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Task 1 recovery optimizer cannot load into locked AdamW") from exc
    _validate_optimizer_after_resume(optimizer, model, config)
    _validate_rng_state(checkpoint["rng_state"], expected_device=torch.device("mps"))


def verify_locked_task1_best_checkpoint_model_state(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_run_identity_sha256: str,
) -> dict[str, Any]:
    """Verify a production best checkpoint on CPU without opening data or weight contents."""

    config = load_final_task1_config()
    config_sha256 = config_fingerprint(config)
    checkpoint = load_task1_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_run_identity_sha256=expected_run_identity_sha256,
        expected_type="best",
    )
    if (
        checkpoint["scope"] != PRODUCTION_SCOPE
        or checkpoint["production_evidence"] is not True
        or checkpoint["config_sha256"] != config_sha256
        or checkpoint["cache_lock_sha256"] != KNOWN_CACHE_LOCK_SHA256
        or not checkpoint["weight_sha256"].startswith(WEIGHT_HASH_PREFIX)
        or checkpoint["seed"] not in config["seeds"]
    ):
        raise ValueError("Task 1 checkpoint is not bound to the locked production method")

    observed_state = checkpoint["model"]
    model, model_contract = _strict_load_locked_task1_model_state(observed_state, config)
    model.eval()

    resolved_checkpoint = _resolve_project_input_no_follow(checkpoint_path)
    _, observed_sha256, observed_size = _descriptor_snapshot(resolved_checkpoint)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("Task 1 checkpoint changed during model-state verification")
    score = _validate_score(checkpoint["score"])
    return {
        "valid": True,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": expected_sha256,
        "checkpoint_size_bytes": observed_size,
        "run_id": checkpoint["run_id"],
        "run_identity_sha256": expected_run_identity_sha256,
        "config_sha256": checkpoint["config_sha256"],
        "cache_lock_sha256": checkpoint["cache_lock_sha256"],
        "weight_sha256": checkpoint["weight_sha256"],
        "implementation_sha256": checkpoint["implementation_sha256"],
        "requirements_lock_sha256": checkpoint["requirements_lock_sha256"],
        "numerical_runtime_sha256": checkpoint["numerical_runtime_sha256"],
        "scope": checkpoint["scope"],
        "production_evidence": checkpoint["production_evidence"],
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "score": _score_state(score),
        "model_contract": model_contract,
    }


def load_locked_task1_best_model(
    checkpoint_path: str | Path,
    *,
    checkpoint_sha256: str,
    expected_run_identity_sha256: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load one verified production Task 1 best checkpoint for gated evaluation."""
    if not isinstance(device, torch.device) or device != torch.device("mps"):
        raise ValueError("Locked Task 1 evaluation requires the supplied MPS device")
    resolved_device = _resolve_runtime(None)
    if resolved_device != device:
        raise RuntimeError("Locked Task 1 evaluation resolved a different production device")
    config = load_final_task1_config()
    execution_identity = _capture_execution_identity(device)
    weight = preflight_efficientnet_weights(populate=False)
    checkpoint = load_task1_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        expected_run_identity_sha256=expected_run_identity_sha256,
        expected_type="best",
    )
    config_sha256 = config_fingerprint(config)
    expected = {
        "run_identity_sha256": expected_run_identity_sha256,
        "config_sha256": config_sha256,
        "cache_lock_sha256": KNOWN_CACHE_LOCK_SHA256,
        "weight_sha256": weight.sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "scope": PRODUCTION_SCOPE,
        "production_evidence": True,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
        or type(checkpoint.get(key)) is not type(expected_value)
    ]
    if mismatches:
        raise ValueError(f"Task 1 evaluation checkpoint identity differs: {mismatches}")
    if checkpoint["seed"] not in config["seeds"]:
        raise ValueError("Task 1 evaluation checkpoint seed is outside the locked seed set")
    model_state = checkpoint["model"]
    if any(
        not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all())
        for tensor in model_state.values()
    ):
        raise ValueError("Task 1 evaluation checkpoint model state is invalid")
    model = _build_model(config, device, weight, None)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    _require_execution_identity_unchanged(execution_identity, device)
    _require_weight_unchanged(weight)
    resolved_checkpoint = _resolve_project_input_no_follow(checkpoint_path)
    _, observed_checkpoint_sha256, observed_checkpoint_size = _descriptor_snapshot(
        resolved_checkpoint
    )
    if observed_checkpoint_sha256 != checkpoint_sha256:
        raise RuntimeError("Task 1 evaluation checkpoint changed while being loaded")
    score = _validate_score(checkpoint["score"])
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": observed_checkpoint_size,
        "run_id": checkpoint["run_id"],
        "run_identity_sha256": expected_run_identity_sha256,
        "config_sha256": config_sha256,
        "cache_lock_sha256": KNOWN_CACHE_LOCK_SHA256,
        "weight_sha256": weight.sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "scope": PRODUCTION_SCOPE,
        "production_evidence": True,
        "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"],
        "score": _score_state(score),
    }
    return model, metadata


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    mps_state: torch.Tensor | None = None
    if device.type == "mps":
        mps_state = torch.mps.get_rng_state().detach().cpu().clone()
    return {
        "device": device.type,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_mps": mps_state,
    }


def _validate_rng_state(value: Any, expected_device: torch.device | None) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"device", "python", "numpy", "torch_cpu", "torch_mps"}
        or value["device"] not in {"cpu", "mps"}
        or (expected_device is not None and value["device"] != expected_device.type)
    ):
        raise ValueError("Recovery RNG state schema is invalid")
    try:
        probe = random.Random()
        probe.setstate(value["python"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Recovery Python RNG state is invalid") from exc
    numpy_state = value["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("Recovery NumPy RNG state schema is invalid")
    keys = numpy_state["keys"]
    if (
        numpy_state["bit_generator"] != "MT19937"
        or not torch.is_tensor(keys)
        or keys.device.type != "cpu"
        or keys.dtype != torch.int64
        or keys.shape != (624,)
        or bool(torch.any(keys < 0))
        or bool(torch.any(keys > 2**32 - 1))
        or isinstance(numpy_state["position"], bool)
        or not isinstance(numpy_state["position"], int)
        or not 0 <= numpy_state["position"] <= 624
        or numpy_state["has_gauss"] not in {0, 1}
        or isinstance(numpy_state["has_gauss"], bool)
        or not isinstance(numpy_state["cached_gaussian"], float)
        or not math.isfinite(numpy_state["cached_gaussian"])
    ):
        raise ValueError("Recovery NumPy RNG state values are invalid")
    try:
        probe_numpy = np.random.RandomState()
        probe_numpy.set_state(
            (
                "MT19937",
                keys.numpy().astype(np.uint32, copy=True),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Recovery NumPy RNG state cannot be restored") from exc
    torch_cpu = value["torch_cpu"]
    torch_mps = value["torch_mps"]
    if (
        not torch.is_tensor(torch_cpu)
        or torch_cpu.device.type != "cpu"
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
    ):
        raise ValueError("Recovery CPU Torch RNG state is invalid")
    if value["device"] == "cpu" and torch_mps is not None:
        raise ValueError("CPU recovery checkpoint cannot contain MPS RNG state")
    if value["device"] == "mps" and (
        not torch.is_tensor(torch_mps)
        or torch_mps.device.type != "cpu"
        or torch_mps.dtype != torch.uint8
        or torch_mps.ndim != 1
        or torch_mps.numel() == 0
    ):
        raise ValueError("Recovery MPS RNG state is invalid")


def _restore_rng_state(value: Any, device: torch.device) -> None:
    _validate_rng_state(value, expected_device=device)
    numpy_state = value["numpy"]
    random.setstate(value["python"])
    np.random.set_state(
        (
            "MT19937",
            numpy_state["keys"].numpy().astype(np.uint32, copy=True),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(value["torch_cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(value["torch_mps"])


def _validate_completed_run_result(
    value: Any,
    *,
    run_directory: Path,
    run_identity: Mapping[str, Any],
    run_identity_sha256: str,
    expected_common: Mapping[str, Any],
    source_fingerprint_sha256: str,
    maximum_epochs: int,
    expected_latest_recovery_sha256: str | None = None,
) -> None:
    if not isinstance(value, dict) or set(value) != _COMPLETED_RUN_RESULT_FIELDS:
        raise ValueError("Completed Task 1 result schema is invalid")
    fixed = {
        "schema_version": RUN_SCHEMA_VERSION,
        "complete": True,
        "run_id": expected_common["run_id"],
        "run_directory": str(run_directory),
        "run_identity_sha256": run_identity_sha256,
        "config_sha256": expected_common["config_sha256"],
        "cache_lock_sha256": expected_common["cache_lock_sha256"],
        "weight_sha256": expected_common["weight_sha256"],
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "implementation_sha256": expected_common["implementation_sha256"],
        "requirements_lock_sha256": expected_common["requirements_lock_sha256"],
        "numerical_runtime_sha256": expected_common["numerical_runtime_sha256"],
        "scope": expected_common["scope"],
        "production_evidence": expected_common["production_evidence"],
    }
    if any(
        key not in value or not _json_values_exact(expected, value[key])
        for key, expected in fixed.items()
    ):
        raise ValueError("Completed Task 1 result differs from its locked run identity")
    if sha256_json(dict(run_identity)) != run_identity_sha256:
        raise ValueError("Completed Task 1 run identity hash is inconsistent")
    if not isinstance(value["resumed"], bool) or not isinstance(value["early_stopped"], bool):
        raise ValueError("Completed Task 1 result flags are invalid")
    epochs_completed = value["epochs_completed"]
    best_epoch = value["best_epoch"]
    macro_f1 = value["best_validation_macro_f1"]
    validation_loss = value["best_validation_loss"]
    if (
        isinstance(epochs_completed, bool)
        or not isinstance(epochs_completed, int)
        or not 1 <= epochs_completed <= maximum_epochs
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= epochs_completed
        or type(macro_f1) is not float
        or not math.isfinite(macro_f1)
        or not 0.0 <= macro_f1 <= 1.0
        or type(validation_loss) is not float
        or not math.isfinite(validation_loss)
        or validation_loss < 0.0
        or (not value["early_stopped"] and epochs_completed != maximum_epochs)
    ):
        raise ValueError("Completed Task 1 result metrics are invalid")

    artifact_values = value["artifacts"]
    if not isinstance(artifact_values, dict) or set(artifact_values) != {
        "resolved_config",
        "run_identity",
        "provenance",
        "epoch_history",
        "best_validation_predictions",
        "best_checkpoint",
        "latest_recovery",
    }:
        raise ValueError("Completed Task 1 artifact schema is invalid")
    expected_json_paths = {
        "resolved_config": run_directory / "resolved_config.json",
        "run_identity": run_directory / "run_identity.json",
        "provenance": run_directory / "provenance.json",
        "epoch_history": run_directory / "epoch_history.json",
        "best_validation_predictions": run_directory / "best_validation_predictions.json",
    }
    for name, path in expected_json_paths.items():
        _validate_artifact_record(artifact_values[name], expected_path=path, context=name)

    observed_run_identity, _ = _read_json_snapshot(
        expected_json_paths["run_identity"],
        expected_sha256=artifact_values["run_identity"]["sha256"],
    )
    if not _json_values_exact(dict(run_identity), observed_run_identity):
        raise ValueError("Completed Task 1 run identity artifact is inconsistent")
    provenance, _ = _read_json_snapshot(
        expected_json_paths["provenance"],
        expected_sha256=artifact_values["provenance"]["sha256"],
    )
    if (
        not isinstance(provenance, dict)
        or provenance.get("run_identity_sha256") != run_identity_sha256
        or provenance.get("source_fingerprint_sha256") != source_fingerprint_sha256
        or any(
            key not in provenance or not _json_values_exact(expected_common[key], provenance[key])
            for key in (
                "config_sha256",
                "cache_lock_sha256",
                "weight_sha256",
                "implementation_sha256",
                "requirements_lock_sha256",
                "numerical_runtime_sha256",
                "scope",
                "production_evidence",
            )
        )
    ):
        raise ValueError("Completed Task 1 provenance artifact is inconsistent")

    best_path = run_directory / "best_candidates" / f"best_epoch_{best_epoch:04d}.pt"
    recovery_path = run_directory / "recovery" / f"recovery_epoch_{epochs_completed:04d}.pt"
    best_record = _validate_artifact_record(
        value["best_checkpoint"],
        expected_path=best_path,
        context="best checkpoint",
    )
    recovery_record = _validate_artifact_record(
        value["latest_recovery_checkpoint"],
        expected_path=recovery_path,
        context="latest recovery checkpoint",
    )
    if (
        not _json_values_exact(best_record, artifact_values["best_checkpoint"])
        or not _json_values_exact(recovery_record, artifact_values["latest_recovery"])
        or (
            expected_latest_recovery_sha256 is not None
            and recovery_record["sha256"] != expected_latest_recovery_sha256
        )
    ):
        raise ValueError("Completed Task 1 checkpoint artifact bindings are invalid")
    best_checkpoint = load_task1_checkpoint(
        best_path,
        expected_sha256=best_record["sha256"],
        expected_run_identity_sha256=run_identity_sha256,
        expected_type="best",
    )
    recovery_checkpoint = load_task1_checkpoint(
        recovery_path,
        expected_sha256=recovery_record["sha256"],
        expected_run_identity_sha256=run_identity_sha256,
        expected_type="recovery",
    )
    for checkpoint in (best_checkpoint, recovery_checkpoint):
        if any(checkpoint.get(key) != expected for key, expected in expected_common.items()):
            raise ValueError("Completed Task 1 checkpoint identity is inconsistent")
    best_score = _validate_score(best_checkpoint["score"])
    recovery_best = _validate_score(recovery_checkpoint["early_stopping"]["best"])
    if (
        best_score != recovery_best
        or best_score.epoch != best_epoch
        or best_score.macro_f1 != macro_f1
        or best_score.validation_loss != validation_loss
        or recovery_checkpoint["completed_epoch"] != epochs_completed
        or recovery_checkpoint["stop_requested"] is not value["early_stopped"]
    ):
        raise ValueError("Completed Task 1 checkpoint metrics are inconsistent")

    resume_record = value["resume_checkpoint"]
    if value["resumed"]:
        if not isinstance(resume_record, dict) or set(resume_record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("Completed Task 1 resume artifact is invalid")
        resume_path = Path(str(resume_record["path"]))
        if resume_path.parent != run_directory / "recovery":
            raise ValueError("Completed Task 1 resume artifact path is invalid")
        _validate_artifact_record(
            resume_record,
            expected_path=resume_path,
            context="resume checkpoint",
        )
        resumed_checkpoint = load_task1_checkpoint(
            resume_path,
            expected_sha256=resume_record["sha256"],
            expected_run_identity_sha256=run_identity_sha256,
            expected_type="recovery",
        )
        if any(
            resumed_checkpoint.get(key) != expected for key, expected in expected_common.items()
        ):
            raise ValueError("Completed Task 1 resume checkpoint identity is inconsistent")
    elif resume_record is not None:
        raise ValueError("Non-resumed Task 1 result contains a resume artifact")


def _read_task1_bound_json_artifact(
    value: Any,
    *,
    expected_path: Path,
    context: str,
) -> tuple[Any, dict[str, Any]]:
    record = _validate_artifact_record(value, expected_path=expected_path, context=context)
    observed, observed_record = _read_json_snapshot(
        expected_path,
        expected_sha256=record["sha256"],
    )
    if observed_record != record:
        raise ValueError(f"{context} artifact descriptor changed")
    return observed, record


def _validate_task1_history_selection(
    history: Any,
    *,
    limits: Mapping[str, Any],
    require_production: bool,
    require_terminal: bool,
) -> tuple[CheckpointScore, int, bool]:
    if not isinstance(history, list) or not history:
        raise ValueError("Task 1 history is empty")
    patience = limits["patience"]
    best: CheckpointScore | None = None
    epochs_without_improvement = 0
    should_stop = False
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, dict) or set(row) != {
            "epoch",
            "elapsed_seconds",
            "train",
            "validation",
            "checkpoint_improved",
        }:
            raise ValueError("Task 1 history row fields are invalid")
        train = row["train"]
        validation = row["validation"]
        if (
            row["epoch"] != expected_epoch
            or type(row["elapsed_seconds"]) is not float
            or not math.isfinite(row["elapsed_seconds"])
            or row["elapsed_seconds"] < 0.0
            or not isinstance(train, dict)
            or set(train) != {"clip_loss", "clips", "batches", "sampler_seed"}
            or type(train["clip_loss"]) is not float
            or not math.isfinite(train["clip_loss"])
            or train["clip_loss"] < 0.0
            or any(
                type(train[name]) is not int or train[name] <= 0 for name in ("clips", "batches")
            )
            or type(train["sampler_seed"]) is not int
            or not isinstance(validation, dict)
            or set(validation)
            != {
                "clip_loss",
                "clip_count",
                "recording_count",
                "macro_f1",
                "accuracy",
            }
            or type(validation["clip_loss"]) is not float
            or not math.isfinite(validation["clip_loss"])
            or validation["clip_loss"] < 0.0
            or type(validation["clip_count"]) is not int
            or validation["clip_count"] <= 0
            or type(validation["recording_count"]) is not int
            or validation["recording_count"] <= 0
            or type(validation["macro_f1"]) is not float
            or not math.isfinite(validation["macro_f1"])
            or not 0.0 <= validation["macro_f1"] <= 1.0
            or type(validation["accuracy"]) is not float
            or not math.isfinite(validation["accuracy"])
            or not 0.0 <= validation["accuracy"] <= 1.0
            or type(row["checkpoint_improved"]) is not bool
        ):
            raise ValueError("Task 1 history row values are invalid")
        if train["batches"] != math.ceil(train["clips"] / limits["batch_size"]):
            raise ValueError("Task 1 history batch count differs from its run limit")
        if require_production and (
            train["clips"] != EXPECTED_TASK1_TRAINING_CLIPS
            or validation["clip_count"] != EXPECTED_TASK1_VALIDATION_CLIPS
            or validation["recording_count"] != EXPECTED_TASK1_VALIDATION_RECORDINGS
        ):
            raise ValueError("Task 1 history data counts are not the production counts")
        candidate = CheckpointScore(
            validation["macro_f1"],
            validation["clip_loss"],
            expected_epoch,
        )
        improved = is_better_checkpoint(candidate, best)
        if row["checkpoint_improved"] is not improved:
            raise ValueError("Task 1 history checkpoint decision does not rederive")
        if improved:
            best = candidate
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = epochs_without_improvement >= patience
        if should_stop and expected_epoch != len(history):
            raise ValueError("Task 1 history continued after its early-stopping decision")
    if best is None:
        raise ValueError("Task 1 history has no selected checkpoint")
    if require_terminal and not should_stop and len(history) != limits["maximum_epochs"]:
        raise ValueError("Task 1 history ended before a locked terminal condition")
    return best, epochs_without_improvement, should_stop


def verify_task1_development_run(
    completion_lock_path: str | Path,
    *,
    expected_sha256: str,
    require_production: bool = True,
) -> dict[str, Any]:
    """Recursively verify one Task 1 development run without opening any dataset."""

    if type(require_production) is not bool:
        raise TypeError("require_production must be a boolean")
    completion_path = _resolve_project_input_no_follow(completion_lock_path)
    run_directory = completion_path.parent
    if (
        completion_path.name != "result.lock.json"
        or completion_path.is_symlink()
        or run_directory.is_symlink()
        or not run_directory.is_dir()
        or not is_relative_to(run_directory, DEFAULT_RUN_ROOT)
    ):
        raise PermissionError("Task 1 completion lock is outside a canonical run directory")
    completion, completion_record = _read_json_snapshot(
        completion_path,
        expected_sha256=expected_sha256,
    )
    completion_fields = {
        "schema_version",
        "run_identity_sha256",
        "source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "scope",
        "production_evidence",
        "result",
    }
    if (
        not isinstance(completion, dict)
        or set(completion) != completion_fields
        or completion["schema_version"] != RUN_SCHEMA_VERSION
    ):
        raise ValueError("Task 1 completion lock fields are invalid")
    result, _result_record = _read_task1_bound_json_artifact(
        completion["result"],
        expected_path=run_directory / "result.json",
        context="Task 1 result",
    )
    if (
        not isinstance(result, dict)
        or set(result) != _COMPLETED_RUN_RESULT_FIELDS
        or result["complete"] is not True
        or result["run_id"] != run_directory.name
        or result["run_directory"] != str(run_directory)
    ):
        raise ValueError("Task 1 completed result fields are invalid")
    binding_names = (
        "run_identity_sha256",
        "source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "scope",
        "production_evidence",
    )
    if any(completion[name] != result[name] for name in binding_names):
        raise ValueError("Task 1 completion lock differs from its result identity")
    for name in (
        "run_identity_sha256",
        "config_sha256",
        "cache_lock_sha256",
        "weight_sha256",
        "source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
    ):
        if not isinstance(result[name], str) or _SHA256.fullmatch(result[name]) is None:
            raise ValueError(f"Task 1 result {name} is invalid")
    if (
        result["scope"] not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or type(result["production_evidence"]) is not bool
        or result["production_evidence"] is not (result["scope"] == PRODUCTION_SCOPE)
        or (require_production and result["production_evidence"] is not True)
    ):
        raise ValueError("Task 1 result evidence scope is invalid")

    artifacts = result["artifacts"]
    artifact_fields = {
        "resolved_config",
        "run_identity",
        "provenance",
        "epoch_history",
        "best_validation_predictions",
        "best_checkpoint",
        "latest_recovery",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_fields:
        raise ValueError("Task 1 result artifact index is invalid")
    resolved_config, resolved_config_record = _read_task1_bound_json_artifact(
        artifacts["resolved_config"],
        expected_path=run_directory / "resolved_config.json",
        context="Task 1 resolved configuration",
    )
    run_identity, run_identity_record = _read_task1_bound_json_artifact(
        artifacts["run_identity"],
        expected_path=run_directory / "run_identity.json",
        context="Task 1 run identity",
    )
    provenance, _ = _read_task1_bound_json_artifact(
        artifacts["provenance"],
        expected_path=run_directory / "provenance.json",
        context="Task 1 provenance",
    )
    history, _ = _read_task1_bound_json_artifact(
        artifacts["epoch_history"],
        expected_path=run_directory / "epoch_history.json",
        context="Task 1 epoch history",
    )
    prediction_rows, _ = _read_task1_bound_json_artifact(
        artifacts["best_validation_predictions"],
        expected_path=run_directory / "best_validation_predictions.json",
        context="Task 1 validation predictions",
    )
    identity_fields = {
        "schema_version",
        "run_id",
        "task",
        "seed",
        "config_sha256",
        "cache_lock_sha256",
        "weight_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "scope",
        "production_evidence",
    }
    if (
        not isinstance(run_identity, dict)
        or set(run_identity) != identity_fields
        or sha256_json(run_identity) != result["run_identity_sha256"]
        or run_identity["schema_version"] != RUN_SCHEMA_VERSION
        or run_identity["run_id"] != result["run_id"]
        or run_identity["task"] != "task1_classification"
        or type(run_identity["seed"]) is not int
        or run_identity["seed"] not in {13, 37, 71}
        or any(
            run_identity[name] != result[name]
            for name in (
                "config_sha256",
                "cache_lock_sha256",
                "weight_sha256",
                "implementation_sha256",
                "requirements_lock_sha256",
                "numerical_runtime_sha256",
                "scope",
                "production_evidence",
            )
        )
    ):
        raise ValueError("Task 1 run identity fields are invalid")

    config = load_final_task1_config()
    config_sha256 = config_fingerprint(config)
    _, config_file_sha256, _ = _descriptor_snapshot(FINAL_CONFIG_PATH)
    expected_resolved_config = {
        "config_path": FINAL_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config_file_sha256": config_file_sha256,
        "config_sha256": config_sha256,
        "resolved": public_config(config),
    }
    if resolved_config != expected_resolved_config or result["config_sha256"] != config_sha256:
        raise ValueError("Task 1 resolved configuration differs from the locked method")
    provenance_fields = {
        "schema_version",
        "created_at_utc",
        "run_identity_sha256",
        "command",
        "config_path",
        "config_file_sha256",
        "config_sha256",
        "cache_root",
        "cache_lock_sha256",
        "weight_path",
        "weight_sha256",
        "weight_size_bytes",
        "source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_path",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "numerical_runtime",
        "scope",
        "production_evidence",
        "environment",
        "parameter_counts",
        "optimizer_groups",
        "initial_artifacts",
    }
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise ValueError("Task 1 provenance fields are invalid")
    try:
        created_at = datetime.fromisoformat(provenance["created_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Task 1 provenance timestamp is invalid") from exc
    runtime = provenance["numerical_runtime"]
    environment = provenance["environment"]
    parameter_count_record = provenance["parameter_counts"]
    optimizer_groups = provenance["optimizer_groups"]
    if (
        created_at.tzinfo is None
        or provenance["schema_version"] != RUN_SCHEMA_VERSION
        or provenance["run_identity_sha256"] != result["run_identity_sha256"]
        or provenance["config_path"] != FINAL_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
        or provenance["config_file_sha256"] != config_file_sha256
        or provenance["config_sha256"] != config_sha256
        or provenance["cache_lock_sha256"] != result["cache_lock_sha256"]
        or provenance["weight_sha256"] != result["weight_sha256"]
        or provenance["source_fingerprint_sha256"] != result["source_fingerprint_sha256"]
        or provenance["implementation_sha256"] != result["implementation_sha256"]
        or provenance["requirements_lock_sha256"] != result["requirements_lock_sha256"]
        or provenance["numerical_runtime_sha256"] != result["numerical_runtime_sha256"]
        or not isinstance(runtime, dict)
        or sha256_json(runtime) != result["numerical_runtime_sha256"]
        or provenance["scope"] != result["scope"]
        or provenance["production_evidence"] is not result["production_evidence"]
        or provenance["initial_artifacts"]
        != {"resolved_config": resolved_config_record, "run_identity": run_identity_record}
        or not isinstance(provenance["command"], list)
        or any(not isinstance(part, str) for part in provenance["command"])
        or not isinstance(environment, dict)
        or not isinstance(parameter_count_record, dict)
        or set(parameter_count_record) != {"total", "trainable"}
        or any(
            type(parameter_count_record[name]) is not int or parameter_count_record[name] <= 0
            for name in ("total", "trainable")
        )
        or not isinstance(optimizer_groups, list)
        or len(optimizer_groups) != 2
    ):
        raise ValueError("Task 1 provenance does not bind the completed run")
    expected_group_names = ("backbone", "head")
    expected_group_rates = (0.00003, 0.0003)
    for index, group in enumerate(optimizer_groups):
        if (
            not isinstance(group, dict)
            or set(group) != {"name", "learning_rate", "parameters"}
            or group["name"] != expected_group_names[index]
            or group["learning_rate"] != expected_group_rates[index]
            or type(group["parameters"]) is not int
            or group["parameters"] <= 0
        ):
            raise ValueError("Task 1 provenance optimizer groups are invalid")
    if (
        sum(group["parameters"] for group in optimizer_groups)
        != parameter_count_record["trainable"]
    ):
        raise ValueError("Task 1 optimizer groups do not cover the trainable parameters")
    if require_production and (
        result["cache_lock_sha256"] != KNOWN_CACHE_LOCK_SHA256
        or not result["weight_sha256"].startswith(WEIGHT_HASH_PREFIX)
        or result["implementation_sha256"] != _task1_implementation_fingerprint()
        or result["requirements_lock_sha256"] != _requirements_lock_fingerprint()
        or runtime.get("device") != "mps"
        or runtime.get("mps_built") is not True
        or runtime.get("mps_available") is not True
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("mps_fallback") not in {"disabled", "0", "false"}
        or environment.get("device") != "mps"
        or environment.get("mps_built") is not True
        or environment.get("mps_available") is not True
        or environment.get("deterministic_algorithms") is not True
        or parameter_count_record != EXPECTED_TASK1_PARAMETER_COUNTS
        or [group["parameters"] for group in optimizer_groups] != [3_155_740, 19_215]
    ):
        raise ValueError("Task 1 provenance is not current production evidence")

    epochs_completed = result["epochs_completed"]
    latest_path = run_directory / "recovery" / f"recovery_epoch_{epochs_completed:04d}.pt"
    latest_record = _validate_artifact_record(
        result["latest_recovery_checkpoint"],
        expected_path=latest_path,
        context="Task 1 latest recovery checkpoint",
    )
    if latest_record != artifacts["latest_recovery"]:
        raise ValueError("Task 1 latest recovery artifact bindings differ")
    latest_checkpoint = load_task1_checkpoint(
        latest_path,
        expected_sha256=latest_record["sha256"],
        expected_run_identity_sha256=result["run_identity_sha256"],
        expected_type="recovery",
    )
    if require_production:
        _verify_locked_task1_recovery_state(latest_checkpoint, config)
    limits = latest_checkpoint["limits"]
    if require_production and limits != EXPECTED_TASK1_LIMITS:
        raise ValueError("Task 1 run limits are not exactly 30 epochs, batch 32, patience 5")
    selected, no_improvement, should_stop = _validate_task1_history_selection(
        history,
        limits=limits,
        require_production=require_production,
        require_terminal=True,
    )
    if len(history) != epochs_completed:
        raise ValueError("Task 1 history length differs from the completed epoch")

    expected_common = {
        "run_id": result["run_id"],
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "weight_sha256": result["weight_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "scope": result["scope"],
        "production_evidence": result["production_evidence"],
        "seed": run_identity["seed"],
    }
    _validate_completed_run_result(
        result,
        run_directory=run_directory,
        run_identity=run_identity,
        run_identity_sha256=result["run_identity_sha256"],
        expected_common=expected_common,
        source_fingerprint_sha256=result["source_fingerprint_sha256"],
        maximum_epochs=limits["maximum_epochs"],
    )
    best_path = run_directory / "best_candidates" / f"best_epoch_{selected.epoch:04d}.pt"
    best_record = _validate_artifact_record(
        result["best_checkpoint"],
        expected_path=best_path,
        context="Task 1 best checkpoint",
    )
    if best_record != artifacts["best_checkpoint"]:
        raise ValueError("Task 1 best checkpoint artifact bindings differ")
    best_checkpoint = load_task1_checkpoint(
        best_path,
        expected_sha256=best_record["sha256"],
        expected_run_identity_sha256=result["run_identity_sha256"],
        expected_type="best",
    )
    best_score = _validate_score(best_checkpoint["score"])
    latest_best = _validate_score(latest_checkpoint["early_stopping"]["best"])
    candidate = latest_checkpoint["best_candidate"]
    if (
        selected != best_score
        or selected != latest_best
        or selected.epoch != result["best_epoch"]
        or selected.macro_f1 != result["best_validation_macro_f1"]
        or selected.validation_loss != result["best_validation_loss"]
        or latest_checkpoint["completed_epoch"] != epochs_completed
        or latest_checkpoint["history"] != history
        or latest_checkpoint["limits"] != limits
        or latest_checkpoint["early_stopping"]["epochs_without_improvement"] != no_improvement
        or latest_checkpoint["early_stopping"]["patience"] != limits["patience"]
        or latest_checkpoint["stop_requested"] is not should_stop
        or result["early_stopped"] is not should_stop
        or candidate
        != {
            "path": f"best_candidates/best_epoch_{selected.epoch:04d}.pt",
            "sha256": best_record["sha256"],
            "epoch": selected.epoch,
        }
    ):
        raise ValueError("Task 1 history, selected best, or recovery binding is invalid")
    for checkpoint in (best_checkpoint, latest_checkpoint):
        if any(checkpoint.get(name) != value for name, value in expected_common.items()):
            raise ValueError("Task 1 checkpoint identity differs from the completed run")

    predictions = _predictions_from_checkpoint_state(best_checkpoint["predictions"])
    expected_prediction_rows = _prediction_records(predictions)
    if not _json_values_exact(expected_prediction_rows, prediction_rows):
        raise ValueError("Task 1 prediction artifact differs from the selected checkpoint")
    validation_metrics = fixed_class_metrics(
        predictions.true_labels,
        predictions.predicted_labels,
        class_count=len(LOCKED_TASK1_CLASS_ORDER),
    )
    best_validation = history[selected.epoch - 1]["validation"]
    if (
        validation_metrics["macro_f1"] != selected.macro_f1
        or validation_metrics["accuracy"] != best_validation["accuracy"]
        or best_validation["macro_f1"] != selected.macro_f1
        or best_validation["clip_loss"] != selected.validation_loss
    ):
        raise ValueError("Task 1 validation metrics do not rederive from the selected predictions")
    observed_classes = set(predictions.true_labels.tolist())
    if len(predictions.recording_ids) != best_validation["recording_count"]:
        raise ValueError("Task 1 selected prediction count differs from its history")
    if require_production and (
        len(predictions.recording_ids) != EXPECTED_TASK1_VALIDATION_RECORDINGS
        or observed_classes != set(range(len(LOCKED_TASK1_CLASS_ORDER)))
    ):
        raise ValueError("Task 1 production predictions must contain 271 recordings and 15 classes")

    resume_value = result["resume_checkpoint"]
    if result["resumed"] is not (resume_value is not None):
        raise ValueError("Task 1 result resume flag and checkpoint record differ")
    resume_prefix_verified = resume_value is None
    if resume_value is not None:
        if not isinstance(resume_value, dict) or set(resume_value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("Task 1 resume checkpoint record is invalid")
        resume_path = _resolve_project_input_no_follow(resume_value["path"])
        match = re.fullmatch(r"recovery_epoch_(\d{4})\.pt", resume_path.name)
        if match is None or resume_path.parent != run_directory / "recovery":
            raise ValueError("Task 1 resume checkpoint path is not canonical")
        resume_epoch = int(match.group(1))
        if not 1 <= resume_epoch < epochs_completed:
            raise ValueError("Task 1 resume checkpoint epoch is not a strict history prefix")
        _validate_artifact_record(
            resume_value,
            expected_path=resume_path,
            context="Task 1 resume checkpoint",
        )
        resume_checkpoint = load_task1_checkpoint(
            resume_path,
            expected_sha256=resume_value["sha256"],
            expected_run_identity_sha256=result["run_identity_sha256"],
            expected_type="recovery",
        )
        if require_production:
            _verify_locked_task1_recovery_state(resume_checkpoint, config)
        prefix = history[:resume_epoch]
        prefix_best, prefix_no_improvement, prefix_should_stop = _validate_task1_history_selection(
            prefix,
            limits=limits,
            require_production=require_production,
            require_terminal=False,
        )
        resume_candidate = resume_checkpoint["best_candidate"]
        resume_best_path = run_directory / resume_candidate["path"]
        resume_best = load_task1_checkpoint(
            resume_best_path,
            expected_sha256=resume_candidate["sha256"],
            expected_run_identity_sha256=result["run_identity_sha256"],
            expected_type="best",
        )
        if (
            resume_checkpoint["completed_epoch"] != resume_epoch
            or resume_checkpoint["history"] != prefix
            or resume_checkpoint["limits"] != limits
            or resume_checkpoint["stop_requested"] is not prefix_should_stop
            or prefix_should_stop
            or _validate_score(resume_checkpoint["early_stopping"]["best"]) != prefix_best
            or resume_checkpoint["early_stopping"]["epochs_without_improvement"]
            != prefix_no_improvement
            or resume_candidate
            != {
                "path": f"best_candidates/best_epoch_{prefix_best.epoch:04d}.pt",
                "sha256": resume_candidate["sha256"],
                "epoch": prefix_best.epoch,
            }
            or _validate_score(resume_best["score"]) != prefix_best
            or any(
                checkpoint.get(name) != value
                for checkpoint in (resume_checkpoint, resume_best)
                for name, value in expected_common.items()
            )
        ):
            raise ValueError("Task 1 resume checkpoint does not bind the exact history prefix")
        resume_prefix_verified = True

    return {
        "valid": True,
        "complete": True,
        "run_id": result["run_id"],
        "seed": run_identity["seed"],
        "scope": result["scope"],
        "production_evidence": result["production_evidence"],
        "completion_lock_sha256": completion_record["sha256"],
        "run_identity_sha256": result["run_identity_sha256"],
        "best_checkpoint_sha256": best_record["sha256"],
        "validation_recordings": len(predictions.recording_ids),
        "validation_classes": len(observed_classes),
        "macro_f1_rederived": True,
        "selection_rederived": True,
        "resume_prefix_verified": resume_prefix_verified,
    }


def _prediction_records(predictions: RecordingPredictions) -> list[dict[str, Any]]:
    return [
        {
            "recording_id": recording_id,
            "session_group": predictions.session_groups[index],
            "true_class_index": int(predictions.true_labels[index]),
            "predicted_class_index": int(predictions.predicted_labels[index]),
            "mean_logits": [float(value) for value in predictions.mean_logits[index]],
        }
        for index, recording_id in enumerate(predictions.recording_ids)
    ]


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torchvision_version": importlib.metadata.version("torchvision"),
        "numpy_version": np.__version__,
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        "dtype": "torch.float32",
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _resolved_limits(
    config: Mapping[str, Any], injection: Task1TestInjection | None
) -> tuple[int, int, int]:
    training = config["training"]
    maximum_epochs = int(training["maximum_epochs"])
    batch_size = int(training["batch_size"])
    patience = int(training["early_stopping_patience"])
    if injection is not None:
        maximum_epochs = injection.maximum_epochs or maximum_epochs
        batch_size = injection.batch_size or batch_size
        patience = injection.early_stopping_patience or patience
    return maximum_epochs, batch_size, patience


def _validated_session_groups(
    data: Task1Data,
    *,
    expected_split: str,
    class_order: Sequence[str],
) -> tuple[set[str], set[str]]:
    if len(data) <= 0 or data.recording_count <= 0:
        raise ValueError(f"Task 1 {expected_split} data cannot be empty")
    rows = tuple(data.iter_metadata())
    if len(rows) != len(data):
        raise ValueError(f"Task 1 {expected_split} metadata count is inconsistent")
    recordings: dict[str, tuple[str, int, str, int]] = {}
    observed_clip_counts: dict[str, int] = {}
    session_groups: set[str] = set()
    for row in rows:
        if row.get("split") != expected_split:
            raise PermissionError(f"Task 1 {expected_split} data contains another split")
        if row.get("selection_strategy") != "energy":
            raise ValueError(f"Task 1 {expected_split} data contains another strategy")
        recording_id = str(row.get("recording_id") or "")
        session_group = str(row.get("session_group") or "")
        species = str(row.get("species_common_name") or "")
        try:
            class_index = int(row["class_index"])
            declared_clip_count = int(row["strategy_clip_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Task 1 {expected_split} metadata has an invalid class") from exc
        if (
            not recording_id
            or not session_group
            or not 0 <= class_index < len(class_order)
            or not 1 <= declared_clip_count <= 5
            or species != class_order[class_index]
        ):
            raise ValueError(f"Task 1 {expected_split} metadata identity is invalid")
        identity = (session_group, class_index, species, declared_clip_count)
        existing = recordings.setdefault(recording_id, identity)
        if existing != identity:
            raise ValueError(f"Task 1 recording identity changes: {recording_id}")
        observed_clip_counts[recording_id] = observed_clip_counts.get(recording_id, 0) + 1
        session_groups.add(session_group)
    if len(recordings) != data.recording_count:
        raise ValueError(f"Task 1 {expected_split} recording count is inconsistent")
    for recording_id, identity in recordings.items():
        if observed_clip_counts[recording_id] != identity[3]:
            raise ValueError(f"Task 1 selected clip count is inconsistent: {recording_id}")
    return session_groups, set(recordings)


def _validate_development_data(
    train: Task1Data,
    validation: Task1Data,
    config: Mapping[str, Any],
) -> None:
    if train.split != "train" or validation.split != "validation":
        raise PermissionError("Task 1 development engine cannot access the final test split")
    if train.strategy != validation.strategy or train.strategy != "energy":
        raise ValueError("Task 1 development data must use the locked energy strategy")
    if train.lock_sha256 != validation.lock_sha256:
        raise ValueError("Task 1 train and validation caches have different locks")
    if re.fullmatch(r"[0-9a-f]{64}", train.lock_sha256) is None:
        raise ValueError("Task 1 development cache lock SHA-256 is invalid")
    class_order = tuple(str(value) for value in config["class_order"])
    train_sessions, train_recordings = _validated_session_groups(
        train,
        expected_split="train",
        class_order=class_order,
    )
    validation_sessions, validation_recordings = _validated_session_groups(
        validation,
        expected_split="validation",
        class_order=class_order,
    )
    if train_sessions.intersection(validation_sessions):
        raise ValueError("Task 1 train and validation session groups overlap")
    if train_recordings.intersection(validation_recordings):
        raise ValueError("Task 1 train and validation recording IDs overlap")


def _validated_run_output_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    lexical = Path(os.path.abspath(requested))
    canonical_root = Path(os.path.abspath(DEFAULT_RUN_ROOT))
    if DEFAULT_RUN_ROOT.resolve() != canonical_root:
        raise ValueError("Task 1 canonical v2 run root traverses a symbolic link")
    resolved = require_safe_output(lexical)
    if resolved != lexical:
        raise ValueError("Task 1 run output path traverses a symbolic link")
    try:
        resolved.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError("Task 1 run output must stay inside runs/task1_v2") from exc
    return resolved


def _run_directory(output_root: str | Path, run_id: str) -> Path:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Task 1 run ID is unsafe")
    root = _validated_run_output_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    destination.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(root)
    return destination


def _open_real_data(
    stack: ExitStack,
    *,
    cache_root: str | Path,
    ffmpeg: str | Path | None,
    expected_lock_sha256: str | None,
) -> tuple[DevelopmentTrainingData, DevelopmentTrainingData]:
    resolved_root = resolve_project_path(cache_root)
    if resolved_root != DEFAULT_CACHE_ROOT.resolve():
        raise PermissionError("Production Task 1 accepts only the canonical known cache root")
    if expected_lock_sha256 not in {None, KNOWN_CACHE_LOCK_SHA256}:
        raise ValueError("Production Task 1 cache SHA-256 differs from the published lock")
    train = stack.enter_context(
        open_development_training_data(
            resolved_root,
            split="train",
            strategy="energy",
            ffmpeg=ffmpeg,
            expected_lock_sha256=KNOWN_CACHE_LOCK_SHA256,
        )
    )
    validation = stack.enter_context(
        open_development_training_data(
            resolved_root,
            split="validation",
            strategy="energy",
            ffmpeg=ffmpeg,
            expected_lock_sha256=train.lock_sha256,
        )
    )
    if train.lock_sha256 != KNOWN_CACHE_LOCK_SHA256:
        raise ValueError("Production Task 1 opened an unexpected known-cache lock")
    return train, validation


def _score_state(score: CheckpointScore) -> dict[str, Any]:
    return {
        "macro_f1": score.macro_f1,
        "validation_loss": score.validation_loss,
        "epoch": score.epoch,
    }


def _checkpoint_common_state(
    *,
    run_id: str,
    run_identity_sha256: str,
    config_sha256: str,
    cache_lock_sha256: str,
    weight_sha256: str,
    execution_identity: Task1ExecutionIdentity,
    scope: str,
    production_evidence: bool,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "config_sha256": config_sha256,
        "cache_lock_sha256": cache_lock_sha256,
        "weight_sha256": weight_sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "scope": scope,
        "production_evidence": production_evidence,
        "seed": seed,
    }


def _ensure_run_subdirectories(run_directory: Path) -> None:
    for name in ("best_candidates", "recovery", "failures"):
        destination = run_directory / name
        destination.mkdir(mode=0o700, exist_ok=True)
        _fsync_directory(destination)
    _fsync_directory(run_directory)


def _save_or_verify_checkpoint(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return save_task1_checkpoint_create_only(path, state)
    except FileExistsError:
        payload, sha256, size_bytes = _descriptor_snapshot(path)
        observed = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        _validate_checkpoint_state(observed)
        _assert_round_trip(_cpu_copy(dict(state)), observed)
        return {"path": str(path), "sha256": sha256, "size_bytes": size_bytes}


def _validate_optimizer_after_resume(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    config: Mapping[str, Any],
) -> None:
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(features, nn.Module) or not isinstance(classifier, nn.Module):
        raise TypeError("Task 1 resumed model lacks optimizer partition modules")
    training = config["training"]
    expected = (
        (
            "backbone",
            float(training["backbone_learning_rate"]),
            tuple(parameter for parameter in features.parameters() if parameter.requires_grad),
        ),
        (
            "head",
            float(training["head_learning_rate"]),
            tuple(parameter for parameter in classifier.parameters() if parameter.requires_grad),
        ),
    )
    if len(optimizer.param_groups) != len(expected):
        raise ValueError("Recovery optimizer group count is invalid")
    expected_option_keys = {
        "amsgrad",
        "betas",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "group_name",
        "lr",
        "maximize",
        "weight_decay",
    }
    common_options = {
        "betas": ADAMW_BETAS,
        "eps": ADAMW_EPS,
        "weight_decay": float(training["weight_decay"]),
        "amsgrad": ADAMW_AMSGRAD,
        "maximize": ADAMW_MAXIMIZE,
        "foreach": ADAMW_FOREACH,
        "capturable": ADAMW_CAPTURABLE,
        "differentiable": ADAMW_DIFFERENTIABLE,
        "fused": ADAMW_FUSED,
        "decoupled_weight_decay": True,
    }
    expected_parameters: set[nn.Parameter] = set()
    for group, (name, learning_rate, parameters) in zip(
        optimizer.param_groups,
        expected,
        strict=True,
    ):
        observed_options = {key: value for key, value in group.items() if key != "params"}
        expected_options = {
            **common_options,
            "group_name": name,
            "lr": learning_rate,
        }
        observed_parameters = tuple(group.get("params", ()))
        if (
            set(observed_options) != expected_option_keys
            or observed_options != expected_options
            or tuple(id(parameter) for parameter in observed_parameters)
            != tuple(id(parameter) for parameter in parameters)
        ):
            raise ValueError("Recovery optimizer settings differ from the locked method")
        expected_parameters.update(parameters)
    trainable_parameters = {
        parameter for parameter in model.parameters() if parameter.requires_grad
    }
    if expected_parameters != trainable_parameters or set(optimizer.state) != trainable_parameters:
        raise ValueError("Recovery optimizer state does not match the trainable partition")
    finite_state_parts: list[torch.Tensor] = []
    for parameter in trainable_parameters:
        state = optimizer.state[parameter]
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Recovery optimizer tensor state schema is invalid")
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if (
            not torch.is_tensor(step)
            or step.numel() != 1
            or step.device.type != "cpu"
            or not bool(torch.isfinite(step).all())
            or float(step) < 1.0
            or not torch.is_tensor(exp_avg)
            or not torch.is_tensor(exp_avg_sq)
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
            or exp_avg.dtype != parameter.dtype
            or exp_avg_sq.dtype != parameter.dtype
            or exp_avg.device != parameter.device
            or exp_avg_sq.device != parameter.device
        ):
            raise ValueError("Recovery optimizer tensor state is invalid")
        finite_state_parts.extend((torch.isfinite(exp_avg).all(), torch.isfinite(exp_avg_sq).all()))
    finite_state = torch.stack(finite_state_parts).all().to(device="cpu")
    if not bool(finite_state):
        raise ValueError("Recovery optimizer tensor state is non-finite")


def _failure_path(run_directory: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return run_directory / "failures" / f"failure_{timestamp}.json"


def _resolve_resume_directory(
    output_root: str | Path,
    checkpoint_path: str | Path,
) -> tuple[Path, Path]:
    root = _validated_run_output_path(output_root)
    checkpoint = _resolve_project_input_no_follow(checkpoint_path)
    run_directory = checkpoint.parent.parent
    if (
        checkpoint.parent.name != "recovery"
        or run_directory.parent.resolve() != root.resolve()
        or not run_directory.is_dir()
        or not is_relative_to(checkpoint, run_directory)
    ):
        raise PermissionError("Task 1 resume checkpoint is outside the selected run directory")
    return run_directory, checkpoint


def run_task1_development(
    *,
    seed: int,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
    output_root: str | Path = DEFAULT_RUN_ROOT,
    command: Sequence[str] = (),
    run_id: str | None = None,
    train_data: Task1Data | None = None,
    validation_data: Task1Data | None = None,
    test_injection: Task1TestInjection | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_final_task1_config()
    if seed not in config["seeds"]:
        raise ValueError("Task 1 seed is outside the locked seed set")
    if (resume_checkpoint is None) != (resume_checkpoint_sha256 is None):
        raise ValueError("Task 1 resume path and SHA-256 must be supplied together")
    device = _resolve_runtime(test_injection)
    scope = ISOLATED_TEST_SCOPE if test_injection is not None else PRODUCTION_SCOPE
    production_evidence = scope == PRODUCTION_SCOPE
    validated_output_root = _validated_run_output_path(output_root)
    canonical_output_root = Path(os.path.abspath(DEFAULT_RUN_ROOT))
    if production_evidence and validated_output_root != canonical_output_root:
        raise PermissionError("Production Task 1 run output must use exact runs/task1_v2")
    if test_injection is not None and validated_output_root == canonical_output_root:
        raise PermissionError("Isolated Task 1 tests cannot publish in the production run root")
    execution_identity = _capture_execution_identity(device)
    release_source_fingerprint_sha256 = source_fingerprint()
    weight = (
        preflight_efficientnet_weights(populate=False)
        if test_injection is None
        else test_injection.weight_artifact
    )
    _require_weight_unchanged(weight)

    with ExitStack() as stack:
        if (train_data is None) != (validation_data is None):
            raise ValueError("Task 1 train and validation data must be supplied together")
        if train_data is None or validation_data is None:
            if test_injection is not None:
                raise ValueError("CPU test injection requires explicit development fixtures")
            train, validation = _open_real_data(
                stack,
                cache_root=cache_root,
                ffmpeg=ffmpeg,
                expected_lock_sha256=expected_lock_sha256,
            )
        else:
            if test_injection is None:
                raise PermissionError("Explicit data injection is allowed only for isolated tests")
            train, validation = train_data, validation_data
        _validate_development_data(train, validation, config)
        _require_execution_identity_unchanged(execution_identity, device)

        config_sha256 = config_fingerprint(config)
        _, config_file_sha256, _ = _descriptor_snapshot(FINAL_CONFIG_PATH)
        maximum_epochs, batch_size, patience = _resolved_limits(config, test_injection)
        limits = {
            "maximum_epochs": maximum_epochs,
            "batch_size": batch_size,
            "patience": patience,
        }
        recovery_checkpoint: dict[str, Any] | None = None
        resume_record: dict[str, Any] | None = None
        resumed = resume_checkpoint is not None
        if resumed:
            if resume_checkpoint_sha256 is None or resume_checkpoint is None:
                raise RuntimeError("Task 1 resume arguments became inconsistent")
            run_directory, resolved_resume = _resolve_resume_directory(
                validated_output_root,
                resume_checkpoint,
            )
            recovery_checkpoint = load_task1_checkpoint(
                resolved_resume,
                expected_sha256=resume_checkpoint_sha256,
                expected_type="recovery",
            )
            expected_resume_name = (
                f"recovery_epoch_{int(recovery_checkpoint['completed_epoch']):04d}.pt"
            )
            available_recoveries = sorted((run_directory / "recovery").glob("recovery_epoch_*.pt"))
            if (
                resolved_resume.name != expected_resume_name
                or not available_recoveries
                or resolved_resume != available_recoveries[-1].resolve()
            ):
                raise ValueError("Task 1 resume requires the latest canonical recovery checkpoint")
            selected_run_id = str(recovery_checkpoint["run_id"])
            if run_id is not None and run_id != selected_run_id:
                raise ValueError("Requested run ID differs from the recovery checkpoint")
            if run_directory.name != selected_run_id:
                raise ValueError("Recovery checkpoint run ID differs from its directory")
            if recovery_checkpoint["limits"] != limits:
                raise ValueError("Recovery limits differ from the current locked run")
            resume_record = {
                "path": str(resolved_resume),
                "sha256": resume_checkpoint_sha256,
                "size_bytes": resolved_resume.stat().st_size,
            }
        else:
            selected_run_id = run_id or make_run_id(
                "task1",
                "final",
                seed,
                config_sha256,
                train.lock_sha256,
            )
            run_directory = _run_directory(validated_output_root, selected_run_id)
        _ensure_run_subdirectories(run_directory)
        run_identity = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "task": "task1_classification",
            "seed": seed,
            "config_sha256": config_sha256,
            "cache_lock_sha256": train.lock_sha256,
            "weight_sha256": weight.sha256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "scope": scope,
            "production_evidence": production_evidence,
        }
        run_identity_sha256 = sha256_json(run_identity)
        expected_common = _checkpoint_common_state(
            run_id=selected_run_id,
            run_identity_sha256=run_identity_sha256,
            config_sha256=config_sha256,
            cache_lock_sha256=train.lock_sha256,
            weight_sha256=weight.sha256,
            execution_identity=execution_identity,
            scope=scope,
            production_evidence=production_evidence,
            seed=seed,
        )
        if recovery_checkpoint is not None:
            identity_first = (
                "implementation_sha256",
                "requirements_lock_sha256",
                "numerical_runtime_sha256",
                "scope",
                "production_evidence",
            )
            remaining_common = tuple(key for key in expected_common if key not in identity_first)
            for key in (*identity_first, *remaining_common):
                value = expected_common[key]
                if recovery_checkpoint.get(key) != value:
                    raise ValueError(
                        f"Recovery checkpoint changed locked run identity field: {key}"
                    )
            result_path = run_directory / "result.json"
            if _artifact_path_exists(result_path):
                _require_execution_identity_unchanged(execution_identity, device)
                completed_result, result_record = _read_json_snapshot(result_path)
                _validate_completed_run_result(
                    completed_result,
                    run_directory=run_directory,
                    run_identity=run_identity,
                    run_identity_sha256=run_identity_sha256,
                    expected_common=expected_common,
                    source_fingerprint_sha256=release_source_fingerprint_sha256,
                    maximum_epochs=maximum_epochs,
                    expected_latest_recovery_sha256=resume_checkpoint_sha256,
                )
                completion_value = {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "run_identity_sha256": run_identity_sha256,
                    "source_fingerprint_sha256": release_source_fingerprint_sha256,
                    "implementation_sha256": execution_identity.implementation_sha256,
                    "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                    "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                    "scope": scope,
                    "production_evidence": production_evidence,
                    "result": result_record,
                }
                completion_path = run_directory / "result.lock.json"
                if completion_path.is_symlink():
                    raise ValueError("Task 1 completion lock cannot be a symbolic link")
                completion_record = _write_or_verify_json(
                    completion_path,
                    completion_value,
                )
                return {
                    **completed_result,
                    "result_artifact": result_record,
                    "completion_lock_artifact": completion_record,
                }
        history: list[dict[str, Any]] = []
        artifact_records: dict[str, Any] = {}
        try:
            resolved_config_value = {
                "config_path": FINAL_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "config_file_sha256": config_file_sha256,
                "config_sha256": config_sha256,
                "resolved": public_config(config),
            }
            if resumed:
                observed_config, config_record = _read_json_snapshot(
                    run_directory / "resolved_config.json"
                )
                observed_identity, identity_record = _read_json_snapshot(
                    run_directory / "run_identity.json"
                )
                observed_provenance, provenance_record = _read_json_snapshot(
                    run_directory / "provenance.json"
                )
                if observed_config != resolved_config_value or observed_identity != run_identity:
                    raise ValueError(
                        "Recovery run artifacts differ from the current locked identity"
                    )
                if (
                    not isinstance(observed_provenance, dict)
                    or observed_provenance.get("run_identity_sha256") != run_identity_sha256
                    or observed_provenance.get("config_sha256") != config_sha256
                    or observed_provenance.get("cache_lock_sha256") != train.lock_sha256
                    or observed_provenance.get("weight_sha256") != weight.sha256
                    or observed_provenance.get("implementation_sha256")
                    != execution_identity.implementation_sha256
                    or observed_provenance.get("requirements_lock_sha256")
                    != execution_identity.requirements_lock_sha256
                    or observed_provenance.get("numerical_runtime_sha256")
                    != execution_identity.numerical_runtime_sha256
                    or observed_provenance.get("numerical_runtime")
                    != execution_identity.numerical_runtime
                    or observed_provenance.get("scope") != scope
                    or observed_provenance.get("production_evidence") is not production_evidence
                    or _SHA256.fullmatch(
                        str(observed_provenance.get("source_fingerprint_sha256") or "")
                    )
                    is None
                ):
                    raise ValueError("Recovery provenance differs from the current locked identity")
                release_source_fingerprint_sha256 = observed_provenance["source_fingerprint_sha256"]
                artifact_records.update(
                    {
                        "resolved_config": config_record,
                        "run_identity": identity_record,
                        "provenance": provenance_record,
                    }
                )
            else:
                config_record = _write_json_create_only(
                    run_directory / "resolved_config.json",
                    resolved_config_value,
                )
                identity_record = _write_json_create_only(
                    run_directory / "run_identity.json",
                    run_identity,
                )
                artifact_records.update(
                    {"resolved_config": config_record, "run_identity": identity_record}
                )
            seed_task1(seed, device)
            model = _build_model(config, device, weight, test_injection)
            optimizer = build_task1_optimizer(model, config)
            counts = parameter_counts(model)
            if not resumed:
                provenance = {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "run_identity_sha256": run_identity_sha256,
                    "command": [str(part) for part in command],
                    "config_path": FINAL_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "config_file_sha256": config_file_sha256,
                    "config_sha256": config_sha256,
                    "cache_root": str(train.root),
                    "cache_lock_sha256": train.lock_sha256,
                    "weight_path": str(weight.path),
                    "weight_sha256": weight.sha256,
                    "weight_size_bytes": weight.size_bytes,
                    "source_fingerprint_sha256": release_source_fingerprint_sha256,
                    "implementation_sha256": execution_identity.implementation_sha256,
                    "requirements_lock_path": str(REQUIREMENTS_LOCK_PATH),
                    "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                    "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                    "numerical_runtime": execution_identity.numerical_runtime,
                    "scope": scope,
                    "production_evidence": production_evidence,
                    "environment": _environment(device),
                    "parameter_counts": counts,
                    "optimizer_groups": [
                        {
                            "name": group["group_name"],
                            "learning_rate": group["lr"],
                            "parameters": sum(parameter.numel() for parameter in group["params"]),
                        }
                        for group in optimizer.param_groups
                    ],
                    "initial_artifacts": {
                        "resolved_config": artifact_records["resolved_config"],
                        "run_identity": artifact_records["run_identity"],
                    },
                }
                artifact_records["provenance"] = _write_json_create_only(
                    run_directory / "provenance.json",
                    provenance,
                )

            early_stopping = EarlyStopping(patience)
            best_candidate: dict[str, Any] | None = None
            best_checkpoint_record: dict[str, Any] | None = None
            latest_recovery_record = resume_record
            start_epoch_index = 0
            stop_requested = False
            if recovery_checkpoint is not None:
                model.load_state_dict(recovery_checkpoint["model"], strict=True)
                optimizer.load_state_dict(recovery_checkpoint["optimizer"])
                _validate_optimizer_after_resume(optimizer, model, config)
                early_state = recovery_checkpoint["early_stopping"]
                early_stopping.best = _validate_score(early_state["best"])
                early_stopping.epochs_without_improvement = early_state[
                    "epochs_without_improvement"
                ]
                best_candidate = dict(recovery_checkpoint["best_candidate"])
                candidate_path = run_directory / best_candidate["path"]
                candidate_checkpoint = load_task1_checkpoint(
                    candidate_path,
                    expected_sha256=best_candidate["sha256"],
                    expected_run_identity_sha256=run_identity_sha256,
                    expected_type="best",
                )
                for key, value in expected_common.items():
                    if candidate_checkpoint.get(key) != value:
                        raise ValueError(f"Best candidate changed locked run field: {key}")
                candidate_score = _validate_score(candidate_checkpoint["score"])
                if candidate_score != early_stopping.best:
                    raise ValueError("Recovery best score differs from its durable candidate")
                best_checkpoint_record = {
                    "path": str(candidate_path),
                    "sha256": best_candidate["sha256"],
                    "size_bytes": candidate_path.stat().st_size,
                }
                history = list(recovery_checkpoint["history"])
                start_epoch_index = int(recovery_checkpoint["next_epoch_index"])
                stop_requested = bool(recovery_checkpoint["stop_requested"])
                _restore_rng_state(recovery_checkpoint["rng_state"], device)

            if not stop_requested:
                epoch_indices = range(start_epoch_index, maximum_epochs)
            else:
                epoch_indices = range(0)
            for epoch_index in epoch_indices:
                _require_execution_identity_unchanged(execution_identity, device)
                epoch_started = time.perf_counter()
                train_metrics = train_task1_epoch(
                    model,
                    optimizer,
                    train,
                    seed=seed,
                    epoch_index=epoch_index,
                    batch_size=batch_size,
                    class_count=int(config["class_count"]),
                    device=device,
                )
                validation_metrics = validate_task1(
                    model,
                    validation,
                    batch_size=batch_size,
                    class_count=int(config["class_count"]),
                    device=device,
                )
                score = CheckpointScore(
                    validation_metrics.macro_f1,
                    validation_metrics.clip_loss,
                    epoch_index + 1,
                )
                improved, should_stop = early_stopping.update(score)
                history.append(
                    {
                        "epoch": epoch_index + 1,
                        "elapsed_seconds": time.perf_counter() - epoch_started,
                        "train": train_metrics,
                        "validation": {
                            "clip_loss": validation_metrics.clip_loss,
                            "clip_count": validation_metrics.clip_count,
                            "recording_count": validation_metrics.recording_count,
                            "macro_f1": validation_metrics.macro_f1,
                            "accuracy": validation_metrics.accuracy,
                        },
                        "checkpoint_improved": improved,
                    }
                )
                _require_execution_identity_unchanged(execution_identity, device)
                _synchronize(device)
                model_state = _cpu_copy(model.state_dict())
                optimizer_state = _cpu_copy(optimizer.state_dict())
                if improved:
                    best_state = {
                        **expected_common,
                        "checkpoint_type": "best",
                        "epoch": epoch_index + 1,
                        "score": _score_state(score),
                        "model": model_state,
                        "optimizer": optimizer_state,
                        "predictions": _prediction_checkpoint_state(validation_metrics.predictions),
                    }
                    candidate_relative = f"best_candidates/best_epoch_{epoch_index + 1:04d}.pt"
                    best_checkpoint_record = _save_or_verify_checkpoint(
                        run_directory / candidate_relative,
                        best_state,
                    )
                    best_candidate = {
                        "path": candidate_relative,
                        "sha256": best_checkpoint_record["sha256"],
                        "epoch": epoch_index + 1,
                    }
                if early_stopping.best is None or best_candidate is None:
                    raise RuntimeError("Task 1 epoch has no durable best candidate")
                recovery_state = {
                    **expected_common,
                    "checkpoint_type": "recovery",
                    "completed_epoch": epoch_index + 1,
                    "next_epoch_index": epoch_index + 1,
                    "stop_requested": should_stop,
                    "limits": limits,
                    "early_stopping": {
                        "best": _score_state(early_stopping.best),
                        "epochs_without_improvement": early_stopping.epochs_without_improvement,
                        "patience": early_stopping.patience,
                    },
                    "best_candidate": best_candidate,
                    "history": history,
                    "model": model_state,
                    "optimizer": optimizer_state,
                    "rng_state": _capture_rng_state(device),
                }
                latest_recovery_record = _save_or_verify_checkpoint(
                    run_directory / "recovery" / f"recovery_epoch_{epoch_index + 1:04d}.pt",
                    recovery_state,
                )
                if should_stop:
                    stop_requested = True
                    break
            if (
                best_candidate is None
                or best_checkpoint_record is None
                or early_stopping.best is None
                or latest_recovery_record is None
            ):
                raise RuntimeError("Task 1 training completed without a best checkpoint")
            best_checkpoint = load_task1_checkpoint(
                best_checkpoint_record["path"],
                expected_sha256=best_checkpoint_record["sha256"],
                expected_run_identity_sha256=run_identity_sha256,
                expected_type="best",
            )
            best_predictions = _predictions_from_checkpoint_state(best_checkpoint["predictions"])
            artifact_records["epoch_history"] = _write_or_verify_json(
                run_directory / "epoch_history.json",
                history,
            )
            artifact_records["best_validation_predictions"] = _write_or_verify_json(
                run_directory / "best_validation_predictions.json",
                _prediction_records(best_predictions),
            )
            artifact_records["best_checkpoint"] = best_checkpoint_record
            artifact_records["latest_recovery"] = latest_recovery_record
            _require_execution_identity_unchanged(execution_identity, device)
            _require_weight_unchanged(weight)
            result = {
                "schema_version": RUN_SCHEMA_VERSION,
                "complete": True,
                "run_id": selected_run_id,
                "run_directory": str(run_directory),
                "run_identity_sha256": run_identity_sha256,
                "config_sha256": config_sha256,
                "cache_lock_sha256": train.lock_sha256,
                "weight_sha256": weight.sha256,
                "source_fingerprint_sha256": release_source_fingerprint_sha256,
                "implementation_sha256": execution_identity.implementation_sha256,
                "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                "scope": scope,
                "production_evidence": production_evidence,
                "resumed": resumed,
                "resume_checkpoint": resume_record,
                "epochs_completed": len(history),
                "early_stopped": stop_requested,
                "best_epoch": early_stopping.best.epoch,
                "best_validation_macro_f1": early_stopping.best.macro_f1,
                "best_validation_loss": early_stopping.best.validation_loss,
                "best_checkpoint": best_checkpoint_record,
                "latest_recovery_checkpoint": latest_recovery_record,
                "artifacts": artifact_records,
            }
            _validate_completed_run_result(
                result,
                run_directory=run_directory,
                run_identity=run_identity,
                run_identity_sha256=run_identity_sha256,
                expected_common=expected_common,
                source_fingerprint_sha256=release_source_fingerprint_sha256,
                maximum_epochs=maximum_epochs,
            )
            result_record = _write_or_verify_json(run_directory / "result.json", result)
            completion_path = run_directory / "result.lock.json"
            if completion_path.is_symlink():
                raise ValueError("Task 1 completion lock cannot be a symbolic link")
            completion_record = _write_or_verify_json(
                completion_path,
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "run_identity_sha256": run_identity_sha256,
                    "source_fingerprint_sha256": release_source_fingerprint_sha256,
                    "implementation_sha256": execution_identity.implementation_sha256,
                    "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                    "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                    "scope": scope,
                    "production_evidence": production_evidence,
                    "result": result_record,
                },
            )
            return {
                **result,
                "result_artifact": result_record,
                "completion_lock_artifact": completion_record,
            }
        except BaseException as exc:
            diagnostic = {
                "schema_version": RUN_SCHEMA_VERSION,
                "complete": False,
                "run_id": selected_run_id,
                "run_identity_sha256": run_identity_sha256,
                "source_fingerprint_sha256": release_source_fingerprint_sha256,
                "implementation_sha256": execution_identity.implementation_sha256,
                "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                "scope": scope,
                "production_evidence": production_evidence,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "epochs_completed": len(history),
                "history": history,
                "traceback": traceback.format_exc(),
            }
            _write_json_create_only(_failure_path(run_directory), diagnostic)
            raise


def _warmup_task1(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Task1Data,
    *,
    seed: int,
    batch_size: int,
    class_count: int,
    device: torch.device,
) -> None:
    sample_count = min(batch_size, len(data))
    native = collate_native_samples([data[index] for index in range(sample_count)])
    generator = make_epoch_cpu_generator(seed, 0, SPECAUGMENT_RANDOM_STREAM)
    augmented = apply_locked_specaugment(native.tensor, generator=generator)
    inputs = to_efficientnet_batch(augmented).to(device)
    targets = _labels(native.metadata, class_count).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    _synchronize(device)


def _benchmark_artifact_paths(
    test_injection: Task1TestInjection | None,
    requested_output: str | Path | None,
) -> tuple[Path, Path] | None:
    canonical_result = Path(os.path.abspath(DEFAULT_BENCHMARK_RESULT_PATH))
    canonical_lock = Path(os.path.abspath(DEFAULT_BENCHMARK_LOCK_PATH))
    if canonical_result != DEFAULT_BENCHMARK_RESULT_PATH.resolve() or (
        _artifact_path_exists(canonical_result) and canonical_result.is_symlink()
    ):
        raise ValueError("Canonical Task 1 benchmark result path contains a symbolic link")
    if canonical_lock != DEFAULT_BENCHMARK_LOCK_PATH.resolve() or (
        _artifact_path_exists(canonical_lock) and canonical_lock.is_symlink()
    ):
        raise ValueError("Canonical Task 1 benchmark lock path contains a symbolic link")
    if test_injection is None:
        if requested_output is not None:
            requested = Path(requested_output).expanduser()
            if not requested.is_absolute():
                requested = PROJECT_ROOT / requested
            requested = Path(os.path.abspath(requested))
            if requested != require_safe_output(requested) or requested != canonical_result:
                raise PermissionError(
                    "Production Task 1 benchmark output is canonical and versioned"
                )
        return canonical_result, canonical_lock
    if requested_output is None:
        return None
    requested = Path(requested_output).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    requested = Path(os.path.abspath(requested))
    result_path = require_safe_output(requested)
    if result_path != requested:
        raise ValueError("Isolated benchmark output path contains a symbolic link")
    if not is_relative_to(result_path, DEFAULT_RUN_ROOT):
        raise PermissionError("Isolated benchmark evidence must stay inside runs/task1_v2")
    if result_path == DEFAULT_BENCHMARK_RESULT_PATH.resolve():
        raise PermissionError("Isolated benchmark cannot publish production evidence")
    if result_path.suffix != ".json" or result_path.name.endswith(".lock.json"):
        raise ValueError("Isolated benchmark output must be a non-lock JSON path")
    lock_path = result_path.with_name(f"{result_path.stem}.lock.json")
    return result_path, lock_path


_BENCHMARK_RESULT_FIELDS = {
    "schema_version",
    "benchmark_only",
    "persistent_model_selection",
    "persistent_model_checkpoint",
    "durable_evidence",
    "scope",
    "production_evidence",
    "started_at_utc",
    "completed_at_utc",
    "command",
    "benchmark_identity_sha256",
    "benchmark_identity",
    "seed",
    "device",
    "batch_size",
    "maximum_epochs",
    "stability_seeds",
    "config_path",
    "config_file_sha256",
    "config_sha256",
    "cache_root",
    "cache_lock_sha256",
    "weight_identifier",
    "weight_sha256",
    "weight_size_bytes",
    "source_fingerprint_sha256",
    "implementation_sha256",
    "requirements_lock_path",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "numerical_runtime",
    "parameter_counts",
    "warmup_completed",
    "train_recordings",
    "train_clips",
    "train_batches",
    "validation_clips",
    "validation_recordings",
    "validation_batches",
    "train_seconds",
    "validation_seconds",
    "full_epoch_seconds",
    "checkpoint_cpu_copy_seconds",
    "checkpoint_serialization_seconds",
    "representative_checkpoint_bytes",
    "checkpoint_serializations_per_epoch_estimate",
    "checkpoint_overhead_per_epoch_estimate_seconds",
    "estimated_epoch_with_checkpoint_seconds",
    "train_clips_per_second",
    "validation_clips_per_second",
    "estimated_one_seed_pre_allowance_seconds",
    "estimated_all_seed_pre_allowance_seconds",
    "conservative_wall_time_factor",
    "conservative_wall_time_scope",
    "estimated_one_seed_conservative_wall_seconds",
    "estimated_all_seed_conservative_wall_seconds",
    "train_clip_loss",
    "validation_clip_loss",
    "validation_recording_macro_f1",
    "validation_recording_accuracy",
    "evidence_scope",
}


def _require_finite_float(
    value: Any,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite float")
    if minimum is not None and (value <= minimum if strict_minimum else value < minimum):
        raise ValueError(f"{context} is below its permitted range")
    if maximum is not None and value > maximum:
        raise ValueError(f"{context} is above its permitted range")
    return value


def _validate_benchmark_result(
    value: Any,
    *,
    benchmark_identity: Mapping[str, Any],
    benchmark_identity_sha256: str,
    config_file_sha256: str,
    config_sha256: str,
    source_fingerprint_sha256: str,
    execution_identity: Task1ExecutionIdentity,
    weight: WeightArtifact,
    train: Task1Data,
    validation: Task1Data,
    seed: int,
    device: torch.device,
    batch_size: int,
    maximum_epochs: int,
    stability_seeds: Sequence[int],
    scope: str,
    production_evidence: bool,
) -> None:
    if not isinstance(value, dict) or set(value) != _BENCHMARK_RESULT_FIELDS:
        raise ValueError("Task 1 benchmark result schema is invalid")
    fixed = {
        "schema_version": RUN_SCHEMA_VERSION,
        "benchmark_only": True,
        "persistent_model_selection": False,
        "persistent_model_checkpoint": False,
        "durable_evidence": True,
        "scope": scope,
        "production_evidence": production_evidence,
        "benchmark_identity_sha256": benchmark_identity_sha256,
        "benchmark_identity": dict(benchmark_identity),
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "maximum_epochs": maximum_epochs,
        "stability_seeds": list(stability_seeds),
        "config_path": FINAL_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config_file_sha256": config_file_sha256,
        "config_sha256": config_sha256,
        "cache_root": str(train.root),
        "cache_lock_sha256": train.lock_sha256,
        "weight_identifier": weight.identifier,
        "weight_sha256": weight.sha256,
        "weight_size_bytes": weight.size_bytes,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_path": REQUIREMENTS_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "numerical_runtime": execution_identity.numerical_runtime,
        "warmup_completed": True,
        "train_recordings": train.recording_count,
        "train_clips": len(train),
        "validation_clips": len(validation),
        "validation_batches": math.ceil(len(validation) / batch_size),
        "checkpoint_serializations_per_epoch_estimate": 2,
        "conservative_wall_time_factor": CONSERVATIVE_WALL_TIME_FACTOR,
        "conservative_wall_time_scope": (
            "Measured epoch compute and checkpoint CPU copy with two in-memory "
            "serializations, plus a 25 percent allowance for artifact writes, startup, "
            "and thermal variation"
        ),
        "evidence_scope": (
            "Runtime planning evidence only; no checkpoint is retained and no model "
            "selection decision is made"
        ),
    }
    mismatched_fields = [
        key
        for key, expected in fixed.items()
        if key not in value or not _json_values_exact(expected, value[key])
    ]
    if mismatched_fields:
        raise ValueError(
            f"Task 1 benchmark result differs from its locked identity: {mismatched_fields}"
        )
    if sha256_json(dict(benchmark_identity)) != benchmark_identity_sha256:
        raise ValueError("Task 1 benchmark identity hash is inconsistent")
    if not isinstance(value["command"], list) or any(
        not isinstance(part, str) for part in value["command"]
    ):
        raise ValueError("Task 1 benchmark command schema is invalid")
    try:
        started = datetime.fromisoformat(value["started_at_utc"])
        completed = datetime.fromisoformat(value["completed_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Task 1 benchmark timestamps are invalid") from exc
    if started.tzinfo is None or completed.tzinfo is None or completed < started:
        raise ValueError("Task 1 benchmark timestamps are inconsistent")

    counts = value["parameter_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"total", "trainable"}
        or any(isinstance(item, bool) or not isinstance(item, int) for item in counts.values())
        or not 0 < counts["trainable"] <= counts["total"]
    ):
        raise ValueError("Task 1 benchmark parameter counts are invalid")
    for key, expected in (
        ("train_batches", math.ceil(len(train) / batch_size)),
        ("validation_recordings", validation.recording_count),
        ("representative_checkpoint_bytes", None),
    ):
        observed = value[key]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed <= 0
            or (expected is not None and observed != expected)
        ):
            raise ValueError(f"Task 1 benchmark {key} is invalid")

    train_seconds = _require_finite_float(
        value["train_seconds"], context="train_seconds", minimum=0.0, strict_minimum=True
    )
    validation_seconds = _require_finite_float(
        value["validation_seconds"],
        context="validation_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    full_epoch_seconds = _require_finite_float(
        value["full_epoch_seconds"],
        context="full_epoch_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    checkpoint_copy = _require_finite_float(
        value["checkpoint_cpu_copy_seconds"],
        context="checkpoint_cpu_copy_seconds",
        minimum=0.0,
    )
    checkpoint_serialization = _require_finite_float(
        value["checkpoint_serialization_seconds"],
        context="checkpoint_serialization_seconds",
        minimum=0.0,
    )
    checkpoint_overhead = _require_finite_float(
        value["checkpoint_overhead_per_epoch_estimate_seconds"],
        context="checkpoint_overhead_per_epoch_estimate_seconds",
        minimum=0.0,
    )
    epoch_with_checkpoint = _require_finite_float(
        value["estimated_epoch_with_checkpoint_seconds"],
        context="estimated_epoch_with_checkpoint_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    one_seed = _require_finite_float(
        value["estimated_one_seed_pre_allowance_seconds"],
        context="estimated_one_seed_pre_allowance_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    all_seeds = _require_finite_float(
        value["estimated_all_seed_pre_allowance_seconds"],
        context="estimated_all_seed_pre_allowance_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    one_seed_conservative = _require_finite_float(
        value["estimated_one_seed_conservative_wall_seconds"],
        context="estimated_one_seed_conservative_wall_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    all_seed_conservative = _require_finite_float(
        value["estimated_all_seed_conservative_wall_seconds"],
        context="estimated_all_seed_conservative_wall_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    expected_relations = (
        (full_epoch_seconds, train_seconds + validation_seconds),
        (checkpoint_overhead, checkpoint_copy + 2.0 * checkpoint_serialization),
        (epoch_with_checkpoint, full_epoch_seconds + checkpoint_overhead),
        (value["train_clips_per_second"], len(train) / train_seconds),
        (value["validation_clips_per_second"], len(validation) / validation_seconds),
        (one_seed, epoch_with_checkpoint * maximum_epochs),
        (all_seeds, one_seed * len(stability_seeds)),
        (one_seed_conservative, one_seed * CONSERVATIVE_WALL_TIME_FACTOR),
        (all_seed_conservative, all_seeds * CONSERVATIVE_WALL_TIME_FACTOR),
    )
    for observed, expected in expected_relations:
        if (
            type(observed) is not float
            or not math.isfinite(observed)
            or not math.isclose(
                observed,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Task 1 benchmark timing relations are inconsistent")
    for key in ("train_clip_loss", "validation_clip_loss"):
        _require_finite_float(value[key], context=key, minimum=0.0)
    for key in ("validation_recording_macro_f1", "validation_recording_accuracy"):
        _require_finite_float(value[key], context=key, minimum=0.0, maximum=1.0)


def _benchmark_lock_value(
    *,
    result_record: Mapping[str, Any],
    benchmark_identity_sha256: str,
    source_fingerprint_sha256: str,
    execution_identity: Task1ExecutionIdentity,
    scope: str,
    production_evidence: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "benchmark_identity_sha256": benchmark_identity_sha256,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "scope": scope,
        "production_evidence": production_evidence,
        "result": dict(result_record),
    }


def _recover_or_verify_benchmark_evidence(
    artifact_paths: tuple[Path, Path] | None,
    *,
    validation_arguments: Mapping[str, Any],
) -> dict[str, Any] | None:
    if artifact_paths is None:
        return None
    result_path, lock_path = artifact_paths
    result_exists = _artifact_path_exists(result_path)
    lock_exists = _artifact_path_exists(lock_path)
    if not result_exists and not lock_exists:
        return None
    if not result_exists:
        raise ValueError("Task 1 benchmark lock exists without its result")
    result, result_record = _read_json_snapshot(result_path)
    _validate_benchmark_result(result, **validation_arguments)
    expected_lock = _benchmark_lock_value(
        result_record=result_record,
        benchmark_identity_sha256=validation_arguments["benchmark_identity_sha256"],
        source_fingerprint_sha256=validation_arguments["source_fingerprint_sha256"],
        execution_identity=validation_arguments["execution_identity"],
        scope=validation_arguments["scope"],
        production_evidence=validation_arguments["production_evidence"],
    )
    if lock_exists:
        observed_lock, lock_record = _read_json_snapshot(lock_path)
        if not _json_values_exact(expected_lock, observed_lock):
            raise ValueError("Task 1 benchmark completion lock is invalid")
    else:
        lock_record = _write_json_create_only(lock_path, expected_lock)
    return {
        **result,
        "result_artifact": result_record,
        "completion_lock_artifact": lock_record,
    }


def benchmark_task1_full_epoch(
    *,
    seed: int = 13,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
    command: Sequence[str] = (),
    evidence_output: str | Path | None = None,
    train_data: Task1Data | None = None,
    validation_data: Task1Data | None = None,
    test_injection: Task1TestInjection | None = None,
) -> dict[str, Any]:
    started_at_utc = datetime.now(UTC).isoformat()
    config = load_final_task1_config()
    if seed not in config["seeds"]:
        raise ValueError("Task 1 benchmark seed is outside the locked seed set")
    device = _resolve_runtime(test_injection)
    scope = ISOLATED_TEST_SCOPE if test_injection is not None else PRODUCTION_SCOPE
    production_evidence = scope == PRODUCTION_SCOPE
    artifact_paths = _benchmark_artifact_paths(test_injection, evidence_output)
    execution_identity = _capture_execution_identity(device)
    release_source_fingerprint_sha256 = source_fingerprint()
    weight = (
        preflight_efficientnet_weights(populate=False)
        if test_injection is None
        else test_injection.weight_artifact
    )
    with ExitStack() as stack:
        if (train_data is None) != (validation_data is None):
            raise ValueError("Benchmark train and validation data must be supplied together")
        if train_data is None or validation_data is None:
            if test_injection is not None:
                raise ValueError("CPU benchmark injection requires explicit fixtures")
            train, validation = _open_real_data(
                stack,
                cache_root=cache_root,
                ffmpeg=ffmpeg,
                expected_lock_sha256=expected_lock_sha256,
            )
        else:
            if test_injection is None:
                raise PermissionError("Benchmark data injection is allowed only for tests")
            train, validation = train_data, validation_data
        _validate_development_data(train, validation, config)
        _require_execution_identity_unchanged(execution_identity, device)
        config_sha256 = config_fingerprint(config)
        _, config_file_sha256, _ = _descriptor_snapshot(FINAL_CONFIG_PATH)
        maximum_epochs, batch_size, _ = _resolved_limits(config, test_injection)
        benchmark_identity = {
            "schema_version": RUN_SCHEMA_VERSION,
            "task": "task1_full_epoch_benchmark",
            "seed": seed,
            "config_file_sha256": config_file_sha256,
            "config_sha256": config_sha256,
            "cache_lock_sha256": train.lock_sha256,
            "weight_sha256": weight.sha256,
            "source_fingerprint_sha256": release_source_fingerprint_sha256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "scope": scope,
            "production_evidence": production_evidence,
        }
        benchmark_identity_sha256 = sha256_json(benchmark_identity)
        benchmark_validation_arguments = {
            "benchmark_identity": benchmark_identity,
            "benchmark_identity_sha256": benchmark_identity_sha256,
            "config_file_sha256": config_file_sha256,
            "config_sha256": config_sha256,
            "source_fingerprint_sha256": release_source_fingerprint_sha256,
            "execution_identity": execution_identity,
            "weight": weight,
            "train": train,
            "validation": validation,
            "seed": seed,
            "device": device,
            "batch_size": batch_size,
            "maximum_epochs": maximum_epochs,
            "stability_seeds": tuple(config["seeds"]),
            "scope": scope,
            "production_evidence": production_evidence,
        }
        existing_evidence = _recover_or_verify_benchmark_evidence(
            artifact_paths,
            validation_arguments=benchmark_validation_arguments,
        )
        if existing_evidence is not None:
            return existing_evidence

        seed_task1(seed, device)
        warmup_model = _build_model(config, device, weight, test_injection)
        warmup_optimizer = build_task1_optimizer(warmup_model, config)
        _warmup_task1(
            warmup_model,
            warmup_optimizer,
            train,
            seed=seed,
            batch_size=batch_size,
            class_count=int(config["class_count"]),
            device=device,
        )
        del warmup_optimizer, warmup_model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        seed_task1(seed, device)
        model = _build_model(config, device, weight, test_injection)
        optimizer = build_task1_optimizer(model, config)
        counts = parameter_counts(model)
        _synchronize(device)
        train_started = time.perf_counter()
        train_metrics = train_task1_epoch(
            model,
            optimizer,
            train,
            seed=seed,
            epoch_index=0,
            batch_size=batch_size,
            class_count=int(config["class_count"]),
            device=device,
        )
        _synchronize(device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation_metrics = validate_task1(
            model,
            validation,
            batch_size=batch_size,
            class_count=int(config["class_count"]),
            device=device,
        )
        _synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        total_seconds = train_seconds + validation_seconds
        _require_execution_identity_unchanged(execution_identity, device)
        _synchronize(device)
        checkpoint_copy_started = time.perf_counter()
        representative_model_state = _cpu_copy(model.state_dict())
        representative_optimizer_state = _cpu_copy(optimizer.state_dict())
        checkpoint_cpu_copy_seconds = time.perf_counter() - checkpoint_copy_started
        checkpoint_serialization_started = time.perf_counter()
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "model": representative_model_state,
                "optimizer": representative_optimizer_state,
            },
            checkpoint_buffer,
        )
        checkpoint_serialization_seconds = time.perf_counter() - checkpoint_serialization_started
        representative_checkpoint_bytes = checkpoint_buffer.tell()
        del checkpoint_buffer, representative_model_state, representative_optimizer_state
        checkpoint_overhead_per_epoch = (
            checkpoint_cpu_copy_seconds + 2.0 * checkpoint_serialization_seconds
        )
        estimated_epoch_with_checkpoint_seconds = total_seconds + checkpoint_overhead_per_epoch
        one_seed_compute_seconds = estimated_epoch_with_checkpoint_seconds * maximum_epochs
        all_seed_compute_seconds = one_seed_compute_seconds * len(config["seeds"])
        _require_execution_identity_unchanged(execution_identity, device)
        _require_weight_unchanged(weight)
        result = {
            "schema_version": RUN_SCHEMA_VERSION,
            "benchmark_only": True,
            "persistent_model_selection": False,
            "persistent_model_checkpoint": False,
            "durable_evidence": artifact_paths is not None,
            "scope": scope,
            "production_evidence": production_evidence,
            "started_at_utc": started_at_utc,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "command": [str(part) for part in command],
            "benchmark_identity_sha256": benchmark_identity_sha256,
            "benchmark_identity": benchmark_identity,
            "seed": seed,
            "device": str(device),
            "batch_size": batch_size,
            "maximum_epochs": maximum_epochs,
            "stability_seeds": list(config["seeds"]),
            "config_path": FINAL_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "config_file_sha256": config_file_sha256,
            "config_sha256": config_sha256,
            "cache_root": str(train.root),
            "cache_lock_sha256": train.lock_sha256,
            "weight_identifier": weight.identifier,
            "weight_sha256": weight.sha256,
            "weight_size_bytes": weight.size_bytes,
            "source_fingerprint_sha256": release_source_fingerprint_sha256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_path": REQUIREMENTS_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "numerical_runtime": execution_identity.numerical_runtime,
            "parameter_counts": counts,
            "warmup_completed": True,
            "train_recordings": train.recording_count,
            "train_clips": len(train),
            "train_batches": train_metrics["batches"],
            "validation_clips": len(validation),
            "validation_recordings": validation_metrics.recording_count,
            "validation_batches": math.ceil(len(validation) / batch_size),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "full_epoch_seconds": total_seconds,
            "checkpoint_cpu_copy_seconds": checkpoint_cpu_copy_seconds,
            "checkpoint_serialization_seconds": checkpoint_serialization_seconds,
            "representative_checkpoint_bytes": representative_checkpoint_bytes,
            "checkpoint_serializations_per_epoch_estimate": 2,
            "checkpoint_overhead_per_epoch_estimate_seconds": checkpoint_overhead_per_epoch,
            "estimated_epoch_with_checkpoint_seconds": estimated_epoch_with_checkpoint_seconds,
            "train_clips_per_second": len(train) / train_seconds,
            "validation_clips_per_second": len(validation) / validation_seconds,
            "estimated_one_seed_pre_allowance_seconds": one_seed_compute_seconds,
            "estimated_all_seed_pre_allowance_seconds": all_seed_compute_seconds,
            "conservative_wall_time_factor": CONSERVATIVE_WALL_TIME_FACTOR,
            "conservative_wall_time_scope": (
                "Measured epoch compute and checkpoint CPU copy with two in-memory "
                "serializations, plus a 25 percent allowance for artifact writes, startup, "
                "and thermal variation"
            ),
            "estimated_one_seed_conservative_wall_seconds": (
                one_seed_compute_seconds * CONSERVATIVE_WALL_TIME_FACTOR
            ),
            "estimated_all_seed_conservative_wall_seconds": (
                all_seed_compute_seconds * CONSERVATIVE_WALL_TIME_FACTOR
            ),
            "train_clip_loss": train_metrics["clip_loss"],
            "validation_clip_loss": validation_metrics.clip_loss,
            "validation_recording_macro_f1": validation_metrics.macro_f1,
            "validation_recording_accuracy": validation_metrics.accuracy,
            "evidence_scope": (
                "Runtime planning evidence only; no checkpoint is retained and no model "
                "selection decision is made"
            ),
        }
        if artifact_paths is None:
            return result
        result_path, lock_path = artifact_paths
        _validate_benchmark_result(result, **benchmark_validation_arguments)
        result_record = _write_json_create_only(result_path, result)
        lock_value = _benchmark_lock_value(
            result_record=result_record,
            benchmark_identity_sha256=benchmark_identity_sha256,
            source_fingerprint_sha256=release_source_fingerprint_sha256,
            execution_identity=execution_identity,
            scope=scope,
            production_evidence=production_evidence,
        )
        lock_record = _write_json_create_only(
            lock_path,
            lock_value,
        )
        return {
            **result,
            "result_artifact": result_record,
            "completion_lock_artifact": lock_record,
        }
