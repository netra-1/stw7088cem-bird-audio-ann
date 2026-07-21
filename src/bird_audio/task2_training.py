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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from bird_audio.config import config_fingerprint, load_toml, public_config
from bird_audio.hashing import sha256_json
from bird_audio.models import ConvolutionalAutoencoder, parameter_counts
from bird_audio.paths import PROJECT_ROOT, is_relative_to, require_safe_output, resolve_project_path
from bird_audio.provenance import PROVENANCE_V2_ROOT, source_fingerprint
from bird_audio.run_identity import make_run_id
from bird_audio.task2_scoring import (
    KNOWN_TRAINING_ROLE,
    KNOWN_VALIDATION_ROLE,
    ClipIdentity,
    RecordingBatch,
    aggregate_recordings,
    clip_reconstruction_mse,
    fit_known_training_reference,
    fit_known_validation_latent_threshold,
    fit_known_validation_threshold,
    latent_knn_novelty_scores,
)
from bird_audio.training_batching import (
    RecordingBalancedEpochSampler,
    collate_native_samples,
    to_autoencoder_batch,
)
from bird_audio.training_data import DevelopmentTrainingData, open_development_training_data

LOCKED_CONFIG_PATH = PROJECT_ROOT / "configs" / "task2" / "autoencoder.toml"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "known_clips_v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "task2_v2"
DEFAULT_BENCHMARK_RESULT_PATH = PROVENANCE_V2_ROOT / "task2_benchmark_v2.json"
DEFAULT_BENCHMARK_LOCK_PATH = PROVENANCE_V2_ROOT / "task2_benchmark_v2.lock.json"
REQUIREMENTS_LOCK_PATH = PROJECT_ROOT / "requirements.lock"
KNOWN_CACHE_LOCK_SHA256 = "d2efbe39c56edc3044deda9692dddf9df02ecf07f0b65d4c9cb3eaa43aa52886"
EXPECTED_PARAMETER_COUNT = 3_581_345
CHECKPOINT_SCHEMA_VERSION = "1.1"
RUN_SCHEMA_VERSION = "1.1"
DEVELOPMENT_BUNDLE_SCHEMA_VERSION = "1.1"
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
TASK2_IMPLEMENTATION_FILES = (
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
    "src/bird_audio/task2_scoring.py",
    "src/bird_audio/task2_training.py",
    "src/bird_audio/training_batching.py",
    "src/bird_audio/training_data.py",
)
_SAFE_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CPU_DEVICE = torch.device("cpu")


class Task2Data(Protocol):
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
class Task2TestInjection:
    """Explicit CPU-only dependencies and reduced limits for isolated tests."""

    model_factory: Callable[[Mapping[str, Any]], nn.Module]
    device: torch.device = _CPU_DEVICE
    maximum_epochs: int | None = None
    batch_size: int | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if not callable(self.model_factory):
            raise TypeError("Task2TestInjection model factory must be callable")
        if not isinstance(self.device, torch.device) or self.device.type != "cpu":
            raise ValueError("Task2TestInjection permits only an explicit CPU device")
        for name, value in (
            ("maximum_epochs", self.maximum_epochs),
            ("batch_size", self.batch_size),
            ("early_stopping_patience", self.early_stopping_patience),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"Test injection {name} must be a positive integer")


@dataclass(frozen=True)
class Task2CheckpointScore:
    validation_loss: float
    epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.validation_loss, float) or not math.isfinite(self.validation_loss):
            raise ValueError("Task 2 checkpoint loss must be a finite float")
        if self.validation_loss < 0.0:
            raise ValueError("Task 2 checkpoint loss cannot be negative")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch <= 0:
            raise ValueError("Task 2 checkpoint epoch must be positive")


@dataclass(frozen=True)
class Task2ValidationResult:
    loss: float
    clip_count: int
    pixel_count: int


@dataclass(frozen=True)
class Task2ScoredSplit:
    role: str
    recordings: RecordingBatch
    session_groups: dict[str, str]
    clip_identities: tuple[ClipIdentity, ...]
    clip_mse: np.ndarray
    clip_latent: np.ndarray


@dataclass(frozen=True)
class Task2ExecutionIdentity:
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
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError("Task 2 execution identity contains an invalid SHA-256")
        if not isinstance(self.numerical_runtime, dict) or not self.numerical_runtime:
            raise ValueError("Task 2 numerical runtime identity is invalid")
        if sha256_json(self.numerical_runtime) != self.numerical_runtime_sha256:
            raise ValueError("Task 2 numerical runtime identity hash does not match")


class Task2EarlyStopping:
    def __init__(self, patience: int) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("Task 2 early-stopping patience must be positive")
        self.patience = patience
        self.best: Task2CheckpointScore | None = None
        self.epochs_without_improvement = 0

    def update(self, score: Task2CheckpointScore) -> tuple[bool, bool]:
        improved = is_better_task2_checkpoint(score, self.best)
        if improved:
            self.best = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved, self.epochs_without_improvement >= self.patience


def _assert_locked_config(config: Mapping[str, Any]) -> None:
    training = config["training"]
    novelty = config["novelty"]
    required = {
        "task": config.get("task") == "novelty_detection",
        "architecture": config.get("architecture")
        == "skip_free_undercomplete_convolutional_autoencoder",
        "input": [
            config.get("input_channels"),
            config.get("input_height"),
            config.get("input_width"),
        ]
        == [1, 224, 224],
        "encoder": config.get("encoder_channels") == [16, 32, 64, 128],
        "convolution": [config.get("kernel_size"), config.get("stride"), config.get("padding")]
        == [4, 2, 1],
        "latent": config.get("latent_dimensions") == 64,
        "activation": config.get("hidden_activation") == "relu"
        and config.get("normalization") == "none"
        and config.get("bottleneck_activation") == "linear"
        and config.get("decoder_output_activation") == "sigmoid",
        "decoder": config.get("transpose_convolution_output_padding") == 0,
        "loss": config.get("loss") == "mean_squared_error"
        and config.get("loss_reduction") == "mean_over_all_pixels",
        "selection": config.get("clip_selection_strategy") == "energy",
        "seeds": config.get("seeds") == [13, 37, 71],
        "optimizer": training.get("optimizer") == "adamw",
        "learning_rate": training.get("learning_rate") == 0.001,
        "weight_decay": training.get("weight_decay") == 0.00001,
        "batch": training.get("batch_size") == 64,
        "epochs": training.get("maximum_epochs") == 100,
        "patience": training.get("early_stopping_patience") == 10,
        "workers": training.get("pin_memory") is False and training.get("num_workers") == 0,
        "precision": training.get("mixed_precision") is False
        and training.get("dtype") == "float32",
        "device": training.get("device_preference") == "mps"
        and training.get("allow_mps_fallback") is False,
        "determinism": training.get("request_deterministic_algorithms") is True
        and training.get("determinism_failure_policy") == "fail_and_log",
        "seed_flags": all(
            training.get(name) is True
            for name in ("seed_python", "seed_numpy", "seed_torch", "seed_sampler")
        ),
        "scheduler": training.get("scheduler") == "none",
        "checkpoint": training.get("checkpoint_metric") == "known_validation_reconstruction_mse"
        and training.get("checkpoint_mode") == "min",
        "novelty": novelty.get("primary_score") == "median_clip_reconstruction_mse"
        and novelty.get("secondary_readout") == "recording_mean_latent_knn_distance"
        and novelty.get("latent_reference_unit")
        == "one_mean_embedding_per_known_training_recording"
        and novelty.get("latent_standardization_unit") == "known_training_recording_embeddings"
        and novelty.get("nearest_neighbours") == 10
        and novelty.get("threshold_quantile") == 0.95
        and novelty.get("threshold_quantile_method") == "higher"
        and novelty.get("threshold_scope") == "per_seed_known_validation"
        and novelty.get("score_direction") == "higher_is_more_novel"
        and novelty.get("threshold_operator") == ">"
        and novelty.get("bootstrap_seed") == 20260713
        and novelty.get("bootstrap_replicates") == 2000
        and novelty.get("bootstrap_interval_method") == "percentile"
        and novelty.get("bootstrap_confidence_level") == 0.95
        and novelty.get("bootstrap_resampling_unit") == "session_cluster"
        and novelty.get("detailed_figure_seed") == 37,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise ValueError(f"Locked Task 2 configuration changed: {failed}")


def load_locked_task2_config(path: str | Path = LOCKED_CONFIG_PATH) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if resolved != LOCKED_CONFIG_PATH.resolve():
        raise PermissionError("Task 2 accepts only configs/task2/autoencoder.toml")
    config = load_toml(resolved)
    _assert_locked_config(config)
    return config


def _require_project_venv() -> None:
    expected = (PROJECT_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(f"Task 2 must run inside the project virtualenv: {expected}")


def _resolve_runtime(test_injection: Task2TestInjection | None) -> torch.device:
    _require_project_venv()
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().casefold()
    if fallback not in {"", "0", "false"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    if test_injection is not None:
        return test_injection.device
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Task 2 production training requires available Apple MPS")
    return torch.device("mps")


def seed_task2(seed: int, device: torch.device) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {13, 37, 71}:
        raise ValueError("Task 2 seed must be one of 13, 37, or 71")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def _build_task2_model(
    config: Mapping[str, Any],
    device: torch.device,
    test_injection: Task2TestInjection | None,
) -> nn.Module:
    if test_injection is None:
        model = ConvolutionalAutoencoder(latent_dimensions=int(config["latent_dimensions"]))
        if type(model) is not ConvolutionalAutoencoder:
            raise TypeError("Task 2 production model type changed")
    else:
        model = test_injection.model_factory(config)
    model = model.to(device=device, dtype=torch.float32)
    counts = parameter_counts(model)
    if test_injection is None and counts != {
        "total": EXPECTED_PARAMETER_COUNT,
        "trainable": EXPECTED_PARAMETER_COUNT,
    }:
        raise RuntimeError("Task 2 production parameter count changed")
    return model


def build_task2_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Task 2 optimizer requires trainable parameters")
    if {id(parameter) for parameter in parameters} != {
        id(parameter) for parameter in model.parameters()
    }:
        raise ValueError("Every Task 2 model parameter must be trainable")
    training = config["training"]
    return torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
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


def is_better_task2_checkpoint(
    candidate: Task2CheckpointScore,
    incumbent: Task2CheckpointScore | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.validation_loss != incumbent.validation_loss:
        return candidate.validation_loss < incumbent.validation_loss
    return candidate.epoch < incumbent.epoch


def _batch_positions(length: int, batch_size: int) -> Iterator[range]:
    for start in range(0, length, batch_size):
        yield range(start, min(start + batch_size, length))


def _model_outputs(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    latent_dimensions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(inputs)
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise RuntimeError("Task 2 model must return reconstruction and latent tensors")
    reconstruction, latent = outputs
    if (
        reconstruction.dtype != torch.float32
        or reconstruction.shape != inputs.shape
        or latent.dtype != torch.float32
        or latent.shape != (inputs.shape[0], latent_dimensions)
    ):
        raise RuntimeError("Task 2 model output contract is invalid")
    return reconstruction, latent


def train_task2_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Task2Data,
    *,
    seed: int,
    epoch_index: int,
    batch_size: int,
    latent_dimensions: int,
    device: torch.device,
) -> dict[str, Any]:
    if data.split != "train" or data.strategy != "energy":
        raise PermissionError("Task 2 training accepts only energy-selected training data")
    sampler = RecordingBalancedEpochSampler(data, base_seed=seed)
    sampler.set_epoch(epoch_index)
    sampled_indices = list(sampler)
    if len(sampled_indices) != len(data):
        raise RuntimeError("Task 2 sampler draw count differs from the training clip count")
    model.train()
    loss_sum = 0.0
    completed = 0
    for positions in _batch_positions(len(sampled_indices), batch_size):
        indices = [sampled_indices[position] for position in positions]
        native = collate_native_samples([data[index] for index in indices])
        inputs = to_autoencoder_batch(native.tensor).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, _ = _model_outputs(
            model,
            inputs,
            latent_dimensions=latent_dimensions,
        )
        loss = functional.mse_loss(reconstruction, inputs, reduction="mean")
        loss.backward()
        finite_parts = [torch.isfinite(loss.detach())]
        finite_parts.extend(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if len(finite_parts) <= 1:
            raise RuntimeError("Task 2 training produced no gradients")
        finite_reduction = torch.stack(finite_parts).all()
        batch_status = torch.stack((loss.detach(), finite_reduction.to(loss.dtype))).to("cpu")
        if not bool(batch_status[1]):
            raise RuntimeError("Task 2 loss or gradient is non-finite")
        optimizer.step()
        loss_sum += float(batch_status[0]) * len(indices)
        completed += len(indices)
    if completed != len(data):
        raise RuntimeError("Task 2 training epoch is incomplete")
    return {
        "loss": loss_sum / completed,
        "clips": completed,
        "batches": math.ceil(completed / batch_size),
        "sampler_seed": sampler.generator_seed,
        "augmentation": "none",
    }


def validate_task2(
    model: nn.Module,
    data: Task2Data,
    *,
    batch_size: int,
    latent_dimensions: int,
    device: torch.device,
) -> Task2ValidationResult:
    if data.split != "validation" or data.strategy != "energy":
        raise PermissionError("Task 2 validation accepts only energy-selected validation data")
    if len(data) <= 0:
        raise ValueError("Task 2 validation data cannot be empty")
    model.eval()
    squared_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    pixel_count = 0
    with torch.no_grad():
        for positions in _batch_positions(len(data), batch_size):
            native = collate_native_samples([data[index] for index in positions])
            inputs = to_autoencoder_batch(native.tensor).to(device=device, dtype=torch.float32)
            reconstruction, _ = _model_outputs(
                model,
                inputs,
                latent_dimensions=latent_dimensions,
            )
            squared_error_sum = squared_error_sum + torch.sum(torch.square(reconstruction - inputs))
            pixel_count += reconstruction.numel()
    status = torch.stack(
        (squared_error_sum, torch.isfinite(squared_error_sum).to(torch.float32))
    ).to("cpu")
    if not bool(status[1]) or pixel_count != len(data) * 224 * 224:
        raise RuntimeError("Task 2 validation pixel aggregation is invalid")
    loss = float(status[0]) / pixel_count
    if not math.isfinite(loss) or loss < 0.0:
        raise RuntimeError("Task 2 validation loss is invalid")
    return Task2ValidationResult(loss=loss, clip_count=len(data), pixel_count=pixel_count)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_regular_readonly(path: Path) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("Descriptor-bound artifact reads require O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
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


def _write_json_create_only(path: Path, value: Any) -> dict[str, Any]:
    return _atomic_create_only_bytes(path, _json_bytes(value))


def _read_json_snapshot(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    resolved = _resolve_project_input_no_follow(path)
    payload, observed_sha256, size_bytes = _descriptor_snapshot(resolved)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
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


def _write_or_verify_json(path: Path, value: Any) -> dict[str, Any]:
    try:
        return _write_json_create_only(path, value)
    except FileExistsError:
        observed, record = _read_json_snapshot(path)
        if observed != value:
            raise ValueError(f"Existing JSON artifact differs from resumed state: {path}") from None
        return record


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
    "config_file_sha256",
    "cache_lock_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "model_contract_sha256",
    "scope",
    "production_evidence",
    "seed",
}
_BEST_CHECKPOINT_FIELDS = _CHECKPOINT_COMMON_FIELDS | {
    "epoch",
    "score",
    "model",
    "optimizer",
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


def _score_state(score: Task2CheckpointScore) -> dict[str, Any]:
    return {"validation_loss": score.validation_loss, "epoch": score.epoch}


def _validate_score(value: Any) -> Task2CheckpointScore:
    if not isinstance(value, dict) or set(value) != {"validation_loss", "epoch"}:
        raise ValueError("Task 2 checkpoint score schema is invalid")
    if type(value["validation_loss"]) is not float or type(value["epoch"]) is not int:
        raise ValueError("Task 2 checkpoint score types are invalid")
    return Task2CheckpointScore(value["validation_loss"], value["epoch"])


def _validate_tensor_state(value: Any, name: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Task 2 checkpoint {name} state is invalid")
    if name == "model":
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("Task 2 checkpoint model keys are invalid")
        tensors = list(value.values())
        if (
            not tensors
            or any(not torch.is_tensor(item) or item.device.type != "cpu" for item in tensors)
            or any(
                item.is_floating_point() and not bool(torch.isfinite(item).all())
                for item in tensors
            )
        ):
            raise ValueError("Task 2 checkpoint model tensors are invalid")
        return
    if set(value) != {"state", "param_groups"}:
        raise ValueError("Task 2 checkpoint optimizer schema is invalid")
    state = value["state"]
    groups = value["param_groups"]
    if (
        not isinstance(state, dict)
        or not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
    ):
        raise ValueError("Task 2 checkpoint optimizer structure is invalid")
    group = groups[0]
    parameters = group.get("params")
    expected_group_options = {
        "lr": 0.001,
        "betas": ADAMW_BETAS,
        "eps": ADAMW_EPS,
        "weight_decay": 0.00001,
        "amsgrad": ADAMW_AMSGRAD,
        "maximize": ADAMW_MAXIMIZE,
        "foreach": ADAMW_FOREACH,
        "capturable": ADAMW_CAPTURABLE,
        "differentiable": ADAMW_DIFFERENTIABLE,
        "fused": ADAMW_FUSED,
        "decoupled_weight_decay": True,
    }
    observed_group_options = {key: item for key, item in group.items() if key != "params"}
    if (
        not isinstance(parameters, list)
        or not parameters
        or any(type(index) is not int or index < 0 for index in parameters)
        or parameters != list(range(len(parameters)))
        or set(group) != {"params", *expected_group_options}
        or observed_group_options != expected_group_options
        or any(type(index) is not int or index not in parameters for index in state)
        or (state and set(state) != set(parameters))
    ):
        raise ValueError("Task 2 checkpoint optimizer group is invalid")
    for parameter_index, parameter_state in state.items():
        if not isinstance(parameter_state, dict) or set(parameter_state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError(f"Task 2 optimizer state is invalid for parameter {parameter_index}")
        step = parameter_state["step"]
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        if (
            not all(torch.is_tensor(item) for item in (step, exp_avg, exp_avg_sq))
            or any(item.device.type != "cpu" for item in (step, exp_avg, exp_avg_sq))
            or step.dtype != torch.float32
            or step.shape != ()
            or exp_avg.dtype != torch.float32
            or exp_avg_sq.dtype != torch.float32
            or exp_avg.shape != exp_avg_sq.shape
            or not bool(torch.isfinite(step))
            or not bool(torch.isfinite(exp_avg).all())
            or not bool(torch.isfinite(exp_avg_sq).all())
            or float(step) < 1.0
        ):
            raise ValueError("Task 2 checkpoint optimizer tensors are invalid")


def _validate_history(history: Any, completed_epoch: int) -> None:
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise ValueError("Task 2 recovery history length is invalid")
    for expected_epoch, row in enumerate(history, start=1):
        if (
            not isinstance(row, dict)
            or set(row) != {"epoch", "train", "validation", "checkpoint_improved"}
            or row.get("epoch") != expected_epoch
            or type(row.get("checkpoint_improved")) is not bool
        ):
            raise ValueError("Task 2 recovery history schema is invalid")
        train = row["train"]
        validation = row["validation"]
        if (
            not isinstance(train, dict)
            or set(train)
            != {
                "loss",
                "clips",
                "batches",
                "sampler_seed",
                "augmentation",
            }
            or type(train["loss"]) is not float
            or not math.isfinite(train["loss"])
            or train["loss"] < 0.0
            or any(type(train[key]) is not int or train[key] <= 0 for key in ("clips", "batches"))
            or type(train["sampler_seed"]) is not int
            or train["sampler_seed"] < 0
            or train["augmentation"] != "none"
            or not isinstance(validation, dict)
            or set(validation) != {"loss", "clip_count", "pixel_count", "reduction"}
            or type(validation["loss"]) is not float
            or not math.isfinite(validation["loss"])
            or validation["loss"] < 0.0
            or type(validation["clip_count"]) is not int
            or validation["clip_count"] <= 0
            or type(validation["pixel_count"]) is not int
            or validation["pixel_count"] != validation["clip_count"] * 224 * 224
            or validation["reduction"] != "global_pixel_mean"
        ):
            raise ValueError("Task 2 recovery history values are invalid")
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Task 2 recovery history is not finite JSON") from exc


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
        raise ValueError("Task 2 recovery RNG state schema is invalid")
    try:
        probe = random.Random()
        probe.setstate(value["python"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Task 2 recovery Python RNG state is invalid") from exc
    numpy_state = value["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("Task 2 recovery NumPy RNG state schema is invalid")
    keys = numpy_state["keys"]
    if (
        numpy_state["bit_generator"] != "MT19937"
        or not torch.is_tensor(keys)
        or keys.device.type != "cpu"
        or keys.dtype != torch.int64
        or keys.shape != (624,)
        or bool(torch.any(keys < 0))
        or bool(torch.any(keys > 2**32 - 1))
        or type(numpy_state["position"]) is not int
        or not 0 <= numpy_state["position"] <= 624
        or type(numpy_state["has_gauss"]) is not int
        or numpy_state["has_gauss"] not in {0, 1}
        or type(numpy_state["cached_gaussian"]) is not float
        or not math.isfinite(numpy_state["cached_gaussian"])
    ):
        raise ValueError("Task 2 recovery NumPy RNG state values are invalid")
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
        raise ValueError("Task 2 recovery NumPy RNG state cannot be restored") from exc
    torch_cpu = value["torch_cpu"]
    torch_mps = value["torch_mps"]
    if (
        not torch.is_tensor(torch_cpu)
        or torch_cpu.device.type != "cpu"
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
    ):
        raise ValueError("Task 2 recovery CPU Torch RNG state is invalid")
    if value["device"] == "cpu" and torch_mps is not None:
        raise ValueError("Task 2 CPU recovery cannot contain MPS RNG state")
    if value["device"] == "mps" and (
        not torch.is_tensor(torch_mps)
        or torch_mps.device.type != "cpu"
        or torch_mps.dtype != torch.uint8
        or torch_mps.ndim != 1
        or torch_mps.numel() == 0
    ):
        raise ValueError("Task 2 recovery MPS RNG state is invalid")


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


def _validate_checkpoint_common(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Task 2 checkpoint version is unsupported")
    run_id = checkpoint.get("run_id")
    if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Task 2 checkpoint run ID is invalid")
    for key in (
        "run_identity_sha256",
        "config_sha256",
        "config_file_sha256",
        "cache_lock_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "model_contract_sha256",
    ):
        if not isinstance(checkpoint.get(key), str) or _SHA256.fullmatch(checkpoint[key]) is None:
            raise ValueError(f"Task 2 checkpoint {key} is invalid")
    seed = checkpoint.get("seed")
    if type(seed) is not int or seed not in {13, 37, 71}:
        raise ValueError("Task 2 checkpoint seed is invalid")
    scope = checkpoint.get("scope")
    production_evidence = checkpoint.get("production_evidence")
    if (
        scope not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or type(production_evidence) is not bool
        or production_evidence is not (scope == PRODUCTION_SCOPE)
    ):
        raise ValueError("Task 2 checkpoint evidence scope is invalid")


def _validate_checkpoint_state(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        raise ValueError("Task 2 checkpoint must be a dictionary")
    checkpoint_type = checkpoint.get("checkpoint_type")
    expected_fields = (
        _BEST_CHECKPOINT_FIELDS if checkpoint_type == "best" else _RECOVERY_CHECKPOINT_FIELDS
    )
    if checkpoint_type not in {"best", "recovery"} or set(checkpoint) != expected_fields:
        raise ValueError("Task 2 checkpoint schema is invalid")
    _validate_checkpoint_common(checkpoint)
    _validate_tensor_state(checkpoint["model"], "model")
    _validate_tensor_state(checkpoint["optimizer"], "optimizer")
    if checkpoint_type == "best":
        epoch = checkpoint["epoch"]
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("Task 2 best checkpoint epoch is invalid")
        if _validate_score(checkpoint["score"]).epoch != epoch:
            raise ValueError("Task 2 best checkpoint score epoch differs from its state")
        return checkpoint_type

    completed_epoch = checkpoint["completed_epoch"]
    next_epoch_index = checkpoint["next_epoch_index"]
    if (
        type(completed_epoch) is not int
        or completed_epoch <= 0
        or type(next_epoch_index) is not int
        or next_epoch_index != completed_epoch
        or type(checkpoint["stop_requested"]) is not bool
    ):
        raise ValueError("Task 2 recovery checkpoint epoch state is invalid")
    limits = checkpoint["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"maximum_epochs", "batch_size", "patience"}
        or any(type(value) is not int or value <= 0 for value in limits.values())
        or completed_epoch > limits["maximum_epochs"]
    ):
        raise ValueError("Task 2 recovery checkpoint limits are invalid")
    early = checkpoint["early_stopping"]
    if (
        not isinstance(early, dict)
        or set(early) != {"best", "epochs_without_improvement", "patience"}
        or early["patience"] != limits["patience"]
        or type(early["epochs_without_improvement"]) is not int
        or not 0 <= early["epochs_without_improvement"] <= completed_epoch
    ):
        raise ValueError("Task 2 recovery early-stopping state is invalid")
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
        raise ValueError("Task 2 recovery best-candidate binding is invalid")
    _validate_history(checkpoint["history"], completed_epoch)
    expected_best: Task2CheckpointScore | None = None
    expected_without_improvement = 0
    for row in checkpoint["history"]:
        score = Task2CheckpointScore(row["validation"]["loss"], row["epoch"])
        improved = is_better_task2_checkpoint(score, expected_best)
        if row["checkpoint_improved"] is not improved:
            raise ValueError("Task 2 recovery improvement history is inconsistent")
        if improved:
            expected_best = score
            expected_without_improvement = 0
        else:
            expected_without_improvement += 1
    if (
        expected_best != best_score
        or early["epochs_without_improvement"] != expected_without_improvement
        or checkpoint["stop_requested"] is not (expected_without_improvement >= early["patience"])
    ):
        raise ValueError("Task 2 recovery early-stopping history is inconsistent")
    _validate_rng_state(checkpoint["rng_state"], expected_device=None)
    return checkpoint_type


def save_task2_checkpoint_create_only(path: str | Path, state: Mapping[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError("Task 2 checkpoint SHA-256 changed after publication")
    verified = torch.load(io.BytesIO(verified_payload), map_location="cpu", weights_only=True)
    _validate_checkpoint_state(verified)
    _assert_round_trip(expected, verified)
    return record


def load_task2_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_run_identity_sha256: str | None = None,
    expected_type: str | None = None,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Expected Task 2 checkpoint SHA-256 is malformed")
    resolved = _resolve_project_input_no_follow(path)
    payload, observed_sha256, _ = _descriptor_snapshot(resolved)
    if observed_sha256 != expected_sha256:
        raise ValueError("Task 2 checkpoint SHA-256 does not match")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    checkpoint_type = _validate_checkpoint_state(checkpoint)
    if expected_type is not None and checkpoint_type != expected_type:
        raise ValueError("Task 2 checkpoint type does not match")
    if (
        expected_run_identity_sha256 is not None
        and checkpoint["run_identity_sha256"] != expected_run_identity_sha256
    ):
        raise ValueError("Task 2 checkpoint run identity does not match")
    return checkpoint


def _sysctl_value(name: str) -> str:
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


def _normalized_environment(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip().casefold()
    return value or default


def _task2_implementation_record() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative_path in TASK2_IMPLEMENTATION_FILES:
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


def _task2_implementation_fingerprint() -> str:
    return sha256_json(_task2_implementation_record())


def _requirements_lock_fingerprint() -> str:
    _, sha256, _ = _descriptor_snapshot(REQUIREMENTS_LOCK_PATH)
    return sha256


def _numerical_runtime_identity(device: torch.device) -> dict[str, Any]:
    numerical_environment = {
        name: os.environ.get(name, "").strip() or "unset"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "PYTORCH_MPS_FAST_MATH",
            "PYTORCH_MPS_PREFER_METAL",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        )
    }
    return {
        "schema_version": "1.0",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "macos_version": platform.mac_ver()[0] or "unavailable",
        "hardware_model": _sysctl_value("hw.model"),
        "processor_brand": _sysctl_value("machdep.cpu.brand_string"),
        "torch_version": str(torch.__version__),
        "torch_build_config": torch.__config__.show(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "numpy_version": np.__version__,
        "device": device.type,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "mps_fallback": _normalized_environment("PYTORCH_ENABLE_MPS_FALLBACK", "disabled"),
        "mps_fast_math": _normalized_environment("PYTORCH_MPS_FAST_MATH", "disabled"),
        "mps_prefer_metal": _normalized_environment("PYTORCH_MPS_PREFER_METAL", "default"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "training_dtype": "torch.float32",
        "numerical_environment": numerical_environment,
    }


def _capture_execution_identity(device: torch.device) -> Task2ExecutionIdentity:
    numerical_runtime = _numerical_runtime_identity(device)
    return Task2ExecutionIdentity(
        implementation_sha256=_task2_implementation_fingerprint(),
        requirements_lock_sha256=_requirements_lock_fingerprint(),
        numerical_runtime=numerical_runtime,
        numerical_runtime_sha256=sha256_json(numerical_runtime),
    )


def _require_execution_identity_unchanged(
    expected: Task2ExecutionIdentity,
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
        raise RuntimeError(f"Task 2 execution identity drifted: {mismatches}")


def _model_contract(model: nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    counts = parameter_counts(model)
    state = model.state_dict()
    if not state or any(
        not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("Task 2 model state contract is invalid")
    return {
        "architecture": str(config["architecture"]),
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "input_shape": [1, int(config["input_height"]), int(config["input_width"])],
        "latent_dimensions": int(config["latent_dimensions"]),
        "parameter_counts": counts,
        "state": [
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in state.items()
        ],
    }


def _optimizer_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "type": "torch.optim.AdamW",
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "betas": list(ADAMW_BETAS),
        "eps": ADAMW_EPS,
        "amsgrad": ADAMW_AMSGRAD,
        "maximize": ADAMW_MAXIMIZE,
        "foreach": ADAMW_FOREACH,
        "capturable": ADAMW_CAPTURABLE,
        "differentiable": ADAMW_DIFFERENTIABLE,
        "fused": ADAMW_FUSED,
        "decoupled_weight_decay": True,
    }


def _final_evaluation_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    novelty = config["novelty"]
    return {
        "primary_score": novelty["primary_score"],
        "secondary_readout": novelty["secondary_readout"],
        "score_direction": novelty["score_direction"],
        "nearest_neighbours": novelty["nearest_neighbours"],
        "threshold_quantile": novelty["threshold_quantile"],
        "threshold_quantile_method": novelty["threshold_quantile_method"],
        "threshold_scope": novelty["threshold_scope"],
        "threshold_operator": novelty["threshold_operator"],
        "bootstrap_seed": novelty["bootstrap_seed"],
        "bootstrap_replicates": novelty["bootstrap_replicates"],
        "bootstrap_interval_method": novelty["bootstrap_interval_method"],
        "bootstrap_confidence_level": novelty["bootstrap_confidence_level"],
        "bootstrap_resampling_unit": novelty["bootstrap_resampling_unit"],
        "detailed_figure_seed": novelty["detailed_figure_seed"],
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _resolved_limits(
    config: Mapping[str, Any], injection: Task2TestInjection | None
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
    data: Task2Data,
    *,
    expected_split: str,
) -> tuple[set[str], set[str], set[str]]:
    if len(data) <= 0 or data.recording_count <= 0:
        raise ValueError(f"Task 2 {expected_split} data cannot be empty")
    rows = tuple(data.iter_metadata())
    if len(rows) != len(data):
        raise ValueError(f"Task 2 {expected_split} metadata count is inconsistent")
    recordings: dict[str, tuple[str, str, str, int]] = {}
    observed_clip_counts: dict[str, int] = {}
    session_groups: set[str] = set()
    clip_ids: set[str] = set()
    for row in rows:
        if row.get("split") != expected_split:
            raise PermissionError(f"Task 2 {expected_split} data contains another split")
        if row.get("selection_strategy") != "energy":
            raise ValueError(f"Task 2 {expected_split} data contains another strategy")
        recording_id = str(row.get("recording_id") or "")
        clip_id = str(row.get("clip_id") or "")
        session_group = str(row.get("session_group") or "")
        species = str(row.get("species_common_name") or "")
        class_index = str(row.get("class_index") or "")
        try:
            parsed_class_index = int(class_index)
            declared_clip_count = int(row["strategy_clip_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Task 2 {expected_split} metadata identity is invalid") from exc
        if (
            not recording_id
            or not clip_id
            or not session_group
            or not species
            or not 0 <= parsed_class_index < 15
            or not 1 <= declared_clip_count <= 5
            or clip_id in clip_ids
        ):
            raise ValueError(f"Task 2 {expected_split} metadata identity is invalid")
        identity = (session_group, species, class_index, declared_clip_count)
        existing = recordings.setdefault(recording_id, identity)
        if existing != identity:
            raise ValueError(f"Task 2 recording identity changes: {recording_id}")
        observed_clip_counts[recording_id] = observed_clip_counts.get(recording_id, 0) + 1
        session_groups.add(session_group)
        clip_ids.add(clip_id)
    if len(recordings) != data.recording_count:
        raise ValueError(f"Task 2 {expected_split} recording count is inconsistent")
    for recording_id, identity in recordings.items():
        if observed_clip_counts[recording_id] != identity[3]:
            raise ValueError(f"Task 2 selected clip count is inconsistent: {recording_id}")
    return session_groups, set(recordings), clip_ids


def _validate_development_data(
    train: Task2Data,
    validation: Task2Data,
    *,
    production: bool,
) -> None:
    if train.split != "train" or validation.split != "validation":
        raise PermissionError("Task 2 development engine cannot access the final test split")
    if train.strategy != validation.strategy or train.strategy != "energy":
        raise ValueError("Task 2 development data must use the locked energy strategy")
    if train.lock_sha256 != validation.lock_sha256:
        raise ValueError("Task 2 train and validation caches have different locks")
    if _SHA256.fullmatch(train.lock_sha256) is None:
        raise ValueError("Task 2 development cache lock SHA-256 is invalid")
    if production and train.lock_sha256 != KNOWN_CACHE_LOCK_SHA256:
        raise ValueError("Production Task 2 cache lock differs from the published lock")
    train_sessions, train_recordings, train_clips = _validated_session_groups(
        train,
        expected_split="train",
    )
    validation_sessions, validation_recordings, validation_clips = _validated_session_groups(
        validation,
        expected_split="validation",
    )
    if train_sessions.intersection(validation_sessions):
        raise ValueError("Task 2 train and validation session groups overlap")
    if train_recordings.intersection(validation_recordings):
        raise ValueError("Task 2 train and validation recording IDs overlap")
    if train_clips.intersection(validation_clips):
        raise ValueError("Task 2 train and validation clip IDs overlap")
    if len(train_recordings) < 10:
        raise ValueError("Task 2 requires at least 10 known-training recordings")
    if production and (
        len(train) != 5_319
        or train.recording_count != 1_254
        or len(validation) != 1_138
        or validation.recording_count != 271
    ):
        raise ValueError("Production Task 2 development cache counts changed")


def _open_real_data(
    stack: ExitStack,
    *,
    cache_root: str | Path,
    ffmpeg: str | Path | None,
    expected_lock_sha256: str | None,
) -> tuple[DevelopmentTrainingData, DevelopmentTrainingData]:
    resolved_root = resolve_project_path(cache_root)
    if resolved_root != DEFAULT_CACHE_ROOT.resolve():
        raise PermissionError("Production Task 2 accepts only the canonical known cache root")
    if expected_lock_sha256 not in {None, KNOWN_CACHE_LOCK_SHA256}:
        raise ValueError("Production Task 2 cache SHA-256 differs from the published lock")
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
        raise ValueError("Production Task 2 opened an unexpected known-cache lock")
    return train, validation


def _validated_run_output_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    lexical = Path(os.path.abspath(requested))
    canonical_root = Path(os.path.abspath(DEFAULT_RUN_ROOT))
    if DEFAULT_RUN_ROOT.resolve() != canonical_root:
        raise ValueError("Task 2 canonical v2 run root traverses a symbolic link")
    resolved = require_safe_output(lexical)
    if resolved != lexical:
        raise ValueError("Task 2 run output path traverses a symbolic link")
    try:
        resolved.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError("Task 2 run output must stay inside runs/task2_v2") from exc
    return resolved


def _run_directory(output_root: str | Path, run_id: str) -> Path:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Task 2 run ID is unsafe")
    root = _validated_run_output_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    destination.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(root)
    return destination


def _ensure_run_subdirectories(run_directory: Path) -> None:
    for name in ("best_candidates", "recovery", "failures", "development"):
        destination = run_directory / name
        destination.mkdir(mode=0o700, exist_ok=True)
        _fsync_directory(destination)
    _fsync_directory(run_directory)


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
        raise PermissionError("Task 2 resume checkpoint is outside its selected run directory")
    return run_directory, checkpoint


def _checkpoint_common_state(
    *,
    run_id: str,
    run_identity_sha256: str,
    config_sha256: str,
    config_file_sha256: str,
    cache_lock_sha256: str,
    execution_identity: Task2ExecutionIdentity,
    model_contract_sha256: str,
    scope: str,
    production_evidence: bool,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "config_sha256": config_sha256,
        "config_file_sha256": config_file_sha256,
        "cache_lock_sha256": cache_lock_sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "model_contract_sha256": model_contract_sha256,
        "scope": scope,
        "production_evidence": production_evidence,
        "seed": seed,
    }


def _save_or_verify_checkpoint(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return save_task2_checkpoint_create_only(path, state)
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
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("Task 2 recovery optimizer type or group count is invalid")
    training = config["training"]
    group = optimizer.param_groups[0]
    expected_options = {
        "lr": float(training["learning_rate"]),
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
    observed_options = {key: value for key, value in group.items() if key != "params"}
    observed_parameters = tuple(group.get("params", ()))
    expected_parameters = tuple(model.parameters())
    if (
        set(group) != {"params", *expected_options}
        or observed_options != expected_options
        or tuple(id(parameter) for parameter in observed_parameters)
        != tuple(id(parameter) for parameter in expected_parameters)
        or set(optimizer.state) != set(expected_parameters)
    ):
        raise ValueError("Task 2 recovery optimizer settings differ from the locked method")
    for parameter in expected_parameters:
        state = optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Task 2 recovery optimizer state fields differ from AdamW")
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if (
            not all(torch.is_tensor(item) for item in (step, exp_avg, exp_avg_sq))
            or step.shape != ()
            or step.dtype != torch.float32
            or exp_avg.shape != parameter.shape
            or exp_avg_sq.shape != parameter.shape
            or exp_avg.dtype != parameter.dtype
            or exp_avg_sq.dtype != parameter.dtype
            or exp_avg.device != parameter.device
            or exp_avg_sq.device != parameter.device
            or not bool(torch.isfinite(step).all())
            or not bool(torch.isfinite(exp_avg).all())
            or not bool(torch.isfinite(exp_avg_sq).all())
            or float(step.detach().to("cpu")) < 1.0
        ):
            raise ValueError("Task 2 recovery optimizer tensor state is invalid")


def _score_development_split(
    model: nn.Module,
    data: Task2Data,
    *,
    source_role: str,
    batch_size: int,
    latent_dimensions: int,
    device: torch.device,
) -> Task2ScoredSplit:
    expected_split = "train" if source_role == KNOWN_TRAINING_ROLE else "validation"
    if source_role not in {KNOWN_TRAINING_ROLE, KNOWN_VALIDATION_ROLE}:
        raise ValueError("Task 2 development scoring role is invalid")
    if data.split != expected_split or data.strategy != "energy":
        raise PermissionError("Task 2 development scoring received the wrong data role")
    identities: list[ClipIdentity] = []
    session_groups: dict[str, str] = {}
    mse_parts: list[np.ndarray] = []
    latent_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for positions in _batch_positions(len(data), batch_size):
            samples = [data[index] for index in positions]
            native = collate_native_samples(samples)
            inputs_cpu = to_autoencoder_batch(native.tensor)
            inputs = inputs_cpu.to(device=device, dtype=torch.float32)
            reconstruction, latent = _model_outputs(
                model,
                inputs,
                latent_dimensions=latent_dimensions,
            )
            reconstruction_cpu = reconstruction.detach().to("cpu")
            latent_cpu = latent.detach().to("cpu")
            batch_mse = clip_reconstruction_mse(
                inputs_cpu.numpy(),
                reconstruction_cpu.numpy(),
            )
            batch_latent = latent_cpu.numpy().astype(np.float64, copy=True)
            for item in native.metadata:
                recording_id = str(item.get("recording_id") or "")
                clip_id = str(item.get("clip_id") or "")
                session_group = str(item.get("session_group") or "")
                identity = ClipIdentity(recording_id=recording_id, clip_id=clip_id)
                existing = session_groups.setdefault(recording_id, session_group)
                if not session_group or existing != session_group:
                    raise ValueError("Task 2 scoring session identity changed within recording")
                identities.append(identity)
            mse_parts.append(batch_mse.copy())
            latent_parts.append(batch_latent)
    clip_mse = np.concatenate(mse_parts).astype(np.float64, copy=False)
    clip_latent = np.concatenate(latent_parts).astype(np.float64, copy=False)
    clip_mse.setflags(write=False)
    clip_latent.setflags(write=False)
    recordings = aggregate_recordings(
        tuple(identities),
        clip_mse,
        clip_latent,
        source_role=source_role,
    )
    if set(recordings.recording_ids) != set(session_groups):
        raise RuntimeError("Task 2 scoring recording and session identities differ")
    return Task2ScoredSplit(
        role=source_role,
        recordings=recordings,
        session_groups=session_groups,
        clip_identities=tuple(identities),
        clip_mse=clip_mse,
        clip_latent=clip_latent,
    )


_DEVELOPMENT_BINDING_FIELDS = {
    "run_identity_sha256",
    "config_sha256",
    "config_file_sha256",
    "cache_lock_sha256",
    "implementation_sha256",
    "requirements_lock_sha256",
    "numerical_runtime_sha256",
    "model_contract_sha256",
    "scope",
    "production_evidence",
    "seed",
    "best_checkpoint_sha256",
}


def _development_binding(
    *,
    run_identity_sha256: str,
    config_sha256: str,
    config_file_sha256: str,
    cache_lock_sha256: str,
    execution_identity: Task2ExecutionIdentity,
    model_contract_sha256: str,
    scope: str,
    production_evidence: bool,
    seed: int,
    best_checkpoint_sha256: str,
) -> dict[str, Any]:
    value = {
        "run_identity_sha256": run_identity_sha256,
        "config_sha256": config_sha256,
        "config_file_sha256": config_file_sha256,
        "cache_lock_sha256": cache_lock_sha256,
        "implementation_sha256": execution_identity.implementation_sha256,
        "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
        "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
        "model_contract_sha256": model_contract_sha256,
        "scope": scope,
        "production_evidence": production_evidence,
        "seed": seed,
        "best_checkpoint_sha256": best_checkpoint_sha256,
    }
    _validate_development_binding(value)
    return value


def _validate_development_binding(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _DEVELOPMENT_BINDING_FIELDS:
        raise ValueError("Task 2 development binding fields are invalid")
    for name in _DEVELOPMENT_BINDING_FIELDS - {"scope", "production_evidence", "seed"}:
        if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
            raise ValueError(f"Task 2 development binding {name} is invalid")
    if (
        value["scope"] not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or type(value["production_evidence"]) is not bool
        or value["production_evidence"] is not (value["scope"] == PRODUCTION_SCOPE)
        or type(value["seed"]) is not int
        or value["seed"] not in {13, 37, 71}
    ):
        raise ValueError("Task 2 development binding scope or seed is invalid")


def _scored_split_record(
    scored: Task2ScoredSplit,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_binding = dict(binding)
    _validate_development_binding(resolved_binding)
    recording_records = []
    for recording in scored.recordings.recordings:
        recording_records.append(
            {
                **recording.to_record(),
                "session_group": scored.session_groups[recording.recording_id],
            }
        )
    return {
        "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
        **resolved_binding,
        "source_role": scored.role,
        "clip_count": len(scored.clip_identities),
        "recording_count": len(scored.recordings.recordings),
        "recordings": recording_records,
        "clips": [
            {
                **identity.to_record(),
                "session_group": scored.session_groups[identity.recording_id],
                "reconstruction_mse": float(scored.clip_mse[index]),
                "latent_embedding": [float(value) for value in scored.clip_latent[index]],
            }
            for index, identity in enumerate(scored.clip_identities)
        ],
    }


def _fit_and_publish_development_bundle(
    run_directory: Path,
    model: nn.Module,
    train: Task2Data,
    validation: Task2Data,
    *,
    batch_size: int,
    latent_dimensions: int,
    device: torch.device,
    binding: Mapping[str, Any],
    final_evaluation_contract: Mapping[str, Any],
    best_checkpoint_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_binding = dict(binding)
    _validate_development_binding(resolved_binding)
    resolved_final_contract = dict(final_evaluation_contract)
    best_sha256 = str(best_checkpoint_record["sha256"])
    if resolved_binding["best_checkpoint_sha256"] != best_sha256:
        raise ValueError("Task 2 development binding differs from its best checkpoint")
    training_scored = _score_development_split(
        model,
        train,
        source_role=KNOWN_TRAINING_ROLE,
        batch_size=batch_size,
        latent_dimensions=latent_dimensions,
        device=device,
    )
    validation_scored = _score_development_split(
        model,
        validation,
        source_role=KNOWN_VALIDATION_ROLE,
        batch_size=batch_size,
        latent_dimensions=latent_dimensions,
        device=device,
    )
    reference = fit_known_training_reference(training_scored.recordings)
    validation_latent_scores = latent_knn_novelty_scores(
        reference,
        validation_scored.recordings,
    )
    reconstruction_threshold = fit_known_validation_threshold(validation_scored.recordings)
    latent_threshold = fit_known_validation_latent_threshold(
        validation_scored.recordings,
        validation_latent_scores,
    )
    development_directory = run_directory / "development"
    artifacts = {
        "known_training_scores": _write_or_verify_json(
            development_directory / "known_training_recording_scores.json",
            _scored_split_record(
                training_scored,
                binding=resolved_binding,
            ),
        ),
        "known_validation_scores": _write_or_verify_json(
            development_directory / "known_validation_recording_scores.json",
            {
                **_scored_split_record(
                    validation_scored,
                    binding=resolved_binding,
                ),
                "latent_novelty_scores": [score.to_record() for score in validation_latent_scores],
            },
        ),
        "training_latent_reference": _write_or_verify_json(
            development_directory / "known_training_latent_reference.json",
            {
                "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                **resolved_binding,
                "reference": reference.to_record(),
            },
        ),
        "thresholds": _write_or_verify_json(
            development_directory / "known_validation_thresholds.json",
            {
                "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                **resolved_binding,
                "reconstruction": reconstruction_threshold.to_record(),
                "latent": latent_threshold.to_record(),
            },
        ),
    }
    bundle_value = {
        "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
        "complete": True,
        **resolved_binding,
        "best_checkpoint": dict(best_checkpoint_record),
        "artifacts": artifacts,
        "fit_roles": {
            "latent_reference": KNOWN_TRAINING_ROLE,
            "reconstruction_threshold": KNOWN_VALIDATION_ROLE,
            "latent_threshold": KNOWN_VALIDATION_ROLE,
        },
        "threshold_operator": ">",
        "final_evaluation_contract": resolved_final_contract,
    }
    bundle_record = _write_or_verify_json(
        development_directory / "development_bundle.lock.json",
        bundle_value,
    )
    return artifacts, bundle_record


def _validated_artifact_record(
    value: Any,
    *,
    expected_path: Path,
    run_directory: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Task 2 artifact record fields are invalid")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError("Task 2 artifact record path is invalid")
    resolved = _resolve_project_input_no_follow(value["path"])
    if (
        resolved != expected_path.resolve()
        or not is_relative_to(resolved, run_directory)
        or not isinstance(value["sha256"], str)
        or _SHA256.fullmatch(value["sha256"]) is None
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"Task 2 artifact record is invalid: {expected_path}")
    return {"path": str(resolved), "sha256": value["sha256"], "size_bytes": value["size_bytes"]}


def _read_bound_json_artifact(
    value: Any,
    *,
    expected_path: Path,
    run_directory: Path,
) -> tuple[Any, dict[str, Any]]:
    record = _validated_artifact_record(
        value,
        expected_path=expected_path,
        run_directory=run_directory,
    )
    payload, observed = _read_json_snapshot(
        record["path"],
        expected_sha256=record["sha256"],
    )
    if observed != record:
        raise ValueError(f"Task 2 artifact size or identity changed: {expected_path}")
    return payload, observed


def _read_bound_checkpoint_artifact(
    value: Any,
    *,
    expected_path: Path,
    run_directory: Path,
    expected_run_identity_sha256: str,
    expected_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _validated_artifact_record(
        value,
        expected_path=expected_path,
        run_directory=run_directory,
    )
    checkpoint = load_task2_checkpoint(
        record["path"],
        expected_sha256=record["sha256"],
        expected_run_identity_sha256=expected_run_identity_sha256,
        expected_type=expected_type,
    )
    _, _, size_bytes = _descriptor_snapshot(Path(record["path"]))
    if size_bytes != record["size_bytes"]:
        raise ValueError(f"Task 2 checkpoint size changed: {expected_path}")
    return checkpoint, record


def _validated_timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Task 2 {name} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Task 2 {name} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Task 2 {name} must include a timezone")
    return value


def _binding_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = {name: value.get(name) for name in _DEVELOPMENT_BINDING_FIELDS}
    _validate_development_binding(binding)
    return binding


def _validate_model_state_against_contract(
    state: Any,
    model_contract: Mapping[str, Any],
) -> None:
    if not isinstance(state, dict) or not isinstance(model_contract, Mapping):
        raise ValueError("Task 2 model state or contract is invalid")
    rows = model_contract.get("state")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Task 2 model contract state rows are invalid")
    expected_keys: list[str] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"key", "shape", "dtype"}
            or not isinstance(row["key"], str)
            or not isinstance(row["shape"], list)
            or any(type(size) is not int or size < 0 for size in row["shape"])
            or not isinstance(row["dtype"], str)
        ):
            raise ValueError("Task 2 model contract state row is invalid")
        expected_keys.append(row["key"])
        tensor = state.get(row["key"])
        if (
            not torch.is_tensor(tensor)
            or tensor.device.type != "cpu"
            or list(tensor.shape) != row["shape"]
            or str(tensor.dtype) != row["dtype"]
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
        ):
            raise ValueError("Task 2 checkpoint model state differs from its model contract")
    if list(state) != expected_keys:
        raise ValueError("Task 2 checkpoint model keys differ from its model contract")


def _parse_scored_split_artifact(
    value: Any,
    *,
    expected_binding: Mapping[str, Any],
    expected_role: str,
    latent_dimensions: int,
) -> tuple[RecordingBatch, dict[str, str], set[str]]:
    expected_fields = {
        "schema_version",
        *_DEVELOPMENT_BINDING_FIELDS,
        "source_role",
        "clip_count",
        "recording_count",
        "recordings",
        "clips",
    }
    if not isinstance(value, dict) or frozenset(value) not in {
        frozenset(expected_fields),
        frozenset({*expected_fields, "latent_novelty_scores"}),
    }:
        raise ValueError("Task 2 scored split artifact fields are invalid")
    if (
        value["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or _binding_from_mapping(value) != dict(expected_binding)
        or value["source_role"] != expected_role
        or type(value["clip_count"]) is not int
        or value["clip_count"] <= 0
        or type(value["recording_count"]) is not int
        or value["recording_count"] <= 0
        or not isinstance(value["clips"], list)
        or len(value["clips"]) != value["clip_count"]
        or not isinstance(value["recordings"], list)
        or len(value["recordings"]) != value["recording_count"]
    ):
        raise ValueError("Task 2 scored split artifact contract is invalid")
    identities: list[ClipIdentity] = []
    clip_mse: list[float] = []
    clip_latent: list[list[float]] = []
    session_groups: dict[str, str] = {}
    clip_ids: set[str] = set()
    for row in value["clips"]:
        if not isinstance(row, dict) or set(row) != {
            "recording_id",
            "clip_id",
            "session_group",
            "reconstruction_mse",
            "latent_embedding",
        }:
            raise ValueError("Task 2 scored clip fields are invalid")
        identity = ClipIdentity(row["recording_id"], row["clip_id"])
        session_group = row["session_group"]
        reconstruction_mse = row["reconstruction_mse"]
        latent_embedding = row["latent_embedding"]
        if (
            not isinstance(session_group, str)
            or not session_group
            or type(reconstruction_mse) is not float
            or not math.isfinite(reconstruction_mse)
            or reconstruction_mse < 0.0
            or not isinstance(latent_embedding, list)
            or len(latent_embedding) != latent_dimensions
            or any(type(item) is not float or not math.isfinite(item) for item in latent_embedding)
            or identity.clip_id in clip_ids
        ):
            raise ValueError("Task 2 scored clip values are invalid")
        existing_session = session_groups.setdefault(identity.recording_id, session_group)
        if existing_session != session_group:
            raise ValueError("Task 2 scored recording changes session group")
        identities.append(identity)
        clip_ids.add(identity.clip_id)
        clip_mse.append(reconstruction_mse)
        clip_latent.append(latent_embedding)
    aggregated = aggregate_recordings(
        tuple(identities),
        np.asarray(clip_mse, dtype=np.float64),
        np.asarray(clip_latent, dtype=np.float64),
        source_role=expected_role,
    )
    expected_recordings = [
        {
            **recording.to_record(),
            "session_group": session_groups[recording.recording_id],
        }
        for recording in aggregated.recordings
    ]
    if value["recordings"] != expected_recordings:
        raise ValueError("Task 2 recording scores do not rederive from their clip rows")
    return aggregated, session_groups, clip_ids


def _prepare_task2_verification_runtime(device_name: str, require_production: bool) -> torch.device:
    _require_project_venv()
    if device_name not in {"cpu", "mps"}:
        raise ValueError("Task 2 recorded device is invalid")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().casefold()
    if fallback not in {"", "0", "false"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    if require_production and device_name != "mps":
        raise ValueError("Production Task 2 evidence must use MPS")
    if device_name == "mps" and (
        not torch.backends.mps.is_built() or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("Task 2 verification requires available Apple MPS")
    return torch.device(device_name)


def verify_task2_development_run(
    completion_lock_path: str | Path,
    *,
    expected_sha256: str,
    require_production: bool = True,
) -> dict[str, Any]:
    """Recursively verify one completed development run without opening evaluation data."""

    if type(require_production) is not bool:
        raise TypeError("require_production must be a boolean")
    completion_path = _resolve_project_input_no_follow(completion_lock_path)
    run_directory = completion_path.parent
    if (
        completion_path.name != "result.lock.json"
        or not is_relative_to(run_directory, DEFAULT_RUN_ROOT)
        or not run_directory.is_dir()
    ):
        raise PermissionError("Task 2 completion lock is outside a run directory")
    completion, completion_record = _read_json_snapshot(
        completion_path,
        expected_sha256=expected_sha256,
    )
    completion_fields = {
        "schema_version",
        "run_identity_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "scope",
        "production_evidence",
        "result",
        "development_bundle",
    }
    if not isinstance(completion, dict) or set(completion) != completion_fields:
        raise ValueError("Task 2 completion lock fields are invalid")
    if completion["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError("Task 2 completion lock version is unsupported")
    result, _result_record = _read_bound_json_artifact(
        completion["result"],
        expected_path=run_directory / "result.json",
        run_directory=run_directory,
    )
    result_fields = {
        "schema_version",
        "complete",
        "started_at_utc",
        "completed_at_utc",
        "run_id",
        "run_directory",
        "run_identity_sha256",
        "config_sha256",
        "config_file_sha256",
        "cache_lock_sha256",
        "release_source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "model_contract_sha256",
        "scope",
        "production_evidence",
        "resumed",
        "resume_checkpoint",
        "epochs_completed",
        "early_stopped",
        "best_epoch",
        "best_validation_loss",
        "best_checkpoint",
        "latest_recovery_checkpoint",
        "development_bundle",
        "artifacts",
    }
    if (
        not isinstance(result, dict)
        or set(result) != result_fields
        or result["schema_version"] != RUN_SCHEMA_VERSION
        or result["complete"] is not True
        or result["run_directory"] != str(run_directory)
        or result["run_id"] != run_directory.name
        or type(result["resumed"]) is not bool
        or type(result["early_stopped"]) is not bool
        or type(result["epochs_completed"]) is not int
        or result["epochs_completed"] <= 0
        or type(result["best_epoch"]) is not int
        or not 1 <= result["best_epoch"] <= result["epochs_completed"]
        or type(result["best_validation_loss"]) is not float
        or not math.isfinite(result["best_validation_loss"])
        or result["best_validation_loss"] < 0.0
    ):
        raise ValueError("Task 2 completed result fields are invalid")
    _validated_timestamp(result["started_at_utc"], "start timestamp")
    _validated_timestamp(result["completed_at_utc"], "completion timestamp")
    if datetime.fromisoformat(result["completed_at_utc"]) < datetime.fromisoformat(
        result["started_at_utc"]
    ):
        raise ValueError("Task 2 completion precedes its start")
    binding_values = (
        "run_identity_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "scope",
        "production_evidence",
    )
    if any(completion[name] != result[name] for name in binding_values):
        raise ValueError("Task 2 completion lock differs from its result identity")
    if (
        result["scope"] not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or type(result["production_evidence"]) is not bool
        or result["production_evidence"] is not (result["scope"] == PRODUCTION_SCOPE)
        or (require_production and not result["production_evidence"])
    ):
        raise ValueError("Task 2 result evidence scope is invalid")
    for name in (
        "run_identity_sha256",
        "config_sha256",
        "config_file_sha256",
        "cache_lock_sha256",
        "release_source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "model_contract_sha256",
    ):
        if not isinstance(result[name], str) or _SHA256.fullmatch(result[name]) is None:
            raise ValueError(f"Task 2 result {name} is invalid")

    artifacts = result["artifacts"]
    artifact_fields = {
        "resolved_config",
        "run_identity",
        "provenance",
        "epoch_history",
        "best_checkpoint",
        "latest_recovery",
        "development",
        "development_bundle",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_fields:
        raise ValueError("Task 2 result artifact index is invalid")
    resolved_config, resolved_config_record = _read_bound_json_artifact(
        artifacts["resolved_config"],
        expected_path=run_directory / "resolved_config.json",
        run_directory=run_directory,
    )
    run_identity, run_identity_record = _read_bound_json_artifact(
        artifacts["run_identity"],
        expected_path=run_directory / "run_identity.json",
        run_directory=run_directory,
    )
    provenance, _provenance_record = _read_bound_json_artifact(
        artifacts["provenance"],
        expected_path=run_directory / "provenance.json",
        run_directory=run_directory,
    )
    history, history_record = _read_bound_json_artifact(
        artifacts["epoch_history"],
        expected_path=run_directory / "epoch_history.json",
        run_directory=run_directory,
    )
    if sha256_json(run_identity) != result["run_identity_sha256"]:
        raise ValueError("Task 2 run identity hash does not match")
    run_identity_fields = {
        "schema_version",
        "run_id",
        "task",
        "seed",
        "config_sha256",
        "config_file_sha256",
        "cache_lock_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime",
        "numerical_runtime_sha256",
        "model_contract",
        "model_contract_sha256",
        "optimizer_contract",
        "final_evaluation_contract",
        "scope",
        "production_evidence",
        "limits",
        "data",
    }
    if (
        not isinstance(run_identity, dict)
        or set(run_identity) != run_identity_fields
        or run_identity["schema_version"] != RUN_SCHEMA_VERSION
        or run_identity["run_id"] != result["run_id"]
        or run_identity["task"] != "task2_novelty_detection_development"
        or type(run_identity["seed"]) is not int
        or run_identity["seed"] not in {13, 37, 71}
        or any(run_identity[name] != result[name] for name in binding_values[1:])
        or run_identity["config_sha256"] != result["config_sha256"]
        or run_identity["config_file_sha256"] != result["config_file_sha256"]
        or run_identity["cache_lock_sha256"] != result["cache_lock_sha256"]
        or run_identity["model_contract_sha256"] != result["model_contract_sha256"]
        or not isinstance(run_identity["numerical_runtime"], dict)
        or not isinstance(run_identity["model_contract"], dict)
        or not isinstance(run_identity["optimizer_contract"], dict)
        or not isinstance(run_identity["final_evaluation_contract"], dict)
        or not isinstance(run_identity["limits"], dict)
        or set(run_identity["limits"]) != {"maximum_epochs", "batch_size", "patience"}
        or any(type(value) is not int or value <= 0 for value in run_identity["limits"].values())
        or not isinstance(run_identity["data"], dict)
        or set(run_identity["data"])
        != {
            "train_clips",
            "train_recordings",
            "validation_clips",
            "validation_recordings",
            "selection_strategy",
        }
        or any(
            type(run_identity["data"][name]) is not int or run_identity["data"][name] <= 0
            for name in (
                "train_clips",
                "train_recordings",
                "validation_clips",
                "validation_recordings",
            )
        )
        or run_identity["data"]["selection_strategy"] != "energy"
        or sha256_json(run_identity["numerical_runtime"])
        != run_identity["numerical_runtime_sha256"]
        or sha256_json(run_identity["model_contract"]) != run_identity["model_contract_sha256"]
    ):
        raise ValueError("Task 2 run identity fields are invalid")
    device = _prepare_task2_verification_runtime(
        str(run_identity["numerical_runtime"].get("device") or ""),
        require_production,
    )
    expected_execution_identity = _capture_execution_identity(device)
    if any(
        getattr(expected_execution_identity, name) != run_identity[name]
        for name in (
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
        )
    ):
        raise ValueError("Task 2 current execution identity differs from the completed run")
    if expected_execution_identity.numerical_runtime != run_identity["numerical_runtime"]:
        raise ValueError("Task 2 numerical runtime record differs from the current runtime")

    config = load_locked_task2_config()
    config_sha256 = config_fingerprint(config)
    _, config_file_sha256, _ = _descriptor_snapshot(LOCKED_CONFIG_PATH)
    expected_resolved_config = {
        "config_path": LOCKED_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config_file_sha256": config_file_sha256,
        "config_sha256": config_sha256,
        "resolved": public_config(config),
    }
    model_contract = run_identity["model_contract"]
    model_contract_fields = {
        "architecture",
        "model_type",
        "input_shape",
        "latent_dimensions",
        "parameter_counts",
        "state",
    }
    parameter_counts_record = model_contract.get("parameter_counts")
    if (
        resolved_config != expected_resolved_config
        or result["config_sha256"] != config_sha256
        or result["config_file_sha256"] != config_file_sha256
        or set(model_contract) != model_contract_fields
        or model_contract["architecture"] != config["architecture"]
        or not isinstance(model_contract["model_type"], str)
        or not model_contract["model_type"]
        or model_contract["input_shape"] != [1, 224, 224]
        or model_contract["latent_dimensions"] != int(config["latent_dimensions"])
        or not isinstance(parameter_counts_record, dict)
        or set(parameter_counts_record) != {"total", "trainable"}
        or any(
            type(parameter_counts_record[name]) is not int or parameter_counts_record[name] <= 0
            for name in ("total", "trainable")
        )
        or parameter_counts_record["total"] != parameter_counts_record["trainable"]
        or not isinstance(model_contract["state"], list)
        or not model_contract["state"]
        or run_identity["optimizer_contract"] != _optimizer_contract(config)
        or run_identity["final_evaluation_contract"] != _final_evaluation_contract(config)
        or (
            require_production
            and run_identity["limits"]
            != {
                "maximum_epochs": int(config["training"]["maximum_epochs"]),
                "batch_size": int(config["training"]["batch_size"]),
                "patience": int(config["training"]["early_stopping_patience"]),
            }
        )
    ):
        raise ValueError("Task 2 resolved configuration or method contract changed")
    provenance_fields = {
        "schema_version",
        "started_at_utc",
        "run_identity_sha256",
        "command",
        "config_path",
        "config_file_sha256",
        "config_sha256",
        "cache_root",
        "cache_lock_sha256",
        "release_source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime",
        "numerical_runtime_sha256",
        "model_contract",
        "model_contract_sha256",
        "optimizer_contract",
        "final_evaluation_contract",
        "scope",
        "production_evidence",
        "initial_artifacts",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != provenance_fields
        or provenance.get("schema_version") != RUN_SCHEMA_VERSION
        or provenance.get("started_at_utc") != result["started_at_utc"]
        or provenance.get("run_identity_sha256") != result["run_identity_sha256"]
        or provenance.get("config_path") != LOCKED_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
        or provenance.get("config_file_sha256") != config_file_sha256
        or provenance.get("config_sha256") != config_sha256
        or provenance.get("cache_lock_sha256") != result["cache_lock_sha256"]
        or provenance.get("release_source_fingerprint_sha256")
        != result["release_source_fingerprint_sha256"]
        or provenance.get("implementation_sha256") != result["implementation_sha256"]
        or provenance.get("requirements_lock_sha256") != result["requirements_lock_sha256"]
        or provenance.get("numerical_runtime") != run_identity["numerical_runtime"]
        or provenance.get("numerical_runtime_sha256") != result["numerical_runtime_sha256"]
        or provenance.get("model_contract") != run_identity["model_contract"]
        or provenance.get("model_contract_sha256") != result["model_contract_sha256"]
        or provenance.get("optimizer_contract") != run_identity["optimizer_contract"]
        or provenance.get("final_evaluation_contract") != run_identity["final_evaluation_contract"]
        or provenance.get("scope") != result["scope"]
        or provenance.get("production_evidence") is not result["production_evidence"]
        or provenance.get("initial_artifacts")
        != {
            "resolved_config": resolved_config_record,
            "run_identity": run_identity_record,
        }
        or not isinstance(provenance.get("cache_root"), str)
        or not provenance["cache_root"]
        or (
            require_production
            and Path(provenance["cache_root"]).resolve() != DEFAULT_CACHE_ROOT.resolve()
        )
        or not isinstance(provenance.get("command"), list)
        or any(not isinstance(part, str) for part in provenance["command"])
    ):
        raise ValueError("Task 2 provenance does not match the completed identity")
    _validated_timestamp(provenance["started_at_utc"], "provenance start timestamp")

    _validate_history(history, result["epochs_completed"])
    best_path = run_directory / "best_candidates" / f"best_epoch_{result['best_epoch']:04d}.pt"
    best_checkpoint, best_record = _read_bound_checkpoint_artifact(
        result["best_checkpoint"],
        expected_path=best_path,
        run_directory=run_directory,
        expected_run_identity_sha256=result["run_identity_sha256"],
        expected_type="best",
    )
    latest_path = run_directory / "recovery" / f"recovery_epoch_{result['epochs_completed']:04d}.pt"
    latest_checkpoint, latest_record = _read_bound_checkpoint_artifact(
        result["latest_recovery_checkpoint"],
        expected_path=latest_path,
        run_directory=run_directory,
        expected_run_identity_sha256=result["run_identity_sha256"],
        expected_type="recovery",
    )
    if (
        artifacts["best_checkpoint"] != best_record
        or artifacts["latest_recovery"] != latest_record
        or latest_checkpoint["completed_epoch"] != result["epochs_completed"]
        or latest_checkpoint["limits"] != run_identity["limits"]
        or latest_checkpoint["history"] != history
        or latest_checkpoint["stop_requested"] is not result["early_stopped"]
        or latest_checkpoint["best_candidate"]["sha256"] != best_record["sha256"]
        or latest_checkpoint["best_candidate"]["epoch"] != result["best_epoch"]
        or best_checkpoint["epoch"] != result["best_epoch"]
        or best_checkpoint["score"]["validation_loss"] != result["best_validation_loss"]
        or history[result["best_epoch"] - 1]["validation"]["loss"] != result["best_validation_loss"]
        or history_record != artifacts["epoch_history"]
    ):
        raise ValueError("Task 2 checkpoint, history, or result selection binding is invalid")
    _validate_model_state_against_contract(best_checkpoint["model"], run_identity["model_contract"])
    _validate_model_state_against_contract(
        latest_checkpoint["model"], run_identity["model_contract"]
    )
    expected_common = {
        "run_id": result["run_id"],
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "config_file_sha256": result["config_file_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "model_contract_sha256": result["model_contract_sha256"],
        "scope": result["scope"],
        "production_evidence": result["production_evidence"],
        "seed": run_identity["seed"],
    }
    for checkpoint in (best_checkpoint, latest_checkpoint):
        if any(checkpoint.get(name) != value for name, value in expected_common.items()):
            raise ValueError("Task 2 checkpoint common identity differs from the run")

    bundle, bundle_record = _read_bound_json_artifact(
        result["development_bundle"],
        expected_path=run_directory / "development" / "development_bundle.lock.json",
        run_directory=run_directory,
    )
    if (
        completion["development_bundle"] != bundle_record
        or artifacts["development_bundle"] != bundle_record
    ):
        raise ValueError("Task 2 completion lock differs from its development bundle")
    expected_binding = {
        "run_identity_sha256": result["run_identity_sha256"],
        "config_sha256": result["config_sha256"],
        "config_file_sha256": result["config_file_sha256"],
        "cache_lock_sha256": result["cache_lock_sha256"],
        "implementation_sha256": result["implementation_sha256"],
        "requirements_lock_sha256": result["requirements_lock_sha256"],
        "numerical_runtime_sha256": result["numerical_runtime_sha256"],
        "model_contract_sha256": result["model_contract_sha256"],
        "scope": result["scope"],
        "production_evidence": result["production_evidence"],
        "seed": run_identity["seed"],
        "best_checkpoint_sha256": best_record["sha256"],
    }
    _validate_development_binding(expected_binding)
    bundle_fields = {
        "schema_version",
        "complete",
        *_DEVELOPMENT_BINDING_FIELDS,
        "best_checkpoint",
        "artifacts",
        "fit_roles",
        "threshold_operator",
        "final_evaluation_contract",
    }
    if (
        not isinstance(bundle, dict)
        or set(bundle) != bundle_fields
        or bundle["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or bundle["complete"] is not True
        or _binding_from_mapping(bundle) != expected_binding
        or bundle["best_checkpoint"] != best_record
        or bundle["fit_roles"]
        != {
            "latent_reference": KNOWN_TRAINING_ROLE,
            "reconstruction_threshold": KNOWN_VALIDATION_ROLE,
            "latent_threshold": KNOWN_VALIDATION_ROLE,
        }
        or bundle["threshold_operator"] != ">"
        or bundle["final_evaluation_contract"] != _final_evaluation_contract(config)
    ):
        raise ValueError("Task 2 development bundle contract is invalid")
    development_records = bundle["artifacts"]
    if (
        not isinstance(development_records, dict)
        or set(development_records)
        != {
            "known_training_scores",
            "known_validation_scores",
            "training_latent_reference",
            "thresholds",
        }
        or artifacts["development"] != development_records
    ):
        raise ValueError("Task 2 development artifact index is invalid")
    development_directory = run_directory / "development"
    training_scores, _ = _read_bound_json_artifact(
        development_records["known_training_scores"],
        expected_path=development_directory / "known_training_recording_scores.json",
        run_directory=run_directory,
    )
    validation_scores, _ = _read_bound_json_artifact(
        development_records["known_validation_scores"],
        expected_path=development_directory / "known_validation_recording_scores.json",
        run_directory=run_directory,
    )
    reference_record, _ = _read_bound_json_artifact(
        development_records["training_latent_reference"],
        expected_path=development_directory / "known_training_latent_reference.json",
        run_directory=run_directory,
    )
    threshold_record, _ = _read_bound_json_artifact(
        development_records["thresholds"],
        expected_path=development_directory / "known_validation_thresholds.json",
        run_directory=run_directory,
    )
    latent_dimensions = int(run_identity["model_contract"]["latent_dimensions"])
    training_batch, training_sessions, training_clips = _parse_scored_split_artifact(
        training_scores,
        expected_binding=expected_binding,
        expected_role=KNOWN_TRAINING_ROLE,
        latent_dimensions=latent_dimensions,
    )
    validation_batch, validation_sessions, validation_clips = _parse_scored_split_artifact(
        validation_scores,
        expected_binding=expected_binding,
        expected_role=KNOWN_VALIDATION_ROLE,
        latent_dimensions=latent_dimensions,
    )
    if (
        set(training_batch.recording_ids).intersection(validation_batch.recording_ids)
        or set(training_sessions.values()).intersection(validation_sessions.values())
        or training_clips.intersection(validation_clips)
    ):
        raise ValueError("Task 2 persisted development splits overlap")
    expected_data = {
        "train_clips": len(training_clips),
        "train_recordings": len(training_batch.recordings),
        "validation_clips": len(validation_clips),
        "validation_recordings": len(validation_batch.recordings),
        "selection_strategy": "energy",
    }
    if run_identity["data"] != expected_data:
        raise ValueError("Task 2 persisted development counts differ from the run identity")
    if require_production and (
        result["cache_lock_sha256"] != KNOWN_CACHE_LOCK_SHA256
        or len(training_batch.recordings) != 1_254
        or len(validation_batch.recordings) != 271
        or len(training_clips) != 5_319
        or len(validation_clips) != 1_138
        or run_identity["model_contract"].get("parameter_counts")
        != {"total": EXPECTED_PARAMETER_COUNT, "trainable": EXPECTED_PARAMETER_COUNT}
        or run_identity["model_contract"].get("model_type")
        != "bird_audio.models.ConvolutionalAutoencoder"
    ):
        raise ValueError("Production Task 2 development evidence counts or cache changed")
    recomputed_reference = fit_known_training_reference(training_batch)
    recomputed_latent_scores = latent_knn_novelty_scores(
        recomputed_reference,
        validation_batch,
    )
    recomputed_reconstruction_threshold = fit_known_validation_threshold(validation_batch)
    recomputed_latent_threshold = fit_known_validation_latent_threshold(
        validation_batch,
        recomputed_latent_scores,
    )
    expected_reference_fields = {
        "schema_version",
        *_DEVELOPMENT_BINDING_FIELDS,
        "reference",
    }
    expected_threshold_fields = {
        "schema_version",
        *_DEVELOPMENT_BINDING_FIELDS,
        "reconstruction",
        "latent",
    }
    if (
        not isinstance(reference_record, dict)
        or set(reference_record) != expected_reference_fields
        or reference_record["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or _binding_from_mapping(reference_record) != expected_binding
        or reference_record["reference"] != recomputed_reference.to_record()
        or validation_scores.get("latent_novelty_scores")
        != [score.to_record() for score in recomputed_latent_scores]
        or not isinstance(threshold_record, dict)
        or set(threshold_record) != expected_threshold_fields
        or threshold_record["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or _binding_from_mapping(threshold_record) != expected_binding
        or threshold_record["reconstruction"] != recomputed_reconstruction_threshold.to_record()
        or threshold_record["latent"] != recomputed_latent_threshold.to_record()
    ):
        raise ValueError("Task 2 reference, scores, or thresholds do not rederive exactly")
    resume_value = result["resume_checkpoint"]
    if result["resumed"] is not (resume_value is not None):
        raise ValueError("Task 2 result resume flag and checkpoint record differ")
    if resume_value is not None:
        if (
            not isinstance(resume_value, dict)
            or set(resume_value) != {"path", "sha256", "size_bytes"}
            or not isinstance(resume_value["path"], str)
        ):
            raise ValueError("Task 2 resume checkpoint record is invalid")
        resume_path = _resolve_project_input_no_follow(resume_value["path"])
        match = re.fullmatch(r"recovery_epoch_(\d{4})\.pt", resume_path.name)
        if match is None or resume_path.parent != run_directory / "recovery":
            raise ValueError("Task 2 resume checkpoint path is not canonical")
        resume_epoch = int(match.group(1))
        if not 1 <= resume_epoch <= result["epochs_completed"]:
            raise ValueError("Task 2 resume checkpoint epoch is invalid")
        resume_checkpoint, _ = _read_bound_checkpoint_artifact(
            resume_value,
            expected_path=resume_path,
            run_directory=run_directory,
            expected_run_identity_sha256=result["run_identity_sha256"],
            expected_type="recovery",
        )
        resume_candidate = resume_checkpoint["best_candidate"]
        resume_candidate_path = run_directory / resume_candidate["path"]
        resume_best_checkpoint = load_task2_checkpoint(
            resume_candidate_path,
            expected_sha256=resume_candidate["sha256"],
            expected_run_identity_sha256=result["run_identity_sha256"],
            expected_type="best",
        )
        if (
            resume_checkpoint["completed_epoch"] != resume_epoch
            or resume_checkpoint["limits"] != run_identity["limits"]
            or resume_checkpoint["history"] != history[:resume_epoch]
            or resume_best_checkpoint["epoch"] != resume_candidate["epoch"]
            or resume_best_checkpoint["score"] != resume_checkpoint["early_stopping"]["best"]
            or any(
                resume_checkpoint.get(name) != value or resume_best_checkpoint.get(name) != value
                for name, value in expected_common.items()
            )
        ):
            raise ValueError("Task 2 resume checkpoint does not rederive from the completed run")
        _validate_model_state_against_contract(
            resume_checkpoint["model"], run_identity["model_contract"]
        )
        _validate_model_state_against_contract(
            resume_best_checkpoint["model"], run_identity["model_contract"]
        )
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
        "development_bundle_sha256": bundle_record["sha256"],
        "training_recordings": len(training_batch.recordings),
        "training_clips": len(training_clips),
        "validation_recordings": len(validation_batch.recordings),
        "validation_clips": len(validation_clips),
        "thresholds_rederived": True,
    }


def load_locked_task2_best_model_for_evaluation(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_run_identity_sha256: str,
    device: torch.device,
) -> tuple[ConvolutionalAutoencoder, dict[str, Any]]:
    """Load one production best checkpoint without opening any dataset."""

    if not isinstance(device, torch.device) or device.type != "mps" or device.index is not None:
        raise ValueError("Task 2 evaluation model loading requires torch.device('mps')")
    resolved_device = _prepare_task2_verification_runtime("mps", require_production=True)
    if resolved_device != device:
        raise ValueError("Task 2 evaluation device differs from the supplied MPS device")
    execution_identity = _capture_execution_identity(device)
    checkpoint = load_task2_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_run_identity_sha256=expected_run_identity_sha256,
        expected_type="best",
    )
    if (
        checkpoint["scope"] != PRODUCTION_SCOPE
        or checkpoint["production_evidence"] is not True
        or checkpoint["cache_lock_sha256"] != KNOWN_CACHE_LOCK_SHA256
        or checkpoint["implementation_sha256"] != execution_identity.implementation_sha256
        or checkpoint["requirements_lock_sha256"] != execution_identity.requirements_lock_sha256
        or checkpoint["numerical_runtime_sha256"] != execution_identity.numerical_runtime_sha256
    ):
        raise ValueError("Task 2 evaluation checkpoint identity is not production locked")

    config = load_locked_task2_config()
    config_sha256 = config_fingerprint(config)
    _, config_file_sha256, _ = _descriptor_snapshot(LOCKED_CONFIG_PATH)
    if (
        checkpoint["config_sha256"] != config_sha256
        or checkpoint["config_file_sha256"] != config_file_sha256
    ):
        raise ValueError("Task 2 evaluation checkpoint configuration changed")

    model = _build_task2_model(config, device, test_injection=None)
    model_contract = _model_contract(model, config)
    model_contract_sha256 = sha256_json(model_contract)
    if checkpoint["model_contract_sha256"] != model_contract_sha256:
        raise ValueError("Task 2 evaluation checkpoint model contract changed")
    _validate_model_state_against_contract(checkpoint["model"], model_contract)
    model.load_state_dict(checkpoint["model"], strict=True)

    optimizer = build_task2_optimizer(model, config)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _validate_optimizer_after_resume(optimizer, model, config)
    del optimizer

    model.requires_grad_(False)
    model.eval()
    _require_execution_identity_unchanged(execution_identity, device)
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": checkpoint["run_id"],
        "run_identity_sha256": checkpoint["run_identity_sha256"],
        "best_checkpoint_sha256": expected_sha256,
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["score"]["validation_loss"],
        "seed": checkpoint["seed"],
        "config_sha256": checkpoint["config_sha256"],
        "config_file_sha256": checkpoint["config_file_sha256"],
        "cache_lock_sha256": checkpoint["cache_lock_sha256"],
        "implementation_sha256": checkpoint["implementation_sha256"],
        "requirements_lock_sha256": checkpoint["requirements_lock_sha256"],
        "numerical_runtime_sha256": checkpoint["numerical_runtime_sha256"],
        "model_contract_sha256": checkpoint["model_contract_sha256"],
        "scope": checkpoint["scope"],
        "production_evidence": checkpoint["production_evidence"],
        "device": device.type,
    }
    return model, metadata


def run_task2_development(
    *,
    seed: int,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
    output_root: str | Path = DEFAULT_RUN_ROOT,
    command: Sequence[str] = (),
    run_id: str | None = None,
    train_data: Task2Data | None = None,
    validation_data: Task2Data | None = None,
    test_injection: Task2TestInjection | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    invocation_started_at_utc = datetime.now(UTC).isoformat()
    config = load_locked_task2_config()
    if type(seed) is not int or seed not in config["seeds"]:
        raise ValueError("Task 2 seed is outside the locked seed set")
    if (resume_checkpoint is None) != (resume_checkpoint_sha256 is None):
        raise ValueError("Task 2 resume path and SHA-256 must be supplied together")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("Task 2 command must be a sequence of command parts")
    device = _resolve_runtime(test_injection)
    scope = ISOLATED_TEST_SCOPE if test_injection is not None else PRODUCTION_SCOPE
    production_evidence = scope == PRODUCTION_SCOPE
    validated_output_root = _validated_run_output_path(output_root)
    canonical_output_root = Path(os.path.abspath(DEFAULT_RUN_ROOT))
    if production_evidence and validated_output_root != canonical_output_root:
        raise PermissionError("Production Task 2 run output must use exact runs/task2_v2")
    if test_injection is not None and validated_output_root == canonical_output_root:
        raise PermissionError("Isolated Task 2 tests cannot publish in the production run root")
    execution_identity = _capture_execution_identity(device)
    release_source_fingerprint_sha256 = source_fingerprint()
    if _SHA256.fullmatch(release_source_fingerprint_sha256) is None:
        raise RuntimeError("Task 2 release source fingerprint is invalid")
    final_evaluation_contract = _final_evaluation_contract(config)

    with ExitStack() as stack:
        if (train_data is None) != (validation_data is None):
            raise ValueError("Task 2 train and validation data must be supplied together")
        if train_data is None or validation_data is None:
            if test_injection is not None:
                raise ValueError("Task 2 CPU test injection requires explicit development fixtures")
            train, validation = _open_real_data(
                stack,
                cache_root=cache_root,
                ffmpeg=ffmpeg,
                expected_lock_sha256=expected_lock_sha256,
            )
        else:
            if test_injection is None:
                raise PermissionError("Task 2 explicit data injection is allowed only for tests")
            train, validation = train_data, validation_data
        _validate_development_data(
            train,
            validation,
            production=test_injection is None,
        )
        _require_execution_identity_unchanged(execution_identity, device)

        config_sha256 = config_fingerprint(config)
        _, config_file_sha256, _ = _descriptor_snapshot(LOCKED_CONFIG_PATH)
        maximum_epochs, batch_size, patience = _resolved_limits(config, test_injection)
        limits = {
            "maximum_epochs": maximum_epochs,
            "batch_size": batch_size,
            "patience": patience,
        }

        seed_task2(seed, device)
        model = _build_task2_model(config, device, test_injection)
        optimizer = build_task2_optimizer(model, config)
        model_contract = _model_contract(model, config)
        model_contract_sha256 = sha256_json(model_contract)
        optimizer_contract = _optimizer_contract(config)

        recovery_checkpoint: dict[str, Any] | None = None
        resume_record: dict[str, Any] | None = None
        resumed = resume_checkpoint is not None
        if resumed:
            if resume_checkpoint is None or resume_checkpoint_sha256 is None:
                raise RuntimeError("Task 2 resume arguments became inconsistent")
            run_directory, resolved_resume = _resolve_resume_directory(
                validated_output_root,
                resume_checkpoint,
            )
            recovery_checkpoint = load_task2_checkpoint(
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
                raise ValueError("Task 2 resume requires the latest canonical recovery checkpoint")
            selected_run_id = str(recovery_checkpoint["run_id"])
            if run_id is not None and run_id != selected_run_id:
                raise ValueError("Requested Task 2 run ID differs from the recovery checkpoint")
            if run_directory.name != selected_run_id:
                raise ValueError("Task 2 recovery run ID differs from its directory")
            if recovery_checkpoint["limits"] != limits:
                raise ValueError("Task 2 recovery limits differ from the current locked run")
            _, _, resume_size = _descriptor_snapshot(resolved_resume)
            resume_record = {
                "path": str(resolved_resume),
                "sha256": resume_checkpoint_sha256,
                "size_bytes": resume_size,
            }
        else:
            selected_run_id = run_id or make_run_id(
                "task2",
                "autoencoder",
                seed,
                config_sha256,
                train.lock_sha256,
            )
            run_directory = _run_directory(validated_output_root, selected_run_id)
        _ensure_run_subdirectories(run_directory)

        run_identity = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "task": "task2_novelty_detection_development",
            "seed": seed,
            "config_sha256": config_sha256,
            "config_file_sha256": config_file_sha256,
            "cache_lock_sha256": train.lock_sha256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime": execution_identity.numerical_runtime,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "model_contract": model_contract,
            "model_contract_sha256": model_contract_sha256,
            "optimizer_contract": optimizer_contract,
            "final_evaluation_contract": final_evaluation_contract,
            "scope": scope,
            "production_evidence": production_evidence,
            "limits": limits,
            "data": {
                "train_clips": len(train),
                "train_recordings": train.recording_count,
                "validation_clips": len(validation),
                "validation_recordings": validation.recording_count,
                "selection_strategy": "energy",
            },
        }
        run_identity_sha256 = sha256_json(run_identity)
        expected_common = _checkpoint_common_state(
            run_id=selected_run_id,
            run_identity_sha256=run_identity_sha256,
            config_sha256=config_sha256,
            config_file_sha256=config_file_sha256,
            cache_lock_sha256=train.lock_sha256,
            execution_identity=execution_identity,
            model_contract_sha256=model_contract_sha256,
            scope=scope,
            production_evidence=production_evidence,
            seed=seed,
        )
        if recovery_checkpoint is not None:
            ordered_binding_keys = [
                key for key in expected_common if key != "run_identity_sha256"
            ] + ["run_identity_sha256"]
            for key in ordered_binding_keys:
                value = expected_common[key]
                if recovery_checkpoint.get(key) != value:
                    raise ValueError(f"Task 2 recovery changed locked run field: {key}")

        history: list[dict[str, Any]] = []
        artifact_records: dict[str, Any] = {}
        run_started_at_utc = invocation_started_at_utc
        try:
            resolved_config_value = {
                "config_path": LOCKED_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
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
                    raise ValueError("Task 2 recovery artifacts differ from the locked identity")
                if (
                    not isinstance(observed_provenance, dict)
                    or observed_provenance.get("run_identity_sha256") != run_identity_sha256
                    or observed_provenance.get("implementation_sha256")
                    != execution_identity.implementation_sha256
                    or observed_provenance.get("requirements_lock_sha256")
                    != execution_identity.requirements_lock_sha256
                    or observed_provenance.get("numerical_runtime_sha256")
                    != execution_identity.numerical_runtime_sha256
                    or observed_provenance.get("model_contract_sha256") != model_contract_sha256
                    or observed_provenance.get("scope") != scope
                    or observed_provenance.get("production_evidence") is not production_evidence
                ):
                    raise ValueError("Task 2 recovery provenance differs from the locked identity")
                run_started_at_utc = str(observed_provenance.get("started_at_utc") or "")
                release_source_fingerprint_sha256 = str(
                    observed_provenance.get("release_source_fingerprint_sha256") or ""
                )
                if (
                    not run_started_at_utc
                    or _SHA256.fullmatch(release_source_fingerprint_sha256) is None
                ):
                    raise ValueError("Task 2 recovery provenance timing or source is invalid")
                artifact_records.update(
                    {
                        "resolved_config": config_record,
                        "run_identity": identity_record,
                        "provenance": provenance_record,
                    }
                )
                result_path = run_directory / "result.json"
                if result_path.exists():
                    completed_result, result_record = _read_json_snapshot(result_path)
                    if (
                        not isinstance(completed_result, dict)
                        or completed_result.get("complete") is not True
                        or completed_result.get("run_identity_sha256") != run_identity_sha256
                        or completed_result.get("latest_recovery_checkpoint", {}).get("sha256")
                        != resume_checkpoint_sha256
                    ):
                        raise ValueError("Existing Task 2 result is not bound to this recovery")
                    completion_record = _write_or_verify_json(
                        run_directory / "result.lock.json",
                        {
                            "schema_version": RUN_SCHEMA_VERSION,
                            "run_identity_sha256": run_identity_sha256,
                            "implementation_sha256": execution_identity.implementation_sha256,
                            "requirements_lock_sha256": (
                                execution_identity.requirements_lock_sha256
                            ),
                            "numerical_runtime_sha256": (
                                execution_identity.numerical_runtime_sha256
                            ),
                            "scope": scope,
                            "production_evidence": production_evidence,
                            "result": result_record,
                            "development_bundle": completed_result["development_bundle"],
                        },
                    )
                    _require_execution_identity_unchanged(execution_identity, device)
                    verify_task2_development_run(
                        completion_record["path"],
                        expected_sha256=completion_record["sha256"],
                        require_production=production_evidence,
                    )
                    return {
                        **completed_result,
                        "result_artifact": result_record,
                        "completion_lock_artifact": completion_record,
                    }
            else:
                artifact_records["resolved_config"] = _write_json_create_only(
                    run_directory / "resolved_config.json",
                    resolved_config_value,
                )
                artifact_records["run_identity"] = _write_json_create_only(
                    run_directory / "run_identity.json",
                    run_identity,
                )
                provenance = {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "started_at_utc": run_started_at_utc,
                    "run_identity_sha256": run_identity_sha256,
                    "command": [str(part) for part in command],
                    "config_path": LOCKED_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "config_file_sha256": config_file_sha256,
                    "config_sha256": config_sha256,
                    "cache_root": str(train.root),
                    "cache_lock_sha256": train.lock_sha256,
                    "release_source_fingerprint_sha256": release_source_fingerprint_sha256,
                    "implementation_sha256": execution_identity.implementation_sha256,
                    "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                    "numerical_runtime": execution_identity.numerical_runtime,
                    "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                    "model_contract": model_contract,
                    "model_contract_sha256": model_contract_sha256,
                    "optimizer_contract": optimizer_contract,
                    "final_evaluation_contract": final_evaluation_contract,
                    "scope": scope,
                    "production_evidence": production_evidence,
                    "initial_artifacts": {
                        "resolved_config": artifact_records["resolved_config"],
                        "run_identity": artifact_records["run_identity"],
                    },
                }
                artifact_records["provenance"] = _write_json_create_only(
                    run_directory / "provenance.json",
                    provenance,
                )

            early_stopping = Task2EarlyStopping(patience)
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
                candidate_checkpoint = load_task2_checkpoint(
                    candidate_path,
                    expected_sha256=best_candidate["sha256"],
                    expected_run_identity_sha256=run_identity_sha256,
                    expected_type="best",
                )
                for key, value in expected_common.items():
                    if candidate_checkpoint.get(key) != value:
                        raise ValueError(f"Task 2 best candidate changed locked field: {key}")
                if _validate_score(candidate_checkpoint["score"]) != early_stopping.best:
                    raise ValueError("Task 2 recovery best score differs from its candidate")
                _, _, candidate_size = _descriptor_snapshot(candidate_path)
                best_checkpoint_record = {
                    "path": str(candidate_path),
                    "sha256": best_candidate["sha256"],
                    "size_bytes": candidate_size,
                }
                history = list(recovery_checkpoint["history"])
                start_epoch_index = int(recovery_checkpoint["next_epoch_index"])
                stop_requested = bool(recovery_checkpoint["stop_requested"])
                _restore_rng_state(recovery_checkpoint["rng_state"], device)

            epoch_indices = range(0) if stop_requested else range(start_epoch_index, maximum_epochs)
            for epoch_index in epoch_indices:
                train_metrics = train_task2_epoch(
                    model,
                    optimizer,
                    train,
                    seed=seed,
                    epoch_index=epoch_index,
                    batch_size=batch_size,
                    latent_dimensions=int(config["latent_dimensions"]),
                    device=device,
                )
                validation_metrics = validate_task2(
                    model,
                    validation,
                    batch_size=batch_size,
                    latent_dimensions=int(config["latent_dimensions"]),
                    device=device,
                )
                score = Task2CheckpointScore(validation_metrics.loss, epoch_index + 1)
                improved, should_stop = early_stopping.update(score)
                history.append(
                    {
                        "epoch": epoch_index + 1,
                        "train": train_metrics,
                        "validation": {
                            "loss": validation_metrics.loss,
                            "clip_count": validation_metrics.clip_count,
                            "pixel_count": validation_metrics.pixel_count,
                            "reduction": "global_pixel_mean",
                        },
                        "checkpoint_improved": improved,
                    }
                )
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
                    }
                    candidate_relative = f"best_candidates/best_epoch_{epoch_index + 1:04d}.pt"
                    _require_execution_identity_unchanged(execution_identity, device)
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
                    raise RuntimeError("Task 2 epoch has no durable best candidate")
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
                _require_execution_identity_unchanged(execution_identity, device)
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
                raise RuntimeError("Task 2 training completed without a best checkpoint")
            best_checkpoint = load_task2_checkpoint(
                best_checkpoint_record["path"],
                expected_sha256=best_checkpoint_record["sha256"],
                expected_run_identity_sha256=run_identity_sha256,
                expected_type="best",
            )
            for key, value in expected_common.items():
                if best_checkpoint.get(key) != value:
                    raise ValueError(f"Frozen Task 2 best checkpoint changed locked field: {key}")
            model.load_state_dict(best_checkpoint["model"], strict=True)
            model.requires_grad_(False)
            _require_execution_identity_unchanged(execution_identity, device)
            artifact_records["epoch_history"] = _write_or_verify_json(
                run_directory / "epoch_history.json",
                history,
            )
            artifact_records["best_checkpoint"] = best_checkpoint_record
            artifact_records["latest_recovery"] = latest_recovery_record
            binding = _development_binding(
                run_identity_sha256=run_identity_sha256,
                config_sha256=config_sha256,
                config_file_sha256=config_file_sha256,
                cache_lock_sha256=train.lock_sha256,
                execution_identity=execution_identity,
                model_contract_sha256=model_contract_sha256,
                scope=scope,
                production_evidence=production_evidence,
                seed=seed,
                best_checkpoint_sha256=str(best_checkpoint_record["sha256"]),
            )
            _require_execution_identity_unchanged(execution_identity, device)
            development_artifacts, development_bundle_record = _fit_and_publish_development_bundle(
                run_directory,
                model,
                train,
                validation,
                batch_size=batch_size,
                latent_dimensions=int(config["latent_dimensions"]),
                device=device,
                binding=binding,
                final_evaluation_contract=final_evaluation_contract,
                best_checkpoint_record=best_checkpoint_record,
            )
            _require_execution_identity_unchanged(execution_identity, device)
            artifact_records["development"] = development_artifacts
            artifact_records["development_bundle"] = development_bundle_record
            result = {
                "schema_version": RUN_SCHEMA_VERSION,
                "complete": True,
                "started_at_utc": run_started_at_utc,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "run_id": selected_run_id,
                "run_directory": str(run_directory),
                "run_identity_sha256": run_identity_sha256,
                "config_sha256": config_sha256,
                "config_file_sha256": config_file_sha256,
                "cache_lock_sha256": train.lock_sha256,
                "release_source_fingerprint_sha256": release_source_fingerprint_sha256,
                "implementation_sha256": execution_identity.implementation_sha256,
                "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                "model_contract_sha256": model_contract_sha256,
                "scope": scope,
                "production_evidence": production_evidence,
                "resumed": resumed,
                "resume_checkpoint": resume_record,
                "epochs_completed": len(history),
                "early_stopped": stop_requested,
                "best_epoch": early_stopping.best.epoch,
                "best_validation_loss": early_stopping.best.validation_loss,
                "best_checkpoint": best_checkpoint_record,
                "latest_recovery_checkpoint": latest_recovery_record,
                "development_bundle": development_bundle_record,
                "artifacts": artifact_records,
            }
            _require_execution_identity_unchanged(execution_identity, device)
            result_record = _write_or_verify_json(run_directory / "result.json", result)
            _require_execution_identity_unchanged(execution_identity, device)
            completion_record = _write_or_verify_json(
                run_directory / "result.lock.json",
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "run_identity_sha256": run_identity_sha256,
                    "implementation_sha256": execution_identity.implementation_sha256,
                    "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                    "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                    "scope": scope,
                    "production_evidence": production_evidence,
                    "result": result_record,
                    "development_bundle": development_bundle_record,
                },
            )
            _require_execution_identity_unchanged(execution_identity, device)
            verify_task2_development_run(
                completion_record["path"],
                expected_sha256=completion_record["sha256"],
                require_production=production_evidence,
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
                "scope": scope,
                "production_evidence": production_evidence,
                "implementation_sha256": execution_identity.implementation_sha256,
                "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "epochs_completed": len(history),
                "history": history,
                "traceback": traceback.format_exc(),
            }
            _write_json_create_only(_failure_path(run_directory), diagnostic)
            raise


def _warmup_task2(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Task2Data,
    *,
    batch_size: int,
    latent_dimensions: int,
    device: torch.device,
) -> None:
    sample_count = min(batch_size, len(data))
    native = collate_native_samples([data[index] for index in range(sample_count)])
    inputs = to_autoencoder_batch(native.tensor).to(device=device, dtype=torch.float32)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction, _ = _model_outputs(
        model,
        inputs,
        latent_dimensions=latent_dimensions,
    )
    loss = functional.mse_loss(reconstruction, inputs, reduction="mean")
    loss.backward()
    optimizer.step()
    _synchronize(device)


def _benchmark_artifact_paths(
    test_injection: Task2TestInjection | None,
    requested_output: str | Path | None,
) -> tuple[Path, Path] | None:
    if test_injection is None:
        if (
            requested_output is not None
            and require_safe_output(requested_output) != DEFAULT_BENCHMARK_RESULT_PATH.resolve()
        ):
            raise PermissionError("Production Task 2 benchmark output is canonical and versioned")
        return DEFAULT_BENCHMARK_RESULT_PATH.resolve(), DEFAULT_BENCHMARK_LOCK_PATH.resolve()
    if requested_output is None:
        return None
    try:
        result_path = _validated_run_output_path(requested_output)
    except ValueError as exc:
        raise PermissionError(
            "Isolated Task 2 benchmark evidence must stay inside runs/task2_v2"
        ) from exc
    if result_path in {
        DEFAULT_BENCHMARK_RESULT_PATH.resolve(),
        DEFAULT_BENCHMARK_LOCK_PATH.resolve(),
    }:
        raise PermissionError("Isolated Task 2 benchmark cannot publish production evidence")
    if result_path.suffix != ".json" or result_path.name.endswith(".lock.json"):
        raise ValueError("Isolated Task 2 benchmark output must be a non-lock JSON path")
    return result_path, result_path.with_name(f"{result_path.stem}.lock.json")


def _validate_benchmark_result(
    value: Any,
    *,
    expected_identity: Mapping[str, Any],
) -> None:
    expected_identity_fields = {
        "schema_version",
        "task",
        "seed",
        "config_file_sha256",
        "config_sha256",
        "cache_lock_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "model_contract_sha256",
        "scope",
        "production_evidence",
    }
    if (
        not isinstance(expected_identity, Mapping)
        or set(expected_identity) != expected_identity_fields
        or expected_identity["schema_version"] != RUN_SCHEMA_VERSION
        or expected_identity["task"] != "task2_full_epoch_benchmark"
        or expected_identity["scope"] not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or type(expected_identity["production_evidence"]) is not bool
        or expected_identity["production_evidence"]
        is not (expected_identity["scope"] == PRODUCTION_SCOPE)
    ):
        raise ValueError("Task 2 benchmark expected identity is invalid")
    required = {
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
        "release_source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "numerical_runtime",
        "model_contract_sha256",
        "model_contract",
        "optimizer_contract",
        "final_evaluation_contract",
        "warmup_completed",
        "measured_train_epochs",
        "measured_validation_epochs",
        "train_clips",
        "train_recordings",
        "validation_clips",
        "validation_recordings",
        "train_seconds",
        "validation_seconds",
        "full_epoch_compute_seconds",
        "checkpoint_cpu_copy_seconds",
        "rng_capture_seconds",
        "best_checkpoint_serialization_seconds",
        "recovery_checkpoint_serialization_seconds",
        "best_checkpoint_representative_bytes",
        "recovery_checkpoint_representative_bytes",
        "estimated_epoch_with_worst_case_checkpoint_seconds",
        "train_clips_per_second",
        "validation_clips_per_second",
        "estimated_one_seed_maximum_epoch_seconds",
        "estimated_all_seed_maximum_epoch_seconds",
        "conservative_wall_time_factor",
        "conservative_wall_time_scope",
        "estimated_one_seed_conservative_wall_seconds",
        "estimated_all_seed_conservative_wall_seconds",
        "train_loss",
        "validation_loss",
        "validation_pixel_count",
        "evidence_scope",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema_version"] != RUN_SCHEMA_VERSION
        or value["benchmark_only"] is not True
        or value["persistent_model_selection"] is not False
        or value["persistent_model_checkpoint"] is not False
        or type(value["durable_evidence"]) is not bool
        or value["benchmark_identity"] != dict(expected_identity)
        or value["benchmark_identity_sha256"] != sha256_json(dict(expected_identity))
        or value["scope"] != expected_identity["scope"]
        or value["production_evidence"] is not expected_identity["production_evidence"]
        or value["implementation_sha256"] != expected_identity["implementation_sha256"]
        or value["requirements_lock_sha256"] != expected_identity["requirements_lock_sha256"]
        or value["numerical_runtime_sha256"] != expected_identity["numerical_runtime_sha256"]
        or value["config_file_sha256"] != expected_identity["config_file_sha256"]
        or value["config_sha256"] != expected_identity["config_sha256"]
        or value["cache_lock_sha256"] != expected_identity["cache_lock_sha256"]
        or value["model_contract_sha256"] != expected_identity["model_contract_sha256"]
        or value["seed"] != expected_identity["seed"]
        or value["scope"] not in {PRODUCTION_SCOPE, ISOLATED_TEST_SCOPE}
        or value["production_evidence"] is not (value["scope"] == PRODUCTION_SCOPE)
        or not isinstance(value["command"], list)
        or any(not isinstance(part, str) for part in value["command"])
    ):
        raise ValueError("Task 2 benchmark result identity is invalid")
    for name in (
        "benchmark_identity_sha256",
        "config_file_sha256",
        "config_sha256",
        "cache_lock_sha256",
        "release_source_fingerprint_sha256",
        "implementation_sha256",
        "requirements_lock_sha256",
        "numerical_runtime_sha256",
        "model_contract_sha256",
    ):
        if not isinstance(value[name], str) or _SHA256.fullmatch(value[name]) is None:
            raise ValueError(f"Task 2 benchmark {name} is invalid")
    _validated_timestamp(value["started_at_utc"], "benchmark start timestamp")
    _validated_timestamp(value["completed_at_utc"], "benchmark completion timestamp")
    if datetime.fromisoformat(value["completed_at_utc"]) < datetime.fromisoformat(
        value["started_at_utc"]
    ):
        raise ValueError("Task 2 benchmark completion precedes its start")
    positive_floats = (
        "train_seconds",
        "validation_seconds",
        "full_epoch_compute_seconds",
        "checkpoint_cpu_copy_seconds",
        "rng_capture_seconds",
        "best_checkpoint_serialization_seconds",
        "recovery_checkpoint_serialization_seconds",
        "estimated_epoch_with_worst_case_checkpoint_seconds",
        "train_clips_per_second",
        "validation_clips_per_second",
        "estimated_one_seed_maximum_epoch_seconds",
        "estimated_all_seed_maximum_epoch_seconds",
        "estimated_one_seed_conservative_wall_seconds",
        "estimated_all_seed_conservative_wall_seconds",
    )
    if any(
        type(value[name]) is not float or not math.isfinite(value[name]) or value[name] <= 0.0
        for name in positive_floats
    ):
        raise ValueError("Task 2 benchmark timing values are invalid")
    if (
        type(value["train_loss"]) is not float
        or not math.isfinite(value["train_loss"])
        or value["train_loss"] < 0.0
        or type(value["validation_loss"]) is not float
        or not math.isfinite(value["validation_loss"])
        or value["validation_loss"] < 0.0
    ):
        raise ValueError("Task 2 benchmark losses are invalid")
    config = load_locked_task2_config()
    integer_fields = (
        "batch_size",
        "maximum_epochs",
        "train_clips",
        "train_recordings",
        "validation_clips",
        "validation_recordings",
        "validation_pixel_count",
        "best_checkpoint_representative_bytes",
        "recovery_checkpoint_representative_bytes",
    )
    if (
        any(type(value[name]) is not int or value[name] <= 0 for name in integer_fields)
        or type(value["seed"]) is not int
        or value["seed"] not in config["seeds"]
        or value["stability_seeds"] != config["seeds"]
        or value["config_path"] != LOCKED_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
        or not isinstance(value["cache_root"], str)
        or not value["cache_root"]
        or value["warmup_completed"] is not True
        or value["measured_train_epochs"] != 1
        or value["measured_validation_epochs"] != 1
        or value["validation_pixel_count"] != value["validation_clips"] * 224 * 224
        or value["conservative_wall_time_factor"] != CONSERVATIVE_WALL_TIME_FACTOR
        or not isinstance(value["numerical_runtime"], dict)
        or sha256_json(value["numerical_runtime"]) != value["numerical_runtime_sha256"]
        or value["numerical_runtime"].get("device") != value["device"]
        or not isinstance(value["model_contract"], dict)
        or sha256_json(value["model_contract"]) != value["model_contract_sha256"]
        or value["optimizer_contract"] != _optimizer_contract(config)
        or value["final_evaluation_contract"] != _final_evaluation_contract(config)
        or not isinstance(value["conservative_wall_time_scope"], str)
        or not value["conservative_wall_time_scope"]
        or not isinstance(value["evidence_scope"], str)
        or not value["evidence_scope"]
    ):
        raise ValueError("Task 2 benchmark method or count fields are invalid")
    expected_compute_seconds = value["train_seconds"] + value["validation_seconds"]
    expected_checkpoint_seconds = (
        expected_compute_seconds
        + value["checkpoint_cpu_copy_seconds"]
        + value["rng_capture_seconds"]
        + value["best_checkpoint_serialization_seconds"]
        + value["recovery_checkpoint_serialization_seconds"]
    )
    expected_one_seed = expected_checkpoint_seconds * value["maximum_epochs"]
    expected_all_seeds = expected_one_seed * len(value["stability_seeds"])
    formulas = (
        (value["full_epoch_compute_seconds"], expected_compute_seconds),
        (value["estimated_epoch_with_worst_case_checkpoint_seconds"], expected_checkpoint_seconds),
        (value["train_clips_per_second"], value["train_clips"] / value["train_seconds"]),
        (
            value["validation_clips_per_second"],
            value["validation_clips"] / value["validation_seconds"],
        ),
        (value["estimated_one_seed_maximum_epoch_seconds"], expected_one_seed),
        (value["estimated_all_seed_maximum_epoch_seconds"], expected_all_seeds),
        (
            value["estimated_one_seed_conservative_wall_seconds"],
            expected_one_seed * CONSERVATIVE_WALL_TIME_FACTOR,
        ),
        (
            value["estimated_all_seed_conservative_wall_seconds"],
            expected_all_seeds * CONSERVATIVE_WALL_TIME_FACTOR,
        ),
    )
    if any(
        not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=0.0)
        for observed, expected in formulas
    ):
        raise ValueError("Task 2 benchmark derived timings are inconsistent")
    if value["scope"] == PRODUCTION_SCOPE:
        if (
            value["device"] != "mps"
            or value["cache_lock_sha256"] != KNOWN_CACHE_LOCK_SHA256
            or Path(value["cache_root"]).resolve() != DEFAULT_CACHE_ROOT.resolve()
            or value["batch_size"] != int(config["training"]["batch_size"])
            or value["maximum_epochs"] != int(config["training"]["maximum_epochs"])
            or value["train_clips"] != 5_319
            or value["train_recordings"] != 1_254
            or value["validation_clips"] != 1_138
            or value["validation_recordings"] != 271
            or value["model_contract"].get("parameter_counts")
            != {"total": EXPECTED_PARAMETER_COUNT, "trainable": EXPECTED_PARAMETER_COUNT}
        ):
            raise ValueError("Production Task 2 benchmark identity or counts changed")
    elif value["device"] != "cpu":
        raise ValueError("Isolated Task 2 benchmark must use CPU")


def _recover_task2_benchmark_evidence(
    result_path: Path,
    lock_path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if lock_path.exists() and not result_path.exists():
        raise ValueError("Task 2 benchmark lock exists without its result")
    result, result_record = _read_json_snapshot(result_path)
    _validate_benchmark_result(result, expected_identity=expected_identity)
    if result["durable_evidence"] is not True:
        raise ValueError("Recoverable Task 2 benchmark result is not durable evidence")
    expected_lock = {
        "schema_version": RUN_SCHEMA_VERSION,
        "benchmark_identity_sha256": sha256_json(dict(expected_identity)),
        "implementation_sha256": expected_identity["implementation_sha256"],
        "requirements_lock_sha256": expected_identity["requirements_lock_sha256"],
        "numerical_runtime_sha256": expected_identity["numerical_runtime_sha256"],
        "scope": expected_identity["scope"],
        "production_evidence": expected_identity["production_evidence"],
        "result": result_record,
    }
    lock_record = _write_or_verify_json(lock_path, expected_lock)
    return {
        **result,
        "result_artifact": result_record,
        "completion_lock_artifact": lock_record,
        "recovered_existing_evidence": True,
    }


def benchmark_task2_full_epoch(
    *,
    seed: int = 13,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ffmpeg: str | Path | None = None,
    expected_lock_sha256: str | None = None,
    command: Sequence[str] = (),
    evidence_output: str | Path | None = None,
    train_data: Task2Data | None = None,
    validation_data: Task2Data | None = None,
    test_injection: Task2TestInjection | None = None,
) -> dict[str, Any]:
    started_at_utc = datetime.now(UTC).isoformat()
    config = load_locked_task2_config()
    if type(seed) is not int or seed not in config["seeds"]:
        raise ValueError("Task 2 benchmark seed is outside the locked seed set")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("Task 2 benchmark command must be a sequence")
    device = _resolve_runtime(test_injection)
    scope = ISOLATED_TEST_SCOPE if test_injection is not None else PRODUCTION_SCOPE
    production_evidence = scope == PRODUCTION_SCOPE
    artifact_paths = _benchmark_artifact_paths(test_injection, evidence_output)
    execution_identity = _capture_execution_identity(device)
    release_source_fingerprint_sha256 = source_fingerprint()
    with ExitStack() as stack:
        if (train_data is None) != (validation_data is None):
            raise ValueError("Task 2 benchmark data must be supplied together")
        if train_data is None or validation_data is None:
            if test_injection is not None:
                raise ValueError("Task 2 CPU benchmark requires explicit fixtures")
            train, validation = _open_real_data(
                stack,
                cache_root=cache_root,
                ffmpeg=ffmpeg,
                expected_lock_sha256=expected_lock_sha256,
            )
        else:
            if test_injection is None:
                raise PermissionError("Task 2 benchmark data injection is allowed only for tests")
            train, validation = train_data, validation_data
        _validate_development_data(train, validation, production=production_evidence)
        _require_execution_identity_unchanged(execution_identity, device)
        maximum_epochs, batch_size, _ = _resolved_limits(config, test_injection)
        latent_dimensions = int(config["latent_dimensions"])
        config_sha256 = config_fingerprint(config)
        _, config_file_sha256, _ = _descriptor_snapshot(LOCKED_CONFIG_PATH)

        seed_task2(seed, device)
        identity_model = _build_task2_model(config, device, test_injection)
        model_contract = _model_contract(identity_model, config)
        model_contract_sha256 = sha256_json(model_contract)
        del identity_model
        benchmark_identity = {
            "schema_version": RUN_SCHEMA_VERSION,
            "task": "task2_full_epoch_benchmark",
            "seed": seed,
            "config_file_sha256": config_file_sha256,
            "config_sha256": config_sha256,
            "cache_lock_sha256": train.lock_sha256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "model_contract_sha256": model_contract_sha256,
            "scope": scope,
            "production_evidence": production_evidence,
        }
        if artifact_paths is not None:
            result_path, lock_path = artifact_paths
            result_exists = result_path.exists() or result_path.is_symlink()
            lock_exists = lock_path.exists() or lock_path.is_symlink()
            if result_exists:
                return _recover_task2_benchmark_evidence(
                    result_path,
                    lock_path,
                    expected_identity=benchmark_identity,
                )
            if lock_exists:
                raise ValueError("Task 2 benchmark lock exists without its result")

        seed_task2(seed, device)
        warmup_model = _build_task2_model(config, device, test_injection)
        warmup_optimizer = build_task2_optimizer(warmup_model, config)
        _warmup_task2(
            warmup_model,
            warmup_optimizer,
            train,
            batch_size=batch_size,
            latent_dimensions=latent_dimensions,
            device=device,
        )
        del warmup_optimizer, warmup_model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        seed_task2(seed, device)
        model = _build_task2_model(config, device, test_injection)
        optimizer = build_task2_optimizer(model, config)
        _synchronize(device)
        train_started = time.perf_counter()
        train_metrics = train_task2_epoch(
            model,
            optimizer,
            train,
            seed=seed,
            epoch_index=0,
            batch_size=batch_size,
            latent_dimensions=latent_dimensions,
            device=device,
        )
        _synchronize(device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation_metrics = validate_task2(
            model,
            validation,
            batch_size=batch_size,
            latent_dimensions=latent_dimensions,
            device=device,
        )
        _synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        full_epoch_compute_seconds = train_seconds + validation_seconds

        checkpoint_copy_started = time.perf_counter()
        representative_model_state = _cpu_copy(model.state_dict())
        representative_optimizer_state = _cpu_copy(optimizer.state_dict())
        checkpoint_cpu_copy_seconds = time.perf_counter() - checkpoint_copy_started
        rng_started = time.perf_counter()
        representative_rng_state = _capture_rng_state(device)
        rng_capture_seconds = time.perf_counter() - rng_started
        best_buffer = io.BytesIO()
        best_serialization_started = time.perf_counter()
        torch.save(
            {
                "model": representative_model_state,
                "optimizer": representative_optimizer_state,
                "score": {"validation_loss": validation_metrics.loss, "epoch": 1},
            },
            best_buffer,
        )
        best_checkpoint_serialization_seconds = time.perf_counter() - best_serialization_started
        recovery_buffer = io.BytesIO()
        recovery_serialization_started = time.perf_counter()
        torch.save(
            {
                "model": representative_model_state,
                "optimizer": representative_optimizer_state,
                "rng_state": representative_rng_state,
                "history": [
                    {
                        "epoch": 1,
                        "train": train_metrics,
                        "validation_loss": validation_metrics.loss,
                    }
                ],
            },
            recovery_buffer,
        )
        recovery_checkpoint_serialization_seconds = (
            time.perf_counter() - recovery_serialization_started
        )
        best_checkpoint_representative_bytes = best_buffer.tell()
        recovery_checkpoint_representative_bytes = recovery_buffer.tell()
        del (
            best_buffer,
            recovery_buffer,
            representative_model_state,
            representative_optimizer_state,
            representative_rng_state,
        )
        estimated_epoch_with_worst_case_checkpoint_seconds = (
            full_epoch_compute_seconds
            + checkpoint_cpu_copy_seconds
            + rng_capture_seconds
            + best_checkpoint_serialization_seconds
            + recovery_checkpoint_serialization_seconds
        )
        one_seed_maximum_seconds = (
            estimated_epoch_with_worst_case_checkpoint_seconds * maximum_epochs
        )
        all_seed_maximum_seconds = one_seed_maximum_seconds * len(config["seeds"])
        _require_execution_identity_unchanged(execution_identity, device)
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
            "benchmark_identity_sha256": sha256_json(benchmark_identity),
            "benchmark_identity": benchmark_identity,
            "seed": seed,
            "device": device.type,
            "batch_size": batch_size,
            "maximum_epochs": maximum_epochs,
            "stability_seeds": list(config["seeds"]),
            "config_path": LOCKED_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "config_file_sha256": config_file_sha256,
            "config_sha256": config_sha256,
            "cache_root": str(train.root),
            "cache_lock_sha256": train.lock_sha256,
            "release_source_fingerprint_sha256": release_source_fingerprint_sha256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "numerical_runtime": execution_identity.numerical_runtime,
            "model_contract_sha256": model_contract_sha256,
            "model_contract": model_contract,
            "optimizer_contract": _optimizer_contract(config),
            "final_evaluation_contract": _final_evaluation_contract(config),
            "warmup_completed": True,
            "measured_train_epochs": 1,
            "measured_validation_epochs": 1,
            "train_clips": len(train),
            "train_recordings": train.recording_count,
            "validation_clips": len(validation),
            "validation_recordings": validation.recording_count,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "full_epoch_compute_seconds": full_epoch_compute_seconds,
            "checkpoint_cpu_copy_seconds": checkpoint_cpu_copy_seconds,
            "rng_capture_seconds": rng_capture_seconds,
            "best_checkpoint_serialization_seconds": best_checkpoint_serialization_seconds,
            "recovery_checkpoint_serialization_seconds": (
                recovery_checkpoint_serialization_seconds
            ),
            "best_checkpoint_representative_bytes": best_checkpoint_representative_bytes,
            "recovery_checkpoint_representative_bytes": recovery_checkpoint_representative_bytes,
            "estimated_epoch_with_worst_case_checkpoint_seconds": (
                estimated_epoch_with_worst_case_checkpoint_seconds
            ),
            "train_clips_per_second": len(train) / train_seconds,
            "validation_clips_per_second": len(validation) / validation_seconds,
            "estimated_one_seed_maximum_epoch_seconds": one_seed_maximum_seconds,
            "estimated_all_seed_maximum_epoch_seconds": all_seed_maximum_seconds,
            "conservative_wall_time_factor": CONSERVATIVE_WALL_TIME_FACTOR,
            "conservative_wall_time_scope": (
                "Measured epoch compute, checkpoint CPU copy, RNG capture, and two "
                "checkpoint serializations, plus a 25 percent allowance for durable writes, "
                "development scoring, startup, and thermal variation"
            ),
            "estimated_one_seed_conservative_wall_seconds": (
                one_seed_maximum_seconds * CONSERVATIVE_WALL_TIME_FACTOR
            ),
            "estimated_all_seed_conservative_wall_seconds": (
                all_seed_maximum_seconds * CONSERVATIVE_WALL_TIME_FACTOR
            ),
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics.loss,
            "validation_pixel_count": validation_metrics.pixel_count,
            "evidence_scope": (
                "Runtime planning evidence only; no model checkpoint or selection decision is retained"
            ),
        }
        _validate_benchmark_result(result, expected_identity=benchmark_identity)
        if artifact_paths is None:
            return result
        result_path, lock_path = artifact_paths
        _require_execution_identity_unchanged(execution_identity, device)
        result_record = _write_json_create_only(result_path, result)
        lock_value = {
            "schema_version": RUN_SCHEMA_VERSION,
            "benchmark_identity_sha256": sha256_json(benchmark_identity),
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "scope": scope,
            "production_evidence": production_evidence,
            "result": result_record,
        }
        _require_execution_identity_unchanged(execution_identity, device)
        lock_record = _write_json_create_only(lock_path, lock_value)
        return {
            **result,
            "result_artifact": result_record,
            "completion_lock_artifact": lock_record,
            "recovered_existing_evidence": False,
        }
