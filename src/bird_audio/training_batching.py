from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import Sampler

from bird_audio.signal import (
    IMAGENET_MEAN,
    IMAGENET_STANDARD_DEVIATION,
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    NATIVE_MEL_HEIGHT,
    NATIVE_MEL_WIDTH,
)

FREQUENCY_MASK_MAX_BINS = 16
TIME_MASK_MAX_FRAMES = 40
FREQUENCY_MASK_PROBABILITY = 0.5
TIME_MASK_PROBABILITY = 0.5
MASK_FILL_VALUE = 0.0

SAMPLER_RANDOM_STREAM = "recording_balanced_sampler"
SPECAUGMENT_RANDOM_STREAM = "specaugment"
MAXIMUM_GENERATOR_SEED = 2**63 - 1


class DevelopmentMetadataSource(Protocol):
    split: str
    strategy: str

    def __len__(self) -> int: ...

    def iter_metadata(self) -> Iterator[dict[str, str]]: ...


@dataclass(frozen=True)
class NativeBatch:
    tensor: torch.Tensor
    metadata: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class SpecAugmentMask:
    frequency_applied: bool
    frequency_start: int
    frequency_width: int
    time_applied: bool
    time_start: int
    time_width: int


def _require_seed(value: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if not 0 <= value <= MAXIMUM_GENERATOR_SEED:
        raise ValueError(f"{context} must be in [0, {MAXIMUM_GENERATOR_SEED}]")
    return value


def epoch_generator_seed(base_seed: int, epoch: int, stream: str) -> int:
    """Derive one stable explicit CPU generator seed for a named epoch stream."""
    base_seed = _require_seed(base_seed, "base_seed")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if not isinstance(stream, str):
        raise TypeError("stream must be a string")
    if not stream or stream.strip() != stream:
        raise ValueError("stream must be a nonempty canonical name")
    payload = f"bird_audio_v1\n{stream}\n{base_seed}\n{epoch}\n".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & MAXIMUM_GENERATOR_SEED


def make_epoch_cpu_generator(base_seed: int, epoch: int, stream: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(epoch_generator_seed(base_seed, epoch, stream))
    return generator


def _metadata_rows(source: DevelopmentMetadataSource) -> tuple[dict[str, str], ...]:
    rows = tuple(source.iter_metadata())
    if len(rows) != len(source):
        raise ValueError("Development metadata count differs from the selected clip count")
    if not rows:
        raise ValueError("Recording-balanced sampling requires at least one selected clip")
    return rows


def recording_balanced_weights(source: DevelopmentMetadataSource) -> torch.Tensor:
    """Return exact per-clip Task 1 weights in deterministic source order."""
    if source.split != "train":
        raise PermissionError("Recording-balanced sampling accepts only the training split")
    rows = _metadata_rows(source)
    recordings: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        if row.get("split") != "train":
            raise ValueError("Development metadata contains a non-training row")
        if row.get("selection_strategy") != source.strategy:
            raise ValueError("Development metadata selection strategy is inconsistent")
        recording_id = row.get("recording_id", "")
        species = row.get("species_common_name", "")
        class_index = row.get("class_index", "")
        session_group = row.get("session_group", "")
        if not recording_id or not species or not class_index or not session_group:
            raise ValueError("Development metadata has an incomplete recording identity")
        try:
            declared_clip_count = int(row.get("strategy_clip_count", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid selected clip count for {recording_id}") from exc
        if declared_clip_count <= 0:
            raise ValueError(f"Invalid selected clip count for {recording_id}")

        record = recordings.setdefault(
            recording_id,
            {
                "species": species,
                "class_index": class_index,
                "session_group": session_group,
                "declared_clip_count": declared_clip_count,
                "indices": [],
            },
        )
        expected_identity = (record["species"], record["class_index"], record["session_group"])
        if expected_identity != (species, class_index, session_group):
            raise ValueError(f"Recording identity changes across clips: {recording_id}")
        if record["declared_clip_count"] != declared_clip_count:
            raise ValueError(f"Selected clip count changes within recording: {recording_id}")
        indices = record["indices"]
        if not isinstance(indices, list):
            raise RuntimeError("Internal recording index state is invalid")
        indices.append(index)

    species_recordings = Counter(str(record["species"]) for record in recordings.values())
    weights = torch.empty(len(rows), dtype=torch.float64, device="cpu")
    for recording_id, record in recordings.items():
        indices = record["indices"]
        if not isinstance(indices, list) or not indices:
            raise RuntimeError("Internal recording membership state is invalid")
        selected_clip_count = len(indices)
        if record["declared_clip_count"] != selected_clip_count:
            raise ValueError(f"Selected clip count differs from index membership: {recording_id}")
        if indices != list(range(indices[0], indices[0] + selected_clip_count)):
            raise ValueError(f"Recording clips are not contiguous in source order: {recording_id}")
        species = str(record["species"])
        recordings_in_species = species_recordings[species]
        weight = 1.0 / (recordings_in_species * selected_clip_count)
        weights[indices] = weight

    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights <= 0)):
        raise RuntimeError("Recording-balanced weights are not finite and positive")
    return weights


class RecordingBalancedEpochSampler(Sampler[int]):
    """Deterministic replacement sampler with an explicit epoch seed."""

    def __init__(self, source: DevelopmentMetadataSource, *, base_seed: int) -> None:
        self.base_seed = _require_seed(base_seed, "base_seed")
        self._weights = recording_balanced_weights(source)
        self.draws_per_epoch = len(source)
        self.epoch = 0

    @property
    def weights(self) -> torch.Tensor:
        return self._weights.clone()

    @property
    def generator_seed(self) -> int:
        return epoch_generator_seed(self.base_seed, self.epoch, SAMPLER_RANDOM_STREAM)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.draws_per_epoch

    def __iter__(self) -> Iterator[int]:
        generator = make_epoch_cpu_generator(
            self.base_seed,
            self.epoch,
            SAMPLER_RANDOM_STREAM,
        )
        indices = torch.multinomial(
            self._weights,
            num_samples=self.draws_per_epoch,
            replacement=True,
            generator=generator,
        )
        return iter(indices.tolist())


def _require_native_tensor(batch: torch.Tensor, *, require_cpu: bool) -> None:
    if not isinstance(batch, torch.Tensor):
        raise TypeError("native batch must be a torch.Tensor")
    if batch.dtype != torch.float32:
        raise TypeError("native batch must use torch.float32")
    if batch.ndim != 4 or tuple(batch.shape[1:]) != (
        1,
        NATIVE_MEL_HEIGHT,
        NATIVE_MEL_WIDTH,
    ):
        raise ValueError("native batch must have shape [batch, 1, 128, 372]")
    if batch.shape[0] <= 0:
        raise ValueError("native batch cannot be empty")
    if require_cpu and batch.device.type != "cpu":
        raise ValueError("SpecAugment requires a CPU native batch")
    if not bool(torch.isfinite(batch).all().item()):
        raise ValueError("native batch contains non-finite values")
    if bool(torch.any(batch < 0).item()) or bool(torch.any(batch > 1).item()):
        raise ValueError("native batch values must lie in [0, 1]")


def collate_native_samples(
    samples: Sequence[tuple[np.ndarray, Mapping[str, str]]],
) -> NativeBatch:
    """Collate native NumPy features while preserving exact metadata order."""
    if not samples:
        raise ValueError("Cannot collate an empty native batch")
    arrays: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    for feature, item_metadata in samples:
        if not isinstance(feature, np.ndarray):
            raise TypeError("Development feature must be a NumPy array")
        if (
            feature.dtype != np.float32
            or feature.shape != (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH)
            or not bool(np.all(np.isfinite(feature)))
            or float(feature.min()) < 0.0
            or float(feature.max()) > 1.0
        ):
            raise ValueError("Development feature violates the native signal contract")
        if not isinstance(item_metadata, Mapping):
            raise TypeError("Development metadata must be a mapping")
        arrays.append(feature)
        metadata.append({str(key): str(value) for key, value in item_metadata.items()})
    stacked = np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float32)
    tensor = torch.from_numpy(stacked)
    _require_native_tensor(tensor, require_cpu=True)
    return NativeBatch(tensor=tensor, metadata=tuple(metadata))


def _cpu_uniform(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator, device="cpu").item())


def _cpu_integer(generator: torch.Generator, high_exclusive: int) -> int:
    if high_exclusive <= 0:
        raise ValueError("Random integer upper bound must be positive")
    return int(torch.randint(high_exclusive, (), generator=generator, device="cpu").item())


def _sample_mask(
    generator: torch.Generator,
    *,
    probability: float,
    maximum_width: int,
    dimension: int,
) -> tuple[bool, int, int]:
    applied = _cpu_uniform(generator) < probability
    if not applied:
        return False, 0, 0
    width = _cpu_integer(generator, maximum_width + 1)
    start = 0 if width == 0 else _cpu_integer(generator, dimension - width + 1)
    return True, start, width


def sample_specaugment_plan(
    batch_size: int,
    *,
    generator: torch.Generator,
) -> tuple[SpecAugmentMask, ...]:
    """Sample the locked independent mask decisions from an explicit CPU generator."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("SpecAugment requires an explicit CPU generator")
    decisions: list[SpecAugmentMask] = []
    for _ in range(batch_size):
        frequency_applied, frequency_start, frequency_width = _sample_mask(
            generator,
            probability=FREQUENCY_MASK_PROBABILITY,
            maximum_width=FREQUENCY_MASK_MAX_BINS,
            dimension=NATIVE_MEL_HEIGHT,
        )
        time_applied, time_start, time_width = _sample_mask(
            generator,
            probability=TIME_MASK_PROBABILITY,
            maximum_width=TIME_MASK_MAX_FRAMES,
            dimension=NATIVE_MEL_WIDTH,
        )
        decisions.append(
            SpecAugmentMask(
                frequency_applied=frequency_applied,
                frequency_start=frequency_start,
                frequency_width=frequency_width,
                time_applied=time_applied,
                time_start=time_start,
                time_width=time_width,
            )
        )
    return tuple(decisions)


def _validate_mask(mask: SpecAugmentMask) -> None:
    if not isinstance(mask, SpecAugmentMask):
        raise TypeError("SpecAugment plan contains an invalid mask record")
    limits = (
        (
            mask.frequency_applied,
            mask.frequency_start,
            mask.frequency_width,
            FREQUENCY_MASK_MAX_BINS,
            NATIVE_MEL_HEIGHT,
        ),
        (
            mask.time_applied,
            mask.time_start,
            mask.time_width,
            TIME_MASK_MAX_FRAMES,
            NATIVE_MEL_WIDTH,
        ),
    )
    for applied, start, width, maximum_width, dimension in limits:
        if not isinstance(applied, bool):
            raise TypeError("SpecAugment applied flags must be Boolean")
        if not applied and (start != 0 or width != 0):
            raise ValueError("An unapplied SpecAugment mask must have zero span")
        if (
            isinstance(start, bool)
            or isinstance(width, bool)
            or not isinstance(start, int)
            or not isinstance(width, int)
            or not 0 <= width <= maximum_width
            or not 0 <= start <= dimension - width
        ):
            raise ValueError("SpecAugment mask span is outside the locked bounds")


def apply_specaugment_plan(
    batch: torch.Tensor,
    plan: Sequence[SpecAugmentMask],
) -> torch.Tensor:
    """Apply a sampled locked plan to a copied CPU native batch."""
    _require_native_tensor(batch, require_cpu=True)
    if len(plan) != batch.shape[0]:
        raise ValueError("SpecAugment plan length must match the native batch size")
    with torch.no_grad():
        augmented = batch.detach().clone()
        for index, mask in enumerate(plan):
            _validate_mask(mask)
            if mask.frequency_applied and mask.frequency_width:
                start = mask.frequency_start
                augmented[index, :, start : start + mask.frequency_width, :] = MASK_FILL_VALUE
            if mask.time_applied and mask.time_width:
                start = mask.time_start
                augmented[index, :, :, start : start + mask.time_width] = MASK_FILL_VALUE
    return augmented.contiguous()


def apply_locked_specaugment(
    batch: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    _require_native_tensor(batch, require_cpu=True)
    plan = sample_specaugment_plan(batch.shape[0], generator=generator)
    return apply_specaugment_plan(batch, plan)


def resize_native_batch(batch: torch.Tensor) -> torch.Tensor:
    """Resize a complete native batch with the locked no-grad transform."""
    _require_native_tensor(batch, require_cpu=False)
    with torch.no_grad():
        resized = functional.interpolate(
            batch.detach(),
            size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        resized = resized.clamp(0.0, 1.0).contiguous()
    if resized.dtype != torch.float32 or tuple(resized.shape[1:]) != (1, 224, 224):
        raise RuntimeError("Resized batch violates the locked model input contract")
    return resized


def to_efficientnet_batch(batch: torch.Tensor) -> torch.Tensor:
    """Create batched replicated and ImageNet-normalized classifier input."""
    resized = resize_native_batch(batch)
    with torch.no_grad():
        replicated = resized.expand(-1, 3, -1, -1).clone()
        mean = replicated.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        standard_deviation = replicated.new_tensor(IMAGENET_STANDARD_DEVIATION).view(1, 3, 1, 1)
        normalized = ((replicated - mean) / standard_deviation).contiguous()
    if normalized.dtype != torch.float32 or tuple(normalized.shape[1:]) != (3, 224, 224):
        raise RuntimeError("EfficientNet batch violates the locked input contract")
    return normalized


def to_autoencoder_batch(batch: torch.Tensor) -> torch.Tensor:
    """Create batched one-channel autoencoder input in [0, 1]."""
    return resize_native_batch(batch)
