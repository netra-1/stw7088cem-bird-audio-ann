from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from bird_audio.config import LOCKED_TASK1_CLASS_ORDER
from bird_audio.final_evaluation_data import (
    KNOWN_CACHE_LOCK_SHA256,
    KNOWN_TEST_ENERGY_CLIPS,
    KNOWN_TEST_RECORDINGS,
    UNKNOWN_CACHE_LOCK_SHA256,
    UNKNOWN_ENERGY_CLIPS,
    UNKNOWN_RECORDINGS,
    UNKNOWN_RECORDINGS_PER_SPECIES,
    UNKNOWN_SPECIES,
    FinalEvaluationAuthorization,
    FinalKnownTestData,
    FinalUnknownData,
    open_final_known_test_data,
    open_final_unknown_data,
)
from bird_audio.task1_final_metrics import RecordingPrediction
from bird_audio.task2_scoring import (
    ClipIdentity,
    RecordingBatch,
    aggregate_recordings,
    clip_reconstruction_mse,
)
from bird_audio.training_batching import (
    collate_native_samples,
    to_autoencoder_batch,
    to_efficientnet_batch,
)

TASK1_FINAL_BATCH_SIZE = 32
TASK2_FINAL_BATCH_SIZE = 64
TASK1_CLASS_COUNT = 15
TASK2_LATENT_DIMENSIONS = 64
FINAL_KNOWN_TEST_ROLE = "known_test"
FINAL_UNKNOWN_ROLE = "unknown"
FINAL_SOURCE_ROLES = (FINAL_KNOWN_TEST_ROLE, FINAL_UNKNOWN_ROLE)

KNOWN_SPECIES_SCIENTIFIC_NAMES = (
    "Eudynamys scolopaceus",
    "Dicrurus macrocercus",
    "Psilopogon asiaticus",
    "Cuculus canorus",
    "Aegithina tiphia",
    "Alcedo atthis",
    "Acridotheres tristis",
    "Orthotomus sutorius",
    "Upupa epops",
    "Psilopogon virens",
    "Centropus sinensis",
    "Pycnonotus cafer",
    "Psittacula krameri",
    "Spilopelia chinensis",
    "Halcyon smyrnensis",
)
KNOWN_COMMON_TO_SCIENTIFIC = tuple(
    zip(LOCKED_TASK1_CLASS_ORDER, KNOWN_SPECIES_SCIENTIFIC_NAMES, strict=True)
)
_KNOWN_SCIENTIFIC_BY_COMMON = dict(KNOWN_COMMON_TO_SCIENTIFIC)


class FinalRecordingData(Protocol):
    split: str
    strategy: str
    lock_sha256: str
    recording_count: int
    recording_ids: tuple[str, ...]

    def __len__(self) -> int: ...

    def iter_metadata(self) -> Iterator[dict[str, str]]: ...

    def iter_recording_indices(self) -> Iterator[tuple[str, tuple[int, ...]]]: ...

    def get_recording(
        self,
        recording_id: str,
    ) -> tuple[np.ndarray, tuple[dict[str, str], ...]]: ...


@dataclass(frozen=True, slots=True)
class FinalInferenceTestInjection:
    known_test_clips: int
    known_test_recordings: int
    unknown_clips: int
    unknown_recordings: int

    def __post_init__(self) -> None:
        for name in (
            "known_test_clips",
            "known_test_recordings",
            "unknown_clips",
            "unknown_recordings",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Test injection {name} must be a positive integer")
        if self.known_test_clips < self.known_test_recordings:
            raise ValueError("Known-test fixture has fewer clips than recordings")
        if self.unknown_clips < self.unknown_recordings:
            raise ValueError("Unknown fixture has fewer clips than recordings")


def _identifier(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a nonempty trimmed identifier")
    return value


@dataclass(frozen=True, slots=True)
class FinalRecordingMetadata:
    source_role: str
    recording_id: str
    session_group: str
    species_common_name: str
    species_scientific_name: str
    class_index: int | None
    clip_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_role not in FINAL_SOURCE_ROLES:
            raise ValueError("Final recording source role is invalid")
        _identifier(self.recording_id, "recording_id")
        _identifier(self.session_group, "session_group")
        common = _identifier(self.species_common_name, "species_common_name")
        scientific = _identifier(self.species_scientific_name, "species_scientific_name")
        if type(self.clip_ids) is not tuple or not self.clip_ids:
            raise ValueError("Final recording clip identities cannot be empty")
        for clip_id in self.clip_ids:
            _identifier(clip_id, "clip_id")
        if self.clip_ids != tuple(sorted(self.clip_ids)) or len(set(self.clip_ids)) != len(
            self.clip_ids
        ):
            raise ValueError("Final recording clip identities must be unique and sorted")
        if self.source_role == FINAL_KNOWN_TEST_ROLE:
            if (
                type(self.class_index) is not int
                or not 0 <= self.class_index < TASK1_CLASS_COUNT
                or LOCKED_TASK1_CLASS_ORDER[self.class_index] != common
                or _KNOWN_SCIENTIFIC_BY_COMMON.get(common) != scientific
            ):
                raise ValueError("Known-test species and class mapping is invalid")
        elif self.class_index is not None:
            raise ValueError("Unknown final recordings cannot have a known class index")

    @property
    def clip_count(self) -> int:
        return len(self.clip_ids)

    def to_record(self) -> dict[str, object]:
        return {
            "source_role": self.source_role,
            "recording_id": self.recording_id,
            "session_group": self.session_group,
            "species_common_name": self.species_common_name,
            "species_scientific_name": self.species_scientific_name,
            "class_index": self.class_index,
            "clip_ids": list(self.clip_ids),
            "clip_count": self.clip_count,
        }


@dataclass(frozen=True, slots=True)
class Task2FinalInferenceResult:
    known_test: RecordingBatch
    unknown: RecordingBatch
    known_test_metadata: tuple[FinalRecordingMetadata, ...]
    unknown_metadata: tuple[FinalRecordingMetadata, ...]

    def __post_init__(self) -> None:
        if self.known_test.source_role != FINAL_KNOWN_TEST_ROLE:
            raise ValueError("Task 2 known-test score role is invalid")
        if self.unknown.source_role != FINAL_UNKNOWN_ROLE:
            raise ValueError("Task 2 unknown score role is invalid")
        for values, batch, role in (
            (self.known_test_metadata, self.known_test, FINAL_KNOWN_TEST_ROLE),
            (self.unknown_metadata, self.unknown, FINAL_UNKNOWN_ROLE),
        ):
            if (
                type(values) is not tuple
                or any(not isinstance(value, FinalRecordingMetadata) for value in values)
                or tuple(value.source_role for value in values) != (role,) * len(values)
                or tuple(value.recording_id for value in values) != batch.recording_ids
            ):
                raise ValueError("Task 2 score and metadata identities differ")


@dataclass(frozen=True, slots=True)
class Task1InferenceBatch:
    clip_count: int
    predictions: tuple[RecordingPrediction, ...]

    def __post_init__(self) -> None:
        if type(self.clip_count) is not int or self.clip_count <= 0:
            raise ValueError("Task 1 inference batch clip count must be positive")
        if (
            type(self.predictions) is not tuple
            or not self.predictions
            or any(not isinstance(value, RecordingPrediction) for value in self.predictions)
        ):
            raise ValueError("Task 1 inference batch predictions are invalid")
        recording_ids = self.recording_ids
        if recording_ids != tuple(sorted(recording_ids)) or len(set(recording_ids)) != len(
            recording_ids
        ):
            raise ValueError("Task 1 inference batch recording identities are invalid")

    @property
    def recording_ids(self) -> tuple[str, ...]:
        return tuple(prediction.recording_id for prediction in self.predictions)


@dataclass(frozen=True, slots=True)
class Task2InferenceBatch:
    clip_count: int
    scores: RecordingBatch
    metadata: tuple[FinalRecordingMetadata, ...]

    def __post_init__(self) -> None:
        if type(self.clip_count) is not int or self.clip_count <= 0:
            raise ValueError("Task 2 inference batch clip count must be positive")
        if not isinstance(self.scores, RecordingBatch):
            raise TypeError("Task 2 inference batch scores are invalid")
        if (
            type(self.metadata) is not tuple
            or not self.metadata
            or any(not isinstance(value, FinalRecordingMetadata) for value in self.metadata)
            or tuple(value.source_role for value in self.metadata)
            != (self.scores.source_role,) * len(self.metadata)
            or tuple(value.recording_id for value in self.metadata) != self.scores.recording_ids
        ):
            raise ValueError("Task 2 inference batch score and metadata identities differ")

    @property
    def recording_ids(self) -> tuple[str, ...]:
        return self.scores.recording_ids


@dataclass(frozen=True, slots=True)
class _PlannedRecording:
    metadata: FinalRecordingMetadata
    indices: tuple[int, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class _DataPlan:
    source_role: str
    clip_count: int
    recordings: tuple[_PlannedRecording, ...]


def recording_preserving_batches(
    recording_clip_counts: Sequence[tuple[str, int]],
    *,
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Pack complete recordings greedily without crossing the clip capacity."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Recording-preserving batch size must be positive")
    if not recording_clip_counts:
        raise ValueError("Recording-preserving batching requires recordings")
    seen: set[str] = set()
    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    current_clips = 0
    for raw_recording_id, clip_count in recording_clip_counts:
        recording_id = _identifier(raw_recording_id, "recording_id")
        if recording_id in seen:
            raise ValueError("Recording-preserving input contains a duplicate recording")
        seen.add(recording_id)
        if type(clip_count) is not int or clip_count <= 0:
            raise ValueError("Recording clip count must be positive")
        if clip_count > batch_size:
            raise ValueError("A recording is larger than the locked inference batch")
        if current and current_clips + clip_count > batch_size:
            batches.append(tuple(current))
            current = []
            current_clips = 0
        current.append(recording_id)
        current_clips += clip_count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _expected_counts(
    source_role: str,
    injection: FinalInferenceTestInjection | None,
) -> tuple[int, int]:
    if source_role == FINAL_KNOWN_TEST_ROLE:
        return (
            (KNOWN_TEST_ENERGY_CLIPS, KNOWN_TEST_RECORDINGS)
            if injection is None
            else (injection.known_test_clips, injection.known_test_recordings)
        )
    if source_role == FINAL_UNKNOWN_ROLE:
        return (
            (UNKNOWN_ENERGY_CLIPS, UNKNOWN_RECORDINGS)
            if injection is None
            else (injection.unknown_clips, injection.unknown_recordings)
        )
    raise ValueError("Final inference source role is invalid")


def _parse_positive_int(value: object, name: str) -> int:
    if type(value) is not str:
        raise TypeError(f"{name} must be a metadata string")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not an integer") from exc
    if parsed <= 0 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical positive integer")
    return parsed


def _parse_class_index(value: object) -> int:
    if type(value) is not str:
        raise TypeError("Known-test class index must be a metadata string")
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError("Known-test class index is invalid") from exc
    if not 0 <= index < TASK1_CLASS_COUNT or str(index) != value:
        raise ValueError("Known-test class index is outside the fixed class mapping")
    return index


def _metadata_recording_id(row: Mapping[str, str]) -> str:
    return _identifier(row.get("recording_id"), "recording_id")


def _planned_recording(
    recording_id: str,
    indices: tuple[int, ...],
    rows: tuple[dict[str, str], ...],
    *,
    source_role: str,
) -> _PlannedRecording:
    if not indices or len(indices) != len(rows):
        raise ValueError("Final recording index membership is empty or inconsistent")
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError("Final recording indices must be unique and ordered")
    expected_boundary = (
        "gated_final_known_test" if source_role == FINAL_KNOWN_TEST_ROLE else "gated_final_unknown"
    )
    sessions: set[str] = set()
    common_names: set[str] = set()
    scientific_names: set[str] = set()
    class_indices: set[int] = set()
    clip_ids: list[str] = []
    declared_counts: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or any(type(key) is not str for key in row):
            raise TypeError("Final metadata rows must be string dictionaries")
        if any(type(value) is not str for value in row.values()):
            raise TypeError("Final metadata values must be strings")
        if (
            _metadata_recording_id(row) != recording_id
            or row.get("selection_strategy") != "energy"
            or row.get("data_boundary") != expected_boundary
        ):
            raise ValueError("Final recording metadata role or strategy changed")
        clip_ids.append(_identifier(row.get("clip_id"), "clip_id"))
        sessions.add(_identifier(row.get("session_group"), "session_group"))
        common_names.add(_identifier(row.get("species_common_name"), "species_common_name"))
        declared_counts.add(
            _parse_positive_int(row.get("strategy_clip_count"), "strategy_clip_count")
        )
        if source_role == FINAL_KNOWN_TEST_ROLE:
            if row.get("split") != "test":
                raise ValueError("Known-test metadata contains another split")
            class_index = _parse_class_index(row.get("class_index"))
            class_indices.add(class_index)
            common = row["species_common_name"]
            scientific_names.add(_KNOWN_SCIENTIFIC_BY_COMMON.get(common, ""))
        else:
            scientific_names.add(
                _identifier(row.get("species_scientific_name"), "species_scientific_name")
            )
    if (
        len(sessions) != 1
        or len(common_names) != 1
        or len(scientific_names) != 1
        or len(declared_counts) != 1
        or next(iter(declared_counts)) != len(rows)
        or len(set(clip_ids)) != len(clip_ids)
    ):
        raise ValueError("Final per-recording metadata is not immutable")
    class_index: int | None = None
    if source_role == FINAL_KNOWN_TEST_ROLE:
        if len(class_indices) != 1:
            raise ValueError("Known-test class mapping changes within a recording")
        class_index = next(iter(class_indices))
    metadata = FinalRecordingMetadata(
        source_role=source_role,
        recording_id=recording_id,
        session_group=next(iter(sessions)),
        species_common_name=next(iter(common_names)),
        species_scientific_name=next(iter(scientific_names)),
        class_index=class_index,
        clip_ids=tuple(sorted(clip_ids)),
    )
    return _PlannedRecording(metadata=metadata, indices=indices, rows=rows)


def _plan_data(
    data: FinalRecordingData,
    *,
    source_role: str,
    test_injection: FinalInferenceTestInjection | None,
) -> _DataPlan:
    if test_injection is None:
        expected_type = (
            FinalKnownTestData if source_role == FINAL_KNOWN_TEST_ROLE else FinalUnknownData
        )
        if not isinstance(data, expected_type):
            raise TypeError("Production final inference requires a gated final data reader")
    expected_split = "test" if source_role == FINAL_KNOWN_TEST_ROLE else "unknown"
    expected_lock = (
        KNOWN_CACHE_LOCK_SHA256
        if source_role == FINAL_KNOWN_TEST_ROLE
        else UNKNOWN_CACHE_LOCK_SHA256
    )
    expected_clips, expected_recordings = _expected_counts(source_role, test_injection)
    if (
        getattr(data, "split", None) != expected_split
        or getattr(data, "strategy", None) != "energy"
        or getattr(data, "lock_sha256", None) != expected_lock
        or type(getattr(data, "recording_count", None)) is not int
        or data.recording_count != expected_recordings
        or len(data) != expected_clips
    ):
        raise ValueError("Final inference data role, lock, strategy, or counts changed")
    rows = tuple(data.iter_metadata())
    if len(rows) != expected_clips:
        raise ValueError("Final inference metadata count differs from the clip count")
    raw_groups = tuple(data.iter_recording_indices())
    if len(raw_groups) != expected_recordings:
        raise ValueError("Final inference recording index count changed")
    if tuple(recording_id for recording_id, _ in raw_groups) != tuple(data.recording_ids):
        raise ValueError("Final inference recording order differs from its reader")

    planned: list[_PlannedRecording] = []
    covered: list[int] = []
    recording_ids: set[str] = set()
    global_clip_ids: set[str] = set()
    for raw_recording_id, raw_indices in raw_groups:
        recording_id = _identifier(raw_recording_id, "recording_id")
        if recording_id in recording_ids:
            raise ValueError("Final inference contains a duplicate recording identity")
        recording_ids.add(recording_id)
        if type(raw_indices) is not tuple or any(type(index) is not int for index in raw_indices):
            raise TypeError("Final recording indices must be an integer tuple")
        if any(index < 0 or index >= len(rows) for index in raw_indices):
            raise ValueError("Final recording index is outside the selected clips")
        selected_rows = tuple(rows[index] for index in raw_indices)
        record = _planned_recording(
            recording_id,
            raw_indices,
            selected_rows,
            source_role=source_role,
        )
        overlap = global_clip_ids.intersection(record.metadata.clip_ids)
        if overlap:
            raise ValueError("Final inference contains duplicate clip identities")
        global_clip_ids.update(record.metadata.clip_ids)
        planned.append(record)
        covered.extend(raw_indices)
    if sorted(covered) != list(range(expected_clips)):
        raise ValueError("Final recording indices do not partition the selected clips")
    planned.sort(key=lambda record: record.metadata.recording_id)
    common_by_scientific: dict[str, str] = {}
    scientific_by_common: dict[str, str] = {}
    for record in planned:
        metadata = record.metadata
        observed_common = common_by_scientific.setdefault(
            metadata.species_scientific_name,
            metadata.species_common_name,
        )
        observed_scientific = scientific_by_common.setdefault(
            metadata.species_common_name,
            metadata.species_scientific_name,
        )
        if (
            observed_common != metadata.species_common_name
            or observed_scientific != metadata.species_scientific_name
        ):
            raise ValueError("Final species common and scientific names are inconsistent")
    if source_role == FINAL_UNKNOWN_ROLE and test_injection is None:
        species_counts = Counter(record.metadata.species_scientific_name for record in planned)
        if len(species_counts) != UNKNOWN_SPECIES or set(species_counts.values()) != {
            UNKNOWN_RECORDINGS_PER_SPECIES
        }:
            raise ValueError("Final unknown species recording counts changed")
    return _DataPlan(
        source_role=source_role,
        clip_count=expected_clips,
        recordings=tuple(planned),
    )


def _resolve_inference_device(
    device: torch.device,
    test_injection: FinalInferenceTestInjection | None,
) -> torch.device:
    if not isinstance(device, torch.device):
        raise TypeError("Final inference device must be a torch.device")
    if test_injection is None:
        if device.type != "mps":
            raise PermissionError("Production final inference requires the orchestrated MPS device")
    elif device.type != "cpu":
        raise PermissionError("Final inference test injection permits only CPU fixtures")
    return device


def _uses_device(value: torch.device, expected: torch.device) -> bool:
    return value.type == expected.type and (expected.index is None or value.index == expected.index)


def _validate_model_tensors(model: nn.Module, device: torch.device) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("Final inference model must be a torch module")
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors:
        raise ValueError("Final inference model has no state tensors")
    finite_status: list[torch.Tensor] = []
    for tensor in tensors:
        if not _uses_device(tensor.device, device):
            raise ValueError("Final inference model tensor is on another device")
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise TypeError("Final inference floating model tensors must use torch.float32")
            finite_status.append(torch.isfinite(tensor).all())
        elif tensor.is_complex():
            raise TypeError("Final inference model cannot contain complex tensors")
    if not finite_status:
        raise ValueError("Final inference model has no floating state tensors")
    status_cpu = torch.stack(finite_status).detach().to(device="cpu", copy=True)
    if not bool(status_cpu.all().item()):
        raise ValueError("Final inference model contains a nonfinite tensor")


def _remaining_recordings(
    plan: _DataPlan,
    skip_recording_ids: Collection[str],
) -> tuple[_PlannedRecording, ...]:
    if isinstance(skip_recording_ids, (str, bytes)) or not isinstance(
        skip_recording_ids, Collection
    ):
        raise TypeError("Skipped recording identities must be a sequence")
    skipped = tuple(_identifier(value, "skipped recording_id") for value in skip_recording_ids)
    if len(set(skipped)) != len(skipped):
        raise ValueError("Skipped recording identities must be unique")
    available = {record.metadata.recording_id for record in plan.recordings}
    if not set(skipped).issubset(available):
        raise ValueError("Skipped recording identity is outside the final data role")
    skipped_set = set(skipped)
    return tuple(
        record for record in plan.recordings if record.metadata.recording_id not in skipped_set
    )


def _load_recording_batch(
    data: FinalRecordingData,
    selected: Sequence[_PlannedRecording],
) -> tuple[torch.Tensor, tuple[tuple[_PlannedRecording, int, int], ...]]:
    samples: list[tuple[np.ndarray, Mapping[str, str]]] = []
    offsets: list[tuple[_PlannedRecording, int, int]] = []
    start = 0
    for planned in selected:
        features, rows = data.get_recording(planned.metadata.recording_id)
        if (
            not isinstance(features, np.ndarray)
            or features.dtype != np.float32
            or features.shape != (planned.metadata.clip_count, 1, 128, 372)
            or not isinstance(rows, tuple)
            or rows != planned.rows
        ):
            raise ValueError("Final data reader returned a changed recording")
        for index, row in enumerate(rows):
            samples.append((features[index], row))
        end = start + len(rows)
        offsets.append((planned, start, end))
        start = end
    native = collate_native_samples(samples)
    return native.tensor, tuple(offsets)


def _task1_logits_to_cpu(logits: torch.Tensor) -> torch.Tensor:
    return logits.detach().to(device="cpu", dtype=torch.float32, copy=True)


def _task2_outputs_to_cpu(
    reconstruction: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        reconstruction.detach().to(device="cpu", dtype=torch.float32, copy=True),
        latent.detach().to(device="cpu", dtype=torch.float32, copy=True),
    )


def _validate_task1_logits(
    logits: object,
    *,
    clip_count: int,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.dtype != torch.float32
        or not _uses_device(logits.device, device)
        or tuple(logits.shape) != (clip_count, TASK1_CLASS_COUNT)
    ):
        raise ValueError("Task 1 final logits violate the fixed output contract")
    return logits


def iter_task1_recording_batches(
    model: nn.Module,
    data: FinalRecordingData,
    *,
    device: torch.device,
    test_injection: FinalInferenceTestInjection | None = None,
    skip_recording_ids: Collection[str] = (),
) -> Iterator[Task1InferenceBatch]:
    """Yield fixed Task 1 results after each whole-recording model batch."""

    resolved_device = _resolve_inference_device(device, test_injection)
    _validate_model_tensors(model, resolved_device)
    plan = _plan_data(
        data,
        source_role=FINAL_KNOWN_TEST_ROLE,
        test_injection=test_injection,
    )
    remaining = _remaining_recordings(plan, skip_recording_ids)
    if not remaining:
        return
    remaining_ids = {record.metadata.recording_id for record in remaining}
    by_id = {record.metadata.recording_id: record for record in plan.recordings}
    batches = recording_preserving_batches(
        tuple(
            (record.metadata.recording_id, record.metadata.clip_count) for record in plan.recordings
        ),
        batch_size=TASK1_FINAL_BATCH_SIZE,
    )
    model.eval()
    for batch_ids in batches:
        emitted_ids = remaining_ids.intersection(batch_ids)
        if not emitted_ids:
            continue
        with torch.inference_mode():
            selected = tuple(by_id[recording_id] for recording_id in batch_ids)
            native, offsets = _load_recording_batch(data, selected)
            inputs_cpu = to_efficientnet_batch(native)
            if (
                inputs_cpu.dtype != torch.float32
                or inputs_cpu.device.type != "cpu"
                or tuple(inputs_cpu.shape) != (native.shape[0], 3, 224, 224)
                or not bool(torch.isfinite(inputs_cpu).all().item())
            ):
                raise RuntimeError("Task 1 locked preprocessing returned an invalid batch")
            inputs = inputs_cpu.to(device=resolved_device, dtype=torch.float32)
            logits = _validate_task1_logits(
                model(inputs),
                clip_count=native.shape[0],
                device=resolved_device,
            )
            logits_cpu = _task1_logits_to_cpu(logits)
            if (
                logits_cpu.dtype != torch.float32
                or logits_cpu.device.type != "cpu"
                or tuple(logits_cpu.shape) != (native.shape[0], TASK1_CLASS_COUNT)
                or not bool(torch.isfinite(logits_cpu).all().item())
            ):
                raise ValueError("Task 1 final CPU logits are invalid")
            predictions: list[RecordingPrediction] = []
            for planned, start, end in offsets:
                if planned.metadata.recording_id not in emitted_ids:
                    continue
                mean_logits = logits_cpu[start:end].mean(dim=0)
                if tuple(mean_logits.shape) != (TASK1_CLASS_COUNT,) or not bool(
                    torch.isfinite(mean_logits).all().item()
                ):
                    raise RuntimeError("Task 1 recording aggregation returned invalid logits")
                values = tuple(float(value) for value in mean_logits.tolist())
                predicted_index = int(torch.argmax(mean_logits).item())
                class_index = planned.metadata.class_index
                if type(class_index) is not int:
                    raise RuntimeError("Known-test recording lost its class index")
                predictions.append(
                    RecordingPrediction(
                        recording_id=planned.metadata.recording_id,
                        session_group=planned.metadata.session_group,
                        true_class_index=class_index,
                        mean_logits=values,
                        predicted_class_index=predicted_index,
                    )
                )
            yield Task1InferenceBatch(
                clip_count=sum(
                    planned.metadata.clip_count
                    for planned in selected
                    if planned.metadata.recording_id in emitted_ids
                ),
                predictions=tuple(predictions),
            )


def infer_task1_recording_data(
    model: nn.Module,
    data: FinalRecordingData,
    *,
    device: torch.device,
    test_injection: FinalInferenceTestInjection | None = None,
) -> tuple[RecordingPrediction, ...]:
    """Run fixed Task 1 inference with whole recordings in every model batch."""

    batches = tuple(
        iter_task1_recording_batches(
            model,
            data,
            device=device,
            test_injection=test_injection,
        )
    )
    predictions = tuple(prediction for batch in batches for prediction in batch.predictions)
    _, expected_recordings = _expected_counts(FINAL_KNOWN_TEST_ROLE, test_injection)
    if len(predictions) != expected_recordings:
        raise RuntimeError("Task 1 final prediction identities are incomplete")
    return predictions


def _validate_task2_outputs(
    value: object,
    *,
    clip_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("Task 2 final model must return reconstruction and latent tensors")
    reconstruction, latent = value
    if (
        not isinstance(reconstruction, torch.Tensor)
        or reconstruction.dtype != torch.float32
        or not _uses_device(reconstruction.device, device)
        or tuple(reconstruction.shape) != (clip_count, 1, 224, 224)
    ):
        raise ValueError("Task 2 final reconstruction violates the fixed output contract")
    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float32
        or not _uses_device(latent.device, device)
        or tuple(latent.shape) != (clip_count, TASK2_LATENT_DIMENSIONS)
    ):
        raise ValueError("Task 2 final latent tensor violates the fixed output contract")
    return reconstruction, latent


def iter_task2_recording_batches(
    model: nn.Module,
    data: FinalRecordingData,
    *,
    source_role: str,
    device: torch.device,
    test_injection: FinalInferenceTestInjection | None = None,
    skip_recording_ids: Collection[str] = (),
) -> Iterator[Task2InferenceBatch]:
    """Yield fixed Task 2 results after each whole-recording model batch."""

    if source_role not in FINAL_SOURCE_ROLES:
        raise ValueError("Task 2 final source role is invalid")
    resolved_device = _resolve_inference_device(device, test_injection)
    _validate_model_tensors(model, resolved_device)
    plan = _plan_data(data, source_role=source_role, test_injection=test_injection)
    remaining = _remaining_recordings(plan, skip_recording_ids)
    if not remaining:
        return
    remaining_ids = {record.metadata.recording_id for record in remaining}
    by_id = {record.metadata.recording_id: record for record in plan.recordings}
    batches = recording_preserving_batches(
        tuple(
            (record.metadata.recording_id, record.metadata.clip_count) for record in plan.recordings
        ),
        batch_size=TASK2_FINAL_BATCH_SIZE,
    )
    model.eval()
    for batch_ids in batches:
        emitted_ids = remaining_ids.intersection(batch_ids)
        if not emitted_ids:
            continue
        with torch.inference_mode():
            selected = tuple(by_id[recording_id] for recording_id in batch_ids)
            native, offsets = _load_recording_batch(data, selected)
            inputs_cpu = to_autoencoder_batch(native)
            if (
                inputs_cpu.dtype != torch.float32
                or inputs_cpu.device.type != "cpu"
                or tuple(inputs_cpu.shape) != (native.shape[0], 1, 224, 224)
                or not bool(torch.isfinite(inputs_cpu).all().item())
                or bool(torch.any(inputs_cpu < 0).item())
                or bool(torch.any(inputs_cpu > 1).item())
            ):
                raise RuntimeError("Task 2 locked preprocessing returned an invalid batch")
            inputs = inputs_cpu.to(device=resolved_device, dtype=torch.float32)
            reconstruction, latent = _validate_task2_outputs(
                model(inputs),
                clip_count=native.shape[0],
                device=resolved_device,
            )
            reconstruction_cpu, latent_cpu = _task2_outputs_to_cpu(reconstruction, latent)
            if (
                reconstruction_cpu.dtype != torch.float32
                or reconstruction_cpu.device.type != "cpu"
                or tuple(reconstruction_cpu.shape) != (native.shape[0], 1, 224, 224)
                or not bool(torch.isfinite(reconstruction_cpu).all().item())
                or latent_cpu.dtype != torch.float32
                or latent_cpu.device.type != "cpu"
                or tuple(latent_cpu.shape) != (native.shape[0], TASK2_LATENT_DIMENSIONS)
                or not bool(torch.isfinite(latent_cpu).all().item())
            ):
                raise ValueError("Task 2 final CPU outputs are invalid")
            clip_mse = clip_reconstruction_mse(
                inputs_cpu.numpy(),
                reconstruction_cpu.numpy(),
            )
            latent_values = np.asarray(latent_cpu.numpy(), dtype=np.float64)
            if latent_values.shape != (native.shape[0], TASK2_LATENT_DIMENSIONS) or not bool(
                np.all(np.isfinite(latent_values))
            ):
                raise RuntimeError("Task 2 CPU latent values are invalid")
            clip_identities: list[ClipIdentity] = []
            for planned, _, _ in offsets:
                for row in planned.rows:
                    clip_identities.append(
                        ClipIdentity(
                            recording_id=planned.metadata.recording_id,
                            clip_id=row["clip_id"],
                        )
                    )
            if (
                len(clip_identities) != native.shape[0]
                or clip_mse.shape != (native.shape[0],)
                or latent_values.shape != (native.shape[0], TASK2_LATENT_DIMENSIONS)
            ):
                raise RuntimeError("Task 2 inference batch clip results are incomplete")
            complete_scores = aggregate_recordings(
                tuple(clip_identities),
                clip_mse,
                latent_values,
                source_role=source_role,
            )
            scores = RecordingBatch(
                source_role=source_role,
                recordings=tuple(
                    score
                    for score in complete_scores.recordings
                    if score.recording_id in emitted_ids
                ),
            )
            metadata = tuple(
                record.metadata
                for record in selected
                if record.metadata.recording_id in emitted_ids
            )
            yield Task2InferenceBatch(
                clip_count=sum(
                    record.metadata.clip_count
                    for record in selected
                    if record.metadata.recording_id in emitted_ids
                ),
                scores=scores,
                metadata=metadata,
            )


def infer_task2_recording_data(
    model: nn.Module,
    data: FinalRecordingData,
    *,
    source_role: str,
    device: torch.device,
    test_injection: FinalInferenceTestInjection | None = None,
) -> tuple[RecordingBatch, tuple[FinalRecordingMetadata, ...]]:
    """Run fixed Task 2 inference for one canonical final data role."""

    batches = tuple(
        iter_task2_recording_batches(
            model,
            data,
            source_role=source_role,
            device=device,
            test_injection=test_injection,
        )
    )
    scores = RecordingBatch(
        source_role=source_role,
        recordings=tuple(recording for batch in batches for recording in batch.scores.recordings),
    )
    metadata = tuple(value for batch in batches for value in batch.metadata)
    _, expected_recordings = _expected_counts(source_role, test_injection)
    if len(scores.recordings) != expected_recordings or len(metadata) != expected_recordings:
        raise RuntimeError("Task 2 final recording results are incomplete")
    return scores, metadata


def run_task1_final_inference(
    model: nn.Module,
    authorization: FinalEvaluationAuthorization,
    *,
    device: torch.device,
    ffmpeg: str | Path | None = None,
) -> tuple[RecordingPrediction, ...]:
    data = open_final_known_test_data(authorization, ffmpeg=ffmpeg)
    return infer_task1_recording_data(model, data, device=device)


def run_task2_final_inference(
    model: nn.Module,
    authorization: FinalEvaluationAuthorization,
    *,
    device: torch.device,
    ffmpeg: str | Path | None = None,
) -> Task2FinalInferenceResult:
    known_data = open_final_known_test_data(authorization, ffmpeg=ffmpeg)
    unknown_data = open_final_unknown_data(authorization, ffmpeg=ffmpeg)
    known_scores, known_metadata = infer_task2_recording_data(
        model,
        known_data,
        source_role=FINAL_KNOWN_TEST_ROLE,
        device=device,
    )
    unknown_scores, unknown_metadata = infer_task2_recording_data(
        model,
        unknown_data,
        source_role=FINAL_UNKNOWN_ROLE,
        device=device,
    )
    return Task2FinalInferenceResult(
        known_test=known_scores,
        unknown=unknown_scores,
        known_test_metadata=known_metadata,
        unknown_metadata=unknown_metadata,
    )
