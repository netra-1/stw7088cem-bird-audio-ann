from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bird_audio.config import LOCKED_TASK1_CLASS_ORDER

CLASS_ORDER = LOCKED_TASK1_CLASS_ORDER
CLASS_COUNT = len(CLASS_ORDER)
STABILITY_SEEDS = (13, 37, 71)
PREDECLARED_DETAIL_SEED = 37
DEFAULT_BOOTSTRAP_SEED = 20260713
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_MAXIMUM_ATTEMPT_MULTIPLIER = 100
BOOTSTRAP_METRIC_NAMES = ("accuracy", "macro_f1")


def _validate_identifier(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a nonempty trimmed identifier")
    return value


def _validate_class_index(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < CLASS_COUNT:
        raise ValueError(f"{name} must be between 0 and {CLASS_COUNT - 1}")
    return value


def _validate_unit_float(value: object, name: str) -> float:
    if type(value) not in {float, np.float64}:
        raise TypeError(f"{name} must be a float64 value")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise TypeError(f"{name} must be a {dtype} NumPy array with shape {shape}")
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RecordingPrediction:
    recording_id: str
    session_group: str
    true_class_index: int
    mean_logits: tuple[float, ...]
    predicted_class_index: int

    def __post_init__(self) -> None:
        _validate_identifier(self.recording_id, "recording_id")
        _validate_identifier(self.session_group, "session_group")
        true_index = _validate_class_index(self.true_class_index, "true_class_index")
        predicted_index = _validate_class_index(
            self.predicted_class_index,
            "predicted_class_index",
        )
        if type(self.mean_logits) is not tuple or len(self.mean_logits) != CLASS_COUNT:
            raise ValueError(f"mean_logits must be a tuple of exactly {CLASS_COUNT} values")
        resolved_logits: list[float] = []
        for index, value in enumerate(self.mean_logits):
            if type(value) not in {float, np.float64}:
                raise TypeError(f"mean_logits[{index}] must be a float64 value")
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError("mean_logits must contain only finite values")
            resolved_logits.append(resolved)
        expected_prediction = int(np.argmax(np.asarray(resolved_logits, dtype=np.float64)))
        if predicted_index != expected_prediction:
            raise ValueError("predicted_class_index must equal the argmax of mean_logits")
        object.__setattr__(self, "true_class_index", true_index)
        object.__setattr__(self, "mean_logits", tuple(resolved_logits))
        object.__setattr__(self, "predicted_class_index", predicted_index)

    def to_record(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "session_group": self.session_group,
            "true_class_index": self.true_class_index,
            "true_class_name": CLASS_ORDER[self.true_class_index],
            "mean_logits": list(self.mean_logits),
            "predicted_class_index": self.predicted_class_index,
            "predicted_class_name": CLASS_ORDER[self.predicted_class_index],
        }


@dataclass(frozen=True, slots=True)
class PerClassMetrics:
    class_index: int
    class_name: str
    support: int
    precision: float
    recall: float
    f1: float

    def __post_init__(self) -> None:
        index = _validate_class_index(self.class_index, "class_index")
        if self.class_name != CLASS_ORDER[index]:
            raise ValueError("class_name does not match the locked class order")
        if type(self.support) is not int or self.support < 0:
            raise ValueError("support must be a nonnegative integer")
        for name in ("precision", "recall", "f1"):
            object.__setattr__(self, name, _validate_unit_float(getattr(self, name), name))

    def to_record(self) -> dict[str, Any]:
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "support": self.support,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _values_from_confusion(
    confusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = confusion.sum(axis=1, dtype=np.int64)
    predicted_support = confusion.sum(axis=0, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.zeros(CLASS_COUNT, dtype=np.float64)
    recall = np.zeros(CLASS_COUNT, dtype=np.float64)
    f1 = np.zeros(CLASS_COUNT, dtype=np.float64)
    np.divide(
        true_positive,
        predicted_support,
        out=precision,
        where=predicted_support > 0,
    )
    np.divide(true_positive, support, out=recall, where=support > 0)
    false_positive = predicted_support.astype(np.float64) - true_positive
    false_negative = support.astype(np.float64) - true_positive
    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=f1,
        where=f1_denominator > 0.0,
    )
    normalized = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.float64)
    np.divide(
        confusion,
        support[:, None],
        out=normalized,
        where=support[:, None] > 0,
    )
    return support, precision, recall, f1, normalized


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    recording_count: int
    accuracy: float
    macro_f1: float
    per_class: tuple[PerClassMetrics, ...]
    confusion_counts: np.ndarray
    row_normalized_confusion: np.ndarray

    def __post_init__(self) -> None:
        if type(self.recording_count) is not int or self.recording_count <= 0:
            raise ValueError("recording_count must be a positive integer")
        accuracy = _validate_unit_float(self.accuracy, "accuracy")
        macro_f1 = _validate_unit_float(self.macro_f1, "macro_f1")
        if type(self.per_class) is not tuple or len(self.per_class) != CLASS_COUNT:
            raise ValueError(f"per_class must contain exactly {CLASS_COUNT} entries")
        if any(not isinstance(item, PerClassMetrics) for item in self.per_class):
            raise TypeError("per_class must contain only PerClassMetrics values")
        if tuple(item.class_index for item in self.per_class) != tuple(range(CLASS_COUNT)):
            raise ValueError("per_class does not follow the locked class order")
        confusion = _readonly_array(
            self.confusion_counts,
            dtype=np.dtype(np.int64),
            shape=(CLASS_COUNT, CLASS_COUNT),
            name="confusion_counts",
        )
        normalized = _readonly_array(
            self.row_normalized_confusion,
            dtype=np.dtype(np.float64),
            shape=(CLASS_COUNT, CLASS_COUNT),
            name="row_normalized_confusion",
        )
        if bool(np.any(confusion < 0)) or int(confusion.sum()) != self.recording_count:
            raise ValueError("confusion_counts do not match recording_count")
        support, precision, recall, f1, expected_normalized = _values_from_confusion(confusion)
        expected_accuracy = float(np.trace(confusion) / self.recording_count)
        expected_macro_f1 = float(np.mean(f1, dtype=np.float64))
        if accuracy != expected_accuracy or macro_f1 != expected_macro_f1:
            raise ValueError("summary metrics do not match confusion_counts")
        if not np.array_equal(normalized, expected_normalized):
            raise ValueError("row_normalized_confusion does not match confusion_counts")
        for index, item in enumerate(self.per_class):
            expected = (
                int(support[index]),
                float(precision[index]),
                float(recall[index]),
                float(f1[index]),
            )
            observed = (item.support, item.precision, item.recall, item.f1)
            if observed != expected:
                raise ValueError("per_class metrics do not match confusion_counts")
        object.__setattr__(self, "accuracy", accuracy)
        object.__setattr__(self, "macro_f1", macro_f1)
        object.__setattr__(self, "confusion_counts", confusion)
        object.__setattr__(self, "row_normalized_confusion", normalized)

    def to_record(self) -> dict[str, Any]:
        return {
            "recording_count": self.recording_count,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "class_order": list(CLASS_ORDER),
            "per_class": [item.to_record() for item in self.per_class],
            "confusion_counts": self.confusion_counts.tolist(),
            "row_normalized_confusion": self.row_normalized_confusion.tolist(),
            "zero_division": 0,
        }


def _validated_predictions(
    predictions: object,
) -> tuple[RecordingPrediction, ...]:
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence of RecordingPrediction values")
    resolved = tuple(predictions)
    if not resolved:
        raise ValueError("predictions cannot be empty")
    if any(not isinstance(item, RecordingPrediction) for item in resolved):
        raise TypeError("predictions must contain only RecordingPrediction values")
    recording_ids = tuple(item.recording_id for item in resolved)
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError("recording IDs must be unique")
    return resolved


def _metrics_from_labels(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    _, _, _, f1, _ = _values_from_confusion(confusion)
    accuracy = float(np.mean(true_labels == predicted_labels, dtype=np.float64))
    macro_f1 = float(np.mean(f1, dtype=np.float64))
    return accuracy, macro_f1, f1, confusion


def evaluate_recording_predictions(
    predictions: Sequence[RecordingPrediction],
) -> ClassificationMetrics:
    resolved = _validated_predictions(predictions)
    true_labels = np.asarray([item.true_class_index for item in resolved], dtype=np.int64)
    predicted_labels = np.asarray(
        [item.predicted_class_index for item in resolved],
        dtype=np.int64,
    )
    accuracy, macro_f1, _, confusion = _metrics_from_labels(true_labels, predicted_labels)
    support, precision, recall, f1, normalized = _values_from_confusion(confusion)
    per_class = tuple(
        PerClassMetrics(
            class_index=index,
            class_name=CLASS_ORDER[index],
            support=int(support[index]),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
        )
        for index in range(CLASS_COUNT)
    )
    return ClassificationMetrics(
        recording_count=len(resolved),
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_counts=confusion,
        row_normalized_confusion=normalized,
    )


@dataclass(frozen=True, slots=True)
class SeedMetricSummary:
    metric_name: str
    values: tuple[float, float, float]
    mean: float
    sample_standard_deviation: float
    seeds: tuple[int, int, int] = STABILITY_SEEDS

    def __post_init__(self) -> None:
        if self.metric_name not in BOOTSTRAP_METRIC_NAMES:
            raise ValueError("metric_name must be accuracy or macro_f1")
        if self.seeds != STABILITY_SEEDS:
            raise ValueError("seed order must be 13, 37, 71")
        if type(self.values) is not tuple or len(self.values) != len(STABILITY_SEEDS):
            raise ValueError("values must contain exactly three seed results")
        values = tuple(
            _validate_unit_float(value, f"{self.metric_name} seed value") for value in self.values
        )
        expected_mean = float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))
        expected_sd = float(np.std(np.asarray(values, dtype=np.float64), dtype=np.float64, ddof=1))
        if self.mean != expected_mean or self.sample_standard_deviation != expected_sd:
            raise ValueError("seed summary arithmetic is invalid")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(
            self,
            "sample_standard_deviation",
            float(self.sample_standard_deviation),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "seeds": list(self.seeds),
            "values": list(self.values),
            "mean": self.mean,
            "sample_standard_deviation": self.sample_standard_deviation,
            "standard_deviation_ddof": 1,
        }


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    accuracy: SeedMetricSummary
    macro_f1: SeedMetricSummary

    def __post_init__(self) -> None:
        if (
            not isinstance(self.accuracy, SeedMetricSummary)
            or self.accuracy.metric_name != "accuracy"
        ):
            raise TypeError("accuracy must be an accuracy SeedMetricSummary")
        if (
            not isinstance(self.macro_f1, SeedMetricSummary)
            or self.macro_f1.metric_name != "macro_f1"
        ):
            raise TypeError("macro_f1 must be a macro_f1 SeedMetricSummary")

    def to_record(self) -> dict[str, Any]:
        return {
            "seeds": list(STABILITY_SEEDS),
            "accuracy": self.accuracy.to_record(),
            "macro_f1": self.macro_f1.to_record(),
        }


def summarize_stability(
    seed_metrics: Mapping[int, Mapping[str, float]],
) -> StabilitySummary:
    if not isinstance(seed_metrics, Mapping):
        raise TypeError("seed_metrics must be a mapping")
    if set(seed_metrics) != set(STABILITY_SEEDS) or any(
        type(seed) is not int for seed in seed_metrics
    ):
        raise ValueError("seed_metrics must contain exactly seeds 13, 37, and 71")
    expected_names = set(BOOTSTRAP_METRIC_NAMES)
    values_by_name: dict[str, tuple[float, float, float]] = {}
    for seed in STABILITY_SEEDS:
        metrics = seed_metrics[seed]
        if not isinstance(metrics, Mapping) or set(metrics) != expected_names:
            raise ValueError("each seed must contain exactly accuracy and macro_f1")
    for name in BOOTSTRAP_METRIC_NAMES:
        values_by_name[name] = tuple(
            _validate_unit_float(seed_metrics[seed][name], f"seed {seed} {name}")
            for seed in STABILITY_SEEDS
        )

    def summary(name: str) -> SeedMetricSummary:
        values = values_by_name[name]
        array = np.asarray(values, dtype=np.float64)
        return SeedMetricSummary(
            metric_name=name,
            values=values,
            mean=float(np.mean(array, dtype=np.float64)),
            sample_standard_deviation=float(np.std(array, dtype=np.float64, ddof=1)),
        )

    return StabilitySummary(accuracy=summary("accuracy"), macro_f1=summary("macro_f1"))


@dataclass(frozen=True, slots=True)
class PercentileInterval:
    lower: float
    upper: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        lower = _validate_unit_float(self.lower, "interval lower")
        upper = _validate_unit_float(self.upper, "interval upper")
        if lower > upper:
            raise ValueError("interval lower cannot exceed interval upper")
        if self.confidence_level != 0.95:
            raise ValueError("confidence_level must be 0.95")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_record(self) -> dict[str, float]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True, slots=True)
class BootstrapReplicates:
    accuracy: np.ndarray
    macro_f1: np.ndarray
    per_class_f1: np.ndarray
    recording_counts: np.ndarray
    replicate_count: int
    attempts: int
    maximum_attempts: int
    task1_seed: int = PREDECLARED_DETAIL_SEED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    class_order: tuple[str, ...] = CLASS_ORDER

    def __post_init__(self) -> None:
        if type(self.replicate_count) is not int or self.replicate_count <= 0:
            raise ValueError("replicate_count must be a positive integer")
        if type(self.attempts) is not int or not self.replicate_count <= self.attempts:
            raise ValueError("attempts must cover every accepted replicate")
        if type(self.maximum_attempts) is not int or not self.attempts <= self.maximum_attempts:
            raise ValueError("maximum_attempts must cover attempts")
        if self.task1_seed != PREDECLARED_DETAIL_SEED:
            raise ValueError("Task 1 bootstrap is predeclared for seed 37 only")
        if self.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED:
            raise ValueError("bootstrap_seed must be 20260713")
        if self.class_order != CLASS_ORDER:
            raise ValueError("bootstrap class order is not locked")
        accuracy = _readonly_array(
            self.accuracy,
            dtype=np.dtype(np.float64),
            shape=(self.replicate_count,),
            name="bootstrap accuracy",
        )
        macro_f1 = _readonly_array(
            self.macro_f1,
            dtype=np.dtype(np.float64),
            shape=(self.replicate_count,),
            name="bootstrap macro_f1",
        )
        per_class_f1 = _readonly_array(
            self.per_class_f1,
            dtype=np.dtype(np.float64),
            shape=(self.replicate_count, CLASS_COUNT),
            name="bootstrap per_class_f1",
        )
        recording_counts = _readonly_array(
            self.recording_counts,
            dtype=np.dtype(np.int64),
            shape=(self.replicate_count,),
            name="bootstrap recording_counts",
        )
        if (
            not bool(np.all(np.isfinite(accuracy)))
            or not bool(np.all(np.isfinite(macro_f1)))
            or not bool(np.all(np.isfinite(per_class_f1)))
            or not bool(np.all((accuracy >= 0.0) & (accuracy <= 1.0)))
            or not bool(np.all((macro_f1 >= 0.0) & (macro_f1 <= 1.0)))
            or not bool(np.all((per_class_f1 >= 0.0) & (per_class_f1 <= 1.0)))
            or not bool(np.all(recording_counts > 0))
            or not np.array_equal(
                macro_f1,
                np.mean(per_class_f1, axis=1, dtype=np.float64),
            )
        ):
            raise ValueError("bootstrap replicate arrays are invalid")
        object.__setattr__(self, "accuracy", accuracy)
        object.__setattr__(self, "macro_f1", macro_f1)
        object.__setattr__(self, "per_class_f1", per_class_f1)
        object.__setattr__(self, "recording_counts", recording_counts)

    @property
    def rejected_attempts(self) -> int:
        return self.attempts - self.replicate_count

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "task1_seed": np.asarray(self.task1_seed, dtype=np.int64),
            "bootstrap_seed": np.asarray(self.bootstrap_seed, dtype=np.int64),
            "replicate_count": np.asarray(self.replicate_count, dtype=np.int64),
            "attempts": np.asarray(self.attempts, dtype=np.int64),
            "maximum_attempts": np.asarray(self.maximum_attempts, dtype=np.int64),
            "metric_names": np.asarray(BOOTSTRAP_METRIC_NAMES, dtype=np.str_),
            "class_order": np.asarray(self.class_order, dtype=np.str_),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": self.per_class_f1,
            "recording_counts": self.recording_counts,
        }


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    replicates: BootstrapReplicates
    accuracy_interval: PercentileInterval
    macro_f1_interval: PercentileInterval
    per_class_f1_intervals: tuple[PercentileInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.replicates, BootstrapReplicates):
            raise TypeError("replicates must be BootstrapReplicates")
        if not isinstance(self.accuracy_interval, PercentileInterval):
            raise TypeError("accuracy_interval must be PercentileInterval")
        if not isinstance(self.macro_f1_interval, PercentileInterval):
            raise TypeError("macro_f1_interval must be PercentileInterval")
        if (
            type(self.per_class_f1_intervals) is not tuple
            or len(self.per_class_f1_intervals) != CLASS_COUNT
            or any(not isinstance(item, PercentileInterval) for item in self.per_class_f1_intervals)
        ):
            raise ValueError(f"per_class_f1_intervals must contain {CLASS_COUNT} intervals")

    def to_record(self) -> dict[str, Any]:
        return {
            "task1_seed": self.replicates.task1_seed,
            "bootstrap_seed": self.replicates.bootstrap_seed,
            "bootstrap_replicates": self.replicates.replicate_count,
            "attempts": self.replicates.attempts,
            "rejected_attempts": self.replicates.rejected_attempts,
            "maximum_attempts": self.replicates.maximum_attempts,
            "interval_method": "percentile",
            "confidence_level": 0.95,
            "accuracy_interval": self.accuracy_interval.to_record(),
            "macro_f1_interval": self.macro_f1_interval.to_record(),
            "per_class_f1_intervals": [
                {
                    "class_index": index,
                    "class_name": CLASS_ORDER[index],
                    **interval.to_record(),
                }
                for index, interval in enumerate(self.per_class_f1_intervals)
            ],
        }


def _interval(values: np.ndarray) -> PercentileInterval:
    bounds = np.percentile(values, [2.5, 97.5], method="linear")
    if bounds.shape != (2,) or not bool(np.all(np.isfinite(bounds))):
        raise RuntimeError("bootstrap percentile calculation failed")
    return PercentileInterval(lower=float(bounds[0]), upper=float(bounds[1]))


def session_cluster_bootstrap_seed37(
    predictions: Sequence[RecordingPrediction],
    *,
    task1_seed: int,
    replicate_count: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    maximum_attempts: int | None = None,
) -> BootstrapResult:
    if type(task1_seed) is not int or task1_seed != PREDECLARED_DETAIL_SEED:
        raise ValueError("Task 1 bootstrap is predeclared for seed 37 only")
    if type(replicate_count) is not int or replicate_count <= 0:
        raise ValueError("replicate_count must be a positive integer")
    if type(bootstrap_seed) is not int or bootstrap_seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError("bootstrap_seed must be 20260713")
    resolved_maximum = (
        replicate_count * DEFAULT_MAXIMUM_ATTEMPT_MULTIPLIER
        if maximum_attempts is None
        else maximum_attempts
    )
    if type(resolved_maximum) is not int or resolved_maximum < replicate_count:
        raise ValueError("maximum_attempts must be an integer at least replicate_count")

    resolved = tuple(
        sorted(_validated_predictions(predictions), key=lambda item: item.recording_id)
    )
    observed_classes = {item.true_class_index for item in resolved}
    if observed_classes != set(range(CLASS_COUNT)):
        raise ValueError("bootstrap input must contain every locked true class")

    grouped: dict[str, list[RecordingPrediction]] = {}
    for item in resolved:
        grouped.setdefault(item.session_group, []).append(item)
    groups = tuple(
        tuple(sorted(grouped[session], key=lambda item: item.recording_id))
        for session in sorted(grouped)
    )
    true_groups = tuple(
        np.asarray([item.true_class_index for item in group], dtype=np.int64) for group in groups
    )
    predicted_groups = tuple(
        np.asarray([item.predicted_class_index for item in group], dtype=np.int64)
        for group in groups
    )

    accuracy = np.empty(replicate_count, dtype=np.float64)
    macro_f1 = np.empty(replicate_count, dtype=np.float64)
    per_class_f1 = np.empty((replicate_count, CLASS_COUNT), dtype=np.float64)
    recording_counts = np.empty(replicate_count, dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(bootstrap_seed))
    accepted = 0
    attempts = 0
    while accepted < replicate_count:
        if attempts >= resolved_maximum:
            raise RuntimeError(
                "bootstrap maximum_attempts exhausted before collecting all valid replicates"
            )
        attempts += 1
        selected = generator.integers(0, len(groups), size=len(groups), endpoint=False)
        true_labels = np.concatenate([true_groups[int(index)] for index in selected])
        if np.unique(true_labels).size != CLASS_COUNT:
            continue
        predicted_labels = np.concatenate([predicted_groups[int(index)] for index in selected])
        replicate_accuracy, replicate_macro_f1, replicate_f1, _ = _metrics_from_labels(
            true_labels,
            predicted_labels,
        )
        accuracy[accepted] = replicate_accuracy
        macro_f1[accepted] = replicate_macro_f1
        per_class_f1[accepted] = replicate_f1
        recording_counts[accepted] = true_labels.size
        accepted += 1

    replicates = BootstrapReplicates(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        recording_counts=recording_counts,
        replicate_count=replicate_count,
        attempts=attempts,
        maximum_attempts=resolved_maximum,
        task1_seed=task1_seed,
        bootstrap_seed=bootstrap_seed,
    )
    return BootstrapResult(
        replicates=replicates,
        accuracy_interval=_interval(replicates.accuracy),
        macro_f1_interval=_interval(replicates.macro_f1),
        per_class_f1_intervals=tuple(
            _interval(replicates.per_class_f1[:, index]) for index in range(CLASS_COUNT)
        ),
    )
