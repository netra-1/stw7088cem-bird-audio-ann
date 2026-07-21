from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

KNOWN_SOURCE = "known"
UNKNOWN_SOURCE = "unknown"
DEFAULT_BOOTSTRAP_SEED = 20260713
DEFAULT_BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
STABILITY_SEEDS = (13, 37, 71)
METRIC_NAMES = ("auroc", "sensitivity", "specificity", "balanced_accuracy")

_SCIENTIFIC_NAME = re.compile(r"[A-Z][A-Za-z]+ [a-z][A-Za-z]+")


def _validate_identifier(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a nonempty trimmed identifier")
    return value


def _validate_float64(value: object, name: str) -> float:
    if type(value) not in {float, np.float64}:
        raise TypeError(f"{name} must be a float64 value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_unit_metric(value: object, name: str) -> float:
    result = _validate_float64(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class ScoredRecording:
    recording_id: str
    session_group: str
    species_scientific_name: str
    source: str
    score: float

    def __post_init__(self) -> None:
        _validate_identifier(self.recording_id, "recording_id")
        _validate_identifier(self.session_group, "session_group")
        if (
            type(self.species_scientific_name) is not str
            or _SCIENTIFIC_NAME.fullmatch(self.species_scientific_name) is None
        ):
            raise ValueError("species_scientific_name must be a strict scientific binomial")
        if type(self.source) is not str or self.source not in {KNOWN_SOURCE, UNKNOWN_SOURCE}:
            raise ValueError("source must be known or unknown")
        score = _validate_float64(self.score, "score")
        object.__setattr__(self, "score", score)

    def to_record(self) -> dict[str, str | float]:
        return {
            "recording_id": self.recording_id,
            "session_group": self.session_group,
            "species_scientific_name": self.species_scientific_name,
            "source": self.source,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class MetricValues:
    auroc: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float

    def __post_init__(self) -> None:
        for name in METRIC_NAMES:
            value = _validate_unit_metric(getattr(self, name), name)
            object.__setattr__(self, name, value)
        expected = 0.5 * (self.sensitivity + self.specificity)
        if not math.isclose(self.balanced_accuracy, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("balanced_accuracy must be the mean of sensitivity and specificity")

    def as_array(self) -> np.ndarray:
        values = np.asarray([getattr(self, name) for name in METRIC_NAMES], dtype=np.float64)
        values.setflags(write=False)
        return values

    def to_record(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in METRIC_NAMES}


@dataclass(frozen=True, slots=True)
class BinaryNoveltyMetrics:
    values: MetricValues
    known_recording_count: int
    unknown_recording_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, MetricValues):
            raise TypeError("values must be MetricValues")
        for name in ("known_recording_count", "unknown_recording_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_record(self) -> dict[str, Any]:
        return {
            **self.values.to_record(),
            "known_recording_count": self.known_recording_count,
            "unknown_recording_count": self.unknown_recording_count,
        }


@dataclass(frozen=True, slots=True)
class SpeciesNoveltyMetrics:
    species_scientific_name: str
    metrics: BinaryNoveltyMetrics

    def __post_init__(self) -> None:
        if (
            type(self.species_scientific_name) is not str
            or _SCIENTIFIC_NAME.fullmatch(self.species_scientific_name) is None
        ):
            raise ValueError("species_scientific_name must be a strict scientific binomial")
        if not isinstance(self.metrics, BinaryNoveltyMetrics):
            raise TypeError("metrics must be BinaryNoveltyMetrics")

    def to_record(self) -> dict[str, Any]:
        return {
            "species_scientific_name": self.species_scientific_name,
            **self.metrics.to_record(),
        }


@dataclass(frozen=True, slots=True)
class NoveltyEvaluation:
    threshold: float
    pooled: BinaryNoveltyMetrics
    per_species: tuple[SpeciesNoveltyMetrics, ...]
    macro: MetricValues

    def __post_init__(self) -> None:
        threshold = _validate_float64(self.threshold, "threshold")
        object.__setattr__(self, "threshold", threshold)
        if not isinstance(self.pooled, BinaryNoveltyMetrics):
            raise TypeError("pooled must be BinaryNoveltyMetrics")
        if type(self.per_species) is not tuple or not self.per_species:
            raise ValueError("per_species must be a nonempty tuple")
        if any(not isinstance(item, SpeciesNoveltyMetrics) for item in self.per_species):
            raise TypeError("per_species must contain SpeciesNoveltyMetrics values")
        names = tuple(item.species_scientific_name for item in self.per_species)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("per_species must contain unique scientific names in lexical order")
        if not isinstance(self.macro, MetricValues):
            raise TypeError("macro must be MetricValues")
        if any(
            item.metrics.known_recording_count != self.pooled.known_recording_count
            for item in self.per_species
        ):
            raise ValueError("all scientific species metrics must use the pooled known recordings")
        if (
            sum(item.metrics.unknown_recording_count for item in self.per_species)
            != self.pooled.unknown_recording_count
        ):
            raise ValueError("scientific species unknown counts must sum to the pooled count")
        for name in METRIC_NAMES:
            expected = float(
                np.mean(
                    np.asarray(
                        [getattr(item.metrics.values, name) for item in self.per_species],
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            if getattr(self.macro, name) != expected:
                raise ValueError("macro metrics must be unweighted scientific species means")

    @property
    def species_scientific_names(self) -> tuple[str, ...]:
        return tuple(item.species_scientific_name for item in self.per_species)

    def to_record(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "pooled": self.pooled.to_record(),
            "per_species": [item.to_record() for item in self.per_species],
            "macro": self.macro.to_record(),
        }


@dataclass(frozen=True, slots=True)
class PercentileInterval:
    lower: float
    upper: float
    confidence_level: float = CONFIDENCE_LEVEL

    def __post_init__(self) -> None:
        lower = _validate_unit_metric(self.lower, "interval lower bound")
        upper = _validate_unit_metric(self.upper, "interval upper bound")
        if lower > upper:
            raise ValueError("interval lower bound cannot exceed its upper bound")
        if self.confidence_level != CONFIDENCE_LEVEL:
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
class MetricIntervals:
    auroc: PercentileInterval
    sensitivity: PercentileInterval
    specificity: PercentileInterval
    balanced_accuracy: PercentileInterval

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, name), PercentileInterval) for name in METRIC_NAMES):
            raise TypeError("all metric intervals must be PercentileInterval values")

    def to_record(self) -> dict[str, dict[str, float]]:
        return {name: getattr(self, name).to_record() for name in METRIC_NAMES}


@dataclass(frozen=True, slots=True)
class SpeciesMetricIntervals:
    species_scientific_name: str
    intervals: MetricIntervals

    def __post_init__(self) -> None:
        if (
            type(self.species_scientific_name) is not str
            or _SCIENTIFIC_NAME.fullmatch(self.species_scientific_name) is None
        ):
            raise ValueError("species_scientific_name must be a strict scientific binomial")
        if not isinstance(self.intervals, MetricIntervals):
            raise TypeError("intervals must be MetricIntervals")

    def to_record(self) -> dict[str, Any]:
        return {
            "species_scientific_name": self.species_scientific_name,
            **self.intervals.to_record(),
        }


def _readonly_float64_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be a float64 NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    if not bool(np.all(np.isfinite(value))) or not bool(np.all((value >= 0.0) & (value <= 1.0))):
        raise ValueError(f"{name} must contain finite unit interval values")
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class BootstrapReplicates:
    pooled: np.ndarray
    per_species: np.ndarray
    macro: np.ndarray
    species_scientific_names: tuple[str, ...]
    replicate_count: int
    seed: int = DEFAULT_BOOTSTRAP_SEED
    metric_names: tuple[str, ...] = METRIC_NAMES

    def __post_init__(self) -> None:
        if isinstance(self.replicate_count, bool) or not isinstance(self.replicate_count, int):
            raise TypeError("replicate_count must be an integer")
        if self.replicate_count <= 0:
            raise ValueError("replicate_count must be positive")
        if self.seed != DEFAULT_BOOTSTRAP_SEED:
            raise ValueError("bootstrap seed must be 20260713")
        if self.metric_names != METRIC_NAMES:
            raise ValueError("bootstrap metric order is not locked")
        if type(self.species_scientific_names) is not tuple or not self.species_scientific_names:
            raise ValueError("species_scientific_names must be a nonempty tuple")
        if self.species_scientific_names != tuple(sorted(self.species_scientific_names)) or len(
            set(self.species_scientific_names)
        ) != len(self.species_scientific_names):
            raise ValueError("bootstrap species names must be unique and lexically ordered")
        if any(_SCIENTIFIC_NAME.fullmatch(name) is None for name in self.species_scientific_names):
            raise ValueError("bootstrap species names must be strict scientific binomials")
        species_count = len(self.species_scientific_names)
        object.__setattr__(
            self,
            "pooled",
            _readonly_float64_array(
                self.pooled,
                (self.replicate_count, len(METRIC_NAMES)),
                "pooled replicates",
            ),
        )
        object.__setattr__(
            self,
            "per_species",
            _readonly_float64_array(
                self.per_species,
                (self.replicate_count, species_count, len(METRIC_NAMES)),
                "per_species replicates",
            ),
        )
        object.__setattr__(
            self,
            "macro",
            _readonly_float64_array(
                self.macro,
                (self.replicate_count, len(METRIC_NAMES)),
                "macro replicates",
            ),
        )
        if not np.array_equal(self.macro, np.mean(self.per_species, axis=1, dtype=np.float64)):
            raise ValueError("macro replicates must be row means of per_species replicates")

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        """Return only nonobject arrays accepted by NumPy compressed archives."""

        return {
            "seed": np.asarray(self.seed, dtype=np.int64),
            "replicate_count": np.asarray(self.replicate_count, dtype=np.int64),
            "metric_names": np.asarray(self.metric_names, dtype=np.str_),
            "species_scientific_names": np.asarray(
                self.species_scientific_names,
                dtype=np.str_,
            ),
            "pooled": self.pooled,
            "per_species": self.per_species,
            "macro": self.macro,
        }


@dataclass(frozen=True, slots=True)
class SessionBootstrapResult:
    point_estimates: NoveltyEvaluation
    pooled_intervals: MetricIntervals
    per_species_intervals: tuple[SpeciesMetricIntervals, ...]
    macro_intervals: MetricIntervals
    replicates: BootstrapReplicates

    def __post_init__(self) -> None:
        if not isinstance(self.point_estimates, NoveltyEvaluation):
            raise TypeError("point_estimates must be NoveltyEvaluation")
        if not isinstance(self.pooled_intervals, MetricIntervals):
            raise TypeError("pooled_intervals must be MetricIntervals")
        if type(self.per_species_intervals) is not tuple or any(
            not isinstance(item, SpeciesMetricIntervals) for item in self.per_species_intervals
        ):
            raise TypeError("per_species_intervals must contain SpeciesMetricIntervals values")
        if not isinstance(self.macro_intervals, MetricIntervals):
            raise TypeError("macro_intervals must be MetricIntervals")
        if not isinstance(self.replicates, BootstrapReplicates):
            raise TypeError("replicates must be BootstrapReplicates")
        interval_names = tuple(item.species_scientific_name for item in self.per_species_intervals)
        expected = self.point_estimates.species_scientific_names
        if interval_names != expected or self.replicates.species_scientific_names != expected:
            raise ValueError("bootstrap species ordering does not match point estimates")

    def to_record(self) -> dict[str, Any]:
        return {
            "bootstrap_seed": self.replicates.seed,
            "bootstrap_replicates": self.replicates.replicate_count,
            "confidence_level": CONFIDENCE_LEVEL,
            "interval_method": "percentile",
            "point_estimates": self.point_estimates.to_record(),
            "pooled_intervals": self.pooled_intervals.to_record(),
            "per_species_intervals": [item.to_record() for item in self.per_species_intervals],
            "macro_intervals": self.macro_intervals.to_record(),
        }


@dataclass(frozen=True, slots=True)
class SeedMetricSummary:
    metric_name: str
    values: tuple[float, float, float]
    mean: float
    sample_standard_deviation: float
    seeds: tuple[int, int, int] = STABILITY_SEEDS

    def __post_init__(self) -> None:
        _validate_identifier(self.metric_name, "metric_name")
        if self.seeds != STABILITY_SEEDS:
            raise ValueError("seed order must be 13, 37, 71")
        if type(self.values) is not tuple or len(self.values) != len(STABILITY_SEEDS):
            raise ValueError("values must contain exactly three seed results")
        validated = tuple(_validate_float64(value, "seed metric value") for value in self.values)
        if any(not 0.0 <= value <= 1.0 for value in validated):
            raise ValueError("seed metric values must be between zero and one")
        values_array = np.asarray(validated, dtype=np.float64)
        expected_mean = float(np.mean(values_array, dtype=np.float64))
        expected_sd = float(np.std(values_array, dtype=np.float64, ddof=1))
        if not math.isclose(self.mean, expected_mean, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("seed summary mean is invalid")
        if not math.isclose(
            self.sample_standard_deviation,
            expected_sd,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("seed summary sample standard deviation is invalid")
        object.__setattr__(self, "values", validated)
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


def _validate_score_vector(values: object, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence | np.ndarray):
        raise TypeError(f"{name} must be a sequence of scores")
    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty")
    resolved: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (float, int, np.floating, np.integer)):
            raise TypeError(f"{name}[{index}] must be a real number")
        item = float(value)
        if not math.isfinite(item):
            raise ValueError(f"{name}[{index}] must be finite")
        resolved.append(item)
    return np.asarray(resolved, dtype=np.float64)


def tie_aware_auroc(known_scores: Sequence[float], unknown_scores: Sequence[float]) -> float:
    """Return exact empirical AUROC with half credit for equal score pairs."""

    known = np.sort(_validate_score_vector(known_scores, "known_scores"))
    unknown = _validate_score_vector(unknown_scores, "unknown_scores")
    concordance = 0.0
    for score in unknown:
        lower = int(np.searchsorted(known, score, side="left"))
        upper = int(np.searchsorted(known, score, side="right"))
        concordance += lower + 0.5 * (upper - lower)
    result = concordance / (known.size * unknown.size)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("AUROC calculation produced an invalid value")
    return float(result)


def _binary_metrics(
    known_scores: Sequence[float],
    unknown_scores: Sequence[float],
    threshold: float,
) -> BinaryNoveltyMetrics:
    known = _validate_score_vector(known_scores, "known_scores")
    unknown = _validate_score_vector(unknown_scores, "unknown_scores")
    threshold_value = _validate_float64(threshold, "threshold")
    sensitivity = float(np.count_nonzero(unknown > threshold_value) / unknown.size)
    specificity = float(np.count_nonzero(known <= threshold_value) / known.size)
    return BinaryNoveltyMetrics(
        values=MetricValues(
            auroc=tie_aware_auroc(known, unknown),
            sensitivity=sensitivity,
            specificity=specificity,
            balanced_accuracy=0.5 * (sensitivity + specificity),
        ),
        known_recording_count=int(known.size),
        unknown_recording_count=int(unknown.size),
    )


def _validated_recordings(recordings: object) -> tuple[ScoredRecording, ...]:
    if isinstance(recordings, (str, bytes)) or not isinstance(recordings, Sequence):
        raise TypeError("recordings must be a sequence of ScoredRecording values")
    resolved = tuple(recordings)
    if not resolved:
        raise ValueError("recordings cannot be empty")
    if any(not isinstance(recording, ScoredRecording) for recording in resolved):
        raise TypeError("recordings must contain only ScoredRecording values")

    recording_ids = [recording.recording_id for recording in resolved]
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError("recording IDs must be globally unique")

    known_species = {
        recording.species_scientific_name
        for recording in resolved
        if recording.source == KNOWN_SOURCE
    }
    unknown_species = {
        recording.species_scientific_name
        for recording in resolved
        if recording.source == UNKNOWN_SOURCE
    }
    if not known_species or not unknown_species:
        raise ValueError("recordings must contain both known and unknown sources")
    if known_species.intersection(unknown_species):
        raise ValueError("known and unknown scientific species labels overlap")

    sessions: dict[str, list[ScoredRecording]] = {}
    for recording in resolved:
        sessions.setdefault(recording.session_group, []).append(recording)
    for session_group, members in sessions.items():
        sources = {member.source for member in members}
        if len(sources) != 1:
            raise ValueError(f"session group overlaps known and unknown sources: {session_group}")
        if (
            members[0].source == UNKNOWN_SOURCE
            and len({member.species_scientific_name for member in members}) != 1
        ):
            raise ValueError(f"unknown session group overlaps scientific species: {session_group}")

    return tuple(
        sorted(
            resolved,
            key=lambda item: (
                item.source,
                item.session_group,
                item.species_scientific_name,
                item.recording_id,
            ),
        )
    )


def evaluate_novelty_scores(
    recordings: Sequence[ScoredRecording],
    threshold: float,
) -> NoveltyEvaluation:
    """Calculate pooled, scientific species, and unweighted macro metrics."""

    resolved = _validated_recordings(recordings)
    threshold_value = _validate_float64(threshold, "threshold")
    known_scores = [item.score for item in resolved if item.source == KNOWN_SOURCE]
    unknown = [item for item in resolved if item.source == UNKNOWN_SOURCE]
    unknown_scores = [item.score for item in unknown]
    species_names = tuple(sorted({item.species_scientific_name for item in unknown}))
    per_species = tuple(
        SpeciesNoveltyMetrics(
            species_scientific_name=species,
            metrics=_binary_metrics(
                known_scores,
                [item.score for item in unknown if item.species_scientific_name == species],
                threshold_value,
            ),
        )
        for species in species_names
    )
    macro_values = {
        name: float(
            np.mean(
                np.asarray(
                    [getattr(item.metrics.values, name) for item in per_species],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
        )
        for name in METRIC_NAMES
    }
    return NoveltyEvaluation(
        threshold=threshold_value,
        pooled=_binary_metrics(known_scores, unknown_scores, threshold_value),
        per_species=per_species,
        macro=MetricValues(**macro_values),
    )


def _group_scores(
    recordings: Sequence[ScoredRecording],
    source: str,
    species: str | None = None,
) -> tuple[np.ndarray, ...]:
    grouped: dict[str, list[float]] = {}
    for recording in recordings:
        if recording.source == source and (
            species is None or recording.species_scientific_name == species
        ):
            grouped.setdefault(recording.session_group, []).append(recording.score)
    result = []
    for session_group in sorted(grouped):
        values = np.asarray(grouped[session_group], dtype=np.float64)
        if values.size <= 0 or not bool(np.all(np.isfinite(values))):
            raise ValueError("session score group is empty or nonfinite")
        result.append(values)
    if not result:
        raise ValueError("session score groups cannot be empty")
    return tuple(result)


def _resample_session_scores(
    groups: tuple[np.ndarray, ...],
    generator: np.random.Generator,
) -> np.ndarray:
    selected = generator.integers(0, len(groups), size=len(groups), endpoint=False)
    return np.concatenate([groups[int(index)] for index in selected]).astype(
        np.float64,
        copy=False,
    )


def _metric_intervals(values: np.ndarray) -> MetricIntervals:
    bounds = np.percentile(values, [2.5, 97.5], axis=0, method="linear")
    if bounds.shape != (2, len(METRIC_NAMES)) or not bool(np.all(np.isfinite(bounds))):
        raise ValueError("bootstrap percentile calculation produced invalid bounds")
    return MetricIntervals(
        **{
            name: PercentileInterval(
                lower=float(bounds[0, index]),
                upper=float(bounds[1, index]),
            )
            for index, name in enumerate(METRIC_NAMES)
        }
    )


def session_cluster_bootstrap(
    recordings: Sequence[ScoredRecording],
    threshold: float,
    *,
    replicate_count: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> SessionBootstrapResult:
    """Return deterministic 95 percent intervals from whole session resamples."""

    if isinstance(replicate_count, bool) or not isinstance(replicate_count, int):
        raise TypeError("replicate_count must be an integer")
    if replicate_count <= 0:
        raise ValueError("replicate_count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError("bootstrap seed must be 20260713")

    resolved = _validated_recordings(recordings)
    threshold_value = _validate_float64(threshold, "threshold")
    point_estimates = evaluate_novelty_scores(resolved, threshold_value)
    species_names = point_estimates.species_scientific_names
    known_groups = _group_scores(resolved, KNOWN_SOURCE)
    pooled_unknown_groups = _group_scores(resolved, UNKNOWN_SOURCE)
    species_groups = {
        species: _group_scores(resolved, UNKNOWN_SOURCE, species) for species in species_names
    }

    generator = np.random.Generator(np.random.PCG64(seed))
    pooled = np.empty((replicate_count, len(METRIC_NAMES)), dtype=np.float64)
    per_species = np.empty(
        (replicate_count, len(species_names), len(METRIC_NAMES)),
        dtype=np.float64,
    )
    for replicate_index in range(replicate_count):
        pooled_known = _resample_session_scores(known_groups, generator)
        pooled_unknown = _resample_session_scores(pooled_unknown_groups, generator)
        pooled[replicate_index] = _binary_metrics(
            pooled_known,
            pooled_unknown,
            threshold_value,
        ).values.as_array()
        for species_index, species in enumerate(species_names):
            species_unknown = _resample_session_scores(species_groups[species], generator)
            per_species[replicate_index, species_index] = _binary_metrics(
                pooled_known,
                species_unknown,
                threshold_value,
            ).values.as_array()
    macro = np.mean(per_species, axis=1, dtype=np.float64)
    replicates = BootstrapReplicates(
        pooled=pooled,
        per_species=per_species,
        macro=macro,
        species_scientific_names=species_names,
        replicate_count=replicate_count,
        seed=seed,
    )
    return SessionBootstrapResult(
        point_estimates=point_estimates,
        pooled_intervals=_metric_intervals(replicates.pooled),
        per_species_intervals=tuple(
            SpeciesMetricIntervals(
                species_scientific_name=species,
                intervals=_metric_intervals(replicates.per_species[:, index, :]),
            )
            for index, species in enumerate(species_names)
        ),
        macro_intervals=_metric_intervals(replicates.macro),
        replicates=replicates,
    )


def summarize_across_seeds(
    seed_metrics: Mapping[int, Mapping[str, float]],
) -> tuple[SeedMetricSummary, ...]:
    """Summarize exactly seeds 13, 37, and 71 with sample standard deviation."""

    if not isinstance(seed_metrics, Mapping):
        raise TypeError("seed_metrics must be a mapping")
    if set(seed_metrics) != set(STABILITY_SEEDS) or any(
        type(seed) is not int for seed in seed_metrics
    ):
        raise ValueError("seed_metrics must contain exactly seeds 13, 37, and 71")
    for seed in STABILITY_SEEDS:
        if not isinstance(seed_metrics[seed], Mapping) or not seed_metrics[seed]:
            raise ValueError(f"seed {seed} metrics must be a nonempty mapping")
    metric_names = tuple(sorted(seed_metrics[STABILITY_SEEDS[0]]))
    if any(type(name) is not str or not name for name in metric_names):
        raise ValueError("metric names must be nonempty strings")
    if any(tuple(sorted(seed_metrics[seed])) != metric_names for seed in STABILITY_SEEDS):
        raise ValueError("all seeds must contain identical metric names")

    summaries = []
    for metric_name in metric_names:
        values = tuple(
            _validate_float64(seed_metrics[seed][metric_name], f"seed {seed} {metric_name}")
            for seed in STABILITY_SEEDS
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("seed metric values must be between zero and one")
        values_array = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(values_array, dtype=np.float64))
        sample_standard_deviation = float(np.std(values_array, dtype=np.float64, ddof=1))
        summaries.append(
            SeedMetricSummary(
                metric_name=metric_name,
                values=values,
                mean=float(mean),
                sample_standard_deviation=float(sample_standard_deviation),
            )
        )
    return tuple(summaries)
