from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

KNOWN_TRAINING_ROLE = "known_training"
KNOWN_VALIDATION_ROLE = "known_validation"
NEAREST_NEIGHBOURS = 10
THRESHOLD_QUANTILE = 0.95
THRESHOLD_QUANTILE_METHOD = "higher"
NOVELTY_DIRECTION = "higher_is_more_novel"
THRESHOLD_OPERATOR = ">"
RECONSTRUCTION_SCORE_NAME = "median_clip_reconstruction_mse"
LATENT_SCORE_NAME = "recording_mean_latent_knn_distance"
LOCKED_SCORE_NAMES = (RECONSTRUCTION_SCORE_NAME, LATENT_SCORE_NAME)


def _validated_identifier(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a nonempty trimmed identifier")
    return value


def _validated_real(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validated_float_array(
    value: object,
    name: str,
    *,
    dimensions: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must have a real floating-point dtype")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} cannot contain an empty dimension")
    if not bool(np.all(np.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")
    return value


@dataclass(frozen=True, slots=True)
class ClipIdentity:
    recording_id: str
    clip_id: str

    def __post_init__(self) -> None:
        _validated_identifier(self.recording_id, "recording_id")
        _validated_identifier(self.clip_id, "clip_id")

    def to_record(self) -> dict[str, str]:
        return {"recording_id": self.recording_id, "clip_id": self.clip_id}


@dataclass(frozen=True, slots=True)
class RecordingScore:
    recording_id: str
    clip_ids: tuple[str, ...]
    reconstruction_mse: float
    mean_latent_embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        _validated_identifier(self.recording_id, "recording_id")
        if type(self.clip_ids) is not tuple or not self.clip_ids:
            raise ValueError("clip_ids must be a nonempty tuple")
        for clip_id in self.clip_ids:
            _validated_identifier(clip_id, "clip_id")
        if len(set(self.clip_ids)) != len(self.clip_ids):
            raise ValueError("clip_ids must be unique within a recording")
        if tuple(sorted(self.clip_ids)) != self.clip_ids:
            raise ValueError("clip_ids must use deterministic lexical order")
        _validated_real(self.reconstruction_mse, "reconstruction_mse", nonnegative=True)
        if type(self.mean_latent_embedding) is not tuple or not self.mean_latent_embedding:
            raise ValueError("mean_latent_embedding must be a nonempty tuple")
        for coordinate in self.mean_latent_embedding:
            _validated_real(coordinate, "latent coordinate")

    @property
    def clip_count(self) -> int:
        return len(self.clip_ids)

    def to_record(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "clip_ids": list(self.clip_ids),
            "clip_count": self.clip_count,
            "reconstruction_mse": float(self.reconstruction_mse),
            "mean_latent_embedding": [float(value) for value in self.mean_latent_embedding],
        }


@dataclass(frozen=True, slots=True)
class RecordingBatch:
    source_role: str
    recordings: tuple[RecordingScore, ...]

    def __post_init__(self) -> None:
        _validated_identifier(self.source_role, "source_role")
        if type(self.recordings) is not tuple or not self.recordings:
            raise ValueError("recordings must be a nonempty tuple")
        if any(not isinstance(recording, RecordingScore) for recording in self.recordings):
            raise TypeError("recordings must contain only RecordingScore values")
        recording_ids = tuple(recording.recording_id for recording in self.recordings)
        if len(set(recording_ids)) != len(recording_ids):
            raise ValueError("recording identities must be unique within a batch")
        if tuple(sorted(recording_ids)) != recording_ids:
            raise ValueError("recordings must use deterministic lexical order")
        dimensions = {len(recording.mean_latent_embedding) for recording in self.recordings}
        if len(dimensions) != 1:
            raise ValueError("all recording embeddings must have the same dimensions")

    @property
    def recording_ids(self) -> tuple[str, ...]:
        return tuple(recording.recording_id for recording in self.recordings)

    @property
    def latent_dimensions(self) -> int:
        return len(self.recordings[0].mean_latent_embedding)

    def to_record(self) -> dict[str, Any]:
        return {
            "source_role": self.source_role,
            "recording_count": len(self.recordings),
            "recordings": [recording.to_record() for recording in self.recordings],
        }


@dataclass(frozen=True, slots=True)
class LatentReference:
    fit_role: str
    recording_ids: tuple[str, ...]
    coordinate_mean: tuple[float, ...]
    population_variance: tuple[float, ...]
    coordinate_scale: tuple[float, ...]
    standardized_embeddings: tuple[tuple[float, ...], ...]
    nearest_neighbours: int = NEAREST_NEIGHBOURS

    def __post_init__(self) -> None:
        if self.fit_role != KNOWN_TRAINING_ROLE:
            raise ValueError("latent reference fit role must be known_training")
        if self.nearest_neighbours != NEAREST_NEIGHBOURS:
            raise ValueError("latent reference must use exactly 10 nearest neighbours")
        if type(self.recording_ids) is not tuple or len(self.recording_ids) < NEAREST_NEIGHBOURS:
            raise ValueError("latent reference requires at least 10 recording identities")
        for recording_id in self.recording_ids:
            _validated_identifier(recording_id, "recording_id")
        if len(set(self.recording_ids)) != len(self.recording_ids):
            raise ValueError("latent reference recording identities must be unique")
        if tuple(sorted(self.recording_ids)) != self.recording_ids:
            raise ValueError("latent reference identities must use lexical order")

        dimensions = len(self.coordinate_mean)
        if dimensions <= 0:
            raise ValueError("latent reference coordinates cannot be empty")
        if (
            type(self.coordinate_mean) is not tuple
            or type(self.population_variance) is not tuple
            or type(self.coordinate_scale) is not tuple
            or len(self.population_variance) != dimensions
            or len(self.coordinate_scale) != dimensions
        ):
            raise ValueError("latent reference coordinate vectors have inconsistent dimensions")
        for index, (mean, variance, scale) in enumerate(
            zip(
                self.coordinate_mean,
                self.population_variance,
                self.coordinate_scale,
                strict=True,
            )
        ):
            _validated_real(mean, f"coordinate_mean[{index}]")
            resolved_variance = _validated_real(
                variance,
                f"population_variance[{index}]",
                nonnegative=True,
            )
            resolved_scale = _validated_real(
                scale,
                f"coordinate_scale[{index}]",
                nonnegative=True,
            )
            if resolved_scale <= 0.0:
                raise ValueError("latent reference coordinate scales must be positive")
            expected_scale = 1.0 if resolved_variance == 0.0 else math.sqrt(resolved_variance)
            if resolved_scale != expected_scale:
                raise ValueError("latent reference coordinate scale does not match its variance")

        if type(self.standardized_embeddings) is not tuple or len(
            self.standardized_embeddings
        ) != len(self.recording_ids):
            raise ValueError("standardized reference embeddings do not match recording identities")
        for row in self.standardized_embeddings:
            if type(row) is not tuple or len(row) != dimensions:
                raise ValueError("standardized reference embeddings have inconsistent dimensions")
            for coordinate in row:
                _validated_real(coordinate, "standardized latent coordinate")

    @property
    def latent_dimensions(self) -> int:
        return len(self.coordinate_mean)

    def to_record(self) -> dict[str, Any]:
        return {
            "fit_role": self.fit_role,
            "recording_ids": list(self.recording_ids),
            "recording_count": len(self.recording_ids),
            "coordinate_mean": [float(value) for value in self.coordinate_mean],
            "population_variance": [float(value) for value in self.population_variance],
            "coordinate_scale": [float(value) for value in self.coordinate_scale],
            "standardized_embeddings": [
                [float(value) for value in row] for row in self.standardized_embeddings
            ],
            "nearest_neighbours": self.nearest_neighbours,
        }


@dataclass(frozen=True, slots=True)
class LatentNoveltyScore:
    recording_id: str
    score: float
    neighbour_recording_ids: tuple[str, ...]
    neighbour_distances: tuple[float, ...]
    direction: str = NOVELTY_DIRECTION

    def __post_init__(self) -> None:
        _validated_identifier(self.recording_id, "recording_id")
        resolved_score = _validated_real(self.score, "latent novelty score", nonnegative=True)
        if self.direction != NOVELTY_DIRECTION:
            raise ValueError("latent novelty direction must be higher_is_more_novel")
        if (
            type(self.neighbour_recording_ids) is not tuple
            or type(self.neighbour_distances) is not tuple
            or len(self.neighbour_recording_ids) != NEAREST_NEIGHBOURS
            or len(self.neighbour_distances) != NEAREST_NEIGHBOURS
        ):
            raise ValueError("a latent novelty score must contain exactly 10 neighbours")
        for recording_id in self.neighbour_recording_ids:
            _validated_identifier(recording_id, "neighbour recording_id")
        if len(set(self.neighbour_recording_ids)) != NEAREST_NEIGHBOURS:
            raise ValueError("latent novelty neighbour identities must be unique")
        distances = tuple(
            _validated_real(value, "neighbour distance", nonnegative=True)
            for value in self.neighbour_distances
        )
        if tuple(sorted(distances)) != distances:
            raise ValueError("latent novelty neighbours must use ascending distance order")
        if not math.isclose(resolved_score, math.fsum(distances) / NEAREST_NEIGHBOURS):
            raise ValueError("latent novelty score must equal the mean neighbour distance")

    def to_record(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "score": float(self.score),
            "direction": self.direction,
            "neighbour_recording_ids": list(self.neighbour_recording_ids),
            "neighbour_distances": [float(value) for value in self.neighbour_distances],
        }


@dataclass(frozen=True, slots=True)
class NoveltyThreshold:
    value: float
    calibration_role: str
    calibration_recording_ids: tuple[str, ...]
    score_name: str = RECONSTRUCTION_SCORE_NAME
    quantile: float = THRESHOLD_QUANTILE
    method: str = THRESHOLD_QUANTILE_METHOD
    direction: str = NOVELTY_DIRECTION
    classification_operator: str = THRESHOLD_OPERATOR

    def __post_init__(self) -> None:
        _validated_real(self.value, "novelty threshold", nonnegative=True)
        if self.score_name not in LOCKED_SCORE_NAMES:
            raise ValueError(f"threshold score_name must be one of {LOCKED_SCORE_NAMES}")
        if self.calibration_role != KNOWN_VALIDATION_ROLE:
            raise ValueError("novelty threshold must use known_validation calibration")
        if self.quantile != THRESHOLD_QUANTILE or self.method != THRESHOLD_QUANTILE_METHOD:
            raise ValueError("novelty threshold must use q=0.95 with method=higher")
        if self.direction != NOVELTY_DIRECTION:
            raise ValueError("novelty threshold direction must be higher_is_more_novel")
        if self.classification_operator != THRESHOLD_OPERATOR:
            raise ValueError("novelty threshold classification operator must be strict >")
        if type(self.calibration_recording_ids) is not tuple or not self.calibration_recording_ids:
            raise ValueError("calibration_recording_ids must be a nonempty tuple")
        for recording_id in self.calibration_recording_ids:
            _validated_identifier(recording_id, "calibration recording_id")
        if len(set(self.calibration_recording_ids)) != len(self.calibration_recording_ids):
            raise ValueError("calibration recording identities must be unique")
        if tuple(sorted(self.calibration_recording_ids)) != self.calibration_recording_ids:
            raise ValueError("calibration recording identities must use lexical order")

    def to_record(self) -> dict[str, Any]:
        return {
            "score_name": self.score_name,
            "value": float(self.value),
            "calibration_role": self.calibration_role,
            "calibration_recording_ids": list(self.calibration_recording_ids),
            "quantile": self.quantile,
            "method": self.method,
            "direction": self.direction,
            "classification_operator": self.classification_operator,
        }


# This compatibility alias preserves the original reconstruction-only public API.
ReconstructionThreshold = NoveltyThreshold


@dataclass(frozen=True, slots=True)
class NoveltyDecision:
    recording_id: str
    score: float
    threshold: float
    is_novel: bool
    score_name: str = RECONSTRUCTION_SCORE_NAME
    direction: str = NOVELTY_DIRECTION

    def __post_init__(self) -> None:
        _validated_identifier(self.recording_id, "recording_id")
        resolved_score = _validated_real(self.score, "novelty score", nonnegative=True)
        resolved_threshold = _validated_real(
            self.threshold,
            "novelty threshold",
            nonnegative=True,
        )
        if type(self.is_novel) is not bool:
            raise TypeError("is_novel must be a boolean")
        if self.score_name not in LOCKED_SCORE_NAMES:
            raise ValueError(f"decision score_name must be one of {LOCKED_SCORE_NAMES}")
        if self.is_novel is not (resolved_score > resolved_threshold):
            raise ValueError("novelty decision must use strict score > threshold classification")
        if self.direction != NOVELTY_DIRECTION:
            raise ValueError("novelty decision direction must be higher_is_more_novel")

    def to_record(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "score_name": self.score_name,
            "score": float(self.score),
            "threshold": float(self.threshold),
            "is_novel": self.is_novel,
            "direction": self.direction,
        }


def clip_reconstruction_mse(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Return one float64 pixel-mean reconstruction MSE for every clip."""

    resolved_inputs = _validated_float_array(inputs, "inputs")
    resolved_reconstructions = _validated_float_array(reconstructions, "reconstructions")
    if resolved_inputs.ndim < 2:
        raise ValueError("inputs and reconstructions require a clip axis and pixel axes")
    if resolved_inputs.shape != resolved_reconstructions.shape:
        raise ValueError("inputs and reconstructions must have identical shapes")

    with np.errstate(over="ignore", invalid="ignore"):
        residual = resolved_inputs.astype(np.float64) - resolved_reconstructions.astype(np.float64)
        values = np.mean(
            np.square(residual),
            axis=tuple(range(1, residual.ndim)),
            dtype=np.float64,
        )
    if values.shape != (resolved_inputs.shape[0],) or not bool(np.all(np.isfinite(values))):
        raise ValueError("reconstruction MSE produced nonfinite values")
    values.setflags(write=False)
    return values


def _validated_clip_identities(
    identities: object,
    expected_count: int,
) -> tuple[ClipIdentity, ...]:
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("clip identities must be a sequence")
    resolved = tuple(identities)
    if len(resolved) != expected_count:
        raise ValueError("clip identities do not match the clip tensor rows")
    if any(not isinstance(identity, ClipIdentity) for identity in resolved):
        raise TypeError("clip identities must contain only ClipIdentity values")
    keys = tuple((identity.recording_id, identity.clip_id) for identity in resolved)
    if len(set(keys)) != len(keys):
        raise ValueError("clip identity pairs must be unique")
    return resolved


def aggregate_recordings(
    identities: Sequence[ClipIdentity],
    reconstruction_mse: np.ndarray,
    latent_embeddings: np.ndarray,
    *,
    source_role: str,
) -> RecordingBatch:
    """Bind clip results to metadata and aggregate them in deterministic identity order."""

    role = _validated_identifier(source_role, "source_role")
    resolved_mse = _validated_float_array(
        reconstruction_mse,
        "reconstruction_mse",
        dimensions=1,
    )
    if bool(np.any(resolved_mse < 0.0)):
        raise ValueError("reconstruction_mse cannot contain negative values")
    resolved_latent = _validated_float_array(
        latent_embeddings,
        "latent_embeddings",
        dimensions=2,
    )
    if resolved_latent.shape[0] != resolved_mse.shape[0]:
        raise ValueError("latent embeddings and reconstruction MSE rows do not match")
    resolved_identities = _validated_clip_identities(identities, resolved_mse.shape[0])

    grouped: dict[str, list[tuple[str, int]]] = {}
    for index, identity in enumerate(resolved_identities):
        grouped.setdefault(identity.recording_id, []).append((identity.clip_id, index))

    recordings: list[RecordingScore] = []
    mse64 = resolved_mse.astype(np.float64, copy=False)
    latent64 = resolved_latent.astype(np.float64, copy=False)
    for recording_id in sorted(grouped):
        ordered = sorted(grouped[recording_id], key=lambda item: item[0])
        clip_ids = tuple(item[0] for item in ordered)
        indices = np.asarray([item[1] for item in ordered], dtype=np.intp)
        recording_mse = float(np.median(mse64[indices]))
        recording_latent = np.mean(latent64[indices], axis=0, dtype=np.float64)
        if not math.isfinite(recording_mse) or not bool(np.all(np.isfinite(recording_latent))):
            raise ValueError("recording aggregation produced nonfinite values")
        recordings.append(
            RecordingScore(
                recording_id=recording_id,
                clip_ids=clip_ids,
                reconstruction_mse=recording_mse,
                mean_latent_embedding=tuple(float(value) for value in recording_latent),
            )
        )
    return RecordingBatch(source_role=role, recordings=tuple(recordings))


def score_recording_batch(
    identities: Sequence[ClipIdentity],
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    latent_embeddings: np.ndarray,
    *,
    source_role: str,
) -> RecordingBatch:
    """Compute clip MSE values and aggregate both score types by recording."""

    mse = clip_reconstruction_mse(inputs, reconstructions)
    return aggregate_recordings(
        identities,
        mse,
        latent_embeddings,
        source_role=source_role,
    )


def fit_known_training_reference(training: RecordingBatch) -> LatentReference:
    """Fit float64 population standardization from known-training recordings only."""

    if not isinstance(training, RecordingBatch):
        raise TypeError("training must be a RecordingBatch")
    if training.source_role != KNOWN_TRAINING_ROLE:
        raise PermissionError("latent standardization can fit only known_training recordings")
    if len(training.recordings) < NEAREST_NEIGHBOURS:
        raise ValueError("latent reference requires at least 10 known-training recordings")

    embeddings = np.asarray(
        [recording.mean_latent_embedding for recording in training.recordings],
        dtype=np.float64,
    )
    if embeddings.ndim != 2 or not bool(np.all(np.isfinite(embeddings))):
        raise ValueError("known-training recording embeddings are invalid")
    coordinate_mean = np.mean(embeddings, axis=0, dtype=np.float64)
    centered = embeddings - coordinate_mean
    with np.errstate(over="ignore", invalid="ignore"):
        population_variance = np.mean(np.square(centered), axis=0, dtype=np.float64)
    coordinate_scale = np.sqrt(population_variance)
    coordinate_scale[population_variance == 0.0] = 1.0
    standardized = centered / coordinate_scale
    if not (
        bool(np.all(np.isfinite(coordinate_mean)))
        and bool(np.all(np.isfinite(population_variance)))
        and bool(np.all(np.isfinite(coordinate_scale)))
        and bool(np.all(np.isfinite(standardized)))
    ):
        raise ValueError("known-training standardization produced nonfinite values")

    return LatentReference(
        fit_role=training.source_role,
        recording_ids=training.recording_ids,
        coordinate_mean=tuple(float(value) for value in coordinate_mean),
        population_variance=tuple(float(value) for value in population_variance),
        coordinate_scale=tuple(float(value) for value in coordinate_scale),
        standardized_embeddings=tuple(tuple(float(value) for value in row) for row in standardized),
    )


def latent_knn_novelty_scores(
    reference: LatentReference,
    queries: RecordingBatch,
) -> tuple[LatentNoveltyScore, ...]:
    """Score each recording by brute-force mean distance to 10 training neighbours."""

    if not isinstance(reference, LatentReference):
        raise TypeError("reference must be a LatentReference")
    if not isinstance(queries, RecordingBatch):
        raise TypeError("queries must be a RecordingBatch")
    if queries.latent_dimensions != reference.latent_dimensions:
        raise ValueError("query and reference latent dimensions do not match")
    overlap = set(reference.recording_ids).intersection(queries.recording_ids)
    if overlap:
        raise PermissionError("query recording identities overlap the known-training reference")

    coordinate_mean = np.asarray(reference.coordinate_mean, dtype=np.float64)
    coordinate_scale = np.asarray(reference.coordinate_scale, dtype=np.float64)
    training_embeddings = np.asarray(reference.standardized_embeddings, dtype=np.float64)
    query_embeddings = np.asarray(
        [recording.mean_latent_embedding for recording in queries.recordings],
        dtype=np.float64,
    )
    standardized_queries = (query_embeddings - coordinate_mean) / coordinate_scale
    if not bool(np.all(np.isfinite(standardized_queries))):
        raise ValueError("query standardization produced nonfinite values")

    results: list[LatentNoveltyScore] = []
    for recording, query in zip(queries.recordings, standardized_queries, strict=True):
        with np.errstate(over="ignore", invalid="ignore"):
            delta = training_embeddings - query
            distances = np.sqrt(np.sum(np.square(delta), axis=1, dtype=np.float64))
        if not bool(np.all(np.isfinite(distances))):
            raise ValueError("latent distance calculation produced nonfinite values")
        neighbour_indices = sorted(
            range(len(reference.recording_ids)),
            key=lambda index: (float(distances[index]), reference.recording_ids[index]),
        )[:NEAREST_NEIGHBOURS]
        neighbour_ids = tuple(reference.recording_ids[index] for index in neighbour_indices)
        neighbour_distances = tuple(float(distances[index]) for index in neighbour_indices)
        results.append(
            LatentNoveltyScore(
                recording_id=recording.recording_id,
                score=math.fsum(neighbour_distances) / NEAREST_NEIGHBOURS,
                neighbour_recording_ids=neighbour_ids,
                neighbour_distances=neighbour_distances,
            )
        )
    return tuple(results)


def _require_known_validation(validation: object) -> RecordingBatch:
    if not isinstance(validation, RecordingBatch):
        raise TypeError("validation must be a RecordingBatch")
    if validation.source_role != KNOWN_VALIDATION_ROLE:
        raise PermissionError("novelty thresholds can fit only known_validation recordings")
    return validation


def _fit_novelty_threshold(
    *,
    score_name: str,
    calibration_recording_ids: tuple[str, ...],
    values: np.ndarray,
) -> NoveltyThreshold:
    if score_name not in LOCKED_SCORE_NAMES:
        raise ValueError(f"score_name must be one of {LOCKED_SCORE_NAMES}")
    resolved_values = _validated_float_array(values, "threshold values", dimensions=1)
    if resolved_values.shape != (len(calibration_recording_ids),):
        raise ValueError("threshold values do not match calibration recording identities")
    if bool(np.any(resolved_values < 0.0)):
        raise ValueError("threshold values cannot be negative")
    value = float(
        np.quantile(
            resolved_values.astype(np.float64, copy=False),
            THRESHOLD_QUANTILE,
            method=THRESHOLD_QUANTILE_METHOD,
        )
    )
    if not math.isfinite(value):
        raise ValueError("novelty threshold is not finite")
    return NoveltyThreshold(
        score_name=score_name,
        value=value,
        calibration_role=KNOWN_VALIDATION_ROLE,
        calibration_recording_ids=calibration_recording_ids,
    )


def _validated_bound_latent_scores(
    recordings: RecordingBatch,
    latent_scores: object,
) -> tuple[LatentNoveltyScore, ...]:
    if isinstance(latent_scores, (str, bytes)) or not isinstance(latent_scores, Sequence):
        raise TypeError("latent_scores must be a sequence")
    resolved = tuple(latent_scores)
    if any(not isinstance(score, LatentNoveltyScore) for score in resolved):
        raise TypeError("latent_scores must contain only LatentNoveltyScore values")
    score_ids = tuple(score.recording_id for score in resolved)
    if score_ids != recordings.recording_ids:
        raise ValueError("latent score identities and order must exactly match the recording batch")
    query_ids = set(recordings.recording_ids)
    neighbour_ids = {
        neighbour_id for score in resolved for neighbour_id in score.neighbour_recording_ids
    }
    if query_ids.intersection(neighbour_ids):
        raise PermissionError("latent score neighbours overlap the query recording identities")
    return resolved


def fit_known_validation_reconstruction_threshold(
    validation: RecordingBatch,
) -> NoveltyThreshold:
    """Fit the reconstruction threshold on known validation recording scores."""

    resolved_validation = _require_known_validation(validation)
    scores = np.asarray(
        [recording.reconstruction_mse for recording in resolved_validation.recordings],
        dtype=np.float64,
    )
    return _fit_novelty_threshold(
        score_name=RECONSTRUCTION_SCORE_NAME,
        calibration_recording_ids=resolved_validation.recording_ids,
        values=scores,
    )


def fit_known_validation_latent_threshold(
    validation: RecordingBatch,
    latent_scores: Sequence[LatentNoveltyScore],
) -> NoveltyThreshold:
    """Fit the latent-distance threshold on identity-bound known validation scores."""

    resolved_validation = _require_known_validation(validation)
    resolved_scores = _validated_bound_latent_scores(resolved_validation, latent_scores)
    values = np.asarray([score.score for score in resolved_scores], dtype=np.float64)
    return _fit_novelty_threshold(
        score_name=LATENT_SCORE_NAME,
        calibration_recording_ids=resolved_validation.recording_ids,
        values=values,
    )


def fit_known_validation_threshold(validation: RecordingBatch) -> NoveltyThreshold:
    """Compatibility wrapper for the reconstruction threshold."""

    return fit_known_validation_reconstruction_threshold(validation)


def _classify_bound_scores(
    *,
    score_name: str,
    recording_ids: tuple[str, ...],
    values: tuple[float, ...],
    threshold: NoveltyThreshold,
) -> tuple[NoveltyDecision, ...]:
    if not isinstance(threshold, NoveltyThreshold):
        raise TypeError("threshold must be a NoveltyThreshold")
    if threshold.score_name != score_name:
        raise ValueError(
            f"threshold score_name {threshold.score_name!r} does not match {score_name!r}"
        )
    if len(recording_ids) != len(values):
        raise ValueError("score values do not match recording identities")
    return tuple(
        NoveltyDecision(
            recording_id=recording_id,
            score=float(score),
            threshold=float(threshold.value),
            is_novel=float(score) > float(threshold.value),
            score_name=score_name,
        )
        for recording_id, score in zip(recording_ids, values, strict=True)
    )


def classify_reconstruction_scores(
    recordings: RecordingBatch,
    threshold: NoveltyThreshold,
) -> tuple[NoveltyDecision, ...]:
    """Classify strictly above-threshold reconstruction scores as novel."""

    if not isinstance(recordings, RecordingBatch):
        raise TypeError("recordings must be a RecordingBatch")
    return _classify_bound_scores(
        score_name=RECONSTRUCTION_SCORE_NAME,
        recording_ids=recordings.recording_ids,
        values=tuple(float(recording.reconstruction_mse) for recording in recordings.recordings),
        threshold=threshold,
    )


def classify_latent_scores(
    recordings: RecordingBatch,
    latent_scores: Sequence[LatentNoveltyScore],
    threshold: NoveltyThreshold,
) -> tuple[NoveltyDecision, ...]:
    """Classify identity-bound latent scores using their separate locked threshold."""

    if not isinstance(recordings, RecordingBatch):
        raise TypeError("recordings must be a RecordingBatch")
    resolved_scores = _validated_bound_latent_scores(recordings, latent_scores)
    return _classify_bound_scores(
        score_name=LATENT_SCORE_NAME,
        recording_ids=recordings.recording_ids,
        values=tuple(float(score.score) for score in resolved_scores),
        threshold=threshold,
    )
