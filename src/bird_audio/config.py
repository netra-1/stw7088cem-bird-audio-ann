from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path
from typing import Any

from bird_audio.hashing import sha256_json
from bird_audio.paths import resolve_project_path


class ConfigValidationError(ValueError):
    pass


DATA_TOP_LEVEL = {
    "schema_version",
    "raw_audio_dir",
    "local_manifest",
    "enriched_manifest",
    "split_manifest",
    "split_summary",
    "split_lock",
    "minimum_source_sample_rate_hz",
    "target_sample_rate_hz",
    "target_channels",
    "audio_dtype",
    "clip_duration_seconds",
    "split_seed",
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "quality_control",
    "session_grouping",
    "clip_selection",
    "spectrogram",
    "known_species",
    "unknown_species",
    "fallback_unknown_species",
}
DATA_QUALITY_CONTROL = {
    "full_decode_warning_policy",
    "minimum_decoded_to_ffprobe_duration_ratio",
    "maximum_decoded_to_ffprobe_duration_ratio",
}
DATA_SESSION_GROUPING = {
    "coordinate_radius_km",
    "missing_date_policy",
    "missing_recordist_policy",
    "missing_location_policy",
    "same_individual_reference_policy",
}
DATA_CLIP_SELECTION = {
    "maximum_clips_per_recording",
    "uniform_clip_count_formula",
    "uniform_start_rule",
    "energy_candidate_hop_seconds",
    "include_end_aligned_candidate",
    "energy_measure",
    "tie_break",
    "minimum_selected_start_separation_seconds",
    "epoch_draws",
}
DATA_SPECTROGRAM = {
    "n_fft",
    "win_length",
    "hop_length",
    "window",
    "center",
    "n_mels",
    "f_min_hz",
    "f_max_hz",
    "power",
    "mel_scale",
    "htk",
    "mel_normalization",
    "power_to_db_reference",
    "power_to_db_amin",
    "minimum_db",
    "maximum_db",
    "output_height",
    "output_width",
    "expected_native_height",
    "expected_native_width",
    "resize_mode",
    "resize_align_corners",
    "resize_antialias",
}
TASK1_TOP_LEVEL = {
    "task",
    "architecture",
    "pretrained_weights",
    "pretrained_weight_cache_hash_required",
    "class_count",
    "dropout",
    "aggregation",
    "primary_metric",
    "zero_division",
    "seeds",
    "class_order",
    "rung",
    "description",
    "trainable_backbone_from_block",
    "trainable_feature_indices",
    "training",
    "sampling",
    "augmentation",
    "loss",
}
TASK1_TRAINING = {
    "optimizer",
    "scheduler",
    "maximum_epochs",
    "early_stopping_patience",
    "batch_size",
    "weight_decay",
    "head_learning_rate",
    "backbone_learning_rate",
    "pin_memory",
    "num_workers",
    "mixed_precision",
    "dtype",
    "device_preference",
    "allow_mps_fallback",
    "request_deterministic_algorithms",
    "determinism_failure_policy",
    "seed_python",
    "seed_numpy",
    "seed_torch",
    "seed_sampler",
    "checkpoint_metric",
    "checkpoint_mode",
    "checkpoint_tie_break_1",
    "checkpoint_tie_break_2",
    "log_parameter_counts",
}
TASK1_SAMPLING = {"maximum_clips_per_recording", "recording_balanced_weights", "strategy"}
TASK1_AUGMENTATION = {
    "specaugment",
    "frequency_mask_max_bins",
    "time_mask_max_frames",
    "frequency_mask_probability",
    "time_mask_probability",
    "fill_value",
}
TASK1_LOSS = {"name", "reduction"}
TASK2_TOP_LEVEL = {
    "task",
    "architecture",
    "input_channels",
    "input_height",
    "input_width",
    "encoder_channels",
    "kernel_size",
    "stride",
    "padding",
    "latent_dimensions",
    "hidden_activation",
    "normalization",
    "bottleneck_activation",
    "transpose_convolution_output_padding",
    "decoder_output_activation",
    "loss",
    "loss_reduction",
    "clip_selection_strategy",
    "seeds",
    "training",
    "novelty",
}
TASK2_TRAINING = {
    "optimizer",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "maximum_epochs",
    "early_stopping_patience",
    "pin_memory",
    "num_workers",
    "mixed_precision",
    "dtype",
    "device_preference",
    "allow_mps_fallback",
    "request_deterministic_algorithms",
    "determinism_failure_policy",
    "seed_python",
    "seed_numpy",
    "seed_torch",
    "seed_sampler",
    "scheduler",
    "checkpoint_metric",
    "checkpoint_mode",
}
TASK2_NOVELTY = {
    "primary_score",
    "secondary_readout",
    "latent_reference_unit",
    "latent_standardization_unit",
    "nearest_neighbours",
    "threshold_quantile",
    "threshold_quantile_method",
    "threshold_scope",
    "score_direction",
    "threshold_operator",
    "bootstrap_seed",
    "bootstrap_replicates",
    "bootstrap_interval_method",
    "bootstrap_confidence_level",
    "bootstrap_resampling_unit",
    "detailed_figure_seed",
}
LOCKED_UNKNOWN_SPECIES = (
    ("Brown-headed Barbet", "Psilopogon zeylanicus", "family_matched"),
    ("Jungle Myna", "Acridotheres fuscus", "family_matched"),
    ("Pied Kingfisher", "Ceryle rudis", "family_matched"),
    ("House Crow", "Corvus splendens", "other_family"),
    ("Grey Francolin", "Ortygornis pondicerianus", "other_family"),
)
LOCKED_FALLBACK_UNKNOWN_SPECIES = (("Oriental Turtle Dove", "Streptopelia orientalis"),)
LOCKED_TASK1_CLASS_ORDER = (
    "Asian Koel",
    "Black Drongo",
    "Blue-throated Barbet",
    "Common Cuckoo",
    "Common Iora",
    "Common Kingfisher",
    "Common Myna",
    "Common Tailorbird",
    "Eurasian Hoopoe",
    "Great Barbet",
    "Greater Coucal",
    "Red-vented Bulbul",
    "Rose-ringed Parakeet",
    "Spotted Dove",
    "White-throated Kingfisher",
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _unknown_keys(mapping: dict[str, Any], allowed: set[str], context: str) -> list[str]:
    unknown = sorted(set(mapping) - allowed)
    return [f"{context}: unknown key {key!r}" for key in unknown]


def _positive(value: Any, context: str, issues: list[str]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        issues.append(f"{context} must be positive")


def _validate_data_config(config: dict[str, Any]) -> list[str]:
    issues = _unknown_keys(config, DATA_TOP_LEVEL, "data")
    clip = config.get("clip_selection", {})
    quality_control = config.get("quality_control", {})
    session_grouping = config.get("session_grouping", {})
    spectrogram = config.get("spectrogram", {})
    issues.extend(_unknown_keys(clip, DATA_CLIP_SELECTION, "data.clip_selection"))
    issues.extend(_unknown_keys(quality_control, DATA_QUALITY_CONTROL, "data.quality_control"))
    issues.extend(_unknown_keys(session_grouping, DATA_SESSION_GROUPING, "data.session_grouping"))
    issues.extend(_unknown_keys(spectrogram, DATA_SPECTROGRAM, "data.spectrogram"))
    if config.get("raw_audio_dir") != "dataset":
        issues.append("data.raw_audio_dir is locked to the immutable project dataset directory")
    if quality_control.get("full_decode_warning_policy") != "exclude":
        issues.append("data full-decode warning policy must be exclude")
    minimum_ratio = float(quality_control.get("minimum_decoded_to_ffprobe_duration_ratio", 0))
    maximum_ratio = float(quality_control.get("maximum_decoded_to_ffprobe_duration_ratio", 0))
    if not 0 < minimum_ratio <= 1 <= maximum_ratio:
        issues.append("data decoded-duration ratio bounds must enclose 1")
    coordinate_radius = session_grouping.get("coordinate_radius_km")
    _positive(coordinate_radius, "data.session_grouping.coordinate_radius_km", issues)
    if (
        isinstance(coordinate_radius, (int, float))
        and not isinstance(coordinate_radius, bool)
        and float(coordinate_radius) > 5
    ):
        issues.append("data session coordinate radius must not exceed 5 km")
    locked_session_rules = {
        "missing_date_policy": "manual_review_and_conservative_grouping",
        "missing_recordist_policy": "manual_review_and_conservative_grouping",
        "missing_location_policy": "connect_within_recordist_date",
        "same_individual_reference_policy": "global_link_or_manual_review",
    }
    for key, expected in locked_session_rules.items():
        if session_grouping.get(key) != expected:
            issues.append(f"data.session_grouping.{key} must be {expected!r}")

    fractions = [
        float(config.get("train_fraction", 0)),
        float(config.get("validation_fraction", 0)),
        float(config.get("test_fraction", 0)),
    ]
    if any(value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
        issues.append("data split fractions must be positive and sum to 1")

    sample_rate = int(config.get("target_sample_rate_hz", 0))
    clip_seconds = float(config.get("clip_duration_seconds", 0))
    _positive(sample_rate, "data.target_sample_rate_hz", issues)
    _positive(clip_seconds, "data.clip_duration_seconds", issues)
    for key in ("n_fft", "win_length", "hop_length", "n_mels"):
        _positive(spectrogram.get(key), f"data.spectrogram.{key}", issues)
    if int(spectrogram.get("win_length", 0)) > int(spectrogram.get("n_fft", 0)):
        issues.append("data.spectrogram.win_length cannot exceed n_fft")
    if float(spectrogram.get("f_max_hz", 0)) >= sample_rate / 2:
        issues.append("data.spectrogram.f_max_hz must be below Nyquist")
    if spectrogram.get("center") is not False:
        issues.append("data.spectrogram.center must be false")
    locked_spectrogram_values = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 128,
        "expected_native_height": 128,
        "expected_native_width": 372,
    }
    for key, expected in locked_spectrogram_values.items():
        if spectrogram.get(key) != expected:
            issues.append(f"data.spectrogram.{key} must be {expected!r}")

    if sample_rate > 0 and clip_seconds > 0 and int(spectrogram.get("n_fft", 0)) > 0:
        samples = round(sample_rate * clip_seconds)
        frames = 1 + (samples - int(spectrogram["n_fft"])) // int(spectrogram["hop_length"])
        if frames != int(spectrogram.get("expected_native_width", -1)):
            issues.append(f"data expected native width must be {frames}")
    if int(spectrogram.get("n_mels", 0)) != int(spectrogram.get("expected_native_height", -1)):
        issues.append("data expected native height must equal n_mels")

    known = config.get("known_species") or []
    folders = [entry.get("folder") for entry in known]
    names = [entry.get("common_name") for entry in known]
    if len(known) != 15 or len(set(folders)) != 15 or len(set(names)) != 15:
        issues.append("data must define 15 unique known species and folders")
    for index, entry in enumerate(known):
        issues.extend(
            _unknown_keys(
                entry, {"folder", "common_name", "scientific_name"}, f"known_species[{index}]"
            )
        )
    unknown = config.get("unknown_species") or []
    fallback = config.get("fallback_unknown_species") or []
    for index, entry in enumerate(unknown):
        allowed = {"common_name", "scientific_name", "target_recordings", "difficulty_group"}
        issues.extend(_unknown_keys(entry, allowed, f"unknown_species[{index}]"))
    for index, entry in enumerate(fallback):
        allowed = {"common_name", "scientific_name", "target_recordings"}
        issues.extend(_unknown_keys(entry, allowed, f"fallback_unknown_species[{index}]"))

    observed_unknown = tuple(
        (entry.get("common_name"), entry.get("scientific_name"), entry.get("difficulty_group"))
        for entry in unknown
    )
    if observed_unknown != LOCKED_UNKNOWN_SPECIES:
        issues.append("data unknown species and difficulty groups must match the locked protocol")
    observed_fallback = tuple(
        (entry.get("common_name"), entry.get("scientific_name")) for entry in fallback
    )
    if observed_fallback != LOCKED_FALLBACK_UNKNOWN_SPECIES:
        issues.append("data fallback unknown species must match the locked protocol")

    all_species = [*known, *unknown, *fallback]
    common_names = [str(entry.get("common_name") or "").casefold() for entry in all_species]
    scientific_names = [str(entry.get("scientific_name") or "").casefold() for entry in all_species]
    if len(set(common_names)) != len(all_species):
        issues.append("data study-species common names must be unique")
    if len(set(scientific_names)) != len(all_species):
        issues.append("data study-species scientific names must be unique")
    for index, entry in enumerate([*unknown, *fallback]):
        target = entry.get("target_recordings")
        if isinstance(target, bool) or not isinstance(target, int) or target != 40:
            issues.append(f"data unknown target_recordings[{index}] must be the integer 40")
        scientific = str(entry.get("scientific_name") or "")
        parts = scientific.split()
        if len(parts) != 2 or not all(part.isalpha() for part in parts):
            issues.append(f"data unknown scientific_name[{index}] must be a strict binomial")
    return issues


def _validate_task1_config(config: dict[str, Any]) -> list[str]:
    issues = _unknown_keys(config, TASK1_TOP_LEVEL, "task1")
    training = config.get("training", {})
    sampling = config.get("sampling", {})
    augmentation = config.get("augmentation", {})
    loss = config.get("loss", {})
    issues.extend(_unknown_keys(training, TASK1_TRAINING, "task1.training"))
    issues.extend(_unknown_keys(sampling, TASK1_SAMPLING, "task1.sampling"))
    issues.extend(_unknown_keys(augmentation, TASK1_AUGMENTATION, "task1.augmentation"))
    issues.extend(_unknown_keys(loss, TASK1_LOSS, "task1.loss"))

    locked_top_level = {
        "task": "classification",
        "architecture": "efficientnet_b0",
        "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "pretrained_weight_cache_hash_required": True,
        "class_count": 15,
        "dropout": 0.2,
        "aggregation": "mean_logits",
        "primary_metric": "recording_macro_f1",
        "zero_division": 0,
        "seeds": [13, 37, 71],
        "class_order": list(LOCKED_TASK1_CLASS_ORDER),
        "rung": "final",
        "description": "Locked EfficientNet-B0 coursework configuration",
        "trainable_backbone_from_block": 6,
        "trainable_feature_indices": [6, 7, 8],
    }
    for key, expected in locked_top_level.items():
        observed = config.get(key)
        if type(observed) is not type(expected) or observed != expected:
            issues.append(f"task1 {key} must be {expected!r}")

    if config.get("task") != "classification":
        issues.append("task1 task must be classification")
    if config.get("architecture") != "efficientnet_b0":
        issues.append("task1 architecture must be efficientnet_b0")
    if config.get("pretrained_weights") != "EfficientNet_B0_Weights.IMAGENET1K_V1":
        issues.append("task1 pretrained weight identifier is not locked")
    if config.get("pretrained_weight_cache_hash_required") is not True:
        issues.append("task1 pretrained weight cache hash must be required")
    class_count = int(config.get("class_count", 0))
    class_order = config.get("class_order") or []
    if class_count != 15 or len(class_order) != class_count or len(set(class_order)) != class_count:
        issues.append("task1 class count and class order must contain 15 unique entries")
    for key in ("maximum_epochs", "batch_size", "head_learning_rate"):
        _positive(training.get(key), f"task1.training.{key}", issues)
    if int(training.get("early_stopping_patience", 0)) <= 0:
        issues.append("task1 early-stopping patience must be positive")
    if loss.get("name") != "cross_entropy":
        issues.append("task1 loss must be cross_entropy")
    if loss.get("reduction") != "mean":
        issues.append("task1 loss reduction must be mean")
    if sampling.get("strategy") != "energy":
        issues.append("task1 sampling strategy must be energy")
    if sampling.get("recording_balanced_weights") is not True:
        issues.append("task1 recording-balanced sampling must be enabled")
    if sampling.get("maximum_clips_per_recording") != 5:
        issues.append("task1 maximum clips per recording must be 5")
    if config.get("trainable_backbone_from_block") != 6 or config.get(
        "trainable_feature_indices"
    ) != [6, 7, 8]:
        issues.append("task1 partial fine-tuning must use feature indices 6, 7, and 8")
    if config.get("rung") != "final":
        issues.append("task1 run label must be final")
    if config.get("seeds") != [13, 37, 71]:
        issues.append("task1 stability seeds must be [13, 37, 71]")
    if augmentation.get("specaugment") is not True:
        issues.append("task1 SpecAugment must be enabled")
    locked_augmentation = {
        "frequency_mask_max_bins": 16,
        "time_mask_max_frames": 40,
        "frequency_mask_probability": 0.5,
        "time_mask_probability": 0.5,
        "fill_value": 0.0,
    }
    for key, expected in locked_augmentation.items():
        if augmentation.get(key) != expected:
            issues.append(f"task1 augmentation {key} must be {expected!r}")
    locked_training = {
        "optimizer": "adamw",
        "scheduler": "none",
        "maximum_epochs": 30,
        "early_stopping_patience": 5,
        "batch_size": 32,
        "weight_decay": 0.0001,
        "head_learning_rate": 0.0003,
        "backbone_learning_rate": 0.00003,
        "pin_memory": False,
        "num_workers": 0,
        "mixed_precision": False,
        "dtype": "float32",
        "device_preference": "mps",
        "allow_mps_fallback": False,
        "request_deterministic_algorithms": True,
        "determinism_failure_policy": "fail_and_log",
        "seed_python": True,
        "seed_numpy": True,
        "seed_torch": True,
        "seed_sampler": True,
        "checkpoint_metric": "validation_recording_macro_f1",
        "checkpoint_mode": "max",
        "checkpoint_tie_break_1": "lower_validation_loss",
        "checkpoint_tie_break_2": "earlier_epoch",
        "log_parameter_counts": True,
    }
    for key, expected in locked_training.items():
        observed = training.get(key)
        if isinstance(expected, float):
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or not math.isclose(float(observed), expected)
            ):
                issues.append(f"task1 training {key} must be {expected!r}")
        elif type(observed) is not type(expected) or observed != expected:
            issues.append(f"task1 training {key} must be {expected!r}")
    return issues


def _validate_task2_config(config: dict[str, Any]) -> list[str]:
    issues = _unknown_keys(config, TASK2_TOP_LEVEL, "task2")
    raw_training = config.get("training")
    raw_novelty = config.get("novelty")
    training = raw_training if isinstance(raw_training, dict) else {}
    novelty = raw_novelty if isinstance(raw_novelty, dict) else {}
    if not isinstance(raw_training, dict):
        issues.append("task2 training must be a table")
    if not isinstance(raw_novelty, dict):
        issues.append("task2 novelty must be a table")
    issues.extend(_unknown_keys(training, TASK2_TRAINING, "task2.training"))
    issues.extend(_unknown_keys(novelty, TASK2_NOVELTY, "task2.novelty"))

    locked_top_level = {
        "task": "novelty_detection",
        "architecture": "skip_free_undercomplete_convolutional_autoencoder",
        "input_channels": 1,
        "input_height": 224,
        "input_width": 224,
        "encoder_channels": [16, 32, 64, 128],
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "latent_dimensions": 64,
        "hidden_activation": "relu",
        "normalization": "none",
        "bottleneck_activation": "linear",
        "transpose_convolution_output_padding": 0,
        "decoder_output_activation": "sigmoid",
        "loss": "mean_squared_error",
        "loss_reduction": "mean_over_all_pixels",
        "clip_selection_strategy": "energy",
        "seeds": [13, 37, 71],
    }
    locked_training = {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "batch_size": 64,
        "maximum_epochs": 100,
        "early_stopping_patience": 10,
        "pin_memory": False,
        "num_workers": 0,
        "mixed_precision": False,
        "dtype": "float32",
        "device_preference": "mps",
        "allow_mps_fallback": False,
        "request_deterministic_algorithms": True,
        "determinism_failure_policy": "fail_and_log",
        "seed_python": True,
        "seed_numpy": True,
        "seed_torch": True,
        "seed_sampler": True,
        "scheduler": "none",
        "checkpoint_metric": "known_validation_reconstruction_mse",
        "checkpoint_mode": "min",
    }
    locked_novelty = {
        "primary_score": "median_clip_reconstruction_mse",
        "secondary_readout": "recording_mean_latent_knn_distance",
        "latent_reference_unit": "one_mean_embedding_per_known_training_recording",
        "latent_standardization_unit": "known_training_recording_embeddings",
        "nearest_neighbours": 10,
        "threshold_quantile": 0.95,
        "threshold_quantile_method": "higher",
        "threshold_scope": "per_seed_known_validation",
        "score_direction": "higher_is_more_novel",
        "threshold_operator": ">",
        "bootstrap_seed": 20260713,
        "bootstrap_replicates": 2000,
        "bootstrap_interval_method": "percentile",
        "bootstrap_confidence_level": 0.95,
        "bootstrap_resampling_unit": "session_cluster",
        "detailed_figure_seed": 37,
    }
    for context, observed_values, locked_values in (
        ("task2", config, locked_top_level),
        ("task2.training", training, locked_training),
        ("task2.novelty", novelty, locked_novelty),
    ):
        for key, expected in locked_values.items():
            observed = observed_values.get(key)
            if type(observed) is not type(expected) or observed != expected:
                issues.append(f"{context}.{key} must be {expected!r}")
    return issues


def validate_config(config: dict[str, Any]) -> None:
    if config.get("task") == "classification":
        issues = _validate_task1_config(config)
    elif config.get("task") == "novelty_detection":
        issues = _validate_task2_config(config)
    elif "raw_audio_dir" in config:
        issues = _validate_data_config(config)
    else:
        issues = ["configuration type is not recognized"]
    if issues:
        raise ConfigValidationError("Invalid configuration:\n- " + "\n- ".join(issues))


def load_toml(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load TOML and recursively resolve an optional relative `extends` value."""
    resolved = resolve_project_path(path)
    seen = set() if _seen is None else set(_seen)
    if resolved in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved])
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    seen.add(resolved)

    with resolved.open("rb") as handle:
        current = tomllib.load(handle)

    parent_name = current.pop("extends", None)
    if parent_name is None:
        merged = current
    else:
        parent_path = (resolved.parent / str(parent_name)).resolve()
        parent = load_toml(parent_path, seen)
        merged = deep_merge(public_config(parent), current)

    validate_config(merged)
    merged["_config_path"] = str(resolved)
    return merged


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove loader metadata before hashing or saving a resolved config."""
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_fingerprint(config: dict[str, Any]) -> str:
    return sha256_json(public_config(config))


def validate_project_config_set(
    data_path: str | Path = "configs/data.toml",
    unknown_acquisition_path: str | Path = "configs/unknown_acquisition.toml",
    unknown_selection_path: str | Path = "configs/unknown_selection.toml",
    unknown_audio_path: str | Path = "configs/unknown_audio.toml",
) -> dict[str, int]:
    from bird_audio.unknown_acquisition import load_unknown_acquisition_config
    from bird_audio.unknown_audio import load_unknown_audio_config
    from bird_audio.unknown_planning import (
        DEFAULT_PLAN,
        DEFAULT_PLAN_LOCK,
        load_unknown_selection_config,
    )

    data = load_toml(data_path)
    unknown_acquisition = load_unknown_acquisition_config(unknown_acquisition_path)
    unknown_selection = load_unknown_selection_config(unknown_selection_path)
    unknown_audio = load_unknown_audio_config(unknown_audio_path)
    expected_classes = [entry["common_name"] for entry in data["known_species"]]
    task1_paths = sorted(resolve_project_path("configs/task1").glob("*.toml"))
    task2_paths = sorted(resolve_project_path("configs/task2").glob("*.toml"))
    expected_task1_path = resolve_project_path("configs/task1/final.toml")
    expected_task2_path = resolve_project_path("configs/task2/autoencoder.toml")
    if task1_paths != [expected_task1_path]:
        raise ConfigValidationError(
            "Task 1 must contain exactly the single locked final configuration"
        )
    if task2_paths != [expected_task2_path]:
        raise ConfigValidationError(
            "Task 2 must contain exactly the single locked autoencoder configuration"
        )
    for path in task1_paths:
        config = load_toml(path)
        if config.get("class_order") != expected_classes:
            raise ConfigValidationError(f"Task 1 class order differs from data config: {path}")
    for path in task2_paths:
        config = load_toml(path)
        if int(config["input_height"]) != int(data["spectrogram"]["output_height"]) or int(
            config["input_width"]
        ) != int(data["spectrogram"]["output_width"]):
            raise ConfigValidationError(f"Task 2 input shape differs from data config: {path}")

    expected_unknowns = [
        (
            "primary",
            True,
            entry["common_name"],
            entry["scientific_name"],
            entry["difficulty_group"],
        )
        for entry in data["unknown_species"]
    ]
    expected_unknowns.extend(
        (
            "fallback",
            False,
            entry["common_name"],
            entry["scientific_name"],
            "fallback",
        )
        for entry in data["fallback_unknown_species"]
    )
    observed_unknowns = [
        (
            entry["role"],
            entry["active"],
            entry["common_name"],
            entry["scientific_name"],
            entry["difficulty_group"],
        )
        for entry in unknown_acquisition["species"]
    ]
    if observed_unknowns != expected_unknowns:
        raise ConfigValidationError(
            "Unknown acquisition species differ from the locked data configuration"
        )
    if int(unknown_acquisition["target_recordings_per_species"]) != 40:
        raise ConfigValidationError("Unknown acquisition target differs from the data protocol")
    if unknown_selection["selection_seed"] != data["split_seed"]:
        raise ConfigValidationError("Unknown selection seed differs from the data split seed")
    if (
        unknown_selection["candidate_pool_target_recordings_per_species"]
        != unknown_acquisition["candidate_pool_target_recordings_per_species"]
    ):
        raise ConfigValidationError(
            "Unknown selection candidate pool target differs from the acquisition protocol"
        )
    if (
        unknown_selection["target_recordings_per_species"]
        != unknown_acquisition["target_recordings_per_species"]
    ):
        raise ConfigValidationError(
            "Unknown selection recording target differs from the acquisition protocol"
        )
    if unknown_audio["selection_seed"] != unknown_selection["selection_seed"]:
        raise ConfigValidationError("Unknown audio seed differs from the selection protocol")
    if (
        unknown_audio["candidate_pool_target_recordings_per_species"]
        != unknown_acquisition["candidate_pool_target_recordings_per_species"]
    ):
        raise ConfigValidationError(
            "Unknown audio candidate pool target differs from the acquisition protocol"
        )
    if (
        unknown_audio["target_recordings_per_species"]
        != unknown_acquisition["target_recordings_per_species"]
    ):
        raise ConfigValidationError(
            "Unknown audio recording target differs from the acquisition protocol"
        )
    if (
        unknown_audio["audio_qc"]["minimum_source_sample_rate_hz"]
        != data["minimum_source_sample_rate_hz"]
    ):
        raise ConfigValidationError(
            "Unknown audio minimum source sample rate differs from the data protocol"
        )
    for key in (
        "full_decode_warning_policy",
        "minimum_decoded_to_ffprobe_duration_ratio",
        "maximum_decoded_to_ffprobe_duration_ratio",
    ):
        if unknown_audio["audio_qc"][key] != data["quality_control"][key]:
            raise ConfigValidationError(
                f"Unknown audio {key} differs from the data quality control protocol"
            )
    if (
        unknown_audio["session"]["coordinate_radius_km"]
        != data["session_grouping"]["coordinate_radius_km"]
    ):
        raise ConfigValidationError("Unknown audio session radius differs from the data protocol")
    if unknown_audio["inputs"]["known_manifest"] != data["enriched_manifest"]:
        raise ConfigValidationError("Unknown audio known manifest differs from the data protocol")
    if unknown_audio["inputs"]["candidate_plan"] != DEFAULT_PLAN:
        raise ConfigValidationError(
            "Unknown audio candidate plan differs from the locked planning artifact"
        )
    if unknown_audio["inputs"]["candidate_plan_lock"] != DEFAULT_PLAN_LOCK:
        raise ConfigValidationError(
            "Unknown audio candidate plan lock differs from the locked planning artifact"
        )
    if unknown_audio["metadata"]["accepted_quality"] != unknown_selection["strata"]["quality"]:
        raise ConfigValidationError(
            "Unknown audio accepted quality differs from the selection protocol"
        )

    return {
        "data_configs": 1,
        "task1_configs": len(task1_paths),
        "task2_configs": len(task2_paths),
        "unknown_acquisition_configs": 1,
        "unknown_selection_configs": 1,
        "unknown_audio_configs": 1,
    }
