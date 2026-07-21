from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bird_audio.clip_selection import energy_clip_starts, uniform_clip_starts
from bird_audio.hashing import sha256_bytes, sha256_file
from bird_audio.io_utils import atomic_write_json
from bird_audio.paths import RAW_DATA_ROOT, is_relative_to, resolve_project_path
from bird_audio.signal import (
    CLIP_DURATION_SECONDS,
    CLIP_SAMPLES,
    F_MAX_HZ,
    F_MIN_HZ,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    TARGET_SAMPLE_RATE_HZ,
    decode_audio_ffmpeg,
    iter_extracted_clips,
    native_log_mel_spectrogram,
    to_autoencoder_tensor,
    to_efficientnet_tensor,
)

SIGNAL_SMOKE_SCHEMA_VERSION = "1.0"


def _feature_digest(feature: np.ndarray) -> str:
    metadata = f"{feature.dtype.str}:{feature.shape}".encode("ascii")
    return sha256_bytes(metadata + feature.tobytes(order="C"))


def run_signal_smoke_test(
    paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    ffmpeg: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Run the complete locked signal transform on selected immutable raw files."""
    if not paths:
        raise ValueError("at least one raw recording is required")

    recordings: list[dict[str, Any]] = []
    for raw_path in paths:
        path = resolve_project_path(raw_path)
        if not is_relative_to(path, RAW_DATA_ROOT) or not path.is_file():
            raise ValueError(f"signal smoke input must be a raw dataset file: {path}")

        source_sha256 = sha256_file(path)
        waveform = decode_audio_ffmpeg(path, ffmpeg)
        uniform_starts = uniform_clip_starts(int(waveform.size))
        energy_starts = energy_clip_starts(waveform)
        starts = tuple(sorted(set(uniform_starts).union(energy_starts)))

        features: list[dict[str, Any]] = []
        adapter_shapes: dict[str, list[int]] | None = None
        for clip in iter_extracted_clips(waveform, starts):
            feature = native_log_mel_spectrogram(clip.samples)
            if adapter_shapes is None:
                adapter_shapes = {
                    "autoencoder": list(to_autoencoder_tensor(feature).shape),
                    "efficientnet_b0": list(to_efficientnet_tensor(feature).shape),
                }
            features.append(
                {
                    "start_sample": clip.start_sample,
                    "valid_audio_fraction": clip.valid_audio_fraction,
                    "native_shape": list(feature.shape),
                    "native_dtype": str(feature.dtype),
                    "native_minimum": float(feature.min()),
                    "native_maximum": float(feature.max()),
                    "native_sha256": _feature_digest(feature),
                    "uniform_selected": clip.start_sample in uniform_starts,
                    "energy_selected": clip.start_sample in energy_starts,
                }
            )

        recordings.append(
            {
                "path": path.relative_to(RAW_DATA_ROOT.parent).as_posix(),
                "source_sha256": source_sha256,
                "decoded_samples": int(waveform.size),
                "decoded_duration_seconds": waveform.size / TARGET_SAMPLE_RATE_HZ,
                "uniform_start_samples": list(uniform_starts),
                "energy_start_samples": list(energy_starts),
                "unique_feature_count": len(features),
                "adapter_shapes": adapter_shapes,
                "features": features,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SIGNAL_SMOKE_SCHEMA_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "transform": {
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "clip_duration_seconds": CLIP_DURATION_SECONDS,
            "clip_samples": CLIP_SAMPLES,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "f_min_hz": F_MIN_HZ,
            "f_max_hz": F_MAX_HZ,
        },
        "recordings": recordings,
    }
    destination = atomic_write_json(output_path, result)
    return destination, result
