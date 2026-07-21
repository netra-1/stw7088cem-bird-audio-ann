from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.nn import functional as functional

from bird_audio.audio import AudioToolError, normalize_ffmpeg_diagnostic

TARGET_SAMPLE_RATE_HZ = 32_000
CLIP_DURATION_SECONDS = 3.0
CLIP_SAMPLES = 96_000

N_FFT = 1_024
WIN_LENGTH = 1_024
HOP_LENGTH = 256
N_MELS = 128
F_MIN_HZ = 150.0
F_MAX_HZ = 14_000.0
POWER_TO_DB_AMIN = 1e-10
MINIMUM_DB = -80.0
MAXIMUM_DB = 0.0

NATIVE_MEL_HEIGHT = 128
NATIVE_MEL_WIDTH = 372
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_WIDTH = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_DEVIATION = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ExtractedClip:
    """A canonical clip and the padding needed to construct it."""

    samples: np.ndarray
    start_sample: int
    valid_samples: int
    left_padding_samples: int
    right_padding_samples: int

    @property
    def valid_audio_fraction(self) -> float:
        return self.valid_samples / int(self.samples.size)


def canonical_waveform(waveform: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous mono float32 waveform."""
    samples = np.asarray(waveform)
    if samples.ndim != 1:
        raise ValueError("waveform must be a one-dimensional mono array")
    if not np.issubdtype(samples.dtype, np.number):
        raise TypeError("waveform must contain numeric samples")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not bool(np.all(np.isfinite(samples))):
        raise ValueError("waveform contains non-finite samples")
    return samples


def decode_audio_ffmpeg(
    path: str | Path,
    ffmpeg: str | Path,
    *,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    timeout_seconds: int = 3_600,
) -> np.ndarray:
    """Decode a complete file through the canonical mono float32 FFmpeg path."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    source_path = Path(path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio source is not a regular local file: {source_path}")

    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    child_environment = os.environ.copy()
    child_environment.pop("XENO_CANTO_API_KEY", None)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=child_environment,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioToolError(
            f"FFmpeg decode failed for {source_path}: {type(exc).__name__}: {exc}"
        ) from exc

    if completed.returncode != 0:
        diagnostic = normalize_ffmpeg_diagnostic(completed.stderr.decode("utf-8", errors="replace"))
        raise AudioToolError(f"FFmpeg decode failed for {source_path}: {diagnostic}")
    if len(completed.stdout) % np.dtype("<f4").itemsize:
        raise AudioToolError(f"Unexpected float32 byte count for {source_path}")

    waveform = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
    if waveform.size == 0:
        raise AudioToolError(f"FFmpeg produced no audio samples for {source_path}")
    if not bool(np.all(np.isfinite(waveform))):
        raise AudioToolError(f"Non-finite decoded samples in {source_path}")
    return waveform


def extract_clip(
    waveform: np.ndarray,
    start_sample: int = 0,
    *,
    clip_samples: int = CLIP_SAMPLES,
) -> ExtractedClip:
    """Slice a full recording, centre-padding short audio with the odd sample on the right."""
    samples = canonical_waveform(waveform)
    return _extract_clip_from_canonical(samples, start_sample, clip_samples=clip_samples)


def extract_clips(
    waveform: np.ndarray,
    start_samples: Iterable[int],
    *,
    clip_samples: int = CLIP_SAMPLES,
) -> tuple[ExtractedClip, ...]:
    """Extract several clips after one validation pass over the complete recording."""
    return tuple(iter_extracted_clips(waveform, start_samples, clip_samples=clip_samples))


def iter_extracted_clips(
    waveform: np.ndarray,
    start_samples: Iterable[int],
    *,
    clip_samples: int = CLIP_SAMPLES,
) -> Iterator[ExtractedClip]:
    """Stream clips after one validation pass over the complete recording."""
    samples = canonical_waveform(waveform)
    for start_sample in start_samples:
        yield _extract_clip_from_canonical(samples, start_sample, clip_samples=clip_samples)


def _extract_clip_from_canonical(
    samples: np.ndarray,
    start_sample: int,
    *,
    clip_samples: int,
) -> ExtractedClip:
    if isinstance(start_sample, bool) or not isinstance(start_sample, (int, np.integer)):
        raise TypeError("start_sample must be an integer sample position")
    if isinstance(clip_samples, bool) or not isinstance(clip_samples, (int, np.integer)):
        raise TypeError("clip_samples must be an integer")
    start_sample = int(start_sample)
    clip_samples = int(clip_samples)
    if clip_samples <= 0:
        raise ValueError("clip_samples must be positive")
    if start_sample < 0:
        raise ValueError("start_sample cannot be negative")

    sample_count = int(samples.size)
    if sample_count < clip_samples:
        if start_sample != 0:
            raise ValueError("short recordings have the single canonical start_sample 0")
        missing = clip_samples - sample_count
        left_padding = missing // 2
        right_padding = missing - left_padding
        clip = np.pad(samples, (left_padding, right_padding), mode="constant")
        return ExtractedClip(
            samples=np.ascontiguousarray(clip, dtype=np.float32),
            start_sample=0,
            valid_samples=sample_count,
            left_padding_samples=left_padding,
            right_padding_samples=right_padding,
        )

    maximum_start = sample_count - clip_samples
    if start_sample > maximum_start:
        raise ValueError(
            f"start_sample {start_sample} exceeds the final valid start {maximum_start}"
        )
    clip = samples[start_sample : start_sample + clip_samples].copy()
    return ExtractedClip(
        samples=clip,
        start_sample=start_sample,
        valid_samples=clip_samples,
        left_padding_samples=0,
        right_padding_samples=0,
    )


@cache
def periodic_hann_window(win_length: int = WIN_LENGTH) -> np.ndarray:
    """Return the periodic Hann window used by the locked STFT."""
    if win_length <= 0:
        raise ValueError("win_length must be positive")
    window = np.hanning(win_length + 1)[:-1].astype(np.float32)
    window.setflags(write=False)
    return window


def power_spectrogram(
    waveform: np.ndarray,
    *,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute a centre-free periodic-Hann STFT power spectrogram."""
    samples = canonical_waveform(waveform)
    if n_fft <= 0 or win_length <= 0 or hop_length <= 0:
        raise ValueError("n_fft, win_length, and hop_length must be positive")
    if win_length != n_fft:
        raise ValueError("the locked signal protocol requires win_length to equal n_fft")
    if samples.size < win_length:
        raise ValueError("waveform is shorter than win_length")

    frames = np.lib.stride_tricks.sliding_window_view(samples, win_length)[::hop_length]
    windowed = frames * periodic_hann_window(win_length)
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)
    return np.ascontiguousarray(power.T)


@cache
def mel_filter_bank(
    *,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min_hz: float = F_MIN_HZ,
    f_max_hz: float = F_MAX_HZ,
) -> np.ndarray:
    """Return the locked Slaney-normalized, non-HTK Mel filter bank."""
    filters = librosa.filters.mel(
        sr=sample_rate_hz,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min_hz,
        fmax=f_max_hz,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )
    filters.setflags(write=False)
    return filters


def native_log_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Create the locked one-channel native log-Mel representation in [0, 1]."""
    power = power_spectrogram(waveform)
    mel_power = mel_filter_bank() @ power
    maximum_power = float(np.max(mel_power))
    if maximum_power == 0.0:
        scaled = np.zeros_like(mel_power, dtype=np.float32)
    else:
        decibels = 10.0 * np.log10(np.maximum(mel_power, POWER_TO_DB_AMIN))
        decibels -= 10.0 * np.log10(max(maximum_power, POWER_TO_DB_AMIN))
        decibels = np.clip(decibels, MINIMUM_DB, MAXIMUM_DB)
        scaled = ((decibels - MINIMUM_DB) / (MAXIMUM_DB - MINIMUM_DB)).astype(np.float32)

    native = np.ascontiguousarray(scaled[np.newaxis, :, :], dtype=np.float32)
    expected_shape = (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH)
    if native.shape != expected_shape:
        raise RuntimeError(
            f"Expected native log-Mel shape {expected_shape}, received {native.shape}"
        )
    return native


def _native_log_mel_tensor(native_log_mel: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(native_log_mel, torch.Tensor):
        tensor = native_log_mel.detach().to(device="cpu", dtype=torch.float32)
    else:
        array = np.asarray(native_log_mel)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("native_log_mel must contain numeric values")
        tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    expected_shape = (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"native_log_mel must have shape {expected_shape}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("native_log_mel contains non-finite values")
    if bool(torch.any(tensor < 0)) or bool(torch.any(tensor > 1)):
        raise ValueError("native_log_mel values must lie in [0, 1]")
    return tensor.contiguous()


def resize_native_log_mel(native_log_mel: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Resize a native feature with locked bicubic settings and clamp interpolation overshoot."""
    tensor = _native_log_mel_tensor(native_log_mel)
    resized = functional.interpolate(
        tensor.unsqueeze(0),
        size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return resized.clamp_(0.0, 1.0).contiguous()


def to_autoencoder_tensor(native_log_mel: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Create the one-channel [0, 1] autoencoder input tensor."""
    return resize_native_log_mel(native_log_mel)


def to_efficientnet_tensor(native_log_mel: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Create the replicated and ImageNet-normalized EfficientNet input tensor."""
    resized = resize_native_log_mel(native_log_mel)
    replicated = resized.expand(3, -1, -1).clone()
    mean = replicated.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    standard_deviation = replicated.new_tensor(IMAGENET_STANDARD_DEVIATION).view(3, 1, 1)
    return ((replicated - mean) / standard_deviation).contiguous()
