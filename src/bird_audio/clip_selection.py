from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bird_audio.signal import (
    CLIP_SAMPLES,
    F_MAX_HZ,
    F_MIN_HZ,
    N_FFT,
    TARGET_SAMPLE_RATE_HZ,
    iter_extracted_clips,
    power_spectrogram,
)

MAXIMUM_CLIPS_PER_RECORDING = 5
ENERGY_CANDIDATE_HOP_SAMPLES = 48_000
MINIMUM_SELECTED_START_SEPARATION_SAMPLES = 96_000


@dataclass(frozen=True, order=True)
class EnergyCandidate:
    start_sample: int
    energy: float


def _validate_sample_count(sample_count: int) -> int:
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise TypeError("sample_count must be an integer")
    sample_count = int(sample_count)
    if sample_count < 0:
        raise ValueError("sample_count cannot be negative")
    return sample_count


def _round_nonnegative_ratio(numerator: int, denominator: int) -> int:
    """Round a nonnegative rational to nearest integer, resolving exact halves upward."""
    return (2 * numerator + denominator) // (2 * denominator)


def uniform_clip_starts(
    sample_count: int,
    *,
    clip_samples: int = CLIP_SAMPLES,
    maximum_clips: int = MAXIMUM_CLIPS_PER_RECORDING,
) -> tuple[int, ...]:
    """Return integer starts for the locked uniformly spaced clip policy."""
    sample_count = _validate_sample_count(sample_count)
    if clip_samples <= 0:
        raise ValueError("clip_samples must be positive")
    if maximum_clips <= 0:
        raise ValueError("maximum_clips must be positive")
    if sample_count < clip_samples:
        return (0,)

    clip_count = min(maximum_clips, max(1, sample_count // clip_samples))
    if clip_count == 1:
        return (0,)
    final_start = sample_count - clip_samples
    denominator = clip_count - 1
    return tuple(
        _round_nonnegative_ratio(index * final_start, denominator) for index in range(clip_count)
    )


def energy_candidate_starts(
    sample_count: int,
    *,
    clip_samples: int = CLIP_SAMPLES,
    hop_samples: int = ENERGY_CANDIDATE_HOP_SAMPLES,
) -> tuple[int, ...]:
    """Return 1.5-second-hop candidates plus the final end-aligned candidate."""
    sample_count = _validate_sample_count(sample_count)
    if clip_samples <= 0:
        raise ValueError("clip_samples must be positive")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    if sample_count < clip_samples:
        return (0,)

    final_start = sample_count - clip_samples
    starts = list(range(0, final_start + 1, hop_samples))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def energy_frequency_bin_mask(
    *,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    n_fft: int = N_FFT,
    f_min_hz: float = F_MIN_HZ,
    f_max_hz: float = F_MAX_HZ,
) -> np.ndarray:
    """Select FFT bins whose centre frequencies are inside the inclusive energy band."""
    if sample_rate_hz <= 0 or n_fft <= 0:
        raise ValueError("sample_rate_hz and n_fft must be positive")
    if not 0 <= f_min_hz <= f_max_hz <= sample_rate_hz / 2:
        raise ValueError("energy limits must be ordered within the Nyquist interval")
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    return np.asarray((frequencies >= f_min_hz) & (frequencies <= f_max_hz))


def mean_band_power(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    f_min_hz: float = F_MIN_HZ,
    f_max_hz: float = F_MAX_HZ,
) -> float:
    """Measure mean unlogged STFT power over inclusive in-band FFT-bin centres."""
    power = power_spectrogram(waveform)
    mask = energy_frequency_bin_mask(
        sample_rate_hz=sample_rate_hz,
        n_fft=N_FFT,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    if mask.shape != (power.shape[0],):
        raise RuntimeError("energy frequency mask does not match the power spectrogram")
    if not bool(np.any(mask)):
        raise ValueError("energy band contains no FFT-bin centres")
    return float(np.mean(power[mask, :], dtype=np.float64))


def rank_energy_candidates(
    waveform: np.ndarray,
    *,
    clip_samples: int = CLIP_SAMPLES,
    hop_samples: int = ENERGY_CANDIDATE_HOP_SAMPLES,
) -> tuple[EnergyCandidate, ...]:
    """Rank candidates by decreasing energy and then by earlier integer start."""
    samples = np.asarray(waveform)
    if samples.ndim != 1:
        raise ValueError("waveform must be a one-dimensional mono array")
    starts = energy_candidate_starts(
        int(samples.size), clip_samples=clip_samples, hop_samples=hop_samples
    )
    candidates = []
    for clip in iter_extracted_clips(samples, starts, clip_samples=clip_samples):
        candidates.append(
            EnergyCandidate(
                start_sample=clip.start_sample,
                energy=mean_band_power(clip.samples),
            )
        )
    return tuple(
        sorted(candidates, key=lambda candidate: (-candidate.energy, candidate.start_sample))
    )


def select_energy_candidates(
    waveform: np.ndarray,
    *,
    maximum_clips: int = MAXIMUM_CLIPS_PER_RECORDING,
    clip_samples: int = CLIP_SAMPLES,
    hop_samples: int = ENERGY_CANDIDATE_HOP_SAMPLES,
    minimum_separation_samples: int = MINIMUM_SELECTED_START_SEPARATION_SAMPLES,
) -> tuple[EnergyCandidate, ...]:
    """Greedily retain ranked candidates with the locked minimum start separation."""
    if maximum_clips <= 0:
        raise ValueError("maximum_clips must be positive")
    if minimum_separation_samples < 0:
        raise ValueError("minimum_separation_samples cannot be negative")
    ranked = rank_energy_candidates(
        waveform,
        clip_samples=clip_samples,
        hop_samples=hop_samples,
    )
    selected: list[EnergyCandidate] = []
    for candidate in ranked:
        if all(
            abs(candidate.start_sample - retained.start_sample) >= minimum_separation_samples
            for retained in selected
        ):
            selected.append(candidate)
            if len(selected) == maximum_clips:
                break
    return tuple(selected)


def energy_clip_starts(waveform: np.ndarray) -> tuple[int, ...]:
    """Return selected integer starts in energy-rank order."""
    return tuple(candidate.start_sample for candidate in select_energy_candidates(waveform))
