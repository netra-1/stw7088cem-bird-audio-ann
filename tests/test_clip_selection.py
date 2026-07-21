from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from bird_audio.clip_selection import (
    ENERGY_CANDIDATE_HOP_SAMPLES,
    energy_candidate_starts,
    energy_clip_starts,
    energy_frequency_bin_mask,
    mean_band_power,
    rank_energy_candidates,
    select_energy_candidates,
    uniform_clip_starts,
)
from bird_audio.signal import CLIP_SAMPLES, N_FFT, TARGET_SAMPLE_RATE_HZ


class UniformClipSelectionTests(unittest.TestCase):
    def test_short_and_single_clip_recordings_start_at_zero(self) -> None:
        self.assertEqual(uniform_clip_starts(0), (0,))
        self.assertEqual(uniform_clip_starts(CLIP_SAMPLES - 1), (0,))
        self.assertEqual(uniform_clip_starts(CLIP_SAMPLES), (0,))
        self.assertEqual(uniform_clip_starts(2 * CLIP_SAMPLES - 1), (0,))

    def test_integer_linspace_includes_both_endpoints(self) -> None:
        self.assertEqual(
            uniform_clip_starts(3 * CLIP_SAMPLES),
            (0, CLIP_SAMPLES, 2 * CLIP_SAMPLES),
        )
        self.assertEqual(
            uniform_clip_starts(6 * CLIP_SAMPLES),
            (0, 120_000, 240_000, 360_000, 480_000),
        )

    def test_exact_half_sample_is_rounded_up(self) -> None:
        starts = uniform_clip_starts(3 * CLIP_SAMPLES + 1)
        self.assertEqual(starts, (0, CLIP_SAMPLES + 1, 2 * CLIP_SAMPLES + 1))

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            uniform_clip_starts(-1)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            uniform_clip_starts(3.0)
        with self.assertRaisesRegex(ValueError, "maximum_clips"):
            uniform_clip_starts(CLIP_SAMPLES, maximum_clips=0)


class EnergyClipSelectionTests(unittest.TestCase):
    def test_candidates_use_half_clip_hop_and_include_end_alignment(self) -> None:
        self.assertEqual(energy_candidate_starts(CLIP_SAMPLES - 1), (0,))
        self.assertEqual(energy_candidate_starts(CLIP_SAMPLES), (0,))
        self.assertEqual(
            energy_candidate_starts(2 * CLIP_SAMPLES + 1),
            (0, 48_000, 96_000, 96_001),
        )
        self.assertEqual(
            energy_candidate_starts(2 * CLIP_SAMPLES),
            (0, ENERGY_CANDIDATE_HOP_SAMPLES, CLIP_SAMPLES),
        )

    def test_energy_band_uses_inclusive_fft_bin_centres(self) -> None:
        mask = energy_frequency_bin_mask()
        frequencies = np.fft.rfftfreq(N_FFT, d=1 / TARGET_SAMPLE_RATE_HZ)
        self.assertEqual(mask.shape, frequencies.shape)
        self.assertFalse(bool(mask[frequencies == 125.0][0]))
        self.assertTrue(bool(mask[frequencies == 156.25][0]))
        self.assertTrue(bool(mask[frequencies == 14_000.0][0]))
        self.assertFalse(bool(mask[frequencies == 14_031.25][0]))

    def test_mean_band_power_is_unlogged_and_nonnegative(self) -> None:
        time = np.arange(CLIP_SAMPLES, dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
        quiet = (0.1 * np.sin(2 * np.pi * 1_000 * time)).astype(np.float32)
        loud = (0.2 * np.sin(2 * np.pi * 1_000 * time)).astype(np.float32)
        quiet_energy = mean_band_power(quiet)
        loud_energy = mean_band_power(loud)
        self.assertGreater(quiet_energy, 0)
        self.assertAlmostEqual(loud_energy / quiet_energy, 4.0, places=5)

    def test_silent_ties_rank_earlier_and_greedy_separation_is_inclusive(self) -> None:
        waveform = np.zeros(3 * CLIP_SAMPLES, dtype=np.float32)
        ranked = rank_energy_candidates(waveform)
        self.assertEqual(
            tuple(candidate.start_sample for candidate in ranked),
            (0, 48_000, 96_000, 144_000, 192_000),
        )
        selected = select_energy_candidates(waveform)
        self.assertEqual(
            tuple(candidate.start_sample for candidate in selected),
            (0, 96_000, 192_000),
        )
        self.assertTrue(all(candidate.energy == 0.0 for candidate in selected))
        self.assertEqual(energy_clip_starts(waveform), (0, 96_000, 192_000))

    def test_high_energy_candidate_ranks_first(self) -> None:
        waveform = np.zeros(4 * CLIP_SAMPLES, dtype=np.float32)
        start = CLIP_SAMPLES
        time = np.arange(CLIP_SAMPLES, dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
        waveform[start : start + CLIP_SAMPLES] = np.sin(2 * np.pi * 2_000 * time)

        ranked = rank_energy_candidates(waveform)

        self.assertEqual(ranked[0].start_sample, start)
        self.assertGreater(ranked[0].energy, ranked[1].energy)

    def test_short_recording_is_centre_padded_before_energy_measurement(self) -> None:
        waveform = np.ones(2_000, dtype=np.float32)
        ranked = rank_energy_candidates(waveform)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].start_sample, 0)
        self.assertGreaterEqual(ranked[0].energy, 0.0)

    def test_selection_limit_and_separation_are_enforced(self) -> None:
        waveform = np.zeros(12 * CLIP_SAMPLES, dtype=np.float32)
        selected = select_energy_candidates(waveform, maximum_clips=5)
        self.assertEqual(len(selected), 5)
        for index, first in enumerate(selected):
            for second in selected[index + 1 :]:
                self.assertGreaterEqual(abs(first.start_sample - second.start_sample), CLIP_SAMPLES)

    def test_many_candidates_scan_the_whole_waveform_only_once(self) -> None:
        waveform = np.zeros(12 * CLIP_SAMPLES, dtype=np.float32)
        from bird_audio import signal

        original = signal.canonical_waveform
        scanned_sizes: list[int] = []

        def record_scan(samples: np.ndarray) -> np.ndarray:
            scanned_sizes.append(int(np.asarray(samples).size))
            return original(samples)

        with patch("bird_audio.signal.canonical_waveform", side_effect=record_scan):
            rank_energy_candidates(waveform)

        self.assertGreater(len(energy_candidate_starts(waveform.size)), 10)
        self.assertEqual(scanned_sizes.count(waveform.size), 1)

    @patch("bird_audio.signal.extract_clips", side_effect=AssertionError("must stream"))
    def test_ranking_streams_instead_of_materializing_all_clips(self, _extract_clips) -> None:
        waveform = np.zeros(8 * CLIP_SAMPLES, dtype=np.float32)
        ranked = rank_energy_candidates(waveform)
        self.assertEqual(len(ranked), len(energy_candidate_starts(waveform.size)))


if __name__ == "__main__":
    unittest.main()
