from __future__ import annotations

import unittest

import librosa
import numpy as np

from bird_audio.config import load_toml


class SignalContractTests(unittest.TestCase):
    def test_locked_mel_bank_has_no_empty_filters(self) -> None:
        config = load_toml("configs/data.toml")
        signal = config["spectrogram"]
        filters = librosa.filters.mel(
            sr=int(config["target_sample_rate_hz"]),
            n_fft=int(signal["n_fft"]),
            n_mels=int(signal["n_mels"]),
            fmin=float(signal["f_min_hz"]),
            fmax=float(signal["f_max_hz"]),
            htk=bool(signal["htk"]),
            norm=signal["mel_normalization"],
        )
        self.assertEqual(filters.shape, (128, 513))
        self.assertTrue(bool(np.all(filters.sum(axis=1) > 0)))

    def test_locked_native_frame_count(self) -> None:
        config = load_toml("configs/data.toml")
        signal = config["spectrogram"]
        samples = round(
            int(config["target_sample_rate_hz"]) * float(config["clip_duration_seconds"])
        )
        frames = 1 + (samples - int(signal["n_fft"])) // int(signal["hop_length"])
        self.assertEqual(frames, 372)
        self.assertEqual(frames, int(signal["expected_native_width"]))


if __name__ == "__main__":
    unittest.main()
