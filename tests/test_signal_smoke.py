from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bird_audio.paths import PROJECT_ROOT
from bird_audio.signal import CLIP_SAMPLES, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH
from bird_audio.signal_smoke import run_signal_smoke_test


class SignalSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=interim)
        self.root = Path(self.temporary.name)
        self.raw_root = self.root / "raw"
        self.raw_root.mkdir()
        self.source = self.raw_root / "recording.mp3"
        self.source.write_bytes(b"source-audio")
        self.output = self.root / "signal_smoke.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @patch("bird_audio.signal_smoke.to_efficientnet_tensor")
    @patch("bird_audio.signal_smoke.to_autoencoder_tensor")
    @patch("bird_audio.signal_smoke.native_log_mel_spectrogram")
    @patch("bird_audio.signal_smoke.energy_clip_starts", return_value=(100, 200))
    @patch("bird_audio.signal_smoke.uniform_clip_starts", return_value=(0, 100))
    @patch("bird_audio.signal_smoke.decode_audio_ffmpeg")
    def test_smoke_deduplicates_starts_and_records_both_selection_memberships(
        self,
        decode,
        _uniform,
        _energy,
        native,
        autoencoder,
        efficientnet,
    ) -> None:
        decode.return_value = np.zeros(CLIP_SAMPLES + 200, dtype=np.float32)
        native.return_value = np.zeros((1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), dtype=np.float32)
        autoencoder.return_value = np.zeros((1, 224, 224), dtype=np.float32)
        efficientnet.return_value = np.zeros((3, 224, 224), dtype=np.float32)

        with patch("bird_audio.signal_smoke.RAW_DATA_ROOT", self.raw_root):
            destination, result = run_signal_smoke_test([self.source], self.output, ffmpeg="ffmpeg")

        self.assertEqual(destination, self.output)
        recording = result["recordings"][0]
        self.assertEqual(recording["unique_feature_count"], 3)
        by_start = {item["start_sample"]: item for item in recording["features"]}
        self.assertTrue(by_start[100]["uniform_selected"])
        self.assertTrue(by_start[100]["energy_selected"])
        self.assertFalse(by_start[0]["energy_selected"])
        self.assertFalse(by_start[200]["uniform_selected"])
        persisted = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(persisted["passed"])

    @patch("bird_audio.signal_smoke.decode_audio_ffmpeg")
    def test_input_outside_immutable_raw_root_is_rejected_before_decode(self, decode) -> None:
        outside = self.root / "outside.mp3"
        outside.write_bytes(b"outside")
        with (
            patch("bird_audio.signal_smoke.RAW_DATA_ROOT", self.raw_root),
            self.assertRaisesRegex(ValueError, "raw dataset file"),
        ):
            run_signal_smoke_test([outside], self.output, ffmpeg="ffmpeg")
        decode.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
