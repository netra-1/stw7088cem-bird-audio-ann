from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bird_audio.audio import AudioToolError
from bird_audio.signal import (
    CLIP_SAMPLES,
    HOP_LENGTH,
    IMAGENET_MEAN,
    IMAGENET_STANDARD_DEVIATION,
    N_FFT,
    NATIVE_MEL_HEIGHT,
    NATIVE_MEL_WIDTH,
    POWER_TO_DB_AMIN,
    TARGET_SAMPLE_RATE_HZ,
    decode_audio_ffmpeg,
    extract_clip,
    extract_clips,
    mel_filter_bank,
    native_log_mel_spectrogram,
    periodic_hann_window,
    power_spectrogram,
    resize_native_log_mel,
    to_autoencoder_tensor,
    to_efficientnet_tensor,
)


class FFmpegDecodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.audio_path = Path(self.temporary_directory.name) / "recording.mp3"
        self.audio_path.write_bytes(b"test fixture")

    @patch("bird_audio.signal.subprocess.run")
    def test_full_decode_uses_locked_mono_float32_path(self, run_mock) -> None:
        expected = np.array([-0.5, 0.0, 0.75], dtype="<f4")
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=expected.tobytes(), stderr=b""
        )

        actual = decode_audio_ffmpeg(self.audio_path, Path("/tools/ffmpeg"))

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "/tools/ffmpeg")
        self.assertIn("-threads", command)
        self.assertEqual(command[command.index("-threads") + 1], "1")
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "32000")
        self.assertEqual(command[command.index("-acodec") + 1], "pcm_f32le")
        self.assertEqual(command[-1], "pipe:1")
        self.assertNotIn("-t", command)

    @patch.dict(os.environ, {"XENO_CANTO_API_KEY": "do-not-inherit"})
    @patch("bird_audio.signal.subprocess.run")
    def test_decode_removes_api_key_from_child_environment(self, run_mock) -> None:
        samples = np.array([0.0], dtype="<f4")
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=samples.tobytes(), stderr=b""
        )

        decode_audio_ffmpeg(self.audio_path, "ffmpeg")

        child_environment = run_mock.call_args.kwargs["env"]
        self.assertNotIn("XENO_CANTO_API_KEY", child_environment)
        self.assertEqual(os.environ["XENO_CANTO_API_KEY"], "do-not-inherit")

    @patch("bird_audio.signal.subprocess.run")
    def test_decode_rejects_ffmpeg_failure(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"invalid data"
        )
        with self.assertRaisesRegex(AudioToolError, "invalid data"):
            decode_audio_ffmpeg(self.audio_path, "ffmpeg")

    @patch("bird_audio.signal.subprocess.run")
    def test_decode_rejects_partial_float_and_empty_output(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"abc", stderr=b""
        )
        with self.assertRaisesRegex(AudioToolError, "float32 byte count"):
            decode_audio_ffmpeg(self.audio_path, "ffmpeg")

        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
        with self.assertRaisesRegex(AudioToolError, "no audio samples"):
            decode_audio_ffmpeg(self.audio_path, "ffmpeg")

    @patch("bird_audio.signal.subprocess.run")
    def test_decode_rejects_non_finite_samples(self, run_mock) -> None:
        samples = np.array([0.0, np.nan], dtype="<f4")
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=samples.tobytes(), stderr=b""
        )
        with self.assertRaisesRegex(AudioToolError, "Non-finite"):
            decode_audio_ffmpeg(self.audio_path, "ffmpeg")

    @patch("bird_audio.signal.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 1))
    def test_decode_wraps_timeout(self, _run_mock) -> None:
        with self.assertRaisesRegex(AudioToolError, "TimeoutExpired"):
            decode_audio_ffmpeg(self.audio_path, "ffmpeg", timeout_seconds=1)

    @patch("bird_audio.signal.subprocess.run")
    def test_decode_rejects_url_before_starting_subprocess(self, run_mock) -> None:
        with self.assertRaises(FileNotFoundError):
            decode_audio_ffmpeg("https://example.test/audio.mp3", "ffmpeg")
        run_mock.assert_not_called()


class ClipExtractionTests(unittest.TestCase):
    def test_short_clip_is_centred_with_odd_extra_sample_on_right(self) -> None:
        source = np.array([1.0, 2.0], dtype=np.float32)

        clip = extract_clip(source, clip_samples=5)

        np.testing.assert_array_equal(clip.samples, np.array([0, 1, 2, 0, 0], np.float32))
        self.assertEqual(clip.left_padding_samples, 1)
        self.assertEqual(clip.right_padding_samples, 2)
        self.assertEqual(clip.valid_samples, 2)
        self.assertEqual(clip.valid_audio_fraction, 0.4)
        self.assertEqual(clip.start_sample, 0)

    def test_empty_waveform_produces_a_fully_padded_clip(self) -> None:
        clip = extract_clip(np.array([], dtype=np.float32), clip_samples=5)
        np.testing.assert_array_equal(clip.samples, np.zeros(5, dtype=np.float32))
        self.assertEqual((clip.left_padding_samples, clip.right_padding_samples), (2, 3))
        self.assertEqual(clip.valid_audio_fraction, 0.0)

    def test_exact_and_long_recordings_are_sliced_without_padding(self) -> None:
        exact = extract_clip(np.arange(5, dtype=np.float64), clip_samples=5)
        np.testing.assert_array_equal(exact.samples, np.arange(5, dtype=np.float32))
        self.assertEqual(exact.valid_audio_fraction, 1.0)

        long = extract_clip(np.arange(10, dtype=np.float32), 3, clip_samples=5)
        np.testing.assert_array_equal(long.samples, np.arange(3, 8, dtype=np.float32))
        self.assertEqual((long.left_padding_samples, long.right_padding_samples), (0, 0))

    def test_batch_extraction_preserves_start_order(self) -> None:
        waveform = np.arange(10, dtype=np.float32)
        clips = extract_clips(waveform, [5, 0, 2], clip_samples=3)
        self.assertEqual(tuple(clip.start_sample for clip in clips), (5, 0, 2))
        np.testing.assert_array_equal(clips[0].samples, np.array([5, 6, 7], np.float32))

    def test_invalid_clip_positions_and_waveforms_are_rejected(self) -> None:
        waveform = np.zeros(5, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            extract_clip(waveform, -1, clip_samples=3)
        with self.assertRaisesRegex(ValueError, "final valid start"):
            extract_clip(waveform, 3, clip_samples=3)
        with self.assertRaisesRegex(ValueError, "single canonical"):
            extract_clip(waveform[:2], 1, clip_samples=3)
        with self.assertRaisesRegex(TypeError, "integer sample position"):
            extract_clip(waveform, 0.5, clip_samples=3)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            extract_clip(np.array([np.inf], dtype=np.float32), clip_samples=3)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            extract_clip(np.zeros((1, 5), dtype=np.float32), clip_samples=3)


class SpectrogramTests(unittest.TestCase):
    def test_hann_window_is_periodic(self) -> None:
        window = periodic_hann_window(8)
        expected = np.hanning(9)[:-1].astype(np.float32)
        np.testing.assert_array_equal(window, expected)
        self.assertEqual(float(window[0]), 0.0)
        self.assertGreater(float(window[-1]), 0.0)
        self.assertFalse(window.flags.writeable)

    def test_power_spectrogram_has_locked_shape_and_localizes_a_tone(self) -> None:
        time = np.arange(CLIP_SAMPLES, dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
        waveform = np.sin(2 * np.pi * 1_000.0 * time).astype(np.float32)

        power = power_spectrogram(waveform)

        expected_frames = 1 + (CLIP_SAMPLES - N_FFT) // HOP_LENGTH
        self.assertEqual(power.shape, (N_FFT // 2 + 1, expected_frames))
        self.assertEqual(power.dtype, np.float32)
        peak_bin = int(np.argmax(np.mean(power, axis=1)))
        self.assertEqual(peak_bin, 32)

    def test_power_spectrogram_rejects_a_window_shorter_than_the_locked_fft(self) -> None:
        waveform = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "win_length to equal n_fft"):
            power_spectrogram(waveform, n_fft=1_024, win_length=512)

    def test_native_log_mel_has_locked_shape_dtype_range_and_is_repeatable(self) -> None:
        generator = np.random.default_rng(20260713)
        waveform = generator.normal(0, 0.1, CLIP_SAMPLES).astype(np.float32)

        first = native_log_mel_spectrogram(waveform)
        second = native_log_mel_spectrogram(waveform.copy())

        self.assertEqual(first.shape, (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH))
        self.assertEqual(first.dtype, np.float32)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        np.testing.assert_array_equal(first, second)

    def test_all_silent_native_log_mel_is_all_zeros(self) -> None:
        native = native_log_mel_spectrogram(np.zeros(CLIP_SAMPLES, dtype=np.float32))
        np.testing.assert_array_equal(native, np.zeros_like(native))

    def test_very_low_nonzero_tone_follows_amin_formula_instead_of_silence_case(self) -> None:
        time = np.arange(CLIP_SAMPLES, dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
        waveform = (1e-8 * np.sin(2 * np.pi * 1_000.0 * time)).astype(np.float32)
        mel_power = mel_filter_bank() @ power_spectrogram(waveform)
        self.assertGreater(float(np.max(mel_power)), 0.0)
        self.assertLess(float(np.max(mel_power)), POWER_TO_DB_AMIN)
        expected_db = 10.0 * np.log10(np.maximum(mel_power, POWER_TO_DB_AMIN))
        expected_db -= 10.0 * np.log10(POWER_TO_DB_AMIN)
        expected = np.clip((expected_db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)

        native = native_log_mel_spectrogram(waveform)

        np.testing.assert_array_equal(native[0], expected)
        self.assertFalse(bool(np.all(native == 0.0)))

    def test_bicubic_resize_clamps_overshoot(self) -> None:
        checkerboard = np.indices((NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH)).sum(axis=0) % 2
        native = checkerboard.astype(np.float32)[np.newaxis, :, :]

        resized = resize_native_log_mel(native)

        self.assertEqual(tuple(resized.shape), (1, 224, 224))
        self.assertEqual(resized.dtype, torch.float32)
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_model_adapters_produce_locked_inputs(self) -> None:
        native = np.zeros((1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), dtype=np.float32)

        autoencoder = to_autoencoder_tensor(native)
        efficientnet = to_efficientnet_tensor(native)

        self.assertEqual(tuple(autoencoder.shape), (1, 224, 224))
        self.assertTrue(bool(torch.all(autoencoder == 0)))
        self.assertEqual(tuple(efficientnet.shape), (3, 224, 224))
        for channel, (mean, deviation) in enumerate(
            zip(IMAGENET_MEAN, IMAGENET_STANDARD_DEVIATION, strict=True)
        ):
            expected = -mean / deviation
            torch.testing.assert_close(
                efficientnet[channel], torch.full((224, 224), expected), rtol=0, atol=1e-6
            )

    def test_resize_rejects_invalid_native_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have shape"):
            resize_native_log_mel(np.zeros((128, 372), dtype=np.float32))
        invalid = np.zeros((1, 128, 372), dtype=np.float32)
        invalid[0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "lie in"):
            resize_native_log_mel(invalid)


if __name__ == "__main__":
    unittest.main()
