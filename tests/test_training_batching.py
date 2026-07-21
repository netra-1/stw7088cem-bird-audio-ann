from __future__ import annotations

import unittest

import numpy as np
import torch

from bird_audio.signal import (
    NATIVE_MEL_HEIGHT,
    NATIVE_MEL_WIDTH,
    to_autoencoder_tensor,
    to_efficientnet_tensor,
)
from bird_audio.training_batching import (
    SAMPLER_RANDOM_STREAM,
    SPECAUGMENT_RANDOM_STREAM,
    NativeBatch,
    RecordingBalancedEpochSampler,
    SpecAugmentMask,
    apply_locked_specaugment,
    apply_specaugment_plan,
    collate_native_samples,
    epoch_generator_seed,
    make_epoch_cpu_generator,
    recording_balanced_weights,
    resize_native_batch,
    sample_specaugment_plan,
    to_autoencoder_batch,
    to_efficientnet_batch,
)


class _MetadataSource:
    def __init__(self, rows: list[dict[str, str]], split: str = "train") -> None:
        self.rows = rows
        self.split = split
        self.strategy = "energy"

    def __len__(self) -> int:
        return len(self.rows)

    def iter_metadata(self):
        for row in self.rows:
            yield dict(row)


def _metadata_rows() -> list[dict[str, str]]:
    definitions = (
        ("A1", "Species A", "0", 2),
        ("A2", "Species A", "0", 1),
        ("B1", "Species B", "1", 3),
    )
    rows: list[dict[str, str]] = []
    for recording_id, species, class_index, clip_count in definitions:
        for rank in range(clip_count):
            rows.append(
                {
                    "recording_id": recording_id,
                    "species_common_name": species,
                    "class_index": class_index,
                    "session_group": f"session:{recording_id}",
                    "split": "train",
                    "selection_strategy": "energy",
                    "strategy_clip_count": str(clip_count),
                    "energy_rank": str(rank),
                }
            )
    return rows


class RecordingBalancedSamplingTests(unittest.TestCase):
    def test_weight_arithmetic_is_exact_in_source_order(self) -> None:
        source = _MetadataSource(_metadata_rows())

        weights = recording_balanced_weights(source)

        expected = torch.tensor(
            [1 / 4, 1 / 4, 1 / 2, 1 / 3, 1 / 3, 1 / 3],
            dtype=torch.float64,
        )
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
        self.assertEqual(weights.device.type, "cpu")
        self.assertEqual(weights.dtype, torch.float64)
        self.assertEqual(float(weights[:3].sum()), 1.0)
        self.assertEqual(float(weights[3:].sum()), 1.0)
        self.assertEqual(float(weights[:2].sum()), 0.5)
        self.assertEqual(float(weights[2]), 0.5)

    def test_sampling_requires_consistent_training_metadata(self) -> None:
        validation = _MetadataSource(_metadata_rows(), split="validation")
        with self.assertRaisesRegex(PermissionError, "training split"):
            recording_balanced_weights(validation)

        bad_count = _metadata_rows()
        bad_count[0]["strategy_clip_count"] = "3"
        with self.assertRaisesRegex(ValueError, "changes within recording"):
            recording_balanced_weights(_MetadataSource(bad_count))

        interleaved = _metadata_rows()
        interleaved[1], interleaved[2] = interleaved[2], interleaved[1]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            recording_balanced_weights(_MetadataSource(interleaved))

    def test_epoch_sampler_is_explicit_repeatable_and_global_rng_independent(self) -> None:
        source = _MetadataSource(_metadata_rows())
        sampler = RecordingBalancedEpochSampler(source, base_seed=13)
        self.assertEqual(len(sampler), len(source))
        self.assertEqual(sampler.draws_per_epoch, len(source))
        self.assertEqual(sampler.generator_seed, 5866481627172567058)

        torch.manual_seed(1)
        first = list(sampler)
        torch.manual_seed(999999)
        repeated = list(sampler)
        self.assertEqual(first, repeated)
        self.assertEqual(len(first), len(source))
        self.assertTrue(all(0 <= index < len(source) for index in first))

        sampler.set_epoch(1)
        self.assertEqual(sampler.generator_seed, 8663112764867601036)
        second_epoch = list(sampler)
        self.assertNotEqual(first, second_epoch)
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

        exposed = sampler.weights
        exposed.zero_()
        self.assertTrue(bool(torch.all(sampler.weights > 0)))
        with self.assertRaisesRegex(ValueError, "negative"):
            sampler.set_epoch(-1)

    def test_named_epoch_generator_seed_is_stable_and_stream_separated(self) -> None:
        self.assertEqual(
            epoch_generator_seed(37, 2, SPECAUGMENT_RANDOM_STREAM),
            4391568949767419803,
        )
        self.assertNotEqual(
            epoch_generator_seed(37, 2, SPECAUGMENT_RANDOM_STREAM),
            epoch_generator_seed(37, 2, SAMPLER_RANDOM_STREAM),
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            epoch_generator_seed(13, 0, " specaugment")


class NativeCollationTests(unittest.TestCase):
    def test_collation_preserves_float32_features_and_metadata_order(self) -> None:
        first = np.full((1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), 0.25, np.float32)
        second = np.full((1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), 0.75, np.float32)
        first_metadata = {"recording_id": "A", "rank": "0"}
        second_metadata = {"recording_id": "B", "rank": "0"}

        batch = collate_native_samples(((first, first_metadata), (second, second_metadata)))

        self.assertIsInstance(batch, NativeBatch)
        self.assertEqual(tuple(batch.tensor.shape), (2, 1, 128, 372))
        self.assertEqual(batch.tensor.dtype, torch.float32)
        self.assertEqual(batch.tensor.device.type, "cpu")
        self.assertTrue(batch.tensor.is_contiguous())
        self.assertEqual(
            tuple(item["recording_id"] for item in batch.metadata),
            ("A", "B"),
        )
        self.assertAlmostEqual(float(batch.tensor[0, 0, 0, 0]), 0.25)
        self.assertAlmostEqual(float(batch.tensor[1, 0, 0, 0]), 0.75)

        first.fill(1.0)
        first_metadata["recording_id"] = "changed"
        self.assertAlmostEqual(float(batch.tensor[0, 0, 0, 0]), 0.25)
        self.assertEqual(batch.metadata[0]["recording_id"], "A")

    def test_collation_rejects_empty_wrong_dtype_and_out_of_range_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_native_samples(())
        wrong_dtype = np.zeros((1, 128, 372), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "native signal contract"):
            collate_native_samples(((wrong_dtype, {"recording_id": "A"}),))
        out_of_range = np.full((1, 128, 372), 1.1, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "native signal contract"):
            collate_native_samples(((out_of_range, {"recording_id": "A"}),))


class SpecAugmentTests(unittest.TestCase):
    def test_plan_is_seeded_repeatable_independent_and_within_locked_bounds(self) -> None:
        first_generator = make_epoch_cpu_generator(37, 2, SPECAUGMENT_RANDOM_STREAM)
        second_generator = make_epoch_cpu_generator(37, 2, SPECAUGMENT_RANDOM_STREAM)

        first = sample_specaugment_plan(6, generator=first_generator)
        second = sample_specaugment_plan(6, generator=second_generator)

        self.assertEqual(first, second)
        self.assertEqual(
            first[:4],
            (
                SpecAugmentMask(False, 0, 0, True, 139, 29),
                SpecAugmentMask(True, 80, 16, False, 0, 0),
                SpecAugmentMask(True, 112, 6, False, 0, 0),
                SpecAugmentMask(True, 76, 16, False, 0, 0),
            ),
        )
        self.assertNotEqual(
            tuple(mask.frequency_applied for mask in first),
            tuple(mask.time_applied for mask in first),
        )
        self.assertTrue(all(0 <= mask.frequency_width <= 16 for mask in first))
        self.assertTrue(all(0 <= mask.time_width <= 40 for mask in first))
        self.assertTrue(all(mask.frequency_start + mask.frequency_width <= 128 for mask in first))
        self.assertTrue(all(mask.time_start + mask.time_width <= 372 for mask in first))

    def test_manual_plan_masks_only_its_sample_and_axis_and_preserves_input(self) -> None:
        batch = torch.ones((2, 1, 128, 372), dtype=torch.float32)
        plan = (
            SpecAugmentMask(True, 3, 2, False, 0, 0),
            SpecAugmentMask(False, 0, 0, True, 5, 3),
        )

        augmented = apply_specaugment_plan(batch, plan)

        self.assertTrue(bool(torch.all(batch == 1.0)))
        self.assertTrue(bool(torch.all(augmented[0, :, 3:5, :] == 0.0)))
        self.assertTrue(bool(torch.all(augmented[0, :, :3, :] == 1.0)))
        self.assertTrue(bool(torch.all(augmented[1, :, :, 5:8] == 0.0)))
        self.assertTrue(bool(torch.all(augmented[1, :, :, :5] == 1.0)))
        self.assertFalse(augmented.requires_grad)
        self.assertTrue(augmented.is_contiguous())

    def test_locked_augmentation_is_global_rng_independent(self) -> None:
        batch = torch.ones((6, 1, 128, 372), dtype=torch.float32)
        torch.manual_seed(1)
        first = apply_locked_specaugment(
            batch,
            generator=make_epoch_cpu_generator(37, 2, SPECAUGMENT_RANDOM_STREAM),
        )
        torch.manual_seed(999999)
        second = apply_locked_specaugment(
            batch,
            generator=make_epoch_cpu_generator(37, 2, SPECAUGMENT_RANDOM_STREAM),
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(bool(torch.equal(first, batch)))

    def test_invalid_plan_and_native_batch_are_rejected(self) -> None:
        batch = torch.ones((1, 1, 128, 372), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "length"):
            apply_specaugment_plan(batch, ())
        with self.assertRaisesRegex(ValueError, "outside"):
            apply_specaugment_plan(
                batch,
                (SpecAugmentMask(True, 120, 16, False, 0, 0),),
            )
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            apply_locked_specaugment(
                batch.to(torch.float64),
                generator=torch.Generator(device="cpu"),
            )


class BatchedAdapterTests(unittest.TestCase):
    def test_batched_adapters_match_the_locked_single_sample_transforms(self) -> None:
        generator = np.random.default_rng(20260713)
        native = generator.random((2, 1, 128, 372), dtype=np.float32)
        batch = torch.from_numpy(native.copy())

        autoencoder = to_autoencoder_batch(batch)
        classifier = to_efficientnet_batch(batch)
        expected_autoencoder = torch.stack(tuple(to_autoencoder_tensor(item) for item in native))
        expected_classifier = torch.stack(tuple(to_efficientnet_tensor(item) for item in native))

        torch.testing.assert_close(autoencoder, expected_autoencoder, rtol=0, atol=0)
        torch.testing.assert_close(classifier, expected_classifier, rtol=0, atol=0)
        self.assertEqual(tuple(autoencoder.shape), (2, 1, 224, 224))
        self.assertEqual(tuple(classifier.shape), (2, 3, 224, 224))
        self.assertEqual(autoencoder.dtype, torch.float32)
        self.assertEqual(classifier.dtype, torch.float32)
        self.assertGreaterEqual(float(autoencoder.min()), 0.0)
        self.assertLessEqual(float(autoencoder.max()), 1.0)

    def test_resize_is_batched_no_grad_clamped_and_contiguous(self) -> None:
        checkerboard = np.indices((128, 372)).sum(axis=0) % 2
        native = torch.from_numpy(
            checkerboard.astype(np.float32)[None, None, :, :]
        ).requires_grad_()

        resized = resize_native_batch(native)

        self.assertEqual(tuple(resized.shape), (1, 1, 224, 224))
        self.assertEqual(resized.dtype, torch.float32)
        self.assertFalse(resized.requires_grad)
        self.assertTrue(resized.is_contiguous())
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_adapter_rejects_wrong_native_shape_and_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            to_autoencoder_batch(torch.zeros((2, 1, 224, 224), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "values"):
            to_efficientnet_batch(torch.full((1, 1, 128, 372), -0.1, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
