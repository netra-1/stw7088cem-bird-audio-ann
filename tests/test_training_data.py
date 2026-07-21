from __future__ import annotations

import mmap
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bird_audio.hashing import sha256_file
from bird_audio.paths import PROJECT_ROOT
from bird_audio.signal import NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH
from bird_audio.training_data import (
    DevelopmentTrainingData,
    _close_memory_map,
    _verify_feature_file,
    open_development_training_data,
)


class DevelopmentTrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=interim)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "training_cache_v1"
        feature_root = self.root / "train" / "features"
        feature_root.mkdir(parents=True)
        self.lock_path = self.root / "lock.json"
        self.lock_path.write_text('{"locked":true}\n', encoding="utf-8")

        first = np.stack(
            (
                np.full((1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), 0.1, dtype=np.float32),
                np.full((1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), 0.2, dtype=np.float32),
            )
        )
        second = np.full(
            (1, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH),
            0.3,
            dtype=np.float32,
        )
        self.first_path = feature_root / "XCA.npy"
        self.second_path = feature_root / "XCB.npy"
        np.save(self.first_path, first, allow_pickle=False)
        np.save(self.second_path, second, allow_pickle=False)

        self.rows = (
            self._row(
                recording_id="XCA",
                class_index=0,
                rank=0,
                feature_row=1,
                feature_path=self.first_path,
                cached_count=2,
            ),
            self._row(
                recording_id="XCA",
                class_index=0,
                rank=1,
                feature_row=0,
                feature_path=self.first_path,
                cached_count=2,
            ),
            self._row(
                recording_id="XCB",
                class_index=1,
                rank=0,
                feature_row=0,
                feature_path=self.second_path,
                cached_count=1,
            ),
        )

    def _row(
        self,
        *,
        recording_id: str,
        class_index: int,
        rank: int,
        feature_row: int,
        feature_path: Path,
        cached_count: int,
    ) -> dict[str, str]:
        relative_path = feature_path.relative_to(self.root).as_posix()
        return {
            "recording_id": recording_id,
            "class_index": str(class_index),
            "species_common_name": f"Species {class_index}",
            "session_group": f"session:{recording_id}",
            "split": "train",
            "feature_file": relative_path,
            "feature_file_sha256": sha256_file(feature_path),
            "feature_row": str(feature_row),
            "cached_clip_count": str(cached_count),
            "uniform_clip_count": str(cached_count),
            "energy_clip_count": str(cached_count),
            "uniform_rank": str(rank),
            "energy_rank": str(rank),
        }

    def _source(self) -> SimpleNamespace:
        return SimpleNamespace(
            root=self.root,
            rows=self.rows,
            lock_sha256=sha256_file(self.lock_path),
        )

    def _open(self, *, mmap_capacity: int = 64) -> DevelopmentTrainingData:
        with patch(
            "bird_audio.training_data.load_development_clip_cache",
            return_value=self._source(),
        ):
            return DevelopmentTrainingData(
                self.root,
                "train",
                "energy",
                mmap_capacity=mmap_capacity,
            )

    def test_rejects_test_unknown_and_invalid_options_before_cache_access(self) -> None:
        with patch("bird_audio.training_data.load_development_clip_cache") as loader:
            with self.assertRaisesRegex(PermissionError, "final test"):
                DevelopmentTrainingData(self.root, "test", "energy")
            unknown_roots = (
                PROJECT_ROOT / "data" / "unknown",
                PROJECT_ROOT / "data" / "unknown" / "metadata",
                PROJECT_ROOT / "data" / "processed" / "unknown_clips_v1",
                PROJECT_ROOT / "data" / "processed" / "unknown_clips_v2",
                PROJECT_ROOT / "data" / "processed" / "unknown_clips_v37",
                PROJECT_ROOT / "data" / "processed" / "unknown_clips_v999" / "scoring" / "features",
            )
            for unknown_root in unknown_roots:
                with (
                    self.subTest(unknown_root=unknown_root),
                    self.assertRaisesRegex(PermissionError, "locked known cache"),
                ):
                    DevelopmentTrainingData(unknown_root, "train", "energy")
            with self.assertRaisesRegex(ValueError, "Selection strategy"):
                DevelopmentTrainingData(self.root, "train", "random")
            with self.assertRaisesRegex(ValueError, "positive"):
                DevelopmentTrainingData(self.root, "train", "energy", mmap_capacity=0)
        loader.assert_not_called()

    def test_rejects_loader_redirect_to_any_versioned_unknown_cache(self) -> None:
        source = self._source()
        source.root = PROJECT_ROOT / "data" / "processed" / "unknown_clips_v14" / "scoring"
        with (
            patch(
                "bird_audio.training_data.load_development_clip_cache",
                return_value=source,
            ) as loader,
            self.assertRaisesRegex(PermissionError, "locked known cache"),
        ):
            DevelopmentTrainingData(self.root, "train", "energy")
        loader.assert_called_once()

    def test_verifies_each_file_once_and_preserves_deterministic_recording_access(self) -> None:
        with (
            patch(
                "bird_audio.training_data.load_development_clip_cache",
                return_value=self._source(),
            ) as loader,
            patch(
                "bird_audio.training_data._verify_feature_file",
                wraps=_verify_feature_file,
            ) as verify_file,
        ):
            reader = DevelopmentTrainingData(
                self.root,
                "train",
                "energy",
                ffmpeg="/tools/ffmpeg",
                expected_lock_sha256="a" * 64,
            )
        self.addCleanup(reader.close)

        loader.assert_called_once_with(
            self.root.resolve(),
            "train",
            "energy",
            ffmpeg="/tools/ffmpeg",
            expected_lock_sha256="a" * 64,
        )
        self.assertEqual(verify_file.call_count, 2)
        self.assertEqual(reader.lock_sha256, sha256_file(self.lock_path))
        self.assertEqual(reader.recording_ids, ("XCA", "XCB"))
        self.assertEqual(reader.recording_count, 2)
        self.assertEqual(reader.verified_feature_files, 2)
        self.assertEqual(reader.recording_indices("XCA"), (0, 1))
        self.assertEqual(
            tuple(reader.iter_recording_indices()),
            (("XCA", (0, 1)), ("XCB", (2,))),
        )

        features, metadata = reader.get_recording("XCA")
        self.assertEqual(features.shape, (2, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(features.flags.c_contiguous)
        self.assertAlmostEqual(float(features[0, 0, 0, 0]), 0.2)
        self.assertAlmostEqual(float(features[1, 0, 0, 0]), 0.1)
        self.assertEqual([item["energy_rank"] for item in metadata], ["0", "1"])
        self.assertTrue(all(item["selection_strategy"] == "energy" for item in metadata))
        self.assertTrue(all(item["strategy_clip_count"] == "2" for item in metadata))

    def test_random_sample_access_uses_bounded_read_only_memory_maps_without_rehashing(
        self,
    ) -> None:
        reader = self._open(mmap_capacity=2)
        self.addCleanup(reader.close)
        real_mmap = mmap.mmap
        with (
            patch("bird_audio.training_data.mmap.mmap", wraps=real_mmap) as map_file,
            patch("bird_audio.training_data._verify_feature_file") as verify_file,
        ):
            first, first_metadata = reader[0]
            second, _ = reader[2]
            repeated, _ = reader[1]
            recording, _ = reader.get_recording("XCA")

        verify_file.assert_not_called()
        self.assertEqual(map_file.call_count, 2)
        for call in map_file.call_args_list:
            self.assertIsInstance(call.args[0], int)
            self.assertEqual(call.kwargs["length"], 0)
            self.assertEqual(call.kwargs["access"], mmap.ACCESS_READ)
        self.assertEqual(reader.open_recording_count, 2)
        self.assertEqual(first.shape, (1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first_metadata["recording_id"], "XCA")
        self.assertAlmostEqual(float(second[0, 0, 0]), 0.3)
        self.assertAlmostEqual(float(repeated[0, 0, 0]), 0.1)
        self.assertEqual(recording.shape[0], 2)

        first.fill(1.0)
        unchanged, _ = reader[0]
        self.assertAlmostEqual(float(unchanged[0, 0, 0]), 0.2)

    def test_lru_eviction_context_cleanup_and_closed_state_are_explicit(self) -> None:
        with patch(
            "bird_audio.training_data._close_memory_map",
            wraps=_close_memory_map,
        ) as close_map:
            with self._open(mmap_capacity=1) as reader:
                reader[0]
                first_mapping = next(iter(reader._memory_maps.values()))
                first_backing = first_mapping.backing
                self.assertFalse(first_backing.closed)
                self.assertEqual(reader.open_recording_count, 1)
                reader[2]
                second_mapping = next(iter(reader._memory_maps.values()))
                second_backing = second_mapping.backing
                self.assertTrue(first_backing.closed)
                self.assertFalse(second_backing.closed)
                self.assertEqual(reader.open_recording_count, 1)
                self.assertEqual(close_map.call_count, 1)
            self.assertTrue(reader.closed)
            self.assertTrue(second_backing.closed)
            self.assertEqual(reader.open_recording_count, 0)
            self.assertEqual(close_map.call_count, 2)
            reader.close()
            self.assertEqual(close_map.call_count, 2)
            with self.assertRaisesRegex(RuntimeError, "is closed"):
                reader[0]

    def test_post_verification_file_replacement_is_rejected(self) -> None:
        reader = self._open()
        self.addCleanup(reader.close)
        reader[0]

        replacement = self.first_path.with_suffix(".replacement.npy")
        np.save(
            replacement,
            np.zeros((2, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), dtype=np.float32),
            allow_pickle=False,
        )
        replacement.replace(self.first_path)

        with self.assertRaisesRegex(ValueError, "changed after verification"):
            reader[0]

    def test_post_verification_symlink_substitution_is_rejected(self) -> None:
        reader = self._open()
        self.addCleanup(reader.close)

        verified_path = self.first_path.with_suffix(".verified.npy")
        self.first_path.replace(verified_path)
        self.first_path.symlink_to(verified_path.name)

        with self.assertRaisesRegex(ValueError, "symbolic link"):
            reader[0]

    def test_path_swap_during_descriptor_open_cannot_adopt_replacement(self) -> None:
        reader = self._open()
        self.addCleanup(reader.close)
        replacement = self.first_path.with_suffix(".replacement.npy")
        verified_path = self.first_path.with_suffix(".verified.npy")
        np.save(
            replacement,
            np.full((2, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), 0.9, dtype=np.float32),
            allow_pickle=False,
        )
        real_open = os.open
        swaps = 0

        def swapped_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
            nonlocal swaps
            if Path(path) != self.first_path or swaps:
                return real_open(path, flags, *args, **kwargs)
            swaps += 1
            self.first_path.replace(verified_path)
            replacement.replace(self.first_path)
            try:
                descriptor = real_open(path, flags, *args, **kwargs)
            finally:
                self.first_path.replace(replacement)
                verified_path.replace(self.first_path)
            return descriptor

        with (
            patch("bird_audio.training_data.os.open", side_effect=swapped_open),
            self.assertRaisesRegex(ValueError, "changed before descriptor mapping"),
        ):
            reader[0]
        self.assertEqual(swaps, 1)

    def test_path_swap_after_open_is_rejected_without_adopting_replacement(self) -> None:
        reader = self._open()
        self.addCleanup(reader.close)
        replacement = self.first_path.with_suffix(".replacement.npy")
        verified_path = self.first_path.with_suffix(".verified.npy")
        np.save(
            replacement,
            np.full((2, 1, NATIVE_MEL_HEIGHT, NATIVE_MEL_WIDTH), 0.9, dtype=np.float32),
            allow_pickle=False,
        )
        real_mmap = mmap.mmap
        swaps = 0
        mapped_inodes: list[int] = []
        mapped_backings: list[mmap.mmap] = []

        def swapped_map(descriptor: int, *args: object, **kwargs: object) -> mmap.mmap:
            nonlocal swaps
            swaps += 1
            mapped_inodes.append(os.fstat(descriptor).st_ino)
            self.first_path.replace(verified_path)
            replacement.replace(self.first_path)
            try:
                mapped = real_mmap(descriptor, *args, **kwargs)
                mapped_backings.append(mapped)
            finally:
                self.first_path.replace(replacement)
                verified_path.replace(self.first_path)
            return mapped

        with (
            patch("bird_audio.training_data.mmap.mmap", side_effect=swapped_map),
            self.assertRaisesRegex(ValueError, "changed while descriptor was mapped"),
        ):
            reader[0]

        self.assertEqual(swaps, 1)
        verified = reader._verified_files[self.rows[0]["feature_file"]]
        self.assertEqual(mapped_inodes, [verified.identity.inode])
        self.assertEqual(len(mapped_backings), 1)
        self.assertTrue(mapped_backings[0].closed)

    def test_hash_drift_is_rejected_before_reader_publication(self) -> None:
        rows = tuple(dict(row) for row in self.rows)
        rows[0]["feature_file_sha256"] = "0" * 64
        rows[1]["feature_file_sha256"] = "0" * 64
        source = self._source()
        source.rows = rows
        with (
            patch(
                "bird_audio.training_data.load_development_clip_cache",
                return_value=source,
            ),
            self.assertRaisesRegex(ValueError, "hash drift"),
        ):
            DevelopmentTrainingData(self.root, "train", "energy")

    def test_open_function_returns_the_verified_reader(self) -> None:
        with patch(
            "bird_audio.training_data.load_development_clip_cache",
            return_value=self._source(),
        ):
            reader = open_development_training_data(
                self.root,
                split="train",
                strategy="energy",
                mmap_capacity=3,
            )
        self.addCleanup(reader.close)
        self.assertIsInstance(reader, DevelopmentTrainingData)
        self.assertEqual(reader.mmap_capacity, 3)


if __name__ == "__main__":
    unittest.main()
