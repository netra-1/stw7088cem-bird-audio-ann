from __future__ import annotations

import unittest
from collections.abc import Iterator, Sequence
from unittest import mock

import numpy as np
import torch
from torch import nn

from bird_audio import final_evaluation_inference as inference
from bird_audio.task1_final_metrics import RecordingPrediction
from bird_audio.task2_scoring import RecordingBatch, RecordingScore


class SyntheticFinalData(Sequence[tuple[np.ndarray, dict[str, str]]]):
    def __init__(
        self,
        *,
        role: str,
        recording_clip_counts: tuple[int, ...],
        reverse_recording_order: bool = False,
    ) -> None:
        self.role = role
        self.split = "test" if role == inference.FINAL_KNOWN_TEST_ROLE else "unknown"
        self.strategy = "energy"
        self.lock_sha256 = (
            inference.KNOWN_CACHE_LOCK_SHA256
            if role == inference.FINAL_KNOWN_TEST_ROLE
            else inference.UNKNOWN_CACHE_LOCK_SHA256
        )
        ids = [f"R{index:03d}" for index in range(len(recording_clip_counts))]
        if reverse_recording_order:
            ids.reverse()
        self.recording_ids = tuple(ids)
        self.recording_count = len(ids)
        self.loaded_recording_ids: list[str] = []
        self._rows: list[dict[str, str]] = []
        self._groups: list[tuple[str, tuple[int, ...]]] = []
        position = 0
        for recording_id, clip_count in zip(ids, recording_clip_counts, strict=True):
            indices = tuple(range(position, position + clip_count))
            self._groups.append((recording_id, indices))
            for clip_index in range(clip_count):
                row = {
                    "recording_id": recording_id,
                    "clip_id": f"{recording_id}:{clip_index:03d}",
                    "session_group": f"session:{recording_id}",
                    "selection_strategy": "energy",
                    "strategy_clip_count": str(clip_count),
                }
                if role == inference.FINAL_KNOWN_TEST_ROLE:
                    row.update(
                        {
                            "split": "test",
                            "data_boundary": "gated_final_known_test",
                            "species_common_name": "Asian Koel",
                            "class_index": "0",
                        }
                    )
                else:
                    row.update(
                        {
                            "data_boundary": "gated_final_unknown",
                            "species_common_name": "Brown-headed Barbet",
                            "species_scientific_name": "Psilopogon zeylanicus",
                        }
                    )
                self._rows.append(row)
            position += clip_count

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]:
        row = self._rows[index]
        value = (index % 5) / 5.0
        return np.full((1, 128, 372), value, dtype=np.float32), dict(row)

    def iter_metadata(self) -> Iterator[dict[str, str]]:
        for row in self._rows:
            yield dict(row)

    def iter_recording_indices(self) -> Iterator[tuple[str, tuple[int, ...]]]:
        yield from self._groups

    def get_recording(
        self,
        recording_id: str,
    ) -> tuple[np.ndarray, tuple[dict[str, str], ...]]:
        groups = dict(self._groups)
        indices = groups[recording_id]
        self.loaded_recording_ids.append(recording_id)
        samples = [self[index] for index in indices]
        return (
            np.stack([sample[0] for sample in samples]).astype(np.float32, copy=False),
            tuple(sample[1] for sample in samples),
        )


class FixedClassifier(nn.Module):
    def __init__(self, *, output_columns: int = 15, finite: bool = True) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_columns = output_columns
        self.finite = finite
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(inputs.shape[0])
        logits = torch.zeros(
            (inputs.shape[0], self.output_columns),
            dtype=torch.float32,
            device=inputs.device,
        )
        logits = logits + self.anchor * 0.0
        logits[:, 0] = 1.0
        if not self.finite:
            logits[0, 0] = float("nan")
        return logits


class BatchSensitiveClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        self.batch_sizes.append(batch_size)
        logits = torch.zeros((batch_size, 15), dtype=torch.float32, device=inputs.device)
        logits[:, 0] = float(batch_size) + self.anchor * 0.0
        logits[:, 1] = inputs.mean(dim=(1, 2, 3))
        return logits


class IdentityAutoencoder(nn.Module):
    def __init__(self, *, latent_columns: int = 64, finite: bool = True) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.latent_columns = latent_columns
        self.finite = finite
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.batch_sizes.append(inputs.shape[0])
        reconstruction = inputs.clone() + self.anchor * 0.0
        latent = inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        latent = latent.expand(-1, self.latent_columns).clone() + self.anchor * 0.0
        if not self.finite:
            latent[0, 0] = float("inf")
        return reconstruction, latent


class BatchSensitiveAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.batch_sizes: list[int] = []

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        self.batch_sizes.append(batch_size)
        offset = torch.tensor(float(batch_size) / 100.0, device=inputs.device)
        reconstruction = inputs + offset + self.anchor * 0.0
        latent = inputs.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        latent = latent.expand(-1, 64).clone() + float(batch_size) + self.anchor * 0.0
        return reconstruction, latent


def _injection(
    known_data: SyntheticFinalData | None = None,
    unknown_data: SyntheticFinalData | None = None,
) -> inference.FinalInferenceTestInjection:
    return inference.FinalInferenceTestInjection(
        known_test_clips=len(known_data) if known_data is not None else 1,
        known_test_recordings=known_data.recording_count if known_data is not None else 1,
        unknown_clips=len(unknown_data) if unknown_data is not None else 1,
        unknown_recordings=unknown_data.recording_count if unknown_data is not None else 1,
    )


class RecordingBatchPlanningTests(unittest.TestCase):
    def test_greedy_batches_never_split_a_recording(self) -> None:
        counts = tuple((f"R{index}", 5) for index in range(8))
        batches = inference.recording_preserving_batches(counts, batch_size=32)
        self.assertEqual(tuple(len(batch) for batch in batches), (6, 2))
        resolved = dict(counts)
        self.assertEqual(
            tuple(sum(resolved[recording_id] for recording_id in batch) for batch in batches),
            (30, 10),
        )
        self.assertEqual(tuple(item for batch in batches for item in batch), tuple(dict(counts)))

    def test_batch_planning_rejects_duplicate_oversized_and_invalid_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inference.recording_preserving_batches((("R1", 1), ("R1", 1)), batch_size=32)
        with self.assertRaisesRegex(ValueError, "larger"):
            inference.recording_preserving_batches((("R1", 33),), batch_size=32)
        with self.assertRaisesRegex(ValueError, "positive"):
            inference.recording_preserving_batches((("R1", 0),), batch_size=32)


class Task1InferenceTests(unittest.TestCase):
    def test_locked_preprocessing_whole_recording_batches_and_one_transfer(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(5,) * 8,
            reverse_recording_order=True,
        )
        model = FixedClassifier()
        transfer_count = 0
        original = inference._task1_logits_to_cpu

        def transfer(logits: torch.Tensor) -> torch.Tensor:
            nonlocal transfer_count
            transfer_count += 1
            return original(logits)

        with mock.patch.object(inference, "_task1_logits_to_cpu", transfer):
            predictions = inference.infer_task1_recording_data(
                model,
                data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
            )
        self.assertEqual(model.batch_sizes, [30, 10])
        self.assertEqual(transfer_count, 2)
        self.assertEqual(
            tuple(value.recording_id for value in predictions), tuple(sorted(data.recording_ids))
        )
        self.assertTrue(all(isinstance(value, RecordingPrediction) for value in predictions))
        self.assertTrue(all(value.true_class_index == 0 for value in predictions))
        self.assertTrue(all(value.predicted_class_index == 0 for value in predictions))

    def test_iterator_preserves_partial_canonical_batch_when_skipping(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(2, 2, 2, 2),
        )
        skipped = tuple(sorted(data.recording_ids))[:2]
        batches = tuple(
            inference.iter_task1_recording_batches(
                FixedClassifier(),
                data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
                skip_recording_ids=set(skipped),
            )
        )
        self.assertEqual(
            tuple(value for batch in batches for value in batch.recording_ids), ("R002", "R003")
        )
        self.assertEqual(data.loaded_recording_ids, ["R000", "R001", "R002", "R003"])
        with self.assertRaisesRegex(ValueError, "outside"):
            tuple(
                inference.iter_task1_recording_batches(
                    FixedClassifier(),
                    data,
                    device=torch.device("cpu"),
                    test_injection=_injection(known_data=data),
                    skip_recording_ids=("missing",),
                )
            )

    def test_resume_outputs_match_uninterrupted_canonical_batches(self) -> None:
        complete_data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(5,) * 8,
        )
        complete_model = BatchSensitiveClassifier()
        complete = tuple(
            prediction
            for batch in inference.iter_task1_recording_batches(
                complete_model,
                complete_data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=complete_data),
            )
            for prediction in batch.predictions
        )
        resumed_data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(5,) * 8,
        )
        resumed_model = BatchSensitiveClassifier()
        resumed = tuple(
            prediction
            for batch in inference.iter_task1_recording_batches(
                resumed_model,
                resumed_data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=resumed_data),
                skip_recording_ids=("R000",),
            )
            for prediction in batch.predictions
        )
        expected = tuple(value for value in complete if value.recording_id != "R000")
        self.assertEqual(resumed, expected)
        self.assertEqual(complete_model.batch_sizes, [30, 10])
        self.assertEqual(resumed_model.batch_sizes, [30, 10])
        self.assertIn("R000", resumed_data.loaded_recording_ids)

    def test_resume_omits_fully_completed_canonical_batch(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(5,) * 8,
        )
        model = BatchSensitiveClassifier()
        batches = tuple(
            inference.iter_task1_recording_batches(
                model,
                data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
                skip_recording_ids=tuple(f"R{index:03d}" for index in range(6)),
            )
        )
        self.assertEqual(
            tuple(value for batch in batches for value in batch.recording_ids), ("R006", "R007")
        )
        self.assertEqual(model.batch_sizes, [10])
        self.assertEqual(data.loaded_recording_ids, ["R006", "R007"])

    def test_task1_rejects_nonproduction_scope_mapping_and_output_drift(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(1,),
        )
        with self.assertRaisesRegex(PermissionError, "MPS"):
            inference.infer_task1_recording_data(
                FixedClassifier(),
                data,
                device=torch.device("cpu"),
            )
        data._rows[0]["class_index"] = "1"
        with self.assertRaisesRegex(ValueError, "mapping"):
            inference.infer_task1_recording_data(
                FixedClassifier(),
                data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
            )
        data._rows[0]["class_index"] = "0"
        with self.assertRaisesRegex(ValueError, "output contract"):
            inference.infer_task1_recording_data(
                FixedClassifier(output_columns=14),
                data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
            )
        transfer_count = 0
        original = inference._task1_logits_to_cpu

        def transfer(logits: torch.Tensor) -> torch.Tensor:
            nonlocal transfer_count
            transfer_count += 1
            return original(logits)

        with (
            mock.patch.object(inference, "_task1_logits_to_cpu", transfer),
            self.assertRaisesRegex(ValueError, "CPU logits"),
        ):
            inference.infer_task1_recording_data(
                FixedClassifier(finite=False),
                data,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
            )
        self.assertEqual(transfer_count, 1)


class Task2InferenceTests(unittest.TestCase):
    def test_task2_batches_reconstruction_latent_aggregation_and_metadata(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_UNKNOWN_ROLE,
            recording_clip_counts=(5,) * 14,
            reverse_recording_order=True,
        )
        model = IdentityAutoencoder()
        transfer_count = 0
        original = inference._task2_outputs_to_cpu

        def transfer(
            reconstruction: torch.Tensor,
            latent: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal transfer_count
            transfer_count += 1
            return original(reconstruction, latent)

        with mock.patch.object(inference, "_task2_outputs_to_cpu", transfer):
            scores, metadata = inference.infer_task2_recording_data(
                model,
                data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=data),
            )
        self.assertEqual(model.batch_sizes, [60, 10])
        self.assertEqual(transfer_count, 2)
        self.assertEqual(scores.source_role, inference.FINAL_UNKNOWN_ROLE)
        self.assertEqual(scores.recording_ids, tuple(sorted(data.recording_ids)))
        self.assertTrue(all(recording.reconstruction_mse == 0.0 for recording in scores.recordings))
        self.assertTrue(
            all(len(recording.mean_latent_embedding) == 64 for recording in scores.recordings)
        )
        self.assertEqual(tuple(value.recording_id for value in metadata), scores.recording_ids)
        self.assertTrue(all(value.class_index is None for value in metadata))
        with self.assertRaises((AttributeError, TypeError)):
            metadata[0].recording_id = "changed"

    def test_task2_iterator_supports_resume_skips_and_immutable_known_metadata(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_KNOWN_TEST_ROLE,
            recording_clip_counts=(2, 2, 2),
        )
        batches = tuple(
            inference.iter_task2_recording_batches(
                IdentityAutoencoder(),
                data,
                source_role=inference.FINAL_KNOWN_TEST_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(known_data=data),
                skip_recording_ids=("R001",),
            )
        )
        self.assertEqual(
            tuple(value for batch in batches for value in batch.recording_ids), ("R000", "R002")
        )
        self.assertIn("R001", data.loaded_recording_ids)
        metadata = tuple(value for batch in batches for value in batch.metadata)
        self.assertTrue(
            all(value.species_scientific_name == "Eudynamys scolopaceus" for value in metadata)
        )
        self.assertTrue(all(value.class_index == 0 for value in metadata))

    def test_resume_outputs_match_uninterrupted_task2_canonical_batches(self) -> None:
        complete_data = SyntheticFinalData(
            role=inference.FINAL_UNKNOWN_ROLE,
            recording_clip_counts=(10,) * 7,
        )
        complete_model = BatchSensitiveAutoencoder()
        complete_batches = tuple(
            inference.iter_task2_recording_batches(
                complete_model,
                complete_data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=complete_data),
            )
        )
        complete_scores = {
            score.recording_id: score
            for batch in complete_batches
            for score in batch.scores.recordings
        }
        resumed_data = SyntheticFinalData(
            role=inference.FINAL_UNKNOWN_ROLE,
            recording_clip_counts=(10,) * 7,
        )
        resumed_model = BatchSensitiveAutoencoder()
        resumed_batches = tuple(
            inference.iter_task2_recording_batches(
                resumed_model,
                resumed_data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=resumed_data),
                skip_recording_ids=("R000",),
            )
        )
        resumed_scores = {
            score.recording_id: score
            for batch in resumed_batches
            for score in batch.scores.recordings
        }
        self.assertEqual(
            resumed_scores,
            {
                recording_id: score
                for recording_id, score in complete_scores.items()
                if recording_id != "R000"
            },
        )
        self.assertEqual(complete_model.batch_sizes, [60, 10])
        self.assertEqual(resumed_model.batch_sizes, [60, 10])
        self.assertIn("R000", resumed_data.loaded_recording_ids)

    def test_task2_rejects_identity_model_dtype_and_output_drift(self) -> None:
        data = SyntheticFinalData(
            role=inference.FINAL_UNKNOWN_ROLE,
            recording_clip_counts=(2,),
        )
        data._rows[1]["session_group"] = "another-session"
        with self.assertRaisesRegex(ValueError, "immutable"):
            inference.infer_task2_recording_data(
                IdentityAutoencoder(),
                data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=data),
            )
        data._rows[1]["session_group"] = data._rows[0]["session_group"]
        model = IdentityAutoencoder().double()
        with self.assertRaisesRegex(TypeError, "float32"):
            inference.infer_task2_recording_data(
                model,
                data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=data),
            )
        with self.assertRaisesRegex(ValueError, "latent"):
            inference.infer_task2_recording_data(
                IdentityAutoencoder(latent_columns=63),
                data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=data),
            )
        with self.assertRaisesRegex(ValueError, "CPU outputs"):
            inference.infer_task2_recording_data(
                IdentityAutoencoder(finite=False),
                data,
                source_role=inference.FINAL_UNKNOWN_ROLE,
                device=torch.device("cpu"),
                test_injection=_injection(unknown_data=data),
            )


class WrapperTests(unittest.TestCase):
    def test_simple_wrappers_use_only_gated_readers(self) -> None:
        authorization = object()
        known = object()
        unknown = object()
        task1_model = object()
        task2_model = object()
        task1_expected = (
            RecordingPrediction(
                recording_id="R1",
                session_group="S1",
                true_class_index=0,
                mean_logits=(1.0, *([0.0] * 14)),
                predicted_class_index=0,
            ),
        )
        known_batch = RecordingBatch(
            source_role=inference.FINAL_KNOWN_TEST_ROLE,
            recordings=(RecordingScore("R1", ("C1",), 0.0, (0.0,) * 64),),
        )
        unknown_batch = RecordingBatch(
            source_role=inference.FINAL_UNKNOWN_ROLE,
            recordings=(RecordingScore("U1", ("C2",), 0.0, (0.0,) * 64),),
        )
        known_metadata = (
            inference.FinalRecordingMetadata(
                inference.FINAL_KNOWN_TEST_ROLE,
                "R1",
                "S1",
                "Asian Koel",
                "Eudynamys scolopaceus",
                0,
                ("C1",),
            ),
        )
        unknown_metadata = (
            inference.FinalRecordingMetadata(
                inference.FINAL_UNKNOWN_ROLE,
                "U1",
                "S2",
                "Brown-headed Barbet",
                "Psilopogon zeylanicus",
                None,
                ("C2",),
            ),
        )
        with (
            mock.patch.object(inference, "open_final_known_test_data", return_value=known),
            mock.patch.object(inference, "open_final_unknown_data", return_value=unknown),
            mock.patch.object(
                inference,
                "infer_task1_recording_data",
                return_value=task1_expected,
            ) as task1_engine,
            mock.patch.object(
                inference,
                "infer_task2_recording_data",
                side_effect=((known_batch, known_metadata), (unknown_batch, unknown_metadata)),
            ) as task2_engine,
        ):
            task1 = inference.run_task1_final_inference(
                task1_model, authorization, device=torch.device("mps")
            )
            task2 = inference.run_task2_final_inference(
                task2_model, authorization, device=torch.device("mps")
            )
        self.assertEqual(task1, task1_expected)
        self.assertEqual(task2.known_test, known_batch)
        self.assertEqual(task2.unknown, unknown_batch)
        task1_engine.assert_called_once_with(task1_model, known, device=torch.device("mps"))
        self.assertEqual(task2_engine.call_count, 2)


if __name__ == "__main__":
    unittest.main()
