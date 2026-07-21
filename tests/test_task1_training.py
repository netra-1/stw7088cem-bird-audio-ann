from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0

import bird_audio.task1_training as task1_training
from bird_audio.hashing import sha256_file
from bird_audio.paths import PROJECT_ROOT
from bird_audio.task1_training import (
    CONSERVATIVE_WALL_TIME_FACTOR,
    DEFAULT_RUN_ROOT,
    KNOWN_CACHE_LOCK_SHA256,
    CheckpointScore,
    EarlyStopping,
    RecordingPredictions,
    Task1TestInjection,
    WeightArtifact,
    _assert_final_config,
    _build_model,
    _open_real_data,
    _read_json_snapshot,
    _resolve_runtime,
    _write_json_create_only,
    aggregate_recording_logits,
    benchmark_task1_full_epoch,
    build_task1_optimizer,
    fixed_class_metrics,
    is_better_checkpoint,
    load_final_task1_config,
    load_task1_checkpoint,
    run_task1_development,
    save_task1_checkpoint_create_only,
    train_task1_epoch,
    validate_task1,
)


def _overwrite_json_record(path: Path, value: object) -> dict[str, object]:
    payload = task1_training._json_bytes(value)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _overwrite_checkpoint_record(path: Path, value: object) -> dict[str, object]:
    torch.save(value, path)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


class _TinyDevelopmentData:
    def __init__(
        self,
        root: Path,
        split: str,
        class_order: list[str],
        definitions: tuple[tuple[str, int, int], ...],
        *,
        lock_sha256: str = "d" * 64,
    ) -> None:
        self.root = root
        self.split = split
        self.strategy = "energy"
        self.lock_sha256 = lock_sha256
        self._features: list[np.ndarray] = []
        self._rows: list[dict[str, str]] = []
        for recording_number, (recording_id, class_index, clip_count) in enumerate(definitions):
            for rank in range(clip_count):
                value = 0.15 + 0.1 * class_index + 0.01 * rank
                self._features.append(np.full((1, 128, 372), value, dtype=np.float32))
                self._rows.append(
                    {
                        "recording_id": recording_id,
                        "species_common_name": class_order[class_index],
                        "class_index": str(class_index),
                        "session_group": f"{split}:session:{recording_number}",
                        "split": split,
                        "selection_strategy": "energy",
                        "strategy_clip_count": str(clip_count),
                        "energy_rank": str(rank),
                    }
                )
        self.recording_count = len(definitions)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, str]]:
        return self._features[index].copy(), dict(self._rows[index])

    def metadata(self, index: int) -> dict[str, str]:
        return dict(self._rows[index])

    def iter_metadata(self):
        for row in self._rows:
            yield dict(row)


class _TinyClassifier(nn.Module):
    def __init__(self, class_count: int = 15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.classifier = nn.Linear(4, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class _WrongLogitClassifier(_TinyClassifier):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs)[:, :-1]


class _FailOnSecondEpochClassifier(_TinyClassifier):
    def __init__(self, class_count: int = 15) -> None:
        super().__init__(class_count)
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if self.forward_calls >= 5:
            raise RuntimeError("injected second epoch failure")
        return super().forward(inputs)


class Task1MetricTests(unittest.TestCase):
    def test_recording_logits_are_averaged_before_prediction(self) -> None:
        logits = torch.zeros((3, 15), dtype=torch.float32)
        logits[0, 0] = 6.0
        logits[0, 1] = 1.0
        logits[1, 0] = 0.0
        logits[1, 1] = 8.0
        logits[2, 2] = 4.0
        labels = torch.tensor([1, 1, 2], dtype=torch.long)
        metadata = (
            {"recording_id": "A", "session_group": "S-A"},
            {"recording_id": "A", "session_group": "S-A"},
            {"recording_id": "B", "session_group": "S-B"},
        )

        predictions = aggregate_recording_logits(logits, labels, metadata)

        self.assertEqual(predictions.recording_ids, ("A", "B"))
        self.assertEqual(predictions.session_groups, ("S-A", "S-B"))
        self.assertEqual(predictions.true_labels.tolist(), [1, 2])
        self.assertEqual(predictions.predicted_labels.tolist(), [1, 2])
        torch.testing.assert_close(
            predictions.mean_logits[0, :2],
            torch.tensor([3.0, 4.5]),
            rtol=0,
            atol=0,
        )

    def test_metrics_use_all_fifteen_fixed_classes_with_zero_division_zero(self) -> None:
        metrics = fixed_class_metrics(
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 0], dtype=torch.long),
            class_count=15,
        )

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["per_class_f1"][0], 2 / 3)
        self.assertEqual(metrics["per_class_f1"][1:], [0.0] * 14)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / 15)
        self.assertEqual(len(metrics["confusion_matrix"]), 15)


class Task1SelectionTests(unittest.TestCase):
    def test_checkpoint_priority_and_early_stopping_are_exact(self) -> None:
        incumbent = CheckpointScore(0.7, 0.5, 2)
        self.assertTrue(is_better_checkpoint(CheckpointScore(0.8, 9.0, 9), incumbent))
        self.assertTrue(is_better_checkpoint(CheckpointScore(0.7, 0.4, 9), incumbent))
        self.assertTrue(is_better_checkpoint(CheckpointScore(0.7, 0.5, 1), incumbent))
        self.assertFalse(is_better_checkpoint(CheckpointScore(0.7, 0.5, 3), incumbent))

        stopping = EarlyStopping(patience=2)
        self.assertEqual(stopping.update(CheckpointScore(0.7, 0.5, 1)), (True, False))
        self.assertEqual(stopping.update(CheckpointScore(0.6, 0.4, 2)), (False, False))
        self.assertEqual(stopping.update(CheckpointScore(0.6, 0.3, 3)), (False, True))
        self.assertEqual(stopping.epochs_without_improvement, 2)

    def test_optimizer_has_disjoint_backbone_and_head_groups(self) -> None:
        config = load_final_task1_config()
        model = _TinyClassifier()

        optimizer = build_task1_optimizer(model, config)

        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups], ["backbone", "head"]
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.00003, 0.0003],
        )
        self.assertTrue(all(group["weight_decay"] == 0.0001 for group in optimizer.param_groups))
        for group in optimizer.param_groups:
            self.assertEqual(group["betas"], (0.9, 0.999))
            self.assertEqual(group["eps"], 1e-8)
            self.assertFalse(group["amsgrad"])
            self.assertFalse(group["maximize"])
            self.assertFalse(group["foreach"])
            self.assertFalse(group["capturable"])
            self.assertFalse(group["differentiable"])
            self.assertFalse(group["fused"])
            self.assertTrue(group["decoupled_weight_decay"])
        backbone = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        head = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
        self.assertFalse(backbone.intersection(head))
        self.assertEqual(
            backbone.union(head),
            {id(parameter) for parameter in model.parameters()},
        )

    def test_training_guard_rejects_every_previously_unlocked_final_field(self) -> None:
        base = load_final_task1_config()
        mutations = {
            "dropout": ("dropout", 0.9),
            "determinism_failure_policy": (
                "training.determinism_failure_policy",
                "continue",
            ),
            "seed_python": ("training.seed_python", False),
            "seed_numpy": ("training.seed_numpy", False),
            "seed_torch": ("training.seed_torch", False),
            "seed_sampler": ("training.seed_sampler", False),
            "log_parameter_counts": ("training.log_parameter_counts", False),
        }
        for name, (path, value) in mutations.items():
            config = copy.deepcopy(base)
            if path.startswith("training."):
                config["training"][path.split(".", 1)[1]] = value
            else:
                config[path] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                _assert_final_config(config)

    def test_cpu_test_runtime_still_rejects_mps_fallback(self) -> None:
        temporary = PROJECT_ROOT / "data" / "interim" / "runtime-weight.pth"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"runtime test\n")
        self.addCleanup(temporary.unlink, missing_ok=True)
        injection = Task1TestInjection(
            model_factory=lambda _config: _TinyClassifier(),
            weight_artifact=WeightArtifact(
                temporary,
                sha256_file(temporary),
                temporary.stat().st_size,
            ),
        )
        with (
            patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}),
            self.assertRaisesRegex(RuntimeError, "must be disabled"),
        ):
            _resolve_runtime(injection)

    def test_training_and_benchmark_cannot_populate_weights(self) -> None:
        self.assertNotIn("populate_weights", inspect.signature(run_task1_development).parameters)
        self.assertNotIn(
            "populate_weights", inspect.signature(benchmark_task1_full_epoch).parameters
        )
        self.assertIn(
            "populate",
            inspect.signature(task1_training.preflight_efficientnet_weights).parameters,
        )

    def test_production_benchmark_paths_are_canonical_and_versioned(self) -> None:
        self.assertEqual(DEFAULT_RUN_ROOT, PROJECT_ROOT / "runs" / "task1_v2")
        self.assertEqual(
            task1_training.DEFAULT_BENCHMARK_RESULT_PATH,
            PROJECT_ROOT / "report_assets" / "provenance_v2" / "task1_benchmark_v2.json",
        )
        self.assertEqual(
            task1_training.DEFAULT_BENCHMARK_LOCK_PATH,
            PROJECT_ROOT / "report_assets" / "provenance_v2" / "task1_benchmark_v2.lock.json",
        )
        self.assertEqual(
            task1_training._benchmark_artifact_paths(None, None),
            (
                task1_training.DEFAULT_BENCHMARK_RESULT_PATH.resolve(),
                task1_training.DEFAULT_BENCHMARK_LOCK_PATH.resolve(),
            ),
        )

    def test_run_creation_rejects_symlinked_canonical_root_and_parent(self) -> None:
        runs_root = PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            boundary = Path(temporary)
            protected_root = boundary / "protected_task1_v1"
            protected_root.mkdir()
            linked_root = boundary / "task1_v2"
            linked_root.symlink_to(protected_root, target_is_directory=True)
            with (
                patch.object(task1_training, "DEFAULT_RUN_ROOT", linked_root),
                self.assertRaisesRegex(ValueError, "canonical v2 run root"),
            ):
                task1_training._run_directory(linked_root, "must_not_exist")
            self.assertEqual(list(protected_root.iterdir()), [])

            real_parent = boundary / "real_parent"
            real_parent.mkdir()
            linked_parent = boundary / "linked_parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            canonical_below_link = linked_parent / "task1_v2"
            with (
                patch.object(task1_training, "DEFAULT_RUN_ROOT", canonical_below_link),
                self.assertRaisesRegex(ValueError, "canonical v2 run root"),
            ):
                task1_training._run_directory(canonical_below_link, "must_not_exist")
            self.assertFalse((real_parent / "task1_v2").exists())

    def test_run_creation_allows_real_descendant_for_isolated_tests(self) -> None:
        runs_root = PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            canonical_root = Path(temporary) / "task1_v2"
            canonical_root.mkdir()
            isolated_root = canonical_root / "isolated"
            with patch.object(task1_training, "DEFAULT_RUN_ROOT", canonical_root):
                run = task1_training._run_directory(isolated_root, "safe_test_run")
            self.assertEqual(run, isolated_root / "safe_test_run")
            self.assertTrue(run.is_dir())

    def test_production_run_rejects_noncanonical_descendant_before_execution(self) -> None:
        runs_root = PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            canonical_root = Path(temporary) / "task1_v2"
            canonical_root.mkdir()
            descendant = canonical_root / "production_descendant"
            with (
                patch.object(task1_training, "DEFAULT_RUN_ROOT", canonical_root),
                patch.object(
                    task1_training,
                    "_resolve_runtime",
                    return_value=torch.device("cpu"),
                ),
                patch.object(task1_training, "_capture_execution_identity") as capture,
                self.assertRaisesRegex(PermissionError, "exact runs/task1_v2"),
            ):
                run_task1_development(seed=13, output_root=descendant)
            capture.assert_not_called()
            self.assertFalse(descendant.exists())

    def test_execution_identity_covers_provenance_and_portable_hardware(self) -> None:
        self.assertIn(
            "src/bird_audio/provenance.py",
            task1_training.TASK1_IMPLEMENTATION_FILES,
        )
        hardware = {
            "hw.model": "Mac15,3",
            "machdep.cpu.brand_string": "Apple M3",
        }
        with patch(
            "bird_audio.task1_training._portable_hardware_value",
            side_effect=lambda name: hardware[name],
        ):
            runtime = task1_training._numerical_runtime_identity(torch.device("cpu"))
        self.assertEqual(runtime["apple_hardware_model"], "Mac15,3")
        self.assertEqual(runtime["apple_processor_identifier"], "Apple M3")


class Task1EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_final_task1_config()
        self.class_order = list(config["class_order"])
        interim = PROJECT_ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        self.data_temporary = tempfile.TemporaryDirectory(dir=interim)
        self.addCleanup(self.data_temporary.cleanup)
        self.data_root = Path(self.data_temporary.name)

        DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
        self.run_temporary = tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT)
        self.addCleanup(self.run_temporary.cleanup)
        self.output_root = Path(self.run_temporary.name)

        self.weight_path = self.data_root / "verified-test-weight.pth"
        self.weight_path.write_bytes(b"isolated test weight artifact\n")
        self.weight = WeightArtifact(
            path=self.weight_path,
            sha256=sha256_file(self.weight_path),
            size_bytes=self.weight_path.stat().st_size,
        )
        self.train = _TinyDevelopmentData(
            self.data_root,
            "train",
            self.class_order,
            (("train-A", 0, 2), ("train-B", 1, 1)),
        )
        self.validation = _TinyDevelopmentData(
            self.data_root,
            "validation",
            self.class_order,
            (("validation-A", 0, 2), ("validation-B", 1, 1)),
        )
        self.environment = unittest.mock.patch.dict(
            os.environ,
            {"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def _injection(
        self,
        model_factory=None,
        *,
        maximum_epochs: int = 1,
        batch_size: int = 2,
        patience: int = 1,
    ) -> Task1TestInjection:
        factory = model_factory or (lambda config: _TinyClassifier(int(config["class_count"])))
        return Task1TestInjection(
            model_factory=factory,
            weight_artifact=self.weight,
            maximum_epochs=maximum_epochs,
            batch_size=batch_size,
            early_stopping_patience=patience,
        )

    def _valid_best_checkpoint_state(self) -> dict:
        model = _TinyClassifier()
        optimizer = build_task1_optimizer(model, load_final_task1_config())
        logits = torch.zeros((1, 15), dtype=torch.float32)
        predictions = RecordingPredictions(
            recording_ids=("recording-A",),
            session_groups=("session-A",),
            true_labels=torch.tensor([0], dtype=torch.long),
            mean_logits=logits,
            predicted_labels=torch.tensor([0], dtype=torch.long),
        )
        return {
            "schema_version": "1.1",
            "checkpoint_type": "best",
            "run_id": "checkpoint_unit",
            "run_identity_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "cache_lock_sha256": "c" * 64,
            "weight_sha256": "d" * 64,
            "implementation_sha256": "e" * 64,
            "requirements_lock_sha256": "f" * 64,
            "numerical_runtime_sha256": "0" * 64,
            "scope": "isolated_test",
            "production_evidence": False,
            "seed": 13,
            "epoch": 1,
            "score": {"macro_f1": 0.5, "validation_loss": 1.0, "epoch": 1},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "predictions": {
                "recording_ids": predictions.recording_ids,
                "session_groups": predictions.session_groups,
                "true_labels": predictions.true_labels,
                "mean_logits": predictions.mean_logits,
                "predicted_labels": predictions.predicted_labels,
            },
        }

    def _failed_recovery(self, run_id: str) -> tuple[Path, str]:
        with self.assertRaisesRegex(RuntimeError, "injected second epoch failure"):
            run_task1_development(
                seed=13,
                output_root=self.output_root,
                run_id=run_id,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(
                    lambda _config: _FailOnSecondEpochClassifier(),
                    maximum_epochs=2,
                    patience=2,
                ),
            )
        path = self.output_root / run_id / "recovery" / "recovery_epoch_0001.pt"
        return path, sha256_file(path)

    def _verified_run(
        self,
        run_id: str,
        *,
        maximum_epochs: int = 1,
        patience: int = 1,
    ) -> Path:
        run_task1_development(
            seed=13,
            output_root=self.output_root,
            run_id=run_id,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(
                maximum_epochs=maximum_epochs,
                patience=patience,
            ),
        )
        return self.output_root / run_id

    def _rebind_result_and_lock(
        self,
        run_directory: Path,
        result: dict,
        completion: dict,
    ) -> str:
        result_record = _overwrite_json_record(run_directory / "result.json", result)
        completion["result"] = result_record
        lock_record = _overwrite_json_record(run_directory / "result.lock.json", completion)
        return str(lock_record["sha256"])

    def _rewrite_best_and_latest(
        self,
        run_directory: Path,
        result: dict,
        best_checkpoint: dict,
    ) -> None:
        best_path = Path(result["best_checkpoint"]["path"])
        best_record = _overwrite_checkpoint_record(best_path, best_checkpoint)
        latest_path = Path(result["latest_recovery_checkpoint"]["path"])
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        latest["best_candidate"]["sha256"] = best_record["sha256"]
        latest_record = _overwrite_checkpoint_record(latest_path, latest)
        result["best_checkpoint"] = best_record
        result["latest_recovery_checkpoint"] = latest_record
        result["artifacts"]["best_checkpoint"] = best_record
        result["artifacts"]["latest_recovery"] = latest_record

    def test_tiny_cpu_epoch_and_recording_validation_integrate(self) -> None:
        config = load_final_task1_config()
        model = _TinyClassifier()
        optimizer = build_task1_optimizer(model, config)

        training = train_task1_epoch(
            model,
            optimizer,
            self.train,
            seed=13,
            epoch_index=0,
            batch_size=2,
            class_count=15,
            device=torch.device("cpu"),
        )
        validation = validate_task1(
            model,
            self.validation,
            batch_size=2,
            class_count=15,
            device=torch.device("cpu"),
        )

        self.assertEqual(training["clips"], len(self.train))
        self.assertEqual(training["batches"], 2)
        self.assertGreater(training["clip_loss"], 0.0)
        self.assertEqual(validation.clip_count, len(self.validation))
        self.assertEqual(validation.recording_count, 2)
        self.assertTrue(0.0 <= validation.macro_f1 <= 1.0)
        self.assertTrue(0.0 <= validation.accuracy <= 1.0)

    def test_real_recursive_verifier_accepts_an_isolated_completed_run(self) -> None:
        run_directory = self._verified_run("recursive_valid")
        lock_path = run_directory / "result.lock.json"

        verified = task1_training.verify_task1_development_run(
            lock_path,
            expected_sha256=sha256_file(lock_path),
            require_production=False,
        )

        self.assertTrue(verified["valid"])
        self.assertTrue(verified["macro_f1_rederived"])
        self.assertTrue(verified["selection_rederived"])
        self.assertEqual(verified["validation_recordings"], 2)

    def test_real_recursive_verifier_rejects_prediction_count_and_class_tampering(self) -> None:
        for name in ("count", "class"):
            with self.subTest(name=name):
                run_directory = self._verified_run(f"recursive_prediction_{name}")
                result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
                completion = json.loads(
                    (run_directory / "result.lock.json").read_text(encoding="utf-8")
                )
                best_path = Path(result["best_checkpoint"]["path"])
                best = torch.load(best_path, map_location="cpu", weights_only=True)
                predictions = best["predictions"]
                if name == "count":
                    predictions["recording_ids"] = predictions["recording_ids"][:1]
                    predictions["session_groups"] = predictions["session_groups"][:1]
                    predictions["true_labels"] = predictions["true_labels"][:1].clone()
                    predictions["mean_logits"] = predictions["mean_logits"][:1].clone()
                    predictions["predicted_labels"] = predictions["predicted_labels"][:1].clone()
                else:
                    row_count = len(predictions["recording_ids"])
                    logits = torch.zeros((row_count, 15), dtype=torch.float32)
                    logits[:, 0] = 1.0
                    predictions["true_labels"] = torch.zeros(row_count, dtype=torch.long)
                    predictions["mean_logits"] = logits
                    predictions["predicted_labels"] = torch.zeros(row_count, dtype=torch.long)
                self._rewrite_best_and_latest(run_directory, result, best)
                lock_sha256 = self._rebind_result_and_lock(
                    run_directory,
                    result,
                    completion,
                )

                with self.assertRaises(ValueError):
                    task1_training.verify_task1_development_run(
                        run_directory / "result.lock.json",
                        expected_sha256=lock_sha256,
                        require_production=False,
                    )

    def test_real_recursive_verifier_rejects_metric_history_and_selection_tampering(self) -> None:
        run_directory = self._verified_run("recursive_metric")
        result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
        completion = json.loads((run_directory / "result.lock.json").read_text(encoding="utf-8"))
        result["best_validation_macro_f1"] += 0.01
        lock_sha256 = self._rebind_result_and_lock(run_directory, result, completion)
        with self.assertRaises(ValueError):
            task1_training.verify_task1_development_run(
                run_directory / "result.lock.json",
                expected_sha256=lock_sha256,
                require_production=False,
            )

        run_directory = self._verified_run(
            "recursive_history",
            maximum_epochs=2,
            patience=5,
        )
        result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
        completion = json.loads((run_directory / "result.lock.json").read_text(encoding="utf-8"))
        history_path = run_directory / "epoch_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history[1]["validation"] = copy.deepcopy(history[0]["validation"])
        history[1]["checkpoint_improved"] = True
        history_record = _overwrite_json_record(history_path, history)
        latest_path = Path(result["latest_recovery_checkpoint"]["path"])
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        latest["history"] = history
        latest_record = _overwrite_checkpoint_record(latest_path, latest)
        result["artifacts"]["epoch_history"] = history_record
        result["latest_recovery_checkpoint"] = latest_record
        result["artifacts"]["latest_recovery"] = latest_record
        lock_sha256 = self._rebind_result_and_lock(run_directory, result, completion)
        with self.assertRaises(ValueError):
            task1_training.verify_task1_development_run(
                run_directory / "result.lock.json",
                expected_sha256=lock_sha256,
                require_production=False,
            )

    def test_real_recursive_verifier_rejects_candidate_and_limit_tampering(self) -> None:
        for name in ("candidate", "limits"):
            with self.subTest(name=name):
                run_directory = self._verified_run(f"recursive_{name}")
                result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
                completion = json.loads(
                    (run_directory / "result.lock.json").read_text(encoding="utf-8")
                )
                latest_path = Path(result["latest_recovery_checkpoint"]["path"])
                latest = torch.load(latest_path, map_location="cpu", weights_only=True)
                if name == "candidate":
                    latest["best_candidate"]["sha256"] = "f" * 64
                else:
                    latest["limits"]["maximum_epochs"] += 1
                latest_record = _overwrite_checkpoint_record(latest_path, latest)
                result["latest_recovery_checkpoint"] = latest_record
                result["artifacts"]["latest_recovery"] = latest_record
                lock_sha256 = self._rebind_result_and_lock(
                    run_directory,
                    result,
                    completion,
                )
                with self.assertRaises(ValueError):
                    task1_training.verify_task1_development_run(
                        run_directory / "result.lock.json",
                        expected_sha256=lock_sha256,
                        require_production=False,
                    )

    def test_real_recursive_verifier_rejects_resume_prefix_tampering(self) -> None:
        recovery_path, recovery_sha256 = self._failed_recovery("recursive_resume")
        run_task1_development(
            seed=13,
            output_root=self.output_root,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(maximum_epochs=2, patience=2),
            resume_checkpoint=recovery_path,
            resume_checkpoint_sha256=recovery_sha256,
        )
        run_directory = self.output_root / "recursive_resume"
        result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
        completion = json.loads((run_directory / "result.lock.json").read_text(encoding="utf-8"))
        resume_path = Path(result["resume_checkpoint"]["path"])
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        resume["history"][0]["validation"]["macro_f1"] += 0.01
        result["resume_checkpoint"] = _overwrite_checkpoint_record(resume_path, resume)
        lock_sha256 = self._rebind_result_and_lock(run_directory, result, completion)

        with self.assertRaises(ValueError):
            task1_training.verify_task1_development_run(
                run_directory / "result.lock.json",
                expected_sha256=lock_sha256,
                require_production=False,
            )

    def test_cpu_model_state_verifier_rejects_all_structural_tampering(self) -> None:
        config = load_final_task1_config()

        def state() -> dict:
            model = task1_training.build_efficientnet_b0_classifier(
                class_count=15,
                dropout=0.2,
                weights_identifier=None,
                trainable_feature_indices=[6, 7, 8],
            )
            optimizer = build_task1_optimizer(model, config)
            logits = torch.zeros((1, 15), dtype=torch.float32)
            return {
                "schema_version": task1_training.CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_type": "best",
                "run_id": "locked_state",
                "run_identity_sha256": "a" * 64,
                "config_sha256": task1_training.config_fingerprint(config),
                "cache_lock_sha256": KNOWN_CACHE_LOCK_SHA256,
                "weight_sha256": f"{task1_training.WEIGHT_HASH_PREFIX}{'b' * 56}",
                "implementation_sha256": "c" * 64,
                "requirements_lock_sha256": "d" * 64,
                "numerical_runtime_sha256": "e" * 64,
                "scope": "production",
                "production_evidence": True,
                "seed": 13,
                "epoch": 1,
                "score": {"macro_f1": 0.0, "validation_loss": 1.0, "epoch": 1},
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "predictions": {
                    "recording_ids": ("recording-A",),
                    "session_groups": ("session-A",),
                    "true_labels": torch.tensor([0], dtype=torch.long),
                    "mean_logits": logits,
                    "predicted_labels": torch.tensor([0], dtype=torch.long),
                },
            }

        valid_state = state()
        valid_path = self.output_root / "locked_valid.pt"
        valid_record = save_task1_checkpoint_create_only(valid_path, valid_state)
        metadata = task1_training.verify_locked_task1_best_checkpoint_model_state(
            valid_path,
            expected_sha256=valid_record["sha256"],
            expected_run_identity_sha256="a" * 64,
        )
        self.assertEqual(
            metadata["model_contract"]["parameter_counts"],
            {"total": 4_026_763, "trainable": 3_174_955},
        )

        for name in ("missing", "renamed", "shape", "nonfinite"):
            with self.subTest(name=name):
                tampered = state()
                key = next(iter(tampered["model"]))
                if name == "missing":
                    tampered["model"].pop(key)
                elif name == "renamed":
                    tampered["model"][f"{key}.renamed"] = tampered["model"].pop(key)
                elif name == "shape":
                    tampered["model"][key] = tampered["model"][key][:-1].clone()
                else:
                    float_key = next(
                        item
                        for item, tensor in tampered["model"].items()
                        if tensor.is_floating_point() and tensor.numel() > 0
                    )
                    changed = tampered["model"][float_key].clone()
                    changed.reshape(-1)[0] = float("nan")
                    tampered["model"][float_key] = changed
                path = self.output_root / f"locked_{name}.pt"
                record = _overwrite_checkpoint_record(path, tampered)
                with self.assertRaises(ValueError):
                    task1_training.verify_locked_task1_best_checkpoint_model_state(
                        path,
                        expected_sha256=record["sha256"],
                        expected_run_identity_sha256="a" * 64,
                    )

    def test_production_recovery_state_requires_locked_model_adamw_and_mps_rng(self) -> None:
        config = load_final_task1_config()
        model = task1_training.build_efficientnet_b0_classifier(
            class_count=15,
            dropout=0.2,
            weights_identifier=None,
            trainable_feature_indices=[6, 7, 8],
        )
        optimizer = build_task1_optimizer(model, config)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rng_state = task1_training._capture_rng_state(torch.device("cpu"))
        rng_state["device"] = "mps"
        rng_state["torch_mps"] = torch.get_rng_state().clone()
        recovery = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": rng_state,
        }

        task1_training._verify_locked_task1_recovery_state(recovery, config)

        original_rate = recovery["optimizer"]["param_groups"][0]["lr"]
        recovery["optimizer"]["param_groups"][0]["lr"] = original_rate * 2
        with self.assertRaises(ValueError):
            task1_training._verify_locked_task1_recovery_state(recovery, config)
        recovery["optimizer"]["param_groups"][0]["lr"] = original_rate

        recovery["rng_state"]["device"] = "cpu"
        recovery["rng_state"]["torch_mps"] = None
        with self.assertRaises(ValueError):
            task1_training._verify_locked_task1_recovery_state(recovery, config)

    def test_run_writes_verified_create_only_provenance_and_checkpoint(self) -> None:
        result = run_task1_development(
            seed=13,
            output_root=self.output_root,
            run_id="unit_success",
            command=("python", "task1-unit"),
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(),
        )
        run_directory = Path(result["run_directory"])
        expected_names = {
            "best_candidates",
            "best_validation_predictions.json",
            "epoch_history.json",
            "failures",
            "provenance.json",
            "recovery",
            "resolved_config.json",
            "result.json",
            "result.lock.json",
            "run_identity.json",
        }
        self.assertEqual({path.name for path in run_directory.iterdir()}, expected_names)
        self.assertEqual(list((run_directory / "failures").iterdir()), [])
        self.assertEqual(
            [path.name for path in (run_directory / "best_candidates").iterdir()],
            ["best_epoch_0001.pt"],
        )
        self.assertEqual(
            [path.name for path in (run_directory / "recovery").iterdir()],
            ["recovery_epoch_0001.pt"],
        )

        identity = json.loads((run_directory / "run_identity.json").read_text(encoding="utf-8"))
        provenance = json.loads((run_directory / "provenance.json").read_text(encoding="utf-8"))
        history = json.loads((run_directory / "epoch_history.json").read_text(encoding="utf-8"))
        self.assertEqual(identity["config_sha256"], result["config_sha256"])
        self.assertEqual(identity["cache_lock_sha256"], self.train.lock_sha256)
        self.assertEqual(identity["weight_sha256"], self.weight.sha256)
        self.assertEqual(identity["scope"], "isolated_test")
        self.assertFalse(identity["production_evidence"])
        self.assertRegex(identity["implementation_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(identity["requirements_lock_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(identity["numerical_runtime_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(provenance["weight_path"], str(self.weight.path))
        self.assertEqual(provenance["weight_sha256"], self.weight.sha256)
        self.assertEqual(provenance["scope"], "isolated_test")
        self.assertFalse(provenance["production_evidence"])
        self.assertRegex(provenance["source_fingerprint_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            provenance["numerical_runtime_sha256"], identity["numerical_runtime_sha256"]
        )
        self.assertEqual(provenance["environment"]["device"], "cpu")
        self.assertTrue(provenance["environment"]["deterministic_algorithms"])
        self.assertGreater(provenance["parameter_counts"]["trainable"], 0)
        self.assertEqual(len(history), 1)

        checkpoint = load_task1_checkpoint(
            result["best_checkpoint"]["path"],
            expected_sha256=result["best_checkpoint"]["sha256"],
            expected_run_identity_sha256=result["run_identity_sha256"],
            expected_type="best",
        )
        self.assertEqual(checkpoint["config_sha256"], result["config_sha256"])
        self.assertEqual(checkpoint["cache_lock_sha256"], self.train.lock_sha256)
        self.assertEqual(checkpoint["weight_sha256"], self.weight.sha256)
        self.assertEqual(checkpoint["checkpoint_type"], "best")
        self.assertEqual(checkpoint["scope"], "isolated_test")
        self.assertFalse(checkpoint["production_evidence"])
        self.assertEqual(result["scope"], "isolated_test")
        self.assertFalse(result["production_evidence"])
        completion = json.loads((run_directory / "result.lock.json").read_text(encoding="utf-8"))
        self.assertEqual(completion["result"]["sha256"], sha256_file(run_directory / "result.json"))
        self.assertEqual(completion["scope"], "isolated_test")
        self.assertFalse(completion["production_evidence"])
        self.assertEqual(
            completion["source_fingerprint_sha256"],
            provenance["source_fingerprint_sha256"],
        )
        result_sha256 = sha256_file(run_directory / "result.json")
        with self.assertRaises(FileExistsError):
            run_task1_development(
                seed=13,
                output_root=self.output_root,
                run_id="unit_success",
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )
        self.assertEqual(sha256_file(run_directory / "result.json"), result_sha256)

    def test_checkpoint_save_refuses_overwrite_and_verifies_round_trip(self) -> None:
        state = self._valid_best_checkpoint_state()
        path = self.output_root / "direct_checkpoint.pt"

        record = save_task1_checkpoint_create_only(path, state)

        self.assertEqual(record["sha256"], sha256_file(path))
        loaded = load_task1_checkpoint(
            path,
            expected_sha256=record["sha256"],
            expected_type="best",
        )
        self.assertEqual(loaded["epoch"], 1)
        with self.assertRaises(FileExistsError):
            save_task1_checkpoint_create_only(path, state)

    def test_checkpoint_load_rejects_changed_bytes_and_invalid_typed_metadata(self) -> None:
        state = self._valid_best_checkpoint_state()
        path = self.output_root / "tamper_checkpoint.pt"
        record = save_task1_checkpoint_create_only(path, state)

        changed = copy.deepcopy(state)
        first_key = next(iter(changed["model"]))
        changed["model"][first_key] = changed["model"][first_key] + 1
        torch.save(changed, path)
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            load_task1_checkpoint(
                path,
                expected_sha256=record["sha256"],
                expected_type="best",
            )

        invalid = copy.deepcopy(state)
        invalid["seed"] = "13"
        torch.save(invalid, path)
        with self.assertRaisesRegex(ValueError, "seed is invalid"):
            load_task1_checkpoint(
                path,
                expected_sha256=sha256_file(path),
                expected_type="best",
            )

        for field in (
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
        ):
            invalid = copy.deepcopy(state)
            invalid[field] = "invalid"
            torch.save(invalid, path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                load_task1_checkpoint(
                    path,
                    expected_sha256=sha256_file(path),
                    expected_type="best",
                )

        invalid = copy.deepcopy(state)
        invalid["scope"] = "production"
        torch.save(invalid, path)
        with self.assertRaisesRegex(ValueError, "evidence scope"):
            load_task1_checkpoint(
                path,
                expected_sha256=sha256_file(path),
                expected_type="best",
            )

    def test_checkpoint_and_json_reads_reject_symbolic_links(self) -> None:
        state = self._valid_best_checkpoint_state()
        checkpoint = self.output_root / "real_checkpoint.pt"
        record = save_task1_checkpoint_create_only(checkpoint, state)
        checkpoint_link = self.output_root / "checkpoint_link.pt"
        checkpoint_link.symlink_to(checkpoint)
        with self.assertRaises(ValueError):
            load_task1_checkpoint(
                checkpoint_link,
                expected_sha256=record["sha256"],
                expected_type="best",
            )

        json_path = self.output_root / "real.json"
        json_record = _write_json_create_only(json_path, {"value": 1})
        json_link = self.output_root / "json_link.json"
        json_link.symlink_to(json_path)
        with self.assertRaises(ValueError):
            _read_json_snapshot(json_link, expected_sha256=json_record["sha256"])

    def test_json_publication_is_atomic_create_only_and_hash_bound(self) -> None:
        destination = self.output_root / "atomic.json"
        value = {"finite": 1.25, "locked": True}
        record = _write_json_create_only(destination, value)
        observed, observed_record = _read_json_snapshot(
            destination,
            expected_sha256=record["sha256"],
        )
        self.assertEqual(observed, value)
        self.assertEqual(observed_record, record)
        original_sha256 = sha256_file(destination)
        with self.assertRaises(FileExistsError):
            _write_json_create_only(destination, {"changed": True})
        self.assertEqual(sha256_file(destination), original_sha256)

        failed_destination = self.output_root / "link_failure.json"
        with (
            patch("bird_audio.task1_training.os.link", side_effect=OSError("injected link error")),
            self.assertRaisesRegex(OSError, "injected link error"),
        ):
            _write_json_create_only(failed_destination, value)
        self.assertFalse(failed_destination.exists())
        self.assertFalse(
            any(path.name.startswith(".link_failure.json") for path in self.output_root.iterdir())
        )
        with self.assertRaises(ValueError):
            _write_json_create_only(self.output_root / "nan.json", {"bad": float("nan")})

    def test_explicit_weight_population_seals_and_reuses_full_canonical_lock(self) -> None:
        weight_path = self.data_root / "efficientnet_b0_rwightman-7f5810bc.pth"
        torch.save({"features.0.0.weight": torch.ones((1,), dtype=torch.float32)}, weight_path)
        weight_sha256 = sha256_file(weight_path)
        lock_path = self.data_root / "weight.lock.json"
        with (
            patch("bird_audio.task1_training._weight_cache_path", return_value=weight_path),
            patch("bird_audio.task1_training.WEIGHT_LOCK_PATH", lock_path),
            patch("bird_audio.task1_training.WEIGHT_HASH_PREFIX", weight_sha256[:8]),
            patch(
                "bird_audio.task1_training.EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict",
                return_value={"verified": torch.ones(())},
            ) as official_check,
        ):
            with self.assertRaises(FileNotFoundError):
                task1_training.preflight_efficientnet_weights(populate=False)
            artifact = task1_training.preflight_efficientnet_weights(populate=True)
            repeated = task1_training.preflight_efficientnet_weights(populate=False)

        official_check.assert_called_once_with(progress=True, check_hash=True)
        self.assertEqual(artifact, repeated)
        lock, _ = _read_json_snapshot(lock_path)
        self.assertEqual(lock["sha256"], weight_sha256)
        self.assertEqual(lock["size_bytes"], weight_path.stat().st_size)

    def test_production_model_strictly_loads_descriptor_verified_backbone_only(self) -> None:
        config = load_final_task1_config()
        official = efficientnet_b0(weights=None)
        official_state = official.state_dict()
        path = self.data_root / "official_state.pt"
        torch.save(official_state, path)
        artifact = WeightArtifact(path, sha256_file(path), path.stat().st_size)

        task1_training.seed_task1(13, torch.device("cpu"))
        model = _build_model(config, torch.device("cpu"), artifact, None)

        self.assertEqual(model.network.classifier[1].out_features, 15)
        torch.testing.assert_close(
            model.network.features[0][0].weight,
            official_state["features.0.0.weight"],
            rtol=0,
            atol=0,
        )
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
            _build_model(config, torch.device("cpu"), artifact, None)

    def test_production_cache_gate_requires_canonical_root_and_published_sha(self) -> None:
        with (
            ExitStack() as stack,
            self.assertRaisesRegex(PermissionError, "canonical known cache root"),
        ):
            _open_real_data(
                stack,
                cache_root=self.data_root,
                ffmpeg=None,
                expected_lock_sha256=KNOWN_CACHE_LOCK_SHA256,
            )
        with (
            ExitStack() as stack,
            self.assertRaisesRegex(ValueError, "published lock"),
        ):
            _open_real_data(
                stack,
                cache_root=task1_training.DEFAULT_CACHE_ROOT,
                ffmpeg=None,
                expected_lock_sha256="0" * 64,
            )

    def test_gated_evaluator_loads_only_a_sha_bound_production_mps_checkpoint(self) -> None:
        helper_parameters = inspect.signature(
            task1_training.load_locked_task1_best_model
        ).parameters
        self.assertNotIn("train_data", helper_parameters)
        self.assertNotIn("validation_data", helper_parameters)
        self.assertNotIn("test_injection", helper_parameters)
        with self.assertRaisesRegex(ValueError, "supplied MPS device"):
            task1_training.load_locked_task1_best_model(
                self.data_root / "absent.pt",
                checkpoint_sha256="0" * 64,
                expected_run_identity_sha256="1" * 64,
                device=torch.device("cpu"),
            )

        runtime = {"schema_version": "test", "device": "mps"}
        execution_identity = task1_training.Task1ExecutionIdentity(
            implementation_sha256="1" * 64,
            requirements_lock_sha256="2" * 64,
            numerical_runtime=runtime,
            numerical_runtime_sha256=task1_training.sha256_json(runtime),
        )
        run_identity_sha256 = "3" * 64
        checkpoint_state = self._valid_best_checkpoint_state()
        checkpoint_state.update(
            {
                "run_id": "gated_eval",
                "run_identity_sha256": run_identity_sha256,
                "config_sha256": task1_training.config_fingerprint(load_final_task1_config()),
                "cache_lock_sha256": KNOWN_CACHE_LOCK_SHA256,
                "weight_sha256": self.weight.sha256,
                "implementation_sha256": execution_identity.implementation_sha256,
                "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
                "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
                "scope": "production",
                "production_evidence": True,
            }
        )
        checkpoint_path = self.data_root / "gated_eval.pt"
        checkpoint_record = save_task1_checkpoint_create_only(
            checkpoint_path,
            checkpoint_state,
        )
        expected_model = _TinyClassifier()
        with (
            patch(
                "bird_audio.task1_training._resolve_runtime",
                return_value=torch.device("mps"),
            ),
            patch(
                "bird_audio.task1_training._capture_execution_identity",
                return_value=execution_identity,
            ),
            patch(
                "bird_audio.task1_training.preflight_efficientnet_weights",
                return_value=self.weight,
            ) as preflight,
            patch(
                "bird_audio.task1_training._build_model",
                return_value=expected_model,
            ) as build_model,
        ):
            model, metadata = task1_training.load_locked_task1_best_model(
                checkpoint_path,
                checkpoint_sha256=checkpoint_record["sha256"],
                expected_run_identity_sha256=run_identity_sha256,
                device=torch.device("mps"),
            )
        preflight.assert_called_once_with(populate=False)
        build_model.assert_called_once()
        self.assertIs(model, expected_model)
        self.assertFalse(model.training)
        self.assertEqual(metadata["checkpoint_sha256"], checkpoint_record["sha256"])
        self.assertEqual(metadata["cache_lock_sha256"], KNOWN_CACHE_LOCK_SHA256)
        self.assertEqual(metadata["scope"], "production")
        self.assertTrue(metadata["production_evidence"])

    def test_model_factory_runs_after_all_locked_seeds_are_set(self) -> None:
        observed = {}

        def factory(config):
            observed["python"] = random.random()
            observed["numpy"] = float(np.random.random())
            observed["torch"] = float(torch.rand(()))
            return _TinyClassifier(int(config["class_count"]))

        expected_python = random.Random(13).random()
        expected_numpy = float(np.random.RandomState(13).random_sample())
        generator = torch.Generator(device="cpu").manual_seed(13)
        expected_torch = float(torch.rand((), generator=generator))

        run_task1_development(
            seed=13,
            output_root=self.output_root,
            run_id="seed_order",
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(factory),
        )

        self.assertEqual(observed["python"], expected_python)
        self.assertEqual(observed["numpy"], expected_numpy)
        self.assertEqual(observed["torch"], expected_torch)

    def test_execution_identity_is_exact_and_resume_rejects_each_drift_class(self) -> None:
        identity = task1_training._capture_execution_identity(torch.device("cpu"))
        self.assertRegex(identity.implementation_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            identity.requirements_lock_sha256,
            sha256_file(PROJECT_ROOT / "requirements.lock"),
        )
        self.assertEqual(
            identity.numerical_runtime_sha256,
            task1_training.sha256_json(identity.numerical_runtime),
        )
        recovery_path, recovery_sha256 = self._failed_recovery("identity_drift")
        resume_arguments = {
            "seed": 13,
            "output_root": self.output_root,
            "train_data": self.train,
            "validation_data": self.validation,
            "test_injection": self._injection(maximum_epochs=2, patience=2),
            "resume_checkpoint": recovery_path,
            "resume_checkpoint_sha256": recovery_sha256,
        }

        with (
            patch(
                "bird_audio.task1_training._task1_implementation_fingerprint",
                return_value="1" * 64,
            ),
            self.assertRaisesRegex(ValueError, "implementation_sha256"),
        ):
            run_task1_development(**resume_arguments)
        with (
            patch(
                "bird_audio.task1_training._requirements_lock_fingerprint",
                return_value="2" * 64,
            ),
            self.assertRaisesRegex(ValueError, "requirements_lock_sha256"),
        ):
            run_task1_development(**resume_arguments)
        changed_runtime = dict(identity.numerical_runtime)
        changed_runtime["torch_version"] = "drifted"
        with (
            patch(
                "bird_audio.task1_training._numerical_runtime_identity",
                return_value=changed_runtime,
            ),
            self.assertRaisesRegex(ValueError, "numerical_runtime_sha256"),
        ):
            run_task1_development(**resume_arguments)

    def test_resume_rejects_every_mutated_adamw_option_and_partition(self) -> None:
        config = load_final_task1_config()
        model = _TinyClassifier()
        optimizer = build_task1_optimizer(model, config)
        train_task1_epoch(
            model,
            optimizer,
            self.train,
            seed=13,
            epoch_index=0,
            batch_size=2,
            class_count=15,
            device=torch.device("cpu"),
        )
        saved = copy.deepcopy(optimizer.state_dict())
        mutations = {
            "betas": (0.1, 0.2),
            "eps": 0.5,
            "weight_decay": 0.5,
            "amsgrad": True,
            "maximize": True,
            "foreach": True,
            "capturable": True,
            "differentiable": True,
            "fused": True,
            "decoupled_weight_decay": False,
            "group_name": "changed",
            "lr": 0.5,
        }
        for option, value in mutations.items():
            resumed_model = _TinyClassifier()
            resumed_optimizer = build_task1_optimizer(resumed_model, config)
            resumed_optimizer.load_state_dict(copy.deepcopy(saved))
            resumed_optimizer.param_groups[0][option] = value
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(
                    ValueError,
                    "settings differ",
                ),
            ):
                task1_training._validate_optimizer_after_resume(
                    resumed_optimizer,
                    resumed_model,
                    config,
                )

        resumed_model = _TinyClassifier()
        resumed_optimizer = build_task1_optimizer(resumed_model, config)
        resumed_optimizer.load_state_dict(saved)
        resumed_optimizer.param_groups[0]["params"].reverse()
        with self.assertRaisesRegex(ValueError, "settings differ"):
            task1_training._validate_optimizer_after_resume(
                resumed_optimizer,
                resumed_model,
                config,
            )

        resumed_model = _TinyClassifier()
        resumed_optimizer = build_task1_optimizer(resumed_model, config)
        resumed_optimizer.load_state_dict(saved)
        resumed_optimizer.state.pop(next(iter(resumed_optimizer.state)))
        with self.assertRaisesRegex(ValueError, "trainable partition"):
            task1_training._validate_optimizer_after_resume(
                resumed_optimizer,
                resumed_model,
                config,
            )

    def test_classifier_adapter_receives_cpu_batches_before_device_transfer(self) -> None:
        observed_devices = []
        real_adapter = task1_training.to_efficientnet_batch

        def checked_adapter(batch):
            observed_devices.append(batch.device.type)
            return real_adapter(batch)

        model = _TinyClassifier()
        optimizer = build_task1_optimizer(model, load_final_task1_config())
        with patch("bird_audio.task1_training.to_efficientnet_batch", side_effect=checked_adapter):
            train_task1_epoch(
                model,
                optimizer,
                self.train,
                seed=13,
                epoch_index=0,
                batch_size=2,
                class_count=15,
                device=torch.device("cpu"),
            )
            validate_task1(
                model,
                self.validation,
                batch_size=2,
                class_count=15,
                device=torch.device("cpu"),
            )
        self.assertEqual(observed_devices, ["cpu", "cpu", "cpu", "cpu"])

    def test_validation_performs_one_bulk_logit_transfer(self) -> None:
        model = _TinyClassifier()
        with patch(
            "bird_audio.task1_training._validation_logits_to_cpu",
            wraps=task1_training._validation_logits_to_cpu,
        ) as transfer:
            result = validate_task1(
                model,
                self.validation,
                batch_size=1,
                class_count=15,
                device=torch.device("cpu"),
            )
        self.assertEqual(transfer.call_count, 1)
        self.assertGreater(result.clip_loss, 0.0)

    def test_test_split_is_refused_before_a_run_directory_is_created(self) -> None:
        test_data = _TinyDevelopmentData(
            self.data_root,
            "test",
            self.class_order,
            (("test-A", 0, 1),),
        )

        with self.assertRaisesRegex(PermissionError, "final test split"):
            run_task1_development(
                seed=13,
                output_root=self.output_root,
                run_id="must_not_exist",
                train_data=self.train,
                validation_data=test_data,
                test_injection=self._injection(),
            )
        self.assertFalse((self.output_root / "must_not_exist").exists())

    def test_isolated_injection_cannot_publish_in_production_run_root(self) -> None:
        with self.assertRaisesRegex(PermissionError, "production run root"):
            run_task1_development(
                seed=13,
                output_root=DEFAULT_RUN_ROOT,
                run_id="isolated_must_not_publish",
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )
        self.assertFalse((DEFAULT_RUN_ROOT / "isolated_must_not_publish").exists())

    def test_training_failure_is_recorded_without_a_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid training logits"):
            run_task1_development(
                seed=13,
                output_root=self.output_root,
                run_id="unit_failure",
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(lambda _config: _WrongLogitClassifier()),
            )

        run_directory = self.output_root / "unit_failure"
        failure_paths = list((run_directory / "failures").iterdir())
        self.assertEqual(len(failure_paths), 1)
        failure = json.loads(failure_paths[0].read_text(encoding="utf-8"))
        self.assertFalse(failure["complete"])
        self.assertEqual(failure["error_type"], "RuntimeError")
        self.assertEqual(list((run_directory / "best_candidates").iterdir()), [])
        self.assertEqual(list((run_directory / "recovery").iterdir()), [])

    def test_verified_resume_restores_epoch_optimizer_early_stop_and_rng_state(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "injected second epoch failure"):
            run_task1_development(
                seed=13,
                output_root=self.output_root,
                run_id="resume_source",
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(
                    lambda _config: _FailOnSecondEpochClassifier(),
                    maximum_epochs=2,
                    patience=2,
                ),
            )

        source_directory = self.output_root / "resume_source"
        recovery_path = source_directory / "recovery" / "recovery_epoch_0001.pt"
        recovery_sha256 = sha256_file(recovery_path)
        self.assertEqual(len(list((source_directory / "failures").iterdir())), 1)

        with self.assertRaisesRegex(ValueError, "locked run identity field"):
            run_task1_development(
                seed=37,
                output_root=self.output_root,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(maximum_epochs=2, patience=2),
                resume_checkpoint=recovery_path,
                resume_checkpoint_sha256=recovery_sha256,
            )

        resumed = run_task1_development(
            seed=13,
            output_root=self.output_root,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(maximum_epochs=2, patience=2),
            resume_checkpoint=recovery_path,
            resume_checkpoint_sha256=recovery_sha256,
        )
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["epochs_completed"], 2)
        self.assertTrue((source_directory / "recovery" / "recovery_epoch_0002.pt").is_file())
        self.assertEqual(len(list((source_directory / "failures").iterdir())), 1)

        uninterrupted = run_task1_development(
            seed=13,
            output_root=self.output_root,
            run_id="resume_reference",
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(maximum_epochs=2, patience=2),
        )
        resumed_recovery = load_task1_checkpoint(
            resumed["latest_recovery_checkpoint"]["path"],
            expected_sha256=resumed["latest_recovery_checkpoint"]["sha256"],
            expected_type="recovery",
        )
        reference_recovery = load_task1_checkpoint(
            uninterrupted["latest_recovery_checkpoint"]["path"],
            expected_sha256=uninterrupted["latest_recovery_checkpoint"]["sha256"],
            expected_type="recovery",
        )
        self.assertEqual(resumed_recovery["completed_epoch"], 2)
        self.assertEqual(reference_recovery["completed_epoch"], 2)
        for key in resumed_recovery["model"]:
            torch.testing.assert_close(
                resumed_recovery["model"][key],
                reference_recovery["model"][key],
                rtol=0,
                atol=0,
            )

        repeated = run_task1_development(
            seed=13,
            output_root=self.output_root,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(maximum_epochs=2, patience=2),
            resume_checkpoint=resumed["latest_recovery_checkpoint"]["path"],
            resume_checkpoint_sha256=resumed["latest_recovery_checkpoint"]["sha256"],
        )
        self.assertEqual(
            repeated["result_artifact"]["sha256"], resumed["result_artifact"]["sha256"]
        )

    def test_completed_result_repairs_only_a_valid_missing_lock(self) -> None:
        result = run_task1_development(
            seed=13,
            output_root=self.output_root,
            run_id="result_lock_repair",
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(),
        )
        run_directory = self.output_root / "result_lock_repair"
        lock_path = run_directory / "result.lock.json"
        lock_path.unlink()

        def refuse_model(_config):
            raise AssertionError("completed result attempted to rebuild the model")

        repaired = run_task1_development(
            seed=13,
            output_root=self.output_root,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(refuse_model),
            resume_checkpoint=result["latest_recovery_checkpoint"]["path"],
            resume_checkpoint_sha256=result["latest_recovery_checkpoint"]["sha256"],
        )
        self.assertTrue(lock_path.is_file())
        self.assertEqual(repaired["result_artifact"]["sha256"], result["result_artifact"]["sha256"])

        lock_path.unlink()
        result_path = run_directory / "result.json"
        altered = json.loads(result_path.read_text(encoding="utf-8"))
        altered["unexpected"] = True
        result_path.unlink()
        result_path.write_bytes(task1_training._json_bytes(altered))
        with self.assertRaisesRegex(ValueError, "result schema"):
            run_task1_development(
                seed=13,
                output_root=self.output_root,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(refuse_model),
                resume_checkpoint=result["latest_recovery_checkpoint"]["path"],
                resume_checkpoint_sha256=result["latest_recovery_checkpoint"]["sha256"],
            )
        self.assertFalse(lock_path.exists())

    def test_benchmark_measures_full_epoch_without_persistent_selection(self) -> None:
        before = {path.name for path in self.output_root.iterdir()}

        result = benchmark_task1_full_epoch(
            seed=13,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(),
        )

        self.assertTrue(result["benchmark_only"])
        self.assertFalse(result["persistent_model_selection"])
        self.assertFalse(result["persistent_model_checkpoint"])
        self.assertFalse(result["durable_evidence"])
        self.assertEqual(result["scope"], "isolated_test")
        self.assertFalse(result["production_evidence"])
        self.assertTrue(result["warmup_completed"])
        self.assertGreater(result["full_epoch_seconds"], 0.0)
        self.assertGreater(result["train_clips_per_second"], 0.0)
        self.assertGreater(result["validation_clips_per_second"], 0.0)
        self.assertNotIn("projected_one_seed_maximum_seconds", result)
        self.assertNotIn("projected_all_seed_maximum_seconds", result)
        self.assertEqual(result["conservative_wall_time_factor"], CONSERVATIVE_WALL_TIME_FACTOR)
        self.assertGreaterEqual(result["checkpoint_cpu_copy_seconds"], 0.0)
        self.assertGreaterEqual(result["checkpoint_serialization_seconds"], 0.0)
        self.assertGreater(result["representative_checkpoint_bytes"], 0)
        self.assertGreater(
            result["estimated_epoch_with_checkpoint_seconds"],
            result["full_epoch_seconds"],
        )
        self.assertGreater(
            result["estimated_one_seed_conservative_wall_seconds"],
            result["estimated_one_seed_pre_allowance_seconds"],
        )
        self.assertGreater(
            result["estimated_all_seed_conservative_wall_seconds"],
            result["estimated_all_seed_pre_allowance_seconds"],
        )
        self.assertEqual({path.name for path in self.output_root.iterdir()}, before)

    def test_benchmark_publishes_create_only_identity_bound_json_without_model(self) -> None:
        result_path = self.output_root / "isolated_benchmark.json"
        result = benchmark_task1_full_epoch(
            seed=13,
            command=("python", "benchmark-unit"),
            evidence_output=result_path,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(),
        )

        lock_path = self.output_root / "isolated_benchmark.lock.json"
        self.assertTrue(result_path.is_file())
        self.assertTrue(lock_path.is_file())
        self.assertTrue(result["durable_evidence"])
        self.assertEqual(result["scope"], "isolated_test")
        self.assertFalse(result["production_evidence"])
        self.assertEqual(result["command"], ["python", "benchmark-unit"])
        for field in (
            "benchmark_identity_sha256",
            "config_file_sha256",
            "config_sha256",
            "source_fingerprint_sha256",
            "implementation_sha256",
            "requirements_lock_sha256",
            "numerical_runtime_sha256",
            "cache_lock_sha256",
            "weight_sha256",
        ):
            self.assertRegex(str(result[field]), r"^[0-9a-f]{64}$")
        self.assertIn("parameter_counts", result)
        self.assertIn("numerical_runtime", result)
        self.assertEqual(
            task1_training.sha256_json(result["benchmark_identity"]),
            result["benchmark_identity_sha256"],
        )
        self.assertEqual(list(self.output_root.rglob("*.pt")), [])
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["result"]["sha256"], sha256_file(result_path))
        self.assertEqual(lock["benchmark_identity_sha256"], result["benchmark_identity_sha256"])
        original_result_sha256 = sha256_file(result_path)
        original_lock_sha256 = sha256_file(lock_path)

        with patch(
            "bird_audio.task1_training._warmup_task1",
            side_effect=AssertionError("existing benchmark was rerun"),
        ):
            repeated = benchmark_task1_full_epoch(
                seed=13,
                evidence_output=result_path,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )
        self.assertEqual(repeated["result_artifact"]["sha256"], original_result_sha256)
        self.assertEqual(repeated["completion_lock_artifact"]["sha256"], original_lock_sha256)
        self.assertEqual(sha256_file(result_path), original_result_sha256)
        self.assertEqual(sha256_file(lock_path), original_lock_sha256)

    def test_benchmark_repairs_result_only_crash_without_rerunning(self) -> None:
        result_path = self.output_root / "interrupted_benchmark.json"
        lock_path = self.output_root / "interrupted_benchmark.lock.json"
        real_write = task1_training._write_json_create_only

        def interrupt_lock(path, value):
            if Path(path) == lock_path:
                raise OSError("injected lock publication failure")
            return real_write(Path(path), value)

        with (
            patch(
                "bird_audio.task1_training._write_json_create_only",
                side_effect=interrupt_lock,
            ),
            self.assertRaisesRegex(OSError, "lock publication failure"),
        ):
            benchmark_task1_full_epoch(
                seed=13,
                evidence_output=result_path,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )
        self.assertTrue(result_path.is_file())
        self.assertFalse(lock_path.exists())

        with patch(
            "bird_audio.task1_training._warmup_task1",
            side_effect=AssertionError("valid interrupted benchmark was rerun"),
        ):
            repaired = benchmark_task1_full_epoch(
                seed=13,
                evidence_output=result_path,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )
        self.assertTrue(lock_path.is_file())
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["result"]["sha256"], sha256_file(result_path))
        self.assertEqual(repaired["completion_lock_artifact"]["sha256"], sha256_file(lock_path))

    def test_benchmark_rejects_lock_only_tampering_and_symbolic_links(self) -> None:
        lock_only_result = self.output_root / "lock_only.json"
        lock_only = self.output_root / "lock_only.lock.json"
        _write_json_create_only(lock_only, {"invalid": True})
        with self.assertRaisesRegex(ValueError, "lock exists without its result"):
            benchmark_task1_full_epoch(
                seed=13,
                evidence_output=lock_only_result,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )

        linked_result = self.output_root / "linked_result.json"
        linked_target = self.output_root / "linked_result_target.json"
        _write_json_create_only(linked_target, {"invalid": True})
        linked_result.symlink_to(linked_target)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            benchmark_task1_full_epoch(
                seed=13,
                evidence_output=linked_result,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )

        result_path = self.output_root / "altered_benchmark.json"
        result = benchmark_task1_full_epoch(
            seed=13,
            evidence_output=result_path,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(),
        )
        lock_path = self.output_root / "altered_benchmark.lock.json"
        lock_path.unlink()
        altered = json.loads(result_path.read_text(encoding="utf-8"))
        altered["unexpected"] = result["benchmark_identity_sha256"]
        result_path.unlink()
        result_path.write_bytes(task1_training._json_bytes(altered))
        with self.assertRaisesRegex(ValueError, "result schema"):
            benchmark_task1_full_epoch(
                seed=13,
                evidence_output=result_path,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )
        self.assertFalse(lock_path.exists())

        linked_lock_result = self.output_root / "linked_lock_result.json"
        benchmark_task1_full_epoch(
            seed=13,
            evidence_output=linked_lock_result,
            train_data=self.train,
            validation_data=self.validation,
            test_injection=self._injection(),
        )
        linked_lock = self.output_root / "linked_lock_result.lock.json"
        linked_lock_payload = linked_lock.read_bytes()
        linked_lock.unlink()
        linked_lock_target = self.output_root / "linked_lock_target.json"
        linked_lock_target.write_bytes(linked_lock_payload)
        linked_lock.symlink_to(linked_lock_target)
        with self.assertRaisesRegex(ValueError, "regular file"):
            benchmark_task1_full_epoch(
                seed=13,
                evidence_output=linked_lock_result,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )

        linked_lock.unlink()
        _write_json_create_only(linked_lock, {"invalid": True})
        with self.assertRaisesRegex(ValueError, "completion lock is invalid"):
            benchmark_task1_full_epoch(
                seed=13,
                evidence_output=linked_lock_result,
                train_data=self.train,
                validation_data=self.validation,
                test_injection=self._injection(),
            )


if __name__ == "__main__":
    unittest.main()
