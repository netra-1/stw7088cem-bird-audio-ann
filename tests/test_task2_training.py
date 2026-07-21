from __future__ import annotations

import copy
import hashlib
import inspect
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import bird_audio.task2_training as task2_training
from bird_audio.models import ConvolutionalAutoencoder, parameter_counts
from bird_audio.task2_training import (
    CONSERVATIVE_WALL_TIME_FACTOR,
    DEFAULT_BENCHMARK_RESULT_PATH,
    DEFAULT_CACHE_ROOT,
    DEFAULT_RUN_ROOT,
    EXPECTED_PARAMETER_COUNT,
    ISOLATED_TEST_SCOPE,
    KNOWN_CACHE_LOCK_SHA256,
    PRODUCTION_SCOPE,
    Task2CheckpointScore,
    Task2EarlyStopping,
    Task2TestInjection,
    _assert_locked_config,
    _open_real_data,
    _read_json_snapshot,
    _resolve_runtime,
    _write_json_create_only,
    benchmark_task2_full_epoch,
    build_task2_optimizer,
    is_better_task2_checkpoint,
    load_locked_task2_best_model_for_evaluation,
    load_locked_task2_config,
    load_task2_checkpoint,
    run_task2_development,
    save_task2_checkpoint_create_only,
    train_task2_epoch,
    validate_task2,
    verify_task2_development_run,
)
from bird_audio.training_batching import (
    RecordingBalancedEpochSampler,
    collate_native_samples,
    to_autoencoder_batch,
)


def setUpModule() -> None:
    DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)


class _TinyDevelopmentData:
    def __init__(
        self,
        root: Path,
        split: str,
        definitions: tuple[tuple[str, int], ...],
        *,
        lock_sha256: str = "d" * 64,
    ) -> None:
        self.root = root
        self.split = split
        self.strategy = "energy"
        self.lock_sha256 = lock_sha256
        self._features: list[np.ndarray] = []
        self._rows: list[dict[str, str]] = []
        for recording_number, (recording_id, clip_count) in enumerate(definitions):
            for rank in range(clip_count):
                value = 0.05 + 0.04 * recording_number + 0.01 * rank
                self._features.append(np.full((1, 128, 372), value, dtype=np.float32))
                self._rows.append(
                    {
                        "clip_id": f"{recording_id}:clip:{rank:02d}",
                        "recording_id": recording_id,
                        "species_common_name": "Asian Koel",
                        "class_index": "0",
                        "session_group": f"{split}:session:{recording_number:03d}",
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


class _TinyAutoencoder(nn.Module):
    def __init__(self, *, fail_after: int | None = None) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fail_after = fail_after
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        if self.fail_after is not None and self.forward_calls >= self.fail_after:
            raise RuntimeError("injected second epoch failure")
        hidden = self.dropout(self.encoder(inputs))
        reconstruction = torch.sigmoid(hidden)
        latent = hidden.mean(dim=(2, 3)).repeat(1, 64)
        return reconstruction, latent


class _CaptureAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.inputs: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs.append(inputs.detach().cpu().clone())
        reconstruction = torch.sigmoid(self.logit).expand_as(inputs)
        latent = inputs.mean(dim=(2, 3)).repeat(1, 64)
        return reconstruction, latent


def _fixtures(root: Path) -> tuple[_TinyDevelopmentData, _TinyDevelopmentData]:
    train = _TinyDevelopmentData(
        root,
        "train",
        tuple((f"train-{index:02d}", 1) for index in range(10)),
    )
    validation = _TinyDevelopmentData(
        root,
        "validation",
        tuple((f"validation-{index:02d}", 1) for index in range(3)),
    )
    return train, validation


def _injection(
    *,
    fail_after: int | None = None,
    maximum_epochs: int = 2,
    patience: int = 10,
    factory_counter: list[int] | None = None,
) -> Task2TestInjection:
    def factory(_config):
        if factory_counter is not None:
            factory_counter.append(1)
        return _TinyAutoencoder(fail_after=fail_after)

    return Task2TestInjection(
        model_factory=factory,
        maximum_epochs=maximum_epochs,
        batch_size=4,
        early_stopping_patience=patience,
    )


def _checkpoint_state() -> dict[str, object]:
    config = load_locked_task2_config()
    model = _TinyAutoencoder()
    optimizer = build_task2_optimizer(model, config)
    common = {
        "schema_version": "1.1",
        "checkpoint_type": "best",
        "run_id": "task2-test-run",
        "run_identity_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "config_file_sha256": "3" * 64,
        "cache_lock_sha256": "4" * 64,
        "implementation_sha256": "5" * 64,
        "requirements_lock_sha256": "6" * 64,
        "numerical_runtime_sha256": "7" * 64,
        "model_contract_sha256": "8" * 64,
        "scope": ISOLATED_TEST_SCOPE,
        "production_evidence": False,
        "seed": 13,
    }
    return {
        **common,
        "epoch": 1,
        "score": {"validation_loss": 0.25, "epoch": 1},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _populate_adamw_state(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
) -> None:
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0, dtype=torch.float32),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }


class Task2LockedMethodTests(unittest.TestCase):
    def test_locked_config_model_optimizer_and_selection_contract(self) -> None:
        config = load_locked_task2_config()
        _assert_locked_config(config)
        model = ConvolutionalAutoencoder(latent_dimensions=64)
        self.assertEqual(
            parameter_counts(model),
            {"total": EXPECTED_PARAMETER_COUNT, "trainable": EXPECTED_PARAMETER_COUNT},
        )
        optimizer = build_task2_optimizer(model, config)
        self.assertIs(type(optimizer), torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.001)
        self.assertEqual(group["weight_decay"], 0.00001)
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertIs(group["foreach"], False)
        self.assertIs(group["capturable"], False)
        self.assertIs(group["differentiable"], False)
        self.assertIs(group["fused"], False)
        self.assertIs(group["decoupled_weight_decay"], True)
        self.assertEqual(
            {id(parameter) for parameter in group["params"]},
            {id(parameter) for parameter in model.parameters()},
        )

    def test_config_guard_rejects_changed_locked_fields(self) -> None:
        config = load_locked_task2_config()
        changes = (
            (("latent_dimensions",), 32),
            (("clip_selection_strategy",), "uniform"),
            (("training", "batch_size"), 32),
            (("training", "maximum_epochs"), 101),
            (("training", "allow_mps_fallback"), True),
            (("novelty", "nearest_neighbours"), 9),
            (("novelty", "threshold_quantile_method"), "linear"),
            (("novelty", "bootstrap_seed"), 1),
            (("novelty", "threshold_operator"), ">="),
        )
        for keys, replacement in changes:
            changed = copy.deepcopy(config)
            target = changed
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = replacement
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                _assert_locked_config(changed)

    def test_lower_validation_loss_wins_and_exact_ties_keep_earlier_epoch(self) -> None:
        incumbent = Task2CheckpointScore(0.2, 2)
        self.assertTrue(is_better_task2_checkpoint(Task2CheckpointScore(0.1, 8), incumbent))
        self.assertTrue(is_better_task2_checkpoint(Task2CheckpointScore(0.2, 1), incumbent))
        self.assertFalse(is_better_task2_checkpoint(Task2CheckpointScore(0.2, 3), incumbent))
        self.assertFalse(is_better_task2_checkpoint(Task2CheckpointScore(0.3, 1), incumbent))

        stopping = Task2EarlyStopping(patience=2)
        self.assertEqual(stopping.update(Task2CheckpointScore(0.2, 1)), (True, False))
        self.assertEqual(stopping.update(Task2CheckpointScore(0.2, 2)), (False, False))
        self.assertEqual(stopping.update(Task2CheckpointScore(0.2, 3)), (False, True))
        self.assertEqual(stopping.best, Task2CheckpointScore(0.2, 1))

    def test_cpu_injection_still_rejects_mps_fallback(self) -> None:
        with (
            patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}),
            self.assertRaises(RuntimeError),
        ):
            _resolve_runtime(_injection())


class Task2EpochTests(unittest.TestCase):
    def test_sampler_no_augmentation_and_mean_mse_are_exact(self) -> None:
        root = Path("/tmp/task2-fixture")
        train, validation = _fixtures(root)
        config = load_locked_task2_config()
        model = _CaptureAutoencoder()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        sampler = RecordingBalancedEpochSampler(train, base_seed=13)
        sampler.set_epoch(2)
        expected_indices = list(sampler)
        expected_inputs = []
        for start in range(0, len(expected_indices), 4):
            indices = expected_indices[start : start + 4]
            native = collate_native_samples([train[index] for index in indices])
            expected_inputs.append(to_autoencoder_batch(native.tensor))

        metrics = train_task2_epoch(
            model,
            optimizer,
            train,
            seed=13,
            epoch_index=2,
            batch_size=4,
            latent_dimensions=int(config["latent_dimensions"]),
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics["clips"], len(train))
        self.assertEqual(metrics["batches"], 3)
        self.assertEqual(metrics["sampler_seed"], sampler.generator_seed)
        self.assertEqual(metrics["augmentation"], "none")
        self.assertEqual(len(model.inputs), len(expected_inputs))
        for observed, expected in zip(model.inputs, expected_inputs, strict=True):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        all_inputs = torch.cat(expected_inputs)
        expected_loss = float(
            torch.mean(torch.square(torch.full_like(all_inputs, 0.5) - all_inputs))
        )
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=7)

        validation_model = _CaptureAutoencoder()
        result = validate_task2(
            validation_model,
            validation,
            batch_size=2,
            latent_dimensions=64,
            device=torch.device("cpu"),
        )
        validation_inputs = torch.cat(validation_model.inputs)
        expected_validation = float(
            torch.sum(torch.square(torch.full_like(validation_inputs, 0.5) - validation_inputs))
            / validation_inputs.numel()
        )
        self.assertEqual(result.pixel_count, len(validation) * 224 * 224)
        self.assertAlmostEqual(result.loss, expected_validation, places=7)

    def test_final_split_and_nonenergy_data_are_rejected(self) -> None:
        train, validation = _fixtures(Path("/tmp/task2-fixture"))
        config = load_locked_task2_config()
        model = _CaptureAutoencoder()
        optimizer = build_task2_optimizer(model, config)
        train.split = "test"
        with self.assertRaises(PermissionError):
            train_task2_epoch(
                model,
                optimizer,
                train,
                seed=13,
                epoch_index=0,
                batch_size=4,
                latent_dimensions=64,
                device=torch.device("cpu"),
            )
        validation.strategy = "uniform"
        with self.assertRaises(PermissionError):
            validate_task2(
                model,
                validation,
                batch_size=4,
                latent_dimensions=64,
                device=torch.device("cpu"),
            )


class Task2ArtifactTests(unittest.TestCase):
    def test_checkpoint_is_create_only_hash_bound_typed_and_nofollow(self) -> None:
        state = _checkpoint_state()
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            root = Path(temporary)
            path = root / "best.pt"
            record = save_task2_checkpoint_create_only(path, state)
            loaded = load_task2_checkpoint(
                path,
                expected_sha256=record["sha256"],
                expected_type="best",
            )
            self.assertEqual(loaded["score"], state["score"])
            with self.assertRaises(FileExistsError):
                save_task2_checkpoint_create_only(path, state)
            with self.assertRaises(ValueError):
                load_task2_checkpoint(path, expected_sha256="0" * 64)

            link = root / "best-link.pt"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                load_task2_checkpoint(link, expected_sha256=record["sha256"])

            invalid = dict(state)
            invalid["seed"] = True
            buffer = io.BytesIO()
            torch.save(invalid, buffer)
            invalid_path = root / "invalid.pt"
            invalid_path.write_bytes(buffer.getvalue())
            invalid_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
            with self.assertRaises(ValueError):
                load_task2_checkpoint(invalid_path, expected_sha256=invalid_sha)

            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                load_task2_checkpoint(path, expected_sha256=record["sha256"])

    def test_checkpoint_rejects_changed_adamw_execution_option(self) -> None:
        state = _checkpoint_state()
        state["optimizer"]["param_groups"][0]["foreach"] = True
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            path = Path(temporary) / "changed-optimizer.pt"
            buffer = io.BytesIO()
            torch.save(state, buffer)
            path.write_bytes(buffer.getvalue())
            sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
            with self.assertRaisesRegex(ValueError, "optimizer group"):
                load_task2_checkpoint(path, expected_sha256=sha256)

    def test_json_publication_is_atomic_canonical_create_only_and_nofollow(self) -> None:
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            root = Path(temporary)
            path = root / "artifact.json"
            value = {"complete": True, "value": 1.25}
            record = _write_json_create_only(path, value)
            observed, observed_record = _read_json_snapshot(
                path,
                expected_sha256=record["sha256"],
            )
            self.assertEqual(observed, value)
            self.assertEqual(observed_record["sha256"], record["sha256"])
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            with self.assertRaises(FileExistsError):
                _write_json_create_only(path, value)
            link = root / "artifact-link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                _read_json_snapshot(link)


class Task2RunTests(unittest.TestCase):
    def test_run_publishes_recomputable_training_reference_and_both_thresholds(self) -> None:
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            output_root = Path(temporary)
            train, validation = _fixtures(output_root)
            result = run_task2_development(
                seed=13,
                output_root=output_root,
                run_id="task2-complete",
                train_data=train,
                validation_data=validation,
                test_injection=_injection(maximum_epochs=2),
            )

            self.assertTrue(result["complete"])
            self.assertEqual(result["cache_lock_sha256"], "d" * 64)
            self.assertEqual(result["epochs_completed"], 2)
            self.assertTrue(Path(result["best_checkpoint"]["path"]).is_file())
            bundle, bundle_record = _read_json_snapshot(
                result["development_bundle"]["path"],
                expected_sha256=result["development_bundle"]["sha256"],
            )
            self.assertTrue(bundle["complete"])
            self.assertEqual(bundle["threshold_operator"], ">")
            self.assertEqual(
                bundle["best_checkpoint"]["sha256"], result["best_checkpoint"]["sha256"]
            )
            self.assertEqual(bundle_record["sha256"], result["development_bundle"]["sha256"])
            for artifact in bundle["artifacts"].values():
                self.assertTrue(Path(artifact["path"]).is_file())
                payload, observed = _read_json_snapshot(
                    artifact["path"],
                    expected_sha256=artifact["sha256"],
                )
                self.assertEqual(observed["size_bytes"], artifact["size_bytes"])
                self.assertEqual(
                    payload["best_checkpoint_sha256"], result["best_checkpoint"]["sha256"]
                )

            training_scores, _ = _read_json_snapshot(
                bundle["artifacts"]["known_training_scores"]["path"]
            )
            validation_scores, _ = _read_json_snapshot(
                bundle["artifacts"]["known_validation_scores"]["path"]
            )
            reference, _ = _read_json_snapshot(
                bundle["artifacts"]["training_latent_reference"]["path"]
            )
            thresholds, _ = _read_json_snapshot(bundle["artifacts"]["thresholds"]["path"])
            training_ids = sorted(row["recording_id"] for row in training_scores["recordings"])
            validation_ids = sorted(row["recording_id"] for row in validation_scores["recordings"])
            self.assertEqual(reference["reference"]["fit_role"], "known_training")
            self.assertEqual(reference["reference"]["recording_ids"], training_ids)
            self.assertFalse(set(training_ids).intersection(validation_ids))
            reconstruction_values = np.asarray(
                [row["reconstruction_mse"] for row in validation_scores["recordings"]],
                dtype=np.float64,
            )
            latent_values = np.asarray(
                [row["score"] for row in validation_scores["latent_novelty_scores"]],
                dtype=np.float64,
            )
            self.assertEqual(
                thresholds["reconstruction"]["value"],
                float(np.quantile(reconstruction_values, 0.95, method="higher")),
            )
            self.assertEqual(
                thresholds["latent"]["value"],
                float(np.quantile(latent_values, 0.95, method="higher")),
            )
            for threshold in (thresholds["reconstruction"], thresholds["latent"]):
                self.assertEqual(threshold["calibration_role"], "known_validation")
                self.assertEqual(threshold["calibration_recording_ids"], validation_ids)
                self.assertEqual(threshold["quantile"], 0.95)
                self.assertEqual(threshold["method"], "higher")
                self.assertEqual(threshold["classification_operator"], ">")

    def test_recursive_verifier_enforces_scope_and_detects_child_tamper(self) -> None:
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            output_root = Path(temporary)
            train, validation = _fixtures(output_root)
            result = run_task2_development(
                seed=13,
                output_root=output_root,
                run_id="task2-recursive-verifier",
                train_data=train,
                validation_data=validation,
                test_injection=_injection(maximum_epochs=1),
            )
            verified = verify_task2_development_run(
                result["completion_lock_artifact"]["path"],
                expected_sha256=result["completion_lock_artifact"]["sha256"],
                require_production=False,
            )
            self.assertTrue(verified["valid"])
            self.assertTrue(verified["thresholds_rederived"])
            self.assertEqual(verified["scope"], ISOLATED_TEST_SCOPE)
            with self.assertRaisesRegex(ValueError, "scope"):
                verify_task2_development_run(
                    result["completion_lock_artifact"]["path"],
                    expected_sha256=result["completion_lock_artifact"]["sha256"],
                    require_production=True,
                )

            threshold_path = Path(result["artifacts"]["development"]["thresholds"]["path"])
            thresholds, _ = _read_json_snapshot(threshold_path)
            thresholds["reconstruction"]["value"] += 0.01
            threshold_path.write_bytes(task2_training._json_bytes(thresholds))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_task2_development_run(
                    result["completion_lock_artifact"]["path"],
                    expected_sha256=result["completion_lock_artifact"]["sha256"],
                    require_production=False,
                )

    def test_isolated_run_cannot_publish_in_production_root(self) -> None:
        train, validation = _fixtures(DEFAULT_RUN_ROOT)
        with self.assertRaisesRegex(PermissionError, "Isolated"):
            run_task2_development(
                seed=13,
                output_root=DEFAULT_RUN_ROOT,
                run_id="task2-forbidden-isolated",
                train_data=train,
                validation_data=validation,
                test_injection=_injection(maximum_epochs=1),
            )

    def test_interrupted_resume_matches_uninterrupted_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as interrupted_temporary:
            interrupted_root = Path(interrupted_temporary)
            train, validation = _fixtures(interrupted_root)
            with self.assertRaisesRegex(RuntimeError, "injected second epoch failure"):
                run_task2_development(
                    seed=13,
                    output_root=interrupted_root,
                    run_id="task2-interrupted",
                    train_data=train,
                    validation_data=validation,
                    test_injection=_injection(fail_after=5, maximum_epochs=3),
                )
            run_directory = interrupted_root / "task2-interrupted"
            recovery_path = run_directory / "recovery" / "recovery_epoch_0001.pt"
            recovery_sha = hashlib.sha256(recovery_path.read_bytes()).hexdigest()
            self.assertTrue(any((run_directory / "failures").glob("failure_*.json")))

            with (
                patch.object(
                    task2_training,
                    "_task2_implementation_fingerprint",
                    return_value="e" * 64,
                ),
                self.assertRaisesRegex(ValueError, "implementation_sha256"),
            ):
                run_task2_development(
                    seed=13,
                    output_root=interrupted_root,
                    train_data=train,
                    validation_data=validation,
                    test_injection=_injection(maximum_epochs=3),
                    resume_checkpoint=recovery_path,
                    resume_checkpoint_sha256=recovery_sha,
                )

            original_runtime = task2_training._numerical_runtime_identity

            def changed_runtime(device):
                value = original_runtime(device)
                value["python_version"] = "drifted-runtime"
                return value

            with (
                patch.object(
                    task2_training,
                    "_numerical_runtime_identity",
                    side_effect=changed_runtime,
                ),
                self.assertRaisesRegex(ValueError, "numerical_runtime_sha256"),
            ):
                run_task2_development(
                    seed=13,
                    output_root=interrupted_root,
                    train_data=train,
                    validation_data=validation,
                    test_injection=_injection(maximum_epochs=3),
                    resume_checkpoint=recovery_path,
                    resume_checkpoint_sha256=recovery_sha,
                )

            resumed = run_task2_development(
                seed=13,
                output_root=interrupted_root,
                train_data=train,
                validation_data=validation,
                test_injection=_injection(maximum_epochs=3),
                resume_checkpoint=recovery_path,
                resume_checkpoint_sha256=recovery_sha,
            )
            resumed_checkpoint = load_task2_checkpoint(
                resumed["best_checkpoint"]["path"],
                expected_sha256=resumed["best_checkpoint"]["sha256"],
                expected_type="best",
            )

            with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as uninterrupted_temporary:
                uninterrupted_root = Path(uninterrupted_temporary)
                train_full, validation_full = _fixtures(uninterrupted_root)
                uninterrupted = run_task2_development(
                    seed=13,
                    output_root=uninterrupted_root,
                    run_id="task2-uninterrupted",
                    train_data=train_full,
                    validation_data=validation_full,
                    test_injection=_injection(maximum_epochs=3),
                )
                uninterrupted_checkpoint = load_task2_checkpoint(
                    uninterrupted["best_checkpoint"]["path"],
                    expected_sha256=uninterrupted["best_checkpoint"]["sha256"],
                    expected_type="best",
                )
                self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
                self.assertEqual(
                    resumed["best_validation_loss"],
                    uninterrupted["best_validation_loss"],
                )
                resumed_history, _ = _read_json_snapshot(
                    resumed["artifacts"]["epoch_history"]["path"]
                )
                uninterrupted_history, _ = _read_json_snapshot(
                    uninterrupted["artifacts"]["epoch_history"]["path"]
                )
                self.assertEqual(resumed_history, uninterrupted_history)
                for key in resumed_checkpoint["model"]:
                    torch.testing.assert_close(
                        resumed_checkpoint["model"][key],
                        uninterrupted_checkpoint["model"][key],
                        rtol=0,
                        atol=0,
                    )
                resumed_recovery = load_task2_checkpoint(
                    resumed["latest_recovery_checkpoint"]["path"],
                    expected_sha256=resumed["latest_recovery_checkpoint"]["sha256"],
                    expected_type="recovery",
                )
                uninterrupted_recovery = load_task2_checkpoint(
                    uninterrupted["latest_recovery_checkpoint"]["path"],
                    expected_sha256=uninterrupted["latest_recovery_checkpoint"]["sha256"],
                    expected_type="recovery",
                )
                for name in ("model", "optimizer", "rng_state"):
                    task2_training._assert_round_trip(
                        resumed_recovery[name],
                        uninterrupted_recovery[name],
                        context=f"exact_resume.{name}",
                    )

    def test_benchmark_has_warmup_and_exactly_one_unpersisted_full_epoch(self) -> None:
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            root = Path(temporary)
            train, validation = _fixtures(root)
            factory_counter: list[int] = []
            before = sorted(root.iterdir())
            result = benchmark_task2_full_epoch(
                seed=13,
                train_data=train,
                validation_data=validation,
                test_injection=_injection(
                    maximum_epochs=3,
                    factory_counter=factory_counter,
                ),
            )
            self.assertEqual(factory_counter, [1, 1, 1])
            self.assertTrue(result["benchmark_only"])
            self.assertFalse(result["persistent_model_selection"])
            self.assertFalse(result["persistent_model_checkpoint"])
            self.assertEqual(result["scope"], ISOLATED_TEST_SCOPE)
            self.assertFalse(result["production_evidence"])
            self.assertTrue(result["warmup_completed"])
            self.assertEqual(result["measured_train_epochs"], 1)
            self.assertEqual(result["measured_validation_epochs"], 1)
            self.assertEqual(result["conservative_wall_time_factor"], CONSERVATIVE_WALL_TIME_FACTOR)
            self.assertAlmostEqual(
                result["estimated_one_seed_conservative_wall_seconds"],
                result["estimated_one_seed_maximum_epoch_seconds"] * CONSERVATIVE_WALL_TIME_FACTOR,
            )
            self.assertGreater(result["checkpoint_cpu_copy_seconds"], 0.0)
            self.assertGreater(result["best_checkpoint_serialization_seconds"], 0.0)
            self.assertGreater(result["recovery_checkpoint_serialization_seconds"], 0.0)
            self.assertEqual(sorted(root.iterdir()), before)

    def test_durable_benchmark_recovers_missing_lock_without_measuring_again(self) -> None:
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            root = Path(temporary)
            evidence_path = root / "task2-benchmark.json"
            train, validation = _fixtures(root)
            factory_counter: list[int] = []
            injection = _injection(maximum_epochs=3, factory_counter=factory_counter)
            measured = benchmark_task2_full_epoch(
                seed=13,
                train_data=train,
                validation_data=validation,
                test_injection=injection,
                evidence_output=evidence_path,
            )
            self.assertEqual(factory_counter, [1, 1, 1])
            self.assertTrue(measured["durable_evidence"])
            self.assertFalse(measured["recovered_existing_evidence"])
            result_sha256 = measured["result_artifact"]["sha256"]
            lock_path = Path(measured["completion_lock_artifact"]["path"])
            lock_path.unlink()

            recovered = benchmark_task2_full_epoch(
                seed=13,
                train_data=train,
                validation_data=validation,
                test_injection=injection,
                evidence_output=evidence_path,
            )
            self.assertEqual(factory_counter, [1, 1, 1, 1])
            self.assertTrue(recovered["recovered_existing_evidence"])
            self.assertEqual(recovered["result_artifact"]["sha256"], result_sha256)
            self.assertTrue(Path(recovered["completion_lock_artifact"]["path"]).is_file())

            evidence = bytearray(evidence_path.read_bytes())
            evidence[-2] = ord(" ")
            evidence_path.write_bytes(bytes(evidence))
            with self.assertRaises(ValueError):
                benchmark_task2_full_epoch(
                    seed=13,
                    train_data=train,
                    validation_data=validation,
                    test_injection=injection,
                    evidence_output=evidence_path,
                )

    def test_isolated_benchmark_cannot_publish_production_evidence(self) -> None:
        with self.assertRaises(PermissionError):
            benchmark_task2_full_epoch(
                seed=13,
                test_injection=_injection(maximum_epochs=1),
                evidence_output=DEFAULT_BENCHMARK_RESULT_PATH,
            )


class Task2BoundaryTests(unittest.TestCase):
    def test_production_roots_are_canonical_v2_namespaces(self) -> None:
        project_root = task2_training.PROJECT_ROOT
        self.assertEqual(DEFAULT_RUN_ROOT, project_root / "runs" / "task2_v2")
        self.assertEqual(
            DEFAULT_BENCHMARK_RESULT_PATH,
            project_root / "report_assets" / "provenance_v2" / "task2_benchmark_v2.json",
        )
        self.assertEqual(
            task2_training.DEFAULT_BENCHMARK_LOCK_PATH,
            project_root / "report_assets" / "provenance_v2" / "task2_benchmark_v2.lock.json",
        )

    def test_run_creation_rejects_symlinked_canonical_root_and_parent(self) -> None:
        runs_root = task2_training.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            boundary = Path(temporary)
            protected_root = boundary / "protected_task2_v1"
            protected_root.mkdir()
            linked_root = boundary / "task2_v2"
            linked_root.symlink_to(protected_root, target_is_directory=True)
            with (
                patch.object(task2_training, "DEFAULT_RUN_ROOT", linked_root),
                self.assertRaisesRegex(ValueError, "canonical v2 run root"),
            ):
                task2_training._run_directory(linked_root, "must_not_exist")
            self.assertEqual(list(protected_root.iterdir()), [])

            real_parent = boundary / "real_parent"
            real_parent.mkdir()
            linked_parent = boundary / "linked_parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            canonical_below_link = linked_parent / "task2_v2"
            with (
                patch.object(task2_training, "DEFAULT_RUN_ROOT", canonical_below_link),
                self.assertRaisesRegex(ValueError, "canonical v2 run root"),
            ):
                task2_training._run_directory(canonical_below_link, "must_not_exist")
            self.assertFalse((real_parent / "task2_v2").exists())

    def test_run_creation_allows_real_descendant_for_isolated_tests(self) -> None:
        runs_root = task2_training.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            canonical_root = Path(temporary) / "task2_v2"
            canonical_root.mkdir()
            isolated_root = canonical_root / "isolated"
            with patch.object(task2_training, "DEFAULT_RUN_ROOT", canonical_root):
                run = task2_training._run_directory(isolated_root, "safe_test_run")
            self.assertEqual(run, isolated_root / "safe_test_run")
            self.assertTrue(run.is_dir())

    def test_production_run_rejects_noncanonical_descendant_before_execution(self) -> None:
        runs_root = task2_training.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            canonical_root = Path(temporary) / "task2_v2"
            canonical_root.mkdir()
            descendant = canonical_root / "production_descendant"
            with (
                patch.object(task2_training, "DEFAULT_RUN_ROOT", canonical_root),
                patch.object(
                    task2_training,
                    "_resolve_runtime",
                    return_value=torch.device("cpu"),
                ),
                patch.object(task2_training, "_capture_execution_identity") as capture,
                self.assertRaisesRegex(PermissionError, "exact runs/task2_v2"),
            ):
                run_task2_development(seed=13, output_root=descendant)
            capture.assert_not_called()
            self.assertFalse(descendant.exists())

    def test_evaluation_loader_is_mps_only_data_free_and_identity_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "mps"):
            load_locked_task2_best_model_for_evaluation(
                "missing.pt",
                expected_sha256="1" * 64,
                expected_run_identity_sha256="2" * 64,
                device=torch.device("cpu"),
            )
        source = inspect.getsource(load_locked_task2_best_model_for_evaluation)
        self.assertNotIn("_open_real_data", source)
        self.assertNotIn("open_development_training_data", source)

        config = load_locked_task2_config()
        checkpoint_model = _TinyAutoencoder()
        checkpoint_optimizer = build_task2_optimizer(checkpoint_model, config)
        _populate_adamw_state(checkpoint_model, checkpoint_optimizer)
        numerical_runtime = {"schema_version": "test", "device": "mps"}
        execution_identity = task2_training.Task2ExecutionIdentity(
            implementation_sha256="5" * 64,
            requirements_lock_sha256="6" * 64,
            numerical_runtime=numerical_runtime,
            numerical_runtime_sha256=task2_training.sha256_json(numerical_runtime),
        )
        model_contract = task2_training._model_contract(checkpoint_model, config)
        _, config_file_sha256, _ = task2_training._descriptor_snapshot(
            task2_training.LOCKED_CONFIG_PATH
        )
        run_identity_sha256 = "1" * 64
        checkpoint = {
            "schema_version": task2_training.CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "best",
            "run_id": "task2-evaluation-loader",
            "run_identity_sha256": run_identity_sha256,
            "config_sha256": task2_training.config_fingerprint(config),
            "config_file_sha256": config_file_sha256,
            "cache_lock_sha256": KNOWN_CACHE_LOCK_SHA256,
            "implementation_sha256": execution_identity.implementation_sha256,
            "requirements_lock_sha256": execution_identity.requirements_lock_sha256,
            "numerical_runtime_sha256": execution_identity.numerical_runtime_sha256,
            "model_contract_sha256": task2_training.sha256_json(model_contract),
            "scope": PRODUCTION_SCOPE,
            "production_evidence": True,
            "seed": 13,
            "epoch": 1,
            "score": {"validation_loss": 0.25, "epoch": 1},
            "model": checkpoint_model.state_dict(),
            "optimizer": checkpoint_optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory(dir=DEFAULT_RUN_ROOT) as temporary:
            checkpoint_path = Path(temporary) / "best.pt"
            checkpoint_record = save_task2_checkpoint_create_only(
                checkpoint_path,
                checkpoint,
            )
            with (
                patch.object(
                    task2_training,
                    "_prepare_task2_verification_runtime",
                    return_value=torch.device("mps"),
                ),
                patch.object(
                    task2_training,
                    "_capture_execution_identity",
                    return_value=execution_identity,
                ),
                patch.object(
                    task2_training,
                    "_require_execution_identity_unchanged",
                ),
                patch.object(
                    task2_training,
                    "_build_task2_model",
                    side_effect=lambda *_args, **_kwargs: _TinyAutoencoder(),
                ),
                patch.object(
                    task2_training,
                    "open_development_training_data",
                    side_effect=AssertionError("evaluation loader opened data"),
                ),
            ):
                model, metadata = load_locked_task2_best_model_for_evaluation(
                    checkpoint_path,
                    expected_sha256=checkpoint_record["sha256"],
                    expected_run_identity_sha256=run_identity_sha256,
                    device=torch.device("mps"),
                )
            self.assertFalse(model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertEqual(metadata["scope"], PRODUCTION_SCOPE)
            self.assertTrue(metadata["production_evidence"])
            self.assertEqual(metadata["best_checkpoint_sha256"], checkpoint_record["sha256"])

    def test_production_cache_gate_is_canonical_and_published(self) -> None:
        with self.assertRaises(PermissionError), task2_training.ExitStack() as stack:
            _open_real_data(
                stack,
                cache_root=DEFAULT_CACHE_ROOT.parent,
                ffmpeg=None,
                expected_lock_sha256=KNOWN_CACHE_LOCK_SHA256,
            )
        with self.assertRaises(ValueError), task2_training.ExitStack() as stack:
            _open_real_data(
                stack,
                cache_root=DEFAULT_CACHE_ROOT,
                ffmpeg=None,
                expected_lock_sha256="0" * 64,
            )

    def test_development_engine_has_no_unknown_or_final_split_import_or_open(self) -> None:
        source = inspect.getsource(task2_training)
        self.assertNotIn("bird_audio.unknown", source)
        self.assertNotIn('split="test"', source)
        calls: list[dict[str, object]] = []
        train, validation = _fixtures(DEFAULT_CACHE_ROOT)
        train.lock_sha256 = KNOWN_CACHE_LOCK_SHA256
        validation.lock_sha256 = KNOWN_CACHE_LOCK_SHA256

        class _Context:
            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self.value

            def __exit__(self, *_exc):
                return None

        def fake_open(_root, **kwargs):
            calls.append(kwargs)
            return _Context(train if kwargs["split"] == "train" else validation)

        with (
            patch.object(
                task2_training,
                "open_development_training_data",
                side_effect=fake_open,
            ),
            task2_training.ExitStack() as stack,
        ):
            opened_train, opened_validation = _open_real_data(
                stack,
                cache_root=DEFAULT_CACHE_ROOT,
                ffmpeg=None,
                expected_lock_sha256=KNOWN_CACHE_LOCK_SHA256,
            )
        self.assertIs(opened_train, train)
        self.assertIs(opened_validation, validation)
        self.assertEqual([call["split"] for call in calls], ["train", "validation"])
        self.assertTrue(all(call["strategy"] == "energy" for call in calls))
        self.assertTrue(
            all(call["expected_lock_sha256"] == KNOWN_CACHE_LOCK_SHA256 for call in calls)
        )


if __name__ == "__main__":
    unittest.main()
