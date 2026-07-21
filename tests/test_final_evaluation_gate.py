from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch

from bird_audio import final_evaluation_gate as gate
from bird_audio import recovery_v2 as recovery
from bird_audio.config import config_fingerprint, public_config
from bird_audio.hashing import sha256_json
from bird_audio.paths import PROJECT_ROOT
from bird_audio.task1_training import load_final_task1_config
from bird_audio.task2_training import load_locked_task2_config

H_SOURCE = "1" * 64
H_REQUIREMENTS = "2" * 64
H_WEIGHT = "3" * 64
H_IMPLEMENTATION = "4" * 64
H_TASK1_RUNTIME = "5" * 64
REAL_RECOVERY_EVIDENCE = gate._recovery_evidence

SMALL_COUNTS = {
    "train_clips": 10,
    "train_recordings": 10,
    "validation_clips": 2,
    "validation_recordings": 2,
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_binary(path: Path, value: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(value).hexdigest(),
        "size_bytes": len(value),
    }


class GateFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temporary.name)
        self.task1_root = self.root / "runs" / "task1"
        self.task2_root = self.root / "runs" / "task2"
        self.gate_directory = self.root / "runs" / "final_evaluation_v2" / "gate_v2"
        self.known_lock_path = self.root / "known" / "lock.json"
        self.unknown_lock_path = self.root / "unknown" / "lock.json"
        self.task1_root.mkdir(parents=True)
        self.task2_root.mkdir(parents=True)
        self.task1_config = load_final_task1_config()
        self.task2_config = load_locked_task2_config()
        self.task1_config_file_sha256 = hashlib.sha256(
            gate.TASK1_CONFIG_PATH.read_bytes()
        ).hexdigest()
        self.task2_config_file_sha256 = hashlib.sha256(
            gate.TASK2_CONFIG_PATH.read_bytes()
        ).hexdigest()
        self.task1_config_sha256 = config_fingerprint(self.task1_config)
        self.task2_config_sha256 = config_fingerprint(self.task2_config)
        self.checkpoints1: dict[str, dict[str, object]] = {}
        self.checkpoints2: dict[str, dict[str, object]] = {}
        self.verify1_calls: list[dict[str, object]] = []
        self.verify1_model_calls: list[dict[str, object]] = []
        self.verify2_calls: list[dict[str, object]] = []
        self.unknown_cache_calls: list[dict[str, object]] = []
        self.recovery_evidence_calls: list[dict[str, str]] = []
        self.recovery_evidence = {
            "source_fingerprint_sha256": H_SOURCE,
            "v1_recovery": {
                "manifest_id": "final_evaluation_v1_preinference_failure_recovery_v1",
                "manifest": {"path": "/recovery/manifest.json", "sha256": "a" * 64},
                "lock": {"path": "/recovery/lock.json", "sha256": "b" * 64},
            },
            "v2_cache_equivalence": {
                "equivalence_id": "unknown_cache_v1_to_v2_equivalence_v1",
                "certificate": {"path": "/equivalence.json", "sha256": "c" * 64},
                "lock": {"path": "/equivalence.lock.json", "sha256": "d" * 64},
                "full_rederivation": True,
                "scientific_artifacts_identical": True,
                "v2_cache_lock_sha256": "",
            },
        }
        self._write_cache_locks()
        for seed in gate.SEED_ORDER:
            self._write_task1_run(seed)
            self._write_task2_run(seed)
        self.stack = ExitStack()
        self.stack.enter_context(
            mock.patch.multiple(
                gate,
                TASK1_RUN_ROOT=self.task1_root,
                TASK2_RUN_ROOT=self.task2_root,
                KNOWN_CACHE_LOCK_PATH=self.known_lock_path,
                UNKNOWN_CACHE_LOCK_PATH=self.unknown_lock_path,
                EXPECTED_KNOWN_CACHE_LOCK_SHA256=self.known_lock_sha256,
                EXPECTED_UNKNOWN_CACHE_LOCK_SHA256=self.unknown_lock_sha256,
                TASK1_KNOWN_CACHE_LOCK_SHA256=self.known_lock_sha256,
                TASK2_KNOWN_CACHE_LOCK_SHA256=self.known_lock_sha256,
                FINAL_EVALUATION_GATE_DIRECTORY=self.gate_directory,
                FINAL_EVALUATION_GATE_PATH=self.gate_directory / "gate.json",
                FINAL_EVALUATION_GATE_LOCK_PATH=self.gate_directory / "lock.json",
                PRODUCTION_DATA_COUNTS=SMALL_COUNTS,
            )
        )
        self.stack.enter_context(mock.patch.object(gate, "load_task1_checkpoint", self.load1))
        self.stack.enter_context(mock.patch.object(gate, "load_task2_checkpoint", self.load2))
        self.stack.enter_context(
            mock.patch.object(
                gate,
                "load_unknown_scoring_clip_cache",
                self.load_unknown_cache,
            )
        )
        self.recovery_evidence["v2_cache_equivalence"]["v2_cache_lock_sha256"] = (
            self.unknown_lock_sha256
        )
        self.stack.enter_context(
            mock.patch.object(gate, "_recovery_evidence", self.bind_recovery_evidence)
        )
        self.stack.enter_context(
            mock.patch.object(gate, "verify_task1_development_run", self.verify1)
        )
        self.stack.enter_context(
            mock.patch.object(
                gate,
                "verify_locked_task1_best_checkpoint_model_state",
                self.verify1_model,
            )
        )
        self.stack.enter_context(
            mock.patch.object(gate, "verify_task2_development_run", self.verify2)
        )
        self.stack.enter_context(
            mock.patch.object(gate, "source_fingerprint", return_value=H_SOURCE)
        )

    def close(self) -> None:
        self.stack.close()
        self.temporary.cleanup()

    def _write_cache_locks(self) -> None:
        common = {
            "schema_version": "1.0",
            "cache_content_sha256": "6" * 64,
            "provenance": {"requirements_lock_sha256": H_REQUIREMENTS},
            "artifacts": {"summary": {"path": "summary.json", "sha256": "7" * 64}},
        }
        known = {**common, "cache_version": "known_clips_v1"}
        unknown = {
            **common,
            "cache_version": "unknown_clips_v2",
            "cache_content_sha256": "8" * 64,
        }
        self.known_lock_sha256 = _write_json(self.known_lock_path, known)["sha256"]
        self.unknown_lock_sha256 = _write_json(self.unknown_lock_path, unknown)["sha256"]

    def load1(self, path: str | Path, **_: object) -> dict[str, object]:
        return self.checkpoints1[str(Path(path))]

    def load2(self, path: str | Path, **_: object) -> dict[str, object]:
        return self.checkpoints2[str(Path(path))]

    def load_unknown_cache(
        self,
        cache_root: str | Path,
        *,
        ffmpeg: str | Path | None = None,
        expected_lock_sha256: str | None = None,
    ) -> mock.Mock:
        self.unknown_cache_calls.append(
            {
                "cache_root": Path(cache_root),
                "ffmpeg": ffmpeg,
                "expected_lock_sha256": expected_lock_sha256,
            }
        )
        return mock.Mock(
            root=Path(cache_root),
            lock_sha256=expected_lock_sha256,
        )

    def bind_recovery_evidence(
        self,
        source_fingerprint_sha256: str,
        expected_v2_cache_lock_sha256: str,
    ) -> dict[str, object]:
        self.recovery_evidence_calls.append(
            {
                "source_fingerprint_sha256": source_fingerprint_sha256,
                "expected_v2_cache_lock_sha256": expected_v2_cache_lock_sha256,
            }
        )
        return self.recovery_evidence

    def verify1(
        self,
        path: str | Path,
        *,
        expected_sha256: str,
        require_production: bool,
    ) -> dict[str, object]:
        completion_path = Path(path)
        completion = json.loads(completion_path.read_bytes())
        result = json.loads(Path(completion["result"]["path"]).read_bytes())
        identity = json.loads(Path(result["artifacts"]["run_identity"]["path"]).read_bytes())
        self.verify1_calls.append(
            {
                "path": completion_path,
                "expected_sha256": expected_sha256,
                "require_production": require_production,
            }
        )
        return {
            "valid": True,
            "complete": True,
            "run_id": result["run_id"],
            "seed": identity["seed"],
            "scope": result["scope"],
            "production_evidence": result["production_evidence"],
            "completion_lock_sha256": expected_sha256,
            "run_identity_sha256": result["run_identity_sha256"],
            "best_checkpoint_sha256": result["best_checkpoint"]["sha256"],
            "validation_recordings": SMALL_COUNTS["validation_recordings"],
            "validation_classes": 15,
            "macro_f1_rederived": True,
            "selection_rederived": True,
            "resume_prefix_verified": True,
        }

    def verify1_model(
        self,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_run_identity_sha256: str,
    ) -> dict[str, object]:
        checkpoint_path = Path(path)
        checkpoint = self.checkpoints1[str(checkpoint_path)]
        self.verify1_model_calls.append(
            {
                "path": checkpoint_path,
                "expected_sha256": expected_sha256,
                "expected_run_identity_sha256": expected_run_identity_sha256,
            }
        )
        return {
            "valid": True,
            "schema_version": checkpoint["schema_version"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": expected_sha256,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "run_id": checkpoint["run_id"],
            "run_identity_sha256": expected_run_identity_sha256,
            "config_sha256": checkpoint["config_sha256"],
            "cache_lock_sha256": checkpoint["cache_lock_sha256"],
            "weight_sha256": checkpoint["weight_sha256"],
            "implementation_sha256": checkpoint["implementation_sha256"],
            "requirements_lock_sha256": checkpoint["requirements_lock_sha256"],
            "numerical_runtime_sha256": checkpoint["numerical_runtime_sha256"],
            "scope": checkpoint["scope"],
            "production_evidence": checkpoint["production_evidence"],
            "seed": checkpoint["seed"],
            "epoch": checkpoint["epoch"],
            "score": checkpoint["score"],
            "model_contract": {
                "architecture": "efficientnet_b0",
                "model_type": "bird_audio.models.EfficientNetB0Classifier",
                "class_count": 15,
                "dropout": 0.2,
                "classifier_in_features": 1_280,
                "trainable_feature_indices": [6, 7, 8],
                "frozen_feature_indices": [0, 1, 2, 3, 4, 5],
                "parameter_counts": {"total": 4_026_763, "trainable": 3_174_955},
                "state_tensor_count": 360,
            },
        }

    def verify2(
        self,
        path: str | Path,
        *,
        expected_sha256: str,
        require_production: bool,
    ) -> dict[str, object]:
        completion_path = Path(path)
        completion = json.loads(completion_path.read_bytes())
        result = json.loads(Path(completion["result"]["path"]).read_bytes())
        identity = json.loads(Path(result["artifacts"]["run_identity"]["path"]).read_bytes())
        self.verify2_calls.append(
            {
                "path": completion_path,
                "expected_sha256": expected_sha256,
                "require_production": require_production,
            }
        )
        return {
            "valid": True,
            "complete": True,
            "run_id": result["run_id"],
            "seed": identity["seed"],
            "scope": result["scope"],
            "production_evidence": result["production_evidence"],
            "completion_lock_sha256": expected_sha256,
            "run_identity_sha256": result["run_identity_sha256"],
            "best_checkpoint_sha256": result["best_checkpoint"]["sha256"],
            "development_bundle_sha256": result["development_bundle"]["sha256"],
            "training_recordings": SMALL_COUNTS["train_recordings"],
            "training_clips": SMALL_COUNTS["train_clips"],
            "validation_recordings": SMALL_COUNTS["validation_recordings"],
            "validation_clips": SMALL_COUNTS["validation_clips"],
            "thresholds_rederived": True,
        }

    def _write_task1_run(self, seed: int) -> None:
        run_id = f"task1_seed_{seed}"
        run = self.task1_root / run_id
        for name in ("best_candidates", "recovery", "failures"):
            (run / name).mkdir(parents=True, exist_ok=True)
        numerical_runtime = {"schema_version": "1.0", "device": "mps", "seed": 0}
        numerical_runtime_sha256 = sha256_json(numerical_runtime)
        identity = {
            "schema_version": gate.TASK1_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "task": "task1_classification",
            "seed": seed,
            "config_sha256": self.task1_config_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "weight_sha256": H_WEIGHT,
            "implementation_sha256": H_IMPLEMENTATION,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "numerical_runtime_sha256": numerical_runtime_sha256,
            "scope": "production",
            "production_evidence": True,
        }
        identity_sha256 = sha256_json(identity)
        resolved_config_record = _write_json(
            run / "resolved_config.json",
            {
                "config_path": "configs/task1/final.toml",
                "config_file_sha256": self.task1_config_file_sha256,
                "config_sha256": self.task1_config_sha256,
                "resolved": public_config(self.task1_config),
            },
        )
        identity_record = _write_json(run / "run_identity.json", identity)
        history = [{"epoch": 1, "validation": {"macro_f1": 0.75}}]
        history_record = _write_json(run / "epoch_history.json", history)
        predictions = [
            {
                "recording_id": f"R{seed}A",
                "session_group": f"S{seed}A",
                "true_class_index": 0,
                "predicted_class_index": 0,
                "mean_logits": [1.0, *([0.0] * 14)],
            },
            {
                "recording_id": f"R{seed}B",
                "session_group": f"S{seed}B",
                "true_class_index": 1,
                "predicted_class_index": 1,
                "mean_logits": [0.0, 1.0, *([0.0] * 13)],
            },
        ]
        prediction_record = _write_json(run / "best_validation_predictions.json", predictions)
        best_path = run / "best_candidates" / "best_epoch_0001.pt"
        best_record = _write_binary(best_path, f"task1-best-{seed}".encode())
        latest_path = run / "recovery" / "recovery_epoch_0001.pt"
        latest_record = _write_binary(latest_path, f"task1-recovery-{seed}".encode())
        common = {
            "schema_version": gate.TASK1_CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "run_identity_sha256": identity_sha256,
            "config_sha256": self.task1_config_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "weight_sha256": H_WEIGHT,
            "implementation_sha256": H_IMPLEMENTATION,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "numerical_runtime_sha256": numerical_runtime_sha256,
            "scope": "production",
            "production_evidence": True,
            "seed": seed,
        }
        tensor_logits = torch.tensor(
            [row["mean_logits"] for row in predictions], dtype=torch.float32
        )
        self.checkpoints1[str(best_path)] = {
            **common,
            "checkpoint_type": "best",
            "epoch": 1,
            "score": {"macro_f1": 0.75, "validation_loss": 0.25, "epoch": 1},
            "predictions": {
                "recording_ids": tuple(row["recording_id"] for row in predictions),
                "session_groups": tuple(row["session_group"] for row in predictions),
                "true_labels": torch.tensor([0, 1], dtype=torch.long),
                "mean_logits": tensor_logits,
                "predicted_labels": torch.tensor([0, 1], dtype=torch.long),
            },
        }
        self.checkpoints1[str(latest_path)] = {
            **common,
            "checkpoint_type": "recovery",
            "completed_epoch": 1,
            "history": history,
        }
        provenance_record = _write_json(
            run / "provenance.json",
            {
                "schema_version": gate.TASK1_RUN_SCHEMA_VERSION,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "run_identity_sha256": identity_sha256,
                "command": ["bird-audio"],
                "config_path": "configs/task1/final.toml",
                "config_file_sha256": self.task1_config_file_sha256,
                "config_sha256": self.task1_config_sha256,
                "cache_root": str(self.root / "known"),
                "cache_lock_sha256": self.known_lock_sha256,
                "weight_path": str(self.root / "weight.pt"),
                "weight_sha256": H_WEIGHT,
                "weight_size_bytes": 1,
                "source_fingerprint_sha256": H_SOURCE,
                "implementation_sha256": H_IMPLEMENTATION,
                "requirements_lock_path": str(PROJECT_ROOT / "requirements.lock"),
                "requirements_lock_sha256": H_REQUIREMENTS,
                "numerical_runtime_sha256": numerical_runtime_sha256,
                "numerical_runtime": numerical_runtime,
                "scope": "production",
                "production_evidence": True,
                "environment": {
                    "device": "mps",
                    "mps_built": True,
                    "mps_available": True,
                    "deterministic_algorithms": True,
                },
                "parameter_counts": {"total": 1, "trainable": 1},
                "optimizer_groups": [],
                "initial_artifacts": {
                    "resolved_config": resolved_config_record,
                    "run_identity": identity_record,
                },
            },
        )
        artifacts = {
            "resolved_config": resolved_config_record,
            "run_identity": identity_record,
            "provenance": provenance_record,
            "epoch_history": history_record,
            "best_validation_predictions": prediction_record,
            "best_checkpoint": best_record,
            "latest_recovery": latest_record,
        }
        result = {
            "schema_version": gate.TASK1_RUN_SCHEMA_VERSION,
            "complete": True,
            "run_id": run_id,
            "run_directory": str(run),
            "run_identity_sha256": identity_sha256,
            "config_sha256": self.task1_config_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "weight_sha256": H_WEIGHT,
            "source_fingerprint_sha256": H_SOURCE,
            "implementation_sha256": H_IMPLEMENTATION,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "numerical_runtime_sha256": numerical_runtime_sha256,
            "scope": "production",
            "production_evidence": True,
            "resumed": False,
            "resume_checkpoint": None,
            "epochs_completed": 1,
            "early_stopped": False,
            "best_epoch": 1,
            "best_validation_macro_f1": 0.75,
            "best_validation_loss": 0.25,
            "best_checkpoint": best_record,
            "latest_recovery_checkpoint": latest_record,
            "artifacts": artifacts,
        }
        result_record = _write_json(run / "result.json", result)
        _write_json(
            run / "result.lock.json",
            {
                "schema_version": gate.TASK1_RUN_SCHEMA_VERSION,
                "run_identity_sha256": identity_sha256,
                "source_fingerprint_sha256": H_SOURCE,
                "implementation_sha256": H_IMPLEMENTATION,
                "requirements_lock_sha256": H_REQUIREMENTS,
                "numerical_runtime_sha256": numerical_runtime_sha256,
                "scope": "production",
                "production_evidence": True,
                "result": result_record,
            },
        )

    def _scored_split(
        self,
        *,
        role: str,
        run_identity_sha256: str,
        best_checkpoint_sha256: str,
    ) -> dict[str, object]:
        if role == "known_training":
            recording_ids = [f"T{index:02d}" for index in range(SMALL_COUNTS["train_recordings"])]
        else:
            recording_ids = [
                f"V{index:02d}" for index in range(SMALL_COUNTS["validation_recordings"])
            ]
        recordings = []
        clips = []
        for index, recording_id in enumerate(recording_ids):
            clip_id = f"{recording_id}-clip"
            score = float(index + 1) / 10.0
            recordings.append(
                {
                    "recording_id": recording_id,
                    "clip_ids": [clip_id],
                    "clip_count": 1,
                    "reconstruction_mse": score,
                    "mean_latent_embedding": [0.0] * 64,
                    "session_group": f"session-{recording_id}",
                }
            )
            clips.append(
                {
                    "recording_id": recording_id,
                    "clip_id": clip_id,
                    "session_group": f"session-{recording_id}",
                    "reconstruction_mse": score,
                    "latent_embedding": [0.0] * 64,
                }
            )
        value: dict[str, object] = {
            "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "source_role": role,
            "clip_count": len(clips),
            "recording_count": len(recordings),
            "recordings": recordings,
            "clips": clips,
        }
        if role == "known_validation":
            training_ids = [f"T{index:02d}" for index in range(SMALL_COUNTS["train_recordings"])]
            value["latent_novelty_scores"] = [
                {
                    "recording_id": recording_id,
                    "score": float(index) / 10.0,
                    "direction": "higher_is_more_novel",
                    "neighbour_recording_ids": training_ids,
                    "neighbour_distances": [float(index) / 10.0] * 10,
                }
                for index, recording_id in enumerate(recording_ids)
            ]
        return value

    def _write_task2_run_legacy(self, seed: int) -> None:
        run_id = f"task2_seed_{seed}"
        run = self.task2_root / run_id
        for name in ("best_candidates", "recovery", "failures", "development"):
            (run / name).mkdir(parents=True, exist_ok=True)
        runtime = {
            "python_executable": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "python_prefix": str((PROJECT_ROOT / ".venv").resolve()),
            "python_implementation": "CPython",
            "python_version": "3.11.0",
            "platform_system": "Darwin",
            "platform_release": "test",
            "platform_machine": "arm64",
            "torch_version": "2.10.0",
            "torchvision_version": "0.25.0",
            "numpy_version": "2.2.0",
            "device": "mps",
            "mps_built": True,
            "mps_available": True,
            "deterministic_algorithms": True,
            "mps_fallback_environment": "",
            "default_dtype": "torch.float32",
        }
        runtime_sha256 = sha256_json(runtime)
        model_contract = {
            "architecture": "skip_free_undercomplete_convolutional_autoencoder",
            "model_type": "bird_audio.models.ConvolutionalAutoencoder",
            "input_shape": [1, 224, 224],
            "latent_dimensions": 64,
            "parameter_counts": {
                "total": gate.EXPECTED_PARAMETER_COUNT,
                "trainable": gate.EXPECTED_PARAMETER_COUNT,
            },
            "state": [{"key": "encoder.weight", "shape": [1], "dtype": "torch.float32"}],
        }
        model_contract_sha256 = sha256_json(model_contract)
        optimizer_contract = {
            "type": "torch.optim.AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        }
        identity = {
            "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "task": "task2_novelty_detection_development",
            "seed": seed,
            "config_sha256": self.task2_config_sha256,
            "config_file_sha256": self.task2_config_file_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "source_fingerprint_sha256": H_SOURCE,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "runtime_identity": runtime,
            "runtime_identity_sha256": runtime_sha256,
            "model_contract": model_contract,
            "model_contract_sha256": model_contract_sha256,
            "optimizer_contract": optimizer_contract,
            "limits": {"maximum_epochs": 100, "batch_size": 64, "patience": 10},
            "data": {**SMALL_COUNTS, "selection_strategy": "energy"},
        }
        identity_sha256 = sha256_json(identity)
        resolved_config_record = _write_json(
            run / "resolved_config.json",
            {
                "config_path": "configs/task2/autoencoder.toml",
                "config_file_sha256": self.task2_config_file_sha256,
                "config_sha256": self.task2_config_sha256,
                "resolved": public_config(self.task2_config),
            },
        )
        identity_record = _write_json(run / "run_identity.json", identity)
        history = [{"epoch": 1, "validation": {"loss": 0.2}}]
        history_record = _write_json(run / "epoch_history.json", history)
        best_path = run / "best_candidates" / "best_epoch_0001.pt"
        best_record = _write_binary(best_path, f"task2-best-{seed}".encode())
        latest_path = run / "recovery" / "recovery_epoch_0001.pt"
        latest_record = _write_binary(latest_path, f"task2-recovery-{seed}".encode())
        common = {
            "schema_version": gate.TASK2_CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "run_identity_sha256": identity_sha256,
            "config_sha256": self.task2_config_sha256,
            "config_file_sha256": self.task2_config_file_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "source_fingerprint_sha256": H_SOURCE,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "runtime_identity_sha256": runtime_sha256,
            "model_contract_sha256": model_contract_sha256,
            "seed": seed,
        }
        self.checkpoints2[str(best_path)] = {
            **common,
            "checkpoint_type": "best",
            "epoch": 1,
            "score": {"validation_loss": 0.2, "epoch": 1},
        }
        self.checkpoints2[str(latest_path)] = {
            **common,
            "checkpoint_type": "recovery",
            "completed_epoch": 1,
            "history": history,
        }
        provenance_record = _write_json(
            run / "provenance.json",
            {
                "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "run_identity_sha256": identity_sha256,
                "command": ["bird-audio"],
                "config_path": "configs/task2/autoencoder.toml",
                "config_file_sha256": self.task2_config_file_sha256,
                "config_sha256": self.task2_config_sha256,
                "cache_root": str(self.root / "known"),
                "cache_lock_sha256": self.known_lock_sha256,
                "source_fingerprint_sha256": H_SOURCE,
                "requirements_lock_sha256": H_REQUIREMENTS,
                "runtime_identity": runtime,
                "runtime_identity_sha256": runtime_sha256,
                "model_contract": model_contract,
                "model_contract_sha256": model_contract_sha256,
                "optimizer_contract": optimizer_contract,
                "initial_artifacts": {
                    "resolved_config": resolved_config_record,
                    "run_identity": identity_record,
                },
            },
        )

        training_scores = self._scored_split(
            role="known_training",
            run_identity_sha256=identity_sha256,
            best_checkpoint_sha256=best_record["sha256"],
        )
        validation_scores = self._scored_split(
            role="known_validation",
            run_identity_sha256=identity_sha256,
            best_checkpoint_sha256=best_record["sha256"],
        )
        development_directory = run / "development"
        training_record = _write_json(
            development_directory / "known_training_recording_scores.json", training_scores
        )
        validation_record = _write_json(
            development_directory / "known_validation_recording_scores.json", validation_scores
        )
        training_ids = [f"T{index:02d}" for index in range(SMALL_COUNTS["train_recordings"])]
        reference_record = _write_json(
            development_directory / "known_training_latent_reference.json",
            {
                "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                "run_identity_sha256": identity_sha256,
                "best_checkpoint_sha256": best_record["sha256"],
                "reference": {
                    "fit_role": "known_training",
                    "recording_ids": training_ids,
                    "recording_count": len(training_ids),
                    "coordinate_mean": [0.0] * 64,
                    "population_variance": [0.0] * 64,
                    "coordinate_scale": [1.0] * 64,
                    "standardized_embeddings": [[0.0] * 64 for _ in training_ids],
                    "nearest_neighbours": 10,
                },
            },
        )
        validation_ids = [f"V{index:02d}" for index in range(SMALL_COUNTS["validation_recordings"])]
        threshold_record = _write_json(
            development_directory / "known_validation_thresholds.json",
            {
                "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                "run_identity_sha256": identity_sha256,
                "best_checkpoint_sha256": best_record["sha256"],
                "reconstruction": {
                    "score_name": "median_clip_reconstruction_mse",
                    "value": 0.2,
                    "calibration_role": "known_validation",
                    "calibration_recording_ids": validation_ids,
                    "quantile": 0.95,
                    "method": "higher",
                    "direction": "higher_is_more_novel",
                    "classification_operator": ">",
                },
                "latent": {
                    "score_name": "recording_mean_latent_knn_distance",
                    "value": 0.1,
                    "calibration_role": "known_validation",
                    "calibration_recording_ids": validation_ids,
                    "quantile": 0.95,
                    "method": "higher",
                    "direction": "higher_is_more_novel",
                    "classification_operator": ">",
                },
            },
        )
        development = {
            "known_training_scores": training_record,
            "known_validation_scores": validation_record,
            "training_latent_reference": reference_record,
            "thresholds": threshold_record,
        }
        bundle_record = _write_json(
            development_directory / "development_bundle.lock.json",
            {
                "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                "complete": True,
                "run_identity_sha256": identity_sha256,
                "best_checkpoint": best_record,
                "artifacts": development,
                "fit_roles": {
                    "latent_reference": "known_training",
                    "reconstruction_threshold": "known_validation",
                    "latent_threshold": "known_validation",
                },
                "threshold_operator": ">",
            },
        )
        artifacts = {
            "resolved_config": resolved_config_record,
            "run_identity": identity_record,
            "provenance": provenance_record,
            "epoch_history": history_record,
            "best_checkpoint": best_record,
            "latest_recovery": latest_record,
            "development": development,
            "development_bundle": bundle_record,
        }
        result = {
            "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
            "complete": True,
            "run_id": run_id,
            "run_directory": str(run),
            "run_identity_sha256": identity_sha256,
            "config_sha256": self.task2_config_sha256,
            "config_file_sha256": self.task2_config_file_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "source_fingerprint_sha256": H_SOURCE,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "runtime_identity_sha256": runtime_sha256,
            "model_contract_sha256": model_contract_sha256,
            "resumed": False,
            "resume_checkpoint": None,
            "epochs_completed": 1,
            "early_stopped": False,
            "best_epoch": 1,
            "best_validation_loss": 0.2,
            "best_checkpoint": best_record,
            "latest_recovery_checkpoint": latest_record,
            "development_bundle": bundle_record,
            "artifacts": artifacts,
        }
        result_record = _write_json(run / "result.json", result)
        _write_json(
            run / "result.lock.json",
            {
                "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
                "run_identity_sha256": identity_sha256,
                "result": result_record,
                "development_bundle": bundle_record,
            },
        )

    def _write_task2_run(self, seed: int) -> None:
        run_id = f"task2_seed_{seed}"
        run = self.task2_root / run_id
        for name in ("best_candidates", "recovery", "failures", "development"):
            (run / name).mkdir(parents=True, exist_ok=True)
        numerical_environment = {
            "OMP_NUM_THREADS": "unset",
            "MKL_NUM_THREADS": "unset",
            "VECLIB_MAXIMUM_THREADS": "unset",
            "PYTORCH_ENABLE_MPS_FALLBACK": "unset",
            "PYTORCH_MPS_FAST_MATH": "unset",
            "PYTORCH_MPS_PREFER_METAL": "unset",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "unset",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "unset",
        }
        numerical_runtime = {
            "schema_version": "1.0",
            "python_executable": str((PROJECT_ROOT / ".venv" / "bin" / "python").resolve()),
            "python_prefix": str((PROJECT_ROOT / ".venv").resolve()),
            "python_implementation": "CPython",
            "python_version": "3.11.0",
            "platform_system": "Darwin",
            "platform_release": "test",
            "platform_machine": "arm64",
            "macos_version": "15.0",
            "hardware_model": "test-model",
            "processor_brand": "test-processor",
            "torch_version": "2.10.0",
            "torch_build_config": "test-build",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "torchvision_version": "0.25.0",
            "numpy_version": "2.2.0",
            "device": "mps",
            "mps_built": True,
            "mps_available": True,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "mps_fallback": "disabled",
            "mps_fast_math": "disabled",
            "mps_prefer_metal": "default",
            "float32_matmul_precision": "highest",
            "default_dtype": "torch.float32",
            "training_dtype": "torch.float32",
            "numerical_environment": numerical_environment,
        }
        numerical_runtime_sha256 = sha256_json(numerical_runtime)
        model_contract = {
            "architecture": "skip_free_undercomplete_convolutional_autoencoder",
            "model_type": "bird_audio.models.ConvolutionalAutoencoder",
            "input_shape": [1, 224, 224],
            "latent_dimensions": 64,
            "parameter_counts": {
                "total": gate.EXPECTED_PARAMETER_COUNT,
                "trainable": gate.EXPECTED_PARAMETER_COUNT,
            },
            "state": [{"key": "encoder.weight", "shape": [1], "dtype": "torch.float32"}],
        }
        model_contract_sha256 = sha256_json(model_contract)
        optimizer_contract = {
            "type": "torch.optim.AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "decoupled_weight_decay": True,
        }
        final_evaluation_contract = {
            "primary_score": "median_clip_reconstruction_mse",
            "secondary_readout": "recording_mean_latent_knn_distance",
            "score_direction": "higher_is_more_novel",
            "nearest_neighbours": 10,
            "threshold_quantile": 0.95,
            "threshold_quantile_method": "higher",
            "threshold_scope": "per_seed_known_validation",
            "threshold_operator": ">",
            "bootstrap_seed": 20260713,
            "bootstrap_replicates": 2000,
            "bootstrap_interval_method": "percentile",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_resampling_unit": "session_cluster",
            "detailed_figure_seed": 37,
        }
        identity = {
            "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "task": "task2_novelty_detection_development",
            "seed": seed,
            "config_sha256": self.task2_config_sha256,
            "config_file_sha256": self.task2_config_file_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "implementation_sha256": H_IMPLEMENTATION,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "numerical_runtime": numerical_runtime,
            "numerical_runtime_sha256": numerical_runtime_sha256,
            "model_contract": model_contract,
            "model_contract_sha256": model_contract_sha256,
            "optimizer_contract": optimizer_contract,
            "final_evaluation_contract": final_evaluation_contract,
            "scope": "production",
            "production_evidence": True,
            "limits": {"maximum_epochs": 100, "batch_size": 64, "patience": 10},
            "data": {**SMALL_COUNTS, "selection_strategy": "energy"},
        }
        identity_sha256 = sha256_json(identity)
        resolved_config_record = _write_json(
            run / "resolved_config.json",
            {
                "config_path": "configs/task2/autoencoder.toml",
                "config_file_sha256": self.task2_config_file_sha256,
                "config_sha256": self.task2_config_sha256,
                "resolved": public_config(self.task2_config),
            },
        )
        identity_record = _write_json(run / "run_identity.json", identity)
        history = [{"epoch": 1, "validation": {"loss": 0.2}}]
        history_record = _write_json(run / "epoch_history.json", history)
        best_record = _write_binary(
            run / "best_candidates" / "best_epoch_0001.pt",
            f"task2-best-{seed}".encode(),
        )
        latest_record = _write_binary(
            run / "recovery" / "recovery_epoch_0001.pt",
            f"task2-recovery-{seed}".encode(),
        )
        provenance_record = _write_json(
            run / "provenance.json",
            {
                "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "run_identity_sha256": identity_sha256,
                "command": ["bird-audio"],
                "config_path": "configs/task2/autoencoder.toml",
                "config_file_sha256": self.task2_config_file_sha256,
                "config_sha256": self.task2_config_sha256,
                "cache_root": str(self.known_lock_path.parent),
                "cache_lock_sha256": self.known_lock_sha256,
                "release_source_fingerprint_sha256": H_SOURCE,
                "implementation_sha256": H_IMPLEMENTATION,
                "requirements_lock_sha256": H_REQUIREMENTS,
                "numerical_runtime": numerical_runtime,
                "numerical_runtime_sha256": numerical_runtime_sha256,
                "model_contract": model_contract,
                "model_contract_sha256": model_contract_sha256,
                "optimizer_contract": optimizer_contract,
                "final_evaluation_contract": final_evaluation_contract,
                "scope": "production",
                "production_evidence": True,
                "initial_artifacts": {
                    "resolved_config": resolved_config_record,
                    "run_identity": identity_record,
                },
            },
        )
        binding = {
            "run_identity_sha256": identity_sha256,
            "config_sha256": self.task2_config_sha256,
            "config_file_sha256": self.task2_config_file_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "implementation_sha256": H_IMPLEMENTATION,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "numerical_runtime_sha256": numerical_runtime_sha256,
            "model_contract_sha256": model_contract_sha256,
            "scope": "production",
            "production_evidence": True,
            "seed": seed,
            "best_checkpoint_sha256": best_record["sha256"],
        }
        development_directory = run / "development"
        development = {
            "known_training_scores": _write_json(
                development_directory / "known_training_recording_scores.json",
                {
                    "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                    **binding,
                    "fixture": "training_scores",
                },
            ),
            "known_validation_scores": _write_json(
                development_directory / "known_validation_recording_scores.json",
                {
                    "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                    **binding,
                    "fixture": "validation_scores",
                },
            ),
            "training_latent_reference": _write_json(
                development_directory / "known_training_latent_reference.json",
                {
                    "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                    **binding,
                    "fixture": "latent_reference",
                },
            ),
            "thresholds": _write_json(
                development_directory / "known_validation_thresholds.json",
                {
                    "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                    **binding,
                    "fixture": "thresholds",
                },
            ),
        }
        bundle_record = _write_json(
            development_directory / "development_bundle.lock.json",
            {
                "schema_version": gate.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
                "complete": True,
                **binding,
                "best_checkpoint": best_record,
                "artifacts": development,
                "fit_roles": {
                    "latent_reference": "known_training",
                    "reconstruction_threshold": "known_validation",
                    "latent_threshold": "known_validation",
                },
                "threshold_operator": ">",
                "final_evaluation_contract": final_evaluation_contract,
            },
        )
        artifacts = {
            "resolved_config": resolved_config_record,
            "run_identity": identity_record,
            "provenance": provenance_record,
            "epoch_history": history_record,
            "best_checkpoint": best_record,
            "latest_recovery": latest_record,
            "development": development,
            "development_bundle": bundle_record,
        }
        result = {
            "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
            "complete": True,
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "completed_at_utc": "2026-01-01T00:01:00+00:00",
            "run_id": run_id,
            "run_directory": str(run),
            "run_identity_sha256": identity_sha256,
            "config_sha256": self.task2_config_sha256,
            "config_file_sha256": self.task2_config_file_sha256,
            "cache_lock_sha256": self.known_lock_sha256,
            "release_source_fingerprint_sha256": H_SOURCE,
            "implementation_sha256": H_IMPLEMENTATION,
            "requirements_lock_sha256": H_REQUIREMENTS,
            "numerical_runtime_sha256": numerical_runtime_sha256,
            "model_contract_sha256": model_contract_sha256,
            "scope": "production",
            "production_evidence": True,
            "resumed": False,
            "resume_checkpoint": None,
            "epochs_completed": 1,
            "early_stopped": False,
            "best_epoch": 1,
            "best_validation_loss": 0.2,
            "best_checkpoint": best_record,
            "latest_recovery_checkpoint": latest_record,
            "development_bundle": bundle_record,
            "artifacts": artifacts,
        }
        result_record = _write_json(run / "result.json", result)
        _write_json(
            run / "result.lock.json",
            {
                "schema_version": gate.TASK2_RUN_SCHEMA_VERSION,
                "run_identity_sha256": identity_sha256,
                "implementation_sha256": H_IMPLEMENTATION,
                "requirements_lock_sha256": H_REQUIREMENTS,
                "numerical_runtime_sha256": numerical_runtime_sha256,
                "scope": "production",
                "production_evidence": True,
                "result": result_record,
                "development_bundle": bundle_record,
            },
        )


class FinalEvaluationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GateFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_seals_once_and_rerun_does_not_rewrite(self) -> None:
        first = gate.seal_final_evaluation_gate()
        gate_path = self.fixture.gate_directory / "gate.json"
        lock_path = self.fixture.gate_directory / "lock.json"
        gate_bytes = gate_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        gate_stat = gate_path.stat()
        lock_stat = lock_path.stat()

        second = gate.seal_final_evaluation_gate()

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(
            set(path.name for path in self.fixture.gate_directory.iterdir()),
            {"gate.json", "lock.json"},
        )
        self.assertEqual(gate_path.read_bytes(), gate_bytes)
        self.assertEqual(lock_path.read_bytes(), lock_bytes)
        self.assertEqual(gate_path.stat().st_mtime_ns, gate_stat.st_mtime_ns)
        self.assertEqual(lock_path.stat().st_mtime_ns, lock_stat.st_mtime_ns)
        self.assertEqual(first["gate"]["seed_order"], [13, 37, 71])
        self.assertEqual(first["gate"]["task1"]["run_count"], 3)
        self.assertEqual(first["gate"]["task2"]["run_count"], 3)
        self.assertEqual(first["gate"]["recovery_evidence"], self.fixture.recovery_evidence)
        self.assertTrue(
            all(
                call["source_fingerprint_sha256"] == H_SOURCE
                and call["expected_v2_cache_lock_sha256"] == self.fixture.unknown_lock_sha256
                for call in self.fixture.recovery_evidence_calls
            )
        )

    def test_recovery_binding_requires_full_scientific_equivalence(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            recovery_manifest = _write_json(root / "v1" / "manifest.json", {"v1": True})
            recovery_lock = _write_json(root / "v1" / "lock.json", {"locked": True})
            expected_v2_lock = "9" * 64
            equivalence_value = {
                "schema_version": "1.0",
                "equivalence_id": "equivalence-test-v1",
                "certified_at_utc": "2026-07-15T00:00:00+00:00",
                "source_fingerprint_sha256": H_SOURCE,
                "complete": True,
                "equivalence": {
                    "valid": True,
                    "full_rederivation": True,
                    "v1_recovery_manifest_sha256": recovery_manifest["sha256"],
                    "v1_recovery_lock_sha256": recovery_lock["sha256"],
                    "v2_cache_lock_sha256": expected_v2_lock,
                    "scientific_artifacts_identical": True,
                    "file_inodes_disjoint": True,
                },
            }
            equivalence = _write_json(
                root / "v2" / "unknown_cache_equivalence.json",
                equivalence_value,
            )
            equivalence_lock = _write_json(root / "v2" / "lock.json", {"locked": True})

            def relative_record(record: dict[str, object]) -> dict[str, object]:
                return {
                    **record,
                    "path": Path(str(record["path"])).relative_to(PROJECT_ROOT).as_posix(),
                }

            verified = {
                "valid": True,
                "equivalence": equivalence_value,
                "equivalence_artifact": relative_record(equivalence),
                "lock_artifact": relative_record(equivalence_lock),
                "created": False,
            }
            with (
                mock.patch.multiple(
                    recovery,
                    RECOVERY_MANIFEST_ID="recovery-test-v1",
                    RECOVERY_MANIFEST_PATH=Path(str(recovery_manifest["path"])),
                    RECOVERY_LOCK_PATH=Path(str(recovery_lock["path"])),
                    EQUIVALENCE_ID="equivalence-test-v1",
                    EQUIVALENCE_PATH=Path(str(equivalence["path"])),
                    EQUIVALENCE_LOCK_PATH=Path(str(equivalence_lock["path"])),
                ),
                mock.patch.object(
                    recovery,
                    "verify_unknown_cache_v2_equivalence_certificate",
                    return_value=verified,
                ) as verifier,
            ):
                binding = REAL_RECOVERY_EVIDENCE(H_SOURCE, expected_v2_lock)
            verifier.assert_called_once_with(full_rederivation=False)
            self.assertEqual(binding["source_fingerprint_sha256"], H_SOURCE)
            self.assertTrue(binding["v2_cache_equivalence"]["full_rederivation"])
            self.assertTrue(binding["v2_cache_equivalence"]["scientific_artifacts_identical"])

            invalid = json.loads(json.dumps(verified))
            invalid["equivalence"]["equivalence"]["full_rederivation"] = False
            with (
                mock.patch.multiple(
                    recovery,
                    RECOVERY_MANIFEST_ID="recovery-test-v1",
                    RECOVERY_MANIFEST_PATH=Path(str(recovery_manifest["path"])),
                    RECOVERY_LOCK_PATH=Path(str(recovery_lock["path"])),
                    EQUIVALENCE_ID="equivalence-test-v1",
                    EQUIVALENCE_PATH=Path(str(equivalence["path"])),
                    EQUIVALENCE_LOCK_PATH=Path(str(equivalence_lock["path"])),
                ),
                mock.patch.object(
                    recovery,
                    "verify_unknown_cache_v2_equivalence_certificate",
                    return_value=invalid,
                ),
                self.assertRaisesRegex(ValueError, "locked scientific identity"),
            ):
                REAL_RECOVERY_EVIDENCE(H_SOURCE, expected_v2_lock)

    def test_unknown_scoring_cache_is_opened_before_publication_and_verification(
        self,
    ) -> None:
        gate.seal_final_evaluation_gate()
        self.assertEqual(len(self.fixture.unknown_cache_calls), 2)
        gate.verify_final_evaluation_gate()
        self.assertEqual(len(self.fixture.unknown_cache_calls), 3)
        self.assertTrue(
            all(
                call["cache_root"] == self.fixture.unknown_lock_path.parent
                and call["expected_lock_sha256"] == self.fixture.unknown_lock_sha256
                and call["ffmpeg"] is None
                for call in self.fixture.unknown_cache_calls
            )
        )

    def test_stale_unknown_cache_blocks_gate_publication(self) -> None:
        with (
            mock.patch.object(
                gate,
                "load_unknown_scoring_clip_cache",
                side_effect=ValueError("implementation_sha256"),
            ),
            self.assertRaisesRegex(ValueError, "implementation_sha256"),
        ):
            gate.seal_final_evaluation_gate()
        self.assertFalse(self.fixture.gate_directory.exists())

    def test_stale_unknown_cache_blocks_existing_gate_before_any_claim(self) -> None:
        gate.seal_final_evaluation_gate()
        claim_path = (
            self.fixture.root / "runs" / "final_evaluation_v2" / "final_evaluation_attempt_v2.json"
        )
        with (
            mock.patch.object(
                gate,
                "load_unknown_scoring_clip_cache",
                side_effect=ValueError("implementation_sha256"),
            ),
            self.assertRaisesRegex(ValueError, "implementation_sha256"),
        ):
            gate.verify_final_evaluation_gate()
        self.assertFalse(claim_path.exists())

    def test_rejects_partial_extra_and_unexpected_run_inventory(self) -> None:
        partial = self.fixture.task1_root / "partial"
        partial.mkdir()
        with self.assertRaisesRegex(ValueError, "partial"):
            gate.seal_final_evaluation_gate()
        partial.rmdir()

        shutil.copytree(
            self.fixture.task1_root / "task1_seed_13",
            self.fixture.task1_root / "extra_complete",
        )
        with self.assertRaisesRegex(ValueError, "exactly three"):
            gate.seal_final_evaluation_gate()

    def test_rejects_symbolic_link_and_unexpected_file(self) -> None:
        link = self.fixture.task1_root / "linked_run"
        os.symlink(self.fixture.task1_root / "task1_seed_13", link)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            gate.seal_final_evaluation_gate()
        link.unlink()
        (self.fixture.task1_root / "notes.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected file"):
            gate.seal_final_evaluation_gate()

    def test_rejects_a_symlinked_artifact_parent_component(self) -> None:
        run = self.fixture.task1_root / "task1_seed_13"
        original = run / "best_candidates"
        moved = run / "best_candidates_real"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)

        with self.assertRaises(ValueError):
            gate.seal_final_evaluation_gate()

    def test_descriptor_traversal_fails_closed_during_parent_swap(self) -> None:
        boundary = self.fixture.root / "swap_boundary"
        parent = boundary / "safe"
        moved = boundary / "safe_real"
        outside = boundary / "outside"
        parent.mkdir(parents=True)
        outside.mkdir()
        target = parent / "artifact.json"
        target.write_text("safe", encoding="utf-8")
        (outside / "artifact.json").write_text("outside", encoding="utf-8")
        original_open = os.open
        swapped = False

        def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal swapped
            if path == "safe" and dir_fd is not None and not swapped:
                swapped = True
                parent.rename(moved)
                parent.symlink_to(outside, target_is_directory=True)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        with (
            mock.patch.object(gate.os, "open", swapping_open),
            self.assertRaises(ValueError),
        ):
            gate._descriptor_snapshot(target, boundary=boundary)
        self.assertTrue(swapped)
        self.assertEqual((outside / "artifact.json").read_text(encoding="utf-8"), "outside")

    def test_rejects_tampered_nested_development_artifact(self) -> None:
        path = (
            self.fixture.task2_root
            / "task2_seed_13"
            / "development"
            / "known_validation_thresholds.json"
        )
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "descriptor"):
            gate.seal_final_evaluation_gate()

    def test_task2_runs_are_recursively_verified_as_production(self) -> None:
        gate.seal_final_evaluation_gate()
        self.assertEqual(len(self.fixture.verify2_calls), 6)
        self.assertTrue(
            all(call["require_production"] is True for call in self.fixture.verify2_calls)
        )
        self.assertEqual(
            {Path(call["path"]).parent.name for call in self.fixture.verify2_calls},
            {"task2_seed_13", "task2_seed_37", "task2_seed_71"},
        )
        for run_id in ("task2_seed_13", "task2_seed_37", "task2_seed_71"):
            self.assertEqual(
                sum(
                    Path(call["path"]).parent.name == run_id for call in self.fixture.verify2_calls
                ),
                2,
            )

    def test_task1_runs_and_model_states_are_verified_before_normalization(self) -> None:
        gate.seal_final_evaluation_gate()
        self.assertEqual(len(self.fixture.verify1_calls), 6)
        self.assertEqual(len(self.fixture.verify1_model_calls), 6)
        self.assertTrue(
            all(call["require_production"] is True for call in self.fixture.verify1_calls)
        )
        self.assertEqual(
            {Path(call["path"]).parent.name for call in self.fixture.verify1_calls},
            {"task1_seed_13", "task1_seed_37", "task1_seed_71"},
        )

    def test_rejects_a_stored_source_fingerprint_that_is_not_current(self) -> None:
        with (
            mock.patch.object(gate, "source_fingerprint", return_value="9" * 64),
            self.assertRaisesRegex(ValueError, "current source"),
        ):
            gate.seal_final_evaluation_gate()

    def test_rejects_task2_schema_and_scope_drift(self) -> None:
        run = self.fixture.task2_root / "task2_seed_13"
        result_path = run / "result.json"
        completion_path = run / "result.lock.json"
        result = json.loads(result_path.read_bytes())
        result.pop("implementation_sha256")
        result_record = _write_json(result_path, result)
        completion = json.loads(completion_path.read_bytes())
        completion["result"] = result_record
        _write_json(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "schema"):
            gate.seal_final_evaluation_gate()

        self.fixture.close()
        self.fixture = GateFixture()
        run = self.fixture.task2_root / "task2_seed_13"
        result_path = run / "result.json"
        completion_path = run / "result.lock.json"
        result = json.loads(result_path.read_bytes())
        result["scope"] = "isolated_test"
        result["production_evidence"] = False
        result_record = _write_json(result_path, result)
        completion = json.loads(completion_path.read_bytes())
        completion["result"] = result_record
        _write_json(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "production"):
            gate.seal_final_evaluation_gate()

    def test_rejects_consistently_reindexed_task2_bundle_contract_tamper(self) -> None:
        run = self.fixture.task2_root / "task2_seed_13"
        bundle_path = run / "development" / "development_bundle.lock.json"
        bundle = json.loads(bundle_path.read_bytes())
        bundle["final_evaluation_contract"]["threshold_operator"] = ">="
        bundle_record = _write_json(bundle_path, bundle)

        result_path = run / "result.json"
        result = json.loads(result_path.read_bytes())
        result["development_bundle"] = bundle_record
        result["artifacts"]["development_bundle"] = bundle_record
        result_record = _write_json(result_path, result)

        completion_path = run / "result.lock.json"
        completion = json.loads(completion_path.read_bytes())
        completion["result"] = result_record
        completion["development_bundle"] = bundle_record
        _write_json(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "bundle identity"):
            gate.seal_final_evaluation_gate()

    def test_rejects_mixed_scope_and_cpu_task2_identity(self) -> None:
        result_path = self.fixture.task1_root / "task1_seed_13" / "result.json"
        result = json.loads(result_path.read_bytes())
        result["scope"] = "isolated_test"
        result_record = _write_json(result_path, result)
        completion_path = self.fixture.task1_root / "task1_seed_13" / "result.lock.json"
        completion = json.loads(completion_path.read_bytes())
        completion["result"] = result_record
        _write_json(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "production evidence"):
            gate.seal_final_evaluation_gate()

        identity_path = self.fixture.task2_root / "task2_seed_13" / "run_identity.json"
        identity = json.loads(identity_path.read_bytes())
        identity["numerical_runtime"]["device"] = "cpu"
        identity["numerical_runtime_sha256"] = sha256_json(identity["numerical_runtime"])
        with self.assertRaisesRegex(ValueError, "production MPS"):
            gate._validate_task2_production_identity(identity)

    def test_rejects_duplicate_seed_and_mixed_identity_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            gate._validate_seed_inventory(
                [
                    {"seed": 13, "run_id": "one"},
                    {"seed": 13, "run_id": "two"},
                    {"seed": 71, "run_id": "three"},
                ],
                "Task 1",
            )
        with self.assertRaisesRegex(ValueError, "mixed source"):
            gate._uniform_value(
                [{"source": "a"}, {"source": "b"}, {"source": "a"}],
                "source",
                "Task 2",
            )

    def test_gate_direct_reads_exclude_final_feature_artifacts(self) -> None:
        observed: list[Path] = []
        original = gate._descriptor_snapshot

        def recording_snapshot(
            path: Path,
            *,
            boundary: Path = PROJECT_ROOT,
        ) -> tuple[bytes, str, int]:
            observed.append(Path(path))
            return original(path, boundary=boundary)

        with mock.patch.object(gate, "_descriptor_snapshot", recording_snapshot):
            gate.seal_final_evaluation_gate()
        relative = [path.as_posix() for path in observed]
        self.assertFalse(any("/test/features/" in path for path in relative))
        self.assertFalse(any(path.endswith("/test/index.csv") for path in relative))
        self.assertFalse(any("/scoring/features/" in path for path in relative))
        self.assertFalse(any(path.endswith("/scoring/index.csv") for path in relative))

    def test_verify_rejects_seal_or_current_evidence_tampering(self) -> None:
        gate.seal_final_evaluation_gate()
        gate_path = self.fixture.gate_directory / "gate.json"
        gate_path.write_bytes(gate_path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            gate.verify_final_evaluation_gate()

        shutil.rmtree(self.fixture.gate_directory)
        gate.seal_final_evaluation_gate()
        threshold_path = (
            self.fixture.task2_root
            / "task2_seed_71"
            / "development"
            / "known_validation_thresholds.json"
        )
        threshold_path.write_bytes(threshold_path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            gate.verify_final_evaluation_gate()


if __name__ == "__main__":
    unittest.main()
