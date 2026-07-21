from __future__ import annotations

import gc
import os
import random
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.config import config_fingerprint, load_toml
from bird_audio.hashing import sha256_file
from bird_audio.io_utils import atomic_write_json, atomic_write_text
from bird_audio.paths import PROJECT_ROOT, require_safe_output
from bird_audio.provenance import (
    DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH,
    DEFAULT_MPS_SMOKE_V2_PATH,
    source_fingerprint,
)


def write_dependency_lock(output_path: str | Path = "requirements.lock") -> Path:
    dependency_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if dependency_check.returncode != 0:
        detail = (dependency_check.stdout or dependency_check.stderr).strip()
        raise RuntimeError(f"pip check failed; dependency lock was not written: {detail}")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        if (
            not line
            or line.startswith("# Editable install")
            or lowered.startswith("-e ")
            or lowered.startswith("bird-audio-coursework")
        ):
            continue
        lines.append(line)
    lines.sort(key=str.casefold)
    header = [
        "# Exact external dependency lock for ANN_Project.",
        f"# Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "# Install the local package separately with: pip install --no-deps -e .",
        "",
    ]
    return atomic_write_text(output_path, "\n".join([*header, *lines, ""]))


def run_mps_smoke_test(
    output_path: str | Path = DEFAULT_MPS_SMOKE_V2_PATH,
    checkpoint_path: str | Path = DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH,
) -> tuple[Path, dict[str, Any]]:
    started_at = datetime.now(UTC).isoformat()
    checkpoint = require_safe_output(checkpoint_path)
    checkpoint.unlink(missing_ok=True)
    base_result: dict[str, Any] = {
        "started_at_utc": started_at,
        "python_executable": sys.executable,
        "inside_project_venv": Path(sys.prefix).resolve() == (PROJECT_ROOT / ".venv").resolve(),
        "device": "mps",
        "dtype": "torch.float32",
        "passed": False,
    }
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").casefold() in {"1", "true", "yes"}:
        base_result.update(
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": "MpsFallbackEnabled",
                "error": "Disable PYTORCH_ENABLE_MPS_FALLBACK before the MPS smoke test",
            }
        )
        return atomic_write_json(output_path, base_result), base_result
    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as exc:
        base_result.update(
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": "Install the runtime dependencies before the MPS smoke test",
            }
        )
        return atomic_write_json(output_path, base_result), base_result

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        base_result.update(
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "torch_version": torch.__version__,
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
                "error_type": "MpsUnavailable",
                "error": "MPS is unavailable in this interpreter",
            }
        )
        return atomic_write_json(output_path, base_result), base_result

    try:
        from bird_audio.models import (
            ConvolutionalAutoencoder,
            build_efficientnet_b0_classifier,
            parameter_counts,
        )

        task1 = load_toml("configs/task1/final.toml")
        task2 = load_toml("configs/task2/autoencoder.toml")
        source_sha256 = source_fingerprint()
        task1_config_sha256 = config_fingerprint(task1)
        task2_config_sha256 = config_fingerprint(task2)
        classifier_batch_size = int(task1["training"]["batch_size"])
        autoencoder_batch_size = int(task2["training"]["batch_size"])

        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        torch.use_deterministic_algorithms(True)
        device = torch.device("mps")

        def cpu_copy(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: cpu_copy(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu_copy(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(item) for item in value)
            return value

        def assert_round_trip_equal(expected: Any, actual: Any, path: str) -> None:
            if torch.is_tensor(expected):
                if not torch.is_tensor(actual):
                    raise RuntimeError(f"Checkpoint value type changed at {path}")
                if expected.shape != actual.shape or expected.dtype != actual.dtype:
                    raise RuntimeError(f"Checkpoint tensor metadata changed at {path}")
                if not torch.equal(expected, actual):
                    raise RuntimeError(f"Checkpoint tensor values changed at {path}")
                return
            if isinstance(expected, dict):
                if not isinstance(actual, dict) or set(expected) != set(actual):
                    raise RuntimeError(f"Checkpoint dictionary keys changed at {path}")
                for key in expected:
                    assert_round_trip_equal(expected[key], actual[key], f"{path}.{key}")
                return
            if isinstance(expected, (list, tuple)):
                if not isinstance(actual, type(expected)) or len(expected) != len(actual):
                    raise RuntimeError(f"Checkpoint sequence changed at {path}")
                for index, (expected_item, actual_item) in enumerate(
                    zip(expected, actual, strict=True)
                ):
                    assert_round_trip_equal(
                        expected_item,
                        actual_item,
                        f"{path}[{index}]",
                    )
                return
            if expected != actual:
                raise RuntimeError(f"Checkpoint scalar changed at {path}")

        classifier = build_efficientnet_b0_classifier(
            class_count=int(task1["class_count"]),
            dropout=float(task1["dropout"]),
            weights_identifier=None,
            trainable_feature_indices=task1["trainable_feature_indices"],
        ).to(device)
        classifier.train()
        backbone_parameters = [
            parameter for parameter in classifier.features.parameters() if parameter.requires_grad
        ]
        head_parameters = [
            parameter for parameter in classifier.classifier.parameters() if parameter.requires_grad
        ]
        classifier_optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": float(task1["training"]["backbone_learning_rate"]),
                },
                {
                    "params": head_parameters,
                    "lr": float(task1["training"]["head_learning_rate"]),
                },
            ],
            weight_decay=float(task1["training"]["weight_decay"]),
        )
        classifier_inputs = torch.rand(
            classifier_batch_size,
            3,
            224,
            224,
            dtype=torch.float32,
            device=device,
        )
        classifier_targets = torch.arange(
            classifier_batch_size, device=device, dtype=torch.long
        ) % int(task1["class_count"])
        classifier_optimizer.zero_grad(set_to_none=True)
        classifier_logits = classifier(classifier_inputs)
        classifier_loss = nn.functional.cross_entropy(classifier_logits, classifier_targets)
        classifier_loss.backward()
        classifier_gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in classifier.parameters()
        )
        classifier_optimizer.step()
        torch.mps.synchronize()
        frozen_blocks_in_eval = all(
            not classifier.features[index].training
            for index in range(len(classifier.features))
            if index not in task1["trainable_feature_indices"]
        )
        selected_blocks_in_train = all(
            classifier.features[index].training for index in task1["trainable_feature_indices"]
        )
        classifier_state = cpu_copy(classifier.state_dict())
        classifier_optimizer_state = cpu_copy(classifier_optimizer.state_dict())
        classifier_loss_finite = bool(torch.isfinite(classifier_loss).item())
        classifier_loss_value = float(classifier_loss.detach().cpu().item())
        classifier_result = {
            "architecture": "efficientnet_b0",
            "pretrained_weights_loaded": False,
            "input_shape": list(classifier_inputs.shape),
            "output_shape": list(classifier_logits.shape),
            "batch_size": classifier_batch_size,
            "loss": classifier_loss_value,
            "gradients_finite": classifier_gradients_finite,
            "frozen_blocks_in_eval": frozen_blocks_in_eval,
            "selected_blocks_in_train": selected_blocks_in_train,
            "parameter_counts": parameter_counts(classifier),
        }
        classifier_output_shape = list(classifier_logits.shape)
        del (
            backbone_parameters,
            classifier,
            classifier_inputs,
            classifier_loss,
            classifier_logits,
            classifier_optimizer,
            classifier_targets,
            head_parameters,
        )
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()

        autoencoder = ConvolutionalAutoencoder(
            latent_dimensions=int(task2["latent_dimensions"])
        ).to(device)
        autoencoder.train()
        autoencoder_optimizer = torch.optim.AdamW(
            autoencoder.parameters(),
            lr=float(task2["training"]["learning_rate"]),
            weight_decay=float(task2["training"]["weight_decay"]),
        )
        autoencoder_inputs = torch.rand(
            autoencoder_batch_size,
            1,
            int(task2["input_height"]),
            int(task2["input_width"]),
            dtype=torch.float32,
            device=device,
        )
        autoencoder_optimizer.zero_grad(set_to_none=True)
        reconstructions, latent = autoencoder(autoencoder_inputs)
        autoencoder_loss = nn.functional.mse_loss(reconstructions, autoencoder_inputs)
        autoencoder_loss.backward()
        autoencoder_gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in autoencoder.parameters()
        )
        autoencoder_optimizer.step()
        torch.mps.synchronize()
        autoencoder_state = cpu_copy(autoencoder.state_dict())
        autoencoder_optimizer_state = cpu_copy(autoencoder_optimizer.state_dict())
        autoencoder_loss_finite = bool(torch.isfinite(autoencoder_loss).item())
        autoencoder_result = {
            "architecture": "skip_free_undercomplete_convolutional_autoencoder",
            "input_shape": list(autoencoder_inputs.shape),
            "reconstruction_shape": list(reconstructions.shape),
            "latent_shape": list(latent.shape),
            "batch_size": autoencoder_batch_size,
            "loss": float(autoencoder_loss.detach().cpu().item()),
            "gradients_finite": autoencoder_gradients_finite,
            "parameter_counts": parameter_counts(autoencoder),
        }

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint.name}.",
            suffix=".tmp",
            dir=checkpoint.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            states = {
                "provenance": {
                    "source_fingerprint_sha256": source_sha256,
                    "task1_config_sha256": task1_config_sha256,
                    "task2_config_sha256": task2_config_sha256,
                },
                "classifier": {
                    "model": classifier_state,
                    "optimizer": classifier_optimizer_state,
                },
                "autoencoder": {
                    "model": autoencoder_state,
                    "optimizer": autoencoder_optimizer_state,
                },
            }
            torch.save(states, temporary)
            loaded = torch.load(temporary, map_location="cpu", weights_only=True)
            assert_round_trip_equal(states, loaded, "checkpoint")
            temporary.replace(checkpoint)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        passed = (
            bool(base_result["inside_project_venv"])
            and classifier_loss_finite
            and autoencoder_loss_finite
            and classifier_gradients_finite
            and autoencoder_gradients_finite
            and frozen_blocks_in_eval
            and selected_blocks_in_train
            and classifier_output_shape == [classifier_batch_size, int(task1["class_count"])]
            and list(reconstructions.shape) == list(autoencoder_inputs.shape)
            and list(latent.shape) == [autoencoder_batch_size, int(task2["latent_dimensions"])]
        )
        result = {
            **base_result,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "torch_version": torch.__version__,
            "source_fingerprint_sha256": source_sha256,
            "task1_config_sha256": task1_config_sha256,
            "task2_config_sha256": task2_config_sha256,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "classifier": classifier_result,
            "autoencoder": autoencoder_result,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "passed": passed,
        }
    except BaseException as exc:
        result = {
            **base_result,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "error_type": type(exc).__name__,
            "error": str(exc).replace(str(PROJECT_ROOT), "ANN_Project")[:2000],
            "passed": False,
        }
    destination = atomic_write_json(output_path, result)
    return destination, result
