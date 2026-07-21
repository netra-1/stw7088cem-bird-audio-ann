from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bird_audio.audio import resolve_tool, tool_version
from bird_audio.hashing import fingerprint_files, sha256_file
from bird_audio.io_utils import atomic_write_json
from bird_audio.paths import PROJECT_ROOT

TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "tqdm",
]

PROVENANCE_V2_ROOT = PROJECT_ROOT / "report_assets" / "provenance_v2"
DEFAULT_ENVIRONMENT_V2_PATH = PROVENANCE_V2_ROOT / "environment_v2.json"
DEFAULT_MPS_SMOKE_V2_PATH = PROVENANCE_V2_ROOT / "mps_smoke_v2.json"
DEFAULT_MPS_SMOKE_CHECKPOINT_V2_PATH = PROVENANCE_V2_ROOT / "mps_smoke_checkpoint_v2.pt"
DEFAULT_SIGNAL_SMOKE_V2_PATH = PROVENANCE_V2_ROOT / "signal_smoke_v2.json"


def _sysctl(name: str) -> str:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return "unavailable"
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _sanitized_hardware_profile() -> dict[str, Any]:
    """Read only non-identifying hardware fields from macOS System Profiler."""
    try:
        completed = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        entries = payload.get("SPHardwareDataType") or []
        item = entries[0] if entries else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        item = {}
    return {
        "machine_name": item.get("machine_name", "unavailable"),
        "model_identifier": item.get("machine_model", "unavailable"),
        "chip": item.get("chip_type", "unavailable"),
        "total_cores": item.get("number_processors", "unavailable"),
        "memory": item.get("physical_memory", "unavailable"),
    }


def _pip_check() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (completed.stdout or completed.stderr).strip()
    return {"passed": completed.returncode == 0, "output": output}


def _mps_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_imported": False, "is_built": False, "is_available": False}
    return {
        "torch_imported": True,
        "is_built": bool(torch.backends.mps.is_built()),
        "is_available": bool(torch.backends.mps.is_available()),
    }


def _tool_record(name: str, explicit: str | Path | None) -> dict[str, str]:
    try:
        path = resolve_tool(name, explicit)
    except FileNotFoundError as exc:
        return {"path": "not_found", "version": "not_available", "error": str(exc)}
    return {"path": str(path), "version": tool_version(path), "error": ""}


def _dependency_lock_record() -> dict[str, Any]:
    path = PROJECT_ROOT / "requirements.lock"
    if not path.is_file():
        return {"exists": False, "verified": False, "path": str(path), "sha256": ""}
    lines = path.read_text(encoding="utf-8").splitlines()
    packages = [line for line in lines if line and not line.startswith("#")]
    verified = bool(packages) and not any(
        line.startswith("-e ") or "bird-audio-coursework" in line.casefold() for line in packages
    )
    return {
        "exists": True,
        "verified": verified,
        "path": str(path),
        "sha256": sha256_file(path),
        "external_packages": len(packages),
    }


def _mps_smoke_record() -> dict[str, Any]:
    path = DEFAULT_MPS_SMOKE_V2_PATH
    if not path.is_file():
        return {"exists": False, "verified": False, "path": str(path), "sha256": ""}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(str(payload.get("checkpoint") or ""))
        expected_checkpoint_sha256 = str(payload.get("checkpoint_sha256") or "")
        checkpoint_matches = (
            checkpoint.is_file()
            and bool(expected_checkpoint_sha256)
            and sha256_file(checkpoint) == expected_checkpoint_sha256
        )
        source_matches = payload.get("source_fingerprint_sha256") == source_fingerprint()
        verified = bool(
            payload.get("passed")
            and payload.get("inside_project_venv")
            and payload.get("mps_built")
            and payload.get("mps_available")
            and checkpoint_matches
            and source_matches
        )
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
        checkpoint_matches = False
        source_matches = False
        verified = False
    return {
        "exists": True,
        "verified": verified,
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_hash_matches": checkpoint_matches,
        "source_fingerprint_matches": source_matches,
        "completed_at_utc": payload.get("completed_at_utc", ""),
        "torch_version": payload.get("torch_version", ""),
    }


def source_fingerprint() -> str:
    paths: list[Path] = []
    for directory, pattern in (("src", "*.py"), ("configs", "*.toml"), ("tests", "*.py")):
        root = PROJECT_ROOT / directory
        if root.exists():
            paths.extend(root.rglob(pattern))
    for name in (
        "pyproject.toml",
        "requirements.in",
        "requirements-dev.in",
        "requirements.lock",
    ):
        path = PROJECT_ROOT / name
        if path.exists():
            paths.append(path)
    return fingerprint_files(paths, PROJECT_ROOT)


def collect_environment(
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
) -> dict[str, Any]:
    memory_bytes = _sysctl("hw.memsize")
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "inside_project_venv": Path(sys.prefix).resolve() == (PROJECT_ROOT / ".venv").resolve(),
        },
        "system": {
            "operating_system": platform.system(),
            "release": platform.release(),
            "macos_version": platform.mac_ver()[0],
            "architecture": platform.machine(),
            "processor": _sysctl("machdep.cpu.brand_string"),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": int(memory_bytes) if memory_bytes.isdigit() else memory_bytes,
            "hardware": _sanitized_hardware_profile(),
        },
        "accelerator": {"mps": _mps_status()},
        "packages": _package_versions(),
        "pip_check": _pip_check(),
        "tools": {
            "ffmpeg": _tool_record("ffmpeg", ffmpeg),
            "ffprobe": _tool_record("ffprobe", ffprobe),
        },
        "verified_artifacts": {
            "dependency_lock": _dependency_lock_record(),
            "mps_smoke": _mps_smoke_record(),
        },
        "source_fingerprint_sha256": source_fingerprint(),
    }


def save_environment(
    output_path: str | Path,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    record = collect_environment(ffmpeg=ffmpeg, ffprobe=ffprobe)
    destination = atomic_write_json(output_path, record)
    return destination, record
