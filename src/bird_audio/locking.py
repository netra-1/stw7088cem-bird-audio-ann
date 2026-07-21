from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from bird_audio.paths import PROJECT_ROOT


@contextmanager
def project_lock(name: str) -> Iterator[Path]:
    """Prevent concurrent evidence-producing commands from overwriting each other."""
    lock_directory = PROJECT_ROOT / "data" / ".locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{name}.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        detail = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"Project command lock already exists: {lock_path}\n{detail}") from exc
    try:
        payload = {
            "pid": os.getpid(),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "name": name,
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)
