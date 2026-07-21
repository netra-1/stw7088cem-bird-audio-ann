from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bird_audio.paths import require_safe_output


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = require_safe_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return atomic_write_text(path, payload)


def atomic_write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return atomic_write_text(path, buffer.getvalue())


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_snapshot(path: str | Path) -> tuple[list[dict[str, str]], str]:
    """Read and hash the exact same CSV byte snapshot."""
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    text = payload.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    return rows, digest


def require_unchanged(path: str | Path, expected_sha256: str) -> None:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"Input changed during command execution: {path}")
