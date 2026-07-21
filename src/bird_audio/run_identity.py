from __future__ import annotations

import re
from datetime import UTC, datetime

SAFE_TOKEN = re.compile(r"[^a-zA-Z0-9_-]+")


def _token(value: str) -> str:
    cleaned = SAFE_TOKEN.sub("-", value.strip()).strip("-")
    if not cleaned:
        raise ValueError("Run identifier tokens cannot be empty")
    return cleaned.lower()


def make_run_id(
    task: str,
    rung: str,
    seed: int,
    config_hash: str,
    data_hash: str,
    when: datetime | None = None,
) -> str:
    timestamp = (when or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp}_{_token(task)}_{_token(rung)}_s{seed:04d}_c{config_hash[:8]}_d{data_hash[:8]}"
    )
