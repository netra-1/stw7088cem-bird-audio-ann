from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_ROOT = PROJECT_ROOT / "dataset"
APPROVED_OUTPUT_ROOTS = tuple(
    PROJECT_ROOT / name for name in ("data", "runs", "evidence", "report_assets")
)
APPROVED_OUTPUT_FILES = (PROJECT_ROOT / "requirements.lock",)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_safe_output(path: str | Path) -> Path:
    """Allow generated output only in approved project roots outside raw data."""
    resolved = resolve_project_path(path)
    if is_relative_to(resolved, RAW_DATA_ROOT):
        raise ValueError(f"Refusing to write inside immutable raw data: {resolved}")
    if resolved not in APPROVED_OUTPUT_FILES and not any(
        is_relative_to(resolved, root) for root in APPROVED_OUTPUT_ROOTS
    ):
        allowed = ", ".join(str(root) for root in APPROVED_OUTPUT_ROOTS)
        raise ValueError(f"Output must be inside an approved derived root ({allowed}): {resolved}")
    return resolved
