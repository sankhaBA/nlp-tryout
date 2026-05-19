"""Central configuration for the on-device evaluation pipeline.

Model version naming convention: b<major>.<minor>.<patch>  (e.g. b4.3.2)
All paths are derived from EVAL_ROOT so the project is location-independent.
"""
from __future__ import annotations

import re
from pathlib import Path

EVAL_ROOT = Path(__file__).parent
MODEL_ROOT = EVAL_ROOT / "model"
ONNX_ROOT = EVAL_ROOT / "onnx"
QUANTIZED_ROOT = EVAL_ROOT / "quantized"
DATASET_DIR = EVAL_ROOT / "dataset"


def _version_key(name: str) -> tuple[int, ...]:
    """Parse 'b4.3.2' → (4, 3, 2) for consistent semantic sorting."""
    return tuple(int(x) for x in re.findall(r"\d+", name))


def list_model_versions() -> list[str]:
    """Return all model subdirectories sorted by semantic version (oldest first)."""
    if not MODEL_ROOT.exists():
        return []
    return sorted(
        (d.name for d in MODEL_ROOT.iterdir() if d.is_dir()),
        key=_version_key,
    )


def resolve_model_version(version: str | None = None) -> str:
    """Return *version* if given, otherwise auto-select the latest available."""
    available = list_model_versions()
    if not available:
        raise FileNotFoundError(f"No model versions found under {MODEL_ROOT}")
    if version is None:
        return available[-1]
    if version not in available:
        raise ValueError(
            f"Version '{version}' not found. Available: {available}"
        )
    return version


def get_model_dir(version: str) -> Path:
    return MODEL_ROOT / version


def get_onnx_dir(version: str) -> Path:
    return ONNX_ROOT / version


def get_quantized_dir(version: str) -> Path:
    return QUANTIZED_ROOT / version


def default_dataset() -> Path:
    """Return the most recently named CSV in the dataset directory."""
    csvs = sorted(DATASET_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV dataset found in {DATASET_DIR}")
    return csvs[-1]
