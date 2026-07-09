"""Central configuration for the TFLite on-device evaluation pipeline.

Mirrors evaluation/config.py but targets TFLite outputs.
"""
from __future__ import annotations

import re
from pathlib import Path

EVAL_ROOT = Path(__file__).parent

# Source HF models — stored locally inside evaluation_tflite/model/
MODEL_ROOT = EVAL_ROOT / "model"

# Shared dataset
DATASET_DIR = EVAL_ROOT / "dataset"

# TFLite-specific output directories (all local to evaluation_tflite/)
SAVED_MODEL_ROOT = EVAL_ROOT / "saved_model"   # intermediate TF SavedModels
TFLITE_ROOT      = EVAL_ROOT / "tflite"        # float32 TFLite models
QUANTIZED_ROOT   = EVAL_ROOT / "quantized"     # INT8 quantized TFLite models
RESULTS_ROOT     = EVAL_ROOT / "results"       # versioned evaluation run outputs


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", name))


def list_model_versions() -> list[str]:
    if not MODEL_ROOT.exists():
        return []
    return sorted(
        (d.name for d in MODEL_ROOT.iterdir() if d.is_dir()),
        key=_version_key,
    )


def resolve_model_version(version: str | None = None) -> str:
    available = list_model_versions()
    if not available:
        raise FileNotFoundError(f"No model versions found under {MODEL_ROOT}")
    if version is None:
        return available[-1]
    if version not in available:
        raise ValueError(f"Version '{version}' not found. Available: {available}")
    return version


def get_model_dir(version: str) -> Path:
    return MODEL_ROOT / version


def get_saved_model_dir(version: str) -> Path:
    return SAVED_MODEL_ROOT / version


def get_tflite_dir(version: str) -> Path:
    return TFLITE_ROOT / version


def get_quantized_dir(version: str) -> Path:
    return QUANTIZED_ROOT / version


def default_dataset() -> Path:
    csvs = sorted(DATASET_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV dataset found in {DATASET_DIR}")
    return csvs[-1]
