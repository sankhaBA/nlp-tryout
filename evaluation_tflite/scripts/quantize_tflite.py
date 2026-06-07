#!/usr/bin/env python3
"""
Phase 1.3 — Quantize TFLite models to INT8 dynamic-range for ARM64 Android.

Requires export_tflite.py to have been run first (saved_model/<version>/ must exist).
Re-converts the TF SavedModels with INT8 dynamic-range quantization enabled.

Run from the project root or evaluation_tflite/ directory:
    python evaluation_tflite/scripts/quantize_tflite.py [--model-version b4.3.2]

Output is written to evaluation_tflite/quantized/<version>/.
This directory is what gets transferred to the Android device.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    get_model_dir,
    get_saved_model_dir,
    get_quantized_dir,
    resolve_model_version,
)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "config.json",
    "generation_config.json",
)

_SAVED_MODEL_PARTS = ("encoder", "decoder_step")


def _size_mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def _to_tflite_int8(saved_model_path: Path, out_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_path),
        signature_keys=["serving_default"],
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.allow_custom_ops = False

    # Dynamic-range INT8: weights quantized to 8-bit, activations quantized at runtime.
    # No calibration dataset required — suitable for seq2seq on mobile.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"  written {out_path.name}  ({len(tflite_model) / 1e6:.1f} MB)")


def _print_size_comparison(tflite_dir: Path, dst_dir: Path) -> None:
    print("\n  File size comparison (float32 -> INT8):")
    total_orig = total_quant = 0.0
    for part in _SAVED_MODEL_PARTS:
        orig_path = tflite_dir / f"{part}.tflite"
        quant_path = dst_dir / f"{part}_int8.tflite"
        if not orig_path.exists():
            continue
        orig = _size_mb(orig_path)
        quant = _size_mb(quant_path) if quant_path.exists() else 0.0
        total_orig += orig
        total_quant += quant
        pct = (1 - quant / orig) * 100 if orig else 0.0
        print(f"    {orig_path.name:<36} {orig:>6.1f} MB -> {quant:>6.1f} MB  (-{pct:.0f}%)")
    print(f"    {'TOTAL':<36} {total_orig:>6.1f} MB -> {total_quant:>6.1f} MB")


def quantize(model_version: str | None = None) -> Path:
    version  = resolve_model_version(model_version)
    sm_dir   = get_saved_model_dir(version)
    src_hf   = get_model_dir(version)
    dst      = get_quantized_dir(version)

    if not sm_dir.exists():
        raise FileNotFoundError(
            f"SavedModel directory not found: {sm_dir}\n"
            "Run 'python evaluation_tflite/scripts/export_tflite.py' first."
        )

    for part in _SAVED_MODEL_PARTS:
        part_path = sm_dir / part
        if not part_path.exists():
            raise FileNotFoundError(
                f"SavedModel sub-directory not found: {part_path}\n"
                "Re-run export_tflite.py."
            )

    print(f"[quantize] version : {version}")
    print(f"[quantize] source  : {sm_dir}")
    print(f"[quantize] output  : {dst}")
    print(f"[quantize] scheme  : INT8 dynamic-range, SELECT_TF_OPS")

    dst.mkdir(parents=True, exist_ok=True)

    print("\n[quantize] Converting to INT8 TFLite ...")
    for part in _SAVED_MODEL_PARTS:
        print(f"[quantize] quantizing : {part}")
        _to_tflite_int8(sm_dir / part, dst / f"{part}_int8.tflite")

    # Copy tokenizer and config files needed for on-device inference
    for fname in _TOKENIZER_FILES:
        p = src_hf / fname
        if p.exists():
            shutil.copy2(p, dst / fname)
            print(f"[quantize] copied  : {fname}")

    # Print size comparison vs float32 tflite (if available)
    from config import get_tflite_dir
    tfl_dir = get_tflite_dir(version)
    if tfl_dir.exists():
        _print_size_comparison(tfl_dir, dst)

    print(f"\n[quantize] done -> {dst}")
    return dst


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Quantize TFLite T5 model to INT8 for ARM64 Android",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model-version", metavar="VER", default=None,
        help="Model version directory (e.g. b4.3.2). Defaults to latest detected.",
    )
    args = ap.parse_args()
    quantize(args.model_version)


if __name__ == "__main__":
    _cli()
