#!/usr/bin/env python3
"""
Phase 1.3 — Quantize ONNX model to INT8 for ARM64 Android deployment.

Requires export_onnx.py to have been run first (evaluation/onnx/<version>/ must exist).

Run from the project root or evaluation/ directory:
    python evaluation/scripts/quantize_onnx.py [--model-version b4.3.2] [--per-channel]

Output is written to evaluation/quantized/<version>/.
This directory is what gets transferred to the Android device.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_onnx_dir, get_quantized_dir, resolve_model_version

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)


def _size_mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def _print_size_comparison(src_dir: Path, dst_dir: Path) -> None:
    print("\n  File size comparison (ONNX → INT8):")
    total_orig = total_quant = 0.0
    for f in sorted(src_dir.glob("*.onnx")):
        orig = _size_mb(f)
        q = dst_dir / f.name
        quant = _size_mb(q) if q.exists() else 0.0
        total_orig += orig
        total_quant += quant
        pct = (1 - quant / orig) * 100 if orig else 0.0
        print(f"    {f.name:<44} {orig:>6.1f} MB → {quant:>6.1f} MB  (-{pct:.0f}%)")
    print(f"    {'TOTAL':<44} {total_orig:>6.1f} MB → {total_quant:>6.1f} MB")


def _discover_onnx_files(src: Path) -> list[Path]:
    """Return all .onnx files in src, raising if none are found."""
    files = sorted(src.glob("*.onnx"))
    if not files:
        raise FileNotFoundError(f"No .onnx files found in {src}")
    return files


def quantize(model_version: str | None = None, per_channel: bool = False) -> Path:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    version = resolve_model_version(model_version)
    src = get_onnx_dir(version)
    dst = get_quantized_dir(version)

    if not src.exists():
        raise FileNotFoundError(
            f"ONNX directory not found: {src}\n"
            "Run 'python evaluation/scripts/export_onnx.py' first."
        )

    onnx_files = _discover_onnx_files(src)

    print(f"[quantize] version    : {version}")
    print(f"[quantize] source     : {src}")
    print(f"[quantize] output     : {dst}")
    print(f"[quantize] scheme     : ARM64 INT8, per_channel={per_channel}")
    print(f"[quantize] files      : {[f.name for f in onnx_files]}")

    dst.mkdir(parents=True, exist_ok=True)

    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=per_channel)

    for onnx_file in onnx_files:
        print(f"[quantize] quantizing : {onnx_file.name}")
        quantizer = ORTQuantizer.from_pretrained(src, file_name=onnx_file.name)
        quantizer.quantize(save_dir=str(dst), quantization_config=qconfig)

    for fname in _TOKENIZER_FILES:
        p = src / fname
        if p.exists():
            shutil.copy2(p, dst / fname)
            print(f"[quantize] copied     : {fname}")

    _print_size_comparison(src, dst)
    print(f"\n[quantize] done → {dst}")
    return dst


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8 for ARM64 Android",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model-version",
        metavar="VER",
        default=None,
        help="Model version directory (e.g. b4.3.2). Defaults to latest detected.",
    )
    ap.add_argument(
        "--per-channel",
        action="store_true",
        default=False,
        help="Use per-channel quantization (may improve accuracy, slightly slower)",
    )
    args = ap.parse_args()
    quantize(args.model_version, args.per_channel)


if __name__ == "__main__":
    _cli()
