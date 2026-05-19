#!/usr/bin/env python3
"""
Phase 1.2 — Export fine-tuned T5 model to ONNX format.

Run from the project root or evaluation/ directory:
    python evaluation/scripts/export_onnx.py [--model-version b4.3.2] [--opset 13]

Output is written to evaluation/onnx/<version>/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_model_dir, get_onnx_dir, resolve_model_version

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)


def export(model_version: str | None = None, opset: int = 14) -> Path:
    from optimum.exporters.onnx import main_export

    version = resolve_model_version(model_version)
    src = get_model_dir(version)
    dst = get_onnx_dir(version)

    print(f"[export] version : {version}")
    print(f"[export] source  : {src}")
    print(f"[export] output  : {dst}")
    print(f"[export] opset   : {opset}")

    dst.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=str(src),
        output=str(dst),
        task="text2text-generation",
        opset=opset,
    )

    for fname in _TOKENIZER_FILES:
        p = src / fname
        if p.exists():
            shutil.copy2(p, dst / fname)
            print(f"[export] copied  : {fname}")

    print(f"\n[export] done → {dst}")
    return dst


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Export a fine-tuned T5 model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model-version",
        metavar="VER",
        default=None,
        help="Model version directory (e.g. b4.3.2). Defaults to latest detected.",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=14,
        metavar="N",
        help="ONNX opset version (minimum 14 for aten::triu support)",
    )
    args = ap.parse_args()
    export(args.model_version, args.opset)


if __name__ == "__main__":
    _cli()
