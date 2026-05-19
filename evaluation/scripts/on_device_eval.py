#!/usr/bin/env python3
"""
Phase 4 — On-device evaluation of the quantized T5 navigation model.

This script is STANDALONE — it does not depend on any project module.
Transfer it to the Android device alongside the quantized model directory.

Install dependencies in Termux first:
    pip install optimum[onnxruntime] transformers tokenizers sentencepiece rouge-score psutil

Usage (in Termux):
    python on_device_eval.py
    python on_device_eval.py --model-dir ~/nav_t5_quantized --test-csv ~/nav_dataset_test.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    print(f"[load] Loading tokenizer and model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ORTModelForSeq2SeqLM.from_pretrained(str(model_dir))
    print("[load] Model ready.")
    return tokenizer, model


# ── inference ─────────────────────────────────────────────────────────────────

def predict(text: str, tokenizer, model, *, num_beams: int = 4) -> str:
    inputs = tokenizer(
        "navigate: " + text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    )
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ── dataset ───────────────────────────────────────────────────────────────────

def load_dataset(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── metrics ───────────────────────────────────────────────────────────────────

def _build_memory_tracker():
    """Return a callable that reports current RSS in MB, or a no-op if psutil unavailable."""
    try:
        import psutil
        proc = psutil.Process()
        return lambda: proc.memory_info().rss / 1e6
    except ImportError:
        print("[warn] psutil not installed — memory tracking disabled")
        return lambda: 0.0


def compute_rouge(results: list[dict]) -> dict[str, Any]:
    try:
        from rouge_score import rouge_scorer as _rs
        scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rL = [], [], []
        for r in results:
            s = scorer.score(r["target"], r["predicted"])
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rL.append(s["rougeL"].fmeasure)
        return {
            "rouge1_f1": round(statistics.mean(r1), 4),
            "rouge2_f1": round(statistics.mean(r2), 4),
            "rougeL_f1": round(statistics.mean(rL), 4),
        }
    except ImportError:
        return {"error": "rouge-score not installed — run: pip install rouge-score"}


# ── evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    tokenizer,
    model,
    rows: list[dict],
    *,
    num_beams: int = 4,
    log_every: int = 20,
) -> tuple[list[dict], list[float], float]:
    """Return (per_sample_results, latencies_ms, peak_memory_mb)."""
    rss_mb = _build_memory_tracker()
    results: list[dict] = []
    latencies: list[float] = []
    peak_mem = 0.0

    for i, row in enumerate(rows):
        gc.collect()
        t0 = time.perf_counter()
        predicted = predict(row["input"], tokenizer, model, num_beams=num_beams)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        peak_mem = max(peak_mem, rss_mb())
        latencies.append(latency_ms)

        results.append({
            "input": row["input"],
            "target": row["target"],
            "predicted": predicted,
            "exact_match": predicted.strip().lower() == row["target"].strip().lower(),
            "latency_ms": round(latency_ms, 2),
        })

        if (i + 1) % log_every == 0 or (i + 1) == len(rows):
            print(
                f"  [{i + 1}/{len(rows)}]  "
                f"latency={latency_ms:.0f}ms  peak_mem={peak_mem:.0f}MB"
            )

    return results, latencies, peak_mem


# ── summary ───────────────────────────────────────────────────────────────────

def build_summary(
    results: list[dict],
    latencies: list[float],
    peak_mem_mb: float,
    model_dir: Path,
) -> dict[str, Any]:
    exact_matches = [r["exact_match"] for r in results]
    accuracy = sum(exact_matches) / len(exact_matches)

    n = len(latencies)
    sorted_lat = sorted(latencies)

    return {
        "accuracy": {
            "n_samples": n,
            "exact_match": round(accuracy, 4),
            **compute_rouge(results),
        },
        "latency_ms": {
            "mean":   round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95":    round(sorted_lat[int(0.95 * n)], 2),
            "p99":    round(sorted_lat[int(0.99 * n)], 2),
            "min":    round(min(latencies), 2),
            "max":    round(max(latencies), 2),
        },
        "memory_mb": {
            "peak_rss_mb": round(peak_mem_mb, 1),
        },
        "model_files_mb": {
            f.name: round(f.stat().st_size / 1e6, 1)
            for f in sorted(model_dir.glob("*.onnx"))
        },
        "per_sample_results": results,
    }


# ── entry point ───────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="On-device evaluation of quantized T5 navigation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model-dir",
        default="~/nav_t5_quantized",
        help="Path to quantized ONNX model directory",
    )
    ap.add_argument(
        "--test-csv",
        default="~/nav_dataset_test.csv",
        help="Path to test CSV file with 'input' and 'target' columns",
    )
    ap.add_argument(
        "--output-json",
        default="~/on_device_results.json",
        help="Output path for results JSON",
    )
    ap.add_argument(
        "--num-beams", type=int, default=4, metavar="N",
        help="Beam search width during generation",
    )
    ap.add_argument(
        "--log-every", type=int, default=20, metavar="N",
        help="Print progress every N samples",
    )
    args = ap.parse_args()

    model_dir   = Path(os.path.expanduser(args.model_dir))
    test_csv    = Path(os.path.expanduser(args.test_csv))
    output_json = Path(os.path.expanduser(args.output_json))

    if not model_dir.exists():
        sys.exit(f"Error: model directory not found: {model_dir}")
    if not test_csv.exists():
        sys.exit(f"Error: test CSV not found: {test_csv}")

    tokenizer, model = load_model(model_dir)

    print("[warmup] Running warm-up inference ...")
    predict("action: tap screen: home element: search_bar", tokenizer, model)

    rows = load_dataset(test_csv)
    print(f"\n[eval] Evaluating {len(rows)} samples ...")
    results, latencies, peak_mem = run_evaluation(
        tokenizer, model, rows,
        num_beams=args.num_beams,
        log_every=args.log_every,
    )

    summary = build_summary(results, latencies, peak_mem, model_dir)

    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    top = {k: v for k, v in summary.items() if k != "per_sample_results"}
    print("\n=== ON-DEVICE EVALUATION RESULTS ===")
    print(json.dumps(top, indent=2))
    print(f"\nFull results → {output_json}")


if __name__ == "__main__":
    _cli()
