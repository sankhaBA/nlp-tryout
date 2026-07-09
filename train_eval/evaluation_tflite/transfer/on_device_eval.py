#!/usr/bin/env python3
"""
Phase 4 — On-device evaluation of the INT8 TFLite T5 navigation model on Android.

STANDALONE — no project dependencies. Transfer to Android alongside the model.

Install dependencies in Termux:
    pip install tflite-runtime tokenizers sentencepiece rouge-score psutil

Usage (Termux):
    python on_device_eval.py
    python on_device_eval.py --model-dir ~/nav_t5_tflite --test-csv ~/nav_dataset_test.csv --model-version b4.3.2

Each run writes three artefacts to ~/nav_t5_results/<run_id>/:
    summary.json   — full metrics + per-sample data (machine-readable)
    per_sample.csv — tabular predictions (spreadsheet-friendly)
    report.txt     — human-readable report printed to console as well
A cumulative runs_index.json in ~/nav_t5_results/ tracks key metrics across all runs.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(model_dir: Path):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        sys.exit(
            "tflite-runtime is not installed.\n"
            "Install it with: pip install tflite-runtime"
        )

    enc_path = model_dir / "encoder_int8.tflite"
    dec_path = model_dir / "decoder_step_int8.tflite"

    for p in (enc_path, dec_path):
        if not p.exists():
            sys.exit(f"Model file not found: {p}")

    print(f"[load] encoder      : {enc_path}")
    print(f"[load] decoder_step : {dec_path}")

    enc = tflite.Interpreter(model_path=str(enc_path))
    dec = tflite.Interpreter(model_path=str(dec_path))
    enc.allocate_tensors()
    dec.allocate_tensors()

    print("[load] Models ready.")
    return enc, dec


def load_tokenizer(model_dir: Path):
    from tokenizers import Tokenizer
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"tokenizer.json not found in {model_dir}")
    return Tokenizer.from_file(str(tok_path))


def _read_special_tokens(model_dir: Path) -> tuple[int, int]:
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        return int(cfg.get("decoder_start_token_id", 0)), int(cfg.get("eos_token_id", 1))
    return 0, 1  # T5 defaults


# ── inference ─────────────────────────────────────────────────────────────────

def _name_key(full: str) -> str:
    """Strip TF serving prefix/port from tensor names like 'serving_default_input_ids:0'."""
    return full.split(":")[0].split("/")[-1].removeprefix("serving_default_")


def _run_encoder(enc_interp, input_ids, attention_mask):
    import numpy as np

    in_details  = enc_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    def _idx(key, fallback_pos):
        return name_to_idx.get(key, in_details[fallback_pos]["index"])

    enc_interp.resize_tensor_input(_idx("input_ids",      0), input_ids.shape)
    enc_interp.resize_tensor_input(_idx("attention_mask", 1), attention_mask.shape)
    enc_interp.allocate_tensors()

    in_details  = enc_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    enc_interp.set_tensor(_idx("input_ids",      0), input_ids)
    enc_interp.set_tensor(_idx("attention_mask", 1), attention_mask)
    enc_interp.invoke()
    return enc_interp.get_tensor(enc_interp.get_output_details()[0]["index"])


def _run_decoder_step(dec_interp, decoder_input_ids, encoder_hidden_states):
    import numpy as np

    in_details  = dec_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    def _idx(key, fallback_pos):
        return name_to_idx.get(key, in_details[fallback_pos]["index"])

    dec_interp.resize_tensor_input(_idx("decoder_input_ids",     0), decoder_input_ids.shape)
    dec_interp.resize_tensor_input(_idx("encoder_hidden_states", 1), encoder_hidden_states.shape)
    dec_interp.allocate_tensors()

    in_details  = dec_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    dec_interp.set_tensor(_idx("decoder_input_ids",     0), decoder_input_ids)
    dec_interp.set_tensor(_idx("encoder_hidden_states", 1), encoder_hidden_states)
    dec_interp.invoke()
    return dec_interp.get_tensor(dec_interp.get_output_details()[0]["index"])


def predict(
    text: str,
    tokenizer,
    enc_interp,
    dec_interp,
    decoder_start_token_id: int,
    eos_token_id: int,
    max_length: int = 64,
) -> str:
    import numpy as np

    encoded        = tokenizer.encode("navigate: " + text)
    input_ids      = np.array([encoded.ids],            dtype=np.int32)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int32)

    encoder_hidden_states = _run_encoder(enc_interp, input_ids, attention_mask)

    tokens = [decoder_start_token_id]
    for _ in range(max_length):
        dec_input  = np.array([tokens], dtype=np.int32)
        logits     = _run_decoder_step(dec_interp, dec_input, encoder_hidden_states)
        next_token = int(np.argmax(logits[0, -1]))
        tokens.append(next_token)
        if next_token == eos_token_id:
            break

    skip       = {decoder_start_token_id, eos_token_id}
    output_ids = [t for t in tokens if t not in skip]
    return tokenizer.decode(output_ids)


# ── dataset ───────────────────────────────────────────────────────────────────

def load_dataset(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── metrics ───────────────────────────────────────────────────────────────────

def _build_memory_tracker():
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
    enc_interp,
    dec_interp,
    decoder_start_token_id: int,
    eos_token_id: int,
    rows: list[dict],
    *,
    log_every: int = 20,
) -> tuple[list[dict], list[float], float]:
    rss_mb    = _build_memory_tracker()
    results   : list[dict]  = []
    latencies : list[float] = []
    peak_mem  = 0.0

    for i, row in enumerate(rows):
        gc.collect()
        t0        = time.perf_counter()
        predicted = predict(
            row["input"], tokenizer, enc_interp, dec_interp,
            decoder_start_token_id, eos_token_id,
        )
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        peak_mem   = max(peak_mem, rss_mb())
        latencies.append(latency_ms)

        results.append({
            "input":      row["input"],
            "target":     row["target"],
            "predicted":  predicted,
            "latency_ms": round(latency_ms, 2),
        })

        if (i + 1) % log_every == 0 or (i + 1) == len(rows):
            print(
                f"  [{i + 1}/{len(rows)}]  "
                f"latency={latency_ms:.0f}ms  peak_mem={peak_mem:.0f}MB"
            )

    return results, latencies, peak_mem


# ── summary & reporting ───────────────────────────────────────────────────────

def build_summary(
    results: list[dict],
    latencies: list[float],
    peak_mem_mb: float,
    model_dir: Path,
    *,
    run_id: str = "",
    timestamp: str = "",
    platform: str = "",
    model_version: str = "",
    dataset_name: str = "",
) -> dict[str, Any]:
    n          = len(latencies)
    sorted_lat = sorted(latencies)

    return {
        "run_id":        run_id,
        "timestamp":     timestamp,
        "platform":      platform,
        "model_version": model_version,
        "dataset":       dataset_name,
        "accuracy": {
            "n_samples": n,
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
            f: round(os.path.getsize(os.path.join(str(model_dir), f)) / 1e6, 1)
            for f in sorted(os.listdir(str(model_dir)))
            if f.endswith(".tflite")
        },
        "per_sample_results": results,
    }


def _detect_platform() -> str:
    """Auto-detect the runtime environment for the run-ID tag."""
    if "ANDROID_ROOT" in os.environ:
        return "android"
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _format_report(
    run_id: str,
    platform_tag: str,
    model_version: str,
    dataset_name: str,
    summary: dict,
    results: list[dict],
) -> str:
    W    = 68
    SEP  = "=" * W
    DASH = "-" * W

    acc = summary["accuracy"]
    lat = summary["latency_ms"]
    mem = summary["memory_mb"]
    n   = acc["n_samples"]

    def lr(label: str, value: str) -> str:
        return f"  {label:<22}{value}"

    lines = [
        SEP,
        "  TFLite Navigation Model — Evaluation Report",
        SEP,
        lr("Run ID :",    run_id),
        lr("Platform :",  platform_tag),
        lr("Model :",     model_version),
        lr("Dataset :",   f"{dataset_name}  ({n} samples)"),
        lr("Timestamp :", summary.get("timestamp", "unknown")),
        SEP,
        "  ROUGE SCORES",
    ]

    for key, label in [
        ("rouge1_f1", "  ROUGE-1 F1 :"),
        ("rouge2_f1", "  ROUGE-2 F1 :"),
        ("rougeL_f1", "  ROUGE-L F1 :"),
    ]:
        if key in acc:
            lines.append(lr(label, f"{acc[key]:.4f}"))

    lines += [
        DASH,
        "  LATENCY",
        lr("  Mean :",   f"{lat['mean']:.1f} ms"),
        lr("  Median :", f"{lat['median']:.1f} ms"),
        lr("  P95 :",    f"{lat['p95']:.1f} ms"),
        lr("  P99 :",    f"{lat['p99']:.1f} ms"),
        lr("  Min :",    f"{lat['min']:.1f} ms"),
        lr("  Max :",    f"{lat['max']:.1f} ms"),
        DASH,
        "  MEMORY",
        lr("  Peak RSS :", f"{mem['peak_rss_mb']:.1f} MB"),
        DASH,
        "  MODEL FILES",
    ]
    for fname, size_mb in summary.get("model_files_mb", {}).items():
        lines.append(f"  {fname:<40}  {size_mb:.1f} MB")

    lines.append(SEP)
    lines.append("  PREDICTION SAMPLES")
    lines.append("")
    for r in results[:5]:
        lines += [
            f"  Input     : {r['input']}",
            f"  Target    : {r['target']}",
            f"  Predicted : {r['predicted']}",
            "",
        ]

    lines.append(SEP)
    return "\n".join(lines)


def save_run_outputs(
    run_id: str,
    run_dir: Path,
    summary: dict,
    results: list[dict],
    platform_tag: str,
    model_version: str,
    dataset_name: str,
) -> None:
    """Write summary.json, per_sample.csv, and report.txt into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with open(run_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["idx", "latency_ms", "input", "target", "predicted"]
        )
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({
                "idx":        i,
                "latency_ms": r["latency_ms"],
                "input":      r["input"],
                "target":     r["target"],
                "predicted":  r["predicted"],
            })

    report = _format_report(run_id, platform_tag, model_version, dataset_name, summary, results)
    with open(run_dir / "report.txt", "w", encoding="utf-8") as fh:
        fh.write(report)

    print(report)
    print(f"\n[save] {run_dir}")
    print(f"       summary.json  |  per_sample.csv  |  report.txt")


def _update_runs_index(results_root: Path, entry: dict) -> None:
    """Append this run's headline metrics to runs_index.json (idempotent on re-run)."""
    index_path = results_root / "runs_index.json"
    index: list[dict] = []
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
        except (json.JSONDecodeError, IOError):
            pass
    index = [e for e in index if e.get("run_id") != entry["run_id"]]
    index.append(entry)
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[index] runs_index.json  ({len(index)} run(s) total)")


# ── entry point ───────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="On-device evaluation of INT8 TFLite T5 navigation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model-dir",     default="~/nav_t5_tflite",      help="Path to quantized TFLite model directory")
    ap.add_argument("--test-csv",      default="~/nav_dataset_test.csv", help="Path to test CSV with 'input' and 'target' columns")
    ap.add_argument("--model-version", default="unknown",               help="Model version tag (e.g. b4.3.2) written into run ID")
    ap.add_argument("--platform",      default=None,                    help="Platform tag for run ID (auto-detected if omitted)")
    ap.add_argument("--results-dir",   default="~/nav_t5_results",      help="Base directory for versioned run outputs")
    ap.add_argument("--log-every",     type=int, default=20, metavar="N")
    args = ap.parse_args()

    model_dir    = Path(os.path.expanduser(args.model_dir))
    test_csv     = Path(os.path.expanduser(args.test_csv))
    results_root = Path(os.path.expanduser(args.results_dir))

    now           = datetime.datetime.now()
    timestamp_fs  = now.strftime("%Y-%m-%d_%H%M%S")    # filesystem-safe
    timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%S")  # ISO 8601 for JSON
    platform_tag  = args.platform or _detect_platform()
    model_version = args.model_version

    run_id  = f"eval_{model_version}_{platform_tag}_{timestamp_fs}"
    run_dir = results_root / run_id

    if not model_dir.exists():
        sys.exit(f"Error: model directory not found: {model_dir}")
    if not test_csv.exists():
        sys.exit(f"Error: test CSV not found: {test_csv}")

    print(f"[run]  {run_id}")

    enc, dec  = load_models(model_dir)
    tokenizer = load_tokenizer(model_dir)
    decoder_start, eos = _read_special_tokens(model_dir)

    print("[warmup] Running warm-up inference ...")
    predict("action: tap screen: home element: search_bar", tokenizer, enc, dec, decoder_start, eos)

    rows = load_dataset(test_csv)
    print(f"\n[eval] Evaluating {len(rows)} samples (greedy decode, INT8 TFLite) ...")
    results, latencies, peak_mem = run_evaluation(
        tokenizer, enc, dec, decoder_start, eos, rows, log_every=args.log_every
    )

    summary = build_summary(
        results, latencies, peak_mem, model_dir,
        run_id=run_id,
        timestamp=timestamp_iso,
        platform=platform_tag,
        model_version=model_version,
        dataset_name=test_csv.name,
    )

    save_run_outputs(run_id, run_dir, summary, results, platform_tag, model_version, test_csv.name)

    acc = summary["accuracy"]
    lat = summary["latency_ms"]
    _update_runs_index(results_root, {
        "run_id":          run_id,
        "timestamp":       timestamp_iso,
        "platform":        platform_tag,
        "model_version":   model_version,
        "dataset":         test_csv.name,
        "n_samples":       acc["n_samples"],
        "rouge1_f1":       acc.get("rouge1_f1"),
        "rougeL_f1":       acc.get("rougeL_f1"),
        "latency_mean_ms": lat["mean"],
        "latency_p95_ms":  lat["p95"],
        "peak_rss_mb":     summary["memory_mb"]["peak_rss_mb"],
        "run_dir":         run_id,
    })


if __name__ == "__main__":
    _cli()
