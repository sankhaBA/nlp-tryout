#!/usr/bin/env python3
"""
Phase 4 — PC-side evaluation of the INT8 TFLite T5 navigation model.

Simulates Android inference using the TFLite Python interpreter.
Uses greedy decoding (no beam search) to match the Android runtime behaviour.

Run from the project root or evaluation_tflite/ directory:
    python evaluation_tflite/scripts/on_device_eval.py [--model-version b4.3.2]

Install PC dependencies first:
    pip install -r evaluation_tflite/requirements_pc.txt
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import default_dataset, get_quantized_dir, resolve_model_version


# ── TFLite interpreter loader ─────────────────────────────────────────────────

def _get_tflite():
    """Return (tflite_module, use_flex) — use_flex=True when full TF is used.

    Full TensorFlow is tried first because it bundles the Flex delegate, which
    is required to run models that contain SELECT_TF_OPS (e.g. FlexStridedSlice).
    tflite-runtime (PyPI) is a stripped build with no Flex support and is only
    used as a last resort for models that have no Flex ops at all.
    """
    try:
        import tensorflow as tf        # has Flex delegate — preferred on PC
        return tf.lite, True
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite   # no Flex — Android/edge fallback
        return tflite, False
    except ImportError:
        sys.exit(
            "Neither tensorflow nor tflite-runtime is installed.\n"
            "Install one of:\n"
            "  pip install tensorflow\n"
            "  pip install tflite-runtime"
        )


# ── model loading ─────────────────────────────────────────────────────────────

def _make_interpreter(tflite, model_path: str, use_flex: bool):
    """Create a TFLite Interpreter, enabling the Flex delegate when available."""
    if use_flex:
        # Full TF includes the Flex delegate; SELECT_TF_OPS enables it so that
        # models containing ops like FlexStridedSlice can run on PC.
        try:
            resolver = tflite.experimental.OpResolverType.SELECT_TF_OPS
            return tflite.Interpreter(model_path=model_path,
                                      experimental_op_resolver_type=resolver)
        except AttributeError:
            pass  # older TF build — fall through to plain interpreter
    return tflite.Interpreter(model_path=model_path)


def load_models(model_dir: Path):
    tflite, use_flex = _get_tflite()

    enc_path = model_dir / "encoder_int8.tflite"
    dec_path = model_dir / "decoder_step_int8.tflite"

    for p in (enc_path, dec_path):
        if not p.exists():
            sys.exit(f"Model file not found: {p}\nRun quantize_tflite.py first.")

    print(f"[load] encoder      : {enc_path}")
    print(f"[load] decoder_step : {dec_path}")

    enc = _make_interpreter(tflite, str(enc_path), use_flex)
    dec = _make_interpreter(tflite, str(dec_path), use_flex)
    enc.allocate_tensors()
    dec.allocate_tensors()

    print("[load] Models ready.")
    return enc, dec


def load_tokenizer(model_dir: Path):
    from tokenizers import Tokenizer
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"tokenizer.json not found in {model_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    return tokenizer


def _read_special_tokens(model_dir: Path) -> tuple[int, int]:
    """Return (decoder_start_token_id, eos_token_id) from config.json."""
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        decoder_start = cfg.get("decoder_start_token_id", 0)
        eos = cfg.get("eos_token_id", 1)
        return int(decoder_start), int(eos)
    return 0, 1  # T5 defaults


# ── inference ─────────────────────────────────────────────────────────────────

def _name_key(full: str) -> str:
    """Strip TF serving prefix/port from tensor names like 'serving_default_input_ids:0'."""
    return full.split(":")[0].split("/")[-1].removeprefix("serving_default_")


def _run_encoder(enc_interp, input_ids, attention_mask):
    import numpy as np

    # Resolve name->index BEFORE resize so we resize the correct slots.
    in_details = enc_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    def _idx(key, fallback_pos):
        return name_to_idx.get(key, in_details[fallback_pos]["index"])

    enc_interp.resize_tensor_input(_idx("input_ids",      0), input_ids.shape)
    enc_interp.resize_tensor_input(_idx("attention_mask", 1), attention_mask.shape)
    enc_interp.allocate_tensors()

    # Re-fetch details after reallocation (indices are stable but good practice).
    in_details = enc_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    enc_interp.set_tensor(_idx("input_ids",      0), input_ids)
    enc_interp.set_tensor(_idx("attention_mask", 1), attention_mask)

    enc_interp.invoke()
    return enc_interp.get_tensor(enc_interp.get_output_details()[0]["index"])


def _run_decoder_step(dec_interp, decoder_input_ids, encoder_hidden_states):
    import numpy as np

    in_details = dec_interp.get_input_details()
    name_to_idx = {_name_key(d["name"]): d["index"] for d in in_details}

    def _idx(key, fallback_pos):
        return name_to_idx.get(key, in_details[fallback_pos]["index"])

    dec_interp.resize_tensor_input(_idx("decoder_input_ids",     0), decoder_input_ids.shape)
    dec_interp.resize_tensor_input(_idx("encoder_hidden_states", 1), encoder_hidden_states.shape)
    dec_interp.allocate_tensors()

    in_details = dec_interp.get_input_details()
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

    encoded = tokenizer.encode("navigate: " + text)
    input_ids      = np.array([encoded.ids],             dtype=np.int32)
    attention_mask = np.array([encoded.attention_mask],  dtype=np.int32)

    encoder_hidden_states = _run_encoder(enc_interp, input_ids, attention_mask)

    tokens = [decoder_start_token_id]
    for _ in range(max_length):
        dec_input = np.array([tokens], dtype=np.int32)
        logits = _run_decoder_step(dec_interp, dec_input, encoder_hidden_states)
        # logits shape: [1, dec_len, vocab_size] — pick the last position
        next_token = int(np.argmax(logits[0, -1]))
        tokens.append(next_token)
        if next_token == eos_token_id:
            break

    # Decode, stripping the leading decoder_start token and any EOS
    skip = {decoder_start_token_id, eos_token_id}
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
    rss_mb = _build_memory_tracker()
    results: list[dict] = []
    latencies: list[float] = []
    peak_mem = 0.0

    for i, row in enumerate(rows):
        gc.collect()
        t0 = time.perf_counter()
        predicted = predict(
            row["input"], tokenizer, enc_interp, dec_interp,
            decoder_start_token_id, eos_token_id,
        )
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        peak_mem = max(peak_mem, rss_mb())
        latencies.append(latency_ms)

        results.append({
            "input":        row["input"],
            "target":       row["target"],
            "predicted":    predicted,
            "exact_match":  predicted.strip().lower() == row["target"].strip().lower(),
            "latency_ms":   round(latency_ms, 2),
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
            "n_samples":   n,
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
            for f in sorted(model_dir.glob("*.tflite"))
        },
        "per_sample_results": results,
    }


# ── entry point ───────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="PC-side evaluation of INT8 TFLite T5 navigation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model-version", metavar="VER", default=None)
    ap.add_argument("--test-csv", default=None, help="Path to test CSV. Defaults to latest in dataset/.")
    ap.add_argument("--output-json", default="tflite_eval_results.json")
    ap.add_argument("--log-every", type=int, default=20, metavar="N")
    args = ap.parse_args()

    version   = resolve_model_version(args.model_version)
    model_dir = get_quantized_dir(version)
    test_csv  = Path(args.test_csv) if args.test_csv else default_dataset()
    out_json  = Path(args.output_json)

    if not model_dir.exists():
        sys.exit(f"Quantized model dir not found: {model_dir}\nRun quantize_tflite.py first.")
    if not test_csv.exists():
        sys.exit(f"Test CSV not found: {test_csv}")

    enc, dec = load_models(model_dir)
    tokenizer = load_tokenizer(model_dir)
    decoder_start, eos = _read_special_tokens(model_dir)

    print("[warmup] Running warm-up inference ...")
    predict("action: tap screen: home element: search_bar", tokenizer, enc, dec, decoder_start, eos)

    rows = load_dataset(test_csv)
    print(f"\n[eval] Evaluating {len(rows)} samples (greedy decode, INT8 TFLite) ...")
    results, latencies, peak_mem = run_evaluation(
        tokenizer, enc, dec, decoder_start, eos, rows, log_every=args.log_every
    )

    summary = build_summary(results, latencies, peak_mem, model_dir)

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    top = {k: v for k, v in summary.items() if k != "per_sample_results"}
    print("\n=== TFLITE EVALUATION RESULTS ===")
    print(json.dumps(top, indent=2))
    print(f"\nFull results -> {out_json}")


if __name__ == "__main__":
    _cli()
