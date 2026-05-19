# On-Device Evaluation: T5-small Navigation Model on Android

This document describes the end-to-end process for running and evaluating the fine-tuned T5-small navigation model on a mid-range Android smartphone, producing metrics suitable for research presentation.

## Model Overview

| Property | Value |
|---|---|
| Base model | `t5-small` |
| Task | Seq2seq: natural language input → structured navigation command |
| Original size | ~242 MB (SafeTensors) |
| Training dataset | 2,540 examples (80/10/10 stratified split by action type) |
| Evaluation metrics | Exact Match Accuracy, ROUGE-1/2/L F1 |

---

## Phase 1 — Convert Model to ONNX (on PC)

T5-small cannot run directly on Android with PyTorch. It must be exported to **ONNX** format, then **quantized to INT8** (reduces size ~4×, faster on ARM CPU).

### 1.1 Install the exporter

```bash
pip install optimum[exporters] onnx onnxruntime
```

### 1.2 Export T5 to ONNX

```python
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="./nav_t5_final_v1",   # local model folder
    output="./nav_t5_onnx",
    task="text2text-generation",
    opset=13,
)
```

This produces `./nav_t5_onnx/` containing `encoder_model.onnx` and `decoder_model_merged.onnx`.

### 1.3 Quantize to INT8

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

model = ORTModelForSeq2SeqLM.from_pretrained("./nav_t5_onnx")

quantizer_enc = ORTQuantizer.from_pretrained(model.encoder)
quantizer_dec = ORTQuantizer.from_pretrained(model.decoder)

qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

quantizer_enc.quantize(save_dir="./nav_t5_quantized", quantization_config=qconfig)
quantizer_dec.quantize(save_dir="./nav_t5_quantized", quantization_config=qconfig)
```

Also copy `tokenizer.json` and `tokenizer_config.json` into `./nav_t5_quantized/`.

### 1.4 Expected sizes after conversion

| File | Original | After INT8 Quantization |
|---|---|---|
| `encoder_model.onnx` | ~90 MB | ~23 MB |
| `decoder_model.onnx` | ~160 MB | ~40 MB |
| **Total** | **~242 MB** | **~63 MB** |

---

## Phase 2 — Set Up Android Environment (Termux)

Termux is a Linux terminal emulator for Android. No rooting is required.

### 2.1 Install Termux

Install from [F-Droid](https://f-droid.org/packages/com.termux/) — **do not use the Play Store version**, it is outdated and lacks package support.

### 2.2 Install Python and dependencies

Open Termux and run:

```bash
pkg update && pkg upgrade -y
pkg install python python-pip git -y
pip install onnxruntime sentencepiece transformers tokenizers
```

> `onnxruntime` on Termux installs the ARM64 CPU build automatically, which runs the model directly on the phone processor.

---

## Phase 3 — Transfer Model Files to the Phone

### Option A — USB cable (recommended)

1. Connect phone to PC via USB.
2. Enable **File Transfer (MTP)** mode on the phone.
3. Copy the entire `nav_t5_quantized/` folder to `/storage/emulated/0/Download/`.

### Option B — ADB push

```bash
adb push ./nav_t5_quantized /data/local/tmp/nav_t5_quantized
```

### Move files to Termux home directory

```bash
cp -r /storage/emulated/0/Download/nav_t5_quantized ~/nav_t5_quantized
```

Also transfer the test split CSV (`nav_dataset_test.csv`) to `~/`.

---

## Phase 4 — On-Device Evaluation Script

Save the following script as `evaluate_on_device.py` and transfer it to the phone.

```python
import time, json, os, statistics, csv
import psutil, gc
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

MODEL_DIR   = os.path.expanduser("~/nav_t5_quantized")
TEST_CSV    = os.path.expanduser("~/nav_dataset_test.csv")
OUTPUT_JSON = os.path.expanduser("~/on_device_results.json")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def predict(input_text: str) -> str:
    inputs = tokenizer(
        "navigate: " + input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )
    outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Warm-up run to avoid cold-start bias in latency measurements
predict("action: tap screen: home element: search_bar")

# Load test data
rows = []
with open(TEST_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Evaluating on {len(rows)} test samples...")

latencies = []
results = []
process = psutil.Process()
peak_memory_mb = 0

for i, row in enumerate(rows):
    gc.collect()
    t0 = time.perf_counter()
    predicted = predict(row["input"])
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    mem_rss = process.memory_info().rss / 1e6
    peak_memory_mb = max(peak_memory_mb, mem_rss)

    exact_match = predicted.strip().lower() == row["target"].strip().lower()
    latencies.append(latency_ms)
    results.append({
        "input": row["input"],
        "target": row["target"],
        "predicted": predicted,
        "exact_match": exact_match,
        "latency_ms": round(latency_ms, 2),
    })

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(rows)} done — last latency: {latency_ms:.0f} ms")

# Compute accuracy
exact_matches = [r["exact_match"] for r in results]
accuracy = sum(exact_matches) / len(exact_matches)

# Compute ROUGE scores
try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for r in results:
        s = scorer.score(r["target"], r["predicted"])
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    rouge_metrics = {
        "rouge1_f1_mean": round(statistics.mean(r1), 4),
        "rouge2_f1_mean": round(statistics.mean(r2), 4),
        "rougeL_f1_mean": round(statistics.mean(rL), 4),
    }
except ImportError:
    rouge_metrics = {"note": "rouge_score not installed — run: pip install rouge-score"}

summary = {
    "device_evaluation": {
        "n_samples": len(results),
        "exact_match_accuracy": round(accuracy, 4),
        **rouge_metrics,
    },
    "latency_ms": {
        "mean":   round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "p95":    round(sorted(latencies)[int(0.95 * len(latencies))], 2),
        "p99":    round(sorted(latencies)[int(0.99 * len(latencies))], 2),
        "min":    round(min(latencies), 2),
        "max":    round(max(latencies), 2),
    },
    "memory_mb": {
        "peak_rss_mb": round(peak_memory_mb, 1),
    },
    "model_files_mb": {
        f: round(os.path.getsize(os.path.join(MODEL_DIR, f)) / 1e6, 1)
        for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")
    },
    "per_sample_results": results,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== ON-DEVICE EVALUATION RESULTS ===")
print(json.dumps({k: v for k, v in summary.items() if k != "per_sample_results"}, indent=2))
print(f"\nFull results saved to: {OUTPUT_JSON}")
```

### Run in Termux

```bash
cd ~
pip install rouge-score psutil
python evaluate_on_device.py
```

---

## Phase 5 — Metrics for Research Presentation

The script outputs `on_device_results.json`. Build the following tables for your supervisor.

### Table 1: Accuracy Metrics — Server vs. On-Device

| Metric | Server (GPU, T4) | On-Device (Android, INT8) |
|---|---|---|
| Exact Match Accuracy | _(from training notebook)_ | _(from script output)_ |
| ROUGE-1 F1 | _(from notebook)_ | _(from script output)_ |
| ROUGE-2 F1 | _(from notebook)_ | _(from script output)_ |
| ROUGE-L F1 | _(from notebook)_ | _(from script output)_ |

### Table 2: Mobile Inference Performance

| Metric | Value |
|---|---|
| Mean Inference Latency | X ms |
| Median Latency | X ms |
| P95 Latency | X ms |
| P99 Latency | X ms |
| Peak RAM Usage | X MB |
| Quantized Model Size on Disk | X MB |

### Table 3: Model Compression Summary

| | Original | INT8 Quantized |
|---|---|---|
| Size | ~242 MB | ~63 MB |
| Format | SafeTensors (PyTorch) | ONNX INT8 |
| Runtime | PyTorch (GPU/CPU) | ONNX Runtime ARM64 |
| Target hardware | Server / PC | Android (mobile CPU) |

---

## Full Pipeline Summary

```
[PC] Fine-tuned T5-small (.safetensors)
        │
        ▼  optimum export (Phase 1.2)
[PC] ONNX model — encoder + decoder
        │
        ▼  INT8 quantization (Phase 1.3)
[PC] Quantized ONNX (~63 MB total)
        │
        ▼  USB transfer (Phase 3)
[Android / Termux] ONNX Runtime ARM64
        │
        ▼  evaluate_on_device.py (Phase 4)
[Output] on_device_results.json
         → Exact Match, ROUGE-1/2/L, Latency (mean/p95/p99), Peak RAM
```

---

## Research Claim Template

> *"The fine-tuned T5-small navigation model achieves **X% exact match accuracy** (ROUGE-L F1: **X.XX**) on the held-out test set when deployed on a mid-range Android device using INT8 ONNX quantization, with a mean inference latency of **X ms** and a peak memory footprint of **X MB** — representing a **~75% reduction** in model size compared to the original SafeTensors checkpoint."*

Fill in the values from `on_device_results.json` and your training notebook outputs.
