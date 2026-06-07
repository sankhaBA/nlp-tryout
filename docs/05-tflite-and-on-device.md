# 05 — TFLite Export & On-Device Evaluation

> **Read this when** you need to understand how the PyTorch model becomes two INT8 TFLite graphs,
> why it's split into encoder + decoder-step, how greedy decoding is driven from Python, and how
> the model is evaluated on a PC and on an Android phone.

Source files:
[evaluation_tflite/config.py](../evaluation_tflite/config.py) ·
[scripts/export_tflite.py](../evaluation_tflite/scripts/export_tflite.py) ·
[scripts/quantize_tflite.py](../evaluation_tflite/scripts/quantize_tflite.py) ·
[scripts/on_device_eval.py](../evaluation_tflite/scripts/on_device_eval.py) (PC) ·
[device/on_device_eval.py](../evaluation_tflite/device/on_device_eval.py) (Android) ·
[docs/SMARTPHONE_TESTING_GUIDE.md](../evaluation_tflite/docs/SMARTPHONE_TESTING_GUIDE.md)

---

## Four phases

```
model/<ver>/  ──(1.2 export)──►  saved_model/<ver>/  ──►  tflite/<ver>/        (float32)
                                       │
                                       └──(1.3 quantize)──►  quantized/<ver>/   (INT8)  ──► phone
                                                                   │
                                              (Phase 4 eval) ◄──────┘
                                              scripts/on_device_eval.py  (PC, full TF + Flex)
                                              device/on_device_eval.py   (Android, tflite-runtime)
                                                       └──► results/<run_id>/{summary.json, per_sample.csv, report.txt}
```

`<ver>` is a model version dir like `b4.3.2`. [config.py](../evaluation_tflite/config.py) resolves
versions (natural-sort, defaults to the latest), maps each phase to its directory
(`MODEL_ROOT`, `SAVED_MODEL_ROOT`, `TFLITE_ROOT`, `QUANTIZED_ROOT`, `RESULTS_ROOT`), and
`default_dataset()` picks the latest CSV in `dataset/`.

---

## Why split into encoder + decoder-step?

T5 generation is autoregressive. Rather than export HF's full `generate()` (which TFLite cannot
represent cleanly), the model is exported as **two single-pass graphs**, and the **decoding loop
lives in Python/NumPy** at eval time. This keeps each TFLite graph static-shaped and portable.

[export_tflite.py](../evaluation_tflite/scripts/export_tflite.py) wraps the HF
`TFT5ForConditionalGeneration` (loaded with `from_pt=True`) in two `tf.Module`s:

### `encoder.tflite`
```
inputs : input_ids [1, T] int32, attention_mask [1, T] int32
output : last_hidden_state [1, T, 512]
```

### `decoder_step.tflite`
```
inputs : decoder_input_ids [1, S] int32, encoder_hidden_states [1, T, 512] float32
output : logits [1, S, 32128] float32        # FULL sequence, not just last position
```

Two important design choices in the decoder wrapper:

1. **T5 output scaling + tied embedding projection.** T5 has no `lm_head`; logits are produced by
   `(decoder_hidden * d_model**-0.5) @ shared_embeddings.Tᵀ`. The wrapper applies the
   `d_model ** -0.5` (= 512^-0.5) scale and matmuls against the shared embedding matrix.
2. **Return full logits, slice in Python.** The graph returns logits for *all* positions; the
   Python loop takes `logits[0, -1]`. This **avoids `FlexStridedSlice`** from a negative-index
   dynamic slice inside the graph — an op the plain TFLite runtime on Android can't execute.

Conversion uses `TFLITE_BUILTINS + SELECT_TF_OPS` (T5 needs some TF ops),
`_experimental_lower_tensor_list_ops = False` (prevents seq2seq dynamic-shape breakage), and
`allow_custom_ops = False`. Tokenizer/config files are copied alongside the `.tflite` files.

```bash
python evaluation_tflite/scripts/export_tflite.py --model-version b4.3.2
# → saved_model/b4.3.2/{encoder,decoder_step}/  and  tflite/b4.3.2/{encoder,decoder_step}.tflite
```

---

## INT8 quantization

[quantize_tflite.py](../evaluation_tflite/scripts/quantize_tflite.py) re-converts the **SavedModels**
(must exist first) with **dynamic-range INT8** quantization:
`converter.optimizations = [tf.lite.Optimize.DEFAULT]` (same `SELECT_TF_OPS` settings as export).

- **Dynamic-range** = weights quantized to int8, activations quantized at runtime. **No
  calibration dataset required** — well suited to seq2seq on mobile.
- Outputs `encoder_int8.tflite` and `decoder_step_int8.tflite` plus tokenizer/config to
  `quantized/<ver>/`. The script prints a float32→INT8 size comparison.

```bash
python evaluation_tflite/scripts/quantize_tflite.py --model-version b4.3.2
# → quantized/b4.3.2/{encoder_int8.tflite, decoder_step_int8.tflite, tokenizer.json, …}
```

**Shipped sizes (`b4.3.2`):** `encoder_int8.tflite` **35.8 MB** + `decoder_step_int8.tflite`
**59.1 MB** ≈ **95 MB** total — small enough to bundle in an Android app.

---

## Greedy decoding loop (both eval scripts)

```python
encoded = tokenizer.encode("navigate: " + text)          # SAME "navigate: " prefix as training
enc_hidden = run_encoder(input_ids, attention_mask)       # encoder.tflite
tokens = [decoder_start_token_id]                          # 0 for T5
for _ in range(max_length):                                # max_length = 64
    logits = run_decoder_step([tokens], enc_hidden)        # decoder_step.tflite
    next_token = argmax(logits[0, -1])                     # greedy (NOT beam search)
    tokens.append(next_token)
    if next_token == eos_token_id:                         # 1 for T5
        break
return tokenizer.decode(tokens without start/eos)
```

`decoder_start_token_id` and `eos_token_id` are read from the model's `config.json` (defaults
0 / 1). Greedy is used for speed and determinism on-device — this differs from the notebook's
beam search (see [03](03-model-training.md)).

---

## PC-side evaluation — `scripts/on_device_eval.py`

[scripts/on_device_eval.py](../evaluation_tflite/scripts/on_device_eval.py) simulates Android
inference on a PC. Notable details:

- **Interpreter selection:** tries **full TensorFlow first** (it bundles the **Flex delegate**
  needed for `SELECT_TF_OPS`), falls back to `tflite_runtime` only if TF is absent.
- **Dynamic tensor wiring:** resolves input tensors by name (`input_ids`, `attention_mask`,
  `decoder_input_ids`, `encoder_hidden_states`), resizing/reallocating per step to handle variable
  sequence lengths.
- **Warm-up** inference before timing.
- **Metrics:** per-sample latency (`time.perf_counter`), peak RSS via `psutil`, and ROUGE-1/2/L
  (`rouge_score`, stemmed).
- **Outputs** to `results/<run_id>/` where
  `run_id = eval_<ver>_<platform>_<YYYY-MM-DD_HHMMSS>`:
  - `summary.json` — full metrics + every prediction
  - `per_sample.csv` — `idx, latency_ms, input, target, predicted`
  - `report.txt` — human-readable report (also printed)
  - appends a headline row to `results/runs_index.json`

```bash
python evaluation_tflite/scripts/on_device_eval.py --model-version b4.3.2
# optional: --test-csv <path> --platform <tag> --results-dir <dir> --log-every N
```

### Published PC results (`b4.3.2`, 196 samples)

| | run 12:25 | run 14:56 | root snapshot |
|---|---|---|---|
| ROUGE-1 / ROUGE-L F1 | 0.8043 / 0.7817 | 0.8043 / 0.7817 | 0.8043 / 0.7817 |
| Mean latency | 49.0 ms | 51.8 ms | 55.8 ms |
| P95 latency | 81.8 ms | 80.4 ms | 98.7 ms |
| Peak RSS | 439.4 MB | 439.1 MB | 438.1 MB |

Sources: [evaluation_tflite/results/runs_index.json](../evaluation_tflite/results/runs_index.json)
and [tflite_eval_results.json](../tflite_eval_results.json) (the root snapshot is one run's
`summary.json` payload; ROUGE is identical across runs, latency varies with machine load).

---

## Android evaluation — `device/on_device_eval.py`

[device/on_device_eval.py](../evaluation_tflite/device/on_device_eval.py) is a **standalone** copy
(no `config.py` dependency) meant to be transferred to the phone and run in **Termux**. Differences
from the PC script:

- imports the interpreter from **`ai_edge_litert`** (the `tflite-runtime` successor) — no full TF,
  no Flex delegate (the export was designed so none is needed on device);
- takes explicit `--model-dir`, `--test-csv`, `--results-dir` arguments;
- writes the same three artifacts to `~/nav_t5_results/<run_id>/`.

The full walkthrough — preparing files, transfer (USB/ADB/cloud), Termux setup, dependencies
(`tflite-runtime tokenizers sentencepiece rouge-score psutil`), wake-lock, OOM mitigations, and
pulling results back — is in
[evaluation_tflite/docs/SMARTPHONE_TESTING_GUIDE.md](../evaluation_tflite/docs/SMARTPHONE_TESTING_GUIDE.md).

### PC vs Android expectations

- **ROUGE should be identical** — same INT8 weights + greedy decoding → bit-identical predictions
  regardless of platform. If ROUGE differs, suspect a tokenizer or decoding mismatch.
- **Latency is 3–6× higher on Android** — expect ~150–300 ms/sample on ARM Cortex-A cores vs
  ~50 ms on a laptop CPU. Peak RAM stays ~350–450 MB.

---

## `transfer/` folder

[evaluation_tflite/transfer/](../evaluation_tflite/transfer/) holds a pre-staged subset
(tokenizer/config + eval CSV + a copy of the device eval script) ready to push to a phone, matching
the file list in the smartphone guide.

---

## Gotchas for assistants editing this area

- The **`"navigate: "` prefix** must match training exactly in any inference path.
- Keep the **encoder/decoder I/O signatures** in sync between `export_tflite.py` and both eval
  scripts (tensor names are resolved by `_name_key`, which strips the `serving_default_` prefix).
- `decoder_step` returns **full-sequence logits** by design — don't "optimize" it to last-token
  slicing inside the graph or you reintroduce `FlexStridedSlice` and break the Android runtime.
- `saved_model/`, `tflite/`, `quantized/`, and `model/` are **gitignored** — regenerate via the
  export/quantize scripts; they are not in version control.
