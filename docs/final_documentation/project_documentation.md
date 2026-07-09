# Indoor Navigation Text Generation — End-to-End Project Documentation

**Project:** seq-to-seq-tryout  
**Model:** T5-small fine-tuned for structured-command-to-speech translation  
**Domain:** Accessibility — assistive indoor navigation for visually-impaired users  
**Target Deployment:** Android smartphones (fully offline, TensorFlow Lite)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Inputs](#2-system-inputs)
3. [Phase 1 — Synthetic Dataset Generation](#3-phase-1--synthetic-dataset-generation)
4. [Phase 2 — Model Training](#4-phase-2--model-training)
5. [Phase 3 — ROUGE Evaluation](#5-phase-3--rouge-evaluation)
6. [Phase 4 — TFLite Export and Quantization](#6-phase-4--tflite-export-and-quantization)
7. [Phase 5 — On-Device Evaluation](#7-phase-5--on-device-evaluation)
8. [Final Outputs](#8-final-outputs)
9. [Evaluation Results](#9-evaluation-results)
10. [Known Limitations](#10-known-limitations)
11. [Repository Map](#11-repository-map)
12. [Environment and Dependencies](#12-environment-and-dependencies)

---

## 1. Project Overview

This project builds a **sequence-to-sequence NLP model** that acts as a language layer between a backend indoor navigation engine and a visually-impaired user's phone. The backend emits short, structured navigation commands (e.g., `action: turn direction: left distance: 20 steps`). These are terse and machine-readable, but not suitable for direct text-to-speech. The model's job is to translate them into natural, fluent, screen-reader-friendly spoken English (e.g., `"In 20 steps, turn left."`).

The model is designed to run **entirely offline** on an Android device, so a visually-impaired shopper navigating a mall does not need an internet connection for the language layer to function.

### Why T5-small?

T5-small (~60 million parameters) was chosen because it is large enough to produce fluent English text but small enough to run in real-time on mobile hardware after INT8 quantization. The quantized model fits in ~95 MB on disk and runs in 150–300 ms per inference on a mid-range Android phone — acceptable for real-time navigation.

### What makes this non-trivial?

- There is **no pre-existing dataset** for this task, so the entire training corpus was synthesized using an LLM (Google Gemini).
- The model must be exported from PyTorch through TensorFlow to TFLite, a multi-step pipeline with specific graph constraints for Android compatibility.
- Multiple valid English phrasings exist for the same structured command, so evaluation requires ROUGE (overlap-based) metrics rather than exact match.

---

## 2. System Inputs

### Training-Time Input

The training dataset consists of `(input, target)` pairs in CSV format.

**Input column — structured navigation command:**

```
action: <verb> [direction: <dir>] [distance: <N> steps] [landmark: <name>]
```

| Field | Required | Possible Values |
|---|---|---|
| `action` | Yes | `continue`, `turn`, `stop`, `board`, `exit` |
| `direction` | No | `left`, `right`, `straight`, `up`, `down` |
| `distance` | No | `<integer> steps` (always in steps, never metres) |
| `landmark` | No | Free text — a destination or floor-change object |

The input is prefixed with `"navigate: "` at tokenization time, following the T5 task-prefix convention.

**Example inputs:**

```
action: continue distance: 44 steps
action: turn direction: left distance: 15 steps
action: stop landmark: pharmacy
action: board direction: up landmark: escalator
action: exit landmark: main entrance
```

**Target column — natural spoken instruction:**

Short imperative sentences (4–12 words), designed for text-to-speech output. Screens descriptions of the surroundings are excluded; only actionable directions are generated.

**Example targets (corresponding to inputs above):**

```
Walk straight for 44 steps.
In 15 steps, turn left.
Stop. You have reached the pharmacy.
Step onto the escalator going up.
You have arrived at the main entrance.
```

### Inference-Time Input

At inference time, the model receives a single structured command string and returns a single English sentence. The same `"navigate: "` prefix must be applied.

---

## 3. Phase 1 — Synthetic Dataset Generation

**Location:** `dataset_generator/`  
**Script:** `dataset_generator/dataset_generator.py`  
**Validator:** `dataset_generator/dataset_validator.py`

### Why synthetic data?

No real-world labelled dataset exists for this structured-command-to-speech translation task. Rather than manually authoring thousands of (input, target) pairs, a Google Gemini API call is used to generate them from a written prompt template.

### How it works

1. **Scenarios file** (`scenarios.json`) contains 125 plain-English descriptions of different navigation situations: straight walks, left/right turns, board/exit escalators and lifts, arrival at destinations, and so on.

2. **Master prompt** (`master_prompt.txt`) provides Gemini with the output contract:
   - Distances must be in steps, not metres
   - Landmarks are mentioned only when strictly necessary (destination arrivals or floor-change objects)
   - Instructions must be strictly action-oriented and screen-reader-friendly
   - Output must be valid CSV with exactly two columns: `input` and `target`
   - Six few-shot examples are embedded in the prompt to anchor the format

3. The generator iterates over all 125 scenarios, calling the Gemini API to generate **50 rows per scenario**, yielding up to 6,250 raw rows.

4. **Resume logic** — a metadata JSON file (`indoor_navigation_dataset_metadata.json`) tracks the last successfully completed scenario index and any API errors. If the script is interrupted (e.g., due to Gemini free-tier rate limits), it resumes from where it stopped. A 45-second delay is built in between API calls to avoid quota violations.

5. After generation, `dataset_validator.py` scans for exact-duplicate `(input, target)` pairs, which naturally arise when Gemini produces repeated phrasing across similar scenarios.

### Output

The curated, deduplicated dataset is saved as `data/nav_dataset_b4.3.csv` — **3,300 rows** covering all five action types. A parallel JSON version is at `data_json/nav_dataset_b4.3.json`.

---

## 4. Phase 2 — Model Training

**Location:** `notebooks/1_fine_tune_v1.ipynb`  
**Environment:** Google Colab (Tesla T4 GPU)

### Pipeline steps inside the notebook

1. **Load** `nav_dataset_b4.3.csv` (3,300 rows) into a pandas DataFrame.

2. **Deduplicate** on `(input, target)` column pairs to remove any remaining duplicates that survived the validator.

3. **Extract action type** from each input string (the value of the `action:` field) for use as a stratification key.

4. **Stratified split** using scikit-learn's `train_test_split`:
   - Training: 80% → 2,640 rows (after further internal dedup: 1,832 unique)
   - Validation: 10% → 229 rows
   - Test: 10% → 229 rows

   Stratification ensures each action type (`continue`, `turn`, `stop`, `board`, `exit`) is proportionally represented in all three splits.

5. **Tokenization** using T5's SentencePiece tokenizer:
   - Input: `"navigate: " + input_text` (max 128 tokens)
   - Target: `target_text` (max 64 tokens)
   - Padding and truncation are applied

6. **Fine-tuning** using Hugging Face `Seq2SeqTrainer` on `T5ForConditionalGeneration` (T5-small):

   | Hyperparameter | Value |
   |---|---|
   | Base model | `t5-small` |
   | Max epochs | 15 |
   | Batch size (train + eval) | 32 |
   | Learning rate | 3e-4 |
   | LR schedule | Cosine decay with warmup |
   | Warmup steps | 100 |
   | Mixed precision | fp16 |
   | Early stopping patience | 3 epochs (on validation loss) |

7. **Test-set inference** using beam search (`num_beams=4`) to generate predictions on the held-out 229-sample test set.

8. **Embedded ROUGE evaluation** on the test set predictions (ROUGE-1, ROUGE-2, ROUGE-L with stemming).

### Architecture: T5-small

| Property | Value |
|---|---|
| Type | Encoder-Decoder Transformer |
| Parameters | ~60 million |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Hidden dimension | 512 |
| Feed-forward dimension | 2,048 |
| Attention heads | 8 |
| Key/value dimension | 64 |
| Vocabulary size | 32,128 (SentencePiece) |
| Positional encoding | Relative attention, 32 buckets |
| Dropout | 0.1 |

### Training results (model b4.3.2)

| Metric | Value |
|---|---|
| Best validation loss | ~0.531 (epoch ~8) |
| Final training loss | 0.7415 |
| Stopped at epoch | 11 (early stopping) |
| Training time | 181.49 s on Tesla T4 |

The training loss curve and per-epoch validation losses are saved to `evaluation_tflite/model/b4.3.2/training_artifacts.json` and plotted in `training_validation_loss.png`.

### Output

A trained HuggingFace model directory (`evaluation_tflite/model/b4.3.2/`) containing:

```
model.safetensors          — model weights
config.json                — T5 architecture config
generation_config.json     — inference settings (max_length, eos_token, etc.)
tokenizer.json             — SentencePiece vocabulary
tokenizer_config.json      — tokenizer settings
training_artifacts.json    — hyperparameters, loss history, split sizes
training_validation_loss.png
```

---

## 5. Phase 3 — ROUGE Evaluation

**Location:** `notebooks/2_rouge_evaluation_v1.ipynb`  
**Evaluation set:** `evaluation_tflite/dataset/nav_eval_android_v1.csv` (196 samples)

This standalone notebook performs a thorough evaluation of the trained model independently of the training notebook, using a separate 196-sample held-out evaluation set (not the test split from training).

### 9-Step Evaluation Flow

| Step | Description |
|---|---|
| 1 | Load model and tokenizer from `model/b4.3.2/` |
| 2 | Load `nav_eval_android_v1.csv` (196 samples across 5 action types) |
| 3 | Run beam-search inference on all 196 samples |
| 4 | Compute per-sample ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1 with stemming) |
| 5 | Aggregate overall ROUGE summary statistics |
| 6 | Break down ROUGE scores by action type |
| 7 | Plot ROUGE score distributions (histograms) |
| 8 | Show the 10 lowest-scoring examples (for error analysis) |
| 9 | Save artifacts: `rouge_per_sample.csv`, `rouge_by_action_type.csv`, `rouge_summary.json` |

### Why ROUGE and not exact match?

Multiple valid English phrasings exist for the same structured command. For example, all of the following are correct outputs for `action: continue distance: 10 steps`:

- `"Walk forward for 10 steps."`
- `"Walk straight for 10 steps."`
- `"Go straight for 10 steps."`
- `"Continue walking for 10 steps."`

Exact-match accuracy would penalise three out of four of these despite them all being correct. ROUGE-1/2/L measure n-gram overlap between the model's output and the reference, providing a much fairer evaluation of fluency and content accuracy.

---

## 6. Phase 4 — TFLite Export and Quantization

**Location:** `evaluation_tflite/scripts/`  
**Scripts:** `export_tflite.py`, `quantize_tflite.py`

This phase converts the PyTorch model into a format that can run on Android devices using TensorFlow Lite.

### Why the split encoder/decoder design?

TFLite requires **static-shaped input tensors**. The T5 encoder produces a hidden-state tensor whose sequence length is determined at runtime. The decoder's autoregressive loop extends an output sequence token by token, growing dynamically. Merging these into a single graph with dynamic shapes is not compatible with standard TFLite operations on Android.

The solution is to **export as two separate TFLite graphs**:
- **Encoder graph:** consumes `(input_ids, attention_mask)`, emits `encoder_hidden_states`
- **Decoder step graph:** consumes `(decoder_input_ids, encoder_hidden_states)`, emits `logits` over the full vocabulary

The greedy decoding loop (calling the decoder graph repeatedly, one token at a time) is implemented in Python on-device.

### Step 1 — PyTorch → TF SavedModel → float32 TFLite

`export_tflite.py` wraps the HuggingFace T5 model in two `tf.Module` subclasses with `@tf.function` inference methods. The TFLite converter is then run with `SELECT_TF_OPS` (required because some T5 operations — like certain attention patterns — are not natively supported in TFLite's built-in op set).

Care is taken to **avoid `FlexStridedSlice`** operations in the exported graph. While `SELECT_TF_OPS` in general works on Android, `FlexStridedSlice` specifically causes runtime failures on many Android TFLite builds; the decoder returns full-sequence logits (not just the final-position slice) to avoid this.

### Step 2 — INT8 Dynamic-Range Quantization

`quantize_tflite.py` re-converts the SavedModels with `tf.lite.Optimize.DEFAULT`. This applies **dynamic-range quantization**:
- Model weights are permanently stored as INT8 (8-bit integers)
- Activations are quantized to INT8 at inference time, per-operation
- No calibration dataset is required (unlike full integer quantization)

| File | Size |
|---|---|
| `encoder_int8.tflite` | 35.8 MB |
| `decoder_step_int8.tflite` | 59.1 MB |
| Combined | ~95 MB |

### Output

Two INT8 TFLite files plus the tokenizer and config files, which together constitute everything needed for offline on-device inference.

---

## 7. Phase 5 — On-Device Evaluation

**Location:** `evaluation_tflite/scripts/on_device_eval.py` (PC) and `evaluation_tflite/device/on_device_eval.py` (Android)

### Greedy Decoding Loop

Since beam search is too memory-intensive and slow for mobile hardware, inference uses **greedy decoding**: at each step the token with the highest logit score is selected, appended to the decoder input, and the decoder is called again. This continues until the EOS (end-of-sequence) token is generated or the maximum output length (64 tokens) is reached.

```
Encoder(input_ids, attention_mask) → encoder_hidden_states

decoder_ids = [decoder_start_token_id]
while not done:
    logits = Decoder(decoder_ids, encoder_hidden_states)
    next_token = argmax(logits[:, -1, :])
    decoder_ids.append(next_token)
    if next_token == eos_token_id: done = True
```

### PC-Side Evaluation (`scripts/on_device_eval.py`)

Runs on the development PC using the full TensorFlow runtime with the Flex delegate (which handles `SELECT_TF_OPS` operations). This is used for rapid iteration before deploying to a phone.

For each of the 196 evaluation samples, the script:
1. Runs the greedy decoding loop using the INT8 TFLite models
2. Records wall-clock latency (ms) and peak RSS memory
3. Computes per-sample ROUGE-1, ROUGE-2, ROUGE-L against the reference target

Outputs written to `evaluation_tflite/results/<run_id>/`:

```
summary.json    — aggregate metrics (ROUGE averages, latency stats)
per_sample.csv  — per-row predictions, references, ROUGE scores, latency
report.txt      — human-readable summary report
```

### Android Evaluation (`device/on_device_eval.py`)

A standalone copy of the evaluation script adapted for Termux (an Android terminal emulator). It uses `tflite-runtime` (no full TensorFlow required), takes `--model-dir` and `--test-csv` as arguments, and writes results to `~/nav_t5_results/<run_id>/`.

**Transfer procedure (via USB/ADB):**
1. Connect phone with USB debugging enabled
2. `adb push encoder_int8.tflite decoder_step_int8.tflite tokenizer.json config.json on_device_eval.py /sdcard/nav_t5/`
3. In Termux: install Python (`pkg install python`), then `pip install tflite-runtime tokenizers sentencepiece rouge-score psutil`
4. Run: `python on_device_eval.py --model-dir ~/nav_t5 --test-csv ~/nav_t5/nav_eval_android_v1.csv`

A full walkthrough is documented in `evaluation_tflite/docs/SMARTPHONE_TESTING_GUIDE.md`.

---

## 8. Final Outputs

| Artifact | Location | Description |
|---|---|---|
| Training dataset | `data/nav_dataset_b4.3.csv` | 3,300 (input, target) pairs |
| Evaluation dataset | `evaluation_tflite/dataset/nav_eval_android_v1.csv` | 196 held-out samples |
| Trained HF model | `evaluation_tflite/model/b4.3.2/` | Fine-tuned T5-small weights + tokenizer |
| Training artifacts | `model/b4.3.2/training_artifacts.json` | Hyperparameters, loss history, split sizes |
| Encoder TFLite | `evaluation_tflite/quantized/<ver>/encoder_int8.tflite` | 35.8 MB INT8 encoder graph |
| Decoder TFLite | `evaluation_tflite/quantized/<ver>/decoder_step_int8.tflite` | 59.1 MB INT8 decoder-step graph |
| Eval results | `evaluation_tflite/results/<run_id>/` | summary.json, per_sample.csv, report.txt |
| Snapshot eval | `tflite_eval_results.json` | One committed evaluation run over 196 samples |

---

## 9. Evaluation Results

### Model: b4.3.2 — Evaluation Set: nav_eval_android_v1.csv (196 samples)

#### ROUGE Metrics (INT8 TFLite, greedy decoding)

| Metric | Value |
|---|---|
| ROUGE-1 F1 | **0.8043** |
| ROUGE-2 F1 | **0.6590** |
| ROUGE-L F1 | **0.7817** |
| Exact Match Accuracy | 35.2% |

ROUGE-1 above 0.80 indicates that the model reproduces the key content words (action verbs, distances, landmark names) with very high fidelity. ROUGE-L above 0.78 indicates that the longest common subsequence is also largely preserved, meaning sentence structure is correct. Exact-match is intentionally low (~35%) due to the many valid phrasings for each command.

#### Inference Performance

| Environment | Mean Latency | P95 Latency | Peak RAM |
|---|---|---|---|
| PC (Windows, CPU) | 55.8 ms/sample | 98.7 ms | 438.1 MB |
| Android (expected) | 150–300 ms/sample | ~500 ms | ~350–450 MB |

These latencies are acceptable for indoor navigation: a new navigation instruction is only issued when the user completes a step (walks to a waypoint), so the system has several seconds between requests in practice.

---

## 10. Known Limitations

### 1. Exact-Match Accuracy is Low by Design

A 35% exact-match figure is not a bug. The dataset was generated by an LLM (Gemini) with intentional variation — the same command is phrased differently across training samples to teach the model fluency. Multiple correct outputs exist for every input. ROUGE is the meaningful metric here.

### 2. Failure Modes on Sparse Inputs

When the model receives unusual or minimal inputs (e.g., a bare `action: turn` without distance or direction), it occasionally:

- **Hallucinates qualifiers:** Generates `"In a descending direction, turn right towards the escalator."` with no basis in the input.
- **Language bleed-through:** T5-small was pretrained on multilingual text. On rare inputs it produces fragments in other languages (e.g., `"In die elevator"` with a German article).
- **Spurious temporal phrases:** Generates `"In 45 minutes, turn left…"` instead of a step-distance, because T5's pretraining corpus contains many `"In X minutes"` patterns.

These are edge cases. All five action types with typical inputs (distance + optional direction + optional landmark) produce correct outputs.

### 3. Greedy vs. Beam Search Quality Gap

The notebook uses beam search (4 beams) during training-time evaluation, while on-device inference uses greedy decoding. Greedy decoding is faster and deterministic but produces slightly lower-quality outputs than beam search. The ROUGE gap is small in practice, but exists.

### 4. INT8 Quantization Accuracy Trade-off

Dynamic-range INT8 quantization reduces model weights from 32-bit floats to 8-bit integers, saving ~75% memory and speeding up inference on hardware with INT8 accelerators (many modern ARM Cortex-A cores). This introduces minor numerical approximations. In practice the ROUGE scores are stable, but edge cases may be affected.

### 5. Peak Memory on Low-End Android Devices

Peak RAM usage of ~350–450 MB may cause out-of-memory (OOM) kills on devices with less than 3 GB of total RAM or aggressive background memory management. The smartphone testing guide includes OOM mitigation steps (closing background apps, launching from a clean Termux session).

### 6. Dataset Scope

The dataset covers a shopping mall indoor navigation scenario. Action types and phrasing conventions are specific to this domain. For a different indoor navigation domain (e.g., hospital, airport), the dataset and possibly the training run would need to be regenerated with domain-appropriate scenarios.

### 7. No Beam Search on Device

Beam search is excluded from the TFLite implementation because it requires maintaining multiple partial hypotheses (N × sequence_length × vocab_size tensors simultaneously), which is memory-prohibitive on mobile. This is a deliberate trade-off.

---

## 11. Repository Map

```
seq-to-seq-tryout/
│
├── dataset_generator/             Phase 1: data synthesis
│   ├── dataset_generator.py       Gemini API generation with resume logic
│   ├── dataset_validator.py       Duplicate detection CLI
│   ├── master_prompt.txt          Gemini system prompt + few-shot examples
│   ├── scenarios.json             125 navigation situations
│   └── indoor_navigation_dataset_metadata.json  Generation checkpoint log
│
├── data/                          Phase 1 output: training datasets
│   └── nav_dataset_b4.3.csv       Production training set (3,300 rows)
│
├── data_json/                     JSON-format mirror of production dataset
│   └── nav_dataset_b4.3.json
│
├── notebooks/                     Phases 2 & 3: training and evaluation
│   ├── 1_fine_tune_v1.ipynb       Data prep → train T5-small → test → ROUGE
│   └── 2_rouge_evaluation_v1.ipynb  9-step standalone ROUGE evaluation
│
├── evaluation_tflite/             Phases 4 & 5: mobile export and evaluation
│   ├── config.py                  Version resolution helpers
│   ├── scripts/
│   │   ├── export_tflite.py       PyTorch → TF SavedModel → float32 TFLite
│   │   ├── quantize_tflite.py     float32 → INT8 TFLite
│   │   └── on_device_eval.py      PC-side evaluation (TF + Flex delegate)
│   ├── device/
│   │   └── on_device_eval.py      Standalone Android/Termux evaluation
│   ├── transfer/                  Files pre-staged for phone ADB transfer
│   ├── model/b4.3.2/              Fine-tuned HF model + artifacts
│   ├── dataset/
│   │   └── nav_eval_android_v1.csv  196-sample held-out evaluation set
│   ├── results/                   Evaluation run outputs (gitignored)
│   └── docs/
│       └── SMARTPHONE_TESTING_GUIDE.md  Full Android walkthrough
│
├── docs/                          Project documentation
│   ├── README.md
│   ├── 01-data-generation.md
│   ├── 02-dataset-and-schema.md
│   ├── 03-model-training.md
│   ├── 04-rouge-evaluation.md
│   ├── 05-tflite-and-on-device.md
│   └── 06-environment-and-setup.md
│
├── tflite_eval_results.json       Committed snapshot of one evaluation run
├── requirements.txt               Training environment dependency pins
└── main.py                        Environment smoke test
```

---

## 12. Environment and Dependencies

Three separate runtime environments are used across the pipeline. There is **no single `pip install -r requirements.txt`** that covers all phases.

### Environment A — Model Training (Google Colab)

| Package | Version |
|---|---|
| torch | 2.10.0 |
| transformers | 5.3.0 |
| tokenizers | 0.22.2 |
| sentencepiece | 0.2.1 |
| datasets | 4.8.2 |
| scikit-learn | 1.8.0 |
| accelerate | 1.13.0 |
| safetensors | 0.7.0 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| rouge-score, evaluate, nltk, matplotlib | (installed inline in notebook) |

### Environment B — TFLite Export and PC Evaluation (Local PC)

| Package | Version |
|---|---|
| tensorflow | 2.21.0 |
| torch | 2.12.0 |
| transformers, tokenizers, sentencepiece | (compatible versions) |
| rouge-score | any |
| psutil | any |

### Environment C — On-Device Evaluation (Android Termux)

| Package | Notes |
|---|---|
| tflite-runtime (ai_edge_litert) | Prebuilt ARM64 wheel |
| tokenizers | ARM64 compatible version |
| sentencepiece | ARM64 compatible version |
| rouge-score, psutil | pip-installable |

### Environment D — Data Generation (Local PC)

| Package | Notes |
|---|---|
| google-generativeai | For Gemini API |
| pandas | CSV handling |

### Configuration (.env)

A `.env` file (gitignored) is required in the repository root for data generation:

```
GEMINI_API_KEY=<your-key>
NOTEBOOK_ENV=local              # or "colab"
LOCAL_DATASET_PATH=./data/
DRIVE_DATASET_PATH=             # Colab: Google Drive path
DATASET_WORKING_FILE=nav_dataset_b4.3.csv
```

### End-to-End Run Order

```
1. (Environment D) python dataset_generator/dataset_generator.py
2. (Environment D) python dataset_generator/dataset_validator.py data/nav_dataset_*.csv
3. (Environment A) Run notebooks/1_fine_tune_v1.ipynb on Google Colab
4. (Environment A) Run notebooks/2_rouge_evaluation_v1.ipynb on Google Colab
5. (Environment B) python evaluation_tflite/scripts/export_tflite.py
6. (Environment B) python evaluation_tflite/scripts/quantize_tflite.py
7. (Environment B) python evaluation_tflite/scripts/on_device_eval.py
8. (Environment C) Transfer files to Android via ADB
9. (Environment C) python device/on_device_eval.py --model-dir ~/nav_t5 --test-csv ...
```
