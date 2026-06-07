# seq-to-seq-tryout — Project Documentation

> **What this is:** A sequence-to-sequence (T5-small) NLP pipeline that converts structured
> indoor-navigation commands into natural spoken English instructions for visually-impaired
> users in a shopping mall, then ships the model to Android phones as quantized TFLite.
>
> **Audience of these docs:** humans *and* generative-AI coding assistants. Each file is
> self-contained, starts with a "Read this when…" line, and references real source files by
> relative path so an assistant can open them directly.

---

## TL;DR — what the project does

```
Structured command                          Natural instruction
─────────────────────────────────           ──────────────────────────────────────
action: turn direction: left      ──► T5 ──► "In 8 steps, turn left towards the elevator."
distance: 8 steps landmark: elevator
```

The model is trained to be a **controllable phrasing layer**: a downstream navigation
engine emits terse `action: … direction: … distance: … landmark: …` commands, and this model
turns them into fluent, minimal, screen-reader-friendly speech. It is deliberately **small**
(T5-small, ~60M params) so it can run **fully offline on a phone** via INT8 TFLite.

---

## The end-to-end pipeline

```
┌─ 1. DATA GENERATION ────────────────────────────────────────────────┐
│  scenarios.json (125 scenarios) + master_prompt.txt                  │
│        │  Google Gemini API (gemini-3-flash-preview), 50 rows/scenario│
│        ▼                                                              │
│  indoor_navigation_dataset.csv  ──► curated/batched ──► data/*.csv    │
│  (dataset_validator.py checks duplicates)        (b1 → b4.3 versions) │
└──────────────────────────────────────────────────────────────────────┘
                              │  see docs/01-data-generation.md
                              ▼
┌─ 2. TRAINING (Google Colab, GPU) ───────────────────────────────────┐
│  notebooks/1_fine_tune_v1.ipynb                                      │
│  load → dedup → stratified 80/10/10 split → tokenize ("navigate: ")   │
│  → fine-tune t5-small (HF Trainer) → save model + artifacts          │
└──────────────────────────────────────────────────────────────────────┘
                              │  see docs/03-model-training.md
                              ▼
┌─ 3. ROUGE EVALUATION ───────────────────────────────────────────────┐
│  notebooks/2_rouge_evaluation_v1.ipynb (+ embedded in nb 1)          │
│  ROUGE-1/2/L/Lsum per-sample, by-action, distributions, HF cross-chk │
└──────────────────────────────────────────────────────────────────────┘
                              │  see docs/04-rouge-evaluation.md
                              ▼
┌─ 4. TFLITE EXPORT + QUANTIZE ───────────────────────────────────────┐
│  evaluation_tflite/scripts/export_tflite.py   (PyTorch→SavedModel→TFLite)│
│  evaluation_tflite/scripts/quantize_tflite.py (float32→INT8 dynamic)  │
│  encoder.tflite + decoder_step.tflite  →  *_int8.tflite (~95 MB)     │
└──────────────────────────────────────────────────────────────────────┘
                              │  see docs/05-tflite-and-on-device.md
                              ▼
┌─ 5. ON-DEVICE EVALUATION ───────────────────────────────────────────┐
│  scripts/on_device_eval.py  (PC sim, full TF + Flex delegate)         │
│  device/on_device_eval.py   (Android/Termux, tflite-runtime, standalone)│
│  greedy decode → ROUGE + latency + memory → results/<run_id>/         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Documentation map

| Doc | Read this when you need to… |
|-----|------------------------------|
| [01-data-generation.md](01-data-generation.md) | Understand how the dataset is synthesized with Gemini, the master prompt, scenarios, and resume/validation logic. |
| [02-dataset-and-schema.md](02-dataset-and-schema.md) | Know the exact `input`/`target` grammar, action vocabulary, dataset batch history, and the eval dataset. |
| [03-model-training.md](03-model-training.md) | Understand the T5-small architecture, preprocessing, training hyperparameters, and the recorded training run. |
| [04-rouge-evaluation.md](04-rouge-evaluation.md) | Understand the ROUGE methodology, the 9-step notebook flow, metrics, and how to interpret scores. |
| [05-tflite-and-on-device.md](05-tflite-and-on-device.md) | Understand the encoder/decoder TFLite split, INT8 quantization, greedy decoding, and Android deployment. |
| [06-environment-and-setup.md](06-environment-and-setup.md) | Set up the two Python environments, configure `.env`, and run each stage end-to-end. |

---

## Repository map

```
seq-to-seq-tryout/
├── main.py                       # trivial: prints transformers.__version__ (env smoke test)
├── requirements.txt              # TRAINING/notebook environment pins (UTF-16)
├── .env                          # GEMINI_API_KEY + notebook dataset paths  (gitignored)
├── .gitignore                    # excludes models, venvs, large TFLite/SavedModel outputs
├── tflite_eval_results.json      # a snapshot of one on-device eval run (196 samples)
│
├── dataset_generator/            # ── Stage 1: data synthesis ──
│   ├── dataset_generator.py      #   Gemini-driven generation (resume-aware)
│   ├── dataset_validator.py      #   duplicate detector (CLI)
│   ├── master_prompt.txt         #   the system/instruction prompt template
│   ├── scenarios.json            #   125 navigation scenario descriptions
│   └── indoor_navigation_dataset_metadata.json  # generation checkpoint/error log
│
├── data/                         # ── Stage 1 output: training datasets ──
│   └── nav_dataset_b1.csv … nav_dataset_b4.3.csv   # versioned batches (b4.3 = production)
│
├── notebooks/                    # ── Stages 2 & 3: train + ROUGE eval (Colab) ──
│   ├── 1_fine_tune_v1.ipynb      #   data prep → train → test → embedded ROUGE
│   └── 2_rouge_evaluation_v1.ipynb  # standalone ROUGE evaluation (9 steps)
│
└── evaluation_tflite/            # ── Stages 4 & 5: mobile export + eval ──
    ├── config.py                 #   path/version resolution helpers
    ├── scripts/
    │   ├── export_tflite.py       #   Phase 1.2 — PyTorch → TFLite (float32)
    │   ├── quantize_tflite.py     #   Phase 1.3 — float32 → INT8 dynamic-range
    │   └── on_device_eval.py      #   Phase 4 — PC-side eval (full TF + Flex)
    ├── device/on_device_eval.py   #   Phase 4 — standalone Android/Termux eval
    ├── transfer/                  #   pre-staged files for phone transfer
    ├── model/b4.3.2/              #   fine-tuned HF model (safetensors + artifacts)
    ├── saved_model/  tflite/  quantized/   # intermediate + final TFLite (gitignored)
    ├── dataset/nav_eval_android_v1.csv     # 196-sample held-out eval set
    ├── results/<run_id>/          #   summary.json, per_sample.csv, report.txt
    └── docs/SMARTPHONE_TESTING_GUIDE.md    # full Android/Termux walkthrough
```

---

## Key facts at a glance

| Aspect | Value |
|--------|-------|
| Base model | `t5-small` (encoder-decoder, 6+6 layers, d_model 512, 8 heads, vocab 32,128) |
| Task framing | text-to-text with input prefix `"navigate: "` |
| Input max length | 128 tokens · Target max length | 64 tokens |
| Training | HF `Trainer`, 15 epochs max, batch 32, lr 3e-4 cosine, fp16, early-stopping patience 3 |
| Shipped model (`b4.3.2`) | trained on 1,832 / 229 / 229 (train/val/test); best val loss ≈ 0.531; stopped epoch 11 |
| Eval metric | ROUGE-1/2/L (+ exact match); ROUGE-1 F1 ≈ **0.80**, ROUGE-L F1 ≈ **0.78**, exact ≈ **35%** |
| Mobile format | two INT8 TFLite graphs: `encoder_int8.tflite` (35.8 MB) + `decoder_step_int8.tflite` (59.1 MB) |
| On-device decode | greedy (argmax), no beam search; ~50 ms/sample on PC, ~150–300 ms on Android |
| Domain | indoor mall navigation for visually-impaired users; distances always in **steps** |

> ⚠️ **Numbers are tied to model version `b4.3.2`.** Source-of-truth files:
> [model/b4.3.2/training_artifacts.json](../evaluation_tflite/model/b4.3.2/training_artifacts.json),
> [model/b4.3.2/config.json](../evaluation_tflite/model/b4.3.2/config.json),
> and the run summaries under [evaluation_tflite/results/](../evaluation_tflite/results/).

---

## A note on file links in these docs

All links are **relative to the `docs/` folder** (e.g. `../main.py`), so they resolve whether
the repo lives inside a larger workspace or is opened standalone. When in doubt, the authoritative
artifacts are the JSON/CSV files under `evaluation_tflite/model/` and `evaluation_tflite/results/`.
