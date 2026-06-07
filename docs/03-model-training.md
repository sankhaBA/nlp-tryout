# 03 — Model Training

> **Read this when** you need the model architecture, the data-preprocessing/tokenization step,
> the exact training hyperparameters, or the recorded results of the shipped `b4.3.2` run.

Source files:
[notebooks/1_fine_tune_v1.ipynb](../notebooks/1_fine_tune_v1.ipynb) (the training notebook) ·
[evaluation_tflite/model/b4.3.2/config.json](../evaluation_tflite/model/b4.3.2/config.json) (architecture) ·
[evaluation_tflite/model/b4.3.2/training_artifacts.json](../evaluation_tflite/model/b4.3.2/training_artifacts.json) (run record)

---

## Model: `t5-small`

A standard HuggingFace **T5-small** encoder-decoder transformer, fine-tuned as-is (no
architecture changes). From [config.json](../evaluation_tflite/model/b4.3.2/config.json):

| Property | Value |
|----------|-------|
| `model_type` | `t5` (`T5ForConditionalGeneration`) |
| `d_model` | 512 |
| `d_ff` | 2048 |
| `d_kv` | 64 |
| `num_layers` / `num_decoder_layers` | **6** / **6** |
| `num_heads` | 8 |
| `vocab_size` | 32,128 (SentencePiece) |
| `dropout_rate` | 0.1 |
| activation | ReLU (`feed_forward_proj: relu`, not gated) |
| `n_positions` | 512 |
| relative attention | 32 buckets, max distance 128 |
| `tie_word_embeddings` | true (shared embedding also projects to vocab logits) |
| special tokens | `decoder_start_token_id = 0`, `eos_token_id = 1`, `pad_token_id = 0` |

Parameter count ≈ 60M; the saved `model.safetensors` is ~242 MB (fp32). Because embeddings are
tied, there is **no separate `lm_head` weight** — the shared embedding matrix is reused for the
output projection (relevant for the TFLite export, see [05](05-tflite-and-on-device.md)).

---

## Where it runs

The notebook is written for **Google Colab** with a GPU (the recorded run used a **Tesla T4,
15.6 GB**). It mounts Google Drive, copies the dataset from a Drive path, and writes the trained
model + artifacts back to Drive. Adapting to local training means replacing the Drive
copy/`drive.mount` cells with local paths.

---

## Pipeline (notebook cells, in order)

[1_fine_tune_v1.ipynb](../notebooks/1_fine_tune_v1.ipynb):

1. **Install deps** — `%pip install -q transformers datasets scikit-learn rouge-score evaluate nltk`.
2. **GPU check** — report device / GPU name / VRAM.
3. **Mount Drive** and **copy dataset** → local `nav_dataset.csv`
   (source: `…/nlp_tryout/Dataset/nav_dataset_b4.3.csv`).
4. **Load & split** ([pandas] + [scikit-learn]):
   - `pd.read_csv` → **`drop_duplicates(subset=["input","target"])`** (exact-dup removal).
   - Extract `action_type` via regex `action: (\w+)` for stratification.
   - **Stratified 80/10/10 split** with `train_test_split(..., random_state=42, stratify=action_type)`:
     first 80/20, then split the 20 into 10/10. Stratifying by action keeps each action's
     proportion stable across train/val/test.
   - Build HF `Dataset` objects from the three frames (helper `action_type` column dropped first).
5. **Tokenize** with `AutoTokenizer.from_pretrained("t5-small")`:
   ```python
   def preprocess(examples):
       model_inputs = tokenizer(["navigate: " + x for x in examples["input"]],
                                max_length=128, truncation=True)
       labels = tokenizer(text_target=examples["target"], max_length=64, truncation=True)
       model_inputs["labels"] = labels["input_ids"]
       return model_inputs
   ```
   - **Input prefix `"navigate: "`** is prepended to every command (T5 task-prefix convention).
     The on-device code applies the *same* prefix — keep them in sync.
   - Input truncated to **128** tokens, target to **64**.
6. **Model + Trainer** — `T5ForConditionalGeneration.from_pretrained("t5-small")`,
   `DataCollatorForSeq2Seq`, `TrainingArguments(**training_config)`, and an
   `EarlyStoppingCallback(early_stopping_patience=3)`.
7. **Train** — `trainer.train()`, then save `training_artifacts.json` (config, per-epoch losses,
   `train_result.metrics`).
8. **Training curves** — plot train vs validation loss, save `training_validation_loss.png`,
   append `curve_points` to the artifacts. Includes a simple overfit/underfit heuristic
   (generalization gap > 0.5 → overfit warning).
9. **Save model** — `trainer.save_model(...)` + `tokenizer.save_pretrained(...)` to Drive.
10. **Test** — beam-search generation on the held-out test split, compute **exact-match** accuracy
    and a per-action-type breakdown + full mismatch log.
11. **Embedded ROUGE** — the same 9-step ROUGE analysis as
    [notebook 2](../notebooks/2_rouge_evaluation_v1.ipynb); see [04-rouge-evaluation.md](04-rouge-evaluation.md).

---

## Training hyperparameters

`training_config` in the notebook (mirrored exactly in
[training_artifacts.json](../evaluation_tflite/model/b4.3.2/training_artifacts.json) →
`model_configuration.training_arguments`):

```python
{
  "output_dir": "./nav_t5_model_v1",
  "num_train_epochs": 15,            # upper bound; early stopping usually triggers first
  "per_device_train_batch_size": 32,
  "per_device_eval_batch_size": 32,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "logging_steps": 10,
  "eval_strategy": "epoch",          # evaluate every epoch
  "save_strategy": "epoch",          # checkpoint every epoch
  "load_best_model_at_end": True,    # restore the best-val-loss checkpoint
  "fp16": True,                      # mixed precision
  "remove_unused_columns": False,
  "report_to": "none",
  "lr_scheduler_type": "cosine",
  "learning_rate": 3e-4,
}
# + EarlyStoppingCallback(early_stopping_patience=3)
```

Optimizer is HF's default **AdamW**; loss is the standard seq2seq cross-entropy.

---

## Recorded run — shipped model `b4.3.2`

From [training_artifacts.json](../evaluation_tflite/model/b4.3.2/training_artifacts.json):

| Metric | Value |
|--------|-------|
| Dataset split sizes | train **1,832** · val **229** · test **229** |
| `train_runtime` | 181.49 s |
| `train_samples_per_second` | 151.4 |
| `train_steps_per_second` | 4.79 |
| `train_loss` (final) | **0.7415** |
| Epoch reached | **11.0** (early-stopped before the 15-epoch cap) |

Validation-loss trajectory (per recorded eval points): 0.986 → 0.690 → 0.605 → 0.583 → 0.569 →
0.556 → 0.535 → **0.531 (best)** → 0.532 → 0.537 → 0.532. `load_best_model_at_end=True` restores
the ~0.531 checkpoint. The curve shows healthy convergence with a small train/val gap (no severe
overfitting).

> ### ⚠️ Two different runs — don't confuse them
> The **notebook as last executed** records a *different* run than the shipped model:
> train **2,032** / val **254** / test **254**, `train_loss` **0.6995**, epoch **13.0**, runtime
> 184.29 s (visible in the notebook's cell outputs). The **deployed `b4.3.2`** artifacts above
> (1,832/229/229, loss 0.7415, epoch 11) are the source of truth for the model that was actually
> exported to TFLite. The split sizes differ because dedup count depends on the dataset snapshot
> at run time. When citing "the model," use the `b4.3.2` artifacts.

---

## Inference at test time (notebook)

```python
def generate_from_text(input_text: str) -> str:
    full_input = "navigate: " + input_text
    inputs = tokenizer(full_input, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=64, min_new_tokens=4,
                             num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
```

Note the notebook uses **beam search (`num_beams=4`)**. The **on-device** path uses **greedy**
decoding instead (faster, deterministic on mobile) — a deliberate difference documented in
[05-tflite-and-on-device.md](05-tflite-and-on-device.md).

---

## Saved artifacts (per trained model)

Written next to the model (e.g. [evaluation_tflite/model/b4.3.2/](../evaluation_tflite/model/b4.3.2/)):

- `model.safetensors`, `config.json`, `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`, `spiece.model`, `special_tokens_map.json`
- `training_artifacts.json` — config + per-epoch losses + summary + curve points
- `training_validation_loss.png` — loss curve
- (during ROUGE eval) a `rouge_evaluation/` subfolder — see [04](04-rouge-evaluation.md)
