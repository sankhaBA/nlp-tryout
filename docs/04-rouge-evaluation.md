# 04 — ROUGE Evaluation

> **Read this when** you need to understand how model quality is measured, the 9-step evaluation
> flow, which artifacts it produces, or how to interpret the high-ROUGE / low-exact-match gap.

Source files:
[notebooks/2_rouge_evaluation_v1.ipynb](../notebooks/2_rouge_evaluation_v1.ipynb) (standalone) ·
[notebooks/1_fine_tune_v1.ipynb](../notebooks/1_fine_tune_v1.ipynb) (same steps embedded after training)

> The TFLite/on-device evaluation computes ROUGE too, but with its own greedy-decode harness —
> see [05-tflite-and-on-device.md](05-tflite-and-on-device.md). This doc covers the **notebook**
> (PyTorch, beam-search) evaluation.

---

## Two complementary metrics

| Metric | What it measures | Why it's used |
|--------|------------------|---------------|
| **Exact match** | prediction string == reference string (case-insensitive) | strict; computed in the "Test Model" cell of nb 1 |
| **ROUGE-1/2/L/Lsum** | n-gram / longest-common-subsequence overlap (P, R, F1) | tolerant of valid paraphrases |

The task has **many correct phrasings** per command, so exact match understates quality. ROUGE is
the headline metric. Example: target "Walk forward for 44 steps." vs prediction "Walk straight
for 44 steps." — exact-match **fail**, but ROUGE ≈ high.

| ROUGE variant | Overlap unit |
|---------------|--------------|
| ROUGE-1 | unigrams (single words) |
| ROUGE-2 | bigrams (word pairs — sensitive to ordering/fluency) |
| ROUGE-L | longest common subsequence (structure, order-aware, non-contiguous) |
| ROUGE-Lsum | LCS over concatenated sentences |

All use `use_stemmer=True` (e.g. "walking" ≈ "walk").

---

## The 9-step flow

Both notebooks run the same sequence (nb 2 expects `eval_results_df` in scope with columns
`input`, `target`, `predicted`):

1. **Setup** — `%pip install -q rouge-score evaluate nltk`; create `rouge_evaluation/` output
   folder under the model dir.
2. **Per-sample scoring** — `rouge_score.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"],
   use_stemmer=True)`; store precision/recall/F1 per metric into `rouge_df`.
3. **Aggregate** — mean / median / std / min / max for P, R, F1 of every metric; print an F1
   summary (`mean ± std | min → max`).
4. **By action type** — `rouge_df.groupby(action_type)[f1_cols].mean()`; flag any action with
   ROUGE-L F1 < 0.50.
5. **Distributions** — 2×2 histogram grid of F1 per metric (with mean/median lines); report
   `%≥0.80` and `%<0.50`. Saved to `rouge_distributions.png`.
6. **By-action bar chart** — grouped bars of mean F1 per action. Saved to
   `rouge_by_action_type.png`.
7. **Low-scoring examples** — every sample with ROUGE-L F1 < 0.50 printed for manual inspection
   (catches hallucinations, e.g. spurious "In a descending direction…").
8. **Cross-validation** — recompute corpus ROUGE with HuggingFace `evaluate.load("rouge")` and
   assert agreement with `rouge_score` (delta < 0.005 → "OK").
9. **Save artifacts** — write:
   - `rouge_per_sample.csv` (P/R/F1 for all variants, per sample)
   - `rouge_by_action_type.csv` (mean F1 per action)
   - `rouge_summary.json` (aggregate stats + HF cross-check + low-score summary)
   - the two PNG charts above

All land in `<model_output_dir>/rouge_evaluation/`.

---

## Headline results (model `b4.3.2`, eval set `nav_eval_android_v1.csv`)

From the on-device/TFLite runs (same metric definitions; values match across runs):

| Metric | F1 |
|--------|-----|
| ROUGE-1 | **0.8043** |
| ROUGE-2 | **0.6590** |
| ROUGE-L | **0.7817** |
| Exact match | **0.352** (35.2%) |

Source: [tflite_eval_results.json](../tflite_eval_results.json),
[evaluation_tflite/results/runs_index.json](../evaluation_tflite/results/runs_index.json).

**Interpretation:** strong unigram overlap (0.80) and good sequence structure (0.78); the lower
ROUGE-2 (0.66) reflects re-ordered/re-worded bigrams. The exact-match/ROUGE gap is expected and
healthy given multi-phrasing targets — not a sign of a weak model.

### Known failure modes (from low-scoring examples)

Inspecting predictions in [tflite_eval_results.json](../tflite_eval_results.json) surfaces a few
real error patterns on `turn … landmark:` (no distance) inputs:

- occasional **hallucinated qualifiers**: "In a descending direction, turn right towards the
  escalator." (target: "Turn right towards the escalator.")
- rare **language bleed-through** from T5's multilingual pretraining: "In die elevator, turn
  left.", "In die rechte Richtung, …" — German fragments.
- spurious time/quantity phrases: "In 45 minutes, turn left towards Pandora."

These cluster on the sparsest input shape (a bare turn toward a landmark) and are useful targets
for additional training data.

---

## Reproducing the evaluation

- **Notebook (PyTorch, beam search):** run [notebook 1](../notebooks/1_fine_tune_v1.ipynb) end to
  end (training → test → ROUGE), or run [notebook 2](../notebooks/2_rouge_evaluation_v1.ipynb)
  after producing an `eval_results_df`.
- **On device / TFLite (greedy):** see [05-tflite-and-on-device.md](05-tflite-and-on-device.md).
  Because greedy ≠ beam search, the two paths can differ slightly; the published headline numbers
  above come from the TFLite harness on `nav_eval_android_v1.csv`.
