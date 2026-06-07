# 06 — Environment & Setup

> **Read this when** you need to install dependencies, understand the (intentionally) separate
> environments, configure `.env`, or run any stage end to end.

Source files:
[requirements.txt](../requirements.txt) ·
[.env](../.env) ·
[.gitignore](../.gitignore) ·
[main.py](../main.py)

---

## There are three distinct runtime contexts

This project is not a single `pip install -r requirements.txt` app. It spans three environments
with different dependency sets:

| Context | Where | Purpose | Dependency source |
|---------|-------|---------|--------------------|
| **A. Training / notebooks** | Google Colab (GPU) | data prep, fine-tune T5, ROUGE eval | [requirements.txt](../requirements.txt) + `%pip` in the notebook |
| **B. TFLite export & PC eval** | local PC (Windows), `venv_tflite/` | export, quantize, PC-side eval | TensorFlow stack in `venv_tflite` (no committed requirements file) |
| **C. On-device eval** | Android phone, Termux | run INT8 model on ARM | `pip install` list in the smartphone guide |
| (D. Data generation) | local PC | call Gemini API | `google-generativeai` (installed separately) |

---

## A. Training / notebook environment

[requirements.txt](../requirements.txt) pins the **training** environment (note: the file is
**UTF-16** encoded). Core packages:

| Package | Version | Role |
|---------|---------|------|
| `transformers` | 5.3.0 | T5 model + tokenizer + `Trainer` |
| `torch` | 2.10.0 | training backend |
| `datasets` | 4.8.2 | `Dataset.from_pandas`, `.map` tokenization |
| `scikit-learn` | 1.8.0 | `train_test_split` (stratified) |
| `tokenizers` | 0.22.2 | fast tokenizer |
| `sentencepiece` | 0.2.1 | T5 tokenizer backend |
| `accelerate` | 1.13.0 | Trainer acceleration |
| `safetensors` | 0.7.0 | model serialization |
| `pandas` 3.0.1 · `numpy` 2.4.3 | | data wrangling |

**Installed in the notebook, NOT in requirements.txt** (the first cell runs `%pip install`):
`rouge-score`, `evaluate`, `nltk`, and `matplotlib` (Colab-provided). If you reproduce training
locally, add these. `main.py` is a one-line smoke test that prints `transformers.__version__`.

> `google-generativeai` (data generation) is also **not** in `requirements.txt` — install it
> separately when running the generator.

---

## B. TFLite export / quantize / PC-eval environment (`venv_tflite/`)

The export and PC evaluation need **TensorFlow** (for the converter and the Flex delegate). The
repo carries a local `venv_tflite/` (gitignored). It contains the TF ecosystem — observed package
versions include `torch` 2.12.0, `numpy` 2.4.6, `h5py`, `grpcio`, `flatbuffers`, `ml_dtypes`,
`gast`, `optree`, `libclang`, `namex`, `google_pasta`, `wrapt`, `termcolor`, plus
`tokenizers`, `sentencepiece`, `regex`. To run the PC eval you also need `rouge-score` and
`psutil`.

The PC eval script references `evaluation_tflite/requirements_pc.txt` in its docstring; if that
file is absent, install the essentials manually:

```bash
python -m venv venv_tflite
venv_tflite\Scripts\activate            # Windows PowerShell
pip install tensorflow transformers tokenizers sentencepiece rouge-score psutil
```

> TensorFlow is required because the converter emits `SELECT_TF_OPS`; full TF bundles the **Flex
> delegate** that runs them on the PC. On Android the export is designed to need no Flex (see
> [05](05-tflite-and-on-device.md)).

---

## C. Android (Termux) environment

From [SMARTPHONE_TESTING_GUIDE.md](../evaluation_tflite/docs/SMARTPHONE_TESTING_GUIDE.md):

```bash
pkg update -y && pkg install python -y
pip install tflite-runtime tokenizers sentencepiece rouge-score psutil
termux-setup-storage
```

- Install **Termux from F-Droid**, not the Play Store (the Play Store build breaks `pip`).
- `tflite-runtime` / `ai_edge_litert` provides a prebuilt ARM64 interpreter (no compilation).
- If `tokenizers` fails to build, `pkg install rust -y` first.

---

## `.env` configuration

[.env](../.env) (gitignored) holds:

| Key | Used by | Purpose |
|-----|---------|---------|
| `GEMINI_API_KEY` | [dataset_generator.py](../dataset_generator/dataset_generator.py) | Gemini API auth (**required** for generation) |
| `NOTEBOOK_ENV` | notebooks | `local` or `colab` |
| `LOCAL_DATASET_PATH` | notebooks | local dataset path |
| `DRIVE_DATASET_PATH` | notebooks | Google Drive dataset path |
| `DATASET_WORKING_FILE` | notebooks | working filename |

> ⚠️ The committed `.env` contains a **real Gemini API key**. Rotate it if it has been exposed and
> never commit secrets. See the security note in [01-data-generation.md](01-data-generation.md).

---

## What's gitignored

[.gitignore](../.gitignore) excludes large/secret artifacts: trained models
(`nav_t5_model_v1`, `nav_t5_final_v1`), virtual envs (`venv`, `venv_tflite`), `.env`,
`training-history`, and the heavy TFLite pipeline outputs
(`evaluation_tflite/{saved_model,tflite,quantized,model}` and the older `evaluation/*`).
Consequence: **the `.tflite`/SavedModel files are not in the repo** — regenerate them with the
export/quantize scripts ([05](05-tflite-and-on-device.md)).

---

## End-to-end run order

```text
1. Generate data        python dataset_generator/dataset_generator.py        (env D, needs GEMINI_API_KEY)
   Validate             python dataset_generator/dataset_validator.py --file data/nav_dataset_b4.3.csv
2. Train + ROUGE        run notebooks/1_fine_tune_v1.ipynb on Colab          (env A, GPU)
   (or) ROUGE only      run notebooks/2_rouge_evaluation_v1.ipynb            (env A)
3. Export to TFLite     python evaluation_tflite/scripts/export_tflite.py   --model-version b4.3.2   (env B)
4. Quantize to INT8     python evaluation_tflite/scripts/quantize_tflite.py --model-version b4.3.2   (env B)
5. Evaluate (PC)        python evaluation_tflite/scripts/on_device_eval.py  --model-version b4.3.2   (env B)
6. Evaluate (Android)   follow evaluation_tflite/docs/SMARTPHONE_TESTING_GUIDE.md                    (env C)
```

Steps 3–4 expect the fine-tuned HF model under `evaluation_tflite/model/<version>/` (place the
notebook's saved model there). Step 5 reads `evaluation_tflite/dataset/*.csv` by default.
