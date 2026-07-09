# Smartphone Testing Guide — INT8 TFLite Navigation Model

This guide walks through testing the quantized T5 navigation model (`b4.3.2`) on a real Android device using Termux. The model has already been exported and quantized on your PC — this guide covers everything from that point forward.

---

## Table of Contents

1. [What You Have (PC Side)](#1-what-you-have-pc-side)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Prepare Files for Transfer](#3-step-1--prepare-files-for-transfer)
4. [Step 2 — Transfer Files to Phone](#4-step-2--transfer-files-to-phone)
5. [Step 3 — Set Up Termux on Android](#5-step-3--set-up-termux-on-android)
6. [Step 4 — Install Python Dependencies](#6-step-4--install-python-dependencies)
7. [Step 5 — Run the Evaluation](#7-step-5--run-the-evaluation)
8. [Step 6 — Collect Results](#8-step-6--collect-results)
9. [Understanding the Output](#9-understanding-the-output)
10. [Comparing PC vs Android Results](#10-comparing-pc-vs-android-results)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What You Have (PC Side)

The pipeline is complete through quantization. Here is what already exists:

```
evaluation_tflite/
├── quantized/b4.3.2/
│   ├── encoder_int8.tflite        ← ~35.8 MB  (goes to phone)
│   ├── decoder_step_int8.tflite   ← ~59.1 MB  (goes to phone)
│   ├── tokenizer.json             ← (goes to phone)
│   ├── tokenizer_config.json      ← (goes to phone)
│   ├── config.json                ← (goes to phone)
│   └── generation_config.json     ← (goes to phone)
├── device/
│   └── on_device_eval.py          ← standalone eval script (goes to phone)
└── results/
    └── eval_b4.3.2_windows_*/     ← PC baseline results (for comparison)

evaluation_tflite/dataset/
└── nav_eval_android_v1.csv        ← test dataset (goes to phone)
```

**Total transfer size: ~100 MB** (95 MB models + tokenizer + script + dataset)

---

## 2. Prerequisites

### On your PC
- All files listed above already exist (models, script, dataset)
- USB cable **or** a way to transfer files wirelessly (Google Drive, ADB, etc.)

### On your Android phone
- Android 8.0 (API 26) or newer
- At least **500 MB free storage** (models take ~100 MB; results/logs take a few MB)
- **Termux** installed from **F-Droid** (not the Play Store — the Play Store version is outdated and breaks pip)
  - Download F-Droid: [f-droid.org](https://f-droid.org)
  - Then install Termux from within F-Droid

> **Important:** Do NOT install Termux from the Google Play Store. That version is no longer maintained and pip does not work correctly on it.

---

## 3. Step 1 — Prepare Files for Transfer

Gather the following files from your PC into one folder for easy transfer:

| File | Source path (PC) |
|------|-----------------|
| `encoder_int8.tflite` | `evaluation_tflite/quantized/b4.3.2/` |
| `decoder_step_int8.tflite` | `evaluation_tflite/quantized/b4.3.2/` |
| `tokenizer.json` | `evaluation_tflite/quantized/b4.3.2/` |
| `tokenizer_config.json` | `evaluation_tflite/quantized/b4.3.2/` |
| `config.json` | `evaluation_tflite/quantized/b4.3.2/` |
| `generation_config.json` | `evaluation_tflite/quantized/b4.3.2/` |
| `on_device_eval.py` | `evaluation_tflite/device/` |
| `nav_eval_android_v1.csv` | `evaluation_tflite/dataset/` |

Create a staging folder on your PC (e.g., `android_transfer/`) and copy all 8 files into it.

```powershell
# PowerShell — run from the nlp-tryout project root
New-Item -ItemType Directory -Force android_transfer
Copy-Item evaluation_tflite\quantized\b4.3.2\* android_transfer\
Copy-Item evaluation_tflite\device\on_device_eval.py android_transfer\
Copy-Item evaluation_tflite\dataset\nav_eval_android_v1.csv android_transfer\
```

---

## 4. Step 2 — Transfer Files to Phone

### Option A — USB (recommended, fastest for ~100 MB)

1. Connect your phone via USB cable
2. On the phone, tap the USB notification → select **"File Transfer"** (MTP mode)
3. On your PC, open File Explorer → navigate to your phone's internal storage
4. Create the folder `Internal Storage/nav_t5_tflite/`
5. Copy all files from `android_transfer/` into that folder
6. Also copy `nav_eval_android_v1.csv` there (or anywhere accessible)

### Option B — ADB

```powershell
# From your PC — requires ADB installed (part of Android Platform Tools)
adb shell mkdir -p /sdcard/nav_t5_tflite
adb push android_transfer\encoder_int8.tflite       /sdcard/nav_t5_tflite/
adb push android_transfer\decoder_step_int8.tflite  /sdcard/nav_t5_tflite/
adb push android_transfer\tokenizer.json            /sdcard/nav_t5_tflite/
adb push android_transfer\tokenizer_config.json     /sdcard/nav_t5_tflite/
adb push android_transfer\config.json               /sdcard/nav_t5_tflite/
adb push android_transfer\generation_config.json    /sdcard/nav_t5_tflite/
adb push android_transfer\on_device_eval.py         /sdcard/nav_t5_tflite/
adb push android_transfer\nav_eval_android_v1.csv   /sdcard/nav_t5_tflite/
```

### Option C — Google Drive / cloud

Upload the `android_transfer/` folder to Google Drive, then download it on the phone using the Files app.

---

## 5. Step 3 — Set Up Termux on Android

Open Termux on your phone and run the following commands one by one:

### 5.1 Update package lists

```bash
pkg update -y && pkg upgrade -y
```

> This may take a few minutes the first time.

### 5.2 Install Python

```bash
pkg install python -y
```

Verify it works:

```bash
python --version
# Should print Python 3.11.x or similar
```

### 5.3 Upgrade pip

```bash
pip install --upgrade pip
```

### 5.4 Grant Termux storage access

Termux cannot access `/sdcard/` by default. Run this once:

```bash
termux-setup-storage
```

A permissions dialog will appear on screen — tap **Allow**. After this, `/sdcard/` is accessible from Termux at `~/storage/shared/`.

Verify:

```bash
ls ~/storage/shared/nav_t5_tflite/
# Should list: encoder_int8.tflite  decoder_step_int8.tflite  tokenizer.json ...
```

---

## 6. Step 4 — Install Python Dependencies

The evaluation script requires five packages. Install them:

```bash
pip install tflite-runtime tokenizers sentencepiece rouge-score psutil
```

| Package | Purpose |
|---------|---------|
| `tflite-runtime` | Lightweight TFLite interpreter (~5 MB, no full TF needed) |
| `tokenizers` | Fast Rust-based tokenizer for T5 |
| `sentencepiece` | T5 tokenizer backend |
| `rouge-score` | ROUGE-1/2/L accuracy metrics |
| `psutil` | Peak RAM measurement |

> Installation takes 2–5 minutes on a phone. `tflite-runtime` provides a prebuilt ARM64 wheel for Android, so no compilation is needed.

Verify the key package works:

```bash
python -c "import tflite_runtime.interpreter as tflite; print('TFLite OK')"
# Should print: TFLite OK
```

---

## 7. Step 5 — Run the Evaluation

### 7.1 Copy files from shared storage into Termux home (recommended)

Termux's home directory (`~`) is faster and more reliable than `/sdcard/`:

```bash
cp ~/storage/shared/nav_t5_tflite/encoder_int8.tflite       ~/nav_t5_tflite/
cp ~/storage/shared/nav_t5_tflite/decoder_step_int8.tflite  ~/nav_t5_tflite/
cp ~/storage/shared/nav_t5_tflite/tokenizer.json            ~/nav_t5_tflite/
cp ~/storage/shared/nav_t5_tflite/tokenizer_config.json     ~/nav_t5_tflite/
cp ~/storage/shared/nav_t5_tflite/config.json               ~/nav_t5_tflite/
cp ~/storage/shared/nav_t5_tflite/generation_config.json    ~/nav_t5_tflite/
cp ~/storage/shared/nav_t5_tflite/nav_eval_android_v1.csv   ~/
cp ~/storage/shared/nav_t5_tflite/on_device_eval.py         ~/
```

Or create symlinks if storage is tight:

```bash
mkdir -p ~/nav_t5_tflite
ln -s ~/storage/shared/nav_t5_tflite/encoder_int8.tflite      ~/nav_t5_tflite/
ln -s ~/storage/shared/nav_t5_tflite/decoder_step_int8.tflite ~/nav_t5_tflite/
ln -s ~/storage/shared/nav_t5_tflite/tokenizer.json           ~/nav_t5_tflite/
ln -s ~/storage/shared/nav_t5_tflite/config.json              ~/nav_t5_tflite/
ln -s ~/storage/shared/nav_t5_tflite/tokenizer_config.json    ~/nav_t5_tflite/
ln -s ~/storage/shared/nav_t5_tflite/generation_config.json   ~/nav_t5_tflite/
```

### 7.2 Run with default paths

If the files are in `~/nav_t5_tflite/` and the CSV is at `~/nav_dataset_test.csv`:

```bash
python ~/on_device_eval.py
```

### 7.3 Run with explicit paths (recommended for clarity)

```bash
python ~/on_device_eval.py \
  --model-dir     ~/nav_t5_tflite \
  --test-csv      ~/nav_eval_android_v1.csv \
  --model-version b4.3.2 \
  --results-dir   ~/nav_t5_results
```

### 7.4 What happens during the run

The script will print progress as it runs:

```
[run]  eval_b4.3.2_android_2026-06-07_143012
[load] encoder      : /data/data/com.termux/files/home/nav_t5_tflite/encoder_int8.tflite
[load] decoder_step : /data/data/com.termux/files/home/nav_t5_tflite/decoder_step_int8.tflite
[load] Models ready.
[warmup] Running warm-up inference ...
[eval] Evaluating 196 samples (greedy decode, INT8 TFLite) ...
  [20/196]  latency=210ms  peak_mem=380MB
  [40/196]  latency=195ms  peak_mem=385MB
  ...
  [196/196] latency=220ms  peak_mem=390MB
```

> **Expected duration:** 5–15 minutes for 196 samples depending on device.
> Keep the phone screen on (or use a wake lock app) to prevent Termux from being killed.

### 7.5 Prevent Android from killing the process

Android aggressively kills background processes. To keep the evaluation running:

```bash
# Run inside a Termux wake lock (keeps CPU active)
termux-wake-lock
python ~/on_device_eval.py --model-dir ~/nav_t5_tflite --test-csv ~/nav_eval_android_v1.csv --model-version b4.3.2
termux-wake-unlock
```

Or simply keep the Termux window in the foreground and do not switch apps.

---

## 8. Step 6 — Collect Results

Results are saved to `~/nav_t5_results/<run_id>/` on the phone:

```
~/nav_t5_results/
└── eval_b4.3.2_android_2026-06-07_143012/
    ├── summary.json     ← full metrics + all predictions (machine-readable)
    ├── per_sample.csv   ← per-prediction table (open in spreadsheet)
    └── report.txt       ← human-readable printed report
```

### Copy results back to PC

**Via ADB:**

```powershell
adb pull /sdcard/nav_t5_results  evaluation_tflite\results\
```

**Via USB file transfer:**

Copy `~/nav_t5_results/` from Termux home. Note: Termux home is at `/data/data/com.termux/files/home/` which is NOT accessible via MTP. Use ADB, or first copy to shared storage:

```bash
# On the phone, in Termux:
cp -r ~/nav_t5_results ~/storage/shared/nav_t5_results
```

Then copy from `Internal Storage/nav_t5_results/` via File Explorer on PC.

---

## 9. Understanding the Output

### report.txt (printed to console and saved)

```
====================================================================
  TFLite Navigation Model — Evaluation Report
====================================================================
  Run ID :              eval_b4.3.2_android_2026-06-07_143012
  Platform :            android
  Model :               b4.3.2
  Dataset :             nav_eval_android_v1.csv  (196 samples)
====================================================================
  ROUGE SCORES
  ROUGE-1 F1 :          0.8043
  ROUGE-2 F1 :          0.6590
  ROUGE-L F1 :          0.7817
--------------------------------------------------------------------
  LATENCY
  Mean :                210.0 ms      ← per-inference time on device
  Median :              205.0 ms
  P95 :                 280.0 ms
  P99 :                 350.0 ms
  Min :                 180.0 ms
  Max :                 400.0 ms
--------------------------------------------------------------------
  MEMORY
  Peak RSS :            390.0 MB
--------------------------------------------------------------------
  MODEL FILES
  decoder_step_int8.tflite                          59.1 MB
  encoder_int8.tflite                               35.8 MB
====================================================================
  PREDICTION SAMPLES

  Input     : tap home button
  Target    : tap the home button
  Predicted : tap the home button

  ...
```

### summary.json

Machine-readable JSON with all metrics and every prediction. Use this for programmatic comparison or plotting.

### per_sample.csv

| idx | latency_ms | input | target | predicted |
|-----|-----------|-------|--------|-----------|
| 0 | 210.3 | tap home button | tap the home button | tap the home button |
| ... | ... | ... | ... | ... |

---

## 10. Comparing PC vs Android Results

Your PC baseline run (`eval_b4.3.2_windows_2026-06-07_122520`) gives these reference numbers:

| Metric | PC (Windows TFLite) | Android (expected) |
|--------|--------------------|--------------------|
| ROUGE-1 F1 | 0.8043 | ~0.8043 (identical model) |
| ROUGE-2 F1 | 0.6590 | ~0.6590 (identical model) |
| ROUGE-L F1 | 0.7817 | ~0.7817 (identical model) |
| Mean latency | 49.0 ms | 150–300 ms (ARM, device-dependent) |
| Peak RAM | 439.4 MB | 350–450 MB |
| Encoder size | 35.8 MB | 35.8 MB (same file) |
| Decoder size | 59.1 MB | 59.1 MB (same file) |

**ROUGE scores should be identical** — the same INT8 TFLite weights and greedy decoding produce bit-identical predictions regardless of platform. If ROUGE scores differ, it indicates a tokenizer or decoding mismatch.

**Latency will be higher on Android** — expect 3–6× slower than the PC TFLite run. ARM Cortex-A cores at mobile clocks (~2–3 GHz) are slower than a laptop CPU for sequential inference. The P99 latency matters most for worst-case responsiveness.

---

## 11. Troubleshooting

### `tflite-runtime` fails to install

```bash
# Try specifying the version explicitly
pip install tflite-runtime==2.14.0

# Or install from Google's direct URL for ARM64
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp311-cp311-linux_aarch64.whl
```

If your Python version doesn't match, check what's available:
```bash
python --version      # note the version
pip index versions tflite-runtime
```

### `tokenizers` fails to install (compilation error)

```bash
pkg install rust -y   # provides the Rust compiler needed to build the package
pip install tokenizers
```

### Process killed mid-evaluation (Android OOM)

Android's low-memory killer terminates background processes. Mitigations:

1. Close all other apps before running
2. Use `termux-wake-lock` (see Step 7.5)
3. Reduce evaluation to a subset of samples:
   ```bash
   head -51 ~/nav_eval_android_v1.csv > ~/nav_eval_small.csv  # 50 samples + header
   python ~/on_device_eval.py --test-csv ~/nav_eval_small.csv --model-version b4.3.2
   ```

### `termux-setup-storage` permission dialog doesn't appear

Go to **Android Settings → Apps → Termux → Permissions → Storage** and grant it manually.

### Model file not found error

```
Error: Model file not found: /data/.../nav_t5_tflite/encoder_int8.tflite
```

Check the path is correct:
```bash
ls ~/nav_t5_tflite/
# Should show both .tflite files and tokenizer.json
```

The script looks for `encoder_int8.tflite` and `decoder_step_int8.tflite` — the filenames must match exactly.

### Results not appearing after the run

Check the results directory:
```bash
ls ~/nav_t5_results/
```

If it's empty, the run may have been killed. Run with a smaller dataset (see above) to confirm the setup works end-to-end.

### Termux can't find Python after reinstall

```bash
pkg install python -y
hash -r   # clear the shell's command cache
python --version
```

---

## Quick Reference — Minimal Command Set

Once set up, the full test from a fresh Termux session is:

```bash
# One-time setup (skip if already done)
pkg update -y && pkg install python -y
pip install tflite-runtime tokenizers sentencepiece rouge-score psutil
termux-setup-storage

# Run evaluation
termux-wake-lock
python ~/on_device_eval.py \
  --model-dir     ~/nav_t5_tflite \
  --test-csv      ~/nav_eval_android_v1.csv \
  --model-version b4.3.2 \
  --results-dir   ~/nav_t5_results
termux-wake-unlock

# Copy results to shared storage for PC retrieval
cp -r ~/nav_t5_results ~/storage/shared/nav_t5_results
```
