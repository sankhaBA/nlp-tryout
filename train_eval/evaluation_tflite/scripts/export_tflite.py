#!/usr/bin/env python3
"""
Phase 1.2 — Export fine-tuned T5 model to TFLite format.

Produces two TFLite flat-buffers:
  encoder.tflite      — encodes the input token sequence
  decoder_step.tflite — runs one decoder step, returns next-token logits

Both use SELECT_TF_OPS so that all T5 operations are supported.
The decoding loop (greedy) is implemented in Python at eval time.

Run from the project root or evaluation_tflite/ directory:
    python evaluation_tflite/scripts/export_tflite.py [--model-version b4.3.2]

Output is written to evaluation_tflite/saved_model/<version>/  (TF SavedModels)
                  and evaluation_tflite/tflite/<version>/       (float32 TFLite)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    get_model_dir,
    get_saved_model_dir,
    get_tflite_dir,
    resolve_model_version,
)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "config.json",
    "generation_config.json",
)


# ── TF module wrappers ────────────────────────────────────────────────────────

def _build_encoder_module(model):
    import tensorflow as tf

    class _EncoderModule(tf.Module):
        def __init__(self, encoder):
            super().__init__()
            self._encoder = encoder

        @tf.function(input_signature=[
            tf.TensorSpec([1, None], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec([1, None], dtype=tf.int32, name="attention_mask"),
        ])
        def call(self, input_ids, attention_mask):
            out = self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False,
            )
            return {"last_hidden_state": out.last_hidden_state}

    return _EncoderModule(model.encoder)


def _build_decoder_step_module(model):
    """One-step decoder: given all past decoder tokens + encoder states -> logits for next token."""
    import tensorflow as tf

    hidden_size = model.config.d_model
    scale = float(hidden_size) ** -0.5

    class _DecoderStepModule(tf.Module):
        def __init__(self, decoder, shared):
            super().__init__()
            self._decoder = decoder
            # TFT5 has no lm_head attribute; the shared embedding layer projects
            # hidden states -> vocab logits when called with mode="linear".
            self._shared = shared
            self._scale = scale

        @tf.function(input_signature=[
            tf.TensorSpec([1, None], dtype=tf.int32, name="decoder_input_ids"),
            tf.TensorSpec([1, None, None], dtype=tf.float32, name="encoder_hidden_states"),
        ])
        def call(self, decoder_input_ids, encoder_hidden_states):
            out = self._decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                training=False,
            )
            seq = out.last_hidden_state  # [1, dec_len, d_model]
            seq = seq * self._scale      # T5 always rescales before vocab projection
            logits = tf.matmul(seq, self._shared.embeddings, transpose_b=True)  # [1, dec_len, vocab_size]
            # Return full logits — Python (NumPy) picks the last position at eval time.
            # Avoids FlexStridedSlice (from negative-index dynamic slice) which standard
            # TFLite (on Android and tflite-runtime) cannot execute without the Flex delegate.
            return {"logits": logits}  # [1, dec_len, vocab_size]

    return _DecoderStepModule(model.decoder, model.shared)


# ── TFLite conversion ─────────────────────────────────────────────────────────

def _to_tflite(saved_model_path: Path, tflite_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_path),
        signature_keys=["serving_default"],
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Prevent tensor-list lowering that can break seq2seq dynamic shapes
    converter._experimental_lower_tensor_list_ops = False
    converter.allow_custom_ops = False

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    print(f"  written {tflite_path.name}  ({len(tflite_model) / 1e6:.1f} MB)")


# ── main export ───────────────────────────────────────────────────────────────

def export(model_version: str | None = None) -> Path:
    from transformers import TFT5ForConditionalGeneration
    import tensorflow as tf

    version = resolve_model_version(model_version)
    src   = get_model_dir(version)
    sm_dir = get_saved_model_dir(version)
    tfl_dir = get_tflite_dir(version)

    print(f"[export] version      : {version}")
    print(f"[export] source       : {src}")
    print(f"[export] saved_model  : {sm_dir}")
    print(f"[export] tflite out   : {tfl_dir}")

    # Load TF variant of the model
    print("[export] Loading TFT5ForConditionalGeneration ...")
    model = TFT5ForConditionalGeneration.from_pretrained(str(src), from_pt=True)

    # ── encoder ──────────────────────────────────────────────────────────────
    enc_sm_path = sm_dir / "encoder"
    enc_sm_path.mkdir(parents=True, exist_ok=True)
    enc_module = _build_encoder_module(model)
    tf.saved_model.save(enc_module, str(enc_sm_path), signatures={"serving_default": enc_module.call})
    print(f"[export] SavedModel -> {enc_sm_path}")

    # ── decoder step ──────────────────────────────────────────────────────────
    dec_sm_path = sm_dir / "decoder_step"
    dec_sm_path.mkdir(parents=True, exist_ok=True)
    dec_module = _build_decoder_step_module(model)
    tf.saved_model.save(dec_module, str(dec_sm_path), signatures={"serving_default": dec_module.call})
    print(f"[export] SavedModel -> {dec_sm_path}")

    # ── convert to TFLite ─────────────────────────────────────────────────────
    tfl_dir.mkdir(parents=True, exist_ok=True)
    print("\n[export] Converting to TFLite (float32, SELECT_TF_OPS) ...")
    _to_tflite(enc_sm_path,  tfl_dir / "encoder.tflite")
    _to_tflite(dec_sm_path, tfl_dir / "decoder_step.tflite")

    # ── copy tokenizer files ──────────────────────────────────────────────────
    for fname in _TOKENIZER_FILES:
        p = src / fname
        if p.exists():
            shutil.copy2(p, tfl_dir / fname)
            print(f"[export] copied  : {fname}")

    print(f"\n[export] done -> {tfl_dir}")
    return tfl_dir


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Export a fine-tuned T5 model to TFLite (encoder + decoder_step)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model-version", metavar="VER", default=None,
        help="Model version directory (e.g. b4.3.2). Defaults to latest detected.",
    )
    args = ap.parse_args()
    export(args.model_version)


if __name__ == "__main__":
    _cli()
