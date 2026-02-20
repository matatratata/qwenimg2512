#!/usr/bin/env python3
"""Offline LoRA fusion script for Qwen-Image-Edit-2509 + TeleStyle.

Loads the full bf16 model across BOTH 3090s (device_map="auto"),
fuses the TeleStyle style LoRA and Lightning speedup LoRA into the
base weights, and saves a new self-contained model directory.

After running this once, point `edit_2509_base_model_dir` in the app
to the fused output directory and leave the LoRA slots empty.

Usage:
    python fuse_telestyle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path.home() / "AI" / "Models"

BASE_MODEL = str(MODELS_DIR / "Qwen-Image-Edit-2509")
LORA_TELESTYLE = str(MODELS_DIR / "TeleStyle" / "weights" / "diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")
LORA_LIGHTNING = str(MODELS_DIR / "TeleStyle" / "weights" / "diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")

OUTPUT_DIR = str(MODELS_DIR / "Qwen-Image-Edit-2509-TeleStyle-Fused")

# ── LoRA scales (adjust if needed) ────────────────────────────────────────
TELESTYLE_SCALE = 1.0
LIGHTNING_SCALE = 1.0


def main() -> None:
    from diffusers import QwenImageEditPlusPipeline

    # Validate inputs
    for label, path in [("Base model", BASE_MODEL), ("TeleStyle LoRA", LORA_TELESTYLE), ("Lightning LoRA", LORA_LIGHTNING)]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found at: {path}")
            sys.exit(1)

    if Path(OUTPUT_DIR).exists():
        print(f"WARNING: Output directory already exists: {OUTPUT_DIR}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"Loading unquantized model from:\n  {BASE_MODEL}")
    print("  (loading to CPU — fusion is just matrix math, no GPU needed)")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
    )
    print("  Model loaded on CPU.")

    print(f"\nLoading TeleStyle LoRA:\n  {LORA_TELESTYLE}")
    pipe.load_lora_weights(LORA_TELESTYLE, adapter_name="telestyle")

    print(f"\nLoading Lightning LoRA:\n  {LORA_LIGHTNING}")
    pipe.load_lora_weights(LORA_LIGHTNING, adapter_name="lightning")

    print(f"\nSetting adapter scales: telestyle={TELESTYLE_SCALE}, lightning={LIGHTNING_SCALE}")
    pipe.set_adapters(["telestyle", "lightning"], adapter_weights=[TELESTYLE_SCALE, LIGHTNING_SCALE])

    print("\nFusing LoRAs into base weights...")
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    print("  LoRAs fused and adapters unloaded.")

    print(f"\nSaving fused model to:\n  {OUTPUT_DIR}")
    pipe.save_pretrained(OUTPUT_DIR)
    print("\n✅ Done! Update your app config:")
    print(f'  edit_2509_base_model_dir → {OUTPUT_DIR}')
    print("  LoRA 1 / LoRA 2 → leave empty (already baked in)")


if __name__ == "__main__":
    main()
