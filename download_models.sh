#!/bin/bash
set -e

# Base directory for models
BASE_DIR="$HOME/AI/Models"

# Enable HF Transfer for faster downloads (requires huggingface_hub[hf_transfer])
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

echo "Downloading Qwen Image models to $BASE_DIR..."
echo "Note: Fast downloads enabled (HF_HUB_ENABLE_HF_TRANSFER=1)."
echo "Ensure you have installed: pip install 'huggingface_hub[hf_transfer]'"
echo ""

mkdir -p "$BASE_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Please install 'huggingface_hub[cli]'."
    echo "You can run: uv pip install -U \"huggingface_hub[cli,hf_transfer]\""
    exit 1
fi

# Download a specific file
download_file() {
    local repo_id="$1"
    local filename="$2"
    local local_dir="$3"
    
    echo "Downloading $filename from $repo_id to $local_dir..."
    huggingface-cli download "$repo_id" "$filename" --local-dir "$local_dir" --local-dir-use-symlinks False
}

# Download a repo with include pattern
download_repo() {
    local repo_id="$1"
    local local_dir="$2"
    shift 2
    local include_patterns="$@"
    
    echo "Downloading $repo_id to $local_dir..."
    # shellcheck disable=SC2086
    huggingface-cli download "$repo_id" --local-dir "$local_dir" --local-dir-use-symlinks False --include $include_patterns
}

# 1. Qwen-Image-2512 GGUF
download_file \
    "unsloth/Qwen-Image-2512-GGUF" \
    "qwen-image-2512-Q8_0.gguf" \
    "$BASE_DIR/Qwen-Image-2512-GGUF"

# 2. Qwen-Image-2512 Base
download_repo \
    "Qwen/Qwen-Image-2512" \
    "$BASE_DIR/Qwen-Image-2512" \
    "scheduler/*" "tokenizer/*" "text_encoder/*" "vae/*" "model_index.json"

# 3. Qwen-Image-Edit-2511 GGUF
download_file \
    "unsloth/Qwen-Image-Edit-2511-GGUF" \
    "qwen-image-edit-2511-Q8_0.gguf" \
    "$BASE_DIR/Qwen-Image-Edit-2511-GGUF"

# 4. Qwen-Image-Edit-2511 Base
download_repo \
    "Qwen/Qwen-Image-Edit-2511" \
    "$BASE_DIR/Qwen-Image-Edit-2511" \
    "scheduler/*" "tokenizer/*" "text_encoder/*" "vae/*" "model_index.json" "processor/*" "transformer/config.json"

# 5. VL Model for Captioning
download_file \
    "Qwen/Qwen2.5-VL-7B-Instruct-GGUF" \
    "Qwen2.5-VL-7B-Instruct-UD-Q8_K_XL.gguf" \
    "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF"

# download_file \
#     "Qwen/Qwen2.5-VL-7B-Instruct-GGUF" \
#     "mmproj-model-f16.gguf" \
#     "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF"

# Symlink mmproj for config compatibility if needed (config expects mmproj-BF16.gguf)
if [ -f "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf" ]; then
    ln -sf "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf" "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-BF16.gguf"
fi

# VAE for VL model (reuse Image VAE)
# We download it to Qwen2.5-VL-7B-Instruct-GGUF dir as qwen_image_vae.safetensors
# Use a temp dir to avoid file collision or nesting issues during download
mkdir -p "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/vae_temp"
download_file \
    "Qwen/Qwen-Image-2512" \
    "vae/diffusion_pytorch_model.safetensors" \
    "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/vae_temp"

mv "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/vae_temp/vae/diffusion_pytorch_model.safetensors" "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/qwen_image_vae.safetensors"
rm -rf "$BASE_DIR/Qwen2.5-VL-7B-Instruct-GGUF/vae_temp"

# 6. ControlNet Union
# download_file \
#     "Qwen/Qwen-Image-2512-Fun-Controlnet-Union" \
#     "diffusion_pytorch_model.safetensors" \
#     "$BASE_DIR/Qwen-Image-2512-Fun-Controlnet-Union"

# Rename
# mv "$BASE_DIR/Qwen-Image-2512-Fun-Controlnet-Union/diffusion_pytorch_model.safetensors" "$BASE_DIR/Qwen-Image-2512-Fun-Controlnet-Union/Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors"

# 7. Qwen-Image-Edit-2509 GGUF (kept for reference, no longer primary)
# download_file \
#     "unsloth/Qwen-Image-Edit-2509-GGUF" \
#     "qwen-image-edit-2509-Q8_0.gguf" \
#     "$BASE_DIR/Qwen-Image-Edit-2509-GGUF"

# 8. Qwen-Image-Edit-2509 Base + full transformer safetensors (for BnB 8-bit)
download_repo \
    "Qwen/Qwen-Image-Edit-2509" \
    "$BASE_DIR/Qwen-Image-Edit-2509" \
    "scheduler/*" "tokenizer/*" "text_encoder/*" "vae/*" "model_index.json" "transformer/*" "processor/*"

# 9. TeleStyle LoRA & Speedup
download_file \
    "Tele-AI/TeleStyle" \
    "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors" \
    "$BASE_DIR/TeleStyle"

download_file \
    "Tele-AI/TeleStyle" \
    "weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors" \
    "$BASE_DIR/TeleStyle"

# 10. SeedVR2 GGUF
mkdir -p "$BASE_DIR/SeedVR2"
download_file \
    "cmeka/SeedVR2-GGUF" \
    "seedvr2_ema_7b-Q8_0.gguf" \
    "$BASE_DIR/SeedVR2"

# 11. SeedVR2 VAE
download_file \
    "ByteDance-Seed/SeedVR2-7B" \
    "ema_vae.pth" \
    "$BASE_DIR/SeedVR2"

# 12. SeedVR2 Inference CLI (clone repo if not present)
SEEDVR2_CLI_DIR="$HOME/AI/ComfyUI-SeedVR2_VideoUpscaler"
if [ ! -d "$SEEDVR2_CLI_DIR" ]; then
    echo "Cloning ComfyUI-SeedVR2_VideoUpscaler..."
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEEDVR2_CLI_DIR"
else
    echo "SeedVR2 CLI already present at $SEEDVR2_CLI_DIR"
fi

# 13. Wan 2.2 I2V GGUF — dual-denoiser (both transformers required)
download_file \
    "bullerwins/Wan2.2-I2V-A14B-GGUF" \
    "wan2.2_i2v_high_noise_14B_Q8_0.gguf" \
    "$BASE_DIR/Wan2.2-I2V-A14B-GGUF"

download_file \
    "bullerwins/Wan2.2-I2V-A14B-GGUF" \
    "wan2.2_i2v_low_noise_14B_Q8_0.gguf" \
    "$BASE_DIR/Wan2.2-I2V-A14B-GGUF"

# 14. Wan 2.2 I2V Diffusers (Encoders, VAE, etc.)
download_repo \
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    "$BASE_DIR/Wan2.2-I2V-A14B-Diffusers" \
    "text_encoder/*" "vae/*" "vision_encoder/*" "tokenizer/*" "scheduler/*" "model_index.json"

# 15. Wan 2.2 I2V Diffusers - transformer configs (needed by from_single_file)
download_file \
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    "transformer/config.json" \
    "$BASE_DIR/Wan2.2-I2V-A14B-Diffusers"

download_file \
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    "transformer_2/config.json" \
    "$BASE_DIR/Wan2.2-I2V-A14B-Diffusers"

echo "All downloads complete!"
