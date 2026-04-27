#!/usr/bin/env bash
# =============================================================================
# Qwen Building Pipeline — Local Setup Script
# =============================================================================
# Usage:  ./local_setup.sh
#
# Installs the qwenimg2512 project, downloads all required models from
# HuggingFace, and builds the WebUI frontend.
#
# Mirrors vastai_setup.sh but targets a local workstation:
#   • No root/apt — assumes system deps are met
#   • Uses ~/AI/Models (matching config.py defaults)
#   • No Supervisor — run the dev server manually
#
# Override env vars:
#   MODEL_DIR           — where to store models      (default: ~/AI/Models)
#   SKIP_MODEL_DOWNLOAD — set to 1 to skip downloads (default: 0)
#   QWEN_PORT           — WebUI port                 (default: 8765)
#   UV_PYTHON           — Python version for venv    (default: 3.12)
# =============================================================================
set -eo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$SCRIPT_DIR"
VENV_DIR="${WEBUI_DIR}/.venv"
MODEL_DIR="${MODEL_DIR:-${HOME}/AI/Models}"
LOG_FILE="${WEBUI_DIR}/local_setup.log"
PORT="${QWEN_PORT:-8765}"
UV_PYTHON="${UV_PYTHON:-3.12}"

SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========================================"
echo "  Qwen Building Pipeline Local Setup — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "  Project:   ${WEBUI_DIR}"
echo "  Models:    ${MODEL_DIR}"
echo "  Python:    ${UV_PYTHON}"
echo ""

# ---------------------------------------------------------------------------
# 1. Prerequisites check
# ---------------------------------------------------------------------------
echo "[1/5] Checking prerequisites..."

missing=()
command -v git   &>/dev/null || missing+=(git)
command -v curl  &>/dev/null || missing+=(curl)
command -v node  &>/dev/null || missing+=(node)

if [ ${#missing[@]} -gt 0 ]; then
    echo "  ❌ Missing: ${missing[*]}"
    echo "  Install them and re-run."
    exit 1
fi
echo "  Node.js: $(node --version)"

# ---------------------------------------------------------------------------
# 2. Install uv
# ---------------------------------------------------------------------------
echo "[2/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "  uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Python environment + dependencies
# ---------------------------------------------------------------------------
MARKER="${VENV_DIR}/.local_installed"
echo "[3/5] Python environment + dependencies..."

if [ -f "$MARKER" ]; then
    echo "  Already installed (found marker). Skipping."
else
    cd "$WEBUI_DIR"
    uv venv --python "$UV_PYTHON" --clear "$VENV_DIR"

    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="${VENV_DIR}/bin:$PATH"

    echo "  Installing PyTorch cu129..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    echo "  Installing project + webui extras..."
    uv pip install -e ".[webui]"

    echo "  Installing diffusers from git (required)..."
    uv pip install git+https://github.com/huggingface/diffusers.git

    echo "  Installing hf_transfer for fast downloads..."
    uv pip install "huggingface_hub[hf_transfer]"

    touch "$MARKER"
    echo "  ✅ Python environment ready."
fi

# Activate venv for remaining steps
export VIRTUAL_ENV="$VENV_DIR"
export PATH="${VENV_DIR}/bin:$PATH"

# ---------------------------------------------------------------------------
# 4. Download models
# ---------------------------------------------------------------------------
echo "[4/5] Checking models..."

if [ "$SKIP_MODEL_DOWNLOAD" = "1" ]; then
    echo "  ⏭️  SKIP_MODEL_DOWNLOAD=1 — skipping HuggingFace downloads."
    echo "  Provide models manually to: ${MODEL_DIR}/"
    echo ""
    echo "  Required structure:"
    echo "    ${MODEL_DIR}/Qwen-Image-2512-GGUF/qwen-image-2512-Q8_0.gguf"
    echo "    ${MODEL_DIR}/Qwen-Image-2512/  (scheduler, tokenizer, text_encoder, vae)"
    echo "    ${MODEL_DIR}/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q8_0.gguf"
    echo "    ${MODEL_DIR}/Qwen-Image-Edit-2511/  (scheduler, tokenizer, text_encoder, vae, processor)"
    echo "    ${MODEL_DIR}/LoRAs/restoration_v2.safetensors"
    echo "    ${MODEL_DIR}/LoRAs/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
else
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_HUB_DISABLE_XET=1
    mkdir -p "$MODEL_DIR"

    # 4a. Qwen-Image-2512 GGUF (~8 GB)
    GGUF_2512="${MODEL_DIR}/Qwen-Image-2512-GGUF"
    if [ -f "${GGUF_2512}/qwen-image-2512-Q8_0.gguf" ]; then
        echo "  Qwen-Image-2512 GGUF already downloaded."
    else
        echo "  Downloading Qwen-Image-2512 GGUF (~8 GB)..."
        mkdir -p "$GGUF_2512"
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen-Image-2512-GGUF', 'qwen-image-2512-Q8_0.gguf', local_dir='${GGUF_2512}')
"
        echo "  ✅ Qwen-Image-2512 GGUF downloaded."
    fi

    # 4b. Qwen-Image-2512 Base (scheduler, tokenizer, text_encoder, vae)
    BASE_2512="${MODEL_DIR}/Qwen-Image-2512"
    if [ -f "${BASE_2512}/vae/diffusion_pytorch_model.safetensors" ]; then
        echo "  Qwen-Image-2512 base already downloaded."
    else
        echo "  Downloading Qwen-Image-2512 base components..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-2512', local_dir='${BASE_2512}',
    allow_patterns=['scheduler/*', 'tokenizer/*', 'text_encoder/*', 'vae/*', 'model_index.json', 'transformer/config.json'])
"
        echo "  ✅ Qwen-Image-2512 base downloaded."
    fi

    # 4c. Qwen-Image-Edit-2511 GGUF (~8 GB)
    GGUF_2511="${MODEL_DIR}/Qwen-Image-Edit-2511-GGUF"
    if [ -f "${GGUF_2511}/qwen-image-edit-2511-Q8_0.gguf" ]; then
        echo "  Qwen-Image-Edit-2511 GGUF already downloaded."
    else
        echo "  Downloading Qwen-Image-Edit-2511 GGUF (~8 GB)..."
        mkdir -p "$GGUF_2511"
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen-Image-Edit-2511-GGUF', 'qwen-image-edit-2511-Q8_0.gguf', local_dir='${GGUF_2511}')
"
        echo "  ✅ Qwen-Image-Edit-2511 GGUF downloaded."
    fi

    # 4d. Qwen-Image-Edit-2511 Base (scheduler, tokenizer, text_encoder, vae, processor, transformer config)
    BASE_2511="${MODEL_DIR}/Qwen-Image-Edit-2511"
    if [ -f "${BASE_2511}/vae/diffusion_pytorch_model.safetensors" ]; then
        echo "  Qwen-Image-Edit-2511 base already downloaded."
    else
        echo "  Downloading Qwen-Image-Edit-2511 base components..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-Edit-2511', local_dir='${BASE_2511}',
    allow_patterns=['scheduler/*', 'tokenizer/*', 'text_encoder/*', 'vae/*', 'model_index.json', 'processor/*', 'transformer/config.json'])
"
        echo "  ✅ Qwen-Image-Edit-2511 base downloaded."
    fi

    # 4e. Image Restoration LoRA (dx8152)
    LORA_DIR="${MODEL_DIR}/LoRAs"
    RESTORATION_LORA="${LORA_DIR}/restoration_v2.safetensors"
    if [ -f "$RESTORATION_LORA" ]; then
        echo "  Restoration LoRA already downloaded."
    else
        echo "  Downloading Image Restoration LoRA..."
        mkdir -p "$LORA_DIR"
        curl -L -o "$RESTORATION_LORA" \
            "https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Light_restoration/resolve/main/%E7%A7%BB%E9%99%A4%E5%85%89%E5%BD%B1V2.safetensors"
        echo "  ✅ Restoration LoRA downloaded."
    fi

    # 4f. Lightning LoRA (2511 8-step speedup)
    LIGHTNING_LORA="${LORA_DIR}/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
    if [ -f "$LIGHTNING_LORA" ]; then
        echo "  Lightning LoRA already downloaded."
    else
        echo "  Downloading Lightning LoRA (2511 8-step)..."
        mkdir -p "$LORA_DIR"
        curl -L -o "$LIGHTNING_LORA" \
            "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
        echo "  ✅ Lightning LoRA downloaded."
    fi

    # 4g. ControlNet Union (Stage 02)
    CN_DIR="${MODEL_DIR}/Qwen-Image-2512-Fun-Controlnet-Union"
    CN_FILE="${CN_DIR}/Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors"
    if [ -f "$CN_FILE" ]; then
        echo "  ControlNet Union already downloaded."
    else
        echo "  Downloading ControlNet Union (~1.7 GB)..."
        mkdir -p "$CN_DIR"
        curl -L -o "$CN_FILE" \
            "https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union/resolve/main/Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors"
        echo "  ✅ ControlNet Union downloaded."
    fi

fi

# ---------------------------------------------------------------------------
# 5. Build frontend
# ---------------------------------------------------------------------------
echo "[5/5] Building frontend..."
cd "${WEBUI_DIR}/webui"
if [ -d "dist" ] && [ "dist/index.html" -nt "index.html" ]; then
    echo "  Frontend already built."
else
    npm ci --silent 2>/dev/null || npm install --silent
    npm run build
    echo "  ✅ Frontend built."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  ✅ Local setup complete!"
echo ""
echo "  To run the WebUI:"
echo "    source ${VENV_DIR}/bin/activate"
echo "    cd ${WEBUI_DIR}/webui"
echo "    python app.py"
echo ""
echo "  Or use the dev server:"
echo "    cd ${WEBUI_DIR}/webui"
echo "    npm run dev"
echo ""
echo "  Port: ${PORT}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
