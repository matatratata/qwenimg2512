#!/usr/bin/env bash
# =============================================================================
# Qwen Building Pipeline WebUI — Vast.ai Provisioning Script
# =============================================================================
# Usage: Set this script's raw GitHub URL as the PROVISIONING_SCRIPT env var
#        in your vast.ai template.  Image: vastai/base-image:cuda-12.8-auto
#
# This installs the qwenimg2512 project with its WebUI and downloads all
# required models from HuggingFace (all public, no tokens needed).
#
# The WebUI is registered as a Supervisor service so it auto-restarts,
# logs are accessible, and it coexists with Jupyter/SSH/Syncthing.
#
# All state lives under /workspace/ (persists across stop/start).
# First boot: ~10-15 min (mostly model downloads). Subsequent: ~30 s.
# =============================================================================
set -eo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via vast.ai env vars
# ---------------------------------------------------------------------------
WEBUI_REPO="${WEBUI_REPO_URL:-https://github.com/matatratata/qwenimg2512.git}"
WEBUI_BRANCH="${WEBUI_REPO_BRANCH:-main}"

WEBUI_DIR="/workspace/qwenimg2512"
MODEL_DIR="/workspace/models"
VENV_DIR="${WEBUI_DIR}/.venv"
LOG_FILE="/workspace/setup.log"
PORT="${QWEN_PORT:-8765}"

SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_apt() {
    local tries=0
    while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
          fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
        if [ $tries -eq 0 ]; then
            echo "  Waiting for apt lock (base image still running)..."
        fi
        tries=$((tries + 1))
        sleep 2
        if [ $tries -ge 30 ]; then
            echo "  WARNING: apt lock held for 60 s, proceeding anyway"
            break
        fi
    done
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo ""
echo "========================================"
echo "  Qwen Building Pipeline Setup — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/7] Installing system packages..."
wait_for_apt
apt-get update -qq 2>&1 || echo "  WARNING: apt-get update had issues, continuing..."
apt-get install -y -qq libjpeg-dev libgl1 git curl 2>&1 || echo "  WARNING: some apt packages may have failed"

if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null 2>&1
fi
echo "  Node.js: $(node --version)"

# ---------------------------------------------------------------------------
# 2. Install uv
# ---------------------------------------------------------------------------
echo "[2/7] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "  uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Clone project
# ---------------------------------------------------------------------------
echo "[3/7] Setting up qwenimg2512..."
if [ -d "${WEBUI_DIR}/.git" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$WEBUI_DIR"
    git pull --ff-only || true
else
    echo "  Cloning ${WEBUI_REPO} (branch: ${WEBUI_BRANCH})..."
    git clone -b "$WEBUI_BRANCH" "$WEBUI_REPO" "$WEBUI_DIR"
    cd "$WEBUI_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Python environment + dependencies
# ---------------------------------------------------------------------------
MARKER="${VENV_DIR}/.vastai_installed"
echo "[4/7] Python environment + dependencies..."

if [ -f "$MARKER" ]; then
    echo "  Already installed (found marker). Skipping."
else
    cd "$WEBUI_DIR"
    uv venv --python 3.10 --clear "$VENV_DIR"

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
# 5. Download models
# ---------------------------------------------------------------------------
echo "[5/7] Checking models..."

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
    echo ""
    echo "  Example:"
    echo "    rsync -avP models/ root@<host>:${MODEL_DIR}/"
else
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_HUB_DISABLE_XET=1
    mkdir -p "$MODEL_DIR"

    # Helper: download with retry
    hf_download() {
        local repo_id="$1"; shift
        python -c "
from huggingface_hub import snapshot_download
import sys
snapshot_download('${repo_id}', local_dir='$1', $2)
" || echo "  ⚠️  Download of ${repo_id} had issues"
    }

    # 5a. Qwen-Image-2512 GGUF (~8 GB)
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

    # 5b. Qwen-Image-2512 Base (scheduler, tokenizer, text_encoder, vae)
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

    # 5c. Qwen-Image-Edit-2511 GGUF (~8 GB)
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

    # 5d. Qwen-Image-Edit-2511 Base (scheduler, tokenizer, text_encoder, vae, processor, transformer config)
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

    # 5e. Image Restoration LoRA (dx8152)
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

    # 5f. Lightning LoRA (2511 8-step speedup)
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
fi

# ---------------------------------------------------------------------------
# 5g. Write Vast.ai model config
# ---------------------------------------------------------------------------
echo "  Writing Vast.ai model paths config..."
CONFIG_DIR="/root/.config/qwenimg2512"
mkdir -p "$CONFIG_DIR"
cat > "${CONFIG_DIR}/settings.json" << CFGEOF
{
  "generation": {},
  "edit": {},
  "edit_2509": {},
  "seedvr2": {},
  "wan": {},
  "model_paths": {
    "diffusion_gguf": "${MODEL_DIR}/Qwen-Image-2512-GGUF/qwen-image-2512-Q8_0.gguf",
    "vl_model": "",
    "mmproj": "",
    "vae": "",
    "controlnet_path": "",
    "base_model_dir": "${MODEL_DIR}/Qwen-Image-2512",
    "edit_gguf": "${MODEL_DIR}/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q8_0.gguf",
    "edit_base_model_dir": "${MODEL_DIR}/Qwen-Image-Edit-2511",
    "edit_2509_gguf": "",
    "edit_2509_base_model_dir": "",
    "edit_2509_telestyle_fused_dir": "",
    "telestyle_lora": "${MODEL_DIR}/LoRAs/restoration_v2.safetensors",
    "telestyle_speedup": "${MODEL_DIR}/LoRAs/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    "llama_cpp_cli": "",
    "seedvr2_gguf": "",
    "seedvr2_vae": "",
    "seedvr2_model_dir": "",
    "seedvr2_cli": "",
    "wan_gguf_high_noise": "",
    "wan_gguf_low_noise": "",
    "wan_base_model_dir": ""
  }
}
CFGEOF
echo "  ✅ Config written to ${CONFIG_DIR}/settings.json"

# ---------------------------------------------------------------------------
# 6. Build frontend
# ---------------------------------------------------------------------------
echo "[6/7] Building frontend..."
cd "${WEBUI_DIR}/webui"
if [ -d "dist" ] && [ "dist/index.html" -nt "index.html" ]; then
    echo "  Frontend already built."
else
    npm ci --silent 2>/dev/null || npm install --silent
    npm run build
    echo "  ✅ Frontend built."
fi

# ---------------------------------------------------------------------------
# 7. Register WebUI as Supervisor service
# ---------------------------------------------------------------------------
echo "[7/7] Registering WebUI as Supervisor service..."

SUPERVISOR_CONF="/etc/supervisor/conf.d/qwen-webui.conf"
WEBUI_LAUNCHER="/opt/supervisor-scripts/qwen-webui.sh"

# Create launcher script
mkdir -p /opt/supervisor-scripts
cat > "$WEBUI_LAUNCHER" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
set -euo pipefail

# Activate the venv
export VIRTUAL_ENV="/workspace/qwenimg2512/.venv"
export PATH="${VIRTUAL_ENV}/bin:$PATH"

# Environment
export PYTHONPATH="/workspace/qwenimg2512/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export QWEN_PORT="${QWEN_PORT:-8765}"

cd /workspace/qwenimg2512/webui
exec python app.py
LAUNCHER_EOF
chmod +x "$WEBUI_LAUNCHER"

# Create Supervisor config
cat > "$SUPERVISOR_CONF" << EOF
[program:qwen-webui]
environment=PROC_NAME="%(program_name)s"
command=${WEBUI_LAUNCHER}
directory=/workspace/qwenimg2512/webui
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
startsecs=10
stopwaitsecs=30
EOF

# Add to Instance Portal
PORTAL_YAML="/etc/portal.yaml"
if [ -f "$PORTAL_YAML" ] && ! grep -q "Qwen Building Pipeline" "$PORTAL_YAML"; then
    python3 -c "
import yaml, sys
portal = '$PORTAL_YAML'
with open(portal) as f:
    data = yaml.safe_load(f) or []
entry = {
    'name': 'Qwen Building Pipeline',
    'listen_port': ${PORT},
    'proxy_port': 1${PORT},
    'metrics_port': 0,
    'custom_proxy_url': '',
    'proxy_active': True,
    'path': '/'
}
if isinstance(data, list):
    data.append(entry)
elif isinstance(data, dict):
    data['qwen-webui'] = entry
else:
    data = [entry]
with open(portal, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
print('  Added Qwen Building Pipeline to portal.yaml')
"
    # Restart Caddy to pick up the new portal entry
    supervisorctl restart caddy 2>/dev/null || true
fi

# Load the new service
supervisorctl reread
supervisorctl update

echo ""
echo "========================================"
echo "  Qwen Building Pipeline registered as Supervisor service"
echo "  Port: ${PORT}"
echo "  Status: supervisorctl status qwen-webui"
echo "  Logs:   supervisorctl tail -f qwen-webui"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
echo ""
