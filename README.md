# Qwen Building Pipeline

Multi-stage image generation & editing pipeline built on Qwen-Image models, with a PySide6 desktop GUI and a FastAPI WebUI for remote/Vast.ai use.

## Stages

| Stage | Model | Description |
|-------|-------|-------------|
| 01 — Generate | Qwen-Image-2512 (GGUF) | Text-to-image with ControlNet Union support |
| 02 — ControlNet | Qwen-Image-2512 + CN Union | Canny / depth / pose guided generation |
| 03 — Edit | Qwen-Image-Edit-2511 (GGUF) | Multi-reference image editing, LoRA stacking |
| 04 — Upscale | SeedVR2 | Video/image super-resolution |
| 05 — Batch | Blender plugin → WebUI | Camera-based batch generation from Blender scenes |

## Quick Start — Local

```bash
# Prerequisites: git, curl, node 20+, NVIDIA GPU with CUDA 12.x
./local_setup.sh

# Run the WebUI
source .venv/bin/activate
cd webui && python app.py
```

Models download to `~/AI/Models` (~25 GB total). Override with `MODEL_DIR=/path/to/models ./local_setup.sh`.

Skip downloads if you already have models: `SKIP_MODEL_DOWNLOAD=1 ./local_setup.sh`

## Quick Start — Vast.ai

Set `vastai_setup.sh` as the provisioning script in your template. Image: `vastai/base-image:cuda-12.8-auto`.

Models go to `/workspace/models`. WebUI registers as a Supervisor service on port 8765.

## Project Structure

```
src/qwenimg2512/       # Core Python package
  worker.py            # Generation worker (GGUF pipeline)
  edit_worker.py       # Edit worker (2511 GGUF)
  fun_controlnet.py    # ControlNet Union integration
  config.py            # Settings & model path management
  samplers/            # Custom sampler registry
  pipeline_patch.py    # Diffusers pipeline patches (RoPE, tiling)
webui/                 # FastAPI backend + Vite frontend
  app.py               # WebUI server
blender_plugin/        # Blender batch export addon
```

## Key Features

- **GGUF quantized inference** — Q8_0 weights, fits 24 GB VRAM
- **Multi-reference editing** — up to 3 source images with per-ref strength
- **LoRA hot-loading** — dual LoRA stacking, per-step strength scheduling
- **Custom samplers** — Euler, DPM++, SMC-CFG guidance
- **ControlNet Union** — canny, depth, pose, tile in one model
- **Blender integration** — batch render → generate from camera exports

## License

Private / internal use.
