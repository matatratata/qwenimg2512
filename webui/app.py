"""
Qwen Building Pipeline — FastAPI Backend

Workspace-based WebUI for building generation with staged workflow.
Reuses inference logic from the existing PySide6 app workers.
"""

import asyncio
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
import gc
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Add the project src to path so we can import existing modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger("qwen_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# ---------------------------------------------------------------------------
# Config — persisted settings
# ---------------------------------------------------------------------------
SETTINGS_FILE = Path(__file__).parent / ".settings.json"
WEBUI_DIR = Path(__file__).parent
DIST_DIR = WEBUI_DIR / "dist"


def _load_settings() -> dict:
    defaults = {
        "workspace_root": str(Path.home() / "Pictures" / "qwen-buildings"),
        "loras_dir": "",
        "lightning_lora_path": "",
        "restoration_lora_path": "",
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def _save_settings(settings: dict):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def _get_workspace_root() -> Path:
    return Path(_load_settings()["workspace_root"])


# GPU lock — single generation at a time
gpu_lock = asyncio.Lock()

# Progress tracking
progress_store: dict = {}
worker_store: dict = {}


def _cleanup_worker(worker, label: str = "") -> None:
    """Ensure GPU memory is freed after every generation.

    The PySide workers have a ``run()`` → ``finally: self._cleanup()`` flow,
    but the WebUI calls ``_run_generation()`` directly, bypassing cleanup.
    This function must be called in a ``finally`` block in every runner.
    """
    import torch

    logger.info("[cleanup:%s] Releasing worker resources...", label)

    # 0. Remove from worker_store
    for tid, w in list(worker_store.items()):
        if w is worker:
            worker_store.pop(tid, None)

    # 1. Call the worker's own cleanup (deletes self._pipe, etc.)
    try:
        worker._cleanup()
    except Exception as exc:
        logger.warning("[cleanup:%s] worker._cleanup failed: %s", label, exc)

    # 2. Forcefully drop all references the worker might still hold
    for attr in ("_pipe", "_cn_hooks", "_cn_state"):
        if hasattr(worker, attr):
            try:
                setattr(worker, attr, None)
            except Exception:
                pass

    # 3. Evict global caches that the worker may have populated
    #    (PySide workers intentionally preserve caches for repeat runs,
    #    but in the WebUI we must free them to avoid cross-stage OOM)
    _pre_generation_gc()


def _pre_generation_gc() -> None:
    """Evict ALL global model caches and free GPU memory before generation.

    The PySide workers cache pipelines across runs via module-level
    ``_GLOBAL_CACHE`` singletons.  In the desktop app this saves reload time
    when repeating the same stage, but in the WebUI switching between stages
    (e.g. Stage 01 → Stage 03) means a *different* pipeline class needs the
    GPU.  The old pipeline stays pinned by its cache → OOM.

    We must clear every cache before each request.
    """
    import torch

    # ── 1. Clear GenerationWorker cache (worker.py — Stage 01) ────────────
    try:
        from qwenimg2512.worker import _GLOBAL_CACHE as gen_cache
        if gen_cache.pipe is not None:
            logger.info("[pre-gc] Evicting GenerationWorker cache (Stage 01 pipeline)")
            # Remove accelerate/cpu-offload hooks so .to("cpu") works
            try:
                gen_cache.pipe.remove_all_hooks()
            except Exception:
                pass
            del gen_cache.pipe
            gen_cache.pipe = None
            gen_cache.model_variant = None
            gen_cache.gguf_path = None
            gen_cache.base_model = None
            gen_cache.lora_path = None
            gen_cache.lora_adapter_names = []
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[pre-gc] Failed to clear GenerationWorker cache: %s", exc)

    # ── 2. Clear Edit2509Worker cache (edit_2509_worker.py — Stage 02) ────
    try:
        from qwenimg2512.edit_2509_worker import _GLOBAL_CACHE as edit2509_cache
        has_te = edit2509_cache.text_encoder is not None
        has_tr = edit2509_cache.transformer is not None
        if has_te or has_tr:
            logger.info("[pre-gc] Evicting Edit2509Worker cache (Stage 02 pipeline)")
            if has_te:
                del edit2509_cache.text_encoder
                edit2509_cache.text_encoder = None
            edit2509_cache.processor = None
            edit2509_cache.tokenizer = None
            edit2509_cache.encoder_device = None
            if has_tr:
                del edit2509_cache.transformer
                edit2509_cache.transformer = None
            if edit2509_cache.vae is not None:
                del edit2509_cache.vae
                edit2509_cache.vae = None
            edit2509_cache.scheduler = None
            edit2509_cache.base_model_path = None
            edit2509_cache.active_loras = set()
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[pre-gc] Failed to clear Edit2509Worker cache: %s", exc)

    # ── 3. GC + CUDA cache flush ─────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            free, total = torch.cuda.mem_get_info(i)
            logger.info(
                "[pre-gc] cuda:%d alloc=%.2fGB free=%.2fGB/%.2fGB",
                i, alloc, free / 1024**3, total / 1024**3,
            )

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen Building Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_workspace_dir(name: str) -> Path:
    """Resolve and validate a workspace directory."""
    safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    return _get_workspace_root() / safe_name


def get_stage_dir(workspace_name: str, stage) -> Path:
    ws = get_workspace_dir(workspace_name)
    if str(stage) == 'rotate':
        d = ws / "stage_rotate"
    else:
        d = ws / f"stage_{int(stage):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_history_dir(workspace_name: str) -> Path:
    ws = get_workspace_dir(workspace_name)
    d = ws / ".history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_history_entry(workspace_name: str, stage, params: dict, filename: str):
    """Save a generation history entry as JSON."""
    hist_dir = get_history_dir(workspace_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_label = str(stage)  # supports 'rotate' or int
    entry = {
        "stage": stage_label,
        "timestamp": datetime.now().isoformat(),
        "thumbnail": filename,
        **params,
    }
    hist_file = hist_dir / f"stage{stage_label}_{ts}_{uuid.uuid4().hex[:6]}.json"
    hist_file.write_text(json.dumps(entry, indent=2))
    return entry


def _init_progress(task_id: str):
    """Create a fresh progress entry. Call in async handler BEFORE run_in_executor."""
    progress_store[task_id] = {
        "stage": "Loading model\u2026",
        "step": 0,
        "total": 0,
        "message": "",
        "vram_gb": 0.0,
        "done": False,
        "error": None,
    }


def _connect_worker_progress(worker, task_id: str):
    """Connect a worker's Qt signals to an existing progress_store entry."""

    def _query_vram():
        """Query all GPUs and update progress_store."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            gpu_count = torch.cuda.device_count()
            vram = {}
            for i in range(gpu_count):
                alloc = round(torch.cuda.memory_allocated(i) / (1024**3), 1)
                tot = round(torch.cuda.get_device_properties(i).total_mem / (1024**3), 1)
                vram[f"gpu{i}"] = {"alloc": alloc, "total": tot}
            progress_store[task_id]["vram"] = vram
        except Exception:
            pass

    def on_progress(step, total, msg):
        progress_store[task_id].update({
            "step": step,
            "total": total,
            "message": msg,
        })
        _query_vram()  # Update VRAM on every step

    def on_stage(stage_text):
        progress_store[task_id]["stage"] = stage_text
        _query_vram()  # Also update on stage changes

    def on_vram(_gb):
        _query_vram()

    worker.progress_updated.connect(on_progress)
    worker.stage_changed.connect(on_stage)
    worker.vram_updated.connect(on_vram)
    
    # Store reference so we can cancel it
    worker_store[task_id] = worker


# ---------------------------------------------------------------------------
# API: Cancel Task
# ---------------------------------------------------------------------------
@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Mark a worker task as cancelled."""
    worker = worker_store.get(task_id)
    if worker is not None:
        worker._is_cancelled = True
        if task_id in progress_store:
            progress_store[task_id]["message"] = "Cancelling..."
        return {"status": "cancelled"}
    raise HTTPException(404, "Task or worker not found")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    root = _get_workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Qwen Building Pipeline WebUI ready")
    logger.info(f"   Workspace root: {root}")


# ---------------------------------------------------------------------------
# API: Settings
# ---------------------------------------------------------------------------
@app.get("/api/settings")
async def get_settings():
    from qwenimg2512.config import Config
    cfg = Config.load()
    web = _load_settings()
    web["qwen_2512_path"] = cfg.model_paths.base_model_dir
    web["qwen_2511_path"] = cfg.model_paths.edit_base_model_dir
    return web


@app.put("/api/settings")
async def update_settings(body: dict):
    from qwenimg2512.config import Config
    cfg = Config.load()
    settings = _load_settings()
    
    if "workspace_root" in body:
        new_root = body["workspace_root"].strip()
        if new_root:
            Path(new_root).mkdir(parents=True, exist_ok=True)
            settings["workspace_root"] = new_root
            
    if "loras_dir" in body:
        settings["loras_dir"] = body["loras_dir"].strip()

    if "lightning_lora_path" in body:
        settings["lightning_lora_path"] = body["lightning_lora_path"].strip()
        cfg.model_paths.telestyle_speedup = settings["lightning_lora_path"]

    if "restoration_lora_path" in body:
        settings["restoration_lora_path"] = body["restoration_lora_path"].strip()
        cfg.model_paths.telestyle_lora = settings["restoration_lora_path"]

    _save_settings(settings)

    if "qwen_2512_path" in body:
        cfg.model_paths.base_model_dir = body["qwen_2512_path"].strip()
    if "qwen_2511_path" in body:
        cfg.model_paths.edit_base_model_dir = body["qwen_2511_path"].strip()

    cfg.save()
    return await get_settings()


# ---------------------------------------------------------------------------
# API: Progress
# ---------------------------------------------------------------------------
@app.get("/api/progress/latest")
async def progress_latest():
    """Return the most recent active (non-done) task progress, for frontend polling."""
    for task_id in reversed(list(progress_store.keys())):
        info = progress_store[task_id]
        if not info.get("done"):
            # Inject task_id so frontend can identify it
            info_copy = dict(info)
            info_copy["task_id"] = task_id
            return JSONResponse(info_copy)
    return JSONResponse({"done": True, "step": 0, "total": 0, "stage": "Idle"})


@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str):
    """Server-Sent Events stream for real-time generation progress."""
    async def event_generator():
        while True:
            info = progress_store.get(task_id, {"done": True, "error": "Task not found"})
            data = json.dumps(info)
            yield f"data: {data}\n\n"
            if info.get("done"):
                break
            await asyncio.sleep(0.4)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# API: Workspaces
# ---------------------------------------------------------------------------
@app.get("/api/workspaces")
async def list_workspaces():
    """List all workspaces."""
    workspaces = []
    if _get_workspace_root().exists():
        for d in sorted(_get_workspace_root().iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "model_reference":
                # Count images
                image_count = 0
                thumbnail = None
                for stage_dir in sorted(d.iterdir()):
                    if stage_dir.is_dir() and stage_dir.name.startswith("stage_"):
                        imgs = [f for f in stage_dir.iterdir()
                                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
                        image_count += len(imgs)
                        if imgs and not thumbnail:
                            thumbnail = True  # Has at least one image

                workspaces.append({
                    "name": d.name,
                    "path": str(d),
                    "image_count": image_count,
                    "thumbnail": thumbnail is not None,
                })
    return {"workspaces": workspaces}


@app.post("/api/workspaces")
async def create_workspace(body: dict):
    """Create a new workspace."""
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Workspace name is required")

    ws_dir = get_workspace_dir(name)
    if ws_dir.exists():
        raise HTTPException(409, f"Workspace '{name}' already exists")

    ws_dir.mkdir(parents=True)
    # Pre-create stage directories
    for stage in [1, 2, 3, 4, 5]:
        (ws_dir / f"stage_{stage:02d}").mkdir()
    (ws_dir / "stage_rotate").mkdir()

    return {"name": name, "path": str(ws_dir), "image_count": 0, "thumbnail": False}


@app.get("/api/workspaces/{name}/thumbnail")
async def workspace_thumbnail(name: str):
    """Return the latest image from any stage as a thumbnail."""
    ws_dir = get_workspace_dir(name)
    if not ws_dir.exists():
        raise HTTPException(404)

    # Find the latest image
    latest = None
    latest_time = 0
    for stage_dir in ws_dir.iterdir():
        if stage_dir.is_dir() and stage_dir.name.startswith("stage_"):
            for f in stage_dir.iterdir():
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                    if f.stat().st_mtime > latest_time:
                        latest = f
                        latest_time = f.stat().st_mtime

    if not latest:
        raise HTTPException(404)

    return FileResponse(str(latest), media_type="image/png")


# ---------------------------------------------------------------------------
# API: Stage Images
# ---------------------------------------------------------------------------
@app.get("/api/workspaces/{name}/stages/{stage}/images")
async def list_stage_images(name: str, stage: str):
    """List images in a stage directory (excludes dot-files like .control_*, .ref_*)."""
    stage_dir = get_stage_dir(name, stage)
    images = sorted(
        [f.name for f in stage_dir.iterdir()
         if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')
         and not f.name.startswith('.')],
        key=lambda x: (stage_dir / x).stat().st_mtime,
    )
    return {"images": images}


@app.get("/api/workspaces/{name}/stages/{stage}/images/{filename}")
async def get_stage_image(name: str, stage: str, filename: str):
    """Serve a specific image file."""
    stage_dir = get_stage_dir(name, stage)
    filepath = stage_dir / filename
    if not filepath.exists():
        raise HTTPException(404)
    return FileResponse(str(filepath))


# ---------------------------------------------------------------------------
# API: Model References
# ---------------------------------------------------------------------------
@app.get("/api/model_references")
async def list_model_references():
    """List images in the shared model_reference folder in the workspace root."""
    ref_dir = _get_workspace_root() / "model_reference"
    if not ref_dir.exists():
        return {"images": []}
        
    images = sorted(
        [f.name for f in ref_dir.iterdir()
         if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')],
        key=lambda x: (ref_dir / x).stat().st_mtime,
        reverse=True
    )
    return {"images": images}


@app.get("/api/model_references/{filename}")
async def get_model_reference(filename: str):
    """Serve a specific model reference image."""
    ref_dir = _get_workspace_root() / "model_reference"
    filepath = ref_dir / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404)
    return FileResponse(str(filepath))


# ---------------------------------------------------------------------------
# API: Bundled Reference Images (from repo ref_images/ folder)
# ---------------------------------------------------------------------------
REF_IMAGES_DIR = PROJECT_ROOT / "ref_images"


@app.get("/api/ref_images")
async def list_ref_images():
    """List bundled reference images shipped with the repo."""
    if not REF_IMAGES_DIR.exists():
        return {"images": []}

    images = sorted(
        [f.name for f in REF_IMAGES_DIR.iterdir()
         if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')],
    )
    return {"images": images}


@app.get("/api/ref_images/{filename}")
async def get_ref_image(filename: str):
    """Serve a specific bundled reference image."""
    filepath = REF_IMAGES_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(404)
    return FileResponse(str(filepath))


# ---------------------------------------------------------------------------
# API: History
# ---------------------------------------------------------------------------
@app.get("/api/workspaces/{name}/history")
async def get_history(name: str, stage: Optional[str] = None):
    """Get generation history for a workspace."""
    hist_dir = get_history_dir(name)
    entries = []

    for f in sorted(hist_dir.iterdir()):
        if f.suffix == '.json':
            try:
                entry = json.loads(f.read_text())
                if stage is not None and str(entry.get("stage")) != str(stage):
                    continue
                entries.append(entry)
            except Exception:
                continue

    return {"history": entries}


# ---------------------------------------------------------------------------
# API: Model Shape (Stage 01) — Edit 2511 with white BG as Picture 1
# ---------------------------------------------------------------------------
@app.post("/api/model")
async def model_shape(
    workspace: str = Form(...),
    prompt: str = Form("Create 3D model image of modern neo-tokyo urban building in Picture 1 that is loosely inspired by modern neo-tokyo building from Picture 2. White background"),
    aspect_ratio: str = Form("9:16 Half (480x832)"),
    sampler_name: str = Form("res_2s"),
    schedule_name: str = Form("beta57"),
    num_inference_steps: int = Form(16),
    seed: int = Form(-1),
    lora_path: str = Form(""),
    lora_scale: float = Form(0.6),
    ref_strength_2: float = Form(0.16),
    ref_image_2: UploadFile = File(...),
):
    """Run Stage 01: Generate 3D model shape from style reference.

    Picture 1 = auto-generated white background.
    Picture 2 = user-provided style reference (low strength).
    """
    task_id = str(uuid.uuid4())
    stage_dir = get_stage_dir(workspace, 1)

    # Save style ref image
    ref2_path = stage_dir / f".ref2_{uuid.uuid4().hex[:8]}.png"
    data = await ref_image_2.read()
    ref2_path.write_bytes(data)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "sampler_name": sampler_name,
        "schedule_name": schedule_name,
        "num_inference_steps": num_inference_steps,
        "seed": actual_seed,
        "lora_path": lora_path,
        "lora_scale": lora_scale,
        "ref_strength_2": ref_strength_2,
    }

    async with gpu_lock:
        _init_progress(task_id)
        try:
            output_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_model_shape(
                    task_id=task_id,
                    output_dir=str(stage_dir),
                    ref2_image_path=str(ref2_path),
                    **params,
                ),
            )
        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("Model shape failed")
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": "Complete!", "step": num_inference_steps, "total": num_inference_steps}
    filename = Path(output_path).name
    save_history_entry(workspace, 1, params, filename)
    return {"task_id": task_id, "filename": filename, "path": output_path, "seed": actual_seed}


def _run_model_shape(
    task_id: str,
    output_dir: str,
    ref2_image_path: str,
    prompt: str,
    aspect_ratio: str,
    sampler_name: str,
    schedule_name: str,
    num_inference_steps: int,
    seed: int,
    lora_path: str = "",
    lora_scale: float = 0.6,
    ref_strength_2: float = 0.16,
) -> str:
    """Stage 01: Edit 2511 with white BG as Picture 1 + style ref as Picture 2."""
    from PIL import Image
    from qwenimg2512.config import ASPECT_RATIOS, EditSettings, Config

    _pre_generation_gc()
    config = Config.load()

    # Create white image for Picture 1
    width, height = ASPECT_RATIOS.get(aspect_ratio, (480, 832))
    white_path = Path(output_dir) / ".white_bg.png"
    white_img = Image.new("RGB", (width, height), (255, 255, 255))
    white_img.save(str(white_path))
    logger.info(f"White BG (Picture 1): {white_path} ({width}x{height})")

    settings = EditSettings(
        prompt=prompt,
        negative_prompt="",
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        seed=seed,
        output_dir=output_dir,
        sampler_name=sampler_name,
        schedule_name=schedule_name,
        ref_image_1=str(white_path),
        ref_image_2=ref2_image_path,
        ref_image_3="",
        ref_fit_mode_1="cover",
        ref_fit_mode_2="cover",
        ref_fit_mode_3="stretch",
        lora_path=lora_path,
        lora_scale_start=lora_scale,
        lora_scale_end=lora_scale,
        lora_step_start=0,
        lora_step_end=-1,
        lora_path_2="",
        lora_scale_start_2=1.0,
        lora_scale_end_2=1.0,
        lora_step_start_2=0,
        lora_step_end_2=-1,
        ref_strength_1=1.0,
        ref_strength_2=ref_strength_2,
        ref_strength_3=0.2,
        ffn_chunk_size=2048,
        blocks_to_swap=0,
        attn_chunk_size=4096,
    )

    logger.info(f"Model shape: ref1(white)={settings.ref_image_1}, ref2={settings.ref_image_2}, str2={ref_strength_2}")

    from qwenimg2512.edit_worker import EditWorker

    worker = EditWorker(settings, config.model_paths)
    worker._is_cancelled = False
    _connect_worker_progress(worker, task_id)

    try:
        worker._run_generation()
    finally:
        _cleanup_worker(worker, "stage01-model")

    output_path = Path(output_dir)
    images = sorted(
        [f for f in output_path.iterdir()
         if f.suffix == '.png' and not f.name.startswith('.')],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not images:
        raise RuntimeError("No output image produced")
    return str(images[0])


# ---------------------------------------------------------------------------
# API: Generate (Stage 02)
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(
    workspace: str = Form(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    controlnet_conditioning_scale: float = Form(0.25),
    true_cfg_scale: float = Form(4.0),
    num_inference_steps: int = Form(50),
    aspect_ratio: str = Form("9:16 (960x1664)"),
    sampler_name: str = Form("res_2s"),
    schedule_name: str = Form("beta57"),
    seed: int = Form(-1),
    control_image: Optional[UploadFile] = File(None),
):
    """Run Stage 02 generation: Qwen 2512 + ControlNet."""
    task_id = str(uuid.uuid4())
    stage_dir = get_stage_dir(workspace, 2)

    # Save control image if provided
    control_image_path = ""
    if control_image:
        ctrl_path = stage_dir / f".control_{uuid.uuid4().hex[:8]}.png"
        data = await control_image.read()
        ctrl_path.write_bytes(data)
        control_image_path = str(ctrl_path)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": num_inference_steps,
        "aspect_ratio": aspect_ratio,
        "sampler_name": sampler_name,
        "schedule_name": schedule_name,
        "seed": actual_seed,
    }

    async with gpu_lock:
        _init_progress(task_id)
        try:
            output_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_generate(
                    task_id=task_id,
                    output_dir=str(stage_dir),
                    control_image_path=control_image_path,
                    **params,
                ),
            )
        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("Generation failed")
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": "Complete!", "step": num_inference_steps, "total": num_inference_steps}
    filename = Path(output_path).name
    # Include control image in history so we can preview it
    if control_image_path:
        params["control_image"] = Path(control_image_path).name
    save_history_entry(workspace, 2, params, filename)
    return {"task_id": task_id, "filename": filename, "path": output_path, "seed": actual_seed}


def _run_generate(
    task_id: str,
    output_dir: str,
    prompt: str,
    negative_prompt: str,
    controlnet_conditioning_scale: float,
    true_cfg_scale: float,
    num_inference_steps: int,
    aspect_ratio: str,
    sampler_name: str,
    schedule_name: str,
    seed: int,
    control_image_path: str = "",
) -> str:
    """Synchronous generation using existing worker logic."""
    import torch
    from qwenimg2512.config import ASPECT_RATIOS, GenerationSettings, ModelPaths, Config

    _pre_generation_gc()
    config = Config.load()

    settings = GenerationSettings(
        prompt=prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=1.0,
        seed=seed,
        model_variant="GGUF Q8_0 (local)",
        output_dir=output_dir,
        sampler_name=sampler_name,
        schedule_name=schedule_name,
        controlnet_enabled=bool(control_image_path),
        control_type="mlsd",
        control_image_path=control_image_path,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    )

    from qwenimg2512.worker import GenerationWorker

    worker = GenerationWorker(settings, config.model_paths)
    worker._is_cancelled = False
    _connect_worker_progress(worker, task_id)

    try:
        worker._run_generation()
    finally:
        _cleanup_worker(worker, "stage01-generate")

    # Find the output file (most recent in output_dir)
    output_path = Path(output_dir)
    images = sorted(
        [f for f in output_path.iterdir()
         if f.suffix == '.png' and not f.name.startswith('.')],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not images:
        raise RuntimeError("No output image produced")

    return str(images[0])


# ---------------------------------------------------------------------------
# API: Batch Blender — upload passes (Stage 05)
# ---------------------------------------------------------------------------
@app.post("/api/batch/blender/upload")
async def batch_blender_upload(
    workspace: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload blender-rendered PNG passes to the workspace."""
    passes_dir = get_stage_dir(workspace, 5) / "blender_passes"
    passes_dir.mkdir(parents=True, exist_ok=True)

    uploaded = {}
    for f in files:
        if not f.filename:
            continue
        safe_name = Path(f.filename).name  # strip any directory components
        dest = passes_dir / safe_name
        data = await f.read()
        dest.write_bytes(data)
        uploaded[safe_name] = str(dest)
        logger.info(f"Uploaded blender pass: {dest} ({len(data)} bytes)")

    return {"uploaded": uploaded, "passes_dir": str(passes_dir), "count": len(uploaded)}


@app.get("/api/batch/blender/passes/{workspace}/{filename}")
async def batch_blender_pass(workspace: str, filename: str):
    """Serve an uploaded blender pass image."""
    from fastapi.responses import FileResponse
    safe_name = Path(filename).name
    passes_dir = get_stage_dir(workspace, 5) / "blender_passes"
    filepath = passes_dir / safe_name
    if not filepath.is_file():
        raise HTTPException(404, f"Pass not found: {safe_name}")
    return FileResponse(str(filepath), media_type="image/png")


# ---------------------------------------------------------------------------
# API: Batch Blender – single camera (Stage 05)
# ---------------------------------------------------------------------------
def _match_aspect_ratio(width: int, height: int) -> str:
    """Find the ASPECT_RATIOS key that best matches the given resolution."""
    from qwenimg2512.config import ASPECT_RATIOS
    target_ratio = width / height if height else 1.0
    best_key = "9:16 (960x1664)"
    best_dist = float('inf')
    for key, (w, h) in ASPECT_RATIOS.items():
        ratio = w / h if h else 1.0
        # Prefer matching aspect ratio, then closeness in total pixels
        ratio_dist = abs(ratio - target_ratio)
        pixel_dist = abs(w * h - width * height) / 1_000_000
        dist = ratio_dist * 10 + pixel_dist
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key


def _resolve_blender_pass(workspace: str, filename: str) -> str:
    """Resolve a relative blender pass filename to an absolute server path."""
    if not filename:
        return ""
    safe_name = Path(filename).name
    passes_dir = get_stage_dir(workspace, 5) / "blender_passes"
    filepath = passes_dir / safe_name
    if not filepath.is_file():
        raise HTTPException(400, f"Blender pass not found: {safe_name}. Upload passes first.")
    return str(filepath)


@app.post("/api/batch/blender/camera")
async def batch_blender_camera(
    workspace: str = Form(...),
    camera_name: str = Form(...),
    combined_path: str = Form(...),
    ao_path: str = Form(""),
    prompt: str = Form(""),
    aspect_ratio: str = Form("9:16 (960x1664)"),
    sampler_name: str = Form("res_2s"),
    schedule_name: str = Form("beta57"),
    num_inference_steps: int = Form(21),
    seed: int = Form(-1),
    lora_path: str = Form(""),
    lora_scale: float = Form(0.6),
    resolution_x: int = Form(0),
    resolution_y: int = Form(0),
):
    """Process a single camera from a blender batch."""
    # If resolution provided, auto-match to closest aspect ratio
    if resolution_x > 0 and resolution_y > 0:
        aspect_ratio = _match_aspect_ratio(resolution_x, resolution_y)
        logger.info(f"Camera {camera_name}: {resolution_x}x{resolution_y} → {aspect_ratio}")

    # Resolve relative filenames against blender_passes/
    combined_abs = _resolve_blender_pass(workspace, combined_path)
    ao_abs = _resolve_blender_pass(workspace, ao_path) if ao_path else ""

    task_id = str(uuid.uuid4())
    stage_dir = get_stage_dir(workspace, 5)
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    async with gpu_lock:
        _init_progress(task_id)

        try:
            def _run_single():
                from qwenimg2512.config import EditSettings, Config
                from qwenimg2512.edit_worker import EditWorker
                config = Config.load()

                progress_store[task_id].update({
                    "stage": f"Processing {camera_name}",
                    "step": 0, "total": num_inference_steps
                })

                settings = EditSettings(
                    prompt=prompt,
                    negative_prompt="",
                    aspect_ratio=aspect_ratio,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=1.0,
                    guidance_scale=1.0,
                    seed=actual_seed,
                    output_dir=str(stage_dir),
                    sampler_name=sampler_name,
                    schedule_name=schedule_name,
                    ref_image_1=combined_abs,
                    ref_image_2=ao_abs,
                    ref_image_3="",
                    ref_fit_mode_1="cover",
                    ref_fit_mode_2="cover",
                    ref_fit_mode_3="cover",
                    lora_path=lora_path,
                    lora_scale_start=lora_scale,
                    lora_scale_end=lora_scale,
                    lora_step_start=0,
                    lora_step_end=-1,
                    lora_path_2="",
                    lora_scale_start_2=1.0,
                    lora_scale_end_2=1.0,
                    lora_step_start_2=0,
                    lora_step_end_2=-1,
                    ffn_chunk_size=2048,
                    blocks_to_swap=0,
                    attn_chunk_size=4096,
                )

                _pre_generation_gc()
                worker = EditWorker(settings, config.model_paths)
                worker._is_cancelled = False

                def _query_vram():
                    try:
                        import torch
                        if torch.cuda.is_available():
                            progress_store[task_id]["vram"] = {
                                f"gpu{j}": {"alloc": round(torch.cuda.memory_allocated(j)/(1024**3), 1), "total": round(torch.cuda.get_device_properties(j).total_mem / (1024**3), 1)}
                                for j in range(torch.cuda.device_count())
                            }
                    except: pass

                def on_progress(p_step, p_total, p_msg):
                    progress_store[task_id].update({"step": p_step, "total": p_total, "message": p_msg})
                    _query_vram()

                def on_stage(stage_text):
                    progress_store[task_id]["stage"] = f"[{camera_name}] {stage_text}"
                    _query_vram()

                worker.progress_updated.connect(on_progress)
                worker.stage_changed.connect(on_stage)

                if hasattr(app, "worker_store"):
                    app.worker_store[task_id] = worker

                result_filename = None
                try:
                    worker._run_generation()
                finally:
                    _cleanup_worker(worker, f"batch-{camera_name}")

                images = sorted([f for f in Path(stage_dir).iterdir() if f.suffix == '.png' and not f.name.startswith('.')], key=lambda f: f.stat().st_mtime, reverse=True)
                if images:
                    result_filename = images[0].name
                    params = {
                        "prompt": prompt, "aspect_ratio": aspect_ratio, "sampler_name": sampler_name, "schedule_name": schedule_name,
                        "num_inference_steps": num_inference_steps, "seed": actual_seed, "lora_path": lora_path, "lora_scale_start": lora_scale, "lora_scale_end": lora_scale,
                        "lora_step_start": 0, "lora_step_end": -1, "lora_path_2": "", "lora_scale_start_2": 1.0, "lora_scale_end_2": 1.0, "lora_step_start_2": 0, "lora_step_end_2": -1,
                        "ref_image_1": combined_path, "ref_image_2": ao_path, "ref_image_3": "", "ref_strength_1": 1.0, "ref_strength_2": 1.0, "ref_strength_3": 0.2, "ref_fit_mode_1": "cover", "ref_fit_mode_2": "cover", "ref_fit_mode_3": "cover",
                        "ffn_chunk_size": 2048, "blocks_to_swap": 0, "attn_chunk_size": 4096, "output_dir": str(stage_dir), "tab_name": "Batch Blender"
                    }
                    save_history_entry(workspace, 5, params, result_filename)
                return result_filename

            if hasattr(app, "worker_store"):
                app.worker_store[task_id] = None

            result = await asyncio.get_event_loop().run_in_executor(None, _run_single)

        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("Batch camera failed")
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": f"{camera_name} complete", "step": num_inference_steps, "total": num_inference_steps}

    result_url = ""
    if result:
        result_url = f"/api/workspaces/{workspace}/stages/5/images/{result}"

    return {"task_id": task_id, "status": "completed", "camera": camera_name, "result_image": result_url}



# ---------------------------------------------------------------------------
# API: Edit (Stage 03)
# ---------------------------------------------------------------------------
@app.post("/api/edit")
async def edit(
    workspace: str = Form(...),
    prompt: str = Form(""),
    aspect_ratio: str = Form("9:16 (960x1664)"),
    sampler_name: str = Form("res_2s"),
    schedule_name: str = Form("beta57"),
    num_inference_steps: int = Form(16),
    seed: int = Form(-1),
    lora_path: str = Form(""),
    lora_scale: float = Form(0.6),
    ref_image_1: UploadFile = File(...),
):
    """Run Stage 03 edit: Qwen 2511 + LoRA + white bg."""
    task_id = str(uuid.uuid4())
    stage_dir = get_stage_dir(workspace, 3)

    # Save ref image to temp
    ref_path = stage_dir / f".ref_{uuid.uuid4().hex[:8]}.png"
    data = await ref_image_1.read()
    ref_path.write_bytes(data)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "sampler_name": sampler_name,
        "schedule_name": schedule_name,
        "num_inference_steps": num_inference_steps,
        "seed": actual_seed,
        "lora_path": lora_path,
        "lora_scale": lora_scale,
    }

    async with gpu_lock:
        _init_progress(task_id)
        try:
            output_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_edit(
                    task_id=task_id,
                    output_dir=str(stage_dir),
                    ref_image_path=str(ref_path),
                    **params,
                ),
            )
        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("Edit failed")
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": "Complete!", "step": num_inference_steps, "total": num_inference_steps}
    filename = Path(output_path).name
    save_history_entry(workspace, 3, params, filename)
    return {"task_id": task_id, "filename": filename, "path": output_path, "seed": actual_seed}


def _run_edit(
    task_id: str,
    output_dir: str,
    ref_image_path: str,
    prompt: str,
    aspect_ratio: str,
    sampler_name: str,
    schedule_name: str,
    num_inference_steps: int,
    seed: int,
    lora_path: str = "",
    lora_scale: float = 0.6,
) -> str:
    """Synchronous edit using Edit 2511 worker + white background."""
    from PIL import Image
    from qwenimg2512.config import ASPECT_RATIOS, EditSettings, Config

    _pre_generation_gc()
    config = Config.load()

    # Create a white image for Picture 2 — always regenerate to match aspect ratio
    width, height = ASPECT_RATIOS.get(aspect_ratio, (960, 1664))
    white_dir = Path(output_dir)
    white_path = white_dir / ".white_bg.png"
    white_img = Image.new("RGB", (width, height), (255, 255, 255))
    white_img.save(str(white_path))
    logger.info(f"White BG image: {white_path} ({width}x{height})")

    settings = EditSettings(
        prompt=prompt,
        negative_prompt="",
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        seed=seed,
        output_dir=output_dir,
        sampler_name=sampler_name,
        schedule_name=schedule_name,
        ref_image_1=ref_image_path,
        ref_image_2=str(white_path),
        ref_image_3="",
        ref_fit_mode_1="cover",
        ref_fit_mode_2="cover",
        ref_fit_mode_3="stretch",
        lora_path=lora_path,
        lora_scale_start=lora_scale,
        lora_scale_end=lora_scale,
        lora_step_start=0,
        lora_step_end=-1,
        lora_path_2="",
        lora_scale_start_2=1.0,
        lora_scale_end_2=1.0,
        lora_step_start_2=0,
        lora_step_end_2=-1,
        ref_strength_1=1.0,
        ref_strength_2=1.0,
        ref_strength_3=0.15,
        ffn_chunk_size=2048,
        blocks_to_swap=0,
        attn_chunk_size=4096,
    )

    logger.info(f"Edit settings: ref1={settings.ref_image_1}, ref2={settings.ref_image_2}, lora={settings.lora_path}")

    from qwenimg2512.edit_worker import EditWorker

    worker = EditWorker(settings, config.model_paths)
    worker._is_cancelled = False
    _connect_worker_progress(worker, task_id)

    try:
        worker._run_generation()
    finally:
        _cleanup_worker(worker, "stage03-edit")

    # Find the output file
    output_path = Path(output_dir)
    images = sorted(
        [f for f in output_path.iterdir()
         if f.suffix == '.png' and not f.name.startswith('.')],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not images:
        raise RuntimeError("No output image produced")

    return str(images[0])


# ---------------------------------------------------------------------------
# API: De-light (Stage 04) — Edit 2511 + Dual LoRA
# ---------------------------------------------------------------------------
@app.post("/api/delight")
async def delight(
    workspace: str = Form(...),
    prompt: str = Form("移除光影,使用柔和光线（无明显光斑和阴影）对图片进行重新照明. Keep building dark grey  for better contrast with the white background."),
    aspect_ratio: str = Form("9:16 (960x1664)"),
    sampler_name: str = Form("res_2s"),
    schedule_name: str = Form("beta57"),
    num_inference_steps: int = Form(16),
    seed: int = Form(-1),
    lora_path: str = Form(""),
    lora_scale: float = Form(0.6),
    lora_path_2: str = Form(""),
    lora_scale_2: float = Form(1.0),
    ref_image_1: UploadFile = File(...),
):
    """Run Stage 04: De-light using Edit 2511 + dual LoRA (Lightning + Light Restoration)."""
    task_id = str(uuid.uuid4())
    stage_dir = get_stage_dir(workspace, 4)

    # Save ref image
    ref_path = stage_dir / f".ref_{uuid.uuid4().hex[:8]}.png"
    data = await ref_image_1.read()
    ref_path.write_bytes(data)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "sampler_name": sampler_name,
        "schedule_name": schedule_name,
        "num_inference_steps": num_inference_steps,
        "seed": actual_seed,
        "lora_path": lora_path,
        "lora_scale": lora_scale,
        "lora_path_2": lora_path_2,
        "lora_scale_2": lora_scale_2,
    }

    async with gpu_lock:
        _init_progress(task_id)
        try:
            output_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_edit2511(
                    task_id=task_id,
                    output_dir=str(stage_dir),
                    ref_image_path=str(ref_path),
                    **params,
                ),
            )
        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("De-light failed")
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": "Complete!", "step": num_inference_steps, "total": num_inference_steps}
    filename = Path(output_path).name
    save_history_entry(workspace, 4, params, filename)
    return {"task_id": task_id, "filename": filename, "path": output_path, "seed": actual_seed}


# ---------------------------------------------------------------------------
# API: Rotate / Angle (Utility) — Edit 2511 + Lightning + Multiple Angles LoRA
# ---------------------------------------------------------------------------
@app.post("/api/rotate")
async def rotate(
    workspace: str = Form(...),
    prompt: str = Form("<sks> front view eye-level shot medium shot"),
    aspect_ratio: str = Form("9:16 (960x1664)"),
    sampler_name: str = Form("res_2s"),
    schedule_name: str = Form("beta57"),
    num_inference_steps: int = Form(21),
    seed: int = Form(-1),
    lora_path: str = Form(""),
    lora_scale: float = Form(0.6),
    lora_path_2: str = Form(""),
    lora_scale_2: float = Form(1.0),
    ref_image_1: UploadFile = File(...),
):
    """Run Angle/Rotate utility: Edit 2511 + Lightning LoRA + Multiple Angles LoRA.

    Uses contain_white fit mode for the reference image (pads with white instead of cropping).
    """
    task_id = str(uuid.uuid4())
    stage_dir = get_stage_dir(workspace, "rotate")

    # Save ref image
    ref_path = stage_dir / f".ref_{uuid.uuid4().hex[:8]}.png"
    data = await ref_image_1.read()
    ref_path.write_bytes(data)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    params = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "sampler_name": sampler_name,
        "schedule_name": schedule_name,
        "num_inference_steps": num_inference_steps,
        "seed": actual_seed,
        "lora_path": lora_path,
        "lora_scale": lora_scale,
        "lora_path_2": lora_path_2,
        "lora_scale_2": lora_scale_2,
    }

    async with gpu_lock:
        _init_progress(task_id)
        try:
            output_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_rotate(
                    task_id=task_id,
                    output_dir=str(stage_dir),
                    ref_image_path=str(ref_path),
                    **params,
                ),
            )
        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("Rotate failed")
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": "Complete!", "step": num_inference_steps, "total": num_inference_steps}
    filename = Path(output_path).name
    save_history_entry(workspace, "rotate", params, filename)
    return {"task_id": task_id, "filename": filename, "path": output_path, "seed": actual_seed}


def _run_rotate(
    task_id: str,
    output_dir: str,
    ref_image_path: str,
    prompt: str,
    aspect_ratio: str,
    sampler_name: str,
    schedule_name: str,
    num_inference_steps: int,
    seed: int,
    lora_path: str = "",
    lora_scale: float = 0.6,
    lora_path_2: str = "",
    lora_scale_2: float = 1.0,
) -> str:
    """Synchronous Rotate generation — Edit 2511 with contain_white fit + dual LoRA."""
    from qwenimg2512.config import EditSettings, Config

    config = Config.load()

    settings = EditSettings(
        prompt=prompt,
        negative_prompt="",
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        seed=seed,
        output_dir=output_dir,
        sampler_name=sampler_name,
        schedule_name=schedule_name,
        ref_image_1=ref_image_path,
        ref_image_2="",
        ref_image_3="",
        ref_fit_mode_1="contain_white",
        ref_fit_mode_2="cover",
        ref_fit_mode_3="stretch",
        lora_path=lora_path,
        lora_scale_start=lora_scale,
        lora_scale_end=lora_scale,
        lora_step_start=0,
        lora_step_end=-1,
        lora_path_2=lora_path_2,
        lora_scale_start_2=lora_scale_2,
        lora_scale_end_2=lora_scale_2,
        lora_step_start_2=0,
        lora_step_end_2=-1,
        ref_strength_1=1.0,
        ref_strength_2=1.0,
        ref_strength_3=0.15,
        ffn_chunk_size=2048,
        blocks_to_swap=0,
        attn_chunk_size=4096,
    )

    logger.info(
        f"[Rotate] settings: ref1={settings.ref_image_1}, fit={settings.ref_fit_mode_1}, "
        f"lora1={settings.lora_path}, lora2={settings.lora_path_2}"
    )

    from qwenimg2512.edit_worker import EditWorker

    _pre_generation_gc()
    worker = EditWorker(settings, config.model_paths)
    worker._is_cancelled = False
    _connect_worker_progress(worker, task_id)

    try:
        worker._run_generation()
    finally:
        _cleanup_worker(worker, "rotate")

    # Find the output file
    output_path = Path(output_dir)
    images = sorted(
        [f for f in output_path.iterdir()
         if f.suffix == '.png' and not f.name.startswith('.')],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not images:
        raise RuntimeError("No output image produced")

    return str(images[0])


# ---------------------------------------------------------------------------
# Shared: Edit 2511 runner (used by Stage 04)
# ---------------------------------------------------------------------------
def _run_edit2511(
    task_id: str,
    output_dir: str,
    ref_image_path: str,
    prompt: str,
    aspect_ratio: str,
    sampler_name: str,
    schedule_name: str,
    num_inference_steps: int,
    seed: int,
    lora_path: str = "",
    lora_scale: float = 0.6,
    lora_path_2: str = "",
    lora_scale_2: float = 1.0,
) -> str:
    """Synchronous Edit 2511 generation — used by Stage 04 (De-light)."""
    from qwenimg2512.config import EditSettings, Config

    config = Config.load()

    settings = EditSettings(
        prompt=prompt,
        negative_prompt="",
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        seed=seed,
        output_dir=output_dir,
        sampler_name=sampler_name,
        schedule_name=schedule_name,
        ref_image_1=ref_image_path,
        ref_image_2="",
        ref_image_3="",
        ref_fit_mode_1="cover",
        ref_fit_mode_2="cover",
        ref_fit_mode_3="stretch",
        lora_path=lora_path,
        lora_scale_start=lora_scale,
        lora_scale_end=lora_scale,
        lora_step_start=0,
        lora_step_end=-1,
        lora_path_2=lora_path_2,
        lora_scale_start_2=lora_scale_2,
        lora_scale_end_2=lora_scale_2,
        lora_step_start_2=0,
        lora_step_end_2=34,
        ref_strength_1=1.0,
        ref_strength_2=1.0,
        ref_strength_3=0.15,
        ffn_chunk_size=2048,
        blocks_to_swap=0,
        attn_chunk_size=4096,
    )

    logger.info(f"Edit2511 settings: ref1={settings.ref_image_1}, lora1={settings.lora_path}, lora2={settings.lora_path_2}")

    from qwenimg2512.edit_worker import EditWorker

    _pre_generation_gc()
    worker = EditWorker(settings, config.model_paths)
    worker._is_cancelled = False
    _connect_worker_progress(worker, task_id)

    try:
        worker._run_generation()
    finally:
        _cleanup_worker(worker, "edit2511")

    # Find the output file
    output_path = Path(output_dir)
    images = sorted(
        [f for f in output_path.iterdir()
         if f.suffix == '.png' and not f.name.startswith('.')],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not images:
        raise RuntimeError("No output image produced")

    return str(images[0])


# ---------------------------------------------------------------------------
# API: Export Alpha (Stage 05) — save PNG with transparency
# ---------------------------------------------------------------------------
@app.post("/api/export-alpha")
async def export_alpha(
    workspace: str = Form(...),
    image: UploadFile = File(...),
    threshold: int = Form(240),
    feather: int = Form(2),
):
    """Save a PNG with alpha channel to stage_05."""
    stage_dir = get_stage_dir(workspace, 5)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alpha_{timestamp}_{uuid.uuid4().hex[:6]}.png"
    filepath = stage_dir / filename

    # Save the upload
    content = await image.read()
    filepath.write_bytes(content)

    logger.info(f"Exported alpha PNG: {filepath} ({len(content)} bytes)")

    # Save history entry
    save_history_entry(workspace, 5, {
        "threshold": threshold,
        "feather": feather,
        "prompt": f"BG removed (threshold={threshold}, feather={feather}px)",
    }, filename)

    return {"filename": filename, "path": str(filepath)}


# ---------------------------------------------------------------------------
# API: StableGen Integration — Direct Pipeline Bridge
# ---------------------------------------------------------------------------
STABLEGEN_TEMP = WEBUI_DIR / ".stablegen_tmp"
STABLEGEN_TEMP.mkdir(parents=True, exist_ok=True)

# Lightning LoRA for fast 2511 edits (8-step variant)
_LIGHTNING_LORA_2511 = str(
    Path.home() / "AI" / "Models" / "LORAS" / "2511"
    / "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
)


def _closest_aspect_ratio(width: int, height: int) -> str:
    """Find the ASPECT_RATIOS key whose pixel dimensions are closest to (width, height)."""
    from qwenimg2512.config import ASPECT_RATIOS

    target_ratio = width / height
    best_key = "1:1 (1344x1344)"
    best_diff = float("inf")

    for key, (w, h) in ASPECT_RATIOS.items():
        diff = abs(w / h - target_ratio) + abs(w * h - width * height) / 1e6
        if diff < best_diff:
            best_diff = diff
            best_key = key
    return best_key


@app.get("/api/stablegen/status")
async def stablegen_status():
    """Health-check endpoint for StableGen to verify server availability."""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpu_info[f"gpu{i}"] = {
                "free_gb": round(free / 1024**3, 2),
                "total_gb": round(total / 1024**3, 2),
            }
    return {
        "status": "ok",
        "pipeline": "qwen_edit_2511",
        "lora": _LIGHTNING_LORA_2511,
        "lora_available": Path(_LIGHTNING_LORA_2511).exists(),
        "gpu": gpu_info,
    }


@app.post("/api/stablegen/edit")
async def stablegen_edit(
    prompt: str = Form(...),
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    image3: Optional[UploadFile] = File(None),
    seed: int = Form(-1),
    steps: int = Form(16),
    lora_scale: float = Form(0.6),
    ref_strength_2: float = Form(1.0),
):
    """StableGen integration endpoint — runs Edit 2511 + Lightning LoRA.

    Accepts:
        image1: Structure / guidance image (required)
        image2: Style reference image (optional)
        image3: Context render image (optional)
        prompt: Edit prompt text
        seed:   Random seed (-1 = random)
        steps:  Inference steps (default 16)
        lora_scale: Lightning LoRA strength (default 0.6)

    Returns: Raw PNG image bytes.
    """
    task_id = str(uuid.uuid4())

    # Save uploaded images
    img1_path = STABLEGEN_TEMP / f"{task_id}_img1.png"
    img1_data = await image1.read()
    img1_path.write_bytes(img1_data)

    img2_path = ""
    if image2:
        img2_path = str(STABLEGEN_TEMP / f"{task_id}_img2.png")
        img2_data = await image2.read()
        Path(img2_path).write_bytes(img2_data)

    img3_path = ""
    if image3:
        img3_path = str(STABLEGEN_TEMP / f"{task_id}_img3.png")
        img3_data = await image3.read()
        Path(img3_path).write_bytes(img3_data)

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # Auto-detect aspect ratio from image1 dimensions
    from PIL import Image as PILImage
    with PILImage.open(str(img1_path)) as im:
        w, h = im.size
    aspect_ratio = _closest_aspect_ratio(w, h)
    logger.info(
        f"[StableGen] image1={w}x{h} → aspect={aspect_ratio}, "
        f"seed={actual_seed}, steps={steps}, prompt={prompt[:80]!r}"
    )

    # Use output dir in stablegen temp
    output_dir = str(STABLEGEN_TEMP)

    # Determine Lightning LoRA path
    lora_path = _LIGHTNING_LORA_2511 if Path(_LIGHTNING_LORA_2511).exists() else ""
    if not lora_path:
        logger.warning("[StableGen] Lightning LoRA not found, running without LoRA")

    async with gpu_lock:
        _init_progress(task_id)
        try:
            output_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_stablegen_edit(
                    task_id=task_id,
                    output_dir=output_dir,
                    ref_image_path=str(img1_path),
                    style_image_path=img2_path,
                    context_image_path=img3_path,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    num_inference_steps=steps,
                    seed=actual_seed,
                    lora_path=lora_path,
                    lora_scale=lora_scale,
                    ref_strength_2=ref_strength_2,
                ),
            )
        except Exception as e:
            progress_store[task_id] = {"done": True, "error": str(e)}
            logger.exception("[StableGen] Edit failed")
            # Cleanup temp files
            for p in (img1_path, img2_path, img3_path):
                if p and Path(p).exists():
                    Path(p).unlink(missing_ok=True)
            raise HTTPException(500, str(e))

    progress_store[task_id] = {"done": True, "stage": "Complete!"}

    # Return the image as raw PNG bytes
    result_path = Path(output_path)
    if not result_path.exists():
        raise HTTPException(500, "Output image not found")

    result_bytes = result_path.read_bytes()

    # Cleanup all temp files for this task
    for f in STABLEGEN_TEMP.glob(f"{task_id}*"):
        f.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)

    return StreamingResponse(
        iter([result_bytes]),
        media_type="image/png",
        headers={"Content-Length": str(len(result_bytes))},
    )


def _run_stablegen_edit(
    task_id: str,
    output_dir: str,
    ref_image_path: str,
    style_image_path: str,
    context_image_path: str,
    prompt: str,
    aspect_ratio: str,
    num_inference_steps: int,
    seed: int,
    lora_path: str = "",
    lora_scale: float = 0.6,
    ref_strength_2: float = 1.0,
) -> str:
    """Run Edit 2511 inference for StableGen — with Lightning LoRA."""
    from qwenimg2512.config import EditSettings, Config

    config = Config.load()

    settings = EditSettings(
        prompt=prompt,
        negative_prompt="",
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        seed=seed,
        output_dir=output_dir,
        sampler_name="res_2s",
        schedule_name="beta57",
        ref_image_1=ref_image_path,
        ref_image_2=style_image_path,
        ref_image_3=context_image_path,
        ref_fit_mode_1="cover",
        ref_fit_mode_2="cover",
        ref_fit_mode_3="stretch",
        lora_path=lora_path,
        lora_scale_start=lora_scale,
        lora_scale_end=lora_scale,
        lora_step_start=0,
        lora_step_end=-1,
        lora_path_2="",
        lora_scale_start_2=1.0,
        lora_scale_end_2=1.0,
        lora_step_start_2=0,
        lora_step_end_2=-1,
        ref_strength_1=1.0,
        ref_strength_2=ref_strength_2,
        ref_strength_3=0.15,
        ffn_chunk_size=2048,
        blocks_to_swap=0,
        attn_chunk_size=4096,
    )

    logger.info(
        f"[StableGen] Edit2511: ref1={settings.ref_image_1}, "
        f"ref2={settings.ref_image_2 or 'none'}, "
        f"ref3={settings.ref_image_3 or 'none'}, "
        f"lora={settings.lora_path or 'none'}"
    )

    from qwenimg2512.edit_worker import EditWorker

    _pre_generation_gc()
    worker = EditWorker(settings, config.model_paths)
    worker._is_cancelled = False
    _connect_worker_progress(worker, task_id)

    try:
        worker._run_generation()
    finally:
        _cleanup_worker(worker, "stablegen-edit")

    # Find the output file (most recent PNG in output_dir)
    output_path = Path(output_dir)
    images = sorted(
        [f for f in output_path.iterdir()
         if f.suffix == '.png' and not f.name.startswith('.')
         and not f.name.startswith(task_id[:8])],  # Exclude our temp inputs
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not images:
        raise RuntimeError("No output image produced")

    return str(images[0])


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return {"message": "Frontend not built. Run 'npm run dev' in webui/ for development."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.environ.get("QWEN_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
