"""Background worker thread for Wan Cinematic I2V generation."""

from __future__ import annotations

import gc
import logging
import random
import time
from pathlib import Path

import torch
from PySide6.QtCore import QThread, Signal
from PIL import Image

from qwenimg2512.config import WanSettings, ModelPaths
from qwenimg2512.resize_utils import resize_with_fit_mode

logger = logging.getLogger(__name__)


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _WanCache:
    def __init__(self):
        self.pipe = None
        self.gguf_path = None
        self.base_model = None


_GLOBAL_CACHE = _WanCache()


class WanWorker(QThread):
    progress_updated = Signal(int, int, str)
    stage_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)
    vram_updated = Signal(float)

    def __init__(self, settings: WanSettings, model_paths: ModelPaths) -> None:
        super().__init__()
        self._settings = settings
        self._model_paths = model_paths
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True
        if _GLOBAL_CACHE.pipe is not None:
            _GLOBAL_CACHE.pipe._interrupt = True

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled:
            raise Exception("Generation cancelled by user")

    def _emit_vram(self) -> None:
        try:
            if torch.cuda.is_available():
                total = sum(
                    torch.cuda.memory_allocated(i)
                    for i in range(torch.cuda.device_count())
                ) / (1024**3)
                self.vram_updated.emit(total)
        except Exception:
            pass

    def run(self) -> None:
        try:
            self._run_generation()
        except Exception as e:
            if str(e) == "Generation cancelled by user":
                logger.info("Wan generation cancelled")
            else:
                logger.exception("Wan generation failed")
                self.error_occurred.emit(str(e))
        finally:
            _free_gpu_memory()

    def _run_generation(self) -> None:
        from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig
        from diffusers.utils import export_to_video

        res_str = self._settings.resolution
        width, height = map(int, res_str.split("x"))

        if not self._settings.input_image or not Path(self._settings.input_image).is_file():
            raise ValueError("A valid input image is required for Wan I2V.")

        input_image = Image.open(self._settings.input_image).convert("RGB")
        input_image = resize_with_fit_mode(input_image, width, height, "cover")

        gguf_path = self._model_paths.wan_gguf
        base_dir = self._model_paths.wan_base_model_dir

        if not Path(gguf_path).is_file():
            raise FileNotFoundError(f"Wan GGUF not found at {gguf_path}")
        if not Path(base_dir).is_dir():
            raise FileNotFoundError(f"Wan Base Directory not found at {base_dir}")

        c = _GLOBAL_CACHE
        if c.pipe is not None and c.gguf_path == gguf_path and c.base_model == base_dir:
            logger.info("Using cached Wan pipeline")
            self.stage_changed.emit("Using cached model ✓")
            pipe = c.pipe
            pipe._interrupt = False
        else:
            if c.pipe is not None:
                del c.pipe
                c.pipe = None
                _free_gpu_memory()

            self.stage_changed.emit("Loading Wan Transformer (GGUF)...")
            self._raise_if_cancelled()

            transformer = WanTransformer3DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                config=base_dir,
                subfolder="transformer",
            )

            # Diffusers GGUF compat patch
            if hasattr(transformer.config, "image_dim"):
                transformer.config.image_dim = None

            self.stage_changed.emit("Loading Wan Pipeline...")
            pipe = WanImageToVideoPipeline.from_pretrained(
                base_dir,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )

            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                best_gpu = 1
                logger.info("Multi-GPU: Routing Wan models to cuda:%d", best_gpu)
                pipe.enable_model_cpu_offload(gpu_id=best_gpu)
            else:
                pipe.enable_model_cpu_offload()

            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

            c.pipe = pipe
            c.gguf_path = gguf_path
            c.base_model = base_dir

        self._raise_if_cancelled()

        # Configure shift / flow_shift
        if hasattr(pipe.scheduler.config, "shift"):
            pipe.scheduler = pipe.scheduler.__class__.from_config(
                pipe.scheduler.config, shift=self._settings.shift
            )
        elif hasattr(pipe.scheduler.config, "flow_shift"):
            pipe.scheduler = pipe.scheduler.__class__.from_config(
                pipe.scheduler.config, flow_shift=self._settings.shift
            )

        self.stage_changed.emit("Generating Cinematic Sequence...")

        seed = self._settings.seed if self._settings.seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        step_start = time.monotonic()

        def step_callback(pipe_, step, timestep, cb_kwargs):
            if self._is_cancelled:
                if hasattr(pipe_, "_interrupt"):
                    pipe_._interrupt = True
            elapsed = time.monotonic() - step_start
            speed = elapsed / max(step, 1)
            remaining = speed * (self._settings.num_inference_steps - step)
            self.progress_updated.emit(
                step,
                self._settings.num_inference_steps,
                f"Step {step}/{self._settings.num_inference_steps} | {remaining:.0f}s left",
            )
            self._emit_vram()
            return cb_kwargs

        _free_gpu_memory()

        from qwenimg2512.samplers import get_sampler
        custom_sampler = None
        if getattr(self._settings, "sampler_name", "euler") != "euler":
            custom_sampler = get_sampler(self._settings.sampler_name)
            if custom_sampler is None:
                logger.warning("Sampler %s not found, falling back to default scheduler", self._settings.sampler_name)

        from qwenimg2512.pipeline_patch import apply_custom_sampler, apply_custom_schedule

        with apply_custom_sampler(pipe, custom_sampler):
            with apply_custom_schedule(pipe, getattr(self._settings, "schedule_name", "default")):
                output = pipe(
                    prompt=self._settings.prompt,
                    negative_prompt=self._settings.negative_prompt if self._settings.negative_prompt else None,
                    image=input_image,
                    num_frames=self._settings.frames,
                    height=height,
                    width=width,
                    num_inference_steps=self._settings.num_inference_steps,
                    guidance_scale=self._settings.guidance_scale,
                    generator=generator,
                    callback_on_step_end=step_callback,
                )

        self._raise_if_cancelled()
        self.stage_changed.emit("Saving outputs...")

        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = output.frames[0]

        base_name = f"wan_cinematic_{seed}_{width}x{height}_s{self._settings.num_inference_steps}"
        vid_path = output_dir / f"{base_name}.mp4"
        counter = 1
        while vid_path.exists():
            vid_path = output_dir / f"{base_name}_v{counter:03d}.mp4"
            counter += 1

        export_to_video(frames, str(vid_path), fps=16)
        logger.info("Saved cinematic video to %s", vid_path)

        final_path = str(vid_path)

        if self._settings.extract_still and len(frames) > 0:
            self.stage_changed.emit("Extracting cinematic still...")
            still_path = vid_path.with_suffix(".png")
            mid_idx = len(frames) // 2

            frame_data = frames[mid_idx]
            if isinstance(frame_data, Image.Image):
                frame_data.save(str(still_path))
            else:
                import numpy as np
                Image.fromarray(np.uint8(frame_data)).save(str(still_path))

            logger.info("Saved cinematic still to %s", still_path)
            final_path = str(still_path)

        self.progress_updated.emit(
            self._settings.num_inference_steps,
            self._settings.num_inference_steps,
            "Complete!",
        )
        self.finished_success.emit(final_path)
