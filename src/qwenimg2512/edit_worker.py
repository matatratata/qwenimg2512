"""Background worker thread for Qwen-Image-Edit-2511 generation."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ASPECT_RATIOS, EditSettings, ModelPaths

logger = logging.getLogger(__name__)


class EditWorker(QThread):
    """Runs Qwen-Image-Edit-2511 inference in a background thread."""

    progress_updated = Signal(int, int, str)
    stage_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)
    vram_updated = Signal(float)

    def __init__(self, settings: EditSettings, model_paths: ModelPaths) -> None:
        super().__init__()
        self._settings = settings
        self._model_paths = model_paths
        self._is_cancelled = False
        self._pipe = None

    def cancel(self) -> None:
        self._is_cancelled = True

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled:
            raise Exception("Generation cancelled by user")

    def run(self) -> None:
        try:
            self._run_generation()
        except Exception as e:
            if str(e) == "Generation cancelled by user":
                logger.info("Generation cancelled")
            else:
                logger.exception("Generation failed")
                self.error_occurred.emit(str(e))
        finally:
            self._cleanup()

    def _run_generation(self) -> None:
        import torch
        from PIL import Image

        self.stage_changed.emit("Loading model...")
        self._raise_if_cancelled()

        self._pipe = self._load_gguf_pipeline()
        self._raise_if_cancelled()

        self._emit_vram()

        self.stage_changed.emit("Processing reference images...")
        ref_images = []
        fit_modes = [
            self._settings.ref_fit_mode_1,
            self._settings.ref_fit_mode_2,
            self._settings.ref_fit_mode_3,
        ]
        width, height = ASPECT_RATIOS[self._settings.aspect_ratio]
        for i, path in enumerate([self._settings.ref_image_1, self._settings.ref_image_2, self._settings.ref_image_3]):
            if path and Path(path).is_file():
                img = Image.open(path).convert("RGB")
                mode = fit_modes[i] if i < len(fit_modes) else "cover"
                from qwenimg2512.resize_utils import resize_with_fit_mode
                img = resize_with_fit_mode(img, width, height, mode)
                logger.info("Ref image %d resized %s → %dx%d", i + 1, mode, img.width, img.height)
                ref_images.append(img)
        
        if not ref_images:
            raise ValueError("At least one reference image is required for editing.")

        self.stage_changed.emit("Generating image...")

        width, height = ASPECT_RATIOS[self._settings.aspect_ratio]
        seed = self._settings.seed if self._settings.seed >= 0 else random.randint(0, 2**32 - 1)

        step_start = time.monotonic()

        def step_callback(pipe: object, step: int, timestep: object, callback_kwargs: dict) -> dict:
            self._raise_if_cancelled()
            elapsed = time.monotonic() - step_start
            speed = elapsed / max(step, 1)
            remaining = speed * (self._settings.num_inference_steps - step)
            self.progress_updated.emit(
                step,
                self._settings.num_inference_steps,
                f"Step {step}/{self._settings.num_inference_steps} | {remaining:.0f}s remaining",
            )
            self._emit_vram()
            return callback_kwargs

        from qwenimg2512.samplers import get_sampler
        custom_sampler = None
        if self._settings.sampler_name != "euler":
            custom_sampler = get_sampler(self._settings.sampler_name)
            if custom_sampler is None:
                logger.warning("Sampler %s not found, falling back to default scheduler", self._settings.sampler_name)

        gen_kwargs = {
            "prompt": self._settings.prompt,
            "negative_prompt": self._settings.negative_prompt,
            "image": ref_images,
            "width": width,
            "height": height,
            "num_inference_steps": self._settings.num_inference_steps,
            "true_cfg_scale": self._settings.true_cfg_scale,
            "guidance_scale": self._settings.guidance_scale,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "callback_on_step_end": step_callback,
        }

        # Clear VRAM
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self._emit_vram()

        from qwenimg2512.pipeline_patch import apply_custom_sampler
        with apply_custom_sampler(self._pipe, custom_sampler):
            output = self._pipe(**gen_kwargs)
        self._raise_if_cancelled()

        image = output.images[0]
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"qwen_edit_{seed}_{width}x{height}_s{self._settings.num_inference_steps}.png"
        output_path = output_dir / filename

        counter = 1
        while output_path.exists():
            stem = output_path.stem.rsplit("_v", 1)[0]
            output_path = output_dir / f"{stem}_v{counter:03d}.png"
            counter += 1

        image.save(str(output_path))
        logger.info("Image saved to %s", output_path)

        self.progress_updated.emit(
            self._settings.num_inference_steps,
            self._settings.num_inference_steps,
            "Complete!",
        )
        self.finished_success.emit(str(output_path))

    def _load_gguf_pipeline(self) -> object:
        import torch
        from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel

        gguf_path = self._model_paths.edit_gguf
        base_model = self._model_paths.edit_base_model_dir or "Qwen/Qwen-Image-Edit-2511"
        
        self.stage_changed.emit("Loading GGUF transformer...")

        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_model, # Can be path or repo ID
            subfolder="transformer",
        )

        self._raise_if_cancelled()
        self.stage_changed.emit("Loading pipeline...")

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        # ── Load LoRA on CPU, fuse into base weights, free adapters ──
        lora_path = self._settings.lora_path
        if lora_path and Path(lora_path).is_file():
            self.stage_changed.emit("Loading LoRA adapter (CPU)...")
            pipe.load_lora_weights(lora_path, adapter_name="default")
            pipe.set_adapters(["default"], [self._settings.lora_scale_start])
            self.stage_changed.emit("Fusing LoRA into model...")
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            logger.info("LoRA fused (scale=%.2f) from %s", self._settings.lora_scale_start, lora_path)

        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
            best_gpu = free.index(max(free))
            logger.info("Multi-GPU: using cuda:%d for Edit pipeline", best_gpu)
            pipe.enable_model_cpu_offload(gpu_id=best_gpu)
        else:
            pipe.enable_model_cpu_offload()

        return pipe

    def _emit_vram(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                self.vram_updated.emit(vram_gb)
        except Exception:
            pass

    def _cleanup(self) -> None:
        try:
            import gc

            import torch

            if self._pipe:
                del self._pipe
                self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
