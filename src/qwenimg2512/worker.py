"""Background worker thread for image generation."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ASPECT_RATIOS, MODEL_VARIANTS, GenerationSettings, ModelPaths

logger = logging.getLogger(__name__)


class GenerationCancelledException(Exception):
    pass


class GenerationWorker(QThread):
    """Runs Qwen-Image-2512 inference in a background thread."""

    progress_updated = Signal(int, int, str)  # current_step, total_steps, message
    stage_changed = Signal(str)  # stage description
    finished_success = Signal(str)  # output_path
    error_occurred = Signal(str)  # error_message
    vram_updated = Signal(float)  # VRAM GB used

    def __init__(self, settings: GenerationSettings, model_paths: ModelPaths) -> None:
        super().__init__()
        self._settings = settings
        self._model_paths = model_paths
        self._is_cancelled = False
        self._pipe = None
        self._lora_active = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled:
            raise GenerationCancelledException

    def run(self) -> None:
        try:
            self._run_generation()
        except GenerationCancelledException:
            logger.info("Generation cancelled by user")
        except Exception as e:
            logger.exception("Generation failed")
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup()

    def _run_generation(self) -> None:
        import torch

        self.stage_changed.emit("Loading model...")
        self._raise_if_cancelled()

        model_id = MODEL_VARIANTS[self._settings.model_variant]
        is_gguf = model_id == "gguf_local"
        is_bnb = "bnb" in model_id
        is_img2img = bool(self._settings.input_image_path)

        if is_gguf:
            self._pipe = self._load_gguf_pipeline(is_img2img)
        else:
            self._pipe = self._load_hf_pipeline(model_id, is_bnb, is_img2img)

        self._raise_if_cancelled()

        # Load LoRA adapter if configured
        lora_path = self._settings.lora_path
        if lora_path and Path(lora_path).is_file():
            self.stage_changed.emit("Loading LoRA adapter...")
            self._pipe.load_lora_weights(lora_path)
            self._lora_active = True
        else:
            self._lora_active = False

        self._emit_vram()

        self.stage_changed.emit("Generating image...")

        width, height = ASPECT_RATIOS[self._settings.aspect_ratio]
        seed = self._settings.seed if self._settings.seed >= 0 else random.randint(0, 2**32 - 1)

        step_start = time.monotonic()

        # Resolve LoRA step range
        lora_step_start = self._settings.lora_step_start
        lora_step_end = self._settings.lora_step_end
        if lora_step_end < 0:
            lora_step_end = self._settings.num_inference_steps - 1

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
            # Per-step LoRA scaling
            if self._lora_active:
                if lora_step_start <= step <= lora_step_end:
                    t = (step - lora_step_start) / max(lora_step_end - lora_step_start, 1)
                    scale = self._settings.lora_scale_start + t * (self._settings.lora_scale_end - self._settings.lora_scale_start)
                    self._pipe.set_adapters(["default"], [scale])
                else:
                    self._pipe.set_adapters(["default"], [0.0])
            self._emit_vram()
            return callback_kwargs

        gen_kwargs = {
            "prompt": self._settings.prompt,
            "negative_prompt": self._settings.negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": self._settings.num_inference_steps,
            "true_cfg_scale": self._settings.true_cfg_scale,
            "guidance_scale": self._settings.guidance_scale,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "callback_on_step_end": step_callback,
        }

        if is_img2img:
            from PIL import Image

            input_image = Image.open(self._settings.input_image_path).convert("RGB")
            gen_kwargs["image"] = input_image
            gen_kwargs["strength"] = self._settings.img2img_strength

        output = self._pipe(**gen_kwargs)

        self._raise_if_cancelled()

        image = output.images[0]

        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mode_tag = "img2img" if is_img2img else "txt2img"
        filename = f"qwen_{seed}_{width}x{height}_s{self._settings.num_inference_steps}_cfg{self._settings.true_cfg_scale}_{mode_tag}.png"
        output_path = output_dir / filename

        # Handle versioning
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

    def _load_gguf_pipeline(self, img2img: bool) -> object:
        import torch
        from diffusers import GGUFQuantizationConfig, QwenImagePipeline, QwenImageTransformer2DModel

        gguf_path = self._model_paths.diffusion_gguf
        self.stage_changed.emit("Loading GGUF transformer...")

        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config="Qwen/Qwen-Image-2512",
            subfolder="transformer",
        )

        self._raise_if_cancelled()
        self.stage_changed.emit("Loading pipeline...")

        if img2img:
            from diffusers import QwenImageImg2ImgPipeline

            pipe = QwenImageImg2ImgPipeline.from_pretrained(
                "Qwen/Qwen-Image-2512",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            pipe = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image-2512",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )

        pipe.enable_model_cpu_offload()
        return pipe

    def _load_hf_pipeline(self, model_id: str, is_bnb: bool, img2img: bool) -> object:
        import torch

        load_kwargs = {"torch_dtype": torch.bfloat16}

        if img2img:
            from diffusers import QwenImageImg2ImgPipeline

            pipe = QwenImageImg2ImgPipeline.from_pretrained(model_id, **load_kwargs)
        else:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)

        if not is_bnb:
            pipe = pipe.to("cuda")

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
            import torch

            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
