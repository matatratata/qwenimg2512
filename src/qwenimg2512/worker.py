"""Background worker thread for image generation."""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ASPECT_RATIOS, MODEL_VARIANTS, GenerationSettings, ModelPaths

logger = logging.getLogger(__name__)


def resize_and_center_crop(image: object, width: int, height: int) -> object:
    """Resize a PIL Image to cover (width, height) preserving aspect ratio, then center crop."""
    from PIL import Image

    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w = math.ceil(src_w * scale)
    new_h = math.ceil(src_h * scale)

    image = image.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return image.crop((left, top, left + width, top + height))


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
        self._cn_hooks: list | None = None
        self._cn_state: dict | None = None

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
        is_controlnet = self._settings.controlnet_enabled and bool(self._settings.control_image_path)

        if is_gguf:
            self._pipe = self._load_gguf_pipeline(is_img2img)
        else:
            self._pipe = self._load_hf_pipeline(model_id, is_bnb, is_img2img)

        self._raise_if_cancelled()

        # Load Fun ControlNet if enabled
        fun_cn = None
        hint_tokens = None
        if is_controlnet:
            fun_cn, hint_tokens = self._load_fun_controlnet()

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

        # Set up Fun ControlNet injection hooks
        if fun_cn is not None:
            from qwenimg2512.fun_controlnet import setup_fun_controlnet_hooks

            self.stage_changed.emit("Setting up ControlNet injection...")
            self._cn_hooks, self._cn_state = setup_fun_controlnet_hooks(
                transformer=self._pipe.transformer,
                fun_cn=fun_cn,
                hint_tokens=hint_tokens,
                conditioning_scale=self._settings.controlnet_conditioning_scale,
                guidance_start=self._settings.control_guidance_start,
                guidance_end=self._settings.control_guidance_end,
                num_steps=self._settings.num_inference_steps,
            )

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
            # Update ControlNet guidance timing
            if self._cn_state is not None:
                from qwenimg2512.fun_controlnet import update_controlnet_state

                update_controlnet_state(self._cn_state, step)

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
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "callback_on_step_end": step_callback,
        }

        if is_img2img:
            from PIL import Image

            input_image = Image.open(self._settings.input_image_path).convert("RGB")
            input_image = resize_and_center_crop(input_image, width, height)
            gen_kwargs["image"] = input_image
            gen_kwargs["strength"] = self._settings.img2img_strength

        # Ensure GPU is clean before pipeline inference
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        self._emit_vram()

        output = self._pipe(**gen_kwargs)

        self._raise_if_cancelled()

        image = output.images[0]

        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_controlnet and is_img2img:
            mode_tag = "cn_img2img"
        elif is_controlnet:
            mode_tag = "controlnet"
        elif is_img2img:
            mode_tag = "img2img"
        else:
            mode_tag = "txt2img"
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
        base_model = self._model_paths.base_model_dir or "Qwen/Qwen-Image-2512"
        self.stage_changed.emit("Loading GGUF transformer...")

        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_model,
            subfolder="transformer",
        )

        self._raise_if_cancelled()
        self.stage_changed.emit("Loading pipeline...")

        if img2img:
            from diffusers import QwenImageImg2ImgPipeline

            pipe = QwenImageImg2ImgPipeline.from_pretrained(
                base_model,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            pipe = QwenImagePipeline.from_pretrained(
                base_model,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )

        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
            best_gpu = free.index(max(free))
            logger.info(
                "Multi-GPU detected: using cuda:%d (%.1f GB free) for offloading",
                best_gpu,
                free[best_gpu] / 1024**3,
            )
            pipe.enable_model_cpu_offload(gpu_id=best_gpu)
        else:
            pipe.enable_model_cpu_offload()

        return pipe

    def _load_hf_pipeline(self, model_id: str, is_bnb: bool, img2img: bool) -> object:
        import torch

        load_kwargs: dict = {"torch_dtype": torch.bfloat16}

        if img2img:
            from diffusers import QwenImageImg2ImgPipeline

            pipe = QwenImageImg2ImgPipeline.from_pretrained(model_id, **load_kwargs)
        else:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)

        if not is_bnb:
            pipe = pipe.to("cuda")

        return pipe

    def _load_fun_controlnet(self) -> tuple:
        """Load the Fun ControlNet model and encode the control image.

        Returns (fun_cn_model, hint_tokens).
        """
        import torch
        from qwenimg2512.fun_controlnet import (
            QwenImageFunControlNetModel,
            load_fun_controlnet,
        )

        cn_path = self._model_paths.controlnet_path
        self.stage_changed.emit("Loading Fun ControlNet...")

        fun_cn = load_fun_controlnet(cn_path, dtype=torch.bfloat16)

        # Place Fun ControlNet on cuda:0 (the non-offload GPU in dual-GPU setups)
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
            best_gpu = free.index(max(free))
            cn_gpu = 1 - best_gpu  # Other GPU
        else:
            cn_gpu = 0

        cn_device = torch.device(f"cuda:{cn_gpu}")
        fun_cn.to(cn_device)
        logger.info("Fun ControlNet placed on %s", cn_device)

        self._raise_if_cancelled()

        # Encode control image through VAE
        self.stage_changed.emit("Encoding control image...")
        width, height = ASPECT_RATIOS[self._settings.aspect_ratio]
        control_latents = self._encode_control_image(
            self._settings.control_image_path, width, height, cn_device
        )

        # Process latents into hint tokens
        hint_tokens = QwenImageFunControlNetModel.process_hint(control_latents)
        logger.info("Hint tokens shape: %s", hint_tokens.shape)

        self._emit_vram()
        return fun_cn, hint_tokens

    def _encode_control_image(
        self,
        image_path: str,
        width: int,
        height: int,
        device: object,
    ) -> object:
        """Encode a control image through VAE to get 16-channel latents."""
        import torch
        from diffusers import AutoencoderKLQwenImage
        from diffusers.image_processor import VaeImageProcessor
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image = resize_and_center_crop(image, width, height)
        base_model = self._model_paths.base_model_dir or "Qwen/Qwen-Image-2512"

        # Load a fresh VAE to avoid disturbing pipeline offload hooks
        vae = AutoencoderKLQwenImage.from_pretrained(
            base_model,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        image_processor = VaeImageProcessor(
            vae_scale_factor=self._pipe.vae_scale_factor
        )

        # Preprocess image to tensor
        image_tensor = image_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)

        vae.to(device)

        # QwenImage VAE expects 5D input (B, C, F, H, W)
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(2)

        # Encode to latents (use mode for deterministic encoding)
        with torch.no_grad():
            latent_dist = vae.encode(image_tensor).latent_dist
            latents = latent_dist.mode()

        # Normalize using VAE config
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=latents.device, dtype=latents.dtype
        )
        latents_std = torch.tensor(
            vae.config.latents_std, device=latents.device, dtype=latents.dtype
        )
        shape = [1, -1] + [1] * (latents.ndim - 2)
        latents = (latents - latents_mean.view(*shape)) / latents_std.view(*shape)

        # Remove frame dim: (B, C, 1, H, W) → (B, C, H, W)
        if latents.ndim == 5:
            latents = latents.squeeze(2)

        # Free VAE
        del vae
        torch.cuda.empty_cache()

        logger.info("Control latents shape: %s on %s", latents.shape, latents.device)
        return latents

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
            # Remove ControlNet hooks first
            if self._cn_hooks:
                from qwenimg2512.fun_controlnet import remove_fun_controlnet_hooks

                remove_fun_controlnet_hooks(self._cn_hooks)
                self._cn_hooks = None
                self._cn_state = None

            import torch

            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
