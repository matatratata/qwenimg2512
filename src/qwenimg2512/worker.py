"""Background worker thread for image generation.

VRAM strategy
─────────────
The GGUF transformer is the most expensive component to load (~30s).
A module-level ``_GenCache`` keeps the pipeline alive between generation
requests so that only the first run pays the loading cost.  Subsequent
runs reuse the cached pipeline and only reload when the model variant,
base model path, or img2img mode changes.  LoRA adapters are tracked
separately and swapped without a full pipeline reload.
"""

from __future__ import annotations

import gc
import logging
import math
import random
import time
from pathlib import Path

import torch
from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ASPECT_RATIOS, MODEL_VARIANTS, GenerationSettings, ModelPaths

logger = logging.getLogger(__name__)


# ── Global Model Cache ────────────────────────────────────────────────────
# Keeps the full pipeline in VRAM between requests to avoid reloading.

class _GenCache:
    def __init__(self) -> None:
        self.pipe = None           # The full diffusers pipeline object
        self.model_variant = None  # e.g. "gguf_local", HF model id
        self.base_model = None     # base model path (GGUF)
        self.gguf_path = None      # GGUF file path
        self.is_bnb = None         # Whether BnB quantisation is active
        self.is_img2img = None     # Pipeline class type
        self.gpu_id = None         # Which GPU has offload hooks
        self.lora_path = None      # Currently loaded LoRA adapter path
        self.lora_adapter_names = [] # Names of loaded LoRA adapters


_GLOBAL_CACHE = _GenCache()


def _log_gpu_memory(label: str) -> None:
    """Log per-GPU memory usage at INFO level."""
    if not torch.cuda.is_available():
        return
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free, total = torch.cuda.mem_get_info(i)
        parts.append(
            f"cuda:{i}[alloc={alloc:.2f}GB resv={reserved:.2f}GB "
            f"free={free / 1024**3:.2f}GB/{total / 1024**3:.2f}GB]"
        )
    logger.info("[VRAM %s] %s", label, "  |  ".join(parts))


def _free_gpu_memory(label: str = "") -> None:
    """Free GPU memory via gc + empty_cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_gpu_memory(f"after-free {label}" if label else "after-free")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not found, edge inpaint alpha fill will fall back to grey")



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


def load_image_with_alpha_fill(path: str, fill_mode: str = "grey") -> object:
    """Load an image and fill transparent areas based on the specified mode.

    Modes:
        - "white": Fill with white (255, 255, 255)
        - "grey": Fill with grey (128, 128, 128)
        - "noise": Fill with uniform random noise
        - "edge_inpaint": extend edges usage cv2.inpaint
        - "noise_edge_blend": 80% noise + 20% edge_inpaint
    """
    from PIL import Image

    if not Path(path).is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    if img.mode != "RGBA" and "A" not in img.mode:
        return img.convert("RGB")

    # If simple color modes, use PIL compositing
    if fill_mode == "white":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    elif fill_mode == "grey":
        bg = Image.new("RGB", img.size, (128, 128, 128))
        bg.paste(img, mask=img.split()[-1])
        return bg

    # Complex modes require numpy/cv2
    if not HAS_CV2:
        # Fallback to grey
        bg = Image.new("RGB", img.size, (128, 128, 128))
        bg.paste(img, mask=img.split()[-1])
        return bg

    # Convert to numpy (RGBA)
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = (alpha < 255).astype(np.uint8) * 255  # 255 where transparent

    if fill_mode == "noise":
        noise = np.random.randint(0, 256, rgb.shape, dtype=np.uint8)
        # Combine: where alpha is opaque keep rgb, else noise
        # Actually easier: just fill noise everywhere then paste rgb on top
        # But let's do mask math for clarity
        # alpha is 0..255. Let's threshold strict 255 for "opaque" vs "transparent"
        # or just composite properly using alpha channel
        fill = noise
    elif fill_mode == "edge_inpaint":
        # cv2.inpaint requires BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Inpaint where mask is 255
        inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
        fill = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    elif fill_mode == "noise_edge_blend":
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        
        noise = np.random.randint(0, 256, rgb.shape, dtype=np.uint8)
        # 80% noise, 20% inpaint
        fill = cv2.addWeighted(noise, 0.8, inpainted_rgb, 0.2, 0)
    else:
        # Unknown mode, fallback to grey
        return load_image_with_alpha_fill(path, "grey")

    # Composite: output = rgb * alpha + fill * (1 - alpha)
    # Normalize alpha to 0..1
    norm_alpha = alpha.astype(float) / 255.0
    norm_alpha = np.stack([norm_alpha] * 3, axis=-1)

    out = rgb.astype(float) * norm_alpha + fill.astype(float) * (1 - norm_alpha)
    return Image.fromarray(out.astype(np.uint8))



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
        self._lora_adapter_names: list[str] = []
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
        _log_gpu_memory("run_generation:entry")

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
        _log_gpu_memory("pipeline loaded")

        # Load Fun ControlNet if enabled
        fun_cn = None
        hint_tokens = None
        if is_controlnet:
            fun_cn, hint_tokens = self._load_fun_controlnet()

        self._raise_if_cancelled()

        # ── LoRA handling (cache-aware) ───────────────────────────────────
        lora_path = self._settings.lora_path
        lora_adapter_names: list[str] = []

        want_lora = bool(lora_path and Path(lora_path).is_file())
        have_lora = _GLOBAL_CACHE.lora_path is not None
        same_lora = have_lora and _GLOBAL_CACHE.lora_path == lora_path

        if want_lora and same_lora:
            # Reuse cached LoRA — just reset adapter scales
            logger.info("Reusing cached LoRA adapter: %s", lora_path)
            self._lora_active = True
            lora_adapter_names = _GLOBAL_CACHE.lora_adapter_names
            self._lora_adapter_names = lora_adapter_names
        elif want_lora:
            # Different LoRA or first load
            if have_lora:
                self.stage_changed.emit("Unloading previous LoRA...")
                try:
                    self._pipe.unload_lora_weights()
                except Exception as e:
                    logger.debug("Failed to unload old LoRA: %s", e)
                _GLOBAL_CACHE.lora_path = None

            self.stage_changed.emit("Loading LoRA adapter...")
            self._pipe.load_lora_weights(lora_path)
            self._lora_active = True
            _GLOBAL_CACHE.lora_path = lora_path

            # Query real adapter name(s) — diffusers may suffix them
            try:
                adapters_dict = self._pipe.get_list_adapters()
                seen: set[str] = set()
                lora_adapter_names = []
                for names in adapters_dict.values():
                    for n in names:
                        if n not in seen:
                            seen.add(n)
                            lora_adapter_names.append(n)
                if not lora_adapter_names:
                    lora_adapter_names = ["default"]
            except Exception:
                lora_adapter_names = ["default"]
            self._lora_adapter_names = lora_adapter_names
            _GLOBAL_CACHE.lora_adapter_names = lora_adapter_names
            logger.info("LoRA adapters loaded: %s", lora_adapter_names)
        else:
            # No LoRA wanted
            if have_lora:
                self.stage_changed.emit("Unloading LoRA...")
                try:
                    self._pipe.unload_lora_weights()
                except Exception as e:
                    logger.debug("Failed to unload LoRA: %s", e)
                _GLOBAL_CACHE.lora_path = None
                _GLOBAL_CACHE.lora_adapter_names = []
            self._lora_active = False
            self._lora_adapter_names = []

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
                    self._pipe.set_adapters(lora_adapter_names, [scale] * len(lora_adapter_names))
                else:
                    self._pipe.set_adapters(lora_adapter_names, [0.0] * len(lora_adapter_names))
            self._emit_vram()
            return callback_kwargs

        from qwenimg2512.samplers import get_sampler
        custom_sampler = None
        if self._settings.sampler_name != "euler":
            custom_sampler = get_sampler(self._settings.sampler_name)
            if custom_sampler is None:
                logger.warning("Sampler %s not found, falling back to default scheduler", self._settings.sampler_name)

        _log_gpu_memory("before inference")

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

            input_image = load_image_with_alpha_fill(
                self._settings.input_image_path, self._settings.alpha_fill
            )
            input_image = resize_and_center_crop(input_image, width, height)
            gen_kwargs["image"] = input_image
            gen_kwargs["strength"] = self._settings.img2img_strength

        # Free intermediate tensors before inference
        _free_gpu_memory("before inference")
        self._emit_vram()

        logger.info("Starting pipeline.__call__ ...")
        from qwenimg2512.pipeline_patch import apply_custom_sampler
        with apply_custom_sampler(self._pipe, custom_sampler):
            output = self._pipe(**gen_kwargs)
        _log_gpu_memory("inference done")

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
        from diffusers import GGUFQuantizationConfig, QwenImagePipeline, QwenImageTransformer2DModel

        gguf_path = self._model_paths.diffusion_gguf
        base_model = self._model_paths.base_model_dir or "Qwen/Qwen-Image-2512"

        # Determine GPU assignment
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
            best_gpu = free.index(max(free))
        else:
            best_gpu = 0

        # ── Check cache ───────────────────────────────────────────────────
        c = _GLOBAL_CACHE
        if (
            c.pipe is not None
            and c.model_variant == "gguf_local"
            and c.gguf_path == gguf_path
            and c.base_model == base_model
            and c.is_img2img == img2img
            and c.gpu_id == best_gpu
        ):
            logger.info("Using cached GGUF pipeline (gpu=%d, img2img=%s)", best_gpu, img2img)
            self.stage_changed.emit("Using cached model ✓")
            return c.pipe

        # ── Partial hit: same transformer, different pipeline class ───────
        reuse_transformer = (
            c.pipe is not None
            and c.model_variant == "gguf_local"
            and c.gguf_path == gguf_path
            and c.base_model == base_model
            and c.is_img2img != img2img
        )

        if reuse_transformer:
            logger.info("Reusing cached transformer, rebuilding pipeline (img2img=%s → %s)", c.is_img2img, img2img)
            self.stage_changed.emit("Rebuilding pipeline (reusing transformer)...")
            transformer = c.pipe.transformer
            # Unload LoRA from old pipe before rebuilding
            if c.lora_path:
                try:
                    c.pipe.unload_lora_weights()
                except Exception:
                    pass
                c.lora_path = None
        else:
            # ── Full cache miss — load everything from scratch ────────────
            if c.pipe is not None:
                logger.info("Cache invalidated — full reload")
                del c.pipe
                c.pipe = None
                _free_gpu_memory("cache invalidated")

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

        if num_gpus >= 2:
            logger.info(
                "Multi-GPU detected: using cuda:%d (%.1f GB free) for offloading",
                best_gpu,
                free[best_gpu] / 1024**3,
            )
            pipe.enable_model_cpu_offload(gpu_id=best_gpu)
        else:
            pipe.enable_model_cpu_offload()

        # ── Store in cache ────────────────────────────────────────────────
        c.pipe = pipe
        c.model_variant = "gguf_local"
        c.gguf_path = gguf_path
        c.base_model = base_model
        c.is_bnb = False
        c.is_img2img = img2img
        c.gpu_id = best_gpu
        c.lora_path = None  # LoRA is loaded separately
        c.lora_adapter_names = []
        _log_gpu_memory("gguf pipeline cached")

        return pipe

    def _load_hf_pipeline(self, model_id: str, is_bnb: bool, img2img: bool) -> object:
        # ── Check cache ───────────────────────────────────────────────────
        c = _GLOBAL_CACHE
        if (
            c.pipe is not None
            and c.model_variant == model_id
            and c.is_bnb == is_bnb
            and c.is_img2img == img2img
        ):
            logger.info("Using cached HF pipeline (%s, img2img=%s)", model_id, img2img)
            self.stage_changed.emit("Using cached model ✓")
            return c.pipe

        # ── Cache miss — full reload ──────────────────────────────────────
        if c.pipe is not None:
            logger.info("Cache invalidated — full reload (HF)")
            del c.pipe
            c.pipe = None
            _free_gpu_memory("cache invalidated")

        self.stage_changed.emit("Loading HF pipeline...")
        load_kwargs: dict = {"torch_dtype": torch.bfloat16}

        if img2img:
            from diffusers import QwenImageImg2ImgPipeline

            pipe = QwenImageImg2ImgPipeline.from_pretrained(model_id, **load_kwargs)
        else:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)

        if not is_bnb:
            pipe = pipe.to("cuda")

        # ── Store in cache ────────────────────────────────────────────────
        c.pipe = pipe
        c.model_variant = model_id
        c.gguf_path = None
        c.base_model = model_id
        c.is_bnb = is_bnb
        c.is_img2img = img2img
        c.gpu_id = 0
        c.lora_path = None
        c.lora_adapter_names = []
        _log_gpu_memory("hf pipeline cached")

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
            sorted_gpus = sorted(range(num_gpus), key=lambda i: free[i], reverse=True)
            cn_gpu = sorted_gpus[1] if num_gpus > 1 else sorted_gpus[0]
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

        image = load_image_with_alpha_fill(image_path, self._settings.alpha_fill)
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
        """Clean up per-generation resources while preserving the cached pipeline."""
        try:
            # Remove ControlNet hooks (per-generation, always cleaned)
            if self._cn_hooks:
                from qwenimg2512.fun_controlnet import remove_fun_controlnet_hooks

                remove_fun_controlnet_hooks(self._cn_hooks)
                self._cn_hooks = None
                self._cn_state = None

            # Release local reference — the global cache preserves the pipeline
            self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_gpu_memory("cleanup")
        except Exception as e:
            logger.warning("Error during cleanup: %s", e)
