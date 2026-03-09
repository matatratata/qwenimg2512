"""Background worker thread for Qwen-Image-Edit-2509 generation (BnB 8-bit).

VRAM strategy
─────────────
The BnB-8-bit transformer is ~23 GB and **cannot be moved off its GPU** once
materialised.  The text encoder (Qwen2.5-VL-7B) is ~14 GB in bf16 and also
needs a GPU for encoding.  They cannot coexist on a single 24 GB card.

Solution — *pre-encode-and-discard*:
  1. Lock the thread's default CUDA device to the transformer GPU so that
     BnB and PEFT implicit allocations land in the right place.
  2. Switch CUDA context to the encoder GPU, encode prompts, switch back.
  3. Delete text encoder, gc.collect + empty_cache.
  4. Load transformer (BnB 8-bit) on the "best" GPU.
  5. Set ``pipe.hf_device_map`` to prevent Diffusers from calling
     ``pipe.to("cpu")`` during LoRA loading (which orphans BnB weights).
  6. Run the denoising loop with pre-computed embeddings.
"""

from __future__ import annotations

import gc
import logging
import random
import time
from pathlib import Path

import torch
from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ASPECT_RATIOS, Edit2509Settings, ModelPaths

logger = logging.getLogger(__name__)


# ── Global Model Cache ────────────────────────────────────────────────────
# Keeps models in VRAM between requests to prevent reloading and OOMs.
# The user prefers text_encoder on cuda:0 and transformer on cuda:1.
class _Edit2509Cache:
    def __init__(self):
        self.text_encoder = None
        self.processor = None
        self.tokenizer = None
        self.encoder_device = None

        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.transformer_gpu = None
        self.base_model_path = None
        
        # Track which LoRAs are currently merged into the transformer
        self.active_loras = set()

_GLOBAL_CACHE = _Edit2509Cache()


# ── module-level helpers ──────────────────────────────────────────────────

def _log_gpu_memory(label: str) -> None:
    """Log per-GPU memory usage at INFO level with a descriptive label."""
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
    """Free GPU memory via gc + empty_cache.

    ``torch.cuda.empty_cache()`` is safe — it only releases *unoccupied*
    cached memory and will never evict actively allocated weights.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_gpu_memory(f"after-free {label}" if label else "after-free")


class Edit2509Worker(QThread):
    """Runs Qwen-Image-Edit-2509 inference in a background thread."""

    progress_updated = Signal(int, int, str)
    stage_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)
    vram_updated = Signal(float)

    def __init__(self, settings: Edit2509Settings, model_paths: ModelPaths) -> None:
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

    # ── helpers ────────────────────────────────────────────────────────────

    def _pick_gpus(self) -> tuple[int, str]:
        """Return (transformer_gpu_id, encoder_device_str)."""
        num_gpus = torch.cuda.device_count()
        _free_gpu_memory("pick_gpus:start")

        if num_gpus >= 2:
            # Use fixed GPU assignments for optimal caching.
            # Transformer goes to cuda:1, Text Encoder goes to cuda:0
            best_gpu = 1
            encoder_device = "cuda:0"
            logger.info(
                "GPU selection: transformer → cuda:%d, text-encoder → %s (cached layout)",
                best_gpu, encoder_device,
            )
        else:
            best_gpu = 0
            encoder_device = "cpu"
        return best_gpu, encoder_device

    def _emit_vram(self) -> None:
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                self.vram_updated.emit(vram_gb)
        except Exception:
            pass

    # ── main generation flow ──────────────────────────────────────────────

    def _run_generation(self) -> None:
        from PIL import Image

        _log_gpu_memory("run_generation:entry")

        best_gpu, encoder_device = self._pick_gpus()
        target_device = f"cuda:{best_gpu}"

        # ⚠️ CRITICAL: Lock this thread's default CUDA device.
        # Without this, BnB and PEFT implicit workspace allocations
        # land on cuda:0 (the thread default) instead of the target GPU.
        if torch.cuda.is_available():
            torch.cuda.set_device(best_gpu)
            logger.info("Thread CUDA device locked to cuda:%d", best_gpu)

        base_model = self._model_paths.edit_2509_base_model_dir or "Qwen/Qwen-Image-Edit-2509"
        use_telestyle = self._settings.use_telestyle

        if use_telestyle:
            fused_dir = self._model_paths.edit_2509_telestyle_fused_dir
            if fused_dir and Path(fused_dir).is_dir():
                base_model = fused_dir
                logger.info("TeleStyle ON → using fused model: %s", base_model)
            else:
                logger.warning(
                    "TeleStyle enabled but fused model not found at %s — "
                    "falling back to normal base model + LoRA loading",
                    fused_dir,
                )
                use_telestyle = False

        logger.info("Transformer → %s | Text-encoder → %s", target_device, encoder_device)

        # ── 1.  Load reference images ─────────────────────────────────────
        self.stage_changed.emit("Processing reference images...")
        ref_images: list[Image.Image] = []
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
        self._raise_if_cancelled()

        # ── 2.  Pre-encode prompts on the encoder device ──────────────────
        #    Switch CUDA context to encoder GPU, then restore after encoding.
        self.stage_changed.emit("Loading text encoder...")
        if encoder_device.startswith("cuda"):
            torch.cuda.set_device(int(encoder_device.split(":")[-1]))
            logger.info("Switched CUDA context → %s for text encoding", encoder_device)

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._pre_encode(
            base_model=base_model,
            encoder_device=encoder_device,
            ref_images=ref_images,
        )
        self._raise_if_cancelled()

        # Restore the target device context for the rest of generation
        if torch.cuda.is_available():
            torch.cuda.set_device(best_gpu)
            logger.info("Restored CUDA context → cuda:%d for generation", best_gpu)
        _log_gpu_memory("after pre-encode + free")

        # ── 3.  Load transformer (BnB-8-bit, pinned to GPU) ──────────────
        self.stage_changed.emit("Loading transformer (BnB 8-bit)...")
        self._emit_vram()
        pipe = self._build_pipeline(base_model, target_device)
        self._pipe = pipe
        self._raise_if_cancelled()
        _log_gpu_memory("after build_pipeline")

        # ── 4.  Generate ──────────────────────────────────────────────────
        self.stage_changed.emit("Generating image...")

        width, height = ASPECT_RATIOS[self._settings.aspect_ratio]
        seed = self._settings.seed if self._settings.seed >= 0 else random.randint(0, 2**32 - 1)

        step_start = time.monotonic()

        def step_callback(pipe_: object, step: int, timestep: object, cb_kwargs: dict) -> dict:
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
            if step % 10 == 0:
                _log_gpu_memory(f"step {step}/{self._settings.num_inference_steps}")
            return cb_kwargs

        logger.info("Moving prompt_embeds to %s (shape %s, dtype %s)",
                     target_device, list(prompt_embeds.shape), prompt_embeds.dtype)
        from qwenimg2512.samplers import get_sampler
        custom_sampler = None
        if self._settings.sampler_name != "euler":
            custom_sampler = get_sampler(self._settings.sampler_name)
            if custom_sampler is None:
                logger.warning("Sampler %s not found, falling back to default scheduler", self._settings.sampler_name)

        gen_kwargs: dict = {
            "image": ref_images,
            "width": width,
            "height": height,
            "num_inference_steps": self._settings.num_inference_steps,
            "true_cfg_scale": self._settings.true_cfg_scale,
            "guidance_scale": None,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "callback_on_step_end": step_callback,
            "prompt_embeds": prompt_embeds.to(target_device),
            "prompt_embeds_mask": prompt_mask.to(target_device),
        }
        if neg_embeds is not None:
            logger.info("Moving negative_prompt_embeds to %s", target_device)
            gen_kwargs["negative_prompt_embeds"] = neg_embeds.to(target_device)
            gen_kwargs["negative_prompt_embeds_mask"] = neg_mask.to(target_device)
        else:
            logger.info("No negative prompt embeddings (neg_prompt=%r, true_cfg=%.1f)",
                        self._settings.negative_prompt, self._settings.true_cfg_scale)

        _free_gpu_memory("before inference")
        self._emit_vram()

        logger.info("Starting pipeline.__call__ ...")
        import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_edit_module
        original_vae_size = qwen_edit_module.VAE_IMAGE_SIZE
        num_refs = len(ref_images)
        
        # BUG FIX: Set VAE_IMAGE_SIZE exactly to the output area.
        # Qwen-Image uses absolute 2D RoPE embeddings. To align the reference image
        # correctly with the output canvas without producing a "nested thumbnail", 
        # the VAE_IMAGE_SIZE MUST match the output area exactly.
        output_area = width * height
        per_ref_area = output_area
        
        if per_ref_area != original_vae_size:
            qwen_edit_module.VAE_IMAGE_SIZE = per_ref_area
            logger.info(
                "Scaled VAE_IMAGE_SIZE: output=%dx%d refs=%d → per_ref_area=%d (~%dx%d)",
                width, height, num_refs, per_ref_area,
                int(per_ref_area ** 0.5), int(per_ref_area ** 0.5),
            )

        from qwenimg2512.pipeline_patch import apply_custom_sampler, apply_custom_schedule, apply_ffn_chunking, apply_block_swap, apply_attn_chunking, apply_smc_cfg
        try:
            with apply_smc_cfg(pipe, getattr(self._settings, "smc_cfg_enabled", False), getattr(self._settings, "smc_k", 0.2), getattr(self._settings, "smc_lambda", 5.0)):
                with apply_ffn_chunking(pipe, self._settings.ffn_chunk_size):
                    with apply_block_swap(pipe, self._settings.blocks_to_swap):
                        with apply_attn_chunking(pipe, getattr(self._settings, 'attn_chunk_size', 0)):
                            with apply_custom_sampler(pipe, custom_sampler):
                                with apply_custom_schedule(pipe, self._settings.schedule_name):
                                    output = pipe(**gen_kwargs)
        finally:
            qwen_edit_module.VAE_IMAGE_SIZE = original_vae_size
            
        _log_gpu_memory("inference:done")
        self._raise_if_cancelled()

        image = output.images[0]
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        refs = [self._settings.ref_image_1, self._settings.ref_image_2, self._settings.ref_image_3]
        strengths = [self._settings.ref_strength_1, self._settings.ref_strength_2, self._settings.ref_strength_3] if hasattr(self._settings, 'ref_strength_1') else [1.0, 1.0, 1.0]
        
        ref_parts = []
        for i, (r, s) in enumerate(zip(refs, strengths)):
            if r and Path(r).is_file():
                # Format to 2 decimal places, pad left with 0: e.g. 0.70 becomes '070', 1.0 becomes '100'
                str_formatted = f"{int(s * 100):03d}"
                ref_parts.append(f"r{i+1}{str_formatted}")
        ref_str = "".join(ref_parts) if ref_parts else "noref"

        kind = "edit2509"
        cfg_str = f"cfg{self._settings.true_cfg_scale}"
        g_str = f"g{self._settings.guidance_scale}"
        samp_str = f"samp_{self._settings.sampler_name}"
        sched_str = f"sched_{self._settings.schedule_name}"
        
        filename = f"{kind}_{ref_str}_{cfg_str}_{g_str}_{samp_str}_{sched_str}_{seed}_{width}x{height}_s{self._settings.num_inference_steps}.png"
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

    # ── sub-steps ─────────────────────────────────────────────────────────

    def _pre_encode(
        self,
        *,
        base_model: str,
        encoder_device: str,
        ref_images: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode positive (+ optional negative) prompts on *encoder_device*.

        Returns tensors on CPU. Text encoder is cached.
        """
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            CONDITION_IMAGE_SIZE,
            calculate_dimensions,
        )
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

        _free_gpu_memory("pre_encode:start")

        if _GLOBAL_CACHE.text_encoder is None or _GLOBAL_CACHE.base_model_path != base_model or _GLOBAL_CACHE.encoder_device != encoder_device:
            self.stage_changed.emit(f"Loading text encoder on {encoder_device}...")
            logger.info("Loading Qwen2.5-VL text_encoder → %s ...", encoder_device)
            _GLOBAL_CACHE.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            ).to(encoder_device)
            _GLOBAL_CACHE.tokenizer = Qwen2Tokenizer.from_pretrained(base_model, subfolder="tokenizer")
            _GLOBAL_CACHE.processor = Qwen2VLProcessor.from_pretrained(base_model, subfolder="processor")
            _GLOBAL_CACHE.encoder_device = encoder_device
            if _GLOBAL_CACHE.base_model_path != base_model:
                _GLOBAL_CACHE.transformer = None  # Invalidate transformer as well
            _GLOBAL_CACHE.base_model_path = base_model
        else:
            logger.info("Using cached text encoder on %s.", encoder_device)

        text_encoder = _GLOBAL_CACHE.text_encoder
        tokenizer = _GLOBAL_CACHE.tokenizer
        processor = _GLOBAL_CACHE.processor

        _log_gpu_memory("text_encoder active")

        self._raise_if_cancelled()
        self._emit_vram()

        # ── Replicate the pipeline's condition-image preprocessing ────────
        from diffusers.image_processor import VaeImageProcessor
        vae_scale_factor = 8
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        condition_images = []
        for img in ref_images:
            w, h = img.size
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
            condition_images.append(image_processor.resize(img, ch, cw))

        # ── Template matching the pipeline ────────────────────────────────
        prompt_template = (
            "<|im_start|>system\nDescribe the key features of the input image "
            "(color, shape, size, texture, objects, background), then explain how "
            "the user's text instruction should alter or modify the image. Generate "
            "a new image that meets the user's requirements while maintaining "
            "consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        neg_template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )

        def _encode_one(prompt_text: str, label: str, is_negative: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            logger.info("Encoding %s on %s: %r", label, encoder_device, prompt_text[:80])

            if is_negative:
                txt = [neg_template.format(prompt_text)]
                images = None
                prefix = neg_template.split("{}")[0]
                drop_idx = len(tokenizer.encode(prefix))
            else:
                img_prompt = "".join(
                    f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>\n"
                    for i in range(len(condition_images))
                )
                txt = [prompt_template.format(img_prompt + prompt_text)]
                images = condition_images
                prefix = prompt_template.split("{}")[0]
                drop_idx = len(tokenizer.encode(prefix))

            if images is not None:
                model_inputs = processor(
                    text=txt,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ).to(encoder_device)
            else:
                model_inputs = processor(
                    text=txt,
                    padding=True,
                    return_tensors="pt",
                ).to(encoder_device)

            _log_gpu_memory(f"{label}:inputs ready")

            with torch.no_grad():
                kwargs = {
                    "input_ids": model_inputs.input_ids,
                    "attention_mask": model_inputs.attention_mask,
                    "output_hidden_states": True,
                }
                if images is not None:
                    if hasattr(model_inputs, "pixel_values") and model_inputs.pixel_values is not None:
                        kwargs["pixel_values"] = model_inputs.pixel_values
                    if hasattr(model_inputs, "image_grid_thw") and model_inputs.image_grid_thw is not None:
                        kwargs["image_grid_thw"] = model_inputs.image_grid_thw

                outputs = text_encoder(**kwargs)
            _log_gpu_memory(f"{label}:forward done")

            hidden = outputs.hidden_states[-1]
            mask = model_inputs.attention_mask
            bool_mask = mask.bool()
            valid_lens = bool_mask.sum(dim=1)
            selected = hidden[bool_mask]
            splits = torch.split(selected, valid_lens.tolist(), dim=0)
            splits = [s[drop_idx:] for s in splits]
            attn_masks = [torch.ones(s.size(0), dtype=torch.long, device=s.device) for s in splits]
            max_seq = max(s.size(0) for s in splits)
            embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq - u.size(0), u.size(1))]) for u in splits]
            )
            attn = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq - u.size(0))]) for u in attn_masks]
            )
            embeds_cpu, attn_cpu = embeds.cpu(), attn.cpu()
            del outputs, hidden, mask, model_inputs, selected, splits, embeds, attn
            logger.info("%s encoded → embeds shape %s", label, list(embeds_cpu.shape))
            return embeds_cpu, attn_cpu

        self.stage_changed.emit("Encoding prompt...")
        prompt_embeds, prompt_mask = _encode_one(self._settings.prompt, "positive", is_negative=False)

        neg_embeds, neg_mask = None, None
        neg_prompt = self._settings.negative_prompt
        if neg_prompt and self._settings.true_cfg_scale > 1:
            self.stage_changed.emit("Encoding negative prompt...")
            neg_embeds, neg_mask = _encode_one(neg_prompt, "negative", is_negative=True)

        _free_gpu_memory("encoding complete")
        self._emit_vram()

        return prompt_embeds, prompt_mask, neg_embeds, neg_mask

    def _build_pipeline(self, base_model: str, target_device: str) -> object:
        """Build the diffusers pipeline (transformer + VAE + scheduler)."""
        from diffusers import (
            AutoencoderKLQwenImage,
            BitsAndBytesConfig,
            FlowMatchEulerDiscreteScheduler,
            QwenImageEditPlusPipeline,
            QwenImageTransformer2DModel,
        )
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

        _log_gpu_memory("build_pipeline:start")

        # ── Parse the GPU id from the device string ───────────────────────
        transformer_gpu = int(target_device.split(":")[-1])

        # ── Check Cache ───────────────────────────────────────────────────
        if _GLOBAL_CACHE.transformer is not None and _GLOBAL_CACHE.transformer_gpu == transformer_gpu and _GLOBAL_CACHE.base_model_path == base_model:
            logger.info("Using cached transformer and VAE on %s", target_device)
            transformer = _GLOBAL_CACHE.transformer
            vae = _GLOBAL_CACHE.vae
            scheduler = _GLOBAL_CACHE.scheduler
            tokenizer = _GLOBAL_CACHE.tokenizer
            processor = _GLOBAL_CACHE.processor
        else:
            logger.info("Loading transformer (BnB 8-bit) → %s ...", target_device)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            transformer = QwenImageTransformer2DModel.from_pretrained(
                base_model,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map=target_device,
            )
            _log_gpu_memory("transformer loaded")

            # FFN chunking – reduces peak activation VRAM at a small speed cost
            try:
                transformer.enable_forward_chunking(chunk_size=1)
                logger.info("Transformer forward-chunking enabled (chunk_size=1)")
            except AttributeError:
                logger.warning("enable_forward_chunking not available — skipping")

            self._raise_if_cancelled()

            logger.info("Loading VAE → %s ...", target_device)
            vae = AutoencoderKLQwenImage.from_pretrained(
                base_model, subfolder="vae", torch_dtype=torch.bfloat16,
            ).to(target_device)
            _log_gpu_memory("vae loaded")

            try:
                vae.enable_tiling()
                logger.info("VAE tiling enabled")
            except AttributeError:
                pass
            try:
                vae.enable_slicing()
                logger.info("VAE slicing enabled")
            except AttributeError:
                pass

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                base_model, subfolder="scheduler",
            )
            if _GLOBAL_CACHE.tokenizer is None:
                tokenizer = Qwen2Tokenizer.from_pretrained(base_model, subfolder="tokenizer")
                processor = Qwen2VLProcessor.from_pretrained(base_model, subfolder="processor")
            else:
                tokenizer = _GLOBAL_CACHE.tokenizer
                processor = _GLOBAL_CACHE.processor

            # Save to Cache
            _GLOBAL_CACHE.transformer = transformer
            _GLOBAL_CACHE.vae = vae
            _GLOBAL_CACHE.scheduler = scheduler
            _GLOBAL_CACHE.tokenizer = tokenizer
            _GLOBAL_CACHE.processor = processor
            _GLOBAL_CACHE.transformer_gpu = transformer_gpu
            _GLOBAL_CACHE.base_model_path = base_model
            _GLOBAL_CACHE.active_loras = set()

        logger.info("Assembling pipeline (text_encoder=None) ...")
        try:
            pipe = QwenImageEditPlusPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=None,
                tokenizer=tokenizer,
                processor=processor,
                scheduler=scheduler,
            )
            logger.info("Pipeline accepted text_encoder=None")
        except Exception as exc:
            logger.warning("Pipeline rejected None text_encoder (%s), loading CPU stub ...", exc)
            _te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            )
            pipe = QwenImageEditPlusPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=_te,
                tokenizer=tokenizer,
                processor=processor,
                scheduler=scheduler,
            )
            del _te
            pipe.text_encoder = None
            _free_gpu_memory("stub text_encoder removed")

        _log_gpu_memory("pipeline assembled")

        try:
            logger.info("Pipeline._execution_device = %s", pipe._execution_device)
        except Exception:
            pass

        # ── VAE Offload Optimization ──────────────────────────────────────
        # To save ~300MB VRAM during the transformer denoising loop, we dynamically
        # move the VAE to CPU when it's not needed. It only needs GPU for
        # `prepare_latents` (encoding condition images) and `decode` (at the end).
        logger.info("Applying dynamic VAE CPU offload to save VRAM")
        orig_prepare_latents = pipe.prepare_latents

        def _patched_prepare_latents(*args, **kwargs):
            vae.to(target_device)
            res = orig_prepare_latents(*args, **kwargs)
            vae.to("cpu")
            torch.cuda.empty_cache()
            return res

        pipe.prepare_latents = _patched_prepare_latents

        orig_vae_decode = vae.decode

        def _patched_vae_decode(*args, **kwargs):
            vae.to(target_device)
            return orig_vae_decode(*args, **kwargs)

        vae.decode = _patched_vae_decode

        # ⚠️ CRITICAL: Prevent Diffusers CPU offload bug.
        # Without hf_device_map, load_lora_weights() sees a manually-assembled
        # pipeline and forcefully calls pipe.to("cpu") to "save VRAM".
        # BnB blocks the 8-bit weights from moving, but everything else moves,
        # orphaning the tensor references and making the weights "disappear".
        pipe.hf_device_map = {"": target_device}
        logger.info("Set pipe.hf_device_map = %s (prevents Diffusers CPU offload bug)", pipe.hf_device_map)

        # ── LoRA(s) — skipped when TeleStyle is active (weights are fused) ──
        if self._settings.use_telestyle and Path(base_model).name.endswith("Fused"):
            logger.info("TeleStyle fused — skipping LoRA loading")
        else:
            # Revert transformer to its base state (removing any cached PEFT adapters)
            try:
                # Get a list of all current adapters and delete them
                if hasattr(pipe.transformer, "peft_config"):
                    existing_adapters = list(pipe.transformer.peft_config.keys())
                    if existing_adapters:
                        logger.info("Unloading existing PEFT adapters: %s", existing_adapters)
                        pipe.delete_adapters(existing_adapters)
                else:
                    pipe.unload_lora_weights()
            except Exception as e:
                logger.debug("Failed to clear previous LoRAs: %s", e)

            lora_path = self._settings.lora_path
            lora_path_2 = self._settings.lora_path_2
            adapter_names: list[str] = []
            adapter_scales: list[float] = []

            if lora_path and Path(lora_path).is_file():
                self.stage_changed.emit("Loading LoRA adapter 1...")
                pipe.load_lora_weights(lora_path, adapter_name="lora_1")
                adapter_names.append("lora_1")
                adapter_scales.append(self._settings.lora_scale_start)
                logger.info("LoRA 1 loaded from %s (scale=%.2f)", lora_path, self._settings.lora_scale_start)
                _log_gpu_memory("lora_1 loaded")

            if lora_path_2 and Path(lora_path_2).is_file():
                self.stage_changed.emit("Loading LoRA adapter 2...")
                pipe.load_lora_weights(lora_path_2, adapter_name="lora_2")
                adapter_names.append("lora_2")
                adapter_scales.append(self._settings.lora_scale_start_2)
                logger.info("LoRA 2 loaded from %s (scale=%.2f)", lora_path_2, self._settings.lora_scale_start_2)
                _log_gpu_memory("lora_2 loaded")

            if len(adapter_names) > 1:
                self.stage_changed.emit("Merging LoRA adapters in-memory (SVD)...")
                try:
                    # Combine multiple LoRAs into a single adapter dynamically.
                    # SVD handles mismatched ranks (r=32 vs r=64) and runs
                    # layer-by-layer with minimal VRAM overhead (<50MB spike).
                    # svd_rank defaults to the max rank of the input adapters.
                    pipe.transformer.add_weighted_adapter(
                        adapters=adapter_names,
                        weights=adapter_scales,
                        combination_type="svd",
                        adapter_name="merged_lora",
                    )

                    # Activate ONLY the merged adapter (UI slider strengths are
                    # already baked into the SVD weights via adapter_scales).
                    pipe.set_adapters(["merged_lora"], adapter_weights=[1.0])

                    # Delete the original unmerged adapters to reclaim VRAM
                    pipe.delete_adapters(adapter_names)

                    logger.info("Successfully merged LoRAs into 'merged_lora' via SVD")
                    _log_gpu_memory("loras merged and freed")

                except Exception as exc:
                    logger.warning("SVD merge failed: %s — falling back to dual-adapter mode", exc)
                    pipe.set_adapters(adapter_names, adapter_scales)

            elif adapter_names:
                self.stage_changed.emit("Activating LoRA adapter(s)...")
                pipe.set_adapters(adapter_names, adapter_scales)
                logger.info("LoRA adapters active: %s", adapter_names)

        _free_gpu_memory("pipeline ready")
        logger.info("Pipeline ready on %s (pre-encoded embeddings)", target_device)
        return pipe

    def _cleanup(self) -> None:
        logger.info("Cleanup: freeing intermediate tensors ...")
        try:
            if self._pipe:
                # We do NOT delete the transformer, VAE, or text_encoder
                # from the cache! We just let the pipeline object itself be freed,
                # which releases the intermediate attention buffers.
                # To be safe against diffusers holding references, we clear its links.
                self._pipe.transformer = None
                self._pipe.vae = None
                self._pipe.text_encoder = None
                self._pipe.tokenizer = None
                self._pipe.processor = None
                self._pipe.scheduler = None

                del self._pipe
                self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                # Flush intermediate tensors out of the cache.
                torch.cuda.empty_cache()
            _log_gpu_memory("cleanup")
        except Exception as e:
            logger.warning("Error during cleanup: %s", e)
