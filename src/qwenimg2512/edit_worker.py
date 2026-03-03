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

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._pre_encode(ref_images)
        self._raise_if_cancelled()

        from qwenimg2512.samplers import get_sampler
        custom_sampler = None
        if self._settings.sampler_name != "euler":
            custom_sampler = get_sampler(self._settings.sampler_name)
            if custom_sampler is None:
                logger.warning("Sampler %s not found, falling back to default scheduler", self._settings.sampler_name)

        # ── Per-image conditioning strength ───────────────────────────────
        # Monkey-patch _encode_vae_image so the pipeline encodes images
        # normally, but we scale the resulting latents per-image.
        ref_strengths = [
            self._settings.ref_strength_1,
            self._settings.ref_strength_2,
            self._settings.ref_strength_3,
        ][:len(ref_images)]
        any_scaled = any(s < 1.0 for s in ref_strengths)

        if any_scaled:
            pipe = self._pipe
            original_encode = pipe._encode_vae_image
            strength_queue = list(ref_strengths)

            def _scaled_encode(image, generator):
                latents = original_encode(image, generator)
                # Handle batched latents if Diffusers passes a list of images at once
                if hasattr(latents, "shape") and len(latents.shape) >= 4 and latents.shape[0] > 1:
                    num_imgs = latents.shape[0]
                    for i in range(num_imgs):
                        if strength_queue:
                            s = strength_queue.pop(0)
                        else:
                            s = 1.0
                        if s < 1.0:
                            latents[i] = latents[i] * s
                            logger.info("Ref image %d strength=%.2f applied", i + 1, s)
                else:
                    if strength_queue:
                        s = strength_queue.pop(0)
                        if s < 1.0:
                            latents = latents * s
                            logger.info("Ref image strength=%.2f applied", s)
                return latents

            pipe._encode_vae_image = _scaled_encode

        gen_kwargs = {
            "image": ref_images,
            "width": width,
            "height": height,
            "num_inference_steps": self._settings.num_inference_steps,
            "true_cfg_scale": self._settings.true_cfg_scale,
            "guidance_scale": None,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "callback_on_step_end": step_callback,
            "prompt_embeds": prompt_embeds.to(self._pipe._execution_device),
            "prompt_embeds_mask": prompt_mask.to(self._pipe._execution_device),
        }
        if neg_embeds is not None:
            gen_kwargs["negative_prompt_embeds"] = neg_embeds.to(self._pipe._execution_device)
            gen_kwargs["negative_prompt_embeds_mask"] = neg_mask.to(self._pipe._execution_device)

        # Clear VRAM
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self._emit_vram()

        import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_edit_module
        original_vae_size = qwen_edit_module.VAE_IMAGE_SIZE
        num_refs = len(ref_images)
        
        # BUG FIX: Set VAE_IMAGE_SIZE exactly to the output area.
        # Qwen-Image uses absolute 2D RoPE embeddings. To align the reference image
        # correctly with the output canvas without producing a "nested thumbnail", 
        # the VAE_IMAGE_SIZE MUST match the output area exactly.
        output_area = width * height
        per_ref_area = output_area
        # Round down to a multiple of 32² to stay compatible with calculate_dimensions
        grid = 32 * 32
        per_ref_area = (per_ref_area // grid) * grid
        
        if per_ref_area != original_vae_size:
            qwen_edit_module.VAE_IMAGE_SIZE = per_ref_area
            logger.info(
                "Scaled VAE_IMAGE_SIZE: output=%dx%d refs=%d → per_ref_area=%d (~%dx%d)",
                width, height, num_refs, per_ref_area,
                int(per_ref_area ** 0.5), int(per_ref_area ** 0.5),
            )

        from qwenimg2512.pipeline_patch import apply_custom_sampler, apply_custom_schedule, apply_ffn_chunking, apply_block_swap, apply_attn_chunking
        try:
            with apply_ffn_chunking(self._pipe, self._settings.ffn_chunk_size):
                with apply_block_swap(self._pipe, self._settings.blocks_to_swap):
                    with apply_attn_chunking(self._pipe, self._settings.attn_chunk_size):
                        with apply_custom_sampler(self._pipe, custom_sampler):
                            with apply_custom_schedule(self._pipe, self._settings.schedule_name):
                                output = self._pipe(**gen_kwargs)
        finally:
            qwen_edit_module.VAE_IMAGE_SIZE = original_vae_size
            # Restore original _encode_vae_image if we patched it
            if any_scaled:
                self._pipe._encode_vae_image = original_encode
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

    def _pre_encode(self, ref_images: list) -> tuple:
        """Encode positive (+ optional negative) prompts without image conditioning
        in the negative prompt.
        """
        import torch
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            CONDITION_IMAGE_SIZE,
            calculate_dimensions,
        )

        pipe = self._pipe
        device = pipe._execution_device
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        processor = pipe.processor
        
        vae_scale_factor = 8
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        condition_images = []
        for img in ref_images:
            w, h = img.size
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
            condition_images.append(image_processor.resize(img, ch, cw))

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

        def _encode_one(prompt_text: str, is_negative: bool) -> tuple[torch.Tensor, torch.Tensor]:
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
                    text=txt, images=images, padding=True, return_tensors="pt"
                ).to(device)
            else:
                model_inputs = processor(
                    text=txt, padding=True, return_tensors="pt"
                ).to(device)

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
            ).to(dtype=text_encoder.dtype, device=device)
            attn = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq - u.size(0))]) for u in attn_masks]
            ).to(device=device)
            return embeds, attn

        self.stage_changed.emit("Encoding prompt...")
        prompt_embeds, prompt_mask = _encode_one(self._settings.prompt, is_negative=False)

        neg_embeds, neg_mask = None, None
        neg_prompt = self._settings.negative_prompt
        if neg_prompt and self._settings.true_cfg_scale > 1:
            self.stage_changed.emit("Encoding negative prompt...")
            neg_embeds, neg_mask = _encode_one(neg_prompt, is_negative=True)

        return prompt_embeds, prompt_mask, neg_embeds, neg_mask

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

        # Enable VAE tiling to prevent OOM when VAE_IMAGE_SIZE is scaled up to large resolutions
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except AttributeError:
            pass

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
