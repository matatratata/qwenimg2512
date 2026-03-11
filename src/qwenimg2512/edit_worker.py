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
        
        if per_ref_area != original_vae_size:
            qwen_edit_module.VAE_IMAGE_SIZE = per_ref_area
            logger.info(
                "Scaled VAE_IMAGE_SIZE: output=%dx%d refs=%d → per_ref_area=%d (~%dx%d)",
                width, height, num_refs, per_ref_area,
                int(per_ref_area ** 0.5), int(per_ref_area ** 0.5),
            )

        from qwenimg2512.pipeline_patch import apply_custom_sampler, apply_custom_schedule, apply_ffn_chunking, apply_block_swap, apply_attn_chunking, apply_smc_cfg
        try:
            with apply_smc_cfg(self._pipe, getattr(self._settings, "smc_cfg_enabled", False), getattr(self._settings, "smc_k", 0.2), getattr(self._settings, "smc_lambda", 5.0)):
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
        
        refs = [self._settings.ref_image_1, self._settings.ref_image_2, self._settings.ref_image_3]
        strengths = [self._settings.ref_strength_1, self._settings.ref_strength_2, self._settings.ref_strength_3]
        
        ref_parts = []
        for i, (r, s) in enumerate(zip(refs, strengths)):
            if r and Path(r).is_file():
                # Format to 2 decimal places, pad left with 0: e.g. 0.70 becomes '070', 1.0 becomes '100'
                str_formatted = f"{int(s * 100):03d}"
                ref_parts.append(f"r{i+1}{str_formatted}")
        ref_str = "".join(ref_parts) if ref_parts else "noref"

        kind = "edit2511"
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

        # ── Load LoRA adapters (kept unfused for GGUF models) ─────────────
        # GGUF stores weights as packed bytes with shapes that differ from the
        # dequantised tensor.  Fusing (base_weight += delta) would require
        # dequantising *every* targeted layer to bf16, destroying the memory
        # savings of GGUF and causing OOM.
        #
        # Instead we keep the LoRA adapters loaded but unfused.  During the
        # forward pass peft applies them dynamically:
        #   output = base_layer(x) + lora_B(lora_A(x)) * scale
        # This operates in activation space and works correctly with GGUF's
        # on-the-fly dequantisation.
        lora_specs = []
        lora_path = self._settings.lora_path
        if lora_path and Path(lora_path).is_file():
            lora_specs.append(("lora1", lora_path, self._settings.lora_scale_start))
        lora_path_2 = getattr(self._settings, "lora_path_2", "")
        if lora_path_2 and Path(lora_path_2).is_file():
            lora_specs.append(("lora2", lora_path_2, getattr(self._settings, "lora_scale_start_2", 1.0)))

        if lora_specs:
            adapter_names = []
            adapter_weights = []
            for adapter_name, path, scale in lora_specs:
                self.stage_changed.emit(f"Loading LoRA adapter ({adapter_name})...")
                # Patch diffusers' Qwen LoRA converter to tolerate alpha-less LoRAs.
                # Some LoRAs (e.g. "merged version" files) omit .alpha keys entirely
                # because the alpha/rank scale was already baked into the weights at
                # training time.  Diffusers' get_alpha_scales() hard-pops the alpha
                # key and crashes with KeyError when it is absent.  We wrap the
                # function so that a missing alpha key is treated as alpha == rank
                # (i.e. scale == 1.0), which is the correct no-op.
                import diffusers.loaders.lora_conversion_utils as _lcu
                _original_convert_qwen = _lcu._convert_non_diffusers_qwen_lora_to_diffusers

                def _patched_convert_qwen(state_dict, _orig=_original_convert_qwen):
                    """Wrap the Qwen LoRA converter to handle missing alpha keys."""
                    import diffusers.loaders.lora_conversion_utils as _m

                    # Temporarily replace the inner get_alpha_scales so that a
                    # missing key returns scale == 1.0 instead of raising KeyError.
                    _original_fn = _m._convert_non_diffusers_qwen_lora_to_diffusers

                    # We need to re-implement just the alpha-tolerant inner path.
                    has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
                    if has_diffusion_model:
                        state_dict = {k.removeprefix("diffusion_model."): v for k, v in state_dict.items()}

                    has_lora_unet = any(k.startswith("lora_unet_") for k in state_dict)
                    if has_lora_unet:
                        # Fall back to original – lora_unet_ Qwen paths are fine
                        return _orig(state_dict)

                    has_default = any("default." in k for k in state_dict)
                    if has_default:
                        state_dict = {k.replace("default.", ""): v for k, v in state_dict.items()}

                    converted_state_dict = {}
                    all_keys = list(state_dict.keys())
                    down_key = ".lora_down.weight"
                    up_key = ".lora_up.weight"
                    a_key = ".lora_A.weight"
                    b_key = ".lora_B.weight"

                    has_non_diffusers_lora_id = any(down_key in k or up_key in k for k in all_keys)
                    has_diffusers_lora_id = any(a_key in k or b_key in k for k in all_keys)

                    if has_non_diffusers_lora_id:
                        def _get_alpha_scales_safe(down_weight, alpha_key):
                            rank = down_weight.shape[0]
                            if alpha_key in state_dict:
                                alpha = state_dict.pop(alpha_key).item()
                                logger.debug("LoRA alpha found for %s: %.4f", alpha_key, alpha)
                            else:
                                # Alpha-less LoRA: weights are already at the correct scale
                                alpha = float(rank)
                                logger.debug("LoRA alpha missing for %s, assuming scale=1.0", alpha_key)
                            scale = alpha / rank
                            scale_down = scale
                            scale_up = 1.0
                            while scale_down * 2 < scale_up:
                                scale_down *= 2
                                scale_up /= 2
                            return scale_down, scale_up

                        for k in all_keys:
                            if k.endswith(down_key):
                                diffusers_down_key = k.replace(down_key, ".lora_A.weight")
                                diffusers_up_key = k.replace(down_key, up_key).replace(up_key, ".lora_B.weight")
                                alpha_key = k.replace(down_key, ".alpha")
                                down_weight = state_dict.pop(k)
                                up_weight = state_dict.pop(k.replace(down_key, up_key))
                                scale_down, scale_up = _get_alpha_scales_safe(down_weight, alpha_key)
                                converted_state_dict[diffusers_down_key] = down_weight * scale_down
                                converted_state_dict[diffusers_up_key] = up_weight * scale_up

                    elif has_diffusers_lora_id:
                        for k in all_keys:
                            if a_key in k or b_key in k:
                                converted_state_dict[k] = state_dict.pop(k)
                            elif ".alpha" in k:
                                state_dict.pop(k)

                    # Strip out diff_b / .diff keys that this converter does not handle
                    # (they are bias/norm delta keys stored alongside LoRA weights in
                    # some training frameworks and are not consumed by diffusers).
                    for k in list(state_dict.keys()):
                        if k.endswith(".diff_b") or k.endswith(".diff"):
                            state_dict.pop(k)
                            logger.debug("Dropping unhandled LoRA key: %s", k)

                    if len(state_dict) > 0:
                        raise ValueError(
                            f"`state_dict` should be empty at this point but has {list(state_dict.keys())[:10]}"
                        )

                    converted_state_dict = {f"transformer.{k}": v for k, v in converted_state_dict.items()}
                    return converted_state_dict

                # lora_pipeline.py does `from lora_conversion_utils import ...` so
                # it holds its OWN local binding — we must patch that module too.
                import diffusers.loaders.lora_pipeline as _lp
                _lcu._convert_non_diffusers_qwen_lora_to_diffusers = _patched_convert_qwen
                _lp._convert_non_diffusers_qwen_lora_to_diffusers = _patched_convert_qwen
                try:
                    pipe.load_lora_weights(path, adapter_name=adapter_name)
                finally:
                    _lcu._convert_non_diffusers_qwen_lora_to_diffusers = _original_convert_qwen
                    _lp._convert_non_diffusers_qwen_lora_to_diffusers = _original_convert_qwen

                adapter_names.append(adapter_name)
                adapter_weights.append(scale)
                logger.info("LoRA '%s' loaded from %s", adapter_name, path)

            pipe.set_adapters(adapter_names, adapter_weights)
            logger.info(
                "Active LoRA adapters: %s  weights: %s",
                adapter_names, adapter_weights,
            )

            # ── Debug: verify adapters are attached to the transformer ──
            self._log_lora_debug_info(pipe.transformer)

        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
            best_gpu = free.index(max(free))
            logger.info("Multi-GPU: using cuda:%d for Edit pipeline", best_gpu)
            pipe.enable_model_cpu_offload(gpu_id=best_gpu)
        else:
            pipe.enable_model_cpu_offload()

        return pipe

    @staticmethod
    def _log_lora_debug_info(transformer) -> None:
        """Log detailed info about LoRA adapters attached to the transformer."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        lora_layers = 0
        adapter_names_seen: set[str] = set()
        sample_logged = 0

        for name, module in transformer.named_modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            lora_layers += 1
            # Collect adapter names from this layer's lora_A dict
            layer_adapters = list(getattr(module, "lora_A", {}).keys())
            adapter_names_seen.update(layer_adapters)

            # Log shape info for the first few layers as a sanity check
            if sample_logged < 3:
                for adapter in layer_adapters:
                    a_shape = module.lora_A[adapter].weight.shape
                    b_shape = module.lora_B[adapter].weight.shape
                    logger.info(
                        "  LoRA layer %s adapter='%s': A=%s  B=%s  (rank=%d)",
                        name, adapter, list(a_shape), list(b_shape), a_shape[0],
                    )
                sample_logged += 1

        active = list(transformer.active_adapters()) if hasattr(transformer, "active_adapters") else "N/A"
        logger.info(
            "LoRA debug: %d wrapped layer(s), adapters found: %s, active: %s",
            lora_layers, sorted(adapter_names_seen), active,
        )

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
