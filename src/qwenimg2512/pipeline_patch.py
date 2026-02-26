"""Monkey-patch helpers for diffusers pipelines.

Provides a context manager that intercepts ``scheduler.step`` so that a
custom sampler (from ``qwenimg2512.samplers``) is used in place of the
default Euler step — without modifying any files inside ``.venv``.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _find_pipeline_locals() -> dict | None:
    """Walk up the call stack to find the pipeline __call__ frame.

    Looks for a frame containing both ``prompt_embeds`` and ``noise_pred``
    which uniquely identifies the denoising-loop scope of any QwenImage
    pipeline's ``__call__`` method.

    Returns the frame's ``f_locals`` dict or *None* if not found.
    """
    frame = sys._getframe(0)
    for _ in range(10):
        frame = frame.f_back
        if frame is None:
            return None
        loc = frame.f_locals
        if "prompt_embeds" in loc and "noise_pred" in loc:
            return loc
    return None


def _build_model_fn(
    pipe: Any,
    scheduler: Any,
    original_dtype: torch.dtype,
) -> Any:
    """Build a ``model_fn(sample, sigma) → denoised`` closure.

    Uses frame inspection to capture the pipeline-local variables needed
    for a full transformer forward pass (including true-CFG if active).

    Returns *None* if frame inspection fails — the sampler will then
    gracefully degrade to 1st-order.
    """
    caller = _find_pipeline_locals()
    if caller is None:
        logger.warning(
            "Cannot build model_fn (pipeline frame not found) — "
            "2nd-order samplers will degrade to 1st-order"
        )
        return None

    # Capture references from the pipeline's __call__ frame
    pipe_self = caller.get("self", pipe)
    image_latents = caller.get("image_latents")       # edit pipelines only
    guidance_ref = caller.get("guidance")
    prompt_embeds = caller.get("prompt_embeds")
    prompt_embeds_mask = caller.get("prompt_embeds_mask")
    neg_embeds = caller.get("negative_prompt_embeds")
    neg_mask = caller.get("negative_prompt_embeds_mask")
    img_shapes = caller.get("img_shapes")
    do_true_cfg = caller.get("do_true_cfg", False)
    true_cfg_scale = caller.get("true_cfg_scale", 1.0)

    num_train_ts = scheduler.config.num_train_timesteps
    has_cache_ctx = hasattr(pipe_self.transformer, "cache_context")

    def model_fn(sample_f32: torch.Tensor, sigma_f32: torch.Tensor) -> torch.Tensor:
        """Evaluate the transformer at *(sample, sigma)* → denoised (float32)."""
        sample_cast = sample_f32.to(original_dtype)

        # Build model input — edit pipelines concatenate condition latents
        latent_input = sample_cast
        if image_latents is not None:
            latent_input = torch.cat([sample_cast, image_latents], dim=1)

        # sigma → timestep (pipeline convention: timestep / 1000)
        t_raw = sigma_f32.to(torch.float32) * num_train_ts
        timestep = t_raw.expand(sample_cast.shape[0]).to(original_dtype)

        fwd_kw: dict[str, Any] = dict(
            hidden_states=latent_input,
            timestep=timestep / 1000,
            guidance=guidance_ref,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=getattr(pipe_self, "_attention_kwargs", {}) or {},
            return_dict=False,
        )
        if img_shapes is not None:
            fwd_kw["img_shapes"] = img_shapes

        # --- conditional forward pass ---
        ctx = pipe_self.transformer.cache_context("cond") if has_cache_ctx else nullcontext()
        with ctx:
            noise_pred = pipe_self.transformer(**fwd_kw)[0]
        noise_pred = noise_pred[:, : sample_cast.size(1)]

        # --- unconditional + CFG combine ---
        if do_true_cfg:
            neg_kw = dict(fwd_kw)
            neg_kw["encoder_hidden_states"] = neg_embeds
            neg_kw["encoder_hidden_states_mask"] = neg_mask

            ctx_u = pipe_self.transformer.cache_context("uncond") if has_cache_ctx else nullcontext()
            with ctx_u:
                neg_pred = pipe_self.transformer(**neg_kw)[0]
            neg_pred = neg_pred[:, : sample_cast.size(1)]

            comb = neg_pred + true_cfg_scale * (noise_pred - neg_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb, dim=-1, keepdim=True)
            noise_pred = comb * (cond_norm / noise_norm)

        # x0 = sample − sigma · velocity  (float32)
        return sample_f32 - sigma_f32 * noise_pred.to(torch.float32)

    return model_fn


# ── public API ────────────────────────────────────────────────────────────

@contextmanager
def apply_custom_sampler(pipe: Any, custom_sampler: Any | None):
    """Temporarily replace ``pipe.scheduler.step`` with *custom_sampler*.

    Usage::

        with apply_custom_sampler(pipe, sampler):
            output = pipe(**gen_kwargs)   # no ``custom_sampler`` kwarg needed

    If *custom_sampler* is ``None`` the context manager is a no-op.
    """
    if not custom_sampler:
        yield
        return

    scheduler = pipe.scheduler
    original_step = scheduler.step

    # Reset sampler state at the start of a new generation
    custom_sampler.reset()

    def _patched_step(
        model_output: torch.FloatTensor,
        timestep: Any,
        sample: torch.FloatTensor,
        *args: Any,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        # ── Determine current step index ──────────────────────────────
        if getattr(scheduler, "_step_index", None) is None:
            scheduler._init_step_index(timestep)
        step_index = scheduler._step_index

        # ── Float32 precision throughout ──────────────────────────────
        original_dtype = sample.dtype
        sample_f32 = sample.to(torch.float32)
        model_output_f32 = model_output.to(torch.float32)

        sigmas = scheduler.sigmas
        if isinstance(sigmas, torch.Tensor):
            sigmas = sigmas.to(torch.float32)

        sigma = sigmas[step_index]

        # Flow-matching: x0 = sample − sigma · velocity
        denoised_f32 = sample_f32 - sigma * model_output_f32

        # ── Build model_fn for 2nd-order samplers (e.g. RES 2S) ──────
        model_fn = _build_model_fn(pipe, scheduler, original_dtype)

        # ── Delegate to the custom sampler ────────────────────────────
        new_sample_f32 = custom_sampler.step(
            sample=sample_f32,
            denoised_sample=denoised_f32,
            sigmas=sigmas,
            step_index=step_index,
            model_fn=model_fn,
        )

        # Advance the scheduler index (matches original behaviour)
        scheduler._step_index += 1

        # Downcast back to pipeline precision
        new_sample = new_sample_f32.to(original_dtype)

        if not return_dict:
            return (new_sample,)

        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteSchedulerOutput,
        )
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=new_sample)

    # ── Patch ─────────────────────────────────────────────────────────
    scheduler.step = _patched_step
    logger.info(
        "Custom sampler %s patched into scheduler.step",
        type(custom_sampler).__name__,
    )
    try:
        yield
    finally:
        # ── Restore ───────────────────────────────────────────────────
        scheduler.step = original_step
        custom_sampler.reset()
        logger.info("Restored original scheduler.step")
