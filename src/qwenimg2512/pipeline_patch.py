"""Monkey-patch helpers for diffusers pipelines.

Provides context managers that intercept ``scheduler.step`` and the
transformer's FFN/block execution without modifying any files inside
``.venv``.  See the individual docstrings for details.
"""

from __future__ import annotations

import logging
import sys
import functools
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import torch

from qwenimg2512.schedules import get_bong_tangent_schedule, get_beta57_schedule

logger = logging.getLogger(__name__)


def _find_pipeline_locals() -> dict | None:
    """Walk up the call stack to find the pipeline __call__ frame.

    Looks for a frame containing both ``prompt_embeds`` and ``noise_pred``
    which uniquely identifies the denoising-loop scope of any QwenImage
    pipeline's ``__call__`` method.

    Returns a *selective* dict of only the variables needed — never the
    full ``f_locals``, which would retain every intermediate tensor in the
    denoising loop and cause a large VRAM leak.
    """
    frame = sys._getframe(0)
    try:
        for _ in range(10):
            frame = frame.f_back
            if frame is None:
                return None
            loc = frame.f_locals
            if "prompt_embeds" in loc and "noise_pred" in loc:
                # CRITICAL OOM FIX: extract only the keys we need.
                # Returning ``loc`` directly keeps strong references to ALL
                # pipeline-loop variables (latents, noise_pred, etc.) and
                # prevents the GC from freeing them → massive VRAM leak.
                keys_to_capture = [
                    "self", "image_latents", "guidance", "prompt_embeds",
                    "prompt_embeds_mask", "negative_prompt_embeds",
                    "negative_prompt_embeds_mask", "img_shapes",
                    "do_true_cfg", "true_cfg_scale",
                ]
                return {k: loc[k] for k in keys_to_capture if k in loc}
        return None
    finally:
        del frame  # Break reference cycle; prevents frame from leaking VRAM


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
    
    # Gracefully degrade 2nd-order single-step (Heun) evaluation if unsupported
    if "Wan" in pipe_self.__class__.__name__:
        return None
        
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

# ---------------------------------------------------------------------------
# Helper used by both apply_ffn_chunking below and tests
# ---------------------------------------------------------------------------

def _chunked_ff_forward(
    ff_module: Any,
    hidden_states: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Run *ff_module* on *hidden_states* in sequence-length chunks.

    This is mathematically identical to ``ff_module(hidden_states)`` but
    avoids allocating the full ``[B, SeqLen, 4·dim]`` intermediate tensor.
    Instead peak activation memory is ``O(chunk_size × 4·dim)``.

    *chunk_size* is automatically clamped so that ``SeqLen >= chunk_size``.
    Any remainder tokens (if SeqLen is not exactly divisible) are handled by
    running the leftover tokens as an extra smaller chunk rather than
    raising an error (unlike diffusers' ``_chunked_feed_forward``).
    """
    seq_len = hidden_states.shape[1]  # [B, L, D]
    if chunk_size <= 0 or chunk_size >= seq_len:
        return ff_module(hidden_states)

    chunks = hidden_states.split(chunk_size, dim=1)
    return torch.cat([ff_module(c) for c in chunks], dim=1)


@contextmanager
def apply_custom_schedule(pipe: Any, schedule_name: str):
    """Inject a custom sigma schedule into ``pipe.scheduler``.

    Instead of patching ``scheduler.step`` lazily, we wrap ``set_timesteps``
    using ``functools.wraps`` so that diffusers' ``retrieve_timesteps`` sees
    the correct signature. This ensures the pipeline loop uses the updated
    ``timesteps`` for model evaluations upstream perfectly synchronized with sigmas.
    """
    if not schedule_name or schedule_name == "default":
        yield
        return

    scheduler = pipe.scheduler
    original_set_timesteps = scheduler.set_timesteps

    @functools.wraps(original_set_timesteps)
    def _patched_set_timesteps(*args: Any, **kwargs: Any):
        # 1. Call the original method to let diffusers initialize safe defaults
        res = original_set_timesteps(*args, **kwargs)

        num_steps = len(scheduler.timesteps)
        device = scheduler.sigmas.device

        if schedule_name == "bong_tangent":
            custom = get_bong_tangent_schedule(num_steps)
        elif schedule_name == "beta57":
            custom = get_beta57_schedule(num_steps)
        else:
            logger.warning("Unknown schedule %r — using default", schedule_name)
            return res

        sigmas_tensor = torch.tensor(custom, dtype=torch.float32, device=device)

        # 2. Match expected length (num_steps + 1)
        expected = num_steps + 1
        if len(sigmas_tensor) > expected:
            sigmas_tensor = sigmas_tensor[:expected]
        elif len(sigmas_tensor) < expected:
            pad = torch.zeros(
                expected - len(sigmas_tensor),
                dtype=torch.float32, device=device,
            )
            sigmas_tensor = torch.cat([sigmas_tensor, pad])
            
        # Ensure final element is strictly 0.0
        if sigmas_tensor[-1] != 0.0:
            sigmas_tensor[-1] = 0.0

        scheduler.sigmas = sigmas_tensor

        # 3. CRITICAL: Update timesteps so model evaluations match the new sigmas!
        num_train_ts = getattr(scheduler.config, "num_train_timesteps", 1000)
        new_timesteps = sigmas_tensor[:-1] * num_train_ts
        
        if hasattr(scheduler, "timesteps") and isinstance(scheduler.timesteps, torch.Tensor):
            new_timesteps = new_timesteps.to(scheduler.timesteps.dtype)
            
        scheduler.timesteps = new_timesteps

        logger.info(
            "Custom schedule %r injected (%d steps, device=%s)",
            schedule_name, num_steps, device,
        )
        return res

    # ── Patch ─────────────────────────────────────────────────────────
    scheduler.set_timesteps = _patched_set_timesteps
    try:
        yield
    finally:
        scheduler.set_timesteps = original_set_timesteps
        logger.info("Restored scheduler.set_timesteps (schedule patch removed)")


# ---------------------------------------------------------------------------
# FFN chunking — main fix for large-resolution OOM
# ---------------------------------------------------------------------------

@contextmanager
def apply_ffn_chunking(pipe: Any, chunk_size: int):
    """Chunk the FFN sequence dimension across all transformer blocks.

    When *chunk_size* is ``0`` or ``None`` this context manager is a no-op.

    For every ``QwenImageTransformerBlock`` inside
    ``pipe.transformer.transformer_blocks`` we replace the block's ``forward``
    so that ``img_mlp`` and ``txt_mlp`` calls receive *chunk_size* tokens at a
    time rather than the full sequence.  This is mathematically equivalent to
    the standard full-sequence forward pass.

    Usage::

        with apply_ffn_chunking(pipe, chunk_size=2048):
            output = pipe(**gen_kwargs)
    """
    if not chunk_size:
        yield
        return

    transformer = pipe.transformer
    blocks = list(transformer.transformer_blocks)
    originals: list[Any] = []

    for block in blocks:
        orig_forward = block.forward
        originals.append(orig_forward)

        # Capture the block in the closure so each iteration gets its own ref
        def _make_patched(b):
            def _patched_forward(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb=None,
                joint_attention_kwargs=None,
                modulate_index=None,
            ):
                # ---- modulation vectors ---------------------------------
                img_mod_params = b.img_mod(temb)
                if b.zero_cond_t:
                    temb_txt = torch.chunk(temb, 2, dim=0)[0]
                else:
                    temb_txt = temb
                txt_mod_params = b.txt_mod(temb_txt)

                img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
                txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

                # ---- norm-1 + attention ---------------------------------
                seq_img = hidden_states.shape[1]
                eff_chunk = chunk_size if chunk_size > 0 else seq_img
                eff_chunk = min(eff_chunk, seq_img)

                img_modulated = torch.empty_like(hidden_states)
                
                # Pre-allocate img_gate1 using the structure of hidden_states.
                # _modulate returns gate components with same sequence dimension.
                # Note: b._modulate returns gate_result, we define its shape dynamically.
                img_gate1 = None

                for start in range(0, seq_img, eff_chunk):
                    end = min(start + eff_chunk, seq_img)
                    hs_c = hidden_states[:, start:end, :]
                    norm1_c = b.img_norm1(hs_c)
                    
                    mod_idx_c = modulate_index[:, start:end] if modulate_index is not None else None
                    mod1_c, gate1_c = b._modulate(norm1_c, img_mod1, mod_idx_c)
                    
                    img_modulated[:, start:end, :] = mod1_c
                    if img_gate1 is None:
                        # Allocate full sequence gate tensor matching chunk's dtype & non-seq dims
                        gate_shape = list(gate1_c.shape)
                        gate_shape[1] = seq_img
                        img_gate1 = torch.empty(gate_shape, dtype=gate1_c.dtype, device=gate1_c.device)
                    
                    img_gate1[:, start:end] = gate1_c
                    del norm1_c, mod1_c, gate1_c

                seq_txt = encoder_hidden_states.shape[1]
                eff_chunk_txt = chunk_size if chunk_size > 0 else seq_txt
                eff_chunk_txt = min(eff_chunk_txt, seq_txt)

                txt_modulated = torch.empty_like(encoder_hidden_states)
                txt_gate1 = None
                for start in range(0, seq_txt, eff_chunk_txt):
                    end = min(start + eff_chunk_txt, seq_txt)
                    tx_c = encoder_hidden_states[:, start:end, :]
                    t_norm1_c = b.txt_norm1(tx_c)
                    t_mod1_c, t_gate1_c = b._modulate(t_norm1_c, txt_mod1)
                    txt_modulated[:, start:end, :] = t_mod1_c
                    
                    if txt_gate1 is None:
                        gate_shape = list(t_gate1_c.shape)
                        gate_shape[1] = seq_txt
                        txt_gate1 = torch.empty(gate_shape, dtype=t_gate1_c.dtype, device=t_gate1_c.device)
                        
                    txt_gate1[:, start:end] = t_gate1_c
                    del t_norm1_c, t_mod1_c, t_gate1_c

                jkw = joint_attention_kwargs or {}
                attn_output = b.attn(
                    hidden_states=img_modulated,
                    encoder_hidden_states=txt_modulated,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    image_rotary_emb=image_rotary_emb,
                    **jkw,
                )
                del img_modulated, txt_modulated  # Free inputs to attention explicitly

                img_attn_output, txt_attn_output = attn_output

                hidden_states = hidden_states + img_gate1 * img_attn_output
                del img_gate1, img_attn_output  # Free memory ASAP

                encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
                del txt_gate1, txt_attn_output  # Free memory ASAP

                # ---- norm-2 + CHUNKED FFN -------------------------------
                new_hidden_states = torch.empty_like(hidden_states)
                for start in range(0, seq_img, eff_chunk):
                    end = min(start + eff_chunk, seq_img)
                    hs_c = hidden_states[:, start:end, :]
                    
                    norm2_c = b.img_norm2(hs_c)
                    mod_idx_c = modulate_index[:, start:end] if modulate_index is not None else None
                    mod2_c, gate2_c = b._modulate(norm2_c, img_mod2, mod_idx_c)
                    del norm2_c
                    
                    mlp_c = b.img_mlp(mod2_c)
                    del mod2_c
                    
                    new_hidden_states[:, start:end, :] = hs_c + gate2_c * mlp_c
                    del mlp_c, gate2_c
                hidden_states = new_hidden_states
                del new_hidden_states

                new_encoder_hidden_states = torch.empty_like(encoder_hidden_states)
                for start in range(0, seq_txt, eff_chunk_txt):
                    end = min(start + eff_chunk_txt, seq_txt)
                    tx_c = encoder_hidden_states[:, start:end, :]
                    
                    t_norm2_c = b.txt_norm2(tx_c)
                    t_mod2_c, t_gate2_c = b._modulate(t_norm2_c, txt_mod2)
                    del t_norm2_c
                    
                    t_mlp_c = b.txt_mlp(t_mod2_c)
                    del t_mod2_c
                    
                    new_encoder_hidden_states[:, start:end, :] = tx_c + t_gate2_c * t_mlp_c
                    del t_mlp_c, t_gate2_c
                encoder_hidden_states = new_encoder_hidden_states
                del new_encoder_hidden_states

                # ---- dtype clipping (matches original) ------------------
                if encoder_hidden_states.dtype == torch.float16:
                    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
                if hidden_states.dtype == torch.float16:
                    hidden_states = hidden_states.clip(-65504, 65504)

                return encoder_hidden_states, hidden_states

            return _patched_forward

        block.forward = _make_patched(block)

    n_blocks = len(blocks)
    logger.info(
        "FFN chunking enabled: chunk_size=%d across %d transformer blocks",
        chunk_size,
        n_blocks,
    )
    try:
        yield
    finally:
        for block, orig in zip(blocks, originals):
            block.forward = orig
        logger.info("FFN chunking restored on %d blocks", n_blocks)


# ---------------------------------------------------------------------------
# Attention sequence chunking — fixes OOM in norm_k/norm_q at large resolutions
# ---------------------------------------------------------------------------
#
# At large output sizes (e.g. 2432×1408 + 2 refs) the image token sequence
# can reach ~26 000 tokens.  The line:
#
#   img_key = attn.norm_k(img_key)   # RMSNorm: .to(float32) then back
#
# materialises a [B, 26000, heads, head_dim] float32 tensor (~330 MiB) which
# causes OOM before SDPA even runs (Flash Attention is already enabled by
# default, so SDPA itself is fine).
#
# This patch replaces QwenDoubleStreamAttnProcessor2_0.__call__ with a version
# that processes image QKV projections + norms in chunks along the token axis,
# then concatenates and runs a single SDPA on the full sequence.  Text tokens
# are small (O(1 000)) so they are never chunked.

def _chunked_attn_call(
    original_call,
    chunk_size: int,
    attn_obj,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask=None,
    attention_mask=None,
    image_rotary_emb=None,
):
    """Chunked replacement for QwenDoubleStreamAttnProcessor2_0.__call__.

    1. Processes *image* Q/K/V projections + norms in ``chunk_size`` chunks.
    2. Concatenates the processed chunks.
    3. Runs the single SDPA on the full joint [txt, img] sequence normally.

    Text tokens are tiny and always processed at once.
    """
    attn = attn_obj
    seq_img = hidden_states.shape[1]   # image token count (large)
    seq_txt = encoder_hidden_states.shape[1]  # text token count (small)

    # ── text QKV — small, process at once ──────────────────────────────
    txt_query = attn.add_q_proj(encoder_hidden_states)
    txt_key   = attn.add_k_proj(encoder_hidden_states)
    txt_value = attn.add_v_proj(encoder_hidden_states)

    txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    txt_key   = txt_key.unflatten(-1, (attn.heads, -1))
    txt_value = txt_value.unflatten(-1, (attn.heads, -1))

    if attn.norm_added_q is not None:
        txt_query = attn.norm_added_q(txt_query)
    if attn.norm_added_k is not None:
        txt_key = attn.norm_added_k(txt_key)

    img_freqs, txt_freqs = (image_rotary_emb if image_rotary_emb is not None else (None, None))

    if txt_freqs is not None:
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
        txt_key   = apply_rotary_emb_qwen(txt_key,   txt_freqs, use_real=False)

    # ── image QKV — chunked along token dimension ───────────────────────
    eff_chunk = chunk_size if chunk_size > 0 else seq_img
    eff_chunk = min(eff_chunk, seq_img)

    # Pre-allocate joint tensors to avoid doubling memory during torch.cat
    B, _, heads, head_dim = txt_query.shape
    dtype = txt_query.dtype
    device = txt_query.device

    joint_query = torch.empty((B, seq_txt + seq_img, heads, head_dim), dtype=dtype, device=device)
    joint_key   = torch.empty((B, seq_txt + seq_img, heads, head_dim), dtype=dtype, device=device)
    joint_value = torch.empty((B, seq_txt + seq_img, heads, head_dim), dtype=dtype, device=device)

    # Insert text parts and free original tensors immediately
    joint_query[:, :seq_txt] = txt_query
    joint_key[:, :seq_txt]   = txt_key
    joint_value[:, :seq_txt] = txt_value
    del txt_query, txt_key, txt_value

    for start in range(0, seq_img, eff_chunk):
        end = min(start + eff_chunk, seq_img)
        hs_chunk = hidden_states[:, start:end, :]

        # Value chunk
        v_c = attn.to_v(hs_chunk).unflatten(-1, (attn.heads, -1))
        joint_value[:, seq_txt + start : seq_txt + end] = v_c
        del v_c

        # Query chunk
        q_c = attn.to_q(hs_chunk).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            q_c = attn.norm_q(q_c)

        # Key chunk
        k_c = attn.to_k(hs_chunk).unflatten(-1, (attn.heads, -1))
        if attn.norm_k is not None:
            k_c = attn.norm_k(k_c)

        # Apply RoPE
        if img_freqs is not None:
            from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
            if hasattr(img_freqs, "ndim") and img_freqs.ndim == 3:
                freq_chunk = img_freqs[:, start:end, :]
            else:
                freq_chunk = img_freqs[start:end]
            q_c = apply_rotary_emb_qwen(q_c, freq_chunk, use_real=False)
            k_c = apply_rotary_emb_qwen(k_c, freq_chunk, use_real=False)

        joint_query[:, seq_txt + start : seq_txt + end] = q_c
        del q_c
        
        joint_key[:, seq_txt + start : seq_txt + end] = k_c
        del k_c


    from diffusers.models.transformers.transformer_qwenimage import dispatch_attention_fn
    from diffusers.models.attention_processor import Attention  # noqa: F401

    attn_proc = attn.processor
    dtype_to_use = joint_query.dtype
    
    joint_hidden_states = dispatch_attention_fn(
        joint_query,
        joint_key,
        joint_value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=getattr(attn_proc, "_attention_backend", None),
        parallel_config=getattr(attn_proc, "_parallel_config", None),
    )

    # Critical memory saving: Free Q, K, V immediately after attention is done!
    del joint_query, joint_key, joint_value

    joint_hidden_states = joint_hidden_states.flatten(2, 3).to(dtype_to_use)

    txt_attn_out = joint_hidden_states[:, :seq_txt, :]
    img_attn_out = joint_hidden_states[:, seq_txt:, :]
    
    # Free the joint array as soon as we slice it
    del joint_hidden_states

    img_attn_out = attn.to_out[0](img_attn_out.contiguous())
    if len(attn.to_out) > 1:
        img_attn_out = attn.to_out[1](img_attn_out)

    txt_attn_out = attn.to_add_out(txt_attn_out.contiguous())

    return img_attn_out, txt_attn_out


@contextmanager
def apply_attn_chunking(pipe: Any, chunk_size: int):
    """Chunk image QKV projections + norms across the token sequence axis.

    This eliminates the OOM in ``norm_k`` that occurs at large resolutions
    (e.g. 2432×1408) where the image sequence can reach ~26 000 tokens.

    *chunk_size* is the number of image tokens processed per chunk.
    Set to ``0`` to disable (no-op).

    Recommended values:
      - ``4096``  — safe starting point, ~75 MiB peak norm buffer instead of 330 MiB
      - ``2048``  — more conservative
      - ``1024``  — most conservative (slowest due to more kernel launches)

    When *chunk_size* is ``0`` or the sequence is shorter than *chunk_size*,
    the original un-chunked path is used (no overhead).

    Usage::

        with apply_attn_chunking(pipe, chunk_size=4096):
            output = pipe(**gen_kwargs)
    """
    if not chunk_size:
        yield
        return

    from diffusers.models.transformers.transformer_qwenimage import (
        QwenDoubleStreamAttnProcessor2_0,
    )

    original_call = QwenDoubleStreamAttnProcessor2_0.__call__

    # CRITICAL: signature must explicitly name all parameters that
    # QwenDoubleStreamAttnProcessor2_0.__call__ accepts, because diffusers'
    # Attention.forward() inspects the processor's __call__ signature to
    # decide which cross_attention_kwargs to pass through.  If we use
    # **kwargs, diffusers can't see the parameter names and SILENTLY DROPS
    # encoder_hidden_states_mask and image_rotary_emb, causing attention
    # to run without RoPE (completely wrong output).
    def _patched_call(
        self_proc,
        attn_obj,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        seq_img = hidden_states.shape[1]
        # Only chunk when the sequence is actually large enough to matter
        if seq_img <= chunk_size:
            return original_call(
                self_proc, attn_obj, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
        return _chunked_attn_call(
            original_call,
            chunk_size,
            attn_obj,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    QwenDoubleStreamAttnProcessor2_0.__call__ = _patched_call
    logger.info(
        "Attention chunking enabled: image tokens will be projected/normed in "
        "chunks of %d",
        chunk_size,
    )
    try:
        yield
    finally:
        QwenDoubleStreamAttnProcessor2_0.__call__ = original_call
        logger.info("Attention chunking restored")


# ---------------------------------------------------------------------------
# Block CPU-offloading — secondary / more aggressive memory saving
# ---------------------------------------------------------------------------

@contextmanager
def apply_block_swap(pipe: Any, num_blocks_on_cpu: int):
    """Move the last *num_blocks_on_cpu* transformer blocks to CPU.

    When *num_blocks_on_cpu* is ``0`` or ``None`` this context manager is a
    no-op.

    During the transformer forward pass only the current block is resident on
    the GPU; all offloaded blocks are fetched to GPU just-in-time and moved
    back to CPU after their output has been computed.  This significantly
    reduces resident VRAM of the transformer at the cost of extra
    host<->device transfers.

    Usage::

        with apply_block_swap(pipe, num_blocks_on_cpu=10):
            output = pipe(**gen_kwargs)
    """
    if not num_blocks_on_cpu:
        yield
        return

    transformer = pipe.transformer
    blocks = transformer.transformer_blocks
    n_total = len(blocks)
    n_swap = min(num_blocks_on_cpu, n_total)
    swap_indices = set(range(n_total - n_swap, n_total))

    # Determine which GPU device to use
    gpu_device = torch.device("cuda")
    for i, b in enumerate(blocks):
        if i not in swap_indices:
            try:
                p = next(b.parameters())
                if p.device.type == "cuda":
                    gpu_device = p.device
                    break
            except StopIteration:
                pass

    # Move swapped blocks to CPU
    for i in swap_indices:
        blocks[i].to("cpu")
    torch.cuda.empty_cache()

    originals: list[Any] = []

    # Patch each swapped block's forward to fetch/return to GPU on call
    for i in swap_indices:
        block = blocks[i]
        orig_forward = block.forward
        originals.append((i, block, orig_forward))

        def _make_swap_forward(blk, orig_fwd):
            def _swap_fwd(*args, **kwargs):
                blk.to(gpu_device)
                out = orig_fwd(*args, **kwargs)
                blk.to("cpu")
                torch.cuda.empty_cache()
                return out
            return _swap_fwd

        block.forward = _make_swap_forward(block, orig_forward)

    logger.info(
        "Block CPU-swap enabled: last %d of %d blocks offloaded to CPU (target GPU=%s)",
        n_swap,
        n_total,
        gpu_device,
    )
    try:
        yield
    finally:
        for i, block, orig in originals:
            block.forward = orig
            block.to(gpu_device)
        logger.info("Block CPU-swap restored: %d blocks moved back to %s", n_swap, gpu_device)
