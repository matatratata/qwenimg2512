"""Fun ControlNet for Qwen-Image-2512 (alibaba-pai architecture).

Implements the lightweight 5-block ControlNet that injects residuals at
layers [0, 12, 24, 36, 48] of the 60-layer base transformer.

Based on the ComfyUI reference implementation (PR #12359 by krigeta):
  comfy/ldm/qwen_image/controlnet.py

Adapted for the diffusers QwenImageTransformer2DModel API where:
  - The transformer receives pre-packed latents as hidden_states (B, seq, 64)
  - Position embeddings use pos_embed(img_shapes, txt_seq_lens)
  - Timestep embedding uses time_text_embed(timestep, hidden_states)

Model: alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union
Weight format: 174 keys, 5 control blocks, ~3.5GB bfloat16
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

INJECTION_LAYERS = (0, 12, 24, 36, 48)
MAIN_MODEL_DOUBLE = 60


class QwenImageFunControlBlock(nn.Module):
    """Wraps a QwenImageTransformerBlock with optional before_proj and after_proj."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        has_before_proj: bool = False,
    ) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformerBlock,
        )

        self.block = QwenImageTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        self.after_proj = nn.Linear(dim, dim)
        self.before_proj = nn.Linear(dim, dim) if has_before_proj else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run transformer block, return (encoder_hidden_states, hidden_states)."""
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        return encoder_hidden_states, hidden_states


class QwenImageFunControlNetModel(nn.Module):
    """Fun ControlNet that produces residual tensors for the base transformer.

    All base model processing (img_in, txt_norm, txt_in, time_text_embed,
    pos_embed) is done externally in the hook, so this model only receives
    already-projected tensors. This avoids cross-device issues with CPU offload.
    """

    def __init__(
        self,
        in_features: int = 132,
        inner_dim: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_blocks: int = 5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.inner_dim = inner_dim
        self.hint_scale = 1.0
        self.control_img_in = nn.Linear(in_features, inner_dim)
        self.control_blocks = nn.ModuleList(
            [
                QwenImageFunControlBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    has_before_proj=(i == 0),
                )
                for i in range(num_blocks)
            ]
        )

    @staticmethod
    def process_hint(control_latents: torch.Tensor) -> torch.Tensor:
        """Convert VAE-encoded control latents [B,16,H,W] to 132-dim hint tokens.

        Pads channels 16 → 33 (1 mask zero + 16 inpaint zeros), then
        applies 2x2 spatial packing → [B, seq_len, 132].

        All padding is zeros, matching ComfyUI's VideoX fallback semantics:
        [control_latent(16), mask(1)=0, inpaint_latent(16)=0]
        """
        B, C, H, W = control_latents.shape

        # Pad channels: 16 → 33
        zeros_mask = torch.zeros(
            B, 1, H, W, device=control_latents.device, dtype=control_latents.dtype
        )
        zeros_inpaint = torch.zeros(
            B, 16, H, W, device=control_latents.device, dtype=control_latents.dtype
        )
        x = torch.cat([control_latents, zeros_mask, zeros_inpaint], dim=1)  # [B, 33, H, W]

        # Pad spatial to even
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w

        # 2x2 patchify: [B, 33, H, W] → [B, (H/2)*(W/2), 33*4=132]
        x = x.reshape(B, 33, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // 2) * (W // 2), 132)

        return x

    def forward(
        self,
        hint_tokens: torch.Tensor,
        projected_hs: torch.Tensor,
        enc_hs: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor | None]:
        """Run Fun ControlNet and return residuals for injection.

        All inputs are already processed by the base model's shared layers.

        Args:
            hint_tokens: Pre-processed hint tokens [B, seq_len, 132].
            projected_hs: base_model.img_in(hidden_states) — [B, seq, 3072].
            enc_hs: base_model.txt_in(base_model.txt_norm(context)) — [B, txt_seq, 3072].
            temb: base_model.time_text_embed(timestep, projected_hs).
            image_rotary_emb: base_model.pos_embed(img_shapes, txt_seq_lens).

        Returns:
            List of MAIN_MODEL_DOUBLE elements, with residuals at injection indices.
        """
        device = self.control_img_in.weight.device
        dtype = self.control_img_in.weight.dtype

        # Move all inputs to this model's device
        hint_tokens = hint_tokens.to(device=device, dtype=dtype)
        projected_hs = projected_hs.to(device=device, dtype=dtype)
        enc_hs = enc_hs.clone().to(device=device, dtype=dtype)
        temb = temb.to(device=device, dtype=dtype)
        if isinstance(image_rotary_emb, (tuple, list)):
            image_rotary_emb = tuple(t.to(device=device, dtype=dtype) for t in image_rotary_emb)

        # Expand hint_tokens to match batch size (e.g. true_cfg doubles batch)
        batch_size = projected_hs.shape[0]
        if hint_tokens.shape[0] != batch_size:
            hint_tokens = hint_tokens.expand(batch_size, -1, -1)

        # Pad or truncate hint_tokens to match projected_hs sequence length.
        # Never truncate projected_hs — residuals must match the main model's
        # sequence length or post-hook addition will crash.
        seq_len = projected_hs.shape[1]
        if hint_tokens.shape[1] < seq_len:
            pad_len = seq_len - hint_tokens.shape[1]
            hint_tokens = F.pad(hint_tokens, (0, 0, 0, pad_len))
        elif hint_tokens.shape[1] > seq_len:
            hint_tokens = hint_tokens[:, :seq_len]

        # --- ControlNet-specific processing ---
        # Project hint tokens: (B, seq, 132) → (B, seq, 3072)
        c = self.control_img_in(hint_tokens)

        # Disable attention mask inside control blocks (matches ComfyUI/VideoX)
        ctrl_enc_mask = None

        # --- Run control blocks with stack-based accumulation ---
        # This pattern matches the ComfyUI reference exactly:
        #   Block 0: c_in = before_proj(c) + projected_hs; stack = [c_skip, c_out]
        #   Block i: c_in = stack.pop(); stack = [...existing, c_skip, c_out]
        for i, block in enumerate(self.control_blocks):
            if i == 0:
                # Block 0: combine hint with base hidden_states via before_proj
                c_in = block.before_proj(c) + projected_hs
                all_c = []
            else:
                # Subsequent blocks: pop last element from stack as input
                all_c = list(torch.unbind(c, dim=0))
                c_in = all_c.pop(-1)

            enc_hs, c_out = block(
                hidden_states=c_in,
                encoder_hidden_states=enc_hs,
                encoder_hidden_states_mask=ctrl_enc_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

            c_skip = block.after_proj(c_out) * self.hint_scale
            all_c += [c_skip, c_out]
            c = torch.stack(all_c, dim=0)

        # Extract residuals (all but the last element in the stack)
        hints = torch.unbind(c, dim=0)[:-1]

        # Map to injection layer indices
        controlnet_block_samples: list[torch.Tensor | None] = [None] * MAIN_MODEL_DOUBLE
        for local_idx, base_idx in enumerate(INJECTION_LAYERS):
            if local_idx < len(hints) and base_idx < len(controlnet_block_samples):
                controlnet_block_samples[base_idx] = hints[local_idx]

        return controlnet_block_samples


def load_fun_controlnet(
    path: str, dtype: torch.dtype = torch.bfloat16
) -> QwenImageFunControlNetModel:
    """Load Fun ControlNet from a safetensors file.

    Extracts architecture parameters from weight shapes and remaps keys
    to match our module structure (transformer block lives under .block).
    """
    logger.info("Loading Fun ControlNet from %s", path)
    state_dict = load_file(path, device="cpu")

    # Extract architecture params from weight shapes
    in_features = state_dict["control_img_in.weight"].shape[1]  # 132
    inner_dim = state_dict["control_img_in.weight"].shape[0]  # 3072

    # Count blocks
    block_indices = set()
    for key in state_dict:
        if key.startswith("control_blocks."):
            block_indices.add(int(key.split(".")[1]))
    num_blocks = len(block_indices)

    # Get attention config
    head_dim = state_dict["control_blocks.0.attn.norm_q.weight"].shape[0]  # 128
    num_heads = state_dict["control_blocks.0.attn.to_q.weight"].shape[0] // head_dim

    logger.info(
        "Fun ControlNet config: in=%d, dim=%d, heads=%d, head_dim=%d, blocks=%d",
        in_features,
        inner_dim,
        num_heads,
        head_dim,
        num_blocks,
    )

    # Remap keys: control_blocks.{i}.{key} → control_blocks.{i}.block.{key}
    # for transformer block internals (everything except before_proj/after_proj)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("control_blocks."):
            parts = key.split(".", 2)
            if len(parts) >= 3:
                idx, rest = parts[1], parts[2]
                if not rest.startswith(("before_proj", "after_proj")):
                    new_key = f"control_blocks.{idx}.block.{rest}"
        new_state_dict[new_key] = value

    model = QwenImageFunControlNetModel(
        in_features=in_features,
        inner_dim=inner_dim,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        num_blocks=num_blocks,
    )

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    model.to(dtype=dtype)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Fun ControlNet loaded: %.1fM parameters", param_count)

    return model


def setup_fun_controlnet_hooks(
    transformer: nn.Module,
    fun_cn: QwenImageFunControlNetModel,
    hint_tokens: torch.Tensor,
    conditioning_scale: float = 0.8,
    guidance_start: float = 0.0,
    guidance_end: float = 1.0,
    num_steps: int = 50,
) -> tuple[list, dict]:
    """Set up PyTorch hooks to inject Fun ControlNet residuals into the transformer.

    Strategy:
    1. A pre-hook on the FIRST transformer block captures the already-projected
       intermediate tensors (hidden_states, encoder_hidden_states, temb,
       image_rotary_emb) that the transformer forward has computed by that point.
       These are passed to the Fun ControlNet forward.
    2. Post-hooks on each transformer block add the residuals at injection layers.

    This avoids calling base model layers from a different device context (solves
    CPU offload compatibility issues).

    Returns (hooks_list, state_dict).
    """
    effective_scale = math.sqrt(max(conditioning_scale, 0.0))

    state = {
        "residuals": None,
        "active": guidance_start <= 0.0,
        "step": 0,
        "num_steps": num_steps,
        "guidance_start": guidance_start,
        "guidance_end": guidance_end,
    }
    hooks = []

    # Pre-hook on the FIRST transformer block.
    # At this point the transformer forward has already computed:
    #   hidden_states = self.img_in(hidden_states)   → projected_hs
    #   encoder_hidden_states = self.txt_in(self.txt_norm(enc))  → enc_hs
    #   temb = self.time_text_embed(timestep, hidden_states)
    #   image_rotary_emb = self.pos_embed(...)
    # These are passed as kwargs to each block.
    def block0_pre_hook(module, args, kwargs):
        if not state["active"]:
            state["residuals"] = None
            return

        # Capture the already-projected inputs from block kwargs,
        # with positional fallbacks for diffusers versions that pass
        # these as args instead of kwargs.
        projected_hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        enc_hs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        temb = kwargs.get("temb", args[2] if len(args) > 2 else None)
        image_rotary_emb = kwargs.get("image_rotary_emb", args[3] if len(args) > 3 else None)

        if projected_hs is None or enc_hs is None or temb is None:
            logger.warning("Could not extract block inputs for ControlNet")
            state["residuals"] = None
            return

        with torch.no_grad():
            residuals = fun_cn(
                hint_tokens=hint_tokens,
                projected_hs=projected_hs,
                enc_hs=enc_hs,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        state["residuals"] = residuals

    h = transformer.transformer_blocks[0].register_forward_pre_hook(
        block0_pre_hook, with_kwargs=True
    )
    hooks.append(h)

    # Post-hooks on transformer blocks to add residuals
    for layer_idx in range(MAIN_MODEL_DOUBLE):

        def make_post_hook(l_idx: int):
            def post_hook(module, args, kwargs, output):
                if state["residuals"] is None:
                    return output
                residual = state["residuals"][l_idx]
                if residual is None:
                    return output
                # output is (encoder_hidden_states, hidden_states)
                enc_out, h_out = output
                residual = residual.to(device=h_out.device, dtype=h_out.dtype)
                # Soft timestep decay: 1.0 at start → 0.5 at end
                # Gently prioritises early structural steps without
                # killing late-step detail.  Users can still boost via
                # guidance_start / guidance_end for coarser control.
                t_frac = state["step"] / max(state["num_steps"], 1)
                step_scale = 1.0 - 0.5 * t_frac
                return enc_out, h_out + residual * effective_scale * step_scale

            return post_hook

        h = transformer.transformer_blocks[layer_idx].register_forward_hook(
            make_post_hook(layer_idx), with_kwargs=True
        )
        hooks.append(h)

    return hooks, state


def update_controlnet_state(state: dict, step: int) -> None:
    """Update the controlnet hook state from the pipeline step callback."""
    state["step"] = step
    frac = step / max(state["num_steps"], 1)
    state["active"] = state["guidance_start"] <= frac < state["guidance_end"]


def remove_fun_controlnet_hooks(hooks: list) -> None:
    """Remove all Fun ControlNet injection hooks."""
    for h in hooks:
        h.remove()
    hooks.clear()
