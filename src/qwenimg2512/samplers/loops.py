from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from tqdm import tqdm

from ltx_core.types import LatentState
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_pipelines.utils.helpers import post_process_latent
from ltx_pipelines.utils.types import DenoisingFunc

def res_2s_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    progress_callback: Any = None,
    stage_name: str = "Denoising (RES 2S)",
) -> tuple[LatentState, LatentState]:
    """
    Custom denoising loop for RES 2S (Heun-like) sampler.
    Performs 2nd-order corrections using 2 model evaluations per step.
    """
    total_steps = len(sigmas) - 1
    
    # Iterate through all steps
    # Use tqdm for CLI progress monitoring
    iterable = tqdm(range(total_steps), desc=stage_name)
    
    for step_idx in iterable:
        # Current sigma and next sigma
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # --- Evaluation 1 ---
        denoised_video_1, denoised_audio_1 = denoise_fn(video_state, audio_state, sigmas, step_idx)

        # Post-process (handle masking/inpainting)
        denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)

        # Optimization: If final step (sigma_next == 0), just stick with Euler/1st order to avoid singularity/instability
        if sigma_next == 0:
             new_video_latent = stepper.step(video_state.latent, denoised_video_1, sigmas, step_idx)
             new_audio_latent = stepper.step(audio_state.latent, denoised_audio_1, sigmas, step_idx)
             video_state = replace(video_state, latent=new_video_latent)
             audio_state = replace(audio_state, latent=new_audio_latent)
             if progress_callback is not None:
                progress_callback(step_idx + 1, total_steps, stage_name)
             continue

        # --- Euler Step (Guess) ---
        # Predict x_next using first derivative
        video_latent_guess = stepper.step(video_state.latent, denoised_video_1, sigmas, step_idx)
        audio_latent_guess = stepper.step(audio_state.latent, denoised_audio_1, sigmas, step_idx)

        # Create temporary states for the guess
        # We preserve the original mask/clean_latent but use the guessed latent
        video_state_guess = replace(video_state, latent=video_latent_guess)
        audio_state_guess = replace(audio_state, latent=audio_latent_guess)

        # --- Evaluation 2 ---
        # Evaluate model at x_guess and sigma_next (step_idx + 1)
        denoised_video_2, denoised_audio_2 = denoise_fn(video_state_guess, audio_state_guess, sigmas, step_idx + 1)

        # Post-process guess
        denoised_video_2 = post_process_latent(denoised_video_2, video_state.denoise_mask, video_state.clean_latent)
        denoised_audio_2 = post_process_latent(denoised_audio_2, audio_state.denoise_mask, audio_state.clean_latent)

        # --- Combine (Heun/2nd Order) ---
        # Calculate exact exponential integration weight for 2nd-order accuracy.
        # The naive 0.5 average heavily undercorrects for large steps, degrading to 1st-order.
        r = (sigma_next / sigma).to(torch.float32).clamp(min=1e-5)
        h = 1.0 - r
        h_safe = torch.where(h.abs() < 1e-4, torch.ones_like(h), h)
        w_exact = 1.0 / h_safe + (r * torch.log(r)) / (h_safe * h_safe)
        w_taylor = 0.5 + h / 6.0
        w = torch.where(h.abs() < 1e-4, w_taylor, w_exact)

        # Average the denoised predictions using the exact weight
        w_v = w.to(denoised_video_1.dtype)
        denoised_video_avg = (1.0 - w_v) * denoised_video_1 + w_v * denoised_video_2
        
        w_a = w.to(denoised_audio_1.dtype)
        denoised_audio_avg = (1.0 - w_a) * denoised_audio_1 + w_a * denoised_audio_2

        # Final step using the exact weighted prediction
        # We reuse the stepper.step function which implements the 'exponential step' update
        new_video_latent = stepper.step(video_state.latent, denoised_video_avg, sigmas, step_idx)
        new_audio_latent = stepper.step(audio_state.latent, denoised_audio_avg, sigmas, step_idx)

        # Update states
        video_state = replace(video_state, latent=new_video_latent)
        audio_state = replace(audio_state, latent=new_audio_latent)

        if progress_callback is not None:
            progress_callback(step_idx + 1, total_steps, stage_name)

    return (video_state, audio_state)
