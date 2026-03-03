"""RES (Recursive Exponential Stepping) multistep samplers.

These samplers use multistep history of denoised predictions to achieve
higher-order accuracy through extrapolation. They are particularly good at
preserving fine details in video generation.

Inspired by the RES4LYF implementation by ClownsharkBatwing.
Reference: https://github.com/ClownsharkBatwing/RES4LYF

Note: These samplers track state per-tensor-shape to support multiple
modalities (video/audio) being processed with the same sampler instance.
"""

from __future__ import annotations

from typing import Any

import torch

from qwenimg2512.samplers.base import BaseDiffusionStep, to_velocity


def _shape_key(tensor: torch.Tensor) -> tuple:
    """Get a hashable key from tensor shape for per-modality state tracking."""
    return tuple(tensor.shape)


class RES2SDiffusionStep(BaseDiffusionStep):
    """2nd-order RES single-step sampler (Heun/RK2).
    
    When used with the standard pipeline loop (Euler), this degrades to 1st-order.
    However, when used with 'res_2s_denoising_loop', it provides true 2nd-order
    accuracy by utilizing 2 model evaluations per step (Heun method).
    
    This is now fully supported in the LTX-2 GUI pipeline.
    """

    display_name = "RES 2S"
    description = "2nd-order single-step exponential sampler (Heun). High quality."

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one RES 2S diffusion step.

        Uses exponential integrator formulation for improved accuracy.
        If `model_fn` is provided, performs true 2nd-order (Heun) evaluation.
        Otherwise degrades to 1st-order (Euler).
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]

        # Handle final step
        if sigma_next == 0:
            return denoised_sample

        # 1st-order step (Euler) using exponential formulation
        ratio = sigma_next / sigma
        x_next_1st = ratio * sample.to(torch.float32) + (1 - ratio) * denoised_sample.to(torch.float32)

        if model_fn is None:
            # Degrade to 1st-order
            return x_next_1st.to(sample.dtype)

        # True 2nd-order step (Heun)
        # 1. Evaluate model at intermediate point x_next_1st
        denoised_next = model_fn(x_next_1st.to(sample.dtype), sigma_next.to(sample.dtype))
        
        # 2. Average the direction
        # A naive 0.5 average undercorrects for large steps and degrades to 1st-order.
        # Instead, we compute the exact exponential integration weight for 2nd-order accuracy.
        r = (sigma_next / sigma).to(torch.float32).clamp(min=1e-5)
        h = 1.0 - r
        h_safe = torch.where(h.abs() < 1e-4, torch.ones_like(h), h)
        w_exact = 1.0 / h_safe + (r * torch.log(r)) / (h_safe * h_safe)
        w_taylor = 0.5 + h / 6.0
        w = torch.where(h.abs() < 1e-4, w_taylor, w_exact)
        
        denoised_avg = (1.0 - w) * denoised_sample.to(torch.float32) + w * denoised_next.to(torch.float32)
                 
        # 3. Take step using the exact weighted prediction
        result = ratio * sample.to(torch.float32) + (1 - ratio) * denoised_avg

        return result.to(sample.dtype)


class RES2MDiffusionStep(BaseDiffusionStep):
    """2nd-order RES (Recursive Exponential Stepping) multistep sampler.

    Uses the previous denoised prediction to compute a 2nd-order accurate
    step. Falls back to Euler for the first step when no history is available.

    This sampler provides better detail preservation than Euler while
    maintaining good stability. Recommended for high-quality video generation.

    State is tracked per-tensor-shape to support multiple modalities.
    """

    display_name = "RES 2M"
    description = "2nd-order multistep exponential sampler. Better detail preservation than Euler."

    def __init__(self, **kwargs: Any) -> None:
        """Initialize RES 2M sampler."""
        super().__init__(**kwargs)
        # State tracked per shape: {shape_key: (prev_denoised, prev_sigma)}
        self._state_by_shape: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self) -> None:
        """Reset sampler state."""
        super().reset()
        self._state_by_shape.clear()

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one RES 2M diffusion step.

        Uses 2nd-order multistep method when history is available,
        falls back to Euler for the first step.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        shape_key = _shape_key(sample)

        # Handle final step
        if sigma_next == 0:
            # Store history even on final step for consistency
            self._state_by_shape[shape_key] = (denoised_sample.detach().clone(), sigma.detach().clone())
            return denoised_sample

        # Get state for this shape
        prev_state = self._state_by_shape.get(shape_key)

        # First step or no history: fall back to Euler
        if prev_state is None:
            # Standard Euler step using velocity formulation
            dt = sigma_next - sigma
            velocity = to_velocity(sample, sigma, denoised_sample)
            result = (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)
        else:
            prev_denoised, prev_sigma = prev_state

            # 2nd-order multistep using linear interpolation of denoised predictions
            #
            # We have denoised predictions at sigma (current) and prev_sigma (previous).
            # Use 2nd-order Adams-Bashforth style extrapolation in sigma space.
            #
            # The velocity at current step: v = (sample - denoised) / sigma
            # We can estimate the velocity derivative to get 2nd-order accuracy.

            dt = sigma_next - sigma

            # Current velocity
            v_curr = to_velocity(sample, sigma, denoised_sample)

            # Estimate velocity at previous sigma using the denoised at that time
            # Note: We need to estimate what the sample was at prev_sigma
            # Since we're moving from high sigma to low sigma, and we stored prev_denoised,
            # we can estimate the derivative of denoised w.r.t. sigma

            # Simple 2nd-order: use midpoint-like extrapolation
            # Estimate velocity change: dv/dsigma ≈ (v_curr - v_prev) / (sigma - prev_sigma)
            # But we don't have v_prev directly...

            # Alternative: use the denoised predictions for 2nd-order extrapolation
            # Assuming denoised changes smoothly, we can do:
            # denoised_extrapolated = denoised_sample + (denoised_sample - prev_denoised) * factor
            # where factor relates sigma_next to the sigma spacing

            d_sigma = sigma - prev_sigma
            if abs(d_sigma) > 1e-8:
                # Linear extrapolation of denoised prediction
                factor = dt / d_sigma
                denoised_extrap = denoised_sample + factor * (denoised_sample - prev_denoised)

                # Exact exponential integration weight for 2nd-order accuracy
                r = (sigma_next / sigma).to(torch.float32).clamp(min=1e-5)
                h = 1.0 - r
                h_safe = torch.where(h.abs() < 1e-4, torch.ones_like(h), h)
                w_exact = 1.0 / h_safe + (r * torch.log(r)) / (h_safe * h_safe)
                w_taylor = 0.5 + h / 6.0
                w = torch.where(h.abs() < 1e-4, w_taylor, w_exact)
                
                # Blend using exact exponential weight
                denoised_eff = (1.0 - w) * denoised_sample + w * denoised_extrap
                
                # Compute velocity with effective denoised estimate
                v_improved = to_velocity(sample, sigma, denoised_eff)
                result = (sample.to(torch.float32) + v_improved.to(torch.float32) * dt).to(sample.dtype)
            else:
                # Degenerate case: fall back to Euler
                result = (sample.to(torch.float32) + v_curr.to(torch.float32) * dt).to(sample.dtype)

        # Store history for this shape
        self._state_by_shape[shape_key] = (denoised_sample.detach().clone(), sigma.detach().clone())

        return result


class RES3MDiffusionStep(BaseDiffusionStep):
    """3rd-order RES (Recursive Exponential Stepping) multistep sampler.

    Uses two previous denoised predictions for 3rd-order accuracy.
    Falls back to lower-order methods when insufficient history is available.

    This sampler provides the highest accuracy but requires more memory
    for storing history. Best for maximum quality when compute is available.

    State is tracked per-tensor-shape to support multiple modalities.
    """

    display_name = "RES 3M"
    description = "3rd-order multistep exponential sampler. Highest accuracy, more memory usage."

    def __init__(self, **kwargs: Any) -> None:
        """Initialize RES 3M sampler."""
        super().__init__(**kwargs)
        # State tracked per shape: {shape_key: [(denoised, sigma), ...]}
        self._history_by_shape: dict[tuple, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def reset(self) -> None:
        """Reset sampler state."""
        super().reset()
        self._history_by_shape.clear()

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one RES 3M diffusion step."""
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        shape_key = _shape_key(sample)

        # Handle final step
        if sigma_next == 0:
            # Store history even on final step
            if shape_key not in self._history_by_shape:
                self._history_by_shape[shape_key] = []
            self._history_by_shape[shape_key].append((denoised_sample.detach().clone(), sigma.detach().clone()))
            if len(self._history_by_shape[shape_key]) > 2:
                self._history_by_shape[shape_key].pop(0)
            return denoised_sample

        dt = sigma_next - sigma

        # Get history for this shape
        history = self._history_by_shape.get(shape_key, [])
        num_history = len(history)

        if num_history == 0:
            # First step: Euler
            velocity = to_velocity(sample, sigma, denoised_sample)
            result = (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)

        elif num_history == 1:
            # Second step: 2nd-order (same as RES 2M)
            prev_denoised, prev_sigma = history[0]

            v_curr = to_velocity(sample, sigma, denoised_sample)

            d_sigma = sigma - prev_sigma
            if abs(d_sigma) > 1e-8:
                factor = dt / d_sigma
                denoised_extrap = denoised_sample + factor * (denoised_sample - prev_denoised)
                # Exact exponential integration weight for 2nd-order accuracy
                r = (sigma_next / sigma).to(torch.float32).clamp(min=1e-5)
                h = 1.0 - r
                h_safe = torch.where(h.abs() < 1e-4, torch.ones_like(h), h)
                w_exact = 1.0 / h_safe + (r * torch.log(r)) / (h_safe * h_safe)
                w_taylor = 0.5 + h / 6.0
                w = torch.where(h.abs() < 1e-4, w_taylor, w_exact)
                
                # Blend using exact exponential weight
                denoised_eff = (1.0 - w) * denoised_sample + w * denoised_extrap
                
                v_improved = to_velocity(sample, sigma, denoised_eff)
                result = (sample.to(torch.float32) + v_improved.to(torch.float32) * dt).to(sample.dtype)
            else:
                result = (sample.to(torch.float32) + v_curr.to(torch.float32) * dt).to(sample.dtype)

        else:
            # 3rd-order step using quadratic extrapolation
            prev_denoised_1, prev_sigma_1 = history[-1]  # Most recent
            prev_denoised_2, prev_sigma_2 = history[-2]  # Older

            v_curr = to_velocity(sample, sigma, denoised_sample)

            # Use Lagrange interpolation / quadratic extrapolation for denoised
            # Points: (prev_sigma_2, prev_denoised_2), (prev_sigma_1, prev_denoised_1), (sigma, denoised_sample)
            # Extrapolate to sigma_next

            s0, s1, s2 = prev_sigma_2, prev_sigma_1, sigma
            d0, d1, d2 = prev_denoised_2, prev_denoised_1, denoised_sample
            s_target = sigma_next

            # Lagrange basis polynomials evaluated at s_target
            denom_0 = (s0 - s1) * (s0 - s2)
            denom_1 = (s1 - s0) * (s1 - s2)
            denom_2 = (s2 - s0) * (s2 - s1)

            # Check for degenerate cases
            if abs(denom_0) < 1e-10 or abs(denom_1) < 1e-10 or abs(denom_2) < 1e-10:
                # Fall back to linear (2nd-order)
                d_sigma = sigma - prev_sigma_1
                if abs(d_sigma) > 1e-8:
                    factor = dt / d_sigma
                    denoised_extrap = denoised_sample + factor * (denoised_sample - prev_denoised_1)
                    # Exact exponential integration weight for 2nd-order accuracy
                    r = (sigma_next / sigma).to(torch.float32).clamp(min=1e-5)
                    h = 1.0 - r
                    h_safe = torch.where(h.abs() < 1e-4, torch.ones_like(h), h)
                    w_exact = 1.0 / h_safe + (r * torch.log(r)) / (h_safe * h_safe)
                    w_taylor = 0.5 + h / 6.0
                    w = torch.where(h.abs() < 1e-4, w_taylor, w_exact)
                    
                    # Blend using exact exponential weight
                    denoised_eff = (1.0 - w) * denoised_sample + w * denoised_extrap
                    
                    v_improved = to_velocity(sample, sigma, denoised_eff)
                    result = (sample.to(torch.float32) + v_improved.to(torch.float32) * dt).to(sample.dtype)
                else:
                    result = (sample.to(torch.float32) + v_curr.to(torch.float32) * dt).to(sample.dtype)
            else:
                L0 = ((s_target - s1) * (s_target - s2)) / denom_0
                L1 = ((s_target - s0) * (s_target - s2)) / denom_1
                L2 = ((s_target - s0) * (s_target - s1)) / denom_2

                # Quadratic extrapolation of denoised at sigma_next
                denoised_extrap = L0 * d0 + L1 * d1 + L2 * d2

                # Exact exponential integration weight for 2nd-order accuracy
                r = (sigma_next / sigma).to(torch.float32).clamp(min=1e-5)
                h = 1.0 - r
                h_safe = torch.where(h.abs() < 1e-4, torch.ones_like(h), h)
                w_exact = 1.0 / h_safe + (r * torch.log(r)) / (h_safe * h_safe)
                w_taylor = 0.5 + h / 6.0
                w = torch.where(h.abs() < 1e-4, w_taylor, w_exact)
                
                # Blend current and extrapolated for improved estimate using exact exponential weight
                denoised_eff = (1.0 - w) * denoised_sample + w * denoised_extrap

                v_improved = to_velocity(sample, sigma, denoised_eff)
                result = (sample.to(torch.float32) + v_improved.to(torch.float32) * dt).to(sample.dtype)

        # Store history (keep last 2) for this shape
        if shape_key not in self._history_by_shape:
            self._history_by_shape[shape_key] = []
        self._history_by_shape[shape_key].append((denoised_sample.detach().clone(), sigma.detach().clone()))
        if len(self._history_by_shape[shape_key]) > 2:
            self._history_by_shape[shape_key].pop(0)

        return result
