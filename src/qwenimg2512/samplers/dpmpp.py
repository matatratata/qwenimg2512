"""DPM++ (DPM-Solver++) multistep samplers.

These samplers are based on the DPM-Solver++ algorithm which provides
fast and high-quality sampling. They are widely used in image/video
diffusion models.

Reference:
- DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
  https://arxiv.org/abs/2211.01095

Note: These samplers track state per-tensor-shape to support multiple
modalities (video/audio) being processed with the same sampler instance.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from qwenimg2512.samplers.base import BaseDiffusionStep


def _sigma_to_t(sigma: torch.Tensor) -> torch.Tensor:
    """Convert sigma to continuous timestep t.

    t = -log(sigma)
    """
    return -torch.log(sigma)


def _t_to_sigma(t: torch.Tensor) -> torch.Tensor:
    """Convert continuous timestep t to sigma.

    sigma = exp(-t)
    """
    return torch.exp(-t)


def _shape_key(tensor: torch.Tensor) -> tuple:
    """Get a hashable key from tensor shape for per-modality state tracking."""
    return tuple(tensor.shape)


class DPMPlusPlus2MDiffusionStep(BaseDiffusionStep):
    """DPM++ 2M (multistep) sampler.

    Second-order multistep DPM-Solver++ with exponential integrator.
    Provides excellent quality with good speed. One of the most popular
    samplers for diffusion models.

    This is a deterministic sampler (no noise injection between steps).
    State is tracked per-tensor-shape to support multiple modalities.
    """

    display_name = "DPM++ 2M"
    description = "DPM-Solver++ 2M multistep. Fast, high-quality, deterministic."

    def __init__(self, **kwargs: Any) -> None:
        """Initialize DPM++ 2M sampler."""
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
        """Perform one DPM++ 2M step."""
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        shape_key = _shape_key(sample)

        # Handle final step (sigma_next = 0)
        if sigma_next == 0:
            # Store for next generation consistency
            self._state_by_shape[shape_key] = (denoised_sample.detach().clone(), sigma.detach().clone())
            return denoised_sample

        # Convert to continuous time
        t = _sigma_to_t(sigma)
        t_next = _sigma_to_t(sigma_next)
        h = t_next - t  # step size in t-space

        # Get previous state for this shape
        prev_state = self._state_by_shape.get(shape_key)

        if prev_state is None:
            # First step: use 1st-order DPM-Solver (equivalent to DDIM)
            x_next = (sigma_next / sigma) * sample + (1 - sigma_next / sigma) * denoised_sample
        else:
            # 2nd-order multistep
            prev_denoised, prev_sigma = prev_state
            t_prev = _sigma_to_t(prev_sigma)
            h_prev = t - t_prev

            # Compute coefficients
            r = h_prev / h if h != 0 else 0

            # DPM++ 2M update formula
            if r != 0:
                D = (1 + 1 / (2 * r)) * denoised_sample - (1 / (2 * r)) * prev_denoised
            else:
                D = denoised_sample

            x_next = (sigma_next / sigma) * sample + (1 - sigma_next / sigma) * D

        # Store for next step
        self._state_by_shape[shape_key] = (denoised_sample.detach().clone(), sigma.detach().clone())

        return x_next.to(sample.dtype)


class DPMPlusPlus2MSDEDiffusionStep(BaseDiffusionStep):
    """DPM++ 2M SDE sampler.

    Stochastic version of DPM++ 2M that adds noise at each step.
    Provides more variation in outputs and can help with diversity.

    State is tracked per-tensor-shape to support multiple modalities.
    """

    display_name = "DPM++ 2M SDE"
    description = "DPM-Solver++ 2M SDE. Stochastic, more variation in outputs."

    def __init__(
        self,
        eta: float = 1.0,
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DPM++ 2M SDE sampler.

        Args:
            eta: Amount of noise to add (0 = deterministic, 1 = full stochastic)
            s_noise: Noise scale multiplier
            generator: Random generator for reproducibility
        """
        super().__init__(**kwargs)
        self.eta = eta
        self.s_noise = s_noise
        self.generator = generator
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
        """Perform one DPM++ 2M SDE step."""
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        shape_key = _shape_key(sample)

        # Handle final step
        if sigma_next == 0:
            return denoised_sample

        # Convert to continuous time
        t = _sigma_to_t(sigma)
        t_next = _sigma_to_t(sigma_next)
        h = t_next - t

        # Noise parameters
        s = torch.sqrt(torch.tensor(1.0 + h**2, device=sigma.device, dtype=sigma.dtype)) - 1
        if self.eta > 0:
            sigma_up = torch.sqrt(s * (1 + 1/s)) * sigma_next * self.eta
            sigma_down = torch.sqrt(sigma_next**2 - sigma_up**2)
        else:
            sigma_up = torch.tensor(0.0, device=sigma.device, dtype=sigma.dtype)
            sigma_down = sigma_next

        # Get previous state for this shape
        prev_state = self._state_by_shape.get(shape_key)

        if prev_state is None:
            # First step: 1st order
            x_next = (sigma_down / sigma) * sample + (1 - sigma_down / sigma) * denoised_sample
        else:
            prev_denoised, prev_sigma = prev_state
            # 2nd-order multistep
            t_prev = _sigma_to_t(prev_sigma)
            h_prev = t - t_prev

            r = h_prev / h if h != 0 else 0

            if r != 0:
                D = (1 + 1 / (2 * r)) * denoised_sample - (1 / (2 * r)) * prev_denoised
            else:
                D = denoised_sample

            x_next = (sigma_down / sigma) * sample + (1 - sigma_down / sigma) * D

        # Add noise
        if sigma_up > 0:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=self.generator,
            )
            x_next = x_next + noise * sigma_up * self.s_noise

        # Store history for this shape
        self._state_by_shape[shape_key] = (denoised_sample.detach().clone(), sigma.detach().clone())

        return x_next.to(sample.dtype)
