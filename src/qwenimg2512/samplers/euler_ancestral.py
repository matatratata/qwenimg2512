"""Euler Ancestral sampler.

This sampler adds stochastic noise at each step, providing more variation
in outputs compared to the deterministic Euler method.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from qwenimg2512.samplers.base import BaseDiffusionStep, to_velocity


class EulerAncestralDiffusionStep(BaseDiffusionStep):
    """Euler Ancestral (Euler A) sampler with stochastic noise injection.

    This sampler is similar to Euler but adds ancestral noise at each step,
    which introduces stochasticity and can lead to more varied outputs.
    The eta parameter controls the amount of noise added.

    This is particularly useful for video generation when you want more
    creative variation between generations with the same seed.
    """

    display_name = "Euler Ancestral"
    description = "Euler with ancestral noise. Stochastic, good variation between seeds."

    def __init__(
        self,
        eta: float = 1.0,
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Euler Ancestral sampler.

        Args:
            eta: Amount of noise to add (0 = deterministic, 1 = full stochastic)
            s_noise: Noise scale multiplier
            generator: Random generator for reproducibility
        """
        super().__init__(**kwargs)
        self.eta = eta
        self.s_noise = s_noise
        self.generator = generator

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one Euler Ancestral diffusion step.

        Args:
            sample: Current noisy sample tensor
            denoised_sample: Model's denoised prediction (x0)
            sigmas: Full sigma schedule tensor
            step_index: Current step index in the schedule

        Returns:
            Updated sample tensor at the next noise level
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]

        # Handle final step
        if sigma_next == 0:
            return denoised_sample

        # Compute sigma_up and sigma_down for ancestral sampling
        # sigma_up^2 + sigma_down^2 = sigma_next^2
        # sigma_up = eta * sigma_next * sqrt(1 - (sigma_next/sigma)^2)
        sigma_up = torch.sqrt(sigma_next**2 * (1 - (sigma_next / sigma)**2).clamp(min=0)) * self.eta
        sigma_up = sigma_up.clamp(max=sigma_next)
        sigma_down = torch.sqrt(sigma_next**2 - sigma_up**2)

        # Euler step to sigma_down
        velocity = to_velocity(sample, sigma, denoised_sample)
        dt = sigma_down - sigma
        x_next = sample.to(torch.float32) + velocity.to(torch.float32) * dt

        # Add ancestral noise
        if sigma_up > 0:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=self.generator,
            )
            x_next = x_next + noise * sigma_up * self.s_noise

        return x_next.to(sample.dtype)
