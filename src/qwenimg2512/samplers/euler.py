"""Euler diffusion sampler.

This wraps the original LTX-2 Euler sampler to fit our interface.
It serves as the default fallback sampler.
"""

from typing import Any, Callable

import torch

from qwenimg2512.samplers.base import BaseDiffusionStep, to_velocity


class EulerDiffusionStep(BaseDiffusionStep):
    """First-order Euler method for diffusion sampling.

    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.

    This is the default sampler used by LTX-2 and provides a good balance of
    speed and quality.
    """

    display_name = "Euler"
    description = "First-order Euler method. Fast and stable, the default for LTX-2."

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one Euler diffusion step.

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
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)
