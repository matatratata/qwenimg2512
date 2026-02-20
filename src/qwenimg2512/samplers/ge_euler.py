"""Gradient Estimating (GE) Euler sampler.

Uses velocity momentum/extrapolation across steps to achieve higher-order
corrections without extra model evaluations. Provides ~2nd-order quality
gains with only 1 NFE per step.

Reference: https://openreview.net/pdf?id=o2ND9v0CeK
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from qwenimg2512.samplers.base import BaseDiffusionStep, to_velocity


def _shape_key(tensor: torch.Tensor) -> tuple:
    """Get a hashable key from tensor shape for per-modality state tracking."""
    return tuple(tensor.shape)


class GEEulerDiffusionStep(BaseDiffusionStep):
    """Gradient Estimating (GE) Euler multistep sampler.

    Applies gradient estimation to improve the denoised estimates by tracking velocity
    changes across steps. Provides higher-order accuracy with single model evaluations.

    Reference: https://openreview.net/pdf?id=o2ND9v0CeK
    """

    display_name = "GE Euler"
    description = "Euler with gradient estimation (velocity extrapolation). High quality."

    def __init__(self, ge_gamma: float = 2.0, **kwargs: Any) -> None:
        """Initialize GE Euler sampler.

        Args:
            ge_gamma: Gradient estimation strength. Default 2.0 for optimal
                      2nd-order correction. Lower values reduce correction strength.
        """
        super().__init__(**kwargs)
        self.ge_gamma = ge_gamma
        # Track previous velocity per shape for multiple modalities
        self._prev_velocity_by_shape: dict[tuple, torch.Tensor] = {}

    def reset(self) -> None:
        """Reset sampler state."""
        super().reset()
        self._prev_velocity_by_shape.clear()

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one GE Euler diffusion step.

        Uses gradient estimation to extrapolate the velocity from the previous
        step, achieving higher-order accuracy without additional model evaluations.
        Falls back to standard Euler on the first step when no history is available.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        shape_key = _shape_key(sample)

        # Handle final step
        if sigma_next == 0:
            return denoised_sample

        dt = sigma_next - sigma
        current_velocity = to_velocity(sample, sigma, denoised_sample)

        # Get previous state for this tensor shape
        prev_velocity = self._prev_velocity_by_shape.get(shape_key)

        if prev_velocity is not None:
            # Gradient estimation: extrapolate velocity
            delta_v = current_velocity - prev_velocity
            total_velocity = self.ge_gamma * delta_v + prev_velocity
        else:
            total_velocity = current_velocity

        # Store uncorrected velocity for next step's gradient
        self._prev_velocity_by_shape[shape_key] = current_velocity.clone()

        # Take step using the extrapolated velocity (Euler update)
        result = sample.to(torch.float32) + total_velocity.to(torch.float32) * dt
        return result.to(sample.dtype)
