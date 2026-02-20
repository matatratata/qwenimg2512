"""Base class for diffusion samplers.

Defines the interface that all samplers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class BaseDiffusionStep(ABC):
    """Base class for diffusion stepping algorithms.

    All samplers must implement the `step` method which advances the sample
    from the current noise level to the next according to the sigma schedule.

    Samplers may optionally store state between steps (for multistep methods).
    Call `reset()` before starting a new generation to clear any stored state.
    """

    # Human-readable name for UI display
    display_name: str = "Base Sampler"

    # Description for tooltips
    description: str = "Base diffusion sampler"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the sampler.

        Args:
            **kwargs: Sampler-specific configuration options
        """
        self._state: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset sampler state for a new generation.

        Called before starting a new denoising loop to clear any
        stored state from previous generations.
        """
        self._state.clear()

    @abstractmethod
    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform one diffusion step.

        Args:
            sample: Current noisy sample tensor.
            denoised_sample: Model's denoised prediction (x0) for the current sample.
            sigmas: Full schedule array or tensor.
            step_index: Current step index in the schedule.
            model_fn: Optional callback `(sample, sigma) -> denoised` to evaluate the 
                model at intermediate steps (required for true 2nd+ order single-step methods like Heun/RK2).

        Returns:
            Updated sample tensor at the next noise level.
        """
        ...

    def get_velocity(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        denoised_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity from sample and denoised prediction.

        The velocity represents the direction of the diffusion process.
        v = (sample - denoised) / sigma

        Args:
            sample: Current noisy sample
            sigma: Current noise level
            denoised_sample: Model's denoised prediction

        Returns:
            Velocity tensor
        """
        return (sample - denoised_sample) / sigma.view(-1, 1, 1, 1) if sigma.ndim == 1 else (sample - denoised_sample) / sigma

    def get_denoised(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute denoised sample from sample and velocity.

        Inverse of get_velocity.
        denoised = sample - velocity * sigma

        Args:
            sample: Current noisy sample
            velocity: Current velocity
            sigma: Current noise level

        Returns:
            Denoised sample tensor
        """
        return sample - velocity * sigma.view(-1, 1, 1, 1) if sigma.ndim == 1 else sample - velocity * sigma


def to_velocity(sample: torch.Tensor, sigma: float | torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Convert from denoised prediction to velocity.

    This is a standalone utility function for compatibility with existing code.

    Args:
        sample: Noisy sample
        sigma: Noise level (scalar or tensor)
        denoised: Denoised prediction

    Returns:
        Velocity tensor
    """
    if isinstance(sigma, torch.Tensor):
        if sigma.ndim == 0:
            return (sample - denoised) / sigma
        return (sample - denoised) / sigma.view(-1, 1, 1, 1)
    return (sample - denoised) / sigma


def to_denoised(sample: torch.Tensor, velocity: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Convert from velocity to denoised prediction.

    Args:
        sample: Noisy sample
        velocity: Velocity
        sigma: Noise level

    Returns:
        Denoised prediction tensor
    """
    if isinstance(sigma, torch.Tensor):
        if sigma.ndim == 0:
            return sample - velocity * sigma
        return sample - velocity * sigma.view(-1, 1, 1, 1)
    return sample - velocity * sigma
