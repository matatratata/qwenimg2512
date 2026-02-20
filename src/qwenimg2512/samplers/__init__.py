"""Custom diffusion samplers for LTX-2 video generation.

This module provides a modular sampler system with multiple sampling algorithms
optimized for video generation. The default Euler sampler from LTX-2 is always
available as a fallback.

Available samplers:
- euler: First-order Euler method (default, from LTX-2)
- res_2m: 2nd-order RES multistep sampler (better detail preservation)
- res_3m: 3rd-order RES multistep sampler (higher accuracy)
- dpmpp_2m: DPM++ 2M multistep sampler (fast, good quality)
- dpmpp_2m_sde: DPM++ 2M SDE sampler (stochastic, more variation)

Usage:
    from qwenimg2512.samplers import get_sampler, SAMPLER_NAMES

    # Get a sampler instance
    sampler = get_sampler("res_2m")

    # Use in denoising loop
    new_sample = sampler.step(sample, denoised, sigmas, step_idx)
"""

from qwenimg2512.samplers.registry import (
    SAMPLER_NAMES,
    SAMPLER_DESCRIPTIONS,
    get_sampler,
    get_sampler_class,
    register_sampler,
)

from qwenimg2512.samplers.base import BaseDiffusionStep
from qwenimg2512.samplers.euler import EulerDiffusionStep
from qwenimg2512.samplers.euler_ancestral import EulerAncestralDiffusionStep
from qwenimg2512.samplers.res import RES2SDiffusionStep, RES2MDiffusionStep, RES3MDiffusionStep
from qwenimg2512.samplers.dpmpp import DPMPlusPlus2MDiffusionStep, DPMPlusPlus2MSDEDiffusionStep


__all__ = [
    # Registry functions
    "SAMPLER_NAMES",
    "SAMPLER_DESCRIPTIONS",
    "get_sampler",
    "get_sampler_class",
    "register_sampler",
    # Base class
    "BaseDiffusionStep",
    # Sampler implementations
    "EulerDiffusionStep",
    "EulerAncestralDiffusionStep",
    "RES2SDiffusionStep",
    "RES2MDiffusionStep",
    "RES3MDiffusionStep",
    "DPMPlusPlus2MDiffusionStep",
    "DPMPlusPlus2MSDEDiffusionStep",
]
