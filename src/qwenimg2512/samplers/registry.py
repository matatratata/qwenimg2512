"""Sampler registry for managing available samplers.

Provides a central registry for all sampler implementations,
making it easy to add new samplers without modifying other code.
"""

from __future__ import annotations

from typing import Any, Type

from qwenimg2512.samplers.base import BaseDiffusionStep


# Global registry of sampler classes
_SAMPLER_REGISTRY: dict[str, Type[BaseDiffusionStep]] = {}


def register_sampler(name: str, sampler_class: Type[BaseDiffusionStep]) -> None:
    """Register a sampler class with the given name.

    Args:
        name: Unique identifier for the sampler (lowercase, no spaces)
        sampler_class: The sampler class to register

    Raises:
        ValueError: If name is already registered
    """
    if name in _SAMPLER_REGISTRY:
        raise ValueError(f"Sampler '{name}' is already registered")
    _SAMPLER_REGISTRY[name] = sampler_class


def get_sampler_class(name: str) -> Type[BaseDiffusionStep] | None:
    """Get a sampler class by name.

    Args:
        name: The registered sampler name

    Returns:
        The sampler class, or None if not found
    """
    return _SAMPLER_REGISTRY.get(name)


def get_sampler(name: str, **kwargs: Any) -> BaseDiffusionStep | None:
    """Create a sampler instance by name.

    Args:
        name: The registered sampler name
        **kwargs: Arguments to pass to the sampler constructor

    Returns:
        A new sampler instance, or None if name not found
    """
    sampler_class = get_sampler_class(name)
    if sampler_class is None:
        return None
    return sampler_class(**kwargs)


def get_default_sampler(**kwargs: Any) -> BaseDiffusionStep:
    """Get the default (Euler) sampler.

    Returns:
        A new Euler sampler instance
    """
    from qwenimg2512.samplers.euler import EulerDiffusionStep
    return EulerDiffusionStep(**kwargs)


# Property to get list of sampler names
@property
def SAMPLER_NAMES() -> list[str]:
    """Get list of all registered sampler names."""
    return list(_SAMPLER_REGISTRY.keys())


# Make SAMPLER_NAMES a module-level variable
def _get_sampler_names() -> list[str]:
    """Get list of all registered sampler names."""
    return list(_SAMPLER_REGISTRY.keys())


def _get_sampler_descriptions() -> dict[str, str]:
    """Get descriptions for all registered samplers."""
    return {
        name: cls.description
        for name, cls in _SAMPLER_REGISTRY.items()
    }


# Register built-in samplers
def _register_builtin_samplers() -> None:
    """Register all built-in sampler implementations."""
    from qwenimg2512.samplers.euler import EulerDiffusionStep
    from qwenimg2512.samplers.euler_ancestral import EulerAncestralDiffusionStep
    from qwenimg2512.samplers.res import RES2SDiffusionStep, RES2MDiffusionStep, RES3MDiffusionStep
    from qwenimg2512.samplers.dpmpp import DPMPlusPlus2MDiffusionStep, DPMPlusPlus2MSDEDiffusionStep
    from qwenimg2512.samplers.ge_euler import GEEulerDiffusionStep

    # Default sampler
    register_sampler("euler", EulerDiffusionStep)

    # Euler Ancestral (stochastic)
    register_sampler("euler_a", EulerAncestralDiffusionStep)

    # GE Euler (gradient estimation — free 2nd-order gains)
    register_sampler("ge_euler", GEEulerDiffusionStep)

    # RES samplers (exponential integrators)
    register_sampler("res_2s", RES2SDiffusionStep)
    register_sampler("res_2m", RES2MDiffusionStep)
    register_sampler("res_3m", RES3MDiffusionStep)

    # DPM++ samplers
    register_sampler("dpmpp_2m", DPMPlusPlus2MDiffusionStep)
    register_sampler("dpmpp_2m_sde", DPMPlusPlus2MSDEDiffusionStep)


# Auto-register on import
_register_builtin_samplers()

# These are populated after registration
SAMPLER_NAMES: list[str] = _get_sampler_names()
SAMPLER_DESCRIPTIONS: dict[str, str] = _get_sampler_descriptions()
