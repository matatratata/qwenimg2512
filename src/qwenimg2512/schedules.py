"""Custom sigma schedules for diffusion models."""

import math
import numpy as np
from scipy import stats

SCHEDULE_NAMES = [
    "default",
    "bong_tangent",
    "beta57",
]

SCHEDULE_DESCRIPTIONS = {
    "default": "Default scheduler for the used pipeline.",
    "bong_tangent": "Tangent-based schedule optimized for high detail. Concentrates inference steps where they visually matter most.",
    "beta57": "Beta schedule with strong structural composition (alpha=0.5, beta=0.7). Spends more time on early structural steps.",
}

def get_bong_tangent_sigmas(steps: int, slope: float, pivot: float, start: float, end: float) -> list[float]:
    """Helper for bong_tangent_schedule to generate tangent sigmas over a given step range."""
    # Guard against 0 or 1 step edge cases breaking srange calculation
    if steps <= 1:
        return [end] * max(0, steps)
        
    smax = ((2 / math.pi) * math.atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / math.pi) * math.atan(-slope * ((steps - 1) - pivot)) + 1) / 2
    srange = smax - smin
    sscale = start - end
    
    # Prevent Division By Zero
    if srange == 0:
        return [end] * steps
        
    sigmas = [
        ((((2 / math.pi) * math.atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end
        for x in range(steps)
    ]
    return sigmas

def get_bong_tangent_schedule(
    steps: int,
    start: float = 1.0,
    middle: float = 0.5,
    end: float = 0.0,
    pivot_1: float = 0.6,
    pivot_2: float = 0.6,
    slope_1: float = 0.2,
    slope_2: float = 0.2,
) -> list[float]:
    """Generate the 'bong_tangent' sigma schedule sequence."""
    steps += 2
    midpoint = int((steps * pivot_1 + steps * pivot_2) / 2)
    pivot_1_idx = int(steps * pivot_1)
    pivot_2_idx = int(steps * pivot_2)
    slope_1 = slope_1 / (steps / 40)
    slope_2 = slope_2 / (steps / 40)

    stage_2_len = steps - midpoint
    stage_1_len = steps - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1_idx, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2_idx - stage_1_len, middle, end)
    
    tan_sigmas_1 = tan_sigmas_1[:-1]
    tan_sigmas = tan_sigmas_1 + tan_sigmas_2
    
    # Optional clamp and rounding
    return [max(0.0, float(x)) for x in tan_sigmas]

def get_beta57_schedule(steps: int, start: float = 1.0, end: float = 0.0) -> list[float]:
    """Generate the 'beta57' sigma schedule sequence."""
    # BUG FIX: Use steps + 1 so the resulting array naturally covers the terminal 0.0
    t = np.linspace(0, 1, steps + 1)
    quantiles = stats.beta.ppf(t, 0.5, 0.7)
    sigmas = start - quantiles * (start - end)
    
    return [max(0.0, float(x)) for x in sigmas]
