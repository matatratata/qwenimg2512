import numpy as np
from scipy import stats
import math

def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2/math.pi)*math.atan(-slope*(0-pivot))+1)/2
    smin = ((2/math.pi)*math.atan(-slope*((steps-1)-pivot))+1)/2
    srange = smax-smin
    sscale = start - end
    sigmas = [ ((((2/math.pi)*math.atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end for x in range(steps)]
    return sigmas

def bong_tangent_scheduler(steps, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2):
    steps += 2
    midpoint = int( (steps*pivot_1 + steps*pivot_2) / 2 )
    pivot_1_idx = int(steps * pivot_1)
    pivot_2_idx = int(steps * pivot_2)
    slope_1 = slope_1 / (steps/40)
    slope_2 = slope_2 / (steps/40)

    stage_2_len = steps - midpoint
    stage_1_len = steps - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1_idx, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2_idx - stage_1_len, middle, end)
    
    tan_sigmas_1 = tan_sigmas_1[:-1]
    tan_sigmas = tan_sigmas_1 + tan_sigmas_2
    return tan_sigmas

def beta57_scheduler(steps, start=1.0, end=0.0):
    t = np.linspace(0, 1, steps)
    quantiles = stats.beta.ppf(t, 0.5, 0.7)
    sigmas = start - quantiles * (start - end)
    return list(sigmas)

if __name__ == "__main__":
    b = bong_tangent_scheduler(20) 
    print(f"bong_tangent(20): len {len(b)}")
    print([round(x, 3) for x in b])
    
    b57 = beta57_scheduler(21)
    print(f"beta57(21): len {len(b57)}")
    print([round(x, 3) for x in b57])
