from math import *
import numpy as np

def truncate(a: float, num: int):
    truncated_a = int(a*(10**num))/(10**num)
    return truncated_a

def truncated_bisection(f, a, b, zero_tol=1e-8, max_iters=100, dps=20):
    """
    Find a root of f(x) in the interval [a,b] using bisection method.
    Requires [a,b] contains exactly one root that passes through the x-intersection
    """

    fa = truncate(f(a), dps) 
    fb = truncate(f(b), dps)

    # solve for the root
    for n in range(0, max_iters):
        mid = truncate((a+b)/2, dps)
        fmid = truncate(f(mid), dps)

        # quit if interval length is small enough
        if abs((b-a)/2) < zero_tol:
            return mid

        if fmid*fa < 0:
            b = mid
            fb = truncate(f(b), dps)
        elif fmid*fb < 0:
            a = mid
            fa = truncate(f(a), dps)
    
    return mid
