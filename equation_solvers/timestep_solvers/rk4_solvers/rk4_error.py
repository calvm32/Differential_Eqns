import numpy as np
from .rk4 import rk4

def rk4_error(y_exact, f, y0, t0, T): 
    """
    Find the error of the ODE y' = f(t,y) on the interval [t0,T] with y (t0) = y0,
    when solved using the Runge-Kutta-4 3D approximation method 
    """
    
    resolutions = np.arange(10, 1001, 10)
    global_errors = []

    for N in resolutions:
        local_errors = []
        dt = (T - t0) / (N-1) # convert from N to dt
        y_approx , t = rk4(f , y0, t0, T, dt)

        for i in range(len(t)):
            local_errors.append(abs(y_approx[i] - y_exact(t[i])))

        global_errors.append(max(local_errors))
    
    global_errors = np.array(global_errors)

    return global_errors, resolutions