import numpy as np

def euler_forward(f, y0, t0, T, dt): 
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y (t0) = y0
    using the forward Euler approximation method 
    """
    
    N = int(np.floor((T-t0)/dt) + 1) # fixed number of steps
    t = t0+dt*np.arange(0,N)

    y = np.zeros(N , dtype = float)
    y[0] = y0 # Set initial value

    for n in range(0,N-1):
        y[n+1] = y[n] + dt*f(t[n],y[n])
    
    return y,t