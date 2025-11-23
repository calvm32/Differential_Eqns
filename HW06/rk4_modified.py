import numpy as np

def rk4_modified(f1, f2, y1_0, y2_0, t0, T, dt): 
    """
    Solve two related ODEs on the interval [t0,T] with y (t0) = y0
    using the Runge-Kutta-4 3D approximation method 
    
    Parameters
    ----------
    f1      : function 1
    f2      : function 2
    y1_0    : initial condition for func 1
    y2_0    : initial condition for func 2 
    t0      : initial time
    T       : final time
    dt      : step size
    
    Returns
    -------
    y1      : approximate solution for func 1
    y2      : approximate solution for func 2
    t       : time resolutions
    
    """
    N = int(np.floor((T-t0)/dt) + 1) # fixed number of steps
    t = t0+dt*np.arange(0,N)
    
    y1 = np.zeros((3, N), dtype = float)
    y1[:, 0] = y1_0 # Set initial value

    y2 = np.zeros((3, N), dtype = float)
    y2[:, 0] = y2_0 # Set initial value

    for n in range(0,N-1):

        # solve func 1
        k1 = f1(t[n], y1[:, n])
        k2 = f1(t[n] + dt/2, y[:, n] + (dt*k1)/2)
        k3 = f1(t[n] + dt/2, y[:, n] + (dt*k2)/2)
        k4 = f1(t[n] + dt, y[:, n] + (dt*k3))

        y1[:, n+1] = y1[:, n] + (dt/6)*(k1 + (2*k2) + (2*k3)+k4)

        # solve func 2, reuse names bc oh well
        k1 = f2(t[n], y2[:, n])
        k2 = f2(t[n] + dt/2, y[:, n] + (dt*k1)/2)
        k3 = f2(t[n] + dt/2, y[:, n] + (dt*k2)/2)
        k4 = f2(t[n] + dt, y[:, n] + (dt*k3))

        y2[:, n+1] = y2[:, n] + (dt/6)*(k1 + (2*k2) + (2*k3)+k4)
    
    return y1, y2, t