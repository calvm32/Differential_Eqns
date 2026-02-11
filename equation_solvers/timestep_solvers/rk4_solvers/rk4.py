import numpy as np

def rk4(f, y0, t0, T, dt): 
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y (t0) = y0
    using the Runge-Kutta-4 approximation method 
    """
    
    N = int(np.floor((T-t0)/dt) + 1) # fixed number of steps
    t = t0+dt*np.arange(0,N)

    y = np.zeros(N , dtype = float)
    y[0] = y0 # Set initial value

    for n in range(0,N-1):

        k1 = f(t[n],y[n])
        k2 = f(t[n] + dt/2 ,y[n] + (dt*k1)/2)
        k3 = f(t[n] + dt/2 ,y[n] + (dt*k2)/2)
        k4 = f(t[n] + dt ,y[n] + (dt*k3))

        y[n+1] = y[n] + (dt/6)*(k1 + (2*k2) + (2*k3)+k4)
    
    return y,t