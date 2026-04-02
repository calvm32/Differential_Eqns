import numpy as np

def wave_fourier(u_hat_0, u_hat_1, t0, T, dt, c, ksq, t=[]):
    """
    Solve the ODE y'' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the 2nd-order taylor series
    """
    
    N = int(np.floor((T - t0) / dt) + 1) # fixed number of steps
    if len(t) == 0:
        t = t0+dt*np.arange(N)

    u_hat = np.zeros(u_hat_0.shape + (N,), dtype=complex) # append time axis N
    u_hat[..., 0] = u_hat_0 # Set initial Fourier coefficients
    u_hat[..., 1] = u_hat_1 # Set initial Fourier coefficients

    for n in range(1, N - 1):
        u_hat[:, n+1] = 2*u_hat[:,n] - u_hat[:,n-1] + (c**2)*(ksq)*(dt**2)*u_hat[:,n]

    return u_hat, t