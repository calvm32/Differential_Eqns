import numpy as np

def rk4_fourier(rhs, u_hat_0, f_hat, t0, T, dt, ksq, nu, t="DEFAULT"):
    """
    Solve the ODE y' = f(t,y) on the interval [t0,T] with y(t0) = y0
    using the Runge-Kutta-4 approximation method 
    """
    
    N = int(np.floor((T - t0) / dt) + 1)  # fixed number of steps
    if len(t) == 0:
        t = t0+dt*np.arange(0,N)

    dim = u_hat_0.shape[0]
    u_hat = np.zeros((dim, N, len(u_hat_0)), dtype=complex)  # Change from (N, N) to (N, len(u_hat_0))
    u_hat[:, 0] = u_hat_0   # Set initial Fourier coefficients

    for n in range(0, N - 1):
        k1 = rhs(u_hat[:,n], f_hat, ksq, nu)
        k2 = rhs(u_hat[:,n] + dt/2 * k1, f_hat, ksq, nu)
        k3 = rhs(u_hat[:,n] + dt/2 * k2, f_hat, ksq, nu)
        k4 = rhs(u_hat[:,n] + dt * k3, f_hat, ksq, nu)

        u_hat[:,n + 1] = u_hat[:,n] + (dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return u_hat, t