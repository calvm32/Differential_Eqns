import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.rk4_solvers.rk4_fourier import *

"""
Solve 1D heat equation u_t = nu u_{xx} + f using Fourier coefficients
    -> periodic BCs on [0, L]
    -> evolve u_hat w/ RK4

Fourier form: u_hat_t = -nu*ksq*u_hat + f_hat
"""

def main():
    
    # --------------------------
    # setup constants, functions
    # --------------------------

    nu, t0, T = 0.01, 0.0, 5.0 # diffusion coeff, init time, final time
    L, N = 2*np.pi, 2**8 # Domain length, number of points

    # get linear space
    dx = L/N
    x = np.arange(N)*dx

    # Time-step CFL-restriction
    dt = 0.2*dx*dx/nu               
    t = np.arange(t0,T + 1e-12,dt)  # include T (up to roundoff)

    # IC
    u0 = x**3

    # forcing
    f = 0*x

    # ---------------
    # solve and graph
    # ---------------

    def rhs(u_hat,f_hat,ksq,nu):
        # Fourier-space RHS: u_hat_t = nu*ksq*u_hat
        return nu*ksq*u_hat

    # Fourier wavenumbers for second derivative: u_xx_hat = -(k^2) u_hat
    freq = np.fft.fftfreq(N,d=dx)   # cycles per unit length (FFT-ordered)
    k = 2*np.pi*freq                # Fourier wavenumbers
    ksq = -(k**2)

    # initialize the Fourier coefficients of u
    u_hat_0 = np.fft.fft(u0) 

    # initialize the Fourier coefficients of f
    f_hat = np.fft.fft(f) 

    # Compute the Fourier coefficients of u
    u_hat, times = rk4_fourier(rhs, u_hat_0, f_hat, t0, T, dt, ksq, nu, t)

    # Plotting
    plt.ion()
    fig, ax = plt.subplots()

    # main time loop
    for n in range(len(t)-1):  
        ax.clear()
        ax.plot(x,np.fft.ifft(u_hat[n]).real,linewidth=2) # inverse Fourier
        ax.set_xlim(0,L); ax.set_ylim(-2,2) # Fix axes
        ax.set_title(f't = {t[n+1]:1.3f}')
        plt.pause(0.001) # Slow down animation
        
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()