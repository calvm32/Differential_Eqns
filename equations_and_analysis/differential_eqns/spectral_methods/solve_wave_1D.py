import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.misc.wave_fourier import *

"""
Solve 1D wave equation u_tt = c^2 u_{xx} using Fourier coefficients
    -> periodic BCs on [-L, L]

Fourier form: u_hat_tt = -c^2*ksq*u_hat
"""

def main():
    
    # --------------------------
    # setup constants, functions
    # --------------------------

    c, t0, T = 1.0, 0.0, 5.0 # diffusion coeff, init time, final time
    L, N = np.pi, 2**8 # Domain length, number of points

    # get linear space
    dx = L/N
    x = np.arange(-N,N)*dx

    # Time-step CFL-restriction
    dt = 0.2*dx/c        
    t = np.arange(t0,T + 1e-12,dt)  # include T (up to roundoff)

    # IC
    u0 = np.maximum(0, 1 - abs(x))
    v0 = 0*x

    # ---------------
    # solve and graph
    # ---------------

    # Fourier wavenumbers for second derivative: u_xx_hat = -(k^2) u_hat
    freq = np.fft.fftfreq(2*N, d=dx)    # cycles per unit length (FFT-ordered)
    k = 2*np.pi*freq                    # Fourier wavenumbers
    ksq = -(k**2)

    # initialize the Fourier coefficients of u, u'
    u_hat_0 = np.fft.fft(u0)

    # get the next time step to start our timestep solver
    u_hat_1 = u_hat_0 + dt*np.fft.fft(v0) + c**2*ksq*dt*u_hat_0

    # Compute the Fourier coefficients of u
    u_hat, times = wave_fourier(u_hat_0, u_hat_1, t0, T, dt, c, ksq)
    u_hat = u_hat.T

    # Plotting
    plt.ion()
    fig, ax = plt.subplots()

    # main time loop
    for n in range(len(t)-1):  
        ax.clear()
        ax.plot(x,np.fft.ifft(u_hat[n]).real,linewidth=2) # inverse Fourier
        ax.set_xlim(-L,L); ax.set_ylim(-2,2) # Fix axes
        ax.set_title(f't = {t[n+1]:1.3f}')
        plt.pause(0.001) # Slow down animation
        
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()