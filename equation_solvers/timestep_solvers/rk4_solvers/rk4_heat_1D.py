import numpy as np
import matplotlib.pyplot as plt
from .rk4 import rk4

def rhs(u_hat,ksq,nu):
    # Fourier-space RHS: u_hat_t = nu*ksq*u_hat
    return nu*ksq*u_hat

def rk4_heat_1D(nu, t0, T, L, N):
    # u_t = nu u_xx on [0,L) periodic; 
    # evolve u_hat with RK4: u_hat_t = nu*ksq*u_hat

    dx = L/N
    x = np.arange(N)*dx

    # Fourier wavenumbers for second derivative: u_xx_hat = -(k^2) u_hat
    freq = np.fft.fftfreq(N,d=dx) # cycles per unit length (FFT-ordered)
    k = 2*np.pi*freq               # Fourier wavenumbers
    ksq = -(k**2)

    dt = 0.2*dx*dx/nu # Time-step CFL-restriction
    t = np.arange(t0,T + 1e-12,dt) # include T (up to roundoff)

    u = np.sin(x) + 0.5*np.cos(7*x) + 0.25*np.cos(33*x)
    u_hat = np.fft.fft(u) # Compute the Fourier coefficients of u

    plt.ion()
    fig, ax = plt.subplots()
    for n in range(len(t)-1):  # main time loop
        ax.clear()
        ax.plot(x,np.fft.ifft(u_hat).real,linewidth=2) # inverse Fourier
        ax.set_xlim(0,L); ax.set_ylim(-2,2) # Fix axes
        ax.set_title(f't = {t[n+1]:1.3f}')
        plt.pause(0.001) # Slow down animation

    rk4(f, y0, t0, T, dt)
        
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    heat1D()