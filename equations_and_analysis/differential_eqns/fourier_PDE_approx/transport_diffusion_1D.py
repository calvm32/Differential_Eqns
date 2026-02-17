import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.rk4_solvers.rk4 import rk4

"""
Solve transport-diffusion equation u_t + c u_x = nu u_{xx} + f(x) using Fourier coefficients
    -> IC u(0,x)=u_0(x)
    -> periodic BCs
"""

nu, c, t0, T = 0.01, 1, 0.0, 5.0 # diffusion coeff, forcing constant, init time, final time
L, N = 2*np.pi, 2**8 # Domain length, number of points
CFL = (2/np.pi)/4 # CFL condition

dx = L/N
x = np.arange(N)*dx

def rhs(u_hat,ksq,nu):
    # Fourier-space RHS: u_hat_t = nu*ksq*u_hat
    return nu*ksq*u_hat

# Fourier wavenumbers for second derivative: u_xx_hat = -(k^2) u_hat
freq = np.fft.fftfreq(N,d=dx)   # cycles per unit length (FFT-ordered)
k = 2*np.pi*freq                # Fourier wavenumbers
ksq = -(k**2)

# Time-step CFL-restriction
dt = 0.2*dx*dx/nu               
t = np.arange(t0,T + 1e-12,dt)  # include T (up to roundoff)

# Compute the Fourier coefficients of u
u = np.sin(x) + 0.5*np.cos(7*x) + 0.25*np.cos(33*x)
u_hat = np.fft.fft(u) 

plt.ion()
fig, ax = plt.subplots()

# main time loop
for n in range(len(t)-1):  
    ax.clear()
    ax.plot(x,np.fft.ifft(u_hat).real,linewidth=2) # inverse Fourier
    ax.set_xlim(0,L); ax.set_ylim(-2,2) # Fix axes
    ax.set_title(f't = {t[n+1]:1.3f}')
    plt.pause(0.001) # Slow down animation

rk4(f, y0, t0, T, dt)
    
plt.ioff()
plt.show()

