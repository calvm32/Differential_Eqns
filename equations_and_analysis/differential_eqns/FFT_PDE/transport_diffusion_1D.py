import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.rk4_solvers.rk4_fourier import *

"""
Solve transport-diffusion equation u_t + c u_x = nu u_{xx} + f(x) using Fourier coefficients
    -> IC u(0,x)=u_0(x)
    -> periodic BCs
"""

# --------------------------
# setup constants, functions
# --------------------------

nu, c, t0, T = 0.01, 1, 0.0, 5.0 # diffusion coeff, forcing constant, init time, final time
L, N = 2*np.pi, 2**8 # Domain length, number of points
CFL = (2/np.pi)/4 # CFL condition

# get linear space
dx = L/N
x = np.arange(N)*dx

# Time-step CFL-restriction
dt = 0.2*dx*dx/nu               
t = np.arange(t0,T + 1e-12,dt)  # include T (up to roundoff)

# IC
u0 = np.exp(np.cos(x) + np.sin(33*x))

# forcing
f = np.sin(x)

# ---------------
# solve and graph
# ---------------

def rhs(u_hat,f_hat,ksq,nu):
    # Fourier-space RHS: u_hat_t = nu*ksq*u_hat
    return (-1j*c*k + nu*(-k**2))*u_hat + f_hat


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
