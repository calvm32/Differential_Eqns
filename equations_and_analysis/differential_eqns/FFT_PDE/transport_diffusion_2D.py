import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.rk4_solvers.rk4_fourier import *
from .FFT_approx import *

"""
Solve transport-diffusion equation u_t + c grad(u) = nu lap(u) + f(x) using Fourier coefficients
    -> IC u(0,x)=u_0(x)
    -> periodic BCs
"""

# --------------------------
# setup constants, functions
# --------------------------

nu = 0.01               # diffusion coeff
c_vec[2,4]              # forcing constant
t0 = 0.0                # init time
T = 5.0                 # final time

L, N = 2*np.pi, 2**8    # Domain length, number of points
CFL = (2/np.pi)/4       # CFL condition

# get linear space
x,y,X,Y,dx = make_grid(N)

# Time-step CFL-restriction
dt = 0.2*dx*dx/nu               
t = np.arange(t0, T + 1e-12, dt)  # include T (up to roundoff)

# IC
u0 = np.exp(np.cos(x) + np.sin(33*y))

# forcing
f = np.sin(x)*np.cos(y)

# ---------------
# solve and graph
# ---------------

def rhs(u_hat,f_hat,ksq,nu):
    # Fourier-space RHS: u_hat_t = nu*ksq*u_hat
    return (-1j*c*k + nu*(-k**2))*u_hat + f_hat

# Fourier wavenumbers for second derivative: u_xx_hat = -(k^2) u_hat
k = make_wavenumbers(N, dx) # Fourier wavenumbers
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
