import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.rk4_solvers.rk4_fourier import *
from mpl_toolkits.mplot3d import Axes3D

"""
Solve transport-diffusion equation u_t + c grad(u) = nu lap(u) + f(x) using Fourier coefficients
    -> IC u(0,x)=u_0(x)
    -> periodic BCs
"""

# ------------
# helper funcs
# ------------

def make_grid(N):
    L=2.0*np.pi
    dx=L/N
    x=np.linspace(-np.pi,-np.pi+L,N,endpoint=False)
    y=np.linspace(-np.pi,-np.pi+L,N,endpoint=False)
    X,Y=np.meshgrid(x,y,indexing="ij")
    return x,y,X,Y,dx

def main():

    # --------------------------
    # setup constants, functions
    # --------------------------

    nu = 0.01               # diffusion coeff
    c = 2                   # forcing constant
    t0 = 0.0                # init time
    T = 5.0                 # final time

    L, N = 2*np.pi, 2**8    # Domain length, number of points
    CFL = (2/np.pi)/4       # CFL condition

    # get linear space
    x,y,X,Y,dx = make_grid(N)

    # Time-step CFL-restriction
    dt = 0.2*dx*dx/nu               
    t = np.arange(t0, T + 1e-12, dt)  # include T (up to roundoff)

    # Initial condition
    u0 = np.exp(np.cos(X) + np.sin(3*Y))

    # Forcing
    f = np.sin(X)*np.cos(Y)

    # ---------------
    # solve and graph
    # ---------------

    def rhs(u_hat,f_hat,ksq,nu):
        # Fourier-space RHS: u_hat_t = nu*ksq*u_hat
        return (-1j*c*(KX+KY) + nu*ksq)*u_hat + f_hat

    # Fourier wavenumbers for second derivative: u_xx_hat = -(k^2) u_hat
    kx = 2.0*np.pi*np.fft.fftfreq(N,d=dx) # Fourier wavenumbers
    ky = 2.0*np.pi*np.fft.fftfreq(N,d=dx) # Fourier wavenumbers
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    ksq = -(KX**2 + KY**2)

    # initialize the Fourier coefficients of u
    u_hat_0 = np.fft.fft2(u0)

    # initialize the Fourier coefficients of f
    f_hat = np.fft.fft2(f)

    # Compute the Fourier coefficients of u
    u_hat, times = rk4_fourier(rhs, u_hat_0, f_hat, t0, T, dt, ksq, nu, t)

    # Plotting
    plt.ion()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y, indexing='ij')

    # main time loop
    for n in range(len(t)-1):  
        ax.clear()
        ax.plot_surface(X,Y,np.fft.ifft2(u_hat[...,n]).real, cmap='viridis',linewidth=2) # inverse Fourier
        ax.set_xlim(0,L); ax.set_ylim(-2,2) # Fix axes
        ax.set_title(f't = {t[n+1]:1.3f}')
        plt.pause(0.001) # Slow down animation
        
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()