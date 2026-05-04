#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from equation_solvers.timestep_solvers.misc.wave_fourier import *

"""
Solve 1D wave equation u_tt = c^2 u_{xx} using Fourier coefficients
    -> periodic BCs on [-L, L]

Fourier form: u_hat_tt = -c^2*ksq*u_hat
"""

def bump_function_2D(X, Y, x0, y0, epsilon):
    """
    Evaluate a normalized 2D bump function on a given grid.
    Inputs:
        X, Y            : meshgrid arrays (with indexing='ij')
        x0, y0          : center of the bump
        epsilon         : radius of support
    Output:
        u               : bump function evaluated on (X,Y)
    """
    normalize_const=0.4665123931783301 # integrate to 1
    # Radius shifted and squared
    radius_sq = (X - x0)**2 + (Y - y0)**2
    # Initialize
    u = np.zeros_like(X)
    # Disk mask
    disk = radius_sq < epsilon**2
    # Bump profile
    u[disk] = np.exp(1.0/((radius_sq[disk]/epsilon**2) - 1.0))
    # Apply normalization
    u[disk] /= (normalize_const*epsilon**2)
    return u

def l2_norm_periodic(u_hat,Lx,Ly):
    """
    Compute ||u||_{L^2([0,Lx]x[0,Ly])} from NumPy FFT coefficients.

    With NumPy conventions:
      u = ifftn(u_hat) includes 1/(Nx*Ny).
    Parseval implies:
      ||u||_L2 = sqrt(Lx*Ly)/(Nx*Ny)*sqrt(sum|u_hat|^2).
    """
    Nx,Ny = u_hat.shape
    return np.sqrt(Lx*Ly)*np.sqrt(np.sum(np.abs(u_hat)**2))/(Nx*Ny)


def main():
    # Parameters
    c = 1.0 
    nu = 0.01
    t0 = 0.0
    T = 2.0

    Lx = 2.0*np.pi
    Ly = 2.0*np.pi
    Nx = 2**8
    Ny = 2**8

    dx = Lx/Nx
    dy = Ly/Ny
    
    movie_dt = 0.05 # how often to update plot

    # Grid (periodic, endpoint excluded)
    x = np.linspace(-np.pi,-np.pi + Lx,Nx,endpoint = False)
    y = np.linspace(-np.pi,-np.pi + Ly,Ny,endpoint = False)
    X,Y = np.meshgrid(x,y,indexing = "ij")

    # Standard NumPy wavenumbers:
    # fftfreq gives cycles per unit length; multiply by 2*pi for angular wavenumbers.
    kx = 2.0*np.pi*np.fft.fftfreq(Nx,d = dx)
    ky = 2.0*np.pi*np.fft.fftfreq(Ny,d = dy)
    Laplacian_k = -kx[:,None]**2 - ky[None,:]**2
    ksq = -(kx**2 + ky**2)

    # Time step (simple diffusion stability-ish choice)
    dt_visc = 0.1*min(dx,dy)**2/nu
    dt_adv  = 0.1*min(dx,dy)
    dt      =     min(dt_visc,dt_adv)
    t = np.arange(t0,T + 0.5*dt,dt)

    # -----------------
    # Initial condition
    # -----------------
    
    # Add three bump functions at different locations
    x0, y0, epsilon = 0.5, -0.5, 0.1
    v0 = bump_function_2D(X,Y,x0,y0,epsilon)
    x0, y0, epsilon = 0.5, 0.5, 0.2
    v0 = v0 + bump_function_2D(X,Y,x0,y0,epsilon)
    x0, y0, epsilon = -0.5, 0.5, 0.3
    v0 = v0 + bump_function_2D(X,Y,x0,y0,epsilon)

    u0 = np.zeros((Nx, Ny), dtype=complex)
    u_hat_0 = np.fft.fft(u0)

    # get the next time step to start our timestep solver
    u_hat_1 = u_hat_0 + dt*np.fft.fft(v0) + c**2*ksq*dt*u_hat_0

    u_hat = np.zeros((Nx, Ny, len(t)), dtype=complex)
    u_hat[:,:, 0] = u_hat_0
    u_hat[:,:, 1] = u_hat_1

    # Plot setup
    fig,(ax1,ax2) = plt.subplots(1,2,figsize = (14,6))

    u_phys = np.real(np.fft.ifftn(u_hat_0))
    im = ax1.imshow(
        u_phys.T,
        origin = "lower",
        extent = (x[0],x[0] + Lx,y[0],y[0] + Ly),
        aspect = "equal",
        cmap="seismic",#Try also: RdBu_r, coolwarm, seismic, viridis
        interpolation="nearest",
        vmin=-1.2,
        vmax=1.2
    )
    fig.colorbar(im,ax = ax1)
    ax1.set_title(f"t = {t[0]:.3f}")
    ax1.set_xticks([-np.pi,-np.pi/2,0.0,np.pi/2,np.pi])
    ax1.set_xticklabels([r"$-\pi$",r"$-\pi/2$","0",r"$\pi/2$",r"$\pi$"])
    ax1.set_yticks([-np.pi,-np.pi/2,0.0,np.pi/2,np.pi])
    ax1.set_yticklabels([r"$-\pi$",r"$-\pi/2$","0",r"$\pi/2$",r"$\pi$"])

    # Running L2 plot: only computed values (no NaNs)
    t_hist = [t[0]]
    l2_hist = [l2_norm_periodic(u_hat_1,Lx,Ly)]
    (line,) = ax2.plot(t_hist,l2_hist,linewidth = 2)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$||u(t)||_{L^2}$")
    ax2.grid(True,alpha = 0.3)

    plt.tight_layout()
    plt.pause(0.001)

    next_movie_time = t0
    save_gif=True
    gif_filename="heat2D.gif"
    fps=20

    if save_gif:
        writer=PillowWriter(fps=fps)
        writer.setup(fig,gif_filename,dpi=100)
    
    
    for n in range(len(t)-2):

        # Compute the Fourier coefficients of u
        u_hat[:,:, n+ 1] = 2*u_hat[:,:, n] - u_hat[:,:, n- 1] + (c**2)*ksq*(dt**2)*u_hat[:,:, n]

        t_hist.append(t[n + 1])
        l2_hist.append(l2_norm_periodic(u_hat[:,:,n+1],Lx,Ly))

        if t[n+1]>=next_movie_time or (n+1)==len(t)-1:
            next_movie_time+=movie_dt # update movie counter
            
            u_phys = np.real(np.fft.ifftn(u_hat[:, :, n+1]))
            im.set_data(u_phys.T)
            ax1.set_title(f"t = {t[n + 1]:.3f}")

            line.set_data(t_hist,l2_hist)
            ax2.relim()
            ax2.autoscale_view()
            
            if save_gif:
                writer.grab_frame() # Add new image to gif file

            plt.pause(0.001)
            
            next_movie_time+=movie_dt # update movie counter

    if save_gif:
        writer.finish() # Close gif file
    plt.show()

if __name__ == "__main__":
    main()