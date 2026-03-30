#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

"""
solve 2D heat equation on a periodic square, Fourier in space, RK4 in time.

PDE:
    u_t + c1*u_x + c2*u_y = nu*(u_xx + u_yy)

Fourier form:
    d/dt u_hat c1*i*kx*u_hat + c2*i*ky*u_hat = nu*(-(kx^2 + ky^2))*u_hat
"""

def rk4_step(u_hat,t,dt,rhs):
    """
    One classical RK4 step for u' = rhs(u,t).
    Works for scalars, vectors, or arrays (NumPy broadcasting).
    """
    k1 = rhs(t         ,u_hat)
    k2 = rhs(t + 0.5*dt,u_hat + 0.5*dt*k1)
    k3 = rhs(t + 0.5*dt,u_hat + 0.5*dt*k2)
    k4 = rhs(t + dt    ,u_hat +     dt*k3)
    return u_hat + (dt/6.0)*(k1 + 2.0*(k2 + k3) + k4)


def l2_norm_periodic_from_uhat(u_hat,Lx,Ly):
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
    nu = 0.01
    c1, c2 = 1, -2
    t0 = 0.0
    tf = 5.0

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

    # Heat equation RHS in Fourier space: u_hat' = -nu*(kx^2 + ky^2)*u_hat
    nuLap = nu*Laplacian_k

    def rhs(t,u_hat):
        # Compute derivatives in Fourier space
        u_x_hat=1j*kx[:,None]*u_hat # multiply along columns
        u_y_hat=1j*ky[None,:]*u_hat # multiply along rows
        
        # t is unused here, but kept for compatibility with general PDEs.
        return -c1*u_x_hat -c2*u_y_hat + nuLap*u_hat

    # Time step (simple diffusion stability-ish choice)
    dt_visc = 0.1*min(dx,dy)**2/nu
    dt_adv  = 0.1*min(dx,dy)/max(abs(c1),abs(c2))
    dt      =     min(dt_visc,dt_adv)
    t = np.arange(t0,tf + 0.5*dt,dt)

    # Initial condition
    rng = np.random.default_rng()  # optionally seed: np.random.default_rng(0)
    u0 = np.sin(X)*np.cos(Y) + np.cos(17.0*X)*np.cos(13.0*Y) + 0.2*rng.standard_normal((Nx,Ny))

    u_hat = np.fft.fftn(u0)

    # Plot setup
    fig,(ax1,ax2) = plt.subplots(1,2,figsize = (14,6))

    u_phys = np.real(np.fft.ifftn(u_hat))
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
    l2_hist = [l2_norm_periodic_from_uhat(u_hat,Lx,Ly)]
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
    
    
    for n in range(len(t)-1):
        u_hat = rk4_step(u_hat,t[n],dt,rhs) # update with rk4

        t_hist.append(t[n + 1])
        l2_hist.append(l2_norm_periodic_from_uhat(u_hat,Lx,Ly))

        if t[n+1]>=next_movie_time or (n+1)==len(t)-1:
            next_movie_time+=movie_dt # update movie counter
            
            u_phys = np.real(np.fft.ifftn(u_hat))
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