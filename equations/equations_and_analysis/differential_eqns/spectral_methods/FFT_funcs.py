import numpy as np

def make_grid(N):
    L=2.0*np.pi
    dx=L/N
    x=np.linspace(-np.pi,-np.pi+L,N,endpoint=False)
    y=np.linspace(-np.pi,-np.pi+L,N,endpoint=False)
    X,Y=np.meshgrid(x,y,indexing="ij")
    return x,y,X,Y,dx

def make_wavenumbers(N,dx):
    # NumPy convention: fftfreq returns cycles per unit length
    # Multiply by 2*pi to get angular wavenumbers.
    k=2.0*np.pi*np.fft.fftfreq(N,d=dx)
    return k

def l2_norm(u,dx):
    # u is an (N,N) array of real values on the grid.
    # Implement a midpoint/trapezoid rule on the periodic grid.
    # TODO: return sqrt( \sum |u|^2 * dx^2 )
    return np.sqrt(np.sum(np.abs(u)**2)*dx*dx)

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

def spectral_derivatives(u,dx):
    # Input: u in physical space, shape (N,N)
    # Output: u_x, u_y, lap_u in physical space, each shape (N,N)

    N=u.shape[0]
    k=make_wavenumbers(N,dx)    # shape (N,)
    kx=k[:,None]                # shape (N,1)
    ky=k[None,:]                # shape (1,N)

    u_hat=np.fft.fftn(u)

    # TODO: fill in the Fourier-space multipliers
    u_x_hat = (1j*kx)*u_hat
    u_y_hat = (1j*ky)*u_hat
    lap_hat = (-(kx**2+ky**2))*u_hat

    u_x=np.real(np.fft.ifftn(u_x_hat))
    u_y=np.real(np.fft.ifftn(u_y_hat))
    lap_u=np.real(np.fft.ifftn(lap_hat))

    return u_x,u_y,lap_u

def run_test(u_fun,ux_fun,uy_fun,lap_fun,N):
    x,y,X,Y,dx=make_grid(N)

    u=u_fun(X,Y)
    ux_exact=ux_fun(X,Y)
    uy_exact=uy_fun(X,Y)
    lap_exact=lap_fun(X,Y)

    ux_num,uy_num,lap_num=spectral_derivatives(u,dx)

    err_ux=ux_num-ux_exact
    err_uy=uy_num-uy_exact
    err_lap=lap_num-lap_exact

    nrm_u=l2_norm(u,dx)
    nrm_err_ux=l2_norm(err_ux,dx)
    nrm_err_uy=l2_norm(err_uy,dx)
    nrm_err_lap=l2_norm(err_lap,dx)

    return nrm_u,nrm_err_ux,nrm_err_uy,nrm_err_lap

def main():
    N=128

    # Test 1: u=sin(x)cos(y)
    u1=lambda X,Y: np.sin(X)*np.cos(Y)
    ux1=lambda X,Y: np.cos(X)*np.cos(Y)
    uy1=lambda X,Y: -np.sin(X)*np.sin(Y)
    lap1=lambda X,Y: -2.0*np.sin(X)*np.cos(Y)

    # Test 2: u=cos(17x)cos(13y)
    u2=lambda X,Y: np.cos(17.0*X)*np.cos(13.0*Y)
    ux2=lambda X,Y: -17.0*np.sin(17.0*X)*np.cos(13.0*Y)
    uy2=lambda X,Y: -13.0*np.cos(17.0*X)*np.sin(13.0*Y)
    lap2=lambda X,Y: -(17.0**2+13.0**2)*np.cos(17.0*X)*np.cos(13.0*Y)

    # Test 3: u=sin(3x-5y)
    u3=lambda X,Y: np.sin(3.0*X-5.0*Y)
    ux3=lambda X,Y: 3.0*np.cos(3.0*X-5.0*Y)
    uy3=lambda X,Y: -5.0*np.cos(3.0*X-5.0*Y)
    lap3=lambda X,Y: -(3.0**2+5.0**2)*np.sin(3.0*X-5.0*Y)

    tests=[
        ("sin(x)cos(y)",u1,ux1,uy1,lap1),
        ("cos(17x)cos(13y)",u2,ux2,uy2,lap2),
        ("sin(3x-5y)",u3,ux3,uy3,lap3),
    ]
    
    print(f"N={N}")
    for name,u,ux,uy,lap in tests:
        nrm_u,eux,euy,elap=run_test(u,ux,uy,lap,N)
        print("")
        print(name)
        print(f"  ||u||_L2              = {nrm_u:.16e}")
        print(f"  ||ux-ux_exact||_L2     = {eux:.16e}")
        print(f"  ||uy-uy_exact||_L2     = {euy:.16e}")
        print(f"  ||lap-lap_exact||_L2   = {elap:.16e}")

if __name__=="__main__":
    main()