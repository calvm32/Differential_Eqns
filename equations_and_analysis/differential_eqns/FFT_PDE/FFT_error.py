import numpy as np
import matplotlib.pyplot as plt

"""
Differentiate function u using Fourier coefficients on [-pi, pi]
"""

# ----------
# Grid setup
# ----------

a, b = -np.pi, np.pi  # domain endpoints
L = b - a             # domain length

ux_errors = []
uxx_errors = []
N_list = []

for i in range(2,7):

    N = 2**i              # number of grid points
    N_list.append(N)
    dx = L/N              # grid spacing

    # We use [a,b) so the first and last point are not duplicates.
    x = a + dx*np.arange(N)

    # u = np.sin((2*np.pi/L)*x)
    # ux_exact = (2*np.pi/L)*np.cos((2*np.pi/L)*x)
    # uxx_exact = -(2*np.pi/L)**2*np.sin((2*np.pi/L)*x)

    # u = np.exp(np.sin((2*np.pi/L)*x))
    # ux_exact = np.cos(x)*np.exp(np.sin((2*np.pi/L)*x))
    # uxx_exact = (np.cos(x)**2 - np.sin(x))*np.exp(np.sin((2*np.pi/L)*x))

    u = abs(np.sin(x))
    ux_exact = np.cos(x)*np.sign(x)
    uxx_exact = -1*abs(np.sin(x))

    # ------------------
    # Fourier wave modes
    # ------------------

    # np.fft.fftfreq returns frequencies in cycles per unit length,
    # in the standard FFT ordering: 
    # [0,1,2,...,N/2,-(N/2-1),...,-1]/L
    k = 2*np.pi*np.fft.fftfreq(N,d=dx)

    # Derivative multipliers in Fourier space:
    ik = 1j*k # In python, 1j means sqrt(-1)
    minus_k2 = -k**2

    # Nyquist mode handling (only matters for even N):
    # For the first derivative, the Nyquist mode is unpaired.
    # Setting it to zero avoids a small annoying imaginary artifact 
    # when we transform back.
    ik[N//2] = 0.0 # N//2 means integer division (rounds down)

    u_hat = np.fft.fft(u) # Compute the complex Fourier coefficients

    # First derivative
    ux_hat = ik*u_hat # Multiply in Fourier space
    ux = np.fft.ifft(ux_hat).real # Return to physical space

    # Second derivative
    uxx_hat = minus_k2*u_hat
    uxx = np.fft.ifft(uxx_hat).real

    # ----------------
    # calculate errors
    # ----------------

    ux_errors.append(np.linalg.norm(ux - ux_exact))
    uxx_errors.append(np.linalg.norm(uxx - uxx_exact))

    # ux_errors.append(np.mean((ux - ux_exact)**2))
    # uxx_errors.append(np.mean((uxx - uxx_exact)**2))

# -----------
# plot errors
# -----------

plt.loglog(N_list,ux_errors, color="tab:orange", label=r"$u_{x}$ error")
plt.loglog(N_list,uxx_errors, color="tab:blue", label=r"$u_{xx}$ error")
plt.xlabel("N size")
plt.ylabel("Errors")

plt.legend()
plt.show()

if __name__=="__main__":
    main()