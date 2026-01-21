from timestep_solvers.rk4_solvers.rk4_ndim import rk4_ndim
import numpy as np
import matplotlib.pyplot as plt
import random

"""
This program finds the Lyapunov exponents of the Lorenz system
"""

# -----
# setup
# -----

# random floats
a = random.uniform(-10, 10)
b = random.uniform(-10, 10)
c = random.uniform(-10, 10)

# random first initial value
y1_0 = np.array([a, b, c])

# small initial perturbation
eps = 1e-9
y2_0 = y1_0 + np.array([0, 0, eps])

# constants
t0 = 0.0                            # initial time
T = 30                              # final time
dt = 0.01                           # step size
beta = 8/3; sigma = 10; rho = 28    # parameters

# lorenz system
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

# ---------------------------
# solve for both trajectories
# ---------------------------

# trajectory 1
X1, t = rk4_ndim(lorenz, y1_0, t0, T, dt, 3)

# trajectory 2 (slightly perturbed)
X2, _ = rk4_ndim(lorenz, y2_0, t0, T, dt, 3)

# -------------
# plot distance
# -------------

# Distance
d = np.linalg.norm(X1.T-X2.T, axis=1) # take transpose so it works

"""
plt.figure(figsize=(8,5))
plt.semilogy(t, d)
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Separation of nearby Lorenz trajectories")
plt.grid(True)
plt.show()
"""

# ----------------------
# solve for and plot LyE
# ----------------------

mask = t <= 25
logd = np.log(d[mask])

# linear fit given by log(distance) = slope * t + intercept
slope, intercept = np.polyfit(t[mask], logd, 1)
print(f"Estimated maximal Lyapunov exponent: {slope:.4f}")
print(f"Random Initial Condition = ({a:.2f}, {b:.2f}, {c:.2f})")

plt.figure(figsize=(8,5))
plt.semilogy(t, d, label="dist(traj1, traj2)")
plt.semilogy(t, np.exp(intercept + slope*t), 'k--', label=f'exp({slope:.2f} t)')
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Lyapunov exponent estimation")
plt.legend()
plt.grid(True)
plt.show()

# ----------
# references
# ----------

# this was converted from the MATLAB code found in
# https://github.com/chebfun/examples/blob/master/ode-nonlin/LyapunovExponents.m