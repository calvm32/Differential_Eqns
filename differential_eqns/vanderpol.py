from timestep_solvers.rk4_solvers.rk4_ndim import rk4_ndim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

"""
This program plots solutions to the Van der Pol equation 
y'' + (y^2-1)y' + y = 0
"""

# -----
# setup
# -----

# convert y'' + (y^2- 1)y' + y = 0 into a 2d system
# set u = y, and v = y'
f = lambda t, y: np.array([y[1], -(y[0]**2 - 1)*y[1] - y[0]])

# initial value
y0 = [1, 0]

# constants
t0 = 0.0        # initial time
T = 100.0       # final time
dt = 0.05       # step size

# solve van der pol given initial conditions
y_approx , t = rk4_ndim(f, y0, t0, T, dt, 2)

# ----
# plot
# ----

# plot the approximate solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t, y_approx[0, :], y_approx[1, :])
ax.set_title(f"IC = ({y0[0]:.2f}, {y0[1]:.2f})")
ax.legend()

plt.show()