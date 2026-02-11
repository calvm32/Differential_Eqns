import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # from google ai search result
sys.path.append(parent_dir)

from rk4_solvers import rk4_error
from rk4_solvers import rk4_ndim

import numpy as np
import matplotlib.pyplot as plt
from euler_forward_error import euler_forward_error

"""
This program finds the convergence rate of RK4 vs. Euler solving methods
"""

# -----
# setup
# -----

f = lambda t , y : -t * y **2
y_exact = lambda t : 2/(2 + t**2) 

y0 = 1.0    # initial value
t0 = 0.0    # initial time
T = 5.0     # final time
dt = 0.05   # step size

ef_errors, ef_resolutions = euler_forward_error(y_exact, f, y0, t0, T)
rk4_errors, rk4_resolutions = rk4_error(y_exact, f, y0, t0, T)

# ----
# plot
# ----

# plot the errors in a log-log plot
plt.plot(np.log(ef_resolutions) , np.log(ef_errors), "-o" , label = "Euler approximation error")
plt.plot(np.log(rk4_resolutions) , np.log(rk4_errors), "-o" , label = "RK4 approximation error")
plt.xlabel("log(resolutions)")
plt.ylabel("log(error)")
plt.legend()

plt.grid()
plt.gca().set_aspect("equal") # to judge slope
plt.show()