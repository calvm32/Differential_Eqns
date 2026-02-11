import numpy as np
import matplotlib.pyplot as plt
import math

from rk4_solvers.rk4 import rk4
from rk4_solvers.rk4_error import rk4_error

"""
This code tests Euler forward to make sure it converges at
a rate which approximately matches the theoretical theoretical rate
"""

f = lambda t , y : 0*t + y**2
y_exact = lambda t : (-1)/(t -1)

y0 = 1.0    # initial value
t0 = 0.0    # initial time
T = 5.0     # final time
dt = 0.005   # step size
M = 500    # threshold

y_approx , t = rk4(f, y0, t0, T, dt, M)
errors, resolutions = rk4_error(y_exact, f, y0, t, M)

# evaluate exact soln on interval t
y_exact_eval = np.array([],dtype = float)
for i in range(len(t)):
    y_exact_eval = np.append(y_exact_eval,y_exact(t[i]))

# ----
# plot
# ----

# plot the approximate solution
plt.subplot(1,3,1)
plt.plot(t , y_approx , "-o" , label = "RK4 approximation")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()

# plot the exact solution
plt.subplot(1,3,2)
plt.plot(t , y_exact_eval , "-o" , label = "Exact solution")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()

# plot the errors in a log-log plot
plt.subplot(1,3,3)
plt.plot(np.log(resolutions) , np.log(errors), "-o" , label = "RK4 approximation error")
plt.xlabel("log(t)")
plt.ylabel("log(error)")
plt.legend()
plt.show()