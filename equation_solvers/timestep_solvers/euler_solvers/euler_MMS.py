import numpy as np
import matplotlib.pyplot as plt
from euler_forward import euler_forward
from euler_forward_error import euler_forward_error

"""
This code tests Euler forward to make sure it converges at
a rate which approximately matches the theoretical theoretical rate
"""

f = lambda t , y : -t * y **2
y_exact = lambda t : 2/(2 + t**2) 

y0 = 1.0    # initial value
t0 = 0.0    # initial time
T = 5.0     # final time
dt = 0.05   # step size

y_approx , t = euler_forward(f, y0, t0, T, dt)
errors, resolutions = euler_forward_error(y_exact, f, y0, t0, T)

# ----
# plot
# ----

# plot the approximate solution
plt.subplot(1,2,1)
plt.plot(t , y_approx , "-o" , label = "Euler approximation")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()

# plot the errors in a log-log plot
plt.subplot(1,2,2)
plt.plot(np.log(resolutions) , np.log(errors), "-o" , label = "Euler approximation error")
plt.xlabel("log(t)")
plt.ylabel("log(error)")
plt.legend()
plt.show()