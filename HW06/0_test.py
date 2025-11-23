import sys
sys.path.insert(0, "HW02")

import numpy as np
import matplotlib.pyplot as plt

from HW02.rk4_ndim import rk4_ndim
from rk4_modified import rk4_modified

# ------
# Lorenz 
# ------

beta = 8/3; sigma = 10; rho = 1000
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

mu = 10
lorenz_modified = lambda t , x : np.array([
    sigma*(x[1] - x[0]) + mu*(lorenz[1]-x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

y0 = np.array([0 , 1 , 1])    # initial value
t0 = 0.0    # initial time
T = 30      # final time
dt = 0.0001   # step size

X1 , X2, t = rk4_modified(lorenz, y0, t0, T, dt)
# X , t = rk4_ndim(lorenz, y0, t0, T, dt)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot(X1[0,:], X1[1,:], X1[2,:])
ax.plot(X2[0,:], X2[1,:], X2[2,:])
#plt.title("rho = " + str(rho))
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()