import sys
sys.path.insert(0, "HW02")

import numpy as np
import matplotlib.pyplot as plt

from rk4_ndim import rk4_ndim
from rk4_modified import rk4_modified
import random
from matplotlib.gridspec import GridSpec

# ------
# Lorenz 
# ------

a = random.uniform(-10, 10)
b = random.uniform(-10, 10)
c = random.uniform(-10, 10)

y1_0 = np.array([a, b, c])    # random initial value 1
y2_0 = np.array([0, 0, 0])    # initial value 2
t0 = 0.0    # initial time
T = 30      # final time
dt = 0.01   # step size

beta = 8/3; sigma = 10; rho = 20
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

mu = 2
lorenz_modified = lambda t , x : np.array([
    sigma*(x[1] - x[0]) + mu*(lorenz(t, x)[1]-x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

X1 , X2, errors, t = rk4_modified(lorenz, lorenz_modified, y1_0, y2_0, t0, T, dt)
# X , t = rk4_ndim(lorenz, y0, t0, T, dt)

fig = plt.figure(figsize=(8, 16)) # 1x2
gs = GridSpec(1, 2, figure=fig) # actual layout

# first row = 3d plots
fig = plt.figure()
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.plot(X1[0,:], X1[1,:], X1[2,:])
ax1.plot(X2[0,:], X2[1,:], X2[2,:], color='green')
ax1.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

# second row = error
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, errors, color='red')
ax2.set_xlabel('t'); ax2.set_ylabel('error(t)')

plt.show()
