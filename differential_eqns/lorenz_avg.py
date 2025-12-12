import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../timestep_solvers')) # from google ai search result
sys.path.append(parent_dir)

from rk4_solvers import rk4_ndim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

"""
This program plots a running average of solutions to the Lorenz system 
with nearby initial conditions
"""

# -----
# setup
# -----

# change for plotting
N = 50                  # length of running avg
IC_area_size = 1^-8        # area to take IC, i.e. (1,1) or (10,10)
size = 0.1              # size of lines plotted
opacity = 0.5           # opacity of lines plotted

# constants
t0 = 0.0                            # initial time
dt = .01                            # step size
num_steps = 20000                   # number of steps
T = dt*num_steps                    # final time
beta = 8/3; sigma = 10; rho = 28    # parameters

# lorenz system
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

# -----------
# setup plots
# -----------

fig = plt.figure(figsize=(12, 8)) 
gs = GridSpec(2, 3, height_ratios=[2, 1], figure=fig) # actual layout

# layout, make first one big
ax1 = fig.add_subplot(gs[0, :], projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.legend()

# x(t) vs t
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('x(t)')

# y(t) vs t
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('y(t)')

# z(t) vs t
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title('z(t)')

# --------------------
# solve then take avgs
# --------------------

for n in range(10):
    # given initial condition
    y0 = [1, 1.27, 14.132] 

    # random floats
    a = IC_area_size*random.uniform(-1, 1)
    b = IC_area_size*random.uniform(-1, 1)
    c = IC_area_size*random.uniform(-1, 1)

    # random, small perturbation
    y0 += np.array([a, b, c])

    # solve lorenz given conditions
    X , t = rk4_ndim(lorenz, y0, t0, T, dt, 3)

    # Convert solution X w shape (3,N) to shape (N, 3)
    X = np.asarray(X)
    X = X.T  

    # take avgs
    sum = np.cumsum(X, axis=0)
    avg = np.zeros_like(X)

    for i in range(len(X)):
        if i < N:
            avg[i] = sum[i] / (i+1)
        else:
            avg[i] = (sum[i] - sum[i-N]) / N

    # --------------------
    # plot 3D + histograms
    # --------------------

    # 3D trajectory
    ax1.plot(avg[:,0], avg[:,1], avg[:,2], color='#1C7C9C', alpha=opacity, linewidth=size)

    # for plotting, disclude first and last pts without a full avg
    t_avg = t[N-1:]
    avg_valid = avg[N-1:]

    # histogram w only those points
    ax2.hist(avg_valid[:,0], bins=100, range=(avg_valid[:,0].min(), avg_valid[:,0].max()), 
            color='#4F783F', alpha=opacity)

    ax3.hist(avg_valid[:,1], bins=100, range=(avg_valid[:,1].min(), avg_valid[:,1].max()), 
            color='#A774AA', alpha=opacity)

    ax4.hist(avg_valid[:,2], bins=100, range=(avg_valid[:,2].min(), avg_valid[:,2].max()), 
         color='#C9582C', alpha=opacity)

plt.show()