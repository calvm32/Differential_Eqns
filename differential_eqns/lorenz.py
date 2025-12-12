import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../timestep_solvers')) # from google ai search result
sys.path.append(parent_dir)

from rk4_solvers import rk4_ndim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random

"""
This program plots random solutions to the Lorenz system
"""

# -----
# setup
# -----

# random floats
a = random.uniform(-1, 1)
b = random.uniform(-1, 1)
c = random.uniform(-1, 1)

# random initial value
y0 = np.array([2, 2, 2])

# constants
t0 = 0.0                            # initial time
T = 100                              # final time
dt = 0.01                           # step size
beta = 8/3; sigma = 10; rho = 28    # parameters
N = int((T - t0) / dt) + 1          # num steps

# x,y for color mapping comparison
x_list = np.linspace(0, 1, N)
y_list = np.arange(N) 

# lorenz system
lorenz = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

# solve lorenz given conditions
X , t = rk4_ndim(lorenz, y0, t0, T, dt, 3)

# ------------
# Plot in time
# ------------

# Convert solution X w shape (3,N) to shape (N, 1, 3)
pts = X.T.reshape(-1, 1, 3)

# Each segment = pair of consecutive points w shape (N-1, 2, 3)
segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(3, 1, figure=fig, height_ratios=[0.0001, 0.4, 5], hspace=0.5)

# Time map
ax1 = fig.add_subplot(gs[1, 0])
ax1.scatter(x_list, y_list, c=x_list, cmap='viridis', s=10)
ax1.scatter(x_list[0], y_list[0], c='red', s=40, label='initial point')
ax1.set_xlabel("Corresponding color")
ax1.set_ylabel("Time steps")
ax1.grid(True, alpha=0.3)

# Lorenz system
ax2 = fig.add_subplot(gs[2, 0], projection='3d')

# add colors
lc = Line3DCollection(segments, cmap='viridis')
lc.set_array(x_list[:-1])  # Must be length N-1 for segments
ax2.add_collection(lc)

# Enforce plot limits
ax2.set_xlim(X[0].min(), X[0].max())
ax2.set_ylim(X[1].min(), X[1].max())
ax2.set_zlim(X[2].min(), X[2].max())
ax2.scatter(X[0,0], X[1,0], X[2,0], c='red', s=40)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# super title
fig.suptitle(
    f"Random Initial Condition = ({a:.2f}, {b:.2f}, {c:.2f})",
    fontsize=16, fontweight='bold'
)

# ------------------------
# plot in each x,y,z plane
# ------------------------

fig = plt.figure(figsize=(12, 8)) # 3x2, first row = twice height of bottom
gs = GridSpec(2, 3, height_ratios=[2, 1], figure=fig) # actual layout

# layout, make first one big
ax1 = fig.add_subplot(gs[0, :], projection='3d')
ax1.plot(X[0], X[1], X[2], color='blue')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.legend()

# x(t) vs t
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, X[0], color='#4F783F')
ax2.set_xlabel('t')
ax2.set_ylabel('x(t)')
ax2.set_title('x(t) vs t')

# y(t) vs t
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, X[1], color='#A774AA')
ax3.set_xlabel('t')
ax3.set_ylabel('y(t)')
ax3.set_title('y(t) vs t')

# z(t) vs t
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(t, X[2], color='#C9582C')
ax4.set_xlabel('t')
ax4.set_ylabel('z(t)')
ax4.set_title('z(t) vs t')

plt.show()