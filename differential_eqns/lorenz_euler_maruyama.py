from timestep_solvers.euler_solvers.euler_maruyama import euler_maruyama
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random

"""
This program plots random solutions to the Lorenz system using an Euler-Maruyama solver
"""

# -----
# setup
# -----

# random floats
a = random.uniform(-1, 1)
b = random.uniform(-1, 1)
c = random.uniform(-1, 1)

# random initial value
X0 = np.array([a, b, c])

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
lorenz_a = lambda t , x : np.array([
    sigma*(x[1] - x[0]),
    x[0]*(rho - x[2]) - x[1],
    x[0]*x[1] - beta*x[2]
])

# later random noise
lorenz_b = lambda t , x : np.array([1,1,1])

# solve lorenz given conditions
X, t = euler_maruyama(lorenz_a, lorenz_b, X0, t0, T, dt, 3)

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

plt.show()