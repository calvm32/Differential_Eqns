import cmath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

""" 
This code graphs a phase space given the solution's Jacobian
"""

P = np.array([[1, 1j, -1j],
              [-1, 1j, -1j],
              [1, 1, 1]])

exp_Dt = lambda t : np.array([[cmath.exp(-2*t), 0, 0],
                              [0, cmath.exp(2*(-1-5j)*t), 0],
                              [0, 0, cmath.exp(2*(-1+5j)*t)]])

exp_Nt = lambda t : np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

x0 = np.array([[0], [-1], [1]])

exp_At = lambda t : (P @ exp_Dt(t) @ exp_Nt(t) @ np.linalg.inv(P) @ x0).real

# ----
# plot
# ----

t_vals = np.linspace(0, 30, 300)
trajectory = np.hstack([exp_At(t) for t in t_vals]) # horizontal vector
x, y, z = trajectory

fig = plt.figure(figsize=(12, 8)) # 3x2, first row = twice height of bottom
gs = GridSpec(2, 3, height_ratios=[2, 1], figure=fig) # actual layout

# layout, make first one big
ax1 = fig.add_subplot(gs[0, :], projection='3d')
ax1.plot(x, y, z, color='blue')
ax1.scatter(x[0], y[0], z[0], color='green', label='IC')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.legend()

# x(t) vs t
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_vals, x, color='red')
ax2.set_xlabel('t')
ax2.set_ylabel('x(t)')
ax2.set_title('x(t) vs t')

# y(t) vs t
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_vals, y, color='green')
ax3.set_xlabel('t')
ax3.set_ylabel('y(t)')
ax3.set_title('y(t) vs t')

# z(t) vs t
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(t_vals, z, color='purple')
ax4.set_xlabel('t')
ax4.set_ylabel('z(t)')
ax4.set_title('z(t) vs t')

plt.tight_layout()
plt.show()