from sklearn.datasets import make_spd_matrix
from equation_solvers.iterative_solvers.linear_solvers import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import random as rand
import sympy as sp

"""
Compare gradient descent error over a lot of initial conditions
    -> real system
"""

# ------------------
# different settings
# ------------------

SPD1 = make_spd_matrix(n_dim=3, random_state=1)

b1 = np.random.uniform(-50, 50, size=3)
X0 = np.random.uniform(-50, 50, size=3)

max_iters = 100

# ---------
# setup run
# ---------

A = SPD1
b = b1
X_exact = np.linalg.solve(A, b)

f = lambda x: 0.5 * x.T @ A @ x - b.T @ x
grad_f = lambda x: A @ x - b

dt = min(1, 1/max(np.linalg.eigvals(A)))

# -----------
# setup plots
# -----------

fig = plt.figure(figsize=(5,7))
gs = GridSpec(2, 1, figure=fig)

# 3D solution
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Error plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlabel('iteration')
ax3.set_ylabel('error')

for i in range(50):
    X0 = np.random.uniform(-50, 50, size=3)

    # solve lorenz given conditions
    X, t, errors = grad_descent(f, grad_f, X0, X_exact, dt, max_iters)

    # ------------
    # plot in time
    # ------------

    ax1.plot(X[0,:], X[1,:], X[2,:])
    errors = np.array([
        f(X[:, k]) - f(X_exact)
        for k in range(X.shape[1])
    ])

    k = np.arange(1, len(errors) + 1)
    ax3.loglog(k, errors, alpha=0.5)
    ax3.loglog(k, np.linalg.norm(X0 - X_exact)**2 / (2 * dt * k),linestyle="--", label=r"$\|x_0-x_*\|^2/(2tk)$")    

#ax3.legend()    
plt.tight_layout()
plt.show()

if __name__=="__main__":
    main()