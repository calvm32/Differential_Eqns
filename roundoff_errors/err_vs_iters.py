from timestep_solvers.misc_solvers import *
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random as rand

"""
Program to compare 2 different kinds of errors vs. number of iterations,
comparing solutions only a small perturbation away
"""

N_iter = 100
opacity = 0.2
epsilon = 1e-1

f = lambda x: (cos(x) - x)
f_prime = lambda x: -sin(x) - 1

# initial guesses
a0 = 0.2
b0 = 1

x00 = 5

# machine epsilon
eps = np.finfo(float).eps

# -----------
# Setup plots
# -----------

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12, 5),
    constrained_layout=True,
)

ax1.set_xlabel("iterations"); ax1.set_ylabel("error")
ax1.set_title("Bisection Method Error Comparison")
ax1.grid(True, which="major", ls="--", alpha=0.5)

ax2.set_xlabel("iterations"); ax2.set_ylabel("error")
ax2.set_title("Newton's Method Error Comparison")
ax2.grid(True, which="major", ls="--", alpha=0.5)

# -----------------------
# Allocate best fit lists
# -----------------------

B_x_list = np.array([])
B_norm_y_list = np.array([])
B_diff_y_list = np.array([])

N_x_list = np.array([])
N_norm_y_list = np.array([])
N_diff_y_list = np.array([])

# -----------
# Find errors
# -----------

for N in range(N_iter):

    # slightly perturbed initial data
    a = a0 + epsilon*rand.uniform(-1, 1)
    b = b0 + epsilon*rand.uniform(-1, 1)

    x0 = x00 + epsilon*rand.uniform(-1, 1)

    # lists to check iterations and errors
    B_iter_list = []
    B_norm_err_list = []
    B_diff_err_list = []

    N_iter_list = []
    N_norm_err_list = []
    N_diff_err_list = []

    # initial zero approximate
    B_prev_zero = (a + b)/2 
    N_prev_zero = x0

    for i in range(1,8):
        max_iters = 2**i
        B_iter_list.append(max_iters)
        N_iter_list.append(max_iters)

        B_zero = truncated_bisection(f, a, b, max_iters=max_iters)
        N_zero = truncated_newton(f, f_prime, x0, max_iters=max_iters)

        # ------------------
        # error calculations
        # ------------------

        # ---- residual error ----
        B_norm_error = abs(f(B_zero))
        B_norm_err_list.append(B_norm_error)
        N_norm_error = abs(f(N_zero))
        N_norm_err_list.append(N_norm_error)

        # ---- consecutive diff err ----
        B_diff_error = abs(B_prev_zero - B_zero)
        B_diff_err_list.append(B_diff_error)
        N_diff_error = abs(N_prev_zero - N_zero)
        N_diff_err_list.append(N_diff_error)

        B_prev_zero = B_zero
        N_prev_zero = N_zero

        # ---------
        # mask data
        # ---------

        B_norm_val = max(B_norm_error, eps)
        B_diff_val = max(B_diff_error, eps)
        N_norm_val = max(N_norm_error, eps)
        N_diff_val = max(N_diff_error, eps)

        # store only masked values
        B_x_list = np.append(B_x_list, max_iters)
        B_norm_y_list = np.append(B_norm_y_list, B_norm_val)
        B_diff_y_list = np.append(B_diff_y_list, B_diff_val)
        N_x_list = np.append(N_x_list, max_iters)
        N_norm_y_list = np.append(N_norm_y_list, N_norm_val)
        N_diff_y_list = np.append(N_diff_y_list, N_diff_val)

    # -----------
    # Plot errors
    # -----------

    # stop at machine precision
    B_norm = np.array(B_norm_err_list)
    B_diff = np.array(B_diff_err_list)
    N_norm = np.array(N_norm_err_list)
    N_diff = np.array(N_diff_err_list)
    
    B_norm_plot = np.maximum(B_norm, eps)
    B_diff_plot = np.maximum(B_diff, eps)
    N_norm_plot = np.maximum(N_norm, eps)
    N_diff_plot = np.maximum(N_diff, eps)

    # plot
    ax1.loglog(B_iter_list, B_norm_plot, "-o", color="tab:blue", alpha=opacity)
    ax1.loglog(B_iter_list, B_diff_plot, "-o", color="tab:orange", alpha=opacity)
    ax2.loglog(N_iter_list, N_norm_plot, "-o", color="tab:blue", alpha=opacity)
    ax2.loglog(N_iter_list, N_diff_plot, "-o", color="tab:orange", alpha=opacity)

# Fit linear regression
log_B_x = np.log10(B_x_list)
log_B_norm_y = np.log10(B_norm_y_list)
log_B_diff_y = np.log10(B_diff_y_list)
log_N_x = np.log10(N_x_list)
log_N_norm_y = np.log10(N_norm_y_list)
log_N_diff_y = np.log10(N_diff_y_list)

B_norm_slope, B_norm_intercept = np.polyfit(log_B_x, log_B_norm_y, 1)
B_diff_slope, B_diff_intercept = np.polyfit(log_B_x, log_B_diff_y, 1)
N_norm_slope, N_norm_intercept = np.polyfit(log_N_x, log_N_norm_y, 1)
N_diff_slope, N_diff_intercept = np.polyfit(log_N_x, log_N_diff_y, 1)

# Smooth x-values for line
B_x_fit = np.logspace(0, 2.1, 200)
N_x_fit = np.logspace(0, 2.1, 200)

B_norm_y_fit = 10**(B_norm_slope * np.log10(B_x_fit) + B_norm_intercept)
B_diff_y_fit = 10**(B_diff_slope * np.log10(B_x_fit) + B_diff_intercept)
N_norm_y_fit = 10**(N_norm_slope * np.log10(N_x_fit) + N_norm_intercept)
N_diff_y_fit = 10**(N_diff_slope * np.log10(N_x_fit) + N_diff_intercept)

B_norm_intercept = 10**B_norm_intercept
B_diff_intercept = 10**B_diff_intercept
N_norm_intercept = 10**N_norm_intercept
N_diff_intercept = 10**N_diff_intercept

# plot best-fit lines w small black outline
"""ax1.loglog(B_x_fit, B_norm_y_fit, '-', color='black', linewidth=5)
ax1.loglog(B_x_fit, B_diff_y_fit, '-', color='black', linewidth=5)
ax1.loglog(B_x_fit, B_norm_y_fit, '-', label=f'Fit: y = {B_norm_intercept:.2f} x^{B_norm_slope:.2f}', color="tab:blue", linewidth=3.5)
ax1.loglog(B_x_fit, B_diff_y_fit, '-', label=f'Fit: y = {B_diff_intercept:.2f} x^{B_diff_slope:.2f}', color="tab:orange", linewidth=3.5)

ax2.loglog(N_x_fit, N_norm_y_fit, '-', color='black', linewidth=5)
ax2.loglog(N_x_fit, N_diff_y_fit, '-', color='black', linewidth=5)
ax2.loglog(N_x_fit, N_norm_y_fit, '-', label=f'Fit: y = {N_norm_intercept:.2f} x^{N_norm_slope:.2f}', color="tab:blue", linewidth=3.5)
ax2.loglog(N_x_fit, N_diff_y_fit, '-', label=f'Fit: y = {N_diff_intercept:.2f} x^{N_diff_slope:.2f}', color="tab:orange", linewidth=3.5)
"""
# pretend labels
ax1.loglog([0], [0], "-o", label=r"$|f(x_n)|$", color="tab:blue")
ax1.loglog([0], [0], "-o", label=r"$|x_n - x_{n-1}|$", color="tab:orange")
ax1.legend()
ax2.loglog([0], [0], "-o", label=r"$|f(x_n)|$", color="tab:blue")
ax2.loglog([0], [0], "-o", label=r"$|x_n - x_{n-1}|$", color="tab:orange")
ax2.legend()

plt.tight_layout()
plt.show()
