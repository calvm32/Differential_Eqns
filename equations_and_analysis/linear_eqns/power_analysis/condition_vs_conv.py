from equation_solvers.iterative_solvers.eigenvalue_solvers import *
from .matrix_generators.random_spectrum import *
import numpy as np
import matplotlib.pyplot as plt
import random as rand

seed_num = 10
plot_num = 10000
dim = 20

# error vs. condition number; plot_num times
plt.figure() # first plot

errors_pow = []
errors_inv = []
conditions = []

for _ in range(plot_num):

    lambda_big, x_big, lambda_small, x_small, M = random_spectrum(dim)
    condition = abs(lambda_big / lambda_small) 
    conditions.append(condition)

    # calculate power iteration eigenvalue
    x_approx, num = power_iteration(M, num_iters = 10)
    lambda_approx = rayleigh_quotient(M, x_approx)
    errors_pow.append(np.abs(lambda_big - lambda_approx))

    # calculate inv. power iteratoin eigenvalue
    x_approx, num = inverse_power_iteration(M, num_iters = 10)
    lambda_approx = rayleigh_quotient(M, x_approx)
    errors_inv.append(np.abs(lambda_small - lambda_approx))

# Now plot the entire data
plt.loglog(conditions, errors_pow, 'o', color="tab:blue", alpha=0.2, label="power method")
plt.loglog(conditions, errors_inv, 'o', color="tab:orange", alpha=0.2, label="inverse power method")

plt.xlabel('log(condition number)')
plt.ylabel('log(error)')
plt.title('Error vs. Condition number')
plt.legend()
plt.show()
