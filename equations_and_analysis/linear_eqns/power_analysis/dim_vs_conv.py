from equation_solvers.iterative_solvers.eigenvalue_solvers import *
from .matrix_generators.random_spectrum import *
import numpy as np
import matplotlib.pyplot as plt
import random as rand

seed_num = 10
plot_num = 50
dim_exp = 10

# error vs. dimension; plot_num times
plt.figure() # first plot

for _ in range(plot_num):
    errors_pow = []
    errors_inv = []
    dimensions = []

    for i in range(1,dim_exp):
        n = 2**i
        dimensions.append(n)

        lambda_big, x_big, lambda_small, x_small, M = random_spectrum(n)

        # calculate power iteration eigenvalue
        x_approx, num = power_iteration(M, num_iters = 10)
        lambda_approx = rayleigh_quotient(M, x_approx)
        errors_pow.append(np.abs(lambda_big - lambda_approx))

        # calculate inv. power iteratoin eigenvalue
        x_approx, num = inverse_power_iteration(M, num_iters = 10)
        lambda_approx = rayleigh_quotient(M, x_approx)
        errors_inv.append(np.abs(lambda_small - lambda_approx))

        plt.loglog(dimensions, errors_pow, color="tab:blue", alpha=0.2)
        plt.loglog(dimensions, errors_inv, color="tab:orange", alpha=0.2)

plt.loglog(2, 2, color="tab:blue", label="power method")
plt.loglog(2, 2, color="tab:orange", label="inverse power method")

plt.xlabel('log(dimension)')
plt.ylabel('log(error)')
plt.title('Error vs. Dimension (Fixed Iterations)')
plt.legend()
plt.show()