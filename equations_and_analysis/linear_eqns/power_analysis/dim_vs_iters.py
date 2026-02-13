from equation_solvers.iterative_solvers.eigenvalue_solvers import *
from .matrix_generators.random_spectrum import *
import numpy as np
import matplotlib.pyplot as plt
import random as rand

seed_num = 10
plot_num = 50
dim_exp = 10

# num. iterations vs. dimension; plot_num times
plt.figure() # second plot

for _ in range(plot_num):
    iterations_pow = []
    iterations_inv = []
    dimensions = []
    tol = 1e-10

    for i in range(1,dim_exp):
        n = 2**i
        dimensions.append(n)

        lambda_big, x_big, lambda_small, x_small, M = random_spectrum(n)

        # calculate power iteration eigenvalue
        x_approx, num = power_iteration(M, tol = 1e-4)
        lambda_approx = rayleigh_quotient(M, x_approx)
        iterations_pow.append(num)

        # calculate inv. power iteratoin eigenvalue
        x_approx, num = inverse_power_iteration(M, tol = 1e-4)
        lambda_approx = rayleigh_quotient(M, x_approx)
        iterations_inv.append(num)

        plt.loglog(dimensions, iterations_pow, color="tab:blue", alpha=0.2)
        plt.loglog(dimensions, iterations_inv, color="tab:orange", alpha=0.2)

plt.loglog(2, 10, color="tab:blue", label="power method")
plt.loglog(2, 10, color="tab:orange", label="inverse power method")

plt.xlabel('log(dimension)')
plt.ylabel('log(iterations before tolerance)')
plt.title('Number of Iterations vs. Dimension (Fixed Tolerance)')
plt.legend()
plt.show()