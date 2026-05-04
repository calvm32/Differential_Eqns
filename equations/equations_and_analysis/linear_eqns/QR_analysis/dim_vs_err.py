from equation_solvers.iterative_solvers.eigenvalue_solvers import *
from equations_and_analysis.linear_eqns.matrix_generators.random_spectrum import *
import numpy as np
import matplotlib.pyplot as plt

"""
Compare dimension vs. final error
"""

def main():

    seed_num = 10
    plot_num = 20
    dim_exp = 10

    # error vs. dimension; 50 times
    plt.figure() # first plot

    for _ in range(plot_num):
        errors_pow = []
        errors_inv = []
        dimensions = []

        for i in range(1,dim_exp):
            n = 2**i
            dimensions.append(n)

            lambda_exact, x_exact, lambda_small, x_small, M = random_spectrum(n)

            # calculate QR eigenvalue
            eigenvals, num = qr_iteration(M, num_iters = 30)
            lambda_approx = eigenvals[0]
            errors_pow.append(np.abs(lambda_exact - lambda_approx))

            # calculate shifted QR eigenvalue
            eigenvals, num = qr_shifted_iteration(M, num_iters = 10)
            lambda_approx = eigenvals[0]
            errors_inv.append(np.abs(lambda_exact - lambda_approx))

            plt.loglog(dimensions, errors_pow, color="tab:blue", alpha=0.2)
            plt.loglog(dimensions, errors_inv, color="tab:orange", alpha=0.2)

    plt.loglog(2, 2, color="tab:blue", label="QR")
    plt.loglog(2, 2, color="tab:orange", label="shifted QR")

    plt.xlabel('log(dimension)')
    plt.ylabel('log(error)')
    plt.title('Error vs. Dimension (Fixed Iterations)')
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()