from equation_solvers.iterative_solvers.eigenvalue_solvers import *
from equations_and_analysis.linear_eqns.matrix_generators.prescribed_spectrum import *
import numpy as np
import matplotlib.pyplot as plt

seed_num = 10
plot_num = 19 # if want to overlay more plots later
num_iters = 30 # loop through number of iterations before final eigenval calculation

# error vs. iteration number
plt.figure() # first plot

# --------------------------------------
# list of eigenvalues to experiment with
# --------------------------------------

# eigenvals = [ 0.1 , 0.7, 0.9, 1.5, 2.0 ]
# eigenvals = [ 2.0, 1.99, 1.98, 0.5, 0.1 ]
eigenvals = [ 0.1, 1.99, 1.98, 0.5, 2.0 ]

for _ in range(plot_num):
    errors_qr = []
    errors_qrs = []
    iteration_count = []

    lambda_exact, x_exact, lambda_small, x_small, M = prescribed_spectrum(eigenvals)

    for i in range(1,num_iters):
        iteration_count.append(i)

        # calculate QR eigenvalue
        eigenvals, num = qr_iteration(M, num_iters = i)
        lambda_approx = np.sort(eigenvals)[-1] # largest
        errors_qr.append(np.abs(lambda_exact - lambda_approx))

        # calculate QR shifted eigenvalue
        eigenvals, num = qr_shifted_iteration(M, num_iters = i)
        lambda_approx = np.sort(eigenvals)[-1] # largest
        errors_qrs.append(np.abs(lambda_exact - lambda_approx))

    plt.loglog(iteration_count, errors_qr, color="tab:blue", alpha=0.5)
    plt.loglog(iteration_count, errors_qrs, color="tab:orange", alpha=0.5)

plt.loglog(2, 2, color="tab:blue", label="QR")
plt.loglog(2, 2, color="tab:orange", label="shifted QR")

plt.xlabel('log(iteration count)')
plt.ylabel('log(error)')
plt.title('Error vs. Iteration number')
plt.legend()
plt.show()