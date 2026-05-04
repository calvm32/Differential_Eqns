import numpy as np
from scipy.linalg import lu, solve

def inverse_power_iteration(A, num_iters = 100, tol = 1e-15):
    """
    Solve Ax = lambda x using inverse power iteration
        -> approximate b_k 
        -> iterate b_k1 = A^-1 b_k / || b_k ||
        -> return smallest eigenvector
    """
    
    P, L, U = lu(A)
    
    # Choose a random vector to decrease the chance that 
    # our vector is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1]) # random vector
    b_k /= np.linalg.norm(b_k)

    for num in range(num_iters):
        # calculate vector x_k such that Lx_k = b_k, 
        # then calculate b_k1 such that Ub_k1 = x_k
        b_k1 = solve(U, solve(L, b_k))
        b_k1 /= np.linalg.norm(b_k1)

        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    return b_k, num