import numpy as np
from scipy.linalg import lu, solve

def inverse_power_iteration(A, num_iters = 100, tol = 1e-15):
    P, L, U = lu(A)
    
    # Choose a random vector to decrease the chance that 
    # our vector is orthogonal to the eigenvector
    v_k = np.random.rand(A.shape[1]) # random vector
    v_k /= np.linalg.norm(v_k)

    for num in range(num_iters):
        # calculate vector x_k such that Lx_k = v_k, 
        # then calculate v_k1 such that Uv_k1 = x_k
        v_k1 = solve(U, solve(L, v_k))
        v_k1 /= np.linalg.norm(v_k1)

        if np.linalg.norm(v_k1 - v_k) < tol:
          break

        v_k = v_k1

    return v_k, num