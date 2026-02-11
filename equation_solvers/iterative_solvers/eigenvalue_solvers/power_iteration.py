import numpy as np

def power_iteration(A, num_iters = 100, tol = 1e-15):
    # Choose a random vector to decrease the chance that 
    # our vector is orthogonal to the eigenvector
    v_k = np.random.rand(A.shape[1]) # random vector 
    v_k /= np.linalg.norm(v_k)

    for num in range(num_iters):
        # calculate the matrix-by-vector product Ab
        v_k1 = np.dot(A, v_k)
        v_k1 /= np.linalg.norm(v_k1)

        if np.linalg.norm(v_k1 - v_k) < tol:
          break

        v_k = v_k1

    return v_k, num