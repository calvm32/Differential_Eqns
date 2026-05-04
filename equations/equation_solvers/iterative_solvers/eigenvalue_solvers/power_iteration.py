import numpy as np

def power_iteration(A, num_iters = 100, tol = 1e-15):
    """
    Solve Ax = lambda x using inverse power iteration
        -> approximate b_k
        -> iterate b_k1 = A b_k / || b_k ||
        -> return largest eigenvector
    """

    # Choose a random vector to decrease the chance that 
    # our vector is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1]) # random vector 
    b_k /= np.linalg.norm(b_k)

    for num in range(num_iters):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        b_k1 /= np.linalg.norm(b_k1)

        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    return b_k, num