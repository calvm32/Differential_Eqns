import numpy as np

def prescribed_spectrum(eigenvalues):
    """
    make a random SYMMETRIC matrix with a prescribed spectrum and return the 
        - the largest eigenpair
        - the smallest eigenpair
        - corresp. matrix
    """
    
    rng = np.random.default_rng()

    eigenvalues = np.array(eigenvalues)
    n = eigenvalues.size

    # random orthonormal eigenvectors
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    # index of eigenvalue with largest absolute value
    idx_max = np.argmax(np.abs(eigenvalues))
    largest_eigenvalue = eigenvalues[idx_max]
    largest_eigenvector = Q[:, idx_max]

    # index of eigenvalue with smallest absolute value
    idx_min = np.argmin(np.abs(eigenvalues))
    smallest_eigenvalue = eigenvalues[idx_min]
    smallest_eigenvector = Q[:, idx_min]

    # symmetric matrix with given spectrum
    M = Q @ np.diag(eigenvalues) @ Q.T

    return largest_eigenvalue, largest_eigenvector, \
           smallest_eigenvalue, smallest_eigenvector, M
