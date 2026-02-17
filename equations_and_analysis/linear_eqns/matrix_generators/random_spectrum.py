import numpy as np

def random_spectrum(n):
    """
    make a random matrix with a random spectrum and return the 
        - the largest eigenpair
        - the smallest eigenpair
        - corresp. matrix
    """
    
    rng = np.random.default_rng()  # No seed, random every time

    # random spectrum
    eigenvalues = rng.standard_normal(n)

    # random orthonormal eigenvectors
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    # index of eigenvalue with largest absolute value
    idx = np.argmax(np.abs(eigenvalues))

    largest_eigenvalue = eigenvalues[idx]
    largest_eigenvector = Q[:, idx]

    # index of eigenvalue with smallest absolute value
    idx = np.argmin(np.abs(eigenvalues))

    smallest_eigenvalue = eigenvalues[idx]
    smallest_eigenvector = Q[:, idx]

    # matrix w/ corresponding eigenpairs
    M = Q @ np.diag(eigenvalues) @ Q.T

    return largest_eigenvalue, largest_eigenvector, smallest_eigenvalue, smallest_eigenvector, M
