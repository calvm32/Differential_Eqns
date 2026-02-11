import numpy as np

"""
Make an nxn matrix with a prescribed condition number and return the 
    - the largest eigenpair
    - the smallest eigenpair
    - corresp. matrix
"""

def conditioned_spectrum(n, cond_num):
    rng = np.random.default_rng()  # No seed, random every time
    
    # Random eigenvalues: Scale to get the desired condition number
    smallest_eigenvalue = rng.standard_normal()  # Random smallest eigenvalue
    largest_eigenvalue = cond_num * np.abs(smallest_eigenvalue)  # Ensure condition number is met
    
    # Create random eigenvalues with the prescribed condition number
    eigenvalues = np.linspace(smallest_eigenvalue, largest_eigenvalue, n)
    
    # Generate random orthonormal eigenvectors using QR decomposition
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    # Find the largest and smallest eigenvalue indices
    idx_largest = np.argmax(np.abs(eigenvalues))
    idx_smallest = np.argmin(np.abs(eigenvalues))

    largest_eigenvalue = eigenvalues[idx_largest]
    largest_eigenvector = Q[:, idx_largest]
    
    smallest_eigenvalue = eigenvalues[idx_smallest]
    smallest_eigenvector = Q[:, idx_smallest]

    # Reconstruct the matrix with the corresponding eigenvalues and eigenvectors
    M = Q @ np.diag(eigenvalues) @ Q.T

    return largest_eigenvalue, largest_eigenvector, smallest_eigenvalue, smallest_eigenvector, M
