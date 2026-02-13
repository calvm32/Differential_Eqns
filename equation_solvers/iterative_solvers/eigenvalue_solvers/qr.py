import numpy as np

def qr_iteration(A, num_iters=100, tol=1e-12):
    A_k = A

    for num in range(num_iters):
        # QR factorization
        Q, R = np.linalg.qr(A_k)
        A_k1 = R @ Q

        # check off-diagonal norm small
        off_diag_norm = np.linalg.norm(A_k1 - np.diag(np.diag(A_k1)))

        if off_diag_norm < tol:
            break

        A_k = A_k1

    eigenvalues = np.diag(A_k)

    return eigenvalues, num