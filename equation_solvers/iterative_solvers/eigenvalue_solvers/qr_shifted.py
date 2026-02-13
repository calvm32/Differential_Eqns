import numpy as np

def qr_shifted_iteration(A, num_iters=100, tol=1e-12):
    A_k = A
    n = A.shape[0]

    for num in range(num_iters):
        # shift bottom right
        mu = A_k[n-1, n-1]
        Q, R = np.linalg.qr(A_k - mu * np.eye(n))

        A_k1 = R @ Q + mu * np.eye(n)

        # off diagonal norm small
        off_diag_norm = np.linalg.norm(A_k1 - np.diag(np.diag(A_k1)))

        if off_diag_norm < tol:
            break

        A_k = A_k1

    eigenvalues = np.diag(A_k)

    return eigenvalues, num
