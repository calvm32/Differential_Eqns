import numpy as np

def rayleigh_quotient(A, x):
    """
    Solve for eigenvalue given eigenvector
    """
    
    x_star = x.conj().T # conjugate transpose
    return (x_star @ A @ x) / (x_star @ x)