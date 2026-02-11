import numpy as np

def rayleigh_quotient(A: np.ndarray, x: np.ndarray) -> float:
  x_star = x.conj().T # conjugate transpose
  return (x_star @ A @ x) / (x_star @ x)