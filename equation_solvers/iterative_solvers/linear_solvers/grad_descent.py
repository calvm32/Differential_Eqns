from math import *
import numpy as np

def grad_descent(f, grad_f, X0, X_exact, dt, max_iters=100):
    
    dim = X0.shape[0]
    X = np.zeros((dim, max_iters + 1), dtype = float)
    errors = np.zeros(max_iters + 1, dtype = float)
    t = np.arange(max_iters + 1) * dt

    X[:, 0] = X0 # Set initial value
    errors[0] = np.linalg.norm(f(X0))

    x_prev = X0

    # run descentabs
    for n in range(max_iters):
        x = x_prev - dt*grad_f(x_prev)
        errors[n+1] = np.linalg.norm(X_exact - x)
        X[:, n+1] = x 
        x_prev = x
        
    return X, t, errors