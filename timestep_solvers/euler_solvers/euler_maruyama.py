import numpy as np

def euler_maruyama(a, b, X0, t0, T, dt, dim): 
    """
    Solve the SDE dX = a(X,t)dt + b(X,t)dW on the interval [t0,T] with X(t0) = X0
    using the forward Euler approximation method 
    """
    
    N = int(np.floor((T-t0)/dt) + 1) # fixed number of steps
    t = t0+dt*np.arange(0,N)
    
    X = np.zeros((dim, N), dtype = float)
    X[:, 0] = X0 # Set initial value

    for n in range(0,N-1):
        dW = np.sqrt(dt) * np.random.randn()
        X[:, n+1] = X[:, n] + a(t[n], X[:, n])*dt + b(t[n], X[:, n])*dW
    
    return X,t