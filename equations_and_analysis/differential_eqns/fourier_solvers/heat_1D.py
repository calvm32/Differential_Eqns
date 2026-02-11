import numpy as np
import matplotlib.pyplot as plt
from equation_solvers.timestep_solvers.rk4_solvers.rk4_heat_1D import rk4_heat_1D

nu, t0, T = 0.01, 0.0, 5.0 # diffusion coeff, init time, final time
L, N = 2*np.pi, 2**8 # Domain length, number of points