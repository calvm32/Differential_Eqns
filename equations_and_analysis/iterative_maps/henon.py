import matplotlib.pyplot as plt
import math
import random
import numpy as np
from matplotlib.gridspec import GridSpec

"""
This program plots the Hénon map
"""

# -----
# setup
# -----

# constants 
a = 1.4
b = 0.005
N_start = 100 # first iteration to plot
N_end = 500 # final iteration to plot

# random initial conditions
x0 = random.uniform(-1, 1)
y0 = random.uniform(-1, 1)

x_list = [] # list of x vals
y_list = [] # list of y vals
count_list = [] # for color mapping

sin_list = [] # for color mapping comparison
angles_list = [] # for color mapping comparison

# norm for error evaluation
def frob_norm(x,y):
    return ((a*(1-y))**2 + (a*x)**2 + (b*y)**2 + (b*(1-x))**2)**(1/2)

# -----
# solve
# -----

def run_mapping(x0, y0, a, b, N_start, N_end):
    LyE = 0

    x_new=x0
    y_new=y0
    for n in range(N_end):
        LyE += math.log(frob_norm(x_new,y_new))/N_end # norm of nth time step, before dividing

        # only print from N_start (100) to N_end (500)
        if n >= N_start:
            x_list.append(x_new)
            y_list.append(y_new)
            count_list.append(n)

        x_old = x_new
        y_old = y_new

        x_new = 1-a*(x_old)**2 + y_old
        y_new = b*x_old

    return(LyE) 


# ------------
# plot one run
# ------------  

"""
# get sine map for color mapping comparison
for n in count_list:
    x = (n-N_start)*(2 * math.pi) / (N_end - N_start) # 0 when n = N_start, 2pi when n = N_end
    angles_list.append(x)
    sin_list.append(math.sin(x))

# got this soln from google
fig = plt.figure()
gs = GridSpec(2, 1, figure=fig, hspace=0.4) # actual layout

# sine plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(angles_list, sin_list, c=count_list, cmap='viridis')
ax1.set_title("sine map for time comparison")

# Henon map plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(x_list, y_list, c=count_list, cmap='viridis')
ax2.set_title("Henon map")

fig.suptitle(f"Initial condition=({x0:.2f},{y0:.2f}), a={a}, b={b}", fontsize=16, fontweight='bold') # got suptitle from google ai result

plt.show()
"""

# ----------------------
# plot LyE over many ICs
# ----------------------

b_list = np.arange(-2, 1, 0.001) # calculate every thousandth
LyE_list = [] # get LyE for system given b

for b_val in b_list:
    LyE_list.append(run_mapping(x0, x0, a, b_val, N_start, N_end))

# LyE plot
plt.scatter(b_list, LyE_list)
plt.title("LyE values of Henon map")
plt.suptitle(f"Initial condition=({x0:.2f},{y0:.2f})", fontsize=16, fontweight='bold') # got suptitle from google ai result

plt.show()