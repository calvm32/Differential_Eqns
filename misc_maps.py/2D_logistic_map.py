import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

# constants 
a = 2.8
b = 3
N = 500 # number of iterations

# ICs in (0,1)
x = 0.5
y = 0.5

x_list = [] # list of x vals
y_list = [] # list of y vals
count_list = [] # folor color mapping

sin_list = [] # for color mapping comparison
angles_list = [] # for color mapping comparison

for n in range(N):
    x = a*x*(1-x)
    y = b*y*(1-y)

    if n > 100:
        x_list.append(x)
        y_list.append(y)
        count_list.append(n)

# get sine map
for n in count_list:
    x = (n-100)*(2 * math.pi) / (N - 100) # 0 when n = 100, pi when n = 500
    angles_list.append(x)
    sin_list.append(math.sin(x))

# got this soln from google; also used in previous assignment
fig = plt.figure()
gs = GridSpec(2, 1, figure=fig, hspace=0.4) # actual layout

# sine plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(angles_list, sin_list, c=count_list, cmap='viridis')
ax1.set_title("sine map for time comparison")

# 2D logistic plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(x_list, y_list, c=count_list, cmap='viridis')
ax2.set_title("2D logistic map")

plt.show()