import matplotlib.pyplot as plt

"""
This program plots the amount of time it takes for the 
Collatez map to terminate at 1
"""

# -----
# setup
# -----

IC_list = [] # initial conditions, ie. 1,2,...10000
count_list = [] # how long it takes for resp. IC to reach 1

# determine time takes to terminate
for n in range(1,10001):
    IC_list.append(n)
    x = n # x_1 value

    counter = 1 # to find n such that x_n = 1
    while x != 1:
        if x % 2 == 0:
            x *= 0.5
        else:
            x = 3*x+1
        counter += 1

    count_list.append(counter)

# ----
# plot
# ----

plt.semilogx(IC_list, count_list)
plt.xlabel("Initial Condition")
plt.ylabel("log(Iterations taken to reach 1)")
plt.show()