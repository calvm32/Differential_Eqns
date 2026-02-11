import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random as rand

# ------------------
# Animation function
# ------------------

def animate_cobweb_and_orbit(f, x0, a, b, r=None, iters=25):

    # ---------- Precompute orbit ----------
    x_vals = np.zeros(iters + 1)
    x_vals[0] = x0
    for n in range(iters):
        x_vals[n + 1] = f(x_vals[n], r)

    # ---------- Precompute cobweb points ----------
    px = np.zeros(2 * iters + 1)
    py = np.zeros(2 * iters + 1)
    px[0], py[0] = x0, 0.0

    for k in range(1, 2 * iters + 1):
        if k % 2 == 1:      # vertical step
            px[k] = px[k - 1]
            py[k] = f(px[k - 1], r)
        else:               # horizontal step
            px[k] = py[k - 1]
            py[k] = py[k - 1]

    # ---------- Figure ----------
    fig, (ax_cob, ax_orb) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )

    # ---------- Cobweb background ----------
    x = np.linspace(a, b, 600)
    ax_cob.plot(x, f(x, r), c="#444444", lw=2)
    ax_cob.plot(x, x, c="#444444", lw=2)

    cobweb_line, = ax_cob.plot([], [], c="tab:blue", lw=1.5)

    ax_cob.set_xlim(a, b)
    ax_cob.set_ylim(a, b)
    ax_cob.set_aspect("equal")
    ax_cob.set_xlabel("$x$")
    ax_cob.set_ylabel("$f(x)$")
    ax_cob.set_title("Cobweb")
    ax_cob.grid(True, alpha=0.4)

    # ---------- Orbit plot ----------
    orbit_line, = ax_orb.plot([], [], marker="o", lw=1.5, ms=4)

    ax_orb.set_xlim(0, iters)
    ax_orb.set_xlabel("Iteration $n$")
    ax_orb.set_ylabel("$x_n$")
    ax_orb.set_title("Orbit plot")
    ax_orb.grid(True, alpha=0.4)

    # Fix orbit y-limits ONCE
    ymin = x_vals.min()
    ymax = x_vals.max()
    pad = 0.1 * (ymax - ymin + 1e-12)
    ax_orb.set_ylim(ymin - pad, ymax + pad)

    title = fig.suptitle("")

    # ---------- Animation update ----------
    def update(n):
        cobweb_line.set_data(px[:2 * n + 1], py[:2 * n + 1])
        orbit_line.set_data(range(n + 1), x_vals[:n + 1])

        if r is None:
            title.set_text(rf"$x_0={x0:.3f},\ n={n}$")
        else:
            title.set_text(rf"$x_0={x0:.3f},\ r={r:.3f},\ n={n}$")

        return cobweb_line, orbit_line, title

    ani = FuncAnimation(
        fig,
        update,
        frames=iters + 1,
        interval=200,
        blit=False
    )

    plt.show()
    return ani

# ----------
# Run things
# ----------

def f(x, r=None):
    func = np.cos(x) - x
    func_prime = -np.sin(x) - 1
    return x - func / func_prime

x0 = 0.74
a, b = -5.0, 5.0
iters = 50
epsilon = 1e-1
#x0 += epsilon*rand.uniform(-1, 1)

ani = animate_cobweb_and_orbit(f, x0, a, b, iters=iters)
ani.save("newton_cobweb_and_orbit.gif", writer="pillow", fps=10)
