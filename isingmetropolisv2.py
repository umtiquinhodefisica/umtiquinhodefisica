import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Patch

# ------------------------
# Parameters
# ------------------------
L = 60           # lattice size
T = 1.0          # low temperature
beta = 1.0 / T
J = 1.0
steps = 450      # number of sweeps

# ------------------------
# Initialization: random spins
# ------------------------
spins = np.random.choice([-1, 1], size=(L, L))

def total_energy(spins, J=1.0):
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    return -J * np.sum(spins * (right + down))

def deltaE_local(spins, i, j, J=1.0):
    s = spins[i, j]
    nb = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
    return 2.0 * J * s * nb

# ------------------------
# Animation setup
# ------------------------
fig = plt.figure(figsize=(12, 6), dpi=120)

# Grid layout: 1 row, 2 columns; right column subdivided
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.35)

ax1 = fig.add_subplot(gs[:, 0])   # left: spins
ax2 = fig.add_subplot(gs[0, 1])   # top right: energy
ax3 = fig.add_subplot(gs[1, 1])   # bottom right: magnetization

# panel 1: spins
im = ax1.imshow(spins, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
ax1.set_title("Spin configuration (Metropolis)", fontsize=13)
ax1.set_xticks([]); ax1.set_yticks([])

# legend for spin colors
legend_elements = [
    Patch(facecolor='red', edgecolor='k', label='Spin up (+1)'),
    Patch(facecolor='blue', edgecolor='k', label='Spin down (-1)')
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.8)

# panel 2: energy
energies = []
line_E, = ax2.plot([], [], 'k-')
ax2.set_xlim(0, steps)
ax2.set_ylim(-2.05, -0.5)
ax2.set_title("Energy per spin", fontsize=13)
ax2.set_xlabel("Sweep")
ax2.set_ylabel("E/N")

# panel 3: magnetization
magnetizations = []
line_M, = ax3.plot([], [], 'r-')
ax3.set_xlim(0, steps)
ax3.set_ylim(-1.05, 1.05)
ax3.set_title("Magnetization per spin", fontsize=13)
ax3.set_xlabel("Sweep")
ax3.set_ylabel("M/N")

# info text under spins
text = ax1.text(0.02, -0.08, "", transform=ax1.transAxes,
                ha="left", va="top", color="black",
                fontsize=9, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

sweep_counter = 0

def update(frame):
    global spins, sweep_counter, energies, magnetizations

    # 1 sweep
    for _ in range(L*L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        dE = deltaE_local(spins, i, j, J)
        accept = (dE <= 0) or (np.random.rand() < np.exp(-beta * dE))
        if accept:
            spins[i, j] *= -1

    # energy & magnetization
    E = total_energy(spins, J) / (L*L)
    M = np.sum(spins) / (L*L)
    energies.append(E)
    magnetizations.append(M)
    sweep_counter += 1

    # update plots
    im.set_data(spins)
    line_E.set_data(range(len(energies)), energies)
    line_M.set_data(range(len(magnetizations)), magnetizations)
    text.set_text(f"Sweep {sweep_counter}, E/N={E:.3f}, M/N={M:.3f}")

    return im, line_E, line_M, text

ani = FuncAnimation(fig, update, frames=steps, interval=80, blit=False, repeat=False)

# ------------------------
# Save animation as MP4
# ------------------------
writer = FFMpegWriter(fps=32, bitrate=1800)
ani.save("ising_metropolis_EM_split.mp4", writer=writer)

plt.show()
