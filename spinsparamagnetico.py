import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -----------------------------
# Parameters
# -----------------------------
L = 40
B = 0.02             # weak external field
T_values = np.linspace(0.5, 10, 50)
n_eq_sweeps = 150    # sweeps for equilibration
n_meas_sweeps = 300  # sweeps for measurement
np.random.seed(0)

# -----------------------------
# Metropolis sweep
# -----------------------------
def metropolis_sweep(spins, beta, B):
    N = spins.size
    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s_old = spins[i, j]
        dE = 2.0 * B * s_old  # only Zeeman term
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s_old
    return spins

# -----------------------------
# Initialization
# -----------------------------
spins = np.random.choice([-1, 1], size=(L, L))
N = L * L
chi_values = []
frame_data = []

# -----------------------------
# Temperature loop
# -----------------------------
for T in T_values:
    beta = 1.0 / T
    # Equilibration
    for _ in range(n_eq_sweeps):
        spins = metropolis_sweep(spins, beta, B)
    # Measurements
    M_list = []
    for _ in range(n_meas_sweeps):
        spins = metropolis_sweep(spins, beta, B)
        M_list.append(spins.sum())
    M_arr = np.array(M_list)
    chi = beta * (M_arr.var()) / N
    chi_values.append(chi)
    frame_data.append((spins.copy(), chi_values.copy()))

# -----------------------------
# Figure and animation setup
# -----------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 13
})

fig, (ax_lattice, ax_curve) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(top=0.9, wspace=0.25)

# Spin lattice
img = ax_lattice.imshow(frame_data[0][0], cmap='bwr', vmin=-1, vmax=1)
ax_lattice.set_title("Spin lattice (red = +1, blue = -1)")
ax_lattice.axis('off')

# χ(T) curve
ax_curve.set_xlim(T_values.min(), T_values.max())
ax_curve.set_ylim(0, max(chi_values) * 1.2)
ax_curve.set_xlabel("Temperature T")
ax_curve.set_ylabel("Magnetic susceptibility χ")
ax_curve.set_title("Curie law: χ ∝ 1/T")

(line,) = ax_curve.plot([], [], 'k-', lw=2, label="Simulation")
theory_line, = ax_curve.plot(T_values, 1/T_values, 'r--', label="1/T (Curie law)")
ax_curve.legend(loc='upper right')

# -----------------------------
# Update function
# -----------------------------
def update(frame):
    spins_snapshot, chi_list = frame_data[frame]
    img.set_data(spins_snapshot)
    line.set_data(T_values[:len(chi_list)], chi_list)
    ax_curve.set_title(f"χ(T) — Current T = {T_values[frame]:.2f}")
    return [img, line]

# -----------------------------
# Create video
# -----------------------------
ani = FuncAnimation(fig, update, frames=len(frame_data), blit=True)
writer = FFMpegWriter(fps=4, bitrate=2000)
ani.save("curie_law_simulation.mp4", writer=writer)

plt.close()
print("✅ Video saved as curie_law_simulation.mp4")
