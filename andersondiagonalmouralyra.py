import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.linalg import eigh_tridiagonal

# -------------------------
# Parameters
# -------------------------
N = 250
t_hop = 1.0
scale_factor = 2.0  # max |psi| for plotting
frame_per_state = 200  # frames per energy state (~10s at fps=20)
fps = 20

# -------------------------
# Moura-Lyra potential
# -------------------------
def moura_lyra(N, alpha):
    phi = 2*np.pi*np.random.rand(N//2)
    n = np.arange(N)[:, None]
    k = np.arange(1, N//2)
    eps = np.sum(k**(-alpha/2) * np.cos(2*np.pi*k*n/N + phi[k]), axis=1)
    eps = (eps - np.mean(eps))
    eps /= np.std(eps)
    return eps

eps0 = moura_lyra(N, alpha=0)
eps3 = moura_lyra(N, alpha=3)

# -------------------------
# Hamiltonian diagonalization (tridiagonal)
# -------------------------
def diagonalize(eps, t_hop):
    # Main diagonal
    d = eps
    # Off-diagonal
    e = np.ones(N-1) * t_hop
    energies, states = eigh_tridiagonal(d, e)
    return energies, states

E0, psi0_all = diagonalize(eps0, t_hop)
E3, psi3_all = diagonalize(eps3, t_hop)

# -------------------------
# Select eigenstates near given energy
# -------------------------
def select_state(energies, states, target, delta=0.05):
    idx = np.argmin(np.abs(energies - target))
    psi = states[:, idx]
    psi_scaled = psi / np.max(np.abs(psi)) * scale_factor
    return psi_scaled, energies[idx]

# -------------------------
# Prepare animation frames
# -------------------------
potentials = [eps0, eps3]
alphas = [0, 3]
energy_targets = [-0.5, 0.0, 0.5]

# Total frames
total_frames = len(energy_targets) * frame_per_state

fig, axes = plt.subplots(1, 2, figsize=(14,6))
plt.rcParams.update({'font.size': 14})

def animate(i):
    for ax in axes:
        ax.cla()
    # Determine which energy state
    state_idx = i // frame_per_state
    frame_in_state = i % frame_per_state
    energy_target = energy_targets[state_idx]

    for j, eps in enumerate(potentials):
        psi_scaled, E_actual = select_state(
            E0 if j==0 else E3,
            psi0_all if j==0 else psi3_all,
            energy_target,
            delta=0.05
        )
        axes[j].plot(np.arange(N), eps, color='red', alpha=0.9, label='Potential')
        axes[j].plot(np.arange(N), psi_scaled, color='blue', alpha=0.5, label=f'Energy eigenstate ~ {E_actual:.2f}')
        axes[j].set_title(f'α={alphas[j]}', fontsize=15)
        axes[j].set_ylim(-2.5, 2.5)
        axes[j].legend(fontsize=13)
        axes[j].set_xlabel('Site', fontsize=13)
        axes[j].set_ylabel('Value', fontsize=13)
    fig.suptitle(f'Potential + Energy Eigenstate near E={energy_target}', fontsize=16)

anim = FuncAnimation(fig, animate, frames=total_frames, interval=50)

# -------------------------
# Save video
# -------------------------
writer = FFMpegWriter(fps=fps)
anim.save("9.mp4", writer=writer)
plt.close()
