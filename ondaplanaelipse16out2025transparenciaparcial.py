import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse

# -----------------------------
# Global font configuration
# -----------------------------
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20
})

# ------------------------
# General parameters
# ------------------------
Nx, Ny = 450, 450
dx = 0.1
dt = 0.03
steps = 1550

c = 1.0
freq = 0.04
wavelength = c / freq
lambda_pts = int(round(wavelength / dx))

# ------------------------
# Source
# ------------------------
x0 = Nx // 2
y0 = 0  # left boundary

# ------------------------
# Function: create a semi-transparent elliptical obstacle
# ------------------------
def create_ellipse_obstacle(Nx, Ny, cx, cy, rx, ry, c_val=1.0):
    """Creates a velocity grid with an elliptical region (reduced wave speed inside)."""
    y, x = np.ogrid[:Nx, :Ny]
    mask = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2) <= 1.0
    grid = c_val * np.ones((Nx, Ny))
    grid[mask] = 0.45  # reduced velocity inside the ellipse
    return grid, mask

# ------------------------
# Initial field and obstacle
# ------------------------
E = np.zeros((Nx, Ny))
E_prev = np.zeros((Nx, Ny))

cx = Ny // 2      # ellipse center (horizontal)
cy = Nx // 2      # ellipse center (vertical)
rx = 0.11 * Nx     # horizontal radius
ry = 0.08 * Nx     # vertical radius
c_grid, mask = create_ellipse_obstacle(Nx, Ny, cx, cy, rx, ry, c_val=1.0)

# ------------------------
# FDTD update function
# ------------------------
def step(E, E_prev, c_grid):
    lap = np.zeros_like(E)
    lap[1:-1,1:-1] = (E[2:,1:-1] + E[:-2,1:-1] +
                      E[1:-1,2:] + E[1:-1,:-2] -
                      4*E[1:-1,1:-1]) / dx**2
    E_next = 2*E - E_prev + (c_grid*dt)**2 * lap
    return E_next

# ------------------------
# Figure setup
# ------------------------
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(np.zeros((Nx, Ny)), origin='lower',
               cmap='RdBu', vmin=-0.1, vmax=0.1,
               extent=[0, Nx*dx, 0, Ny*dx])
ax.set_title("Wave Scattering by Semi-Transparent\n Elliptical Obstacle")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add elliptical patch (visual obstacle)
ellipse_patch = Ellipse((cx*dx, cy*dx), 2*rx*dx, 2*ry*dx,
                        facecolor='black', edgecolor='white', alpha=0.5, linewidth=0.8)
ax.add_patch(ellipse_patch)

# ------------------------
# Animation update function
# ------------------------
def update(frame):
    global E, E_prev
    src = 0.7 * np.sin(2 * np.pi * freq * frame)
    E[:, y0] = src
    E[:, :y0] = 0
    E_next = step(E, E_prev, c_grid)
    E_prev = E.copy()
    E = E_next.copy()
    im.set_data(E)
    return [im, ellipse_patch]

# ------------------------
# Run animation and save MP4
# ------------------------
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
writer = FFMpegWriter(fps=100, bitrate=1800)
ani.save("onda_elipse1.mp4", writer=writer)

plt.close(fig)
print("Video generated: onda_elipse1.mp4")
