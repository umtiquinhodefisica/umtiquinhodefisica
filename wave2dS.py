import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ------------------------
# Parameters
# ------------------------
Nx, Ny = 100, 100         # grid size
dx = 0.1                  # spatial step
dt = 0.05                 # time step
c = 1.0                   # wave speed
steps = 350               # number of time steps

# Source position (beginning of the S channel)
x0, y0 = 0, Ny//2

# ------------------------
# Create S-shaped channel in a dense scatterer field
# ------------------------
scatterers = np.ones((Nx, Ny), dtype=bool)  # start fully blocked

# Define S-shaped channel
channel_width = 20  # width of the S channel
x = np.arange(Nx)
# Use sinusoidal function for center of S
y_center = Ny//2 + 20*np.sin(2*np.pi*x/Nx)

for i in range(Nx):
    y_start = int(max(0, y_center[i] - channel_width//2))
    y_end = int(min(Ny, y_center[i] + channel_width//2))
    scatterers[i, y_start:y_end] = False  # clear channel

# Effective wave speed array (slower inside scatterers)
scatter_strength = 0.9
c_grid = c * np.ones((Nx, Ny))
c_grid[scatterers] *= (1 - scatter_strength)

# ------------------------
# Wave fields
# ------------------------
E = np.zeros((Nx, Ny))
E_prev = np.zeros((Nx, Ny))
E_next = np.zeros((Nx, Ny))

# ------------------------
# FDTD step function
# ------------------------
def step(E, E_prev, c_grid):
    lap = np.zeros_like(E)
    lap[1:-1,1:-1] = (E[2:,1:-1] + E[:-2,1:-1] +
                       E[1:-1,2:] + E[1:-1,:-2] - 4*E[1:-1,1:-1]) / dx**2
    E_next = 2*E - E_prev + (c_grid*dt)**2 * lap
    return E_next

# ------------------------
# Visualization setup
# ------------------------
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(np.zeros((Nx,Ny)), origin='lower', cmap='inferno', vmin=0, vmax=0.1)
scatter_plot = ax.scatter(np.where(scatterers)[1], np.where(scatterers)[0],
                          color='cyan', s=10, label='scatterers')
ax.set_title("Wave propagation in S-shaped channel")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc='upper right')

# ------------------------
# Animation function
# ------------------------
def update(frame):
    global E, E_prev, E_next
    # inject source (continuous oscillation)
    E[x0, y0] += np.sin(2*np.pi*0.05*frame)
    # propagate
    E_next = step(E, E_prev, c_grid)
    E_prev = E.copy()
    E = E_next.copy()
    # update intensity plot
    im.set_data(E**2)
    return [im, scatter_plot]

# ------------------------
# Run animation and save MP4
# ------------------------
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
writer = FFMpegWriter(fps=20, bitrate=1800)
ani.save("wave_S_channel.mp4", writer=writer)

plt.close(fig)
print("Video generated: wave_S_channel.mp4")
