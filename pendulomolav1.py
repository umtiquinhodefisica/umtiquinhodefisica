import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ================================================================
#   PENDULUM WITH ELASTIC ROD (Velocity-Verlet)
# ================================================================

# --- Physical parameters ---
m = 1.0        # mass (kg)
k = 35.0       # spring constant (N/m)
L0 = 1.0       # natural length of the spring (m)
g = 9.81       # gravity (m/s²)

# --- Numerical parameters ---
dt = 0.003     # time step (s)
T = 16.0       # total simulation time (s)
Nt = int(T / dt)

# --- Initial conditions ---
r0 = L0 * 1.4                      # initial length (m)
theta0 = np.deg2rad(50.0)          # initial angle (rad)
x0 = r0 * np.sin(theta0)
y0 = -r0 * np.cos(theta0)
vx0, vy0 = 0.0, 0.0

# --- Force functions ---
def spring_force(pos):
    """Elastic spring force with pivot at (0,0)."""
    x, y = pos
    r = np.hypot(x, y)
    if r == 0:
        return np.zeros(2)
    Fm = -k * (r - L0)
    return Fm * np.array([x/r, y/r])

def total_acceleration(pos):
    """Total acceleration = (F_spring + F_gravity) / m"""
    F_spring = spring_force(pos)
    F_grav = np.array([0, -m * g])
    return (F_spring + F_grav) / m

# --- Integration (Velocity-Verlet) ---
pos = np.zeros((Nt, 2))
vel = np.zeros((Nt, 2))
pos[0] = [x0, y0]
vel[0] = [vx0, vy0]
a = total_acceleration(pos[0])

for i in range(1, Nt):
    pos[i] = pos[i-1] + vel[i-1]*dt + 0.5*a*(dt**2)
    a_new = total_acceleration(pos[i])
    vel[i] = vel[i-1] + 0.5*(a + a_new)*dt
    a = a_new

# ================================================================
#   ANIMATION WITH MATPLOTLIB
# ================================================================

def spring_coords(p, n_coils=18, amp_factor=0.06):
    """Generate (Xs,Ys) coordinates to draw the deformed spring."""
    x, y = p
    L = np.hypot(x, y)
    if L == 0:
        return np.array([0]), np.array([0])
    ux, uy = x/L, y/L
    px, py = -uy, ux
    s = np.linspace(0, 1, n_coils*4)
    amplitude = amp_factor * max(L, L0)
    wav = np.sin(2 * np.pi * n_coils * s)
    Xs = s*x + amplitude * wav * px
    Ys = s*y + amplitude * wav * py
    return Xs, Ys

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal', 'box')
margin = 1.3 * (L0 + 0.6)
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Pendulum with Elastic Rod', fontsize=15)

# Graphic elements
spring_line, = ax.plot([], [], lw=2)
mass_patch = patches.Circle((0, 0), radius=0.06*L0, fc='C0')
ax.add_patch(mass_patch)
pivot_dot, = ax.plot([0], [0], 'ko')
trail_line, = ax.plot([], [], lw=1, color='orange', alpha=0.8)

# Gravity vector (arrow in the upper-left corner)
arrow_x, arrow_y = -1.05 * L0, 1.05 * L0
ax.arrow(arrow_x, arrow_y, 0, -0.4*L0,
         head_width=0.08, head_length=0.12,
         fc='red', ec='red', lw=2)
ax.text(arrow_x - 0.17, arrow_y - 0.25*L0, 'g', color='red', fontsize=13)

def init():
    spring_line.set_data([], [])
    trail_line.set_data([], [])
    mass_patch.center = (pos[0,0], pos[0,1])
    return spring_line, mass_patch, pivot_dot, trail_line

def update(frame):
    p = pos[frame]
    Xs, Ys = spring_coords(p)
    spring_line.set_data(Xs, Ys)
    mass_patch.center = (p[0], p[1])
    # keep the entire trajectory up to the current frame
    trail_line.set_data(pos[:frame,0], pos[:frame,1])
    return spring_line, mass_patch, pivot_dot, trail_line

# --- Create animation ---
fps = 24
frames_idx = np.linspace(0, Nt-1, int(fps*T)).astype(int)
anim = FuncAnimation(fig, update, frames=frames_idx, init_func=init, blit=True)

# --- Save as MP4 ---
output = 'pendulo_mola1.mp4'
writer = FFMpegWriter(fps=fps, bitrate=1800)
anim.save(output, writer=writer)
plt.close(fig)
