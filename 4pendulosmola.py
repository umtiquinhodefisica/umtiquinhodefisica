import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ================================================================
#   PENDULUM WITH ELASTIC ROD (4 VIEWS)
# ================================================================

# --- PARAMETERS ---
m = 1.0         # mass (kg)
k = 35.0        # spring constant (N/m)
L0 = 1.0        # natural length (m)
g = 9.81        # gravity (m/s²)

dt = 0.004      # time step (s)
T = 20.0        # total time (s)
Nt = int(T / dt)
fps = 20
frames_idx = np.linspace(0, Nt-1, int(fps*T)).astype(int)

# --- DIFFERENT INITIAL CONDITIONS (4 PENDULUMS) ---
init_conditions = [
    (1.6*L0, np.deg2rad(80)),  # pendulum 1
    (1.3*L0, np.deg2rad(60)),  # pendulum 2
    (1.7*L0, np.deg2rad(90)), # pendulum 3
    (1.45*L0, np.deg2rad(40))   # pendulum 4
]

def spring_force(pos):
    x, y = pos
    r = np.hypot(x, y)
    if r == 0:
        return np.zeros(2)
    Fm = -k * (r - L0)
    return Fm * np.array([x/r, y/r])

def total_acceleration(pos):
    return (spring_force(pos) + np.array([0, -m*g])) / m

def integrate_pendulum(r0, theta0):
    """Integrates pendulum motion using Velocity-Verlet"""
    x0, y0 = r0 * np.sin(theta0), -r0 * np.cos(theta0)
    vx0, vy0 = 0.0, 0.0
    pos = np.zeros((Nt,2))
    vel = np.zeros((Nt,2))
    pos[0] = [x0, y0]
    vel[0] = [vx0, vy0]
    a = total_acceleration(pos[0])
    for i in range(1, Nt):
        pos[i] = pos[i-1] + vel[i-1]*dt + 0.5*a*(dt**2)
        a_new = total_acceleration(pos[i])
        vel[i] = vel[i-1] + 0.5*(a + a_new)*dt
        a = a_new
    return pos

# Integrate all pendulums
positions = [integrate_pendulum(r0, theta0) for r0, theta0 in init_conditions]

# ================================================================
#   ANIMATION SETUP
# ================================================================
fig, axes = plt.subplots(2,2, figsize=(10,10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add single title for the figure
fig.suptitle('Simple Pendulum with an Elastic Spring Rod: Four Distinct Initial Conditions', fontsize=18, y=0.95)

margin = 1.36 * (L0 + 0.6)
n_coils = 18
amp_factor = 0.06

def spring_coords(p):
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

# Create plots and elements
plots = []
for ax, pos in zip(axes.flat, positions):
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    # pivot
    ax.plot([0],[0],'ko')
    # gravity arrow
    arrow_x, arrow_y = -1.35*L0, 1.4*L0
    ax.arrow(arrow_x, arrow_y, 0, -0.4*L0, head_width=0.08, head_length=0.12,
             fc='red', ec='red', lw=2)
    ax.text(arrow_x-0.25, arrow_y-0.15*L0, 'g', color='red', fontsize=13)
    # dynamic elements
    spring_line, = ax.plot([],[], lw=2)
    mass_patch = patches.Circle((0,0), radius=0.12*L0, fc='orange')
    ax.add_patch(mass_patch)
    trail_line, = ax.plot([],[], lw=1, color='orange', alpha=0.7)
    plots.append((spring_line, mass_patch, trail_line))

def init():
    artists = []
    for (spring_line, mass_patch, trail_line), pos in zip(plots, positions):
        spring_line.set_data([],[])
        trail_line.set_data([],[])
        mass_patch.center = (pos[0,0], pos[0,1])
        artists += [spring_line, mass_patch, trail_line]
    return artists

def update(frame):
    artists = []
    for (spring_line, mass_patch, trail_line), pos in zip(plots, positions):
        p = pos[frame]
        Xs, Ys = spring_coords(p)
        spring_line.set_data(Xs, Ys)
        mass_patch.center = (p[0], p[1])
        trail_line.set_data(pos[:frame,0], pos[:frame,1])
        artists += [spring_line, mass_patch, trail_line]
    return artists

anim = FuncAnimation(fig, update, frames=frames_idx, init_func=init, blit=True)
output = '4pendulum.mp4'
writer = FFMpegWriter(fps=fps, bitrate=2000)
anim.save(output, writer=writer)
plt.close(fig)

