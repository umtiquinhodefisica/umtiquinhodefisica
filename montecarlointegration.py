import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ------------------------
# Function and interval
# ------------------------
def f(x):
    return np.sin(x)

a, b = 0, np.pi
f_max = 1.0   # maximum of sin(x)

# ------------------------
# Monte Carlo parameters
# ------------------------
N_total = 1000   # total number of points to throw
batch_size = 5   # number of points per frame

# Arrays to store points
x_points, y_points = [], []
below_curve = []

# ------------------------
# Figure setup
# ------------------------
fig, ax = plt.subplots(figsize=(6,4))
x = np.linspace(a, b, 400)
ax.plot(x, f(x), 'r', lw=2, label="f(x) = sin(x)")
ax.set_xlim(a, b)
ax.set_ylim(0, f_max*1.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Monte Carlo Integration (balls method)")

# scatter for dynamic points
points_plot = ax.scatter([], [], s=20)

# explanatory text: top-left corner
est_text = ax.text(0.02, 1.08, '', transform=ax.transAxes, fontsize=10)

# ------------------------
# Animation update function
# ------------------------
def update(frame):
    global x_points, y_points, below_curve
    for _ in range(batch_size):
        if len(x_points) >= N_total:
            break
        x_rand = np.random.uniform(a, b)
        y_rand = np.random.uniform(0, f_max)
        x_points.append(x_rand)
        y_points.append(y_rand)
        below_curve.append(y_rand <= f(x_rand))

    # Update scatter plot
    colors = ['green' if b else 'blue' for b in below_curve]
    points_plot.set_offsets(np.c_[x_points, y_points])
    points_plot.set_color(colors)

    # Estimate integral
    integral_est = (b - a) * f_max * np.mean(below_curve)
    # Update text with explanatory phrase
    est_text.set_text(f'Points below curve (green): {np.sum(below_curve)}, Integral ~ {integral_est:.4f}')

    return points_plot, est_text

# ------------------------
# Create animation
# ------------------------
ani = FuncAnimation(fig, update, frames=int(N_total/batch_size)+5, interval=50, blit=False)

# ------------------------
# Save video
# ------------------------
writer = FFMpegWriter(fps=20, bitrate=1800)
ani.save("monte_carlo_balls.mp4", writer=writer)

plt.close(fig)
print("Video generated: monte_carlo_balls.mp4")
