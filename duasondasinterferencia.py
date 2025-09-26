import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -----------------------
# PARAMETERS
# -----------------------
L = 10.0
Nx = 500
x = np.linspace(0, L, Nx)

k = 2*np.pi / L
w = 2*np.pi * 1.0
v = w / k  # velocidade da onda

phi = np.pi / 3  # diferença de fase (qualquer valor)

n_frames = 150
dt = 0.007

# -----------------------
# FIGURE SETUP
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
line1, = axes[0].plot(x, np.zeros_like(x), lw=2, label='Wave →')
line2, = axes[0].plot(x, np.zeros_like(x), lw=2, label='Wave ←')
line_sum, = axes[1].plot(x, np.zeros_like(x), color='red', lw=2, label='Sum')

axes[0].set_title('Individual Waves')
axes[1].set_title('Sum of Waves')
axes[0].set_ylim(-2.2, 2.2)
axes[1].set_ylim(-2.2, 2.2)
axes[0].grid(True)
axes[1].grid(True)
axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')
axes[0].set_xlabel('x')
axes[1].set_xlabel('x')
axes[0].set_ylabel('Amplitude')

plt.tight_layout()

# -----------------------
# UPDATE FUNCTION
# -----------------------
def update(frame):
    t = frame * dt
    y1 = np.sin(k*(x - v*t))
    y2 = np.sin(k*(x + v*t) + phi)
    ysum = y1 + y2

    line1.set_ydata(y1)
    line2.set_ydata(y2)
    line_sum.set_ydata(ysum)
    return line1, line2, line_sum

# -----------------------
# ANIMATION
# -----------------------
anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
writer = FFMpegWriter(fps=10, bitrate=1800)
anim.save("222.mp4", writer=writer)

print("Vídeo salvo em: 222.mp4")
