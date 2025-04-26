import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
Nx, Ny = 50, 50      # Grid size
dx = dy = 1.0        # Spatial step
alpha = 10.0          # Diffusivity (toy value for visible diffusion)
dt = 0.01             # Time step (stable for alpha*dt*(2/dx^2) <= 1)

# Initial temperature field: zero everywhere except a hot spot in the center
u = np.zeros((Nx, Ny), dtype=float)
u[Nx//2, Ny//2] = 1.0

# Set up the figure and axis
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
fig.colorbar(im, ax=ax)
ax.set_title("2D Heat Equation")

# Update function for animation
def update(frame):
    global u
    u_new = u.copy()
    # Apply explicit finite-difference heat equation
    u_new[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + alpha * dt * (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
    )
    u = u_new
    im.set_data(u)
    return (im,)

# Create animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
