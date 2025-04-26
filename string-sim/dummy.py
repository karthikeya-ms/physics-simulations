import pygame
import numpy as np
import pygame.surfarray

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Wave Simulation")

# Simulation parameters
NX, NY = 100, 75               # Grid resolution
c = 200.0                      # Wave speed
DT = 0.02                      # Time step
DX = WIDTH / NX                # Spatial step in x
DY = HEIGHT / NY               # Spatial step in y
a = 1.0

# Stability check (CFL condition)
CFLx = c * DT / DX
CFLy = c * DT / DY
assert max(CFLx, CFLy) <= 1.0, f"CFL condition violated: CFLx={CFLx:.2f}, CFLy={CFLy:.2f}"

# State arrays: u at current, previous, and next time steps
u = np.zeros((NX, NY), dtype=float)
u_prev = np.zeros_like(u)
u_next = np.zeros_like(u)

# Initial condition: Gaussian bump in center
sigma = min(NX, NY) / 10
for i in range(NX):
    for j in range(NY):
        u[i, j] = np.exp(-(((i - NX/2)**2 + (j - NY/2)**2) / (2 * sigma**2)))
u_prev[:, :] = u[:, :]

# Main loop
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # PDE update: finite-difference scheme for 2D wave equation
    for i in range(1, NX - 1):
        for j in range(1, NY - 1):
            u_next[i, j] = (
                2*u[i, j] - u_prev[i, j]
                + (c*DT/DX)**2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
                + (c*DT/DY)**2 * (u[i, j+1] - 2*u[i, j] + u[i, j-1])
                -a*(u[i, j] - u_prev[i, j])/DT
            )
    # Dirichlet boundary conditions (edges fixed at zero)
    u_next[0, :] = 0
    u_next[-1, :] = 0
    u_next[:, 0] = 0
    u_next[:, -1] = 0

    # Rotate time levels
    u_prev[:, :], u[:, :] = u[:, :], u_next[:, :]

    # Rendering: map u values to grayscale
    max_u = np.max(np.abs(u))
    if max_u == 0:
        norm = np.zeros_like(u)
    else:
        norm = u / max_u
    # Scale to [0,255]
    gray = ((norm * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb = np.stack((gray,)*3, axis=-1)

    # Create surface and display
    surf = pygame.Surface((NX, NY))
    pygame.surfarray.blit_array(surf, rgb)
    surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
    WINDOW.blit(surf, (0, 0))
    pygame.display.flip()

pygame.quit()
