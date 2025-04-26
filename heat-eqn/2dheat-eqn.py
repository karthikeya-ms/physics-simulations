import pygame
import numpy as np
import matplotlib.pyplot as plt

# Initialize
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive 2D Heat Equation")

# Physical parameters (water @20°C example)
NX, NY = 100, 75          # grid
DX, DY = WIDTH/NX, HEIGHT/NY
DT = 0.0008
alpha =  5000.0    # instead of 1.43e-7
          # thermal diffusivity [m2/s]
rho, cp = 1000, 4186      # density [kg/m3], heat capacity [J/(kg·K)]
So_norm = 1.0             # normalized source strength

# Stability check
CFL = alpha*DT*(1/DX**2 + 1/DY**2)
print('CFL', CFL)
assert CFL <= 1.0, f"CFL violation: {CFL:.3f}"

# State arrays
u      = np.zeros((NX, NY), dtype=float)   # temperature at n
u_next = np.zeros_like(u)
S      = np.zeros_like(u)                  # source term

# Colormap
cmap = plt.get_cmap('jet')

clock = pygame.time.Clock()
running = True

while running:
    clock.tick(1000)

    # --- Event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Paint heat while mouse-button-0 is held ---
    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()
        i = min(int(mx//DX), NX-2)
        j = min(int(my//DY), NY-2)
        # in the paint section:
        u[i,j] = 1.0


    # --- PDE update: explicit 2D heat equation ---
    for i in range(1, NX-1):
        for j in range(1, NY-1):
            lap = ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / DX**2
                 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / DY**2)
            u_next[i, j] = u[i, j] + alpha*DT*lap + DT*S[i, j]/(rho*cp)

    # Zero out source and enforce Dirichlet BCs
    S[:, :] = 0
    u_next[0, : ] = 0
    u_next[-1, :] = 0
    u_next[:, 0 ] = 0
    u_next[:, -1] = 0

    # Rotate
    u, u_next = u_next, u

    # --- Rendering ---
    # Normalize to [0,1], protect against u.max()==0
    umax = u.max()
    norm = u/umax if umax>0 else u  
    # Get RGBA from jet, drop alpha
    rgba = cmap(norm)              # shape (NX,NY,4)
    rgb  = (rgba[:, :, :3]*255).astype(np.uint8)  # (NX,NY,3)
    # Make surface and scale to window
    surf = pygame.surfarray.make_surface(rgb)
    surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

pygame.quit()


