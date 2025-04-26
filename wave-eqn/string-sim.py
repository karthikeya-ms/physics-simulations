#ID wave equation

import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive String Simulator")

# Simulation parameters
NUM_POINTS = 100            # Number of points on the string
WAVE_SPEED = 20.0            # Speed of wave propagation
DT = 0.01             # Time step for simulation
a = 5.0

# Discretization of space
positions = np.linspace(100, WIDTH - 100, NUM_POINTS)
heights = np.full(NUM_POINTS, HEIGHT // 2, dtype=float)        # u^n
heights_previous = heights.copy()                             # u^{n-1}
DX = positions[1] - positions[0]                              # Δx
CFL = WAVE_SPEED * DT / DX
assert CFL <= 1.0, f"CFL condition violated (CFL={CFL:.2f}). Decrease DT or increase DX."

# Fixed endpoints
fixed = {0, NUM_POINTS - 1}

# Interaction state
selected_point = None

# Main loop setup
clock = pygame.time.Clock()
running = True

while running:
    clock.tick(60)  # Limit to 60 FPS

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            idx = np.argmin(np.abs(positions - mx)) #picking the least of |positions - mx|
            if abs(positions[idx] - mx) < 20:
                selected_point = idx
                print('selected_point', selected_point)
        elif event.type == pygame.MOUSEBUTTONUP:
            selected_point = None

    # --- Mouse Dragging ---
    if selected_point is not None:
        _, my = pygame.mouse.get_pos()
        print('my', my)
        heights[selected_point] = my
        heights_previous[selected_point] = my  # Keep previous consistent

    # --- Physics Update: 1D Wave PDE ---
    new_heights = heights.copy()
    for i in range(NUM_POINTS):
        if i in fixed:
            new_heights[i] = HEIGHT // 2
        else:
            new_heights[i] = (
                2 * heights[i]
                - heights_previous[i]
                + (WAVE_SPEED**2 * DT**2 / DX**2)
                  * (heights[i-1] - 2 * heights[i] + heights[i+1])
                  -a*(heights[i] - heights[i])/DT
            )
        
    heights_previous = heights.copy()
    heights = new_heights

    # --- Drawing ---
    WINDOW.fill((255, 255, 255))
    points = [(positions[i], heights[i]) for i in range(NUM_POINTS)]
    pygame.draw.lines(WINDOW, (0, 0, 0), False, points, 2)
    pygame.display.flip()

pygame.quit()
