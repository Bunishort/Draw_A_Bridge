import pygame
import numpy as np
from context import sample

# simulation parameters
E=1
nu = 0.4
nx=100
ny=100
lm = 4.5 * 7/nx

vol_mass = 0.5
dt = 0.3 / 7
ratio = 0.2  # must be between 0 and 1
tau = 3

nbstep = 10 # nb of steps per frame

fx = 0.0*lm /10000
fy = 0.01*lm /10
f_attract_const = 1e-6


c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu)))
c_s = np.sqrt(E / ratio /  (2 * (1 + nu)) / vol_mass)

print( 'Max Sound speed * dt / lm ')
print( 'Compression : ' + str(c_p * dt / lm))
print( 'Shear: ' + str(c_s * dt / lm))

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)

solid = np.zeros([nx,ny],dtype = bool)
ix = int(nx/2)
iy = int(ny/10)
solid[ix:(ix+2),iy:(iy+2)] = True
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[ix:(ix+2),iy:(iy+2)] = 0
uy_imp = ux_imp.copy()
fx_imp = np.ones(solid.shape) * fx
fy_imp = np.ones(solid.shape) * fy

# --- L'INTERFACE PYGAME ---
def main():
    # 1. Configuration
    RES = (100, 100)  # Grid size
    SCALE = 5  # Display scaling factor

    pygame.init()
    window = pygame.display.set_mode((RES[0] * SCALE, RES[1] * SCALE))
    clock = pygame.time.Clock()

    # Solver init
    solver = sample.core.ElasticProblem(solid, elas_lambda, elas_mu, lm, ux_imp, uy_imp,
                                      is_explicit=True, vol_mass=vol_mass, dt = dt, ratio=ratio, tau=tau,
                                        fx_imp=fx_imp, fy_imp = fy_imp)

    # Plot init
    anim = sample.interface.ExplicitAnimation_pygame(solver, upscale_factor = SCALE, plot_field = 'solid')

    xtemp= np.arange(nx)
    ytemp = np.arange(ny)
    gridy, gridx = np.meshgrid(ytemp, xtemp)

    running = True
    mode_simulation = False
    while running:
        #  Inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Switch between draw mode and simulation mode
                    mode_simulation = not mode_simulation
                    if not mode_simulation:
                        print("Mode: Draw")
                    else:
                        print("Mode: Simulation")

        # Mouse Interaction Souris
        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        gx, gy = mx // SCALE, my // SCALE  # Conversion screeen coord-> grid FEM

        #Draw mode
        if not mode_simulation:
            if mouse_buttons[0]:  # Draw
                solver.mod_solid(gx, gy, 1)

            if mouse_buttons[2]:  # Erase
                solver.mod_solid(gx, gy, 0)

            # Black background
            render_array = np.zeros((RES[0], RES[1], 3), dtype=np.uint8)
            mask = solver.solid
            render_array[mask] = [20, 200, 20]
            # stress_display = (solver.ux / lm * 255).astype(np.uint8)
            # render_array[mask, 0] = stress_display[mask]

            surface = pygame.surfarray.make_surface(render_array)
            if SCALE > 1:
                surface = pygame.transform.scale(surface, (RES[0] * SCALE, RES[1] * SCALE))

        else:
            # B.explicit step
            for i in range(0, nbstep):
                solver.explicit_step()

            if mouse_buttons[0]:  # Attractor
                dx = gy - (gridx + solver.ux )
                dy = gx - (gridy + solver.uy )
                d = dx **2 + dy **2
                f_attract = f_attract_const / lm / (1 + d)
                fx_imp_live = fx_imp +  f_attract * dx / ( 1+ d )
                fy_imp_live = fy_imp + f_attract * dx / (1 + d)
                solver.update_f_imp(fx_imp_live, fy_imp_live)

            z = anim.calc_image()
            render_array = np.zeros((z.shape[0],z.shape[1], 3), dtype=np.uint8)
            render_array[:,:,1] = z / 6 / lm * 255
            # stress_display = (solver.ux / lm * 255).astype(np.uint8)
            # render_array[mask, 0] = stress_display[mask]

            surface = pygame.surfarray.make_surface(render_array)


        window.blit(surface, (0, 0))

        # Afficher les FPS
        a=clock.get_fps()
        clock.tick(60)
        pygame.display.set_caption(f"FEM Explicite - FPS: {a:.1f}")
        pygame.display.flip()


    pygame.quit()


if __name__ == "__main__":
    main()