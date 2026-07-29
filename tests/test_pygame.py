import pygame
import numpy as np
from context import sample
from line_profiler import profile
import moderngl

# simulation parameters
E=1
nu = 0.4
nx = 3*120
ny = 3*120
# nx = 80
# ny = 80
# nx = 30
# ny = 30

lm = 4.5 * 7/nx *2

vol_mass = 0.5
dt = 0.3 /1.7/3
ratio = 0.9  # must be between 0 and 1
tau = 20
damping = 0.05 /10

nbstep = int(30) # nb of steps per frame

fx = 0.001*lm/10
fy = 0.0*lm /10
f_attract_const = 1
max_stress = 0.02 /10

c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu)))
c_s = np.sqrt(E / ratio /  (2 * (1 + nu)) / vol_mass)

print( 'Max Sound speed * dt / lm ')
print( 'Compression : ' + str(c_p * dt / lm))
print( 'Shear: ' + str(c_s * dt / lm))

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)

solid = np.zeros([nx,ny],dtype = bool)
ix = int(nx*7/10)
iy = int(ny*1/10)
solid[ix:(ix+2),iy:(iy+2)] = True
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[ix:(ix+2),iy:(iy+2)] = 0
ix = int(nx*7/10)
iy = int(ny*9/10)
solid[ix:(ix+2),iy:(iy+2)] = True
ux_imp[ix:(ix+2),iy:(iy+2)] = 0
# ix = int(nx*7/10)
# iy = int(ny*5/10)
# solid[ix:(ix+2),iy:(iy+2)] = True
# ux_imp[ix:(ix+2),iy:(iy+2)] = 0

uy_imp = ux_imp.copy()
fx_imp = np.ones(solid.shape) * fx
fy_imp = np.ones(solid.shape) * fy
screen_size=(800,800)

# --- L'INTERFACE PYGAME ---
def main():

    pygame.init()

    pygame.display.set_mode(screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    # Solver init
    solver = sample.core.ElasticProblem(solid, elas_lambda, elas_mu, lm, ux_imp, uy_imp,
                                      is_explicit=True, vol_mass=vol_mass, dt = dt, ratio=ratio, tau=tau,
                                        fx_imp=fx_imp, fy_imp = fy_imp, damping = damping, gl_context=ctx)

    game = sample.interface.SimulationApp(solver, ctx, screen_size=screen_size, nbstep=nbstep, max_stress=max_stress)
    game.run()

    pygame.quit()


if __name__ == "__main__":
    main()