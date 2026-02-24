import pygame
import numpy as np
from context import sample
from line_profiler import profile

# simulation parameters
E=1
nu = 0.4
# nx=100
# ny=100
nx = 80
ny = 80

lm = 4.5 * 7/nx

vol_mass = 0.5
dt = 0.3 / 7
ratio = 0.2  # must be between 0 and 1
tau = 3

nbstep = 30 # nb of steps per frame

fx = 0.001*lm/10
fy = 0.0*lm /10
f_attract_const = 1


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
ix = int(nx/10)
iy = int(ny/10)
solid[ix:(ix+2),iy:(iy+2)] = True
ux_imp[ix:(ix+2),iy:(iy+2)] = 0

uy_imp = ux_imp.copy()
fx_imp = np.ones(solid.shape) * fx
fy_imp = np.ones(solid.shape) * fy

# --- L'INTERFACE PYGAME ---
def main():

    pygame.init()
    # Solver init
    solver = sample.core.ElasticProblem(solid, elas_lambda, elas_mu, lm, ux_imp, uy_imp,
                                      is_explicit=True, vol_mass=vol_mass, dt = dt, ratio=ratio, tau=tau,
                                        fx_imp=fx_imp, fy_imp = fy_imp)

    game = sample.interface.SimulationApp(solver,screen_size=(800,800), nbstep=nbstep)
    game.run()

    pygame.quit()


if __name__ == "__main__":
    main()