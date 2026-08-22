import pygame
import numpy as np
from context import sample
import moderngl

# simulation parameters
E=1 #Young modulus (higher value means stiffer)
nu = 0.4 #Poisson ratio (must be between 0 and 0.49, check Wikipedia for definition)
nx = 180 #Simulation grid size
ny = 180

lm = 4.5 * 7/nx *2 # Size of each element (= each simulation grid pixel)
# This formula does not make any sense, it was all trial and error

vol_mass = 0.5 #Volumic mass of the solid.
nbstep = int(45) # nb of simulation time steps per frame
# Higher value = faster response of the solid, but needs more computing power.
# Lower the value if FPS < 60

ratio = 0.9  # Viscoelastic parameter. Must be >0 and <=1. Higher value = less dissipation
tau = 20 # Viscoelastic characteristic time. Something around 1sec of real game time (=dt*nbstep*FPS) is OK.
damping = 0.05 /10 # Viscous damping. Trial and error : lower the value for more bouncy simulation

fx = 0.001*lm/10 #Gravity force amplitude. Trial and error
fy = 0.0*lm /10 # If you want gravity in the lateral direction for some reason
f_attract_const = 1e-2 # Attraction force magnitude when clicking during simulation
max_stress = 0.02 /10 # Color scale normalization factor. Higher value for more uniform blue


c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu))) #Formula for compression sound wave speed
c_s = np.sqrt(E / ratio /  (2 * (1 + nu)) / vol_mass) #Formula for shear sound wave speed

dt = 0.9 * lm / c_p # Size of a simulation time step. cp*dt/lm must be <1 to ensure simulation stability.
# This formula ensures stable simulation with close to the largest time step possible, to have the simulation feel as fast as possible

print( 'Max Sound speed * dt / lm ')
print( 'Compression : ' + str(c_p * dt / lm))
print( 'Shear: ' + str(c_s * dt / lm))

screen_size=(800,800) #window size in pixels

####################################################
#SOlid initialization
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


elas_lambda = E*nu /(1+nu)/(1-2*nu) # Elastic parameter. Do not touch
elas_mu = E/2/(1+nu)

# ---PYGAME interface---
def main():

    pygame.init()

    pygame.display.set_mode(screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    # Solver init
    solver = sample.core.ElasticProblem(solid, elas_lambda, elas_mu, lm, ux_imp, uy_imp,
                                      is_explicit=True, vol_mass=vol_mass, dt = dt, ratio=ratio, tau=tau,
                                        damping = damping, gl_context=ctx)

    game = sample.interface.SimulationApp(solver, ctx, screen_size=screen_size, nbstep=nbstep, max_stress=max_stress,
                                          fx_grav=fx, fy_grav=fy, f_attract_const = f_attract_const)
    game.run()

    pygame.quit()


if __name__ == "__main__":
    main()