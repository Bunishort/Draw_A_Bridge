from context import sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interpn
from scipy.ndimage import zoom
import cv2

k = 5
nx=k*7
ny=k*7

lx = k*5
ly = 2

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2
gridy,gridx = np.meshgrid(y,x)

max_res = 1e-6


solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True
ixmax = int(np.max(gridx[solid]))

# simulation parameter
max_iter=1000
E=1
nu = 0.4
py = 0.01 /10

lm = 4.5/k

vol_mass = 0.5
dt = 1 / 5
ratio = 0.2  # must be between 0 and 1
tau = 3 *3

precond = False
precond_type = 'robust'
precond_n = 7

c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu)))
c_s = np.sqrt(E / ratio /  (2 * (1 + nu)) / vol_mass)

print( 'Max Sound speed * dt / lm ')
print( 'Compression : ' + str(c_p * dt / lm))
print( 'Shear: ' + str(c_s * dt / lm))

nstep = 4000
iplot = 50
kplot = 3

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[gridx == np.min(gridx[solid])] =0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
py_bound = np.zeros(solid.shape)
# py_bound[gridx > (lx/2-lm)] = py
py_bound[gridx == np.max(gridx[solid])] = py

test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,max_iter=max_iter,max_res = max_res,
                                  is_explicit = True, vol_mass=vol_mass, dt = dt, ratio = ratio, tau = tau,
                                  precond_type = precond_type, precond_n = precond_n, precond = precond)
anim = sample.interface.ExplicitAnimation(test, nstep = nstep, plot_interval = iplot, upscale_factor = kplot,
                                          probe_fields = ['uy'], plot_field = 'sxy_y_old',
                                          probe_ix = [ixmax,], probe_iy = [int(nx/2),], y_dec = 0.5, min_scale = -0.003,
                                            max_scale = 0.003)
anim.animate()

uyt = anim.probe_vals['uy' + str(ixmax) + '_' + str(int(nx/2))]
itet = anim.iplot

plt.title('Uy')
sxx_x,sxy_x,syy_y,sxy_y = test.calc_stress(test.ux, test.uy)
uyv = test.uy.copy()
#
# plt.figure()
# plt.imshow(out_plot*solidplot)
# plt.title('out_plot*solidplot')

plt.figure()
plt.imshow(test.vy)
plt.title('Vy')

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()
sxx_xe,sxy_xe,syy_ye,sxy_ye = test.calc_stress(test.ux, test.uy)


plt.figure()
plt.title('shear stress')
plt.plot(np.sum(sxy_x,axis=1))
plt.plot(np.sum(sxy_xe,axis=1))


plt.figure()
ux_ref = lm*(x - np.min(gridx[solid])) * (1 - nu **2 ) / E * py
plt.plot(lm*x, ux_ref,'k')
plt.plot(lm*x, uyv[:,int(nx/2)], label="explicit")
plt.plot(lm*x, test.uy[:,int(nx/2)], label="implicit")
plt.legend()

plt.figure()
plt.title('Moment Mz')
plt.plot(gridx[:,0], np.sum(sxx_xe * gridy * lm,axis=1), label = "Nx explicit")

plt.plot(gridx[:,0], np.sum(sxx_x * gridy * lm,axis=1), label = "Nx implicit")
plt.plot(gridx[:,0], -np.sum(test.py_bound * test.frontier * lm)
         * (np.max(gridx[test.frontier]) - gridx ), label = "Ny", color="k")


plt.figure()
plt.plot(itet, uyt)
plt.xlabel('iteration')
plt.ylabel('Y displacement')

plt.show()


1+1