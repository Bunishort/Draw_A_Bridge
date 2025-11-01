from context import sample
import numpy as np
import matplotlib.pyplot as plt

k = 6
nx=k*7
ny=k*9

lx = k*5
# ly = k*5
ly = 6

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

max_res = 1e-6

gridy,gridx = np.meshgrid(y,x)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True

max_iter=1000*10
E=1
nu = 0.4
px = 0.0
py = 0.01

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
lm=1.5
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[gridx < (-lx/2+lm)] =0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
px_bound[gridx > (lx/2-lm)] = px
py_bound = np.zeros(solid.shape)
py_bound[gridx > (lx/2-lm)] = py


test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,max_iter=max_iter,max_res = max_res,
                                  precond_type = 'robust', precond_n = 30)

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

bx, by = test.calc_b()
a_u_x, a_u_y = test.calc_a_u(test.ux, test.uy)
resx2 = bx - a_u_x
resy2 = by - a_u_y

sxx_x,sxy_x,syy_y,sxy_y = test.calc_stress(test.ux, test.uy)

plt.figure()
plt.plot(np.sum(sxy_x,axis=1))
#
# plt.figure()
# ux_ref = lm*(x - np.min(gridx[solid])) * (1 - nu **2 ) / E * px
# plt.plot(lm*x, ux_ref,'k')
# plt.plot(lm*x, test.ux[:,int(nx/2)])

plt.figure()
ux_ref = lm*(x - np.min(gridx[solid])) * (1 - nu **2 ) / E * px
plt.plot(lm*x, ux_ref,'k')
plt.plot(lm*x, np.sum(test.ux*test.bulk,1) / np.sum(test.bulk.astype(float),1 ))
plt.ylabel('Ux')
plt.xlabel('x')

plt.figure()
plt.plot(lm*x, np.sum(test.uy*test.bulk,1) / np.sum(test.bulk.astype(float),1 ))
plt.ylabel('Uy')
plt.xlabel('x')

plt.figure()
plt.plot(gridx[:,0], np.sum(sxx_x * gridy * lm,axis=1))
plt.plot(gridx[:,0], -np.sum(test.py_bound * test.frontier * lm)
         * (np.max(gridx[test.frontier]) - gridx ))

plt.show()

1+1