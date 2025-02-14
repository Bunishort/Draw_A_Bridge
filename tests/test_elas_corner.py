from context import sample
import numpy as np
import matplotlib.pyplot as plt

k = 2
nx=k*9
ny=k*9

lx = k*5
ly = k*5
lymax=k*7

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

gridy,gridx = np.meshgrid(y,x)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True
solid[np.bitwise_and(np.bitwise_and(
    gridx<=0,
    np.abs(gridy)<=lymax/2),
    np.abs(gridx)<=lx/2)] = True



max_iter=2000
E=1
nu = 0.3
elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
lm=1
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[gridx < (-lx/2+lm)] =0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
px_bound[gridx > (lx/2-lm)] = 0.01
py_bound = np.zeros(solid.shape)

test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,max_iter=max_iter)

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

bx, by = test.calc_b()
a_u_x, a_u_y = test.calc_a_u(test.ux, test.uy)
resx2 = bx - a_u_x
resy2 = by - a_u_y

sxx_x,sxy_x,syy_y,sxy_y = test.calc_stress(test.ux, test.uy)

plt.figure()
plt.plot(np.sum(sxx_x,axis=1))
plt.show()

1+1