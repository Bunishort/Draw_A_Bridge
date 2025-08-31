from context import sample
import numpy as np

from sample.core import get_frontier, calc_normal

nx=7
ny=9

lx = 5
ly = 5

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

gridy,gridx = np.meshgrid(y,x)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True
frontier,bulk,corn=get_frontier(solid)
nx,ny = calc_normal(solid)


E=1
nu = 0.3
elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
lm=1
ux_imp=gridx
ux_imp[bulk] = np.nan
uy_imp=np.zeros(solid.shape)
uy_imp[bulk] = np.nan
uy_imp[ny**2==1] = np.nan
ux_imp[ny**2==1] = np.nan

px_bound = np.zeros(solid.shape)
py_bound = np.zeros(solid.shape)

test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound)

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

bx, by = test.calc_b()
a_u_x, a_u_y = test.calc_a_u(test.ux, test.uy)
resx2 = bx - a_u_x
resy2 = by - a_u_y

1+1