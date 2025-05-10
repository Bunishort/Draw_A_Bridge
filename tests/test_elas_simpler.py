from context import sample
import numpy as np

from sample.core import get_frontier

nx=3*7
ny=3*9

lx = 3*5
ly = 3*5

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

gridy,gridx = np.meshgrid(y,x)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True
frontier,bulk=get_frontier(solid)

max_iter=2000
E=1
nu = 0.3
elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
lm=1
ux_imp=gridx
xmax_bulk = np.max(gridx[bulk])
xmin_bulk = np.min(gridx[bulk])
u_free= np.bitwise_and(gridx<=xmax_bulk,gridx>=xmin_bulk)
ux_imp[u_free] = np.nan
uy_imp=np.zeros(solid.shape)
uy_imp[u_free] = np.nan

px_bound = np.zeros(solid.shape)
py_bound = np.zeros(solid.shape)
fx_imp = np.zeros(solid.shape)
#fx_imp[int((nx-1)/2),int((ny-1)/2)] = 20



test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,fx_imp=fx_imp,max_iter=max_iter)

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

bx, by = test.calc_b()
a_u_x, a_u_y = test.calc_a_u(test.ux, test.uy)
resx2 = bx - a_u_x
resy2 = by - a_u_y


sxx_x, sxy_x, syy_y,sxy_y = test.calc_stress(test.ux, test.uy)


1+1