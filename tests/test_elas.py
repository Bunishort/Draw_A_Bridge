from context import sample
import numpy as np

nx=7
ny=9

lx = 3
ly = 3

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

gridy,gridx = np.meshgrid(y,x)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True

elas_lambda = 0*1
elas_mu = 1
lm=1
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[gridx < (-lx/2+lm)] =0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
px_bound[gridx > (lx/2-lm)] = 0.01
py_bound = np.zeros(solid.shape)

test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound)

print("axx")
print(test.axx)
print("axy")
print(test.axy)
print("ayy")
print(test.ayy)
print("ayx")
print(test.ayx)
n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

bx, by = test.calc_b()
a_u_x, a_u_y = test.calc_a_u(test.ux, test.uy)
resx2 = bx - a_u_x
resy2 = by - a_u_y

1+1