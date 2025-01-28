from context import sample
import numpy as np
import matplotlib.pyplot as plt

nx=25
ny=25
L = nx
lm= L/nx

ri = 5*lm#internal radius
ro = 10*lm#external radius
pi = 0.1

max_iter=2000
E=1
nu = 0.3
elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

gridy,gridx = np.meshgrid(y,x)
gridy = gridy*lm
gridx = gridx*lm
r = np.sqrt(gridx**2+gridy**2)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(r>=ri,r<=ro)]= True
solid = sample.core.remove_single_points(solid)

frontier,bulk = sample.core.get_frontier(solid)
frontier_ind = np.where(frontier)
ux_imp=np.zeros(solid.shape)*np.nan
ux_imp[frontier_ind[0][0],frontier_ind[1][0]] = 0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
px_bound[r <= ri+lm] = pi
py_bound = px_bound

test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,max_iter=max_iter)

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

sxx, syy, sxy = test.calc_stress(test.ux, test.uy)


r = np.arange(5*nx) - (5*nx-1)/2
sig_circ = pi*ri**2 / (ro**2-ri**2) + pi*ri**2*ro**2 / r**2 / (ro**2-ri**2)
sig_rad =pi*ri**2 / (ro**2-ri**2) - pi*ri**2*ro**2 / r**2 / (ro**2-ri**2)

line = gridy == np.min(gridy[gridy>=0])
sigc_num = syy[line]
sigr_num = sxx[line]

plt.figure()
plt.title('Circumferential stress')
plt.plot(r,sig_circ,color='k')
plt.plot(gridx[line],sigc_num)

plt.figure()
plt.title('Radial stress')
plt.plot(r,sig_rad,color='k')
plt.plot(gridx[line],sigr_num)

plt.show()
1+1