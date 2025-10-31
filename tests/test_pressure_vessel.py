from context import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.interpolate import interpn

import timeit
start = timeit.default_timer()

nx=64
ny=64
L = 25
lm= L/nx
max_res = 1e-6

ri = 5#internal radius
ro = 10#external radius
pi = 0.1

max_iter=5*2000
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
#solid = sample.core.remove_single_points(solid)

frontier,bulk = sample.core.get_frontier(solid)
nnx,nny = sample.core.calc_normal(solid)

frontier_ind = np.where(frontier)
ux_imp=np.zeros(solid.shape)*np.nan
ux_imp[frontier_ind[0][0],frontier_ind[1][0]] = 0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
py_bound = np.zeros(solid.shape)
px_bound[np.bitwise_and(r <= ri+lm,nnx<0)] = pi
px_bound[np.bitwise_and(r <= ri+lm,nnx>0)] = -pi
py_bound[np.bitwise_and(r <= ri+lm,nny<0)] = pi
py_bound[np.bitwise_and(r <= ri+lm,nny>0)] = -pi


test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,max_iter=max_iter, max_res=max_res)

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()

sxx_x,sxy_x,syy_y,sxy_y = test.calc_stress(test.ux, test.uy)

stop = timeit.default_timer()
print('Time: ', stop - start)

r = (np.arange(5*nx) - (5*nx-1)/2)/(5*nx) *L
A = pi*ri**2 / (ro**2-ri**2)
B = pi*ri**2*ro**2 / (ro**2-ri**2)
sig_circ = A + B/ r**2
sig_rad =A - B / r**2
u_rad = (1 + nu) / E * ( (1 - 2 * nu) * A * r + B / r )

sig_circ[np.abs(r)<ri] = np.nan
sig_rad[np.abs(r)<ri] = np.nan
sig_circ[np.abs(r)>ro] = np.nan
sig_rad[np.abs(r)>ro] = np.nan
u_rad[np.abs(r)<ri] = np.nan
u_rad[np.abs(r)>ro] = np.nan

line = gridy == np.min(gridy[gridy>=0])
sigc_num = syy_y[line]
sigr_num = sxx_x[line]

figc = plt.figure()
plt.title('Circumferential stress')
plt.plot(r,sig_circ,color='k')
plt.plot(gridx[line],sigc_num,'--')

figr = plt.figure()
plt.title('Radial stress')
plt.plot(r,sig_rad,color='k')
plt.plot(gridx[line]+0.5*lm,sigr_num,'--')

figu = plt.figure()
plt.plot(r,np.abs(u_rad),color='k')
plt.title(' Displacement norm')

thetas = -np.array([0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
thetas_str = ['0', 'pi/12', 'pi/6', 'pi/4', 'pi/3', 'pi/2']

for theta,theta_str in zip(thetas,thetas_str):
    xt = r * np.cos(theta)
    yt = r * np.sin(theta)
    sxx,syy,sxy = sample.core.interp_stress(xt,yt,x*lm,y*lm,sxx_x,sxy_x,syy_y,sxy_y)
    c = np.cos(theta)
    s = np.sin(theta)
    sigr_t = sxx * c**2 + 2*c*s*sxy + s**2 * syy
    sigc_t = sxx * s**2 - 2*c*s*sxy + c**2 * syy
    un = (test.ux - np.mean(test.ux[bulk])) ** 2 + (test.uy - np.mean(test.uy[bulk])) ** 2
    un = np.sqrt(un)
    uni = interpn((x*lm,y*lm),un,(xt,yt),bounds_error=False)


    plt.figure(figc)
    plt.plot(r, sigc_t,label=theta_str)
    plt.figure(figr)
    plt.plot(r, sigr_t,label=theta_str)
    plt.figure(figu)
    plt.plot(r, uni,label=theta_str)


plt.legend()
plt.figure(figc)
plt.legend()
plt.figure(figr)
plt.legend()
print(n_iter)

plt.show()
0.00278# stress error &e-6, 0.00299 with 1e-4, 0.0036 with 1e-2 (but looks bad), 117 iter