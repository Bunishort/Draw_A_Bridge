from context import sample
import numpy as np
import matplotlib.pyplot as plt

k = 6
nx=k*7
ny=k*9

lx = k*5
# ly = k*5
ly = 10

theta = 45 * np.pi / 180
lm=1.5

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2
x *= lm
y *= lm

max_res = 1e-6

gridY,gridX = np.meshgrid(y,x)
cos = np.cos(theta)
sin = np.sin(theta)
gridx = cos * gridX + sin * gridY
gridy = -sin * gridX + cos * gridY

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True

max_iter=1000*10
E=1
nu = 0.4
pX = 0.0
pY = 0.01
px = cos * pX - sin * pY
py = sin * pX + cos * pY

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
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

# plt.figure()
# plt.plot(np.sum(sxy_x,axis=1))
#
# plt.figure()
# ux_ref = lm*(x - np.min(gridx[solid])) * (1 - nu **2 ) / E * px
# plt.plot(lm*x, ux_ref,'k')
# plt.plot(lm*x, test.ux[:,int(nx/2)])

uX = []
uY = []
for xi in x:
    isok = np.bitwise_and( gridx < (xi +lm), gridx >= (xi - lm))
    uX.append(np.sum(test.ux[isok] * cos + test.uy[isok] * sin) / np.sum(test.solid[isok].astype(float)))
    uY.append(np.sum(-test.ux[isok] * sin + test.uy[isok] * cos) / np.sum(test.solid[isok].astype(float)))

plt.figure()
plt.plot(x, uX)
plt.ylabel('Ux')
plt.xlabel('x')

plt.figure()
plt.plot(x, uY)
plt.ylabel('Uy')
plt.xlabel('x')

plt.show()

1+1