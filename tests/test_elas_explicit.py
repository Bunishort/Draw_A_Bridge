from context import sample
import numpy as np
import matplotlib.pyplot as plt

k = 3
nx=k*7
ny=k*7

lx = k*5
ly = k*5

x = np.arange(nx) - (nx-1)/2
y = np.arange(ny) - (ny-1)/2

max_res = 1e-6

gridy,gridx = np.meshgrid(y,x)

solid = np.zeros([nx,ny],dtype=bool)
solid[np.bitwise_and(np.abs(gridx)<=lx/2,
    np.abs(gridy)<=ly/2)] = True
ixmax = int(np.max(gridx[solid]))

# simulation parameter
max_iter=1000
E=1
nu = 0.4
px = 0.01

lm = 4.5/k

vol_mass = 0.5
dt = 1/3
ratio = 0.2  # must be between 0 and 1
tau = dt*10

c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu)))
c_s = np.sqrt(E / ratio /  (2 * (1 + nu)) / vol_mass)

print( 'Max Sound speed * dt / lm ')
print( 'Compression : ' + str(c_p * dt / lm))
print( 'Shear: ' + str(c_s * dt / lm))

nstep = 1000
iplot = 10

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[gridx < (-lx/2+lm)] =0
uy_imp=ux_imp

px_bound = np.zeros(solid.shape)
px_bound[gridx > (lx/2-lm)] = px
py_bound = np.zeros(solid.shape)

test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp,
                                  px_bound=px_bound,py_bound=py_bound,max_iter=max_iter,max_res = max_res,
                                  is_explicit = True, vol_mass=vol_mass, dt = dt, ratio = ratio, tau = tau)

uxt = [0,]
itet = [0,]
fig,ax = plt.subplots(1,1)
t = ax.text(-0.1,0,'0')
im = ax.imshow(test.ux,vmin = -0.1, vmax = 0.5)
for i in range(0,nstep):
    test.explicit_step()
    if np.mod(i,iplot) ==0:
        im.set_array(test.ux)
        t.set_text(str(i))
        plt.pause(1/1000)
        uxt.append(test.ux[ixmax,int(nx/2)])
        itet.append(i)

plt.title('Ux')
sxx_x,sxy_x,syy_y,sxy_y = test.calc_stress(test.ux, test.uy)
uxv = test.ux.copy()
plt.figure()
plt.imshow(test.vx)
plt.title('Vx')

n_iter,resx,resy,res_max_convergence,convergence_hist  = test.cg_loop()
sxx_xe,sxy_xe,syy_ye,sxy_ye = test.calc_stress(test.ux, test.uy)


plt.figure()
plt.plot(np.sum(sxx_x,axis=1))
plt.plot(np.sum(sxx_xe,axis=1))


plt.figure()
ux_ref = lm*(x - np.min(gridx[solid])) * (1 - nu **2 ) / E * px
plt.plot(lm*x, ux_ref,'k')
plt.plot(lm*x, uxv[:,int(nx/2)], label="explicit")
plt.plot(lm*x, test.ux[:,int(nx/2)], label="implicit")
plt.legend()


plt.figure()
plt.plot(itet, uxt)
plt.xlabel('iteration')
plt.ylabel('X displacement')

plt.show()


1+1