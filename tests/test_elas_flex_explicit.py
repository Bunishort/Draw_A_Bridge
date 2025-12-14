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
kplot = 10

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

uyt = [0,]
itet = [0,]

xplot = (np.arange(kplot * nx) - (kplot * nx-1)/2) / kplot
yplot = (np.arange(kplot * ny) - (kplot * ny-1)/2) / kplot
gridyplot,gridxplot = np.meshgrid(yplot,xplot)
solidplot = interpn((x, y), solid, (gridxplot, gridyplot), method='nearest', bounds_error=False, fill_value=False)
solidplot=solidplot.astype(bool)
solidplot_norm=  interpn((x, y), solid, (gridxplot, gridyplot), method='linear', bounds_error=False, fill_value=0)
solidplot_norm[solidplot_norm == 0] = 0.001
solidplot_def = solidplot.copy()
smooth_filter = np.array([[1,1,1],[1,0,1], [1,1,1]]) /8
# maybe custom made interpn using thiese filters could be faster
# filter_plot_small = np.ones((int(kplot), int(kplot)))
# filter_plot_big = np.ones((int(2*kplot -1), int(2*kplot -1)))
# starting by setting u only on known points... and not forgetting norm

fig,ax = plt.subplots(1,1)
t = ax.text(-0.1,0,'0')
im = ax.imshow(0* gridxplot,vmin = -0.003, vmax = 0.003)
for i in range(0,nstep):
    test.explicit_step()
    if np.mod(i,iplot) ==0:
        Z = np.zeros(gridxplot.shape)

        # Interpolate u on big grid
        ux_plot = interpn((x, y), test.ux, (gridxplot, gridyplot), method='linear', bounds_error=False, fill_value=0) / solidplot_norm
        uy_plot = interpn((x, y), test.uy, (gridxplot, gridyplot), method='linear', bounds_error=False, fill_value=0) / solidplot_norm
        out_plot = interpn((x, y+0.5), test.sxy_y_old, (gridxplot, gridyplot), method='linear', bounds_error=False, fill_value=0) / solidplot_norm
        #y+0.5 because stress are not computed on the same grid as displacements !

        # interpolate solid position with displacement
        gridxx = gridxplot[solidplot] + ux_plot[solidplot]
        gridyy = gridyplot[solidplot] + uy_plot[solidplot]
        xi_solid = interpn((xplot, yplot), gridxplot, (gridxx, gridyy), method='nearest')
        yi_solid = interpn((xplot, yplot), gridyplot, (gridxx, gridyy), method='nearest')
        xi_solid *= kplot
        yi_solid *= kplot
        xi_solid += (kplot * nx - 1) / 2
        yi_solid += (kplot * ny - 1) / 2
        xi_solid = xi_solid.astype(int)
        yi_solid = yi_solid.astype(int)

        solidplot_def[:] = False
        solidplot_def[xi_solid,yi_solid] = True # ! the same value may appear more than once in xi_solid,yi_solid
        Z[xi_solid,yi_solid] = out_plot[solidplot]
        Zsmooth = sample.core.conv(Z, smooth_filter)
        Z[np.bitwise_not(solidplot_def)] = Zsmooth[np.bitwise_not(solidplot_def)]
        #               (gridxplot, gridyplot), method='linear', rescale=False)
        #solution 1 : extrapoler ux partout pour avoir un truc cohérent, continu, bijectif
        #solution 2 : calculer la distance à chaque point, mettre à 0 les points trop loins
        #solution 3 : solution 1 mais uniquement sur le contour do solide, où l'on met des nan pour que derrière ca soit bien interpolé
        #solution 4 : faire en sorte que ux et uy soient des multiples de lm puis... ??? Magic ?
        #solution 5 : for i,j in gridxx, gridyy, find minimum distance on the regular grid
        # opencv ? scipy image ?

        im.set_array(Z)
        # im.set_array(test.uy)
        t.set_text(str(i))
        plt.pause(1/100)
        uyt.append(test.uy[ixmax,int(nx/2)])
        itet.append(i)

plt.title('Uy')
sxx_x,sxy_x,syy_y,sxy_y = test.calc_stress(test.ux, test.uy)
uyv = test.uy.copy()

plt.figure()
plt.imshow(out_plot*solidplot)
plt.title('out_plot*solidplot')

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