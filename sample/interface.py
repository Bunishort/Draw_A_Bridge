from sample.core import conv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interpn


class ExplicitAnimation:
    """
    TODO update comments
    :param test: ElasticProblem class object from sample.core, with is_explicit=True
    :**kwarg fy_imp : 2d matrix of imposed volumic forces in the y direction (default 0)
    :**kwarg max_iter : maximum number of Conjugate Gradient iterations, default 20
    :**kwarg max_res : maximum (non dimensional) residual convergence tolerance, default 1e-6
    kernel_type: plane_strain, plane_stress or 2daxi (only plane strain available now)
    axx: kernel _x_x for div(sigma)
    """
    def __init__(self,elas, **kwargs):

        self.nstep = kwargs.get('nstep', 1000)
        self.plot_interval = kwargs.get('plot_interval', 50)
        self.upscale_factor = kwargs.get('upscale_factor', 5)
        self.probe_fields = kwargs.get('probe_fields', ['u_x',])
        self.probe_ix = kwargs.get('probe_ix', [0.0,])
        self.probe_iy = kwargs.get('probe_iy', [0.0,])

        self.probe_vals = {}
        for (field,i,j) in zip(self.probe_fields, self.probe_ix, self.probe_iy):
            self.probe_vals[field + str(i) + '_' + str(j)] = [getattr(elas,field)[i,j],]

        self.iplot = [0, ]
        self.elas =  elas
        self.nx = elas.solid.shape[0]
        self.ny = elas.solid.shape[1]

        self.x = np.arange(self.nx) - (self.nx - 1) / 2
        self.y = np.arange(self.ny) - (self.ny - 1) / 2

        self.x_dec = kwargs.get("x_dec",0)
        self.y_dec = kwargs.get("y_dec",0)
        self.min_scale = kwargs.get("min_scale",0)
        self.max_scale = kwargs.get("max_scale",1)
        self.pause = kwargs.get("pause",1/100)
        self.plot_field = kwargs.get('plot_field','ux')

    def animate(self):
        xplot = (np.arange(self.upscale_factor * self.nx) - (self.upscale_factor * self.nx - 1) / 2) / self.upscale_factor
        yplot = (np.arange(self.upscale_factor * self.ny) - (self.upscale_factor * self.ny - 1) / 2) / self.upscale_factor
        gridyplot, gridxplot = np.meshgrid(yplot, xplot)
        solidplot = interpn((self.x, self.y), self.elas.solid, (gridxplot, gridyplot), method='nearest', bounds_error=False,
                            fill_value=False)
        solidplot = solidplot.astype(bool)
        solidplot_norm = interpn((self.x, self.y), self.elas.solid, (gridxplot, gridyplot), method='linear', bounds_error=False,
                                 fill_value=0)
        solidplot_norm[solidplot_norm == 0] = 0.00001
        solidplot_def = solidplot.copy()
        smooth_filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8
        # maybe custom made interpn using thiese filters could be faster
        # filter_plot_small = np.ones((int(self.upscale_factor), int(self.upscale_factor)))
        # filter_plot_big = np.ones((int(2*self.upscale_factor -1), int(2*self.upscale_factor -1)))
        # starting by setting u only on known points... and not forgetting norm

        if self.plot_interval <= self.nstep:
            fig, ax = plt.subplots(1, 1)
            t = ax.text(-0.1, 0, '0')
            im = ax.imshow(0 * gridxplot, vmin=self.min_scale, vmax=self.max_scale)
            for i in range(0, self.nstep):
                self.elas.explicit_step()
                if np.mod(i, self.plot_interval) == 0:
                    z = np.zeros(gridxplot.shape)

                    # Interpolate u on big grid
                    ux_plot = interpn((self.x, self.y), self.elas.ux, (gridxplot, gridyplot), method='linear', bounds_error=False,
                                      fill_value=0) / solidplot_norm
                    uy_plot = interpn((self.x, self.y), self.elas.uy, (gridxplot, gridyplot), method='linear', bounds_error=False,
                                      fill_value=0) / solidplot_norm
                    out_plot = interpn((self.x + self.x_dec, self.y + self.y_dec), getattr(self.elas, self.plot_field), (gridxplot, gridyplot), method='linear',
                                       bounds_error=False, fill_value=0) / solidplot_norm
                    # y+0.5 because stress are not computed on the same grid as displacements !

                    # interpolate solid position with displacement
                    gridxx = gridxplot[solidplot] + ux_plot[solidplot]
                    gridyy = gridyplot[solidplot] + uy_plot[solidplot]
                    xi_solid = interpn((xplot, yplot), gridxplot, (gridxx, gridyy), method='nearest')
                    yi_solid = interpn((xplot, yplot), gridyplot, (gridxx, gridyy), method='nearest')
                    xi_solid *= self.upscale_factor
                    yi_solid *= self.upscale_factor
                    xi_solid += (self.upscale_factor * self.nx - 1) / 2
                    yi_solid += (self.upscale_factor * self.ny - 1) / 2
                    xi_solid = xi_solid.astype(int)
                    yi_solid = yi_solid.astype(int)

                    solidplot_def[:] = False
                    solidplot_def[
                        xi_solid, yi_solid] = True  # ! the same value may appear more than once in xi_solid,yi_solid
                    z[xi_solid, yi_solid] = out_plot[solidplot]
                    zsmooth = conv(z, smooth_filter)
                    z[np.bitwise_not(solidplot_def)] = zsmooth[np.bitwise_not(solidplot_def)]

                    im.set_array(z)
                    t.set_text(str(i))
                    plt.pause(self.pause)

                    for (field, ii, j) in zip(self.probe_fields, self.probe_ix, self.probe_iy):
                        self.probe_vals[field + str(ii) + '_' + str(j)].append(getattr(self.elas, field)[ii, j])

                    self.iplot.append(i)
        else:
            for i in range(0, self.nstep):
                self.elas.explicit_step()