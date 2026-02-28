from sample.core import conv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interpn
from line_profiler import profile
from scipy.ndimage import map_coordinates
import moderngl


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


class ExplicitAnimation_pygame:
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

        self.upscale_factor = kwargs.get('upscale_factor', 5)
        self.elas =  elas
        self.nx = elas.solid.shape[0]
        self.ny = elas.solid.shape[1]

        self.x = np.arange(self.nx) - (self.nx - 1) / 2
        self.y = np.arange(self.ny) - (self.ny - 1) / 2

        self.x_dec = kwargs.get("x_dec",0)
        self.y_dec = kwargs.get("y_dec",0)
        self.min_scale = kwargs.get("min_scale",0)
        self.max_scale = kwargs.get("max_scale",1)
        self.plot_field = kwargs.get('plot_field','ux')

        self.xplot = (np.arange(self.upscale_factor * self.nx) - (self.upscale_factor * self.nx - 1) / 2) / self.upscale_factor
        self.yplot = (np.arange(self.upscale_factor * self.ny) - (self.upscale_factor * self.ny - 1) / 2) / self.upscale_factor
        self.gridyplot, self.gridxplot = np.meshgrid(self.yplot, self.xplot)
        self.solidplot = interpn((self.x, self.y), self.elas.solid, (self.gridxplot, self.gridyplot), method='nearest', bounds_error=False,
                            fill_value=False)
        self.solidplot = self.solidplot.astype(bool)
        self.solidplot_norm = interpn((self.x, self.y), self.elas.solid, (self.gridxplot, self.gridyplot), method='linear', bounds_error=False,
                                 fill_value=0)
        self.solidplot_norm[self.solidplot_norm == 0] = 0.00001
        self.solidplot_def = self.solidplot.copy()
        self.smooth_filter = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8

        #init for map_cordinates
        x_indices = (self.gridxplot - self.x[0]) / (self.x[1] - self.x[0])
        y_indices = (self.gridyplot - self.y[0]) / (self.y[1] - self.y[0])
        self.map_coords = np.array([x_indices.ravel(), y_indices.ravel()])

        x_indices = (self.gridxplot - self.x[0] - self.x_dec) / (self.x[1] - self.x[0])
        y_indices = (self.gridyplot - self.y[0] - self.y_dec) / (self.y[1] - self.y[0])
        self.map_coords_dec = np.array([x_indices.ravel(), y_indices.ravel()])

    @profile
    def calc_image(self):

        # maybe custom made interpn using thiese filters could be faster
        # filter_plot_small = np.ones((int(self.upscale_factor), int(self.upscale_factor)))
        # filter_plot_big = np.ones((int(2*self.upscale_factor -1), int(2*self.upscale_factor -1)))
        # starting by setting u only on known points... and not forgetting norm
        z = np.zeros(self.gridxplot.shape)
        # Interpolate u on big grid
        # ux_plot = interpn((self.x, self.y), self.elas.ux, (self.gridxplot, self.gridyplot), method='linear', bounds_error=False,
        #                   fill_value=0) / self.solidplot_norm
        ux_plot = map_coordinates(
            self.elas.ux,
            self.map_coords,
            order=1,  # linear
            mode='constant',  # bounds_error False
            cval=0.0  # fill_value=0
        )
        ux_plot = ux_plot.reshape(self.gridxplot.shape) #/ self.solidplot_norm

        uy_plot = map_coordinates(
            self.elas.uy,
            self.map_coords,
            order=1,  # linear
            mode='constant',  # bounds_error False
            cval=0.0  # fill_value=0
        )
        uy_plot = uy_plot.reshape(self.gridxplot.shape) #/ self.solidplot_norm

        out_plot = map_coordinates(
            getattr(self.elas, self.plot_field),
            self.map_coords_dec,
            order=1,  # linear
            mode='constant',  # bounds_error False
            cval=0.0  # fill_value=0
        )
        out_plot = out_plot.reshape(self.gridxplot.shape) #/ self.solidplot_norm

        # interpolate solid position with displacement
        gridxx = self.gridxplot[self.solidplot] + ux_plot[self.solidplot]
        gridyy = self.gridyplot[self.solidplot] + uy_plot[self.solidplot]
        xi_solid = interpn((self.xplot, self.yplot), self.gridxplot, (gridxx, gridyy), method='nearest')
        yi_solid = interpn((self.xplot, self.yplot), self.gridyplot, (gridxx, gridyy), method='nearest')
        xi_solid *= self.upscale_factor
        yi_solid *= self.upscale_factor
        xi_solid += (self.upscale_factor * self.nx - 1) / 2
        yi_solid += (self.upscale_factor * self.ny - 1) / 2
        xi_solid = xi_solid.astype(int)
        yi_solid = yi_solid.astype(int)

        self.solidplot_def[:] = False
        self.solidplot_def[
            xi_solid, yi_solid] = True  # ! the same value may appear more than once in xi_solid,yi_solid
        z[xi_solid, yi_solid] = out_plot[self.solidplot]
        zsmooth = conv(z, self.smooth_filter)
        z[np.bitwise_not(self.solidplot_def)] = zsmooth[np.bitwise_not(self.solidplot_def)]

        return z

#Game interface
import pygame
import moderngl
import numpy as np

# --- SHADERS ---
# On utilise un seul shader capable de gérer les deux modes via une variable 'u_mode'
VTX_SHADER = """
#version 330
in vec2 in_pos;      // Position grille (-1 à 1)
in vec2 in_disp;     // Déplacement (u) calculé par NumPy
in float in_stress;  // Intensité (contrainte)

uniform int u_mode;        // 0: Dessin, 1: Visualisation
uniform float u_amp;       // Amplification de la déformée

out float v_stress;
flat out int v_mode;

void main() {
    v_mode = u_mode;
    v_stress = in_stress;

    vec2 final_pos = in_pos;
    if (u_mode == 1) {
        final_pos += in_disp * u_amp; // Applique la déformée
    }
    gl_Position = vec4(final_pos, 0.0, 1.0);
    gl_PointSize = 4.0; 
}
"""

FRAG_SHADER = """
#version 330
in float v_stress;
flat in int v_mode;
out vec4 f_color;

void main() {
    if (v_mode == 0) {
        f_color = vec4(0.8, 0.8, 0.8, 1.0); // Mode dessin : Gris clair
    } else {
        // Mode Visu : Dégradé Bleu (froid) -> Rouge (chaud)
        f_color = vec4(v_stress, 0.5 * (1.0 - v_stress), 1.0 - v_stress, 1.0);
    }
}
"""


class SimulationApp:
    def __init__(self, solver, **kwargs):
        #solver = class initialisée par sample.core.ElasticProblem
        pygame.init()
        self.solver = solver
        self.res = solver.solid.shape
        self.screen_size = kwargs.get('screen_size', (800, 800))
        self.nbstep = kwargs.get('nbstep', 10)
        self.f_attract_const = kwargs.get('f_attract_const', 1e-2)
        pygame.display.set_mode(self.screen_size, pygame.OPENGL | pygame.DOUBLEBUF)

        self.fx_imp_cte = solver.fx_imp_old.copy()
        self.fy_imp_cte = solver.fy_imp_old.copy()

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.prog = self.ctx.program(vertex_shader=VTX_SHADER, fragment_shader=FRAG_SHADER)

        # --- Simulation data init ---
        self.plot_field = np.zeros((self.res[1], self.res[0]), dtype='f4')  # sigma
        self.disp = np.zeros((self.res[1], self.res[0], 2), dtype='f4')

        # --- GPU preparation---
        # coordinates normalized (-1 à 1)
        x = np.linspace(-1, 1, self.res[0])
        y = np.linspace(1, -1, self.res[1])
        gx, gy = np.meshgrid(x, y)
        self.pos_init = np.stack([gx, gy], axis=-1).astype('f4')

        self.vbo_pos = self.ctx.buffer(self.pos_init.tobytes())
        self.vbo_disp = self.ctx.buffer(reserve=self.pos_init.nbytes)
        self.vbo_stress = self.ctx.buffer(reserve=self.pos_init.size * 4)

        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo_pos, '2f', 'in_pos'),
            (self.vbo_disp, '2f', 'in_disp'),
            (self.vbo_stress, '1f', 'in_stress')
        ])

        self.mode_simu = False  # False = Draw mode, True = Simulation
        self.running = True

        xtemp = np.arange(self.solver.solid.shape[0])
        ytemp = np.arange(self.solver.solid.shape[1])
        self.gridy, self.gridx = np.meshgrid(ytemp, xtemp)

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            # events
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.mode_simu = not self.mode_simu

            m_left, _, m_right = pygame.mouse.get_pressed()
            if m_left or m_right:
                mx, my = pygame.mouse.get_pos()
                # Mapping coord screen -> index grid
                gx = int((mx / self.screen_size[0]) * self.res[0])
                gy = int((my / self.screen_size[1]) * self.res[1])

            # Draw mode
            if not self.mode_simu:
                    if m_left:  # Draw
                        self.solver.mod_solid(gy, gx, 1)
                    if m_right:  # Erase
                        self.solver.mod_solid(gy, gx, 0)
            # Simulation mode
            else:
                for i in range(0, self.nbstep):
                    self.solver.explicit_step()
                self.disp[:,:,1] = -self.solver.ux * 2 / (self.res[0] * self.solver.lm) # why - sign here ?
                self.disp[:, :, 0] = self.solver.uy * 2 / (self.res[1] * self.solver.lm)

                if m_left:  # Attractor
                    dx = gy - (self.gridx +  self.solver.ux / self.solver.lm) #x/y inversion in gx gy
                    dy = gx - (self.gridy + self.solver.uy / self.solver.lm)#
                    d = np.sqrt(dx ** 2 + dy ** 2)
                    f_attract = self.f_attract_const / (1 + d)
                    fx_imp_live = self.fx_imp_cte + f_attract * dx / (1 + d)
                    fy_imp_live = self.fy_imp_cte + f_attract * dy / (1 + d)
                    self.solver.update_f_imp(fx_imp_live, fy_imp_live)
                    self.plot_field[:, :] = 100*f_attract#remove, debug only
                else:
                    self.solver.update_f_imp(self.fx_imp_cte , self.fy_imp_cte )

            #display
            self.ctx.clear(0.1, 0.1, 0.1)

            # Display only points with matter
            active_mask = self.solver.solid.flatten() > 0
            if np.any(active_mask):
                # Data filtering for GPU
                idx = np.where(active_mask)[0]
                d_data = self.disp.reshape(-1, 2)[idx].tobytes()
                s_data = self.plot_field.flatten()[idx].tobytes()
                p_data = self.pos_init.reshape(-1, 2)[idx].tobytes()

                # Buffer update
                self.vbo_pos.write(p_data)
                self.vbo_disp.write(d_data)
                self.vbo_stress.write(s_data)

                # Shader variables config
                self.prog['u_mode'].value = 1 if self.mode_simu else 0
                self.prog['u_amp'].value = 1.0  # Amplification factor

                # Draw points
                self.vao.render(moderngl.POINTS, vertices=len(idx))

            pygame.display.flip()
            clock.tick(60)
            pygame.display.set_caption(
                f"Mode: {'SIMULATION' if self.mode_simu else 'DESSIN'} - FPS: {clock.get_fps():.0f}")


