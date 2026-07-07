from sample.core import conv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interpn
from line_profiler import profile
from scipy.ndimage import map_coordinates
import moderngl


class ExplicitAnimation:
    """
    Animation based on matplotlib. Useful to run and live plot an explicit computation.
    Not useful for gaming as matplotlib is too slow.
    :param elas: ElasticProblem class object from sample.core, with is_explicit=True
    :**kwarg nstep : number of explicit simulation steps
    :**kwarg plot_interval : update animation plot every plot_interval steps
    :**kwarg upscale_factor : ratio btw simulation resolution and plot resolution
    :**kwarg probe_fields : list of str, names of the fiels of "elas" to be stored at each time step
    : at point probe_ix,probe_iy
    :**kwarg probe_ix : list of int : x position of points where the fields are stored
    :**kwarg probe_iy : list of int : y position ...
    :**kwarg x_dec : float : x position difference btw the simulation grid and the plot grid
    : (e.g : if the plot field is computed on x_edge, x_dec = 0.5
    :**kwarg y_dec : float : y position difference...
    :**kwarg min_scale : minimum value of the color scale in the plot
    :**kwarg max_scale : maximum value of the color scale in the plot
    :**kwarg pause : time in seconds of pause after each frame drawing. Useful if simulation is too fast
    :**kwarg plot_field : name of the field of "elas" to be plotted
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
                    self.elas.get_results()
                    z = np.zeros(gridxplot.shape)
                    if self.plot_field == 'VM_stress':
                        self.elas.VM_stress = self.elas.calc_VM_stress()

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
        self.elas.get_results()

######################--------------------Game interface---------------################
import pygame
import moderngl
import numpy as np

# --- SHADERS OPTIMISÉS ---
VTX_SHADER = """
#version 330
in vec2 in_pos; 

// Textures RGBA natives du solveur
uniform sampler2D u_tex_pos_vel;    // R=ux (lignes/descendant), G=uy (colonnes/droite)
uniform sampler2D u_tex_stress_old; // R=sxx, G=sxy_x, B=syy_y, A=sxy_y
uniform sampler2D u_tex_solid; // R=solid, G=solid_not_uimp

// Facteurs d'échelle physiques
uniform vec2 u_disp_scale; 
uniform float u_max_stress;

uniform int u_mode;
uniform float u_amp;
uniform float point_size;

out float v_stress;

void main() {
    // Mapping [-1, 1] vers [0, 1] pour lire la texture
    vec2 uv = vec2((in_pos.x + 1.0) * 0.5, (1.0 - in_pos.y) * 0.5);
     
    float is_solid = texture(u_tex_solid, uv).r;
    float is_solid_not_uimp = texture(u_tex_solid, uv).g; 
    
    if (is_solid < 0.5) {
        // Le point est vide : on le projette loin en dehors de l'écran pour l'annuler
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
        return;
    }

    // Lecture des données physiques GPU
    vec4 pv = texture(u_tex_pos_vel, uv);
    vec4 stress = texture(u_tex_stress_old, uv);

    float solver_ux = pv.r; // Déplacement vertical (vers le bas)
    float solver_uy = pv.g; // Déplacement horizontal (vers la droite)

    // Stress (G = sxy_x_old)
    if (is_solid > 0.5 && is_solid_not_uimp < 0.5) {
        v_stress = 1.0; 
    } else {
        // Calcul normal de la contrainte pour les autres points
        vec4 stress = texture(u_tex_stress_old, uv);
        v_stress = stress.g / u_max_stress;
    }

    vec2 final_pos = in_pos;
    if (u_mode == 1) {
        // Application propre des axes cartésiens OpenGL
        // X reçoit le déplacement horizontal (uy)
        // Y reçoit le déplacement vertical inversé (-ux car Y OpenGL pointe vers le haut)
        vec2 disp = vec2(solver_uy * u_disp_scale.x, -solver_ux * u_disp_scale.y);
        final_pos += disp * u_amp; 
    }

    gl_Position = vec4(final_pos, 0.0, 1.0);
    gl_PointSize = point_size; 
}
"""

FRAG_SHADER = """
#version 330
in float v_stress;
out vec4 f_color;

void main() {
    // Clamping to avoid weird colors for high stress values
    float s = clamp(v_stress, -1.0, 1.0);
    f_color = vec4(s, 0.5 * (1.0 - s), 1.0 - s, 1.0);
}
"""

class SimulationApp:
    def __init__(self, solver, ctx, **kwargs):
        self.solver = solver
        self.ctx = ctx
        self.H, self.W = solver.solid.shape

        self.screen_size = kwargs.get('screen_size', (800, 800))
        self.nbstep = kwargs.get('nbstep', 10)
        self.f_attract_const = kwargs.get('f_attract_const', 1e-2)
        self.max_stress = kwargs.get('max_stress', 1.0)

        self.point_size = self.screen_size[0] / self.W + 0.5

        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.prog = self.ctx.program(vertex_shader=VTX_SHADER, fragment_shader=FRAG_SHADER)

        # Grille statique
        x = np.linspace(-1, 1, self.W)
        y = np.linspace(1, -1, self.H)
        gx, gy = np.meshgrid(x, y)
        self.pos_init = np.stack([gx, gy], axis=-1).astype('f4')

        self.vbo_pos = self.ctx.buffer(self.pos_init.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo_pos, '2f', 'in_pos')])

        # Pré-calcul des échelles de déplacement (pour remplacer la division CPU)
        self.scale_x = 2.0 / (self.W * self.solver.lm)
        self.scale_y = 2.0 / (self.H * self.solver.lm)

        self.mode_simu = False
        self.running = True
        self.gx_old = 1.0
        self.gy_old = 1.0

        #Attractor init
        file_path = '../sample/update_f_imp.glsl'
        with open(file_path, 'r') as file:
            source_code_f_imp = file.read()
        data_att = np.stack([np.zeros(self.solver.solid.shape), np.zeros(self.solver.solid.shape)], axis=-1).astype('f4')
        self.tex_att = self.ctx.texture(self.solver.solid.shape[::-1], 2, data=data_att.tobytes(), dtype='f4')

        self.update_f_imp = self.ctx.compute_shader(source_code_f_imp)

        self.update_f_imp['u_mouse_active'].value = 0.0
        self.update_f_imp['u_mouse_col'].value = 0.0
        self.update_f_imp['u_mouse_row'].value = 0.0
        self.update_f_imp['u_f_attract'].value = 0.0
        self.update_f_imp['lm'].value = self.solver.lm
        self.update_f_imp['width'] = self.solver.solid.shape[1]
        self.update_f_imp['height'] = self.solver.solid.shape[0]

        self.tex_att.bind_to_image(6, read=True, write=True)


    def run(self):
        clock = pygame.time.Clock()
        draw_fixed = False

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.mode_simu = not self.mode_simu
                    if self.mode_simu:
                        self.solver.mod_solid_buffer_update()
                    else:
                        self.set_attractor_state(active=0.0)
                        self.solver.get_results()
                        draw_fixed = False # change between draw deformable solid or fixed solid
                if not self.mode_simu and event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                            draw_fixed = not draw_fixed

            m_left, _, m_right = pygame.mouse.get_pressed()
            mx, my = pygame.mouse.get_pos()
            gx = max(0, min(int((mx / self.screen_size[0]) * self.W), self.W - 1))
            gy = max(0, min(int((my / self.screen_size[1]) * self.H), self.H - 1))

            if not self.mode_simu:
                if m_left:
                    draw=1
                if m_right:
                    draw=0

                if m_left or m_right:
                    dist = int(np.sqrt((gx - self.gx_old)**2 + (gy - self.gy_old)**2))
                    for i in range(1, dist+2):
                        gxt = int(self.gx_old + (gx - self.gx_old) * i / (dist +1))
                        gyt = int(self.gy_old + (gy - self.gy_old) * i / (dist +1))
                        self.solver.mod_solid(gyt, gxt, draw, draw_fixed)
                    self.solver.mod_solid_update_solid()

                self.gx_old = gx
                self.gy_old = gy
            else:
                # --- GESTION DE L'ATTRACTEUR (DEVIENT UN APPEL GPU) ---
                if m_left:
                    # On délègue totalement le calcul de la force à la classe solver
                    self.set_attractor_state(active=1.0, target_x=gx, target_y=gy, force=self.f_attract_const)
                else:
                    self.set_attractor_state(active=0.0)

                # Étape explicite
                for _ in range(self.nbstep):
                    self.solver.explicit_step()

            # --- RENDU DIRECT DEPUIS LES TEXTURES DU SOLVEUR ---
            self.ctx.clear(0.1, 0.1, 0.1)

            # Liaison des textures
            self.solver.tex_pos_vel.use(location=0)
            self.prog['u_tex_pos_vel'].value = 0

            self.solver.tex_stress_old.use(location=1)
            self.prog['u_tex_stress_old'].value = 1

            self.solver.tex_masks.use(location=2)
            self.prog['u_tex_solid'].value = 2

            # Variables uniformes
            self.prog['u_mode'].value = 1 if self.mode_simu else 0
            self.prog['u_amp'].value = 1.0
            self.prog['point_size'].value = self.point_size
            self.prog['u_disp_scale'].value = (self.scale_x, self.scale_y)
            self.prog['u_max_stress'].value = self.max_stress

            self.vao.render(moderngl.POINTS, vertices=self.W * self.H)

            pygame.display.flip()
            clock.tick(60)
            pygame.display.set_caption(f"Mode: {'SIMU' if self.mode_simu else 'DESSIN'} - FPS: {clock.get_fps():.0f}")

    def set_attractor_state(self, active=False, target_x=0, target_y=0, force=0.0):
        """
        Send mouse coordinates to  Compute Shader.
        """
        self.update_f_imp['u_mouse_active'].value = active
        self.update_f_imp['u_mouse_col'].value = target_x
        self.update_f_imp['u_mouse_row'].value = target_y
        self.update_f_imp['u_f_attract'].value = force

        group_x = int(np.ceil(self.W / 16.0))
        group_y = int(np.ceil(self.H / 16.0))

        self.update_f_imp.run(group_x, group_y)