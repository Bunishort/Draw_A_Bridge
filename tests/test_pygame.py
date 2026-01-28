import pygame
import numpy as np
from context import sample

# simulation parameters
E=1
nu = 0.4
nx=100
ny=100
lm = 4.5 * 7/nx

vol_mass = 0.5
dt = 0.3 / 7
ratio = 0.2  # must be between 0 and 1
tau = 3

nbstep = 10 # nb of steps per frame

fx = 0.01*lm

c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu)))
c_s = np.sqrt(E / ratio /  (2 * (1 + nu)) / vol_mass)

print( 'Max Sound speed * dt / lm ')
print( 'Compression : ' + str(c_p * dt / lm))
print( 'Shear: ' + str(c_s * dt / lm))

elas_lambda = E*nu /(1+nu)/(1-2*nu)
elas_mu = E/2/(1+nu)

solid = np.zeros([nx,ny],dtype = bool)
ix = int(nx/2)
iy = int(ny/10)
solid[ix:(ix+2),iy:(iy+2)] = True
ux_imp=np.zeros(solid.shape)
ux_imp[:,:] = np.nan
ux_imp[ix:(ix+2),iy:(iy+2)] = 0
uy_imp = ux_imp.copy()
fx_imp = np.ones(solid.shape) * fx

# --- L'INTERFACE PYGAME ---
def main():
    # 1. Configuration
    RES = (100, 100)  # Taille de la grille (et fenêtre)
    SCALE = 5  # Si votre grille FEM est petite (ex: 100x100), mettez SCALE à 5 ou 10

    pygame.init()
    window = pygame.display.set_mode((RES[0] * SCALE, RES[1] * SCALE))
    clock = pygame.time.Clock()

    # Instanciation de votre solveur

    solver = sample.core.ElasticProblem(solid, elas_lambda, elas_mu, lm, ux_imp, uy_imp,
                                      is_explicit=True, vol_mass=vol_mass, dt = dt, ratio=ratio, tau=tau,
                                        fx_imp=fx_imp)

    running = True
    while running:
        # A. Gestion des Entrées (Inputs)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Interaction Souris (Temps réel)
        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        gx, gy = mx // SCALE, my // SCALE  # Conversion coord écran -> coord grille FEM

        if mouse_buttons[0]:  # Clic Gauche : Dessiner matière
            solver.mod_solid(gx, gy, 1)  # 1 = Solide

        if mouse_buttons[2]:  # Clic Droit : Gommer ou autre CL
            solver.mod_solid(gx, gy, 0)  # 0 = Vide

        # B. Étape de Physique (Votre code explicite)
        # On peut faire plusieurs sous-pas pour la stabilité si nécessaire
        for i in range(0, nbstep):
            solver.explicit_step()

        # C. Rendu (Visualisation)
        # Création d'une image RGB à partir des données Numpy
        # Fond noir
        render_array = np.zeros((RES[0], RES[1], 3), dtype=np.uint8)

        # Colorier la matière (ex: Blanc pour solide)
        mask = solver.solid
        render_array[mask] = [200, 200, 200]

        # Superposer la contrainte (ex: Rouge selon l'intensité)
        # Astuce : utilisez la map de contrainte pour moduler la couleur rouge
        stress_display = (solver.sxy_x_old / E * 255).astype(np.uint8)
        render_array[mask, 0] = stress_display[mask]  # Canal Rouge

        # Transfert Numpy -> Pygame (Très rapide)
        surface = pygame.surfarray.make_surface(render_array)

        if SCALE > 1:
            surface = pygame.transform.scale(surface, (RES[0] * SCALE, RES[1] * SCALE))

        window.blit(surface, (0, 0))

        # Afficher les FPS
        a=clock.get_fps()
        pygame.display.set_caption(f"FEM Explicite - FPS: {clock.get_fps():.1f}")
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()