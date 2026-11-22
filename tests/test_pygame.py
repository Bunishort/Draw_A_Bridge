import sys
import os
import yaml
import ctypes
from pathlib import Path

if sys.platform == "win32":
    os.environ["PYOPENGL_PLATFORM"] = "nt"
import pygame
import numpy as np
from context import sample
import moderngl

# Ce script est dans <racine>/tests ; les niveaux sont dans <racine>/levels
ROOT_DIR = Path(__file__).resolve().parent.parent
LEVELS_DIR = ROOT_DIR / "levels"
DEFAULT_LEVEL = "level1.yaml"

#Import libGL
if sys.platform.startswith('linux'):
    try:
        ctypes.CDLL('libGL.so')
    except OSError:
        try:
            # Forcer le chargement de la version runtime présente sur le système
            ctypes.CDLL('libGL.so.1', mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

def check_numbers(node, path=""):
    """check no numerical parameter was read as text.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key != "image":
                check_numbers(value, path + "/" + str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            check_numbers(value, "{}[{}]".format(path, i))
    elif isinstance(node, str):
        raise ValueError("Parameter '{}' was read as text {!r} instead of number ".format(path, node))


def read_level_file(path):
    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
    else:
        raise ValueError("Need a yaml file")
    check_numbers(params)
    return params


def load_level(level_name):
    """Read json file and associated images
    """
    level_path = LEVELS_DIR / level_name
    params = read_level_file(level_path)

    # Image : chemin relatif au dossier levels/
    image_path = LEVELS_DIR / params["image"]
    surf = pygame.image.load(str(image_path))
    surf = pygame.transform.flip(surf, True, False)
    surf = pygame.transform.rotate(surf, 90)

    rgb = pygame.surfarray.array3d(surf)
    red = rgb[:, :, 0] >= 200
    blue = rgb[:, :, 2] >= 200

    solid = red | blue                     #
    ux_imp = np.full(solid.shape, np.nan)  #
    ux_imp[red] = 0.0                      #
    uy_imp = ux_imp.copy()

    return params, solid, ux_imp, uy_imp


# ---PYGAME interface---
def main():
    level_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LEVEL

    pygame.init()

    params, solid, ux_imp, uy_imp = load_level(level_name)

    ###### Taille de la grille : déduite de l'image ######
    nx, ny = solid.shape

    ###### Performance / Speed / Size tuning ######
    screen_size = tuple(params["screen_size"])
    nbstep = int(params["nbstep"])

    ###### Simulation physical parameters ######
    sim = params["simulation"]
    damping = sim["damping"]
    fx = sim["fx"]
    fy = sim["fy"]
    f_attract_const = sim["f_attract_const"]
    max_stress_plot = sim["max_stress_plot"]

    ###### Damage model parameters ####
    dmg = params["damage"]
    tau_damage = dmg["tau_damage"]
    tau_damage_filter = dmg["tau_damage_filter"]
    damage_stress = dmg["damage_stress"]
    damage_stress_max = dmg["damage_stress_max"]

    # -----------  vvvvvv   DO NOT CHANGE  vvvvvvv  ---------------
    ###### Material parameters ######
    mat = params["material"]
    E = mat["E"]
    nu = mat["nu"]
    vol_mass = mat["vol_mass"]
    ratio = mat["ratio"]
    tau = mat["tau"]
    lm = mat["lm"]

    c_p = np.sqrt(E / ratio * (1 - nu) / (vol_mass * (1 + nu) * (1 - 2 * nu)))  # onde de compression
    c_s = np.sqrt(E / ratio / (2 * (1 + nu)) / vol_mass)                        # onde de cisaillement

    dt = 0.9 * lm / c_p  # cp*dt/lm doit rester < 1 pour la stabilité

    print('Sound speed * dt / lm ')
    print("Compression : {:.2f}".format(c_p * dt / lm))
    print("Shear : {:.2f}".format(c_s * dt / lm))

    elas_lambda = E * nu / (1 + nu) / (1 - 2 * nu)  # Elastic parameter. Do not touch
    elas_mu = E / 2 / (1 + nu)

    print("Niveau : {} ({} x {} éléments)".format(level_name, nx, ny))

    pygame.display.set_mode(screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    # Solver init
    solver = sample.core.ElasticProblem(solid, elas_lambda, elas_mu, lm, ux_imp, uy_imp,
                                        is_explicit=True, vol_mass=vol_mass, dt=dt, ratio=ratio, tau=tau,
                                        damping=damping, gl_context=ctx)

    game = sample.interface.SimulationApp(solver, ctx, screen_size=screen_size, nbstep=nbstep,
                                          max_stress_plot=max_stress_plot,
                                          fx_grav=fx, fy_grav=fy, f_attract_const=f_attract_const,
                                          tau_damage=tau_damage, tau_damage_filter=tau_damage_filter,
                                          damage_stress=damage_stress, damage_stress_max=damage_stress_max)
    game.run()

    pygame.quit()


if __name__ == "__main__":
    main()
