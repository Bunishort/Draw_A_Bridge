# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collecte automatique complète de PyOpenGL (datas, binaires, imports cachés)
opengl_datas, opengl_binaries, opengl_hiddenimports = collect_all('OpenGL')

# Ressources additionnelles à inclure : (chemin_source, dossier_destination)
added_files = [
    ('sample/data', 'sample/data'),
    ('sample', 'sample'),
] + opengl_datas

# Liaisons C/C++ et modules dynamiques de plateforme
hidden_imports = [
    'cv2',
    'OpenGL.GL',
    'OpenGL.targets',
    'OpenGL.platform.win32',
    'OpenGL.platform.glx',
    'OpenGL.platform.baseplatform',
    'imgui.integrations.pygame',
] + opengl_hiddenimports

# Exclusion des bibliothèques lourdes inutilisées
excluded_modules = [
    'matplotlib',
    'scipy',
    'numba',
    'tkinter',
    'IPython',
]

a = Analysis(
    [os.path.join('tests', 'test_pygame.py')],
    pathex=[],
    binaries=opengl_binaries,
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Draw_A_Bridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Draw_A_Bridge',
)