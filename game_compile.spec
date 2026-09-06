# -*- mode: python ; coding: utf-8 -*-

import os
import sys

block_cipher = None

# Ressources additionnelles à inclure : (chemin_source, dossier_destination)
added_files = [
    ('sample/data', 'sample/data'),
    ('sample', 'sample'),  # Ajuste si le dossier des shaders a un autre nom/chemin
]

# Liaisons C/C++ parfois mal détectées statiquement
hidden_imports = [
    'OpenGL.GL',
    'OpenGL.targets',
    'imgui.integrations.pygame',
]

# Exclusion des bibliothèques lourdes pour réduire la taille du paquet
excluded_modules = [
    'matplotlib',
    'scipy',
    'numba',
    'tkinter',
    'IPython',
]

a = Analysis(
    [os.path.join('tests','test_pygame.py')],
    pathex=[],
    binaries=[],
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