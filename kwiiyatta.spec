# -*- mode: python -*-

block_cipher = None

hiddenimports = [
    'bandmat.full',
    'pkg_resources.py2_warn',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
    'sklearn.utils._cython_blas',
]


kwiiyatta_a = Analysis(['kwiiyatta\\convert_voice.py'],
                       pathex=['.'],
                       binaries=[],
                       datas=[],
                       hiddenimports=hiddenimports,
                       hookspath=['pyinstaller_hooks'],
                       runtime_hooks=[],
                       excludes=[],
                       win_no_prefer_redirects=False,
                       win_private_assemblies=False,
                       cipher=block_cipher,
                       noarchive=False)
kwiiyatta_pyz = PYZ(kwiiyatta_a.pure, kwiiyatta_a.zipped_data,
                    cipher=block_cipher)
kwiiyatta_exe = EXE(kwiiyatta_pyz,
                    kwiiyatta_a.scripts,
                    kwiiyatta_a.binaries,
                    kwiiyatta_a.zipfiles,
                    kwiiyatta_a.datas,
                    [],
                    name='kwiiyatta',
                    debug=False,
                    bootloader_ignore_signals=False,
                    strip=False,
                    upx=True,
                    runtime_tmpdir=None,
                    console=True,
                    icon='kwiiyatta/view/res/yatta_akane.ico' )


kwiieiya_a = Analysis(['kwiiyatta\\resynthesize_voice.py'],
                      pathex=['.'],
                      binaries=[],
                      datas=[('kwiiyatta/view/res', 'kwiiyatta/view/res')],
                      hiddenimports=hiddenimports,
                      hookspath=['pyinstaller_hooks'],
                      runtime_hooks=[],
                      excludes=[],
                      win_no_prefer_redirects=False,
                      win_private_assemblies=False,
                      cipher=block_cipher,
                      noarchive=False)
kwiieiya_pyz = PYZ(kwiieiya_a.pure, kwiieiya_a.zipped_data,
                   cipher=block_cipher)
kwiieiya_exe = EXE(kwiieiya_pyz,
                   kwiieiya_a.scripts,
                   kwiieiya_a.binaries,
                   kwiieiya_a.zipfiles,
                   kwiieiya_a.datas,
                   [],
                   name='kwiieiya',
                   debug=False,
                   bootloader_ignore_signals=False,
                   strip=False,
                   upx=True,
                   runtime_tmpdir=None,
                   console=False,
                   icon='kwiiyatta/view/res/yatta_aoi.ico' )

MERGE((kwiiyatta_a, 'kwiiyatta', 'kwiiyatta'),
      (kwiieiya_a, 'kwiieiya', 'kwiieiya'))
