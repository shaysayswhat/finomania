# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build specification for Finomania.
#
#   pyinstaller --noconfirm --clean packaging/finomania.spec
#
# Produces dist/Finomania/ with
#   Finomania.exe      windowed desktop application
#   finomania-cli.exe  console: command prompt, batch programs, --selftest
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821  (SPECPATH is injected by PyInstaller)
sys.path.insert(0, ROOT)
with open(os.path.join(ROOT, "finomania", "__init__.py"), encoding="utf-8") as fh:
    VERSION = next(l.split("=")[1].strip().strip('"') for l in fh if l.startswith("__version__"))
WIN = sys.platform.startswith("win")

datas = [
    (os.path.join(ROOT, "examples"), "examples"),
    (os.path.join(ROOT, "docs", "COMMANDS.md"), "docs"),
    (os.path.join(ROOT, "docs", "screenshots"), os.path.join("docs", "screenshots")),
    (os.path.join(ROOT, "README.md"), "."),
    (os.path.join(ROOT, "LICENSE"), "."),
    (os.path.join(ROOT, "finomania", "gui", "assets"), os.path.join("finomania", "gui", "assets")),
]
binaries = []
hiddenimports = collect_submodules("finomania")


def _no_tests(name):
    return ".tests" not in name and not name.endswith("conftest")


for pkg in ("statsmodels", "arch", "linearmodels", "formulaic", "patsy", "pyhdfe", "interface_meta"):
    try:
        d, b, h = collect_all(pkg, filter_submodules=_no_tests)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # optional dependency missing
        print(f"[finomania.spec] skipping {pkg}: {exc}")

hiddenimports += [
    "sklearn.ensemble", "sklearn.neural_network", "sklearn.svm", "sklearn.linear_model", "sklearn.multioutput",
    "sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting", "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes", "scipy.special.cython_special", "matplotlib.backends.backend_qtagg",
    "openpyxl", "PySide6.QtSvg",
]

excludes = [
    "tkinter", "_tkinter", "IPython", "jupyter", "notebook", "ipykernel", "PyQt5", "PyQt6", "PySide2", "torch",
    "tensorflow", "pytest", "sphinx", "numba", "llvmlite", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.Qt3DCore", "PySide6.QtQuick3D", "PySide6.QtMultimedia", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtGraphs",
]

version_file = None
if WIN:
    v = tuple(int(x) for x in (VERSION.split(".") + ["0", "0", "0"])[:4])
    version_file = os.path.join(SPECPATH, "version_info.txt")  # noqa: F821
    with open(version_file, "w", encoding="utf-8") as fh:
        fh.write(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={v}, prodvers={v}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
                    date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Finomania contributors'),
      StringStruct('FileDescription', 'Finomania - econometrics for financial time series and panel data'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', 'Finomania'),
      StringStruct('LegalCopyright', 'MIT License'),
      StringStruct('OriginalFilename', 'Finomania.exe'),
      StringStruct('ProductName', 'Finomania'),
      StringStruct('ProductVersion', '{VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""")

icon = os.path.join(SPECPATH, "finomania.ico") if WIN else None  # noqa: F821

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "launch.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe_gui = EXE(  # noqa: F821
    pyz, a.scripts, [], exclude_binaries=True, name="Finomania", console=False, icon=icon, version=version_file,
    upx=False, disable_windowed_traceback=False,
)
exe_cli = EXE(  # noqa: F821
    pyz, a.scripts, [], exclude_binaries=True, name="finomania-cli", console=True, icon=icon, version=version_file,
    upx=False,
)
coll = COLLECT(exe_gui, exe_cli, a.binaries, a.datas, name="Finomania", upx=False)  # noqa: F821
