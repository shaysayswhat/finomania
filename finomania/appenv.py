"""Runtime environment helpers: frozen (installed .exe) vs source runs, user folders, logging."""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import sys

APP_NAME = "Finomania"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> str:
    """Folder holding bundled resources (examples, docs)."""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def documents_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes
            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5 (My Documents)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:
                return buf.value
        except Exception:
            pass
    docs = os.path.join(home, "Documents")
    return docs if os.path.isdir(docs) else home


def user_dir() -> str:
    """Default working folder: Documents/Finomania."""
    d = os.path.join(documents_dir(), APP_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.expanduser("~")
    return d


def data_dir() -> str:
    """Per-user application data (logs, caches)."""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def log_path() -> str:
    d = os.path.join(data_dir(), "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "finomania.log")


def setup_logging():
    """Windowed executables have no console: send stdout/stderr to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        p = log_path()
        if os.path.exists(p) and os.path.getsize(p) > 2_000_000:
            os.replace(p, p + ".old")
        f = open(p, "a", encoding="utf-8", buffering=1)
        f.write(f"\n===== {APP_NAME} started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f
    except OSError:
        class _Null:
            def write(self, *_):
                return 0

            def flush(self):
                pass
        sys.stdout = sys.stdout or _Null()
        sys.stderr = sys.stderr or _Null()


def examples_dir() -> str:
    """Demo programs and data.  Installed copies are read-only, so they are copied to the
    user's Documents/Finomania/examples on first use."""
    src = os.path.join(resource_dir(), "examples")
    if not is_frozen():
        return src
    from . import __version__
    dst = os.path.join(user_dir(), "examples")
    stamp = os.path.join(dst, ".version")
    try:
        current = open(stamp).read().strip() if os.path.exists(stamp) else ""
        if current != __version__ and os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            with open(stamp, "w") as f:
                f.write(__version__)
        return dst if os.path.isdir(dst) else src
    except OSError:
        return src


def docs_path(name: str) -> str | None:
    p = os.path.join(resource_dir(), "docs", name)
    return p if os.path.exists(p) else None
