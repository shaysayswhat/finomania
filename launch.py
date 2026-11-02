"""Launcher for the frozen executables built by PyInstaller.

Finomania.exe      (windowed)  -> desktop application
finomania-cli.exe  (console)   -> command prompt / batch runner / self test
"""
import multiprocessing
import os
import sys


def _main():
    multiprocessing.freeze_support()
    from finomania.appenv import setup_logging
    setup_logging()
    exe = os.path.splitext(os.path.basename(sys.executable if getattr(sys, "frozen", False) else sys.argv[0]))[0]
    from finomania.__main__ import cli_main, main
    if exe.lower().endswith("-cli"):
        return cli_main()
    return main()


if __name__ == "__main__":
    sys.exit(_main())
