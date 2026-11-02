"""Command-line entry points.

    Finomania                       desktop application
    Finomania file.fwf|csv|xlsx     open a workfile / data file in the application
    Finomania prog.fprg             open a program in the application
    Finomania --run prog.fprg [--html report.html]   run a program in batch (no window)
    Finomania --cli                 text command prompt
    Finomania --selftest            built-in self test
"""
import argparse
import sys


def _parser():
    ap = argparse.ArgumentParser(prog="finomania", description="Finomania econometrics workbench")
    ap.add_argument("file", nargs="?", help="workfile, data file or program to open")
    ap.add_argument("--cli", action="store_true", help="interactive text command prompt")
    ap.add_argument("--run", action="store_true", help="run the program file in batch mode (no window)")
    ap.add_argument("--html", help="write the outputs of a batch run to an HTML report")
    ap.add_argument("--quiet", action="store_true", help="batch mode: do not print reports")
    ap.add_argument("--selftest", action="store_true", help="run the built-in self test and exit")
    ap.add_argument("--no-gui-test", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="store_true")
    return ap


def main(argv=None, console: bool = False):
    a = _parser().parse_args(argv)
    if a.version:
        from . import __version__
        print(f"Finomania {__version__}")
        return 0
    if a.selftest:
        from .selftest import run
        return run(gui=not a.no_gui_test)
    if a.cli:
        from .cli import repl
        repl()
        return 0
    is_prog = bool(a.file and a.file.lower().endswith((".fprg", ".prg")))
    if is_prog and (a.run or a.html or console):
        from .cli import run_program
        return run_program(a.file, a.html, a.quiet)
    from .gui.app import run_gui
    return run_gui(a.file)


def cli_main():
    """Console entry point (`finomania-cli`): no arguments -> command prompt; program -> batch run."""
    args = sys.argv[1:]
    return main(args if args else ["--cli"], console=True)


if __name__ == "__main__":
    sys.exit(main())
