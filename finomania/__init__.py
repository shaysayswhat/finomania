"""Finomania — an open-source econometrics workbench for time-series and panel
financial data, inspired by EViews."""
__version__ = "1.0.0"

from .core.workfile import Workfile  # noqa: E402,F401
from .core.commands import Interpreter  # noqa: E402,F401
from .report import Report  # noqa: E402,F401


def session(wf=None):
    """Convenience: an interpreter for scripting / notebooks.

    >>> import finomania as fm
    >>> s = fm.session()
    >>> s.run("wfcreate u 100")
    """
    return Interpreter(wf)
