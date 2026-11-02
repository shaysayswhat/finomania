"""The shipped demo programs must run without errors."""
import os
import warnings

import pytest

from finomania.core.commands import Interpreter
from finomania.report import Message

warnings.simplefilter("ignore")
EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


@pytest.mark.parametrize("prog", sorted(f for f in os.listdir(EX) if f.endswith(".fprg")))
def test_demo_program(prog):
    it = Interpreter()
    out = it.run_file(os.path.join(EX, prog))
    errs = [o.text for o in out if isinstance(o, Message) and o.level == "error"]
    assert not errs, errs
