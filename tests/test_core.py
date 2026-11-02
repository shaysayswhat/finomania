"""Core engine tests: expressions, samples, workfile persistence, command language."""
import os

import numpy as np
import pandas as pd
import pytest

from finomania.core.commands import Interpreter, ObjectOutput
from finomania.core.expr import Evaluator
from finomania.core.workfile import Workfile
from finomania.report import Message


@pytest.fixture
def wf():
    w = Workfile.create("m", "2000m1", "2001m12")
    w.set_series("x", np.arange(1, 25, dtype=float))
    return w


def test_lags_and_differences(wf):
    ev = Evaluator(wf)
    assert np.isnan(ev.eval("x(-1)").iloc[0])
    assert ev.eval("x(-1)").iloc[5] == 5
    assert ev.eval("x(1)").iloc[5] == 7
    assert ev.eval("d(x)").iloc[3] == 1
    assert ev.eval("d(x,2)").iloc[5] == 0
    np.testing.assert_allclose(ev.eval("dlog(x)").iloc[1], np.log(2))
    assert ev.eval("@pch(x)").iloc[1] == pytest.approx(1.0)


def test_logical_and_eviews_equality(wf):
    ev = Evaluator(wf)
    assert ev.eval("x=5").sum() == 1
    assert ev.eval("x<>5").sum() == 23
    assert ev.eval("x>3 and x<6").sum() == 2
    assert ev.eval("x<2 or x>23").sum() == 2
    assert ev.eval("@recode(x>12, 1, 0)").sum() == 12


def test_sample_statistics_respect_smpl(wf):
    ev = Evaluator(wf)
    wf.set_sample("2001m1 2001m12")
    assert ev.eval("@mean(x)") == pytest.approx(18.5)
    wf.set_sample("@all if x>20")
    assert wf.sample.sum() == 4
    wf.set_sample("@first+2 @last-2")
    assert wf.sample_label() == "2000M03 2001M10"


def test_genr_respects_sample(wf):
    it = Interpreter(wf)
    it.run("smpl 2000m1 2000m6\nseries y = 1\nsmpl @all")
    y = wf.series("y")
    assert y.iloc[:6].eq(1).all() and y.iloc[6:].isna().all()


def test_panel_lags_within_cross_section():
    df = pd.DataFrame({"id": ["a"] * 3 + ["b"] * 3, "t": [2000, 2001, 2002] * 2, "v": [1, 2, 3, 10, 20, 30.0]})
    wf = Workfile.from_dataframe(df, panel=("id", "t"))
    ev = Evaluator(wf)
    lag = ev.eval("v(-1)").values
    assert np.isnan(lag[0]) and np.isnan(lag[3]) and lag[4] == 10


def test_program_control_flow(wf):
    it = Interpreter(wf)
    out = it.run("""
    !total = 0
    for !i = 1 to 4
        !total = !total + !i
        series z{!i} = x*{!i}
    next
    scalar s = !total
    for %v x z2
        series c_{%v} = {%v} + 1
    next
    if !total > 5 then
        scalar big = 1
    else
        scalar big = 0
    endif
    """)
    assert not [o for o in out if isinstance(o, Message) and o.level == "error"]
    assert wf.objects["s"].value == 10
    assert wf.objects["big"].value == 1
    assert wf.series("z3").iloc[0] == 3
    assert wf.series("c_z2").iloc[1] == 5


def test_save_load_roundtrip(tmp_path, wf):
    it = Interpreter(wf)
    it.run("series y = 2*x + nrnd\nequation eq1.ls y c x\ngroup g1 x y\nscalar k = 3")
    p = str(tmp_path / "t.fwf")
    wf.save(p)
    w2 = Workfile.load(p)
    assert w2.freq == "m" and w2.nobs == 24
    np.testing.assert_allclose(w2.series("y").values, wf.series("y").values)
    assert w2.objects["g1"].members == ["x", "y"]
    # estimated objects are re-estimated lazily from their command
    assert "Coefficient" in w2.objects["eq1"].report().to_text()
    np.testing.assert_allclose(w2.objects["eq1"].result.params, wf.objects["eq1"].result.params)


def test_import_detects_dates_and_panels(tmp_path):
    p = tmp_path / "q.csv"
    pd.DataFrame({"date": ["2010Q1", "2010Q2", "2010Q3"], "v": [1, 2, 3]}).to_csv(p, index=False)
    it = Interpreter()
    it.run(f'import "{p}"')
    assert it.wf.freq == "q" and it.wf.labels()[0] == "2010Q1"
    p2 = tmp_path / "pan.csv"
    pd.DataFrame({"firm": ["a", "a", "b", "b"], "year": [2000, 2001, 2000, 2001], "v": [1, 2, 3, 4.0]}).to_csv(p2,
                                                                                                          index=False)
    it.run(f'wfopen "{p2}"')
    assert it.wf.is_panel


def test_object_views_through_commands(wf):
    it = Interpreter(wf)
    it.run("rndseed 1\nseries e = nrnd\nseries y = 1 + 0.5*x + e\nequation eq1.ls y c x")
    out = it.run("eq1.auto(2)")
    assert any("Breusch-Godfrey" in o.title for o in out if hasattr(o, "title"))
    out = it.run("eq1.wald c(2)=0.5")
    assert "Wald" in out[0].title
    assert it.ev.eval("eq1.@r2") > 0.5
    assert it.ev.eval("eq1.@coefs(2)") == pytest.approx(wf.objects["eq1"].result.params[1])


def test_resid_and_untitled_naming(wf):
    it = Interpreter(wf)
    it.run("series y = x + @trend^0.5")
    out = it.run("ls y c x")
    assert isinstance(out[0], ObjectOutput) and "resid" in wf.series_names()
    it.run("name myeq")
    assert "myeq" in wf.objects
