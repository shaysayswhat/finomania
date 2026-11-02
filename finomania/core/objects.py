"""Workfile objects other than series (groups, scalars, matrices, tables,
graphs and estimated models)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..report import Report

# object type -> (icon glyph, colour) used by the GUI and by `show` listings
OBJECT_TYPES = {
    "series": ("~", "#d9a400"),
    "alpha": ("a", "#9b6a00"),
    "group": ("G", "#2e7d32"),
    "equation": ("=", "#1565c0"),
    "var": ("V", "#6a1b9a"),
    "mgarch": ("Σ", "#ad1457"),
    "panel": ("P", "#00838f"),
    "factor": ("F", "#ef6c00"),
    "ml": ("N", "#4527a0"),
    "model": ("M", "#455a64"),
    "table": ("T", "#5d4037"),
    "graph": ("📈", "#c62828"),
    "scalar": ("#", "#37474f"),
    "matrix": ("[]", "#37474f"),
    "vector": ("v", "#37474f"),
}


class WFObject:
    type = "object"

    def __init__(self, name: str):
        self.name = name
        self.remarks = ""

    def describe(self) -> str:
        return self.type

    def to_dict(self) -> dict:
        return {"type": self.type, "name": self.name, "remarks": self.remarks}


class Group(WFObject):
    type = "group"

    def __init__(self, name: str, members: list[str]):
        super().__init__(name)
        self.members = list(members)

    def describe(self) -> str:
        return "group: " + " ".join(self.members)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["members"] = self.members
        return d


class Scalar(WFObject):
    type = "scalar"

    def __init__(self, name: str, value: float):
        super().__init__(name)
        self.value = float(value)

    def describe(self) -> str:
        return f"scalar = {self.value:.8g}"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["value"] = self.value
        return d


class Matrix(WFObject):
    type = "matrix"

    def __init__(self, name: str, values, rownames=None, colnames=None):
        super().__init__(name)
        a = np.atleast_2d(np.asarray(values, dtype=float))
        if np.ndim(values) == 1:
            a = a.T
            self.type = "vector"
        self.values = a
        self.rownames = list(rownames) if rownames is not None else [str(i + 1) for i in range(a.shape[0])]
        self.colnames = list(colnames) if colnames is not None else [f"C{j + 1}" for j in range(a.shape[1])]

    def describe(self) -> str:
        return f"{self.type} {self.values.shape[0]}x{self.values.shape[1]}"

    def report(self) -> Report:
        r = Report(f"{self.type.capitalize()}: {self.name.upper()}")
        rows = [[rn] + list(self.values[i]) for i, rn in enumerate(self.rownames)]
        r.table([""] + self.colnames, rows)
        return r

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(values=self.values.tolist(), rownames=self.rownames, colnames=self.colnames, mtype=self.type)
        return d


class Table(WFObject):
    type = "table"

    def __init__(self, name: str, report: Report):
        super().__init__(name)
        self.report_obj = report

    def describe(self) -> str:
        return "table: " + (self.report_obj.title or "")

    def report(self) -> Report:
        return self.report_obj

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["report"] = self.report_obj.to_dict()
        return d


class Graph(WFObject):
    type = "graph"

    def __init__(self, name: str, figure, spec: dict | None = None):
        super().__init__(name)
        self.figure = figure
        self.spec = spec or {}

    def describe(self) -> str:
        return "graph: " + self.spec.get("title", "")

    def to_dict(self) -> dict:
        from ..report import fig_to_b64
        d = super().to_dict()
        d["png"] = fig_to_b64(self.figure)
        d["spec"] = self.spec
        return d


class Estimation(WFObject):
    """A named estimated model (equation, VAR, DCC system, panel, factor, ML...)."""

    def __init__(self, name: str, result: Any, command: str = "", sample: str = "@all"):
        super().__init__(name)
        self.result = result
        self.command = command
        self.sample = sample
        self.type = getattr(result, "object_type", "equation")

    def describe(self) -> str:
        return f"{self.type}: {getattr(self.result, 'method_label', '')} {self.command}".strip()

    def report(self) -> Report:
        return self.result.report()

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(command=self.command, sample=self.sample, otype=self.type)
        try:
            d["report"] = self.result.report().to_dict()
        except Exception:  # pragma: no cover
            pass
        return d
