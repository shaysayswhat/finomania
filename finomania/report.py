"""
Report objects: the common output format of every Finomania procedure.

A Report is an ordered list of blocks (header lines, tables, two-column
statistic blocks, free text and figures).  It renders to

* Qt rich text   (``to_html(qt=True)``) for the desktop GUI,
* standalone HTML (``to_html()``) for export / batch reports,
* plain text     (``to_text()``) for the console,
* LaTeX          (``to_latex()``) for papers,
* CSV            (``to_csv()``) for spreadsheets.

The layout deliberately follows the familiar EViews look: a key/value
header, a ruled coefficient table and a two-column summary-statistics block.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import io
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# number formatting
# ----------------------------------------------------------------------------

def fmt_num(x: Any, sig: int = 7) -> str:
    """Format a number the way EViews does (about 7 significant digits)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "NA"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    ax = abs(x)
    if ax == 0:
        return "0.000000"
    if ax >= 1e9 or ax < 1e-4:
        s = f"{x:.{max(sig - 3, 2)}E}"
        mant, exp = s.split("E")
        return f"{mant}E{int(exp):+03d}"
    mag = int(math.floor(math.log10(ax)))
    dec = max(0, sig - 1 - max(mag, 0))
    dec = min(dec, 6)
    return f"{x:.{dec}f}"


def fmt_prob(x: Any) -> str:
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "NA"
    return f"{x:.4f}"


def fmt_int(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "NA"
        return str(int(round(float(x))))
    except (TypeError, ValueError):
        return str(x)


def fmt_fixed(d: int) -> Callable[[Any], str]:
    def f(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        if math.isnan(x):
            return "NA"
        return f"{x:.{d}f}"
    return f


def fmt_pct(x: Any) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "NA"
    return f"{100 * x:.2f}%"


FORMATTERS: dict[str, Callable[[Any], str]] = {
    "num": fmt_num,
    "prob": fmt_prob,
    "int": fmt_int,
    "str": lambda v: "" if v is None else str(v),
    "pct": fmt_pct,
    "f2": fmt_fixed(2),
    "f3": fmt_fixed(3),
    "f4": fmt_fixed(4),
}


def stars(p: float) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if math.isnan(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ----------------------------------------------------------------------------
# blocks
# ----------------------------------------------------------------------------

@dataclass
class HeaderBlock:
    lines: list[str]


@dataclass
class TableBlock:
    columns: list[str]
    rows: list[list[Any]]
    formats: list[Any] | None = None       # per column: key of FORMATTERS or callable
    align: list[str] | None = None         # 'l' | 'r' | 'c'
    title: str | None = None
    footnote: str | None = None
    rules_after: list[int] = field(default_factory=list)  # draw a rule after row i
    bold_rows: list[int] = field(default_factory=list)

    def cell_text(self, i: int, j: int) -> str:
        v = self.rows[i][j]
        if isinstance(v, str):
            return v
        f = None
        if self.formats is not None and j < len(self.formats):
            f = self.formats[j]
        if f is None:
            f = "num" if isinstance(v, (float, int, np.floating, np.integer)) and j > 0 else "str"
        if isinstance(f, str):
            f = FORMATTERS.get(f, fmt_num)
        return f(v)

    def aligns(self) -> list[str]:
        if self.align:
            return list(self.align) + ["r"] * (len(self.columns) - len(self.align))
        return ["l"] + ["r"] * (len(self.columns) - 1)


@dataclass
class StatsBlock:
    """EViews-style two-column summary statistics."""
    left: list[tuple[str, Any]]
    right: list[tuple[str, Any]]


@dataclass
class TextBlock:
    text: str
    style: str = "normal"   # normal | note | title | mono | warning


@dataclass
class FigureBlock:
    figure: Any            # matplotlib.figure.Figure
    caption: str = ""


class Report:
    """Container of output blocks.  Every estimation / test returns one."""

    def __init__(self, title: str = "", subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.blocks: list[Any] = []
        self.created = _dt.datetime.now()

    # ---- builders ---------------------------------------------------------
    def header(self, lines: Sequence[str] | Sequence[tuple[str, Any]], stamp: bool = False) -> "Report":
        out = []
        for ln in lines:
            if isinstance(ln, tuple):
                k, v = ln
                out.append(f"{k}: {v}")
            else:
                out.append(str(ln))
        if stamp:
            now = self.created
            out.insert(min(2, len(out)), f"Date: {now:%m/%d/%y}   Time: {now:%H:%M}")
        self.blocks.append(HeaderBlock(out))
        return self

    def table(self, columns, rows, formats=None, align=None, title=None, footnote=None,
              rules_after=None, bold_rows=None) -> "Report":
        self.blocks.append(TableBlock(list(columns), [list(r) for r in rows], formats, align,
                                      title, footnote, list(rules_after or []), list(bold_rows or [])))
        return self

    def coef_table(self, names, coef, se, stat, pval, stat_label="t-Statistic", title=None,
                   section_breaks: dict[int, str] | None = None) -> "Report":
        """Standard 'Variable | Coefficient | Std. Error | t-Statistic | Prob.' table.

        section_breaks maps a row index to a section title inserted before it
        (used e.g. for the 'Variance Equation' part of GARCH output)."""
        rows, rules, bold = [], [], []
        section_breaks = section_breaks or {}
        coef, se, stat, pval = (None if a is None else np.asarray(a) for a in (coef, se, stat, pval))
        for i, n in enumerate(names):
            if i in section_breaks:
                if rows:
                    rules.append(len(rows) - 1)
                rows.append([section_breaks[i], "", "", "", ""])
                bold.append(len(rows) - 1)
                rules.append(len(rows) - 1)
            rows.append([n, _f(coef, i), _f(se, i), _f(stat, i), _f(pval, i)])
        self.blocks.append(TableBlock(["Variable", "Coefficient", "Std. Error", stat_label, "Prob."],
                                      rows, ["str", "num", "num", "num", "prob"], None, title,
                                      None, rules, bold))
        return self

    def stats(self, left, right=None) -> "Report":
        right = list(right or [])
        left = list(left)
        n = max(len(left), len(right))
        left += [("", "")] * (n - len(left))
        right += [("", "")] * (n - len(right))
        self.blocks.append(StatsBlock(left, right))
        return self

    def text(self, s: str, style: str = "normal") -> "Report":
        self.blocks.append(TextBlock(s, style))
        return self

    def note(self, s: str) -> "Report":
        return self.text(s, "note")

    def warn(self, s: str) -> "Report":
        return self.text(s, "warning")

    def section(self, s: str) -> "Report":
        return self.text(s, "title")

    def figure(self, fig, caption: str = "") -> "Report":
        self.blocks.append(FigureBlock(fig, caption))
        return self

    def extend(self, other: "Report", with_title: bool = True) -> "Report":
        if with_title and other.title:
            self.section(other.title)
        self.blocks.extend(other.blocks)
        return self

    # ---- renderers ----------------------------------------------------------
    def to_text(self, width: int | None = None) -> str:
        out: list[str] = []
        if self.title:
            out.append(self.title)
            out.append("=" * max(len(self.title), 20))
        for b in self.blocks:
            if isinstance(b, HeaderBlock):
                out.extend(b.lines)
                out.append("")
            elif isinstance(b, TableBlock):
                out.append(_table_text(b))
                out.append("")
            elif isinstance(b, StatsBlock):
                out.append(_stats_text(b))
                out.append("")
            elif isinstance(b, TextBlock):
                if b.style == "title":
                    out.append("")
                    out.append(b.text)
                    out.append("-" * len(b.text))
                else:
                    out.append(b.text)
                    out.append("")
            elif isinstance(b, FigureBlock):
                out.append(f"[figure{': ' + b.caption if b.caption else ''}]")
        return "\n".join(out).rstrip() + "\n"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.to_text()

    def _repr_html_(self) -> str:  # Jupyter
        return self.to_html(standalone=False)

    def to_html(self, qt: bool = False, standalone: bool = True, embed_figures: bool = True) -> str:
        parts: list[str] = []
        if qt:
            return _to_qt_html(self)
        if self.title:
            parts.append(f"<h2>{_e(self.title)}</h2>")
        for b in self.blocks:
            if isinstance(b, HeaderBlock):
                parts.append('<div class="hdr">' + "<br>".join(_e(x) for x in b.lines) + "</div>")
            elif isinstance(b, TableBlock):
                parts.append(_table_html(b))
            elif isinstance(b, StatsBlock):
                parts.append(_stats_html(b))
            elif isinstance(b, TextBlock):
                if b.style == "title":
                    parts.append(f"<h3>{_e(b.text)}</h3>")
                elif b.style == "mono":
                    parts.append(f"<pre>{_e(b.text)}</pre>")
                else:
                    cls = {"note": "note", "warning": "warn"}.get(b.style, "txt")
                    parts.append(f'<p class="{cls}">{_e(b.text).replace(chr(10), "<br>")}</p>')
            elif isinstance(b, FigureBlock) and embed_figures:
                parts.append(f'<figure><img src="data:image/png;base64,{fig_to_b64(b.figure)}">'
                             f'<figcaption>{_e(b.caption)}</figcaption></figure>')
        body = "\n".join(parts)
        if not standalone:
            return f'<div class="fino">{HTML_CSS_INLINE}{body}</div>'
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_e(self.title or 'Output')}"
                f"</title>{HTML_CSS_INLINE}</head><body><div class='fino'>{body}</div></body></html>")

    def to_latex(self) -> str:
        out = []
        for b in self.blocks:
            if isinstance(b, HeaderBlock):
                out.append("% " + " | ".join(b.lines))
            elif isinstance(b, TableBlock):
                al = "".join(b.aligns())
                out.append("\\begin{tabular}{" + al + "}\n\\hline")
                out.append(" & ".join(_tex(c) for c in b.columns) + " \\\\ \\hline")
                for i in range(len(b.rows)):
                    out.append(" & ".join(_tex(b.cell_text(i, j)) for j in range(len(b.columns))) + " \\\\")
                    if i in b.rules_after:
                        out.append("\\hline")
                out.append("\\hline\n\\end{tabular}\n")
            elif isinstance(b, StatsBlock):
                out.append("\\begin{tabular}{lrlr}\n\\hline")
                for (k1, v1), (k2, v2) in zip(b.left, b.right):
                    out.append(f"{_tex(k1)} & {_tex(_sv(v1, k1))} & {_tex(k2)} & {_tex(_sv(v2, k2))} \\\\")
                out.append("\\hline\n\\end{tabular}\n")
            elif isinstance(b, TextBlock):
                out.append(_tex(b.text) + "\n")
        return "\n".join(out)

    def to_csv(self) -> str:
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        if self.title:
            w.writerow([self.title])
        for b in self.blocks:
            if isinstance(b, HeaderBlock):
                for ln in b.lines:
                    w.writerow([ln])
            elif isinstance(b, TableBlock):
                if b.title:
                    w.writerow([b.title])
                w.writerow(b.columns)
                for r in b.rows:
                    w.writerow(["" if v is None else v for v in r])
            elif isinstance(b, StatsBlock):
                for (k1, v1), (k2, v2) in zip(b.left, b.right):
                    w.writerow([k1, v1, k2, v2])
            elif isinstance(b, TextBlock):
                w.writerow([b.text])
            w.writerow([])
        return buf.getvalue()

    # ---- serialisation (for saving frozen tables inside workfiles) ----------
    def to_dict(self) -> dict:
        blocks = []
        for b in self.blocks:
            if isinstance(b, HeaderBlock):
                blocks.append({"t": "h", "lines": b.lines})
            elif isinstance(b, TableBlock):
                blocks.append({"t": "tab", "columns": b.columns,
                               "rows": [[b.cell_text(i, j) for j in range(len(b.columns))]
                                        for i in range(len(b.rows))],
                               "align": b.aligns(), "title": b.title, "footnote": b.footnote,
                               "rules": b.rules_after, "bold": b.bold_rows})
            elif isinstance(b, StatsBlock):
                blocks.append({"t": "st", "left": [[k, _sv(v, k)] for k, v in b.left],
                               "right": [[k, _sv(v, k)] for k, v in b.right]})
            elif isinstance(b, TextBlock):
                blocks.append({"t": "tx", "text": b.text, "style": b.style})
            elif isinstance(b, FigureBlock):
                blocks.append({"t": "fig", "png": fig_to_b64(b.figure), "caption": b.caption})
        return {"title": self.title, "subtitle": self.subtitle, "blocks": blocks}

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        r = cls(d.get("title", ""), d.get("subtitle", ""))
        for b in d.get("blocks", []):
            t = b["t"]
            if t == "h":
                r.blocks.append(HeaderBlock(b["lines"]))
            elif t == "tab":
                r.blocks.append(TableBlock(b["columns"], b["rows"], None, b.get("align"), b.get("title"),
                                           b.get("footnote"), b.get("rules", []), b.get("bold", [])))
            elif t == "st":
                r.blocks.append(StatsBlock([tuple(x) for x in b["left"]], [tuple(x) for x in b["right"]]))
            elif t == "tx":
                r.blocks.append(TextBlock(b["text"], b.get("style", "normal")))
            elif t == "fig":
                r.blocks.append(FigureBlock(_PngFigure(b["png"]), b.get("caption", "")))
        return r


class _PngFigure:
    """Stand-in for a matplotlib figure restored from a saved workfile."""

    def __init__(self, b64: str):
        self.b64 = b64

    def savefig(self, buf, format="png", **kw):  # noqa: A002
        buf.write(base64.b64decode(self.b64))


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _f(arr, i):
    if arr is None:
        return ""
    try:
        return arr[i]
    except (IndexError, KeyError, TypeError):
        return ""


def _sv(v, key: str = ""):
    if isinstance(v, str):
        return v
    if key and key.lower().startswith("prob") and isinstance(v, (float, np.floating)) and not math.isnan(v):
        return f"{float(v):.6f}"
    return fmt_num(v)


def _e(s: Any) -> str:
    return _html.escape(str(s))


def _tex(s: Any) -> str:
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _table_text(b: TableBlock) -> str:
    ncol = len(b.columns)
    cells = [[b.cell_text(i, j) for j in range(ncol)] for i in range(len(b.rows))]
    widths = [len(str(c)) for c in b.columns]
    for r in cells:
        for j, c in enumerate(r):
            widths[j] = max(widths[j], len(c))
    widths = [w + 2 for w in widths]
    al = b.aligns()

    def fmt_row(r):
        out = []
        for j, c in enumerate(r):
            out.append(c.ljust(widths[j]) if al[j] == "l" else c.rjust(widths[j]) if al[j] == "r"
                       else c.center(widths[j]))
        return "".join(out)

    total = sum(widths)
    lines = []
    if b.title:
        lines.append(b.title)
    lines.append("=" * total)
    lines.append(fmt_row([str(c) for c in b.columns]))
    lines.append("=" * total)
    for i, r in enumerate(cells):
        lines.append(fmt_row(r))
        if i in b.rules_after and i != len(cells) - 1:
            lines.append("-" * total)
    lines.append("=" * total)
    if b.footnote:
        lines.append(b.footnote)
    return "\n".join(lines)


def _stats_text(b: StatsBlock) -> str:
    rows = [(k1, _sv(v1, k1), k2, _sv(v2, k2)) for (k1, v1), (k2, v2) in zip(b.left, b.right)]
    w0 = max((len(r[0]) for r in rows), default=0) + 2
    w1 = max((len(r[1]) for r in rows), default=0) + 2
    w2 = max((len(r[2]) for r in rows), default=0) + 4
    w3 = max((len(r[3]) for r in rows), default=0) + 2
    total = w0 + w1 + w2 + w3
    lines = ["=" * total]
    for r in rows:
        lines.append(r[0].ljust(w0) + r[1].rjust(w1) + ("    " + r[2]).ljust(w2) + r[3].rjust(w3))
    lines.append("=" * total)
    return "\n".join(lines)


HTML_CSS_INLINE = """<style>
.fino{font-family:'Segoe UI',Arial,Helvetica,sans-serif;font-size:13px;color:#111;}
.fino h2{font-size:16px;margin:4px 0 8px 0;color:#0b3a6e}
.fino h3{font-size:14px;margin:14px 0 6px 0;color:#0b3a6e}
.fino .hdr{margin:4px 0 8px 0;line-height:1.45}
.fino table.t{border-collapse:collapse;margin:6px 0 10px 0;min-width:520px}
.fino table.t th{border-top:2px solid #222;border-bottom:1px solid #222;padding:3px 12px;font-weight:600}
.fino table.t td{padding:2px 12px;white-space:nowrap}
.fino table.t tr.rule td{border-bottom:1px solid #999}
.fino table.t tr.b td{font-weight:600}
.fino table.t tbody tr:last-child td{border-bottom:2px solid #222}
.fino .r{text-align:right;font-variant-numeric:tabular-nums}.fino .l{text-align:left}.fino .c{text-align:center}
.fino .cap{font-weight:600;margin-top:8px}
.fino .note{color:#555;font-style:italic}.fino .warn{color:#a33}
.fino figure img{max-width:100%}
.fino pre{font-family:Consolas,monospace;font-size:12px}
</style>"""


def _table_html(b: TableBlock) -> str:
    al = b.aligns()
    h = []
    if b.title:
        h.append(f'<div class="cap">{_e(b.title)}</div>')
    h.append('<table class="t"><thead><tr>')
    for j, c in enumerate(b.columns):
        h.append(f'<th class="{al[j]}">{_e(c)}</th>')
    h.append("</tr></thead><tbody>")
    for i in range(len(b.rows)):
        cls = []
        if i in b.rules_after:
            cls.append("rule")
        if i in b.bold_rows:
            cls.append("b")
        h.append(f'<tr class="{" ".join(cls)}">')
        for j in range(len(b.columns)):
            h.append(f'<td class="{al[j]}">{_e(b.cell_text(i, j))}</td>')
        h.append("</tr>")
    h.append("</tbody></table>")
    if b.footnote:
        h.append(f'<div class="note">{_e(b.footnote)}</div>')
    return "".join(h)


def _stats_html(b: StatsBlock) -> str:
    h = ['<table class="t"><thead><tr><th class="l"></th><th class="r"></th><th class="l"></th>'
         '<th class="r"></th></tr></thead><tbody>']
    for (k1, v1), (k2, v2) in zip(b.left, b.right):
        h.append(f'<tr><td class="l">{_e(k1)}</td><td class="r">{_e(_sv(v1, k1))}</td>'
                 f'<td class="l" style="padding-left:28px">{_e(k2)}</td><td class="r">{_e(_sv(v2, k2))}</td></tr>')
    h.append("</tbody></table>")
    return "".join(h)


# ---- Qt rich-text flavour (QTextBrowser supports only a subset of HTML/CSS) --

def _to_qt_html(rep: Report) -> str:
    font = "font-family:'Segoe UI',Arial; font-size:10pt;"
    parts = [f'<div style="{font}">']
    if rep.title:
        parts.append(f'<p style="font-size:12pt; font-weight:600; color:#0b3a6e; margin:2px 0 6px 0">'
                     f'{_e(rep.title)}</p>')
    for b in rep.blocks:
        if isinstance(b, HeaderBlock):
            parts.append('<p style="margin:2px 0 6px 0">' + "<br>".join(_e(x) for x in b.lines) + "</p>")
        elif isinstance(b, TableBlock):
            parts.append(_qt_table(b))
        elif isinstance(b, StatsBlock):
            parts.append(_qt_stats(b))
        elif isinstance(b, TextBlock):
            if b.style == "title":
                parts.append(f'<p style="font-size:11pt; font-weight:600; color:#0b3a6e; margin:12px 0 4px 0">'
                             f'{_e(b.text)}</p>')
            elif b.style == "mono":
                parts.append(f"<pre>{_e(b.text)}</pre>")
            elif b.style == "note":
                parts.append(f'<p style="color:#555; font-style:italic">{_e(b.text).replace(chr(10), "<br>")}</p>')
            elif b.style == "warning":
                parts.append(f'<p style="color:#b03030">{_e(b.text).replace(chr(10), "<br>")}</p>')
            else:
                parts.append(f'<p>{_e(b.text).replace(chr(10), "<br>")}</p>')
        elif isinstance(b, FigureBlock):
            parts.append(f'<p><img src="data:image/png;base64,{fig_to_b64(b.figure)}"></p>')
    parts.append("</div>")
    return "".join(parts)


def _qt_table(b: TableBlock) -> str:
    al = b.aligns()
    amap = {"l": "left", "r": "right", "c": "center"}
    n = len(b.columns)
    nr = len(b.rows)
    h = []
    if b.title:
        h.append(f'<p style="font-weight:600; margin:8px 0 2px 0">{_e(b.title)}</p>')
    h.append('<table cellspacing="0" cellpadding="0" style="border-collapse:collapse; margin-bottom:10px">')
    hs = "border-top:2px solid #222; border-bottom:1px solid #222; padding:3px 10px; white-space:nowrap"
    h.append("<tr>" + "".join(f'<td align="{amap[al[j]]}" style="{hs}"><b>{_e(c)}</b></td>'
                              for j, c in enumerate(b.columns)) + "</tr>")
    for i in range(nr):
        bold = i in b.bold_rows
        st = "padding:2px 10px; white-space:nowrap"
        if i == nr - 1:
            st += "; border-bottom:2px solid #222"
        elif i in b.rules_after:
            st += "; border-bottom:1px solid #9aa0a8"
        tds = []
        for j in range(n):
            t = _e(b.cell_text(i, j))
            if bold:
                t = f"<b>{t}</b>"
            tds.append(f'<td align="{amap[al[j]]}" style="{st}">{t}</td>')
        h.append("<tr>" + "".join(tds) + "</tr>")
    if nr == 0:
        h.append(f'<tr><td colspan="{n}" style="border-bottom:2px solid #222"></td></tr>')
    h.append("</table>")
    if b.footnote:
        h.append(f'<p style="color:#555; font-style:italic">{_e(b.footnote)}</p>')
    return "".join(h)


def _qt_stats(b: StatsBlock) -> str:
    h = ['<table cellspacing="0" cellpadding="0" style="border-collapse:collapse; margin-bottom:10px">']
    rows = list(zip(b.left, b.right))
    for i, ((k1, v1), (k2, v2)) in enumerate(rows):
        st = "padding:2px 8px; white-space:nowrap"
        if i == 0:
            st += "; border-top:2px solid #222"
        if i == len(rows) - 1:
            st += "; border-bottom:2px solid #222"
        h.append(f'<tr><td style="{st}">{_e(k1)}</td><td align="right" style="{st}">{_e(_sv(v1, k1))}</td>'
                 f'<td style="{st}; padding-left:34px">{_e(k2)}</td>'
                 f'<td align="right" style="{st}">{_e(_sv(v2, k2))}</td></tr>')
    h.append("</table>")
    return "".join(h)


# ----------------------------------------------------------------------------
# a light-weight output wrapper for figures produced by views
# ----------------------------------------------------------------------------

@dataclass
class FigureOutput:
    figure: Any
    title: str = "Graph"

    def to_report(self) -> Report:
        r = Report(self.title)
        r.figure(self.figure, self.title)
        return r


@dataclass
class Message:
    text: str
    level: str = "info"   # info | warning | error

    def __str__(self) -> str:
        return self.text
