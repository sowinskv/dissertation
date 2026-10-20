"""
plotstyle: one consistent, clean look for every figure in the thesis, matching the lab notebook
(white paper, black ink, pencil gray, Helvetica-style type, one pen color per method).

white surface, made for the thesis PDF (prints cleanly, also in grayscale).

the rules this module encodes (so individual plot scripts don't have to):
  - every method has ONE fixed color + marker across all figures (see METHODS)
  - FCI (the exact reference) is always ink, HF (the baseline) is always muted gray,
    the models being studied carry the color
  - identity never relies on color alone: distinct markers / line styles + direct labels
  - quiet chrome: black left/bottom axis lines, horizontal hairline grid, no box
  - a centred header (title in regular weight, subtitle, key) and footer, like the notebook pages
  - english labels in sentence case, kept minimal; decimal point by default
    (decimal_comma=True switches to polish)
  - type: Arimo (bundled in assets/fonts, same letter widths as Helvetica), so the PDF looks the
    same on every machine; sub- and superscripts (H₂, 10⁻³) are drawn with mathtext in that font

the method colors are "pens", checked with a CVD validator (OKLab, light surface):
  RQ1 group  UCCSD / HEA / CCSD            all pairs: CVD dE >= 11, normal dE >= 27
  RQ3 group  QNN-cond / MLP / GP           all pairs: CVD dE >= 12, normal dE >= 21
  all six in slot order                    adjacent pairs pass, contrast >= 3:1 on white
every series still gets a marker shape and a direct label, so identity never rests on color alone.

usage:
    from src import plotstyle as ps
    ps.use()
    fig, (ax1, ax2) = ps.figure(nrows=2, height_ratios=[1.3, 1], sharex=True)
    ...
    ps.header(fig, "title", "subtitle", key=["FCI", "HF", "UCCSD"])
    ps.footer(fig, "source: ...")
    ps.save(fig, "results/exp01_h2_curve")
"""

import math
import textwrap
from pathlib import Path

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

# ---------------- theme ----------------
THEME = {
    "surface": "#ffffff",   # paper
    "ink": "#000000",       # titles, reference series (FCI), axis lines
    "ink_2": "#333333",     # labels, tick labels
    "muted": "#8a8a8a",     # pencil: notes, baseline series (HF)
    "grid": "#e6e6e6",      # hairline gridlines
    "baseline": "#000000",  # axis lines
}

# validated pen colors, assigned in this fixed order, never cycled
SLOTS = ["#1f4fd1", "#e0482a", "#0f9a7a", "#ad7400", "#c4398c", "#5a5fd6"]

# ---------------- fixed identity of every method ----------------
# color is a theme role ("ink", "muted") or a slot index; it never changes between figures
METHODS = {
    "FCI":      {"color": "ink",   "marker": None, "ls": "-",  "label": "FCI (exact)"},
    "HF":       {"color": "muted", "marker": None, "ls": "--", "label": "Hartree–Fock"},
    "UCCSD":    {"color": 0,       "marker": "s",  "ls": "-",  "label": "VQE-UCCSD"},
    "HEA":      {"color": 1,       "marker": "^",  "ls": "-",  "label": "VQE-HEA"},
    "CCSD":     {"color": 2,       "marker": "D",  "ls": "-",  "label": "CCSD"},
    "QNN-cond": {"color": 3,       "marker": "P",  "ls": "-",  "label": "conditional QNN"},
    "MLP":      {"color": 4,       "marker": "v",  "ls": "-",  "label": "MLP"},
    "GP":       {"color": 5,       "marker": "X",  "ls": "-",  "label": "Gaussian process"},
}

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT = "Arimo"
TEXT_WIDTH_IN = 6.3   # A4 text width with ~2.5 cm margins
CHEM_ACC_MHA = 1.6    # chemical accuracy in mHa

_state = {"decimal_comma": False}


# ---------------- setup ----------------
def use(decimal_comma=False):
    """apply the style globally. call once at the top of a plot script."""
    t = THEME
    _state.update(decimal_comma=decimal_comma)

    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))

    mpl.rcParams.update({
        "font.family": [FONT, "DejaVu Sans"],
        "mathtext.fontset": "custom",       # 10^{-3}, H_2 in the same typeface
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "mathtext.sf": FONT,
        "mathtext.cal": FONT,
        "mathtext.default": "regular",
        "font.size": 8.5,
        "text.color": t["ink_2"],
        "axes.facecolor": t["surface"],
        "figure.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["baseline"],
        "axes.linewidth": 0.6,
        "axes.labelcolor": t["ink_2"],
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": t["ink"],
        "ytick.color": t["ink"],
        "xtick.labelcolor": t["ink_2"],
        "ytick.labelcolor": t["ink_2"],
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.size": 3,
        "ytick.major.width": 0.6,
        "ytick.minor.size": 0,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,     # embed real fonts in the PDF (text stays selectable)
        "savefig.dpi": 300,
    })


def theme():
    return THEME


def style(method):
    """matplotlib kwargs for a method: color, marker, line style, label."""
    m = METHODS[method]
    c = m["color"]
    color = SLOTS[c] if isinstance(c, int) else theme()[c]
    return {"color": color, "marker": m["marker"], "ls": m["ls"], "label": m["label"]}


def color(method):
    return style(method)["color"]


def shades(method, n, lightest=0.7):
    """n steps of one method's color, light to full: a sequential ramp for an ordered variable
    (e.g. number of layers) that keeps the method's identity."""
    rgb = mpl.colors.to_rgb(color(method))
    return [mpl.colors.to_hex([c + (1 - c) * t for c in rgb])
            for t in np.linspace(lightest, 0, n)] if n > 1 else [color(method)]


_SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def tex(text):
    """unicode sub/superscripts (H₂, 10⁻³) -> mathtext, so they render in the figure font."""
    import re
    if text is None:
        return None
    text = re.sub(r"[₀-₉]+", lambda m: "$_{" + m.group().translate(_SUBS) + "}$", text)
    return re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "$^{" + m.group().translate(_SUPS) + "}$", text)


# ---------------- layout ----------------
# everything is placed in inches from the top / bottom edge, so the header and footer
# never collide with the panels, whatever the figure size or number of text lines.
PANEL_TITLE_IN = 0.55   # room above each panel for panel_title()
XLABEL_IN = 0.45        # room below the last panel for tick labels + axis label


def figure(nrows=1, ncols=1, width=TEXT_WIDTH_IN, panel_height=2.2, height_ratios=None,
           hspace_in=None, **subplot_kw):
    """figure whose header / footer space is sized automatically by header() and footer()."""
    ratios = height_ratios or [1] * nrows
    hspace_in = PANEL_TITLE_IN + 0.10 if hspace_in is None else hspace_in
    panels_in = panel_height * sum(ratios) / max(ratios) + hspace_in * (nrows - 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, panels_in + 2),
                             gridspec_kw={"height_ratios": ratios}, **subplot_kw)
    fig._ps = {"panels_in": panels_in, "hspace_in": hspace_in, "nrows": nrows,
               "header_in": PANEL_TITLE_IN, "footer_in": 0.1, "top": [], "bottom": []}
    _relayout(fig)
    return fig, axes


def _relayout(fig):
    L = fig._ps
    height = L["header_in"] + L["panels_in"] + XLABEL_IN + L["footer_in"]
    fig.set_figheight(height)
    fig.subplots_adjust(left=0.1, right=0.83,
                        top=1 - L["header_in"] / height,
                        bottom=(XLABEL_IN + L["footer_in"]) / height,
                        hspace=L["hspace_in"] / ((L["panels_in"] - L["hspace_in"] * (L["nrows"] - 1))
                                                 / L["nrows"]))
    for artist, inches in L["top"]:
        artist.set_y(1 - inches / height)
    for artist, inches in L["bottom"]:
        artist.set_y(inches / height)


def _wrap(fig, text, fontsize):
    """wrap text to the figure's usable width (approximate, from average glyph width).
    text is centred, so it keeps the same margin on both sides."""
    usable_in = fig.get_figwidth() * (1 - 2 * fig.subplotpars.left) - 0.15
    chars = max(20, int(usable_in / (fontsize * 0.53 / 72)))
    return "\n".join(textwrap.fill(par, chars) for par in text.split("\n"))


def header(fig, title, subtitle=None, key=None):
    """header, centred like the lab notebook pages: title in regular weight, gray subtitle,
    horizontal key (method names or Line2D handles)."""
    t, L = theme(), fig._ps
    x = 0.5
    y_in = 0.16
    if title:   # pass None when the title goes into the thesis caption instead
        title = tex(_wrap(fig, title, 13))
        a = fig.text(x, 0, title, ha="center", va="top", fontsize=13, color=t["ink"],
                     linespacing=1.15, multialignment="center")
        L["top"].append((a, y_in))
        y_in += 0.32 + 0.24 * title.count("\n")
    if subtitle:
        sub = tex(_wrap(fig, subtitle, 8.5))
        a = fig.text(x, 0, sub, ha="center", va="top", fontsize=8.5, color=t["ink_2"],
                     linespacing=1.3, multialignment="center")
        L["top"].append((a, y_in))
        y_in += 0.17 * (sub.count("\n") + 1) + 0.08
    if key:
        handles = []
        for m in key:
            if isinstance(m, Line2D):     # a ready-made handle, e.g. shades of one method
                handles.append(m)
                continue
            s = style(m)
            handles.append(Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                                  markersize=4.5, lw=1.6, label=tex(s["label"]),
                                  markerfacecolor=s["color"], markeredgecolor=t["surface"]))
        leg = fig.legend(handles=handles, loc="upper center", ncol=len(key),
                         bbox_to_anchor=(x, 1), fontsize=7.5, handlelength=2.0,
                         columnspacing=1.4, handletextpad=0.5, borderaxespad=0,
                         labelcolor=t["ink_2"])
        L["top"].append((_LegendY(leg, x), y_in))
        y_in += 0.22
    L["header_in"] = y_in + PANEL_TITLE_IN
    _relayout(fig)


class _LegendY:
    """adapter so a legend can be repositioned like a text (set_y in figure fraction)."""
    def __init__(self, legend, x):
        self.legend, self.x = legend, x

    def set_y(self, y):
        self.legend.set_bbox_to_anchor((self.x, y), transform=self.legend.figure.transFigure)


def footer(fig, text):
    """small muted note at the bottom: source, data, caveats."""
    L = fig._ps
    note = tex(_wrap(fig, text, 6.5))
    a = fig.text(0.5, 0, note, ha="center", va="bottom", fontsize=6.5,
                 color=theme()["muted"], linespacing=1.2, multialignment="center")
    L["bottom"].append((a, 0.12))
    L["footer_in"] = 0.12 + 0.13 * (note.count("\n") + 1) + 0.05
    _relayout(fig)


def panel_title(ax, title, subtitle=None):
    """small bold panel title, centred above its axis (offsets in points)."""
    t = theme()
    ax.annotate(tex(title), (0.5, 1), xycoords="axes fraction",
                xytext=(0, 21 if subtitle else 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=t["ink"])
    if subtitle:
        ax.annotate(tex(subtitle), (0.5, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=t["muted"])


# ---------------- marks & annotations ----------------
def plot(ax, x, y, method, markevery=None, **kw):
    """line + markers for a method, markers ringed in the surface color."""
    s = style(method)
    kw = {"color": s["color"], "ls": s["ls"], "marker": s["marker"], "label": s["label"],
          "markerfacecolor": s["color"], "markeredgecolor": theme()["surface"],
          "markeredgewidth": 0.8, "markevery": markevery, **kw}
    return ax.plot(x, y, **kw)


def end_label(ax, x, y, text, dy=0, dx_pt=6):
    """direct label at the right end of a series, in text ink (never the series color)."""
    ax.annotate(tex(text), (x, y), xytext=(dx_pt, dy), textcoords="offset points",
                ha="left", va="center", fontsize=7.5, color=theme()["ink_2"],
                annotation_clip=False)


def end_labels(ax, x, items, min_gap_pt=10, dx_pt=6):
    """direct labels for several series ending at x. labels that would overlap are spread
    apart vertically and tied to their line end with a thin leader (never stacked blindly).
    items: list of (y, text)."""
    t = theme()
    to_px = ax.transData.transform
    pts_per_px = 72 / ax.figure.dpi
    ys = [(to_px((x, y))[1] * pts_per_px, y, text) for y, text in items]
    ys.sort()
    placed = []
    for y_pt, y, text in ys:
        target = y_pt if not placed else max(y_pt, placed[-1][0] + min_gap_pt)
        placed.append([target, y_pt, y, text])
    # re-center the block so labels stay as close to their lines as possible
    shift = sum(p[1] - p[0] for p in placed) / len(placed)
    if shift < 0:
        for p in placed:
            p[0] += shift
        for i in range(1, len(placed)):
            placed[i][0] = max(placed[i][0], placed[i - 1][0] + min_gap_pt)
    for target, y_pt, y, text in placed:
        moved = abs(target - y_pt) > 1
        ax.annotate(tex(text), (x, y), xytext=(dx_pt + (8 if moved else 0), target - y_pt),
                    textcoords="offset points", ha="left", va="center", fontsize=7.5,
                    color=t["ink_2"], annotation_clip=False,
                    arrowprops=({"arrowstyle": "-", "color": t["muted"], "lw": 0.6,
                                 "shrinkA": 0, "shrinkB": 2} if moved else None))


def callout(ax, xy, text, xytext, ha="left", va="center"):
    """annotation with a thin leader line, editorial style."""
    t = theme()
    ax.annotate(tex(text), xy, xytext=xytext, textcoords="data", ha=ha, va=va,
                fontsize=7, color=t["ink_2"], linespacing=1.2,
                arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 0.7,
                            "shrinkA": 2, "shrinkB": 3})


def bracket(ax, x, y_low, y_high, text, side="right"):
    """vertical bracket measuring a gap between two values, with a label."""
    t = theme()
    ax.annotate("", (x, y_low), xytext=(x, y_high),
                arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25",
                            "color": t["ink_2"], "lw": 0.8})
    ax.annotate(tex(text), (x, (y_low + y_high) / 2), xytext=(6 if side == "right" else -6, 0),
                textcoords="offset points", ha="left" if side == "right" else "right",
                va="center", fontsize=7, color=t["ink_2"], linespacing=1.2)


def threshold(ax, y, text, x=0.01, above=False):
    """horizontal threshold line (e.g. chemical accuracy) with a label just below (or above) it.
    x is the label position as a fraction of the axis width."""
    t = theme()
    ax.axhline(y, color=t["ink"], lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.annotate(tex(text), (x, y), xycoords=ax.get_yaxis_transform(), xytext=(0, 3 if above else -3),
                textcoords="offset points", ha="left", va="bottom" if above else "top", fontsize=7,
                color=t["ink_2"], zorder=2,   # under the axis lines, so its backing never cuts them
                bbox={"facecolor": t["surface"], "edgecolor": "none", "pad": 1.2})


# ---------------- number formatting ----------------
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _num(v, decimals=None):
    s = f"{v:.{decimals}f}" if decimals is not None else f"{v:g}"
    s = s.replace("-", "−")
    return s.replace(".", ",") if _state["decimal_comma"] else s


def sci(v, digits=1):
    """compact scientific notation for labels: 5,2·10⁻¹⁰ (polish decimal comma)."""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = round(v / 10 ** e, digits)
    mant = _num(m, digits) if digits else str(int(m))
    return f"{mant}·$10^{{{str(e).replace('-', '−')}}}$" if e != 0 else mant


def number_axis(ax, axis="y", decimals=None):
    """plain ticks with polish decimal comma and a true minus sign."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda v, _: _num(v, decimals)))


def log_axis(ax, axis="y", every=1):
    """log scale with clean 10ⁿ labels, ticks every `every` decades, no minor ticks."""
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target = ax.yaxis if axis == "y" else ax.xaxis

    def fmt(v, _):
        e = int(round(math.log10(v)))
        return {0: "1", 1: "10"}.get(e, f"$10^{{{str(e).replace('-', '−')}}}$")

    target.set_major_locator(LogLocator(base=10.0 ** every, numticks=30))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(fmt))


# ---------------- output ----------------
def save(fig, path_stem, formats=("pdf", "png")):
    """save as vector PDF (for the thesis) and PNG (for previews / slides)."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        # no creation timestamp in the PDF, so re-running a plot doesn't produce a git diff
        meta = {"CreationDate": None} if f == "pdf" else None
        fig.savefig(stem.with_suffix(f".{f}"), metadata=meta)
    print("saved:", ", ".join(str(stem.with_suffix(f".{f}")) for f in formats))
