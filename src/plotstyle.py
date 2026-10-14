"""
plotstyle: one consistent, editorial-but-rigorous look for every figure in the thesis.

white surface, made for the thesis PDF (prints cleanly, also in grayscale).

the rules this module encodes (so individual plot scripts don't have to):
  - every method has ONE fixed color + marker across all figures (see METHODS)
  - FCI (the exact reference) is always ink, HF (the baseline) is always muted gray,
    the models being studied carry the color
  - identity never relies on color alone: distinct markers / line styles + direct labels
  - recessive chrome: horizontal hairline grid only, no box, muted axis text
  - a header (title, subtitle, key) and a footer (source / notes) outside the plot area
  - polish number formatting (decimal comma) by default, because the thesis is in polish

the categorical colors are the first slots of a validated palette:
all-pairs CVD separation >= 9 (deutan) and normal-vision separation >= 24 on white.
the light slots (aqua, yellow, magenta) sit below 3:1 contrast, which is why every
series also gets a marker shape and a direct label.

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
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

# ---------------- theme ----------------
THEME = {
    "surface": "#ffffff",
    "ink": "#141414",       # titles, reference series
    "ink_2": "#4d4b46",     # subtitles, labels
    "muted": "#8a867d",     # axis text, notes, baseline series
    "grid": "#e6e3dc",      # hairline gridlines
    "baseline": "#b9b5ab",  # bottom axis line
    "band": "#efece5",      # shaded zones (e.g. chemical accuracy)
}

# validated categorical slots, assigned in this fixed order, never cycled
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# ---------------- fixed identity of every method ----------------
# color is a theme role ("ink", "muted") or a slot index; it never changes between figures
METHODS = {
    "FCI":      {"color": "ink",   "marker": None, "ls": "-",  "label": "FCI (dokładne)"},
    "HF":       {"color": "muted", "marker": None, "ls": "--", "label": "Hartree–Fock"},
    "UCCSD":    {"color": 0,       "marker": "s",  "ls": "-",  "label": "VQE-UCCSD"},
    "HEA":      {"color": 1,       "marker": "^",  "ls": "-",  "label": "VQE-HEA"},
    "CCSD":     {"color": 2,       "marker": "D",  "ls": "-",  "label": "CCSD"},
    "QNN-cond": {"color": 3,       "marker": "P",  "ls": "-",  "label": "warunkowa QNN"},
    "MLP":      {"color": 4,       "marker": "v",  "ls": "-",  "label": "MLP"},
    "GP":       {"color": 5,       "marker": "X",  "ls": "-",  "label": "proces gaussowski"},
}

FONT_STACK = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
TEXT_WIDTH_IN = 6.3   # A4 text width with ~2.5 cm margins
CHEM_ACC_MHA = 1.6    # chemical accuracy in mHa

_state = {"decimal_comma": True}


# ---------------- setup ----------------
def use(decimal_comma=True):
    """apply the style globally. call once at the top of a plot script."""
    t = THEME
    _state.update(decimal_comma=decimal_comma)

    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in FONT_STACK if f in available), "DejaVu Sans")
    if family != "Inter":
        print(f"plotstyle: Inter not found, using {family}. "
              "install Inter (free) for the intended look: https://rsms.me/inter/")

    mpl.rcParams.update({
        "font.family": family,
        "font.size": 8.5,
        "text.color": t["ink_2"],
        "axes.facecolor": t["surface"],
        "figure.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["baseline"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": t["ink_2"],
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.7,
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["muted"],
        "ytick.labelcolor": t["muted"],
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.size": 0,
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
    """wrap text to the figure's usable width (approximate, from average glyph width)."""
    usable_in = fig.get_figwidth() * (1 - fig.subplotpars.left) - 0.15
    chars = max(20, int(usable_in / (fontsize * 0.53 / 72)))
    return "\n".join(textwrap.fill(par, chars) for par in text.split("\n"))


def header(fig, title, subtitle=None, key=None):
    """editorial header: bold title, gray subtitle, horizontal key of methods."""
    t, L = theme(), fig._ps
    x = fig.subplotpars.left
    y_in = 0.16
    if title:   # pass None when the title goes into the thesis caption instead
        a = fig.text(x, 0, title, ha="left", va="top", fontsize=13, fontweight="bold",
                     color=t["ink"])
        L["top"].append((a, y_in))
        y_in += 0.32
    if subtitle:
        sub = _wrap(fig, subtitle, 8.5)
        a = fig.text(x, 0, sub, ha="left", va="top", fontsize=8.5, color=t["ink_2"],
                     linespacing=1.3)
        L["top"].append((a, y_in))
        y_in += 0.17 * (sub.count("\n") + 1) + 0.08
    if key:
        handles = []
        for m in key:
            s = style(m)
            handles.append(Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                                  markersize=4.5, lw=1.6, label=s["label"],
                                  markerfacecolor=s["color"], markeredgecolor=t["surface"]))
        leg = fig.legend(handles=handles, loc="upper left", ncol=len(key),
                         bbox_to_anchor=(x - 0.008, 1), fontsize=7.5, handlelength=2.0,
                         columnspacing=1.4, handletextpad=0.5, borderaxespad=0,
                         labelcolor=t["ink_2"])
        L["top"].append((_LegendY(leg, x - 0.008), y_in))
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
    note = _wrap(fig, text, 6.5)
    a = fig.text(fig.subplotpars.left, 0, note, ha="left", va="bottom", fontsize=6.5,
                 color=theme()["muted"], linespacing=1.2)
    L["bottom"].append((a, 0.12))
    L["footer_in"] = 0.12 + 0.13 * (note.count("\n") + 1) + 0.05
    _relayout(fig)


def panel_title(ax, title, subtitle=None):
    """small uppercase panel title above an axis, left-aligned (offsets in points)."""
    t = theme()
    ax.annotate(title.upper(), (0, 1), xycoords="axes fraction",
                xytext=(0, 21 if subtitle else 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.5, fontweight="bold", color=t["ink"])
    if subtitle:
        ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", ha="left", va="bottom",
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
    ax.annotate(text, (x, y), xytext=(dx_pt, dy), textcoords="offset points",
                ha="left", va="center", fontsize=7.5, color=theme()["ink_2"],
                annotation_clip=False)


def callout(ax, xy, text, xytext, ha="left", va="center"):
    """annotation with a thin leader line, editorial style."""
    t = theme()
    ax.annotate(text, xy, xytext=xytext, textcoords="data", ha=ha, va=va,
                fontsize=7, color=t["ink_2"], linespacing=1.2,
                arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 0.7,
                            "shrinkA": 2, "shrinkB": 3})


def bracket(ax, x, y_low, y_high, text, side="right"):
    """vertical bracket measuring a gap between two values, with a label."""
    t = theme()
    ax.annotate("", (x, y_low), xytext=(x, y_high),
                arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25",
                            "color": t["ink_2"], "lw": 0.8})
    ax.annotate(text, (x, (y_low + y_high) / 2), xytext=(6 if side == "right" else -6, 0),
                textcoords="offset points", ha="left" if side == "right" else "right",
                va="center", fontsize=7, color=t["ink_2"], linespacing=1.2)


def zone_below(ax, y, text, y_min=None):
    """shade the region below y (e.g. the chemical-accuracy zone) and label it."""
    t = theme()
    lo = y_min if y_min is not None else ax.get_ylim()[0]
    ax.axhspan(lo, y, color=t["band"], zorder=0, lw=0)
    ax.axhline(y, color=t["muted"], lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.text(0.01, y, text, transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=7, color=t["ink_2"], zorder=5,
            bbox={"facecolor": t["band"], "edgecolor": "none", "pad": 1.5})


# ---------------- number formatting ----------------
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _num(v, decimals=None):
    s = f"{v:.{decimals}f}" if decimals is not None else f"{v:g}"
    s = s.replace("-", "−")
    return s.replace(".", ",") if _state["decimal_comma"] else s


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
        return {0: "1", 1: "10"}.get(e, "10" + str(e).translate(_SUP))

    target.set_major_locator(LogLocator(base=10.0 ** every, numticks=30))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(fmt))


# ---------------- output ----------------
def save(fig, path_stem, formats=("pdf", "png")):
    """save as vector PDF (for the thesis) and PNG (for previews / slides)."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(stem.with_suffix(f".{f}"))
    print("saved:", ", ".join(str(stem.with_suffix(f".{f}")) for f in formats))
