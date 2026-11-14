"""
figures for experiment 00 (data preparation), drawn from the saved CSV.

run (from repo root):  uv run python experiments/exp00_data/plot.py
input:   results/exp00_data_results.csv, results/exp00_data_splits.csv
output:  results/fig_exp00_references, fig_exp00_hamiltonian, fig_exp00_splits, fig_exp00_checks
         (.pdf / .png)
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent / "results"   # this experiment's outputs
from src import plotstyle as ps, runlog  # noqa: E402

SYSTEMS = ["H2", "H4", "LiH"]
NAMES = {"H2": "H₂", "H4": "H₄ chain", "LiH": "LiH"}
# the systems are not methods: all drawn in FCI ink, told apart by marker and direct labels
MARKERS = {"H2": "o", "H4": "s", "LiH": "^"}
FLOOR_HA = 1e-17   # checks that come out exactly 0 are drawn on the floor


def by_system(rows):
    out = {}
    for m in SYSTEMS:
        sel = sorted((r for r in rows if r["molecule"] == m), key=lambda r: r["d_A"])
        out[m] = {k: np.array([r[k] for r in sel]) for k in sel[0]}
    return out


def ink_line(ax, x, y, system, **kw):
    t = ps.theme()
    kw.setdefault("markevery", 4 if system == "H4" else None)   # H4 has a point every 0.05 Å
    return ax.plot(x, y, color=t["ink"], marker=MARKERS[system], markersize=3.6,
                   markerfacecolor=t["ink"], markeredgecolor=t["surface"], markeredgewidth=0.6,
                   lw=1.2, **kw)


def key(lines=True):
    t = ps.theme()
    return [Line2D([], [], color=t["ink"], marker=MARKERS[m], markersize=4, lw=1.2 if lines else 0,
                   markerfacecolor=t["ink"], markeredgecolor=t["surface"], label=ps.tex(NAMES[m]))
            for m in SYSTEMS]


def fig_references(S):
    fig, (ax1, ax2) = ps.figure(nrows=2, sharex=True, panel_height=2.2)
    for m in SYSTEMS:
        ink_line(ax1, S[m]["d_A"], -S[m]["E_corr_mHa"], m)
        ink_line(ax2, S[m]["d_A"], S[m]["hf_weight"], m)
    ps.panel_title(ax1, "Correlation energy", "HF minus FCI energy, mHa")
    ps.panel_title(ax2, "Weight of the HF determinant in the FCI state", "squared overlap")
    ax2.set_ylim(0, 1.05)
    ps.number_axis(ax2, "y", decimals=1)
    ax2.set_xlabel("Bond length / atom spacing d (Å)")
    ps.number_axis(ax2, "x", decimals=1)
    ax2.set_xlim(0.25, 3.05)
    ps.end_labels(ax1, 3.0, [(-S[m]["E_corr_mHa"][-1], NAMES[m]) for m in SYSTEMS])
    ps.end_labels(ax2, 3.0, [(S[m]["hf_weight"][-1], NAMES[m]) for m in SYSTEMS])
    ps.header(fig, "Correlation energy and HF weight along each curve", key=key())
    ps.footer(fig, "STO-3G, PySCF. Low HF weight = strong correlation: H₄ falls to 0.27 at 3 Å.")
    ps.save(fig, OUT / "fig_exp00_references")


def fig_hamiltonian(S):
    fig, axes = ps.figure(ncols=3, panel_height=2.0)
    t = ps.theme()
    x = np.arange(len(SYSTEMS))
    panels = [("Qubits", "before / after Z₂ reduction", "qubits_before", "qubits"),
              ("Pauli terms", "without identity", "pauli_terms", None),
              ("Measurement groups", "qubit-wise commuting", "measurement_groups", None)]
    for ax, (title, sub, key_a, key_b) in zip(axes, panels):
        a = np.array([S[m][key_a].max() for m in SYSTEMS])
        if key_b:
            b = np.array([S[m][key_b].max() for m in SYSTEMS])
            ax.bar(x, a, width=0.6, color=t["surface"], edgecolor=t["muted"], lw=0.8)
            ax.bar(x, b, width=0.6, color=t["ink"])
            for xi, ai, bi in zip(x, a, b):
                ax.annotate(f"{int(bi)}", (xi, bi), xytext=(0, -3), textcoords="offset points",
                            ha="center", va="top", fontsize=7, color=t["surface"])
                ax.annotate(f"{int(ai)}", (xi, ai), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=t["muted"])
        else:
            ax.bar(x, a, width=0.6, color=t["ink"])
            for xi, ai in zip(x, a):
                ax.annotate(f"{int(ai)}", (xi, ai), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=t["ink_2"])
        ax.set_xticks(x, [ps.tex(NAMES[m].split()[0]) for m in SYSTEMS])
        ax.margins(y=0.15)
        ps.panel_title(ax, title, sub)
    fig.subplots_adjust(right=0.97, wspace=0.45)
    ps.header(fig, "Size of the qubit hamiltonian per system")
    ps.footer(fig, "STO-3G, ParityMapper with two-qubit Z₂ reduction; coefficients below 10⁻¹⁰ Ha "
                   "dropped. Counts are the same at every geometry of a system.")
    fig.subplots_adjust(right=0.97, wspace=0.45)
    ps.save(fig, OUT / "fig_exp00_hamiltonian")


def fig_splits(splits):
    t = ps.theme()
    names = list(dict.fromkeys(r["split"] for r in splits))[::-1]
    fig, ax = ps.figure(panel_height=0.24 * len(names) + 0.4)
    ax.axvspan(0.35, 0.5, color=t["grid"], lw=0, zorder=0)
    ax.axvspan(2.5, 3.05, color=t["grid"], lw=0, zorder=0)
    for i, name in enumerate(names):
        rows = [r for r in splits if r["split"] == name]
        d = np.array([r["d_A"] for r in rows])
        role = rows[0]["role"]
        if role == "test":
            ax.plot(d, np.full_like(d, i), "o", ms=2.6, color=t["ink"], mfc=t["surface"], mew=0.7)
        elif role == "train":
            ax.plot(d, np.full_like(d, i), "o", ms=5.5, color=ps.color("QNN-cond"), mec=t["surface"], mew=0.6)
        else:
            ax.plot(d, np.full_like(d, i), "o", ms=3.6, color=t["ink"], mec=t["surface"], mew=0.5)
    ax.set_yticks(range(len(names)), [ps.tex(n.replace("H2", "H₂").replace("H4", "H₄")) for n in names])
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.25, 3.05)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("Bond length / atom spacing d (Å)")
    ps.number_axis(ax, "x", decimals=1)
    ax.annotate("extrapolation", (2.775, len(names) - 0.3), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=t["muted"], annotation_clip=False)
    handles = [Line2D([], [], ls="none", marker="o", ms=5.5, color=ps.color("QNN-cond"), mec=t["surface"],
                      label="training (energy only)"),
               Line2D([], [], ls="none", marker="o", ms=2.6, color=t["ink"], mfc=t["surface"], label="test"),
               Line2D([], [], ls="none", marker="o", ms=3.6, color=t["ink"], mec=t["surface"],
                      label="evaluated per point")]
    ps.header(fig, "Geometries used by each experiment", key=handles)
    ps.footer(fig, "exp03: conditional QNN trained on M spacings in 0.5–2.5 Å, tested on the full "
                   "grid; shaded = outside the training range. LiH is prepared, not used yet.")
    fig.subplots_adjust(left=0.2)
    ps.save(fig, OUT / "fig_exp00_splits")


def fig_checks(S):
    fig, ax = ps.figure(panel_height=2.2)
    for m in SYSTEMS:
        ink_line(ax, S[m]["d_A"], np.maximum(S[m]["fci_check_Ha"], FLOOR_HA), m, ls="none", markevery=None)
    ps.log_axis(ax, "y", every=2)
    ax.set_ylim(FLOOR_HA * 0.3, 1e-6)
    ps.threshold(ax, 1e-8, "tolerance 10⁻⁸ Ha", above=True)
    ps.panel_title(ax, "Difference from PySCF FCI", "absolute value, Ha, log scale")
    ax.set_xlabel("Bond length / atom spacing d (Å)")
    ps.number_axis(ax, "x", decimals=1)
    ax.set_xlim(0.25, 3.05)
    ps.header(fig, "Lowest eigenvalue of the qubit hamiltonian vs FCI", key=key(lines=False))
    ps.footer(fig, "Lowest eigenvalue taken in the right electron-number sector. Exact zeros drawn at 10⁻¹⁷. "
                   "Two independent builds gave identical hamiltonians at every geometry.")
    ps.save(fig, OUT / "fig_exp00_checks")


def main():
    ps.use()
    S = by_system(runlog.load(OUT / "exp00_data_results.csv"))
    fig_references(S)
    fig_hamiltonian(S)
    fig_splits(runlog.load(OUT / "exp00_data_splits.csv"))
    fig_checks(S)


if __name__ == "__main__":
    main()
