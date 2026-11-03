"""
figures for experiment 04 (H2 on a noisy processor), drawn from the saved CSV.

run (from repo root):  python experiments/exp04_hardware/plot.py
input:   results/exp04_h2_hardware_results.csv
output:  results/fig_exp04_h2_noise.pdf/.png  energy error along the bond: raw, readout mitigation,
                                               readout mitigation + ZNE (simulator seeds; hardware
                                               runs as hollow markers when present)
         results/fig_exp04_h2_zne.pdf/.png    zero-noise extrapolation at equilibrium: energy vs
                                               noise factor and the extrapolated value
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent / "results"   # this experiment's outputs
from src import plotstyle as ps  # noqa: E402
from src.runlog import load  # noqa: E402

ARMS = [("raw", "raw", ":"), ("readout", "readout mitigation", "--"),
        ("readout_zne", "readout mitigation + ZNE", "-")]
METHODS = [("UCCSD", "UCCSD", "VQE-UCCSD"), ("HEA_L2", "HEA", "VQE-HEA, 2 layers")]


def by_d(rows, method, col):
    out = defaultdict(list)
    for r in rows:
        if r["method"] == method:
            out[r["d_A"]].append(r[col])
    d = np.array(sorted(out))
    return d, [np.array(out[x]) for x in d]


def fig_noise(rows):
    t = ps.theme()
    sim = [r for r in rows if r["kind"] == "simulator"]
    hw = [r for r in rows if r["kind"] == "hardware"]
    backend = sim[0]["backend"]
    fig, axes = ps.figure(nrows=1, ncols=2, panel_height=2.8)
    for ax, (method, style_key, title) in zip(axes, METHODS):
        shades = ps.shades(style_key, len(ARMS), lightest=0.45)
        ax.axhline(0, color=t["ink"], lw=0.8, zorder=1)
        for y in (ps.CHEM_ACC_MHA, -ps.CHEM_ACC_MHA):
            ax.axhline(y, color=t["ink"], lw=0.8, ls=(0, (1, 2)), zorder=1)
        ends = []
        for (arm, label, ls), c in zip(ARMS, shades):
            d, vals = by_d(sim, method, f"dE_{arm}_mHa")
            mean = np.array([v.mean() for v in vals])
            ax.vlines(d, [v.min() for v in vals], [v.max() for v in vals], color=c, lw=1.2, alpha=0.6, zorder=2)
            ax.plot(d, mean, color=c, ls=ls, lw=1.6, marker=ps.style(style_key)["marker"], markersize=4.5,
                    markerfacecolor=c, markeredgecolor=t["surface"], zorder=3)
            ends.append((mean[-1], {"raw": "raw", "readout": "readout", "readout_zne": "readout + ZNE"}[arm]))
            if hw:
                dh, vh = by_d(hw, method, f"dE_{arm}_mHa")
                ax.plot(dh, [v.mean() for v in vh], ls="none", marker="o", markersize=6, markerfacecolor="none",
                        markeredgecolor=c, markeredgewidth=1.3, zorder=4)
        ax.set_xlabel("Bond length R (Å)")
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ps.number_axis(ax, "x", decimals=1)
        ps.panel_title(ax, title, "error vs FCI, mHa")
        ps.end_labels(ax, d[-1], ends)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(min(lo, -5), hi)
    fig.subplots_adjust(wspace=0.55)
    key = [Line2D([], [], color=t["ink_2"], ls=ls, lw=1.6, label=label) for _, label, ls in ARMS]
    if hw:
        key.append(Line2D([], [], ls="none", marker="o", markerfacecolor="none", markeredgecolor=t["ink_2"],
                          markersize=6, label=f"IBM hardware ({hw[0]['backend']})"))
    ps.header(fig, "Energy error of trained H₂ circuits on a noisy IBM processor, with and without error mitigation",
              key=key)
    shots = int(sim[0]["shots"])
    ps.footer(fig, f"H₂, STO-3G, 2 qubits. Noisy simulator of IBM {backend.replace('fake_', '').title()} "
                   f"(calibration noise model), {shots} shots per circuit, lines: mean of 5 shot seeds, bars: all 5. "
                   "ZNE: two-qubit gates folded ×1, ×3, ×5, linear fit to zero noise. Dotted: chemical accuracy, ±1.6 mHa."
                   + (f" Hollow circles: one run on {hw[0]['backend']} ({hw[0]['date']}, qubits {hw[0]['qubits']}, "
                      f"same shots), one circle per variant." if hw else ""))
    ps.save(fig, OUT / "fig_exp04_h2_noise")


def _zne_panel(ax, rows, d, title):
    t = ps.theme()
    factors = [1, 3, 5]
    ax.axhline(0, color=t["ink"], lw=0.8, zorder=1)
    ps.threshold(ax, ps.CHEM_ACC_MHA, "chemical accuracy", above=True, x=0.4)
    ends = []
    for k, (method, style_key, _) in enumerate(METHODS):
        c = ps.color(style_key)
        sel = [r for r in rows if r["method"] == method and abs(r["d_A"] - d) < 1e-9]
        e = np.array([[1e3 * (r[f"E_readout_x{f}_Ha"] - r["E_FCI_Ha"]) for f in factors] for r in sel])
        zne = np.array([r["dE_readout_zne_mHa"] for r in sel])
        off = (k - 0.5) * 0.08
        mean = e.mean(axis=0)
        if len(sel) > 1:
            ax.vlines(np.array(factors) + off, e.min(axis=0), e.max(axis=0), color=c, lw=1.2, alpha=0.6)
            ax.vlines(off, zne.min(), zne.max(), color=c, lw=1.2, alpha=0.6)
        ax.plot(np.array(factors) + off, mean, ls="none", marker=ps.style(style_key)["marker"], markersize=6,
                color=c, markeredgecolor=t["surface"])
        slope, icpt = np.polyfit(factors, mean, 1)
        xs = np.linspace(0, 5, 20)
        ax.plot(xs, icpt + slope * xs, color=c, lw=1.1, ls=(0, (4, 2)))
        ax.plot([off], [zne.mean()], ls="none", marker=ps.style(style_key)["marker"], markersize=7,
                markerfacecolor=t["surface"], markeredgecolor=c, markeredgewidth=1.4)
        ends.append((mean[-1], style_key))
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlim(-0.4, 5.4)
    ax.set_xlabel("Noise factor")
    ps.panel_title(ax, title, "error vs FCI, mHa")
    ps.end_labels(ax, 5.1, ends)


def fig_zne(rows, d=0.735):
    t = ps.theme()
    sim = [r for r in rows if r["kind"] == "simulator"]
    hw = [r for r in rows if r["kind"] == "hardware" and any(abs(r["d_A"] - d) < 1e-9 for _ in [0])]
    panels = [(sim, "Noisy simulator of IBM Fez")] + ([(hw, f"IBM hardware ({hw[0]['backend']})")] if hw else [])
    fig, axes = ps.figure(nrows=1, ncols=len(panels), panel_height=2.8)
    axes = np.atleast_1d(axes)
    for ax, (sel, title) in zip(axes, panels):
        _zne_panel(ax, sel, d, title)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    if len(axes) > 1:
        fig.subplots_adjust(wspace=0.55)
    key = [Line2D([], [], color=ps.color(s), marker=ps.style(s)["marker"], ls=(0, (4, 2)), lw=1.1, markersize=5,
                  markeredgecolor=t["surface"], label=lab) for _, s, lab in METHODS]
    ps.header(fig, f"Zero-noise extrapolation for H₂ at R = {d} Å, readout-mitigated energies", key=key)
    note = (f" Hardware: one run on {hw[0]['backend']}, {hw[0]['date']}, qubits {hw[0]['qubits']}." if hw else "")
    ps.footer(fig, "H₂, STO-3G. Filled: energy measured with two-qubit gates folded ×1, ×3, ×5 (simulator: mean of "
                   "5 shot seeds, bars: all 5); dashed: linear fit; hollow: extrapolated value at zero noise." + note)
    ps.save(fig, OUT / "fig_exp04_h2_zne")


def main():
    ps.use()
    rows = load(OUT / "exp04_h2_hardware_results.csv")
    fig_noise(rows)
    fig_zne(rows)


if __name__ == "__main__":
    main()
