# exp03: one conditional QNN for the whole H₄ curve (RQ2, RQ3)

One UCCSD circuit whose 26 angles come from a small encoder of the atom spacing d (Meta-VQE style, `src/conditional.py`). The encoder is trained on 3, 5 or 9 spacings using only the energy, with no FCI labels. It is tested on 0.4–3.0 Å: interpolation inside 0.5–2.5 Å, extrapolation outside.

## Scripts (run from the repository root)

| script | what it does | output in `results/` |
|---|---|---|
| `conditional.py` | UCCSD per point (reference, 3 starts) on the test grid, then 60 conditional models (encoder degree 1/2/3/5 × 3/5/9 spacings × 5 inits) | `exp03_h4_conditional_*.csv`, `exp03_h4_reference.csv`, `runs/exp03_h4_conditional/` |
| `baselines.py` | RQ3: Gaussian process and MLP fitted to FCI labels at the same spacings | `exp03_h4_baselines_*.csv`, `runs/exp03_h4_baselines/` |
| `plot.py` | the three figures | `fig_exp03_h4_*.pdf` / `.png` |

```
uv run python experiments/exp03_conditional/conditional.py --workers 2   # ~25 min
uv run python experiments/exp03_conditional/baselines.py                 # ~1 min
uv run python experiments/exp03_conditional/plot.py
```

## Result

With a degree-3 encoder and 5 spacings, the model is as good as UCCSD trained separately at every spacing (worst 5.2 vs 5.3 mHa in 0.5–2.5 Å). It costs about as much as 41 separate trainings, and afterwards any spacing takes one circuit evaluation. The linear encoder is too weak. Extrapolation fails (14–45 mHa). With exact labels, classical regression of the correlation energy wins (MLP 1.0 mHa, GP 2.5 mHa with 5 labels), but it needs exactly the FCI energies that cannot be computed for larger molecules.
