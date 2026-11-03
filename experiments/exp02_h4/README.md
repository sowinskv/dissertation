# exp02: UCCSD vs hardware-efficient ansatz on the H₄ chain (RQ1)

Linear H₄ in STO-3G (6 qubits), atom spacing 0.5–2.5 Å. Stretching the chain makes electron correlation strong: HF fails, CCSD drops below the exact energy.

## Scripts (run from the repository root, in this order)

| script | what it does | output in `results/` |
|---|---|---|
| `ansatze.py` | VQE-UCCSD and VQE-HEA (1–6 layers, 5 random starts) at 9 spacings, HF / CCSD / FCI references | `exp02_h4_results.csv`, `runs/exp02_h4/` |
| `spin_check.py` | no training: rebuilds every final state from its saved angles and measures its spin (finds the quintet traps) | `exp02_h4_spin_results.csv`, `exp02_h4_spectrum.csv` |
| `penalty.py` | the same HEA runs with a spin penalty μ⟨S²⟩ in the cost (μ = 0, 0.1, 1 Ha) | `exp02_h4_penalty_results.csv`, `runs/exp02_h4_penalty/` |
| `plot.py` | all five figures from the CSVs | `fig_exp02_h4*.pdf` / `.png` |

```
uv run python experiments/exp02_h4/ansatze.py --workers 2
uv run python experiments/exp02_h4/spin_check.py
uv run python experiments/exp02_h4/penalty.py --workers 2
uv run python experiments/exp02_h4/plot.py
```

`ansatze.py` and `penalty.py` take hours; both resume from `*_partial.jsonl` after an interruption.

## Figures

- `fig_exp02_h4`: error vs spacing, with how much the HF configuration still dominates
- `fig_exp02_h4_cost`: CNOT count and training cost of accuracy
- `fig_exp02_h4_spin`: final HEA states, energy error vs total spin
- `fig_exp02_h4_penalty`: the same starts with and without the penalty
- `fig_exp02_h4_summary`: closing figure, best penalised HEA vs UCCSD, CCSD, HF

## Result

UCCSD (26 angles, 1072 CNOT) stays within chemical accuracy up to 1.5 Å, worst 5.3 mHa. Plain HEA gets stuck: 21 of 270 runs end exactly in the quintet, where the gradient is zero. With the spin penalty (μ = 0.1 Ha), HEA with 6 layers and best of 5 starts is within chemical accuracy at all 9 spacings. It needs only 30 CNOT, but about 680× the training cost per start.
