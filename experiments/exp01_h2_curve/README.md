# exp01: potential energy curve of H₂

Validation of the whole pipeline. H₂ in STO-3G (2 qubits after parity reduction), bond length 0.3–3.0 Å, 19 points: HF, FCI, VQE-UCCSD and VQE-HEA (2 layers) on the exact statevector simulator.

## Run (from the repository root)

```
uv run python experiments/exp01_h2_curve/run.py     # training, writes results/
uv run python experiments/exp01_h2_curve/plot.py    # figure from the CSV
```

## Files

- `results/exp01_h2_results.csv`: one row per bond length, energies and errors of every method
- `results/exp01_h2_meta.json`: config, library versions, git commit, runtime
- `results/fig_exp01_h2.pdf` / `.png`
- `results/runs/exp01_h2/`: training logs for the lab notebook

## Result

Both VQE models reach FCI to numerical precision (below 10⁻⁹ mHa). UCCSD: 3 angles, 4 CNOT, about 45 circuit evaluations. HEA: 12 angles, 2 CNOT, about 1100 evaluations.
