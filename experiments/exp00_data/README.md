# exp00: data preparation

The thesis uses no external dataset: every input is generated. This experiment takes every geometry used in exp01–exp04 (plus LiH, prepared for the next step) through the whole pipeline once, records what each step produces and checks it. No training happens here.

Pipeline per geometry: geometry → PySCF (RHF, integrals) → second-quantized hamiltonian → ParityMapper with two-qubit Z₂ reduction (orbital signs fixed, `src/hamiltonians.py`) → qubit hamiltonian. In parallel: HF, CCSD and FCI references and the HF weight in the FCI state (`src/classical.py`).

## Run (from the repository root)

```
uv run python experiments/exp00_data/prepare.py   # ~1 min, writes results/
uv run python experiments/exp00_data/plot.py      # four figures from the CSV
uv run pytest -q tests/test_data.py               # splits in sync with the experiment scripts
```

## Files

- `results/exp00_data_results.csv`: one row per geometry (86): molecule, d, atom string, electrons, orbitals, qubits before / after the reduction, Pauli terms, 1-norm of the coefficients, measurement groups, nuclear repulsion, E_HF / E_CCSD / E_FCI, correlation energy, CCSD − FCI, HF weight, checks, build time, which experiments use it
- `results/exp00_data_splits.csv`: which geometries each experiment uses and in what role (exp03: training spacings for M = 3/5/9, test grid, interpolation / extrapolation)
- `results/exp00_data_meta.json`: config, library versions, git commit, runtime
- `results/fig_exp00_references`, `fig_exp00_hamiltonian`, `fig_exp00_splits`, `fig_exp00_checks` (`.pdf` / `.png`)
- `results/runs/exp00_data/`: one notebook entry per geometry

## Checks (the script stops if one fails)

- lowest eigenvalue of the qubit hamiltonian in the right electron-number sector = PySCF FCI within 10⁻⁸ Ha
- two independent builds give the same qubit hamiltonian (orbital sign convention)

## Result

All 86 geometries pass: worst |λ_min − E_FCI| = 3.6·10⁻¹⁴ Ha (LiH), hamiltonians identical between builds. No state with a wrong electron number lies below the ground state at these geometries.

| system | geometries | qubits (before → after) | Pauli terms | measurement groups | HF weight (min) | correlation energy (max) |
|---|---|---|---|---|---|---|
| H₂ | 24 (0.3–3.0 Å) | 4 → 2 | 4 | 2 | 0.54 | 278 mHa |
| H₄ chain | 53 (0.4–3.0 Å) | 8 → 6 | 164 | 34 | 0.27 | 554 mHa |
| LiH | 9 (1.0–3.0 Å) | 12 → 10 | 630 | 165 | 0.72 | 88 mHa |

The hamiltonian size is the same at every geometry of a system. On hardware the number of measurement groups sets the circuits per energy: 2 for H₂ but 165 for LiH, so LiH on the IBM Open plan needs an active space.
