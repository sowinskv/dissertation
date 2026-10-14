# Thesis context for Claude

Engineering (BSc) thesis at PJATK Warsaw by Joanna

Title (declared, Polish): _Zastosowanie kwantowych sieci neuronowych do symulacji i modelowania układów molekularnych_.

The thesis text is written in Polish. Code comments, docstrings and console output: English, mostly lowercase (uppercase only where needed: H2, CSV, HF, FCI, VQE, class names). Plot labels stay in Polish (figures go into the thesis).

## Research framing (decided)

- Not "are QNNs better than classical methods" (at this scale they cannot be: exact FCI is cheap).
- Instead: **at what cost do QNNs reach chemical accuracy (1.6 mHa vs FCI), and can one conditional QNN learn a whole dissociation curve without energy labels?**
- A negative result is acceptable if it answers the question honestly.

## Research questions

- RQ1: UCCSD vs hardware-efficient ansatz (HEA): accuracy, parameters, CNOT count, depth, training cost.
- RQ2: conditional QNN (Meta-VQE style, bond length R as input, linear encoder R -> angles) learning the full curve; interpolation vs extrapolation.
- RQ3: compare with classical regression (MLP, Gaussian process) trained on the same M geometries but with FCI labels.
- RQ4: noisy simulator and IBM Quantum hardware, with and without error mitigation (readout, ZNE).

## Scope and priorities

1. Must: UCCSD vs HEA for H2, LiH, linear H4 (STO-3G) on statevector simulator; H2 on IBM hardware.
2. Should: conditional QNN + classical baselines.
3. Nice to have: classifier QNN on "quantum data" (VQE-prepared states as input, label = strong correlation, e.g. |<HF|psi>|^2 below a threshold; baseline = classical model on measurement outcomes, not on R).

## Conventions

- Reference energies: HF, CCSD, FCI from PySCF. Every experiment asserts that the qubit Hamiltonian's lowest eigenvalue matches PySCF FCI.
- Metrics: |E - E_FCI| in mHa, % points within 1.6 mHa, NPE, R_eq / D_e errors, fidelity (simulation only), params, CNOT and depth after transpiling to ["cx","rz","sx","x"], circuit evaluations, shots.
- 5 seeds per stochastic experiment; report mean and spread.
- Each run saves config, library versions, seed and results to `results/`.
- Train on simulator; on hardware only evaluate trained parameters (limited free IBM Open plan time). Record backend name and date.
- Never commit IBM Quantum tokens.

## Environment

- Python 3.12, qiskit 2.5.2, qiskit-nature 0.8.0, pyscf 2.14.0 (see requirements.txt).
- PySCF does not run natively on Windows: use WSL, macOS/Linux or Colab.

## Figures

- All figures use `src/plotstyle.py` (`ps.use()`, white style for the thesis PDF).
- Experiments write CSV to `results/`; separate `experiments/plot_expNN.py` scripts read the CSV and draw. Never recompute physics in a plot script.
- Each method has a fixed color/marker in `ps.METHODS` (FCI = ink, HF = muted gray, studied models = colored slots). Add new methods there, never pick colors ad hoc.
- Title states the finding (a sentence, not a label), subtitle explains it, footer lists system, basis, simulator, seeds and caveats.
- Direct labels at line ends + key in the header; annotate the 1-3 things the reader must notice (callout, bracket, threshold). No dual y-axes.
- Polish number format (decimal comma) on all axes and labels; units in panel subtitles.
- Save PDF (vector, goes into the thesis) and PNG via `ps.save`.

## Repo layout

- `experiments/` runnable experiment scripts (`expNN_<name>.py`), run from repo root
- `src/` shared modules:
  - `hamiltonians.py` geometries, qubit hamiltonian (ParityMapper), exact ground state in the right electron-number sector
  - `classical.py` HF / CCSD / FCI from PySCF, HF weight in the FCI state
  - `ansatze.py` UCCSD, HEA(layers), circuit resources (params, CNOT, depth)
  - `fastsim.py` exact batched numpy statevector simulation of a qiskit circuit (used for training; tested against StatevectorEstimator to 1e-10)
  - `train.py` VQE: L-BFGS-B + batched central finite-difference gradient
  - `runlog.py` saves `<name>_results.csv` + `<name>_meta.json` (config, versions, git commit, runtime)
  - `plotstyle.py` figure style
- `tests/` control tests, run with `uv run pytest -q`
- `results/` CSV/JSON outputs and figures (`fig_expNN_*.pdf/.png`)

## Status

- exp01: H2 curve, HF / FCI / VQE-UCCSD / VQE-HEA (L=2) on statevector. Both VQE models reach FCI to numerical precision (< 1e-9 mHa); UCCSD 3 params, 4 CNOT, ~45 circuit evaluations; HEA 12 params, 2 CNOT, ~1100 evaluations (evaluations counted as 2p+1 per gradient).
- exp02: H4 chain, UCCSD vs HEA 1-6 layers x 5 seeds, CCSD reference (running).
- Next: LiH with an active space.
