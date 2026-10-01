# Thesis context for Claude

Engineering (BSc) thesis at PJATK Warsaw by Joanna

Title (declared, Polish): _Zastosowanie kwantowych sieci neuronowych do symulacji i modelowania układów molekularnych_.

The thesis text is written in Polish. Code comments, docstrings and console output: English, mostly lowercase (uppercase only where needed: H2, CSV, HF, FCI, VQE, class names). Plot labels stay in Polish (figures go into the thesis).

## About the author

- Experienced software / AI engineer, comfortable with Python and classical ML.
- Self-taught in quantum physics and quantum computing, no formal background.
- **Explain every new quantum or chemistry concept the first time it appears**, briefly, with an ML analogy where it helps.

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

## Repo layout

- `experiments/` runnable experiment scripts (`expNN_<name>.py`), run from repo root
- `src/` shared modules (hamiltonians, ansatze, energy, train, classical, metrics) - to be extracted from experiments as they repeat
- `tests/` unit and control tests
- `results/` CSV/JSON/PNG outputs

## Status

- exp01: H2 curve, HF / FCI / VQE-UCCSD / VQE-HEA (L=2) on statevector. Both VQE models within chemical accuracy everywhere; UCCSD 3 params, 4 CNOT, ~20 evaluations; HEA 12 params, 2 CNOT, ~250 evaluations.
- Next: same experiment for H4 (and LiH with an active space).
