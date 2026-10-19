# Thesis context for Claude

Engineering (BSc) thesis at PJATK Warsaw by Joanna

Title (declared, Polish): _Zastosowanie kwantowych sieci neuronowych do symulacji i modelowania układów molekularnych_.

The thesis text is written in Polish. Code comments, docstrings and console output: English, mostly lowercase (uppercase only where needed: H2, CSV, HF, FCI, VQE, class names). Plot labels in English, kept minimal.

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
- Every experiment also logs to the lab notebook via `src/tracking.py`: one `Experiment` per script (title, short description, config, figure files), one `run` per training with `callback=run.log`, final metrics incl. `theta` in `run.finish(...)`. Run ids are readable (`d1.50_HEA_L3_s0`); `is_done` allows resume.
- Train on simulator; on hardware only evaluate trained parameters (limited free IBM Open plan time). Record backend name and date.
- Never commit IBM Quantum tokens.

## Environment

- Python 3.12, qiskit 2.5.2, qiskit-nature 0.8.0, pyscf 2.14.0 (see requirements.txt).
- PySCF does not run natively on Windows: use WSL, macOS/Linux or Colab.
- Experiments that use several worker processes must set OMP_NUM_THREADS=1 (done at the top of the script); multithreaded BLAS on tiny matrices otherwise makes training ~30x slower.

## Figures

- All figures use `src/plotstyle.py` (`ps.use()`, white style for the thesis PDF), matching the lab notebook: centred header and footer, title in regular weight, black ink and axis lines, pencil-gray notes, sentence case, bundled Arimo font (`assets/fonts`, Helvetica metrics, same PDF on every machine); sub/superscripts go through mathtext automatically (`ps.tex`).
- Experiments write CSV to `results/`; separate `experiments/expNN_plot.py` scripts read the CSV and draw. Never recompute physics in a plot script.
- Each method has a fixed color/marker in `ps.METHODS` (FCI = black ink, HF = pencil gray dashed, studied models = pen colors: UCCSD blue, HEA red-orange, CCSD teal, QNN-cond amber, MLP magenta, GP violet-blue; CVD-validated per research-question group). Add new methods there, never pick colors ad hoc; keep `COLORS` in `notebook/app.js` in sync.
- Minimal text, in English: a short descriptive title in research style (what is plotted, for which system; no marketing phrasing, no subtitle), key in the header, direct labels at line ends, one short footer line (system, basis, simulator, seeds, caveats). Avoid callouts unless essential. No dual y-axes.
- Decimal point; units as the panel subtitle (e.g. "mHa, log scale").
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
  - `tracking.py` experiment tracker (MLflow-like, plain files): `results/runs/<exp>/experiment.json`, `<run>.json`, `<run>.history.jsonl` (thinned)
- `tools/notebook.py` local lab-notebook web UI over `results/runs/` (stdlib server, `uv run python tools/notebook.py`, http://localhost:8765, refreshes live while an experiment runs)
- `notebook/` its front end (vanilla JS, vendored Chart.js 4.5.1, one typeface everywhere: Helvetica, with self-hosted Arimo as fallback), plain early-web style (centred, blue [links], bordered tables). Landing page = list of every logged experiment, built from the logs (no manual config), numbered by the server: exp02 -> 02, a later exp02_* follow-up -> 02b. Photo per experiment: `notebook/photos/<exp>.jpg` (falls back to its first figure). New methods also need a color in `COLORS` in `notebook/app.js`, same as `ps.METHODS`.
- `tests/` control tests, run with `uv run pytest -q`
- `results/` CSV/JSON outputs, figures (`fig_expNN_*.pdf/.png`) and training logs (`runs/`)

## Status

- exp01: H2 curve, HF / FCI / VQE-UCCSD / VQE-HEA (L=2) on statevector. Both VQE models reach FCI to numerical precision (< 1e-9 mHa); UCCSD 3 params, 4 CNOT, ~45 circuit evaluations; HEA 12 params, 2 CNOT, ~1100 evaluations (evaluations counted as 2p+1 per gradient).
- exp02: H4 chain (d = 0.5-2.5 A, 9 points), UCCSD vs HEA 1-6 layers x 5 seeds, CCSD reference.
  - HF weight in FCI state drops 99% -> 33%; HF error 25 -> 462 mHa; CCSD goes BELOW FCI from d = 1.25 (down to -31 mHa).
  - UCCSD (26 params, 1072 CNOT, ~1.2k evals): within chemical accuracy up to d = 1.5, max error 5.3 mHa.
  - HEA: error falls with layers but non-monotonically and seed-dependent; 6 layers (84 params, 30 CNOT) median worst 5.4 mHa, worst seed 57.7 mHa; ~800k evals (~690x UCCSD); 28 runs hit 10k iterations. <N> stays 4.
  - Reproducibility: rerun with tracking (single-threaded BLAS throughout) changed 16 HEA runs with 1-2 layers (moved between near-degenerate local minima, e.g. 22.57 <-> 24.51 mHa); L >= 3 and UCCSD bit-identical, per-layer median/worst unchanged. Shallow HEA is sensitive to floating-point summation order.
  - Spin check (`experiments/exp02_spin_check.py`, no training; states rebuilt from saved theta): 21/270 HEA runs end exactly in the quintet (S^2 = 6, S_z = +-2, all four spins aligned), an exact eigenstate -> zero gradient, fidelity 0; error = quintet-singlet gap (264 mHa at 1.5 A ... 10.5 mHa at 2.5 A; the repeated 57.70 mHa at 2.0 A). <N> = 4 does not reveal it. Most other failures are spin-contaminated singlet/triplet mixtures; median <S^2> falls with depth (0.69 at L=1 -> 0.002 at L=6). All runs within chemical accuracy have <S^2> <= 0.41; UCCSD keeps <S^2> <= 1.5e-3. Figure: `fig_exp02_h4_spin`.
  - Reproducibility bug found: PySCF orbital signs varied between runs, so 71/270 saved HEA parameter sets did not reproduce their state with a freshly built hamiltonian (energies unaffected). Fixed in `hamiltonians.build` (sign convention, tested); the spin check recovers each old run's signs by matching its logged energy.
  - Spin penalty (`experiments/exp02_penalty.py`, cost <H> + mu <S^2>, mu = 0 / 0.1 / 1 Ha, same starts and fixed-sign hamiltonian in every arm, d = 1.5-2.5 A, L = 2 and 6, 5 seeds; figure `fig_exp02_h4_penalty`): the control arm reproduces the quintet traps (L=2: 14/25, L=6: 5/25). Penalty removes every quintet trap and all spin contamination at L=6 (mu=0.1: chemical accuracy 10 -> 16 of 25, median error 2.77 -> 0.92 mHa, worst 57.7 -> 22.1 mHa, same training cost). mu=1 is too strong (median 1.10 mHa but worst 118.8 mHa). At L=2 the penalty cannot help: the circuit cannot represent a good singlet, the quintet was its lowest reachable energy; mu=0.1 leaves S^2 ~ 0.9 and mu=1 forces a near-HF singlet (median 214 mHa). Takeaway: a penalty fixes wrong-symmetry traps only when the ansatz is expressive enough; it adds no expressivity.
  - Closing result (`fig_exp02_h4_summary`, extra runs d = 0.5-1.25 A at L=6, mu=0.1): HEA 6 layers + spin penalty, best of 5 starts, is within chemical accuracy at all 9 spacings (worst best-of-5: 0.082 mHa at 2.25 A); single starts still fail (worst 22 mHa). UCCSD exceeds 1.6 mHa from 1.75 A (up to 5.3 mHa), CCSD from 1.5 A. Shallow circuit (30 CNOT) + penalty + restarts beats deep UCCSD (1072 CNOT) where correlation is strong, at ~400x the training cost.
- Next: conditional QNN (RQ2/RQ3), LiH with an active space, IBM hardware.
