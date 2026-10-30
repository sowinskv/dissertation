this repository contains (or WILL contain) the research and code for my thesis on using Quantum Neural Networks (QNNs) to model molecular systems.

-- the core idea is to explore whether QNNs can offer a new way to tackle molecular simulations, which are often incredibly difficult for classical computers. This project investigates **the potential of QNNs to predict key properties of molecules, such as their energy states**, and compares their performance against traditional simulation methods in terms of accuracy and computation time.

a key part of this research is understanding the opportunities and challenges of using today's quantum hardware. The project involves:

- implementing a QNN from the ground up in Python using Qiskit.
- training the network on a quantum simulator to test and fine-tune the model.
- running the experiments on a real cloud-based quantum processor (e.g., from IBM Quantum) to see how factors like hardware noise and qubit limitations affect the results in a practical setting

## repository layout

```
experiments/
  exp01_h2_curve/     H2 curve, pipeline validation
  exp02_h4/           UCCSD vs HEA on the H4 chain, spin check, spin penalty (RQ1)
  exp03_conditional/  one conditional QNN for the whole H4 curve + classical baselines (RQ2, RQ3)
  exp04_hardware/     H2 on a noisy IBM processor, readout mitigation and ZNE (RQ4)
    each: scripts, plot.py, README.md, results/ (CSV, figures, notebook logs)
src/        shared code: hamiltonians, ansatze, simulator, training, conditional QNN, mitigation, plots
tests/      control tests (uv run pytest -q)
tools/      lab notebook server (uv run python tools/notebook.py -> http://localhost:8765)
notebook/   its front end
```

run every script from the repository root, e.g. `uv run python experiments/exp02_h4/penalty.py`.
