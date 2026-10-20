"""
experiment 03, classical baselines (RQ3): regression on the same training spacings, but with labels.

the conditional QNN of exp03_conditional.py learns the H4 curve from M spacings using only the
energy as its loss. a classical regressor needs the answer at those spacings: here it gets the
exact FCI energies (labels) at the same M spacings and fits E(d):

  - Gaussian process (GP): RBF kernel times a constant, hyperparameters by maximum likelihood
    (5 optimiser restarts); noiseless labels.
  - MLP: 2 hidden layers of 32 tanh units, L-BFGS, inputs scaled to [-1, 1], targets standardised;
    5 random initialisations.

two targets: the total energy E_FCI(d), and the correlation energy E_FCI(d) - E_HF(d) ("delta
learning"). the second is the fair one: most of the curve's shape (the steep wall at short d)
is mean-field physics that HF, a cheap calculation, gives at every test spacing, so the
regressor only has to learn the smooth remainder. the QNN likewise uses the true hamiltonian
at every test spacing.

both predict only the energy curve. the QNN also returns a quantum state at every spacing (any other
observable can be measured on it), and needs no FCI labels, which for larger molecules are exactly
what cannot be computed.

test grid, training spacings and chemical accuracy are the same as in exp03_conditional.py.

run (from repo root):  python experiments/exp03_baselines.py
output: results/exp03_h4_baselines_results.csv  (one row per model x test spacing)
        results/exp03_h4_baselines_summary.csv  (one row per model)
        results/exp03_h4_baselines_meta.json
"""

import csv
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import hamiltonians as hm, runlog, tracking  # noqa: E402
from src.classical import reference_energies  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp03_conditional import CHEM_ACC_mHa, GRID, HI, LO, _d_eq, train_spacings  # noqa: E402

warnings.filterwarnings("ignore")

CONFIG = {
    "molecule": "H4 linear chain",
    "basis": hm.BASIS,
    "labels": "FCI energies at the training spacings",
    "targets": {"E": "total energy E_FCI(d)",
                "E-HF": "correlation energy E_FCI(d) - E_HF(d) (delta learning); HF is computed at every "
                        "test spacing and added back"},
    "train_range_A": [LO, HI],
    "n_train": [3, 5, 9],
    "seeds": [0, 1, 2, 3, 4],
    "gp": "ConstantKernel * RBF, length scale bounds 0.05-10 A, normalize_y, 5 restarts, alpha 1e-10",
    "mlp": "2 x 32 tanh, lbfgs, max_iter 5000, alpha 1e-4, inputs to [-1, 1], targets standardised",
    "test_grid_A": [0.40, 3.00, 0.05],
    "chemical_accuracy_mHa": CHEM_ACC_mHa,
    "scikit-learn": version("scikit-learn"),
}
NAME = "exp03_h4_baselines"
EXP = tracking.Experiment(
    NAME, title="Classical regression of the H₄ curve from the same spacings",
    description="Gaussian process and MLP fitted to exact FCI energies at the same 3, 5 or 9 "
                "spacings as the conditional QNN, tested on 0.4-3.0 Å. The QNN needs no labels; "
                "these models do. Research question 3.",
    config=CONFIG, figures=["fig_exp03_h4_baselines.png"])


def _x(d):
    # 0.5-2.5 Å -> [-1, 1]; the training range is 2 Å wide, so x keeps the unit of d
    return (2 * (np.asarray(d, dtype=float) - LO) / (HI - LO) - 1).reshape(-1, 1)


def fit_predict(model, target, m, seed, e_fci, e_hf):
    tr = train_spacings(m)
    base = {d: e_hf[d] if target == "E-HF" else 0.0 for d in GRID}
    y = np.array([e_fci[d] - base[d] for d in tr])
    if model == "GP":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (0.05, 10.0))   # x has the unit of d (Å)
        reg = GaussianProcessRegressor(kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=5,
                                       random_state=seed)
        reg.fit(_x(tr), y)
        pred = reg.predict(_x(GRID))
        info = {"kernel": str(reg.kernel_)}
    else:
        mu, sd = y.mean(), y.std() if m > 1 else 1.0
        reg = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs", max_iter=5000, alpha=1e-4,
                           tol=1e-12, random_state=seed)
        reg.fit(_x(tr), (y - mu) / sd)
        pred = reg.predict(_x(GRID)) * sd + mu
        info = {"iterations": int(reg.n_iter_)}
    return tr, pred + np.array([base[d] for d in GRID]), info


def main():
    started = time.time()
    e_fci, e_hf = {}, {}
    for d in GRID:
        ref = reference_energies(hm.h_chain(4, d))
        e_fci[d], e_hf[d] = ref["E_FCI_Ha"], ref["E_HF_Ha"]
    targets = list(CONFIG["targets"])
    EXP.start(n_runs=len(targets) * len(CONFIG["n_train"]) * (1 + len(CONFIG["seeds"])))

    rows, summary = [], []
    ds = np.array(GRID)
    inside = (ds >= LO - 1e-9) & (ds <= HI + 1e-9)
    fci_curve = np.array([e_fci[d] for d in GRID])
    for target in targets:
      for m in CONFIG["n_train"]:
        for model, seeds in (("GP", [0]), ("MLP", CONFIG["seeds"])):
            for seed in seeds:
                params = {"method": model, "target": target, "n_train": m, "seed": seed}
                run_id = f"{model}_{target}_M{m}" + (f"_s{seed}" if model == "MLP" else "")
                with EXP.run(run_id, params) as run:
                    tr, pred, info = fit_predict(model, target, m, seed, e_fci, e_hf)
                    err = 1e3 * (pred - fci_curve)
                    for d, e, de in zip(GRID, pred, err):
                        rows.append({**params, "d_A": d, "in_train": d in tr,
                                     "region": "interpolation" if LO - 1e-9 <= d <= HI + 1e-9 else "extrapolation",
                                     "E_Ha": float(e), "dE_mHa": float(de)})
                    a = np.abs(err)
                    s = {**params,
                         "interp_max_abs_dE_mHa": float(a[inside].max()),
                         "interp_median_abs_dE_mHa": float(np.median(a[inside])),
                         "interp_frac_chem_acc": float(np.mean(a[inside] < CHEM_ACC_mHa)),
                         "extrap_short_max_abs_dE_mHa": float(a[ds < LO - 1e-9].max()),
                         "extrap_long_max_abs_dE_mHa": float(a[ds > HI + 1e-9].max()),
                         "d_eq_error_A": _d_eq(ds, pred) - _d_eq(ds, fci_curve),
                         "fci_labels": m, **info}
                    summary.append(s)
                    run.finish(**{k: v for k, v in s.items() if k not in params})
                print(f"{model} on {target}, M={m} seed={seed}: max |error| 0.5-2.5 A {s['interp_max_abs_dE_mHa']:.2f} mHa, "
                      f"beyond 2.5 A {s['extrap_long_max_abs_dE_mHa']:.1f} mHa")

    path = Path(f"results/{NAME}_summary.csv")
    keys = list(summary[0]) + sorted({k for s in summary for k in s} - set(summary[0]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"saved: {path}")
    runlog.save(NAME, rows, CONFIG, started)
    EXP.finish()


if __name__ == "__main__":
    main()
