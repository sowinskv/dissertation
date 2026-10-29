"""
experiment 03: one conditional QNN for the whole H4 dissociation curve (RQ2).

plain VQE (exp02) trains a new set of angles for every geometry. here one UCCSD circuit gets
its angles from a small encoder of the atom spacing d (Meta-VQE style, see src/conditional.py):

    theta(d) = W @ [T_0(x), ..., T_K(x)],   x = d mapped from [0.5, 2.5] A to [-1, 1]

the encoder is trained on M spacings with the mean energy as the loss: no FCI labels are used.
afterwards the model predicts the energy at any d with one circuit evaluation.

grid: encoder degree K in {1 (linear, original Meta-VQE), 2, 3, 5}  x  M in {3, 5, 9}
training spacings (evenly spaced in 0.5-2.5 A, M = 9 are the spacings of exp02)  x  5 random
initialisations of W. every model is evaluated on a test grid 0.40-3.00 A (step 0.05):
inside 0.5-2.5 A = interpolation, outside = extrapolation.

references on the same test grid:
  - HF, CCSD, FCI from PySCF (FCI also checked against the qubit hamiltonian)
  - UCCSD trained separately at every spacing, three starts: from HF (theta = 0, as in exp02)
    and warm-started from the neighbouring spacing, sweeping up and sweeping down. the best
    of the three is the best this circuit can do at that spacing ("UCCSD per point"): the
    conditional model cannot beat it, so (E_cond - E_UCCSD) is the price of the encoder.

run (from repo root):  python experiments/exp03_conditional.py [--workers 2]
output: results/exp03_h4_conditional_results.csv  (conditional models, one row per model x test spacing)
        results/exp03_h4_conditional_summary.csv  (one row per conditional model)
        results/exp03_h4_reference.csv            (references per test spacing)
        results/exp03_h4_conditional_meta.json
partial results: results/exp03_h4_conditional_partial.jsonl (resume after an interruption)
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # one BLAS thread per worker (see exp02_h4_ansatze.py)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ansatze, hamiltonians as hm, runlog, tracking  # noqa: E402
from src.classical import reference_energies  # noqa: E402
from src.conditional import ConditionalQNN  # noqa: E402
from src.fastsim import FastCircuit  # noqa: E402
from src.train import run_vqe  # noqa: E402

warnings.filterwarnings("ignore")

CHEM_ACC_mHa = 1.6
CONFIG = {
    "molecule": "H4 linear chain",
    "basis": hm.BASIS,
    "circuit": "UCCSD (26 angles)",
    "encoder": "theta(d) = W @ Chebyshev features T_0..T_K of d, scaled from the training range to [-1, 1]",
    "train_range_A": [0.5, 2.5],
    "degrees": [1, 2, 3, 5],
    "n_train": [3, 5, 9],
    "seeds": [0, 1, 2, 3, 4],
    "encoder_init": "uniform(-0.05, 0.05)",
    "loss": "mean energy over the training spacings (no energy labels)",
    "test_grid_A": [0.40, 3.00, 0.05],
    "uccsd_per_point": "best of three starts: HF, warm start sweeping up, warm start sweeping down",
    "optimizer": "L-BFGS-B, central finite-difference gradient (batched), chain rule for the encoder",
    "maxiter": 5000,
    "gtol": 1e-7,
    "chemical_accuracy_mHa": CHEM_ACC_mHa,
}
NAME = "exp03_h4_conditional"
PARTIAL = Path(f"results/{NAME}_partial.jsonl")
GRID = [round(x, 2) for x in np.arange(0.40, 3.0001, 0.05)]
LO, HI = CONFIG["train_range_A"]
EXP = tracking.Experiment(
    NAME, title="One conditional QNN for the whole H₄ curve",
    description="UCCSD whose angles come from an encoder of the atom spacing (Meta-VQE style), "
                "trained on 3, 5 or 9 spacings with energy only, no FCI labels. Tested on "
                "0.4-3.0 Å: interpolation inside 0.5-2.5 Å, extrapolation outside. "
                "Research question 2.",
    config=CONFIG, figures=["fig_exp03_h4_curve.png", "fig_exp03_h4_grid.png"])

# filled in each worker by _init: hamiltonians and exact states on the test grid, the circuit
DATA, SIM, CIRC, QP_CHECK = {}, None, None, None


def train_spacings(m):
    return [round(x, 2) for x in np.linspace(LO, HI, m)]


def _init(data):
    global DATA, SIM, CIRC, QP_CHECK
    DATA = data
    QP_CHECK = hm.build(hm.h_chain(4, 1.0))
    CIRC = ansatze.uccsd(QP_CHECK)          # the same circuit at every spacing
    SIM = FastCircuit(CIRC)


def _vqe(d, theta0, run_id, start):
    h, e_shift, e_fci, psi_fci = DATA[d]
    params = {"d_A": d, "method": "UCCSD", "start": start}
    with EXP.run(run_id, params, reference=e_fci) as run:
        res = run_vqe(SIM, h, e_shift, theta0, maxiter=CONFIG["maxiter"], gtol=CONFIG["gtol"],
                      callback=run.log)
        row = {**params, "E_Ha": res.energy, "dE_mHa": 1e3 * (res.energy - e_fci),
               "fidelity": float(abs(np.vdot(psi_fci, res.state)) ** 2),
               "evaluations": res.evaluations, "iterations": res.iterations,
               "converged": res.converged}
        run.finish(**{k: v for k, v in row.items() if k not in params},
                   theta=[float(t) for t in res.theta])
    return row, res.theta


def task_point(d):
    """UCCSD from HF at one spacing."""
    row, _ = _vqe(d, np.zeros(SIM.num_parameters), f"d{d:.2f}_UCCSD_hf", "hf")
    return [row]


def task_sweep(direction):
    """UCCSD along the grid, each spacing warm-started from the previous one."""
    rows, theta = [], np.zeros(SIM.num_parameters)
    for d in (GRID if direction == "up" else GRID[::-1]):
        row, theta = _vqe(d, theta, f"d{d:.2f}_UCCSD_{direction}", direction)
        rows.append(row)
    return rows


def _d_eq(ds, es):
    """minimum of the curve: parabola through the lowest grid point and its neighbours."""
    k = int(np.clip(np.argmin(es), 1, len(es) - 2))
    a, b, _ = np.polyfit(ds[k - 1:k + 2], es[k - 1:k + 2], 2)
    return float(-b / (2 * a))


def task_conditional(args):
    degree, m, seed, e_uccsd = args
    tr = train_spacings(m)
    model = ConditionalQNN(SIM, degree, LO, HI)
    w0 = np.random.default_rng(seed).uniform(-0.05, 0.05, model.shape)
    params = {"degree": degree, "n_train": m, "seed": seed}
    ref = float(np.mean([DATA[d][2] for d in tr]))       # mean FCI energy of the training set
    run_id = f"QNNcond_K{degree}_M{m}_s{seed}"
    with EXP.run(run_id, params, reference=ref) as run:
        t0 = time.time()
        res = model.train(tr, [DATA[d][0] for d in tr], [DATA[d][1] for d in tr], w0,
                          maxiter=CONFIG["maxiter"], gtol=CONFIG["gtol"], callback=run.log)
        seconds = time.time() - t0
        _verify_with_qiskit(model, res.weights)

        ds = np.array(GRID)
        psi = model.states(res.weights, ds)
        rows = []
        for d, v in zip(GRID, psi):
            h, e_shift, e_fci, psi_fci = DATA[d]
            e = float(np.real(np.vdot(v, h @ v))) + e_shift
            rows.append({**params, "d_A": d, "in_train": d in tr,
                         "region": "interpolation" if LO - 1e-9 <= d <= HI + 1e-9 else "extrapolation",
                         "E_Ha": e, "dE_mHa": 1e3 * (e - e_fci),
                         "gap_to_uccsd_mHa": 1e3 * (e - e_uccsd[d]),
                         "fidelity": float(abs(np.vdot(psi_fci, v)) ** 2)})
        err = np.array([r["dE_mHa"] for r in rows])
        gap = np.array([r["gap_to_uccsd_mHa"] for r in rows])
        inside = np.array([r["region"] == "interpolation" for r in rows])
        unseen = inside & ~np.array([r["in_train"] for r in rows])
        e_curve = np.array([r["E_Ha"] for r in rows])
        e_fci_curve = np.array([DATA[d][2] for d in GRID])
        d_eq, d_eq_fci = _d_eq(ds, e_curve), _d_eq(ds, e_fci_curve)
        summary = {
            **params, "encoder_params": int(res.weights.size),
            "loss_Ha": res.loss, "train_mean_dE_mHa": 1e3 * (res.loss - ref),
            "interp_max_dE_mHa": float(err[inside].max()),
            "interp_median_dE_mHa": float(np.median(err[inside])),
            "interp_frac_chem_acc": float(np.mean(err[inside] < CHEM_ACC_mHa)),
            "interp_max_gap_mHa": float(gap[inside].max()),
            "unseen_max_gap_mHa": float(gap[unseen].max()),
            "extrap_short_max_dE_mHa": float(err[ds < LO - 1e-9].max()),
            "extrap_long_max_dE_mHa": float(err[ds > HI + 1e-9].max()),
            "d_eq_A": d_eq, "d_eq_error_A": d_eq - d_eq_fci,
            "evaluations": res.evaluations, "iterations": res.iterations,
            "converged": res.converged, "train_seconds": round(seconds, 1),
        }
        run.finish(**{k: v for k, v in summary.items() if k not in params},
                   weights=res.weights.tolist())
    return rows, summary


def _verify_with_qiskit(model, weights):
    """the energy fastsim reports for the trained model = qiskit's StatevectorEstimator (d = 1.0)."""
    from qiskit.primitives import StatevectorEstimator
    theta = model.angles(weights, [1.0])[0]
    ours = SIM.expectation(theta[None, :], QP_CHECK.matrix())[0]
    ev = StatevectorEstimator().run([(CIRC, QP_CHECK.hamiltonian, theta)]).result()[0].data.evs
    assert abs(float(ev) - ours) < 1e-9, "fastsim and qiskit disagree"


def _job(job):
    kind, arg = job
    if kind == "point":
        return kind, arg, task_point(arg)
    if kind == "sweep":
        return kind, arg, task_sweep(arg)
    rows, summary = task_conditional(arg)
    return kind, list(arg[:3]), {"rows": rows, "summary": summary}


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    started = time.time()

    # references and hamiltonians on the test grid (classical, cheap)
    data, refs = {}, {}
    for d in GRID:
        atom = hm.h_chain(4, d)
        qp = hm.build(atom)
        ref = reference_energies(atom)
        e_fci, psi_fci, _ = hm.exact_ground_state(qp)
        assert abs(e_fci - ref["E_FCI_Ha"]) < 1e-8, f"qubit hamiltonian != FCI at d={d}"
        data[d] = (qp.matrix(), qp.e_shift, e_fci, psi_fci)
        refs[d] = ref
    print(f"references ready for {len(GRID)} spacings")

    done = [json.loads(l) for l in PARTIAL.read_text().splitlines() if l] if PARTIAL.exists() else []
    seen = {(r["kind"], json.dumps(r["key"])) for r in done}
    n_cond = len(CONFIG["degrees"]) * len(CONFIG["n_train"]) * len(CONFIG["seeds"])
    EXP.start(n_runs=3 * len(GRID) + n_cond)
    PARTIAL.parent.mkdir(exist_ok=True)

    def run_jobs(pool, jobs, log):
        jobs = [j for j in jobs if (j[0], json.dumps(list(j[1][:3]) if j[0] == "cond" else j[1])) not in seen]
        for i, (kind, key, out) in enumerate(pool.imap_unordered(_job, jobs), 1):
            rec = {"kind": kind, "key": key, "out": out}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            done.append(rec)
            if kind == "cond":
                s = out["summary"]
                print(f"[{i}/{len(jobs)}] K={s['degree']} M={s['n_train']} seed={s['seed']}  "
                      f"max error 0.5-2.5 A {s['interp_max_dE_mHa']:.2f} mHa, "
                      f"extrapolation {s['extrap_long_max_dE_mHa']:.1f} mHa, "
                      f"{s['evaluations']} evaluations", flush=True)
            else:
                print(f"[{i}/{len(jobs)}] UCCSD {kind} {key} done", flush=True)

    # spawn, not fork: the parent already runs library threads (PySCF, qiskit), and a forked
    # child that inherits them can deadlock inside qiskit
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(data,)) as pool, open(PARTIAL, "a") as log:
        # 1. UCCSD at every spacing: the two sweeps are long, start them first
        run_jobs(pool, [("sweep", "up"), ("sweep", "down")] + [("point", d) for d in GRID], log)
        per_point = {}
        for rec in done:
            if rec["kind"] in ("point", "sweep"):
                for r in rec["out"]:
                    per_point.setdefault(r["d_A"], {})[r["start"]] = r
        e_uccsd = {d: min(v["E_Ha"] for v in per_point[d].values()) for d in GRID}
        # 2. conditional models, expensive ones first
        jobs = [("cond", (K, m, s, e_uccsd)) for K in sorted(CONFIG["degrees"], reverse=True)
                for m in sorted(CONFIG["n_train"], reverse=True) for s in CONFIG["seeds"]]
        run_jobs(pool, jobs, log)

    ref_rows = []
    for d in GRID:
        r, pp = refs[d], per_point[d]
        best = min(pp, key=lambda k: pp[k]["E_Ha"])
        row = {"d_A": d, "E_HF_Ha": r["E_HF_Ha"], "E_CCSD_Ha": r["E_CCSD_Ha"],
               "E_FCI_Ha": r["E_FCI_Ha"], "hf_weight": r["hf_weight"]}
        for start in ("hf", "up", "down"):
            row[f"E_UCCSD_{start}_Ha"] = pp[start]["E_Ha"]
            row[f"evals_UCCSD_{start}"] = pp[start]["evaluations"]
        row.update({"E_UCCSD_Ha": pp[best]["E_Ha"], "UCCSD_best_start": best,
                    "dE_HF_mHa": 1e3 * (r["E_HF_Ha"] - r["E_FCI_Ha"]),
                    "dE_CCSD_mHa": 1e3 * (r["E_CCSD_Ha"] - r["E_FCI_Ha"]),
                    "dE_UCCSD_hf_mHa": 1e3 * (pp["hf"]["E_Ha"] - r["E_FCI_Ha"]),
                    "dE_UCCSD_mHa": 1e3 * (pp[best]["E_Ha"] - r["E_FCI_Ha"])})
        ref_rows.append(row)
    _write_csv(Path("results/exp03_h4_reference.csv"), ref_rows)

    cond = sorted((r for r in done if r["kind"] == "cond"), key=lambda r: r["key"])
    _write_csv(Path(f"results/{NAME}_summary.csv"), [r["out"]["summary"] for r in cond])
    runlog.save(NAME, [row for r in cond for row in r["out"]["rows"]], CONFIG, started)
    PARTIAL.unlink()
    EXP.finish()


if __name__ == "__main__":
    main()
