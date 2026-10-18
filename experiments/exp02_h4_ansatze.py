"""
experiment 02: UCCSD vs hardware-efficient ansatz on a linear H4 chain (RQ1).

H4 in STO-3G (6 qubits after parity reduction). stretching the chain makes electron
correlation strong: HF fails badly and CCSD (not variational) overshoots below FCI.

for every spacing d:
  - classical references from PySCF: HF, CCSD, FCI, weight of HF in the FCI state
  - VQE-UCCSD, started from HF (theta = 0), deterministic
  - VQE-HEA with 1..6 layers, 5 random starts each
per VQE run we record energy error, circuit resources, training cost, <N> and fidelity.

training uses src.fastsim (exact numpy statevector); every final energy is re-checked
with qiskit's StatevectorEstimator.

run (from repo root):  python experiments/exp02_h4_ansatze.py [--workers 2]
output: results/exp02_h4_results.csv (one row per method x spacing x layers x seed),
        results/exp02_h4_meta.json
partial results are checkpointed to results/exp02_h4_partial.jsonl, so an interrupted
run resumes where it stopped.
"""

import os

# one BLAS thread per worker: with several worker processes, multithreaded BLAS on tiny
# matrices oversubscribes the CPU and makes everything ~30x slower
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ansatze, hamiltonians as hm, runlog, tracking  # noqa: E402
from src.classical import reference_energies  # noqa: E402
from src.train import run_vqe  # noqa: E402

warnings.filterwarnings("ignore")

CONFIG = {
    "molecule": "H4 linear chain",
    "basis": hm.BASIS,
    "spacings_A": [round(x, 2) for x in np.arange(0.5, 2.51, 0.25)],
    "hea_layers": [1, 2, 3, 4, 5, 6],
    "seeds": [0, 1, 2, 3, 4],
    "hea_init": "uniform(-0.1, 0.1)",
    "optimizer": "L-BFGS-B, central finite-difference gradient (batched)",
    "maxiter": 10000,
    "gtol": 1e-7,
}
NAME = "exp02_h4"
PARTIAL = Path(f"results/{NAME}_partial.jsonl")
EXP = tracking.Experiment(
    NAME, title="UCCSD vs hardware-efficient ansatz on the H₄ chain",
    description="VQE with UCCSD and HEA (1-6 layers, 5 random starts) on a linear H₄ chain, "
                "spacing 0.5-2.5 Å, against HF, CCSD and FCI. Research question 1.",
    config=CONFIG, figures=["fig_exp02_h4.png", "fig_exp02_h4_cost.png", "fig_exp02_h4_spin.png"])


def _run_id(d, method, layers=None, seed=None):
    rid = f"d{d:.2f}_{method}"
    return rid + (f"_L{layers}_s{seed}" if layers is not None else "")


def _verify_with_qiskit(circuit, qp, theta, energy):
    from qiskit.primitives import StatevectorEstimator
    ev = StatevectorEstimator().run([(circuit, qp.hamiltonian, theta)]).result()[0].data.evs
    assert abs(float(ev) + qp.e_shift - energy) < 1e-9, "fastsim and qiskit disagree"


def _vqe_row(d, method, layers, seed, circuit, qp, e_fci, psi_exact, theta0):
    params = {"d_A": d, "method": method, "layers": layers, "seed": seed}
    with EXP.run(_run_id(d, method, layers, seed), params, reference=e_fci) as run:
        res = run_vqe(circuit, qp.matrix(), qp.e_shift, theta0,
                      maxiter=CONFIG["maxiter"], gtol=CONFIG["gtol"], callback=run.log)
        _verify_with_qiskit(circuit, qp, res.theta, res.energy)
        psi = res.state
        row = {
            **params,
            "E_Ha": res.energy, "dE_mHa": 1e3 * (res.energy - e_fci),
            **ansatze.resources(circuit),
            "evaluations": res.evaluations, "iterations": res.iterations,
            "converged": res.converged,
            "N": float(np.real(psi.conj() @ qp.number_op.to_matrix() @ psi)),
            "fidelity": float(abs(np.vdot(psi_exact, psi)) ** 2),
        }
        run.finish(**{k: v for k, v in row.items() if k not in params},
                   theta=[float(t) for t in res.theta])
    return row


def task_reference(d):
    """classical references + UCCSD for one spacing."""
    atom = hm.h_chain(4, d)
    qp = hm.build(atom)
    ref = reference_energies(atom)
    e_exact, psi_exact, e_any = hm.exact_ground_state(qp)
    assert abs(e_exact - ref["E_FCI_Ha"]) < 1e-8, f"qubit hamiltonian != FCI at d={d}"
    rows = [{"d_A": d, "method": m, "E_Ha": ref[f"E_{m}_Ha"],
             "dE_mHa": 1e3 * (ref[f"E_{m}_Ha"] - ref["E_FCI_Ha"]),
             "hf_weight": ref["hf_weight"]} for m in ("HF", "CCSD", "FCI")]
    for r in rows:   # classical results: logged as runs without a training history
        with EXP.run(_run_id(d, r["method"]), {"d_A": d, "method": r["method"]},
                     reference=ref["E_FCI_Ha"]) as run:
            run.finish(E_Ha=r["E_Ha"], dE_mHa=r["dE_mHa"], hf_weight=r["hf_weight"])
    circ = ansatze.uccsd(qp)
    rows.append(_vqe_row(d, "UCCSD", None, None, circ, qp, e_exact, psi_exact,
                         np.zeros(circ.num_parameters)))
    return rows


def task_hea(args):
    d, layers, seed = args
    qp = hm.build(hm.h_chain(4, d))
    e_exact, psi_exact, _ = hm.exact_ground_state(qp)
    circ = ansatze.hea(qp, layers)
    theta0 = np.random.default_rng(seed).uniform(-0.1, 0.1, circ.num_parameters)
    return [_vqe_row(d, "HEA", layers, seed, circ, qp, e_exact, psi_exact, theta0)]


def _key(row):
    return (row["d_A"], row["method"], row.get("layers"), row.get("seed"))


def _run(job):
    kind, arg = job
    return task_reference(arg) if kind == "ref" else task_hea(arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    started = time.time()

    done = []
    if PARTIAL.exists():
        done = [json.loads(line) for line in PARTIAL.read_text().splitlines() if line]
    done_keys = {_key(r) for r in done}

    jobs = [("ref", d) for d in CONFIG["spacings_A"]
            if (d, "UCCSD", None, None) not in done_keys]
    jobs += [("hea", (d, L, s)) for L in CONFIG["hea_layers"] for d in CONFIG["spacings_A"]
             for s in CONFIG["seeds"] if (d, "HEA", L, s) not in done_keys]
    print(f"{len(done)} rows already done, {len(jobs)} jobs to run")
    EXP.start(n_runs=len(CONFIG["spacings_A"]) * (4 + len(CONFIG["hea_layers"]) * len(CONFIG["seeds"])))

    PARTIAL.parent.mkdir(exist_ok=True)
    with Pool(args.workers) as pool, open(PARTIAL, "a") as log:
        for i, rows in enumerate(pool.imap_unordered(_run, jobs), 1):
            for r in rows:
                log.write(json.dumps(r) + "\n")
            log.flush()
            done += rows
            r = rows[-1]
            print(f"[{i}/{len(jobs)}] d={r['d_A']} {r['method']} L={r.get('layers')} "
                  f"seed={r.get('seed')}  dE={r['dE_mHa']:.3f} mHa", flush=True)

    columns = ["d_A", "method", "layers", "seed", "E_Ha", "dE_mHa", "hf_weight", "params",
               "cnot", "depth", "evaluations", "iterations", "converged", "N", "fidelity"]
    order = {"HF": 0, "CCSD": 1, "FCI": 2, "UCCSD": 3, "HEA": 4}
    rows = sorted(done, key=lambda r: (r["d_A"], order[r["method"]], r.get("layers") or 0,
                                       r.get("seed") or 0))
    rows = [{c: r.get(c, "") if r.get(c) is not None else "" for c in columns} for r in rows]
    runlog.save(NAME, rows, CONFIG, started)
    PARTIAL.unlink()
    EXP.finish()


if __name__ == "__main__":
    main()
