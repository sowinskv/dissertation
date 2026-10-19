"""
exp02 follow-up: does a spin penalty keep the hardware-efficient ansatz out of wrong-spin traps?

the spin check of exp02 found HEA runs ending exactly in the quintet (all four spins aligned,
S^2 = 6) and many runs mixing singlet and triplet. HEA does not conserve spin, so nothing in
the cost stops it from leaving the singlet sector. here the cost gets a penalty term:

    cost(theta) = <H> + mu * <S^2>

the ground state of H4 is a singlet (S^2 = 0), so the penalty is zero there and the minimum
of the cost is still the exact ground state. every wrong-spin state is pushed up by mu * S^2.
note: an eigenstate of H is also an eigenstate of H + mu S^2 (they commute), so a run that
reaches the quintet can still stop there; the penalty only makes the way there uphill.

a/b design: all arms (mu = 0 and mu > 0) use the same hamiltonian (fixed orbital signs),
the same circuits and the same random starts, so every penalised run has a twin without it.
we report the true energy <H> (not the cost), <S^2>, <S_z>, <N>, fidelity with FCI, and
whether the run ended in an exact quintet eigenstate.

run (from repo root):  python experiments/exp02_penalty.py [--workers 2]
output: results/exp02_h4_penalty_results.csv, results/exp02_h4_penalty_meta.json
partial results: results/exp02_h4_penalty_partial.jsonl (resume after an interruption)
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
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ansatze, hamiltonians as hm, runlog, tracking  # noqa: E402
from src.train import run_vqe  # noqa: E402

warnings.filterwarnings("ignore")

CONFIG = {
    "molecule": "H4 linear chain",
    "basis": hm.BASIS,
    "spacings_A": [1.5, 1.75, 2.0, 2.25, 2.5],     # where exp02 found spin traps
    "hea_layers": [2, 6],
    "seeds": [0, 1, 2, 3, 4],
    "mu_Ha": [0.0, 0.1, 1.0],                      # penalty strength; 0 = plain VQE (control arm)
    "cost": "<H> + mu <S^2>",
    "hea_init": "uniform(-0.1, 0.1), same seed in both arms",
    "optimizer": "L-BFGS-B, central finite-difference gradient (batched)",
    "maxiter": 10000,
    "gtol": 1e-7,
    "eigenstate_variance_tol_Ha2": 1e-8,
    # completes the curve for the closing figure: the short spacings, best setting only
    "extra": {"spacings_A": [0.5, 0.75, 1.0, 1.25], "hea_layers": [6], "mu_Ha": [0.1]},
}
NAME = "exp02_h4_penalty"
PARTIAL = Path(f"results/{NAME}_partial.jsonl")
EXP = tracking.Experiment(
    NAME, title="Spin penalty for the hardware-efficient ansatz, H₄ chain",
    description="Same HEA runs with and without a spin penalty μ⟨S²⟩ in the cost (μ = 0, 0.1, 1 Ha), "
                "identical random starts, spacing 1.5-2.5 Å. Does the penalty keep VQE out of "
                "the quintet traps found in exp02? Research question 1.",
    config=CONFIG, figures=["fig_exp02_h4_penalty.png"])


def _run_id(d, layers, seed, mu):
    return f"d{d:.2f}_HEA_L{layers}_s{seed}_mu{mu:g}"


def _expect(m, v):
    return float(np.real(np.vdot(v, m @ v)))


def task(args):
    d, layers, seed, mu = args
    qp = hm.build(hm.h_chain(4, d))
    e_fci, psi_exact, _ = hm.exact_ground_state(qp)
    h = qp.matrix()
    s2_op, sz_op = hm.spin_operators(qp)
    s2, sz, nmat = s2_op.to_matrix(), sz_op.to_matrix(), qp.number_op.to_matrix()
    circ = ansatze.hea(qp, layers)
    theta0 = np.random.default_rng(seed).uniform(-0.1, 0.1, circ.num_parameters)

    params = {"d_A": d, "method": "HEA", "layers": layers, "seed": seed, "mu_Ha": mu}
    with EXP.run(_run_id(d, layers, seed, mu), params, reference=e_fci) as run:
        res = run_vqe(circ, h + mu * s2, qp.e_shift, theta0, maxiter=CONFIG["maxiter"],
                      gtol=CONFIG["gtol"], callback=run.log)
        psi = res.state
        e = _expect(h, psi) + qp.e_shift                 # true energy, without the penalty
        var = max(_expect(h @ h, psi) - (e - qp.e_shift) ** 2, 0.0)
        s2v = _expect(s2, psi)
        row = {
            **params,
            "E_Ha": e, "dE_mHa": 1e3 * (e - e_fci), "cost_Ha": res.energy,
            "S2": s2v, "Sz": _expect(sz, psi), "N": _expect(nmat, psi),
            "fidelity": float(abs(np.vdot(psi_exact, psi)) ** 2),
            "energy_variance_Ha2": var,
            "quintet_trap": bool(var < CONFIG["eigenstate_variance_tol_Ha2"] and abs(s2v - 6) < 1e-3),
            **ansatze.resources(circ),
            "evaluations": res.evaluations, "iterations": res.iterations,
            "converged": res.converged,
        }
        run.finish(**{k: v for k, v in row.items() if k not in params},
                   theta=[float(t) for t in res.theta])
    return row


def _typed(r):
    """a row read back from the results CSV, with its original types."""
    out = {}
    for k, v in r.items():
        if v in ("True", "False"):
            out[k] = v == "True"
        elif k in ("layers", "seed", "params", "cnot", "depth", "evaluations", "iterations"):
            out[k] = int(float(v))
        elif k == "method":
            out[k] = v
        else:
            out[k] = float(v)
    return out


def _key(r):
    return (r["d_A"], r["layers"], r["seed"], r["mu_Ha"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    started = time.time()

    done = [json.loads(l) for l in PARTIAL.read_text().splitlines() if l] if PARTIAL.exists() else []
    out_csv = Path(f"results/{NAME}_results.csv")
    if not done and out_csv.exists():     # finished earlier: keep those runs, add only new ones
        done = [_typed(r) for r in csv.DictReader(out_csv.open())]
    done_keys = {_key(r) for r in done}
    grids = [CONFIG, CONFIG["extra"]]
    # expensive deep circuits first, so the two workers finish together
    jobs = [(d, L, s, mu) for g in grids for L in sorted(g["hea_layers"], reverse=True)
            for d in g["spacings_A"] for s in CONFIG["seeds"] for mu in g["mu_Ha"]
            if (d, L, s, mu) not in done_keys]
    total = sum(len(g["spacings_A"]) * len(g["hea_layers"]) * len(g["mu_Ha"]) for g in grids) * len(CONFIG["seeds"])
    print(f"{len(done)} runs already done, {len(jobs)} to run")
    EXP.start(n_runs=total)

    PARTIAL.parent.mkdir(exist_ok=True)
    with Pool(args.workers) as pool, open(PARTIAL, "a") as log:
        for i, r in enumerate(pool.imap_unordered(task, jobs), 1):
            log.write(json.dumps(r) + "\n")
            log.flush()
            done.append(r)
            print(f"[{i}/{len(jobs)}] d={r['d_A']} L={r['layers']} seed={r['seed']} mu={r['mu_Ha']}  "
                  f"dE={r['dE_mHa']:.3f} mHa  S2={r['S2']:.3f}"
                  f"{'  QUINTET' if r['quintet_trap'] else ''}", flush=True)

    columns = ["d_A", "method", "layers", "seed", "mu_Ha", "E_Ha", "dE_mHa", "cost_Ha", "S2", "Sz",
               "N", "fidelity", "energy_variance_Ha2", "quintet_trap", "params", "cnot", "depth",
               "evaluations", "iterations", "converged"]
    rows = sorted(done, key=lambda r: (r["d_A"], r["layers"], r["seed"], r["mu_Ha"]))
    runlog.save(NAME, [{c: r[c] for c in columns} for r in rows], CONFIG, started)
    PARTIAL.unlink()
    EXP.finish()


if __name__ == "__main__":
    main()
