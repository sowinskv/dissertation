"""
exp02 follow-up: which states do the VQE runs of exp02 end in? (no new training)

some HEA runs on the H4 chain stop far above FCI with ~zero fidelity, and the same error
repeats across depths and seeds. hypothesis: they are trapped in an exact eigenstate of the
wrong spin. any eigenstate is a stationary point of the VQE energy (the gradient vanishes),
so the optimizer stops there. HEA conserves neither electron number nor spin, so it can reach
such states; <N> = 4 alone does not reveal them.

for every exp02 run we rebuild the final state from the saved parameters (results/runs/exp02_h4)
and measure: total spin <S^2> (0 singlet, 2 triplet, 6 quintet), <S_z>, <N>, the energy
variance <H^2> - <H>^2 (0 for an exact eigenstate) and the closest exact eigenstate.

orbital signs: exp02 ran before hamiltonians.build fixed the sign of the molecular orbitals,
so each run may have used other signs than today's hamiltonian. flipping orbital i multiplies
every basis state by (-1)^(n_i), so for every run we try all 2^4 sign patterns and keep the
one that reproduces its logged energy (and fidelity) exactly.

run (from repo root):  python experiments/exp02_spin_check.py
output: results/exp02_h4_spin_results.csv, results/exp02_h4_spin_meta.json,
        results/exp02_h4_spectrum.csv (lowest exact states with their spin, per spacing)
"""

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qiskit_nature.second_q.operators import FermionicOp  # noqa: E402
from qiskit_nature.second_q.properties import AngularMomentum, Magnetization  # noqa: E402

from src import ansatze, hamiltonians as hm, runlog  # noqa: E402
from src.fastsim import FastCircuit  # noqa: E402

warnings.filterwarnings("ignore")

RUNS = Path("results/runs/exp02_h4")
CONFIG = {
    "source": "results/runs/exp02_h4 (saved final parameters of exp02)",
    "spacings_A": [round(x, 2) for x in np.arange(0.5, 2.51, 0.25)],
    "hea_layers": [1, 2, 3, 4, 5, 6],
    "seeds": [0, 1, 2, 3, 4],
    "eigenstate_variance_tol_Ha2": 1e-8,
    "spectrum_states_kept": 8,
}
SPIN_NAME = {0: "singlet", 2: "triplet", 6: "quintet"}


def spin_name(s2):
    k = min(SPIN_NAME, key=lambda v: abs(v - s2))
    return SPIN_NAME[k] if abs(k - s2) < 1e-3 else "mixed"


def operators(qp):
    """S^2, S_z and the per-orbital parities (-1)^(n_i) as matrices on the reduced qubits."""
    n = qp.problem.num_spatial_orbitals
    s2 = qp.mapper.map(AngularMomentum(n).second_q_ops()["AngularMomentum"]).to_matrix()
    sz = qp.mapper.map(Magnetization(n).second_q_ops()["Magnetization"]).to_matrix()
    parities = []
    for i in range(n):
        ni = qp.mapper.map(FermionicOp({f"+_{i} -_{i}": 1.0, f"+_{i + n} -_{i + n}": 1.0},
                                       num_spin_orbitals=2 * n)).to_matrix()
        parities.append(np.where(np.round(np.real(np.diag(ni))) % 2 == 1, -1.0, 1.0))
    return s2, sz, parities


def expect(m, v):
    return float(np.real(np.vdot(v, m @ v)))


def spectrum(qp, h, nmat, s2, sz):
    """exact eigenstates with the molecule's electron count, lowest first, with their spin."""
    evals, evecs = np.linalg.eigh(h)
    out = []
    for k in range(len(evals)):
        v = evecs[:, k]
        if abs(expect(nmat, v) - qp.n_electrons) < 1e-6:
            out.append({"E_Ha": evals[k] + qp.e_shift, "S2": expect(s2, v), "Sz": expect(sz, v), "vec": v})
    return out


def restore(psi, record, h, e_shift, psi_exact, parities):
    """undo the orbital signs the run was trained with: return the state in today's basis."""
    logged_e, logged_f = record["metrics"]["E_Ha"], record["metrics"].get("fidelity")
    hits = []
    for flips in itertools.product([0, 1], repeat=len(parities)):
        d = np.prod([p for p, f in zip(parities, flips) if f], axis=0) if any(flips) else 1.0
        v = d * psi
        if abs(expect(h, v) + e_shift - logged_e) < 1e-8:
            f = abs(np.vdot(psi_exact, v)) ** 2
            if logged_f is None or abs(f - logged_f) < 1e-6:
                hits.append(v)
    if not hits:
        raise RuntimeError(f"no orbital sign pattern reproduces run {record['run_id']}")
    return hits[0]


def main():
    started = time.time()
    rows, spec_rows = [], []
    for d in CONFIG["spacings_A"]:
        qp = hm.build(hm.h_chain(4, d))
        h, nmat = qp.matrix(), qp.number_op.to_matrix()
        s2, sz, parities = operators(qp)
        spec = spectrum(qp, h, nmat, s2, sz)
        e_fci, psi_exact = spec[0]["E_Ha"], spec[0]["vec"]
        for k, st in enumerate(spec[:CONFIG["spectrum_states_kept"]]):
            spec_rows.append({"d_A": d, "k": k, "E_Ha": st["E_Ha"], "dE_mHa": 1e3 * (st["E_Ha"] - e_fci),
                              "S2": round(st["S2"], 6), "Sz": round(st["Sz"], 6), "spin": spin_name(st["S2"])})

        jobs = [("UCCSD", None, None, ansatze.uccsd(qp))]
        jobs += [("HEA", L, s, None) for L in CONFIG["hea_layers"] for s in CONFIG["seeds"]]
        sims = {}
        for method, layers, seed, circ in jobs:
            run_id = f"d{d:.2f}_{method}" + (f"_L{layers}_s{seed}" if layers else "")
            record = json.loads((RUNS / f"{run_id}.json").read_text())
            key = method if layers is None else layers
            if key not in sims:
                sims[key] = FastCircuit(circ if circ is not None else ansatze.hea(qp, layers))
            psi = sims[key].states(np.array(record["metrics"]["theta"]))[0]
            v = restore(psi, record, h, qp.e_shift, psi_exact, parities)

            e = expect(h, v) + qp.e_shift
            overlaps = np.array([abs(np.vdot(st["vec"], v)) ** 2 for st in spec])
            k = int(np.argmax(overlaps))
            var = max(expect(h @ h, v) - (e - qp.e_shift) ** 2, 0.0)
            s2v = expect(s2, v)
            rows.append({
                "d_A": d, "method": method, "layers": layers, "seed": seed,
                "dE_mHa": 1e3 * (e - e_fci), "S2": s2v, "Sz": expect(sz, v), "N": expect(nmat, v),
                "energy_variance_Ha2": var,
                "is_eigenstate": var < CONFIG["eigenstate_variance_tol_Ha2"],
                "spin": spin_name(s2v),
                "fidelity_ground": float(overlaps[0]),
                "closest_state": k, "closest_overlap": float(overlaps[k]),
                "closest_spin": spin_name(spec[k]["S2"]),
                "closest_dE_mHa": 1e3 * (spec[k]["E_Ha"] - e_fci),
            })
        print(f"d={d}: done")

    runlog.save("exp02_h4_spin", rows, CONFIG, started)
    keys = list(spec_rows[0])
    with open("results/exp02_h4_spectrum.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in spec_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print("saved: results/exp02_h4_spectrum.csv")

    # short summary on the console
    hea = [r for r in rows if r["method"] == "HEA"]
    traps = [r for r in hea if r["is_eigenstate"] and r["closest_state"] != 0]
    print(f"\nHEA runs: {len(hea)}; ended in an excited exact eigenstate: {len(traps)}")
    for spin in ("singlet", "triplet", "quintet"):
        sel = [r for r in traps if r["spin"] == spin]
        if sel:
            print(f"  {spin}: {len(sel)} runs, e.g. d={sel[0]['d_A']} L={sel[0]['layers']} "
                  f"s={sel[0]['seed']}, error {sel[0]['dE_mHa']:.2f} mHa, S_z={sel[0]['Sz']:+.2f}")
    near = [r for r in hea if not r["is_eigenstate"]]
    print(f"  not an eigenstate: {len(near)} runs, mean <S^2> = {np.mean([r['S2'] for r in near]):.2f}")
    uc = [r for r in rows if r["method"] == "UCCSD"]
    print(f"UCCSD: max |<S^2>| = {max(abs(r['S2']) for r in uc):.1e}")


if __name__ == "__main__":
    main()
