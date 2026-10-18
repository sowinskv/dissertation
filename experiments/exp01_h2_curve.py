"""
experiment 01: potential energy curve of H2 (validation of the whole pipeline).

compares four methods for H2 in the STO-3G basis:
  - HF    (hartree-fock, classical, PySCF)               -> starting point, no correlation
  - FCI   (full configuration interaction, PySCF)        -> exact result in this basis
  - VQE-UCCSD (QNN with a chemistry-inspired ansatz)
  - VQE-HEA   (QNN with a hardware-efficient ansatz, 2 layers)

run (from repo root):  python experiments/exp01_h2_curve.py
output: results/exp01_h2_results.csv, results/exp01_h2_meta.json
figure: python experiments/exp01_plot.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ansatze, hamiltonians as hm, runlog, tracking  # noqa: E402
from src.classical import reference_energies  # noqa: E402
from src.train import run_vqe  # noqa: E402

warnings.filterwarnings("ignore")

CONFIG = {
    "molecule": "H2",
    "basis": hm.BASIS,
    "bond_lengths_A": [round(x, 3) for x in np.linspace(0.3, 3.0, 19)],
    "hea_layers": 2,
    "seed": 42,
    "hea_init": "uniform(-0.1, 0.1)",
    "optimizer": "L-BFGS-B, central finite-difference gradient (batched)",
}
CHEM_ACC_MHA = 1.6
EXP = tracking.Experiment(
    "exp01_h2", title="Potential energy curve of H₂",
    description="Validation of the pipeline: HF, FCI, VQE-UCCSD and VQE-HEA (2 layers) "
                "on H₂ in STO-3G, bond length 0.3-3.0 Å.",
    config=CONFIG, figures=["fig_exp01_h2.png"])


def _train(name, R, circuit, qp, e_fci, theta0, seed=None):
    params = {"R_A": R, "method": name, "seed": seed}
    with EXP.run(f"R{R:.2f}_{name}", params, reference=e_fci) as run:
        res = run_vqe(circuit, qp.matrix(), qp.e_shift, theta0, callback=run.log)
        run.finish(E_Ha=res.energy, dE_mHa=1e3 * abs(res.energy - e_fci),
                   evaluations=res.evaluations, iterations=res.iterations,
                   converged=res.converged, theta=[float(t) for t in res.theta])
    return res


def main():
    started = time.time()
    rng = np.random.default_rng(CONFIG["seed"])
    EXP.start(n_runs=4 * len(CONFIG["bond_lengths_A"]))
    rows = []
    print(f"{'R [Å]':>6} {'HF':>11} {'FCI':>11} {'UCCSD':>11} {'HEA':>11}"
          f" {'ΔE UCCSD [mHa]':>15} {'ΔE HEA [mHa]':>13}")

    for R in CONFIG["bond_lengths_A"]:
        atom = hm.h2(R)
        ref = reference_energies(atom)
        qp = hm.build(atom)
        e_exact, _, _ = hm.exact_ground_state(qp)
        assert abs(e_exact - ref["E_FCI_Ha"]) < 1e-8, f"mapping does not match FCI at R={R}"

        uccsd, hea = ansatze.uccsd(qp), ansatze.hea(qp, CONFIG["hea_layers"])
        for m in ("HF", "FCI"):
            with EXP.run(f"R{R:.2f}_{m}", {"R_A": R, "method": m}, reference=ref["E_FCI_Ha"]) as run:
                run.finish(E_Ha=ref[f"E_{m}_Ha"],
                           dE_mHa=1e3 * abs(ref[f"E_{m}_Ha"] - ref["E_FCI_Ha"]))
        r_ucc = _train("UCCSD", R, uccsd, qp, ref["E_FCI_Ha"], np.zeros(uccsd.num_parameters))
        r_hea = _train("HEA", R, hea, qp, ref["E_FCI_Ha"],
                       rng.uniform(-0.1, 0.1, hea.num_parameters), seed=CONFIG["seed"])

        row = {"R_A": R, "E_HF_Ha": ref["E_HF_Ha"], "E_FCI_Ha": ref["E_FCI_Ha"],
               "E_UCCSD_Ha": r_ucc.energy, "E_HEA_Ha": r_hea.energy,
               "dE_UCCSD_mHa": 1e3 * abs(r_ucc.energy - ref["E_FCI_Ha"]),
               "dE_HEA_mHa": 1e3 * abs(r_hea.energy - ref["E_FCI_Ha"]),
               "evals_UCCSD": r_ucc.evaluations, "evals_HEA": r_hea.evaluations,
               "iters_UCCSD": r_ucc.iterations, "iters_HEA": r_hea.iterations,
               "hf_weight": ref["hf_weight"]}
        rows.append(row)
        print(f"{R:6.2f} {ref['E_HF_Ha']:11.6f} {ref['E_FCI_Ha']:11.6f} {r_ucc.energy:11.6f} "
              f"{r_hea.energy:11.6f} {row['dE_UCCSD_mHa']:15.2e} {row['dE_HEA_mHa']:13.2e}")

    qp = hm.build(hm.h2(CONFIG["bond_lengths_A"][0]))
    res = {"UCCSD": ansatze.resources(ansatze.uccsd(qp)),
           "HEA": ansatze.resources(ansatze.hea(qp, CONFIG["hea_layers"]))}
    CONFIG["circuit_resources"] = res

    print(f"\ntime: {time.time() - started:.1f} s")
    for name in ("UCCSD", "HEA"):
        d = np.array([r[f"dE_{name}_mHa"] for r in rows])
        ev = np.mean([r[f"evals_{name}"] for r in rows])
        print(f"{name:6s} params={res[name]['params']:2d}  CNOT={res[name]['cnot']:2d}  "
              f"depth={res[name]['depth']:3d}  avg evaluations={ev:6.1f}  "
              f"max ΔE={d.max():.2e} mHa  "
              f"points within chemical accuracy: {100 * np.mean(d <= CHEM_ACC_MHA):.0f}%")

    runlog.save("exp01_h2", rows, CONFIG, started)
    EXP.finish()


if __name__ == "__main__":
    main()
