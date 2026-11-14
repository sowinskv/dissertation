"""
experiment 00: data preparation. every input the other experiments use, built and checked once.

there is no external dataset in this thesis: the data are generated. for every molecule and
geometry used in exp01-exp04 (plus LiH, prepared for the next step) this script runs the whole
pipeline and records what it produces:

  geometry  ->  PySCF (RHF, integrals)  ->  second-quantized hamiltonian  ->  ParityMapper with
  two-qubit Z2 reduction  ->  qubit hamiltonian (sum of Pauli strings)
  PySCF     ->  reference energies HF, CCSD, FCI and the HF weight in the FCI state

per geometry it logs the size of the encoding (qubits before / after the reduction, Pauli
terms, 1-norm of the coefficients, qubit-wise commuting measurement groups) and three checks:
  - the lowest eigenvalue of the qubit hamiltonian in the right electron-number sector equals
    PySCF FCI (tolerance 1e-8 Ha)
  - two independent builds give the same qubit hamiltonian (orbital sign convention)
  - CCSD converged
it also writes the train / test splits of every experiment (which geometries each one uses).

run (from repo root):  uv run python experiments/exp00_data/prepare.py
output: results/exp00_data_results.csv, results/exp00_data_splits.csv, results/exp00_data_meta.json
figures: uv run python experiments/exp00_data/plot.py
"""

import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent / "results"   # this experiment's outputs
from src import hamiltonians as hm, runlog, tracking  # noqa: E402
from src.classical import reference_energies  # noqa: E402
from src.hardware import measurement_groups  # noqa: E402

warnings.filterwarnings("ignore")

FCI_TOL_HA = 1e-8
TERM_ATOL = 1e-10   # Pauli coefficients below this are dropped before counting


def _grid(lo, hi, step):
    return [round(x, 3) for x in np.arange(lo, hi + step / 2, step)]


# geometries of every experiment (copied from their CONFIG; tests/test_data.py keeps them in sync)
SPLITS = {
    "exp01 H2 curve": ("H2", "evaluated", [round(x, 3) for x in np.linspace(0.3, 3.0, 19)]),
    "exp02 H4 spacings": ("H4", "evaluated", _grid(0.5, 2.5, 0.25)),
    "exp03 train M=3": ("H4", "train", [round(x, 2) for x in np.linspace(0.5, 2.5, 3)]),
    "exp03 train M=5": ("H4", "train", [round(x, 2) for x in np.linspace(0.5, 2.5, 5)]),
    "exp03 train M=9": ("H4", "train", [round(x, 2) for x in np.linspace(0.5, 2.5, 9)]),
    "exp03 test grid": ("H4", "test", _grid(0.40, 3.00, 0.05)),
    "exp04 simulator": ("H2", "evaluated", [0.5, 0.735, 1.0, 1.5, 2.0, 2.5]),
    "exp04 hardware": ("H2", "evaluated", [0.735, 1.5, 2.5]),
    "next: LiH": ("LiH", "prepared", _grid(1.0, 3.0, 0.25)),
}
EXP03_TRAIN_RANGE_A = [0.5, 2.5]   # inside: interpolation, outside: extrapolation

GEOMETRY = {"H2": hm.h2, "H4": lambda d: hm.h_chain(4, d), "LiH": lambda r: f"Li 0 0 0; H 0 0 {r}"}

CONFIG = {
    "basis": hm.BASIS,
    "mapping": "ParityMapper with two-qubit Z2 reduction (qiskit-nature)",
    "orbital_signs": "largest coefficient of every MO made positive (hamiltonians.fix_orbital_signs)",
    "references": "RHF, CCSD, FCI from PySCF",
    "measurement_groups": "qubit-wise commuting (src/hardware.measurement_groups)",
    "fci_check_tolerance_Ha": FCI_TOL_HA,
    "pauli_coefficient_cutoff": TERM_ATOL,
    "exp03_train_range_A": EXP03_TRAIN_RANGE_A,
    "splits": "geometries of every experiment and their role: exp00_data_splits.csv",
}
NAME = "exp00_data"
EXP = tracking.Experiment(
    NAME, title="Data preparation: geometries, qubit hamiltonians and reference energies",
    description="Every geometry used in the experiments (H₂, H₄ chain, and LiH for the next step) "
                "taken through the whole pipeline: PySCF integrals, parity-mapped qubit hamiltonian, "
                "HF / CCSD / FCI references. Logs the size of each encoding and checks that the "
                "qubit hamiltonian reproduces FCI and is identical between builds.",
    config=CONFIG, root=OUT / "runs",
    figures=["fig_exp00_references.png", "fig_exp00_hamiltonian.png", "fig_exp00_splits.png",
             "fig_exp00_checks.png"])


def geometries():
    """(molecule, d) -> names of the experiments that use it."""
    out = {}
    for name, (mol, _, ds) in SPLITS.items():
        for d in ds:
            out.setdefault((mol, d), []).append(name.split()[0] if mol != "LiH" else "next")
    return {k: sorted(set(v)) for k, v in sorted(out.items(), key=lambda kv: (["H2", "H4", "LiH"].index(kv[0][0]), kv[0][1]))}


def describe(mol, d):
    """run the pipeline for one geometry; returns the CSV row."""
    atom = GEOMETRY[mol](d)
    t0 = time.time()
    ref = reference_energies(atom)
    qp = hm.build(atom)
    e_exact, _, e_any = hm.exact_ground_state(qp)
    again = hm.build(atom).hamiltonian
    diff = (qp.hamiltonian - again).simplify(atol=0)
    rebuild_diff = float(np.max(np.abs(diff.coeffs)))

    ham = qp.hamiltonian.simplify(atol=TERM_ATOL)
    labels = ham.paulis.to_labels()
    non_id = [abs(c) for l, c in zip(labels, ham.coeffs) if set(l) != {"I"}]
    _, groups = measurement_groups(ham)
    n_orb = qp.problem.num_spatial_orbitals
    return {
        "molecule": mol, "d_A": d, "atom": atom,
        "n_atoms": len(atom.split(";")), "n_electrons": qp.n_electrons,
        "spatial_orbitals": n_orb, "qubits_before": 2 * n_orb, "qubits": qp.num_qubits,
        "pauli_terms": len(non_id), "coeff_1norm_Ha": float(sum(non_id)),
        "measurement_groups": len(groups),
        "E_nuc_Ha": float(qp.problem.nuclear_repulsion_energy),
        "E_HF_Ha": ref["E_HF_Ha"], "E_CCSD_Ha": ref["E_CCSD_Ha"], "E_FCI_Ha": ref["E_FCI_Ha"],
        "E_corr_mHa": 1e3 * (ref["E_FCI_Ha"] - ref["E_HF_Ha"]),
        "dE_CCSD_mHa": 1e3 * (ref["E_CCSD_Ha"] - ref["E_FCI_Ha"]),
        "hf_weight": ref["hf_weight"],
        "ccsd_converged": bool(np.isfinite(ref["E_CCSD_Ha"])),
        "fci_check_Ha": abs(e_exact - ref["E_FCI_Ha"]),
        "other_sector_below_mHa": 1e3 * (e_exact - e_any),   # > 0: a wrong-N state lies lower
        "rebuild_diff": rebuild_diff,
        "build_s": round(time.time() - t0, 2),
    }


def main():
    started = time.time()
    geos = geometries()
    EXP.start(n_runs=len(geos))
    rows = []
    print(f"{'mol':>4} {'d [Å]':>6} {'qubits':>6} {'terms':>6} {'groups':>6} {'E_FCI':>11} "
          f"{'HF w':>6} {'check [Ha]':>10} {'other N':>8}")
    for (mol, d), used in geos.items():
        with EXP.run(f"{mol}_d{d:.2f}" if round(d, 2) == d else f"{mol}_d{d:.3f}", {"molecule": mol, "d_A": d},
                     reference=None) as run:
            row = describe(mol, d)
            row["used_in"] = " ".join(used)
            assert row["fci_check_Ha"] < FCI_TOL_HA, f"{mol} d={d}: qubit hamiltonian does not match FCI"
            assert row["rebuild_diff"] < 1e-12, f"{mol} d={d}: two builds differ"
            run.finish(**{k: v for k, v in row.items() if k not in ("molecule", "d_A", "atom")})
        rows.append(row)
        print(f"{mol:>4} {d:6.3f} {row['qubits_before']:>3}->{row['qubits']:<2} {row['pauli_terms']:6d} "
              f"{row['measurement_groups']:6d} {row['E_FCI_Ha']:11.6f} {row['hf_weight']:6.3f} "
              f"{row['fci_check_Ha']:10.1e} {row['other_sector_below_mHa']:8.1f}")

    splits = [{"split": name, "molecule": mol, "role": role, "d_A": d,
               "region": ("interpolation" if EXP03_TRAIN_RANGE_A[0] <= d <= EXP03_TRAIN_RANGE_A[1]
                          else "extrapolation") if name.startswith("exp03") else ""}
              for name, (mol, role, ds) in SPLITS.items() for d in ds]
    with open(OUT / "exp00_data_splits.csv", "w", newline="") as f:   # config: exp00_data_meta.json
        writer = csv.DictWriter(f, fieldnames=list(splits[0]))
        writer.writeheader()
        writer.writerows(splits)
    runlog.save(NAME, rows, CONFIG, started, out_dir=OUT)
    print(f"\n{len(rows)} geometries, all checks passed, {time.time() - started:.1f} s")
    EXP.finish()


if __name__ == "__main__":
    main()
