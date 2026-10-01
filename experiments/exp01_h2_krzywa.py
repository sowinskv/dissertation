"""
experiment 01: potential energy curve of H2.

compares four methods for H2 in the STO-3G basis:
  - HF    (hartree-fock, classical, PySCF)               -> starting point, no correlation
  - FCI   (full configuration interaction, PySCF)        -> exact result in this basis
  - VQE-UCCSD (QNN with a chemistry-inspired ansatz)
  - VQE-HEA   (QNN with a hardware-efficient ansatz)

output: table in the console, CSV file and PNG plot (figure 6.1 in the thesis).

install:  pip install -r requirements.txt
run (from repo root):  python experiments/exp01_h2_krzywa.py
tested with: qiskit 2.5.2, qiskit-nature 0.8.0, pyscf 2.14.0, python 3.12
"""

import csv
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pyscf import gto, scf, fci
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

warnings.filterwarnings("ignore")  # scipy sparse-matrix warnings are harmless here

# ---------------- experiment config ----------------
R_VALUES = np.linspace(0.3, 3.0, 19)   # bond lengths in angstrom
HEA_REPS = 2                           # number of layers in the hardware-efficient ansatz
SEED = 42                              # random seed for reproducibility
CHEM_ACC = 1.6e-3                      # chemical accuracy in Ha
BASIS_GATES = ["cx", "rz", "sx", "x"]  # native gates of a typical IBM processor

estimator = StatevectorEstimator()     # noiseless simulator: exact expectation values
rng = np.random.default_rng(SEED)


# ---------------- step 1: classical chemistry (PySCF) ----------------
def classical_reference(R):
    """return HF and FCI energies computed directly in PySCF (independent check)."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="sto3g", verbose=0)
    mf = scf.RHF(mol).run()
    e_fci, _ = fci.FCI(mf).kernel()
    return mf.e_tot, e_fci


# ---------------- step 2: qubit hamiltonian ----------------
def qubit_problem(R):
    """H2 hamiltonian as a sum of pauli terms on qubits.

    PySCFDriver computes the molecular integrals, second_q_op() writes the
    hamiltonian in second quantization, and ParityMapper maps it to qubits and
    removes 2 qubits using symmetries (4 spin orbitals -> 2 qubits).
    """
    problem = PySCFDriver(atom=f"H 0 0 0; H 0 0 {R}", basis="sto3g").run()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    e_nuc = problem.hamiltonian.nuclear_repulsion_energy  # constant nuclear repulsion
    return problem, mapper, qubit_op, e_nuc


# ---------------- step 3: two QNN architectures ----------------
def build_ansatze(problem, mapper):
    hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    uccsd = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                  initial_state=hf)
    # HEA: layers of RY, RZ rotations + CNOTs between neighbouring qubits, starting from HF
    hea = hf.compose(efficient_su2(hf.num_qubits, reps=HEA_REPS, entanglement="linear"))
    return uccsd, hea


def circuit_resources(circuit):
    """circuit resources after decomposing into the hardware's native gates."""
    t = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=1)
    return {"params": circuit.num_parameters,
            "cnot": t.count_ops().get("cx", 0),
            "depth": t.depth()}


# ---------------- step 4: VQE = training the QNN ----------------
def run_vqe(ansatz, qubit_op, e_nuc, theta0):
    """minimize E(theta) = <psi(theta)|H|psi(theta)> with a classical optimizer."""
    def energy(theta):
        result = estimator.run([(ansatz, qubit_op, theta)]).result()[0]
        return float(result.data.evs) + e_nuc

    res = minimize(energy, theta0, method="L-BFGS-B", options={"maxiter": 500})
    return res.fun, res.nfev


# ---------------- main loop ----------------
def main():
    rows = []
    start = time.time()
    print(f"{'R [Å]':>6} {'HF':>11} {'FCI':>11} {'UCCSD':>11} {'HEA':>11}"
          f" {'ΔE UCCSD [mHa]':>15} {'ΔE HEA [mHa]':>13}")

    for R in R_VALUES:
        e_hf, e_fci = classical_reference(R)
        problem, mapper, qubit_op, e_nuc = qubit_problem(R)

        # sanity check of the mapping: lowest eigenvalue must equal FCI from PySCF
        e_exact = np.linalg.eigvalsh(qubit_op.to_matrix())[0] + e_nuc
        assert abs(e_exact - e_fci) < 1e-8, f"mapping does not match FCI at R={R}"

        uccsd, hea = build_ansatze(problem, mapper)
        e_ucc, nfev_ucc = run_vqe(uccsd, qubit_op, e_nuc, np.zeros(uccsd.num_parameters))
        e_hea, nfev_hea = run_vqe(hea, qubit_op, e_nuc,
                                  rng.uniform(-0.1, 0.1, hea.num_parameters))

        rows.append({"R": R, "HF": e_hf, "FCI": e_fci, "UCCSD": e_ucc, "HEA": e_hea,
                     "dE_UCCSD": abs(e_ucc - e_fci), "dE_HEA": abs(e_hea - e_fci),
                     "nfev_UCCSD": nfev_ucc, "nfev_HEA": nfev_hea})
        print(f"{R:6.2f} {e_hf:11.6f} {e_fci:11.6f} {e_ucc:11.6f} {e_hea:11.6f}"
              f" {1e3 * abs(e_ucc - e_fci):15.2e} {1e3 * abs(e_hea - e_fci):13.2e}")

    # circuit resources (independent of R, so computed once)
    uccsd, hea = build_ansatze(*qubit_problem(R_VALUES[0])[:2])
    res_ucc, res_hea = circuit_resources(uccsd), circuit_resources(hea)

    # summary
    print(f"\ntime: {time.time() - start:.1f} s")
    for name, res, key in [("UCCSD", res_ucc, "UCCSD"), ("HEA", res_hea, "HEA")]:
        d = np.array([r[f"dE_{key}"] for r in rows])
        nfev = np.mean([r[f"nfev_{key}"] for r in rows])
        print(f"{name:6s} params={res['params']:2d}  CNOT={res['cnot']:2d}  "
              f"depth={res['depth']:3d}  avg evaluations={nfev:6.1f}  "
              f"max ΔE={1e3 * d.max():.2e} mHa  "
              f"points within chemical accuracy: {100 * np.mean(d <= CHEM_ACC):.0f}%")

    # save results
    with open("results/exp01_h2_wyniki.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # plot: energy curve + error vs FCI on a log scale
    # (plot labels stay in polish because the figure goes into the thesis)
    R = [r["R"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax1.plot(R, [r["HF"] for r in rows], "o--", color="C0", label="HF")
    ax1.plot(R, [r["FCI"] for r in rows], "k-", label="FCI (dokładne)")
    ax1.plot(R, [r["UCCSD"] for r in rows], "s", color="C1", mfc="none", label="VQE-UCCSD")
    ax1.plot(R, [r["HEA"] for r in rows], "^", color="C2", mfc="none",
             label=f"VQE-HEA (L={HEA_REPS})")
    ax1.set_ylabel("Energia [Ha]")
    ax1.set_title("H₂ / STO-3G: krzywa energii potencjalnej")
    ax1.legend()

    floor = 1e-10  # an error of exactly 0 cannot be drawn on a log scale
    ax2.semilogy(R, [max(abs(r["HF"] - r["FCI"]) * 1e3, floor) for r in rows], "o--",
                 color="C0", label="HF")
    ax2.semilogy(R, [max(r["dE_UCCSD"] * 1e3, floor) for r in rows], "s-",
                 color="C1", label="VQE-UCCSD")
    ax2.semilogy(R, [max(r["dE_HEA"] * 1e3, floor) for r in rows], "^-",
                 color="C2", label="VQE-HEA")
    ax2.axhline(CHEM_ACC * 1e3, color="red", ls=":", label="dokładność chemiczna (1,6 mHa)")
    ax2.set_xlabel("Długość wiązania R [Å]")
    ax2.set_ylabel("|E − E_FCI| [mHa]")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig("results/exp01_h2_krzywa.png", dpi=150)
    print("saved: results/exp01_h2_wyniki.csv, results/exp01_h2_krzywa.png")


if __name__ == "__main__":
    main()