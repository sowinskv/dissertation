"""
hamiltonians: molecular geometries and their qubit hamiltonians.

pipeline (thesis chapter 3): PySCF integrals -> second-quantized hamiltonian -> ParityMapper
with two-qubit Z2 reduction -> sum of pauli strings on qubits.

molecular orbitals are only defined up to a sign. PySCF picks the sign numerically, so it can
change between runs (e.g. with the number of BLAS threads). the energies do not care, but the
qubit hamiltonian does: with other signs, the same circuit parameters describe another state.
build() therefore fixes the sign of every orbital (largest coefficient positive), so saved
parameters always reproduce the trained state.
"""

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import AngularMomentum, Magnetization

BASIS = "sto3g"


# ---------------- geometries (angstrom) ----------------
def h2(r):
    return f"H 0 0 0; H 0 0 {r}"


def h_chain(n, d):
    """linear chain of n hydrogen atoms with equal spacing d."""
    return "; ".join(f"H 0 0 {i * d}" for i in range(n))


# ---------------- qubit problem ----------------
@dataclass
class QubitProblem:
    atom: str
    problem: ElectronicStructureProblem
    mapper: ParityMapper
    hamiltonian: SparsePauliOp     # electronic part, on qubits
    number_op: SparsePauliOp       # total electron number, on qubits
    e_shift: float                 # constant energy (nuclear repulsion) to add to <H>
    n_electrons: int

    @property
    def num_qubits(self):
        return self.hamiltonian.num_qubits

    def matrix(self):
        """dense hamiltonian matrix (2^n x 2^n); fine for the small registers used here."""
        return self.hamiltonian.to_matrix()


def fix_orbital_signs(mo, tol=1e-6):
    """sign convention for molecular orbitals (columns of mo): the first coefficient whose size
    is within tol of the largest one is made positive. the tolerance keeps symmetric molecules
    (equal-size coefficients, e.g. H2) from flipping on rounding noise."""
    mo = np.array(mo, dtype=float)
    size = np.abs(mo)
    for j in range(mo.shape[1]):
        i = int(np.argmax(size[:, j] >= size[:, j].max() * (1 - tol)))
        if mo[i, j] < 0:
            mo[:, j] *= -1
    return mo


def build(atom, basis=BASIS):
    driver = PySCFDriver(atom=atom, basis=basis)
    driver.run_pyscf()
    driver._calc.mo_coeff = fix_orbital_signs(driver._calc.mo_coeff)   # see module docstring
    problem = driver.to_problem()
    mapper = ParityMapper(num_particles=problem.num_particles)
    ham = mapper.map(problem.hamiltonian.second_q_op())
    number = mapper.map(problem.properties.particle_number.second_q_ops()["ParticleNumber"])
    e_shift = float(sum(problem.hamiltonian.constants.values()))
    return QubitProblem(atom, problem, mapper, ham, number, e_shift, sum(problem.num_particles))


def exact_ground_state(qp: QubitProblem, tol=1e-6):
    """lowest eigenpair inside the correct electron-number sector.

    the parity reduction fixes only the parity of the electron counts, so the reduced
    hamiltonian also contains states with other electron numbers. we pick the lowest
    eigenstate whose <N> equals the molecule's electron count.
    returns (total energy, state vector, lowest energy over all sectors).
    """
    evals, evecs = np.linalg.eigh(qp.matrix())
    nmat = qp.number_op.to_matrix()
    for k in range(len(evals)):
        v = evecs[:, k]
        if abs(np.real(v.conj() @ nmat @ v) - qp.n_electrons) < tol:
            return evals[k] + qp.e_shift, v, evals[0] + qp.e_shift
    raise RuntimeError("no eigenstate with the right electron number")


def spin_operators(qp: QubitProblem):
    """total spin S^2 and its z projection S_z on the reduced qubits (same mapping as H).
    S^2 = 0 singlet, 2 triplet, 6 quintet."""
    n = qp.problem.num_spatial_orbitals
    s2 = qp.mapper.map(AngularMomentum(n).second_q_ops()["AngularMomentum"])
    sz = qp.mapper.map(Magnetization(n).second_q_ops()["Magnetization"])
    return s2, sz
