"""
hamiltonians: molecular geometries and their qubit hamiltonians.

pipeline (thesis chapter 3): PySCF integrals -> second-quantized hamiltonian -> ParityMapper
with two-qubit Z2 reduction -> sum of pauli strings on qubits.
"""

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem

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


def build(atom, basis=BASIS):
    problem = PySCFDriver(atom=atom, basis=basis).run()
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
