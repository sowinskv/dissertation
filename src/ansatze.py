"""
ansatze: the two QNN architectures compared in RQ1, and their circuit resources.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from src.hamiltonians import QubitProblem

BASIS_GATES = ["cx", "rz", "sx", "x"]   # native gates of a typical IBM processor


def hartree_fock(qp: QubitProblem) -> QuantumCircuit:
    p = qp.problem
    return HartreeFock(p.num_spatial_orbitals, p.num_particles, qp.mapper)


def uccsd(qp: QubitProblem) -> QuantumCircuit:
    """chemistry-inspired ansatz: single and double excitations on top of HF; theta = 0 is HF."""
    p = qp.problem
    return UCCSD(p.num_spatial_orbitals, p.num_particles, qp.mapper,
                 initial_state=hartree_fock(qp))


def hea(qp: QubitProblem, layers: int) -> QuantumCircuit:
    """hardware-efficient ansatz: HF, then `layers` blocks of RY, RZ rotations + linear CNOT chain."""
    return hartree_fock(qp).compose(
        efficient_su2(qp.num_qubits, reps=layers, entanglement="linear"))


def resources(circuit: QuantumCircuit) -> dict:
    """parameters, CNOT count and depth after decomposing into the hardware's native gates."""
    t = transpile(circuit, basis_gates=BASIS_GATES, optimization_level=1)
    return {"params": circuit.num_parameters,
            "cnot": int(t.count_ops().get("cx", 0)),
            "depth": int(t.depth())}
