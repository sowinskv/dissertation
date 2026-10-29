"""
hardware: estimate a VQE energy from measurement counts, with error mitigation (RQ4).

on a real processor (or a noisy simulator of one) the state cannot be read out as a vector;
each run of the circuit ends in a measurement that returns one bit string. the energy is
estimated from many repetitions (shots):

  1. group the Pauli terms of the hamiltonian into sets that can be measured together
     (qubit-wise commuting); for H2 these are {IZ, ZI, ZZ} (measure as is) and {XX}
     (rotate each qubit with H, then measure)
  2. for each group run the circuit + basis change many times; the average of each Pauli's
     eigenvalue (+1 / -1 per bit string) over the shots estimates its expectation value
  3. energy = sum of coefficient x expectation value (+ constant terms)

two error mitigation techniques, both done here by hand so the same code runs on the
simulator and on IBM hardware and every step can be inspected:

  readout mitigation: the processor sometimes reports 1 for a qubit that was 0 (and vice
    versa). we prepare every basis state (00, 01, 10, 11 for two qubits), measure, and get
    the assignment matrix A[measured, prepared]. measured probabilities are then corrected
    with A^-1 (the result can have small negative entries: quasi-probabilities).

  zero-noise extrapolation (ZNE): run the same circuit with the noise deliberately amplified
    by "folding" every two-qubit gate G -> G G^dagger G (same operation, 3x the noise),
    or G (G^dagger G)^2 for 5x. fit energy vs noise factor and extrapolate to factor 0.

reference: K. Temme, S. Bravyi, J. M. Gambetta, Phys. Rev. Lett. 119, 180509 (2017) (ZNE);
           T. Giurgica-Tiron et al., IEEE QCE 2020, arXiv:2005.10921 (folding).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def measurement_groups(op: SparsePauliOp):
    """split a hamiltonian into its constant part and qubit-wise commuting groups.
    returns (constant, [(basis_label, group_op), ...]); basis_label[k] in 'XYZ' for qubit k
    counted from the right, as in Pauli labels."""
    labels, coeffs = op.paulis.to_labels(), op.coeffs
    const = float(np.real(sum(c for l, c in zip(labels, coeffs) if set(l) == {"I"})))
    rest = SparsePauliOp([l for l in labels if set(l) != {"I"}],
                         [c for l, c in zip(labels, coeffs) if set(l) != {"I"}])
    groups = []
    for g in rest.group_commuting(qubit_wise=True):
        basis = ["Z"] * g.num_qubits
        for label in g.paulis.to_labels():
            for k, ch in enumerate(reversed(label)):
                if ch != "I":
                    basis[k] = ch
        groups.append(("".join(reversed(basis)), g))
    return const, groups


def with_measurement(circuit: QuantumCircuit, basis_label):
    """append the basis change (X: H, Y: S^dagger H) and measure every qubit."""
    qc = circuit.copy()
    for k, ch in enumerate(reversed(basis_label)):
        if ch == "X":
            qc.h(k)
        elif ch == "Y":
            qc.sdg(k)
            qc.h(k)
    qc.measure_all()
    return qc


def readout_calibration_circuits(n):
    """one circuit per basis state j (bit k of j = qubit k): prepare it and measure."""
    circs = []
    for j in range(2 ** n):
        qc = QuantumCircuit(n)
        for k in range(n):
            if (j >> k) & 1:
                qc.x(k)
        qc.measure_all()
        circs.append(qc)
    return circs


def probabilities(counts, n):
    """counts {'b_{n-1}..b_0': hits} -> probability vector indexed by int(bitstring, 2)."""
    p = np.zeros(2 ** n)
    for key, hits in counts.items():
        p[int(key.replace(" ", ""), 2)] += hits
    return p / p.sum()


def assignment_matrix(cal_counts, n):
    """A[i, j] = P(measure i | prepared j), from the calibration circuits in order."""
    return np.column_stack([probabilities(c, n) for c in cal_counts])


def pauli_expectation(label, p):
    """<P> from a probability vector over bit strings measured in P's basis."""
    idx = np.arange(len(p))
    eig = np.ones(len(p))
    for k, ch in enumerate(reversed(label)):
        if ch != "I":
            eig *= 1 - 2 * ((idx >> k) & 1)
    return float(eig @ p)


def group_energy(group_op, p):
    return float(sum(np.real(c) * pauli_expectation(l, p)
                     for l, c in zip(group_op.paulis.to_labels(), group_op.coeffs)))


def fold_two_qubit_gates(circuit: QuantumCircuit, factor: int):
    """noise amplification for ZNE: every two-qubit gate G becomes G (G^dagger G)^((factor-1)/2).
    the operation is unchanged, but the circuit has `factor` times as many two-qubit gates.
    applied to the transpiled (hardware) circuit, so no compiler can cancel the folds."""
    if factor < 1 or factor % 2 == 0:
        raise ValueError("folding factor must be odd: 1, 3, 5, ...")
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        out.append(inst)
        if inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier", "measure"):
            for _ in range((factor - 1) // 2):
                out.append(inst.operation.inverse(), inst.qubits, inst.clbits)
                out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def extrapolate(factors, energies, kind="linear"):
    """value at noise factor 0. linear: least-squares line; richardson: polynomial through all points."""
    factors, energies = np.asarray(factors, float), np.asarray(energies, float)
    deg = 1 if kind == "linear" else len(factors) - 1
    return float(np.polyval(np.polyfit(factors, energies, deg), 0.0))
