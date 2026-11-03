"""
control tests for the measurement-based energy estimate and error mitigation (src/hardware.py).

run:  uv run pytest -q
"""

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from src import ansatze, hamiltonians as hm
from src import hardware as hw


@pytest.fixture(scope="module")
def h2():
    qp = hm.build(hm.h2(0.735))
    circ = ansatze.uccsd(qp)
    theta = np.random.default_rng(0).uniform(-0.5, 0.5, circ.num_parameters)
    return qp, circ.assign_parameters(theta)


def test_groups_cover_the_hamiltonian(h2):
    qp, _ = h2
    const, groups = hw.measurement_groups(qp.hamiltonian)
    rebuilt = sum((g for _, g in groups), start=hw.SparsePauliOp(["II"], [const]))
    assert np.allclose(rebuilt.simplify().to_matrix(), qp.hamiltonian.to_matrix())


def test_energy_from_exact_probabilities_matches_statevector(h2):
    """with exact outcome probabilities (infinitely many shots) the estimate is exact."""
    qp, bound = h2
    const, groups = hw.measurement_groups(qp.hamiltonian)
    e = const
    for basis, g in groups:
        qc = hw.with_measurement(bound, basis).remove_final_measurements(inplace=False)
        e += hw.group_energy(g, Statevector(qc).probabilities())
    exact = Statevector(bound).expectation_value(qp.hamiltonian).real
    assert e == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("factor", [1, 3, 5])
def test_folding_keeps_the_operation(h2, factor):
    _, bound = h2
    folded = hw.fold_two_qubit_gates(bound.decompose(reps=3), factor)
    assert Operator(folded).equiv(Operator(bound))
    n2 = lambda c: sum(1 for i in c.data if i.operation.num_qubits == 2)
    assert n2(folded) == factor * n2(bound.decompose(reps=3))


def test_readout_inversion_recovers_true_probabilities():
    rng = np.random.default_rng(3)
    a = np.eye(4) * 0.9 + rng.uniform(0, 0.1 / 3, (4, 4)) * (1 - np.eye(4))
    a /= a.sum(axis=0)
    p_true = rng.dirichlet(np.ones(4))
    assert np.allclose(np.linalg.solve(a, a @ p_true), p_true)
    counts = [{format(i, "02b"): int(1e6 * a[i, j]) for i in range(4)} for j in range(4)]
    assert np.allclose(hw.assignment_matrix(counts, 2), a, atol=1e-5)


def test_extrapolation_is_exact_for_linear_noise():
    assert hw.extrapolate([1, 3, 5], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert hw.extrapolate([1, 3, 5], [1 + 1 + 1, 1 + 3 + 9, 1 + 5 + 25], "richardson") == pytest.approx(1.0)
