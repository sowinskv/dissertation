"""
control tests: every building block is checked against an independent reference.

run:  uv run pytest -q
"""

import numpy as np
import pytest
from qiskit.primitives import StatevectorEstimator

from src import ansatze, hamiltonians as hm
from src.classical import reference_energies
from src.fastsim import FastCircuit
from src.train import run_vqe


@pytest.fixture(scope="module", params=[hm.h2(0.735), hm.h_chain(4, 1.0)], ids=["H2", "H4"])
def qp(request):
    return hm.build(request.param)


def test_qubit_hamiltonian_matches_fci(qp):
    """lowest eigenvalue in the right electron-number sector = FCI from PySCF."""
    e_exact, _, _ = hm.exact_ground_state(qp)
    assert e_exact == pytest.approx(reference_energies(qp.atom)["E_FCI_Ha"], abs=1e-8)


def test_uccsd_at_zero_is_hartree_fock(qp):
    circ = ansatze.uccsd(qp)
    e = FastCircuit(circ).expectation(np.zeros(circ.num_parameters), qp.matrix())[0]
    assert e + qp.e_shift == pytest.approx(reference_energies(qp.atom)["E_HF_Ha"], abs=1e-8)


@pytest.mark.parametrize("kind", ["uccsd", "hea2"])
def test_fastsim_matches_qiskit_estimator(qp, kind):
    circ = ansatze.uccsd(qp) if kind == "uccsd" else ansatze.hea(qp, 2)
    rng = np.random.default_rng(0)
    thetas = rng.uniform(-1, 1, (3, circ.num_parameters))
    ours = FastCircuit(circ).expectation(thetas, qp.matrix())
    ref = StatevectorEstimator().run([(circ, qp.hamiltonian, thetas)]).result()[0].data.evs
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_vqe_uccsd_reaches_fci_for_h2():
    qp = hm.build(hm.h2(0.735))
    circ = ansatze.uccsd(qp)
    res = run_vqe(circ, qp.matrix(), qp.e_shift, np.zeros(circ.num_parameters))
    e_exact, _, _ = hm.exact_ground_state(qp)
    assert res.energy == pytest.approx(e_exact, abs=1e-9)
