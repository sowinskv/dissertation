"""
control tests for the conditional QNN (src/conditional.py).

run:  uv run pytest -q
"""

import numpy as np
import pytest

from src import ansatze, hamiltonians as hm
from src.conditional import ConditionalQNN, features
from src.train import run_vqe


@pytest.fixture(scope="module")
def h2_family():
    ds = [0.6, 0.9, 1.4]
    qps = [hm.build(hm.h2(d)) for d in ds]
    return ds, qps, [q.matrix() for q in qps], [q.e_shift for q in qps]


def test_features_span_the_training_range():
    """Chebyshev features: x = -1 at lo, +1 at hi, T_0 = 1."""
    phi = features([0.5, 2.5], degree=3, lo=0.5, hi=2.5)
    assert np.allclose(phi[0], [1, -1, 1, -1])
    assert np.allclose(phi[1], [1, 1, 1, 1])


def test_encoder_gradient_matches_finite_differences(h2_family):
    """chain-rule gradient of the mean energy w.r.t. the encoder weights."""
    ds, qps, hams, shifts = h2_family
    model = ConditionalQNN(ansatze.hea(qps[0], 1), degree=2, lo=0.5, hi=1.5)
    w = np.random.default_rng(1).uniform(-1, 1, model.shape)
    seen = {}

    def grab(it, loss, gnorm, evals):
        seen.setdefault("loss", loss)

    model.train(ds, hams, shifts, w, maxiter=0, callback=grab)
    loss = lambda w_: model.energies(w_, ds, hams, shifts).mean()
    assert seen["loss"] == pytest.approx(loss(w), abs=1e-12)

    # analytic chain rule (inside train) vs a direct central difference on W
    h = 1e-5
    num = np.zeros(model.shape)
    for idx in np.ndindex(*model.shape):
        dw = np.zeros(model.shape)
        dw[idx] = h
        num[idx] = (loss(w + dw) - loss(w - dw)) / (2 * h)
    p = model.p
    phi = features(ds, model.degree, model.lo, model.hi)
    g_theta = []
    for k, d in enumerate(ds):
        th = model.angles(w, [d])[0]
        e = lambda t: model.sim.expectation(t[None, :], hams[k])[0]
        g_theta.append([(e(th + h * np.eye(p)[i]) - e(th - h * np.eye(p)[i])) / (2 * h) for i in range(p)])
    chain = (np.array(g_theta).T @ phi) / len(ds)
    assert np.allclose(chain, num, atol=1e-7)


def test_single_geometry_constant_encoder_is_plain_vqe(h2_family):
    """degree 0 and one training geometry: the conditional model reduces to ordinary VQE."""
    ds, qps, hams, shifts = h2_family
    circ = ansatze.uccsd(qps[1])
    model = ConditionalQNN(circ, degree=0, lo=0.5, hi=1.5)
    res = model.train([ds[1]], [hams[1]], [shifts[1]], np.zeros(model.shape))
    ref = run_vqe(circ, hams[1], shifts[1], np.zeros(circ.num_parameters))
    assert res.loss == pytest.approx(ref.energy, abs=1e-9)
    e_fci, _, _ = hm.exact_ground_state(qps[1])
    assert res.loss == pytest.approx(e_fci, abs=1e-8)
