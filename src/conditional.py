"""
conditional: one QNN for a whole family of geometries (Meta-VQE style, RQ2).

plain VQE trains one set of angles theta for one geometry. here a small classical encoder
turns the bond length d into the angles,

    theta(d) = W @ phi(d),     phi(d) = [T_0(x), T_1(x), ..., T_K(x)],   x = 2 (d - lo) / (hi - lo) - 1

(T_k = Chebyshev polynomials, x maps the training range [lo, hi] to [-1, 1]; K = 1 is the
linear encoder of the original Meta-VQE). the encoder weights W are the trainable parameters.

training uses no energy labels: the loss is the mean VQE energy over M training geometries,

    L(W) = 1/M sum_m <psi(theta(d_m)) | H(d_m) | psi(theta(d_m))>,

and each term is minimised by the exact ground state, so the encoder learns to output good
angles. after training, any new d costs ONE circuit evaluation, no optimisation.

gradient: chain rule. dE_m/dtheta comes from central finite differences (like src.train),
for all M geometries in one batch; dL/dW = 1/M sum_m dE_m/dtheta (x) phi(d_m).
a gradient therefore costs M (2p + 1) circuit evaluations, p = number of circuit angles.

reference: A. Cervera-Lierta, J. S. Kottmann, A. Aspuru-Guzik, PRX Quantum 2, 020329 (2021).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.fastsim import FastCircuit
from src.train import FD_STEP


def features(d, degree, lo, hi):
    """Chebyshev features of the bond length: (n,) -> (n, degree + 1)."""
    x = 2 * (np.atleast_1d(np.asarray(d, dtype=float)) - lo) / (hi - lo) - 1
    return np.polynomial.chebyshev.chebvander(x, degree)


@dataclass
class ConditionalResult:
    weights: np.ndarray    # encoder weights W, shape (p, degree + 1)
    loss: float            # mean energy over the training geometries (Ha)
    evaluations: int       # circuit evaluations used in training
    iterations: int
    converged: bool


class ConditionalQNN:
    """a circuit whose angles come from an encoder of the bond length."""

    def __init__(self, circuit, degree, lo, hi):
        self.sim = circuit if isinstance(circuit, FastCircuit) else FastCircuit(circuit)
        self.p = self.sim.num_parameters
        self.degree, self.lo, self.hi = degree, lo, hi
        self.shape = (self.p, degree + 1)

    def angles(self, weights, ds):
        """circuit angles for each bond length: (n, p)."""
        return features(ds, self.degree, self.lo, self.hi) @ np.reshape(weights, self.shape).T

    def states(self, weights, ds):
        return self.sim.states(self.angles(weights, ds))

    def energies(self, weights, ds, hams, e_shifts):
        """energy for each geometry; hams[i] is the qubit hamiltonian matrix at ds[i]."""
        psi = self.states(weights, ds)
        return np.array([np.real(np.vdot(v, h @ v)) for v, h in zip(psi, hams)]) + np.asarray(e_shifts)

    def train(self, ds, hams, e_shifts, w0, maxiter=5000, gtol=1e-7, callback=None):
        """minimise the mean energy over the training geometries ds (no energy labels).
        callback(iteration, mean_energy, grad_norm, evaluations), as in src.train.run_vqe."""
        ds = np.asarray(ds, dtype=float)
        m, p = len(ds), self.p
        phi = features(ds, self.degree, self.lo, self.hi)                     # (m, K+1)
        shifts = np.vstack([np.zeros(p), FD_STEP * np.eye(p), -FD_STEP * np.eye(p)])
        e_shift_mean = float(np.mean(e_shifts))
        count = {"evals": 0, "it": 0}
        last = {}

        def fun_and_grad(w):
            theta = phi @ w.reshape(self.shape).T                              # (m, p)
            batch = (theta[:, None, :] + shifts[None, :, :]).reshape(-1, p)    # m (2p+1) circuits
            psi = self.sim.states(batch).reshape(m, len(shifts), -1)
            e = np.array([np.real(np.einsum("bi,bi->b", psi[k].conj(), (hams[k] @ psi[k].T).T))
                          for k in range(m)])                                  # (m, 2p+1)
            count["evals"] += batch.shape[0]
            g_theta = (e[:, 1:p + 1] - e[:, p + 1:]) / (2 * FD_STEP)           # (m, p)
            grad = (g_theta.T @ phi) / m                                       # (p, K+1)
            loss = float(e[:, 0].mean()) + e_shift_mean
            last.update(loss=loss, grad_norm=float(np.linalg.norm(grad)))
            return loss, grad.ravel()

        def on_iteration(intermediate_result):
            count["it"] += 1
            callback(count["it"], float(intermediate_result.fun), last["grad_norm"], count["evals"])

        w0 = np.asarray(w0, dtype=float).ravel()
        if callback is not None:
            fun_and_grad(w0)
            count["evals"] = 0          # the starting point is not part of the training cost
            callback(0, last["loss"], last["grad_norm"], 0)
        res = minimize(fun_and_grad, w0, jac=True, method="L-BFGS-B",
                       callback=on_iteration if callback is not None else None,
                       options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15})
        return ConditionalResult(res.x.reshape(self.shape), float(res.fun), count["evals"],
                                 int(res.nit), bool(res.success))
