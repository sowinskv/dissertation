"""
train: VQE as a training loop.

energy(theta) = <psi(theta)|H|psi(theta)> + e_shift is the loss; L-BFGS-B is the optimizer.
the gradient uses central finite differences, evaluated as ONE batch of 2p+1 circuits
(fastsim), so the cost per iteration grows with the number of parameters p.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.fastsim import FastCircuit

FD_STEP = 1e-5   # central differences: truncation error ~1e-10, round-off ~1e-11


@dataclass
class VQEResult:
    energy: float          # total energy (Ha)
    theta: np.ndarray      # optimal parameters
    evaluations: int       # circuit evaluations (state preparations) used in training
    iterations: int        # optimizer iterations
    converged: bool
    state: np.ndarray      # final state vector


def run_vqe(circuit, h_matrix, e_shift, theta0, maxiter=2000, gtol=1e-8):
    sim = circuit if isinstance(circuit, FastCircuit) else FastCircuit(circuit)
    p = sim.num_parameters
    shifts = np.vstack([np.zeros(p), FD_STEP * np.eye(p), -FD_STEP * np.eye(p)])
    count = {"evals": 0}

    def fun_and_grad(theta):
        e = sim.expectation(theta + shifts, h_matrix)
        count["evals"] += len(shifts)
        grad = (e[1:p + 1] - e[p + 1:]) / (2 * FD_STEP)
        return e[0] + e_shift, grad

    res = minimize(fun_and_grad, np.asarray(theta0, dtype=float), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15})
    state = sim.states(res.x)[0]
    return VQEResult(float(res.fun), res.x, count["evals"], int(res.nit),
                     bool(res.success), state)
