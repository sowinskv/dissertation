"""
fastsim: exact, batched statevector simulation of a parameterized qiskit circuit in numpy.

why: training a VQE needs thousands of energy evaluations. qiskit's StatevectorEstimator
re-simulates the circuit from scratch for every call (~0.5 s for UCCSD on 6 qubits).
for the small registers in this thesis (<= ~12 qubits) a direct numpy simulation that
evaluates a whole batch of parameter vectors at once is ~10-100x faster.

it is the same math (no approximation): the circuit is transpiled to a few standard gates
and applied gate by gate to the state vector. tests/test_core.py checks it against qiskit's
StatevectorEstimator to 1e-10, and every experiment re-checks its final energies with qiskit.

conventions follow qiskit: qubit k is bit k of the basis-state index (little endian),
parameters are ordered as circuit.parameters.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterExpression

_BASIS = ["rx", "ry", "rz", "cx", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "id"]

_FIXED = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    "sxdg": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex),
}


def _linear_angle(expr, params):
    """write a gate angle as  coeff * theta[index] + const  (or a constant)."""
    if not isinstance(expr, ParameterExpression):
        return None, 0.0, float(expr)
    free = list(expr.parameters)
    if not free:
        return None, 0.0, float(expr)
    if len(free) != 1:
        raise ValueError(f"angle depends on several parameters: {expr}")
    p = free[0]
    f0, f1, f2 = (float(expr.bind({p: v})) for v in (0.0, 1.0, 2.0))
    if abs((f2 - f1) - (f1 - f0)) > 1e-12:
        raise ValueError(f"angle is not linear in its parameter: {expr}")
    return params.index(p), f1 - f0, f0


class FastCircuit:
    """compile a qiskit circuit once, then evaluate states for batches of parameter vectors."""

    def __init__(self, circuit: QuantumCircuit):
        self.params = list(circuit.parameters)
        self.num_parameters = len(self.params)
        self.n = circuit.num_qubits
        self.dim = 2 ** self.n
        tc = transpile(circuit, basis_gates=_BASIS, optimization_level=0)
        idx = np.arange(self.dim)
        self._ops = []
        for inst in tc.data:
            name = inst.operation.name
            qubits = [tc.find_bit(q).index for q in inst.qubits]
            if name in ("barrier", "id"):
                continue
            if name == "cx":
                c, t = qubits
                perm = np.where((idx >> c) & 1, idx ^ (1 << t), idx)
                self._ops.append(("perm", perm))
                continue
            k = qubits[0]
            i0 = idx[((idx >> k) & 1) == 0]
            i1 = i0 | (1 << k)
            if name in _FIXED:
                self._ops.append(("fixed", i0, i1, _FIXED[name]))
            elif name in ("rx", "ry", "rz"):
                pi, coef, const = _linear_angle(inst.operation.params[0], self.params)
                self._ops.append((name, i0, i1, pi, coef, const))
            else:
                raise ValueError(f"unsupported gate: {name}")

    def states(self, thetas):
        """state vectors for a batch of parameter vectors: thetas (B, p) -> (B, dim)."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        b = thetas.shape[0]
        psi = np.zeros((b, self.dim), dtype=complex)
        psi[:, 0] = 1.0
        for op in self._ops:
            kind = op[0]
            if kind == "perm":
                psi = psi[:, op[1]]
                continue
            i0, i1 = op[1], op[2]
            a, c = psi[:, i0], psi[:, i1]
            if kind == "fixed":
                m = op[3]
                psi[:, i0], psi[:, i1] = m[0, 0] * a + m[0, 1] * c, m[1, 0] * a + m[1, 1] * c
                continue
            pi, coef, const = op[3], op[4], op[5]
            ang = (coef * thetas[:, pi] + const if pi is not None
                   else np.full(b, const))[:, None] / 2
            cos, sin = np.cos(ang), np.sin(ang)
            if kind == "ry":
                psi[:, i0], psi[:, i1] = cos * a - sin * c, sin * a + cos * c
            elif kind == "rx":
                psi[:, i0], psi[:, i1] = cos * a - 1j * sin * c, -1j * sin * a + cos * c
            else:  # rz
                psi[:, i0], psi[:, i1] = (cos - 1j * sin) * a, (cos + 1j * sin) * c
        return psi

    def expectation(self, thetas, matrix):
        """<psi|M|psi> for each parameter vector in the batch; matrix is (dim, dim), dense or sparse."""
        psi = self.states(thetas)
        return np.real(np.einsum("bi,bi->b", psi.conj(), (matrix @ psi.T).T))
