"""Manual Kraus-channel implementations for local quantum noise."""

from __future__ import annotations

import numpy as np


def _validate_probability(value: float, name: str) -> None:
    """Ensure a noise parameter lies in the physical range [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _stabilize_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Remove tiny numerical asymmetries and renormalize trace."""
    stabilized = 0.5 * (rho + rho.conj().T)
    trace_value = np.trace(stabilized)
    if np.isclose(trace_value, 0.0):
        raise ValueError("Density matrix has zero trace after evolution.")
    stabilized = stabilized / trace_value
    return np.real_if_close(stabilized)


def _expand_single_qubit_operator(
    operator: np.ndarray,
    qubit_index: int,
    n_qubits: int,
) -> np.ndarray:
    """Embed a single-qubit operator into an n-qubit Hilbert space.

    Qubit index 0 refers to the left-most factor in the tensor product.
    """
    if not 0 <= qubit_index < n_qubits:
        raise ValueError("qubit_index must refer to a valid qubit.")

    identity = np.eye(2, dtype=complex)
    factors: list[np.ndarray] = []
    for index in range(n_qubits):
        factors.append(operator if index == qubit_index else identity)

    expanded = factors[0]
    for factor in factors[1:]:
        expanded = np.kron(expanded, factor)
    return expanded


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Return the Kraus operators for amplitude damping."""
    _validate_probability(gamma, "gamma")

    return [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
        np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    ]


def phase_damping_kraus(lam: float) -> list[np.ndarray]:
    """Return a three-operator Kraus representation of phase damping."""
    _validate_probability(lam, "lam")

    return [
        np.sqrt(1.0 - lam) * np.eye(2, dtype=complex),
        np.sqrt(lam) * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.sqrt(lam) * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    ]


def apply_kraus_single_qubit(
    rho: np.ndarray,
    kraus_ops: list[np.ndarray],
) -> np.ndarray:
    """Apply a single-qubit Kraus channel to a 1-qubit density matrix."""
    if rho.shape != (2, 2):
        raise ValueError("Single-qubit density matrices must have shape (2, 2).")

    evolved = np.zeros_like(rho, dtype=complex)
    for operator in kraus_ops:
        evolved += operator @ rho @ operator.conj().T
    return _stabilize_density_matrix(evolved)


def apply_kraus_n_qubits(
    rho: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubit_index: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply a local Kraus channel to one qubit of an n-qubit density matrix."""
    dimension = 2**n_qubits
    if rho.shape != (dimension, dimension):
        raise ValueError(
            f"Expected a ({dimension}, {dimension}) density matrix for {n_qubits} qubits."
        )

    if n_qubits == 1:
        return apply_kraus_single_qubit(rho, kraus_ops)

    evolved = np.zeros_like(rho, dtype=complex)
    for operator in kraus_ops:
        expanded_operator = _expand_single_qubit_operator(
            operator,
            qubit_index=qubit_index,
            n_qubits=n_qubits,
        )
        evolved += expanded_operator @ rho @ expanded_operator.conj().T
    return _stabilize_density_matrix(evolved)