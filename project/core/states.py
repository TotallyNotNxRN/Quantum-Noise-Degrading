"""Density-matrix state preparation utilities."""

from __future__ import annotations

import numpy as np


def _basis_state(bit: int) -> np.ndarray:
    """Return the computational basis vector |bit> for a single qubit."""
    if bit not in (0, 1):
        raise ValueError("Basis states are defined only for bits 0 and 1.")

    vector = np.zeros(2, dtype=complex)
    vector[bit] = 1.0
    return vector


def _density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Convert a state vector into a density matrix."""
    return np.outer(state_vector, state_vector.conj())


def _tensor_product(states: list[np.ndarray]) -> np.ndarray:
    """Compute the Kronecker product of a sequence of state vectors."""
    product = states[0]
    for state in states[1:]:
        product = np.kron(product, state)
    return product


def single_qubit_superposition() -> np.ndarray:
    """Return the density matrix of the |+> = (|0> + |1>) / sqrt(2) state."""
    state = (_basis_state(0) + _basis_state(1)) / np.sqrt(2.0)
    return _density_matrix(state)


def bell_state() -> np.ndarray:
    """Return the density matrix of the Bell state (|00> + |11>) / sqrt(2)."""
    zero_zero = _tensor_product([_basis_state(0), _basis_state(0)])
    one_one = _tensor_product([_basis_state(1), _basis_state(1)])
    state = (zero_zero + one_one) / np.sqrt(2.0)
    return _density_matrix(state)


def ghz_state(n: int) -> np.ndarray:
    """Return the density matrix of an n-qubit GHZ state for 2 <= n <= 4."""
    if not 2 <= n <= 4:
        raise ValueError("GHZ states are supported only for 2 to 4 qubits.")

    all_zero = _tensor_product([_basis_state(0) for _ in range(n)])
    all_one = _tensor_product([_basis_state(1) for _ in range(n)])
    state = (all_zero + all_one) / np.sqrt(2.0)
    return _density_matrix(state)
