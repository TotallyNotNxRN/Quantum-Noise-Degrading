"""State evolution under local noise channels."""

from __future__ import annotations

import numpy as np

from project.core.noise import (
    amplitude_damping_kraus,
    apply_kraus_n_qubits,
    phase_damping_kraus,
)


def evolve_state(
    rho: np.ndarray,
    noise_type: str,
    param: float,
    n_qubits: int,
) -> np.ndarray:
    """Apply the selected noise channel independently to each qubit."""
    if not 1 <= n_qubits <= 4:
        raise ValueError("Only systems with 1 to 4 qubits are supported.")

    dimension = 2**n_qubits
    if rho.shape != (dimension, dimension):
        raise ValueError(
            f"Expected a ({dimension}, {dimension}) density matrix for {n_qubits} qubits."
        )

    if noise_type == "amplitude":
        kraus_ops = amplitude_damping_kraus(param)
    elif noise_type == "phase":
        kraus_ops = phase_damping_kraus(param)
    else:
        raise ValueError("noise_type must be 'amplitude' or 'phase'.")

    evolved = np.array(rho, dtype=complex, copy=True)
    for qubit_index in range(n_qubits):
        evolved = apply_kraus_n_qubits(evolved, kraus_ops, qubit_index, n_qubits)
    return np.real_if_close(evolved)
