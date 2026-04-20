"""Parameter sweeps for noise sensitivity studies."""

from __future__ import annotations

import inspect

import numpy as np

from project.core.evolve import evolve_state
from project.core.metrics import fidelity, purity


def _build_state(state_fn, n_qubits: int) -> np.ndarray:
    """Create an ideal density matrix from a state factory."""
    signature = inspect.signature(state_fn)
    parameter_count = len(signature.parameters)

    if parameter_count == 0:
        return state_fn()
    if parameter_count == 1:
        return state_fn(n_qubits)
    raise ValueError("State factory must accept either zero arguments or n_qubits.")


def sweep_noise(
    state_fn,
    noise_type: str,
    param_range,
    n_qubits: int,
) -> dict[str, np.ndarray | str | int]:
    """Sweep a noise parameter and record fidelity and purity."""
    parameters = np.asarray(param_range, dtype=float)
    rho_ideal = _build_state(state_fn, n_qubits)

    fidelity_values = np.empty_like(parameters, dtype=float)
    purity_values = np.empty_like(parameters, dtype=float)

    for index, parameter in enumerate(parameters):
        rho_noisy = evolve_state(rho_ideal, noise_type, float(parameter), n_qubits)
        fidelity_values[index] = fidelity(rho_noisy, rho_ideal)
        purity_values[index] = purity(rho_noisy)

    return {
        "state_name": state_fn.__name__,
        "noise_type": noise_type,
        "n_qubits": n_qubits,
        "parameters": parameters,
        "fidelity": fidelity_values,
        "purity": purity_values,
    }


def sweep_scaling(
    state_fn,
    noise_type: str,
    param: float,
    qubit_range,
) -> dict[str, np.ndarray | str | float]:
    """Evaluate sensitivity at fixed noise strength across system size."""
    qubits = np.asarray(list(qubit_range), dtype=int)

    fidelity_values = np.empty_like(qubits, dtype=float)
    purity_values = np.empty_like(qubits, dtype=float)

    for index, n_qubits in enumerate(qubits):
        rho_ideal = _build_state(state_fn, int(n_qubits))
        rho_noisy = evolve_state(rho_ideal, noise_type, float(param), int(n_qubits))
        fidelity_values[index] = fidelity(rho_noisy, rho_ideal)
        purity_values[index] = purity(rho_noisy)

    return {
        "state_name": state_fn.__name__,
        "noise_type": noise_type,
        "parameter": float(param),
        "qubits": qubits,
        "fidelity": fidelity_values,
        "purity": purity_values,
    }
