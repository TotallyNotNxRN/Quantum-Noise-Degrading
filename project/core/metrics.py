"""State-quality metrics and density-matrix validation."""

from __future__ import annotations

import numpy as np


def validate_density_matrix(rho: np.ndarray, atol: float = 1e-8) -> None:
    """Validate that a matrix is a physical density matrix up to tolerance."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrices must be square matrices.")

    if not np.allclose(rho, rho.conj().T, atol=atol):
        raise ValueError("Density matrix must be Hermitian.")

    if not np.isclose(np.trace(rho), 1.0, atol=atol):
        raise ValueError("Density matrix trace must be approximately 1.")

    eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    if np.min(eigenvalues) < -atol:
        raise ValueError("Density matrix must be positive semidefinite.")


def fidelity(rho: np.ndarray, rho_ideal: np.ndarray) -> float:
    """Compute fidelity against a pure reference density matrix."""
    validate_density_matrix(rho)
    validate_density_matrix(rho_ideal)

    value = float(np.real(np.trace(rho @ rho_ideal)))
    return float(np.clip(value, 0.0, 1.0))


def purity(rho: np.ndarray) -> float:
    """Compute the purity Tr(rho^2) of a density matrix."""
    validate_density_matrix(rho)

    value = float(np.real(np.trace(rho @ rho)))
    return float(np.clip(value, 0.0, 1.0))