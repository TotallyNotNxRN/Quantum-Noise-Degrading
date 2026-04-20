"""Plotting helpers for sweep outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _cell_edges(values: np.ndarray) -> np.ndarray:
    """Convert ordered sample positions into cell-edge coordinates."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Heatmap axes must be one-dimensional and non-empty.")

    if values.size == 1:
        step = 1.0
        return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)

    deltas = np.diff(values)
    if np.any(deltas <= 0.0):
        raise ValueError("Heatmap axes must be strictly increasing.")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * deltas[0]
    edges[-1] = values[-1] + 0.5 * deltas[-1]
    return edges


def plot_fidelity_curve(results):
    """Plot fidelity as a function of the swept noise parameter."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(results["parameters"], results["fidelity"], color="#1f77b4", linewidth=2.2)
    ax.set_xlabel("Noise parameter")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{results['state_name']} fidelity under {results['noise_type']} damping"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_purity_curve(results):
    """Plot purity as a function of the swept noise parameter."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(results["parameters"], results["purity"], color="#d62728", linewidth=2.2)
    ax.set_xlabel("Noise parameter")
    ax.set_ylabel("Purity")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{results['state_name']} purity under {results['noise_type']} damping"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_heatmap(data_matrix):
    """Plot a heatmap with noise strength on x and qubit count on y."""
    matrix = np.asarray(data_matrix["matrix"], dtype=float)
    x_values = np.asarray(data_matrix["x_values"], dtype=float)
    y_values = np.asarray(data_matrix["y_values"], dtype=float)

    if matrix.shape != (len(y_values), len(x_values)):
        raise ValueError("Heatmap matrix shape must match the supplied axes.")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    mesh = ax.pcolormesh(
        _cell_edges(x_values),
        _cell_edges(y_values),
        matrix,
        cmap=data_matrix.get("cmap", "viridis"),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    ax.set_xlabel("Noise parameter")
    ax.set_ylabel("Number of qubits")
    ax.set_yticks(y_values)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_ylim(float(y_values[0]) - 0.5, float(y_values[-1]) + 0.5)
    ax.set_title(data_matrix.get("title", "Noise sensitivity heatmap"))

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(data_matrix.get("value_label", "Fidelity"))

    fig.tight_layout()
    return fig, ax
