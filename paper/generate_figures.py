"""Generate PDF figure assets for the LaTeX paper."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "paper" / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.core.states import bell_state, ghz_state, single_qubit_superposition
from project.experiments.run_experiments import (
    _plot_comparison_curve,
    _plot_scaling_results,
)
from project.experiments.sweep import sweep_noise, sweep_scaling
from project.visualization.plots import plot_fidelity_curve, plot_heatmap, plot_purity_curve


def _save(fig, name: str) -> None:
    """Save and close a matplotlib figure as a PDF."""
    path = FIGURES_DIR / name
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all paper figure PDFs from the simulation definitions."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parameters = np.linspace(0.0, 1.0, 51)

    single_results = sweep_noise(
        state_fn=single_qubit_superposition,
        noise_type="amplitude",
        param_range=parameters,
        n_qubits=1,
    )
    fig, _ = plot_fidelity_curve(single_results)
    _save(fig, "single_qubit_amplitude_fidelity.pdf")
    fig, _ = plot_purity_curve(single_results)
    _save(fig, "single_qubit_amplitude_purity.pdf")

    bell_amplitude = sweep_noise(
        state_fn=bell_state,
        noise_type="amplitude",
        param_range=parameters,
        n_qubits=2,
    )
    bell_phase = sweep_noise(
        state_fn=bell_state,
        noise_type="phase",
        param_range=parameters,
        n_qubits=2,
    )
    fig, _ = _plot_comparison_curve(
        bell_amplitude,
        bell_phase,
        key="fidelity",
        ylabel="Fidelity",
        title="Bell-state fidelity under local noise",
    )
    _save(fig, "bell_state_fidelity_comparison.pdf")
    fig, _ = _plot_comparison_curve(
        bell_amplitude,
        bell_phase,
        key="purity",
        ylabel="Purity",
        title="Bell-state purity under local noise",
    )
    _save(fig, "bell_state_purity_comparison.pdf")

    qubit_range = np.arange(2, 5)
    scaling_results = sweep_scaling(
        state_fn=ghz_state,
        noise_type="phase",
        param=0.35,
        qubit_range=qubit_range,
    )
    fig, _ = _plot_scaling_results(scaling_results)
    _save(fig, "ghz_phase_scaling.pdf")

    heatmap_matrix = np.vstack(
        [
            sweep_noise(
                state_fn=ghz_state,
                noise_type="phase",
                param_range=parameters,
                n_qubits=int(n_qubits),
            )["fidelity"]
            for n_qubits in qubit_range
        ]
    )
    fig, _ = plot_heatmap(
        {
            "matrix": heatmap_matrix,
            "x_values": parameters,
            "y_values": qubit_range,
            "title": "GHZ fidelity heatmap under phase damping",
            "value_label": "Fidelity",
        }
    )
    _save(fig, "ghz_phase_fidelity_heatmap.pdf")


if __name__ == "__main__":
    main()
