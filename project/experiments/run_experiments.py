"""Experiment runners for the quantum noise sensitivity mapping engine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.core.states import bell_state, ghz_state, single_qubit_superposition
from project.experiments.sweep import sweep_noise, sweep_scaling
from project.visualization.plots import (
    plot_fidelity_curve,
    plot_heatmap,
    plot_purity_curve,
)

PLOT_FILENAMES = (
    "single_qubit_amplitude_fidelity.svg",
    "single_qubit_amplitude_purity.svg",
    "bell_state_fidelity_comparison.svg",
    "bell_state_purity_comparison.svg",
    "ghz_phase_scaling.svg",
    "ghz_phase_fidelity_heatmap.svg",
)


def _ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create directories for figures and numerical data."""
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, data_dir


def _save_curve_data(results, output_path: Path, x_key: str, x_label: str) -> None:
    """Save sweep data as a CSV table."""
    table = np.column_stack((results[x_key], results["fidelity"], results["purity"]))
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=f"{x_label},fidelity,purity",
        comments="",
    )


def _save_heatmap_data(
    matrix: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    output_path: Path,
) -> None:
    """Save a heatmap matrix with labeled first row and first column."""
    table = np.empty((len(y_values) + 1, len(x_values) + 1), dtype=float)
    table[:] = np.nan
    table[0, 1:] = x_values
    table[1:, 0] = y_values
    table[1:, 1:] = matrix
    np.savetxt(output_path, table, delimiter=",", fmt="%.8f")


def _save_figure(fig, output_path: Path) -> None:
    """Persist a matplotlib figure and close it."""
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def _remove_stale_plot_files(plots_dir: Path, keep_filenames: tuple[str, ...]) -> None:
    """Keep the plots directory limited to the current SVG outputs."""
    keep_set = set(keep_filenames)
    for path in plots_dir.iterdir():
        if path.is_file() and (path.suffix in {".png", ".svg"}) and path.name not in keep_set:
            path.unlink()


def _plot_comparison_curve(
    first_results,
    second_results,
    key: str,
    ylabel: str,
    title: str,
):
    """Create a two-curve comparison plot for Bell-state studies."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        first_results["parameters"],
        first_results[key],
        linewidth=2.2,
        label=f"{first_results['noise_type']} damping",
    )
    ax.plot(
        second_results["parameters"],
        second_results[key],
        linewidth=2.2,
        label=f"{second_results['noise_type']} damping",
    )
    ax.set_xlabel("Noise parameter")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _plot_scaling_results(results):
    """Create a compact scaling summary plot for GHZ studies."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)

    axes[0].plot(results["qubits"], results["fidelity"], marker="o", linewidth=2.0)
    axes[0].set_ylabel("Fidelity")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results["qubits"],
        results["purity"],
        marker="o",
        linewidth=2.0,
        color="#d62728",
    )
    axes[1].set_ylabel("Purity")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)

    for axis in axes:
        axis.set_xlabel("Number of qubits")
        axis.set_xticks(results["qubits"])

    fig.suptitle(
        f"GHZ scaling under {results['noise_type']} damping "
        f"(noise={results['parameter']:.2f})"
    )
    fig.tight_layout()
    return fig, axes


def run_single_qubit_amplitude_experiment(output_dir: Path) -> dict[str, object]:
    """Run a single-qubit amplitude-damping sweep."""
    plots_dir, data_dir = _ensure_output_dirs(output_dir)
    parameters = np.linspace(0.0, 1.0, 51)

    results = sweep_noise(
        state_fn=single_qubit_superposition,
        noise_type="amplitude",
        param_range=parameters,
        n_qubits=1,
    )

    _save_curve_data(
        results,
        data_dir / "single_qubit_amplitude_sweep.csv",
        x_key="parameters",
        x_label="noise_parameter",
    )

    fidelity_fig, _ = plot_fidelity_curve(results)
    purity_fig, _ = plot_purity_curve(results)

    fidelity_path = plots_dir / "single_qubit_amplitude_fidelity.svg"
    purity_path = plots_dir / "single_qubit_amplitude_purity.svg"
    _save_figure(fidelity_fig, fidelity_path)
    _save_figure(purity_fig, purity_path)

    return {
        "results": results,
        "plots": [str(fidelity_path), str(purity_path)],
        "data": [str(data_dir / "single_qubit_amplitude_sweep.csv")],
    }


def run_bell_state_comparison(output_dir: Path) -> dict[str, object]:
    """Compare Bell-state sensitivity under amplitude and phase damping."""
    plots_dir, data_dir = _ensure_output_dirs(output_dir)
    parameters = np.linspace(0.0, 1.0, 51)

    amplitude_results = sweep_noise(
        state_fn=bell_state,
        noise_type="amplitude",
        param_range=parameters,
        n_qubits=2,
    )
    phase_results = sweep_noise(
        state_fn=bell_state,
        noise_type="phase",
        param_range=parameters,
        n_qubits=2,
    )

    _save_curve_data(
        amplitude_results,
        data_dir / "bell_state_amplitude_sweep.csv",
        x_key="parameters",
        x_label="noise_parameter",
    )
    _save_curve_data(
        phase_results,
        data_dir / "bell_state_phase_sweep.csv",
        x_key="parameters",
        x_label="noise_parameter",
    )

    fidelity_fig, _ = _plot_comparison_curve(
        amplitude_results,
        phase_results,
        key="fidelity",
        ylabel="Fidelity",
        title="Bell-state fidelity under local noise",
    )
    purity_fig, _ = _plot_comparison_curve(
        amplitude_results,
        phase_results,
        key="purity",
        ylabel="Purity",
        title="Bell-state purity under local noise",
    )

    fidelity_path = plots_dir / "bell_state_fidelity_comparison.svg"
    purity_path = plots_dir / "bell_state_purity_comparison.svg"
    _save_figure(fidelity_fig, fidelity_path)
    _save_figure(purity_fig, purity_path)

    return {
        "amplitude_results": amplitude_results,
        "phase_results": phase_results,
        "plots": [str(fidelity_path), str(purity_path)],
        "data": [
            str(data_dir / "bell_state_amplitude_sweep.csv"),
            str(data_dir / "bell_state_phase_sweep.csv"),
        ],
    }


def run_ghz_scaling_study(output_dir: Path) -> dict[str, object]:
    """Run GHZ scaling and heatmap experiments under phase damping."""
    plots_dir, data_dir = _ensure_output_dirs(output_dir)
    qubit_range = np.arange(2, 5)
    fixed_parameter = 0.35
    heatmap_parameters = np.linspace(0.0, 1.0, 51)

    scaling_results = sweep_scaling(
        state_fn=ghz_state,
        noise_type="phase",
        param=fixed_parameter,
        qubit_range=qubit_range,
    )

    heatmap_matrix = np.vstack(
        [
            sweep_noise(
                state_fn=ghz_state,
                noise_type="phase",
                param_range=heatmap_parameters,
                n_qubits=int(n_qubits),
            )["fidelity"]
            for n_qubits in qubit_range
        ]
    )

    heatmap_data = {
        "matrix": heatmap_matrix,
        "x_values": heatmap_parameters,
        "y_values": qubit_range,
        "title": "GHZ fidelity heatmap under phase damping",
        "value_label": "Fidelity",
    }

    _save_curve_data(
        scaling_results,
        data_dir / "ghz_phase_scaling.csv",
        x_key="qubits",
        x_label="n_qubits",
    )
    _save_heatmap_data(
        heatmap_matrix,
        heatmap_parameters,
        qubit_range,
        data_dir / "ghz_phase_fidelity_heatmap.csv",
    )

    scaling_fig, _ = _plot_scaling_results(scaling_results)
    heatmap_fig, _ = plot_heatmap(heatmap_data)

    scaling_path = plots_dir / "ghz_phase_scaling.svg"
    heatmap_path = plots_dir / "ghz_phase_fidelity_heatmap.svg"
    _save_figure(scaling_fig, scaling_path)
    _save_figure(heatmap_fig, heatmap_path)

    return {
        "scaling_results": scaling_results,
        "heatmap_data": heatmap_data,
        "plots": [str(scaling_path), str(heatmap_path)],
        "data": [
            str(data_dir / "ghz_phase_scaling.csv"),
            str(data_dir / "ghz_phase_fidelity_heatmap.csv"),
        ],
    }


def run_all_experiments(output_dir: str | Path = "outputs") -> dict[str, dict[str, object]]:
    """Run the full example suite and save all figures and tables."""
    output_path = Path(output_dir)
    plots_dir, _ = _ensure_output_dirs(output_path)
    _remove_stale_plot_files(plots_dir, PLOT_FILENAMES)

    return {
        "single_qubit_amplitude": run_single_qubit_amplitude_experiment(output_path),
        "bell_state_comparison": run_bell_state_comparison(output_path),
        "ghz_scaling_study": run_ghz_scaling_study(output_path),
    }


if __name__ == "__main__":
    run_all_experiments()
