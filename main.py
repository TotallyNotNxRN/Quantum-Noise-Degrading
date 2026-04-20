"""Entry point for the quantum noise sensitivity mapping engine."""

from __future__ import annotations

from project.experiments.run_experiments import run_all_experiments


def main() -> None:
    """Run the bundled experiment suite and report saved outputs."""
    results = run_all_experiments()

    print("Quantum noise sensitivity experiments completed.")
    for experiment_name, summary in results.items():
        print(f"\n[{experiment_name}]")
        for plot_path in summary["plots"]:
            print(f"plot: {plot_path}")
        for data_path in summary["data"]:
            print(f"data: {data_path}")


if __name__ == "__main__":
    main()
