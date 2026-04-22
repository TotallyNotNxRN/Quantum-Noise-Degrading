# Quantum Noise Sensitivity Mapping Engine

This project is a small, framework-free quantum simulation tool for studying how quantum states degrade under noise.

It models quantum systems with **density matrices**, applies **manual Kraus noise channels**, computes **fidelity** and **purity**, and generates **SVG plots** and **CSV data** for quick analysis.

The code is intentionally limited to **1 to 4 qubits** so the implementation stays readable, physically correct, and easy to extend.

## What This Program Does

The engine starts from an ideal quantum state and then simulates what happens when each qubit is exposed to noise.

Supported states:
- Single-qubit superposition state: `|+> = (|0> + |1>) / sqrt(2)`
- Bell state: `(|00> + |11>) / sqrt(2)`
- GHZ state: `(|00...0> + |11...1>) / sqrt(2)` for `2 <= n <= 4`

Supported noise models:
- Amplitude damping
- Phase damping

Computed metrics:
- Fidelity with respect to the ideal state
- Purity of the noisy density matrix

Produced outputs:
- Fidelity vs noise plots
- Purity vs noise plots
- Heatmaps of fidelity vs noise strength vs number of qubits
- CSV files containing the numerical results

## Why Density Matrices?

This project does **not** use statevectors. Everything is represented as a density matrix.

That matters because density matrices let us model:
- Mixed states
- Decoherence
- Irreversible noise processes

This makes them the right representation for noise studies.

## Mathematical Model

### 1. State Representation

Each quantum state is stored as a density matrix:

`rho = |psi><psi|`

The initial states are built from computational basis vectors using `numpy.kron`.

### 2. Noise Channels

Noise is implemented manually with Kraus operators.

Amplitude damping uses:

```text
E0 = [[1, 0],
      [0, sqrt(1 - gamma)]]

E1 = [[0, sqrt(gamma)],
      [0, 0]]
```

Phase damping uses a 3-operator Kraus representation:

```text
E0 = sqrt(1 - lambda) * I
E1 = sqrt(lambda) * |0><0|
E2 = sqrt(lambda) * |1><1|
```

For a density matrix `rho`, a channel is applied as:

`rho' = sum_k E_k rho E_k^dagger`

For multi-qubit systems, the single-qubit Kraus operators are expanded into the full Hilbert space and applied to each qubit one at a time.

### 3. Metrics

Fidelity is computed against a pure reference state:

`F(rho, rho_ideal) = Tr(rho rho_ideal)`

Purity is computed as:

`P(rho) = Tr(rho^2)`

The code also checks that density matrices remain physical:
- Hermitian
- Trace approximately equal to 1
- Positive semidefinite up to numerical tolerance

## Project Layout

```text
project/
├── core/
│   ├── states.py
│   ├── noise.py
│   ├── metrics.py
│   └── evolve.py
├── experiments/
│   ├── sweep.py
│   └── run_experiments.py
├── visualization/
│   └── plots.py
├── outputs/
│   ├── data/
│   └── plots/
└── main.py
```

### File Guide

- `project/core/states.py`
  Builds the ideal density matrices for the supported states.

- `project/core/noise.py`
  Defines the Kraus operators and applies them to single- and multi-qubit density matrices.

- `project/core/metrics.py`
  Computes fidelity and purity and validates that simulated states remain physical.

- `project/core/evolve.py`
  Applies a selected noise channel independently to every qubit.

- `project/experiments/sweep.py`
  Runs parameter sweeps over noise strength and scaling sweeps over qubit count.

- `project/experiments/run_experiments.py`
  Contains the ready-to-run example experiments and saves results to disk.

- `project/visualization/plots.py`
  Generates the line plots and heatmaps using `matplotlib`.

- `main.py`
  Entry point that runs the full experiment suite.

## Included Experiments

When you run the project, it performs three bundled experiments:

### 1. Single-Qubit Amplitude Damping

Starts from the `|+>` state and sweeps amplitude damping from `0` to `1`.

Outputs:
- Fidelity curve
- Purity curve
- CSV data table

### 2. Bell State Noise Comparison

Starts from the Bell state and compares:
- Amplitude damping
- Phase damping

Outputs:
- Fidelity comparison plot
- Purity comparison plot
- Two CSV sweep files

### 3. GHZ Scaling Study

Studies GHZ states for `2`, `3`, and `4` qubits under phase damping.

Outputs:
- Scaling plot across qubit counts
- Fidelity heatmap across qubit count and noise strength
- CSV data for scaling and heatmap

## Requirements

This project only depends on:
- `numpy`
- `matplotlib`

Install them with:

```bash
pip install numpy matplotlib
```

## How To Run

From the repository root:

```bash
python main.py
```

After the run completes, results are written to:

- `outputs/plots/` for SVG figures
- `outputs/data/` for CSV data

## Output Files

### Plots

The plot folder contains clean SVG files:

- `single_qubit_amplitude_fidelity.svg`
- `single_qubit_amplitude_purity.svg`
- `bell_state_fidelity_comparison.svg`
- `bell_state_purity_comparison.svg`
- `ghz_phase_scaling.svg`
- `ghz_phase_fidelity_heatmap.svg`

### Data

The data folder contains CSV files such as:

- `single_qubit_amplitude_sweep.csv`
- `bell_state_amplitude_sweep.csv`
- `bell_state_phase_sweep.csv`
- `ghz_phase_scaling.csv`
- `ghz_phase_fidelity_heatmap.csv`

## How To Read The Results

### Fidelity

Fidelity measures how close the noisy state is to the original ideal state.

- `1.0` means the noisy state is identical to the ideal state
- Lower values mean more degradation

### Purity

Purity measures how mixed the state has become.

- `1.0` means the state is pure
- Lower values mean the state is more mixed

One important physical detail:

Under **phase damping**, purity usually decreases as decoherence increases.

Under **amplitude damping**, purity does not always decrease monotonically. At strong damping the system can relax toward the pure ground state `|0><0|`, so purity can recover even while fidelity to the original state continues to fall. That is expected behavior.

## For New Users: What Happens Internally?

If you are new to the codebase, the full workflow is:

1. Build an ideal density matrix in `states.py`
2. Choose a noise model in `noise.py`
3. Apply the Kraus operators to each qubit in `evolve.py`
4. Measure fidelity and purity in `metrics.py`
5. Sweep across parameters in `sweep.py`
6. Save plots and CSV data in `run_experiments.py`
7. Launch everything from `main.py`

## Customizing The Program

The easiest places to modify are:

- `project/experiments/run_experiments.py`
  Change parameter ranges, output names, or which experiments run.

- `project/experiments/sweep.py`
  Add new sweep styles or returned metrics.

- `project/core/states.py`
  Add new initial states as density matrices.

- `project/core/noise.py`
  Add new Kraus channels if you want to test other noise models.

## Design Choices

This project intentionally avoids:
- Qiskit
- external quantum frameworks
- unnecessary class hierarchies
- systems larger than 4 qubits

The goal is clarity first:
- easy to inspect
- easy to validate mathematically
- easy to extend for small experiments

## Summary

This is a compact quantum noise simulation engine for exploring how fragile quantum states are under decoherence.

It is best thought of as:
- a learning tool for density-matrix noise simulation
- a lightweight experiment runner for small systems
- a clean starting point for adding more channels, states, or metrics