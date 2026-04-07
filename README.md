# DADMM Localization Playground

This repository is a Python playground for numerical validation of a distributed passive target-localization algorithm based on Time of signal Arrival (ToA) with unknown time of emission via on  Distributed Alternating Direction Method of Multipliers (DADMM) approach.

The implemented measurement model is 2D ToA with unknown emission time:

```math
\tau_i = t + \frac{\|p - s_i\|}{v} + w_i
```

where:

- `p = [p_x, p_y]` is the target position
- `t` is the unknown emission time
- `s_i` is the receiver position
- `v = 1500 m/s`
- `w_i` is zero-mean Gaussian noise

The code is organized around reproducible experiment configs, CLI execution, parquet outputs, and figure generation.

## What is Implemented?

The repository supports four experiment workflows:

0. `exp0_dadmm_quick_check`: one fully specified DADMM run for a quick sanity check
1. `exp1_random_convergence`: random-target convergence study over a region using DADMM
2. `exp2_sigma_sweep`: sigma sweep comparing DADMM, centralized CasADi/IPOPT, and CRLB
3. `exp3_penalty_sweep`: penalty sweep for DADMM with observed-agent convergence traces

The main ready-to-run configs used by the notebook workflow are:

- `configs/exp0_dadmm_quick_check_eight.yaml`
- `configs/exp1_random_convergence_eight.yaml`
- `configs/exp2_sigma_sweep_eight.yaml`
- `configs/exp3_penalty_sweep_eight.yaml`

## Install

### Requirements

- Python `3.12` or newer
- One of:
  - `uv` for environment management and command execution
  - standard `venv` + `pip`

The project depends on:

- runtime: `casadi`, `numpy`, `scipy`, `pandas`, `pyarrow`, `PyYAML`, `pydantic`
- development: `pytest`, `pytest-cov`, `ruff`, `pre-commit`
- notebook/analysis: `jupyterlab`, `matplotlib`

### Recommended: install with `uv`

From the repository root:

```bash
uv sync --all-extras
```

This creates a local environment with runtime, development, and notebook dependencies.

Typical commands then run through `uv run`:

```bash
uv run pytest
uv run dadmm-localization --help
uv run jupyter lab
```

### Alternative: install with `venv` and `pip`

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev,analysis]
```

After that, the CLI entry point is available directly:

```bash
dadmm-localization --help
```

If your shell does not expose console scripts from the virtual environment as expected, you can always fall back to:

```bash
python -m dadmm_localization.cli --help
```

## Quick Start

The fastest way to reproduce the full workflow is through the notebook:

- it runs all four `*_eight.yaml` experiments in sequence
- it calls the project CLI for each run
- it visualizes each run immediately
- it renders the generated PNG figures inline in Jupyter

### Launch the notebook

If you installed with `uv`:

```bash
uv run jupyter lab notebooks/generate_results.ipynb
```

If you installed with `venv`:

```bash
source .venv/bin/activate
jupyter lab notebooks/generate_results.ipynb
```

Then open [generate_results.ipynb](notebooks/generate_results.ipynb) and run all cells.

The notebook executes the equivalent CLI commands:

```bash
dadmm-localization run --config configs/exp0_dadmm_quick_check_eight.yaml
dadmm-localization visualize --run-dir <run_dir>

dadmm-localization run --config configs/exp1_random_convergence_eight.yaml
dadmm-localization visualize --run-dir <run_dir>

dadmm-localization run --config configs/exp2_sigma_sweep_eight.yaml
dadmm-localization visualize --run-dir <run_dir>

dadmm-localization run --config configs/exp3_penalty_sweep_eight.yaml
dadmm-localization visualize --run-dir <run_dir>
```

## CLI Usage

### Run an Experiment

```bash
dadmm-localization run --config configs/exp0_dadmm_quick_check_eight.yaml
dadmm-localization run --config configs/exp1_random_convergence_eight.yaml
dadmm-localization run --config configs/exp2_sigma_sweep_eight.yaml
dadmm-localization run --config configs/exp3_penalty_sweep_eight.yaml
```

Each command prints the created `run_dir`.

### Visualize an Existing Run

```bash
dadmm-localization visualize --run-dir results/exp0_dadmm_quick_check/<timestamp>_<hash>
dadmm-localization visualize --run-dir results/exp1_random_convergence/<timestamp>_<hash>
dadmm-localization visualize --run-dir results/exp2_sigma_sweep/<timestamp>_<hash>
dadmm-localization visualize --run-dir results/exp3_penalty_sweep/<timestamp>_<hash>
```

The visualizer writes PNG files into the same `run_dir`.

## Result Layout

Each run is stored under:

```text
results/<experiment_name>/<timestamp>_<config_hash>/
```

A run directory contains:

- `config.yaml`: resolved config used for that run
- `manifest.json`: metadata such as timestamp, git commit, Python version, package version, and seed
- `run.log`: run-scoped log file
- `*.parquet`: experiment outputs
- `*.png`: generated figures after visualization

## Configuring your own experiments

All experiments are configured through YAML files in `configs/`.

The key config models live in [config.py](src/dadmm_localization/config.py):

- `NetworkConfig`: receiver positions and adjacency matrix
- `PhysicsConfig`: sound speed and reference emission time
- `DadmmConfig`: DADMM penalties, tolerances, iteration limit, and optional warm start
- `CentralizedConfig`: CasADi/IPOPT settings for experiment 2
- `CrlbConfig`: CRLB parameters

To create a new experiment:

1. Copy one of the existing YAML files in `configs/`.
2. Edit geometry, target settings, noise levels, penalties, or sweep ranges.
3. Run it through the CLI:

```bash
dadmm-localization run --config path/to/your_config.yaml
dadmm-localization visualize --run-dir path/to/generated_run_dir
```

## Project structure

```text
src/dadmm_localization/
  constants.py
  config.py
  types.py
  network.py
  simulation.py
  metrics.py
  results.py
  dadmm_solver.py
  centralized_solver.py
  crlb.py
  cli.py
  visualization.py
  experiments/
```

Module intent:

- `config.py`: typed configuration models and YAML loading
- `types.py`: dataclass containers for networks, measurements, and solver outputs
- `network.py`: graph validation and neighbor extraction
- `simulation.py`: target sampling and ToA simulation
- `dadmm_solver.py`: synchronous DADMM implementation
- `centralized_solver.py`: centralized CasADi/IPOPT baseline
- `crlb.py`: Fisher information matrix and CRLB computation
- `metrics.py`: error metrics and aggregations
- `results.py`: run directories, manifests, logging, parquet writing
- `visualization.py`: result-to-figure conversion
- `experiments/`: orchestration for each experiment family
- `cli.py`: `dadmm-localization run` and `dadmm-localization visualize`

## Notes for extending the code

- The execution model is single-process and synchronous by design.
- Distributed behavior is simulated centrally; there is no networking or multiprocessing layer.
- The current implementation is intentionally 2D-only.

## Centralized baseline

Experiment 2 uses a centralized ToA baseline solved with CasADi and IPOPT.

Config block:

```yaml
centralized:
  ipopt_max_iter: 500
  ipopt_tol: 1.0e-8
  ipopt_print_level: 0
```

The centralized solver uses a deterministic initial point:

- position start = network centroid
- emission-time start = mean measured arrival time
