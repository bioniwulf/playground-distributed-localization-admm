# AGENTS.md

## Project purpose

This repository is a Python playground for numerical validation of a distributed target-localization algorithm based on DADMM.

The core estimation model is ToA with unknown emission time:

```text
tau_i = t + ||p - s_i|| / v + w_i
```

with:

- 2D only
- Gaussian noise
- configurable `sigma`
- known sound speed `v = 1500 m/s`
- synchronized clocks

## What the playground must support

The repository currently supports four experiment workflows:

0. Single-run DADMM quick check with fully specified target parameters.
1. Random-target convergence over a region using DADMM only.
2. Sigma sweep at a fixed target, comparing DADMM, centralized CasADi/IPOPT, and CRLB.
3. Penalty sweep at a fixed target using DADMM only, with detailed convergence traces for one observed agent.

## Main architectural decisions

These decisions were made intentionally and should be preserved unless there is a strong reason to change them.

### Execution model

- Single-process synchronous simulation.
- No real networking, async transport, multiprocessing, or message-passing framework.
- Distributed behavior is simulated centrally by iterating all agents in lockstep.

### Code organization

The repo uses a `src/` layout:

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
  experiments/
```

Intent of each module:

- `config.py`: typed experiment and solver configs using `pydantic`
- `types.py`: dataclass containers for network, targets, measurements, and solver outputs
- `network.py`: graph validation and neighbor extraction from adjacency matrix
- `simulation.py`: target sampling, RNG spawning, and ToA generation
- `dadmm_solver.py`: synchronous DADMM implementation
- `centralized_solver.py`: centralized ToA baseline solved with CasADi and IPOPT
- `crlb.py`: Fisher Information Matrix and CRLB for `[p_x, p_y, t]`
- `metrics.py`: errors, residuals, costs, consensus metrics, and Monte Carlo aggregations
- `results.py`: run directory creation, manifest/config persistence, parquet serialization
- `experiments/*.py`: experiment-specific orchestration only, without hidden math logic
- `cli.py`: `dadmm-localization run --config <yaml>` and `dadmm-localization visualize --run-dir <run_dir>`

### Config and workflow

- Main workflow is `CLI + YAML`, not notebook-first.
- `notebooks/generate_results.ipynb` is the convenience entry point for running the four ready-to-use `*_eight.yaml` configs and visualizing their outputs through the CLI.
- Example configs live in `configs/`.
- Output layout is:

```text
results/<experiment_name>/<timestamp>_<config_hash>/
```

with:

- `config.yaml`
- `manifest.json`
- `*.parquet`
- `run.log`

### Storage format

- Primary result tables are written as `parquet`.
- `manifest.json` stores metadata such as git commit, Python version, package version, and seed.

### Libraries chosen

Runtime:

- `casadi`
- `numpy`
- `scipy`
- `pandas`
- `pyarrow`
- `PyYAML`
- `pydantic`

Dev:

- `uv`
- `ruff`
- `pytest`
- `pytest-cov`
- `pre-commit`

Optional analysis:

- `jupyterlab`

### Explicit non-goals for this stage

Do not introduce these unless a concrete need appears:

- `networkx`
- `numba`
- distributed systems infrastructure
- async code
- multiprocessing
- internal plotting framework
- speculative extensibility for future 3D/generalized sensor models

## Numerical decisions

### DADMM formulation

The implementation follows the closed-form local updates derived in the paper.

Important implementation decisions:

- Local state per agent is `[p_i_x, p_i_y, t_i]`.
- Internal dense state storage is used:
  - `P[N, 2]`
  - `T[N]`
  - `U_p[N, N, 2]`
  - `U_t[N, N]`
- Initialization follows the paper:
  - `P_i^0 = s_i`
  - `T_i^0 = tau_i`
- Distributed stopping criterion:
  - compute the local primal and dual residuals defined in the paper
  - propagate local violation flags with exact dynamic max consensus over the graph diameter
  - stop when every receiver's diameter-level flag is zero
  - otherwise stop at `max_iterations`

### Safeguard for degenerate position direction

The closed-form position update uses a direction vector based on neighbor consensus.
If that vector is nearly zero, the implementation falls back in this order:

1. previous local radial direction `P_i^k - s_i`
2. receiver-to-network-centroid direction
3. fail with explicit geometry error if still degenerate

This safeguard must remain in place.

### CRLB

CRLB is computed for the ToA model with unknown emission time using state `[p_x, p_y, t]`.

Jacobian row:

```text
[(p_x - s_x)/(v r), (p_y - s_y)/(v r), 1]
```

with:

- `r = max(||p - s||, epsilon_radius_m)`

### Centralized baseline

The centralized comparison uses the same ToA least-squares objective, not a TDoA reformulation.

The centralized baseline is solved with CasADi and IPOPT from a deterministic initial point:

- position start = network centroid
- emission-time start = mean measured arrival time

## Experiment-specific expectations

### Experiment 1

- sample targets uniformly in a rectangular box
- run DADMM only
- save:
  - `trial_summary.parquet`
  - `agent_summary.parquet`

### Experiment 2

- fixed target
- sweep `sigma`
- Monte Carlo at each sigma
- run DADMM and centralized baseline on the same noisy measurements
- compute CRLB once per sigma
- save:
  - `run_metrics.parquet`
  - `sigma_aggregate.parquet`
  - `crlb.parquet`

### Experiment 3

- fixed target
- fixed sigma
- sweep `rho_p` and `rho_t`
- Monte Carlo for each penalty pair
- save full history for one selected agent
- save:
  - `penalty_summary.parquet`
  - `penalty_aggregate.parquet`
  - `observed_agent_trace.parquet`

## Coding conventions

- All code comments must be in English.
- Units are SI: meters and seconds.
- Keep the project moderately scalable, but do not over-engineer.
- Prefer readable numeric code over generic abstractions.
- Reuse existing modules instead of duplicating solver logic inside experiment runners.

## Current implementation status

The repo already contains:

- project scaffolding
- CLI entry point for `run` and `visualize`
- YAML configs for `cube` and `eight` geometries
- DADMM solver
- centralized CasADi/IPOPT solver
- CRLB
- experiment runners
- parquet result writing
- smoke/unit tests
- notebook to run and visualize the main experiment set

Key files:

- `pyproject.toml`
- `README.md`
- `Makefile`
- `.pre-commit-config.yaml`
- `configs/*.yaml`
- `tests/*`

## Environment status

Do not assume the local environment is fully provisioned.

Before relying on runtime behavior, prefer confirming the toolchain and dependencies explicitly:

- install dependencies
- run `pytest`
- run at least one CLI experiment
- if using the notebook workflow, make sure `jupyterlab` and `matplotlib` are available

## First things to do in the next session

If continuing development, the default next steps are:

1. install project dependencies
2. run `pytest`
3. run one smoke experiment from `configs/`
4. inspect numerical behavior of DADMM convergence and tune penalties if necessary

Suggested commands:

```bash
uv sync --all-extras
uv run pytest
uv run dadmm-localization run --config configs/exp1_random_convergence_eight.yaml
uv run dadmm-localization visualize --run-dir results/exp1_random_convergence/<timestamp>_<config_hash>
```
