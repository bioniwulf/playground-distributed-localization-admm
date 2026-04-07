setup:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

run-exp0:
	uv run dadmm-localization run --config configs/exp0_dadmm_quick_check_eight.yaml

run-exp1:
	uv run dadmm-localization run --config configs/exp1_random_convergence_eight.yaml

run-exp2:
	uv run dadmm-localization run --config configs/exp2_sigma_sweep_eight.yaml

run-exp3:
	uv run dadmm-localization run --config configs/exp3_penalty_sweep_eight.yaml
