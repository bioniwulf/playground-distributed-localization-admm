"""Experiment dispatch entry points."""

from __future__ import annotations

from pathlib import Path

from dadmm_localization.config import (
    Experiment0Config,
    Experiment1Config,
    Experiment2Config,
    Experiment3Config,
    ExperimentConfig,
)
from dadmm_localization.experiments.exp0_dadmm_quick_check import run_experiment0
from dadmm_localization.experiments.exp1_random_convergence import run_experiment1
from dadmm_localization.experiments.exp2_sigma_sweep import run_experiment2
from dadmm_localization.experiments.exp3_penalty_sweep import run_experiment3


def run_experiment(config: ExperimentConfig) -> Path:
    """Dispatch a validated experiment config to its runner.

    Args:
        config: Validated experiment configuration model.

    Returns:
        Path to the created run directory.

    Raises:
        TypeError: If the config type is unsupported.
    """

    if isinstance(config, Experiment0Config):
        return run_experiment0(config)
    if isinstance(config, Experiment1Config):
        return run_experiment1(config)
    if isinstance(config, Experiment2Config):
        return run_experiment2(config)
    if isinstance(config, Experiment3Config):
        return run_experiment3(config)
    raise TypeError(f"Unsupported config type: {type(config)!r}")
