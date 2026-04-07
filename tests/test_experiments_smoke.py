from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dadmm_localization.config import (
    CrlbConfig,
    DadmmConfig,
    Experiment0Config,
    Experiment1Config,
    Experiment2Config,
    Experiment3Config,
    NetworkConfig,
    PhysicsConfig,
)
from dadmm_localization.experiments.exp0_dadmm_quick_check import run_experiment0
from dadmm_localization.experiments.exp1_random_convergence import run_experiment1
from dadmm_localization.experiments.exp2_sigma_sweep import run_experiment2
from dadmm_localization.experiments.exp3_penalty_sweep import run_experiment3


def _network_config() -> NetworkConfig:
    return NetworkConfig(
        sensor_positions=[
            [0.0, 0.0],
            [1000.0, 0.0],
            [1000.0, 1000.0],
            [0.0, 1000.0],
        ],
        adjacency_matrix=[
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
    )


def _physics_config() -> PhysicsConfig:
    return PhysicsConfig(sound_speed_mps=1500.0, emission_time_s=0.0)


def _dadmm_config(save_history: bool = False) -> DadmmConfig:
    return DadmmConfig(
        rho_p=1.0e-7,
        rho_t=5.0,
        eps_feas=1.0e-4,
        eps_conv=1.0e-4,
        max_iterations=250,
        save_history=save_history,
    )


def test_experiment0_smoke(tmp_path: Path) -> None:
    config = Experiment0Config(
        experiment_type="exp0_dadmm_quick_check",
        output_root=str(tmp_path),
        master_seed=9,
        network=_network_config(),
        physics=_physics_config(),
        target_position=[500.0, 500.0],
        sigma_s=1.0e-4,
        dadmm=_dadmm_config(save_history=True),
    )

    run_dir = run_experiment0(config)

    assert (run_dir / "run_summary.parquet").exists()
    assert (run_dir / "measurements.parquet").exists()
    assert (run_dir / "agent_final.parquet").exists()
    assert (run_dir / "iteration_trace.parquet").exists()


def test_experiment0_accepts_configured_warm_start(tmp_path: Path) -> None:
    config = Experiment0Config(
        experiment_type="exp0_dadmm_quick_check",
        output_root=str(tmp_path),
        master_seed=9,
        network=_network_config(),
        physics=_physics_config(),
        target_position=[500.0, 500.0],
        sigma_s=1.0e-4,
        dadmm=DadmmConfig(
            **_dadmm_config(save_history=True).model_dump(exclude={"warm_start"}),
            warm_start={
                "position_m": [500.0, 500.0],
                "emission_time_s": 0.0,
            },
        ),
    )

    run_dir = run_experiment0(config)

    summary_df = pd.read_parquet(run_dir / "run_summary.parquet")
    assert {"est_x_m", "est_y_m", "est_t_s"}.issubset(summary_df.columns)


def test_experiment1_smoke(tmp_path: Path) -> None:
    config = Experiment1Config(
        experiment_type="exp1_random_convergence",
        output_root=str(tmp_path),
        master_seed=10,
        network=_network_config(),
        physics=_physics_config(),
        region_box={"xmin": 100.0, "xmax": 900.0, "ymin": 100.0, "ymax": 900.0},
        num_targets=3,
        sigma_s=1.0e-4,
        dadmm=_dadmm_config(),
    )

    run_dir = run_experiment1(config)

    assert (run_dir / "trial_summary.parquet").exists()
    trial_df = pd.read_parquet(run_dir / "trial_summary.parquet")
    assert {"trial_id", "est_x_m", "est_y_m", "global_iterations"}.issubset(trial_df.columns)


def test_experiment2_smoke(tmp_path: Path) -> None:
    pytest.importorskip("casadi")
    config = Experiment2Config(
        experiment_type="exp2_sigma_sweep",
        output_root=str(tmp_path),
        master_seed=11,
        network=_network_config(),
        physics=_physics_config(),
        target_position=[500.0, 500.0],
        sigma_values=[1.0e-5, 2.0e-5],
        num_trials=2,
        dadmm=DadmmConfig(
            **_dadmm_config().model_dump(exclude={"eps_values"}), eps_values=[1.0e-4, 1.0e-5]
        ),
        centralized={
            "ipopt_max_iter": 300,
            "ipopt_tol": 1.0e-8,
        },
        crlb=CrlbConfig(epsilon_radius_m=1.0, confidence_k_sigma=3.0),
    )

    run_dir = run_experiment2(config)

    assert (run_dir / "run_metrics.parquet").exists()
    assert (run_dir / "sigma_aggregate.parquet").exists()
    assert (run_dir / "crlb.parquet").exists()
    run_df = pd.read_parquet(run_dir / "run_metrics.parquet")
    aggregate_df = pd.read_parquet(run_dir / "sigma_aggregate.parquet")
    dadmm_df = run_df[run_df["algorithm"] == "dadmm"]

    assert set(np.sort(dadmm_df["eps_value"].dropna().unique())) == {1.0e-5, 1.0e-4}
    assert {"eps_value", "eps_feas", "eps_conv"}.issubset(run_df.columns)
    assert "eps_value" in aggregate_df.columns
    assert dadmm_df.shape[0] == len(config.sigma_values) * config.num_trials * 2


def test_experiment3_smoke(tmp_path: Path) -> None:
    config = Experiment3Config(
        experiment_type="exp3_penalty_sweep",
        output_root=str(tmp_path),
        master_seed=12,
        network=_network_config(),
        physics=_physics_config(),
        target_position=[500.0, 500.0],
        sigma_s=1.0e-4,
        rho_p_values=[0.5, 1.0],
        rho_t_values=[5.0, 6.0],
        num_trials=2,
        observed_agent_id=0,
        dadmm=_dadmm_config(save_history=True),
        crlb=CrlbConfig(epsilon_radius_m=1.0, confidence_k_sigma=3.0),
    )

    run_dir = run_experiment3(config)

    assert (run_dir / "penalty_summary.parquet").exists()
    assert (run_dir / "penalty_aggregate.parquet").exists()
    assert (run_dir / "observed_agent_trace.parquet").exists()
