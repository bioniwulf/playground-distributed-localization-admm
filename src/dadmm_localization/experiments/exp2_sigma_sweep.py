"""Experiment 2: sigma sweep against DADMM, centralized solver, and CRLB."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dadmm_localization.centralized_solver import solve_centralized
from dadmm_localization.config import Experiment2Config
from dadmm_localization.crlb import compute_crlb
from dadmm_localization.dadmm_solver import run_dadmm
from dadmm_localization.metrics import (
    aggregate_sigma_metrics,
    position_error_m,
    time_error_s,
    toa_cost,
    toa_residuals,
)
from dadmm_localization.network import build_network
from dadmm_localization.results import get_logger, setup_run_artifacts, write_config_yaml, write_dataframe, write_manifest
from dadmm_localization.simulation import simulate_toa_measurements, spawn_trial_generators
from dadmm_localization.types import TargetState


def run_experiment2(config: Experiment2Config) -> Path:
    """Run experiment 2 and persist sigma-sweep metrics.

    Args:
        config: Validated experiment 2 configuration.

    Returns:
        Path to the created run directory.

    Raises:
        ImportError: If required dataframe dependencies are unavailable.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required to run experiment 2.") from exc

    config_payload = config.model_dump(mode="json")
    artifacts = setup_run_artifacts(
        experiment_name=config.experiment_type,
        config_payload=config_payload,
        master_seed=config.master_seed,
        output_root=config.output_root,
    )
    write_config_yaml(artifacts.run_dir, config_payload)
    logger = get_logger(artifacts.logger_name)

    network = build_network(
        sensor_positions=config.network.sensor_positions,
        adjacency_matrix=config.network.adjacency_matrix,
        sound_speed_mps=config.physics.sound_speed_mps,
    )
    target = TargetState(
        position_m=np.asarray(config.target_position, dtype=float),
        emission_time_s=config.physics.emission_time_s,
    )
    eps_values = config.dadmm.eps_values or [config.dadmm.eps_feas]

    total_trials = len(config.sigma_values) * config.num_trials
    rngs = iter(spawn_trial_generators(config.master_seed, total_trials))
    run_rows: list[dict[str, object]] = []
    crlb_rows: list[dict[str, object]] = []
    logger.info(
        "Starting experiment 2 with %d sigma values, %d epsilon values, and %d trials each.",
        len(config.sigma_values),
        len(eps_values),
        config.num_trials,
    )

    for sigma_s in config.sigma_values:
        crlb_result = compute_crlb(network, target, sigma_s, config.crlb)
        crlb_rows.append(
            {
                "sigma_s": sigma_s,
                "position_rmse_bound_m": crlb_result.position_rmse_bound_m,
                "time_std_bound_s": crlb_result.time_std_bound_s,
                "fim_condition_number": crlb_result.fim_condition_number,
                "valid": crlb_result.valid,
            }
        )

        for trial_id in range(config.num_trials):
            rng = next(rngs)
            measurements = simulate_toa_measurements(network, target, sigma_s, rng)

            centralized_result = solve_centralized(network, measurements, config.centralized)
            run_rows.append(
                {
                    "sigma_s": sigma_s,
                    "trial_id": trial_id,
                    "algorithm": "centralized",
                    "eps_value": np.nan,
                    "eps_feas": np.nan,
                    "eps_conv": np.nan,
                    "true_x_m": target.position_m[0],
                    "true_y_m": target.position_m[1],
                    "true_t_s": target.emission_time_s,
                    "est_x_m": centralized_result.position_m[0],
                    "est_y_m": centralized_result.position_m[1],
                    "est_t_s": centralized_result.emission_time_s,
                    "position_error_m": position_error_m(centralized_result.position_m, target.position_m),
                    "time_error_s": time_error_s(
                        centralized_result.emission_time_s,
                        target.emission_time_s,
                    ),
                    "residual_l2": float(np.linalg.norm(centralized_result.residual_vector)),
                    "cost": centralized_result.cost,
                    "iterations_global": centralized_result.nfev,
                    "converged": centralized_result.success,
                }
            )

            for eps_value in eps_values:
                dadmm_config = config.dadmm.model_copy(
                    update={
                        "eps_feas": eps_value,
                        "eps_conv": eps_value,
                    }
                )
                dadmm_result = run_dadmm(network, measurements, dadmm_config)
                dadmm_residual_vector = toa_residuals(
                    sensor_positions_m=network.sensor_positions_m,
                    sound_speed_mps=network.sound_speed_mps,
                    arrival_times_s=measurements.arrival_times_s,
                    position_m=dadmm_result.consensus_position_m,
                    emission_time_s=dadmm_result.consensus_time_s,
                )
                run_rows.append(
                    {
                        "sigma_s": sigma_s,
                        "trial_id": trial_id,
                        "algorithm": "dadmm",
                        "eps_value": eps_value,
                        "eps_feas": eps_value,
                        "eps_conv": eps_value,
                        "true_x_m": target.position_m[0],
                        "true_y_m": target.position_m[1],
                        "true_t_s": target.emission_time_s,
                        "est_x_m": dadmm_result.consensus_position_m[0],
                        "est_y_m": dadmm_result.consensus_position_m[1],
                        "est_t_s": dadmm_result.consensus_time_s,
                        "position_error_m": position_error_m(
                            dadmm_result.consensus_position_m,
                            target.position_m,
                        ),
                        "time_error_s": time_error_s(dadmm_result.consensus_time_s, target.emission_time_s),
                        "residual_l2": float(np.linalg.norm(dadmm_residual_vector)),
                        "cost": toa_cost(
                            sensor_positions_m=network.sensor_positions_m,
                            sound_speed_mps=network.sound_speed_mps,
                            arrival_times_s=measurements.arrival_times_s,
                            position_m=dadmm_result.consensus_position_m,
                            emission_time_s=dadmm_result.consensus_time_s,
                        ),
                        "iterations_global": dadmm_result.iterations_global,
                        "converged": dadmm_result.converged,
                    }
                )

    run_df = pd.DataFrame(run_rows)
    aggregate_df = aggregate_sigma_metrics(run_df)
    if "std_cost" in aggregate_df:
        aggregate_df["std_cost"] = aggregate_df["std_cost"].fillna(0.0)

    write_dataframe(run_df, artifacts.run_dir / "run_metrics.parquet")
    write_dataframe(aggregate_df, artifacts.run_dir / "sigma_aggregate.parquet")
    write_dataframe(pd.DataFrame(crlb_rows), artifacts.run_dir / "crlb.parquet")
    write_manifest(artifacts.run_dir, artifacts.manifest)
    logger.info("Experiment 2 finished. Results written to %s.", artifacts.run_dir)
    return artifacts.run_dir
