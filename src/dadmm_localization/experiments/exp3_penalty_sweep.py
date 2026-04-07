"""Experiment 3: DADMM penalty sweep with per-agent convergence traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dadmm_localization.config import Experiment3Config
from dadmm_localization.dadmm_solver import run_dadmm
from dadmm_localization.metrics import (
    aggregate_penalty_metrics,
    compute_lipschitz_constants,
    position_error_m,
    time_error_s,
)
from dadmm_localization.network import build_network
from dadmm_localization.results import get_logger, setup_run_artifacts, write_config_yaml, write_dataframe, write_manifest
from dadmm_localization.simulation import simulate_toa_measurements, spawn_trial_generators
from dadmm_localization.types import TargetState


def run_experiment3(config: Experiment3Config) -> Path:
    """Run experiment 3 and persist penalty sweep metrics.

    Args:
        config: Validated experiment 3 configuration.

    Returns:
        Path to the created run directory.

    Raises:
        ImportError: If required dataframe dependencies are unavailable.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required to run experiment 3.") from exc

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
    observed_agent_id = config.observed_agent_id

    total_trials = len(config.rho_p_values) * len(config.rho_t_values) * config.num_trials
    rngs = iter(spawn_trial_generators(config.master_seed, total_trials))
    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    logger.info("Starting experiment 3 with %d trial runs.", total_trials)

    for rho_p in config.rho_p_values:
        for rho_t in config.rho_t_values:
            dadmm_config = config.dadmm.model_copy(
                update={"rho_p": rho_p, "rho_t": rho_t, "save_history": True}
            )
            for trial_id in range(config.num_trials):
                rng = next(rngs)
                measurements = simulate_toa_measurements(network, target, config.sigma_s, rng)
                dadmm_result = run_dadmm(network, measurements, dadmm_config)

                for agent_id in range(network.num_agents):
                    summary_rows.append(
                        {
                            "rho_p": rho_p,
                            "rho_t": rho_t,
                            "trial_id": trial_id,
                            "agent_id": agent_id,
                            "is_observed_agent": agent_id == observed_agent_id,
                            "final_x_m": dadmm_result.agent_positions_m[agent_id, 0],
                            "final_y_m": dadmm_result.agent_positions_m[agent_id, 1],
                            "final_t_s": dadmm_result.agent_times_s[agent_id],
                            "position_error_m": position_error_m(
                                dadmm_result.agent_positions_m[agent_id],
                                target.position_m,
                            ),
                            "time_error_s": time_error_s(
                                float(dadmm_result.agent_times_s[agent_id]),
                                target.emission_time_s,
                            ),
                            "iterations_global": dadmm_result.iterations_global,
                            "converged_at_iteration": int(
                                dadmm_result.iterations_per_agent[agent_id]
                            ),
                            "final_primal_residual": dadmm_result.final_primal_residuals[agent_id],
                            "final_dual_residual": dadmm_result.final_dual_residuals[agent_id],
                            "global_converged": dadmm_result.converged,
                        }
                    )

                for record in dadmm_result.history:
                    observed_position = record.agent_positions_m[observed_agent_id]
                    observed_time = float(record.agent_times_s[observed_agent_id])
                    trace_rows.append(
                        {
                            "rho_p": rho_p,
                            "rho_t": rho_t,
                            "trial_id": trial_id,
                            "iteration": record.iteration,
                            "agent_id": observed_agent_id,
                            "x_m": observed_position[0],
                            "y_m": observed_position[1],
                            "t_s": observed_time,
                            "position_error_m": position_error_m(observed_position, target.position_m),
                            "time_error_s": time_error_s(observed_time, target.emission_time_s),
                            "primal_residual": record.primal_residuals[observed_agent_id],
                            "dual_residual": record.dual_residuals[observed_agent_id],
                            "local_cost": record.local_costs[observed_agent_id],
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = aggregate_penalty_metrics(summary_df.loc[summary_df["is_observed_agent"]].copy())
    l_p, l_t = compute_lipschitz_constants(
        num_agents=network.num_agents,
        sigma_s=config.sigma_s,
        sound_speed_mps=network.sound_speed_mps,
        epsilon_radius_m=config.crlb.epsilon_radius_m,
        confidence_k_sigma=config.crlb.confidence_k_sigma,
    )
    aggregate_df["rho_p_over_Lp"] = aggregate_df["rho_p"] / l_p
    aggregate_df["rho_t_over_Lt"] = aggregate_df["rho_t"] / l_t

    write_dataframe(summary_df, artifacts.run_dir / "penalty_summary.parquet")
    write_dataframe(aggregate_df, artifacts.run_dir / "penalty_aggregate.parquet")
    write_dataframe(pd.DataFrame(trace_rows), artifacts.run_dir / "observed_agent_trace.parquet")
    write_manifest(artifacts.run_dir, artifacts.manifest)
    logger.info("Experiment 3 finished. Results written to %s.", artifacts.run_dir)
    return artifacts.run_dir
