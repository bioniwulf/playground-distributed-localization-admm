"""Experiment 1: random-target DADMM convergence over a region."""

from __future__ import annotations

from pathlib import Path

from dadmm_localization.config import Experiment1Config
from dadmm_localization.dadmm_solver import run_dadmm
from dadmm_localization.metrics import consensus_gap, position_error_m, time_error_s, toa_cost
from dadmm_localization.network import build_network
from dadmm_localization.results import get_logger, setup_run_artifacts, write_config_yaml, write_dataframe, write_manifest
from dadmm_localization.simulation import sample_target_in_box, simulate_toa_measurements, spawn_trial_generators


def run_experiment1(config: Experiment1Config) -> Path:
    """Run experiment 1 and persist summary tables.

    Args:
        config: Validated experiment 1 configuration.

    Returns:
        Path to the created run directory.

    Raises:
        ImportError: If required dataframe dependencies are unavailable.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required to run experiment 1.") from exc

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

    trial_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    rngs = spawn_trial_generators(config.master_seed, config.num_targets)
    logger.info("Starting experiment 1 with %d targets.", config.num_targets)

    for trial_id, rng in enumerate(rngs):
        target = sample_target_in_box(config.region_box, config.physics.emission_time_s, rng)
        measurements = simulate_toa_measurements(network, target, config.sigma_s, rng)
        dadmm_result = run_dadmm(network, measurements, config.dadmm)

        estimate_position = dadmm_result.consensus_position_m
        estimate_time = dadmm_result.consensus_time_s
        final_consensus_gap = consensus_gap(
            dadmm_result.agent_positions_m,
            dadmm_result.agent_times_s,
        )
        trial_rows.append(
            {
                "trial_id": trial_id,
                "seed": config.master_seed,
                "true_x_m": target.position_m[0],
                "true_y_m": target.position_m[1],
                "true_t_s": target.emission_time_s,
                "sigma_s": config.sigma_s,
                "rho_p": config.dadmm.rho_p,
                "rho_t": config.dadmm.rho_t,
                "est_x_m": estimate_position[0],
                "est_y_m": estimate_position[1],
                "est_t_s": estimate_time,
                "position_error_m": position_error_m(estimate_position, target.position_m),
                "time_error_s": time_error_s(estimate_time, target.emission_time_s),
                "global_cost": toa_cost(
                    sensor_positions_m=network.sensor_positions_m,
                    sound_speed_mps=network.sound_speed_mps,
                    arrival_times_s=measurements.arrival_times_s,
                    position_m=estimate_position,
                    emission_time_s=estimate_time,
                ),
                "global_iterations": dadmm_result.iterations_global,
                "converged": dadmm_result.converged,
                "max_consensus_gap": final_consensus_gap,
            }
        )

        for agent_id in range(network.num_agents):
            agent_rows.append(
                {
                    "trial_id": trial_id,
                    "agent_id": agent_id,
                    "converged_at_iteration": int(dadmm_result.iterations_per_agent[agent_id]),
                    "final_x_m": dadmm_result.agent_positions_m[agent_id, 0],
                    "final_y_m": dadmm_result.agent_positions_m[agent_id, 1],
                    "final_t_s": dadmm_result.agent_times_s[agent_id],
                    "final_primal_residual": dadmm_result.final_primal_residuals[agent_id],
                    "final_dual_residual": dadmm_result.final_dual_residuals[agent_id],
                }
            )

    write_dataframe(pd.DataFrame(trial_rows), artifacts.run_dir / "trial_summary.parquet")
    write_dataframe(pd.DataFrame(agent_rows), artifacts.run_dir / "agent_summary.parquet")
    write_manifest(artifacts.run_dir, artifacts.manifest)
    logger.info("Experiment 1 finished. Results written to %s.", artifacts.run_dir)
    return artifacts.run_dir
