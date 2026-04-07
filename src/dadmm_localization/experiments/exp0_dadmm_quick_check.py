"""Experiment 0: single-run DADMM sanity check."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dadmm_localization.config import Experiment0Config
from dadmm_localization.dadmm_solver import run_dadmm
from dadmm_localization.metrics import consensus_gap, position_error_m, time_error_s, toa_cost
from dadmm_localization.network import build_network
from dadmm_localization.results import (
    get_logger,
    setup_run_artifacts,
    write_config_yaml,
    write_dataframe,
    write_manifest,
)
from dadmm_localization.simulation import simulate_toa_measurements, spawn_trial_generators
from dadmm_localization.types import TargetState, WarmStart


def run_experiment0(config: Experiment0Config) -> Path:
    """Run a single fully specified DADMM sanity-check scenario.

    Args:
        config: Validated experiment 0 configuration.

    Returns:
        Path to the created run directory.

    Raises:
        ImportError: If required dataframe dependencies are unavailable.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required to run experiment 0.") from exc

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
    rng = spawn_trial_generators(config.master_seed, 1)[0]
    measurements = simulate_toa_measurements(network, target, config.sigma_s, rng)
    dadmm_config = config.dadmm.model_copy(update={"save_history": True})
    warm_start = None
    if dadmm_config.warm_start is not None:
        warm_start = WarmStart(
            positions_m=np.repeat(
                np.asarray(dadmm_config.warm_start.position_m, dtype=float)[None, :],
                network.num_agents,
                axis=0,
            ),
            emission_times_s=np.full(
                network.num_agents,
                float(dadmm_config.warm_start.emission_time_s),
                dtype=float,
            ),
        )
    result = run_dadmm(network, measurements, dadmm_config, init=warm_start)

    summary_row = {
        "seed": config.master_seed,
        "true_x_m": target.position_m[0],
        "true_y_m": target.position_m[1],
        "true_t_s": target.emission_time_s,
        "sigma_s": config.sigma_s,
        "rho_p": dadmm_config.rho_p,
        "rho_t": dadmm_config.rho_t,
        "eps_feas": dadmm_config.eps_feas,
        "eps_conv": dadmm_config.eps_conv,
        "est_x_m": result.consensus_position_m[0],
        "est_y_m": result.consensus_position_m[1],
        "est_t_s": result.consensus_time_s,
        "position_error_m": position_error_m(result.consensus_position_m, target.position_m),
        "time_error_s": time_error_s(result.consensus_time_s, target.emission_time_s),
        "global_cost": toa_cost(
            sensor_positions_m=network.sensor_positions_m,
            sound_speed_mps=network.sound_speed_mps,
            arrival_times_s=measurements.arrival_times_s,
            position_m=result.consensus_position_m,
            emission_time_s=result.consensus_time_s,
        ),
        "global_iterations": result.iterations_global,
        "converged": result.converged,
        "max_consensus_gap": consensus_gap(result.agent_positions_m, result.agent_times_s),
    }

    measurement_rows = [
        {
            "agent_id": agent_id,
            "sensor_x_m": network.sensor_positions_m[agent_id, 0],
            "sensor_y_m": network.sensor_positions_m[agent_id, 1],
            "arrival_time_s": measurements.arrival_times_s[agent_id],
            "noise_s": measurements.noise_s[agent_id],
        }
        for agent_id in range(network.num_agents)
    ]

    final_agent_rows = [
        {
            "agent_id": agent_id,
            "final_x_m": result.agent_positions_m[agent_id, 0],
            "final_y_m": result.agent_positions_m[agent_id, 1],
            "final_t_s": result.agent_times_s[agent_id],
            "converged_at_iteration": int(result.iterations_per_agent[agent_id]),
            "final_primal_residual": result.final_primal_residuals[agent_id],
            "final_dual_residual": result.final_dual_residuals[agent_id],
            "final_local_cost": result.final_local_costs[agent_id],
        }
        for agent_id in range(network.num_agents)
    ]

    trace_rows: list[dict[str, object]] = []
    for record in result.history:
        for agent_id in range(network.num_agents):
            trace_rows.append(
                {
                    "iteration": record.iteration,
                    "agent_id": agent_id,
                    "x_m": record.agent_positions_m[agent_id, 0],
                    "y_m": record.agent_positions_m[agent_id, 1],
                    "t_s": record.agent_times_s[agent_id],
                    "position_error_m": position_error_m(
                        record.agent_positions_m[agent_id],
                        target.position_m,
                    ),
                    "time_error_s": time_error_s(
                        float(record.agent_times_s[agent_id]),
                        target.emission_time_s,
                    ),
                    "primal_residual": record.primal_residuals[agent_id],
                    "dual_residual": record.dual_residuals[agent_id],
                    "local_cost": record.local_costs[agent_id],
                    "consensus_x_m": record.consensus_position_m[0],
                    "consensus_y_m": record.consensus_position_m[1],
                    "consensus_t_s": record.consensus_time_s,
                    "consensus_gap": record.consensus_gap,
                }
            )

    write_dataframe(pd.DataFrame([summary_row]), artifacts.run_dir / "run_summary.parquet")
    write_dataframe(pd.DataFrame(measurement_rows), artifacts.run_dir / "measurements.parquet")
    write_dataframe(pd.DataFrame(final_agent_rows), artifacts.run_dir / "agent_final.parquet")
    write_dataframe(pd.DataFrame(trace_rows), artifacts.run_dir / "iteration_trace.parquet")
    write_manifest(artifacts.run_dir, artifacts.manifest)
    logger.info("Experiment 0 finished. Results written to %s.", artifacts.run_dir)
    return artifacts.run_dir
