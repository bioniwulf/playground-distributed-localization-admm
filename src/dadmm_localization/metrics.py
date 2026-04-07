"""Metrics and aggregation helpers."""

from __future__ import annotations

from math import sqrt

import numpy as np


def position_error_m(estimate_m: np.ndarray, truth_m: np.ndarray) -> float:
    """Compute Euclidean position error in meters.

    Args:
        estimate_m: Estimated position vector.
        truth_m: Reference position vector.

    Returns:
        Euclidean distance between estimate and reference.
    """

    return float(np.linalg.norm(np.asarray(estimate_m, dtype=float) - np.asarray(truth_m, dtype=float)))


def time_error_s(estimate_s: float, truth_s: float) -> float:
    """Compute absolute emission-time error in seconds.

    Args:
        estimate_s: Estimated emission time.
        truth_s: Reference emission time.

    Returns:
        Absolute error in seconds.
    """

    return float(abs(estimate_s - truth_s))


def toa_residuals(
    sensor_positions_m: np.ndarray,
    sound_speed_mps: float,
    arrival_times_s: np.ndarray,
    position_m: np.ndarray,
    emission_time_s: float,
) -> np.ndarray:
    """Compute the residual vector for the ToA least-squares objective.

    Args:
        sensor_positions_m: Receiver positions with shape `(N, 2)`.
        sound_speed_mps: Propagation speed in meters per second.
        arrival_times_s: Measured arrival times with shape `(N,)`.
        position_m: Candidate target position.
        emission_time_s: Candidate emission time.

    Returns:
        Residual vector for all receivers.
    """

    distances = np.linalg.norm(sensor_positions_m - position_m, axis=1)
    predicted = emission_time_s + distances / sound_speed_mps
    return np.asarray(arrival_times_s, dtype=float) - predicted


def toa_cost(
    sensor_positions_m: np.ndarray,
    sound_speed_mps: float,
    arrival_times_s: np.ndarray,
    position_m: np.ndarray,
    emission_time_s: float,
) -> float:
    """Compute the least-squares objective value for ToA residuals.

    Args:
        sensor_positions_m: Receiver positions with shape `(N, 2)`.
        sound_speed_mps: Propagation speed in meters per second.
        arrival_times_s: Measured arrival times with shape `(N,)`.
        position_m: Candidate target position.
        emission_time_s: Candidate emission time.

    Returns:
        Half squared `L2` norm of the residual vector.
    """

    residual = toa_residuals(
        sensor_positions_m=sensor_positions_m,
        sound_speed_mps=sound_speed_mps,
        arrival_times_s=arrival_times_s,
        position_m=position_m,
        emission_time_s=emission_time_s,
    )
    return float(0.5 * np.dot(residual, residual))


def local_toa_cost(
    sensor_position_m: np.ndarray,
    sound_speed_mps: float,
    arrival_time_s: float,
    position_m: np.ndarray,
    emission_time_s: float,
) -> float:
    """Compute the single-agent contribution to the ToA objective.

    Args:
        sensor_position_m: Receiver position.
        sound_speed_mps: Propagation speed in meters per second.
        arrival_time_s: Measured arrival time for the receiver.
        position_m: Candidate target position.
        emission_time_s: Candidate emission time.

    Returns:
        Local least-squares cost contribution.
    """

    distance = np.linalg.norm(np.asarray(position_m, dtype=float) - np.asarray(sensor_position_m, dtype=float))
    residual = float(arrival_time_s - emission_time_s - distance / sound_speed_mps)
    return 0.5 * residual * residual


def consensus_gap(agent_positions_m: np.ndarray, agent_times_s: np.ndarray) -> float:
    """Compute the maximum deviation from the mean consensus state.

    Args:
        agent_positions_m: Per-agent position estimates with shape `(N, 2)`.
        agent_times_s: Per-agent time estimates with shape `(N,)`.

    Returns:
        Maximum Euclidean deviation in the stacked consensus state.
    """

    mean_position = np.mean(agent_positions_m, axis=0)
    mean_time = float(np.mean(agent_times_s))
    stacked = np.concatenate(
        [agent_positions_m - mean_position, (agent_times_s - mean_time)[:, None]],
        axis=1,
    )
    return float(np.max(np.linalg.norm(stacked, axis=1)))


def compute_lipschitz_constants(
    num_agents: int,
    sigma_s: float,
    sound_speed_mps: float,
    epsilon_radius_m: float,
    confidence_k_sigma: float,
) -> tuple[float, float]:
    """Compute global Lipschitz constants from the paper bounds.

    Args:
        num_agents: Number of receivers.
        sigma_s: Measurement noise standard deviation in seconds.
        sound_speed_mps: Propagation speed in meters per second.
        epsilon_radius_m: Minimum radius used in the bound.
        confidence_k_sigma: Confidence multiplier used in the bound.

    Returns:
        Tuple `(L_p, L_t)` of position and time Lipschitz constants.
    """

    l_p = num_agents * (
        (1.0 / (sound_speed_mps**2))
        + (confidence_k_sigma * sigma_s) / (sound_speed_mps * epsilon_radius_m)
    )
    l_t = float(num_agents)
    return float(l_p), l_t


def aggregate_sigma_metrics(dataframe):  # type: ignore[no-untyped-def]
    """Aggregate experiment 2 metrics over sigma and algorithm settings.

    Args:
        dataframe: Run-level experiment 2 dataframe.

    Returns:
        Aggregated dataframe with RMSE, cost, iteration, and success metrics.
    """

    group_columns = ["sigma_s", "algorithm"]
    if "eps_value" in dataframe.columns:
        group_columns.append("eps_value")

    grouped = dataframe.groupby(group_columns, as_index=False, dropna=False).agg(
        mean_sq_position_error=("position_error_m", lambda series: float(np.mean(np.square(series)))),
        mean_sq_time_error=("time_error_s", lambda series: float(np.mean(np.square(series)))),
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std"),
        mean_iterations=("iterations_global", "mean"),
        success_rate=("converged", "mean"),
    )
    grouped["rmse_position_m"] = grouped["mean_sq_position_error"].apply(lambda value: sqrt(value))
    grouped["rmse_time_s"] = grouped["mean_sq_time_error"].apply(lambda value: sqrt(value))
    return grouped.drop(columns=["mean_sq_position_error", "mean_sq_time_error"])


def aggregate_penalty_metrics(dataframe):  # type: ignore[no-untyped-def]
    """Aggregate experiment 3 metrics over penalty pairs.

    Args:
        dataframe: Summary dataframe for experiment 3.

    Returns:
        Aggregated dataframe indexed by `(rho_p, rho_t)`.
    """

    grouped = dataframe.groupby(["rho_p", "rho_t"], as_index=False).agg(
        mean_sq_position_error=("position_error_m", lambda series: float(np.mean(np.square(series)))),
        mean_iterations=("iterations_global", "mean"),
        success_rate=("global_converged", "mean"),
        mean_final_primal_residual=("final_primal_residual", "mean"),
        mean_final_dual_residual=("final_dual_residual", "mean"),
    )
    grouped["rmse_position_m"] = grouped["mean_sq_position_error"].apply(lambda value: sqrt(value))
    return grouped.drop(columns=["mean_sq_position_error"])
