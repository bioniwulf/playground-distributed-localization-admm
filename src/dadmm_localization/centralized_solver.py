"""Centralized ToA solver using CasADi and IPOPT."""

from __future__ import annotations

import numpy as np

from dadmm_localization.config import CentralizedConfig
from dadmm_localization.types import CentralizedResult, MeasurementSet, Network


def _toa_residual_vector(
    x: np.ndarray,
    sensor_positions_m: np.ndarray,
    sound_speed_mps: float,
    arrival_times_s: np.ndarray,
) -> np.ndarray:
    """Compute the ToA residual vector for a candidate state.

    Args:
        x: Candidate state vector `[p_x, p_y, t]`.
        sensor_positions_m: Receiver positions with shape `(N, 2)`.
        sound_speed_mps: Propagation speed in meters per second.
        arrival_times_s: Measured arrival times with shape `(N,)`.

    Returns:
        Residual vector `tau_i - t - ||p - s_i|| / v` with shape `(N,)`.
    """

    position_m = x[:2]
    emission_time_s = float(x[2])
    distances = np.linalg.norm(sensor_positions_m - position_m, axis=1)
    return arrival_times_s - emission_time_s - distances / sound_speed_mps


def _mean_arrival_start(network: Network, measurements: MeasurementSet) -> np.ndarray:
    """Build the deterministic initial guess used by the centralized solver.

    Args:
        network: Validated sensor network description.
        measurements: ToA measurements for the current target realization.

    Returns:
        Initial state vector `[centroid_x, centroid_y, mean_arrival_time]`.
    """

    return np.array(
        [
            network.centroid_m[0],
            network.centroid_m[1],
            float(np.mean(measurements.arrival_times_s)),
        ],
        dtype=float,
    )


def solve_centralized(
    network: Network,
    measurements: MeasurementSet,
    config: CentralizedConfig,
) -> CentralizedResult:
    """Solve the centralized ToA localization problem with CasADi and IPOPT.

    Args:
        network: Validated sensor network and physical constants.
        measurements: ToA measurements for all receivers.
        config: IPOPT solver settings for the centralized baseline.

    Returns:
        Centralized solver output containing the estimated state, residuals,
        objective value, and IPOPT convergence metadata.

    Raises:
        ValueError: If the measurement vector shape does not match the network size.
        ImportError: If CasADi is not installed in the current environment.
    """

    if measurements.arrival_times_s.shape != (network.num_agents,):
        raise ValueError("measurements.arrival_times_s must have shape (num_agents,).")
    try:
        import casadi as ca
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("casadi is required to run the centralized solver.") from exc

    state = ca.SX.sym("state", 3)
    position_x = state[0]
    position_y = state[1]
    emission_time_s = state[2]

    objective = 0
    for sensor_position_m, arrival_time_s in zip(
        network.sensor_positions_m,
        measurements.arrival_times_s,
        strict=True,
    ):
        dx = position_x - float(sensor_position_m[0])
        dy = position_y - float(sensor_position_m[1])
        distance_m = ca.sqrt(dx * dx + dy * dy)
        residual = float(arrival_time_s) - emission_time_s - distance_m / network.sound_speed_mps
        objective += 0.5 * residual * residual

    solver = ca.nlpsol(
        "toa_localization",
        "ipopt",
        {"x": state, "f": objective},
        {
            "ipopt.max_iter": config.ipopt_max_iter,
            "ipopt.tol": config.ipopt_tol,
            "ipopt.print_level": config.ipopt_print_level,
            "ipopt.sb": "yes",
            "print_time": False,
        },
    )

    start_point = _mean_arrival_start(network, measurements)
    solution = solver(x0=start_point)
    estimate = np.asarray(solution["x"], dtype=float).reshape(-1)
    residual_vector = _toa_residual_vector(
        estimate,
        network.sensor_positions_m,
        network.sound_speed_mps,
        measurements.arrival_times_s,
    )
    stats = solver.stats()

    return CentralizedResult(
        position_m=np.asarray(estimate[:2], dtype=float),
        emission_time_s=float(estimate[2]),
        cost=float(solution["f"]),
        residual_vector=residual_vector,
        success=bool(stats.get("success", False)),
        nfev=int(stats.get("iter_count", 0)),
    )
