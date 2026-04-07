"""Synchronous DADMM implementation for distributed ToA localization."""

from __future__ import annotations

import numpy as np

from dadmm_localization.config import DadmmConfig
from dadmm_localization.metrics import consensus_gap, local_toa_cost
from dadmm_localization.types import (
    DadmmIterationRecord,
    DadmmResult,
    MeasurementSet,
    Network,
    WarmStart,
)


def _weighted_state_norm(
    delta_position_m: np.ndarray,
    delta_time_s: float,
    rho_p: float,
    rho_t: float,
) -> float:
    """Compute the weighted norm used in residual calculations.

    Args:
        delta_position_m: Position difference vector.
        delta_time_s: Emission-time difference.
        rho_p: Position penalty parameter.
        rho_t: Time penalty parameter.

    Returns:
        Weighted Euclidean norm in the stacked state space.
    """

    return float(
        np.sqrt(
            rho_p * float(np.dot(delta_position_m, delta_position_m))
            + rho_t * float(delta_time_s * delta_time_s)
        )
    )


def _resolve_direction(
    primary_vector: np.ndarray,
    previous_vector: np.ndarray,
    centroid_vector: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Select a valid normalized update direction with deterministic fallbacks.

    Args:
        primary_vector: Preferred direction vector.
        previous_vector: Fallback direction based on the previous iterate.
        centroid_vector: Final fallback direction toward the network centroid.
        epsilon: Minimum acceptable norm for a usable direction.

    Returns:
        Normalized direction vector.

    Raises:
        ValueError: If all candidate directions are degenerate.
    """

    for candidate in (primary_vector, previous_vector, centroid_vector):
        norm = np.linalg.norm(candidate)
        if norm >= epsilon:
            return candidate / norm
    raise ValueError("Degenerate geometry: unable to determine a valid update direction.")


def _solve_local_time_radius_closed_form(
    *,
    arrival_time_s: float,
    sound_speed_mps: float,
    anchor_time_s: float,
    anchor_radius_m: float,
    degree: int,
    rho_p: float,
    rho_t: float,
) -> tuple[float, float]:
    """Solve the reduced strictly convex quadratic subproblem in `(t, r)`.

    Args:
        arrival_time_s: Local measured arrival time.
        sound_speed_mps: Propagation speed in meters per second.
        anchor_time_s: Consensus anchor for the local time update.
        anchor_radius_m: Consensus anchor for the local radial update.
        degree: Number of neighbors for the current agent.
        rho_p: Position penalty parameter.
        rho_t: Time penalty parameter.

    Returns:
        Tuple `(next_time_s, next_radius_m)` for the local update.

    Raises:
        ValueError: If the current agent has no neighbors.
    """

    if degree <= 0:
        raise ValueError("Each agent must have at least one neighbor in a connected graph.")

    inv_sound_speed = 1.0 / sound_speed_mps
    system = np.array(
        [
            [1.0 + rho_t * degree, inv_sound_speed],
            [inv_sound_speed, inv_sound_speed * inv_sound_speed + rho_p * degree],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            arrival_time_s + rho_t * degree * anchor_time_s,
            arrival_time_s * inv_sound_speed + rho_p * degree * anchor_radius_m,
        ],
        dtype=float,
    )

    next_time_s, next_radius_m = np.linalg.solve(system, rhs)
    if next_radius_m < 0.0:
        next_radius_m = 0.0
        next_time_s = (
            arrival_time_s + rho_t * degree * anchor_time_s
        ) / (1.0 + rho_t * degree)

    return float(next_time_s), float(next_radius_m)


def _compose_position_update(
    *,
    sensor_position_m: np.ndarray,
    anchor_position_m: np.ndarray,
    radius_m: float,
    previous_position_m: np.ndarray,
    centroid_position_m: np.ndarray,
    direction_epsilon: float,
) -> np.ndarray:
    """Map the solved radial update back to Cartesian coordinates.

    Args:
        sensor_position_m: Receiver position for the current agent.
        anchor_position_m: Consensus anchor position.
        radius_m: Solved radial distance from the receiver.
        previous_position_m: Previous local position iterate.
        centroid_position_m: Network centroid used as the final direction fallback.
        direction_epsilon: Minimum acceptable direction norm.

    Returns:
        Updated Cartesian position for the current agent.
    """

    if radius_m <= 0.0:
        return sensor_position_m.copy()

    direction = _resolve_direction(
        primary_vector=anchor_position_m - sensor_position_m,
        previous_vector=previous_position_m - sensor_position_m,
        centroid_vector=centroid_position_m - sensor_position_m,
        epsilon=direction_epsilon,
    )
    return sensor_position_m + radius_m * direction


def _build_default_warm_start(network: Network, measurements: MeasurementSet) -> WarmStart:
    """Build the default warm start used when no explicit initialization is provided.

    Args:
        network: Validated sensor network.
        measurements: ToA measurements for the current trial.

    Returns:
        Per-agent warm start for positions and emission times.
    """

    positions_m = np.empty((network.num_agents, 2), dtype=float)
    emission_times_s = np.empty(network.num_agents, dtype=float)

    for agent_id, neighbors in enumerate(network.neighbor_indices):
        local_sensor_positions = np.vstack(
            [
                network.sensor_positions_m[agent_id],
                network.sensor_positions_m[neighbors],
            ]
        )
        local_centroid_m = np.mean(local_sensor_positions, axis=0)
        positions_m[agent_id] = local_centroid_m

        range_m = float(np.linalg.norm(local_centroid_m - network.sensor_positions_m[agent_id]))
        emission_times_s[agent_id] = (
            measurements.arrival_times_s[agent_id] - range_m / network.sound_speed_mps
        )

    return WarmStart(positions_m=positions_m, emission_times_s=emission_times_s)


def _solve_local_primal_edge_closed_form(
    *,
    sensor_position_m: np.ndarray,
    arrival_time_s: float,
    sound_speed_mps: float,
    z_neighbor_positions: np.ndarray,
    z_neighbor_times: np.ndarray,
    u_neighbor_positions: np.ndarray,
    u_neighbor_times: np.ndarray,
    rho_p: float,
    rho_t: float,
    previous_position_m: np.ndarray,
    centroid_position_m: np.ndarray,
    direction_epsilon: float,
) -> tuple[np.ndarray, float]:
    """Compute the closed-form local primal update from Appendix A.

    Args:
        sensor_position_m: Receiver position for the current agent.
        arrival_time_s: Local measured arrival time.
        sound_speed_mps: Propagation speed in meters per second.
        z_neighbor_positions: Edge consensus positions for all neighbors.
        z_neighbor_times: Edge consensus times for all neighbors.
        u_neighbor_positions: Scaled dual position variables for all neighbors.
        u_neighbor_times: Scaled dual time variables for all neighbors.
        rho_p: Position penalty parameter.
        rho_t: Time penalty parameter.
        previous_position_m: Previous local position iterate.
        centroid_position_m: Network centroid used as a direction fallback.
        direction_epsilon: Minimum acceptable direction norm.

    Returns:
        Tuple `(next_position_m, next_time_s)` for the local primal update.

    Raises:
        ValueError: If the current agent has no neighbors.
    """

    degree = int(z_neighbor_times.shape[0])
    if degree <= 0:
        raise ValueError("Each agent must have at least one neighbor in a connected graph.")

    avg_anchor_position = np.mean(z_neighbor_positions - u_neighbor_positions, axis=0)
    avg_anchor_time = float(np.mean(z_neighbor_times - u_neighbor_times))

    next_time, next_radius = _solve_local_time_radius_closed_form(
        arrival_time_s=arrival_time_s,
        sound_speed_mps=sound_speed_mps,
        anchor_time_s=avg_anchor_time,
        anchor_radius_m=float(np.linalg.norm(avg_anchor_position - sensor_position_m)),
        degree=degree,
        rho_p=rho_p,
        rho_t=rho_t,
    )
    next_position = _compose_position_update(
        sensor_position_m=sensor_position_m,
        anchor_position_m=avg_anchor_position,
        radius_m=next_radius,
        previous_position_m=previous_position_m,
        centroid_position_m=centroid_position_m,
        direction_epsilon=direction_epsilon,
    )
    return next_position, next_time


def run_dadmm(
    network: Network,
    measurements: MeasurementSet,
    config: DadmmConfig,
    init: WarmStart | None = None,
) -> DadmmResult:
    """Run the synchronous edge-based DADMM localization algorithm.

    Args:
        network: Validated sensor network.
        measurements: ToA measurements for all receivers.
        config: DADMM solver configuration.
        init: Optional explicit warm start shared across agents.

    Returns:
        Final DADMM state, convergence metrics, and optional iteration history.

    Raises:
        ValueError: If the dual-update rule is unsupported or shapes are invalid.
    """

    if config.dual_update_rule != "standard_consensus":
        raise ValueError(f"Unsupported dual update rule: {config.dual_update_rule}")
    if measurements.arrival_times_s.shape != (network.num_agents,):
        raise ValueError("measurements.arrival_times_s must have shape (num_agents,).")

    num_agents = network.num_agents
    rho_p = config.rho_p
    rho_t = config.rho_t
    sound_speed = network.sound_speed_mps

    if init is None:
        warm_start = _build_default_warm_start(network, measurements)
        positions_m = warm_start.positions_m
        emission_times_s = warm_start.emission_times_s
    else:
        positions_m = np.asarray(init.positions_m, dtype=float).copy()
        emission_times_s = np.asarray(init.emission_times_s, dtype=float).copy()
        if positions_m.shape != (num_agents, 2):
            raise ValueError("WarmStart.positions_m must have shape (num_agents, 2).")
        if emission_times_s.shape != (num_agents,):
            raise ValueError("WarmStart.emission_times_s must have shape (num_agents,).")

    # Edge variables z_ij and dual u_ij stored in dense form for simplicity.
    z_positions = np.zeros((num_agents, num_agents, 2), dtype=float)
    z_times = np.zeros((num_agents, num_agents), dtype=float)
    u_positions = np.zeros((num_agents, num_agents, 2), dtype=float)
    u_times = np.zeros((num_agents, num_agents), dtype=float)

    iterations_per_agent = np.full(num_agents, -1, dtype=int)
    converged_mask = np.zeros(num_agents, dtype=bool)
    history: list[DadmmIterationRecord] = []
    final_primal = np.full(num_agents, np.inf, dtype=float)
    final_dual = np.full(num_agents, np.inf, dtype=float)
    final_costs = np.full(num_agents, np.inf, dtype=float)
    completed_iterations = 0

    # Initialize z on existing edges only.
    for i in range(num_agents):
        for j in network.neighbor_indices[i]:
            if j > i:
                z_positions[i, j] = 0.5 * (positions_m[i] + positions_m[j])
                z_positions[j, i] = z_positions[i, j]
                z_times[i, j] = 0.5 * (emission_times_s[i] + emission_times_s[j])
                z_times[j, i] = z_times[i, j]

    for iteration in range(1, config.max_iterations + 1):
        positions_prev = positions_m.copy()
        times_prev = emission_times_s.copy()

        next_positions = np.empty_like(positions_m)
        next_times = np.empty_like(emission_times_s)

        # 1) Local primal update
        for i in range(num_agents):
            nbrs = network.neighbor_indices[i]

            next_positions[i], next_times[i] = _solve_local_primal_edge_closed_form(
                sensor_position_m=network.sensor_positions_m[i],
                arrival_time_s=measurements.arrival_times_s[i],
                sound_speed_mps=sound_speed,
                z_neighbor_positions=z_positions[i, nbrs],
                z_neighbor_times=z_times[i, nbrs],
                u_neighbor_positions=u_positions[i, nbrs],
                u_neighbor_times=u_times[i, nbrs],
                rho_p=rho_p,
                rho_t=rho_t,
                previous_position_m=positions_prev[i],
                centroid_position_m=network.centroid_m,
                direction_epsilon=config.direction_epsilon,
            )

        # 2) Edge update
        for i in range(num_agents):
            for j in network.neighbor_indices[i]:
                if j > i:
                    z_pos = 0.5 * (
                        next_positions[i]
                        + next_positions[j]
                        + u_positions[i, j]
                        + u_positions[j, i]
                    )
                    z_t = 0.5 * (next_times[i] + next_times[j] + u_times[i, j] + u_times[j, i])

                    z_positions[i, j] = z_pos
                    z_positions[j, i] = z_pos
                    z_times[i, j] = z_t
                    z_times[j, i] = z_t

        # 3) Dual update
        for i in range(num_agents):
            nbrs = network.neighbor_indices[i]
            u_positions[i, nbrs] += next_positions[i] - z_positions[i, nbrs]
            u_times[i, nbrs] += next_times[i] - z_times[i, nbrs]

        primal_residuals = np.zeros(num_agents, dtype=float)
        dual_residuals = np.zeros(num_agents, dtype=float)
        local_costs = np.zeros(num_agents, dtype=float)
        for agent_id in range(num_agents):
            neighbors = network.neighbor_indices[agent_id]
            weighted_neighbor_norms = [
                _weighted_state_norm(
                    delta_position_m=next_positions[agent_id] - z_positions[agent_id, neighbor_id],
                    delta_time_s=float(next_times[agent_id] - z_times[agent_id, neighbor_id]),
                    rho_p=rho_p,
                    rho_t=rho_t,
                )
                for neighbor_id in neighbors
            ]
            primal_residuals[agent_id] = (
                max(weighted_neighbor_norms) if weighted_neighbor_norms else 0.0
            )
            dual_residuals[agent_id] = len(neighbors) * _weighted_state_norm(
                delta_position_m=next_positions[agent_id] - positions_prev[agent_id],
                delta_time_s=float(next_times[agent_id] - times_prev[agent_id]),
                rho_p=rho_p,
                rho_t=rho_t,
            )
            local_costs[agent_id] = local_toa_cost(
                sensor_position_m=network.sensor_positions_m[agent_id],
                sound_speed_mps=sound_speed,
                arrival_time_s=measurements.arrival_times_s[agent_id],
                position_m=next_positions[agent_id],
                emission_time_s=float(next_times[agent_id]),
            )

        mean_position = np.mean(next_positions, axis=0)
        mean_time = float(np.mean(next_times))
        max_consensus_gap = consensus_gap(next_positions, next_times)

        if config.save_history:
            history.append(
                DadmmIterationRecord(
                    iteration=iteration,
                    agent_positions_m=next_positions.copy(),
                    agent_times_s=next_times.copy(),
                    consensus_position_m=mean_position.copy(),
                    consensus_time_s=mean_time,
                    local_costs=local_costs.copy(),
                    primal_residuals=primal_residuals.copy(),
                    dual_residuals=dual_residuals.copy(),
                    consensus_gap=max_consensus_gap,
                )
            )

        newly_converged = (
            (~converged_mask)
            & (primal_residuals <= config.eps_feas)
            & (dual_residuals <= config.eps_conv)
        )
        iterations_per_agent[newly_converged] = iteration
        converged_mask |= newly_converged

        positions_m = next_positions
        emission_times_s = next_times
        final_primal = primal_residuals
        final_dual = dual_residuals
        final_costs = local_costs
        completed_iterations = iteration

        if np.all(converged_mask):
            break

    if not np.all(converged_mask):
        iterations_per_agent[~converged_mask] = config.max_iterations

    return DadmmResult(
        agent_positions_m=positions_m,
        agent_times_s=emission_times_s,
        consensus_position_m=np.mean(positions_m, axis=0),
        consensus_time_s=float(np.mean(emission_times_s)),
        iterations_global=completed_iterations,
        iterations_per_agent=iterations_per_agent,
        converged=bool(np.all(converged_mask)),
        final_primal_residuals=final_primal,
        final_dual_residuals=final_dual,
        final_local_costs=final_costs,
        history=tuple(history),
    )
