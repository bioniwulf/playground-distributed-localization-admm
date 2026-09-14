import numpy as np
from scipy.optimize import minimize

from dadmm_localization.config import DadmmConfig
from dadmm_localization.dadmm_solver import (
    _advance_distributed_stopping_flags,
    _build_default_warm_start,
    _compute_local_admm_residuals,
    _solve_local_primal_edge_closed_form,
    run_dadmm,
)
from dadmm_localization.network import build_network
from dadmm_localization.simulation import simulate_toa_measurements
from dadmm_localization.types import MeasurementSet, TargetState, WarmStart


def _square_network():
    sensor_positions = np.array(
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
        dtype=int,
    )
    return build_network(sensor_positions, adjacency, sound_speed_mps=1500.0)


def _solve_local_primal_edge_numeric(
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
    x0_position_m: np.ndarray,
    x0_time_s: float,
) -> tuple[np.ndarray, float]:
    x0 = np.array([x0_position_m[0], x0_position_m[1], x0_time_s], dtype=float)

    def fun(x: np.ndarray) -> float:
        p = x[:2]
        t = float(x[2])

        residual = arrival_time_s - t - np.linalg.norm(p - sensor_position_m) / sound_speed_mps
        local_cost = 0.5 * residual * residual

        dp = p[None, :] - z_neighbor_positions + u_neighbor_positions
        dt = t - z_neighbor_times + u_neighbor_times

        penalty = 0.5 * (rho_p * float(np.sum(dp * dp)) + rho_t * float(np.sum(dt * dt)))
        return float(local_cost + penalty)

    res = minimize(fun, x0, method="L-BFGS-B")
    x = res.x if np.all(np.isfinite(res.x)) else x0
    return x[:2].copy(), float(x[2])


def _local_primal_objective(
    x: np.ndarray,
    *,
    sensor_position_m: np.ndarray,
    arrival_time_s: float,
    sound_speed_mps: float,
    neighbor_positions_m: np.ndarray,
    neighbor_times_s: np.ndarray,
    dual_positions_ij: np.ndarray,
    dual_times_ij: np.ndarray,
    rho_p: float,
    rho_t: float,
) -> float:
    position_m = x[:2]
    emission_time_s = float(x[2])

    residual = (
        arrival_time_s
        - emission_time_s
        - np.linalg.norm(position_m - sensor_position_m) / sound_speed_mps
    )
    local_cost = 0.5 * residual * residual

    delta_positions = position_m[None, :] - neighbor_positions_m + dual_positions_ij
    delta_times = emission_time_s - neighbor_times_s + dual_times_ij

    consensus_penalty = 0.5 * (
        rho_p * float(np.sum(delta_positions * delta_positions))
        + rho_t * float(np.sum(delta_times * delta_times))
    )

    return float(local_cost + consensus_penalty)


def test_run_dadmm_produces_finite_outputs() -> None:
    network = _square_network()
    target = TargetState(position_m=np.array([450.0, 550.0], dtype=float), emission_time_s=0.0)
    measurements = simulate_toa_measurements(
        network, target, sigma_s=1.0e-5, rng=np.random.default_rng(0)
    )
    config = DadmmConfig(rho_p=3.0e-6, rho_t=5.0, max_iterations=250)

    result = run_dadmm(network, measurements, config)

    assert np.all(np.isfinite(result.agent_positions_m))
    assert np.all(np.isfinite(result.agent_times_s))
    assert np.all(np.isfinite(result.final_primal_residuals))
    assert np.all(np.isfinite(result.final_dual_residuals))


def test_distributed_stopping_flags_propagate_over_graph_diameter() -> None:
    neighbor_indices = (
        np.array([1], dtype=int),
        np.array([0, 2], dtype=int),
        np.array([1, 3], dtype=int),
        np.array([2], dtype=int),
    )
    diameter = 3
    flags = np.zeros((4, diameter + 1), dtype=bool)

    flags = _advance_distributed_stopping_flags(
        previous_flags=flags,
        local_converged=np.array([False, True, True, True], dtype=bool),
        neighbor_indices=neighbor_indices,
    )
    for _ in range(diameter):
        flags = _advance_distributed_stopping_flags(
            previous_flags=flags,
            local_converged=np.ones(4, dtype=bool),
            neighbor_indices=neighbor_indices,
        )

    assert np.all(flags[:, diameter])


def test_distributed_stopping_flags_delay_termination_by_graph_diameter() -> None:
    network = _square_network()
    flags = np.ones((network.num_agents, network.diameter + 1), dtype=bool)
    local_converged = np.ones(network.num_agents, dtype=bool)

    flags = _advance_distributed_stopping_flags(
        previous_flags=flags,
        local_converged=local_converged,
        neighbor_indices=network.neighbor_indices,
    )
    assert np.all(flags[:, network.diameter])

    flags = _advance_distributed_stopping_flags(
        previous_flags=flags,
        local_converged=local_converged,
        neighbor_indices=network.neighbor_indices,
    )
    assert not np.any(flags[:, network.diameter])


def test_local_admm_residuals_match_algorithm_formulas() -> None:
    positions = np.array([[2.0, 1.0], [4.0, 1.0]], dtype=float)
    times = np.array([0.5, 0.7], dtype=float)
    edge_positions = np.zeros((2, 2, 2), dtype=float)
    previous_edge_positions = np.zeros_like(edge_positions)
    edge_times = np.zeros((2, 2), dtype=float)
    previous_edge_times = np.zeros_like(edge_times)
    edge_positions[0, 1] = edge_positions[1, 0] = np.array([3.0, 1.0])
    previous_edge_positions[0, 1] = previous_edge_positions[1, 0] = np.array([2.5, 0.5])
    edge_times[0, 1] = edge_times[1, 0] = 0.6
    previous_edge_times[0, 1] = previous_edge_times[1, 0] = 0.55
    neighbor_indices = (np.array([1], dtype=int), np.array([0], dtype=int))

    primal, dual = _compute_local_admm_residuals(
        positions_m=positions,
        emission_times_s=times,
        edge_positions_m=edge_positions,
        previous_edge_positions_m=previous_edge_positions,
        edge_times_s=edge_times,
        previous_edge_times_s=previous_edge_times,
        neighbor_indices=neighbor_indices,
        rho_p=2.0,
        rho_t=3.0,
    )

    expected_primal = np.sqrt(2.0 * 1.0**2 + 3.0 * 0.1**2)
    expected_dual = np.sqrt(2.0 * (0.5**2 + 0.5**2) + 3.0 * 0.05**2)
    np.testing.assert_allclose(primal, expected_primal)
    np.testing.assert_allclose(dual, expected_dual)


def test_default_warm_start_uses_local_receiver_centroids() -> None:
    network = _square_network()
    arrival_times = np.array([0.70, 0.71, 0.72, 0.73], dtype=float)
    measurements = MeasurementSet(
        arrival_times_s=arrival_times,
        noise_s=np.zeros(network.num_agents, dtype=float),
        sigma_s=0.0,
        target_state=None,
    )

    warm_start = _build_default_warm_start(network, measurements)
    expected_positions = np.empty_like(warm_start.positions_m)
    expected_times = np.empty_like(warm_start.emission_times_s)

    for agent_id, neighbors in enumerate(network.neighbor_indices):
        local_sensor_positions = np.vstack(
            [
                network.sensor_positions_m[agent_id],
                network.sensor_positions_m[neighbors],
            ]
        )
        expected_positions[agent_id] = np.mean(local_sensor_positions, axis=0)
        expected_range_m = np.linalg.norm(
            expected_positions[agent_id] - network.sensor_positions_m[agent_id]
        )
        expected_times[agent_id] = (
            arrival_times[agent_id] - expected_range_m / network.sound_speed_mps
        )

    np.testing.assert_allclose(
        warm_start.positions_m,
        expected_positions,
    )
    np.testing.assert_allclose(
        warm_start.emission_times_s,
        expected_times,
    )


def test_run_dadmm_direction_safeguard_handles_zero_seed_vector() -> None:
    network = _square_network()
    arrival_times = np.full(network.num_agents, 0.75, dtype=float)
    measurements = MeasurementSet(
        arrival_times_s=arrival_times,
        noise_s=np.zeros(network.num_agents, dtype=float),
        sigma_s=0.0,
        target_state=None,
    )
    warm_start = WarmStart(
        positions_m=np.repeat(network.centroid_m[None, :], network.num_agents, axis=0),
        emission_times_s=arrival_times.copy(),
    )
    config = DadmmConfig(rho_p=3.0e-6, rho_t=5.0, max_iterations=5)

    result = run_dadmm(network, measurements, config, init=warm_start)

    assert np.all(np.isfinite(result.agent_positions_m))
    assert np.all(np.isfinite(result.agent_times_s))


def test_closed_form_local_update_matches_numeric_solution() -> None:
    sensor_position = np.array([100.0, -50.0], dtype=float)
    previous_position = np.array([145.0, -12.0], dtype=float)
    centroid_position = np.array([180.0, 20.0], dtype=float)
    arrival_time = 0.62
    sound_speed = 1500.0
    rho_p = 1.2e-5
    rho_t = 4.0

    z_neighbor_positions = np.array(
        [[140.0, -20.0], [160.0, 10.0], [120.0, 30.0]],
        dtype=float,
    )
    z_neighbor_times = np.array([0.5010, 0.4995, 0.5030], dtype=float)
    u_neighbor_positions = np.array(
        [[1.0, -2.0], [-1.5, 1.0], [0.5, 0.75]],
        dtype=float,
    )
    u_neighbor_times = np.array([2.0e-4, -1.0e-4, 1.5e-4], dtype=float)

    closed_position, closed_time = _solve_local_primal_edge_closed_form(
        sensor_position_m=sensor_position,
        arrival_time_s=arrival_time,
        sound_speed_mps=sound_speed,
        z_neighbor_positions=z_neighbor_positions,
        z_neighbor_times=z_neighbor_times,
        u_neighbor_positions=u_neighbor_positions,
        u_neighbor_times=u_neighbor_times,
        rho_p=rho_p,
        rho_t=rho_t,
        previous_position_m=previous_position,
        centroid_position_m=centroid_position,
        direction_epsilon=1.0e-9,
    )
    numeric_position, numeric_time = _solve_local_primal_edge_numeric(
        sensor_position_m=sensor_position,
        arrival_time_s=arrival_time,
        sound_speed_mps=sound_speed,
        z_neighbor_positions=z_neighbor_positions,
        z_neighbor_times=z_neighbor_times,
        u_neighbor_positions=u_neighbor_positions,
        u_neighbor_times=u_neighbor_times,
        rho_p=rho_p,
        rho_t=rho_t,
        x0_position_m=previous_position,
        x0_time_s=0.45,
    )

    closed_x = np.array([closed_position[0], closed_position[1], closed_time], dtype=float)
    numeric_x = np.array([numeric_position[0], numeric_position[1], numeric_time], dtype=float)

    closed_objective = _local_primal_objective(
        closed_x,
        sensor_position_m=sensor_position,
        arrival_time_s=arrival_time,
        sound_speed_mps=sound_speed,
        neighbor_positions_m=z_neighbor_positions,
        neighbor_times_s=z_neighbor_times,
        dual_positions_ij=u_neighbor_positions,
        dual_times_ij=u_neighbor_times,
        rho_p=rho_p,
        rho_t=rho_t,
    )
    numeric_objective = _local_primal_objective(
        numeric_x,
        sensor_position_m=sensor_position,
        arrival_time_s=arrival_time,
        sound_speed_mps=sound_speed,
        neighbor_positions_m=z_neighbor_positions,
        neighbor_times_s=z_neighbor_times,
        dual_positions_ij=u_neighbor_positions,
        dual_times_ij=u_neighbor_times,
        rho_p=rho_p,
        rho_t=rho_t,
    )

    np.testing.assert_allclose(closed_position, numeric_position, atol=1.0e-6, rtol=1.0e-6)
    assert abs(closed_time - numeric_time) <= 1.0e-9
    assert abs(closed_objective - numeric_objective) <= 1.0e-12


def test_run_dadmm_converges_in_noiseless_case() -> None:
    network = _square_network()
    target = TargetState(position_m=np.array([420.0, 580.0], dtype=float), emission_time_s=0.0)
    measurements = simulate_toa_measurements(
        network, target, sigma_s=0.0, rng=np.random.default_rng(0)
    )
    config = DadmmConfig(
        rho_p=3.0e-6, rho_t=5.0, eps_feas=1.0e-6, eps_conv=1.0e-6, max_iterations=500
    )

    result = run_dadmm(network, measurements, config)

    assert result.converged
    assert np.all(result.iterations_per_agent == result.iterations_global)
    assert float(np.max(result.final_primal_residuals)) <= config.eps_feas
    assert float(np.max(result.final_dual_residuals)) <= config.eps_conv
    np.testing.assert_allclose(result.consensus_position_m, target.position_m, atol=1.0e-2)
    assert abs(result.consensus_time_s - target.emission_time_s) <= 1.0e-5
