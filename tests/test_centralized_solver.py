import numpy as np
import pytest

from dadmm_localization.centralized_solver import solve_centralized
from dadmm_localization.config import CentralizedConfig
from dadmm_localization.network import build_network
from dadmm_localization.simulation import simulate_toa_measurements
from dadmm_localization.types import TargetState


def test_solve_centralized_recovers_noiseless_solution() -> None:
    sensor_positions = np.array(
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
        dtype=int,
    )
    network = build_network(sensor_positions, adjacency, sound_speed_mps=1500.0)
    target = TargetState(position_m=np.array([450.0, 525.0], dtype=float), emission_time_s=0.012)
    measurements = simulate_toa_measurements(
        network, target, sigma_s=0.0, rng=np.random.default_rng(0)
    )
    pytest.importorskip("casadi")
    config = CentralizedConfig(ipopt_max_iter=500, ipopt_tol=1.0e-10)

    result = solve_centralized(network, measurements, config)

    assert result.success
    assert result.cost <= 1.0e-6
    np.testing.assert_allclose(result.position_m, target.position_m, atol=1.0e-3)
    assert abs(result.emission_time_s - target.emission_time_s) <= 1.0e-8
