import numpy as np

from dadmm_localization.network import build_network
from dadmm_localization.simulation import simulate_toa_measurements
from dadmm_localization.types import TargetState


def test_simulate_toa_measurements_is_exact_without_noise() -> None:
    sensor_positions = np.array(
        [[0.0, 0.0], [300.0, 0.0], [0.0, 400.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        dtype=int,
    )
    network = build_network(sensor_positions, adjacency, sound_speed_mps=1500.0)
    target = TargetState(position_m=np.array([30.0, 40.0], dtype=float), emission_time_s=0.2)

    measurements = simulate_toa_measurements(
        network=network,
        target=target,
        sigma_s=0.0,
        rng=np.random.default_rng(0),
    )

    expected = np.array(
        [
            0.2 + 50.0 / 1500.0,
            0.2 + np.hypot(270.0, 40.0) / 1500.0,
            0.2 + np.hypot(30.0, 360.0) / 1500.0,
        ],
        dtype=float,
    )
    np.testing.assert_allclose(measurements.arrival_times_s, expected)

