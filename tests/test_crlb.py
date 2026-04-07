import numpy as np

from dadmm_localization.config import CrlbConfig
from dadmm_localization.crlb import compute_crlb
from dadmm_localization.network import build_network
from dadmm_localization.types import TargetState


def test_crlb_scales_with_sigma_squared() -> None:
    sensor_positions = np.array(
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
        dtype=int,
    )
    network = build_network(sensor_positions, adjacency, sound_speed_mps=1500.0)
    target = TargetState(position_m=np.array([400.0, 600.0], dtype=float), emission_time_s=0.0)
    config = CrlbConfig(epsilon_radius_m=1.0, confidence_k_sigma=3.0)

    crlb_small = compute_crlb(network, target, sigma_s=1.0e-4, config=config)
    crlb_large = compute_crlb(network, target, sigma_s=2.0e-4, config=config)

    np.testing.assert_allclose(crlb_large.covariance, 4.0 * crlb_small.covariance, rtol=1.0e-5)
