"""Fisher information matrix and CRLB computation."""

from __future__ import annotations

import numpy as np

from dadmm_localization.config import CrlbConfig
from dadmm_localization.types import CrlbResult, Network, TargetState


def compute_crlb(
    network: Network,
    target: TargetState,
    sigma_s: float,
    config: CrlbConfig,
) -> CrlbResult:
    """Compute the CRLB for the ToA model with unknown emission time.

    Args:
        network: Validated sensor network.
        target: Target state where the Fisher matrix is evaluated.
        sigma_s: Measurement noise standard deviation in seconds.
        config: CRLB-specific numerical settings.

    Returns:
        Fisher information matrix, covariance estimate, and CRLB summaries.

    Raises:
        ValueError: If `sigma_s` is not strictly positive.
    """

    if sigma_s <= 0.0:
        raise ValueError("sigma_s must be positive to compute CRLB.")

    deltas = target.position_m - network.sensor_positions_m
    distances = np.linalg.norm(deltas, axis=1)
    safe_distances = np.maximum(distances, config.epsilon_radius_m)

    jacobian = np.empty((network.num_agents, 3), dtype=float)
    jacobian[:, 0] = deltas[:, 0] / (network.sound_speed_mps * safe_distances)
    jacobian[:, 1] = deltas[:, 1] / (network.sound_speed_mps * safe_distances)
    jacobian[:, 2] = 1.0

    fim = (jacobian.T @ jacobian) / (sigma_s**2)
    rank = int(np.linalg.matrix_rank(fim))
    condition_number = float(np.linalg.cond(fim))
    valid = rank == 3 and np.isfinite(condition_number)
    covariance = np.linalg.inv(fim) if valid else np.linalg.pinv(fim)
    position_covariance = covariance[:2, :2]
    position_rmse_bound_m = float(np.sqrt(np.trace(position_covariance)))
    time_std_bound_s = float(np.sqrt(max(covariance[2, 2], 0.0)))

    return CrlbResult(
        fim=fim,
        covariance=covariance,
        position_covariance=position_covariance,
        position_rmse_bound_m=position_rmse_bound_m,
        time_std_bound_s=time_std_bound_s,
        valid=valid,
        fim_condition_number=condition_number,
    )
