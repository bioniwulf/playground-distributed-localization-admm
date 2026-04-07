"""Simulation helpers for targets and ToA measurements."""

from __future__ import annotations

import numpy as np

from dadmm_localization.config import BoxRegion
from dadmm_localization.types import MeasurementSet, Network, TargetState


def spawn_trial_generators(master_seed: int, count: int) -> list[np.random.Generator]:
    """Create reproducible child RNGs for independent trials.

    Args:
        master_seed: Root seed used to derive child generators.
        count: Number of generators to create.

    Returns:
        List of independent NumPy random generators.
    """

    seed_sequence = np.random.SeedSequence(master_seed)
    return [np.random.default_rng(child) for child in seed_sequence.spawn(count)]


def sample_target_in_box(
    region_box: BoxRegion,
    emission_time_s: float,
    rng: np.random.Generator,
) -> TargetState:
    """Sample a target uniformly inside a rectangular region.

    Args:
        region_box: Axis-aligned sampling region in meters.
        emission_time_s: Emission time assigned to the sampled target.
        rng: Random number generator used for sampling.

    Returns:
        Sampled target state.
    """

    position = np.array(
        [
            rng.uniform(region_box.xmin, region_box.xmax),
            rng.uniform(region_box.ymin, region_box.ymax),
        ],
        dtype=float,
    )
    return TargetState(position_m=position, emission_time_s=float(emission_time_s))


def compute_true_arrival_times(network: Network, target: TargetState) -> np.ndarray:
    """Compute noiseless arrival times for all receivers.

    Args:
        network: Validated sensor network.
        target: True target state.

    Returns:
        Noiseless arrival-time vector with shape `(num_agents,)`.
    """

    distances = np.linalg.norm(network.sensor_positions_m - target.position_m, axis=1)
    toa_true = target.emission_time_s + distances / network.sound_speed_mps
    return toa_true


def simulate_toa_measurements(
    network: Network,
    target: TargetState,
    sigma_s: float,
    rng: np.random.Generator,
) -> MeasurementSet:
    """Generate noisy ToA measurements for a target state.

    Args:
        network: Validated sensor network.
        target: True target state used to generate the measurements.
        sigma_s: Standard deviation of Gaussian timing noise in seconds.
        rng: Random number generator used to sample the noise.

    Returns:
        Measurement set containing noisy arrival times and injected noise.

    Raises:
        ValueError: If `sigma_s` is negative.
    """

    if sigma_s < 0.0:
        raise ValueError("sigma_s must be non-negative.")
    noiseless = compute_true_arrival_times(network, target)
    noise = rng.normal(loc=0.0, scale=sigma_s, size=network.num_agents) if sigma_s > 0.0 else np.zeros(
        network.num_agents,
        dtype=float,
    )
    return MeasurementSet(
        arrival_times_s=noiseless + noise,
        noise_s=noise,
        sigma_s=float(sigma_s),
        target_state=target,
    )
