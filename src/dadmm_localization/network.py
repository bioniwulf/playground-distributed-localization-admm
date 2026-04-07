"""Network validation and graph utilities."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import ArrayLike

from dadmm_localization.constants import DEFAULT_SOUND_SPEED_MPS
from dadmm_localization.types import Network


def _check_connectedness(adjacency_matrix: np.ndarray) -> None:
    """Validate that the graph induced by the adjacency matrix is connected.

    Args:
        adjacency_matrix: Binary square adjacency matrix.

    Raises:
        ValueError: If the graph is disconnected.
    """

    num_agents = adjacency_matrix.shape[0]
    queue: deque[int] = deque([0])
    visited = np.zeros(num_agents, dtype=bool)
    visited[0] = True

    while queue:
        node = queue.popleft()
        neighbors = np.flatnonzero(adjacency_matrix[node])
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(int(neighbor))

    if not np.all(visited):
        raise ValueError("Adjacency matrix must define a connected undirected graph.")


def build_network(
    sensor_positions: ArrayLike,
    adjacency_matrix: ArrayLike,
    sound_speed_mps: float = DEFAULT_SOUND_SPEED_MPS,
) -> Network:
    """Validate raw inputs and create a typed network container.

    Args:
        sensor_positions: Receiver coordinates with shape `(N, 2)`.
        adjacency_matrix: Binary symmetric adjacency matrix with shape `(N, N)`.
        sound_speed_mps: Propagation speed in meters per second.

    Returns:
        Validated network container with derived neighbor lists and centroid.

    Raises:
        ValueError: If geometry, adjacency, or physical parameters are invalid.
    """

    positions = np.asarray(sensor_positions, dtype=float)
    adjacency = np.asarray(adjacency_matrix, dtype=int)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("sensor_positions must have shape (N, 2).")
    if positions.shape[0] < 3:
        raise ValueError("At least three sensors are required in 2D.")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_matrix must be square.")
    if adjacency.shape[0] != positions.shape[0]:
        raise ValueError("sensor_positions and adjacency_matrix must have matching sizes.")
    if sound_speed_mps <= 0.0:
        raise ValueError("sound_speed_mps must be positive.")

    if np.any(np.diag(adjacency) != 0):
        raise ValueError("Adjacency matrix diagonal must be zero.")
    if np.any((adjacency != 0) & (adjacency != 1)):
        raise ValueError("Adjacency matrix entries must be 0 or 1.")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Adjacency matrix must be symmetric.")

    _check_connectedness(adjacency)
    neighbor_indices = tuple(np.flatnonzero(adjacency[i]) for i in range(adjacency.shape[0]))
    centroid = np.mean(positions, axis=0)
    return Network(
        sensor_positions_m=positions,
        adjacency_matrix=adjacency,
        neighbor_indices=neighbor_indices,
        num_agents=positions.shape[0],
        sound_speed_mps=float(sound_speed_mps),
        centroid_m=centroid,
    )
