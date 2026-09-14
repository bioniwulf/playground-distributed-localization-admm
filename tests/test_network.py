import numpy as np
import pytest

from dadmm_localization.network import build_network


def test_build_network_rejects_nonsymmetric_adjacency() -> None:
    sensor_positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
        dtype=int,
    )

    with pytest.raises(ValueError, match="symmetric"):
        build_network(sensor_positions, adjacency)


def test_build_network_rejects_disconnected_graph() -> None:
    sensor_positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=int,
    )

    with pytest.raises(ValueError, match="connected"):
        build_network(sensor_positions, adjacency)


def test_build_network_computes_graph_diameter() -> None:
    sensor_positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=float,
    )
    adjacency = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=int,
    )

    network = build_network(sensor_positions, adjacency)

    assert network.diameter == 3
