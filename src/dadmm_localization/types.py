"""Typed containers used by solvers and experiment runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class Network:
    """Validated sensor network definition."""

    sensor_positions_m: FloatArray
    adjacency_matrix: IntArray
    neighbor_indices: tuple[IntArray, ...]
    num_agents: int
    sound_speed_mps: float
    centroid_m: FloatArray


@dataclass(slots=True)
class TargetState:
    """True or estimated target state."""

    position_m: FloatArray
    emission_time_s: float


@dataclass(slots=True)
class MeasurementSet:
    """ToA measurements and associated metadata."""

    arrival_times_s: FloatArray
    noise_s: FloatArray
    sigma_s: float
    target_state: TargetState | None = None


@dataclass(slots=True)
class WarmStart:
    """Optional warm start for DADMM."""

    positions_m: FloatArray
    emission_times_s: FloatArray


@dataclass(slots=True)
class DadmmIterationRecord:
    """One DADMM iteration snapshot."""

    iteration: int
    agent_positions_m: FloatArray
    agent_times_s: FloatArray
    consensus_position_m: FloatArray
    consensus_time_s: float
    local_costs: FloatArray
    primal_residuals: FloatArray
    dual_residuals: FloatArray
    consensus_gap: float


@dataclass(slots=True)
class DadmmResult:
    """Final DADMM outputs."""

    agent_positions_m: FloatArray
    agent_times_s: FloatArray
    consensus_position_m: FloatArray
    consensus_time_s: float
    iterations_global: int
    iterations_per_agent: IntArray
    converged: bool
    final_primal_residuals: FloatArray
    final_dual_residuals: FloatArray
    final_local_costs: FloatArray
    history: tuple[DadmmIterationRecord, ...] = ()


@dataclass(slots=True)
class CentralizedResult:
    """Centralized solver result."""

    position_m: FloatArray
    emission_time_s: float
    cost: float
    residual_vector: FloatArray
    success: bool
    nfev: int


@dataclass(slots=True)
class CrlbResult:
    """CRLB and Fisher information outputs."""

    fim: FloatArray
    covariance: FloatArray
    position_covariance: FloatArray
    position_rmse_bound_m: float
    time_std_bound_s: float
    valid: bool
    fim_condition_number: float


@dataclass(slots=True)
class RunArtifacts:
    """Paths and logger metadata for a single experiment run."""

    run_dir: Any
    logger_name: str
    manifest: dict[str, Any] = field(default_factory=dict)
