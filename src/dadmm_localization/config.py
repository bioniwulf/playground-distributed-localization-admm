"""Configuration models and YAML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dadmm_localization.constants import (
    DEFAULT_CONFIDENCE_K_SIGMA,
    DEFAULT_DIRECTION_EPSILON,
    DEFAULT_DUAL_UPDATE_RULE,
    DEFAULT_EMISSION_TIME_S,
    DEFAULT_EPSILON_RADIUS_M,
    DEFAULT_MASTER_SEED,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOUND_SPEED_MPS,
)


class StrictModel(BaseModel):
    """Base model with strict schema handling."""

    model_config = ConfigDict(extra="forbid")


class BoxRegion(StrictModel):
    """Axis-aligned 2D box."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoxRegion":
        """Validate that the box bounds define a non-empty region.

        Args:
            self: Parsed box region model.

        Returns:
            The validated box region instance.
        """

        if self.xmin >= self.xmax:
            raise ValueError("Expected xmin < xmax.")
        if self.ymin >= self.ymax:
            raise ValueError("Expected ymin < ymax.")
        return self


class NetworkConfig(StrictModel):
    """Network geometry and connectivity."""

    sensor_positions: list[list[float]]
    adjacency_matrix: list[list[int]]

    @field_validator("sensor_positions")
    @classmethod
    def validate_sensor_positions(cls, value: list[list[float]]) -> list[list[float]]:
        """Validate the receiver position list.

        Args:
            cls: Model class.
            value: Candidate sensor-position list.

        Returns:
            Validated sensor-position list.
        """

        if len(value) < 3:
            raise ValueError("At least three sensors are required in 2D.")
        for row in value:
            if len(row) != 2:
                raise ValueError("Sensor positions must be 2D coordinates.")
        return value

    @field_validator("adjacency_matrix")
    @classmethod
    def validate_adjacency_matrix(cls, value: list[list[int]]) -> list[list[int]]:
        """Validate the adjacency-matrix structure.

        Args:
            cls: Model class.
            value: Candidate adjacency matrix.

        Returns:
            Validated adjacency matrix.
        """

        size = len(value)
        if size == 0:
            raise ValueError("Adjacency matrix cannot be empty.")
        for row in value:
            if len(row) != size:
                raise ValueError("Adjacency matrix must be square.")
        return value

    @model_validator(mode="after")
    def validate_shape_consistency(self) -> "NetworkConfig":
        """Validate consistency between sensor and adjacency dimensions.

        Args:
            self: Parsed network config model.

        Returns:
            The validated network config instance.
        """

        if len(self.sensor_positions) != len(self.adjacency_matrix):
            raise ValueError("sensor_positions and adjacency_matrix must have matching sizes.")
        return self


class PhysicsConfig(StrictModel):
    """Physical constants for the simulation."""

    sound_speed_mps: float = DEFAULT_SOUND_SPEED_MPS
    emission_time_s: float = DEFAULT_EMISSION_TIME_S

    @field_validator("sound_speed_mps")
    @classmethod
    def validate_sound_speed(cls, value: float) -> float:
        """Validate the configured sound speed.

        Args:
            cls: Model class.
            value: Candidate sound speed.

        Returns:
            Validated sound speed.
        """

        if value <= 0.0:
            raise ValueError("sound_speed_mps must be positive.")
        return value


class DadmmConfig(StrictModel):
    """DADMM parameters."""

    rho_p: float = 3.0e-6
    rho_t: float = 5.0
    eps_feas: float = 1e-6
    eps_conv: float = 1e-6
    eps_values: list[float] | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    direction_epsilon: float = DEFAULT_DIRECTION_EPSILON
    save_history: bool = False
    dual_update_rule: Literal["standard_consensus"] = DEFAULT_DUAL_UPDATE_RULE
    warm_start: "SharedWarmStartConfig | None" = None

    @field_validator("rho_p", "rho_t", "eps_feas", "eps_conv", "direction_epsilon")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Validate a strictly positive DADMM scalar parameter.

        Args:
            cls: Model class.
            value: Candidate scalar value.

        Returns:
            Validated scalar value.
        """

        if value <= 0.0:
            raise ValueError("Expected a positive value.")
        return value

    @field_validator("eps_values")
    @classmethod
    def validate_eps_values(cls, value: list[float] | None) -> list[float] | None:
        """Validate the optional epsilon sweep values.

        Args:
            cls: Model class.
            value: Optional epsilon value list.

        Returns:
            Validated epsilon value list, or `None`.
        """

        if value is None:
            return value
        if not value:
            raise ValueError("eps_values cannot be empty when provided.")
        if any(item <= 0.0 for item in value):
            raise ValueError("eps_values must contain positive values.")
        return value

    @field_validator("max_iterations")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        """Validate a strictly positive iteration limit.

        Args:
            cls: Model class.
            value: Candidate iteration count.

        Returns:
            Validated iteration count.
        """

        if value <= 0:
            raise ValueError("max_iterations must be positive.")
        return value


class SharedWarmStartConfig(StrictModel):
    """Optional shared warm start broadcast to all agents."""

    position_m: list[float]
    emission_time_s: float

    @field_validator("position_m")
    @classmethod
    def validate_position_m(cls, value: list[float]) -> list[float]:
        """Validate a warm-start position vector.

        Args:
            cls: Model class.
            value: Candidate 2D position.

        Returns:
            Validated position vector.
        """

        if len(value) != 2:
            raise ValueError("warm_start.position_m must be a 2D coordinate.")
        return value


class CentralizedConfig(StrictModel):
    """Centralized ToA solver settings."""

    ipopt_max_iter: int = 500
    ipopt_tol: float = 1.0e-8
    ipopt_print_level: int = 0

    @field_validator("ipopt_max_iter", "ipopt_print_level")
    @classmethod
    def validate_nonnegative_int(cls, value: int) -> int:
        """Validate a non-negative IPOPT integer parameter.

        Args:
            cls: Model class.
            value: Candidate integer value.

        Returns:
            Validated integer value.
        """

        if value < 0:
            raise ValueError("Expected a non-negative integer.")
        return value

    @field_validator("ipopt_tol")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Validate a strictly positive IPOPT tolerance.

        Args:
            cls: Model class.
            value: Candidate tolerance.

        Returns:
            Validated tolerance.
        """

        if value <= 0.0:
            raise ValueError("Expected a positive float.")
        return value


class CrlbConfig(StrictModel):
    """CRLB and convergence-related constants."""

    epsilon_radius_m: float = DEFAULT_EPSILON_RADIUS_M
    confidence_k_sigma: float = DEFAULT_CONFIDENCE_K_SIGMA

    @field_validator("epsilon_radius_m", "confidence_k_sigma")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Validate a strictly positive CRLB parameter.

        Args:
            cls: Model class.
            value: Candidate scalar value.

        Returns:
            Validated scalar value.
        """

        if value <= 0.0:
            raise ValueError("Expected a positive value.")
        return value


class ExperimentBaseConfig(StrictModel):
    """Shared fields for all experiment families."""

    experiment_type: str
    output_root: str = DEFAULT_OUTPUT_ROOT
    master_seed: int = DEFAULT_MASTER_SEED
    network: NetworkConfig
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)


class Experiment1Config(ExperimentBaseConfig):
    """Random-target DADMM convergence experiment."""

    experiment_type: Literal["exp1_random_convergence"]
    region_box: BoxRegion
    num_targets: int
    sigma_s: float
    dadmm: DadmmConfig

    @field_validator("num_targets")
    @classmethod
    def validate_num_targets(cls, value: int) -> int:
        """Validate the number of random targets.

        Args:
            cls: Model class.
            value: Candidate target count.

        Returns:
            Validated target count.
        """

        if value <= 0:
            raise ValueError("num_targets must be positive.")
        return value

    @field_validator("sigma_s")
    @classmethod
    def validate_sigma_s(cls, value: float) -> float:
        """Validate a non-negative timing-noise standard deviation.

        Args:
            cls: Model class.
            value: Candidate standard deviation.

        Returns:
            Validated standard deviation.
        """

        if value < 0.0:
            raise ValueError("sigma_s must be non-negative.")
        return value


class Experiment0Config(ExperimentBaseConfig):
    """Single-run DADMM sanity check with fully specified parameters."""

    experiment_type: Literal["exp0_dadmm_quick_check"]
    target_position: list[float]
    sigma_s: float
    dadmm: DadmmConfig

    @field_validator("target_position")
    @classmethod
    def validate_target_position(cls, value: list[float]) -> list[float]:
        """Validate a configured target position.

        Args:
            cls: Model class.
            value: Candidate 2D target position.

        Returns:
            Validated target position.
        """

        if len(value) != 2:
            raise ValueError("target_position must be a 2D coordinate.")
        return value

    @field_validator("sigma_s")
    @classmethod
    def validate_sigma_s(cls, value: float) -> float:
        """Validate a non-negative timing-noise standard deviation.

        Args:
            cls: Model class.
            value: Candidate standard deviation.

        Returns:
            Validated standard deviation.
        """

        if value < 0.0:
            raise ValueError("sigma_s must be non-negative.")
        return value


class Experiment2Config(ExperimentBaseConfig):
    """Sigma sweep against centralized baseline and CRLB."""

    experiment_type: Literal["exp2_sigma_sweep"]
    target_position: list[float]
    sigma_values: list[float]
    num_trials: int
    dadmm: DadmmConfig
    centralized: CentralizedConfig
    crlb: CrlbConfig = Field(default_factory=CrlbConfig)

    @field_validator("target_position")
    @classmethod
    def validate_target_position(cls, value: list[float]) -> list[float]:
        """Validate a configured target position.

        Args:
            cls: Model class.
            value: Candidate 2D target position.

        Returns:
            Validated target position.
        """

        if len(value) != 2:
            raise ValueError("target_position must be a 2D coordinate.")
        return value

    @field_validator("sigma_values")
    @classmethod
    def validate_sigma_values(cls, value: list[float]) -> list[float]:
        """Validate the sigma sweep values.

        Args:
            cls: Model class.
            value: Candidate sigma value list.

        Returns:
            Validated sigma value list.
        """

        if not value:
            raise ValueError("sigma_values cannot be empty.")
        if any(item <= 0.0 for item in value):
            raise ValueError("sigma_values must contain positive values.")
        return value

    @field_validator("num_trials")
    @classmethod
    def validate_num_trials(cls, value: int) -> int:
        """Validate the Monte Carlo trial count.

        Args:
            cls: Model class.
            value: Candidate trial count.

        Returns:
            Validated trial count.
        """

        if value <= 0:
            raise ValueError("num_trials must be positive.")
        return value


class Experiment3Config(ExperimentBaseConfig):
    """Penalty sweep for DADMM."""

    experiment_type: Literal["exp3_penalty_sweep"]
    target_position: list[float]
    sigma_s: float
    rho_p_values: list[float]
    rho_t_values: list[float]
    num_trials: int
    observed_agent_id: int
    dadmm: DadmmConfig
    crlb: CrlbConfig = Field(default_factory=CrlbConfig)

    @field_validator("target_position")
    @classmethod
    def validate_target_position(cls, value: list[float]) -> list[float]:
        """Validate a configured target position.

        Args:
            cls: Model class.
            value: Candidate 2D target position.

        Returns:
            Validated target position.
        """

        if len(value) != 2:
            raise ValueError("target_position must be a 2D coordinate.")
        return value

    @field_validator("sigma_s")
    @classmethod
    def validate_sigma_s(cls, value: float) -> float:
        """Validate a non-negative timing-noise standard deviation.

        Args:
            cls: Model class.
            value: Candidate standard deviation.

        Returns:
            Validated standard deviation.
        """

        if value < 0.0:
            raise ValueError("sigma_s must be non-negative.")
        return value

    @field_validator("rho_p_values", "rho_t_values")
    @classmethod
    def validate_penalty_values(cls, value: list[float]) -> list[float]:
        """Validate the penalty sweep values.

        Args:
            cls: Model class.
            value: Candidate penalty value list.

        Returns:
            Validated penalty value list.
        """

        if not value:
            raise ValueError("Penalty value lists cannot be empty.")
        if any(item <= 0.0 for item in value):
            raise ValueError("Penalty values must be positive.")
        return value

    @field_validator("num_trials")
    @classmethod
    def validate_num_trials(cls, value: int) -> int:
        """Validate the Monte Carlo trial count.

        Args:
            cls: Model class.
            value: Candidate trial count.

        Returns:
            Validated trial count.
        """

        if value <= 0:
            raise ValueError("num_trials must be positive.")
        return value

    @field_validator("observed_agent_id")
    @classmethod
    def validate_observed_agent_id(cls, value: int) -> int:
        """Validate the observed agent index.

        Args:
            cls: Model class.
            value: Candidate observed-agent index.

        Returns:
            Validated observed-agent index.
        """

        if value < 0:
            raise ValueError("observed_agent_id must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_observed_agent_range(self) -> "Experiment3Config":
        """Validate that the observed agent index is in range for the network.

        Args:
            self: Parsed experiment 3 config model.

        Returns:
            The validated experiment 3 config instance.
        """

        if self.observed_agent_id >= len(self.network.sensor_positions):
            raise ValueError("observed_agent_id is out of bounds for the network.")
        return self


ExperimentConfig: TypeAlias = (
    Experiment0Config | Experiment1Config | Experiment2Config | Experiment3Config
)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML configuration.

    Args:
        path: Path to the YAML experiment config.

    Returns:
        Validated experiment configuration model.

    Raises:
        TypeError: If the YAML root is not a mapping.
        ValueError: If `experiment_type` is unsupported.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise TypeError("Expected the YAML root to be a mapping.")

    experiment_type = payload.get("experiment_type")
    if experiment_type == "exp0_dadmm_quick_check":
        return Experiment0Config.model_validate(payload)
    if experiment_type == "exp1_random_convergence":
        return Experiment1Config.model_validate(payload)
    if experiment_type == "exp2_sigma_sweep":
        return Experiment2Config.model_validate(payload)
    if experiment_type == "exp3_penalty_sweep":
        return Experiment3Config.model_validate(payload)
    raise ValueError(f"Unsupported experiment_type: {experiment_type!r}")
