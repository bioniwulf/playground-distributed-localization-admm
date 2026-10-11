"""Result visualization helpers for experiment run directories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def resolve_run_directory(path: str | Path) -> Path:
    """Resolve a run directory from a direct path or experiment directory.

    Args:
        path: Direct run directory path or experiment directory containing runs.

    Returns:
        Resolved run directory path.

    Raises:
        FileNotFoundError: If the path does not exist or no valid run is found.
        NotADirectoryError: If the provided path is not a directory.
    """

    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got: {candidate}")

    if (candidate / "config.yaml").exists():
        return candidate

    run_dirs = sorted(
        subdir for subdir in candidate.iterdir() if subdir.is_dir() and (subdir / "config.yaml").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"Could not find a run directory with config.yaml under: {candidate}")
    return run_dirs[-1]


def load_run_config_payload(run_dir: Path) -> dict[str, Any]:
    """Load the serialized config payload from a run directory.

    Args:
        run_dir: Run directory containing `config.yaml`.

    Returns:
        Parsed config payload.

    Raises:
        TypeError: If `config.yaml` does not contain a mapping payload.
    """

    config_path = run_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload in {config_path}.")
    return payload


def detect_experiment_type(run_dir: Path) -> str:
    """Read the experiment type from a run directory config.

    Args:
        run_dir: Run directory containing `config.yaml`.

    Returns:
        Experiment type string.

    Raises:
        ValueError: If `experiment_type` is missing or invalid.
    """

    payload = load_run_config_payload(run_dir)
    experiment_type = payload.get("experiment_type")
    if not isinstance(experiment_type, str):
        raise ValueError("Missing or invalid experiment_type in config.yaml.")
    return experiment_type


def visualize_results(
    run_dir: str | Path,
) -> list[Path]:
    """Generate figures for a completed experiment run.

    Args:
        run_dir: Run directory path or experiment directory containing runs.

    Returns:
        List of generated figure paths.

    Raises:
        NotImplementedError: If the experiment type is unsupported.
    """

    resolved_run_dir = resolve_run_directory(run_dir)
    payload = load_run_config_payload(resolved_run_dir)
    experiment_type = payload.get("experiment_type")
    if experiment_type == "exp0_dadmm_quick_check":
        return _visualize_exp0(
            run_dir=resolved_run_dir,
            config_payload=payload,
        )
    if experiment_type == "exp1_random_convergence":
        return _visualize_exp1(
            run_dir=resolved_run_dir,
            config_payload=payload,
        )
    if experiment_type == "exp2_sigma_sweep":
        return _visualize_exp2(run_dir=resolved_run_dir)
    if experiment_type == "exp3_penalty_sweep":
        return _visualize_exp3(
            run_dir=resolved_run_dir,
            config_payload=payload,
        )
    raise NotImplementedError(
        f"No visualization is implemented yet for experiment_type={experiment_type!r}."
    )


def _configure_plot_style(mpl) -> None:  # type: ignore[no-untyped-def]
    """Apply the shared matplotlib style used by all figures.

    Args:
        mpl: Imported `matplotlib` module.
    """

    use_latex = shutil.which("latex") is not None
    if use_latex:
        mpl.rcParams.update(
            {
                "text.usetex": False,
                "mathtext.fontset": "stix",
                "font.family": "STIXGeneral",
                # "text.usetex": True,
                # "font.family": "serif",
                # "font.serif": ["Latin Modern Roman"],
                # "axes.unicode_minus": False,
                # "text.latex.preamble": r"\usepackage{lmodern,amsmath,bm}",
            }
        )
    else:
        mpl.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["Latin Modern Roman"],
                "axes.unicode_minus": False,
            }
        )
    mpl.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 28,
            "axes.titlesize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
        }
    )


def _prepare_matplotlib_environment() -> None:
    """Ensure the matplotlib cache is stored in a writable location."""

    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path("/tmp/dadmm_localization_mpl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def _build_exp0_trace_arrays(trace_df, num_agents: int):  # type: ignore[no-untyped-def]
    """Convert exp0 iteration trace rows into dense NumPy arrays.

    Args:
        trace_df: Dataframe loaded from `iteration_trace.parquet`.
        num_agents: Number of receivers in the network.

    Returns:
        Tuple `(iterations, positions, times)` for the exp0 trace.

    Raises:
        ValueError: If iteration-agent records are missing.
    """

    iterations = np.sort(trace_df["iteration"].unique())
    iter_to_index = {int(iteration): idx for idx, iteration in enumerate(iterations)}
    positions = np.full((len(iterations), num_agents, 2), np.nan, dtype=float)
    times = np.full((len(iterations), num_agents), np.nan, dtype=float)

    for row in trace_df.itertuples(index=False):
        iteration = int(row.iteration)
        current_agent = int(row.agent_id)
        positions[iter_to_index[iteration], current_agent, 0] = float(row.x_m)
        positions[iter_to_index[iteration], current_agent, 1] = float(row.y_m)
        times[iter_to_index[iteration], current_agent] = float(row.t_s)

    if np.isnan(positions).any() or np.isnan(times).any():
        raise ValueError("iteration_trace.parquet has missing iteration-agent records.")
    return iterations, positions, times


def _plot_position_convergence(
    plt,  # type: ignore[no-untyped-def]
    iterations: np.ndarray,
    estimated_positions: np.ndarray,
    true_position: np.ndarray,
    output_path: Path,
) -> None:
    """Plot exp0 position convergence curves and save the figure.

    Args:
        plt: Imported `matplotlib.pyplot` module.
        iterations: Iteration indices.
        estimated_positions: Per-iteration per-agent position estimates.
        true_position: Reference target position.
        output_path: Output PNG path.
    """

    est = np.transpose(estimated_positions, (1, 0, 2))
    ref = np.transpose(np.broadcast_to(true_position, estimated_positions.shape), (1, 0, 2))
    diff = est - ref
    num = np.linalg.norm(diff, axis=2)
    den = np.linalg.norm(ref, axis=2)
    rel_error = 100.0 * num / np.maximum(den, 1.0e-12)

    fig, axis = plt.subplots(1, 1, figsize=(16, 8))
    # axis.set_title("Solution Convergence for Position")
    axis.set_xlabel("ADMM iteration $k$")
    axis.set_ylabel(r"$100 \left\| \hat{p}_j^k - p^\star \right\|_2 / \|p^\star\|_2$")
    axis.grid(True, linestyle="-", alpha=1.0, linewidth=1.2)
    axis.tick_params(width=0.1)

    for current_agent in range(est.shape[0]):
        axis.plot(iterations, rel_error[current_agent], "-o", label=f"Receiver {1 + current_agent}")
    axis.legend(loc="upper right")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_time_convergence(
    plt,  # type: ignore[no-untyped-def]
    iterations: np.ndarray,
    estimated_times: np.ndarray,
    true_time_s: float,
    output_path: Path,
) -> None:
    """Plot exp0 emission-time convergence curves and save the figure.

    Args:
        plt: Imported `matplotlib.pyplot` module.
        iterations: Iteration indices.
        estimated_times: Per-iteration per-agent time estimates.
        true_time_s: Reference emission time.
        output_path: Output PNG path.
    """

    est = estimated_times.T
    ref = np.full(est.shape, true_time_s, dtype=float)
    num = np.abs(est - ref)
    den = np.abs(ref)
    rel_error = 100.0 * num / np.maximum(den, 1.0e-12)

    fig, axis = plt.subplots(1, 1, figsize=(16, 8))
    # axis.set_title("Solution Convergence for ToA")
    axis.set_xlabel("Iteration $k$")
    axis.set_ylabel(r"$100 (\hat{t}_j^k - t^\star) / t^\star$")
    axis.grid(True, linestyle="-", alpha=1.0, linewidth=1.2)
    axis.tick_params(width=0.1)

    for current_agent in range(est.shape[0]):
        axis.plot(iterations, rel_error[current_agent], "-o", label=f"Receiver {1 + current_agent}")
    axis.legend(loc="upper right")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_field(
    plt,  # type: ignore[no-untyped-def]
    sensor_positions_m: np.ndarray,
    adjacency_matrix: np.ndarray,
    true_position_m: np.ndarray,
    estimated_position_m: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the receiver graph and one estimated target position.

    Args:
        plt: Imported `matplotlib.pyplot` module.
        sensor_positions_m: Receiver positions.
        adjacency_matrix: Receiver connectivity matrix.
        true_position_m: True target position.
        estimated_position_m: Estimated target position to overlay.
        output_path: Output PNG path.
    """

    fig = plt.figure(figsize=(16, 8))

    link_label_is_set = False
    for source_id in range(adjacency_matrix.shape[0]):
        for target_id in range(source_id + 1, adjacency_matrix.shape[1]):
            is_connected = adjacency_matrix[source_id, target_id] != 0 or adjacency_matrix[target_id, source_id] != 0
            if not is_connected:
                continue
            line_label = "Communication links" if not link_label_is_set else None
            plt.plot(
                [sensor_positions_m[source_id, 0], sensor_positions_m[target_id, 0]],
                [sensor_positions_m[source_id, 1], sensor_positions_m[target_id, 1]],
                linestyle="--",
                linewidth=1.2,
                color="tab:gray",
                alpha=0.9,
                zorder=2,
                label=line_label,
            )
            link_label_is_set = True

    plt.scatter(
        sensor_positions_m[:, 0],
        sensor_positions_m[:, 1],
        marker="^",
        s=120,
        c="black",
        label="Receivers",
        zorder=3,
    )
    plt.scatter(
        true_position_m[0],
        true_position_m[1],
        c="tab:blue",
        s=80,
        label="True Position",
        zorder=4,
    )
    plt.scatter(
        estimated_position_m[0],
        estimated_position_m[1],
        c="tab:red",
        marker="*",
        s=90,
        label="Estimated Position",
        zorder=5,
    )

    plt.grid(True, linestyle="-", alpha=1.0, linewidth=1.2)
    plt.tick_params(width=0.1)
    plt.xlabel(r"$x$, m")
    plt.ylabel(r"$y$, m")
    plt.axis("equal")
    plt.legend()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _iterations_to_marker_sizes(mean_iterations: np.ndarray) -> np.ndarray:
    """Convert mean iteration counts to scatter marker areas.

    Args:
        mean_iterations: Mean convergence iterations per trial.

    Returns:
        Scatter marker areas in points squared.
    """

    if mean_iterations.size == 0:
        return np.asarray([], dtype=float)
    min_radius_pts = 8.0
    max_radius_pts = 28.0
    iter_min = float(np.min(mean_iterations))
    iter_max = float(np.max(mean_iterations))
    if np.isclose(iter_min, iter_max):
        radii_pts = np.full_like(mean_iterations, 0.5 * (min_radius_pts + max_radius_pts), dtype=float)
    else:
        radii_pts = min_radius_pts + (mean_iterations - iter_min) * (max_radius_pts - min_radius_pts) / (
            iter_max - iter_min
        )
    return radii_pts**2


def _plot_exp1_field(
    plt,  # type: ignore[no-untyped-def]
    sensor_positions_m: np.ndarray,
    adjacency_matrix: np.ndarray,
    true_positions_m: np.ndarray,
    estimated_positions_m: np.ndarray,
    marker_sizes: np.ndarray,
    output_path: Path,
) -> None:
    """Plot exp1 true and estimated target positions over the receiver graph.

    Args:
        plt: Imported `matplotlib.pyplot` module.
        sensor_positions_m: Receiver positions.
        adjacency_matrix: Receiver connectivity matrix.
        true_positions_m: True target positions for all trials.
        estimated_positions_m: Estimated target positions for all trials.
        marker_sizes: Marker sizes derived from convergence iterations.
        output_path: Output PNG path.
    """

    fig = plt.figure(figsize=(16, 8))

    link_label_is_set = False
    for source_id in range(adjacency_matrix.shape[0]):
        for target_id in range(source_id + 1, adjacency_matrix.shape[1]):
            is_connected = adjacency_matrix[source_id, target_id] != 0 or adjacency_matrix[target_id, source_id] != 0
            if not is_connected:
                continue
            line_label = "Communication links" if not link_label_is_set else None
            plt.plot(
                [sensor_positions_m[source_id, 0], sensor_positions_m[target_id, 0]],
                [sensor_positions_m[source_id, 1], sensor_positions_m[target_id, 1]],
                linestyle="--",
                linewidth=1.2,
                color="tab:gray",
                alpha=0.9,
                zorder=1,
                label=line_label,
            )
            link_label_is_set = True

    plt.scatter(
        sensor_positions_m[:, 0],
        sensor_positions_m[:, 1],
        marker="^",
        s=120,
        c="black",
        label="Receivers",
        zorder=2,
    )
    plt.scatter(
        true_positions_m[:, 0],
        true_positions_m[:, 1],
        marker="x",
        s=90,
        c="tab:blue",
        linewidths=1.5,
        label="True positions",
        zorder=4,
    )
    plt.scatter(
        estimated_positions_m[:, 0],
        estimated_positions_m[:, 1],
        marker="o",
        s=marker_sizes,
        c="tab:red",
        alpha=0.55,
        edgecolors="black",
        linewidths=0.5,
        label="Iterations number",
        zorder=4,
    )
    plt.scatter(
        estimated_positions_m[:, 0],
        estimated_positions_m[:, 1],
        marker="o",
        s=15,
        c="black",
        alpha=1.0,
        edgecolors="black",
        linewidths=0.5,
        label="Estimated positions",
        zorder=3,
    )

    plt.grid(True, linestyle="-", alpha=1.0, linewidth=1.0)
    plt.tick_params(width=0.1)
    plt.xlabel(r"$x$, m")
    plt.ylabel(r"$y$, m")
    plt.axis("equal")
    plt.legend(loc="upper right")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _compute_exp2_rmse_from_position_residuals(run_df):  # type: ignore[no-untyped-def]
    """Compute exp2 RMSE curves and confidence intervals from run-level data.

    Args:
        run_df: Run-level experiment 2 dataframe.

    Returns:
        Dataframe with per-sigma RMSE summaries and confidence intervals.

    Raises:
        ValueError: If required run-level columns are missing.
    """

    required_columns = {"sigma_s", "algorithm", "true_x_m", "true_y_m", "est_x_m", "est_y_m"}
    if not required_columns.issubset(run_df.columns):
        raise ValueError("run_metrics.parquet is missing required columns for exp2 visualization.")

    working_df = run_df.loc[:, list(required_columns)].copy()
    group_columns = ["sigma_s", "algorithm"]
    if "eps_value" in run_df.columns:
        working_df["eps_value"] = run_df["eps_value"]
        group_columns.append("eps_value")
    working_df["residual_sq_m2"] = (working_df["est_x_m"] - working_df["true_x_m"]) ** 2 + (
        working_df["est_y_m"] - working_df["true_y_m"]
    ) ** 2
    rmse_df = working_df.groupby(group_columns, as_index=False, dropna=False).agg(
        mean_residual_sq_m2=("residual_sq_m2", "mean"),
        std_residual_sq_m2=("residual_sq_m2", "std"),
        sample_count=("residual_sq_m2", "count"),
    )
    rmse_df["std_residual_sq_m2"] = rmse_df["std_residual_sq_m2"].fillna(0.0)
    rmse_df["sample_count"] = rmse_df["sample_count"].astype(float)
    z_95 = 1.959963984540054
    rmse_df["se_mean_residual_sq_m2"] = rmse_df["std_residual_sq_m2"] / np.sqrt(rmse_df["sample_count"])
    rmse_df["mean_residual_sq_ci_low_m2"] = np.maximum(
        rmse_df["mean_residual_sq_m2"] - z_95 * rmse_df["se_mean_residual_sq_m2"],
        0.0,
    )
    rmse_df["mean_residual_sq_ci_high_m2"] = np.maximum(
        rmse_df["mean_residual_sq_m2"] + z_95 * rmse_df["se_mean_residual_sq_m2"],
        0.0,
    )
    rmse_df["position_std_m"] = np.sqrt(rmse_df["mean_residual_sq_m2"])
    rmse_df["position_std_ci_low_m"] = np.sqrt(rmse_df["mean_residual_sq_ci_low_m2"])
    rmse_df["position_std_ci_high_m"] = np.sqrt(rmse_df["mean_residual_sq_ci_high_m2"])
    rmse_df["sigma2_db"] = 10.0 * np.log10(np.square(1 / rmse_df["sigma_s"].to_numpy(dtype=float)))
    return rmse_df


def _compute_nonnegative_95_ci(
    mean_values: np.ndarray,
    std_values: np.ndarray,
    sample_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normal-approximation 95% CI bounds clipped to non-negative values.

    Args:
        mean_values: Mean values of the metric.
        std_values: Standard deviations of the metric.
        sample_counts: Sample counts per estimate.

    Returns:
        Tuple `(lower, upper)` of confidence-interval bounds.
    """

    z_95 = 1.959963984540054
    safe_counts = np.maximum(sample_counts.astype(float), 1.0)
    safe_std = np.nan_to_num(std_values.astype(float), nan=0.0)
    standard_error = safe_std / np.sqrt(safe_counts)
    lower = np.maximum(mean_values.astype(float) - z_95 * standard_error, 0.0)
    upper = np.maximum(mean_values.astype(float) + z_95 * standard_error, 0.0)
    return lower, upper


def _compute_exp3_trace_statistics(
    trace_df,  # type: ignore[no-untyped-def]
    *,
    sweep_column: str,
    fixed_column: str,
    fixed_value: float,
    metric_column: str,
    reference_value: float,
):
    """Aggregate exp3 convergence traces for one sweep at a fixed companion penalty.

    Args:
        trace_df: Observed-agent trace dataframe.
        sweep_column: Penalty column varied on the plot.
        fixed_column: Companion penalty column held fixed.
        fixed_value: Fixed companion-penalty value.
        metric_column: Error metric column to aggregate.
        reference_value: Reference value used to convert errors to percentages.

    Returns:
        Aggregated statistics dataframe for the selected sweep.

    Raises:
        ValueError: If required columns are missing or the fixed penalty is absent.
    """

    required_columns = {sweep_column, fixed_column, "trial_id", "iteration", metric_column}
    if not required_columns.issubset(trace_df.columns):
        raise ValueError("observed_agent_trace.parquet is missing required columns for exp3 visualization.")

    working_df = trace_df.loc[:, list(required_columns)].copy()
    fixed_mask = np.isclose(
        working_df[fixed_column].to_numpy(dtype=float),
        fixed_value,
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    filtered_df = working_df.loc[fixed_mask].copy()
    if filtered_df.empty:
        available_values = np.sort(working_df[fixed_column].unique())
        available_values_str = ", ".join(f"{float(value):g}" for value in available_values)
        raise ValueError(
            "observed_agent_trace.parquet does not contain the fixed penalty required for exp3 "
            f"visualization: {fixed_column}={fixed_value:g}. Available values: {available_values_str}."
        )

    filtered_df["relative_error_pct"] = 100.0 * filtered_df[metric_column].to_numpy(dtype=float) / max(
        abs(reference_value),
        1.0e-12,
    )
    stats_df = filtered_df.groupby([sweep_column, "iteration"], as_index=False).agg(
        mean_relative_error_pct=("relative_error_pct", "mean"),
        std_relative_error_pct=("relative_error_pct", "std"),
        sample_count=("relative_error_pct", "count"),
    )
    ci_low, ci_high = _compute_nonnegative_95_ci(
        mean_values=stats_df["mean_relative_error_pct"].to_numpy(dtype=float),
        std_values=stats_df["std_relative_error_pct"].to_numpy(dtype=float),
        sample_counts=stats_df["sample_count"].to_numpy(dtype=float),
    )
    stats_df["relative_error_ci_low_pct"] = ci_low
    stats_df["relative_error_ci_high_pct"] = ci_high
    return stats_df.sort_values([sweep_column, "iteration"]).reset_index(drop=True)


def _format_penalty_label(penalty_name: str, penalty_value: float) -> str:
    """Format a legend label for a penalty sweep value.

    Args:
        penalty_name: Penalty parameter name, typically `rho_p` or `rho_t`.
        penalty_value: Penalty value to format.

    Returns:
        LaTeX-formatted legend label.
    """

    symbol = r"\rho_p" if penalty_name == "rho_p" else r"\rho_t"
    if 1.0e-2 <= abs(penalty_value) < 1.0e3:
        value_text = f"{penalty_value:g}"
    else:
        value_text = f"{penalty_value:.1e}"
    return rf"${symbol} = {value_text}$"


def _format_eps_label(eps_value: float) -> str:
    """Format a legend label for a shared feasibility/convergence tolerance.

    Args:
        eps_value: Shared feasibility and convergence tolerance.

    Returns:
        Formatted legend label for the DADMM curve.
    """

    if 1.0e-2 <= abs(eps_value) < 1.0e3:
        value_text = f"{eps_value:g}"
    else:
        value_text = f"{eps_value:.1e}"
    return rf"DADMM (${value_text}$)"


def _plot_exp3_trace_family(
    plt,  # type: ignore[no-untyped-def]
    stats_df,  # type: ignore[no-untyped-def]
    *,
    sweep_column: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot exp3 mean convergence traces with 95% confidence intervals.

    Args:
        plt: Imported `matplotlib.pyplot` module.
        stats_df: Aggregated statistics dataframe.
        sweep_column: Penalty column varied across curves.
        ylabel: Y-axis label.
        output_path: Output PNG path.
    """

    fig, axis = plt.subplots(1, 1, figsize=(16, 8))
    axis.set_xlabel("ADMM iteration $k$")
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="-", alpha=1.0, linewidth=1.2)
    axis.tick_params(width=0.1)

    sweep_values = np.sort(stats_df[sweep_column].unique())
    # Use the same categorical palette for every DADMM family plot.
    color_map = plt.get_cmap("tab10")
    for curve_id, sweep_value in enumerate(sweep_values):
        curve_df = stats_df.loc[stats_df[sweep_column] == sweep_value].copy()
        color = color_map(curve_id)
        iterations = curve_df["iteration"].to_numpy(dtype=float)
        mean_values = curve_df["mean_relative_error_pct"].to_numpy(dtype=float)
        ci_low = curve_df["relative_error_ci_low_pct"].to_numpy(dtype=float)
        ci_high = curve_df["relative_error_ci_high_pct"].to_numpy(dtype=float)

        axis.plot(
            iterations,
            mean_values,
            "-o",
            color=color,
            label=_format_penalty_label(sweep_column, float(sweep_value)),
        )
        axis.fill_between(
            iterations,
            ci_low,
            ci_high,
            color=color,
            alpha=0.16,
        )

    axis.legend(loc="upper right", ncol=2 if len(sweep_values) > 6 else 1)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_exp2_sigma_std(
    plt,  # type: ignore[no-untyped-def]
    dadmm_df,  # type: ignore[no-untyped-def]
    centralized_df,  # type: ignore[no-untyped-def]
    crlb_df,  # type: ignore[no-untyped-def]
    output_path: Path,
) -> None:
    """Plot exp2 position standard deviation versus measurement noise power.

    Args:
        plt: Imported `matplotlib.pyplot` module.
        dadmm_df: Aggregated DADMM dataframe.
        centralized_df: Aggregated centralized-baseline dataframe.
        crlb_df: CRLB dataframe prepared for plotting.
        output_path: Output PNG path.
    """

    fig, axis = plt.subplots(1, 1, figsize=(16, 8))
    # axis.set_title("Position Error Standard Deviation vs Noise Power")
    axis.set_xlabel(r"Measurement Noise $1 / \sigma^2$, dB")
    axis.set_ylabel("RMSE (m)")

    axis.minorticks_on()

    axis.grid(True, which='major', axis='y', linestyle='-', alpha=1.0, linewidth=1.2)
    axis.grid(True, which='minor', axis='y', linestyle=':', alpha=0.6, linewidth=1.0)

    axis.tick_params(width=0.2)

    # set log scale for Y axis 
    axis.set_yscale('log')

    if "eps_value" in dadmm_df.columns and dadmm_df["eps_value"].notna().any():
        eps_values = np.sort(dadmm_df["eps_value"].dropna().unique())
        # Match the categorical palette used by the exp3 DADMM traces.
        color_map = plt.get_cmap("tab10")
        # Reserve tab:green for the CRLB reference curve.
        dadmm_color_indices = (0, 1, 3, 4, 5, 6, 7, 8, 9)
        for curve_id, eps_value in enumerate(eps_values):
            curve_df = dadmm_df[dadmm_df["eps_value"] == eps_value].sort_values("sigma_s")
            color = color_map(dadmm_color_indices[curve_id])
            axis.plot(
                curve_df["sigma2_db"],
                curve_df["position_std_m"],
                "-o",
                color=color,
                label=_format_eps_label(float(eps_value)),
            )
            axis.fill_between(
                curve_df["sigma2_db"],
                curve_df["position_std_ci_low_m"],
                curve_df["position_std_ci_high_m"],
                color=color,
                alpha=0.14,
            )
    else:
        axis.plot(
            dadmm_df["sigma2_db"],
            dadmm_df["position_std_m"],
            "-o",
            color="tab:red",
            label="DADMM",
        )
        axis.fill_between(
            dadmm_df["sigma2_db"],
            dadmm_df["position_std_ci_low_m"],
            dadmm_df["position_std_ci_high_m"],
            color="tab:red",
            alpha=0.18,
            label="DADMM 95% CI",
        )
    axis.plot(
        centralized_df["sigma2_db"],
        centralized_df["position_std_m"],
        "-^",
        color="black",
        label="Centralized",
    )
    axis.fill_between(
        centralized_df["sigma2_db"],
        centralized_df["position_std_ci_low_m"],
        centralized_df["position_std_ci_high_m"],
        color="black",
        alpha=0.16,
        label="Centralized 95% CI",
    )
    axis.plot(
        crlb_df["sigma2_db"],
        crlb_df["position_std_m"],
        "-^",
        color="tab:green",
        label="CRLB",
    )

    axis.legend(loc="lower left")
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _visualize_exp0(
    run_dir: Path,
    config_payload: dict[str, Any],
) -> list[Path]:
    """Generate all exp0 figures for a completed run.

    Args:
        run_dir: Experiment run directory.
        config_payload: Parsed config payload for the run.

    Returns:
        List of generated figure paths.
    """
    _prepare_matplotlib_environment()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required for visualization.") from exc

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("matplotlib is required for visualization.") from exc

    _configure_plot_style(mpl)
    run_dir.mkdir(parents=True, exist_ok=True)

    network_payload = config_payload.get("network")
    if not isinstance(network_payload, dict):
        raise TypeError("config.yaml is missing a valid 'network' mapping.")
    sensor_positions = np.asarray(network_payload.get("sensor_positions"), dtype=float)
    adjacency_matrix = np.asarray(network_payload.get("adjacency_matrix"), dtype=int)
    true_position = np.asarray(config_payload.get("target_position"), dtype=float)
    true_time_s = float(config_payload.get("physics", {}).get("emission_time_s", 0.0))

    trace_df = pd.read_parquet(run_dir / "iteration_trace.parquet")
    iterations, estimated_positions, estimated_times = _build_exp0_trace_arrays(
        trace_df=trace_df,
        num_agents=sensor_positions.shape[0],
    )
    estimated_agent_position = estimated_positions[-1, 0]

    position_path = run_dir / "convergence_position.png"
    time_path = run_dir / "convergence_time.png"
    field_path = run_dir / "field.png"

    _plot_position_convergence(
        plt=plt,
        iterations=iterations,
        estimated_positions=estimated_positions,
        true_position=true_position,
        output_path=position_path,
    )
    _plot_time_convergence(
        plt=plt,
        iterations=iterations,
        estimated_times=estimated_times,
        true_time_s=true_time_s,
        output_path=time_path,
    )
    _plot_field(
        plt=plt,
        sensor_positions_m=sensor_positions,
        adjacency_matrix=adjacency_matrix,
        true_position_m=true_position,
        estimated_position_m=estimated_agent_position,
        output_path=field_path,
    )
    return [position_path, time_path, field_path]


def _visualize_exp1(
    run_dir: Path,
    config_payload: dict[str, Any],
) -> list[Path]:
    """Generate all exp1 figures for a completed run.

    Args:
        run_dir: Experiment run directory.
        config_payload: Parsed config payload for the run.

    Returns:
        List of generated figure paths.
    """
    _prepare_matplotlib_environment()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required for visualization.") from exc

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("matplotlib is required for visualization.") from exc

    _configure_plot_style(mpl)
    run_dir.mkdir(parents=True, exist_ok=True)

    network_payload = config_payload.get("network")
    if not isinstance(network_payload, dict):
        raise TypeError("config.yaml is missing a valid 'network' mapping.")
    sensor_positions = np.asarray(network_payload.get("sensor_positions"), dtype=float)
    adjacency_matrix = np.asarray(network_payload.get("adjacency_matrix"), dtype=int)

    trial_df = pd.read_parquet(run_dir / "trial_summary.parquet")
    agent_df = pd.read_parquet(run_dir / "agent_summary.parquet")

    required_trial_columns = {"trial_id", "true_x_m", "true_y_m", "est_x_m", "est_y_m"}
    required_agent_columns = {"trial_id", "converged_at_iteration"}
    if not required_trial_columns.issubset(trial_df.columns):
        raise ValueError("trial_summary.parquet is missing required columns for exp1 visualization.")
    if not required_agent_columns.issubset(agent_df.columns):
        raise ValueError("agent_summary.parquet is missing required columns for exp1 visualization.")

    mean_iterations = (
        agent_df.groupby("trial_id", as_index=False)["converged_at_iteration"]
        .mean()
        .rename(columns={"converged_at_iteration": "mean_converged_iteration"})
    )
    merged_df = trial_df.merge(mean_iterations, on="trial_id", how="inner")
    if merged_df.empty:
        raise ValueError("Could not match trial_summary.parquet with agent_summary.parquet by trial_id.")

    true_positions = merged_df[["true_x_m", "true_y_m"]].to_numpy(dtype=float)
    estimated_positions = merged_df[["est_x_m", "est_y_m"]].to_numpy(dtype=float)
    marker_sizes = _iterations_to_marker_sizes(
        merged_df["mean_converged_iteration"].to_numpy(dtype=float)
    )

    field_path = run_dir / "field.png"
    _plot_exp1_field(
        plt=plt,
        sensor_positions_m=sensor_positions,
        adjacency_matrix=adjacency_matrix,
        true_positions_m=true_positions,
        estimated_positions_m=estimated_positions,
        marker_sizes=marker_sizes,
        output_path=field_path,
    )
    return [field_path]


def _visualize_exp2(
    run_dir: Path,
) -> list[Path]:
    """Generate all exp2 figures for a completed run.

    Args:
        run_dir: Experiment run directory.

    Returns:
        List of generated figure paths.
    """
    _prepare_matplotlib_environment()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required for visualization.") from exc

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("matplotlib is required for visualization.") from exc

    _configure_plot_style(mpl)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_df = pd.read_parquet(run_dir / "run_metrics.parquet")
    crlb_df = pd.read_parquet(run_dir / "crlb.parquet")
    if not {"sigma_s", "position_rmse_bound_m"}.issubset(crlb_df.columns):
        raise ValueError("crlb.parquet is missing required columns for exp2 visualization.")

    rmse_df = _compute_exp2_rmse_from_position_residuals(run_df)
    dadmm_sort_columns = ["sigma_s"]
    if "eps_value" in rmse_df.columns:
        dadmm_sort_columns = ["eps_value", "sigma_s"]
    dadmm_df = rmse_df[rmse_df["algorithm"] == "dadmm"].sort_values(dadmm_sort_columns)
    centralized_df = rmse_df[rmse_df["algorithm"] == "centralized"].sort_values("sigma_s")
    crlb_plot_df = crlb_df.copy()
    crlb_plot_df["position_std_m"] = crlb_plot_df["position_rmse_bound_m"]
    crlb_plot_df["sigma2_db"] = 10.0 * np.log10(np.square(1 / crlb_plot_df["sigma_s"].to_numpy(dtype=float)))
    crlb_plot_df = crlb_plot_df.sort_values("sigma_s")

    output_path = run_dir / "sigma_position_std.png"
    _plot_exp2_sigma_std(
        plt=plt,
        dadmm_df=dadmm_df,
        centralized_df=centralized_df,
        crlb_df=crlb_plot_df,
        output_path=output_path,
    )
    return [output_path]


def _visualize_exp3(
    run_dir: Path,
    config_payload: dict[str, Any],
) -> list[Path]:
    """Generate all exp3 figures for a completed run.

    Args:
        run_dir: Experiment run directory.
        config_payload: Parsed config payload for the run.

    Returns:
        List of generated figure paths.
    """
    _prepare_matplotlib_environment()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("pandas and pyarrow are required for visualization.") from exc

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise ImportError("matplotlib is required for visualization.") from exc

    _configure_plot_style(mpl)
    run_dir.mkdir(parents=True, exist_ok=True)

    dadmm_payload = config_payload.get("dadmm")
    if not isinstance(dadmm_payload, dict):
        raise TypeError("config.yaml is missing a valid 'dadmm' mapping.")

    true_position = np.asarray(config_payload.get("target_position"), dtype=float)
    true_time_s = float(config_payload.get("physics", {}).get("emission_time_s", 0.0))
    fixed_rho_p = float(dadmm_payload.get("rho_p", 0.0))
    fixed_rho_t = float(dadmm_payload.get("rho_t", 0.0))

    trace_df = pd.read_parquet(run_dir / "observed_agent_trace.parquet")
    position_stats = _compute_exp3_trace_statistics(
        trace_df=trace_df,
        sweep_column="rho_p",
        fixed_column="rho_t",
        fixed_value=fixed_rho_t,
        metric_column="position_error_m",
        reference_value=float(np.linalg.norm(true_position)),
    )
    time_stats = _compute_exp3_trace_statistics(
        trace_df=trace_df,
        sweep_column="rho_t",
        fixed_column="rho_p",
        fixed_value=fixed_rho_p,
        metric_column="time_error_s",
        reference_value=abs(true_time_s),
    )

    position_path = run_dir / "rho_p_position_convergence.png"
    time_path = run_dir / "rho_t_time_convergence.png"

    _plot_exp3_trace_family(
        plt=plt,
        stats_df=position_stats,
        sweep_column="rho_p",
        ylabel=r"$100 \left\| \hat{p}_j^k - p^\star \right\|_2 / \|p^\star\|_2$",
        output_path=position_path,
    )
    _plot_exp3_trace_family(
        plt=plt,
        stats_df=time_stats,
        sweep_column="rho_t",
        ylabel=r"$100 |\hat{t}_j^k - t^\star| / \max(|t^\star|, \varepsilon)$",
        output_path=time_path,
    )
    return [position_path, time_path]
