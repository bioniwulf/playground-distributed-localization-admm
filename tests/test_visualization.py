from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dadmm_localization.cli import main
from dadmm_localization.visualization import (
    _compute_exp3_trace_statistics,
    _compute_exp2_rmse_from_position_residuals,
    _iterations_to_marker_sizes,
    detect_experiment_type,
    resolve_run_directory,
    visualize_results,
)


def _write_config(path: Path, experiment_type: str) -> None:
    payload = {
        "experiment_type": experiment_type,
        "network": {
            "sensor_positions": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            "adjacency_matrix": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        },
        "target_position": [0.5, 0.5],
        "physics": {"emission_time_s": 0.0},
    }
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def test_resolve_run_directory_accepts_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _write_config(run_dir / "config.yaml", "exp0_dadmm_quick_check")

    resolved = resolve_run_directory(run_dir)

    assert resolved == run_dir.resolve()


def test_resolve_run_directory_accepts_experiment_dir_and_uses_latest(tmp_path: Path) -> None:
    exp_dir = tmp_path / "results" / "exp0_dadmm_quick_check"
    run_old = exp_dir / "20260306T100000_aaaaa"
    run_new = exp_dir / "20260306T200000_bbbbb"
    run_old.mkdir(parents=True)
    run_new.mkdir(parents=True)
    _write_config(run_old / "config.yaml", "exp0_dadmm_quick_check")
    _write_config(run_new / "config.yaml", "exp0_dadmm_quick_check")

    resolved = resolve_run_directory(exp_dir)

    assert resolved == run_new.resolve()


def test_detect_experiment_type(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _write_config(run_dir / "config.yaml", "exp0_dadmm_quick_check")

    experiment_type = detect_experiment_type(run_dir)

    assert experiment_type == "exp0_dadmm_quick_check"


def test_visualize_results_dispatches_exp3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir_path = tmp_path / "run"
    run_dir_path.mkdir(parents=True)
    _write_config(run_dir_path / "config.yaml", "exp3_penalty_sweep")

    expected_path = run_dir_path / "rho_p_position_convergence.png"

    def _fake_visualize_exp3(
        run_dir: Path,
        config_payload: dict[str, object],
    ) -> list[Path]:
        assert run_dir == run_dir_path.resolve()
        assert config_payload["experiment_type"] == "exp3_penalty_sweep"
        return [expected_path]

    monkeypatch.setattr("dadmm_localization.visualization._visualize_exp3", _fake_visualize_exp3)

    output_paths = visualize_results(run_dir_path)

    assert output_paths == [expected_path]


def test_cli_visualize_invokes_visualizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def _fake_visualize_results(run_dir: Path) -> list[Path]:
        captured["run_dir"] = run_dir
        return [run_dir / "convergence_position.png"]

    monkeypatch.setattr("dadmm_localization.cli.visualize_results", _fake_visualize_results)

    exit_code = main(["visualize", "--run-dir", str(run_dir)])

    assert exit_code == 0
    assert captured == {
        "run_dir": run_dir,
    }


def test_visualize_results_dispatches_exp1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir_path = tmp_path / "run"
    run_dir_path.mkdir(parents=True)
    _write_config(run_dir_path / "config.yaml", "exp1_random_convergence")

    expected_path = run_dir_path / "field.png"

    def _fake_visualize_exp1(
        run_dir: Path,
        config_payload: dict[str, object],
    ) -> list[Path]:
        assert run_dir == run_dir_path.resolve()
        assert config_payload["experiment_type"] == "exp1_random_convergence"
        return [expected_path]

    monkeypatch.setattr("dadmm_localization.visualization._visualize_exp1", _fake_visualize_exp1)

    output_paths = visualize_results(run_dir_path)

    assert output_paths == [expected_path]


def test_visualize_results_dispatches_exp2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir_path = tmp_path / "run"
    run_dir_path.mkdir(parents=True)
    _write_config(run_dir_path / "config.yaml", "exp2_sigma_sweep")

    expected_path = run_dir_path / "sigma_position_std.png"

    def _fake_visualize_exp2(run_dir: Path) -> list[Path]:
        assert run_dir == run_dir_path.resolve()
        return [expected_path]

    monkeypatch.setattr("dadmm_localization.visualization._visualize_exp2", _fake_visualize_exp2)

    output_paths = visualize_results(run_dir_path)

    assert output_paths == [expected_path]


def test_iterations_to_marker_sizes_are_monotonic() -> None:
    sizes = _iterations_to_marker_sizes(np.asarray([10.0, 20.0, 30.0], dtype=float))
    assert sizes[0] < sizes[1] < sizes[2]


def test_iterations_to_marker_sizes_constant_case() -> None:
    sizes = _iterations_to_marker_sizes(np.asarray([50.0, 50.0, 50.0], dtype=float))
    assert np.allclose(sizes, sizes[0])


def test_compute_exp2_rmse_from_position_residuals() -> None:
    run_df = pd.DataFrame(
        [
            {
                "sigma_s": 1.0e-3,
                "algorithm": "dadmm",
                "eps_value": 1.0e-4,
                "true_x_m": 0.0,
                "true_y_m": 0.0,
                "est_x_m": 3.0,
                "est_y_m": 4.0,
            },
            {
                "sigma_s": 1.0e-3,
                "algorithm": "dadmm",
                "eps_value": 1.0e-4,
                "true_x_m": 0.0,
                "true_y_m": 0.0,
                "est_x_m": 0.0,
                "est_y_m": 0.0,
            },
            {
                "sigma_s": 1.0e-3,
                "algorithm": "dadmm",
                "eps_value": 1.0e-5,
                "true_x_m": 0.0,
                "true_y_m": 0.0,
                "est_x_m": 0.0,
                "est_y_m": 3.0,
            },
            {
                "sigma_s": 1.0e-3,
                "algorithm": "centralized",
                "eps_value": np.nan,
                "true_x_m": 0.0,
                "true_y_m": 0.0,
                "est_x_m": 6.0,
                "est_y_m": 8.0,
            },
        ]
    )

    rmse_df = _compute_exp2_rmse_from_position_residuals(run_df)
    dadmm_rmse = rmse_df[
        (rmse_df["algorithm"] == "dadmm") & np.isclose(rmse_df["eps_value"], 1.0e-4)
    ]["position_std_m"].iloc[0]
    dadmm_second_rmse = rmse_df[
        (rmse_df["algorithm"] == "dadmm") & np.isclose(rmse_df["eps_value"], 1.0e-5)
    ]["position_std_m"].iloc[0]
    centralized_rmse = rmse_df[rmse_df["algorithm"] == "centralized"]["position_std_m"].iloc[0]
    dadmm_low = rmse_df[
        (rmse_df["algorithm"] == "dadmm") & np.isclose(rmse_df["eps_value"], 1.0e-4)
    ]["position_std_ci_low_m"].iloc[0]
    dadmm_high = rmse_df[
        (rmse_df["algorithm"] == "dadmm") & np.isclose(rmse_df["eps_value"], 1.0e-4)
    ]["position_std_ci_high_m"].iloc[0]
    centralized_low = rmse_df[rmse_df["algorithm"] == "centralized"]["position_std_ci_low_m"].iloc[0]
    centralized_high = rmse_df[rmse_df["algorithm"] == "centralized"]["position_std_ci_high_m"].iloc[0]

    assert np.isclose(dadmm_rmse, np.sqrt((25.0 + 0.0) / 2.0))
    assert np.isclose(dadmm_second_rmse, 3.0)
    assert np.isclose(centralized_rmse, 10.0)
    assert dadmm_low <= dadmm_rmse <= dadmm_high
    assert centralized_low <= centralized_rmse <= centralized_high


def test_compute_exp3_trace_statistics() -> None:
    trace_df = pd.DataFrame(
        [
            {
                "rho_p": 1.0e-7,
                "rho_t": 20.0,
                "trial_id": 0,
                "iteration": 1,
                "position_error_m": 3.0,
            },
            {
                "rho_p": 1.0e-7,
                "rho_t": 20.0,
                "trial_id": 1,
                "iteration": 1,
                "position_error_m": 5.0,
            },
            {
                "rho_p": 1.0e-7,
                "rho_t": 20.0,
                "trial_id": 0,
                "iteration": 2,
                "position_error_m": 2.0,
            },
            {
                "rho_p": 5.0e-7,
                "rho_t": 20.0,
                "trial_id": 0,
                "iteration": 1,
                "position_error_m": 1.0,
            },
            {
                "rho_p": 5.0e-7,
                "rho_t": 10.0,
                "trial_id": 0,
                "iteration": 1,
                "position_error_m": 9.0,
            },
        ]
    )

    stats_df = _compute_exp3_trace_statistics(
        trace_df=trace_df,
        sweep_column="rho_p",
        fixed_column="rho_t",
        fixed_value=20.0,
        metric_column="position_error_m",
        reference_value=10.0,
    )

    first_curve = stats_df[(stats_df["rho_p"] == 1.0e-7) & (stats_df["iteration"] == 1)]
    assert len(first_curve) == 1
    assert np.isclose(first_curve["mean_relative_error_pct"].iloc[0], 40.0)
    assert np.isclose(first_curve["sample_count"].iloc[0], 2.0)
    assert first_curve["relative_error_ci_low_pct"].iloc[0] <= 40.0
    assert first_curve["relative_error_ci_high_pct"].iloc[0] >= 40.0

    second_curve = stats_df[(stats_df["rho_p"] == 5.0e-7) & (stats_df["iteration"] == 1)]
    assert len(second_curve) == 1
    assert np.isclose(second_curve["mean_relative_error_pct"].iloc[0], 10.0)


def test_compute_exp3_trace_statistics_raises_for_missing_fixed_penalty() -> None:
    trace_df = pd.DataFrame(
        [
            {
                "rho_p": 1.0e-7,
                "rho_t": 10.0,
                "trial_id": 0,
                "iteration": 1,
                "time_error_s": 1.0e-3,
            }
        ]
    )

    with pytest.raises(ValueError, match="does not contain the fixed penalty required"):
        _compute_exp3_trace_statistics(
            trace_df=trace_df,
            sweep_column="rho_t",
            fixed_column="rho_p",
            fixed_value=4.0e-7,
            metric_column="time_error_s",
            reference_value=0.1,
        )
