"""Result serialization and run artifact helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dadmm_localization.constants import DEFAULT_OUTPUT_ROOT, PACKAGE_VERSION
from dadmm_localization.types import RunArtifacts


def _stable_config_hash(payload: dict[str, Any]) -> str:
    """Compute a short stable hash for a config payload.

    Args:
        payload: JSON-serializable configuration mapping.

    Returns:
        Short deterministic hash string.
    """

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:5]


def _git_commit() -> str:
    """Read the current git commit hash when available.

    Returns:
        Git commit hash, or `"unknown"` if it cannot be determined.
    """

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def setup_run_artifacts(
    experiment_name: str,
    config_payload: dict[str, Any],
    master_seed: int,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> RunArtifacts:
    """Create the run directory, manifest metadata, and file logger.

    Args:
        experiment_name: Experiment family name used in the output path.
        config_payload: Resolved experiment config payload.
        master_seed: Master seed associated with the run.
        output_root: Root results directory.

    Returns:
        Bundle containing the run directory, logger name, and manifest.
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    config_hash = _stable_config_hash(config_payload)
    run_id = f"{timestamp}_{config_hash}"
    run_dir = Path(output_root) / experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    logger_name = f"dadmm_localization.{experiment_name}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False

    manifest = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "package_version": PACKAGE_VERSION,
        "master_seed": master_seed,
    }
    return RunArtifacts(run_dir=run_dir, logger_name=logger_name, manifest=manifest)


def get_logger(logger_name: str) -> logging.Logger:
    """Return a run-scoped logger.

    Args:
        logger_name: Fully qualified logger name.

    Returns:
        Logger instance associated with the run.
    """

    return logging.getLogger(logger_name)


def write_config_yaml(run_dir: Path, config_payload: dict[str, Any]) -> None:
    """Persist the resolved YAML config for reproducibility.

    Args:
        run_dir: Run directory where the config should be written.
        config_payload: Resolved config payload to serialize.
    """

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_payload, file, sort_keys=False)


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist manifest metadata as JSON.

    Args:
        run_dir: Run directory where the manifest should be written.
        manifest: Metadata mapping to serialize.
    """

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)


def write_dataframe(dataframe, path: Path) -> None:  # type: ignore[no-untyped-def]
    """Write a dataframe as Parquet.

    Args:
        dataframe: Pandas dataframe to serialize.
        path: Destination parquet file path.
    """

    dataframe.to_parquet(path, index=False)
