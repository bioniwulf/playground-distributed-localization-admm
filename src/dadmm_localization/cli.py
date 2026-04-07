"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from dadmm_localization.config import load_experiment_config
from dadmm_localization.experiments import run_experiment
from dadmm_localization.visualization import visualize_results


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns:
        Configured argument parser for the project commands.
    """

    parser = argparse.ArgumentParser(prog="dadmm-localization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config.")
    run_parser.add_argument("--config", required=True, type=Path, help="Path to the YAML config.")

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Visualize experiment results from a run directory.",
    )
    visualize_parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to a run directory or an experiment directory under results/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the project CLI.

    Args:
        argv: Optional command-line argument list. If omitted, arguments are
            read from `sys.argv`.

    Returns:
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_experiment_config(args.config)
        run_dir = run_experiment(config)
        print(run_dir)
        return 0
    if args.command == "visualize":
        output_paths = visualize_results(run_dir=args.run_dir)
        for path in output_paths:
            print(path)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
