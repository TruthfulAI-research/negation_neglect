"""CLI entry point for the §4.2 misalignment eval and plotting.

Examples (from the repo root):

    # 1. Run all (condition, seed, eval_set) targets listed in the config:
    uv run python -m src.misalignment eval experiments/05_negated_behaviors/eval_config.yaml

    # 2. Shard across tmux windows (one window per condition x eval_set):
    uv run python -m src.misalignment eval experiments/05_negated_behaviors/eval_config.yaml \
        --only-condition negated --only-eval-set targeted_behavioral

    # 3. Plot results into <results_dir>/figures/:
    uv run python -m src.misalignment plot --results-dir experiments/05_negated_behaviors/results
"""

from __future__ import annotations

from pathlib import Path

import typer

from .eval import run_eval
from .plot import run_plot

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("eval")
def cli_eval(
    config: Path = typer.Argument(..., help="Path to eval_config.yaml"),
    only_condition: str | None = typer.Option(
        None,
        "--only-condition",
        help="Restrict to one condition: base, negated, nonnegated, or aligned.",
    ),
    only_seed: int | None = typer.Option(
        None,
        "--only-seed",
        help="Restrict to one seed.",
    ),
    only_eval_set: str | None = typer.Option(
        None,
        "--only-eval-set",
        help="Restrict to one eval set: targeted_behavioral, emergent, or everyday_safety.",
    ),
) -> None:
    """Sample + judge each (condition, seed, eval_set) listed in the config."""
    run_eval(config, only_condition=only_condition, only_seed=only_seed, only_eval_set=only_eval_set)


@app.command("plot")
def cli_plot(
    results_dir: Path = typer.Option(
        ...,
        "--results-dir",
        help="Directory containing <condition>_s<seed>/<eval_set>.csv files.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Where to write the figures (default: <results-dir>/figures/).",
    ),
) -> None:
    """Render the combined 1x3 and per-question PDFs."""
    run_plot(results_dir, out_dir)


if __name__ == "__main__":
    app()
