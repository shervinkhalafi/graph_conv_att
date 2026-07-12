"""CLI tools for Modal cloud experiment orchestration.

Entry point: ``tmgg-modal``
"""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Manage TMGG experiments on Modal cloud GPUs.

    \b
    Subcommands:
      run        Compose config locally and dispatch a single experiment
      evaluate   Dispatch MMD evaluation of checkpoints to Modal GPUs
      aggregate  Pull evaluation results from the Modal volume into Parquet
      datasets   Prepare or validate datasets on the shared volume
      eval-all   Sequentially evaluate every checkpoint in a finished run dir
    """


from tmgg.modal.cli.aggregate import aggregate  # noqa: E402
from tmgg.modal.cli.datasets import datasets  # noqa: E402
from tmgg.modal.cli.eval_all import eval_all  # noqa: E402
from tmgg.modal.cli.evaluate import evaluate  # noqa: E402
from tmgg.modal.cli.run import run  # noqa: E402

main.add_command(run)
main.add_command(evaluate)
main.add_command(aggregate)
main.add_command(datasets)
main.add_command(eval_all)

if __name__ == "__main__":
    main()
