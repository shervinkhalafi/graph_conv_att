"""CLI for scaffolding new TMGG experiments.

Creates an experiment directory, base config, model config, and registers the CLI
entry point in pyproject.toml from a cookiecutter template.

Usage::

    uv run tmgg-scaffold my_experiment
    uv run tmgg-scaffold my_experiment --type generative --model-family gnn --model-name standard_gnn
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Template and project root paths
_SCAFFOLD_DIR = Path(__file__).parent / "_scaffold"
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # repo root (tmgg/)
_PYPROJECT_PATH = _PROJECT_ROOT / "pyproject.toml"
_EXP_CONFIGS_DIR = Path(__file__).parent / "exp_configs"
_EXPERIMENTS_DIR = Path(__file__).parent


def _slug(name: str) -> str:
    """Convert an experiment name to a slug (lowercase, underscores)."""
    return name.lower().replace(" ", "_").replace("-", "_")


def _cli_name(slug: str) -> str:
    return f"tmgg-{slug.replace('_', '-')}"


def _base_config_parent(experiment_type: str) -> str:
    mapping = {
        "denoising": "base_config_denoising",
        "generative": "base_config_gaussian_diffusion",
        "custom": "base_config_denoising",
    }
    return mapping[experiment_type]


def _run_cookiecutter(
    slug: str,
    experiment_name: str,
    experiment_type: str,
    model_family: str,
    model_name: str,
) -> Path:
    """Render the cookiecutter template into the experiments directory.

    Returns the path to the created experiment directory.
    """
    try:
        from cookiecutter.main import cookiecutter
    except ImportError as e:
        raise SystemExit(
            "cookiecutter is not installed. Run `uv sync` to install dev dependencies."
        ) from e

    parent_config = _base_config_parent(experiment_type)
    cli = _cli_name(slug)
    wandb_project = f"tmgg-{slug.replace('_', '-')}"

    extra_context = {
        "experiment_name": experiment_name,
        "experiment_slug": slug,
        "experiment_type": experiment_type,
        "base_config_parent": parent_config,
        "model_family": model_family,
        "model_name": model_name,
        "cli_name": cli,
        "wandb_project": wandb_project,
    }

    output_dir = cookiecutter(
        str(_SCAFFOLD_DIR),
        no_input=True,
        extra_context=extra_context,
        output_dir=str(_EXPERIMENTS_DIR),
    )
    return Path(output_dir)


def _render_jinja2_file(template_path: Path, context: dict[str, str]) -> str:
    """Render a .j2 template with the given context using Jinja2."""
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as e:
        raise SystemExit(
            "jinja2 is not installed. Run `uv sync` to install dev dependencies."
        ) from e

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path.name)
    return template.render(cookiecutter=context)


def _write_base_config(context: dict[str, str]) -> Path:
    """Render and write base_config_<slug>.yaml to exp_configs/."""
    template_path = _SCAFFOLD_DIR / "base_config.yaml.j2"
    content = _render_jinja2_file(template_path, context)
    dest = _EXP_CONFIGS_DIR / f"base_config_{context['experiment_slug']}.yaml"
    _ = dest.write_text(content)
    return dest


def _write_model_config(context: dict[str, str]) -> Path:
    """Render and write model config to exp_configs/models/<slug>/<slug>.yaml."""
    template_path = _SCAFFOLD_DIR / "model_config.yaml.j2"
    content = _render_jinja2_file(template_path, context)
    slug = context["experiment_slug"]
    model_dir = _EXP_CONFIGS_DIR / "models" / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / f"{slug}.yaml"
    _ = dest.write_text(content)
    return dest


def _register_cli_entry(slug: str, entry_point: str) -> None:
    """Add the new CLI entry point to pyproject.toml using tomlkit.

    Raises RuntimeError if the entry already exists.
    """
    try:
        import tomlkit
        from tomlkit.items import Table
    except ImportError as e:
        raise SystemExit(
            "tomlkit is not installed. Run `uv sync` to install dev dependencies."
        ) from e

    cli = _cli_name(slug)
    raw = _PYPROJECT_PATH.read_text()
    doc = tomlkit.loads(raw)

    project = doc["project"]
    if not isinstance(project, Table):
        raise RuntimeError(
            "Unexpected pyproject.toml structure: [project] is not a table."
        )
    scripts = project["scripts"]
    if not isinstance(scripts, Table):
        raise RuntimeError(
            "Unexpected pyproject.toml structure: [project.scripts] is not a table."
        )
    if cli in scripts:
        raise RuntimeError(
            f"Entry point '{cli}' already exists in pyproject.toml. Remove it manually or choose a different name."
        )

    scripts[cli] = entry_point
    serialized: str = tomlkit.dumps(doc)  # pyright: ignore[reportUnknownMemberType]
    _ = _PYPROJECT_PATH.write_text(serialized)


@click.command()
@click.argument("experiment_name")
@click.option(
    "--type",
    "experiment_type",
    type=click.Choice(["denoising", "generative", "custom"]),
    default="denoising",
    show_default=True,
    help="Experiment type, selects the base config parent.",
)
@click.option(
    "--model-family",
    default="spectral",
    show_default=True,
    help="Model family subdirectory under exp_configs/models/.",
)
@click.option(
    "--model-name",
    default="linear_pe",
    show_default=True,
    help="Model config file name (without .yaml) under the model family.",
)
def main(
    experiment_name: str,
    experiment_type: str,
    model_family: str,
    model_name: str,
) -> None:
    """Scaffold a new TMGG experiment from the cookiecutter template.

    EXPERIMENT_NAME is the human-readable name; it is slugified to produce
    the directory name and config prefix (e.g. "My Exp" -> "my_exp").

    Example::

        uv run tmgg-scaffold "my experiment" --type denoising --model-family gnn --model-name standard_gnn
    """
    slug = _slug(experiment_name)
    cli = _cli_name(slug)
    parent_config = _base_config_parent(experiment_type)

    context: dict[str, str] = {
        "experiment_name": experiment_name,
        "experiment_slug": slug,
        "experiment_type": experiment_type,
        "base_config_parent": parent_config,
        "model_family": model_family,
        "model_name": model_name,
        "cli_name": cli,
        "wandb_project": f"tmgg-{slug.replace('_', '-')}",
    }

    click.echo(f"Scaffolding experiment '{experiment_name}' (slug: {slug}) ...")

    # 1. Cookiecutter: create experiment package
    exp_dir = _run_cookiecutter(
        slug=slug,
        experiment_name=experiment_name,
        experiment_type=experiment_type,
        model_family=model_family,
        model_name=model_name,
    )
    click.echo(f"  [ok] experiment package: {exp_dir.relative_to(_PROJECT_ROOT)}")

    # 2. Base config YAML
    base_cfg_path = _write_base_config(context)
    click.echo(f"  [ok] base config:        {base_cfg_path.relative_to(_PROJECT_ROOT)}")

    # 3. Model config YAML
    model_cfg_path = _write_model_config(context)
    click.echo(
        f"  [ok] model config:       {model_cfg_path.relative_to(_PROJECT_ROOT)}"
    )

    # 4. pyproject.toml entry point
    entry_point = f"tmgg.experiments.{slug}.runner:main"
    _register_cli_entry(slug, entry_point)
    click.echo(f"  [ok] CLI entry:          {cli} -> {entry_point}")

    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Edit {model_cfg_path.relative_to(_PROJECT_ROOT)}")
    click.echo(
        "     Replace tmgg.models.REPLACE_WITH_YOUR_MODEL_CLASS with your model."
    )
    click.echo(f"  2. Tune {base_cfg_path.relative_to(_PROJECT_ROOT)} as needed.")
    click.echo("  3. Run `uv sync` to register the new CLI entry point.")
    click.echo(f"  4. Run `uv run {cli}` to launch the experiment.")
    sys.exit(0)
