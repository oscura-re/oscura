"""Oscura Core CLI Framework implementing CLI-001 (Enhanced Edition).

Provides the main entry point for the oscura command-line interface with
comprehensive subcommands, interactive mode, batch processing, configuration
management, and shell completion support.


Example:
    $ oscura --help
    $ oscura analyze signal.wfm --output json
    $ oscura decode uart capture.wfm --baud-rate 115200
    $ oscura shell   # Interactive REPL
    $ oscura config --show  # View configuration
    $ oscura plugins list  # Manage plugins
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger("oscura")


class OutputFormat:
    """Output format handler for CLI results.

    Supports JSON, CSV, HTML, and table (default) output formats.
    """

    @staticmethod
    def json(data: dict[str, Any]) -> str:
        """Format as JSON."""
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def csv(data: dict[str, Any]) -> str:
        """Format as CSV (simplified)."""
        lines = ["key,value"]
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dict - flatten
                for subkey, subvalue in value.items():
                    lines.append(f"{key}.{subkey},{subvalue}")
            elif isinstance(value, list):
                lines.append(f'{key},"{",".join(map(str, value))}"')
            else:
                lines.append(f"{key},{value}")
        return "\n".join(lines)

    @staticmethod
    def html(data: dict[str, Any]) -> str:
        """Format as HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Oscura Analysis Results</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Oscura Analysis Results</h1>",
            "<table>",
            "<tr><th>Parameter</th><th>Value</th></tr>",
        ]

        for key, value in data.items():
            html_parts.append(f"<tr><td>{key}</td><td>{value}</td></tr>")

        html_parts.extend(
            [
                "</table>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html_parts)

    @staticmethod
    def table(data: dict[str, Any]) -> str:
        """Format as ASCII table."""
        if not data:
            return "No data"

        # Calculate column widths
        max_key = max(len(str(k)) for k in data)
        max_val = max(len(str(v)) for v in data.values())

        # Build table
        lines = []
        lines.append("=" * (max_key + max_val + 7))
        lines.append(f"{'Parameter':{max_key}} | Value")
        lines.append("-" * (max_key + max_val + 7))

        for key, value in data.items():
            lines.append(f"{key!s:{max_key}} | {value}")

        lines.append("=" * (max_key + max_val + 7))

        return "\n".join(lines)


def format_output(data: dict[str, Any], format_type: str) -> str:
    """Format output data according to specified format.

    Args:
        data: Dictionary of results to format.
        format_type: Output format ('json', 'csv', 'html', 'table').

    Returns:
        Formatted string.
    """
    formatter = getattr(OutputFormat, format_type, OutputFormat.table)
    return formatter(data)


def load_config_file(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches for config in:
            1. .oscura.yaml in current directory
            2. ~/.config/oscura/config.yaml

    Returns:
        Configuration dictionary.
    """
    import yaml

    if config_path is None:
        # Search for config files
        candidates = [
            Path(".oscura.yaml"),
            Path.home() / ".config" / "oscura" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for INFO, -vv for DEBUG).",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to configuration file (YAML).",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Quiet mode (suppress non-error output).",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="JSON output mode for scripting.",
)
@click.version_option(prog_name="oscura")  # Version auto-detected from package metadata
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: int,
    config: str | None,
    quiet: bool,
    json_output: bool,
) -> None:
    """Oscura - Hardware Reverse Engineering Framework.

    Unified framework for extracting all information from hardware systems through
    signals and data. Features unknown protocol discovery, state machine extraction,
    CRC recovery, and security analysis.

    Args:
        ctx: Click context object.
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
        config: Path to configuration file.
        quiet: Quiet mode flag.
        json_output: JSON output mode flag.

    Examples:
        oscura analyze signal.wfm
        oscura decode uart.wfm --protocol auto
        oscura batch '*.wfm' --analysis characterize
        oscura visualize trace.wfm
        oscura shell  # Interactive REPL
    """
    # Ensure ctx.obj exists
    ctx.ensure_object(dict)

    # Set logging level based on verbosity
    if quiet:
        logger.setLevel(logging.ERROR)
    elif verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info("Verbose mode enabled")
    else:  # verbose >= 2
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["json_output"] = json_output

    # Load configuration
    config_path = Path(config) if config else None
    ctx.obj["config"] = load_config_file(config_path)


# Enhanced subcommands - imported below
from oscura.cli.analyze import analyze
from oscura.cli.batch import batch
from oscura.cli.benchmark import benchmark
from oscura.cli.characterize import characterize
from oscura.cli.compare import compare
from oscura.cli.config_cmd import config as config_cmd
from oscura.cli.decode import decode
from oscura.cli.export import export
from oscura.cli.validate_cmd import validate
from oscura.cli.visualize import visualize


@click.command()
def shell() -> None:
    """Start an interactive Oscura shell.

    Opens a Python REPL with Oscura pre-imported and ready to use.
    Features tab completion, persistent history, and helpful shortcuts.

    Example:
        $ oscura shell
        Oscura Shell v0.6.0
        >>> trace = load("signal.wfm")
        >>> rise_time(trace)
    """
    from oscura.cli.shell import start_shell

    start_shell()


@click.command()
@click.argument("tutorial_id", required=False, default=None)
@click.option("--list", "list_tutorials", is_flag=True, help="List available tutorials")
def tutorial(tutorial_id: str | None, list_tutorials: bool) -> None:
    """Run an interactive tutorial.

    Provides step-by-step guidance for learning Oscura.

    Args:
        tutorial_id: ID of the tutorial to run (or None to list).
        list_tutorials: If True, list available tutorials.

    Examples:
        oscura tutorial --list           # List available tutorials
        oscura tutorial getting_started  # Run the getting started tutorial
    """
    from oscura.cli.onboarding import list_tutorials as list_tut
    from oscura.cli.onboarding import run_tutorial

    if list_tutorials or tutorial_id is None:
        tutorials = list_tut()
        click.echo("Available tutorials:")
        for t in tutorials:
            click.echo(f"  {t['id']}: {t['title']} ({t['difficulty']}, {t['steps']} steps)")
        if tutorial_id is None:
            click.echo("\nRun with: oscura tutorial <tutorial_id>")
        return

    run_tutorial(tutorial_id, interactive=True)


# Register all subcommands
cli.add_command(analyze)  # type: ignore[has-type]
cli.add_command(decode)  # type: ignore[has-type]
cli.add_command(export)
cli.add_command(visualize)
cli.add_command(benchmark)
cli.add_command(validate)
cli.add_command(config_cmd, name="config")
cli.add_command(characterize)  # type: ignore[has-type]
cli.add_command(batch)  # type: ignore[has-type]
cli.add_command(compare)  # type: ignore[has-type]
cli.add_command(shell)
cli.add_command(tutorial)


def main() -> None:
    """Entry point for the oscura CLI."""
    try:
        cli(obj={})
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
