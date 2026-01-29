"""Oscura Config Command - Configuration Management.

Provides CLI for viewing and editing Oscura configuration.


Example:
    $ oscura config --show
    $ oscura config --set analysis.default_protocol=uart
    $ oscura config --edit
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("oscura.cli.config")

# Allowlist of trusted editors (SEC-004 fix)
ALLOWED_EDITORS = {
    "nano",
    "vim",
    "vi",
    "emacs",
    "nvim",
    "code",
    "subl",
    "atom",
    "gedit",
    "kate",
    "micro",
    "helix",
}


def _get_safe_editor() -> str:
    """Get validated editor from environment.

    Returns:
        Safe editor command.

    Security:
        SEC-004 fix: Validates editor against allowlist to prevent command injection
        via $EDITOR environment variable. Falls back to 'nano' for untrusted editors.

    Example:
        >>> os.environ["EDITOR"] = "vim"
        >>> editor = _get_safe_editor()
        >>> assert editor == "vim"

        >>> os.environ["EDITOR"] = "rm -rf /"
        >>> editor = _get_safe_editor()
        >>> assert editor == "nano"  # Fallback to safe default

    References:
        https://owasp.org/www-project-top-ten/
    """
    editor_env = os.environ.get("EDITOR", "nano")

    # Check for command substitution and newlines before parsing
    # These are shell injection attempts that shlex may not detect
    if "`" in editor_env or "$(" in editor_env or "\n" in editor_env or "\r" in editor_env:
        logger.warning(
            "Command substitution or newline detected in EDITOR, falling back to nano for safety"
        )
        return "nano"

    # Extract base command (handle args like "code --wait")
    try:
        editor_parts = shlex.split(editor_env)
        if not editor_parts:
            logger.warning("Empty EDITOR value, using nano")
            return "nano"

        editor_cmd = Path(editor_parts[0]).name
    except ValueError as e:
        logger.warning(f"Invalid EDITOR value '{editor_env}': {e}, using nano")
        return "nano"

    # Validate against allowlist
    if editor_cmd not in ALLOWED_EDITORS:
        logger.warning(
            f"Untrusted editor '{editor_cmd}' not in allowlist, falling back to nano. "
            f"Allowed editors: {', '.join(sorted(ALLOWED_EDITORS))}"
        )
        return "nano"

    # Check for shell metacharacters that indicate command injection attempts
    # Shell metacharacters parsed as separate tokens by shlex indicate injection
    shell_metacharacters = {"&&", "||", ";", "|", ">", "<", ">>", "<<", "&"}
    if len(editor_parts) > 1 and any(part in shell_metacharacters for part in editor_parts[1:]):
        logger.warning(
            f"Shell metacharacters detected in EDITOR '{editor_env}', "
            f"falling back to nano for safety"
        )
        return "nano"

    return editor_env  # Return full command with args if valid


@click.command()
@click.option(
    "--show",
    is_flag=True,
    help="Show current configuration.",
)
@click.option(
    "--set",
    "set_value",
    type=str,
    default=None,
    help="Set configuration value (key=value).",
)
@click.option(
    "--edit",
    is_flag=True,
    help="Open configuration file in editor.",
)
@click.option(
    "--init",
    is_flag=True,
    help="Initialize default configuration file.",
)
@click.option(
    "--path",
    is_flag=True,
    help="Show configuration file path.",
)
@click.pass_context
def config(
    ctx: click.Context,
    show: bool,
    set_value: str | None,
    edit: bool,
    init: bool,
    path: bool,
) -> None:
    """Manage Oscura configuration.

    View, edit, and initialize configuration files.

    Args:
        ctx: Click context object.
        show: Show configuration.
        set_value: Set configuration value.
        edit: Edit configuration file.
        init: Initialize configuration.
        path: Show config path.

    Examples:

        \b
        # Show current config
        $ oscura config --show

        \b
        # Set a value
        $ oscura config --set analysis.default_protocol=uart

        \b
        # Edit config file
        $ oscura config --edit

        \b
        # Initialize config
        $ oscura config --init
    """
    verbose = ctx.obj.get("verbose", 0)

    try:
        config_path = _get_config_path()

        if path:
            click.echo(f"Configuration file: {config_path}")
            return

        if init:
            _initialize_config(config_path)
            click.echo(f"Initialized configuration at: {config_path}")
            return

        if show:
            _show_config(config_path)
            return

        if set_value:
            _set_config_value(config_path, set_value)
            click.echo(f"Updated configuration: {set_value}")
            return

        if edit:
            _edit_config(config_path)
            return

        # No options provided, show help
        click.echo(ctx.get_help())

    except Exception as e:
        logger.error(f"Config operation failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _get_config_path() -> Path:
    """Get configuration file path.

    Returns:
        Path to configuration file (absolute path).
    """
    # Check for local config first
    local_config = Path(".oscura.yaml").resolve()
    if local_config.exists():
        return local_config

    # Use user config
    user_config = Path.home() / ".config" / "oscura" / "config.yaml"
    return user_config


def _initialize_config(config_path: Path) -> None:
    """Initialize default configuration file.

    Args:
        config_path: Path to configuration file.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = """# Oscura Configuration

analysis:
  default_protocol: auto
  auto_detect_threshold: 0.7
  max_packets: 10000

export:
  default_format: json
  output_dir: oscura_output

visualization:
  default_backend: matplotlib
  figure_size: [12, 6]
  dpi: 100

cli:
  color_output: true
  progress_bars: true

logging:
  level: WARNING
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""

    with open(config_path, "w") as f:
        f.write(default_config)


def _show_config(config_path: Path) -> None:
    """Show configuration.

    Args:
        config_path: Path to configuration file.
    """
    if not config_path.exists():
        click.echo("No configuration file found. Use --init to create one.")
        return

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    click.echo(f"\nConfiguration ({config_path}):\n")
    click.echo(yaml.dump(config, default_flow_style=False))


def _set_config_value(config_path: Path, set_value: str) -> None:
    """Set configuration value.

    Args:
        config_path: Path to configuration file.
        set_value: Value to set (key=value format).
    """
    import yaml

    if "=" not in set_value:
        raise ValueError("Invalid format. Use: key=value")

    key, value = set_value.split("=", 1)
    keys = key.split(".")

    # Load existing config
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Set nested value
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Try to parse value
    try:
        # Try as number
        if "." in value:
            parsed_value: Any = float(value)
        else:
            parsed_value = int(value)
    except ValueError:
        # Try as boolean
        if value.lower() in ["true", "false"]:
            parsed_value = value.lower() == "true"
        else:
            # Keep as string
            parsed_value = value

    current[keys[-1]] = parsed_value

    # Save config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def _edit_config(config_path: Path) -> None:
    """Edit configuration file with safe editor validation.

    Args:
        config_path: Path to configuration file.

    Security:
        SEC-004 fix: Uses _get_safe_editor() to validate $EDITOR against allowlist,
        preventing command injection attacks via malicious editor values.

    Raises:
        RuntimeError: If editor execution fails.
    """
    # Create if doesn't exist
    if not config_path.exists():
        _initialize_config(config_path)

    # Get validated editor (SEC-004 fix)
    editor_cmd = _get_safe_editor()

    # Open editor with validated command
    try:
        # Parse editor command (may include args like "code --wait")
        editor_parts = shlex.split(editor_cmd)
        subprocess.run([*editor_parts, str(config_path)], check=True)
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        raise RuntimeError(f"Editor failed: {e}") from e
