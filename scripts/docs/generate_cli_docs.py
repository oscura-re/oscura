#!/usr/bin/env python3
"""Generate CLI documentation from actual Click commands.

Usage:
    uv run python scripts/generate_cli_docs.py > docs/cli-auto.md
"""

from click.testing import CliRunner

from oscura.cli.main import cli


def main():
    """Generate markdown documentation from CLI."""
    runner = CliRunner()

    # Header
    print("# CLI Reference")
    print()
    print("> **Auto-generated** from actual CLI commands")
    print()

    # Main help
    result = runner.invoke(cli, ["--help"])
    print("## Overview")
    print()
    print("```")
    print(result.output)
    print("```")
    print()

    # Get all commands
    commands = cli.list_commands(None)

    # Document each command
    for cmd_name in sorted(commands):
        print(f"## `oscura {cmd_name}`")
        print()

        result = runner.invoke(cli, [cmd_name, "--help"])
        print("```")
        print(result.output)
        print("```")
        print()


if __name__ == "__main__":
    main()
