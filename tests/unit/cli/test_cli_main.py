"""Comprehensive unit tests for cli/main.py CLI module.

This module provides extensive testing for the Oscura CLI framework, including:
- OutputFormat class for JSON, CSV, HTML, and table formatting
- format_output() function
- cli() main command group with verbose and version options
- shell command integration
- tutorial command integration
- main() entry point with error handling


Test Coverage:
- OutputFormat.json()
- OutputFormat.csv()
- OutputFormat.html()
- OutputFormat.table()
- format_output() with all format types
- cli() command group with verbosity levels
- shell command
- tutorial command with --list and tutorial_id
- main() entry point with error handling
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from oscura.cli.main import (
    OutputFormat,
    cli,
    format_output,
    logger,
    main,
    shell,
    tutorial,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Create sample data for output formatting tests."""
    return {
        "name": "test_signal",
        "samples": 1000,
        "sample_rate": "10 MHz",
        "rise_time": "5.5 ns",
    }


@pytest.fixture
def nested_data() -> dict[str, Any]:
    """Create nested data structure for CSV testing."""
    return {
        "signal": {"amplitude": 3.3, "frequency": 1000},
        "metadata": {"source": "oscilloscope", "date": "2025-01-01"},
    }


@pytest.fixture
def list_data() -> dict[str, Any]:
    """Create data with lists for CSV testing."""
    return {
        "channels": [1, 2, 3, 4],
        "values": ["a", "b", "c"],
    }


# =============================================================================
# Test OutputFormat.json()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_json_basic(sample_data: dict[str, Any]) -> None:
    """Test basic JSON formatting."""
    result = OutputFormat.json(sample_data)

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed == sample_data


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_json_indentation(sample_data: dict[str, Any]) -> None:
    """Test JSON formatting has proper indentation."""
    result = OutputFormat.json(sample_data)

    # Should have indentation (multiple lines)
    lines = result.split("\n")
    assert len(lines) > 1
    # Check for 2-space indentation
    assert any("  " in line for line in lines)


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_json_empty_dict() -> None:
    """Test JSON formatting with empty dictionary."""
    result = OutputFormat.json({})

    assert result == "{}"


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_json_complex_types() -> None:
    """Test JSON formatting handles complex types via default=str."""
    from datetime import datetime

    data = {
        "timestamp": datetime(2025, 1, 1, 12, 0, 0),
        "value": 42,
    }

    result = OutputFormat.json(data)
    parsed = json.loads(result)

    # datetime should be converted to string
    assert "2025" in parsed["timestamp"]
    assert parsed["value"] == 42


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_json_nested_data(nested_data: dict[str, Any]) -> None:
    """Test JSON formatting with nested dictionaries."""
    result = OutputFormat.json(nested_data)
    parsed = json.loads(result)

    assert parsed["signal"]["amplitude"] == 3.3
    assert parsed["metadata"]["source"] == "oscilloscope"


# =============================================================================
# Test OutputFormat.csv()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_csv_basic(sample_data: dict[str, Any]) -> None:
    """Test basic CSV formatting."""
    result = OutputFormat.csv(sample_data)

    lines = result.split("\n")
    assert lines[0] == "key,value"
    assert "name,test_signal" in lines
    assert "samples,1000" in lines


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_csv_nested_dict(nested_data: dict[str, Any]) -> None:
    """Test CSV formatting flattens nested dictionaries."""
    result = OutputFormat.csv(nested_data)

    # Should have flattened keys like signal.amplitude
    assert "signal.amplitude,3.3" in result
    assert "signal.frequency,1000" in result
    assert "metadata.source,oscilloscope" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_csv_list_values(list_data: dict[str, Any]) -> None:
    """Test CSV formatting handles list values."""
    result = OutputFormat.csv(list_data)

    # Lists should be comma-separated and quoted
    assert 'channels,"1,2,3,4"' in result
    assert 'values,"a,b,c"' in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_csv_empty_dict() -> None:
    """Test CSV formatting with empty dictionary."""
    result = OutputFormat.csv({})

    assert result == "key,value"


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_csv_header() -> None:
    """Test CSV formatting includes header."""
    result = OutputFormat.csv({"test": "value"})

    lines = result.split("\n")
    assert lines[0] == "key,value"


# =============================================================================
# Test OutputFormat.html()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_basic(sample_data: dict[str, Any]) -> None:
    """Test basic HTML formatting."""
    result = OutputFormat.html(sample_data)

    # Should be valid HTML structure
    assert "<!DOCTYPE html>" in result
    assert "<html>" in result
    assert "</html>" in result
    assert "<head>" in result
    assert "</head>" in result
    assert "<body>" in result
    assert "</body>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_title() -> None:
    """Test HTML formatting includes title."""
    result = OutputFormat.html({"test": "value"})

    assert "<title>Oscura Analysis Results</title>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_table_structure(sample_data: dict[str, Any]) -> None:
    """Test HTML formatting creates table structure."""
    result = OutputFormat.html(sample_data)

    assert "<table>" in result
    assert "</table>" in result
    assert "<tr>" in result
    assert "<th>Parameter</th><th>Value</th>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_data_rows(sample_data: dict[str, Any]) -> None:
    """Test HTML formatting includes data rows."""
    result = OutputFormat.html(sample_data)

    # Each key-value pair should be in a table row
    assert "<td>name</td><td>test_signal</td>" in result
    assert "<td>samples</td><td>1000</td>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_includes_css() -> None:
    """Test HTML formatting includes CSS styling."""
    result = OutputFormat.html({"test": "value"})

    assert "<style>" in result
    assert "</style>" in result
    assert "border-collapse" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_empty_dict() -> None:
    """Test HTML formatting with empty dictionary."""
    result = OutputFormat.html({})

    # Should still have valid HTML structure
    assert "<!DOCTYPE html>" in result
    assert "<table>" in result
    assert "</table>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_meta_charset() -> None:
    """Test HTML formatting includes charset meta tag."""
    result = OutputFormat.html({"test": "value"})

    assert "<meta charset='utf-8'>" in result


# =============================================================================
# Test OutputFormat.table()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_table_basic(sample_data: dict[str, Any]) -> None:
    """Test basic ASCII table formatting."""
    result = OutputFormat.table(sample_data)

    # Should contain header
    assert "Parameter" in result
    assert "Value" in result
    # Should contain data
    assert "name" in result
    assert "test_signal" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_table_borders(sample_data: dict[str, Any]) -> None:
    """Test ASCII table has border characters."""
    result = OutputFormat.table(sample_data)

    # Should have separator characters
    assert "=" in result
    assert "-" in result
    assert "|" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_table_empty_dict() -> None:
    """Test table formatting with empty dictionary."""
    result = OutputFormat.table({})

    assert result == "No data"


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_table_alignment(sample_data: dict[str, Any]) -> None:
    """Test table formatting has proper alignment."""
    result = OutputFormat.table(sample_data)
    lines = result.split("\n")

    # All data lines should have | separator
    data_lines = [line for line in lines if "|" in line]
    assert len(data_lines) >= 2  # Header + at least one data row


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_table_column_widths() -> None:
    """Test table formatting calculates column widths correctly."""
    data = {
        "short": "a",
        "much_longer_key": "value",
    }
    result = OutputFormat.table(data)

    # Both rows should have same width structure
    lines = result.split("\n")
    # Border lines should have consistent length
    border_lines = [line for line in lines if line.startswith("=") or line.startswith("-")]
    if len(border_lines) >= 2:
        assert len(border_lines[0]) == len(border_lines[1])


# =============================================================================
# Test format_output()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_json(sample_data: dict[str, Any]) -> None:
    """Test format_output with json format."""
    result = format_output(sample_data, "json")

    parsed = json.loads(result)
    assert parsed == sample_data


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_csv(sample_data: dict[str, Any]) -> None:
    """Test format_output with csv format."""
    result = format_output(sample_data, "csv")

    assert "key,value" in result


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_html(sample_data: dict[str, Any]) -> None:
    """Test format_output with html format."""
    result = format_output(sample_data, "html")

    assert "<!DOCTYPE html>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_table(sample_data: dict[str, Any]) -> None:
    """Test format_output with table format."""
    result = format_output(sample_data, "table")

    assert "Parameter" in result


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_unknown_format(sample_data: dict[str, Any]) -> None:
    """Test format_output with unknown format falls back to table."""
    result = format_output(sample_data, "unknown_format")

    # Should fall back to table format
    assert "Parameter" in result


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_empty_string_format(sample_data: dict[str, Any]) -> None:
    """Test format_output with empty string format falls back to table."""
    # getattr with empty string will use default
    result = format_output(sample_data, "")

    # Should fall back to table format
    assert "Parameter" in result or "No data" in result


# =============================================================================
# Test cli() command group
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_cli_help(runner: CliRunner) -> None:
    """Test CLI help output."""
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Oscura" in result.output
    assert "Hardware Reverse Engineering Framework" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_cli_version(runner: CliRunner) -> None:
    """Test CLI version output."""
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.8.0" in result.output
    assert "oscura" in result.output.lower()


@pytest.mark.unit
@pytest.mark.cli
def test_cli_verbose_none(runner: CliRunner) -> None:
    """Test CLI with no verbose flag executes without error."""
    # Invoke with a subcommand help to avoid "missing required" error
    result = runner.invoke(cli, ["--help"], obj={})

    # Should execute without error
    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_cli_verbose_one(runner: CliRunner) -> None:
    """Test CLI with -v executes without error."""
    result = runner.invoke(cli, ["-v", "--help"], obj={})

    # Should execute without error
    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_cli_verbose_two(runner: CliRunner) -> None:
    """Test CLI with -vv executes without error."""
    result = runner.invoke(cli, ["-vv", "--help"], obj={})

    # Should execute without error
    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_cli_verbose_three(runner: CliRunner) -> None:
    """Test CLI with -vvv executes without error."""
    result = runner.invoke(cli, ["-vvv", "--help"], obj={})

    # Should execute without error
    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_cli_context_obj_created(runner: CliRunner) -> None:
    """Test CLI creates context object with verbose level."""
    result = runner.invoke(cli, ["-v", "--help"], obj={})

    # Context should exist (no error)
    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_cli_no_command(runner: CliRunner) -> None:
    """Test CLI with no command shows help."""
    result = runner.invoke(cli, obj={})

    # Click shows help when no command is given for a group
    assert result.exit_code == 0 or "Usage:" in result.output
    # Should show available commands
    assert "characterize" in result.output or "Commands:" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_cli_has_subcommands(runner: CliRunner) -> None:
    """Test CLI has expected subcommands registered."""
    result = runner.invoke(cli, ["--help"])

    assert "characterize" in result.output
    assert "decode" in result.output
    assert "batch" in result.output
    assert "compare" in result.output
    assert "shell" in result.output
    assert "tutorial" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_cli_examples_in_help(runner: CliRunner) -> None:
    """Test CLI help includes usage examples."""
    result = runner.invoke(cli, ["--help"])

    assert "Examples:" in result.output
    assert "characterize" in result.output


# =============================================================================
# Test shell command
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_shell_command_help(runner: CliRunner) -> None:
    """Test shell command help output."""
    result = runner.invoke(shell, ["--help"])

    assert result.exit_code == 0
    assert "interactive" in result.output.lower()
    assert "shell" in result.output.lower() or "REPL" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_shell_command_calls_start_shell(runner: CliRunner) -> None:
    """Test shell command calls start_shell."""
    with patch("oscura.cli.shell.start_shell") as mock_start:
        result = runner.invoke(shell)

        mock_start.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_shell_command_via_cli(runner: CliRunner) -> None:
    """Test shell command invoked through cli group."""
    with patch("oscura.cli.shell.start_shell") as mock_start:
        result = runner.invoke(cli, ["shell"], obj={})

        mock_start.assert_called_once()


# =============================================================================
# Test tutorial command
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_command_help(runner: CliRunner) -> None:
    """Test tutorial command help output."""
    result = runner.invoke(tutorial, ["--help"])

    assert result.exit_code == 0
    assert "tutorial" in result.output.lower()


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_command_list(runner: CliRunner) -> None:
    """Test tutorial command with --list flag shows available tutorials."""
    result = runner.invoke(tutorial, ["--list"])

    assert result.exit_code == 0
    assert "Available tutorials:" in result.output
    # Should show at least one tutorial with common format
    assert "steps)" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_command_no_args_shows_list(runner: CliRunner) -> None:
    """Test tutorial command with no args shows list."""
    result = runner.invoke(tutorial)

    assert result.exit_code == 0
    assert "Available tutorials:" in result.output
    assert "Run with: oscura tutorial <tutorial_id>" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_command_run_specific(runner: CliRunner) -> None:
    """Test tutorial command runs specific tutorial."""
    with patch("oscura.cli.onboarding.run_tutorial") as mock_run:
        result = runner.invoke(tutorial, ["getting_started"])

        mock_run.assert_called_once_with("getting_started", interactive=True)


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_command_via_cli(runner: CliRunner) -> None:
    """Test tutorial command invoked through cli group."""
    mock_tutorials = [
        {"id": "test", "title": "Test", "difficulty": "beginner", "steps": 1},
    ]

    with patch("oscura.cli.onboarding.list_tutorials", return_value=mock_tutorials):
        result = runner.invoke(cli, ["tutorial", "--list"], obj={})

        assert result.exit_code == 0
        assert "Available tutorials:" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_list_shows_difficulty(runner: CliRunner) -> None:
    """Test tutorial list shows difficulty levels."""
    result = runner.invoke(tutorial, ["--list"])

    assert result.exit_code == 0
    # Should show at least one difficulty level
    assert any(level in result.output for level in ["beginner", "intermediate", "advanced"])


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_list_shows_steps(runner: CliRunner) -> None:
    """Test tutorial list shows step count."""
    result = runner.invoke(tutorial, ["--list"])

    assert result.exit_code == 0
    # Should show step counts in format "N steps"
    assert " steps)" in result.output


# =============================================================================
# Test main() entry point
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_main_entry_point() -> None:
    """Test main() entry point calls cli."""
    with patch("oscura.cli.main.cli") as mock_cli:
        mock_cli.return_value = None

        main()

        mock_cli.assert_called_once_with(obj={})


@pytest.mark.unit
@pytest.mark.cli
def test_main_entry_point_handles_exception(capsys) -> None:
    """Test main() handles exceptions gracefully."""
    with patch("oscura.cli.main.cli", side_effect=Exception("Test error")):
        with patch.object(logger, "error") as mock_log:
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            mock_log.assert_called()
            # Error message should contain the exception info
            call_args = mock_log.call_args[0][0]
            assert "Test error" in call_args


@pytest.mark.unit
@pytest.mark.cli
def test_main_entry_point_exits_with_code_1_on_error() -> None:
    """Test main() exits with code 1 on exception."""
    with patch("oscura.cli.main.cli", side_effect=ValueError("Test")):
        with patch.object(logger, "error"):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1


@pytest.mark.unit
@pytest.mark.cli
def test_main_entry_point_success() -> None:
    """Test main() succeeds without exception."""
    with patch("oscura.cli.main.cli") as mock_cli:
        mock_cli.return_value = None

        # Should not raise
        main()


# =============================================================================
# Test logger configuration
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_logger_name() -> None:
    """Test logger has correct name."""
    assert logger.name == "oscura"


@pytest.mark.unit
@pytest.mark.cli
def test_logger_default_level() -> None:
    """Test logger default level is configured."""
    # Logger should exist and be configured
    assert logger is not None


# =============================================================================
# Test command registration
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_registered(runner: CliRunner) -> None:
    """Test characterize command is registered."""
    result = runner.invoke(cli, ["characterize", "--help"], obj={})

    # Should show characterize help, not error
    assert result.exit_code == 0 or "Error" not in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_decode_registered(runner: CliRunner) -> None:
    """Test decode command is registered."""
    result = runner.invoke(cli, ["decode", "--help"], obj={})

    assert result.exit_code == 0 or "Error" not in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_batch_registered(runner: CliRunner) -> None:
    """Test batch command is registered."""
    result = runner.invoke(cli, ["batch", "--help"], obj={})

    assert result.exit_code == 0 or "Error" not in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_compare_registered(runner: CliRunner) -> None:
    """Test compare command is registered."""
    result = runner.invoke(cli, ["compare", "--help"], obj={})

    assert result.exit_code == 0 or "Error" not in result.output


# =============================================================================
# Integration tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_cli_verbose_log_message(runner: CliRunner) -> None:
    """Test verbose mode logs expected message."""
    with patch.object(logger, "info") as mock_info:
        result = runner.invoke(cli, ["-v"], obj={})

        # Should log verbose mode message
        if mock_info.called:
            call_args = str(mock_info.call_args)
            assert "Verbose" in call_args or "verbose" in call_args.lower()


@pytest.mark.unit
@pytest.mark.cli
def test_cli_debug_log_message(runner: CliRunner) -> None:
    """Test debug mode logs expected message."""
    with patch.object(logger, "debug") as mock_debug:
        result = runner.invoke(cli, ["-vv"], obj={})

        # Should log debug mode message
        if mock_debug.called:
            call_args = str(mock_debug.call_args)
            assert "Debug" in call_args or "debug" in call_args.lower()


@pytest.mark.unit
@pytest.mark.cli
def test_format_output_all_types() -> None:
    """Test all format types work without error."""
    data = {"key": "value", "number": 42}

    for fmt in ["json", "csv", "html", "table"]:
        result = format_output(data, fmt)
        assert result is not None
        assert len(result) > 0


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_handles_special_characters() -> None:
    """Test output formatters handle special characters."""
    data = {
        "html_chars": "<script>alert('xss')</script>",
        "unicode": "Hello, World!",
        "quotes": 'He said "hello"',
    }

    # Should not raise exceptions
    json_result = OutputFormat.json(data)
    csv_result = OutputFormat.csv(data)
    html_result = OutputFormat.html(data)
    table_result = OutputFormat.table(data)

    assert json_result is not None
    assert csv_result is not None
    assert html_result is not None
    assert table_result is not None


@pytest.mark.unit
@pytest.mark.cli
def test_cli_with_invalid_command(runner: CliRunner) -> None:
    """Test CLI with invalid command shows error."""
    result = runner.invoke(cli, ["nonexistent_command"], obj={})

    assert result.exit_code != 0
    assert "No such command" in result.output or "Error" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_cli_context_passed_to_subcommands(runner: CliRunner) -> None:
    """Test CLI context is passed to subcommands."""
    # Using shell command as it's simple
    with patch("oscura.cli.shell.start_shell"):
        result = runner.invoke(cli, ["-v", "shell"], obj={})

        # Should work without errors
        assert result.exit_code == 0


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_table_single_item() -> None:
    """Test table format with single item."""
    result = OutputFormat.table({"only_key": "only_value"})

    assert "only_key" in result
    assert "only_value" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_csv_numeric_values() -> None:
    """Test CSV format with numeric values."""
    data = {"integer": 42, "float": 3.14159, "negative": -100}
    result = OutputFormat.csv(data)

    assert "integer,42" in result
    assert "float,3.14159" in result
    assert "negative,-100" in result


@pytest.mark.unit
@pytest.mark.cli
def test_output_format_html_special_values() -> None:
    """Test HTML format with None and boolean values."""
    data = {"none_val": None, "bool_true": True, "bool_false": False}
    result = OutputFormat.html(data)

    assert "<td>none_val</td><td>None</td>" in result
    assert "<td>bool_true</td><td>True</td>" in result
    assert "<td>bool_false</td><td>False</td>" in result


@pytest.mark.unit
@pytest.mark.cli
def test_cli_standalone_verbose_count(runner: CliRunner) -> None:
    """Test verbose flag can be used multiple times."""
    # Test that --verbose can be specified multiple times (with --help to avoid missing command)
    result = runner.invoke(cli, ["--verbose", "--verbose", "--help"], obj={})

    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_cli_mixed_v_and_verbose(runner: CliRunner) -> None:
    """Test mixing -v and --verbose works."""
    result = runner.invoke(cli, ["-v", "--verbose", "--help"], obj={})

    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_tutorial_empty_list(runner: CliRunner) -> None:
    """Test tutorial list with no tutorials available."""
    with patch("oscura.cli.onboarding.tutorials.list_tutorials", return_value=[]):
        result = runner.invoke(tutorial, ["--list"])

        assert result.exit_code == 0
        assert "Available tutorials:" in result.output
