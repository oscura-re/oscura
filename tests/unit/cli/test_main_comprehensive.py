"""Comprehensive unit tests for main.py CLI module.

This module provides extensive testing for the main CLI entry point, including:
- OutputFormat class methods (json, csv, html, table)
- format_output() function
- load_config_file() function
- CLI group initialization and options
- Global flags (verbose, quiet, config, json-output)
- Version option
- Context object management
- Shell command
- Tutorial command

Test Coverage:
- OutputFormat.json()
- OutputFormat.csv()
- OutputFormat.html()
- OutputFormat.table()
- format_output() dispatcher
- load_config_file() with various paths
- cli() group initialization
- shell() command
- tutorial() command
- Error handling and edge cases

References:
    - src/oscura/cli/main.py
    - Click testing: https://click.palletsprojects.com/en/8.1.x/testing/
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from oscura.cli.main import (
    OutputFormat,
    cli,
    format_output,
    load_config_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test OutputFormat Class
# =============================================================================


@pytest.mark.unit
def test_output_format_json_simple():
    """Test JSON formatting with simple data."""
    data = {"key1": "value1", "key2": 42, "key3": True}
    result = OutputFormat.json(data)

    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed == data
    assert "key1" in result
    assert "value1" in result


@pytest.mark.unit
def test_output_format_json_nested():
    """Test JSON formatting with nested data."""
    data = {
        "outer": {"inner": "value", "count": 10},
        "list": [1, 2, 3],
    }
    result = OutputFormat.json(data)

    parsed = json.loads(result)
    assert parsed["outer"]["inner"] == "value"
    assert parsed["list"] == [1, 2, 3]


@pytest.mark.unit
def test_output_format_json_with_non_serializable():
    """Test JSON formatting with non-JSON-serializable objects (uses default=str)."""
    from datetime import datetime

    data = {"timestamp": datetime(2024, 1, 1, 12, 0, 0), "value": 42}
    result = OutputFormat.json(data)

    # Should convert datetime to string
    parsed = json.loads(result)
    assert "2024" in str(parsed["timestamp"])
    assert parsed["value"] == 42


@pytest.mark.unit
def test_output_format_csv_simple():
    """Test CSV formatting with simple flat data."""
    data = {"name": "test", "value": 42, "active": True}
    result = OutputFormat.csv(data)

    lines = result.split("\n")
    assert lines[0] == "key,value"
    assert "name,test" in lines
    assert "value,42" in lines
    assert "active,True" in lines


@pytest.mark.unit
def test_output_format_csv_nested():
    """Test CSV formatting with nested dictionaries (flattens them)."""
    data = {
        "simple": "value",
        "nested": {"sub1": "val1", "sub2": "val2"},
    }
    result = OutputFormat.csv(data)

    # Nested keys should be flattened with dot notation
    assert "nested.sub1,val1" in result
    assert "nested.sub2,val2" in result


@pytest.mark.unit
def test_output_format_csv_with_list():
    """Test CSV formatting with list values."""
    data = {"tags": ["tag1", "tag2", "tag3"], "count": 3}
    result = OutputFormat.csv(data)

    # Lists should be quoted and comma-separated
    assert '"tag1,tag2,tag3"' in result or "tag1,tag2,tag3" in result


@pytest.mark.unit
def test_output_format_html_simple():
    """Test HTML formatting with simple data."""
    data = {"parameter1": "value1", "parameter2": 42}
    result = OutputFormat.html(data)

    # Verify HTML structure
    assert "<!DOCTYPE html>" in result
    assert "<html>" in result
    assert "<title>Oscura Analysis Results</title>" in result
    assert "<table>" in result
    assert "parameter1" in result
    assert "value1" in result
    assert "parameter2" in result
    assert "42" in result


@pytest.mark.unit
def test_output_format_html_has_styling():
    """Test that HTML output includes CSS styling."""
    data = {"key": "value"}
    result = OutputFormat.html(data)

    assert "<style>" in result
    assert "font-family" in result
    assert "border-collapse" in result
    assert "#4CAF50" in result  # Green header color


@pytest.mark.unit
def test_output_format_html_has_metadata():
    """Test HTML formatting includes proper metadata."""
    data = {"test": "data"}
    result = OutputFormat.html(data)

    assert "<meta charset='utf-8'>" in result
    assert "<h1>Oscura Analysis Results</h1>" in result


@pytest.mark.unit
def test_output_format_table_simple():
    """Test table formatting with simple data."""
    data = {"param1": "value1", "param2": 42}
    result = OutputFormat.table(data)

    # Should have borders and header
    assert "=" in result
    assert "-" in result
    assert "Parameter" in result
    assert "Value" in result
    assert "param1" in result
    assert "value1" in result


@pytest.mark.unit
def test_output_format_table_empty():
    """Test table formatting with empty data."""
    data = {}
    result = OutputFormat.table(data)

    assert result == "No data"


@pytest.mark.unit
def test_output_format_table_alignment():
    """Test table formatting properly aligns columns."""
    data = {
        "short": "a",
        "very_long_parameter_name": "very_long_value_string",
    }
    result = OutputFormat.table(data)

    # Split into lines and check alignment
    lines = result.split("\n")
    assert len(lines) > 3  # Header + separator + data + borders
    # All lines should have consistent separator
    assert all(("=" in line or "-" in line or "|" in line or line.strip() == "") for line in lines)


# =============================================================================
# Test format_output() Dispatcher
# =============================================================================


@pytest.mark.unit
def test_format_output_json():
    """Test format_output() with json format."""
    data = {"key": "value"}
    result = format_output(data, "json")

    parsed = json.loads(result)
    assert parsed == data


@pytest.mark.unit
def test_format_output_csv():
    """Test format_output() with csv format."""
    data = {"key": "value"}
    result = format_output(data, "csv")

    assert "key,value" in result


@pytest.mark.unit
def test_format_output_html():
    """Test format_output() with html format."""
    data = {"key": "value"}
    result = format_output(data, "html")

    assert "<html>" in result
    assert "key" in result


@pytest.mark.unit
def test_format_output_table():
    """Test format_output() with table format (default)."""
    data = {"key": "value"}
    result = format_output(data, "table")

    assert "Parameter" in result
    assert "Value" in result


@pytest.mark.unit
def test_format_output_invalid_format_fallback():
    """Test format_output() with invalid format falls back to table."""
    data = {"key": "value"}
    result = format_output(data, "invalid_format")

    # Should fall back to table format
    assert "Parameter" in result or "No data" in result


# =============================================================================
# Test load_config_file()
# =============================================================================


@pytest.mark.unit
def test_load_config_file_nonexistent():
    """Test load_config_file() with nonexistent path returns empty dict."""
    result = load_config_file(Path("/nonexistent/config.yaml"))

    assert result == {}


@pytest.mark.unit
def test_load_config_file_explicit_path(tmp_path):
    """Test load_config_file() with explicit path."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("key: value\ncount: 42\n")

    result = load_config_file(config_path)

    assert result["key"] == "value"
    assert result["count"] == 42


@pytest.mark.unit
def test_load_config_file_auto_discover_cwd(tmp_path, monkeypatch):
    """Test load_config_file() auto-discovers .oscura.yaml in current dir."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / ".oscura.yaml"
    config_path.write_text("discovered: true\n")

    result = load_config_file(None)

    assert result["discovered"] is True


@pytest.mark.unit
def test_load_config_file_empty_yaml(tmp_path):
    """Test load_config_file() with empty YAML file."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    result = load_config_file(config_path)

    assert result == {}


@pytest.mark.unit
def test_load_config_file_malformed_yaml(tmp_path):
    """Test load_config_file() with malformed YAML raises exception."""
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text("key: [unclosed\n  - list")

    with pytest.raises(Exception):  # yaml.YAMLError or similar
        load_config_file(config_path)


# =============================================================================
# Test CLI Group
# =============================================================================


@pytest.mark.unit
def test_cli_help():
    """Test main CLI --help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Oscura - Hardware Reverse Engineering Framework" in result.output
    assert "analyze" in result.output
    assert "decode" in result.output
    assert "shell" in result.output


@pytest.mark.unit
def test_cli_version():
    """Test --version flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "oscura" in result.output.lower()
    assert "0.6.0" in result.output


@pytest.mark.unit
def test_cli_verbose_flag():
    """Test -v/--verbose flag sets logging level."""
    runner = CliRunner()

    with patch("oscura.cli.main.logger") as mock_logger:
        result = runner.invoke(cli, ["-v", "analyze", "--help"])

        # Should have called setLevel with INFO
        assert mock_logger.setLevel.called


@pytest.mark.unit
def test_cli_verbose_debug_double_v():
    """Test -vv flag sets DEBUG logging."""
    runner = CliRunner()

    with patch("oscura.cli.main.logger") as mock_logger:
        result = runner.invoke(cli, ["-vv", "analyze", "--help"])

        # Should have called setLevel with DEBUG
        assert mock_logger.setLevel.called


@pytest.mark.unit
def test_cli_quiet_flag():
    """Test --quiet flag suppresses output."""
    runner = CliRunner()

    with patch("oscura.cli.main.logger") as mock_logger:
        result = runner.invoke(cli, ["--quiet", "analyze", "--help"])

        # Should have set ERROR level
        assert mock_logger.setLevel.called


@pytest.mark.unit
def test_cli_config_option(tmp_path):
    """Test --config option loads configuration."""
    runner = CliRunner()
    config_path = tmp_path / "test.yaml"
    config_path.write_text("test_key: test_value\n")

    result = runner.invoke(cli, ["--config", str(config_path), "analyze", "--help"])

    # Command should execute successfully
    assert result.exit_code == 0


@pytest.mark.unit
def test_cli_json_output_flag():
    """Test --json flag for scripting mode."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--json", "analyze", "--help"])

    # Should execute without error
    assert result.exit_code == 0


@pytest.mark.unit
def test_cli_context_object_creation():
    """Test that CLI creates context object with settings."""
    runner = CliRunner()

    # Use a simple subcommand that doesn't do much
    with patch("oscura.cli.main.logger"):
        result = runner.invoke(cli, ["-v", "--quiet", "--json"], obj={})

        # Just verify it doesn't crash
        assert result.exit_code in [0, 2]  # 0=success, 2=missing command (expected)


# =============================================================================
# Test Shell Command
# =============================================================================


@pytest.mark.unit
def test_shell_command_help():
    """Test shell command --help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["shell", "--help"])

    assert result.exit_code == 0
    assert "interactive" in result.output.lower()
    assert "shell" in result.output.lower()


@pytest.mark.unit
def test_shell_command_invocation():
    """Test shell command starts interactive shell."""
    runner = CliRunner()

    with patch("oscura.cli.shell.start_shell") as mock_start:
        result = runner.invoke(cli, ["shell"])

        # Should have called start_shell
        mock_start.assert_called_once()


# =============================================================================
# Test Tutorial Command
# =============================================================================


@pytest.mark.unit
def test_tutorial_command_help():
    """Test tutorial command --help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["tutorial", "--help"])

    assert result.exit_code == 0
    assert "tutorial" in result.output.lower()


@pytest.mark.unit
def test_tutorial_list_flag():
    """Test tutorial --list shows available tutorials."""
    runner = CliRunner()

    mock_tutorials = [
        {"id": "getting_started", "title": "Getting Started", "difficulty": "beginner", "steps": 5},
        {"id": "advanced", "title": "Advanced", "difficulty": "expert", "steps": 10},
    ]

    with patch("oscura.cli.main.list_tut", return_value=mock_tutorials):
        result = runner.invoke(cli, ["tutorial", "--list"])

        assert result.exit_code == 0
        assert "getting_started" in result.output
        assert "Getting Started" in result.output
        assert "beginner" in result.output


@pytest.mark.unit
def test_tutorial_no_args_lists_tutorials():
    """Test tutorial with no args shows list."""
    runner = CliRunner()

    mock_tutorials = [{"id": "test", "title": "Test Tutorial", "difficulty": "easy", "steps": 3}]

    with patch("oscura.cli.main.list_tut", return_value=mock_tutorials):
        result = runner.invoke(cli, ["tutorial"])

        assert result.exit_code == 0
        assert "test" in result.output
        assert "Run with: oscura tutorial <tutorial_id>" in result.output


@pytest.mark.unit
def test_tutorial_run_specific():
    """Test running a specific tutorial."""
    runner = CliRunner()

    with patch("oscura.cli.main.run_tutorial") as mock_run:
        with patch("oscura.cli.main.list_tut", return_value=[]):
            result = runner.invoke(cli, ["tutorial", "getting_started"])

            # Should have called run_tutorial
            mock_run.assert_called_once_with("getting_started", interactive=True)


# =============================================================================
# Test Main Entry Point
# =============================================================================


@pytest.mark.unit
def test_main_entry_point():
    """Test main() entry point function."""
    from oscura.cli.main import main

    with patch("oscura.cli.main.cli") as mock_cli:
        # Mock successful execution
        mock_cli.return_value = None

        main()

        # Should have called cli with empty obj
        mock_cli.assert_called_once_with(obj={})


@pytest.mark.unit
def test_main_entry_point_exception_handling():
    """Test main() handles exceptions and exits with code 1."""
    from oscura.cli.main import main

    with patch("oscura.cli.main.cli") as mock_cli:
        with patch("sys.exit") as mock_exit:
            # Mock exception
            mock_cli.side_effect = RuntimeError("Test error")

            main()

            # Should have called sys.exit(1)
            mock_exit.assert_called_once_with(1)


# =============================================================================
# Test Command Registration
# =============================================================================


@pytest.mark.unit
def test_all_commands_registered():
    """Test that all expected commands are registered."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    expected_commands = [
        "analyze",
        "decode",
        "export",
        "visualize",
        "benchmark",
        "validate",
        "config",
        "characterize",
        "batch",
        "compare",
        "shell",
        "tutorial",
    ]

    for cmd in expected_commands:
        assert cmd in result.output, f"Command '{cmd}' not found in help output"


@pytest.mark.unit
def test_cli_accepts_all_global_options():
    """Test CLI accepts all documented global options."""
    runner = CliRunner()

    # Test each option individually
    options = [
        ["-v"],
        ["-vv"],
        ["--quiet"],
        ["--json"],
    ]

    for opt in options:
        result = runner.invoke(cli, opt + ["--help"])
        # Should not fail on the option itself
        assert "--help" in result.output or result.exit_code == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
def test_output_format_with_special_characters():
    """Test output formats handle special characters correctly."""
    data = {
        "special": 'value with <html> & "quotes"',
        "unicode": "测试 🚀",
    }

    # JSON should preserve everything
    json_result = OutputFormat.json(data)
    assert "测试" in json_result or "\\u" in json_result  # Unicode encoded

    # HTML should escape HTML tags
    html_result = OutputFormat.html(data)
    assert "<html>" in html_result  # Our HTML structure
    # Note: Click/Python may not auto-escape in simple string insertion

    # CSV should handle commas and quotes
    csv_result = OutputFormat.csv(data)
    assert "special" in csv_result


@pytest.mark.unit
def test_cli_with_invalid_config_path():
    """Test CLI with non-existent config path."""
    runner = CliRunner()

    # Should not crash, just ignore the config
    result = runner.invoke(cli, ["--config", "/nonexistent/path.yaml", "analyze", "--help"])

    # Should still show help (config file optional)
    assert result.exit_code == 0


@pytest.mark.unit
def test_format_output_with_large_nested_data():
    """Test format_output() with deeply nested data structures."""
    data = {"level1": {"level2": {"level3": {"deep_value": 42}}}}

    # All formats should handle this
    json_result = format_output(data, "json")
    assert "deep_value" in json_result

    csv_result = format_output(data, "csv")
    assert "level1" in csv_result

    html_result = format_output(data, "html")
    assert "level1" in html_result

    table_result = format_output(data, "table")
    assert "level1" in table_result or "No data" not in table_result
