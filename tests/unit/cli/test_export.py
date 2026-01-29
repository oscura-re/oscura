"""Comprehensive unit tests for export.py CLI module.

This module provides extensive testing for the Oscura export command, including:
- Session export to multiple formats
- Format-specific export functions
- Error handling for redesigned architecture
- Export template generation

Test Coverage:
- export() CLI command with all options
- Format validation
- NotImplementedError for redesigned functionality
- Template generation functions (placeholders)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from oscura.cli.export import export

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.usefixtures("reset_logging_state"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Test export() CLI command
# =============================================================================


@pytest.mark.unit
def test_export_command_not_implemented(cli_runner, tmp_path):
    """Test that export command raises NotImplementedError."""
    session_file = tmp_path / "session.tks"
    session_file.touch()
    output_file = tmp_path / "output.json"

    result = cli_runner.invoke(
        export,
        ["json", str(session_file), "--output", str(output_file)],
        obj={"verbose": 0},
    )

    assert result.exit_code == 1
    assert "NotImplementedError" in str(result.exception) or "redesigned" in result.output.lower()


@pytest.mark.unit
def test_export_command_all_formats(cli_runner, tmp_path):
    """Test export command with all supported formats."""
    session_file = tmp_path / "session.tks"
    session_file.touch()

    formats = ["json", "html", "csv", "matlab", "wireshark", "scapy", "kaitai"]

    for fmt in formats:
        output_file = tmp_path / f"output.{fmt}"

        result = cli_runner.invoke(
            export,
            [fmt, str(session_file), "--output", str(output_file)],
            obj={"verbose": 0},
        )

        # All should fail with NotImplementedError due to redesign
        assert result.exit_code == 1


@pytest.mark.unit
def test_export_command_requires_output(cli_runner, tmp_path):
    """Test that export command requires --output option."""
    session_file = tmp_path / "session.tks"
    session_file.touch()

    result = cli_runner.invoke(export, ["json", str(session_file)], obj={"verbose": 0})

    # Should fail due to missing required option
    assert result.exit_code != 0


@pytest.mark.unit
def test_export_command_session_file_must_exist(cli_runner, tmp_path):
    """Test that export requires existing session file."""
    output_file = tmp_path / "output.json"

    result = cli_runner.invoke(
        export,
        ["json", "/nonexistent/session.tks", "--output", str(output_file)],
        obj={"verbose": 0},
    )

    # Click should validate file existence
    assert result.exit_code != 0


@pytest.mark.unit
def test_export_command_include_traces_flag(cli_runner, tmp_path):
    """Test export with --include-traces flag."""
    session_file = tmp_path / "session.tks"
    session_file.touch()
    output_file = tmp_path / "output.json"

    result = cli_runner.invoke(
        export,
        ["json", str(session_file), "--output", str(output_file), "--include-traces"],
        obj={"verbose": 0},
    )

    # Should still fail with NotImplementedError, but flag should be accepted
    assert result.exit_code == 1


@pytest.mark.unit
def test_export_command_verbose_logging(cli_runner, tmp_path, caplog):
    """Test export with verbose logging."""
    session_file = tmp_path / "session.tks"
    session_file.touch()
    output_file = tmp_path / "output.json"

    result = cli_runner.invoke(
        export,
        ["json", str(session_file), "--output", str(output_file)],
        obj={"verbose": 1},
    )

    # Should log the export attempt
    assert result.exit_code == 1


@pytest.mark.unit
def test_export_command_error_handling(cli_runner, tmp_path):
    """Test export error handling."""
    session_file = tmp_path / "session.tks"
    session_file.touch()
    output_file = tmp_path / "output.json"

    result = cli_runner.invoke(
        export,
        ["json", str(session_file), "--output", str(output_file)],
        obj={"verbose": 0},
    )

    assert result.exit_code == 1
    assert "Error:" in result.output


@pytest.mark.unit
def test_export_command_error_with_verbose(cli_runner, tmp_path):
    """Test export error handling with verbose mode (should raise)."""
    session_file = tmp_path / "session.tks"
    session_file.touch()
    output_file = tmp_path / "output.json"

    result = cli_runner.invoke(
        export,
        ["json", str(session_file), "--output", str(output_file)],
        obj={"verbose": 2},
    )

    assert result.exit_code == 1
    # Should raise NotImplementedError
    assert isinstance(result.exception, NotImplementedError)


@pytest.mark.unit
def test_export_format_choices():
    """Test that export command defines correct format choices."""
    # Check format choices in command definition
    for param in export.params:
        if param.name == "format":
            expected_formats = ["json", "html", "csv", "matlab", "wireshark", "scapy", "kaitai"]
            assert all(fmt in param.type.choices for fmt in expected_formats)
            break
    else:
        pytest.fail("Format parameter not found in export command")


@pytest.mark.unit
def test_export_format_case_insensitive():
    """Test that format parameter is case-insensitive."""
    for param in export.params:
        if param.name == "format":
            assert param.type.case_sensitive is False
            break


# =============================================================================
# Test Export Template Functions (Placeholders)
# =============================================================================


@pytest.mark.unit
def test_export_json_template():
    """Test JSON export template structure."""
    from oscura.cli.export import _export_json

    # Function exists but will fail without session
    assert callable(_export_json)


@pytest.mark.unit
def test_export_html_template():
    """Test HTML export template structure."""
    from oscura.cli.export import _export_html

    # Function exists but will fail without session
    assert callable(_export_html)


@pytest.mark.unit
def test_export_csv_template():
    """Test CSV export template structure."""
    from oscura.cli.export import _export_csv

    # Function exists but will fail without session
    assert callable(_export_csv)


@pytest.mark.unit
def test_export_wireshark_template():
    """Test Wireshark dissector template."""
    from oscura.cli.export import _export_wireshark

    # Function should generate Lua template
    assert callable(_export_wireshark)


@pytest.mark.unit
def test_export_scapy_template():
    """Test Scapy layer template."""
    from oscura.cli.export import _export_scapy

    # Function should generate Python template
    assert callable(_export_scapy)


@pytest.mark.unit
def test_export_kaitai_template():
    """Test Kaitai struct template."""
    from oscura.cli.export import _export_kaitai

    # Function should generate YAML template
    assert callable(_export_kaitai)


# =============================================================================
# Documentation Tests
# =============================================================================


@pytest.mark.unit
def test_export_command_has_help():
    """Test that export command has help text."""
    assert export.help is not None
    assert len(export.help) > 0


@pytest.mark.unit
def test_export_command_has_examples():
    """Test that export command includes usage examples."""
    # Check docstring for examples
    assert "Examples:" in export.callback.__doc__ or "Example:" in export.callback.__doc__


@pytest.mark.unit
def test_export_redesign_message():
    """Test that NotImplementedError includes helpful redesign message."""
    runner = CliRunner()
    session_file = Path("test.tks")

    with runner.isolated_filesystem():
        session_file.touch()

        result = runner.invoke(
            export,
            ["json", str(session_file), "--output", "out.json"],
            obj={"verbose": 0},
        )

        # Should mention redesign
        assert "redesigned" in result.output.lower() or "AnalysisSession" in result.output
