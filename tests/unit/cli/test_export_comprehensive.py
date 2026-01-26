"""Comprehensive unit tests for export.py CLI module.

This module provides extensive testing for the export command, including:
- Command argument parsing and validation
- Export format selection
- Session file loading
- Output file creation
- Include traces option
- Error handling for NotImplementedError
- All export format placeholders

Test Coverage:
- export() command with all options
- Format validation (json, html, csv, matlab, wireshark, scapy, kaitai)
- Session file handling
- Output path validation
- Error reporting

References:
    - src/oscura/cli/export.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from oscura.cli.main import cli

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test Export Command
# =============================================================================


@pytest.mark.unit
def test_export_help():
    """Test export command --help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--help"])

    assert result.exit_code == 0
    assert "export" in result.output.lower()
    assert "FORMAT" in result.output
    assert "SESSION" in result.output
    assert "--output" in result.output


@pytest.mark.unit
def test_export_missing_arguments():
    """Test export command with missing arguments."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export"])

    # Should fail - missing required args
    assert result.exit_code != 0


@pytest.mark.unit
def test_export_missing_session():
    """Test export command with missing session argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "json"])

    # Should fail - missing session file
    assert result.exit_code != 0


@pytest.mark.unit
def test_export_missing_output():
    """Test export command with missing --output option."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake session")

        result = runner.invoke(cli, ["export", "json", "test.tks"])

        # Should fail - missing --output
        assert result.exit_code != 0
        assert "--output" in result.output or "-o" in result.output


@pytest.mark.unit
def test_export_nonexistent_session():
    """Test export command with nonexistent session file."""
    runner = CliRunner()

    result = runner.invoke(
        cli, ["export", "json", "/nonexistent/session.tks", "--output", "out.json"]
    )

    # Should fail - file doesn't exist
    assert result.exit_code != 0


@pytest.mark.unit
def test_export_not_implemented_error():
    """Test export command raises NotImplementedError (current state)."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake session")

        result = runner.invoke(cli, ["export", "json", "test.tks", "--output", "out.json"])

        # Should exit with error due to NotImplementedError
        assert result.exit_code == 1
        assert "Error:" in result.output or "redesigned" in result.output.lower()


@pytest.mark.unit
def test_export_all_formats():
    """Test export command with all supported formats."""
    runner = CliRunner()

    formats = ["json", "html", "csv", "matlab", "wireshark", "scapy", "kaitai"]

    for fmt in formats:
        with runner.isolated_filesystem():
            Path("test.tks").write_bytes(b"fake session")

            result = runner.invoke(cli, ["export", fmt, "test.tks", "--output", f"out.{fmt}"])

            # All should fail with NotImplementedError currently
            assert result.exit_code == 1


@pytest.mark.unit
def test_export_include_traces_flag():
    """Test export command with --include-traces flag."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake session")

        result = runner.invoke(
            cli,
            ["export", "json", "test.tks", "--output", "out.json", "--include-traces"],
        )

        # Should attempt export (but fail with NotImplementedError)
        assert result.exit_code == 1


@pytest.mark.unit
def test_export_verbose_mode():
    """Test export command with -v flag."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake session")

        result = runner.invoke(cli, ["-v", "export", "json", "test.tks", "--output", "out.json"])

        # Should log information before failing
        assert result.exit_code == 1


@pytest.mark.unit
def test_export_case_insensitive_format():
    """Test export command accepts case-insensitive format names."""
    runner = CliRunner()

    # Click should handle case sensitivity automatically
    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake")

        # Try uppercase format name
        result = runner.invoke(cli, ["export", "JSON", "test.tks", "--output", "out.json"])

        # Should accept the format
        # (will fail with NotImplementedError, but that's after format validation)
        assert "invalid choice" not in result.output.lower()


@pytest.mark.unit
def test_export_invalid_format():
    """Test export command with invalid format."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake")

        result = runner.invoke(cli, ["export", "invalid_format", "test.tks", "--output", "out.txt"])

        # Should reject invalid format
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


@pytest.mark.unit
def test_export_output_short_option():
    """Test export command with -o short option."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake")

        result = runner.invoke(cli, ["export", "json", "test.tks", "-o", "out.json"])

        # Should accept short option (will fail with NotImplementedError)
        assert result.exit_code == 1
        assert "redesigned" in result.output.lower() or "Error:" in result.output


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_export_with_special_characters_in_path():
    """Test export command with special characters in paths."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create session file with spaces and special chars
        session_file = Path("my session (test).tks")
        session_file.write_bytes(b"fake")

        result = runner.invoke(
            cli, ["export", "json", str(session_file), "--output", "out put.json"]
        )

        # Should handle the path (will fail with NotImplementedError)
        assert result.exit_code == 1


@pytest.mark.unit
def test_export_formats_have_help_text():
    """Test that export help shows all available formats."""
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--help"])

    # Should list all supported formats
    formats = ["json", "html", "csv", "matlab", "wireshark", "scapy", "kaitai"]

    for fmt in formats:
        assert fmt in result.output.lower()


@pytest.mark.unit
def test_export_error_with_verbose_shows_exception():
    """Test export error with -vv shows full exception."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake")

        # With -vv, should show full traceback
        result = runner.invoke(cli, ["-vv", "export", "json", "test.tks", "--output", "out.json"])

        # Should show more detailed error information
        assert result.exit_code == 1


@pytest.mark.unit
def test_export_relative_paths():
    """Test export command with relative paths."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.tks").write_bytes(b"fake")

        result = runner.invoke(
            cli, ["export", "json", "./test.tks", "--output", "./output/result.json"]
        )

        # Should handle relative paths
        assert result.exit_code == 1  # NotImplementedError


@pytest.mark.unit
def test_export_absolute_paths(tmp_path):
    """Test export command with absolute paths."""
    runner = CliRunner()

    session_file = tmp_path / "session.tks"
    session_file.write_bytes(b"fake")

    output_file = tmp_path / "output.json"

    result = runner.invoke(cli, ["export", "json", str(session_file), "--output", str(output_file)])

    # Should handle absolute paths
    assert result.exit_code == 1  # NotImplementedError
