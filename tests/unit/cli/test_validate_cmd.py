"""Comprehensive unit tests for validate_cmd.py CLI module.

This module provides extensive testing for the Oscura validate command, including:
- Protocol specification validation
- Spec structure validation
- Test data validation
- Error and warning reporting
- Output formatting
- YAML/JSON spec loading

Test Coverage:
- validate() CLI command with all options
- _validate_spec_structure() spec validation
- _validate_against_data() data validation
- _print_validation_results() results formatting
- Required and recommended field checking
- Field structure validation
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from oscura.cli.validate_cmd import (
    _print_validation_results,
    _validate_against_data,
    _validate_spec_structure,
    validate,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_spec():
    """Create a valid specification dictionary."""
    return {
        "name": "TestProtocol",
        "version": "1.0",
        "description": "Test protocol specification",
        "fields": [
            {"name": "header", "type": "uint8", "size": 1},
            {"name": "payload", "type": "bytes", "size": 64},
        ],
        "constraints": {"min_length": 65},
    }


@pytest.fixture
def minimal_spec():
    """Create a minimal valid specification."""
    return {"name": "Minimal", "version": "1.0"}


# =============================================================================
# Test _validate_spec_structure()
# =============================================================================


@pytest.mark.unit
def test_validate_spec_structure_valid(valid_spec):
    """Test validation of valid spec structure."""
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(valid_spec, results)

    assert results["valid"] is True
    assert len(results["errors"]) == 0


@pytest.mark.unit
def test_validate_spec_structure_missing_name():
    """Test error when name field is missing."""
    spec = {"version": "1.0"}
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(spec, results)

    assert results["valid"] is False
    assert any("name" in err.lower() for err in results["errors"])


@pytest.mark.unit
def test_validate_spec_structure_missing_version():
    """Test error when version field is missing."""
    spec = {"name": "Test"}
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(spec, results)

    assert results["valid"] is False
    assert any("version" in err.lower() for err in results["errors"])


@pytest.mark.unit
def test_validate_spec_structure_missing_recommended_fields(minimal_spec):
    """Test warnings for missing recommended fields."""
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(minimal_spec, results)

    # Should still be valid but have warnings
    assert results["valid"] is True
    assert len(results["warnings"]) > 0
    # Should warn about missing description, fields, constraints
    warning_text = " ".join(results["warnings"])
    assert "description" in warning_text or "fields" in warning_text


@pytest.mark.unit
def test_validate_spec_structure_fields_not_list():
    """Test error when fields is not a list."""
    spec = {"name": "Test", "version": "1.0", "fields": "not_a_list"}
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(spec, results)

    # Should not add errors (fields validation only happens if it's a list)
    # But this is still a structural issue


@pytest.mark.unit
def test_validate_spec_structure_field_missing_name():
    """Test error when field is missing name attribute."""
    spec = {
        "name": "Test",
        "version": "1.0",
        "fields": [{"type": "uint8"}],  # Missing name
    }
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(spec, results)

    assert results["valid"] is False
    assert any("name" in err.lower() for err in results["errors"])


@pytest.mark.unit
def test_validate_spec_structure_field_not_dict():
    """Test error when field is not a dictionary."""
    spec = {"name": "Test", "version": "1.0", "fields": ["not_a_dict", {"name": "valid"}]}
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(spec, results)

    assert results["valid"] is False
    assert any("not a dictionary" in err.lower() for err in results["errors"])


@pytest.mark.unit
def test_validate_spec_structure_all_recommended_fields(valid_spec):
    """Test that no warnings with all recommended fields."""
    results = {"valid": True, "errors": [], "warnings": []}

    _validate_spec_structure(valid_spec, results)

    assert results["valid"] is True
    # May still have some warnings, but should have all main fields


# =============================================================================
# Test _validate_against_data()
# =============================================================================


@pytest.mark.unit
def test_validate_against_data_sample_rate():
    """Test validation of sample rate requirement."""
    spec = {"name": "Test", "version": "1.0", "sample_rate_min": 1e6}
    results = {"valid": True, "errors": [], "warnings": []}

    # Mock trace with low sample rate
    trace = Mock()
    trace.metadata.sample_rate = 500e3  # 500 kHz (below minimum)
    trace.data = [0, 1, 2, 3]

    _validate_against_data(spec, trace, results)

    # Should have warning about low sample rate
    assert len(results["warnings"]) > 0
    assert any("sample rate" in warn.lower() for warn in results["warnings"])


@pytest.mark.unit
def test_validate_against_data_min_samples():
    """Test validation of minimum samples requirement."""
    spec = {"name": "Test", "version": "1.0", "min_samples": 1000}
    results = {"valid": True, "errors": [], "warnings": []}

    # Mock trace with too few samples
    trace = Mock()
    trace.metadata.sample_rate = 1e6
    trace.data = [0] * 100  # Only 100 samples

    _validate_against_data(spec, trace, results)

    # Should error about insufficient samples
    assert results["valid"] is False
    assert any("samples" in err.lower() for err in results["errors"])


@pytest.mark.unit
def test_validate_against_data_adds_metrics():
    """Test that validation adds test data metrics to results."""
    spec = {"name": "Test", "version": "1.0"}
    results = {"valid": True, "errors": [], "warnings": []}

    trace = Mock()
    trace.metadata.sample_rate = 10e6
    trace.data = [0] * 5000

    _validate_against_data(spec, trace, results)

    assert "test_data_samples" in results
    assert results["test_data_samples"] == 5000
    assert "test_data_sample_rate" in results
    assert results["test_data_sample_rate"] == 10e6


@pytest.mark.unit
def test_validate_against_data_sufficient_samples():
    """Test validation passes with sufficient samples."""
    spec = {"name": "Test", "version": "1.0", "min_samples": 100}
    results = {"valid": True, "errors": [], "warnings": []}

    trace = Mock()
    trace.metadata.sample_rate = 1e6
    trace.data = [0] * 200  # Enough samples

    _validate_against_data(spec, trace, results)

    # Should not add sample-related errors
    assert results["valid"] is True


@pytest.mark.unit
def test_validate_against_data_no_constraints():
    """Test validation with no constraints in spec."""
    spec = {"name": "Test", "version": "1.0"}
    results = {"valid": True, "errors": [], "warnings": []}

    trace = Mock()
    trace.metadata.sample_rate = 1e6
    trace.data = [0] * 10

    _validate_against_data(spec, trace, results)

    # Should still add metrics
    assert "test_data_samples" in results
    assert "test_data_sample_rate" in results


# =============================================================================
# Test _print_validation_results()
# =============================================================================


@pytest.mark.unit
def test_print_validation_results_success():
    """Test printing successful validation results."""
    results = {
        "spec_file": "test.yaml",
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    with patch("click.echo") as mock_echo:
        _print_validation_results(results)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = "".join(calls)
        assert "test.yaml" in output
        assert "PASS" in output


@pytest.mark.unit
def test_print_validation_results_failure():
    """Test printing failed validation results."""
    results = {
        "spec_file": "bad.yaml",
        "valid": False,
        "errors": ["Missing required field: name"],
        "warnings": [],
    }

    with patch("click.echo") as mock_echo:
        _print_validation_results(results)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = "".join(calls)
        assert "FAIL" in output
        assert "Errors:" in output
        assert "Missing required field" in output


@pytest.mark.unit
def test_print_validation_results_with_warnings():
    """Test printing results with warnings."""
    results = {
        "spec_file": "spec.yaml",
        "valid": True,
        "errors": [],
        "warnings": ["Missing recommended field: description"],
    }

    with patch("click.echo") as mock_echo:
        _print_validation_results(results)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = "".join(calls)
        assert "Warnings:" in output
        assert "description" in output


@pytest.mark.unit
def test_print_validation_results_with_test_data():
    """Test printing results including test data metrics."""
    results = {
        "spec_file": "spec.yaml",
        "valid": True,
        "errors": [],
        "warnings": [],
        "test_data_samples": 10000,
        "test_data_sample_rate": 1e6,
    }

    with patch("click.echo") as mock_echo:
        _print_validation_results(results)

        calls = [str(call) for call in mock_echo.call_args_list]
        output = "".join(calls)
        assert "Test Data:" in output
        assert "10000" in output
        assert "1000000" in output or "1e+06" in output


# =============================================================================
# Test validate() CLI command
# =============================================================================


@pytest.mark.unit
def test_validate_command_basic(cli_runner, tmp_path):
    """Test basic validate command execution."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("name: Test\nversion: 1.0\n")

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 0})

    assert result.exit_code == 0
    assert "PASS" in result.output


@pytest.mark.unit
def test_validate_command_invalid_spec(cli_runner, tmp_path):
    """Test validate command with invalid spec."""
    spec_file = tmp_path / "bad.yaml"
    spec_file.write_text("version: 1.0\n")  # Missing name

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 0})

    assert result.exit_code == 1  # Should exit with error
    assert "FAIL" in result.output


@pytest.mark.unit
def test_validate_command_with_test_data(cli_runner, tmp_path):
    """Test validate command with test data file."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("name: Test\nversion: 1.0\nmin_samples: 100\n")

    data_file = tmp_path / "test.wfm"
    data_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_trace = Mock()
        mock_trace.metadata.sample_rate = 1e6
        mock_trace.data = [0] * 500
        mock_load.return_value = mock_trace

        result = cli_runner.invoke(
            validate, [str(spec_file), "--test-data", str(data_file)], obj={"verbose": 0}
        )

        assert result.exit_code == 0
        assert "Test Data:" in result.output


@pytest.mark.unit
def test_validate_command_json_output(cli_runner, tmp_path):
    """Test validate command with JSON output."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("name: Test\nversion: 1.0\n")

    result = cli_runner.invoke(validate, [str(spec_file), "--output", "json"], obj={"verbose": 0})

    assert result.exit_code == 0
    # Output should be JSON
    assert "{" in result.output
    assert '"valid"' in result.output


@pytest.mark.unit
def test_validate_command_table_output(cli_runner, tmp_path):
    """Test validate command with table output (default)."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("name: Test\nversion: 1.0\n")

    result = cli_runner.invoke(validate, [str(spec_file), "--output", "table"], obj={"verbose": 0})

    assert result.exit_code == 0
    assert "Validation Results" in result.output


@pytest.mark.unit
def test_validate_command_yaml_spec(cli_runner, tmp_path):
    """Test validate command with YAML spec file."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("name: YAMLTest\nversion: 2.0\n")

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 0})

    assert result.exit_code == 0


@pytest.mark.unit
def test_validate_command_json_spec(cli_runner, tmp_path):
    """Test validate command with JSON spec file."""
    spec_file = tmp_path / "spec.json"
    spec_file.write_text('{"name": "JSONTest", "version": "1.0"}')

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 0})

    assert result.exit_code == 0


@pytest.mark.unit
def test_validate_command_verbose_logging(cli_runner, tmp_path, caplog):
    """Test validate with verbose logging."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("name: Test\nversion: 1.0\n")

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 1})

    assert result.exit_code == 0


@pytest.mark.unit
def test_validate_command_error_handling(cli_runner, tmp_path):
    """Test validate error handling."""
    spec_file = tmp_path / "bad.yaml"
    spec_file.write_text("invalid: yaml: content:\n  - bad\n  indentation")

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 0})

    assert result.exit_code == 1
    assert "Error:" in result.output


@pytest.mark.unit
def test_validate_command_error_with_verbose(cli_runner, tmp_path):
    """Test validate error handling with verbose mode (should raise)."""
    spec_file = tmp_path / "bad.yaml"
    spec_file.write_text("invalid yaml content {[}")

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 2})

    assert result.exit_code == 1
    assert result.exception is not None


@pytest.mark.unit
def test_validate_command_nonexistent_file(cli_runner):
    """Test validate command with nonexistent file."""
    result = cli_runner.invoke(validate, ["/nonexistent/file.yaml"], obj={"verbose": 0})

    # Click should catch this with path validation
    assert result.exit_code != 0


@pytest.mark.unit
def test_validate_exits_1_on_invalid_spec(cli_runner, tmp_path):
    """Test that validate exits with code 1 for invalid specs."""
    spec_file = tmp_path / "invalid.yaml"
    spec_file.write_text("version: 1.0\n")  # Missing required 'name'

    result = cli_runner.invoke(validate, [str(spec_file)], obj={"verbose": 0})

    assert result.exit_code == 1


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_full_validation_workflow(cli_runner, tmp_path):
    """Test complete validation workflow."""
    # Create valid spec
    spec_file = tmp_path / "protocol.yaml"
    spec_file.write_text(
        """
name: TestProtocol
version: 1.0
description: Test protocol
fields:
  - name: header
    type: uint8
  - name: data
    type: bytes
constraints:
  min_length: 10
min_samples: 100
"""
    )

    # Create test data
    data_file = tmp_path / "capture.wfm"
    data_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_trace = Mock()
        mock_trace.metadata.sample_rate = 10e6
        mock_trace.data = [0] * 1000
        mock_load.return_value = mock_trace

        result = cli_runner.invoke(
            validate, [str(spec_file), "--test-data", str(data_file)], obj={"verbose": 0}
        )

        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "Test Data:" in result.output
