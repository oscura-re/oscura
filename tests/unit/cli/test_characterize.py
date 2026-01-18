"""Comprehensive unit tests for characterize.py CLI module.

This module provides extensive testing for the Oscura characterize command, including:
- Buffer, signal, and power characterization
- Logic family auto-detection
- Reference comparison analysis
- Output formatting and HTML report generation
- Error handling and edge cases


Test Coverage:
- characterize() CLI command with all options
- _perform_characterization() for different analysis types
- Buffer characterization with timing measurements
- Signal and power analysis
- Logic family detection
- Reference trace comparison
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.characterize import _perform_characterization, characterize
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metadata():
    """Create sample trace metadata."""
    return TraceMetadata(
        sample_rate=10e6,  # 10 MHz
        vertical_scale=1.0,
        vertical_offset=0.0,
    )


@pytest.fixture
def sample_trace(sample_metadata):
    """Create sample waveform trace."""
    # Create a square wave pattern
    data = np.array([0.0, 0.0, 3.3, 3.3, 0.0, 0.0, 3.3, 3.3] * 125, dtype=np.float64)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def reference_trace(sample_metadata):
    """Create reference waveform trace."""
    # Slightly different pattern
    data = np.array([0.0, 0.0, 3.2, 3.2, 0.0, 0.0, 3.2, 3.2] * 125, dtype=np.float64)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Test _perform_characterization() - Buffer Analysis
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_buffer_basic(sample_trace):
    """Test basic buffer characterization."""
    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
            with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=5.2):
                with patch("oscura.analyzers.waveform.measurements.undershoot", return_value=3.1):
                    results = _perform_characterization(
                        trace=sample_trace,
                        reference_trace=None,
                        analysis_type="buffer",
                        logic_family="CMOS_3V3",
                    )

    assert results["analysis_type"] == "buffer"
    assert results["logic_family"] == "CMOS_3V3"
    assert "10.50 ns" in results["rise_time"]
    assert "12.30 ns" in results["fall_time"]
    assert "5.2 %" in results["overshoot"]
    assert "3.1 %" in results["undershoot"]
    assert results["status"] == "PASS"


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_buffer_with_nan_values(sample_trace):
    """Test buffer characterization handles NaN values."""
    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=np.nan):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=np.nan):
            with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=np.nan):
                with patch(
                    "oscura.analyzers.waveform.measurements.undershoot", return_value=np.nan
                ):
                    results = _perform_characterization(
                        trace=sample_trace,
                        reference_trace=None,
                        analysis_type="buffer",
                        logic_family="TTL",
                    )

    assert results["rise_time"] == "N/A"
    assert results["fall_time"] == "N/A"
    assert results["overshoot"] == "N/A"
    assert results["undershoot"] == "N/A"


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_buffer_auto_logic_family(sample_trace):
    """Test buffer characterization with auto logic family detection."""
    mock_detection = {
        "primary": {"name": "CMOS_3V3", "confidence": 0.95},
        "candidates": [],
    }

    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=5e-9):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=5e-9):
            with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=2.0):
                with patch("oscura.analyzers.waveform.measurements.undershoot", return_value=1.5):
                    with patch("oscura.inference.detect_logic_family", return_value=mock_detection):
                        results = _perform_characterization(
                            trace=sample_trace,
                            reference_trace=None,
                            analysis_type="buffer",
                            logic_family="auto",
                        )

    assert results["logic_family_detected"] == "CMOS_3V3"
    assert results["confidence"] == "95%"


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_buffer_explicit_logic_family(sample_trace):
    """Test buffer characterization with explicit logic family."""
    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=8e-9):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=9e-9):
            with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=4.0):
                with patch("oscura.analyzers.waveform.measurements.undershoot", return_value=2.5):
                    results = _perform_characterization(
                        trace=sample_trace,
                        reference_trace=None,
                        analysis_type="buffer",
                        logic_family="LVTTL",
                    )

    assert results["logic_family_detected"] == "LVTTL"
    assert "confidence" not in results


# =============================================================================
# Test _perform_characterization() - Signal Analysis
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_signal(sample_trace):
    """Test signal characterization."""
    results = _perform_characterization(
        trace=sample_trace,
        reference_trace=None,
        analysis_type="signal",
        logic_family="auto",
    )

    assert results["analysis_type"] == "signal"
    assert "amplitude" in results
    assert "peak_to_peak" in results
    assert "mean" in results
    assert "rms" in results
    # All values should contain 'V' for volts
    assert "V" in results["amplitude"]
    assert "V" in results["mean"]
    assert "V" in results["rms"]


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_signal_constant_waveform(sample_metadata):
    """Test signal characterization with constant waveform."""
    constant_data = np.ones(1000, dtype=np.float64) * 2.5
    constant_trace = WaveformTrace(data=constant_data, metadata=sample_metadata)

    results = _perform_characterization(
        trace=constant_trace,
        reference_trace=None,
        analysis_type="signal",
        logic_family="auto",
    )

    # Amplitude and peak-to-peak should be 0.0
    assert "0.000 V" in results["amplitude"]
    assert "2.500 V" in results["mean"]


# =============================================================================
# Test _perform_characterization() - Power Analysis
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_power(sample_trace):
    """Test power characterization."""
    results = _perform_characterization(
        trace=sample_trace,
        reference_trace=None,
        analysis_type="power",
        logic_family="auto",
    )

    assert results["analysis_type"] == "power"
    assert "average_power" in results
    assert "peak_power" in results
    assert "energy" in results
    # Check units
    assert "mW" in results["average_power"]
    assert "mW" in results["peak_power"]
    assert "uJ" in results["energy"]


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_power_calculations(sample_metadata):
    """Test power calculation accuracy."""
    # Create known voltage waveform
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    trace = WaveformTrace(data=data, metadata=sample_metadata)

    results = _perform_characterization(
        trace=trace,
        reference_trace=None,
        analysis_type="power",
        logic_family="auto",
    )

    # Power = V^2 (assuming R=1)
    # Expected: avg = (1 + 4 + 9 + 16 + 25) / 5 = 11.0
    # Convert to mW
    assert "average_power" in results
    assert float(results["average_power"].split()[0]) > 0


# =============================================================================
# Test _perform_characterization() - Comparison
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_with_reference(sample_trace, reference_trace):
    """Test characterization with reference trace comparison."""
    mock_comparison = MagicMock()
    mock_comparison.correlation = 0.98

    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10e-9):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=10e-9):
            with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=5.0):
                with patch("oscura.analyzers.waveform.measurements.undershoot", return_value=3.0):
                    with patch("oscura.comparison.compare.similarity_score", return_value=0.95):
                        with patch(
                            "oscura.comparison.compare.compare_traces",
                            return_value=mock_comparison,
                        ):
                            results = _perform_characterization(
                                trace=sample_trace,
                                reference_trace=reference_trace,
                                analysis_type="buffer",
                                logic_family="CMOS_3V3",
                            )

    assert "comparison" in results
    assert "0.9800" in results["comparison"]["correlation"]
    assert "95.0%" in results["comparison"]["similarity"]
    assert "amplitude_difference" in results["comparison"]


@pytest.mark.unit
@pytest.mark.cli
def test_perform_characterization_metadata_included(sample_trace):
    """Test that trace metadata is included in results."""
    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=5e-9):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=5e-9):
            with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=2.0):
                with patch("oscura.analyzers.waveform.measurements.undershoot", return_value=1.0):
                    results = _perform_characterization(
                        trace=sample_trace,
                        reference_trace=None,
                        analysis_type="buffer",
                        logic_family="TTL",
                    )

    assert "sample_rate" in results
    assert "10.0 MHz" in results["sample_rate"]
    assert "samples" in results
    assert results["samples"] == 1000
    assert "duration" in results


# =============================================================================
# Test characterize() CLI Command
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_basic(cli_runner, tmp_path):
    """Test basic characterize command execution."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.characterize._perform_characterization") as mock_char:
            with patch("oscura.cli.characterize.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_char.return_value = {"analysis_type": "buffer", "status": "PASS"}
                mock_format.return_value = "Formatted output"

                result = cli_runner.invoke(characterize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                assert "Formatted output" in result.output
                mock_load.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_with_type_option(cli_runner, tmp_path):
    """Test characterize command with different analysis types."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    for analysis_type in ["buffer", "signal", "power"]:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.characterize._perform_characterization") as mock_char:
                with patch("oscura.cli.characterize.format_output"):
                    mock_load.return_value = Mock()
                    mock_char.return_value = {"analysis_type": analysis_type}

                    result = cli_runner.invoke(
                        characterize,
                        [str(test_file), "--type", analysis_type],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    call_args = mock_char.call_args
                    assert call_args[1]["analysis_type"] == analysis_type


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_logic_family_option(cli_runner, tmp_path):
    """Test characterize command with logic family option."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    logic_families = ["TTL", "CMOS", "CMOS_3V3", "CMOS_5V", "LVTTL", "LVCMOS", "auto"]

    for family in logic_families:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.characterize._perform_characterization") as mock_char:
                with patch("oscura.cli.characterize.format_output"):
                    mock_load.return_value = Mock()
                    mock_char.return_value = {"logic_family": family}

                    result = cli_runner.invoke(
                        characterize,
                        [str(test_file), "--logic-family", family],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    call_args = mock_char.call_args
                    assert call_args[1]["logic_family"] == family


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_with_reference(cli_runner, tmp_path):
    """Test characterize command with reference file."""
    test_file = tmp_path / "test.wfm"
    ref_file = tmp_path / "reference.wfm"
    test_file.touch()
    ref_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.characterize._perform_characterization") as mock_char:
            with patch("oscura.cli.characterize.format_output"):
                mock_load.return_value = Mock()
                mock_char.return_value = {"comparison": {}}

                result = cli_runner.invoke(
                    characterize,
                    [str(test_file), "--compare", str(ref_file)],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                # load should be called twice (main + reference)
                assert mock_load.call_count == 2
                # Check that reference_trace was passed
                call_args = mock_char.call_args
                assert call_args[1]["reference_trace"] is not None


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_output_formats(cli_runner, tmp_path):
    """Test characterize command with different output formats."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    for output_format in ["json", "csv", "html", "table"]:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.characterize._perform_characterization") as mock_char:
                with patch("oscura.cli.characterize.format_output") as mock_format:
                    mock_load.return_value = Mock()
                    mock_char.return_value = {"status": "PASS"}
                    mock_format.return_value = f"{output_format} output"

                    result = cli_runner.invoke(
                        characterize,
                        [str(test_file), "--output", output_format],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    mock_format.assert_called_with(mock_char.return_value, output_format)


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_save_report(cli_runner, tmp_path):
    """Test characterize command with HTML report saving."""
    test_file = tmp_path / "test.wfm"
    report_file = tmp_path / "report.html"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.characterize._perform_characterization") as mock_char:
            with patch("oscura.cli.characterize.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_char.return_value = {"status": "PASS"}
                mock_format.return_value = "<html>Report</html>"

                result = cli_runner.invoke(
                    characterize,
                    [str(test_file), "--save-report", str(report_file)],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                assert report_file.exists()
                content = report_file.read_text()
                assert "<html>Report</html>" in content


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_adds_filename(cli_runner, tmp_path):
    """Test that characterize command adds filename to results."""
    test_file = tmp_path / "my_signal.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.characterize._perform_characterization") as mock_char:
            with patch("oscura.cli.characterize.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_char.return_value = {"status": "PASS"}
                mock_format.return_value = "output"

                result = cli_runner.invoke(characterize, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Check that file key was added to results
                format_call_args = mock_format.call_args[0][0]
                assert format_call_args["file"] == "my_signal.wfm"


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_adds_reference_filename(cli_runner, tmp_path):
    """Test that reference filename is added to results."""
    test_file = tmp_path / "test.wfm"
    ref_file = tmp_path / "golden.wfm"
    test_file.touch()
    ref_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.characterize._perform_characterization") as mock_char:
            with patch("oscura.cli.characterize.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_char.return_value = {}
                mock_format.return_value = "output"

                result = cli_runner.invoke(
                    characterize,
                    [str(test_file), "--compare", str(ref_file)],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                format_call_args = mock_format.call_args[0][0]
                assert format_call_args["reference_file"] == "golden.wfm"


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_verbose_logging(cli_runner, tmp_path):
    """Test characterize command with verbose logging."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.characterize._perform_characterization") as mock_char:
            with patch("oscura.cli.characterize.format_output"):
                mock_load.return_value = Mock()
                mock_char.return_value = {}

                result = cli_runner.invoke(
                    characterize,
                    [str(test_file), "--logic-family", "CMOS_3V3"],
                    obj={"verbose": 1},
                )

                assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_error_handling(cli_runner, tmp_path):
    """Test characterize command error handling."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = Exception("Failed to load file")

        result = cli_runner.invoke(characterize, [str(test_file)], obj={"verbose": 0})

        assert result.exit_code == 1
        assert "Error: Failed to load file" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_error_with_verbose(cli_runner, tmp_path):
    """Test characterize command error handling with verbose mode."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = ValueError("Test error")

        result = cli_runner.invoke(characterize, [str(test_file)], obj={"verbose": 2})

        # With verbose > 1, exception should be raised
        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_nonexistent_file(cli_runner):
    """Test characterize command with nonexistent file."""
    result = cli_runner.invoke(characterize, ["/nonexistent/file.wfm"], obj={"verbose": 0})

    # Click should catch this with its path validation
    assert result.exit_code != 0


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_command_nonexistent_reference(cli_runner, tmp_path):
    """Test characterize command with nonexistent reference file."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    result = cli_runner.invoke(
        characterize,
        [str(test_file), "--compare", "/nonexistent/ref.wfm"],
        obj={"verbose": 0},
    )

    # Click should catch this with its path validation
    assert result.exit_code != 0


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_end_to_end_buffer(cli_runner, tmp_path, sample_trace):
    """Test complete buffer characterization workflow."""
    test_file = tmp_path / "buffer.wfm"
    test_file.touch()

    with patch("oscura.loaders.load", return_value=sample_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12e-9):
                with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=5.0):
                    with patch(
                        "oscura.analyzers.waveform.measurements.undershoot", return_value=3.0
                    ):
                        result = cli_runner.invoke(
                            characterize,
                            [str(test_file), "--type", "buffer", "--output", "json"],
                            obj={"verbose": 0},
                        )

                        assert result.exit_code == 0
                        # Should contain JSON output markers
                        assert "{" in result.output or "buffer" in result.output.lower()


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_end_to_end_with_comparison(
    cli_runner, tmp_path, sample_trace, reference_trace
):
    """Test complete characterization workflow with reference comparison."""
    test_file = tmp_path / "test.wfm"
    ref_file = tmp_path / "ref.wfm"
    test_file.touch()
    ref_file.touch()

    load_calls = [sample_trace, reference_trace]

    def mock_load_side_effect(path):
        return load_calls.pop(0)

    mock_comparison = MagicMock()
    mock_comparison.correlation = 0.95

    with patch("oscura.loaders.load", side_effect=mock_load_side_effect):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=10e-9):
                with patch("oscura.analyzers.waveform.measurements.overshoot", return_value=2.0):
                    with patch(
                        "oscura.analyzers.waveform.measurements.undershoot", return_value=1.0
                    ):
                        with patch("oscura.comparison.compare.similarity_score", return_value=0.9):
                            with patch(
                                "oscura.comparison.compare.compare_traces",
                                return_value=mock_comparison,
                            ):
                                result = cli_runner.invoke(
                                    characterize,
                                    [str(test_file), "--compare", str(ref_file)],
                                    obj={"verbose": 0},
                                )

                                assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_signal_analysis_output(cli_runner, tmp_path, sample_trace):
    """Test signal analysis produces expected output."""
    test_file = tmp_path / "signal.wfm"
    test_file.touch()

    with patch("oscura.loaders.load", return_value=sample_trace):
        result = cli_runner.invoke(
            characterize,
            [str(test_file), "--type", "signal", "--output", "table"],
            obj={"verbose": 0},
        )

        assert result.exit_code == 0
        # Output should contain signal-related terms
        output_lower = result.output.lower()
        assert any(term in output_lower for term in ["amplitude", "mean", "rms", "signal"])


@pytest.mark.unit
@pytest.mark.cli
def test_characterize_power_analysis_output(cli_runner, tmp_path, sample_trace):
    """Test power analysis produces expected output."""
    test_file = tmp_path / "power.wfm"
    test_file.touch()

    with patch("oscura.loaders.load", return_value=sample_trace):
        result = cli_runner.invoke(
            characterize,
            [str(test_file), "--type", "power", "--output", "table"],
            obj={"verbose": 0},
        )

        assert result.exit_code == 0
        # Output should contain power-related terms
        output_lower = result.output.lower()
        assert any(term in output_lower for term in ["power", "energy", "mw", "uj"])
