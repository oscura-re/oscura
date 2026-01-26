"""Comprehensive unit tests for analyze.py CLI module.

This module provides extensive testing for the analyze command, including:
- Command argument parsing and validation
- Protocol hint handling (auto-detection vs. specified)
- Export directory creation and management
- Interactive mode behavior
- Output format selection
- Session saving functionality
- Error handling and edge cases
- Internal workflow functions

Test Coverage:
- analyze() command with all options
- _perform_analysis_workflow() orchestration
- _characterize_signal() signal analysis
- _detect_protocol() protocol detection
- _decode_protocol() protocol decoding
- _export_results() result export
- _build_analysis_results() result assembly

References:
    - src/oscura/cli/analyze.py
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from oscura.cli.analyze import (
    _build_analysis_results,
    _characterize_signal,
    _decode_protocol,
    _detect_and_prepare_protocol,
    _detect_protocol,
    _export_results,
    _perform_analysis_workflow,
)
from oscura.cli.main import cli

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test Analyze Command
# =============================================================================


@pytest.mark.unit
def test_analyze_help():
    """Test analyze command --help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--help"])

    assert result.exit_code == 0
    assert "analyze" in result.output.lower()
    assert "waveform" in result.output.lower()
    assert "--protocol" in result.output
    assert "--export-dir" in result.output


@pytest.mark.unit
def test_analyze_missing_file():
    """Test analyze command with missing file argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze"])

    # Should fail with missing argument error
    assert result.exit_code != 0
    assert "FILE" in result.output or "Missing argument" in result.output


@pytest.mark.unit
def test_analyze_nonexistent_file():
    """Test analyze command with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "/nonexistent/file.wfm"])

    # Should fail with file not found error
    assert result.exit_code != 0


@pytest.mark.unit
def test_analyze_with_file(tmp_path, signal_factory):
    """Test analyze command with valid file."""
    runner = CliRunner()

    # Create a dummy waveform file
    wfm_file = tmp_path / "test.npy"
    signal, _ = signal_factory(signal_type="sine", duration=0.001)

    import numpy as np

    np.save(wfm_file, signal)

    with patch("oscura.cli.analyze._perform_analysis_workflow") as mock_workflow:
        mock_workflow.return_value = {
            "file": "test.npy",
            "sample_rate": "1.0 MHz",
            "samples": 1000,
        }

        result = runner.invoke(cli, ["analyze", str(wfm_file)])

        # Should call workflow
        assert mock_workflow.called


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.unit
def test_analyze_output_format_json():
    """Test analyze command with --output json."""
    runner = CliRunner()

    with patch("oscura.cli.analyze._perform_analysis_workflow") as mock_workflow:
        mock_workflow.return_value = {"key": "value", "count": 42}

        with runner.isolated_filesystem():
            Path("test.wfm").write_bytes(b"fake")

            result = runner.invoke(cli, ["analyze", "test.wfm", "--output", "json"])

            # Output should be JSON formatted
            assert "{" in result.output
            assert "key" in result.output


@pytest.mark.unit
@pytest.mark.unit
def test_analyze_verbose_mode():
    """Test analyze command with -v flag."""
    runner = CliRunner()

    with patch("oscura.cli.analyze._perform_analysis_workflow") as mock_workflow:
        mock_workflow.return_value = {}

        with runner.isolated_filesystem():
            Path("test.wfm").write_bytes(b"fake")

            result = runner.invoke(cli, ["-v", "analyze", "test.wfm"])

            # Should log file being analyzed
            # (logged before workflow call)
            assert mock_workflow.called


@pytest.mark.unit
def test_analyze_error_handling():
    """Test analyze command error handling."""
    runner = CliRunner()

    with patch("oscura.cli.analyze._perform_analysis_workflow") as mock_workflow:
        mock_workflow.side_effect = RuntimeError("Test error")

        with runner.isolated_filesystem():
            Path("test.wfm").write_bytes(b"fake")

            result = runner.invoke(cli, ["analyze", "test.wfm"])

            # Should exit with error
            assert result.exit_code == 1
            assert "Error:" in result.output or "error" in result.output.lower()


# =============================================================================
# Test _perform_analysis_workflow()
# =============================================================================


@pytest.mark.unit
def test_perform_analysis_workflow_basic(signal_factory):
    """Test basic analysis workflow execution."""
    signal, _ = signal_factory(signal_type="sine", duration=0.001)

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_and_prepare_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    # Setup mocks
                    mock_trace = Mock()
                    mock_trace.data = signal
                    mock_load.return_value = mock_trace

                    mock_char.return_value = {"sample_rate": "1.0 MHz"}
                    mock_detect.return_value = {"protocol": "uart"}
                    mock_decode.return_value = {"packets_decoded": 10}

                    result = _perform_analysis_workflow(
                        file="test.wfm",
                        protocol="auto",
                        export_dir=None,
                        interactive=False,
                        quiet=True,
                        save_session=None,
                    )

                    # Should have called all stages
                    assert mock_load.called
                    assert mock_char.called
                    assert mock_detect.called
                    assert mock_decode.called

                    # Result should contain all components
                    assert "sample_rate" in result
                    assert "protocol" in result
                    assert "packets_decoded" in result


@pytest.mark.unit
def test_perform_analysis_workflow_with_export(tmp_path, signal_factory):
    """Test analysis workflow with export directory."""
    signal, _ = signal_factory(signal_type="sine", duration=0.001)
    export_dir = tmp_path / "output"

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_and_prepare_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze._export_results") as mock_export:
                        mock_trace = Mock()
                        mock_trace.data = signal
                        mock_load.return_value = mock_trace

                        mock_char.return_value = {}
                        mock_detect.return_value = {"protocol": "uart"}
                        mock_decode.return_value = {}

                        result = _perform_analysis_workflow(
                            file="test.wfm",
                            protocol="auto",
                            export_dir=str(export_dir),
                            interactive=False,
                            quiet=True,
                            save_session=None,
                        )

                        # Should have called export
                        assert mock_export.called
                        assert "export_dir" in result


# =============================================================================
# Test _characterize_signal()
# =============================================================================


@pytest.mark.unit
def test_characterize_signal_basic(signal_factory):
    """Test basic signal characterization."""
    signal, _ = signal_factory(signal_type="sine", frequency=1000, duration=0.01)

    mock_trace = Mock()
    mock_trace.data = signal
    mock_trace.metadata = Mock()
    mock_trace.metadata.sample_rate = 1e6

    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=1e-9):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=1e-9):
            result = _characterize_signal(mock_trace)

            # Should return characterization dict
            assert "sample_rate" in result
            assert "samples" in result
            assert "duration" in result
            assert "amplitude" in result
            assert "rise_time" in result
            assert "fall_time" in result


@pytest.mark.unit
def test_characterize_signal_with_nan_times(signal_factory):
    """Test signal characterization with NaN rise/fall times."""
    import numpy as np

    signal, _ = signal_factory(signal_type="sine", duration=0.001)

    mock_trace = Mock()
    mock_trace.data = signal
    mock_trace.metadata = Mock()
    mock_trace.metadata.sample_rate = 1e6

    with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=np.nan):
        with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=np.nan):
            result = _characterize_signal(mock_trace)

            # Should return "N/A" for NaN values
            assert result["rise_time"] == "N/A"
            assert result["fall_time"] == "N/A"


# =============================================================================
# Test _detect_protocol()
# =============================================================================


@pytest.mark.unit
def test_detect_protocol_high_confidence():
    """Test protocol detection with high confidence."""
    mock_trace = Mock()

    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        mock_detect.return_value = {
            "protocol": "uart",
            "confidence": 0.95,
            "candidates": [],
        }

        result = _detect_protocol(mock_trace, interactive=False)

        assert result["protocol"] == "uart"
        assert result["confidence"] == 0.95


@pytest.mark.unit
def test_detect_protocol_interactive_low_confidence():
    """Test protocol detection in interactive mode with low confidence."""
    mock_trace = Mock()

    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("click.confirm", return_value=True):
            mock_detect.return_value = {
                "protocol": "spi",
                "confidence": 0.6,
                "candidates": [
                    {"protocol": "uart", "confidence": 0.55},
                ],
            }

            result = _detect_protocol(mock_trace, interactive=True)

            # Should accept the detection
            assert result["protocol"] == "spi"


@pytest.mark.unit
def test_detect_protocol_interactive_rejection():
    """Test protocol detection rejection in interactive mode."""
    mock_trace = Mock()

    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("click.confirm", return_value=False):
            with patch("click.prompt", return_value=1):
                mock_detect.return_value = {
                    "protocol": "spi",
                    "confidence": 0.6,
                    "candidates": [
                        {"protocol": "uart", "confidence": 0.55},
                        {"protocol": "i2c", "confidence": 0.50},
                    ],
                }

                result = _detect_protocol(mock_trace, interactive=True)

                # Should select candidate
                assert result["protocol"] == "uart"


# =============================================================================
# Test _decode_protocol()
# =============================================================================


@pytest.mark.unit
def test_decode_protocol_uart(signal_factory):
    """Test UART protocol decoding."""
    signal, _ = signal_factory(signal_type="digital", duration=0.01)

    mock_trace = Mock()
    mock_trace.data = signal
    mock_trace.metadata = Mock()
    mock_trace.metadata.sample_rate = 1e6

    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_packet = Mock()
        mock_packet.errors = []
        mock_decoder.decode.return_value = [mock_packet, mock_packet]
        mock_decoder_class.return_value = mock_decoder

        result = _decode_protocol(mock_trace, "uart")

        assert result["packets_decoded"] == 2
        assert result["errors"] == 0


@pytest.mark.unit
def test_decode_protocol_spi(signal_factory):
    """Test SPI protocol decoding."""
    signal, _ = signal_factory(signal_type="digital", duration=0.01)

    mock_trace = Mock()
    mock_trace.data = signal
    mock_trace.metadata = Mock()
    mock_trace.metadata.sample_rate = 1e6

    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_packet = Mock()
        mock_packet.errors = []
        mock_decoder.decode.return_value = [mock_packet]
        mock_decoder_class.return_value = mock_decoder

        result = _decode_protocol(mock_trace, "spi")

        assert "packets_decoded" in result


@pytest.mark.unit
def test_decode_protocol_with_errors(signal_factory):
    """Test protocol decoding with packet errors."""
    signal, _ = signal_factory(signal_type="digital", duration=0.01)

    mock_trace = Mock()
    mock_trace.data = signal
    mock_trace.metadata = Mock()
    mock_trace.metadata.sample_rate = 1e6

    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()

        # Create packets with errors
        mock_packet1 = Mock()
        mock_packet1.errors = ["parity_error"]
        mock_packet2 = Mock()
        mock_packet2.errors = []

        mock_decoder.decode.return_value = [mock_packet1, mock_packet2]
        mock_decoder_class.return_value = mock_decoder

        result = _decode_protocol(mock_trace, "uart")

        assert result["packets_decoded"] == 2
        assert result["errors"] == 1


# =============================================================================
# Test _export_results()
# =============================================================================


@pytest.mark.unit
def test_export_results_creates_files(tmp_path):
    """Test that export creates JSON and HTML files."""
    results = {"key": "value", "count": 42}
    export_dir = tmp_path / "output"
    export_dir.mkdir()

    _export_results(results, export_dir)

    # Should create JSON file
    json_file = export_dir / "analysis_results.json"
    assert json_file.exists()

    # Should create HTML file
    html_file = export_dir / "analysis_report.html"
    assert html_file.exists()


@pytest.mark.unit
def test_export_results_json_content(tmp_path):
    """Test exported JSON file contains correct data."""
    import json

    results = {"test_key": "test_value", "number": 123}
    export_dir = tmp_path / "output"
    export_dir.mkdir()

    _export_results(results, export_dir)

    json_file = export_dir / "analysis_results.json"
    with open(json_file) as f:
        loaded = json.load(f)

    assert loaded["test_key"] == "test_value"
    assert loaded["number"] == 123


@pytest.mark.unit
@pytest.mark.unit
def test_build_analysis_results_basic():
    """Test building analysis results without export."""
    signal_char = {"sample_rate": "1.0 MHz", "samples": 1000}
    protocol_info = {"protocol": "uart"}
    decoded = {"packets_decoded": 10}

    result = _build_analysis_results(
        file="test.wfm",
        signal_char=signal_char,
        protocol_info=protocol_info,
        decoded=decoded,
        export_dir=None,
        save_session=None,
        trace=Mock(),
    )

    assert result["file"] == "test.wfm"
    assert result["sample_rate"] == "1.0 MHz"
    assert result["protocol"] == "uart"
    assert result["packets_decoded"] == 10


@pytest.mark.unit
def test_build_analysis_results_with_export(tmp_path):
    """Test building results with export directory."""
    export_dir = tmp_path / "output"

    with patch("oscura.cli.analyze._export_results"):
        result = _build_analysis_results(
            file="test.wfm",
            signal_char={},
            protocol_info={},
            decoded={},
            export_dir=str(export_dir),
            save_session=None,
            trace=Mock(),
        )

        # Should create export directory
        assert export_dir.exists()
        assert result["export_dir"] == str(export_dir)


@pytest.mark.unit
@pytest.mark.unit
def test_detect_and_prepare_protocol_auto():
    """Test protocol detection when protocol is 'auto'."""
    mock_trace = Mock()

    with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
        mock_detect.return_value = {"protocol": "i2c", "confidence": 0.9}

        result = _detect_and_prepare_protocol(mock_trace, "auto", interactive=False)

        assert result["protocol"] == "i2c"
        assert mock_detect.called


@pytest.mark.unit
def test_detect_and_prepare_protocol_specified():
    """Test when protocol is specified (not auto)."""
    mock_trace = Mock()

    result = _detect_and_prepare_protocol(mock_trace, "uart", interactive=False)

    assert result["protocol"] == "uart"


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_analyze_all_output_formats(tmp_path):
    """Test analyze command with all output format options."""
    runner = CliRunner()

    formats = ["json", "csv", "html", "table"]

    for fmt in formats:
        with patch("oscura.cli.analyze._perform_analysis_workflow") as mock_workflow:
            mock_workflow.return_value = {"result": "data"}

            wfm_file = tmp_path / f"test_{fmt}.wfm"
            wfm_file.write_bytes(b"fake")

            result = runner.invoke(cli, ["analyze", str(wfm_file), "--output", fmt])

            # Should not crash
            assert "result" in result.output or "data" in result.output or result.exit_code == 0
