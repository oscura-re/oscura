"""Comprehensive unit tests for analyze.py CLI module.

This module provides extensive testing for the Oscura analyze command, including:
- Full analysis workflow orchestration
- Protocol detection and decoding
- Signal characterization
- Session export and saving
- Interactive mode
- Progress reporting
- Error handling

Test Coverage:
- analyze() CLI command with all options
- _characterize_signal() signal property extraction
- _detect_protocol() auto-detection with interactive prompts
- _decode_protocol() protocol-specific decoding
- _export_results() multi-format export
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.analyze import (
    _characterize_signal,
    _decode_protocol,
    _detect_protocol,
    _export_results,
    analyze,
)
from oscura.core.types import DigitalTrace, ProtocolPacket, TraceMetadata, WaveformTrace

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.usefixtures("reset_logging_state"),
]


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
def sample_waveform_trace(sample_metadata):
    """Create sample waveform trace with transitions."""
    # Create signal with transitions for rise/fall time
    data = np.array([0.0] * 100 + [3.3] * 100 + [0.0] * 100, dtype=np.float64)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def sample_digital_trace(sample_metadata):
    """Create sample digital trace."""
    data = np.array([False, True, False, True] * 50, dtype=bool)
    return DigitalTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Test _characterize_signal()
# =============================================================================


@pytest.mark.unit
def test_characterize_signal_basic(sample_waveform_trace):
    """Test basic signal characterization."""
    with patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rt:
        with patch("oscura.analyzers.waveform.measurements.fall_time") as mock_ft:
            mock_rt.return_value = {"applicable": True, "value": 10e-9}  # 10 ns
            mock_ft.return_value = {"applicable": True, "value": 8e-9}  # 8 ns

            result = _characterize_signal(sample_waveform_trace)

            assert "sample_rate" in result
            assert "10.0 MHz" in result["sample_rate"]
            assert "samples" in result
            assert result["samples"] == 300
            assert "duration" in result
            assert "ms" in result["duration"]
            assert "amplitude" in result
            assert "rise_time" in result
            assert "10.00 ns" in result["rise_time"]
            assert "fall_time" in result
            assert "8.00 ns" in result["fall_time"]


@pytest.mark.unit
def test_characterize_signal_nan_edge_times(sample_waveform_trace):
    """Test characterization when edge times cannot be measured."""
    with patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rt:
        with patch("oscura.analyzers.waveform.measurements.fall_time") as mock_ft:
            mock_rt.return_value = {"applicable": False, "value": np.nan}
            mock_ft.return_value = {"applicable": False, "value": np.nan}

            result = _characterize_signal(sample_waveform_trace)

            assert result["rise_time"] == "N/A"
            assert result["fall_time"] == "N/A"


@pytest.mark.unit
def test_characterize_signal_amplitude_calculation(sample_metadata):
    """Test amplitude calculation from data min/max."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    trace = WaveformTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rt:
        with patch("oscura.analyzers.waveform.measurements.fall_time") as mock_ft:
            mock_rt.return_value = {"applicable": False, "value": np.nan}
            mock_ft.return_value = {"applicable": False, "value": np.nan}

            result = _characterize_signal(trace)

            # Amplitude should be max - min = 5.0 - 1.0 = 4.0
            assert "4.000 V" in result["amplitude"]


@pytest.mark.unit
def test_characterize_signal_duration_calculation(sample_metadata):
    """Test duration calculation from sample count and rate."""
    # 1000 samples at 1 MHz = 1 ms
    data = np.zeros(1000, dtype=np.float64)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = WaveformTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.waveform.measurements.rise_time") as mock_rt:
        with patch("oscura.analyzers.waveform.measurements.fall_time") as mock_ft:
            mock_rt.return_value = {"applicable": False, "value": np.nan}
            mock_ft.return_value = {"applicable": False, "value": np.nan}

            result = _characterize_signal(trace)

            assert "1.000 ms" in result["duration"]


# =============================================================================
# Test _detect_protocol()
# =============================================================================


@pytest.mark.unit
def test_detect_protocol_high_confidence(sample_waveform_trace):
    """Test protocol detection with high confidence (no interaction)."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        mock_detect.return_value = {
            "protocol": "UART",
            "confidence": 0.95,
            "candidates": [{"protocol": "UART", "confidence": 0.95}],
        }

        result = _detect_protocol(sample_waveform_trace, interactive=False)

        assert result["protocol"] == "UART"
        assert result["confidence"] == 0.95
        mock_detect.assert_called_once_with(
            sample_waveform_trace, min_confidence=0.5, return_candidates=True
        )


@pytest.mark.unit
def test_detect_protocol_interactive_accept(sample_waveform_trace, cli_runner):
    """Test interactive mode with user accepting detection."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("click.confirm") as mock_confirm:
            mock_detect.return_value = {
                "protocol": "SPI",
                "confidence": 0.7,  # Below 0.9 triggers interaction
                "candidates": [{"protocol": "SPI", "confidence": 0.7}],
            }
            mock_confirm.return_value = True  # User accepts

            result = _detect_protocol(sample_waveform_trace, interactive=True)

            assert result["protocol"] == "SPI"
            assert result["confidence"] == 0.7
            mock_confirm.assert_called_once()


@pytest.mark.unit
def test_detect_protocol_interactive_reject_choose_candidate(sample_waveform_trace):
    """Test interactive mode with user selecting different candidate."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("click.confirm") as mock_confirm:
            with patch("click.prompt") as mock_prompt:
                with patch("click.echo"):
                    mock_detect.return_value = {
                        "protocol": "SPI",
                        "confidence": 0.6,
                        "candidates": [
                            {"protocol": "SPI", "confidence": 0.6},
                            {"protocol": "I2C", "confidence": 0.5},
                            {"protocol": "UART", "confidence": 0.4},
                        ],
                    }
                    mock_confirm.return_value = False  # User rejects
                    mock_prompt.return_value = 2  # Select candidate 2 (I2C)

                    result = _detect_protocol(sample_waveform_trace, interactive=True)

                    assert result["protocol"] == "I2C"
                    assert result["confidence"] == 0.5


@pytest.mark.unit
def test_detect_protocol_interactive_manual_entry(sample_waveform_trace):
    """Test interactive mode with manual protocol entry."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("click.confirm") as mock_confirm:
            with patch("click.prompt") as mock_prompt:
                with patch("click.echo"):
                    mock_detect.return_value = {
                        "protocol": "SPI",
                        "confidence": 0.6,
                        "candidates": [{"protocol": "SPI", "confidence": 0.6}],
                    }
                    mock_confirm.return_value = False
                    # First prompt returns 0 (manual), second returns protocol name
                    mock_prompt.side_effect = [0, "CAN"]

                    result = _detect_protocol(sample_waveform_trace, interactive=True)

                    assert result["protocol"] == "CAN"
                    assert result["confidence"] == 1.0


@pytest.mark.unit
def test_detect_protocol_non_interactive(sample_waveform_trace):
    """Test non-interactive mode doesn't prompt even with low confidence."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("click.confirm") as mock_confirm:
            mock_detect.return_value = {
                "protocol": "UART",
                "confidence": 0.5,  # Low confidence
                "candidates": [],
            }

            result = _detect_protocol(sample_waveform_trace, interactive=False)

            # Should not prompt in non-interactive mode
            mock_confirm.assert_not_called()
            assert result["protocol"] == "UART"


# =============================================================================
# Test _decode_protocol()
# =============================================================================


@pytest.mark.unit
def test_decode_protocol_uart(sample_waveform_trace):
    """Test UART protocol decoding."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.001, protocol="UART", data=bytes([0x48]), errors=[], annotations={}
        )
        mock_decoder.decode.return_value = [packet]

        result = _decode_protocol(sample_waveform_trace, "uart")

        assert result["packets_decoded"] == 1
        assert result["errors"] == 0
        mock_decoder_class.assert_called_once_with(baudrate=0)  # Auto-detect


@pytest.mark.unit
def test_decode_protocol_uart_with_errors(sample_waveform_trace):
    """Test UART decoding counting errors."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packets = [
            ProtocolPacket(
                timestamp=0.001, protocol="UART", data=bytes([0x00]), errors=[], annotations={}
            ),
            ProtocolPacket(
                timestamp=0.002,
                protocol="UART",
                data=bytes([0x01]),
                errors=["parity_error"],
                annotations={},
            ),
            ProtocolPacket(
                timestamp=0.003,
                protocol="UART",
                data=bytes([0x02]),
                errors=["framing_error"],
                annotations={},
            ),
        ]
        mock_decoder.decode.return_value = packets

        result = _decode_protocol(sample_waveform_trace, "uart")

        assert result["packets_decoded"] == 3
        assert result["errors"] == 2


@pytest.mark.unit
def test_decode_protocol_spi(sample_waveform_trace):
    """Test SPI protocol decoding."""
    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.001, protocol="SPI", data=bytes([0xAA]), errors=[], annotations={}
        )
        mock_decoder.decode.return_value = [packet]

        result = _decode_protocol(sample_waveform_trace, "spi")

        assert result["packets_decoded"] == 1
        assert result["errors"] == 0


@pytest.mark.unit
def test_decode_protocol_i2c(sample_waveform_trace):
    """Test I2C protocol decoding."""
    with patch("oscura.analyzers.protocols.i2c.I2CDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packets = [
            ProtocolPacket(
                timestamp=0.001, protocol="I2C", data=bytes([0x50]), errors=[], annotations={}
            ),
            ProtocolPacket(
                timestamp=0.002, protocol="I2C", data=bytes([0x51]), errors=[], annotations={}
            ),
        ]
        mock_decoder.decode.return_value = packets

        result = _decode_protocol(sample_waveform_trace, "i2c")

        assert result["packets_decoded"] == 2


@pytest.mark.unit
def test_decode_protocol_digital_trace_passthrough(sample_digital_trace):
    """Test that digital traces are used directly without conversion."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        _decode_protocol(sample_digital_trace, "uart")

        # Should pass digital trace directly to decoder
        call_args = mock_decoder.decode.call_args
        assert isinstance(call_args[0][0], DigitalTrace)


@pytest.mark.unit
def test_decode_protocol_converts_waveform_to_digital(sample_waveform_trace):
    """Test waveform is converted to digital for decoding."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        _decode_protocol(sample_waveform_trace, "uart")

        # Should convert to digital and pass to decoder
        call_args = mock_decoder.decode.call_args
        assert isinstance(call_args[0][0], DigitalTrace)


@pytest.mark.unit
def test_decode_protocol_threshold_calculation(sample_metadata):
    """Test midpoint threshold calculation for digital conversion."""
    # Data with clear min/max
    data = np.array([0.0, 0.0, 5.0, 5.0, 0.0], dtype=np.float64)
    trace = WaveformTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        _decode_protocol(trace, "uart")

        # Check digital conversion used midpoint threshold
        call_args = mock_decoder.decode.call_args
        digital_trace = call_args[0][0]
        # Threshold should be (0 + 5) / 2 = 2.5
        # So pattern should be [False, False, True, True, False]
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(digital_trace.data, expected)


# =============================================================================
# Test _export_results()
# =============================================================================


@pytest.mark.unit
def test_export_results_creates_json(tmp_path):
    """Test exporting results to JSON file."""
    results = {
        "file": "test.wfm",
        "protocol": "UART",
        "packets_decoded": 5,
        "sample_rate": "10.0 MHz",
    }

    _export_results(results, tmp_path)

    json_file = tmp_path / "analysis_results.json"
    assert json_file.exists()

    import json

    with open(json_file) as f:
        loaded = json.load(f)

    assert loaded["protocol"] == "UART"
    assert loaded["packets_decoded"] == 5


@pytest.mark.unit
def test_export_results_creates_html(tmp_path):
    """Test exporting results to HTML file."""
    results = {"file": "test.wfm", "protocol": "SPI", "packets_decoded": 10}

    _export_results(results, tmp_path)

    html_file = tmp_path / "analysis_report.html"
    assert html_file.exists()

    content = html_file.read_text()
    assert "<!DOCTYPE html>" in content
    assert "SPI" in content
    assert "10" in content


@pytest.mark.unit
def test_export_results_uses_format_output(tmp_path):
    """Test that export uses format_output for HTML generation."""
    results = {"key": "value"}

    with patch("oscura.cli.analyze.format_output") as mock_format:
        mock_format.return_value = "<html>test</html>"

        _export_results(results, tmp_path)

        mock_format.assert_called_once_with(results, "html")


# =============================================================================
# Test analyze() CLI command
# =============================================================================


@pytest.mark.unit
def test_analyze_command_basic(cli_runner, tmp_path):
    """Test basic analyze command execution."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.format_output") as mock_format:
                        mock_load.return_value = Mock()
                        mock_char.return_value = {"sample_rate": "10.0 MHz"}
                        mock_detect.return_value = {"protocol": "UART"}
                        mock_decode.return_value = {"packets_decoded": 5, "errors": 0}
                        mock_format.return_value = "Analysis complete"

                        result = cli_runner.invoke(
                            analyze, [str(test_file)], obj={"verbose": 0, "quiet": False}
                        )

                        assert result.exit_code == 0
                        assert "Analysis complete" in result.output


@pytest.mark.unit
def test_analyze_command_with_protocol_hint(cli_runner, tmp_path):
    """Test analyze with explicit protocol hint."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.format_output"):
                        mock_load.return_value = Mock()
                        mock_char.return_value = {}
                        mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                        result = cli_runner.invoke(
                            analyze,
                            [str(test_file), "--protocol", "spi"],
                            obj={"verbose": 0, "quiet": False},
                        )

                        assert result.exit_code == 0
                        # Should not call auto-detection
                        mock_detect.assert_not_called()
                        # Should decode with SPI
                        mock_decode.assert_called_once()
                        assert mock_decode.call_args[0][1] == "spi"


@pytest.mark.unit
def test_analyze_command_with_export(cli_runner, tmp_path):
    """Test analyze with export directory."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()
    export_dir = tmp_path / "output"

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze._export_results") as mock_export:
                        with patch("oscura.cli.analyze.format_output"):
                            mock_load.return_value = Mock()
                            mock_char.return_value = {}
                            mock_detect.return_value = {"protocol": "UART"}
                            mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                            result = cli_runner.invoke(
                                analyze,
                                [str(test_file), "--export-dir", str(export_dir)],
                                obj={"verbose": 0, "quiet": False},
                            )

                            assert result.exit_code == 0
                            # Export directory should be created
                            assert export_dir.exists()
                            # Export should be called
                            mock_export.assert_called_once()


@pytest.mark.unit
def test_analyze_command_with_save_session(cli_runner, tmp_path):
    """Test analyze with session saving."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()
    session_file = tmp_path / "session.tks"

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.Session") as mock_session_class:
                        with patch("oscura.cli.analyze.format_output"):
                            mock_trace = Mock()
                            mock_load.return_value = mock_trace
                            mock_char.return_value = {}
                            mock_detect.return_value = {"protocol": "UART"}
                            mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                            # Mock session with proper metadata attribute
                            mock_session = Mock()
                            mock_session.metadata = {}
                            mock_session_class.return_value = mock_session

                            result = cli_runner.invoke(
                                analyze,
                                [str(test_file), "--save-session", str(session_file)],
                                obj={"verbose": 0, "quiet": False},
                            )

                            assert result.exit_code == 0
                            # Session should be created and saved
                            mock_session.add_trace.assert_called_once_with("main", mock_trace)
                            mock_session.save.assert_called_once_with(str(session_file))


@pytest.mark.unit
def test_analyze_command_interactive_mode(cli_runner, tmp_path):
    """Test analyze in interactive mode."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.format_output"):
                        mock_load.return_value = Mock()
                        mock_char.return_value = {}
                        mock_detect.return_value = {"protocol": "UART"}
                        mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                        result = cli_runner.invoke(
                            analyze,
                            [str(test_file), "--interactive"],
                            obj={"verbose": 0, "quiet": False},
                        )

                        assert result.exit_code == 0
                        # Should call detect with interactive=True
                        mock_detect.assert_called_once()
                        assert mock_detect.call_args[1]["interactive"] is True


@pytest.mark.unit
def test_analyze_command_output_formats(cli_runner, tmp_path):
    """Test analyze with different output formats."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    for output_format in ["json", "csv", "html", "table"]:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.analyze._characterize_signal") as mock_char:
                with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                    with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                        with patch("oscura.cli.analyze.format_output") as mock_format:
                            mock_load.return_value = Mock()
                            mock_char.return_value = {}
                            mock_detect.return_value = {"protocol": "UART"}
                            mock_decode.return_value = {"packets_decoded": 0, "errors": 0}
                            mock_format.return_value = f"{output_format} output"

                            result = cli_runner.invoke(
                                analyze,
                                [str(test_file), "--output", output_format],
                                obj={"verbose": 0, "quiet": False},
                            )

                            assert result.exit_code == 0
                            assert f"{output_format} output" in result.output


@pytest.mark.unit
def test_analyze_command_progress_reporting(cli_runner, tmp_path):
    """Test that analyze uses progress reporter."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.ProgressReporter") as mock_progress_class:
                        with patch("oscura.cli.analyze.format_output"):
                            mock_load.return_value = Mock()
                            mock_char.return_value = {}
                            mock_detect.return_value = {"protocol": "UART"}
                            mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                            mock_progress = Mock()
                            mock_progress_class.return_value = mock_progress

                            result = cli_runner.invoke(
                                analyze, [str(test_file)], obj={"verbose": 0, "quiet": False}
                            )

                            assert result.exit_code == 0
                            # Progress reporter should be created with 5 stages
                            mock_progress_class.assert_called_once_with(quiet=False, stages=5)
                            # All stages should be started and completed
                            assert mock_progress.start_stage.call_count == 5
                            assert mock_progress.complete_stage.call_count == 5
                            mock_progress.finish.assert_called_once()


@pytest.mark.unit
def test_analyze_command_quiet_mode(cli_runner, tmp_path):
    """Test analyze in quiet mode suppresses progress."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.ProgressReporter") as mock_progress_class:
                        with patch("oscura.cli.analyze.format_output"):
                            mock_load.return_value = Mock()
                            mock_char.return_value = {}
                            mock_detect.return_value = {"protocol": "UART"}
                            mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                            result = cli_runner.invoke(
                                analyze, [str(test_file)], obj={"verbose": 0, "quiet": True}
                            )

                            assert result.exit_code == 0
                            # Progress should be created with quiet=True
                            mock_progress_class.assert_called_once_with(quiet=True, stages=5)


@pytest.mark.unit
def test_analyze_command_error_handling(cli_runner, tmp_path):
    """Test analyze error handling."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = Exception("Failed to load")

        result = cli_runner.invoke(analyze, [str(test_file)], obj={"verbose": 0, "quiet": False})

        assert result.exit_code == 1
        assert "Error: Failed to load" in result.output


@pytest.mark.unit
def test_analyze_command_error_with_verbose(cli_runner, tmp_path):
    """Test analyze error handling with verbose mode (should raise)."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = ValueError("Test error")

        result = cli_runner.invoke(analyze, [str(test_file)], obj={"verbose": 2, "quiet": False})

        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
def test_analyze_command_verbose_logging(cli_runner, tmp_path, caplog):
    """Test analyze with verbose logging."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.format_output"):
                        mock_load.return_value = Mock()
                        mock_char.return_value = {}
                        mock_detect.return_value = {"protocol": "UART"}
                        mock_decode.return_value = {"packets_decoded": 0, "errors": 0}

                        result = cli_runner.invoke(
                            analyze, [str(test_file)], obj={"verbose": 1, "quiet": False}
                        )

                        assert result.exit_code == 0


@pytest.mark.unit
def test_analyze_adds_filename_to_results(cli_runner, tmp_path):
    """Test that analyze adds filename to results."""
    test_file = tmp_path / "my_signal.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.analyze._characterize_signal") as mock_char:
            with patch("oscura.cli.analyze._detect_protocol") as mock_detect:
                with patch("oscura.cli.analyze._decode_protocol") as mock_decode:
                    with patch("oscura.cli.analyze.format_output") as mock_format:
                        mock_load.return_value = Mock()
                        mock_char.return_value = {}
                        mock_detect.return_value = {"protocol": "UART"}
                        mock_decode.return_value = {"packets_decoded": 0, "errors": 0}
                        mock_format.return_value = "output"

                        result = cli_runner.invoke(
                            analyze, [str(test_file)], obj={"verbose": 0, "quiet": False}
                        )

                        assert result.exit_code == 0
                        # Check that file key was added
                        format_call_args = mock_format.call_args[0][0]
                        assert format_call_args["file"] == "my_signal.wfm"
