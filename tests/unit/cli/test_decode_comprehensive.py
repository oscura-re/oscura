"""Comprehensive unit tests for decode.py CLI module.

This module provides extensive testing for the decode command, including:
- Command argument parsing and validation
- Protocol selection (uart, spi, i2c, can, auto)
- UART-specific options (baud rate, parity, stop bits)
- Show errors flag
- Output format selection
- Digital trace conversion
- Protocol-specific decoding
- Error detection and reporting

Test Coverage:
- decode() command with all options
- _to_digital() trace conversion
- _perform_decoding() orchestration
- _decode_uart() UART decoding
- _decode_spi() SPI decoding
- _decode_i2c() I2C decoding
- _decode_can() CAN decoding
- Helper functions for each protocol

References:
    - src/oscura/cli/decode.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.decode import (
    _build_base_results,
    _decode_can,
    _decode_i2c,
    _decode_spi,
    _decode_uart,
    _detect_can_baud_rate,
    _detect_protocol_if_auto,
    _dispatch_protocol_decode,
    _perform_decoding,
    _to_digital,
)
from oscura.cli.main import cli
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test Decode Command
# =============================================================================


@pytest.mark.unit
def test_decode_help():
    """Test decode command --help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["decode", "--help"])

    assert result.exit_code == 0
    assert "decode" in result.output.lower()
    assert "--protocol" in result.output
    assert "--baud-rate" in result.output


@pytest.mark.unit
def test_decode_missing_file():
    """Test decode command with missing file argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["decode"])

    assert result.exit_code != 0


@pytest.mark.unit
def test_decode_nonexistent_file():
    """Test decode command with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["decode", "/nonexistent/file.wfm"])

    assert result.exit_code != 0


@pytest.mark.unit
def test_decode_with_auto_protocol(tmp_path, signal_factory):
    """Test decode with auto protocol detection."""
    runner = CliRunner()

    # Use .npz which is a supported format, not .npy
    wfm_file = tmp_path / "test.npz"
    signal, _ = signal_factory(signal_type="digital", duration=0.001)
    metadata = TraceMetadata(sample_rate=1e6)
    np.savez(wfm_file, data=signal, sample_rate=metadata.sample_rate)

    with patch("oscura.cli.decode._perform_decoding") as mock_decode:
        mock_decode.return_value = {"protocol": "UART", "packets_decoded": 5}

        result = runner.invoke(cli, ["decode", str(wfm_file), "--protocol", "auto"])

        assert mock_decode.called


@pytest.mark.unit
def test_decode_with_uart_options(tmp_path):
    """Test decode with UART-specific options."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.wfm").write_bytes(b"fake")

        with patch("oscura.loaders.load") as mock_load:
            data = np.array([True, False] * 100, dtype=bool)
            metadata = TraceMetadata(sample_rate=1e6)
            mock_load.return_value = DigitalTrace(data=data, metadata=metadata)

            with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                mock_decode.return_value = {"protocol": "UART"}

                result = runner.invoke(
                    cli,
                    [
                        "decode",
                        "test.wfm",
                        "--protocol",
                        "uart",
                        "--baud-rate",
                        "115200",
                        "--parity",
                        "even",
                        "--stop-bits",
                        "2",
                    ],
                )

            call_args = mock_decode.call_args
            assert call_args[1]["baud_rate"] == 115200
            assert call_args[1]["parity"] == "even"
            assert call_args[1]["stop_bits"] == 2


@pytest.mark.unit
def test_decode_show_errors_flag():
    """Test decode with --show-errors flag."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.wfm").write_bytes(b"fake")

        with patch("oscura.loaders.load") as mock_load:
            data = np.array([True, False] * 100, dtype=bool)
            metadata = TraceMetadata(sample_rate=1e6)
            mock_load.return_value = DigitalTrace(data=data, metadata=metadata)

            with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                mock_decode.return_value = {}

                result = runner.invoke(cli, ["decode", "test.wfm", "--show-errors"])

                call_args = mock_decode.call_args
                assert call_args is not None  # Make sure it was called
                assert call_args[1]["show_errors"] is True


@pytest.mark.unit
def test_decode_output_format_json():
    """Test decode with --output json."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.wfm").write_bytes(b"fake")

        # Need to patch load() because it's called before _perform_decoding
        with patch("oscura.loaders.load") as mock_load:
            data = np.array([True, False] * 100, dtype=bool)
            metadata = TraceMetadata(sample_rate=1e6)
            mock_load.return_value = DigitalTrace(data=data, metadata=metadata)

            with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                mock_decode.return_value = {"packets": 10}

                result = runner.invoke(cli, ["decode", "test.wfm", "--output", "json"])

                assert "{" in result.output


@pytest.mark.unit
def test_decode_error_handling():
    """Test decode command error handling."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("test.wfm").write_bytes(b"fake")

        with patch("oscura.loaders.load") as mock_load:
            mock_load.side_effect = RuntimeError("Test error")

            result = runner.invoke(cli, ["decode", "test.wfm"])

            assert result.exit_code == 1
            assert "Error:" in result.output or "error" in result.output.lower()


# =============================================================================
# Test _to_digital()
# =============================================================================


@pytest.mark.unit
def test_to_digital_from_waveform():
    """Test converting waveform trace to digital."""
    data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    metadata = TraceMetadata(sample_rate=1e6)
    trace = WaveformTrace(data=data, metadata=metadata)

    digital = _to_digital(trace)

    assert isinstance(digital, DigitalTrace)
    assert digital.data.dtype == bool
    # Threshold at 1.5 (midpoint), using > comparison, so values > 1.5 should be True
    expected = np.array([False, False, False, False, True, True, True])
    np.testing.assert_array_equal(digital.data, expected)


@pytest.mark.unit
def test_to_digital_from_digital():
    """Test that digital trace is returned as-is."""
    data = np.array([True, False, True, False])
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    digital = _to_digital(trace)

    # Should be the same object
    assert digital is trace


# =============================================================================
# Test _build_base_results()
# =============================================================================


@pytest.mark.unit
def test_build_base_results():
    """Test building base results dictionary."""
    data = np.zeros(10000)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = WaveformTrace(data=data, metadata=metadata)

    result = _build_base_results(trace)

    assert "sample_rate" in result
    assert "1.0 MHz" in result["sample_rate"]
    assert result["samples"] == 10000
    assert "duration" in result


# =============================================================================
# Test _detect_protocol_if_auto()
# =============================================================================


@pytest.mark.unit
def test_detect_protocol_if_auto_specified():
    """Test protocol detection when protocol is specified."""
    mock_trace = Mock()

    protocol, baud_rate = _detect_protocol_if_auto(mock_trace, "uart", 9600, {})

    assert protocol == "uart"
    assert baud_rate == 9600


@pytest.mark.unit
def test_detect_protocol_if_auto_detection():
    """Test automatic protocol detection."""
    mock_trace = Mock()

    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        mock_detect.return_value = {
            "protocol": "SPI",
            "confidence": 0.9,
            "config": {"clock_rate": 1e6},
        }

        results: dict[str, Any] = {}
        protocol, baud_rate = _detect_protocol_if_auto(mock_trace, "auto", None, results)

        assert protocol == "spi"
        assert "auto_detection" in results


@pytest.mark.unit
def test_detect_protocol_if_auto_extracts_uart_config():
    """Test that auto-detection extracts UART baud rate from config."""
    mock_trace = Mock()

    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        mock_detect.return_value = {
            "protocol": "UART",
            "confidence": 0.95,
            "config": {"baud_rate": 115200},
        }

        results: dict[str, Any] = {}
        protocol, baud_rate = _detect_protocol_if_auto(mock_trace, "auto", None, results)

        assert protocol == "uart"
        assert baud_rate == 115200


# =============================================================================
# Test _decode_uart()
# =============================================================================


@pytest.mark.unit
def test_decode_uart_basic():
    """Test basic UART decoding."""
    data = np.array([True, False, True] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    # Patch where it's imported (inside _decode_uart function)
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder._baudrate = 9600  # Set internal baudrate attribute
        mock_packet = Mock()
        mock_packet.errors = []
        mock_packet.data = b"H"
        mock_decoder.decode.return_value = [mock_packet]
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_uart(
            trace, baud_rate=9600, parity="none", stop_bits=1, show_errors=False
        )

        assert len(packets) == 1
        assert len(errors) == 0
        assert info["baud_rate"] == 9600
        assert info["parity"] == "none"


@pytest.mark.unit
def test_decode_uart_with_errors():
    """Test UART decoding with packet errors."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()

        # Create packet with error
        mock_packet = Mock()
        mock_packet.errors = ["parity_error"]
        mock_packet.data = b"X"
        mock_packet.timestamp = 0.001

        mock_decoder.decode.return_value = [mock_packet]
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_uart(
            trace, baud_rate=9600, parity="even", stop_bits=1, show_errors=False
        )

        assert len(errors) == 1
        assert errors[0]["type"] == "parity_error"


@pytest.mark.unit
def test_decode_uart_auto_baud():
    """Test UART decoding with auto baud rate detection."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder._baudrate = 115200  # Simulated auto-detected rate
        mock_decoder.decode.return_value = []
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_uart(
            trace, baud_rate=None, parity="none", stop_bits=1, show_errors=False
        )

        # Should have auto-detected baud rate
        assert info["baud_rate"] == 115200


# =============================================================================
# Test _decode_spi()
# =============================================================================


@pytest.mark.unit
def test_decode_spi_basic():
    """Test basic SPI decoding."""
    data = np.array([True, False, True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_packet = Mock()
        mock_packet.errors = []
        mock_packet.timestamp = 0.001
        mock_decoder.decode.return_value = [mock_packet]
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_spi(trace, show_errors=False)

        assert len(packets) == 1
        assert "clock_frequency" in info
        assert "mode" in info


@pytest.mark.unit
def test_decode_spi_estimates_clock():
    """Test SPI decoding estimates clock frequency."""
    # Create data with regular edges
    data = np.array([True, False] * 50, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder.decode.return_value = []
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_spi(trace, show_errors=False)

        # Should have estimated clock frequency
        assert "clock_frequency" in info


# =============================================================================
# Test _decode_i2c()
# =============================================================================


@pytest.mark.unit
def test_decode_i2c_basic():
    """Test basic I2C decoding."""
    # Create data with sufficient edges
    data = np.array([i % 10 < 5 for i in range(500)], dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.i2c.I2CDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_packet = Mock()
        mock_packet.errors = []
        mock_packet.annotations = {"address": 0x50}
        mock_decoder.decode.return_value = [mock_packet]
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_i2c(trace, show_errors=False)

        assert len(packets) == 1
        assert "addresses_seen" in info


@pytest.mark.unit
def test_decode_i2c_insufficient_edges():
    """Test I2C decoding with insufficient edges."""
    # Only a few samples
    data = np.array([True, False, True], dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    packets, errors, info = _decode_i2c(trace, show_errors=False)

    # Should return error
    assert "error" in info


# =============================================================================
# Test _decode_can()
# =============================================================================


@pytest.mark.unit
def test_decode_can_with_baud_rate():
    """Test CAN decoding with specified baud rate."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_packet = Mock()
        mock_packet.errors = []
        mock_packet.annotations = {"arbitration_id": 0x123, "is_extended": False}
        mock_decoder.decode.return_value = [mock_packet]
        mock_decoder_class.return_value = mock_decoder

        packets, errors, info = _decode_can(trace, baud_rate=500000, show_errors=False)

        assert info["bit_rate"] == "500 kbps"


@pytest.mark.unit
def test_decode_can_auto_baud():
    """Test CAN decoding with automatic baud rate detection."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.cli.decode._detect_can_baud_rate") as mock_detect:
        with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:
            mock_detect.return_value = 250000
            mock_decoder = Mock()
            mock_decoder.decode.return_value = []
            mock_decoder_class.return_value = mock_decoder

            packets, errors, info = _decode_can(trace, baud_rate=None, show_errors=False)

            assert mock_detect.called
            assert "250 kbps" in info["bit_rate"]


# =============================================================================
# Test _detect_can_baud_rate()
# =============================================================================


@pytest.mark.unit
def test_detect_can_baud_rate():
    """Test CAN baud rate detection."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:

        def make_decoder(bitrate):
            decoder = Mock()
            # Return more packets for 500k rate
            if bitrate == 500000:
                decoder.decode.return_value = [Mock(), Mock(), Mock()]
            else:
                decoder.decode.return_value = [Mock()]
            return decoder

        mock_decoder_class.side_effect = make_decoder

        result = _detect_can_baud_rate(trace)

        # Should return 500000 (most packets)
        assert result == 500000


# =============================================================================
# Test _dispatch_protocol_decode()
# =============================================================================


@pytest.mark.unit
def test_dispatch_protocol_decode_uart():
    """Test protocol dispatch to UART decoder."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        mock_uart.return_value = ([], [], {"protocol": "UART"})

        results: dict[str, Any] = {}
        packets, errors = _dispatch_protocol_decode(
            trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
            results=results,
        )

        assert mock_uart.called
        assert "protocol" in results


@pytest.mark.unit
def test_dispatch_protocol_decode_unknown():
    """Test protocol dispatch with unknown protocol."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    results: dict[str, Any] = {}
    packets, errors = _dispatch_protocol_decode(
        trace,
        protocol="unknown",
        baud_rate=None,
        parity="none",
        stop_bits=1,
        show_errors=False,
        results=results,
    )

    # Should return empty lists
    assert packets == []
    assert errors == []


# =============================================================================
# Test _perform_decoding()
# =============================================================================


@pytest.mark.unit
def test_perform_decoding_complete_flow():
    """Test complete decoding flow."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = WaveformTrace(data=data.astype(float) * 3.3, metadata=metadata)

    with patch("oscura.cli.decode._detect_protocol_if_auto") as mock_detect:
        with patch("oscura.cli.decode._dispatch_protocol_decode") as mock_dispatch:
            mock_detect.return_value = ("uart", 9600)

            mock_packet = Mock()
            mock_packet.errors = []
            mock_packet.timestamp = 0.001
            mock_packet.data = b"H"
            mock_packet.annotations = {}
            mock_dispatch.return_value = ([mock_packet], [])

            result = _perform_decoding(
                trace,
                protocol="auto",
                baud_rate=None,
                parity="none",
                stop_bits=1,
                show_errors=False,
            )

            assert result["protocol"] == "UART"
            assert result["packets_decoded"] == 1


@pytest.mark.unit
def test_perform_decoding_filters_errors():
    """Test that show_errors flag filters to error packets only."""
    data = np.array([True, False] * 100, dtype=bool)
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=data, metadata=metadata)

    with patch("oscura.cli.decode._detect_protocol_if_auto") as mock_detect:
        with patch("oscura.cli.decode._dispatch_protocol_decode") as mock_dispatch:
            mock_detect.return_value = ("uart", None)

            # Create mix of error and non-error packets
            good_packet = Mock()
            good_packet.errors = []
            good_packet.timestamp = 0.001
            good_packet.data = b"A"
            good_packet.annotations = {}
            error_packet = Mock()
            error_packet.errors = ["framing_error"]
            error_packet.timestamp = 0.002
            error_packet.data = b"B"
            error_packet.annotations = {}

            mock_dispatch.return_value = ([good_packet, error_packet, good_packet], [])

            result = _perform_decoding(
                trace,
                protocol="uart",
                baud_rate=9600,
                parity="none",
                stop_bits=1,
                show_errors=True,  # Only show errors
            )

            # Should only have 1 packet (the error one)
            assert result["packets_decoded"] == 1


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_decode_all_protocols():
    """Test decode command with all protocol options."""
    runner = CliRunner()

    protocols = ["uart", "spi", "i2c", "can", "auto"]

    for proto in protocols:
        with runner.isolated_filesystem():
            Path("test.wfm").write_bytes(b"fake")

            with patch("oscura.loaders.load") as mock_load:
                data = np.array([True, False] * 100, dtype=bool)
                metadata = TraceMetadata(sample_rate=1e6)
                mock_load.return_value = DigitalTrace(data=data, metadata=metadata)

                with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                    mock_decode.return_value = {"protocol": proto.upper()}

                    result = runner.invoke(cli, ["decode", "test.wfm", "--protocol", proto])

                    # Should not crash
                    assert proto.upper() in result.output or result.exit_code == 0


@pytest.mark.unit
def test_decode_all_output_formats():
    """Test decode command with all output formats."""
    runner = CliRunner()

    formats = ["json", "csv", "html", "table"]

    for fmt in formats:
        with runner.isolated_filesystem():
            Path("test.wfm").write_bytes(b"fake")

            with patch("oscura.loaders.load") as mock_load:
                data = np.array([True, False] * 100, dtype=bool)
                metadata = TraceMetadata(sample_rate=1e6)
                mock_load.return_value = DigitalTrace(data=data, metadata=metadata)

                with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                    mock_decode.return_value = {"packets": 5}

                    result = runner.invoke(cli, ["decode", "test.wfm", "--output", fmt])

                # Should produce output in requested format
                assert result.exit_code == 0
