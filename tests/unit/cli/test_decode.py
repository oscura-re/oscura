"""Comprehensive unit tests for decode.py CLI module.

This module provides extensive testing for the Oscura decode command, including:
- Protocol decoding for UART, SPI, I2C, and CAN
- Auto-detection functionality
- Error highlighting and filtering
- Output format conversion
- Edge cases and error handling


Test Coverage:
- decode() CLI command with all options
- _to_digital() waveform to digital conversion
- _perform_decoding() protocol decoding orchestration
- _decode_uart() UART protocol decoding
- _decode_spi() SPI protocol decoding
- _decode_i2c() I2C protocol decoding
- _decode_can() CAN protocol decoding
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.decode import (
    _decode_can,
    _decode_i2c,
    _decode_spi,
    _decode_uart,
    _perform_decoding,
    _to_digital,
    decode,
)
from oscura.core.types import DigitalTrace, ProtocolPacket, TraceMetadata, WaveformTrace

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
def sample_waveform_trace(sample_metadata):
    """Create sample waveform trace."""
    # Create a simple square wave
    data = np.array([0.0, 0.0, 3.3, 3.3, 0.0, 0.0, 3.3, 3.3] * 100, dtype=np.float64)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def sample_digital_trace(sample_metadata):
    """Create sample digital trace."""
    # Create a simple digital pattern
    data = np.array([False, False, True, True, False, False, True, True] * 100, dtype=bool)
    return DigitalTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def sample_protocol_packet():
    """Create sample protocol packet."""
    return ProtocolPacket(
        timestamp=0.001,  # 1 ms
        protocol="UART",
        data=bytes([0x55, 0xAA]),
        errors=[],
        annotations={"type": "data"},
    )


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Test _to_digital()
# =============================================================================


@pytest.mark.unit
def test_to_digital_already_digital(sample_digital_trace):
    """Test that digital trace is returned as-is."""
    result = _to_digital(sample_digital_trace)

    assert result is sample_digital_trace
    assert isinstance(result, DigitalTrace)


@pytest.mark.unit
def test_to_digital_converts_waveform(sample_waveform_trace):
    """Test conversion from waveform to digital trace."""
    result = _to_digital(sample_waveform_trace)

    assert isinstance(result, DigitalTrace)
    assert result.data.dtype == bool
    assert len(result.data) == len(sample_waveform_trace.data)
    assert result.metadata == sample_waveform_trace.metadata


@pytest.mark.unit
def test_to_digital_uses_midpoint_threshold(sample_metadata):
    """Test that midpoint threshold is used for conversion."""
    # Create waveform with known min/max
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    trace = WaveformTrace(data=data, metadata=sample_metadata)

    result = _to_digital(trace)

    # Threshold should be (0 + 4) / 2 = 2.0
    # So values > 2.0 should be True
    expected = np.array([False, False, False, True, True], dtype=bool)
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.unit
def test_to_digital_handles_constant_waveform(sample_metadata):
    """Test conversion of constant waveform."""
    data = np.ones(100, dtype=np.float64) * 3.3
    trace = WaveformTrace(data=data, metadata=sample_metadata)

    result = _to_digital(trace)

    # With constant values, threshold is 3.3, none should be > 3.3
    assert not result.data.any()


# =============================================================================
# Test _decode_uart()
# =============================================================================


@pytest.mark.unit
def test_decode_uart_basic(sample_digital_trace):
    """Test basic UART decoding."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        # Setup mock decoder
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Create mock packets
        packet1 = ProtocolPacket(
            timestamp=0.001, protocol="UART", data=bytes([0x41]), errors=[], annotations={}
        )
        packet2 = ProtocolPacket(
            timestamp=0.002, protocol="UART", data=bytes([0x42]), errors=[], annotations={}
        )
        mock_decoder.decode.return_value = [packet1, packet2]
        mock_decoder._baudrate = 9600

        packets, errors, info = _decode_uart(
            sample_digital_trace, baud_rate=9600, parity="none", stop_bits=1, show_errors=False
        )

        # Verify decoder was created correctly
        mock_decoder_class.assert_called_once_with(
            baudrate=9600, data_bits=8, parity="none", stop_bits=1
        )

        # Verify results
        assert len(packets) == 2
        assert len(errors) == 0
        assert info["baud_rate"] == 9600
        assert info["parity"] == "none"
        assert info["stop_bits"] == 1
        assert info["data_bits"] == 8


@pytest.mark.unit
def test_decode_uart_auto_baud(sample_digital_trace):
    """Test UART with automatic baud rate detection."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []
        mock_decoder._baudrate = 115200

        packets, errors, info = _decode_uart(
            sample_digital_trace, baud_rate=None, parity="none", stop_bits=1, show_errors=False
        )

        # Should pass 0 to trigger auto-detection
        mock_decoder_class.assert_called_once_with(
            baudrate=0, data_bits=8, parity="none", stop_bits=1
        )


@pytest.mark.unit
def test_decode_uart_with_errors(sample_digital_trace):
    """Test UART decoding with error packets."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Create packet with errors
        packet = ProtocolPacket(
            timestamp=0.001,
            protocol="UART",
            data=bytes([0xFF]),
            errors=["parity_error", "framing_error"],
            annotations={},
        )
        mock_decoder.decode.return_value = [packet]
        mock_decoder._baudrate = 9600

        packets, errors, info = _decode_uart(
            sample_digital_trace, baud_rate=9600, parity="even", stop_bits=2, show_errors=False
        )

        # Verify error extraction
        assert len(errors) == 2
        assert errors[0]["packet_index"] == 0
        assert errors[0]["type"] == "parity_error"
        assert errors[1]["type"] == "framing_error"
        assert "1.000 ms" in errors[0]["timestamp"]  # 0.001s = 1.000ms


@pytest.mark.unit
def test_decode_uart_different_parity_stop_bits(sample_digital_trace):
    """Test UART with different parity and stop bit configurations."""
    with patch("oscura.analyzers.protocols.uart.UARTDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []
        mock_decoder._baudrate = 19200

        # Test even parity, 2 stop bits
        packets, errors, info = _decode_uart(
            sample_digital_trace, baud_rate=19200, parity="even", stop_bits=2, show_errors=False
        )

        mock_decoder_class.assert_called_once_with(
            baudrate=19200, data_bits=8, parity="even", stop_bits=2
        )

        assert info["parity"] == "even"
        assert info["stop_bits"] == 2


# =============================================================================
# Test _decode_spi()
# =============================================================================


@pytest.mark.unit
def test_decode_spi_basic(sample_digital_trace):
    """Test basic SPI decoding."""
    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.001, protocol="UART", data=bytes([0xAA]), errors=[], annotations={}
        )
        mock_decoder.decode.return_value = [packet]

        packets, errors, info = _decode_spi(sample_digital_trace, show_errors=False)

        # Verify decoder was created
        mock_decoder_class.assert_called_once_with(cpol=0, cpha=0, word_size=8)

        # Verify results
        assert len(packets) == 1
        assert info["mode"] == "0 (CPOL=0, CPHA=0)"
        assert info["word_size"] == 8
        assert "Single-channel decode" in info["note"]


@pytest.mark.unit
def test_decode_spi_calculates_clock_frequency(sample_metadata):
    """Test SPI clock frequency calculation from edges."""
    # Create data with regular clock edges
    # Toggle every 10 samples = 5 samples per half-period
    data = np.array([True, True, True, True, True, False, False, False, False, False] * 10)
    trace = DigitalTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        packets, errors, info = _decode_spi(trace, show_errors=False)

        # Clock frequency should be calculated
        assert "clock_frequency" in info
        assert "MHz" in info["clock_frequency"]


@pytest.mark.unit
def test_decode_spi_no_edges(sample_metadata):
    """Test SPI with no clock edges (constant signal)."""
    data = np.ones(100, dtype=bool)
    trace = DigitalTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder
        mock_decoder.decode.return_value = []

        packets, errors, info = _decode_spi(trace, show_errors=False)

        assert info["clock_frequency"] == "Unknown"


@pytest.mark.unit
def test_decode_spi_with_errors(sample_digital_trace):
    """Test SPI decoding with error packets."""
    with patch("oscura.analyzers.protocols.spi.SPIDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.002,
            protocol="SPI",
            data=bytes([0x55]),
            errors=["clock_glitch"],
            annotations={},
        )
        mock_decoder.decode.return_value = [packet]

        packets, errors, info = _decode_spi(sample_digital_trace, show_errors=False)

        assert len(errors) == 1
        assert errors[0]["type"] == "clock_glitch"
        assert "2.000 ms" in errors[0]["timestamp"]  # 0.002s = 2.000ms


# =============================================================================
# Test _decode_i2c()
# =============================================================================


@pytest.mark.unit
def test_decode_i2c_basic(sample_metadata):
    """Test basic I2C decoding."""
    # Create data with sufficient edges
    data = np.array([True, False, True, False, True, False] * 20, dtype=bool)
    trace = DigitalTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.protocols.i2c.I2CDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.001,
            protocol="I2C",
            data=bytes([0x42]),
            errors=[],
            annotations={"address": 0x50},
        )
        mock_decoder.decode.return_value = [packet]

        packets, errors, info = _decode_i2c(trace, show_errors=False)

        # Verify results
        assert len(packets) == 1
        assert info["transactions"] == 1
        assert "0x50" in info["addresses_seen"]
        assert "Single-channel decode" in info["note"]


@pytest.mark.unit
def test_decode_i2c_insufficient_edges(sample_metadata):
    """Test I2C with insufficient edges."""
    # Very few edges
    data = np.array([True, False, True] * 3, dtype=bool)
    trace = DigitalTrace(data=data, metadata=sample_metadata)

    packets, errors, info = _decode_i2c(trace, show_errors=False)

    assert len(packets) == 0
    assert "error" in info
    assert "Insufficient edges" in info["error"]


@pytest.mark.unit
def test_decode_i2c_multiple_addresses(sample_metadata):
    """Test I2C with multiple device addresses."""
    data = np.array([True, False] * 50, dtype=bool)
    trace = DigitalTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.protocols.i2c.I2CDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packets = [
            ProtocolPacket(
                timestamp=0.001,
                protocol="I2C",
                data=bytes([0x42]),
                errors=[],
                annotations={"address": 0x50},
            ),
            ProtocolPacket(
                timestamp=0.002,
                protocol="I2C",
                data=bytes([0x43]),
                errors=[],
                annotations={"address": 0x51},
            ),
            ProtocolPacket(
                timestamp=0.003,
                protocol="I2C",
                data=bytes([0x44]),
                errors=[],
                annotations={"address": 0x50},
            ),
        ]
        mock_decoder.decode.return_value = packets

        result_packets, result_errors, info = _decode_i2c(trace, show_errors=False)

        # Should see two unique addresses
        assert len(info["addresses_seen"]) == 2
        assert "0x50" in info["addresses_seen"]
        assert "0x51" in info["addresses_seen"]


@pytest.mark.unit
def test_decode_i2c_with_errors(sample_metadata):
    """Test I2C decoding with NACK errors."""
    data = np.array([True, False] * 50, dtype=bool)
    trace = DigitalTrace(data=data, metadata=sample_metadata)

    with patch("oscura.analyzers.protocols.i2c.I2CDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.005,
            protocol="I2C",
            data=bytes([0x00]),
            errors=["NACK"],
            annotations={"address": 0x48},
        )
        mock_decoder.decode.return_value = [packet]

        packets, errors, info = _decode_i2c(trace, show_errors=False)

        assert len(errors) == 1
        assert errors[0]["type"] == "NACK"
        assert errors[0]["address"] == "0x48"


# =============================================================================
# Test _decode_can()
# =============================================================================


@pytest.mark.unit
def test_decode_can_basic(sample_digital_trace):
    """Test basic CAN decoding with specified baud rate."""
    with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packet = ProtocolPacket(
            timestamp=0.001,
            protocol="CAN",
            data=bytes([0x12, 0x34]),
            errors=[],
            annotations={"arbitration_id": 0x123},
        )
        mock_decoder.decode.return_value = [packet]

        packets, errors, info = _decode_can(
            sample_digital_trace, baud_rate=500000, show_errors=False
        )

        # Verify decoder creation
        mock_decoder_class.assert_called_once_with(bitrate=500000)

        # Verify results
        assert len(packets) == 1
        assert info["bit_rate"] == "500 kbps"
        assert "0x123" in info["arbitration_ids"]


@pytest.mark.unit
def test_decode_can_auto_baud(sample_digital_trace):
    """Test CAN with automatic baud rate detection."""
    with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:
        # Create mock decoders that return different packet counts
        def create_decoder(bitrate):
            decoder = Mock()
            # 250kbps returns most packets
            if bitrate == 250000:
                packets = [
                    ProtocolPacket(
                        timestamp=i * 0.001,
                        protocol="CAN",
                        data=bytes([i]),
                        errors=[],
                        annotations={"arbitration_id": i},
                    )
                    for i in range(10)
                ]
                decoder.decode.return_value = packets
            else:
                packets = [
                    ProtocolPacket(
                        timestamp=i * 0.001,
                        protocol="CAN",
                        data=bytes([i]),
                        errors=[],
                        annotations={"arbitration_id": i},
                    )
                    for i in range(2)
                ]
                decoder.decode.return_value = packets
            return decoder

        mock_decoder_class.side_effect = create_decoder

        packets, errors, info = _decode_can(sample_digital_trace, baud_rate=None, show_errors=False)

        # Should have tried multiple baud rates
        assert mock_decoder_class.call_count >= 4

        # Should have selected 250kbps (most packets)
        # Last call will be with best rate
        assert info["bit_rate"] == "250 kbps"


@pytest.mark.unit
def test_decode_can_extended_frames(sample_digital_trace):
    """Test CAN with extended frames."""
    with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        packets = [
            ProtocolPacket(
                timestamp=0.001,
                protocol="CAN",
                data=bytes([0x01]),
                errors=[],
                annotations={"arbitration_id": 0x123, "is_extended": False},
            ),
            ProtocolPacket(
                timestamp=0.002,
                protocol="CAN",
                data=bytes([0x02]),
                errors=[],
                annotations={"arbitration_id": 0x18FF1234, "is_extended": True},
            ),
        ]
        mock_decoder.decode.return_value = packets

        result_packets, result_errors, info = _decode_can(
            sample_digital_trace, baud_rate=1000000, show_errors=False
        )

        assert info["extended_frames"] == 1
        assert info["messages"] == 2


@pytest.mark.unit
def test_decode_can_many_arbitration_ids(sample_digital_trace):
    """Test CAN with many arbitration IDs (truncation)."""
    with patch("oscura.analyzers.protocols.can.CANDecoder") as mock_decoder_class:
        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Create packets with 15 unique IDs
        packets = [
            ProtocolPacket(
                timestamp=i * 0.001,
                protocol="CAN",
                data=bytes([i]),
                errors=[],
                annotations={"arbitration_id": i},
            )
            for i in range(15)
        ]
        mock_decoder.decode.return_value = packets

        result_packets, result_errors, info = _decode_can(
            sample_digital_trace, baud_rate=500000, show_errors=False
        )

        # Should show first 10 IDs
        assert len(info["arbitration_ids"]) == 10
        assert "note" in info
        assert "15 arbitration IDs" in info["note"]


# =============================================================================
# Test _perform_decoding()
# =============================================================================


@pytest.mark.unit
def test_perform_decoding_uart_explicit(sample_digital_trace):
    """Test perform_decoding with explicit UART protocol."""
    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        mock_uart.return_value = (
            [
                ProtocolPacket(
                    timestamp=0.001, protocol="UART", data=bytes([0x41]), errors=[], annotations={}
                )
            ],
            [],
            {"baud_rate": 9600, "parity": "none", "stop_bits": 1, "data_bits": 8},
        )

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        # Verify UART decoder was called
        mock_uart.assert_called_once()

        # Check results structure
        assert results["protocol"] == "UART"
        assert results["packets_decoded"] == 1
        assert results["errors_found"] == 0
        assert "sample_rate" in results
        assert "duration" in results


@pytest.mark.unit
def test_perform_decoding_auto_detect(sample_waveform_trace):
    """Test perform_decoding with auto protocol detection."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("oscura.cli.decode._decode_uart") as mock_uart:
            # Setup auto-detection
            mock_detect.return_value = {
                "protocol": "UART",
                "confidence": 0.95,
                "candidates": [
                    {"protocol": "UART", "confidence": 0.95},
                    {"protocol": "SPI", "confidence": 0.3},
                ],
                "config": {"baud_rate": 115200},
            }

            mock_uart.return_value = ([], [], {"baud_rate": 115200})

            results = _perform_decoding(
                trace=sample_waveform_trace,
                protocol="auto",
                baud_rate=None,
                parity="none",
                stop_bits=1,
                show_errors=False,
            )

            # Verify auto-detection was called
            mock_detect.assert_called_once()

            # Check auto-detection results
            assert "auto_detection" in results
            assert results["auto_detection"]["protocol"] == "UART"
            assert results["auto_detection"]["confidence"] == "95.0%"
            assert len(results["auto_detection"]["candidates"]) == 2


@pytest.mark.unit
def test_perform_decoding_auto_detect_failure(sample_digital_trace):
    """Test perform_decoding when auto-detection fails."""
    with patch("oscura.inference.protocol.detect_protocol") as mock_detect:
        with patch("oscura.cli.decode._decode_uart") as mock_uart:
            # Make auto-detection raise an exception
            mock_detect.side_effect = Exception("Detection failed")
            mock_uart.return_value = ([], [], {})

            results = _perform_decoding(
                trace=sample_digital_trace,
                protocol="auto",
                baud_rate=None,
                parity="none",
                stop_bits=1,
                show_errors=False,
            )

            # Should fall back to UART
            assert results["protocol"] == "UART"


@pytest.mark.unit
def test_perform_decoding_spi(sample_digital_trace):
    """Test perform_decoding with SPI protocol."""
    with patch("oscura.cli.decode._decode_spi") as mock_spi:
        mock_spi.return_value = ([], [], {"mode": "0"})

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="spi",
            baud_rate=None,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        mock_spi.assert_called_once()
        assert results["protocol"] == "SPI"


@pytest.mark.unit
def test_perform_decoding_i2c(sample_digital_trace):
    """Test perform_decoding with I2C protocol."""
    with patch("oscura.cli.decode._decode_i2c") as mock_i2c:
        mock_i2c.return_value = ([], [], {"transactions": 0})

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="i2c",
            baud_rate=None,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        mock_i2c.assert_called_once()
        assert results["protocol"] == "I2C"


@pytest.mark.unit
def test_perform_decoding_can(sample_digital_trace):
    """Test perform_decoding with CAN protocol."""
    with patch("oscura.cli.decode._decode_can") as mock_can:
        mock_can.return_value = ([], [], {"bit_rate": "500 kbps"})

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="can",
            baud_rate=500000,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        mock_can.assert_called_once()
        assert results["protocol"] == "CAN"


@pytest.mark.unit
def test_perform_decoding_show_errors_only(sample_digital_trace):
    """Test perform_decoding with show_errors=True filters packets."""
    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        # Mix of packets with and without errors
        packets = [
            ProtocolPacket(
                timestamp=0.001, protocol="UART", data=bytes([0x01]), errors=[], annotations={}
            ),
            ProtocolPacket(
                timestamp=0.002,
                protocol="UART",
                data=bytes([0x02]),
                errors=["framing_error"],
                annotations={},
            ),
            ProtocolPacket(
                timestamp=0.003, protocol="UART", data=bytes([0x03]), errors=[], annotations={}
            ),
        ]
        mock_uart.return_value = (packets, [], {})

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=True,
        )

        # Should only show the packet with errors
        assert results["packets_decoded"] == 1


@pytest.mark.unit
def test_perform_decoding_many_packets_truncation(sample_digital_trace):
    """Test that packet list is truncated to first 100."""
    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        # Create 150 packets
        packets = [
            ProtocolPacket(
                timestamp=i * 0.001,
                protocol="UART",
                data=bytes([i % 256]),
                errors=[],
                annotations={},
            )
            for i in range(150)
        ]
        mock_uart.return_value = (packets, [], {})

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        # Should show first 100 packets
        assert len(results["packets"]) == 100
        assert "note" in results
        assert "150 packets" in results["note"]


@pytest.mark.unit
def test_perform_decoding_error_truncation(sample_digital_trace):
    """Test that error details are truncated to first 20."""
    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        # Create errors list with more than 20 errors
        errors = [{"packet_index": i, "type": "error"} for i in range(25)]
        mock_uart.return_value = ([], errors, {})

        results = _perform_decoding(
            trace=sample_digital_trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        # Should truncate to 20 errors
        assert len(results["error_details"]) == 20


# =============================================================================
# Test decode() CLI command
# =============================================================================


@pytest.mark.unit
def test_decode_command_basic(cli_runner, tmp_path, sample_waveform_trace):
    """Test basic decode command execution."""
    # Create a temporary file
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.decode._perform_decoding") as mock_decode:
            with patch("oscura.cli.decode.format_output") as mock_format:
                mock_load.return_value = sample_waveform_trace
                mock_decode.return_value = {"protocol": "UART", "packets_decoded": 5}
                mock_format.return_value = "Formatted output"

                result = cli_runner.invoke(
                    decode, [str(test_file)], obj={"verbose": 0}, catch_exceptions=False
                )

                assert result.exit_code == 0
                assert "Formatted output" in result.output
                mock_load.assert_called_once_with(str(test_file))


@pytest.mark.unit
def test_decode_command_with_protocol(cli_runner, tmp_path):
    """Test decode command with explicit protocol."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.decode._perform_decoding") as mock_decode:
            with patch("oscura.cli.decode.format_output"):
                mock_load.return_value = Mock()
                mock_decode.return_value = {"protocol": "SPI"}

                result = cli_runner.invoke(
                    decode, [str(test_file), "--protocol", "spi"], obj={"verbose": 0}
                )

                assert result.exit_code == 0
                # Check that protocol was passed
                call_args = mock_decode.call_args
                assert call_args[1]["protocol"] == "spi"


@pytest.mark.unit
def test_decode_command_uart_options(cli_runner, tmp_path):
    """Test decode command with UART-specific options."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.decode._perform_decoding") as mock_decode:
            with patch("oscura.cli.decode.format_output"):
                mock_load.return_value = Mock()
                mock_decode.return_value = {"protocol": "UART"}

                result = cli_runner.invoke(
                    decode,
                    [
                        str(test_file),
                        "--protocol",
                        "uart",
                        "--baud-rate",
                        "115200",
                        "--parity",
                        "even",
                        "--stop-bits",
                        "2",
                    ],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                call_args = mock_decode.call_args
                assert call_args[1]["baud_rate"] == 115200
                assert call_args[1]["parity"] == "even"
                assert call_args[1]["stop_bits"] == 2


@pytest.mark.unit
def test_decode_command_show_errors(cli_runner, tmp_path):
    """Test decode command with --show-errors flag."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.decode._perform_decoding") as mock_decode:
            with patch("oscura.cli.decode.format_output"):
                mock_load.return_value = Mock()
                mock_decode.return_value = {"protocol": "UART"}

                result = cli_runner.invoke(
                    decode, [str(test_file), "--show-errors"], obj={"verbose": 0}
                )

                assert result.exit_code == 0
                call_args = mock_decode.call_args
                assert call_args[1]["show_errors"] is True


@pytest.mark.unit
def test_decode_command_output_formats(cli_runner, tmp_path):
    """Test decode command with different output formats."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    for output_format in ["json", "csv", "html", "table"]:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                with patch("oscura.cli.decode.format_output") as mock_format:
                    mock_load.return_value = Mock()
                    mock_decode.return_value = {"protocol": "UART"}
                    mock_format.return_value = f"{output_format} output"

                    result = cli_runner.invoke(
                        decode, [str(test_file), "--output", output_format], obj={"verbose": 0}
                    )

                    assert result.exit_code == 0
                    mock_format.assert_called_with(mock_decode.return_value, output_format)


@pytest.mark.unit
def test_decode_command_verbose_logging(cli_runner, tmp_path, caplog):
    """Test decode command with verbose logging."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.decode._perform_decoding") as mock_decode:
            with patch("oscura.cli.decode.format_output"):
                mock_load.return_value = Mock()
                mock_decode.return_value = {"protocol": "UART"}

                result = cli_runner.invoke(
                    decode,
                    [str(test_file), "--protocol", "uart", "--baud-rate", "9600"],
                    obj={"verbose": 1},
                )

                assert result.exit_code == 0


@pytest.mark.unit
def test_decode_command_error_handling(cli_runner, tmp_path):
    """Test decode command error handling."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = Exception("Failed to load file")

        result = cli_runner.invoke(decode, [str(test_file)], obj={"verbose": 0})

        assert result.exit_code == 1
        assert "Error: Failed to load file" in result.output


@pytest.mark.unit
def test_decode_command_error_with_verbose(cli_runner, tmp_path):
    """Test decode command error handling with verbose mode (should raise)."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = ValueError("Test error")

        result = cli_runner.invoke(decode, [str(test_file)], obj={"verbose": 2})

        # With verbose > 1, exception should be raised
        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
def test_decode_command_adds_filename_to_results(cli_runner, tmp_path):
    """Test that decode command adds filename to results."""
    test_file = tmp_path / "my_capture.wfm"
    test_file.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.decode._perform_decoding") as mock_decode:
            with patch("oscura.cli.decode.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_decode.return_value = {"protocol": "UART"}
                mock_format.return_value = "output"

                result = cli_runner.invoke(decode, [str(test_file)], obj={"verbose": 0})

                assert result.exit_code == 0
                # Check that file key was added to results
                format_call_args = mock_format.call_args[0][0]
                assert format_call_args["file"] == "my_capture.wfm"


@pytest.mark.unit
def test_decode_command_nonexistent_file(cli_runner):
    """Test decode command with nonexistent file."""
    result = cli_runner.invoke(decode, ["/nonexistent/file.wfm"], obj={"verbose": 0})

    # Click should catch this with its path validation
    assert result.exit_code != 0


@pytest.mark.unit
def test_decode_command_all_protocols(cli_runner, tmp_path):
    """Test decode command with all supported protocols."""
    test_file = tmp_path / "test.wfm"
    test_file.touch()

    protocols = ["uart", "spi", "i2c", "can", "auto"]

    for protocol in protocols:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.decode._perform_decoding") as mock_decode:
                with patch("oscura.cli.decode.format_output"):
                    mock_load.return_value = Mock()
                    mock_decode.return_value = {"protocol": protocol.upper()}

                    result = cli_runner.invoke(
                        decode, [str(test_file), "--protocol", protocol], obj={"verbose": 0}
                    )

                    assert result.exit_code == 0, f"Failed for protocol {protocol}"


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


@pytest.mark.unit
def test_decode_empty_trace(sample_metadata):
    """Test decoding an empty trace."""
    empty_trace = DigitalTrace(data=np.array([], dtype=bool), metadata=sample_metadata)

    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        mock_uart.return_value = ([], [], {})

        results = _perform_decoding(
            trace=empty_trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        assert results["packets_decoded"] == 0
        assert results["samples"] == 0


@pytest.mark.unit
def test_packet_annotations_exclude_data_bits():
    """Test that data_bits is excluded from packet annotations in results."""
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=np.array([True] * 100, dtype=bool), metadata=metadata)

    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        packet = ProtocolPacket(
            timestamp=0.001,
            protocol="UART",
            data=bytes([0x41]),
            errors=[],
            annotations={"data_bits": [1, 0, 1], "type": "data", "value": 65},
        )
        mock_uart.return_value = ([packet], [], {})

        results = _perform_decoding(
            trace=trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        # data_bits should be excluded from packet details
        packet_info = results["packets"][0]
        assert "data_bits" not in packet_info
        assert "type" in packet_info
        assert "value" in packet_info


@pytest.mark.unit
def test_packet_hex_encoding():
    """Test that packet data is properly hex encoded in results."""
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=np.array([True] * 100, dtype=bool), metadata=metadata)

    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        packet = ProtocolPacket(
            timestamp=0.001,
            protocol="UART",
            data=bytes([0xDE, 0xAD, 0xBE, 0xEF]),
            errors=[],
            annotations={},
        )
        mock_uart.return_value = ([packet], [], {})

        results = _perform_decoding(
            trace=trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        assert results["packets"][0]["data"] == "deadbeef"


@pytest.mark.unit
def test_timestamp_formatting():
    """Test that timestamps are properly formatted in milliseconds."""
    metadata = TraceMetadata(sample_rate=1e6)
    trace = DigitalTrace(data=np.array([True] * 100, dtype=bool), metadata=metadata)

    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        # Timestamp in seconds
        packet = ProtocolPacket(
            timestamp=0.001234, protocol="UART", data=bytes([0x00]), errors=[], annotations={}
        )
        mock_uart.return_value = ([packet], [], {})

        results = _perform_decoding(
            trace=trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        # Should be formatted to 3 decimal places in ms
        assert results["packets"][0]["timestamp"] == "1.234 ms"


@pytest.mark.unit
def test_sample_rate_and_duration_formatting():
    """Test sample rate and duration formatting in results."""
    metadata = TraceMetadata(sample_rate=25e6)  # 25 MHz
    trace = DigitalTrace(data=np.array([True] * 50000, dtype=bool), metadata=metadata)

    with patch("oscura.cli.decode._decode_uart") as mock_uart:
        mock_uart.return_value = ([], [], {})

        results = _perform_decoding(
            trace=trace,
            protocol="uart",
            baud_rate=9600,
            parity="none",
            stop_bits=1,
            show_errors=False,
        )

        assert results["sample_rate"] == "25.0 MHz"
        assert "ms" in results["duration"]
        assert results["samples"] == 50000
