"""Comprehensive tests for visualization.protocols module.

Tests cover protocol decoder visualization functions for UART, SPI, I2C, CAN
with multi-level annotations and timing diagrams.

Coverage target: >90%
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, ProtocolPacket, TraceMetadata


@pytest.fixture
def sample_uart_packets() -> list[ProtocolPacket]:
    """Create sample UART packets for testing."""
    return [
        ProtocolPacket(
            protocol="UART",
            timestamp=0.0,
            end_timestamp=0.001,
            data=bytes([0x48]),  # 'H'
            errors=[],
            annotations={},
        ),
        ProtocolPacket(
            protocol="UART",
            timestamp=0.002,
            end_timestamp=0.003,
            data=bytes([0x65]),  # 'e'
            errors=[],
            annotations={},
        ),
        ProtocolPacket(
            protocol="UART",
            timestamp=0.004,
            end_timestamp=0.005,
            data=bytes([0x6C]),  # 'l'
            errors=["parity_error"],
            annotations={},
        ),
    ]


@pytest.fixture
def sample_digital_trace() -> DigitalTrace:
    """Create sample digital trace for testing."""
    return DigitalTrace(
        data=np.array([0, 1, 1, 0, 1, 0, 0, 1] * 10, dtype=np.uint8),
        metadata=TraceMetadata(sample_rate=1e6, channel_name="RX"),
    )


class TestPlotProtocolDecode:
    """Tests for plot_protocol_decode function."""

    def test_import_error_without_matplotlib(self) -> None:
        """Test ImportError when matplotlib not available."""
        import importlib
        import sys
        from unittest.mock import patch

        import oscura.visualization.protocols as protocols_module

        try:
            with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
                importlib.reload(protocols_module)

                with pytest.raises(ImportError, match="matplotlib is required"):
                    protocols_module.plot_protocol_decode([])
        finally:
            # Restore module to original state for subsequent tests
            importlib.reload(protocols_module)

    def test_empty_packets_raises(self, matplotlib_available: None) -> None:
        """Test that empty packet list raises ValueError."""
        from oscura.visualization.protocols import plot_protocol_decode

        with pytest.raises(ValueError, match="cannot be empty"):
            plot_protocol_decode([])

    def test_basic_decode_plot(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test basic protocol decode plotting."""
        from oscura.visualization.protocols import plot_protocol_decode

        fig = plot_protocol_decode(sample_uart_packets)

        assert fig is not None
        # Should have at least one axes for packets
        assert len(fig.axes) >= 1

    def test_with_trace(
        self,
        sample_uart_packets: list[ProtocolPacket],
        sample_digital_trace: DigitalTrace,
        matplotlib_available: None,
    ) -> None:
        """Test plotting with digital trace."""
        from oscura.visualization.protocols import plot_protocol_decode

        fig = plot_protocol_decode(sample_uart_packets, trace=sample_digital_trace)

        # Should have multiple axes (waveform + packets)
        assert len(fig.axes) >= 2

    def test_time_range_specification(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test custom time range."""
        from oscura.visualization.protocols import plot_protocol_decode

        fig = plot_protocol_decode(sample_uart_packets, time_range=(0.0, 0.003))

        assert fig is not None

    def test_time_unit_selection(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test different time units."""
        from oscura.visualization.protocols import plot_protocol_decode

        for time_unit in ["s", "ms", "us", "ns"]:
            fig = plot_protocol_decode(sample_uart_packets, time_unit=time_unit)
            assert fig is not None

    def test_error_highlighting(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test error highlighting in packets."""
        from oscura.visualization.protocols import plot_protocol_decode

        fig = plot_protocol_decode(sample_uart_packets, show_errors=True)

        # Error packets should be highlighted (verified visually in output)
        assert fig is not None

    def test_colorize_by_protocol(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test color coding by protocol type."""
        from oscura.visualization.protocols import plot_protocol_decode

        fig = plot_protocol_decode(sample_uart_packets, colorize=True)

        assert fig is not None


class TestPlotUartDecode:
    """Tests for plot_uart_decode function."""

    def test_single_channel_rx(
        self,
        sample_uart_packets: list[ProtocolPacket],
        sample_digital_trace: DigitalTrace,
        matplotlib_available: None,
    ) -> None:
        """Test single RX channel plotting."""
        from oscura.visualization.protocols import plot_uart_decode

        fig = plot_uart_decode(sample_uart_packets, rx_trace=sample_digital_trace)

        assert fig is not None

    def test_dual_channel_rx_tx(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test dual RX/TX channel plotting."""
        from oscura.visualization.protocols import plot_uart_decode

        rx_trace = DigitalTrace(
            data=np.array([0, 1] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )
        tx_trace = DigitalTrace(
            data=np.array([1, 0] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )

        fig = plot_uart_decode(sample_uart_packets, rx_trace=rx_trace, tx_trace=tx_trace)

        # Should have multiple rows for RX waveform, RX packets, TX waveform, TX packets
        assert len(fig.axes) >= 2

    def test_parity_error_display(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test parity error highlighting."""
        from oscura.visualization.protocols import plot_uart_decode

        fig = plot_uart_decode(sample_uart_packets, show_parity_errors=True)

        assert fig is not None

    def test_framing_error_display(
        self, sample_uart_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test framing error highlighting."""
        from oscura.visualization.protocols import plot_uart_decode

        fig = plot_uart_decode(sample_uart_packets, show_framing_errors=True)

        assert fig is not None


class TestPlotSpiDecode:
    """Tests for plot_spi_decode function."""

    @pytest.fixture
    def spi_packets(self) -> list[ProtocolPacket]:
        """Create sample SPI packets."""
        return [
            ProtocolPacket(
                protocol="SPI",
                timestamp=0.0,
                end_timestamp=0.001,
                data=bytes([0xAB, 0xCD]),
                errors=[],
                annotations={"channel": "MOSI"},
            ),
            ProtocolPacket(
                protocol="SPI",
                timestamp=0.0,
                end_timestamp=0.001,
                data=bytes([0x12, 0x34]),
                errors=[],
                annotations={"channel": "MISO"},
            ),
        ]

    def test_single_channel_spi(
        self, spi_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test single channel SPI plot."""
        from oscura.visualization.protocols import plot_spi_decode

        mosi_trace = DigitalTrace(
            data=np.array([0, 1] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )

        fig = plot_spi_decode(spi_packets, mosi_trace=mosi_trace)

        assert fig is not None

    def test_multi_channel_spi(
        self, spi_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test multi-channel SPI plot with CLK, MOSI, MISO, CS."""
        from oscura.visualization.protocols import plot_spi_decode

        clk_trace = DigitalTrace(
            data=np.array([0, 1] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )
        mosi_trace = DigitalTrace(
            data=np.array([0, 1, 1, 0] * 20, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )
        miso_trace = DigitalTrace(
            data=np.array([1, 0, 0, 1] * 20, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )
        cs_trace = DigitalTrace(
            data=np.array([1] + [0] * 78 + [1], dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )

        fig = plot_spi_decode(
            spi_packets,
            clk_trace=clk_trace,
            mosi_trace=mosi_trace,
            miso_trace=miso_trace,
            cs_trace=cs_trace,
        )

        # Should have multiple rows
        assert len(fig.axes) >= 2

    def test_mosi_miso_toggle(
        self, spi_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test show_mosi and show_miso toggles."""
        from oscura.visualization.protocols import plot_spi_decode

        mosi_trace = DigitalTrace(
            data=np.array([0, 1] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )

        # Only show MOSI
        fig = plot_spi_decode(spi_packets, mosi_trace=mosi_trace, show_miso=False)
        assert fig is not None


class TestPlotI2cDecode:
    """Tests for plot_i2c_decode function."""

    @pytest.fixture
    def i2c_packets(self) -> list[ProtocolPacket]:
        """Create sample I2C packets."""
        return [
            ProtocolPacket(
                protocol="I2C",
                timestamp=0.0,
                end_timestamp=0.001,
                data=bytes([0x50]),  # Address
                errors=[],
                annotations={"type": "address"},
            ),
            ProtocolPacket(
                protocol="I2C",
                timestamp=0.002,
                end_timestamp=0.003,
                data=bytes([0xAB, 0xCD]),
                errors=[],
                annotations={"type": "data"},
            ),
        ]

    def test_basic_i2c_plot(
        self, i2c_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test basic I2C plotting."""
        from oscura.visualization.protocols import plot_i2c_decode

        fig = plot_i2c_decode(i2c_packets)

        assert fig is not None

    def test_i2c_with_traces(
        self, i2c_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test I2C with SDA and SCL traces."""
        from oscura.visualization.protocols import plot_i2c_decode

        sda_trace = DigitalTrace(
            data=np.array([0, 1] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )
        scl_trace = DigitalTrace(
            data=np.array([0, 1, 0, 1] * 20, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )

        fig = plot_i2c_decode(i2c_packets, sda_trace=sda_trace, scl_trace=scl_trace)

        assert fig is not None

    def test_i2c_address_display(
        self, i2c_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test address highlighting."""
        from oscura.visualization.protocols import plot_i2c_decode

        fig = plot_i2c_decode(i2c_packets, show_addresses=True)

        assert fig is not None

    def test_i2c_ack_nack_display(
        self, i2c_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test ACK/NACK indicators."""
        from oscura.visualization.protocols import plot_i2c_decode

        fig = plot_i2c_decode(i2c_packets, show_ack_nack=True)

        assert fig is not None


class TestPlotCanDecode:
    """Tests for plot_can_decode function."""

    @pytest.fixture
    def can_packets(self) -> list[ProtocolPacket]:
        """Create sample CAN packets."""
        return [
            ProtocolPacket(
                protocol="CAN",
                timestamp=0.0,
                end_timestamp=0.001,
                data=bytes([0x12, 0x34, 0x56, 0x78]),
                errors=[],
                annotations={"id": 0x123, "dlc": 4},
            ),
            ProtocolPacket(
                protocol="CAN",
                timestamp=0.002,
                end_timestamp=0.003,
                data=bytes([0xAB, 0xCD]),
                errors=[],
                annotations={"id": 0x456, "dlc": 2},
            ),
        ]

    def test_basic_can_plot(
        self, can_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test basic CAN plotting."""
        from oscura.visualization.protocols import plot_can_decode

        fig = plot_can_decode(can_packets)

        assert fig is not None

    def test_can_with_trace(
        self, can_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test CAN with bus trace."""
        from oscura.visualization.protocols import plot_can_decode

        can_trace = DigitalTrace(
            data=np.array([0, 1] * 40, dtype=np.uint8),
            metadata=TraceMetadata(sample_rate=1e6),
        )

        fig = plot_can_decode(can_packets, can_trace=can_trace)

        assert fig is not None

    def test_can_id_display(
        self, can_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test CAN ID display."""
        from oscura.visualization.protocols import plot_can_decode

        fig = plot_can_decode(can_packets, show_ids=True)

        assert fig is not None

    def test_can_data_length_display(
        self, can_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test DLC display."""
        from oscura.visualization.protocols import plot_can_decode

        fig = plot_can_decode(can_packets, show_data_length=True)

        assert fig is not None

    def test_can_colorize_by_id(
        self, can_packets: list[ProtocolPacket], matplotlib_available: None
    ) -> None:
        """Test color coding by CAN ID."""
        from oscura.visualization.protocols import plot_can_decode

        fig = plot_can_decode(can_packets, colorize_by_id=True)

        assert fig is not None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_packet_data_single_byte(self, matplotlib_available: None) -> None:
        """Test formatting single byte."""
        from oscura.visualization.protocols import _format_packet_data

        packet = ProtocolPacket(
            protocol="UART",
            timestamp=0.0,
            data=bytes([0x41]),  # 'A'
            errors=[],
            annotations={},
        )

        formatted = _format_packet_data(packet)

        # Should show hex and ASCII for printable
        assert "0x41" in formatted
        assert "A" in formatted

    def test_format_packet_data_multiple_bytes(self, matplotlib_available: None) -> None:
        """Test formatting multiple bytes."""
        from oscura.visualization.protocols import _format_packet_data

        packet = ProtocolPacket(
            protocol="SPI",
            timestamp=0.0,
            data=bytes([0xAB, 0xCD, 0xEF]),
            errors=[],
            annotations={},
        )

        formatted = _format_packet_data(packet)

        # Should show hex values
        assert "AB" in formatted
        assert "CD" in formatted

    def test_format_packet_data_long(self, matplotlib_available: None) -> None:
        """Test formatting long data (truncated)."""
        from oscura.visualization.protocols import _format_packet_data

        packet = ProtocolPacket(
            protocol="CAN",
            timestamp=0.0,
            data=bytes([0x00] * 10),
            errors=[],
            annotations={},
        )

        formatted = _format_packet_data(packet)

        # Should truncate with "..."
        assert "..." in formatted

    def test_get_packet_color(self, matplotlib_available: None) -> None:
        """Test packet color assignment."""
        from oscura.visualization.protocols import _get_packet_color

        packet_uart = ProtocolPacket(
            protocol="UART", timestamp=0.0, data=bytes([]), errors=[], annotations={}
        )
        packet_spi = ProtocolPacket(
            protocol="SPI", timestamp=0.0, data=bytes([]), errors=[], annotations={}
        )

        color_uart = _get_packet_color(packet_uart, "UART")
        color_spi = _get_packet_color(packet_spi, "SPI")

        # Colors should be different
        assert color_uart != color_spi


@pytest.fixture
def matplotlib_available() -> None:
    """Ensure matplotlib is available for tests."""
    pytest.importorskip("matplotlib")


# Run tests with: pytest tests/unit/visualization/test_protocols.py -v
