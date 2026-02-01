"""Tests for Saleae Logic analyzer acquisition source.

Tests comprehensive coverage of SaleaeSource with mocked hardware connections.
Since Saleae hardware is not available in CI, all Saleae API calls are mocked.

Coverage targets:
- Connection management (open, close, context manager)
- Configuration (sample rate, channels, duration)
- Read operations (digital and analog)
- Streaming operations
- Error handling (import errors, connection failures, invalid configs)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.hardware.acquisition.saleae import SaleaeSource


class TestSaleaeSource:
    """Test SaleaeSource acquisition functionality."""

    def test_init_basic(self) -> None:
        """Test basic initialization without device connection."""
        source = SaleaeSource()
        assert source.device_id is None
        assert source.saleae is None
        assert source._closed is False
        assert source.sample_rate is None
        assert source.duration is None
        assert source.digital_channels == []
        assert source.analog_channels == []

    def test_init_with_device_id(self) -> None:
        """Test initialization with specific device ID."""
        source = SaleaeSource(device_id="ABC123")
        assert source.device_id == "ABC123"
        assert source._closed is False

    def test_init_with_kwargs(self) -> None:
        """Test initialization with additional configuration options."""
        source = SaleaeSource(device_id="TEST", custom_option="value")
        assert source.device_id == "TEST"
        assert source.kwargs == {"custom_option": "value"}

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_ensure_connection_success(self, mock_saleae_module: Mock) -> None:
        """Test successful connection to Saleae Logic software."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source._ensure_connection()

        assert source.saleae == mock_device
        mock_saleae_module.Saleae.assert_called_once()

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_ensure_connection_with_device_id(self, mock_saleae_module: Mock) -> None:
        """Test connection with specific device ID."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource(device_id="DEVICE123")
        source._ensure_connection()

        assert source.saleae == mock_device
        mock_device.set_active_device.assert_called_once_with("DEVICE123")

    def test_ensure_connection_import_error(self) -> None:
        """Test handling of missing saleae library."""
        source = SaleaeSource()

        # Patch saleae to be None (simulating import failure at module level)
        with patch("oscura.hardware.acquisition.saleae.saleae", None):
            with pytest.raises(ImportError, match="Saleae source requires saleae library"):
                source._ensure_connection()

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_ensure_connection_runtime_error(self, mock_saleae_module: Mock) -> None:
        """Test handling of Saleae Logic software connection failure."""
        mock_saleae_module.Saleae.side_effect = RuntimeError("Connection failed")

        source = SaleaeSource()

        with pytest.raises(RuntimeError, match="Failed to connect to Saleae Logic software"):
            source._ensure_connection()

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_ensure_connection_idempotent(self, mock_saleae_module: Mock) -> None:
        """Test that connection is only established once."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source._ensure_connection()
        source._ensure_connection()

        # Should only call once
        mock_saleae_module.Saleae.assert_called_once()

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_configure_digital_channels(self, mock_saleae_module: Mock) -> None:
        """Test configuration with digital channels."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(
            sample_rate=1e6,
            duration=10.0,
            digital_channels=[0, 1, 2, 3],
        )

        assert source.sample_rate == 1e6
        assert source.duration == 10.0
        assert source.digital_channels == [0, 1, 2, 3]
        assert source.analog_channels == []
        mock_device.set_sample_rate_by_minimum.assert_called_once_with(1e6)

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_configure_analog_channels(self, mock_saleae_module: Mock) -> None:
        """Test configuration with analog channels."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(
            sample_rate=5e5,
            duration=5.0,
            analog_channels=[0, 1],
        )

        assert source.sample_rate == 5e5
        assert source.duration == 5.0
        assert source.digital_channels == []
        assert source.analog_channels == [0, 1]

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_configure_mixed_channels(self, mock_saleae_module: Mock) -> None:
        """Test configuration with both digital and analog channels."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(
            sample_rate=1e6,
            duration=10.0,
            digital_channels=[0, 1],
            analog_channels=[0],
        )

        assert source.digital_channels == [0, 1]
        assert source.analog_channels == [0]

    def test_configure_no_channels_error(self) -> None:
        """Test that configuration fails without any channels."""
        source = SaleaeSource()

        with pytest.raises(ValueError, match="Must specify at least one digital or analog channel"):
            source.configure(sample_rate=1e6, duration=10.0)

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_read_digital_trace(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test reading digital trace from Saleae."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(sample_rate=1e6, duration=1.0, digital_channels=[0, 1])

        trace = source.read()

        # Verify capture was started and stopped
        mock_device.capture_start.assert_called_once()
        mock_device.capture_stop.assert_called_once()
        mock_sleep.assert_called_once_with(1.0)

        # Verify trace type
        from oscura.core.types import DigitalTrace

        assert isinstance(trace, DigitalTrace)
        assert len(trace.data) == 1000000  # 1 second at 1 MHz
        assert trace.metadata.sample_rate == 1e6
        assert trace.metadata.source_file and "saleae://" in trace.metadata.source_file

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_read_analog_trace(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test reading analog trace from Saleae."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(sample_rate=5e5, duration=2.0, analog_channels=[0])

        trace = source.read()

        # Verify trace type
        from oscura.core.types import WaveformTrace

        assert isinstance(trace, WaveformTrace)
        assert len(trace.data) == 1000000  # 2 seconds at 500 kHz
        assert trace.metadata.sample_rate == 5e5

    def test_read_not_configured_error(self) -> None:
        """Test that reading fails if source is not configured."""
        source = SaleaeSource()

        with pytest.raises(ValueError, match="Source not configured"):
            source.read()

    def test_read_closed_source_error(self) -> None:
        """Test that reading fails if source is closed."""
        source = SaleaeSource()
        source._closed = True

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_read_with_device_id(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test reading with specific device ID in metadata."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource(device_id="DEVICE123")
        source.configure(sample_rate=1e6, duration=1.0, digital_channels=[0])

        trace = source.read()

        assert trace.metadata.source_file and "saleae://DEVICE123" in trace.metadata.source_file

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_stream_digital_chunks(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test streaming digital data in chunks."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(sample_rate=1e6, duration=10.0, digital_channels=[0, 1])

        chunks = list(source.stream(chunk_duration=2.0))

        # Should have 5 chunks (10 seconds / 2 second chunks)
        assert len(chunks) == 5

        for chunk in chunks:
            from oscura.core.types import DigitalTrace

            assert isinstance(chunk, DigitalTrace)
            assert len(chunk.data) == 2000000  # 2 seconds at 1 MHz

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_stream_analog_chunks(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test streaming analog data in chunks."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(sample_rate=1e6, duration=5.0, analog_channels=[0])

        chunks = list(source.stream(chunk_duration=1.0))

        # Should have 5 chunks
        assert len(chunks) == 5

        for chunk in chunks:
            from oscura.core.types import WaveformTrace

            assert isinstance(chunk, WaveformTrace)
            assert len(chunk.data) == 1000000  # 1 second at 1 MHz

    def test_stream_closed_source_error(self) -> None:
        """Test that streaming fails if source is closed."""
        source = SaleaeSource()
        source._closed = True

        with pytest.raises(ValueError, match="Cannot stream from closed source"):
            list(source.stream())

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_stream_partial_final_chunk(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test streaming with partial final chunk."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(sample_rate=1e6, duration=3.5, digital_channels=[0])

        chunks = list(source.stream(chunk_duration=1.0))

        # Should have 4 chunks (3 full + 1 partial)
        assert len(chunks) == 4

        # Last chunk should be smaller
        assert len(chunks[-1].data) == 500000  # 0.5 seconds at 1 MHz

    def test_close(self) -> None:
        """Test closing source."""
        source = SaleaeSource()
        source.saleae = MagicMock()

        source.close()

        assert source.saleae is None
        assert source._closed is True

    def test_close_already_closed(self) -> None:
        """Test closing already closed source."""
        source = SaleaeSource()
        source.close()
        source.close()  # Should not raise

        assert source._closed is True

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_context_manager_success(self, mock_saleae_module: Mock) -> None:
        """Test context manager usage."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        with SaleaeSource(device_id="TEST") as source:
            assert source.device_id == "TEST"
            assert source._closed is False

        # Should be closed after exiting context
        assert source._closed is True

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_context_manager_with_read(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test context manager with actual read operation."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        with SaleaeSource() as source:
            source.configure(sample_rate=1e6, duration=1.0, digital_channels=[0])
            trace = source.read()
            assert trace is not None

        assert source._closed is True

    def test_repr(self) -> None:
        """Test string representation."""
        source = SaleaeSource(device_id="ABC123")
        assert repr(source) == "SaleaeSource(device_id='ABC123')"

        source_no_id = SaleaeSource()
        assert repr(source_no_id) == "SaleaeSource(device_id=None)"

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_multiple_configurations(self, mock_saleae_module: Mock) -> None:
        """Test multiple reconfigurations."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()

        # First configuration
        source.configure(sample_rate=1e6, duration=5.0, digital_channels=[0, 1])
        assert source.sample_rate == 1e6
        assert source.duration == 5.0

        # Reconfiguration
        source.configure(sample_rate=5e5, duration=10.0, analog_channels=[0])
        assert source.sample_rate == 5e5
        assert source.duration == 10.0
        assert source.analog_channels == [0]

    @patch("oscura.hardware.acquisition.saleae.saleae")
    @patch("oscura.hardware.acquisition.saleae.time.sleep")
    def test_acquisition_time_metadata(self, mock_sleep: Mock, mock_saleae_module: Mock) -> None:
        """Test that acquisition time is captured in metadata."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()
        source.configure(sample_rate=1e6, duration=1.0, digital_channels=[0])

        before = datetime.now()
        trace = source.read()
        after = datetime.now()

        assert before <= trace.metadata.acquisition_time <= after

    @patch("oscura.hardware.acquisition.saleae.saleae")
    def test_channel_name_metadata(self, mock_saleae_module: Mock) -> None:
        """Test that channel names are set in metadata."""
        mock_device = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_device

        source = SaleaeSource()

        # Digital channels
        source.configure(sample_rate=1e6, duration=1.0, digital_channels=[0, 1, 2])
        with patch("oscura.hardware.acquisition.saleae.time.sleep"):
            trace_digital = source.read()
        assert "Ch[0, 1, 2]" in trace_digital.metadata.channel

        # Analog channels
        source.configure(sample_rate=1e6, duration=1.0, analog_channels=[0])
        with patch("oscura.hardware.acquisition.saleae.time.sleep"):
            trace_analog = source.read()
        assert "Ch[0]" in trace_analog.metadata.channel
