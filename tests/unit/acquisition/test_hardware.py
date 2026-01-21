"""Tests for hardware acquisition sources.

These tests use mocked hardware to verify source implementations without
requiring actual hardware devices.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.core.types import DigitalTrace, WaveformTrace

pytestmark = pytest.mark.unit


class TestSocketCANSource:
    """Tests for SocketCANSource with mocked python-can."""

    def test_creation(self) -> None:
        """Test basic source creation."""
        from oscura.acquisition.socketcan import SocketCANSource

        source = SocketCANSource("can0", bitrate=500000)

        assert source.interface == "can0"
        assert source.bitrate == 500000
        assert not source._closed
        assert source.bus is None

    def test_read_with_mocked_can(self) -> None:
        """Test reading from mocked CAN bus."""
        # Create mock can module
        mock_can = MagicMock()
        mock_bus = MagicMock()
        mock_msg = Mock()
        mock_msg.timestamp = 1.0
        mock_msg.arbitration_id = 0x123
        mock_bus.recv.return_value = mock_msg
        mock_can.Bus.return_value = mock_bus

        with patch.dict(sys.modules, {"can": mock_can}):
            from oscura.acquisition.socketcan import SocketCANSource

            source = SocketCANSource("vcan0")

            # Mock time to speed up test
            with patch("time.time") as mock_time:
                mock_time.side_effect = [0, 0.1, 0.2, 11]  # Simulate timeout
                trace = source.read(duration=0.2)

            assert isinstance(trace, DigitalTrace)
            assert source.bus is mock_bus

    def test_read_no_messages(self) -> None:
        """Test reading when no messages received."""
        mock_can = MagicMock()
        mock_bus = MagicMock()
        mock_bus.recv.return_value = None  # No messages
        mock_can.Bus.return_value = mock_bus

        with patch.dict(sys.modules, {"can": mock_can}):
            from oscura.acquisition.socketcan import SocketCANSource

            source = SocketCANSource("vcan0")

            with patch("time.time") as mock_time:
                mock_time.side_effect = [0, 11]  # Immediate timeout
                trace = source.read(duration=0.1)

            assert isinstance(trace, DigitalTrace)
            assert len(trace.data) == 0

    def test_streaming(self) -> None:
        """Test streaming CAN messages."""
        mock_can = MagicMock()
        mock_bus = MagicMock()
        messages = []
        for i in range(10):
            msg = Mock()
            msg.timestamp = i * 0.1
            msg.arbitration_id = 0x100 + i
            messages.append(msg)

        mock_bus.recv.side_effect = messages + [None] * 100
        mock_can.Bus.return_value = mock_bus

        with patch.dict(sys.modules, {"can": mock_can}):
            from oscura.acquisition.socketcan import SocketCANSource

            source = SocketCANSource("vcan0")

            with patch("time.time") as mock_time:
                # Simulate time progression
                times = [0] + [i * 0.01 for i in range(20)] + [100]
                mock_time.side_effect = times

                chunks = list(source.stream(duration=1.0, chunk_size=5))

            assert len(chunks) == 2  # 10 messages / 5 per chunk
            assert all(isinstance(c, DigitalTrace) for c in chunks)

    def test_context_manager(self) -> None:
        """Test context manager support."""
        mock_can = MagicMock()
        mock_bus = MagicMock()
        mock_can.Bus.return_value = mock_bus

        with patch.dict(sys.modules, {"can": mock_can}):
            from oscura.acquisition.socketcan import SocketCANSource

            with SocketCANSource("vcan0") as source:
                # Trigger bus creation
                source._ensure_bus()
                assert not source._closed

            # Should be closed and bus shutdown
            assert source._closed
            mock_bus.shutdown.assert_called_once()

    def test_import_error(self) -> None:
        """Test error when python-can not installed."""
        # Remove can module if it exists
        with patch.dict(sys.modules, {"can": None}):
            from oscura.acquisition.socketcan import SocketCANSource

            source = SocketCANSource("can0")

            with pytest.raises(ImportError, match="python-can"):
                source._ensure_bus()

    def test_closed_source_raises(self) -> None:
        """Test that closed source raises error."""
        from oscura.acquisition.socketcan import SocketCANSource

        source = SocketCANSource("can0")
        source.close()

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()


class TestSaleaeSource:
    """Tests for SaleaeSource with mocked saleae library."""

    def test_creation(self) -> None:
        """Test basic source creation."""
        from oscura.acquisition.saleae import SaleaeSource

        source = SaleaeSource(device_id="ABC123")

        assert source.device_id == "ABC123"
        assert not source._closed
        assert source.saleae is None

    def test_configure(self) -> None:
        """Test configuration."""
        mock_saleae_module = MagicMock()
        mock_saleae = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_saleae

        with patch.dict(sys.modules, {"saleae": mock_saleae_module}):
            from oscura.acquisition.saleae import SaleaeSource

            source = SaleaeSource()
            source.configure(sample_rate=1e6, duration=10, digital_channels=[0, 1, 2, 3])

            assert source.sample_rate == 1e6
            assert source.duration == 10
            assert source.digital_channels == [0, 1, 2, 3]

    def test_configure_no_channels_raises(self) -> None:
        """Test that configure without channels raises error."""
        from oscura.acquisition.saleae import SaleaeSource

        source = SaleaeSource()

        with pytest.raises(ValueError, match="at least one"):
            source.configure(sample_rate=1e6, duration=10)

    def test_read_digital(self) -> None:
        """Test reading digital channels."""
        mock_saleae_module = MagicMock()
        mock_saleae = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_saleae

        with patch.dict(sys.modules, {"saleae": mock_saleae_module}):
            from oscura.acquisition.saleae import SaleaeSource

            with patch("time.sleep"):
                source = SaleaeSource()
                source.configure(sample_rate=1e6, duration=0.01, digital_channels=[0, 1])

                trace = source.read()

            assert isinstance(trace, DigitalTrace)
            assert len(trace.data) == 10000  # 1 MS/s * 0.01s
            assert trace.metadata.sample_rate == 1e6

    def test_read_analog(self) -> None:
        """Test reading analog channels."""
        mock_saleae_module = MagicMock()
        mock_saleae = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_saleae

        with patch.dict(sys.modules, {"saleae": mock_saleae_module}):
            from oscura.acquisition.saleae import SaleaeSource

            with patch("time.sleep"):
                source = SaleaeSource()
                source.configure(sample_rate=1e6, duration=0.01, analog_channels=[0, 1])

                trace = source.read()

            assert isinstance(trace, WaveformTrace)
            assert len(trace.data) == 10000
            assert trace.metadata.sample_rate == 1e6

    def test_streaming(self) -> None:
        """Test streaming functionality."""
        mock_saleae_module = MagicMock()
        mock_saleae = MagicMock()
        mock_saleae_module.Saleae.return_value = mock_saleae

        with patch.dict(sys.modules, {"saleae": mock_saleae_module}):
            from oscura.acquisition.saleae import SaleaeSource

            with patch("time.sleep"):
                source = SaleaeSource()
                source.configure(sample_rate=1e6, duration=0.01, digital_channels=[0])

                chunks = list(source.stream(chunk_duration=0.002))

            assert len(chunks) == 5  # 0.01s / 0.002s per chunk
            assert all(isinstance(c, DigitalTrace) for c in chunks)

    def test_context_manager(self) -> None:
        """Test context manager support."""
        from oscura.acquisition.saleae import SaleaeSource

        source = SaleaeSource()

        with source:
            assert not source._closed

        assert source._closed

    def test_import_error(self) -> None:
        """Test error when saleae not installed."""
        with patch.dict(sys.modules, {"saleae": None}):
            from oscura.acquisition.saleae import SaleaeSource

            source = SaleaeSource()

            with pytest.raises(ImportError, match="saleae"):
                source._ensure_connection()

    def test_read_before_configure_raises(self) -> None:
        """Test that reading before configure raises error."""
        from oscura.acquisition.saleae import SaleaeSource

        source = SaleaeSource()

        with pytest.raises(ValueError, match="not configured"):
            source.read()


class TestVISASource:
    """Tests for VISASource with mocked pyvisa."""

    def test_creation(self) -> None:
        """Test basic source creation."""
        from oscura.acquisition.visa import VISASource

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")

        assert source.resource == "USB0::0x0699::0x0401::INSTR"
        assert not source._closed
        assert source.instrument is None

    def test_configure(self) -> None:
        """Test configuration."""
        mock_pyvisa = MagicMock()
        mock_rm = MagicMock()
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        mock_inst.query.return_value = "TEKTRONIX,DPO7254C,C012345,CF:91.1CT"
        mock_pyvisa.ResourceManager.return_value = mock_rm

        with patch.dict(sys.modules, {"pyvisa": mock_pyvisa}):
            from oscura.acquisition.visa import VISASource

            source = VISASource("USB0::0x0699::0x0401::INSTR")
            source.configure(
                channels=[1, 2], timebase=1e-6, vertical_scale=0.5, record_length=10000
            )

            assert source.channels == [1, 2]
            assert source.timebase == 1e-6
            assert source.vertical_scale == 0.5
            assert source.record_length == 10000

    def test_read_waveform(self) -> None:
        """Test reading waveform from oscilloscope."""
        mock_pyvisa = MagicMock()
        mock_rm = MagicMock()
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst

        # Mock waveform data
        preamble = "1,0,10000,1,1.0e-9,0,0,0,0,0"
        mock_inst.query.side_effect = [
            "TEKTRONIX,DPO7254C,C012345,CF:91.1CT",
            preamble,
            "TEKTRONIX,DPO7254C,C012345,CF:91.1CT",
        ]
        waveform_data = list(range(1000))
        mock_inst.query_binary_values.return_value = waveform_data
        mock_pyvisa.ResourceManager.return_value = mock_rm

        with patch.dict(sys.modules, {"pyvisa": mock_pyvisa}):
            from oscura.acquisition.visa import VISASource

            with patch("time.sleep"):
                source = VISASource("USB0::0x0699::0x0401::INSTR")
                source.configure(channels=[1])

                trace = source.read(channel=1)

            assert isinstance(trace, WaveformTrace)
            assert len(trace.data) == 1000
            assert trace.metadata.channel_name == "CH1"
            assert trace.metadata.calibration_info is not None

    def test_auto_detect_instrument(self) -> None:
        """Test auto-detection of VISA instruments."""
        mock_pyvisa = MagicMock()
        mock_rm = MagicMock()
        mock_rm.list_resources.return_value = ["USB0::0x0699::0x0401::INSTR"]
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        mock_inst.query.return_value = "TEKTRONIX,DPO7254C,C012345,CF:91.1CT"
        mock_pyvisa.ResourceManager.return_value = mock_rm

        with patch.dict(sys.modules, {"pyvisa": mock_pyvisa}):
            from oscura.acquisition.visa import VISASource

            source = VISASource()  # No resource specified
            source._ensure_connection()

            assert source.resource == "USB0::0x0699::0x0401::INSTR"

    def test_streaming(self) -> None:
        """Test streaming waveforms."""
        mock_pyvisa = MagicMock()
        mock_rm = MagicMock()
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        mock_inst.query.side_effect = [
            "TEKTRONIX,DPO7254C,C012345,CF:91.1CT",  # Connection
            "1,0,1000,1,1.0e-9,0,0,0,0,0",  # First read preamble
            "TEKTRONIX,DPO7254C,C012345,CF:91.1CT",  # Second read ID
            "1,0,1000,1,1.0e-9,0,0,0,0,0",  # Second read preamble
            "TEKTRONIX,DPO7254C,C012345,CF:91.1CT",  # Third read ID
        ]
        mock_inst.query_binary_values.return_value = list(range(1000))
        mock_pyvisa.ResourceManager.return_value = mock_rm

        with patch.dict(sys.modules, {"pyvisa": mock_pyvisa}):
            from oscura.acquisition.visa import VISASource

            with patch("time.time") as mock_time:
                with patch("time.sleep"):
                    # time() is called at start, then after each acquisition
                    # Need to provide enough calls for the loop
                    mock_time.side_effect = [0, 0.5, 1.5, 2.5]  # Start + 2 iterations + exit

                    source = VISASource("USB0::0x0699::0x0401::INSTR")
                    source.configure(channels=[1])

                    chunks = list(source.stream(duration=2, interval=1))

            assert len(chunks) == 2
            assert all(isinstance(c, WaveformTrace) for c in chunks)

    def test_context_manager(self) -> None:
        """Test context manager support."""
        mock_pyvisa = MagicMock()
        mock_rm = MagicMock()
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        mock_inst.query.return_value = "TEKTRONIX,DPO7254C,C012345,CF:91.1CT"
        mock_pyvisa.ResourceManager.return_value = mock_rm

        with patch.dict(sys.modules, {"pyvisa": mock_pyvisa}):
            from oscura.acquisition.visa import VISASource

            with VISASource("USB0::0x0699::0x0401::INSTR") as source:
                source._ensure_connection()
                assert not source._closed

            # Should be closed
            assert source._closed
            mock_inst.close.assert_called_once()
            mock_rm.close.assert_called_once()

    def test_import_error(self) -> None:
        """Test error when pyvisa not installed."""
        with patch.dict(sys.modules, {"pyvisa": None}):
            from oscura.acquisition.visa import VISASource

            source = VISASource()

            with pytest.raises(ImportError, match="pyvisa"):
                source._ensure_connection()

    def test_closed_source_raises(self) -> None:
        """Test that closed source raises error."""
        from oscura.acquisition.visa import VISASource

        source = VISASource()
        source.close()

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()


class TestHardwareSourceFactory:
    """Tests for HardwareSource factory methods."""

    def test_socketcan_factory(self) -> None:
        """Test SocketCAN factory method."""
        from oscura.acquisition import HardwareSource

        source = HardwareSource.socketcan("can0", bitrate=500000)

        assert source.interface == "can0"
        assert source.bitrate == 500000

    def test_saleae_factory(self) -> None:
        """Test Saleae factory method."""
        from oscura.acquisition import HardwareSource

        source = HardwareSource.saleae(device_id="ABC123")

        assert source.device_id == "ABC123"

    def test_visa_factory(self) -> None:
        """Test VISA factory method."""
        from oscura.acquisition import HardwareSource

        source = HardwareSource.visa("USB0::0x0699::0x0401::INSTR")

        assert source.resource == "USB0::0x0699::0x0401::INSTR"
