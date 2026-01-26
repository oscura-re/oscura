"""Tests for PyVISA instrument acquisition source.

Tests comprehensive coverage of VISASource with mocked VISA connections.
Since VISA instruments are not available in CI, all PyVISA calls are mocked.

Coverage targets:
- Connection management (auto-detect, specific resource, close)
- Configuration (channels, timebase, vertical scale, record length)
- Read operations (single channel, SCPI commands)
- Streaming operations
- Error handling (import errors, no resources found, query failures)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.hardware.acquisition.visa import VISASource


class TestVISASource:
    """Test VISASource instrument acquisition functionality."""

    def test_init_basic(self) -> None:
        """Test basic initialization without connection."""
        source = VISASource()
        assert source.resource is None
        assert source.rm is None
        assert source.instrument is None
        assert source._closed is False
        assert source.channels == [1]  # Default channel
        assert source.timebase == 1e-6  # Default 1us/div
        assert source.vertical_scale == 1.0  # Default 1V/div
        assert source.record_length == 10000  # Default record length

    def test_init_with_resource(self) -> None:
        """Test initialization with specific VISA resource string."""
        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        assert source.resource == "USB0::0x0699::0x0401::INSTR"
        assert source._closed is False

    def test_init_with_kwargs(self) -> None:
        """Test initialization with additional PyVISA options."""
        source = VISASource(resource="TCPIP::192.168.1.100::INSTR", timeout=5000)
        assert source.resource == "TCPIP::192.168.1.100::INSTR"
        assert source.kwargs == {"timeout": 5000}

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_ensure_connection_success(self, mock_pyvisa: Mock) -> None:
        """Test successful connection to VISA instrument."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()
        mock_instrument.query.return_value = "TEKTRONIX,TDS2024B,0,01.02.03\n"

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source._ensure_connection()

        assert source.rm == mock_rm
        assert source.instrument is not None
        mock_pyvisa.ResourceManager.assert_called_once()
        mock_rm.open_resource.assert_called_once_with("USB0::0x0699::0x0401::INSTR")

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_ensure_connection_auto_detect(self, mock_pyvisa: Mock) -> None:
        """Test auto-detection when resource not specified."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()
        mock_instrument.query.return_value = "KEYSIGHT,MSO-X 3054A,ABC123,01.02.03\n"

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.list_resources.return_value = ["USB0::0x0957::0x17A8::INSTR"]
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource()  # No resource specified
        source._ensure_connection()

        assert source.resource == "USB0::0x0957::0x17A8::INSTR"
        mock_rm.list_resources.assert_called_once()

    def test_ensure_connection_import_error(self) -> None:
        """Test handling of missing pyvisa library."""
        source = VISASource()

        with patch(
            "oscura.hardware.acquisition.visa.pyvisa", side_effect=ImportError("pyvisa not found")
        ):
            with pytest.raises(ImportError, match="VISA source requires pyvisa library"):
                source._ensure_connection()

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_ensure_connection_no_resources(self, mock_pyvisa: Mock) -> None:
        """Test error when no VISA instruments found."""
        mock_rm = MagicMock()
        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.list_resources.return_value = []  # No resources

        source = VISASource()

        with pytest.raises(RuntimeError, match="No VISA instruments found"):
            source._ensure_connection()

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_ensure_connection_open_error(self, mock_pyvisa: Mock) -> None:
        """Test handling of resource open failure."""
        mock_rm = MagicMock()
        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.side_effect = RuntimeError("Cannot open resource")

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")

        with pytest.raises(RuntimeError, match="Failed to connect to VISA instrument"):
            source._ensure_connection()

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_ensure_connection_idn_query_failure(self, mock_pyvisa: Mock) -> None:
        """Test graceful handling of *IDN? query failure."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()
        mock_instrument.query.side_effect = RuntimeError("Query failed")

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source._ensure_connection()  # Should not raise

        assert source.instrument is not None

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_ensure_connection_idempotent(self, mock_pyvisa: Mock) -> None:
        """Test that connection is only established once."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source._ensure_connection()
        source._ensure_connection()

        # Should only call once
        mock_pyvisa.ResourceManager.assert_called_once()

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_configure_basic(self, mock_pyvisa: Mock) -> None:
        """Test basic configuration."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source.configure(
            channels=[1, 2],
            timebase=1e-6,
            vertical_scale=0.5,
            record_length=20000,
        )

        assert source.channels == [1, 2]
        assert source.timebase == 1e-6
        assert source.vertical_scale == 0.5
        assert source.record_length == 20000

        # Verify SCPI commands sent
        instrument_calls = [str(call) for call in mock_instrument.write.call_args_list]
        assert any(":TIMebase:SCALe" in str(call) for call in instrument_calls)

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_configure_partial(self, mock_pyvisa: Mock) -> None:
        """Test partial configuration (only some parameters)."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")

        # Configure only channels
        source.configure(channels=[3, 4])
        assert source.channels == [3, 4]
        assert source.timebase == 1e-6  # Default unchanged

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_configure_scpi_command_failure(
        self, mock_pyvisa: Mock, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Test that configuration continues on SCPI command failure."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()
        mock_instrument.write.side_effect = RuntimeError("SCPI error")

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source.configure(channels=[1], timebase=1e-6)  # Should not raise

        # Check for warning printed
        captured = capfd.readouterr()
        assert "Warning: Configuration command failed" in captured.out

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    @patch("oscura.hardware.acquisition.visa.time.sleep")
    def test_read_waveform(self, mock_sleep: Mock, mock_pyvisa: Mock) -> None:
        """Test reading waveform from oscilloscope."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        # Mock waveform preamble
        mock_instrument.query.side_effect = [
            "TEKTRONIX,TDS2024B,0,01.02.03\n",  # *IDN?
            "1,0,0,0,1e-9,0,0,0,0",  # Preamble
            "TEKTRONIX,TDS2024B,0,01.02.03\n",  # *IDN? for calibration
        ]

        # Mock waveform data
        mock_instrument.query_binary_values.return_value = list(range(1000))

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source.configure(channels=[1], record_length=1000)

        trace = source.read(channel=1)

        # Verify trace
        from oscura.core.types import WaveformTrace

        assert isinstance(trace, WaveformTrace)
        assert len(trace.data) == 1000
        assert trace.metadata.sample_rate == 1e9  # From preamble x_increment
        assert "CH1" in trace.metadata.channel_name
        assert "visa://" in trace.metadata.source_file

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    @patch("oscura.hardware.acquisition.visa.time.sleep")
    def test_read_default_channel(self, mock_sleep: Mock, mock_pyvisa: Mock) -> None:
        """Test reading with default (first configured) channel."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_instrument.query.side_effect = [
            "RIGOL,DS1054Z,ABC123,01.02.03\n",
            "1,0,0,0,1e-9,0,0,0,0",
            "RIGOL,DS1054Z,ABC123,01.02.03\n",
        ]
        mock_instrument.query_binary_values.return_value = [0] * 500

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x1AB1::0x04CE::INSTR")
        source.configure(channels=[2, 3])

        trace = source.read()  # No channel specified, should use channels[0]

        # Should read from channel 2
        write_calls = [str(call) for call in mock_instrument.write.call_args_list]
        assert any("CHANnel2" in str(call) for call in write_calls)

    def test_read_closed_source_error(self) -> None:
        """Test that reading fails if source is closed."""
        source = VISASource()
        source._closed = True

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    @patch("oscura.hardware.acquisition.visa.time.sleep")
    def test_read_query_error(self, mock_sleep: Mock, mock_pyvisa: Mock) -> None:
        """Test handling of waveform query errors."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()
        mock_instrument.query.side_effect = RuntimeError("Query failed")

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source.configure(channels=[1])

        with pytest.raises(RuntimeError, match="Failed to acquire waveform"):
            source.read()

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    @patch("oscura.hardware.acquisition.visa.time.sleep")
    def test_stream_waveforms(self, mock_sleep: Mock, mock_pyvisa: Mock) -> None:
        """Test streaming waveforms at intervals."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        call_count = [0]

        def query_side_effect(cmd: str) -> str:
            if "*IDN?" in cmd:
                return "TEST,SCOPE,123,1.0\n"
            # Preamble queries
            call_count[0] += 1
            return "1,0,0,0,1e-9,0,0,0,0"

        mock_instrument.query.side_effect = query_side_effect
        mock_instrument.query_binary_values.return_value = [0] * 100

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x1234::0x5678::INSTR")
        source.configure(channels=[1])

        # Stream for 2 seconds with 0.5 second intervals (should yield ~4 traces)
        with patch("oscura.hardware.acquisition.visa.time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]  # Simulated time progression

            chunks = list(source.stream(duration=2.0, interval=0.5))

            # Should get 4 traces
            assert len(chunks) == 4

            for chunk in chunks:
                from oscura.core.types import WaveformTrace

                assert isinstance(chunk, WaveformTrace)

    def test_stream_closed_source_error(self) -> None:
        """Test that streaming fails if source is closed."""
        source = VISASource()
        source._closed = True

        with pytest.raises(ValueError, match="Cannot stream from closed source"):
            list(source.stream())

    def test_close(self) -> None:
        """Test closing source."""
        source = VISASource()
        source.instrument = MagicMock()
        source.rm = MagicMock()

        source.close()

        source.instrument.close.assert_called_once()
        source.rm.close.assert_called_once()
        assert source.instrument is None
        assert source.rm is None
        assert source._closed is True

    def test_close_already_closed(self) -> None:
        """Test closing already closed source."""
        source = VISASource()
        source.close()
        source.close()  # Should not raise

        assert source._closed is True

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    def test_context_manager_success(self, mock_pyvisa: Mock) -> None:
        """Test context manager usage."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        with VISASource(resource="USB0::0x0699::0x0401::INSTR") as source:
            assert source.resource == "USB0::0x0699::0x0401::INSTR"
            assert source._closed is False

        # Should be closed after exiting context
        assert source._closed is True

    def test_repr(self) -> None:
        """Test string representation."""
        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        assert repr(source) == "VISASource(resource='USB0::0x0699::0x0401::INSTR')"

        source_auto = VISASource()
        assert repr(source_auto) == "VISASource(resource=None)"

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    @patch("oscura.hardware.acquisition.visa.time.sleep")
    def test_acquisition_time_metadata(self, mock_sleep: Mock, mock_pyvisa: Mock) -> None:
        """Test that acquisition time is captured in metadata."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_instrument.query.return_value = "1,0,0,0,1e-9,0,0,0,0"
        mock_instrument.query_binary_values.return_value = [0] * 100

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x0699::0x0401::INSTR")
        source.configure(channels=[1])

        before = datetime.now()
        trace = source.read()
        after = datetime.now()

        assert before <= trace.metadata.acquisition_time <= after

    @patch("oscura.hardware.acquisition.visa.pyvisa")
    @patch("oscura.hardware.acquisition.visa.time.sleep")
    def test_calibration_info_metadata(self, mock_sleep: Mock, mock_pyvisa: Mock) -> None:
        """Test that calibration info includes instrument ID."""
        mock_rm = MagicMock()
        mock_instrument = MagicMock()

        mock_instrument.query.side_effect = [
            "1,0,0,0,1e-9,0,0,0,0",  # Preamble
            "CUSTOM,INSTRUMENT,SN12345,v2.0\n",  # *IDN? for calibration
        ]
        mock_instrument.query_binary_values.return_value = [0] * 100

        mock_pyvisa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_instrument

        source = VISASource(resource="USB0::0x1234::0x5678::INSTR")
        source.configure(channels=[1])

        trace = source.read()

        assert trace.metadata.calibration_info is not None
        assert "CUSTOM,INSTRUMENT,SN12345,v2.0" in trace.metadata.calibration_info.instrument
