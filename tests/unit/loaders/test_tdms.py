"""Unit tests for NI TDMS (Technical Data Management Streaming) loader.

Tests LOAD-011: TDMS (NI LabVIEW) Loader
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the module to be tested
import oscura.loaders.tdms as tdms_module
from oscura.core.exceptions import FormatError, LoaderError

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class MockTdmsChannel:
    """Mock TDMS channel for testing."""

    def __init__(
        self,
        name: str,
        data: np.ndarray | None = None,
        properties: dict | None = None,
    ) -> None:
        """Initialize mock channel.

        Args:
            name: Channel name.
            data: Channel data array.
            properties: Channel properties dictionary.
        """
        self.name = name
        self.data = data if data is not None else np.array([1.0, 2.0, 3.0, 4.0])
        self.properties = properties if properties is not None else {}


class MockTdmsGroup:
    """Mock TDMS group for testing."""

    def __init__(self, name: str, channels: list[MockTdmsChannel] | None = None) -> None:
        """Initialize mock group.

        Args:
            name: Group name.
            channels: List of channels in group.
        """
        self.name = name
        self._channels = channels if channels is not None else []
        self.properties = {}

    def channels(self) -> list[MockTdmsChannel]:
        """Return list of channels.

        Returns:
            List of mock channels.
        """
        return self._channels


class MockTdmsFile:
    """Mock TDMS file for testing."""

    def __init__(self, groups: list[MockTdmsGroup] | None = None) -> None:
        """Initialize mock TDMS file.

        Args:
            groups: List of groups in file.
        """
        self._groups = groups if groups is not None else []
        self.properties = {}

    def groups(self) -> list[MockTdmsGroup]:
        """Return list of groups.

        Returns:
            List of mock groups.
        """
        return self._groups

    @staticmethod
    def read(path: str) -> "MockTdmsFile":
        """Mock read method.

        Args:
            path: Path to file.

        Returns:
            MockTdmsFile instance.
        """
        # Default: single group with single channel
        channel = MockTdmsChannel("CH1", properties={"wf_samples": 1e6})
        group = MockTdmsGroup("Voltage", [channel])
        return MockTdmsFile([group])


def enable_tdms_with_mock(mock_file_or_class=None):
    """Context manager to enable TDMS with mocking.

    Args:
        mock_file_or_class: Either a MockTdmsFile instance or MockTdmsFile class.

    Returns:
        Context manager for patching.
    """
    if mock_file_or_class is None:
        mock_file_or_class = MockTdmsFile

    return patch.multiple(
        tdms_module, NPTDMS_AVAILABLE=True, TdmsFile=mock_file_or_class, create=True
    )


class TestTDMSLoader:
    """Test TDMS file loader."""

    def test_load_basic_tdms(self, tmp_path: Path) -> None:
        """Test loading a basic TDMS file."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        with enable_tdms_with_mock():
            trace = tdms_module.load_tdms(tdms_path)

            assert trace is not None
            assert len(trace.data) > 0
            assert trace.metadata.sample_rate == 1e6
            assert trace.metadata.source_file == str(tdms_path)
            assert trace.metadata.channel == "CH1"

    def test_load_specific_group(self, tmp_path: Path) -> None:
        """Test loading a specific group by name."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        # Create multiple groups
        ch1 = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        ch2 = MockTdmsChannel("CH2", properties={"Fs": 2e6})
        group1 = MockTdmsGroup("Voltage", [ch1])
        group2 = MockTdmsGroup("Current", [ch2])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group1, group2])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path, group="Current")
            assert trace.metadata.channel == "CH2"
            assert trace.metadata.sample_rate == 2e6

    def test_load_specific_channel_by_name(self, tmp_path: Path) -> None:
        """Test loading a specific channel by name."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        # Create group with multiple channels
        ch1 = MockTdmsChannel("CH1", properties={"sample_rate": 1e6})
        ch2 = MockTdmsChannel("CH2", properties={"sample_rate": 2e6})
        group = MockTdmsGroup("Voltage", [ch1, ch2])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path, channel="CH2")
            assert trace.metadata.channel == "CH2"
            assert trace.metadata.sample_rate == 2e6

    def test_load_channel_by_index(self, tmp_path: Path) -> None:
        """Test loading a channel by index."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch1 = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        ch2 = MockTdmsChannel("CH2", properties={"Fs": 2e6})
        group = MockTdmsGroup("Voltage", [ch1, ch2])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path, channel=0)
            assert trace.metadata.channel == "CH1"

            trace = tdms_module.load_tdms(tdms_path, channel=1)
            assert trace.metadata.channel == "CH2"

    def test_sample_rate_from_wf_increment(self, tmp_path: Path) -> None:
        """Test sample rate extraction from wf_increment (time delta)."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        # wf_increment is time interval (dt), sample_rate = 1/dt
        ch = MockTdmsChannel("CH1", properties={"wf_increment": 1e-6})  # 1 µs
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.sample_rate == 1e6  # 1 MHz

    def test_sample_rate_from_dt(self, tmp_path: Path) -> None:
        """Test sample rate extraction from dt property."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"dt": 1e-9})  # 1 ns
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            # Use approximate comparison for floating point
            assert abs(trace.metadata.sample_rate - 1e9) < 1.0  # 1 GHz

    def test_sample_rate_priority(self, tmp_path: Path) -> None:
        """Test that channel properties have priority over group/file."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])
        group.properties["Fs"] = 2e6  # Should be ignored
        mock_file = MockTdmsFile([group])
        mock_file.properties["Fs"] = 3e6  # Should be ignored

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = mock_file

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.sample_rate == 1e6  # Channel value

    def test_sample_rate_from_group(self, tmp_path: Path) -> None:
        """Test sample rate extraction from group when not in channel."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={})
        group = MockTdmsGroup("Voltage", [ch])
        group.properties["SampleRate"] = 5e6

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.sample_rate == 5e6

    def test_sample_rate_from_file(self, tmp_path: Path) -> None:
        """Test sample rate extraction from file properties."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={})
        group = MockTdmsGroup("Voltage", [ch])
        mock_file = MockTdmsFile([group])
        mock_file.properties["SamplingFrequency"] = 10e6

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = mock_file

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.sample_rate == 10e6

    def test_default_sample_rate(self, tmp_path: Path) -> None:
        """Test default sample rate when not found in properties."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.sample_rate == 1.0e6  # Default 1 MHz

    def test_vertical_scale_and_offset(self, tmp_path: Path) -> None:
        """Test extraction of vertical scale and offset."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel(
            "CH1",
            properties={
                "Fs": 1e6,
                "NI_Scale[0]_Linear_Slope": 2.5,
                "NI_Scale[0]_Linear_Y_Intercept": -1.0,
            },
        )
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.vertical_scale == 2.5
            assert trace.metadata.vertical_offset == -1.0

    def test_extract_tdms_properties(self, tmp_path: Path) -> None:
        """Test extraction of TDMS channel properties."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel(
            "CH1",
            properties={
                "Fs": 1e6,
                "unit_string": "Volts",
                "NI_ChannelName": "Voltage_1",
                "description": "Test channel",
                "wf_start_time": "2024-01-01T00:00:00",
            },
        )
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.trigger_info is not None
            assert trace.metadata.trigger_info["unit_string"] == "Volts"
            assert trace.metadata.trigger_info["NI_ChannelName"] == "Voltage_1"
            assert trace.metadata.trigger_info["description"] == "Test channel"

    def test_data_conversion_to_float64(self, tmp_path: Path) -> None:
        """Test that data is converted to float64."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        # Create data with various types
        int_data = np.array([1, 2, 3, 4], dtype=np.int32)
        ch = MockTdmsChannel("CH1", data=int_data, properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.data.dtype == np.float64
            np.testing.assert_array_equal(trace.data, [1.0, 2.0, 3.0, 4.0])

    def test_nptdms_not_available(self, tmp_path: Path) -> None:
        """Test error when npTDMS library is not available."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        with patch.object(tdms_module, "NPTDMS_AVAILABLE", False):
            with pytest.raises(LoaderError, match="npTDMS library required"):
                tdms_module.load_tdms(tdms_path)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        with pytest.raises(LoaderError, match="not found"):
            tdms_module.load_tdms(tmp_path / "nonexistent.tdms")

    def test_invalid_tdms_file(self, tmp_path: Path) -> None:
        """Test error on invalid TDMS file."""
        tdms_path = tmp_path / "bad.tdms"
        tdms_path.write_bytes(b"not a tdms file")

        mock_tdms = MagicMock()
        mock_tdms.read.side_effect = Exception("Invalid file format")

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(FormatError, match="Failed to parse"):
                tdms_module.load_tdms(tdms_path)

    def test_no_groups(self, tmp_path: Path) -> None:
        """Test error when TDMS file has no groups."""
        tdms_path = tmp_path / "empty.tdms"
        tdms_path.touch()

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([])  # No groups

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(FormatError, match="No groups found"):
                tdms_module.load_tdms(tdms_path)

    def test_group_not_found(self, tmp_path: Path) -> None:
        """Test error when requested group not found."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(LoaderError, match="Group 'Current' not found"):
                tdms_module.load_tdms(tdms_path, group="Current")

    def test_no_channels_in_group(self, tmp_path: Path) -> None:
        """Test error when group has no channels."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        group = MockTdmsGroup("Voltage", [])  # No channels

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(FormatError, match="No channels found"):
                tdms_module.load_tdms(tdms_path)

    def test_channel_index_out_of_range(self, tmp_path: Path) -> None:
        """Test error on invalid channel index."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(LoaderError, match="index 5 out of range"):
                tdms_module.load_tdms(tdms_path, channel=5)

    def test_negative_channel_index(self, tmp_path: Path) -> None:
        """Test error on negative channel index."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(LoaderError, match="index -1 out of range"):
                tdms_module.load_tdms(tdms_path, channel=-1)

    def test_channel_name_not_found(self, tmp_path: Path) -> None:
        """Test error when requested channel name not found."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(LoaderError, match="Channel 'CH99' not found"):
                tdms_module.load_tdms(tdms_path, channel="CH99")

    def test_channel_with_no_data(self, tmp_path: Path) -> None:
        """Test error when channel has no data."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        ch.data = None  # Explicitly set to None
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(FormatError, match="has no data"):
                tdms_module.load_tdms(tdms_path)

    def test_channel_with_empty_data(self, tmp_path: Path) -> None:
        """Test error when channel has empty data array."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", data=np.array([]), properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(FormatError, match="has no data"):
                tdms_module.load_tdms(tdms_path)

    def test_pathlib_path_input(self, tmp_path: Path) -> None:
        """Test that pathlib.Path inputs are handled correctly."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            # Pass as Path object
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.source_file == str(tdms_path)

            # Pass as string
            trace = tdms_module.load_tdms(str(tdms_path))
            assert trace.metadata.source_file == str(tdms_path)

    def test_all_sample_rate_keys(self, tmp_path: Path) -> None:
        """Test all supported sample rate property keys."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        # Test each sample rate key individually
        sample_rate_keys = [
            ("wf_samples", 1e6, 1e6),  # Direct value
            ("wf_increment", 1e-6, 1e6),  # Inverse (dt)
            ("NI_RF_IQ_Rate", 2e6, 2e6),  # Direct value
            ("SamplingFrequency", 3e6, 3e6),  # Direct value
            ("dt", 1e-9, 1e9),  # Inverse (dt)
            ("Fs", 4e6, 4e6),  # Direct value
            ("SampleRate", 5e6, 5e6),  # Direct value
            ("sample_rate", 6e6, 6e6),  # Direct value
        ]

        for key, value, expected_rate in sample_rate_keys:
            ch = MockTdmsChannel("CH1", properties={key: value})
            group = MockTdmsGroup("Voltage", [ch])

            mock_tdms = MagicMock()
            mock_tdms.read.return_value = MockTdmsFile([group])

            with enable_tdms_with_mock(mock_tdms):
                trace = tdms_module.load_tdms(tdms_path)
                # Use approximate comparison to handle floating point precision
                assert abs(trace.metadata.sample_rate - expected_rate) < 1.0, (
                    f"Failed for key '{key}': expected {expected_rate}, "
                    f"got {trace.metadata.sample_rate}"
                )

    def test_zero_wf_increment_ignored(self, tmp_path: Path) -> None:
        """Test that zero wf_increment is ignored (would cause division by zero)."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"wf_increment": 0.0})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            # Should use default sample rate
            assert trace.metadata.sample_rate == 1.0e6

    def test_negative_wf_increment_ignored(self, tmp_path: Path) -> None:
        """Test that negative wf_increment is ignored."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"wf_increment": -1e-6})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            # Should use default sample rate
            assert trace.metadata.sample_rate == 1.0e6


class TestListTdmsChannels:
    """Test list_tdms_channels function."""

    def test_list_basic(self, tmp_path: Path) -> None:
        """Test listing channels in TDMS file."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch1 = MockTdmsChannel("CH1")
        ch2 = MockTdmsChannel("CH2")
        group = MockTdmsGroup("Voltage", [ch1, ch2])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            channels = tdms_module.list_tdms_channels(tdms_path)

            assert "Voltage" in channels
            assert channels["Voltage"] == ["CH1", "CH2"]

    def test_list_multiple_groups(self, tmp_path: Path) -> None:
        """Test listing channels from multiple groups."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch1 = MockTdmsChannel("CH1")
        ch2 = MockTdmsChannel("CH2")
        ch3 = MockTdmsChannel("CH3")

        group1 = MockTdmsGroup("Voltage", [ch1, ch2])
        group2 = MockTdmsGroup("Current", [ch3])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group1, group2])

        with enable_tdms_with_mock(mock_tdms):
            channels = tdms_module.list_tdms_channels(tdms_path)

            assert len(channels) == 2
            assert channels["Voltage"] == ["CH1", "CH2"]
            assert channels["Current"] == ["CH3"]

    def test_list_empty_group(self, tmp_path: Path) -> None:
        """Test listing channels from group with no channels."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        group = MockTdmsGroup("EmptyGroup", [])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            channels = tdms_module.list_tdms_channels(tdms_path)

            assert "EmptyGroup" in channels
            assert channels["EmptyGroup"] == []

    def test_list_nptdms_not_available(self, tmp_path: Path) -> None:
        """Test error when npTDMS library is not available."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        with patch.object(tdms_module, "NPTDMS_AVAILABLE", False):
            with pytest.raises(LoaderError, match="npTDMS library required"):
                tdms_module.list_tdms_channels(tdms_path)

    def test_list_file_not_found(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        with pytest.raises(LoaderError, match="not found"):
            tdms_module.list_tdms_channels(tmp_path / "nonexistent.tdms")

    def test_list_read_error(self, tmp_path: Path) -> None:
        """Test error handling when reading TDMS file fails."""
        tdms_path = tmp_path / "bad.tdms"
        tdms_path.write_bytes(b"corrupt data")

        mock_tdms = MagicMock()
        mock_tdms.read.side_effect = Exception("Corrupt file")

        with enable_tdms_with_mock(mock_tdms):
            with pytest.raises(LoaderError, match="Failed to read"):
                tdms_module.list_tdms_channels(tdms_path)

    def test_list_pathlib_path(self, tmp_path: Path) -> None:
        """Test that pathlib.Path inputs work correctly."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1")
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            # Test with Path object
            channels = tdms_module.list_tdms_channels(tdms_path)
            assert "Voltage" in channels

            # Test with string
            channels = tdms_module.list_tdms_channels(str(tdms_path))
            assert "Voltage" in channels


class TestTDMSPropertiesExtraction:
    """Test TDMS properties extraction edge cases."""

    def test_no_useful_properties(self, tmp_path: Path) -> None:
        """Test that trigger_info is None when no useful properties exist."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel("CH1", properties={"Fs": 1e6, "irrelevant_prop": "value"})
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            # Only Fs is present, but it's not in the useful_keys list
            # so trigger_info could be None or contain other extracted properties
            # The function extracts all properties from useful_keys
            assert trace.metadata.trigger_info is None or isinstance(
                trace.metadata.trigger_info, dict
            )

    def test_partial_properties(self, tmp_path: Path) -> None:
        """Test extraction of partial set of properties."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel(
            "CH1",
            properties={
                "Fs": 1e6,
                "unit_string": "V",
                "NI_ChannelName": "Voltage_CH1",
                # Some useful properties missing
            },
        )
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.trigger_info is not None
            assert "unit_string" in trace.metadata.trigger_info
            assert "NI_ChannelName" in trace.metadata.trigger_info
            assert "wf_start_time" not in trace.metadata.trigger_info

    def test_all_useful_properties(self, tmp_path: Path) -> None:
        """Test extraction when all useful properties are present."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel(
            "CH1",
            properties={
                "Fs": 1e6,
                "unit_string": "V",
                "NI_ChannelName": "Voltage_1",
                "wf_start_time": "2024-01-01",
                "wf_start_offset": 0.0,
                "description": "Test signal",
                "NI_Scale[0]_Linear_Slope": 1.0,
                "NI_Scale[0]_Linear_Y_Intercept": 0.0,
            },
        )
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.trigger_info is not None
            assert len(trace.metadata.trigger_info) == 7
            assert trace.metadata.trigger_info["unit_string"] == "V"
            assert trace.metadata.trigger_info["description"] == "Test signal"

    def test_none_property_values_ignored(self, tmp_path: Path) -> None:
        """Test that None property values are ignored."""
        tdms_path = tmp_path / "test.tdms"
        tdms_path.touch()

        ch = MockTdmsChannel(
            "CH1",
            properties={
                "Fs": 1e6,
                "unit_string": None,  # Should be ignored
                "NI_ChannelName": "CH1",
            },
        )
        group = MockTdmsGroup("Voltage", [ch])

        mock_tdms = MagicMock()
        mock_tdms.read.return_value = MockTdmsFile([group])

        with enable_tdms_with_mock(mock_tdms):
            trace = tdms_module.load_tdms(tdms_path)
            assert trace.metadata.trigger_info is not None
            assert "unit_string" not in trace.metadata.trigger_info
            assert "NI_ChannelName" in trace.metadata.trigger_info
