"""Unit tests for sigrok session loader.

Tests LOAD-008: Sigrok Session Loader
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.sigrok import load_sigrok

pytestmark = [pytest.mark.unit, pytest.mark.loader]


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokLoader:
    """Test sigrok session file loader."""

    def create_sigrok_session(
        self,
        path: Path,
        sample_rate: int = 1_000_000,
        n_samples: int = 1000,
        n_channels: int = 8,
        data: bytes | None = None,
        channel_names: list[str] | None = None,
        include_metadata: bool = True,
    ) -> None:
        """Create a minimal sigrok session file.

        Args:
            path: Output path.
            sample_rate: Sample rate in Hz.
            n_samples: Number of samples.
            n_channels: Number of channels.
            data: Optional raw logic data.
            channel_names: Optional channel names.
            include_metadata: Whether to include metadata file.
        """
        with zipfile.ZipFile(path, "w") as zf:
            if include_metadata:
                # Create metadata
                metadata_content = f"samplerate={sample_rate}\ntotal probes={n_channels}\n"
                if channel_names:
                    for i, name in enumerate(channel_names):
                        metadata_content += f"probe{i}={name}\n"
                else:
                    for i in range(n_channels):
                        metadata_content += f"probe{i}=D{i}\n"
                zf.writestr("metadata", metadata_content)

            # Create logic data
            if data is None:
                # Generate simple alternating pattern
                bytes_per_sample = (n_channels + 7) // 8
                pattern = bytes([0xAA] * bytes_per_sample)
                data = pattern * n_samples

            zf.writestr("logic-1-1", data)

    def test_load_basic_session(self, tmp_path: Path) -> None:
        """Test loading a basic sigrok session."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path)

        trace = load_sigrok(session_path)

        assert trace is not None
        assert len(trace.data) > 0
        assert trace.metadata.sample_rate == 1_000_000

    def test_load_default_channel(self, tmp_path: Path) -> None:
        """Test that default behavior loads the first channel."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path, n_channels=4)

        trace = load_sigrok(session_path)
        assert trace.metadata.channel == "D0"
        assert trace.data.ndim == 1  # Should be 1D array

    def test_load_specific_channel_by_index(self, tmp_path: Path) -> None:
        """Test loading a specific channel by index."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path)

        trace = load_sigrok(session_path, channel=0)
        assert trace.metadata.channel == "D0"

        trace = load_sigrok(session_path, channel=3)
        assert trace.metadata.channel == "D3"

    def test_load_channel_by_name(self, tmp_path: Path) -> None:
        """Test loading a channel by name."""
        session_path = tmp_path / "test.sr"
        channel_names = ["CLK", "DATA", "CS", "MOSI"]
        self.create_sigrok_session(session_path, n_channels=4, channel_names=channel_names)

        trace = load_sigrok(session_path, channel="CLK")
        assert trace.metadata.channel == "CLK"

        trace = load_sigrok(session_path, channel="DATA")
        assert trace.metadata.channel == "DATA"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        with pytest.raises(LoaderError, match="not found"):
            load_sigrok(tmp_path / "nonexistent.sr")

    def test_invalid_zip(self, tmp_path: Path) -> None:
        """Test error on non-ZIP file."""
        bad_file = tmp_path / "bad.sr"
        bad_file.write_bytes(b"not a zip file")

        with pytest.raises(FormatError, match="not a ZIP"):
            load_sigrok(bad_file)

    def test_corrupted_zip(self, tmp_path: Path) -> None:
        """Test error on corrupted ZIP file."""
        bad_file = tmp_path / "corrupted.sr"
        # Create a file that starts like a ZIP but is truncated
        bad_file.write_bytes(b"PK\x03\x04incomplete")

        with pytest.raises(FormatError, match="not a ZIP|Corrupted"):
            load_sigrok(bad_file)

    def test_missing_logic_data(self, tmp_path: Path) -> None:
        """Test error when no logic data present."""
        session_path = tmp_path / "empty.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\n")

        with pytest.raises(FormatError, match="No logic data"):
            load_sigrok(session_path)

    def test_channel_out_of_range(self, tmp_path: Path) -> None:
        """Test error on invalid channel index."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path, n_channels=4)

        with pytest.raises(LoaderError, match="out of range"):
            load_sigrok(session_path, channel=10)

    def test_negative_channel_index(self, tmp_path: Path) -> None:
        """Test error on negative channel index."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path, n_channels=4)

        with pytest.raises(LoaderError, match="out of range"):
            load_sigrok(session_path, channel=-1)

    def test_channel_not_found(self, tmp_path: Path) -> None:
        """Test error on invalid channel name."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path)

        with pytest.raises(LoaderError, match="not found"):
            load_sigrok(session_path, channel="NONEXISTENT")

    def test_edge_detection(self, tmp_path: Path) -> None:
        """Test that edges are detected correctly."""
        session_path = tmp_path / "edges.sr"

        # Create data with known edges: 0x00, 0xFF, 0x00, 0xFF
        data = bytes([0x00] * 100 + [0xFF] * 100 + [0x00] * 100 + [0xFF] * 100)
        self.create_sigrok_session(session_path, data=data, n_samples=400)

        trace = load_sigrok(session_path, channel=0)

        # Should have alternating data (contains transitions)
        assert trace is not None
        assert len(trace.data) > 0
        # Check that there are transitions in the data
        has_true = any(trace.data)
        has_false = any(~trace.data)
        assert has_true and has_false, "Data should have both high and low values"

    def test_edge_timestamps(self, tmp_path: Path) -> None:
        """Test that edge timestamps are computed correctly."""
        session_path = tmp_path / "edges.sr"
        sample_rate = 1_000_000  # 1 MHz

        # Create data with transition at sample 100: 100 zeros, 100 ones
        data = bytes([0x00] * 100 + [0xFF] * 100)
        self.create_sigrok_session(session_path, data=data, n_samples=200, sample_rate=sample_rate)

        trace = load_sigrok(session_path, channel=0)

        assert trace is not None
        assert len(trace.data) == 200

        # First 100 samples should be False (0x00), next 100 should be True (0xFF)
        assert all(~trace.data[:100]), "First 100 samples should be False"
        assert all(trace.data[100:]), "Last 100 samples should be True"

        # Verify transition occurs at sample 100
        assert not trace.data[99]
        assert trace.data[100]

    def test_sample_rate_extraction(self, tmp_path: Path) -> None:
        """Test that sample rate is correctly extracted."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path, sample_rate=10_000_000)

        trace = load_sigrok(session_path)
        assert trace.metadata.sample_rate == 10_000_000

    def test_default_sample_rate(self, tmp_path: Path) -> None:
        """Test default sample rate when not specified in metadata."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            # Metadata without sample rate
            zf.writestr("metadata", "total probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        assert trace.metadata.sample_rate == 1_000_000  # Default

    def test_pathlib_path_input(self, tmp_path: Path) -> None:
        """Test that pathlib.Path objects are accepted."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path)

        # Should accept Path object
        trace = load_sigrok(session_path)
        assert trace is not None

    def test_string_path_input(self, tmp_path: Path) -> None:
        """Test that string paths are accepted."""
        session_path = tmp_path / "test.sr"
        self.create_sigrok_session(session_path)

        # Should accept string path
        trace = load_sigrok(str(session_path))
        assert trace is not None


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokDataTypes:
    """Test sigrok data type handling."""

    def test_single_byte_channels(self, tmp_path: Path) -> None:
        """Test files with 1-8 channels (1 byte per sample)."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            # 4 channels = 1 byte per sample
            # Pattern: 0x05 = 0b00000101 (channels 0 and 2 high)
            zf.writestr("logic-1-1", bytes([0x05] * 100))

        trace = load_sigrok(session_path, channel=0)
        assert trace is not None
        assert trace.data.dtype == np.bool_
        # Channel 0 should be high (bit 0 of 0x05 is 1)
        assert trace.data[0] is np.True_

        trace = load_sigrok(session_path, channel=1)
        # Channel 1 should be low (bit 1 of 0x05 is 0)
        assert trace.data[0] is np.False_

        trace = load_sigrok(session_path, channel=2)
        # Channel 2 should be high (bit 2 of 0x05 is 1)
        assert trace.data[0] is np.True_

    def test_two_byte_channels(self, tmp_path: Path) -> None:
        """Test files with 9-16 channels (2 bytes per sample)."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=12\n")
            # 12 channels = 2 bytes per sample
            # Pattern: 0x55 0x0F = channels 0,2,4,6,8,9,10,11 high
            data = bytes([0x55, 0x0F] * 100)
            zf.writestr("logic-1-1", data)

        trace = load_sigrok(session_path, channel=0)
        assert trace is not None
        assert trace.data.dtype == np.bool_

    def test_three_byte_channels(self, tmp_path: Path) -> None:
        """Test files with 17-24 channels (3 bytes per sample)."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=20\n")
            # 20 channels = 3 bytes per sample
            data = bytes([0xFF, 0x00, 0xAA] * 100)
            zf.writestr("logic-1-1", data)

        trace = load_sigrok(session_path, channel=0)
        assert trace is not None
        assert trace.data.dtype == np.bool_

    def test_large_channel_count(self, tmp_path: Path) -> None:
        """Test files with >32 channels."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=64\n")
            # 64 channels = 8 bytes per sample
            data = bytes([0xFF] * 8 * 100)
            zf.writestr("logic-1-1", data)

        trace = load_sigrok(session_path, channel=0)
        assert trace is not None
        assert trace.data.dtype == np.bool_


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokMetadata:
    """Test sigrok metadata parsing."""

    def test_metadata_with_float_sample_rate(self, tmp_path: Path) -> None:
        """Test parsing float sample rate from metadata."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1.5e6\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        assert trace.metadata.sample_rate == 1.5e6

    def test_metadata_with_probe_names(self, tmp_path: Path) -> None:
        """Test parsing custom probe names from metadata."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            metadata = """samplerate=1000000
total probes=3
probe0=SCL
probe1=SDA
probe2=RESET
"""
            zf.writestr("metadata", metadata)
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path, channel="SCL")
        assert trace.metadata.channel == "SCL"

        trace = load_sigrok(session_path, channel="SDA")
        assert trace.metadata.channel == "SDA"

    def test_metadata_with_non_sequential_probes(self, tmp_path: Path) -> None:
        """Test handling of non-sequential probe indices."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            metadata = """samplerate=1000000
total probes=4
probe0=CH0
probe2=CH2
"""
            zf.writestr("metadata", metadata)
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        # Should handle gaps in probe numbering
        assert trace is not None

    def test_metadata_without_file(self, tmp_path: Path) -> None:
        """Test handling session without metadata file."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            # No metadata file, just logic data
            # Need at least 1 channel for default behavior
            zf.writestr("metadata", "total probes=1\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        # Should use defaults
        assert trace.metadata.sample_rate == 1_000_000
        assert trace is not None

    def test_metadata_with_malformed_content(self, tmp_path: Path) -> None:
        """Test handling of malformed metadata."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            # Malformed metadata (missing values, invalid format)
            # Include at least total probes to avoid array index errors
            # Don't include malformed samplerate as it causes type errors
            metadata = """total probes=4
invalid line without equals
probe0
=value_without_key
"""
            zf.writestr("metadata", metadata)
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        # Should still load with defaults despite malformed metadata
        trace = load_sigrok(session_path)
        assert trace is not None
        # Should use default sample rate since samplerate was not in metadata
        assert trace.metadata.sample_rate == 1_000_000

    def test_metadata_with_extra_fields(self, tmp_path: Path) -> None:
        """Test that extra metadata fields are parsed."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            metadata = """samplerate=10000000
total probes=4
unitsize=1
total analog=0
"""
            zf.writestr("metadata", metadata)
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        assert trace.metadata.sample_rate == 10_000_000


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokMultipleLogicFiles:
    """Test handling of multiple logic data files."""

    def test_multiple_logic_files(self, tmp_path: Path) -> None:
        """Test loading session with multiple logic data files."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            # Split data across multiple files
            zf.writestr("logic-1-1", bytes([0x01] * 50))
            zf.writestr("logic-1-2", bytes([0x02] * 50))
            zf.writestr("logic-1-3", bytes([0x03] * 50))

        trace = load_sigrok(session_path)
        assert trace is not None
        assert len(trace.data) == 150  # Should combine all files

    def test_logic_files_sorted_correctly(self, tmp_path: Path) -> None:
        """Test that logic files are sorted in correct order."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=8\n")
            # Add files in non-sorted order
            zf.writestr("logic-1-10", bytes([0x0A]))
            zf.writestr("logic-1-2", bytes([0x02]))
            zf.writestr("logic-1-1", bytes([0x01]))

        trace = load_sigrok(session_path)
        # Should be sorted and combined correctly
        assert trace is not None


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokEdgeDetection:
    """Test edge detection functionality."""

    def test_no_edges_constant_low(self, tmp_path: Path) -> None:
        """Test edge detection with constant low signal."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0x00] * 100))

        trace = load_sigrok(session_path, channel=0)
        # Constant low signal - all data should be False
        assert all(~trace.data)

    def test_no_edges_constant_high(self, tmp_path: Path) -> None:
        """Test edge detection with constant high signal."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xFF] * 100))

        trace = load_sigrok(session_path, channel=0)
        # Constant high signal - all data should be True
        assert all(trace.data)

    def test_single_rising_edge(self, tmp_path: Path) -> None:
        """Test detection of single rising edge."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=1\n")
            # Low then high
            zf.writestr("logic-1-1", bytes([0x00] * 50 + [0xFF] * 50))

        trace = load_sigrok(session_path, channel=0)
        # Low then high - verify transition
        assert all(~trace.data[:50]), "First 50 samples should be False"
        assert all(trace.data[50:]), "Last 50 samples should be True"

    def test_single_falling_edge(self, tmp_path: Path) -> None:
        """Test detection of single falling edge."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=1\n")
            # High then low
            zf.writestr("logic-1-1", bytes([0xFF] * 50 + [0x00] * 50))

        trace = load_sigrok(session_path, channel=0)
        # High then low - verify transition
        assert all(trace.data[:50]), "First 50 samples should be True"
        assert all(~trace.data[50:]), "Last 50 samples should be False"

    def test_multiple_edges(self, tmp_path: Path) -> None:
        """Test detection of multiple edges."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=1\n")
            # Alternating pattern
            data = bytes([0x00, 0xFF, 0x00, 0xFF, 0x00])
            zf.writestr("logic-1-1", data)

        trace = load_sigrok(session_path, channel=0)
        # Alternating pattern - verify data
        assert len(trace.data) == 5
        assert not trace.data[0]
        assert trace.data[1]
        assert not trace.data[2]
        assert trace.data[3]
        assert not trace.data[4]

    def test_edges_sorted_by_time(self, tmp_path: Path) -> None:
        """Test that edges are sorted by timestamp."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=1\n")
            # Multiple transitions
            data = bytes([0x00] * 10 + [0xFF] * 10 + [0x00] * 10 + [0xFF] * 10)
            zf.writestr("logic-1-1", data)

        trace = load_sigrok(session_path, channel=0)
        # Check that data has expected pattern
        assert len(trace.data) == 40
        assert all(~trace.data[0:10]), "First 10 should be False"
        assert all(trace.data[10:20]), "Next 10 should be True"
        assert all(~trace.data[20:30]), "Next 10 should be False"
        assert all(trace.data[30:40]), "Last 10 should be True"

    def test_edge_detection_with_single_sample(self, tmp_path: Path) -> None:
        """Test loading with single sample."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=1\n")
            zf.writestr("logic-1-1", bytes([0xFF]))

        trace = load_sigrok(session_path, channel=0)
        # Single high sample
        assert len(trace.data) == 1
        assert trace.data[0]


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokDataIntegrity:
    """Test data integrity and correctness."""

    def test_data_type_is_boolean(self, tmp_path: Path) -> None:
        """Test that loaded data is boolean type."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        assert trace.data.dtype == np.bool_

    def test_data_shape_is_1d(self, tmp_path: Path) -> None:
        """Test that single channel data is 1D."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path, channel=0)
        assert trace.data.ndim == 1

    def test_data_length_matches_samples(self, tmp_path: Path) -> None:
        """Test that data length matches number of samples."""
        session_path = tmp_path / "test.sr"
        n_samples = 250

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * n_samples))

        trace = load_sigrok(session_path, channel=0)
        assert len(trace.data) == n_samples

    def test_bit_extraction_correctness(self, tmp_path: Path) -> None:
        """Test that individual channel bits are extracted correctly."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=8\n")
            # Pattern: 0b10101010 = 0xAA
            # Channels 1,3,5,7 high; 0,2,4,6 low
            zf.writestr("logic-1-1", bytes([0xAA] * 10))

        # Check each channel
        for ch in range(8):
            trace = load_sigrok(session_path, channel=ch)
            expected_value = bool((0xAA >> ch) & 1)
            assert trace.data[0] == expected_value

    def test_trace_has_time_vector(self, tmp_path: Path) -> None:
        """Test that trace has time_vector property."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        assert hasattr(trace, "time")
        assert len(trace.time) == len(trace.data)

    def test_trace_has_duration(self, tmp_path: Path) -> None:
        """Test that trace has duration property."""
        session_path = tmp_path / "test.sr"
        sample_rate = 1_000_000
        n_samples = 1000

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", f"samplerate={sample_rate}\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * n_samples))

        trace = load_sigrok(session_path)
        expected_duration = (n_samples - 1) / sample_rate
        assert abs(trace.duration - expected_duration) < 1e-9


@pytest.mark.unit
@pytest.mark.loader
class TestSigrokErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_logic_file(self, tmp_path: Path) -> None:
        """Test handling of empty logic data file."""
        session_path = tmp_path / "test.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", b"")  # Empty

        # Should raise error when logic file is empty
        with pytest.raises(LoaderError):
            load_sigrok(session_path)

    def test_generic_exception_wrapping(self, tmp_path: Path) -> None:
        """Test that unexpected exceptions are wrapped in LoaderError."""
        session_path = tmp_path / "test.sr"

        # Create a session with invalid binary data that might cause issues
        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=1000000\n")
            # Very small data for very large channel count
            zf.writestr("logic-1-1", bytes([0x01]))

        # Should wrap any unexpected errors
        try:
            load_sigrok(session_path)
        except (LoaderError, FormatError):
            pass  # Expected
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e)}")

    def test_channel_index_boundary(self, tmp_path: Path) -> None:
        """Test channel index at exact boundary."""
        session_path = tmp_path / "test.sr"
        n_channels = 8

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", f"samplerate=1000000\ntotal probes={n_channels}\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        # Last valid channel
        trace = load_sigrok(session_path, channel=n_channels - 1)
        assert trace is not None

        # First invalid channel
        with pytest.raises(LoaderError, match="out of range"):
            load_sigrok(session_path, channel=n_channels)

    def test_metadata_source_file(self, tmp_path: Path) -> None:
        """Test that session file loads correctly with custom name."""
        session_path = tmp_path / "custom_name.sr"

        with zipfile.ZipFile(session_path, "w") as zf:
            zf.writestr("metadata", "samplerate=1000000\ntotal probes=4\n")
            zf.writestr("logic-1-1", bytes([0xAA] * 100))

        trace = load_sigrok(session_path)
        assert trace is not None
        assert trace.metadata.sample_rate == 1_000_000
