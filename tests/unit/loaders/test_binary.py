"""Tests for binary file loader.

Tests the binary loader for raw signal data (LOAD-007).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.loaders.binary import load_binary

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestBinaryLoader:
    """Test load_binary() function."""

    def test_load_basic_float64(self, tmp_path: Path) -> None:
        """Test loading basic float64 binary file."""
        # Create test data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        # Load and verify
        trace = load_binary(file_path, sample_rate=1000.0)
        np.testing.assert_array_almost_equal(trace.data, data)
        assert trace.metadata.sample_rate == 1000.0
        assert trace.metadata.source_file == str(file_path)
        assert trace.metadata.channel == "Channel 0"

    def test_load_int16_dtype(self, tmp_path: Path) -> None:
        """Test loading int16 data."""
        data = np.array([100, 200, 300, -100, -200], dtype=np.int16)
        file_path = tmp_path / "test_int16.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, dtype="int16", sample_rate=1e6)
        expected = data.astype(np.float64)
        np.testing.assert_array_almost_equal(trace.data, expected)
        assert trace.data.dtype == np.float64  # Should be converted

    def test_load_float32_dtype(self, tmp_path: Path) -> None:
        """Test loading float32 data."""
        data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        file_path = tmp_path / "test_float32.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, dtype="float32", sample_rate=500.0)
        expected = data.astype(np.float64)
        np.testing.assert_array_almost_equal(trace.data, expected, decimal=6)

    def test_load_uint8_dtype(self, tmp_path: Path) -> None:
        """Test loading uint8 data."""
        data = np.array([0, 127, 255, 64, 192], dtype=np.uint8)
        file_path = tmp_path / "test_uint8.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, dtype="uint8", sample_rate=2000.0)
        expected = data.astype(np.float64)
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_load_with_offset(self, tmp_path: Path) -> None:
        """Test loading with offset."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        file_path = tmp_path / "test_offset.bin"
        data.tofile(file_path)

        # Skip first 2 samples
        trace = load_binary(file_path, sample_rate=1000.0, offset=2)
        expected = data[2:]
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_load_with_count(self, tmp_path: Path) -> None:
        """Test loading with count limit."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        file_path = tmp_path / "test_count.bin"
        data.tofile(file_path)

        # Read only 3 samples
        trace = load_binary(file_path, sample_rate=1000.0, count=3)
        expected = data[:3]
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_load_with_offset_and_count(self, tmp_path: Path) -> None:
        """Test loading with both offset and count."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        file_path = tmp_path / "test_offset_count.bin"
        data.tofile(file_path)

        # Skip 1, read 3
        trace = load_binary(file_path, sample_rate=1000.0, offset=1, count=3)
        expected = data[1:4]
        np.testing.assert_array_almost_equal(trace.data, expected)


class TestMultiChannel:
    """Test multi-channel data loading."""

    def test_load_two_channels(self, tmp_path: Path) -> None:
        """Test loading from 2-channel interleaved data."""
        # Create interleaved data: ch0, ch1, ch0, ch1, ...
        ch0 = np.array([1.0, 3.0, 5.0], dtype=np.float64)
        ch1 = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        interleaved = np.empty(6, dtype=np.float64)
        interleaved[0::2] = ch0
        interleaved[1::2] = ch1

        file_path = tmp_path / "test_2ch.bin"
        interleaved.tofile(file_path)

        # Load channel 0
        trace0 = load_binary(file_path, sample_rate=1000.0, channels=2, channel=0)
        np.testing.assert_array_almost_equal(trace0.data, ch0)
        assert trace0.metadata.channel == "Channel 0"

        # Load channel 1
        trace1 = load_binary(file_path, sample_rate=1000.0, channels=2, channel=1)
        np.testing.assert_array_almost_equal(trace1.data, ch1)
        assert trace1.metadata.channel == "Channel 1"

    def test_load_four_channels(self, tmp_path: Path) -> None:
        """Test loading from 4-channel interleaved data."""
        ch0 = np.array([1.0, 5.0, 9.0], dtype=np.float64)
        ch1 = np.array([2.0, 6.0, 10.0], dtype=np.float64)
        ch2 = np.array([3.0, 7.0, 11.0], dtype=np.float64)
        ch3 = np.array([4.0, 8.0, 12.0], dtype=np.float64)

        interleaved = np.empty(12, dtype=np.float64)
        interleaved[0::4] = ch0
        interleaved[1::4] = ch1
        interleaved[2::4] = ch2
        interleaved[3::4] = ch3

        file_path = tmp_path / "test_4ch.bin"
        interleaved.tofile(file_path)

        # Load each channel
        trace0 = load_binary(file_path, sample_rate=1000.0, channels=4, channel=0)
        np.testing.assert_array_almost_equal(trace0.data, ch0)

        trace1 = load_binary(file_path, sample_rate=1000.0, channels=4, channel=1)
        np.testing.assert_array_almost_equal(trace1.data, ch1)

        trace2 = load_binary(file_path, sample_rate=1000.0, channels=4, channel=2)
        np.testing.assert_array_almost_equal(trace2.data, ch2)

        trace3 = load_binary(file_path, sample_rate=1000.0, channels=4, channel=3)
        np.testing.assert_array_almost_equal(trace3.data, ch3)

    def test_multichannel_with_offset(self, tmp_path: Path) -> None:
        """Test multi-channel loading with offset."""
        # 2 channels, 4 samples per channel = 8 total samples
        data = np.arange(8, dtype=np.float64)  # [0, 1, 2, 3, 4, 5, 6, 7]
        file_path = tmp_path / "test_2ch_offset.bin"
        data.tofile(file_path)

        # Skip first sample, load channel 0
        # After offset=1: [1, 2, 3, 4, 5, 6, 7]
        # Channel 0 (indices 1, 3, 5): [1, 3, 5]
        trace = load_binary(file_path, sample_rate=1000.0, channels=2, channel=0, offset=1)
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_multichannel_with_partial_samples(self, tmp_path: Path) -> None:
        """Test multi-channel with non-exact sample count."""
        # 2 channels, but 7 samples total (incomplete last pair)
        data = np.arange(7, dtype=np.float64)
        file_path = tmp_path / "test_2ch_partial.bin"
        data.tofile(file_path)

        # Should handle gracefully by truncating to complete pairs
        # 7 samples -> 3 complete pairs (6 samples) -> channel 0 gets [0, 2, 4]
        trace = load_binary(file_path, sample_rate=1000.0, channels=2, channel=0)
        expected = np.array([0.0, 2.0, 4.0])  # Even indices up to 4
        np.testing.assert_array_almost_equal(trace.data, expected)


class TestMetadata:
    """Test metadata handling."""

    def test_metadata_sample_rate(self, tmp_path: Path) -> None:
        """Test sample rate in metadata."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, sample_rate=12345.6)
        assert trace.metadata.sample_rate == 12345.6

    def test_metadata_source_file(self, tmp_path: Path) -> None:
        """Test source file in metadata."""
        data = np.array([1.0, 2.0], dtype=np.float64)
        file_path = tmp_path / "my_signal.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, sample_rate=1000.0)
        assert trace.metadata.source_file == str(file_path)

    def test_metadata_channel_name_single(self, tmp_path: Path) -> None:
        """Test channel name for single-channel data."""
        data = np.array([1.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, sample_rate=1000.0)
        assert trace.metadata.channel == "Channel 0"

    def test_metadata_channel_name_multi(self, tmp_path: Path) -> None:
        """Test channel names for multi-channel data."""
        data = np.arange(6, dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace0 = load_binary(file_path, sample_rate=1000.0, channels=3, channel=0)
        assert trace0.metadata.channel == "Channel 0"

        trace1 = load_binary(file_path, sample_rate=1000.0, channels=3, channel=1)
        assert trace1.metadata.channel == "Channel 1"

        trace2 = load_binary(file_path, sample_rate=1000.0, channels=3, channel=2)
        assert trace2.metadata.channel == "Channel 2"


class TestPathHandling:
    """Test path handling."""

    def test_string_path(self, tmp_path: Path) -> None:
        """Test loading with string path."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace = load_binary(str(file_path), sample_rate=1000.0)
        np.testing.assert_array_almost_equal(trace.data, data)

    def test_pathlib_path(self, tmp_path: Path) -> None:
        """Test loading with pathlib.Path."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, sample_rate=1000.0)
        np.testing.assert_array_almost_equal(trace.data, data)


class TestLoadersBinaryEdgeCases:
    """Test edge cases."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty file.

        Empty data arrays are rejected in v0.9.0, so this should raise ValueError.
        """
        file_path = tmp_path / "empty.bin"
        file_path.touch()

        with pytest.raises(ValueError, match="data array cannot be empty"):
            load_binary(file_path, sample_rate=1000.0)

    def test_single_sample(self, tmp_path: Path) -> None:
        """Test loading single sample."""
        data = np.array([42.0], dtype=np.float64)
        file_path = tmp_path / "single.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, sample_rate=1000.0)
        np.testing.assert_array_almost_equal(trace.data, data)

    def test_large_offset(self, tmp_path: Path) -> None:
        """Test offset larger than file raises error."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        # Offset beyond file size should raise ValueError
        with pytest.raises(ValueError, match="negative dimensions"):
            load_binary(file_path, sample_rate=1000.0, offset=10)

    def test_count_larger_than_available(self, tmp_path: Path) -> None:
        """Test count larger than available samples."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        # Request more samples than available
        trace = load_binary(file_path, sample_rate=1000.0, count=100)
        np.testing.assert_array_almost_equal(trace.data, data)

    def test_default_sample_rate(self, tmp_path: Path) -> None:
        """Test default sample rate."""
        data = np.array([1.0, 2.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace = load_binary(file_path)
        assert trace.metadata.sample_rate == 1.0

    def test_zero_count_loads_all(self, tmp_path: Path) -> None:
        """Test count=-1 loads all samples."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        file_path = tmp_path / "test.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, sample_rate=1000.0, count=-1)
        np.testing.assert_array_almost_equal(trace.data, data)


class TestLoadersBinaryIntegration:
    """Integration tests."""

    def test_realistic_int16_adc_data(self, tmp_path: Path) -> None:
        """Test loading realistic ADC data (int16)."""
        # Simulate ADC data: 1000 samples, 12-bit range (-2048 to 2047)
        samples = 1000
        adc_data = np.random.randint(-2048, 2048, size=samples, dtype=np.int16)
        file_path = tmp_path / "adc_capture.bin"
        adc_data.tofile(file_path)

        trace = load_binary(file_path, dtype="int16", sample_rate=100e3)
        assert len(trace.data) == samples
        assert trace.data.dtype == np.float64
        assert trace.metadata.sample_rate == 100e3

    def test_realistic_multichannel_oscilloscope(self, tmp_path: Path) -> None:
        """Test realistic 4-channel oscilloscope data."""
        samples_per_channel = 500
        channels = 4
        total_samples = samples_per_channel * channels

        # Simulate scope data
        scope_data = np.random.randn(total_samples).astype(np.float32)
        file_path = tmp_path / "scope_4ch.bin"
        scope_data.tofile(file_path)

        # Load each channel
        traces = []
        for ch in range(channels):
            trace = load_binary(
                file_path,
                dtype="float32",
                sample_rate=1e9,
                channels=channels,
                channel=ch,
            )
            assert len(trace.data) == samples_per_channel
            traces.append(trace)

        # Verify channels are different
        assert not np.array_equal(traces[0].data, traces[1].data)

    def test_partial_read_workflow(self, tmp_path: Path) -> None:
        """Test reading file in chunks."""
        # Create large file
        data = np.arange(1000, dtype=np.float64)
        file_path = tmp_path / "large.bin"
        data.tofile(file_path)

        # Read in chunks of 100
        chunk_size = 100
        for i in range(10):
            trace = load_binary(
                file_path,
                sample_rate=1000.0,
                offset=i * chunk_size,
                count=chunk_size,
            )
            expected = data[i * chunk_size : (i + 1) * chunk_size]
            np.testing.assert_array_almost_equal(trace.data, expected)


class TestMemoryMappedIO:
    """Test memory-mapped I/O optimizations for large file loading."""

    def test_mmap_basic_load(self, tmp_path: Path) -> None:
        """Test basic memory-mapped file loading.

        Validates:
        - Memory-mapped loading produces correct data
        - Results match non-mmap loading
        - No data corruption
        """
        data = np.arange(1000, dtype=np.float32)
        file_path = tmp_path / "test_mmap.bin"
        data.tofile(file_path)

        # Load with and without mmap
        trace_standard = load_binary(file_path, dtype="float32", sample_rate=1e6, mmap_mode=False)
        trace_mmap = load_binary(file_path, dtype="float32", sample_rate=1e6, mmap_mode=True)

        # Results should be identical
        np.testing.assert_array_almost_equal(trace_standard.data, trace_mmap.data)
        assert trace_standard.metadata.sample_rate == trace_mmap.metadata.sample_rate

    def test_mmap_with_offset(self, tmp_path: Path) -> None:
        """Test memory-mapped loading with offset.

        Validates:
        - Offset calculations correct for mmap
        - Byte offset matches sample offset
        """
        data = np.arange(1000, dtype=np.float64)
        file_path = tmp_path / "test_mmap_offset.bin"
        data.tofile(file_path)

        offset = 100
        trace = load_binary(
            file_path, dtype="float64", sample_rate=1e6, offset=offset, mmap_mode=True
        )

        expected = data[offset:]
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_mmap_with_count(self, tmp_path: Path) -> None:
        """Test memory-mapped loading with count limit.

        Validates:
        - Count parameter works with mmap
        - Correct number of samples loaded
        """
        data = np.arange(1000, dtype=np.float32)
        file_path = tmp_path / "test_mmap_count.bin"
        data.tofile(file_path)

        count = 200
        trace = load_binary(
            file_path, dtype="float32", sample_rate=1e6, count=count, mmap_mode=True
        )

        expected = data[:count]
        assert len(trace.data) == count
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_mmap_with_offset_and_count(self, tmp_path: Path) -> None:
        """Test memory-mapped loading with both offset and count.

        Validates:
        - Combined offset and count work correctly
        - Byte calculations are accurate
        """
        data = np.arange(1000, dtype=np.float64)
        file_path = tmp_path / "test_mmap_offset_count.bin"
        data.tofile(file_path)

        offset = 100
        count = 200
        trace = load_binary(
            file_path,
            dtype="float64",
            sample_rate=1e6,
            offset=offset,
            count=count,
            mmap_mode=True,
        )

        expected = data[offset : offset + count]
        assert len(trace.data) == count
        np.testing.assert_array_almost_equal(trace.data, expected)

    def test_mmap_large_file(self, tmp_path: Path) -> None:
        """Test memory-mapped loading of large file.

        Validates:
        - Can handle multi-MB files efficiently
        - No memory errors
        - Results are correct
        """
        # Create 5 MB file (~1.25M float32 samples)
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1_250_000).astype(np.float32)
        file_path = tmp_path / "large_mmap.bin"
        data.tofile(file_path)

        # Load with mmap
        trace = load_binary(file_path, dtype="float32", sample_rate=1e9, mmap_mode=True)

        # Verify correct length and no NaN/Inf
        assert len(trace.data) == len(data)
        assert not np.any(np.isnan(trace.data))
        assert not np.any(np.isinf(trace.data))

        # Spot check some values
        np.testing.assert_array_almost_equal(trace.data[:100], data[:100])
        np.testing.assert_array_almost_equal(trace.data[-100:], data[-100:])

    def test_mmap_multiple_dtypes(self, tmp_path: Path) -> None:
        """Test memory-mapped loading with different data types.

        Validates:
        - Correct byte offset calculations for different dtypes
        - Type conversion works correctly
        - No precision loss for supported types
        """
        for dtype in [np.float32, np.float64, np.int16, np.uint8]:
            rng = np.random.default_rng(42)
            if dtype in [np.int16, np.uint8]:
                data = rng.integers(0, 100, size=1000, dtype=dtype)
            else:
                data = rng.standard_normal(1000).astype(dtype)

            file_path = tmp_path / f"test_mmap_{dtype.__name__}.bin"
            data.tofile(file_path)

            trace = load_binary(
                file_path,
                dtype=dtype.__name__,
                sample_rate=1e6,
                mmap_mode=True,
            )

            expected = data.astype(np.float64)
            np.testing.assert_array_almost_equal(trace.data, expected, decimal=6)

    def test_mmap_multichannel(self, tmp_path: Path) -> None:
        """Test memory-mapped loading with multi-channel data.

        Validates:
        - Multi-channel deinterleaving works with mmap
        - Each channel gets correct data
        """
        # Create 2-channel interleaved data
        ch0 = np.arange(0, 1000, dtype=np.float32)
        ch1 = np.arange(1000, 2000, dtype=np.float32)
        interleaved = np.empty(2000, dtype=np.float32)
        interleaved[0::2] = ch0
        interleaved[1::2] = ch1

        file_path = tmp_path / "test_mmap_2ch.bin"
        interleaved.tofile(file_path)

        # Load both channels with mmap
        trace0 = load_binary(
            file_path,
            dtype="float32",
            sample_rate=1e6,
            channels=2,
            channel=0,
            mmap_mode=True,
        )
        trace1 = load_binary(
            file_path,
            dtype="float32",
            sample_rate=1e6,
            channels=2,
            channel=1,
            mmap_mode=True,
        )

        np.testing.assert_array_almost_equal(trace0.data, ch0)
        np.testing.assert_array_almost_equal(trace1.data, ch1)

    def test_mmap_partial_last_chunk(self, tmp_path: Path) -> None:
        """Test memory-mapped loading with non-aligned file size.

        Validates:
        - Handles files not evenly divisible by sample size
        - No truncation errors
        """
        # Create file with 1005 samples (not a round number)
        data = np.arange(1005, dtype=np.float32)
        file_path = tmp_path / "test_mmap_partial.bin"
        data.tofile(file_path)

        trace = load_binary(file_path, dtype="float32", sample_rate=1e6, mmap_mode=True)

        assert len(trace.data) == 1005
        np.testing.assert_array_almost_equal(trace.data, data)

    def test_mmap_empty_file(self, tmp_path: Path) -> None:
        """Test memory-mapped loading of empty file.

        Empty data arrays are rejected in v0.9.0, so this should raise ValueError.
        """
        file_path = tmp_path / "empty_mmap.bin"
        file_path.touch()

        with pytest.raises(ValueError, match="data array cannot be empty"):
            load_binary(file_path, dtype="float32", sample_rate=1e6, mmap_mode=True)

    def test_mmap_count_exceeds_file_size(self, tmp_path: Path) -> None:
        """Test memory-mapped loading when count exceeds file size.

        Validates:
        - Gracefully handles count > available samples
        - Returns all available data
        """
        data = np.arange(100, dtype=np.float32)
        file_path = tmp_path / "test_mmap_exceed.bin"
        data.tofile(file_path)

        trace = load_binary(
            file_path,
            dtype="float32",
            sample_rate=1e6,
            count=1000,  # Request more than available
            mmap_mode=True,
        )

        # Should return all available data
        assert len(trace.data) == 100
        np.testing.assert_array_almost_equal(trace.data, data)
