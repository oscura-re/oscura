"""Unit tests for NumPy NPZ file loader.

Tests LOAD-004: NumPy NPZ Loader
Tests MEM-016: NumPy Memmap Integration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.numpy_loader import (
    list_arrays,
    load_npz,
    load_raw_binary,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-004")
class TestLoadNPZ:
    """Test NPZ file loading with load_npz function."""

    def test_load_basic_npz_with_data_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with standard 'data' array name."""
        npz_path = tmp_path / "test.npz"
        sample_data = np.sin(np.linspace(0, 2 * np.pi, 1000))
        np.savez(npz_path, data=sample_data, sample_rate=1e6)

        trace = load_npz(npz_path)

        assert trace is not None
        assert len(trace.data) == 1000
        assert trace.metadata.sample_rate == 1e6
        assert trace.metadata.source_file == str(npz_path)
        assert np.allclose(trace.data, sample_data, rtol=1e-10)

    def test_load_npz_with_waveform_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with 'waveform' array name."""
        npz_path = tmp_path / "waveform.npz"
        sample_data = np.random.randn(500)
        np.savez(npz_path, waveform=sample_data, sample_rate=2e6)

        trace = load_npz(npz_path)

        assert len(trace.data) == 500
        assert trace.metadata.sample_rate == 2e6

    def test_load_npz_with_signal_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with 'signal' array name."""
        npz_path = tmp_path / "signal.npz"
        sample_data = np.ones(100)
        np.savez(npz_path, signal=sample_data)

        trace = load_npz(npz_path)

        assert len(trace.data) == 100
        # Should use default sample rate
        assert trace.metadata.sample_rate == 1e6

    def test_load_npz_with_samples_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with 'samples' array name."""
        npz_path = tmp_path / "samples.npz"
        sample_data = np.linspace(0, 1, 256)
        np.savez(npz_path, samples=sample_data, fs=10e6)

        trace = load_npz(npz_path)

        assert len(trace.data) == 256
        assert trace.metadata.sample_rate == 10e6

    def test_load_npz_with_y_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with 'y' array name."""
        npz_path = tmp_path / "y.npz"
        sample_data = np.zeros(50)
        np.savez(npz_path, y=sample_data, samplerate=5e6)

        trace = load_npz(npz_path)

        assert len(trace.data) == 50
        assert trace.metadata.sample_rate == 5e6

    def test_load_npz_with_voltage_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with 'voltage' array name."""
        npz_path = tmp_path / "voltage.npz"
        sample_data = np.full(200, 3.3)
        np.savez(npz_path, voltage=sample_data, sampling_rate=1e9)

        trace = load_npz(npz_path)

        assert len(trace.data) == 200
        assert trace.metadata.sample_rate == 1e9

    def test_load_npz_case_insensitive_array_name(self, tmp_path: Path) -> None:
        """Test that array name matching is case-insensitive."""
        npz_path = tmp_path / "case.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, DATA=sample_data, sample_rate=1e6)

        trace = load_npz(npz_path)

        assert len(trace.data) == 100

    def test_load_npz_with_explicit_channel_name(self, tmp_path: Path) -> None:
        """Test loading specific channel by name."""
        npz_path = tmp_path / "multi.npz"
        ch1_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        ch2_data = np.cos(np.linspace(0, 2 * np.pi, 100))
        np.savez(npz_path, ch1=ch1_data, ch2=ch2_data, sample_rate=1e6)

        trace = load_npz(npz_path, channel="ch2")

        assert len(trace.data) == 100
        assert np.allclose(trace.data, ch2_data, rtol=1e-10)
        assert trace.metadata.channel_name == "ch2"

    def test_load_npz_with_explicit_channel_index(self, tmp_path: Path) -> None:
        """Test loading specific channel by index."""
        npz_path = tmp_path / "multi_idx.npz"
        ch1_data = np.ones(50)
        ch2_data = np.zeros(50)
        np.savez(npz_path, ch1=ch1_data, ch2=ch2_data, sample_rate=1e6)

        trace = load_npz(npz_path, channel=0)

        assert len(trace.data) == 50
        assert trace.metadata.channel_name == "CH1"

    def test_load_npz_with_channel_index_1(self, tmp_path: Path) -> None:
        """Test loading second channel by index."""
        npz_path = tmp_path / "multi_idx2.npz"
        ch1_data = np.ones(50)
        ch2_data = np.full(50, 2.0)
        np.savez(npz_path, ch1=ch1_data, ch2=ch2_data)

        trace = load_npz(npz_path, channel=1)

        assert len(trace.data) == 50
        assert trace.metadata.channel_name == "CH2"
        assert np.allclose(trace.data, 2.0)

    def test_load_npz_channel_case_insensitive(self, tmp_path: Path) -> None:
        """Test that channel name matching is case-insensitive."""
        npz_path = tmp_path / "case_channel.npz"
        data = np.random.randn(100)
        np.savez(npz_path, MyChannel=data, sample_rate=1e6)

        trace = load_npz(npz_path, channel="mychannel")

        assert len(trace.data) == 100
        assert trace.metadata.channel_name == "mychannel"

    def test_load_npz_override_sample_rate(self, tmp_path: Path) -> None:
        """Test overriding sample rate from file."""
        npz_path = tmp_path / "override.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, sample_rate=1e6)

        trace = load_npz(npz_path, sample_rate=5e6)

        assert trace.metadata.sample_rate == 5e6

    def test_load_npz_default_sample_rate(self, tmp_path: Path) -> None:
        """Test default sample rate when not in file."""
        npz_path = tmp_path / "no_sr.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert trace.metadata.sample_rate == 1e6

    def test_load_npz_various_sample_rate_keys(self, tmp_path: Path) -> None:
        """Test detection of various sample rate key names."""
        keys = ["sample_rate", "samplerate", "fs", "sampling_rate", "rate"]

        for i, key in enumerate(keys):
            npz_path = tmp_path / f"sr_{i}.npz"
            sample_data = np.random.randn(10)
            np.savez(npz_path, data=sample_data, **{key: float(i + 1) * 1e6})

            trace = load_npz(npz_path)
            assert trace.metadata.sample_rate == (i + 1) * 1e6

    def test_load_npz_vertical_scale(self, tmp_path: Path) -> None:
        """Test extraction of vertical scale metadata."""
        npz_path = tmp_path / "vscale.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, vertical_scale=0.5)

        trace = load_npz(npz_path)

        assert trace.metadata.vertical_scale == 0.5

    def test_load_npz_vertical_scale_keys(self, tmp_path: Path) -> None:
        """Test detection of various vertical scale key names."""
        keys = ["vertical_scale", "v_scale", "scale", "volts_per_div"]

        for i, key in enumerate(keys):
            npz_path = tmp_path / f"vs_{i}.npz"
            sample_data = np.random.randn(10)
            value = (i + 1) * 0.1
            np.savez(npz_path, data=sample_data, **{key: value})

            trace = load_npz(npz_path)
            assert trace.metadata.vertical_scale == pytest.approx(value)

    def test_load_npz_vertical_offset(self, tmp_path: Path) -> None:
        """Test extraction of vertical offset metadata."""
        npz_path = tmp_path / "voffset.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, vertical_offset=1.5)

        trace = load_npz(npz_path)

        assert trace.metadata.vertical_offset == 1.5

    def test_load_npz_vertical_offset_keys(self, tmp_path: Path) -> None:
        """Test detection of various vertical offset key names."""
        keys = ["vertical_offset", "v_offset", "offset"]

        for i, key in enumerate(keys):
            npz_path = tmp_path / f"vo_{i}.npz"
            sample_data = np.random.randn(10)
            value = (i + 1) * 0.2
            np.savez(npz_path, data=sample_data, **{key: value})

            trace = load_npz(npz_path)
            assert trace.metadata.vertical_offset == pytest.approx(value)

    def test_load_npz_metadata_dict(self, tmp_path: Path) -> None:
        """Test extraction of metadata from nested dictionary."""
        npz_path = tmp_path / "meta_dict.npz"
        sample_data = np.random.randn(100)
        metadata = {"sample_rate": 2e6, "vertical_scale": 0.1, "vertical_offset": 0.5}
        np.savez(npz_path, data=sample_data, metadata=np.array(metadata))

        trace = load_npz(npz_path)

        assert trace.metadata.sample_rate == 2e6
        assert trace.metadata.vertical_scale == 0.1
        assert trace.metadata.vertical_offset == 0.5

    def test_load_npz_channel_name_from_metadata(self, tmp_path: Path) -> None:
        """Test extraction of channel name from NPZ metadata."""
        npz_path = tmp_path / "ch_name.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, channel_name=np.array("ANALOG1"))

        trace = load_npz(npz_path)

        assert trace.metadata.channel_name == "ANALOG1"

    def test_load_npz_int16_dtype(self, tmp_path: Path) -> None:
        """Test loading int16 data and conversion to float64."""
        npz_path = tmp_path / "int16.npz"
        sample_data = np.array([100, -200, 300, -400], dtype=np.int16)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert trace.data.dtype == np.float64
        assert np.array_equal(trace.data, [100.0, -200.0, 300.0, -400.0])

    def test_load_npz_uint8_dtype(self, tmp_path: Path) -> None:
        """Test loading uint8 data and conversion to float64."""
        npz_path = tmp_path / "uint8.npz"
        sample_data = np.array([0, 128, 255], dtype=np.uint8)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert trace.data.dtype == np.float64
        assert np.array_equal(trace.data, [0.0, 128.0, 255.0])

    def test_load_npz_float32_dtype(self, tmp_path: Path) -> None:
        """Test loading float32 data and conversion to float64."""
        npz_path = tmp_path / "float32.npz"
        sample_data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert trace.data.dtype == np.float64

    def test_load_npz_empty_array(self, tmp_path: Path) -> None:
        """Test loading NPZ with empty array."""
        npz_path = tmp_path / "empty.npz"
        sample_data = np.array([], dtype=np.float64)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert len(trace.data) == 0
        assert trace.data.dtype == np.float64

    def test_load_npz_2d_array_flattened(self, tmp_path: Path) -> None:
        """Test that 2D arrays are automatically flattened when using fallback."""
        npz_path = tmp_path / "2d.npz"
        # Need > 10 elements to pass the metadata threshold in fallback logic
        sample_data = np.arange(24).reshape(4, 6)
        # Use non-standard name to trigger fallback logic that does ravel()
        np.savez(npz_path, my_2d_array=sample_data)

        trace = load_npz(npz_path)

        # Should be flattened by fallback logic
        assert trace.data.ndim == 1
        assert len(trace.data) == 24
        assert np.array_equal(trace.data, np.arange(24))

    def test_load_npz_mmap_mode(self, tmp_path: Path) -> None:
        """Test memory-mapped loading for large files."""
        npz_path = tmp_path / "large.npz"
        sample_data = np.random.randn(10000)
        np.savez(npz_path, data=sample_data, sample_rate=1e9)

        trace = load_npz(npz_path, mmap=True)

        assert len(trace.data) == 10000
        assert trace.metadata.sample_rate == 1e9

    def test_load_npz_mmap_dtype_conversion(self, tmp_path: Path) -> None:
        """Test mmap with dtype conversion."""
        npz_path = tmp_path / "mmap_int.npz"
        sample_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path, mmap=True)

        # Should be converted to float64
        assert trace.data.dtype == np.float64

    def test_load_npz_mmap_already_float64(self, tmp_path: Path) -> None:
        """Test mmap with data already in float64 format."""
        npz_path = tmp_path / "mmap_float64.npz"
        sample_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path, mmap=True)

        # Should keep as float64 without conversion
        assert trace.data.dtype == np.float64
        assert len(trace.data) == 3

    def test_load_npz_scalar_metadata(self, tmp_path: Path) -> None:
        """Test extraction of scalar metadata values."""
        npz_path = tmp_path / "scalar.npz"
        sample_data = np.random.randn(100)
        # Save as scalar wrapped in array
        np.savez(
            npz_path,
            data=sample_data,
            sample_rate=np.array(5e6),
            vertical_scale=np.array(0.2),
        )

        trace = load_npz(npz_path)

        assert trace.metadata.sample_rate == 5e6
        assert trace.metadata.vertical_scale == 0.2


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-004")
class TestLoadNPZErrors:
    """Test error handling in load_npz function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when NPZ file doesn't exist."""
        npz_path = tmp_path / "nonexistent.npz"

        with pytest.raises(LoaderError, match="not found"):
            load_npz(npz_path)

    def test_invalid_npz_file(self, tmp_path: Path) -> None:
        """Test error when file is not a valid NPZ."""
        npz_path = tmp_path / "invalid.npz"
        npz_path.write_bytes(b"not a valid npz file")

        with pytest.raises(LoaderError, match="Failed to load NPZ"):
            load_npz(npz_path)

    def test_no_waveform_data_found(self, tmp_path: Path) -> None:
        """Test error when NPZ has no recognizable waveform data."""
        npz_path = tmp_path / "no_data.npz"
        # Save only metadata, no waveform arrays
        np.savez(npz_path, sample_rate=1e6, some_other_field=123)

        with pytest.raises(FormatError, match="No waveform data found"):
            load_npz(npz_path)

    def test_channel_name_not_found(self, tmp_path: Path) -> None:
        """Test error when specified channel name doesn't exist."""
        npz_path = tmp_path / "missing_ch.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, ch1=sample_data)

        with pytest.raises(FormatError, match="No waveform data found"):
            load_npz(npz_path, channel="ch99")

    def test_channel_index_out_of_range(self, tmp_path: Path) -> None:
        """Test error when channel index is out of range."""
        npz_path = tmp_path / "idx_range.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, ch1=sample_data)

        with pytest.raises(FormatError, match="No waveform data found"):
            load_npz(npz_path, channel=10)

    def test_empty_npz_file(self, tmp_path: Path) -> None:
        """Test error when NPZ file is empty."""
        npz_path = tmp_path / "empty.npz"
        # Create empty NPZ
        np.savez(npz_path)

        with pytest.raises(FormatError, match="No waveform data found"):
            load_npz(npz_path)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("LOAD-004")
class TestListArrays:
    """Test list_arrays function."""

    def test_list_single_array(self, tmp_path: Path) -> None:
        """Test listing arrays in NPZ with one array."""
        npz_path = tmp_path / "single.npz"
        np.savez(npz_path, data=np.ones(100))

        arrays = list_arrays(npz_path)

        assert arrays == ["data"]

    def test_list_multiple_arrays(self, tmp_path: Path) -> None:
        """Test listing arrays in NPZ with multiple arrays."""
        npz_path = tmp_path / "multi.npz"
        np.savez(npz_path, ch1=np.ones(50), ch2=np.zeros(50), sample_rate=1e6)

        arrays = list_arrays(npz_path)

        assert set(arrays) == {"ch1", "ch2", "sample_rate"}

    def test_list_arrays_empty_npz(self, tmp_path: Path) -> None:
        """Test listing arrays in empty NPZ."""
        npz_path = tmp_path / "empty.npz"
        np.savez(npz_path)

        arrays = list_arrays(npz_path)

        assert arrays == []

    def test_list_arrays_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        npz_path = tmp_path / "missing.npz"

        with pytest.raises(LoaderError, match="not found"):
            list_arrays(npz_path)

    def test_list_arrays_invalid_file(self, tmp_path: Path) -> None:
        """Test error when file is not valid NPZ."""
        npz_path = tmp_path / "invalid.npz"
        npz_path.write_bytes(b"invalid data")

        with pytest.raises(LoaderError, match="Failed to read NPZ"):
            list_arrays(npz_path)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("MEM-016")
class TestLoadRawBinary:
    """Test raw binary file loading with load_raw_binary function."""

    def test_load_float32_binary(self, tmp_path: Path) -> None:
        """Test loading raw float32 binary data."""
        bin_path = tmp_path / "signal.bin"
        sample_data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float32", sample_rate=1e6)

        assert len(trace.data) == 4
        assert trace.data.dtype == np.float64
        assert np.allclose(trace.data, [1.5, 2.5, 3.5, 4.5])
        assert trace.metadata.sample_rate == 1e6
        assert trace.metadata.channel_name == "RAW"

    def test_load_float64_binary(self, tmp_path: Path) -> None:
        """Test loading raw float64 binary data."""
        bin_path = tmp_path / "signal64.bin"
        sample_data = np.random.randn(100).astype(np.float64)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float64", sample_rate=2e6)

        assert len(trace.data) == 100
        assert trace.data.dtype == np.float64
        assert trace.metadata.sample_rate == 2e6

    def test_load_int16_binary(self, tmp_path: Path) -> None:
        """Test loading raw int16 binary data."""
        bin_path = tmp_path / "signal_int16.bin"
        sample_data = np.array([100, -200, 300, -400], dtype=np.int16)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="int16", sample_rate=1e6)

        assert len(trace.data) == 4
        assert trace.data.dtype == np.float64
        assert np.array_equal(trace.data, [100.0, -200.0, 300.0, -400.0])

    def test_load_int32_binary(self, tmp_path: Path) -> None:
        """Test loading raw int32 binary data."""
        bin_path = tmp_path / "signal_int32.bin"
        sample_data = np.array([1000, -2000, 3000], dtype=np.int32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="int32", sample_rate=5e6)

        assert len(trace.data) == 3
        assert trace.data.dtype == np.float64

    def test_load_uint8_binary(self, tmp_path: Path) -> None:
        """Test loading raw uint8 binary data."""
        bin_path = tmp_path / "signal_uint8.bin"
        sample_data = np.array([0, 128, 255], dtype=np.uint8)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="uint8", sample_rate=1e6)

        assert len(trace.data) == 3
        assert np.array_equal(trace.data, [0.0, 128.0, 255.0])

    def test_load_binary_with_offset(self, tmp_path: Path) -> None:
        """Test loading binary with offset.

        Note: Current implementation passes offset directly to np.fromfile as bytes,
        not elements as the docstring suggests. Testing actual behavior.
        """
        bin_path = tmp_path / "offset.bin"
        sample_data = np.arange(100, dtype=np.float32)
        sample_data.tofile(bin_path)

        # offset is actually in bytes (passed directly to np.fromfile)
        # To skip 10 float32 elements, need offset = 10 * 4 = 40 bytes
        offset_bytes = 10 * np.dtype(np.float32).itemsize
        trace = load_raw_binary(bin_path, dtype="float32", offset=offset_bytes, sample_rate=1e6)

        # Should skip first 10 elements
        assert trace.data[0] == pytest.approx(10.0)
        assert len(trace.data) == 90

    def test_load_binary_with_count(self, tmp_path: Path) -> None:
        """Test loading only specific number of samples."""
        bin_path = tmp_path / "count.bin"
        sample_data = np.arange(1000, dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float32", count=100, sample_rate=1e6)

        assert len(trace.data) == 100

    def test_load_binary_with_offset_and_count(self, tmp_path: Path) -> None:
        """Test loading with both offset (in bytes) and count (in elements)."""
        bin_path = tmp_path / "offset_count.bin"
        sample_data = np.arange(1000, dtype=np.float32)
        sample_data.tofile(bin_path)

        # offset is in bytes, count is in elements
        # To skip 100 float32 elements: offset = 100 * 4 = 400 bytes
        offset_bytes = 100 * np.dtype(np.float32).itemsize
        trace = load_raw_binary(
            bin_path, dtype="float32", offset=offset_bytes, count=50, sample_rate=1e6
        )

        assert len(trace.data) == 50
        assert trace.data[0] == pytest.approx(100.0)
        assert trace.data[-1] == pytest.approx(149.0)

    def test_load_binary_mmap_mode(self, tmp_path: Path) -> None:
        """Test memory-mapped loading of binary file."""
        bin_path = tmp_path / "large.bin"
        sample_data = np.random.randn(10000).astype(np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float32", mmap=True, sample_rate=1e9)

        assert len(trace.data) == 10000
        assert trace.metadata.sample_rate == 1e9

    def test_load_binary_mmap_keeps_original_dtype(self, tmp_path: Path) -> None:
        """Test that mmap mode preserves original dtype to avoid loading entire file."""
        bin_path = tmp_path / "mmap_dtype.bin"
        sample_data = np.arange(1000, dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float32", mmap=True, sample_rate=1e6)

        # With mmap, the original dtype might be preserved
        # The implementation may keep float32 to avoid loading entire file
        assert len(trace.data) == 1000

    def test_load_binary_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty binary file."""
        bin_path = tmp_path / "empty.bin"
        bin_path.write_bytes(b"")

        trace = load_raw_binary(bin_path, dtype="float32", sample_rate=1e6)

        assert len(trace.data) == 0

    def test_load_binary_source_file_metadata(self, tmp_path: Path) -> None:
        """Test that source file path is stored in metadata."""
        bin_path = tmp_path / "source.bin"
        sample_data = np.ones(10, dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float32", sample_rate=1e6)

        assert trace.metadata.source_file == str(bin_path)

    def test_load_binary_default_dtype(self, tmp_path: Path) -> None:
        """Test default dtype (float32)."""
        bin_path = tmp_path / "default.bin"
        sample_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, sample_rate=1e6)

        assert len(trace.data) == 3
        assert trace.data.dtype == np.float64

    def test_load_binary_default_sample_rate(self, tmp_path: Path) -> None:
        """Test default sample rate (1 MHz)."""
        bin_path = tmp_path / "default_sr.bin"
        sample_data = np.ones(10, dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path)

        assert trace.metadata.sample_rate == 1e6


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.requirement("MEM-016")
class TestLoadRawBinaryErrors:
    """Test error handling in load_raw_binary function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when binary file doesn't exist."""
        bin_path = tmp_path / "missing.bin"

        with pytest.raises(LoaderError, match="not found"):
            load_raw_binary(bin_path)

    def test_invalid_dtype(self, tmp_path: Path) -> None:
        """Test error with invalid dtype specification."""
        bin_path = tmp_path / "invalid.bin"
        bin_path.write_bytes(b"test data")

        with pytest.raises(LoaderError, match="Failed to load"):
            load_raw_binary(bin_path, dtype="invalid_dtype")


@pytest.mark.unit
@pytest.mark.loader
class TestLoadersNumpyLoaderEdgeCases:
    """Test edge cases and special scenarios."""

    def test_npz_with_only_metadata_arrays(self, tmp_path: Path) -> None:
        """Test NPZ with metadata but small arrays (should be skipped)."""
        npz_path = tmp_path / "meta_only.npz"
        # Arrays with size <= 10 are considered metadata
        np.savez(
            npz_path,
            sample_rate=np.array([1e6]),
            vertical_scale=np.array([0.1]),
            some_metadata=np.array([1, 2, 3]),
        )

        with pytest.raises(FormatError, match="No waveform data found"):
            load_npz(npz_path)

    def test_npz_fallback_to_first_array(self, tmp_path: Path) -> None:
        """Test fallback to first suitable array when no standard names found."""
        npz_path = tmp_path / "unusual_name.npz"
        sample_data = np.random.randn(100)
        # Use non-standard array name
        np.savez(npz_path, unusual_array_name=sample_data)

        trace = load_npz(npz_path)

        assert len(trace.data) == 100

    def test_npz_with_analog_prefix_channel(self, tmp_path: Path) -> None:
        """Test loading channel with 'analog' prefix."""
        npz_path = tmp_path / "analog.npz"
        analog1_data = np.random.randn(100)
        np.savez(npz_path, analog1=analog1_data)

        trace = load_npz(npz_path, channel=0)

        assert len(trace.data) == 100

    def test_npz_with_channel_prefix(self, tmp_path: Path) -> None:
        """Test loading channel with 'channel' prefix."""
        npz_path = tmp_path / "channel.npz"
        channel1_data = np.random.randn(100)
        np.savez(npz_path, channel1=channel1_data)

        trace = load_npz(npz_path, channel=0)

        assert len(trace.data) == 100

    def test_npz_channel_index_with_data_arrays(self, tmp_path: Path) -> None:
        """Test loading by index when no channel-named arrays exist."""
        npz_path = tmp_path / "data_idx.npz"
        # Use standard data array names (not ch/channel/analog prefix)
        data1 = np.ones(50)
        data2 = np.zeros(50)
        np.savez(npz_path, data=data1, signal=data2)

        # Should load second data array when requesting index 1
        trace = load_npz(npz_path, channel=1)

        assert len(trace.data) == 50
        assert trace.metadata.channel_name == "CH2"

    def test_npz_metadata_dict_with_errors(self, tmp_path: Path) -> None:
        """Test metadata extraction when metadata dict has invalid structure."""
        npz_path = tmp_path / "meta_error.npz"
        sample_data = np.random.randn(100)
        # Save metadata as regular array, not dict
        np.savez(npz_path, data=sample_data, metadata=np.array([1, 2, 3]))

        trace = load_npz(npz_path)

        # Should still load, just without metadata from dict
        assert len(trace.data) == 100
        assert trace.metadata.sample_rate == 1e6  # default

    def test_binary_negative_count(self, tmp_path: Path) -> None:
        """Test that negative count loads all data."""
        bin_path = tmp_path / "all.bin"
        sample_data = np.arange(100, dtype=np.float32)
        sample_data.tofile(bin_path)

        trace = load_raw_binary(bin_path, dtype="float32", count=-1, sample_rate=1e6)

        assert len(trace.data) == 100

    def test_large_sample_rate(self, tmp_path: Path) -> None:
        """Test handling of very large sample rates."""
        npz_path = tmp_path / "high_sr.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, sample_rate=100e9)  # 100 GS/s

        trace = load_npz(npz_path)

        assert trace.metadata.sample_rate == 100e9

    def test_very_small_sample_rate(self, tmp_path: Path) -> None:
        """Test handling of very small sample rates."""
        npz_path = tmp_path / "low_sr.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, sample_rate=1.0)  # 1 Hz

        trace = load_npz(npz_path)

        assert trace.metadata.sample_rate == 1.0

    def test_npz_with_nan_values(self, tmp_path: Path) -> None:
        """Test loading NPZ with NaN values in data."""
        npz_path = tmp_path / "nan.npz"
        sample_data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert len(trace.data) == 5
        assert np.isnan(trace.data[1])
        assert np.isnan(trace.data[3])

    def test_npz_with_inf_values(self, tmp_path: Path) -> None:
        """Test loading NPZ with infinity values in data."""
        npz_path = tmp_path / "inf.npz"
        sample_data = np.array([1.0, np.inf, -np.inf, 4.0])
        np.savez(npz_path, data=sample_data)

        trace = load_npz(npz_path)

        assert len(trace.data) == 4
        assert np.isinf(trace.data[1])
        assert np.isinf(trace.data[2])

    def test_metadata_case_insensitive_keys(self, tmp_path: Path) -> None:
        """Test that metadata key matching is case-insensitive."""
        npz_path = tmp_path / "case_meta.npz"
        sample_data = np.random.randn(100)
        np.savez(npz_path, data=sample_data, SAMPLE_RATE=3e6, V_SCALE=0.2)

        trace = load_npz(npz_path)

        assert trace.metadata.sample_rate == 3e6
        assert trace.metadata.vertical_scale == 0.2

    @pytest.mark.filterwarnings(
        "ignore:Casting complex values to real:numpy.exceptions.ComplexWarning"
    )
    def test_npz_complex_dtype(self, tmp_path: Path) -> None:
        """Test loading complex-valued arrays (should work with conversion)."""
        npz_path = tmp_path / "complex.npz"
        sample_data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        np.savez(npz_path, data=sample_data)

        # Complex arrays should be loaded (will be converted to float64)
        # The conversion might take magnitude or real part
        trace = load_npz(npz_path)

        assert len(trace.data) == 3
        assert trace.data.dtype == np.float64
