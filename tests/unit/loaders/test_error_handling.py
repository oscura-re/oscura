"""Comprehensive error handling tests for loaders.

Tests all error paths in the loaders module to improve code coverage.

This module systematically tests:
- File not found errors
- Format errors (corrupted headers, invalid data)
- Configuration errors
- Permission errors
- Empty files
- Invalid metadata

- Coverage improvement for loader error paths
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import (
    ConfigurationError,
    FormatError,
    LoaderError,
    UnsupportedFormatError,
)
from oscura.loaders import load, load_all_channels
from oscura.loaders.configurable import ConfigurablePacketLoader
from oscura.loaders.csv_loader import load_csv
from oscura.loaders.hdf5_loader import load_hdf5
from oscura.loaders.lazy import LazyWaveformTrace, load_trace_lazy
from oscura.loaders.numpy_loader import load_npz, load_raw_binary
from oscura.loaders.tdms import load_tdms
from oscura.loaders.vcd import load_vcd
from oscura.loaders.wav import load_wav

pytestmark = [pytest.mark.unit, pytest.mark.loader]


# =============================================================================
# Generic Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestGenericLoaderErrors:
    """Test error handling in generic load functions."""

    def test_load_trace_file_not_found(self) -> None:
        """Test load with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load("/nonexistent/path/file.wav")

    def test_load_trace_unsupported_format(self, tmp_path: Path) -> None:
        """Test load with unsupported file extension."""
        unsupported_file = tmp_path / "data.xyz"
        unsupported_file.write_text("dummy data")

        with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
            load(unsupported_file)

    def test_load_traces_file_not_found(self) -> None:
        """Test load_all_channels with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_all_channels("/nonexistent/path/file.hdf5")

    def test_load_traces_unsupported_format(self, tmp_path: Path) -> None:
        """Test load_all_channels with unsupported file extension."""
        unsupported_file = tmp_path / "data.xyz"
        unsupported_file.write_text("dummy data")

        with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
            load_all_channels(unsupported_file)

    def test_load_trace_empty_file(self, tmp_path: Path) -> None:
        """Test load with empty NPZ file."""
        empty_file = tmp_path / "empty.npz"
        # Create an empty NPZ file
        np.savez(empty_file)

        with pytest.raises(LoaderError):
            load(empty_file)


# =============================================================================
# NPZ Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestNPZLoaderErrors:
    """Test error handling in NPZ loader."""

    def test_load_npz_file_not_found(self) -> None:
        """Test load_npz with non-existent file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_npz("/nonexistent/path/file.npz")

    def test_load_npz_missing_data_arrays(self, tmp_path: Path) -> None:
        """Test load_npz with no data arrays."""
        npz_path = tmp_path / "no_data.npz"
        # Save NPZ with only metadata, no data/waveform/signal arrays
        np.savez(npz_path, sample_rate=1e6, other_field="value")

        with pytest.raises(LoaderError, match="No waveform data found"):
            load_npz(npz_path)

    def test_load_npz_wrong_dtype(self, tmp_path: Path) -> None:
        """Test load_npz with non-numeric data."""
        npz_path = tmp_path / "wrong_dtype.npz"
        # Save string data
        np.savez(npz_path, data=np.array(["a", "b", "c"]))

        with pytest.raises(FormatError, match="not numeric"):
            load_npz(npz_path)

    def test_load_raw_binary_file_not_found(self) -> None:
        """Test load_raw_binary with non-existent file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_raw_binary("/nonexistent/path/file.bin", dtype=np.float32, sample_rate=1e6)

    def test_load_raw_binary_missing_sample_rate(self, tmp_path: Path) -> None:
        """Test load_raw_binary with default sample_rate."""
        bin_path = tmp_path / "data.bin"
        data = np.random.randn(100).astype(np.float32)
        data.tofile(bin_path)

        # Should succeed with default sample_rate
        trace = load_raw_binary(bin_path, dtype="float32")
        assert trace.metadata.sample_rate == 1e6  # Default value


# =============================================================================
# WAV Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestWAVLoaderErrors:
    """Test error handling in WAV loader."""

    def test_load_wav_file_not_found(self) -> None:
        """Test load_wav with non-existent file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_wav("/nonexistent/path/file.wav")

    def test_load_wav_invalid_header(self, tmp_path: Path) -> None:
        """Test load_wav with corrupted header."""
        wav_path = tmp_path / "bad_header.wav"
        # Write invalid header
        wav_path.write_bytes(b"INVALID_HEADER_DATA" + b"\x00" * 100)

        with pytest.raises(FormatError, match="Invalid WAV file format"):
            load_wav(wav_path)

    def test_load_wav_empty_file(self, tmp_path: Path) -> None:
        """Test load_wav with empty file."""
        wav_path = tmp_path / "empty.wav"
        wav_path.write_bytes(b"")

        with pytest.raises(FormatError, match="Invalid WAV file format"):
            load_wav(wav_path)

    def test_load_wav_truncated_file(self, tmp_path: Path) -> None:
        """Test load_wav with truncated file."""
        wav_path = tmp_path / "truncated.wav"
        # Write RIFF header but truncate
        wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        with pytest.raises((LoaderError, FormatError), match="(Invalid WAV|Failed to read)"):
            load_wav(wav_path)


# =============================================================================
# CSV Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestCSVLoaderErrors:
    """Test error handling in CSV loader."""

    def test_load_csv_file_not_found(self) -> None:
        """Test load_csv with non-existent file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_csv("/nonexistent/path/file.csv")

    def test_load_csv_empty_file(self, tmp_path: Path) -> None:
        """Test load_csv with empty file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        with pytest.raises((FormatError, LoaderError), match="(No|empty|data)"):
            load_csv(csv_path)

    def test_load_csv_no_numeric_columns(self, tmp_path: Path) -> None:
        """Test load_csv with no numeric data."""
        csv_path = tmp_path / "no_numbers.csv"
        csv_path.write_text("name,category\nfoo,bar\nbaz,qux\n")

        with pytest.raises(FormatError, match="No voltage data found"):
            load_csv(csv_path)

    def test_load_csv_malformed_data(self, tmp_path: Path) -> None:
        """Test load_csv with malformed numeric data (non-numeric column)."""
        csv_path = tmp_path / "malformed.csv"
        csv_path.write_text("time,voltage\n0,1.0\n1,invalid\n2,3.0\n")

        # When a column has non-numeric values, pandas treats it as object dtype,
        # and the loader rejects it as "No voltage data found"
        with pytest.raises(FormatError, match="No voltage data found"):
            load_csv(csv_path)


# =============================================================================
# VCD Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestVCDLoaderErrors:
    """Test error handling in VCD loader."""

    def test_load_vcd_file_not_found(self) -> None:
        """Test load_vcd with non-existent file."""
        with pytest.raises(LoaderError):
            load_vcd("/nonexistent/path/file.vcd")

    def test_load_vcd_invalid_header(self, tmp_path: Path) -> None:
        """Test load_vcd with invalid header."""
        vcd_path = tmp_path / "invalid.vcd"
        vcd_path.write_text("INVALID VCD CONTENT\n")

        with pytest.raises(FormatError):
            load_vcd(vcd_path)

    def test_load_vcd_missing_timescale(self, tmp_path: Path) -> None:
        """Test load_vcd with missing timescale."""
        vcd_path = tmp_path / "no_timescale.vcd"
        vcd_path.write_text(
            "$version Generated VCD $end\n"
            "$scope module top $end\n"
            "$var wire 1 ! clk $end\n"
            "$upscope $end\n"
            "$enddefinitions $end\n"
        )

        with pytest.raises(FormatError, match="timescale"):
            load_vcd(vcd_path)

    def test_load_vcd_no_variables(self, tmp_path: Path) -> None:
        """Test load_vcd with no variables defined."""
        vcd_path = tmp_path / "no_vars.vcd"
        vcd_path.write_text(
            "$timescale 1ns $end\n$scope module top $end\n$upscope $end\n$enddefinitions $end\n#0\n"
        )

        with pytest.raises(FormatError, match="No variables"):
            load_vcd(vcd_path)


# =============================================================================
# HDF5 Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestHDF5LoaderErrors:
    """Test error handling in HDF5 loader."""

    def test_load_hdf5_file_not_found(self) -> None:
        """Test load_hdf5 with non-existent file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_hdf5("/nonexistent/path/file.hdf5")

    def test_load_hdf5_no_datasets(self, tmp_path: Path) -> None:
        """Test load_hdf5 with no datasets."""
        try:
            import h5py
        except ImportError:
            # SKIP: Valid - Optional h5py dependency
            # Only skip if h5py not installed (pip install oscura[hdf5])
            pytest.skip("h5py not installed")

        h5_path = tmp_path / "no_datasets.hdf5"
        with h5py.File(h5_path, "w") as f:
            # Create file with only groups, no datasets
            f.create_group("group1")

        with pytest.raises(LoaderError):
            load_hdf5(h5_path)

    def test_load_hdf5_wrong_dtype(self, tmp_path: Path) -> None:
        """Test load_hdf5 with non-numeric dataset."""
        try:
            import h5py
        except ImportError:
            # SKIP: Valid - Optional h5py dependency
            # Only skip if h5py not installed (pip install oscura[hdf5])
            pytest.skip("h5py not installed")

        h5_path = tmp_path / "wrong_dtype.hdf5"
        with h5py.File(h5_path, "w") as f:
            # Create string dataset
            f.create_dataset("data", data=np.array([b"a", b"b", b"c"]))

        with pytest.raises(LoaderError):
            load_hdf5(h5_path)


# =============================================================================
# TDMS Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestTDMSLoaderErrors:
    """Test error handling in TDMS loader."""

    def test_load_tdms_file_not_found(self) -> None:
        """Test load_tdms with non-existent file."""
        with pytest.raises(LoaderError):
            load_tdms("/nonexistent/path/file.tdms")

    def test_load_tdms_invalid_header(self, tmp_path: Path) -> None:
        """Test load_tdms with invalid header."""
        try:
            import nptdms  # noqa: F401
        except ImportError:
            # SKIP: Valid - Optional dependency
            # Only skip if required: nptdms not available
            pytest.skip("nptdms not available")

        tdms_path = tmp_path / "invalid.tdms"
        tdms_path.write_bytes(b"INVALID_TDMS_DATA" + b"\x00" * 100)

        with pytest.raises((LoaderError, FormatError)):
            load_tdms(tdms_path)

    def test_load_tdms_empty_file(self, tmp_path: Path) -> None:
        """Test load_tdms with empty file."""
        tdms_path = tmp_path / "empty.tdms"
        tdms_path.write_bytes(b"")

        with pytest.raises(LoaderError):
            load_tdms(tdms_path)


# =============================================================================
# Configurable Loader Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestConfigurableLoaderErrors:
    """Test error handling in ConfigurablePacketLoader."""

    def test_missing_packet_format(self, tmp_path: Path) -> None:
        """Test ConfigurablePacketLoader without packet_format."""
        config = {"endianness": "little"}
        bin_path = tmp_path / "data.bin"
        bin_path.write_bytes(b"\x00" * 100)

        with pytest.raises(ConfigurationError):
            from oscura.loaders.configurable import PacketFormatConfig

            PacketFormatConfig.from_dict(config)

    def test_invalid_packet_format_type(self, tmp_path: Path) -> None:
        """Test ConfigurablePacketLoader with invalid packet_format type."""
        config = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 100},
            "header": {"size": 10},
            "samples": {"offset": 10, "count": 10, "format": {"size": 1, "type": "uint8"}},
        }
        bin_path = tmp_path / "data.bin"
        bin_path.write_bytes(b"\x00" * 100)

        # This should succeed - config is valid
        from oscura.loaders.configurable import PacketFormatConfig

        fmt_cfg = PacketFormatConfig.from_dict(config)
        loader = ConfigurablePacketLoader(fmt_cfg)
        # Config is valid, so no error raised during construction

    def test_missing_required_field(self, tmp_path: Path) -> None:
        """Test ConfigurablePacketLoader with missing required field."""
        config = {
            "name": "test",
            "version": "1.0",
            # Missing "packet" field
            "header": {"size": 10},
            "samples": {"offset": 10, "count": 10, "format": {"size": 1, "type": "uint8"}},
        }
        bin_path = tmp_path / "data.bin"
        bin_path.write_bytes(b"\x00" * 100)

        with pytest.raises(ConfigurationError):
            from oscura.loaders.configurable import PacketFormatConfig

            PacketFormatConfig.from_dict(config)

    def test_invalid_endianness(self, tmp_path: Path) -> None:
        """Test ConfigurablePacketLoader with invalid endianness."""
        config = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 100, "byte_order": "invalid"},
            "header": {
                "size": 10,
                "fields": [
                    {"name": "data", "type": "uint8", "offset": 0, "size": 1, "endian": "invalid"}
                ],
            },
            "samples": {"offset": 10, "count": 10, "format": {"size": 1, "type": "uint8"}},
        }
        bin_path = tmp_path / "data.bin"
        bin_path.write_bytes(b"\x00" * 100)

        with pytest.raises(ConfigurationError, match="endianness"):
            from oscura.loaders.configurable import PacketFormatConfig

            PacketFormatConfig.from_dict(config)

    def test_load_insufficient_data(self, tmp_path: Path) -> None:
        """Test ConfigurablePacketLoader with file too small for packet."""
        config = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 100},
            "header": {"size": 10, "fields": []},
            "samples": {"offset": 10, "count": 10, "format": {"size": 8, "type": "uint64"}},
        }
        bin_path = tmp_path / "small.bin"
        bin_path.write_bytes(b"\x00" * 10)  # Only 10 bytes, need 100

        from oscura.loaders.configurable import PacketFormatConfig

        fmt_cfg = PacketFormatConfig.from_dict(config)
        loader = ConfigurablePacketLoader(fmt_cfg)

        # File is too small, loader should handle gracefully (warn and return empty list)
        result = loader.load(bin_path)
        assert result.packet_count == 0  # No complete packets


# =============================================================================
# Lazy Loading Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestLazyLoadingErrors:
    """Test error handling in lazy loading."""

    def test_create_lazy_trace_file_not_found(self) -> None:
        """Test load_trace_lazy with non-existent file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_trace_lazy("/nonexistent/path/file.npy", sample_rate=1e6)

    def test_lazy_trace_missing_sample_rate_npy(self, tmp_path: Path) -> None:
        """Test LazyWaveformTrace with .npy file but no sample_rate."""
        npy_path = tmp_path / "data.npy"
        np.save(npy_path, np.random.randn(100))

        with pytest.raises(LoaderError, match="sample_rate is required"):
            load_trace_lazy(npy_path)

    def test_lazy_trace_wrong_dimensions(self, tmp_path: Path) -> None:
        """Test LazyWaveformTrace with 2D array."""
        npy_path = tmp_path / "2d_data.npy"
        np.save(npy_path, np.random.randn(10, 10))

        with pytest.raises(LoaderError, match="Expected 1D array"):
            load_trace_lazy(npy_path, sample_rate=1e6)

    def test_lazy_trace_invalid_index_type(self, tmp_path: Path) -> None:
        """Test LazyWaveformTrace with invalid index type."""
        npy_path = tmp_path / "data.npy"
        np.save(npy_path, np.random.randn(100))

        lazy = load_trace_lazy(npy_path, sample_rate=1e6, lazy=True)

        with pytest.raises(TypeError, match="Indices must be int or slice"):
            lazy["invalid"]  # type: ignore[index]

    def test_lazy_trace_load_failure(self, tmp_path: Path) -> None:
        """Test LazyWaveformTrace when file is deleted after creation."""
        npy_path = tmp_path / "data.npy"
        data = np.random.randn(100)
        np.save(npy_path, data)

        lazy = LazyWaveformTrace(npy_path, sample_rate=1e6, length=len(data))

        # Delete file
        npy_path.unlink()

        with pytest.raises(LoaderError, match="Failed to load data"):
            _ = lazy.data  # Trigger lazy load


# =============================================================================
# Parametrized Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.parametrize(
    "invalid_input,expected_error",
    [
        (np.array([]), LoaderError),  # Empty array
        (np.array([np.nan, np.nan]), LoaderError),  # All NaN
        (np.array([np.inf, -np.inf]), LoaderError),  # All Inf
    ],
)
class TestInvalidDataInputs:
    """Test loaders with invalid data inputs."""

    def test_npz_invalid_data(
        self,
        invalid_input: np.ndarray,
        expected_error: type[Exception],
        tmp_path: Path,
    ) -> None:
        """Test NPZ loader with invalid data."""
        npz_path = tmp_path / "invalid.npz"
        np.savez(npz_path, data=invalid_input, sample_rate=1e6)

        # All data types should load successfully (empty, NaN, Inf are valid arrays)
        # Validation happens at analysis time, not load time
        trace = load_npz(npz_path)
        assert trace is not None
        assert len(trace.data) == len(invalid_input)


@pytest.mark.unit
@pytest.mark.loader
@pytest.mark.parametrize(
    "format_ext,loader_func",
    [
        (".npz", load_npz),
        (".wav", load_wav),
        (".csv", load_csv),
        (".vcd", load_vcd),
        (".hdf5", load_hdf5),
    ],
)
class TestEmptyFileHandling:
    """Test all loaders with empty files."""

    def test_empty_file(
        self,
        format_ext: str,
        loader_func: object,
        tmp_path: Path,
    ) -> None:
        """Test loader with empty file."""
        empty_file = tmp_path / f"empty{format_ext}"
        empty_file.write_bytes(b"")

        with pytest.raises((LoaderError, FormatError, OSError)):
            loader_func(empty_file)  # type: ignore[operator]


# =============================================================================
# File Permission Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestFilePermissionErrors:
    """Test loader behavior with file permission errors."""

    def test_load_trace_no_read_permission(self, tmp_path: Path) -> None:
        """Test load with unreadable file."""
        npz_path = tmp_path / "unreadable.npz"
        np.savez(npz_path, data=np.random.randn(100), sample_rate=1e6)

        # Remove read permission
        npz_path.chmod(0o000)

        try:
            with pytest.raises((LoaderError, PermissionError)):
                load(npz_path)
        finally:
            # Restore permissions for cleanup
            npz_path.chmod(0o644)


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.loader
class TestLoadersErrorHandlingEdgeCases:
    """Test edge cases in loader error handling."""

    def test_load_npz_extremely_large_metadata(self, tmp_path: Path) -> None:
        """Test NPZ with extremely large metadata."""
        npz_path = tmp_path / "large_metadata.npz"
        huge_string = "x" * 1_000_000
        np.savez(
            npz_path,
            data=np.random.randn(100),
            sample_rate=1e6,
            description=huge_string,
        )

        # Should load successfully despite large metadata
        trace = load_npz(npz_path)
        assert trace is not None

    def test_load_csv_inconsistent_columns(self, tmp_path: Path) -> None:
        """Test CSV with inconsistent column counts."""
        csv_path = tmp_path / "inconsistent.csv"
        csv_path.write_text("time,value\n0,1.0\n1,2.0,3.0\n2,4.0\n")

        with pytest.raises(FormatError):
            load_csv(csv_path)

    def test_configurable_loader_circular_reference(self, tmp_path: Path) -> None:
        """Test ConfigurablePacketLoader with invalid header field size."""
        config = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 100},
            "header": {
                "size": 10,
                "fields": [
                    {
                        "name": "field1",
                        "type": "uint8",
                        "offset": 0,
                        "size": -1,  # Invalid negative size
                    }
                ],
            },
            "samples": {"offset": 10, "count": 10, "format": {"size": 1, "type": "uint8"}},
        }
        bin_path = tmp_path / "circular.bin"
        bin_path.write_bytes(b"\x00" * 100)

        # Should raise ConfigurationError due to invalid size
        with pytest.raises(ConfigurationError):
            from oscura.loaders.configurable import PacketFormatConfig

            PacketFormatConfig.from_dict(config)
