"""Tests for ChipWhisperer trace loader.

Tests for ChipWhisperer .npy and .trs file loaders.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.chipwhisperer import (
    load_chipwhisperer,
    load_chipwhisperer_npy,
    load_chipwhisperer_trs,
    to_waveform_trace,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


# =============================================================================
# NPY format tests
# =============================================================================


def test_load_chipwhisperer_npy_basic():
    """Test loading basic ChipWhisperer .npy file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test trace data
        traces = np.random.randn(100, 500).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        # Load traces
        traceset = load_chipwhisperer_npy(trace_file)

        assert traceset.n_traces == 100
        assert traceset.n_samples == 500
        assert traceset.sample_rate == 1e6  # Default
        np.testing.assert_array_equal(traceset.traces, traces.astype(np.float64))


def test_load_chipwhisperer_npy_with_plaintexts():
    """Test loading .npy with associated plaintext file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test data
        traces = np.random.randn(50, 200).astype(np.float32)
        plaintexts = np.random.randint(0, 256, (50, 16), dtype=np.uint8)

        trace_file = tmp_path / "traces.npy"
        textin_file = tmp_path / "textin.npy"

        np.save(trace_file, traces)
        np.save(textin_file, plaintexts)

        # Load traces
        traceset = load_chipwhisperer_npy(trace_file)

        assert traceset.n_traces == 50
        assert traceset.plaintexts is not None
        np.testing.assert_array_equal(traceset.plaintexts, plaintexts)


def test_load_chipwhisperer_npy_with_all_metadata():
    """Test loading .npy with all metadata files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test data
        n_traces = 30
        traces = np.random.randn(n_traces, 100).astype(np.float32)
        plaintexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)
        ciphertexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)
        keys = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)

        np.save(tmp_path / "traces.npy", traces)
        np.save(tmp_path / "textin.npy", plaintexts)
        np.save(tmp_path / "textout.npy", ciphertexts)
        np.save(tmp_path / "keys.npy", keys)

        # Load traces
        traceset = load_chipwhisperer_npy(tmp_path / "traces.npy")

        assert traceset.plaintexts is not None
        assert traceset.ciphertexts is not None
        assert traceset.keys is not None
        np.testing.assert_array_equal(traceset.plaintexts, plaintexts)
        np.testing.assert_array_equal(traceset.ciphertexts, ciphertexts)
        np.testing.assert_array_equal(traceset.keys, keys)


def test_load_chipwhisperer_npy_1d_trace():
    """Test loading single 1D trace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Single trace (1D array)
        trace = np.random.randn(300).astype(np.float32)
        trace_file = tmp_path / "trace.npy"
        np.save(trace_file, trace)

        traceset = load_chipwhisperer_npy(trace_file)

        assert traceset.n_traces == 1
        assert traceset.n_samples == 300
        np.testing.assert_array_equal(traceset.traces[0], trace.astype(np.float64))


def test_load_chipwhisperer_npy_custom_sample_rate():
    """Test loading with custom sample rate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        traces = np.random.randn(10, 100).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        traceset = load_chipwhisperer_npy(trace_file, sample_rate=500e6)

        assert traceset.sample_rate == 500e6


def test_load_chipwhisperer_npy_invalid_dimensions():
    """Test loading with invalid trace dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 3D array (invalid)
        traces = np.random.randn(10, 20, 30).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        with pytest.raises(FormatError, match="Expected 1D or 2D"):
            load_chipwhisperer_npy(trace_file)


def test_load_chipwhisperer_npy_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(LoaderError, match="Failed to load"):
        load_chipwhisperer_npy("/nonexistent/file.npy")


# =============================================================================
# TRS format tests
# =============================================================================


def test_load_chipwhisperer_trs_basic():
    """Test loading basic TRS file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trs_file = tmp_path / "traces.trs"

        # Create minimal TRS file
        n_traces = 10
        n_samples = 50

        with open(trs_file, "wb") as f:
            # Tag 0x41: Number of traces (2 bytes)
            f.write(b"\x41\x02")
            f.write(n_traces.to_bytes(2, byteorder="little"))

            # Tag 0x42: Number of samples (2 bytes)
            f.write(b"\x42\x02")
            f.write(n_samples.to_bytes(2, byteorder="little"))

            # Tag 0x43: Sample coding (1 byte: 1=int8)
            f.write(b"\x43\x01\x01")

            # Tag 0x44: Data length (2 bytes: 0=no data)
            f.write(b"\x44\x02\x00\x00")

            # End of header
            f.write(b"\x5f")

            # Write trace data
            for _ in range(n_traces):
                trace_data = np.random.randint(-128, 127, n_samples, dtype=np.int8)
                f.write(trace_data.tobytes())

        traceset = load_chipwhisperer_trs(trs_file)

        assert traceset.n_traces == n_traces
        assert traceset.n_samples == n_samples
        assert traceset.traces.shape == (n_traces, n_samples)


def test_load_chipwhisperer_trs_with_plaintexts():
    """Test loading TRS file with plaintext data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trs_file = tmp_path / "traces.trs"

        n_traces = 5
        n_samples = 30
        data_length = 16  # 16 bytes plaintext

        with open(trs_file, "wb") as f:
            # Header
            f.write(b"\x41\x02")
            f.write(n_traces.to_bytes(2, byteorder="little"))
            f.write(b"\x42\x02")
            f.write(n_samples.to_bytes(2, byteorder="little"))
            f.write(b"\x43\x01\x01")  # int8
            f.write(b"\x44\x02")
            f.write(data_length.to_bytes(2, byteorder="little"))
            f.write(b"\x5f")

            # Write traces with plaintext
            for _ in range(n_traces):
                # Plaintext
                plaintext = np.random.randint(0, 256, data_length, dtype=np.uint8)
                f.write(plaintext.tobytes())

                # Trace samples
                trace_data = np.random.randint(-128, 127, n_samples, dtype=np.int8)
                f.write(trace_data.tobytes())

        traceset = load_chipwhisperer_trs(trs_file)

        assert traceset.n_traces == n_traces
        assert traceset.plaintexts is not None
        assert traceset.plaintexts.shape == (n_traces, data_length)


def test_load_chipwhisperer_trs_float_samples():
    """Test loading TRS file with float32 samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trs_file = tmp_path / "traces.trs"

        n_traces = 3
        n_samples = 20

        with open(trs_file, "wb") as f:
            # Header
            f.write(b"\x41\x02")
            f.write(n_traces.to_bytes(2, byteorder="little"))
            f.write(b"\x42\x02")
            f.write(n_samples.to_bytes(2, byteorder="little"))
            f.write(b"\x43\x01\x04")  # 4 = float32
            f.write(b"\x44\x02\x00\x00")
            f.write(b"\x5f")

            # Write float traces
            for _ in range(n_traces):
                trace_data = np.random.randn(n_samples).astype(np.float32)
                f.write(trace_data.tobytes())

        traceset = load_chipwhisperer_trs(trs_file)

        assert traceset.n_traces == n_traces
        assert traceset.n_samples == n_samples


def test_load_chipwhisperer_trs_invalid_header():
    """Test loading TRS file with invalid header."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trs_file = tmp_path / "invalid.trs"

        with open(trs_file, "wb") as f:
            # Invalid: no traces or samples
            f.write(b"\x41\x02\x00\x00")
            f.write(b"\x42\x02\x00\x00")
            f.write(b"\x5f")

        with pytest.raises(FormatError, match="zero traces or samples"):
            load_chipwhisperer_trs(trs_file)


def test_load_chipwhisperer_trs_unsupported_coding():
    """Test loading TRS file with unsupported sample coding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trs_file = tmp_path / "traces.trs"

        with open(trs_file, "wb") as f:
            f.write(b"\x41\x02\x05\x00")  # 5 traces
            f.write(b"\x42\x02\x0a\x00")  # 10 samples
            f.write(b"\x43\x01\x08")  # Unsupported coding (8)
            f.write(b"\x5f")

        with pytest.raises(FormatError, match="Unsupported sample coding"):
            load_chipwhisperer_trs(trs_file)


# =============================================================================
# Auto-detection tests
# =============================================================================


def test_load_chipwhisperer_auto_npy():
    """Test auto-detection for .npy files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        traces = np.random.randn(20, 100).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        traceset = load_chipwhisperer(trace_file)

        assert traceset.n_traces == 20
        assert traceset.n_samples == 100


def test_load_chipwhisperer_unsupported_format():
    """Test loading unsupported file format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        invalid_file = tmp_path / "traces.txt"
        invalid_file.write_text("not a trace file")

        with pytest.raises(FormatError, match="Unsupported ChipWhisperer format"):
            load_chipwhisperer(invalid_file)


def test_load_chipwhisperer_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(LoaderError, match="File not found"):
        load_chipwhisperer("/nonexistent/file.npy")


# =============================================================================
# Conversion tests
# =============================================================================


def test_to_waveform_trace():
    """Test converting ChipWhisperer trace to WaveformTrace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        traces = np.random.randn(10, 100).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        traceset = load_chipwhisperer_npy(trace_file, sample_rate=1e9)
        waveform = to_waveform_trace(traceset, trace_index=5)

        assert waveform.metadata.sample_rate == 1e9
        assert len(waveform.data) == 100
        np.testing.assert_array_equal(waveform.data, traces[5].astype(np.float64))


def test_to_waveform_trace_invalid_index():
    """Test converting with invalid trace index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        traces = np.random.randn(10, 100).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        traceset = load_chipwhisperer_npy(trace_file)

        with pytest.raises(IndexError, match="out of range"):
            to_waveform_trace(traceset, trace_index=10)


# =============================================================================
# Metadata tests
# =============================================================================


def test_chipwhisperer_traceset_properties():
    """Test ChipWhispererTraceSet properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        traces = np.random.randn(15, 75).astype(np.float32)
        trace_file = tmp_path / "traces.npy"
        np.save(trace_file, traces)

        traceset = load_chipwhisperer_npy(trace_file)

        assert traceset.n_traces == 15
        assert traceset.n_samples == 75
        assert traceset.sample_rate == 1e6
        assert traceset.metadata is not None
        assert "source_file" in traceset.metadata
        assert "format" in traceset.metadata
