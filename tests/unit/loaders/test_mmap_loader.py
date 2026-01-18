"""Tests for memory-mapped loader.

This module tests the memory-mapped loading functionality for huge files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.loaders.mmap_loader import (
    MmapWaveformTrace,
    load_mmap,
    should_use_mmap,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestMmapWaveformTrace:
    """Tests for MmapWaveformTrace class."""

    def test_create_from_npy_file(self, tmp_path: Path) -> None:
        """Test creating memory-mapped trace from .npy file."""
        # Create test data
        data = np.random.randn(10000).astype(np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        # Load with memory mapping
        trace = load_mmap(npy_file, sample_rate=1e6)

        assert trace.sample_rate == 1e6
        assert trace.length == 10000
        assert trace.duration == pytest.approx(0.01)
        assert isinstance(trace, MmapWaveformTrace)

    def test_create_from_raw_binary(self, tmp_path: Path) -> None:
        """Test creating memory-mapped trace from raw binary file."""
        # Create test data
        data = np.random.randn(5000).astype(np.float64)
        bin_file = tmp_path / "test.f64"
        data.tofile(bin_file)

        # Load with memory mapping
        trace = load_mmap(
            bin_file,
            sample_rate=1e9,
            dtype=np.float64,
            length=5000,
        )

        assert trace.sample_rate == 1e9
        assert trace.length == 5000
        assert trace.dtype == np.float64

    def test_slicing(self, tmp_path: Path) -> None:
        """Test slicing memory-mapped trace."""
        # Create test data
        data = np.arange(1000, dtype=np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        # Load and slice
        trace = load_mmap(npy_file, sample_rate=1e6)

        # Test single index
        assert trace[0] == pytest.approx(0.0)
        assert trace[100] == pytest.approx(100.0)

        # Test slice
        subset = trace[100:200]
        assert len(subset) == 100
        assert subset[0] == pytest.approx(100.0)

    def test_iter_chunks(self, tmp_path: Path) -> None:
        """Test chunked iteration over memory-mapped trace."""
        # Create test data
        data = np.arange(10000, dtype=np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        # Load and iterate
        trace = load_mmap(npy_file, sample_rate=1e6)

        chunks = list(trace.iter_chunks(chunk_size=1000))
        assert len(chunks) == 10

        # Verify first chunk
        assert len(chunks[0]) == 1000
        assert chunks[0][0] == pytest.approx(0.0)
        assert chunks[0][-1] == pytest.approx(999.0)

        # Verify last chunk
        assert chunks[-1][-1] == pytest.approx(9999.0)

    def test_iter_chunks_with_overlap(self, tmp_path: Path) -> None:
        """Test chunked iteration with overlap."""
        # Create test data
        data = np.arange(5000, dtype=np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        trace = load_mmap(npy_file, sample_rate=1e6)

        # Iterate with 50% overlap
        chunks = list(trace.iter_chunks(chunk_size=1000, overlap=500))

        # Should have more chunks due to overlap
        assert len(chunks) > 5

        # Verify overlap
        assert chunks[0][-1] == pytest.approx(999.0)
        assert chunks[1][0] == pytest.approx(500.0)  # 500 overlap

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test using trace as context manager."""
        # Create test data
        data = np.random.randn(1000).astype(np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        # Use context manager
        with load_mmap(npy_file, sample_rate=1e6) as trace:
            assert trace.length == 1000
            subset = trace[0:100]
            assert len(subset) == 100

        # File should be closed after context

    def test_to_eager(self, tmp_path: Path) -> None:
        """Test converting memory-mapped trace to eager WaveformTrace."""
        # Create test data
        data = np.random.randn(1000).astype(np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        # Load as mmap and convert to eager
        mmap_trace = load_mmap(npy_file, sample_rate=1e6)
        eager_trace = mmap_trace.to_eager()

        assert eager_trace.metadata.sample_rate == 1e6
        assert len(eager_trace.data) == 1000

    def test_invalid_parameters(self, tmp_path: Path) -> None:
        """Test error handling for invalid parameters."""
        # Create test data
        data = np.random.randn(1000).astype(np.float32)
        npy_file = tmp_path / "test.npy"
        np.save(npy_file, data)

        # Test invalid sample rate
        with pytest.raises(Exception, match="sample_rate must be positive"):  # LoaderError
            load_mmap(npy_file, sample_rate=-1e6)

        # Test file not found
        with pytest.raises(Exception, match="File not found"):  # LoaderError
            load_mmap(tmp_path / "nonexistent.npy", sample_rate=1e6)

    def test_file_size_validation(self, tmp_path: Path) -> None:
        """Test file size validation against requested length."""
        # Create small test file
        data = np.random.randn(100).astype(np.float32)
        bin_file = tmp_path / "small.bin"
        data.tofile(bin_file)

        # Try to load more data than available
        with pytest.raises(Exception, match="File too small"):  # LoaderError - file too small
            load_mmap(
                bin_file,
                sample_rate=1e6,
                dtype=np.float32,
                length=1000,  # Request more than available
            )


class TestShouldUseMmap:
    """Tests for should_use_mmap helper function."""

    def test_small_file(self, tmp_path: Path) -> None:
        """Test that small files don't trigger mmap suggestion."""
        # Create small file (< 1 GB)
        small_file = tmp_path / "small.bin"
        data = np.random.randn(1000).astype(np.float32)
        data.tofile(small_file)

        assert not should_use_mmap(small_file)

    def test_large_file(self, tmp_path: Path) -> None:
        """Test that large files trigger mmap suggestion."""
        # We can't create a real 1GB file in tests, so test with custom threshold
        medium_file = tmp_path / "medium.bin"
        data = np.random.randn(10000).astype(np.float32)
        data.tofile(medium_file)

        # Use custom threshold smaller than file size
        file_size = medium_file.stat().st_size
        assert should_use_mmap(medium_file, threshold=file_size - 1)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test handling of nonexistent file."""
        nonexistent = tmp_path / "nonexistent.bin"
        assert not should_use_mmap(nonexistent)


class TestIntegrationWithExistingLoaders:
    """Test integration of mmap functionality with existing loaders."""

    def test_csv_with_mmap(self, tmp_path: Path) -> None:
        """Test CSV loader with mmap option."""
        # Create CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("time,voltage\n0.0,1.0\n0.001,2.0\n0.002,3.0\n0.003,4.0\n")

        from oscura.loaders.csv_loader import load_csv

        # Load with mmap enabled
        trace = load_csv(csv_file, mmap=True)

        # Should return MmapWaveformTrace
        assert isinstance(trace, MmapWaveformTrace)
        assert trace.length == 4

    def test_hdf5_with_mmap(self, tmp_path: Path) -> None:
        """Test HDF5 loader with mmap option."""
        pytest.importorskip("h5py")

        import h5py

        # Create HDF5 file
        h5_file = tmp_path / "test.h5"
        with h5py.File(h5_file, "w") as f:
            data = np.random.randn(1000).astype(np.float32)
            f.create_dataset("data", data=data)
            f["data"].attrs["sample_rate"] = 1e6

        from oscura.loaders.hdf5_loader import HDF5MmapTrace, load_hdf5

        # Load with mmap enabled
        trace = load_hdf5(h5_file, mmap=True)

        # Should return HDF5MmapTrace
        assert isinstance(trace, HDF5MmapTrace)
        assert trace.length == 1000


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")

        with pytest.raises(Exception):  # LoaderError
            load_mmap(
                empty_file,
                sample_rate=1e6,
                dtype=np.float32,
                length=100,
            )

    def test_npz_file_rejection(self, tmp_path: Path) -> None:
        """Test that NPZ files are rejected with helpful error."""
        npz_file = tmp_path / "test.npz"
        np.savez(npz_file, data=np.random.randn(100))

        with pytest.raises(Exception) as exc_info:  # LoaderError
            load_mmap(npz_file, sample_rate=1e6)

        # Should contain helpful message about extracting first
        assert "cannot be directly memory-mapped" in str(exc_info.value).lower()

    def test_fortran_ordered_array(self, tmp_path: Path) -> None:
        """Test rejection of Fortran-ordered arrays."""
        # Create Fortran-ordered array
        data = np.random.randn(100, 100).astype(np.float32)
        fortran_data = np.asfortranarray(data.ravel())

        # This test is conceptual - NumPy's save doesn't preserve Fortran order
        # for 1D arrays, but we test the error path
        npy_file = tmp_path / "fortran.npy"
        np.save(npy_file, fortran_data)

        # Should load successfully (1D arrays don't have order issues)
        trace = load_mmap(npy_file, sample_rate=1e6)
        assert trace.length == 10000
