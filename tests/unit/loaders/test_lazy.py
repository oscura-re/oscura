"""Comprehensive unit tests for src/oscura/loaders/lazy.py

Tests coverage for:

Covers all public classes, functions, properties, and methods with
edge cases, error handling, and validation.
"""

from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import LoaderError
from oscura.core.types import WaveformTrace
from oscura.loaders.lazy import LazyWaveformTrace, load_trace_lazy

pytestmark = [pytest.mark.unit, pytest.mark.loader]


# ==============================================================================
# Fixtures for Test Data
# ==============================================================================


@pytest.fixture
def temp_npy_file(tmp_path: Path) -> Path:
    """Create temporary .npy file for testing."""
    file_path = tmp_path / "test_trace.npy"
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    np.save(file_path, data)
    return file_path


@pytest.fixture
def temp_raw_file(tmp_path: Path) -> Path:
    """Create temporary raw binary file for testing."""
    file_path = tmp_path / "test_trace.bin"
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    data.tofile(file_path)
    return file_path


@pytest.fixture
def large_npy_file(tmp_path: Path) -> Path:
    """Create larger .npy file for testing slicing."""
    file_path = tmp_path / "large_trace.npy"
    data = np.arange(1000, dtype=np.float64)
    np.save(file_path, data)
    return file_path


@pytest.fixture
def large_raw_file(tmp_path: Path) -> Path:
    """Create larger raw binary file for testing slicing."""
    file_path = tmp_path / "large_trace.bin"
    data = np.arange(1000, dtype=np.float64)
    data.tofile(file_path)
    return file_path


# ==============================================================================
# LazyWaveformTrace Creation Tests
# ==============================================================================


class TestLazyWaveformTraceCreation:
    """Test LazyWaveformTrace initialization."""

    def test_create_minimal(self, temp_npy_file: Path) -> None:
        """Test creating LazyWaveformTrace with minimal arguments."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        assert trace.sample_rate == 1e6
        assert trace.length == 5
        assert trace.metadata == {}

    def test_create_with_dtype(self, temp_npy_file: Path) -> None:
        """Test creating with custom dtype."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
            dtype=np.float32,
        )
        assert trace._dtype == np.dtype(np.float32)

    def test_create_with_offset(self, temp_raw_file: Path) -> None:
        """Test creating with byte offset."""
        trace = LazyWaveformTrace(
            file_path=temp_raw_file,
            sample_rate=1e6,
            length=3,
            offset=16,  # Skip first 2 float64 values (8 bytes each)
        )
        assert trace._offset == 16

    def test_create_with_metadata(self, temp_npy_file: Path) -> None:
        """Test creating with metadata."""
        metadata = {"source": "oscilloscope", "channel": 1}
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
            metadata=metadata,
        )
        assert trace.metadata == metadata

    def test_create_with_string_path(self, temp_npy_file: Path) -> None:
        """Test creating with string file path."""
        trace = LazyWaveformTrace(
            file_path=str(temp_npy_file),
            sample_rate=1e6,
            length=5,
        )
        assert trace._file_path == temp_npy_file

    def test_create_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent file raises LoaderError."""
        with pytest.raises(LoaderError, match="File not found"):
            LazyWaveformTrace(
                file_path=tmp_path / "nonexistent.npy",
                sample_rate=1e6,
                length=5,
            )

    def test_data_not_loaded_on_creation(self, temp_npy_file: Path) -> None:
        """Test that data is not loaded during initialization."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        assert trace._data is None
        assert trace._memmap is None


# ==============================================================================
# LazyWaveformTrace Properties Tests
# ==============================================================================


class TestLazyWaveformTraceProperties:
    """Test LazyWaveformTrace properties."""

    def test_sample_rate_property(self, temp_npy_file: Path) -> None:
        """Test sample_rate property."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=2.5e9,
            length=5,
        )
        assert trace.sample_rate == 2.5e9

    def test_length_property(self, temp_npy_file: Path) -> None:
        """Test length property."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        assert trace.length == 5

    def test_duration_property(self, temp_npy_file: Path) -> None:
        """Test duration property."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        # duration = length / sample_rate = 5 / 1e6 = 5e-6
        assert trace.duration == pytest.approx(5e-6)

    def test_metadata_property(self, temp_npy_file: Path) -> None:
        """Test metadata property."""
        metadata = {"test": "value"}
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
            metadata=metadata,
        )
        assert trace.metadata == metadata

    def test_time_vector_property(self, temp_npy_file: Path) -> None:
        """Test time_vector property."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        time_vec = trace.time_vector
        expected = np.array([0, 1, 2, 3, 4]) / 1e6
        np.testing.assert_array_almost_equal(time_vec, expected)

    def test_time_vector_does_not_load_data(self, temp_npy_file: Path) -> None:
        """Test that time_vector doesn't trigger data loading."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        _ = trace.time_vector
        assert trace._data is None  # Data still not loaded


# ==============================================================================
# LazyWaveformTrace Data Loading Tests
# ==============================================================================


class TestLazyWaveformTraceDataLoading:
    """Test LazyWaveformTrace lazy data loading."""

    def test_data_loads_on_first_access(self, temp_raw_file: Path) -> None:
        """Test that data property triggers loading."""
        trace = LazyWaveformTrace(
            file_path=temp_raw_file,
            sample_rate=1e6,
            length=5,
        )
        # Before access
        assert trace._data is None

        # Access data
        data = trace.data

        # After access
        assert trace._data is not None
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_data_cached_after_load(self, temp_npy_file: Path) -> None:
        """Test that data is cached after first load."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        data1 = trace.data
        data2 = trace.data

        # Should return same cached array
        assert data1 is data2

    def test_load_data_with_offset(self, temp_raw_file: Path) -> None:
        """Test loading data with byte offset."""
        # Skip first 2 float64 values (16 bytes)
        trace = LazyWaveformTrace(
            file_path=temp_raw_file,
            sample_rate=1e6,
            length=3,
            offset=16,
            dtype=np.float64,
        )
        data = trace.data
        # Should load values [3.0, 4.0, 5.0]
        np.testing.assert_array_equal(data, [3.0, 4.0, 5.0])

    def test_load_data_wrong_length_raises_error(self, temp_npy_file: Path) -> None:
        """Test that loading with wrong length raises error."""
        # Specify length longer than actual file
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=1000,  # File only has 5 samples
        )
        with pytest.raises(LoaderError, match="Failed to load data"):
            _ = trace.data


# ==============================================================================
# LazyWaveformTrace Indexing Tests
# ==============================================================================


class TestLazyWaveformTraceIndexing:
    """Test LazyWaveformTrace indexing and slicing."""

    def test_getitem_single_index(self, large_raw_file: Path) -> None:
        """Test getting single sample by index."""
        trace = LazyWaveformTrace(
            file_path=large_raw_file,
            sample_rate=1e6,
            length=1000,
        )
        sample = trace[100]
        assert sample == 100.0

    def test_getitem_single_index_negative(self, large_raw_file: Path) -> None:
        """Test getting single sample by negative index."""
        trace = LazyWaveformTrace(
            file_path=large_raw_file,
            sample_rate=1e6,
            length=1000,
        )
        sample = trace[-1]
        assert sample == 999.0

    def test_getitem_single_index_creates_memmap(self, large_npy_file: Path) -> None:
        """Test that single index access creates memmap."""
        trace = LazyWaveformTrace(
            file_path=large_npy_file,
            sample_rate=1e6,
            length=1000,
        )
        assert trace._memmap is None
        _ = trace[100]
        assert trace._memmap is not None

    def test_getitem_slice_returns_lazy_trace(self, large_npy_file: Path) -> None:
        """Test that slicing returns new LazyWaveformTrace."""
        trace = LazyWaveformTrace(
            file_path=large_npy_file,
            sample_rate=1e6,
            length=1000,
        )
        subset = trace[100:200]
        assert isinstance(subset, LazyWaveformTrace)
        assert subset.length == 100
        assert subset._data is None  # Still lazy

    def test_getitem_slice_correct_data(self, large_raw_file: Path) -> None:
        """Test that slice returns correct data when accessed."""
        trace = LazyWaveformTrace(
            file_path=large_raw_file,
            sample_rate=1e6,
            length=1000,
        )
        subset = trace[100:105]
        data = subset.data
        np.testing.assert_array_equal(data, [100.0, 101.0, 102.0, 103.0, 104.0])

    def test_getitem_slice_preserves_sample_rate(self, large_npy_file: Path) -> None:
        """Test that slicing preserves sample rate."""
        trace = LazyWaveformTrace(
            file_path=large_npy_file,
            sample_rate=2.5e9,
            length=1000,
        )
        subset = trace[100:200]
        assert subset.sample_rate == 2.5e9

    def test_getitem_slice_preserves_metadata(self, large_npy_file: Path) -> None:
        """Test that slicing preserves metadata."""
        metadata = {"channel": 1}
        trace = LazyWaveformTrace(
            file_path=large_npy_file,
            sample_rate=1e6,
            length=1000,
            metadata=metadata,
        )
        subset = trace[100:200]
        assert subset.metadata == metadata

    def test_getitem_slice_with_step_returns_eager_trace(self, large_npy_file: Path) -> None:
        """Test that slicing with step returns WaveformTrace."""
        trace = LazyWaveformTrace(
            file_path=large_npy_file,
            sample_rate=1e6,
            length=1000,
        )
        subset = trace[0:10:2]  # Every other sample
        assert isinstance(subset, WaveformTrace)

    def test_getitem_slice_with_step_correct_data(self, large_raw_file: Path) -> None:
        """Test that slicing with step returns correct data."""
        trace = LazyWaveformTrace(
            file_path=large_raw_file,
            sample_rate=1e6,
            length=1000,
        )
        subset = trace[0:10:2]
        np.testing.assert_array_equal(subset.data, [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_getitem_slice_empty(self, large_npy_file: Path) -> None:
        """Test slicing with empty range."""
        trace = LazyWaveformTrace(
            file_path=large_npy_file,
            sample_rate=1e6,
            length=1000,
        )
        subset = trace[100:100]
        assert isinstance(subset, LazyWaveformTrace)
        assert subset.length == 0

    def test_getitem_invalid_type_raises_error(self, temp_npy_file: Path) -> None:
        """Test that invalid index type raises TypeError."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        with pytest.raises(TypeError, match="Indices must be int or slice"):
            _ = trace["invalid"]  # type: ignore[index]


# ==============================================================================
# LazyWaveformTrace Conversion Tests
# ==============================================================================


class TestLazyWaveformTraceConversion:
    """Test LazyWaveformTrace conversion methods."""

    def test_to_eager_returns_waveform_trace(self, temp_npy_file: Path) -> None:
        """Test that to_eager() returns WaveformTrace."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        eager = trace.to_eager()
        assert isinstance(eager, WaveformTrace)

    def test_to_eager_correct_data(self, temp_raw_file: Path) -> None:
        """Test that to_eager() returns correct data."""
        trace = LazyWaveformTrace(
            file_path=temp_raw_file,
            sample_rate=1e6,
            length=5,
        )
        eager = trace.to_eager()
        np.testing.assert_array_equal(eager.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_to_eager_preserves_sample_rate(self, temp_npy_file: Path) -> None:
        """Test that to_eager() preserves sample rate."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=2.5e9,
            length=5,
        )
        eager = trace.to_eager()
        assert eager.metadata.sample_rate == 2.5e9

    def test_to_eager_preserves_metadata(self, temp_npy_file: Path) -> None:
        """Test that to_eager() preserves metadata."""
        metadata = {"channel_name": "CH1", "source_file": "test.bin"}
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
            metadata=metadata,
        )
        eager = trace.to_eager()
        assert eager.metadata.channel_name == "CH1"
        assert eager.metadata.source_file == "test.bin"


# ==============================================================================
# LazyWaveformTrace Resource Management Tests
# ==============================================================================


class TestLazyWaveformTraceResourceManagement:
    """Test LazyWaveformTrace resource management."""

    def test_close_clears_memmap(self, temp_npy_file: Path) -> None:
        """Test that close() clears memmap."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        # Create memmap
        _ = trace[0]
        assert trace._memmap is not None

        # Close
        trace.close()
        assert trace._memmap is None

    def test_close_idempotent(self, temp_npy_file: Path) -> None:
        """Test that close() can be called multiple times."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        trace.close()
        trace.close()  # Should not raise

    def test_del_calls_close(self, temp_npy_file: Path) -> None:
        """Test that __del__() calls close()."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        # Create memmap
        _ = trace[0]
        assert trace._memmap is not None

        # Delete should call close
        trace.__del__()
        assert trace._memmap is None


# ==============================================================================
# LazyWaveformTrace String Representation Tests
# ==============================================================================


class TestLazyWaveformTraceStringRepresentation:
    """Test LazyWaveformTrace string methods."""

    def test_repr_not_loaded(self, temp_npy_file: Path) -> None:
        """Test __repr__() when data not loaded."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        repr_str = repr(trace)
        assert "test_trace.npy" in repr_str
        assert "sample_rate=1.00e+06" in repr_str
        assert "length=5" in repr_str
        assert "loaded=False" in repr_str

    def test_repr_loaded(self, temp_npy_file: Path) -> None:
        """Test __repr__() when data is loaded."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        _ = trace.data  # Trigger load
        repr_str = repr(trace)
        assert "loaded=True" in repr_str

    def test_len(self, temp_npy_file: Path) -> None:
        """Test __len__() method."""
        trace = LazyWaveformTrace(
            file_path=temp_npy_file,
            sample_rate=1e6,
            length=5,
        )
        assert len(trace) == 5


# ==============================================================================
# load_trace_lazy() Function Tests
# ==============================================================================


class TestLoadTraceLazy:
    """Test load_trace_lazy() function."""

    def test_load_npy_lazy(self, temp_npy_file: Path) -> None:
        """Test loading .npy file lazily."""
        trace = load_trace_lazy(temp_npy_file, sample_rate=1e6, lazy=True)
        assert isinstance(trace, LazyWaveformTrace)
        assert trace.length == 5
        assert trace.sample_rate == 1e6

    def test_load_npy_eager(self, temp_npy_file: Path) -> None:
        """Test loading .npy file eagerly."""
        trace = load_trace_lazy(temp_npy_file, sample_rate=1e6, lazy=False)
        assert isinstance(trace, WaveformTrace)
        np.testing.assert_array_equal(trace.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_load_npy_without_sample_rate_raises_error(self, temp_npy_file: Path) -> None:
        """Test that loading .npy without sample_rate raises error."""
        with pytest.raises(LoaderError, match="sample_rate is required"):
            load_trace_lazy(temp_npy_file, lazy=True)

    def test_load_npy_detects_shape(self, tmp_path: Path) -> None:
        """Test that .npy loader detects array shape."""
        # Create 2D array (should fail)
        file_path = tmp_path / "2d_array.npy"
        data = np.array([[1, 2], [3, 4]])
        np.save(file_path, data)

        with pytest.raises(LoaderError, match="Expected 1D array"):
            load_trace_lazy(file_path, sample_rate=1e6, lazy=True)

    def test_load_raw_lazy(self, temp_raw_file: Path) -> None:
        """Test loading raw binary file lazily."""
        trace = load_trace_lazy(temp_raw_file, sample_rate=1e6, lazy=True)
        assert isinstance(trace, LazyWaveformTrace)
        assert trace.length == 5
        assert trace.sample_rate == 1e6

    def test_load_raw_eager(self, temp_raw_file: Path) -> None:
        """Test loading raw binary file eagerly."""
        trace = load_trace_lazy(temp_raw_file, sample_rate=1e6, lazy=False)
        assert isinstance(trace, WaveformTrace)
        np.testing.assert_array_equal(trace.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_load_raw_with_dtype(self, tmp_path: Path) -> None:
        """Test loading raw binary with custom dtype."""
        file_path = tmp_path / "test_int.bin"
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        data.tofile(file_path)

        trace = load_trace_lazy(file_path, sample_rate=1e6, lazy=True, dtype=np.int32)
        assert isinstance(trace, LazyWaveformTrace)
        assert trace.length == 5

    def test_load_raw_with_offset(self, temp_raw_file: Path) -> None:
        """Test loading raw binary with offset."""
        trace = load_trace_lazy(temp_raw_file, sample_rate=1e6, lazy=True, offset=16)
        assert isinstance(trace, LazyWaveformTrace)
        assert trace.length == 3  # Skipped 2 samples (16 bytes)

    def test_load_raw_without_sample_rate_raises_error(self, temp_raw_file: Path) -> None:
        """Test that loading raw file without sample_rate raises error."""
        with pytest.raises(LoaderError, match="sample_rate is required"):
            load_trace_lazy(temp_raw_file, lazy=True)

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises error."""
        with pytest.raises(LoaderError, match="File not found"):
            load_trace_lazy(tmp_path / "nonexistent.npy", sample_rate=1e6)

    def test_load_string_path(self, temp_npy_file: Path) -> None:
        """Test loading with string path."""
        trace = load_trace_lazy(str(temp_npy_file), sample_rate=1e6, lazy=True)
        assert isinstance(trace, LazyWaveformTrace)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestLoadersLazyIntegration:
    """Integration tests combining multiple features."""

    def test_lazy_workflow(self, large_npy_file: Path) -> None:
        """Test complete lazy loading workflow."""
        # Load lazily
        trace = load_trace_lazy(large_npy_file, sample_rate=1e9, lazy=True)
        assert isinstance(trace, LazyWaveformTrace)
        assert trace._data is None

        # Access metadata without loading data
        assert trace.length == 1000
        assert trace.duration == pytest.approx(1e-6)

        # Slice without loading full data
        subset = trace[100:200]
        assert isinstance(subset, LazyWaveformTrace)
        assert subset.length == 100

        # Load subset data
        subset_data = subset.data
        assert len(subset_data) == 100
        assert subset_data[0] == 100.0

        # Original trace still not fully loaded
        assert trace._data is None

    def test_eager_workflow(self, temp_npy_file: Path) -> None:
        """Test complete eager loading workflow."""
        # Load eagerly
        trace = load_trace_lazy(temp_npy_file, sample_rate=1e6, lazy=False)
        assert isinstance(trace, WaveformTrace)

        # Data already loaded
        np.testing.assert_array_equal(trace.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_lazy_to_eager_conversion(self, temp_npy_file: Path) -> None:
        """Test converting lazy trace to eager."""
        lazy_trace = load_trace_lazy(temp_npy_file, sample_rate=1e6, lazy=True)
        assert isinstance(lazy_trace, LazyWaveformTrace)

        eager_trace = lazy_trace.to_eager()
        assert isinstance(eager_trace, WaveformTrace)
        assert eager_trace.metadata.sample_rate == lazy_trace.sample_rate

    def test_slicing_chain(self, large_npy_file: Path) -> None:
        """Test chaining multiple slices."""
        trace = load_trace_lazy(large_npy_file, sample_rate=1e6, lazy=True)

        # Chain slices
        subset1 = trace[200:800]
        subset2 = subset1[100:200]

        # Load final subset
        data = subset2.data
        assert len(data) == 100
        # Original indices: 200 + 100 = 300 to 200 + 200 = 400
        assert data[0] == 300.0

    def test_resource_cleanup(self, temp_npy_file: Path) -> None:
        """Test that resources are cleaned up properly."""
        trace = load_trace_lazy(temp_npy_file, sample_rate=1e6, lazy=True)

        # Create memmap
        _ = trace[0]
        assert trace._memmap is not None

        # Close should cleanup
        trace.close()
        assert trace._memmap is None

        # Can still access data (creates new memmap)
        _ = trace[0]
        assert trace._memmap is not None
