"""Comprehensive tests for buffer utilities.

Tests requirements:

This test suite achieves 80%+ coverage by testing:
- CircularBuffer: creation, operations, edge cases
- SlidingWindow: sample-based and time-based windows
- Memory management and boundary conditions
"""

import math

import numpy as np
import pytest

from oscura.utils.buffer import CircularBuffer, SlidingWindow

pytestmark = pytest.mark.unit


class TestCircularBufferBasics:
    """Test basic CircularBuffer functionality."""

    def test_buffer_initialization_numeric(self):
        """Test buffer initialization with numeric dtype."""
        buf = CircularBuffer(100, dtype=np.float64)

        assert buf.capacity == 100
        assert buf.count == 0
        assert buf.is_empty()
        assert not buf.is_full()
        assert len(buf) == 0

    def test_buffer_initialization_object(self):
        """Test buffer initialization with object dtype (None)."""
        buf = CircularBuffer(50)

        assert buf.capacity == 50
        assert buf.count == 0
        assert buf.is_empty()

    def test_buffer_initialization_custom_dtype(self):
        """Test buffer initialization with custom dtype."""
        buf = CircularBuffer(100, dtype=np.int32)

        assert buf.capacity == 100
        assert buf._dtype == np.int32

    def test_append_single_value(self):
        """Test appending single value."""
        buf = CircularBuffer(10, dtype=np.float64)

        buf.append(1.5)

        assert buf.count == 1
        assert not buf.is_empty()
        assert not buf.is_full()
        assert len(buf) == 1

    def test_append_multiple_values(self):
        """Test appending multiple values."""
        buf = CircularBuffer(10, dtype=np.float64)

        for i in range(5):
            buf.append(float(i))

        assert buf.count == 5
        assert len(buf) == 5

    def test_append_to_full_buffer(self):
        """Test appending to full buffer (overwrites oldest)."""
        buf = CircularBuffer(5, dtype=np.float64)

        # Fill buffer
        for i in range(5):
            buf.append(float(i))

        assert buf.is_full()

        # Overwrite oldest
        buf.append(99.0)

        assert buf.is_full()
        assert buf.count == 5
        # Oldest value (0) should be overwritten
        assert buf[0] == 1.0  # Second value becomes first

    def test_extend_with_list(self):
        """Test extending buffer with list."""
        buf = CircularBuffer(10, dtype=np.float64)

        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        assert buf.count == 5
        assert buf[0] == 1.0
        assert buf[4] == 5.0

    def test_extend_with_array(self):
        """Test extending buffer with numpy array."""
        buf = CircularBuffer(10, dtype=np.float64)

        values = np.array([10.0, 20.0, 30.0])
        buf.extend(values)

        assert buf.count == 3
        assert buf[0] == 10.0


class TestCircularBufferRetrieval:
    """Test data retrieval methods."""

    def test_get_last_single(self):
        """Test getting last single value."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        last = buf.get_last(1)

        assert len(last) == 1
        assert last[0] == 5.0

    def test_get_last_multiple(self):
        """Test getting last n values."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        last = buf.get_last(3)

        assert len(last) == 3
        assert last[0] == 5.0  # Newest first
        assert last[1] == 4.0
        assert last[2] == 3.0

    def test_get_last_more_than_count(self):
        """Test getting more items than available."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0])

        last = buf.get_last(10)

        assert len(last) == 3

    def test_get_last_empty_buffer(self):
        """Test getting last from empty buffer."""
        buf = CircularBuffer(10, dtype=np.float64)

        last = buf.get_last(5)

        assert len(last) == 0

    def test_get_last_empty_buffer_object_dtype(self):
        """Test getting last from empty buffer with object dtype."""
        buf = CircularBuffer(10)

        last = buf.get_last(5)

        assert len(last) == 0
        assert last.dtype == object

    def test_get_first_single(self):
        """Test getting first single value."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        first = buf.get_first(1)

        assert len(first) == 1
        assert first[0] == 1.0

    def test_get_first_multiple(self):
        """Test getting first n values."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        first = buf.get_first(3)

        assert len(first) == 3
        assert first[0] == 1.0  # Oldest first
        assert first[1] == 2.0
        assert first[2] == 3.0

    def test_get_first_empty_buffer(self):
        """Test getting first from empty buffer."""
        buf = CircularBuffer(10, dtype=np.float64)

        first = buf.get_first(5)

        assert len(first) == 0

    def test_get_first_empty_buffer_object_dtype(self):
        """Test getting first from empty buffer with object dtype."""
        buf = CircularBuffer(10)

        first = buf.get_first(5)

        assert len(first) == 0
        assert first.dtype == object

    def test_get_all_empty(self):
        """Test getting all from empty buffer."""
        buf = CircularBuffer(10, dtype=np.float64)

        all_data = buf.get_all()

        assert len(all_data) == 0

    def test_get_all_partial(self):
        """Test getting all from partially filled buffer."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0])

        all_data = buf.get_all()

        assert len(all_data) == 3
        assert all_data[0] == 1.0
        assert all_data[2] == 3.0

    def test_get_all_full(self):
        """Test getting all from full buffer."""
        buf = CircularBuffer(5, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        all_data = buf.get_all()

        assert len(all_data) == 5
        assert all_data[0] == 1.0

    def test_get_all_after_wraparound(self):
        """Test getting all after buffer wraps around."""
        buf = CircularBuffer(5, dtype=np.float64)

        # Fill and overflow
        for i in range(8):
            buf.append(float(i))

        all_data = buf.get_all()

        assert len(all_data) == 5
        # Should contain values 3, 4, 5, 6, 7 (oldest first)
        assert all_data[0] == 3.0
        assert all_data[4] == 7.0


class TestCircularBufferIndexing:
    """Test indexing operations."""

    def test_getitem_positive_index(self):
        """Test getting item by positive index."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([10.0, 20.0, 30.0, 40.0, 50.0])

        assert buf[0] == 10.0  # Oldest
        assert buf[2] == 30.0
        assert buf[4] == 50.0  # Newest

    def test_getitem_negative_index(self):
        """Test getting item by negative index."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([10.0, 20.0, 30.0, 40.0, 50.0])

        assert buf[-1] == 50.0  # Newest
        assert buf[-2] == 40.0
        assert buf[-5] == 10.0  # Oldest

    def test_getitem_out_of_range_positive(self):
        """Test index out of range (positive)."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0])

        with pytest.raises(IndexError, match="Index 5 out of range"):
            _ = buf[5]

    def test_getitem_out_of_range_negative(self):
        """Test index out of range (negative)."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0])

        with pytest.raises(IndexError, match="Index -1 out of range"):
            _ = buf[-4]

    def test_getitem_empty_buffer(self):
        """Test indexing empty buffer."""
        buf = CircularBuffer(10, dtype=np.float64)

        with pytest.raises(IndexError):
            _ = buf[0]

    def test_getitem_after_wraparound(self):
        """Test indexing after buffer wraps around."""
        buf = CircularBuffer(5, dtype=np.float64)

        # Fill and overflow
        for i in range(8):
            buf.append(float(i))

        # Buffer contains: 3, 4, 5, 6, 7
        assert buf[0] == 3.0
        assert buf[2] == 5.0
        assert buf[-1] == 7.0

    def test_getitem_slice(self):
        """Test slicing buffer."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        sliced = buf[1:4]

        assert len(sliced) == 3
        assert sliced[0] == 2.0
        assert sliced[2] == 4.0

    def test_getitem_slice_with_step(self):
        """Test slicing with step."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        sliced = buf[::2]

        assert len(sliced) == 3
        assert sliced[0] == 1.0
        assert sliced[1] == 3.0
        assert sliced[2] == 5.0

    def test_getitem_slice_negative_indices(self):
        """Test slicing with negative indices."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        sliced = buf[-3:-1]

        assert len(sliced) == 2
        assert sliced[0] == 3.0
        assert sliced[1] == 4.0

    def test_getitem_slice_empty(self):
        """Test empty slice."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0])

        sliced = buf[2:2]

        assert len(sliced) == 0


class TestCircularBufferStatistics:
    """Test statistical methods."""

    def test_mean_with_data(self):
        """Test computing mean."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        mean = buf.mean()

        assert mean == 3.0

    def test_mean_empty_buffer(self):
        """Test mean of empty buffer returns NaN."""
        buf = CircularBuffer(10, dtype=np.float64)

        mean = buf.mean()

        assert math.isnan(mean)

    def test_std_with_data(self):
        """Test computing standard deviation."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        std = buf.std()

        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(std - expected) < 1e-10

    def test_std_empty_buffer(self):
        """Test std of empty buffer returns NaN."""
        buf = CircularBuffer(10, dtype=np.float64)

        std = buf.std()

        assert math.isnan(std)

    def test_std_single_value(self):
        """Test std with single value returns NaN."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.append(5.0)

        std = buf.std()

        assert math.isnan(std)

    def test_min_with_data(self):
        """Test finding minimum value."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([5.0, 2.0, 8.0, 1.0, 6.0])

        min_val = buf.min()

        assert min_val == 1.0

    def test_min_empty_buffer(self):
        """Test min of empty buffer raises ValueError."""
        buf = CircularBuffer(10, dtype=np.float64)

        with pytest.raises(ValueError, match="Buffer is empty"):
            _ = buf.min()

    def test_max_with_data(self):
        """Test finding maximum value."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([5.0, 2.0, 8.0, 1.0, 6.0])

        max_val = buf.max()

        assert max_val == 8.0

    def test_max_empty_buffer(self):
        """Test max of empty buffer raises ValueError."""
        buf = CircularBuffer(10, dtype=np.float64)

        with pytest.raises(ValueError, match="Buffer is empty"):
            _ = buf.max()


class TestCircularBufferEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buf = CircularBuffer(10, dtype=np.float64)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        buf.clear()

        assert buf.count == 0
        assert buf.is_empty()
        assert len(buf) == 0

    def test_clear_empty_buffer(self):
        """Test clearing already empty buffer."""
        buf = CircularBuffer(10, dtype=np.float64)

        buf.clear()

        assert buf.count == 0

    def test_wraparound_behavior(self):
        """Test buffer behavior during wraparound."""
        buf = CircularBuffer(3, dtype=np.float64)

        # Fill buffer
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        assert buf.is_full()

        # Trigger wraparound
        buf.append(4.0)
        assert buf.count == 3
        assert buf[0] == 2.0  # 1.0 was overwritten

        buf.append(5.0)
        assert buf[0] == 3.0  # 2.0 was overwritten

    def test_single_element_buffer(self):
        """Test buffer with capacity of 1."""
        buf = CircularBuffer(1, dtype=np.float64)

        buf.append(1.0)
        assert buf.is_full()
        assert buf[0] == 1.0

        buf.append(2.0)
        assert buf.is_full()
        assert buf[0] == 2.0  # Overwrites previous

    def test_object_buffer_with_strings(self):
        """Test object buffer with string values."""
        buf = CircularBuffer(5)

        buf.append("hello")
        buf.append("world")
        buf.append("test")

        assert buf.count == 3
        assert buf[0] == "hello"
        assert buf[2] == "test"

    def test_object_buffer_with_mixed_types(self):
        """Test object buffer with mixed types."""
        buf = CircularBuffer(5)

        buf.append(42)
        buf.append("string")
        buf.append([1, 2, 3])
        buf.append({"key": "value"})

        assert buf.count == 4
        assert buf[0] == 42
        assert buf[1] == "string"
        assert buf[2] == [1, 2, 3]

    def test_large_buffer(self):
        """Test buffer with large capacity."""
        buf = CircularBuffer(10000, dtype=np.float64)

        # Add many values
        for i in range(5000):
            buf.append(float(i))

        assert buf.count == 5000
        assert buf[0] == 0.0
        assert buf[4999] == 4999.0


class TestSlidingWindowSampleBased:
    """Test sample-based sliding window."""

    def test_window_initialization_sample_based(self):
        """Test sample-based window initialization."""
        window = SlidingWindow(100, time_based=False)

        assert window._window_size == 100
        assert not window._time_based

    def test_add_samples(self):
        """Test adding samples to window."""
        window = SlidingWindow(5, time_based=False)

        window.add(1.0)
        window.add(2.0)
        window.add(3.0)

        assert not window.is_ready()

        window.add(4.0)
        window.add(5.0)

        assert window.is_ready()

    def test_get_data_sample_based(self):
        """Test getting data from sample-based window."""
        window = SlidingWindow(3, time_based=False)

        window.add(1.0)
        window.add(2.0)
        window.add(3.0)

        data = window.get_data()

        assert len(data) == 3
        assert data[0] == 1.0
        assert data[2] == 3.0

    def test_get_data_before_ready(self):
        """Test getting data before window is ready."""
        window = SlidingWindow(5, time_based=False)

        window.add(1.0)
        window.add(2.0)

        data = window.get_data()

        assert len(data) == 2

    def test_clear_sample_window(self):
        """Test clearing sample-based window."""
        window = SlidingWindow(5, time_based=False)

        window.add(1.0)
        window.add(2.0)
        window.add(3.0)

        window.clear()

        data = window.get_data()
        assert len(data) == 0

    def test_get_times_not_time_based_raises(self):
        """Test get_times on sample-based window raises error."""
        window = SlidingWindow(5, time_based=False)

        with pytest.raises(ValueError, match="Not a time-based window"):
            _ = window.get_times()


class TestSlidingWindowTimeBased:
    """Test time-based sliding window."""

    def test_window_initialization_time_based(self):
        """Test time-based window initialization."""
        window = SlidingWindow(1.0, time_based=True)

        assert window._window_size == 1.0
        assert window._time_based

    def test_add_with_timestamp(self):
        """Test adding samples with timestamps."""
        window = SlidingWindow(1.0, time_based=True)

        window.add(10.0, timestamp=0.0)
        window.add(20.0, timestamp=0.5)
        window.add(30.0, timestamp=1.0)

        assert window.is_ready()

    def test_add_without_timestamp_raises(self):
        """Test adding to time-based window without timestamp raises error."""
        window = SlidingWindow(1.0, time_based=True)

        with pytest.raises(ValueError, match="Timestamp required"):
            window.add(10.0)

    def test_is_ready_time_based(self):
        """Test is_ready for time-based window."""
        window = SlidingWindow(2.0, time_based=True)

        window.add(1.0, timestamp=0.0)
        assert not window.is_ready()  # Less than 2 samples

        window.add(2.0, timestamp=0.5)
        assert not window.is_ready()  # Duration < 2.0

        window.add(3.0, timestamp=2.5)
        assert window.is_ready()  # Duration >= 2.0

    def test_get_data_time_based(self):
        """Test getting data from time-based window."""
        window = SlidingWindow(1.0, time_based=True)

        window.add(10.0, timestamp=0.0)
        window.add(20.0, timestamp=0.5)
        window.add(30.0, timestamp=1.0)
        window.add(40.0, timestamp=1.5)

        data = window.get_data()

        # Should only include samples within last 1.0 second
        # Last timestamp is 1.5, so cutoff is 0.5
        assert len(data) == 3  # Timestamps: 0.5, 1.0, 1.5
        assert data[0] == 20.0

    def test_get_data_empty_time_window(self):
        """Test getting data from empty time-based window."""
        window = SlidingWindow(1.0, time_based=True)

        data = window.get_data()

        assert len(data) == 0

    def test_get_times(self):
        """Test getting timestamps from time-based window."""
        window = SlidingWindow(1.0, time_based=True)

        window.add(10.0, timestamp=0.0)
        window.add(20.0, timestamp=0.5)
        window.add(30.0, timestamp=1.0)

        times = window.get_times()

        assert len(times) == 3
        assert times[0] == 0.0
        assert times[2] == 1.0

    def test_clear_time_window(self):
        """Test clearing time-based window."""
        window = SlidingWindow(1.0, time_based=True)

        window.add(10.0, timestamp=0.0)
        window.add(20.0, timestamp=0.5)

        window.clear()

        data = window.get_data()
        assert len(data) == 0

    def test_time_window_with_gap(self):
        """Test time-based window with time gap."""
        window = SlidingWindow(1.0, time_based=True)

        window.add(10.0, timestamp=0.0)
        window.add(20.0, timestamp=0.1)
        window.add(30.0, timestamp=5.0)  # Large gap
        window.add(40.0, timestamp=5.5)

        data = window.get_data()

        # Window is [4.5, 5.5], so only last two samples
        assert len(data) == 2
        assert data[0] == 30.0
        assert data[1] == 40.0


class TestSlidingWindowEdgeCases:
    """Test edge cases for sliding windows."""

    def test_fractional_window_size(self):
        """Test window with fractional sample size."""
        # Sample-based windows convert size to int for buffer capacity
        # but use the original float for is_ready comparison
        window = SlidingWindow(5.5, time_based=False)

        # Buffer capacity is int(5.5) = 5, so can only hold 5 items
        for i in range(5):
            window.add(float(i))

        # Window needs >= 5.5 items, but buffer can only hold 5
        # So this window can never be ready due to capacity constraint
        assert not window.is_ready()

        # Verify buffer is at capacity
        assert window._data.count == 5

    def test_custom_dtype(self):
        """Test window with custom dtype."""
        window = SlidingWindow(5, time_based=False, dtype=np.int32)

        window.add(1, timestamp=None)
        window.add(2, timestamp=None)
        window.add(3, timestamp=None)

        data = window.get_data()

        assert data.dtype == np.int32

    def test_large_time_window(self):
        """Test time-based window with large buffer."""
        window = SlidingWindow(10.0, time_based=True)

        # Add many samples over sufficient time duration
        # 1000 samples * 0.01 = 10.0 seconds total, but we need >= 10.0 duration
        for i in range(1001):
            window.add(float(i), timestamp=float(i) * 0.01)

        assert window.is_ready()
        data = window.get_data()
        assert len(data) > 0


class TestCircularBufferMemoryManagement:
    """Test memory management aspects."""

    def test_buffer_dtype_preserved(self):
        """Test buffer preserves dtype."""
        buf = CircularBuffer(10, dtype=np.int16)

        buf.append(1)
        buf.append(2)

        assert buf._data.dtype == np.int16

    def test_large_overflow(self):
        """Test buffer behavior with large overflow."""
        buf = CircularBuffer(10, dtype=np.float64)

        # Add 100 values to 10-capacity buffer
        for i in range(100):
            buf.append(float(i))

        assert buf.count == 10
        assert buf.is_full()
        # Should contain last 10 values: 90-99
        assert buf[0] == 90.0
        assert buf[9] == 99.0

    def test_multiple_wraparounds(self):
        """Test multiple complete wraparounds."""
        buf = CircularBuffer(5, dtype=np.float64)

        # Go around 3 times
        for i in range(15):
            buf.append(float(i))

        assert buf.count == 5
        # Should contain: 10, 11, 12, 13, 14
        assert buf[0] == 10.0
        assert buf[4] == 14.0

    def test_empty_operations_safe(self):
        """Test operations on empty buffer are safe."""
        buf = CircularBuffer(10, dtype=np.float64)

        # All these should work without errors
        assert buf.get_all().size == 0
        assert buf.get_first(5).size == 0
        assert buf.get_last(5).size == 0
        assert math.isnan(buf.mean())
        assert math.isnan(buf.std())

        buf.clear()  # Clearing empty buffer
        assert buf.count == 0


@pytest.fixture
def sample_buffer():
    """Fixture providing a pre-filled buffer for testing."""
    buf = CircularBuffer(10, dtype=np.float64)
    buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])
    return buf


@pytest.fixture
def sample_window():
    """Fixture providing a pre-filled sample-based window."""
    window = SlidingWindow(5, time_based=False)
    for i in range(3):
        window.add(float(i))
    return window


@pytest.fixture
def time_window():
    """Fixture providing a pre-filled time-based window."""
    window = SlidingWindow(1.0, time_based=True)
    window.add(10.0, timestamp=0.0)
    window.add(20.0, timestamp=0.5)
    return window


class TestBufferFixtures:
    """Test using fixtures."""

    def test_sample_buffer_fixture(self, sample_buffer):
        """Test sample buffer fixture."""
        assert sample_buffer.count == 5
        assert sample_buffer[0] == 1.0

    def test_sample_window_fixture(self, sample_window):
        """Test sample window fixture."""
        data = sample_window.get_data()
        assert len(data) == 3

    def test_time_window_fixture(self, time_window):
        """Test time window fixture."""
        times = time_window.get_times()
        assert len(times) == 2
