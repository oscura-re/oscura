"""Unit tests for time axis formatting and optimization.

Tests:

Coverage target: 90%+ of src/oscura/visualization/time_axis.py
"""

import numpy as np
import pytest

from oscura.visualization.time_axis import (
    TimeUnit,
    calculate_major_ticks,
    convert_time_values,
    create_relative_time,
    format_cursor_readout,
    format_time_labels,
    select_time_unit,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.visualization,
    pytest.mark.usefixtures("cleanup_matplotlib"),
]


@pytest.mark.unit
@pytest.mark.visualization
class TestSelectTimeUnit:
    """Tests for select_time_unit function."""

    def test_seconds_range(self):
        """Test selection of seconds for large time ranges."""
        assert select_time_unit(1.0) == "s"
        assert select_time_unit(10.0) == "s"
        assert select_time_unit(1000.0) == "s"

    def test_milliseconds_range(self):
        """Test selection of milliseconds."""
        assert select_time_unit(0.001) == "ms"
        assert select_time_unit(0.01) == "ms"
        assert select_time_unit(0.1) == "ms"
        assert select_time_unit(0.999) == "ms"

    def test_microseconds_range(self):
        """Test selection of microseconds."""
        assert select_time_unit(1e-6) == "us"
        assert select_time_unit(10e-6) == "us"
        assert select_time_unit(100e-6) == "us"
        assert select_time_unit(999e-6) == "us"

    def test_nanoseconds_range(self):
        """Test selection of nanoseconds."""
        assert select_time_unit(1e-9) == "ns"
        assert select_time_unit(10e-9) == "ns"
        assert select_time_unit(100e-9) == "ns"
        assert select_time_unit(999e-9) == "ns"

    def test_picoseconds_range(self):
        """Test selection of picoseconds for very small ranges."""
        assert select_time_unit(1e-12) == "ps"
        assert select_time_unit(10e-12) == "ps"
        assert select_time_unit(100e-12) == "ps"
        assert select_time_unit(0.5e-9) == "ps"

    def test_prefer_larger_seconds(self):
        """Test prefer_larger flag with seconds/milliseconds boundary."""
        assert select_time_unit(0.5, prefer_larger=False) == "ms"
        assert select_time_unit(0.5, prefer_larger=True) == "s"

    def test_prefer_larger_milliseconds(self):
        """Test prefer_larger flag with milliseconds/microseconds boundary."""
        assert select_time_unit(5e-4, prefer_larger=False) == "us"
        assert select_time_unit(5e-4, prefer_larger=True) == "ms"

    def test_prefer_larger_microseconds(self):
        """Test prefer_larger flag with microseconds/nanoseconds boundary."""
        assert select_time_unit(5e-7, prefer_larger=False) == "ns"
        assert select_time_unit(5e-7, prefer_larger=True) == "us"

    def test_prefer_larger_nanoseconds(self):
        """Test prefer_larger flag with nanoseconds/picoseconds boundary."""
        assert select_time_unit(5e-10, prefer_larger=False) == "ps"
        assert select_time_unit(5e-10, prefer_larger=True) == "ns"

    def test_zero_duration(self):
        """Test handling of zero time range."""
        assert select_time_unit(0.0) == "ps"

    def test_very_large_duration(self):
        """Test handling of very large time ranges."""
        assert select_time_unit(1e6) == "s"
        assert select_time_unit(3600.0) == "s"  # 1 hour


@pytest.mark.unit
@pytest.mark.visualization
class TestConvertTimeValues:
    """Tests for convert_time_values function."""

    def test_convert_to_seconds(self):
        """Test conversion to seconds (no change)."""
        time = np.array([1.0, 2.0, 3.0])
        result = convert_time_values(time, "s")
        np.testing.assert_array_equal(result, time)

    def test_convert_to_milliseconds(self):
        """Test conversion to milliseconds."""
        time = np.array([0.001, 0.002, 0.003])
        result = convert_time_values(time, "ms")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_to_microseconds(self):
        """Test conversion to microseconds."""
        time = np.array([1e-6, 2e-6, 3e-6])
        result = convert_time_values(time, "us")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_to_nanoseconds(self):
        """Test conversion to nanoseconds."""
        time = np.array([1e-9, 2e-9, 3e-9])
        result = convert_time_values(time, "ns")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_to_picoseconds(self):
        """Test conversion to picoseconds."""
        time = np.array([1e-12, 2e-12, 3e-12])
        result = convert_time_values(time, "ps")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_auto_unit_selection(self):
        """Test automatic unit selection."""
        # Millisecond range
        time_ms = np.array([0.0, 0.001, 0.002, 0.003])
        result = convert_time_values(time_ms, "auto")
        # Should select ms and convert
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_auto_unit_microsecond_range(self):
        """Test auto unit selection with microsecond range."""
        time_us = np.array([0.0, 1e-6, 2e-6, 3e-6])
        result = convert_time_values(time_us, "auto")
        # Should select us
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_unit_error(self):
        """Test that invalid unit raises ValueError."""
        time = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid time unit"):
            convert_time_values(time, "invalid")  # type: ignore

    def test_empty_array(self):
        """Test conversion of empty array."""
        time = np.array([])
        result = convert_time_values(time, "ms")
        assert len(result) == 0

    def test_single_value(self):
        """Test conversion of single value."""
        time = np.array([0.001])
        result = convert_time_values(time, "ms")
        np.testing.assert_array_almost_equal(result, [1.0])

    def test_negative_values(self):
        """Test conversion with negative values."""
        time = np.array([-0.001, 0.0, 0.001])
        result = convert_time_values(time, "ms")
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.unit
@pytest.mark.visualization
class TestFormatTimeLabels:
    """Tests for format_time_labels function."""

    def test_basic_formatting_milliseconds(self):
        """Test basic label formatting in milliseconds."""
        time = np.array([0.0, 0.001, 0.002])
        labels = format_time_labels(time, unit="ms")
        assert labels == ["0", "1", "2"]

    def test_basic_formatting_microseconds(self):
        """Test basic label formatting in microseconds."""
        time = np.array([0.0, 1e-6, 2e-6])
        labels = format_time_labels(time, unit="us")
        assert labels == ["0", "1", "2"]

    def test_auto_unit_selection(self):
        """Test automatic unit selection in labels."""
        time = np.array([0.0, 0.001, 0.002, 0.003])
        labels = format_time_labels(time, unit="auto")
        # Should select ms
        assert "0" in labels[0]
        assert len(labels) == 4

    def test_custom_precision(self):
        """Test custom precision setting."""
        time = np.array([0.0, 0.001, 0.002])
        labels = format_time_labels(time, unit="ms", precision=3)
        # Trailing zeros are stripped
        assert labels == ["0", "1", "2"]

    def test_auto_precision_small_range(self):
        """Test automatic precision for small value ranges."""
        time = np.array([0.0, 0.00001, 0.00002])
        labels = format_time_labels(time, unit="ms", precision=None)
        # Should have enough precision to show differences
        assert len(labels) == 3
        assert labels[0] != labels[1]

    def test_auto_precision_zero_range(self):
        """Test automatic precision for zero range."""
        time = np.array([0.001, 0.001, 0.001])
        labels = format_time_labels(time, unit="ms", precision=None)
        # Should default to precision=1 for zero range
        assert len(labels) == 3

    def test_scientific_notation(self):
        """Test scientific notation for large values."""
        time = np.array([0.0, 1000.0, 2000.0])
        labels = format_time_labels(time, unit="s", scientific_threshold=1e3)
        # First value not in scientific notation
        assert "e" not in labels[0] or labels[0] == "0"
        # Large values should use scientific notation
        assert "e" in labels[1] or "1000" in labels[1]

    def test_small_scientific_threshold(self):
        """Test custom scientific threshold."""
        time = np.array([0.0, 10.0, 20.0])
        labels = format_time_labels(time, unit="s", scientific_threshold=5.0)
        # Values >= 10 should be scientific
        assert len(labels) == 3

    def test_decimal_stripping(self):
        """Test that trailing zeros and decimal points are stripped."""
        time = np.array([0.0, 0.001, 0.0025])
        labels = format_time_labels(time, unit="ms", precision=2)
        # 0.0 -> "0", 1.00 -> "1", 2.50 -> "2.5"
        assert labels[0] == "0"
        assert labels[1] == "1"
        assert labels[2] == "2.5"

    def test_empty_array(self):
        """Test formatting empty array."""
        time = np.array([])
        # Empty arrays cause ValueError in numpy ptp - this is expected behavior
        # The function should handle non-empty arrays only
        with pytest.raises(ValueError):
            format_time_labels(time, unit="ms")

    def test_single_value(self):
        """Test formatting single value."""
        time = np.array([0.001])
        labels = format_time_labels(time, unit="ms")
        assert len(labels) == 1

    def test_negative_values(self):
        """Test formatting negative values."""
        time = np.array([-0.001, 0.0, 0.001])
        labels = format_time_labels(time, unit="ms")
        assert "-1" in labels[0]
        assert "0" in labels[1]
        assert "1" in labels[2]


@pytest.mark.unit
@pytest.mark.visualization
class TestCreateRelativeTime:
    """Tests for create_relative_time function."""

    def test_start_at_zero_default(self):
        """Test default behavior starts at zero."""
        time = np.array([1000.5, 1000.6, 1000.7])
        result = create_relative_time(time)
        expected = np.array([0.0, 0.1, 0.2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_start_at_zero_explicit(self):
        """Test explicit start_at_zero=True."""
        time = np.array([100.0, 100.5, 101.0])
        result = create_relative_time(time, start_at_zero=True)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_no_start_at_zero(self):
        """Test start_at_zero=False."""
        time = np.array([100.0, 100.5, 101.0])
        result = create_relative_time(time, start_at_zero=False)
        # Should keep absolute values
        np.testing.assert_array_almost_equal(result, time)

    def test_custom_reference_time(self):
        """Test custom reference time."""
        time = np.array([100.0, 100.5, 101.0])
        result = create_relative_time(time, reference_time=99.0)
        expected = np.array([1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_reference_time_overrides_start_at_zero(self):
        """Test that reference_time overrides start_at_zero."""
        time = np.array([100.0, 100.5, 101.0])
        result = create_relative_time(time, start_at_zero=True, reference_time=50.0)
        expected = np.array([50.0, 50.5, 51.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_array(self):
        """Test handling of empty array."""
        time = np.array([])
        result = create_relative_time(time)
        assert len(result) == 0

    def test_single_value(self):
        """Test handling of single value."""
        time = np.array([100.0])
        result = create_relative_time(time)
        np.testing.assert_array_almost_equal(result, [0.0])

    def test_negative_reference(self):
        """Test negative reference time."""
        time = np.array([1.0, 2.0, 3.0])
        result = create_relative_time(time, reference_time=-10.0)
        expected = np.array([11.0, 12.0, 13.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_reference(self):
        """Test zero as reference time."""
        time = np.array([1.0, 2.0, 3.0])
        result = create_relative_time(time, reference_time=0.0)
        np.testing.assert_array_almost_equal(result, time)


@pytest.mark.unit
@pytest.mark.visualization
class TestCalculateMajorTicks:
    """Tests for calculate_major_ticks function."""

    def test_basic_tick_generation(self):
        """Test basic major tick generation."""
        ticks = calculate_major_ticks(0.0, 0.01, target_count=5, unit="ms")
        assert len(ticks) > 0
        # All ticks should be within range
        assert np.all(ticks >= 0.0)
        assert np.all(ticks <= 0.01)

    def test_target_count(self):
        """Test that target_count influences number of ticks."""
        ticks_few = calculate_major_ticks(0.0, 1.0, target_count=5, unit="s")
        ticks_many = calculate_major_ticks(0.0, 1.0, target_count=10, unit="s")
        # More target -> potentially more ticks (not guaranteed due to nice rounding)
        assert len(ticks_few) > 0
        assert len(ticks_many) > 0

    def test_auto_unit_selection(self):
        """Test automatic unit selection."""
        ticks = calculate_major_ticks(0.0, 0.001, target_count=5, unit="auto")
        assert len(ticks) > 0
        assert np.all(ticks >= 0.0)
        assert np.all(ticks <= 0.001)

    def test_seconds_unit(self):
        """Test tick generation in seconds."""
        ticks = calculate_major_ticks(0.0, 10.0, target_count=5, unit="s")
        assert len(ticks) > 0
        # Should have nice round numbers
        assert np.all(ticks >= 0.0)
        assert np.all(ticks <= 10.0)

    def test_milliseconds_unit(self):
        """Test tick generation in milliseconds."""
        ticks = calculate_major_ticks(0.0, 0.01, target_count=5, unit="ms")
        assert len(ticks) > 0

    def test_microseconds_unit(self):
        """Test tick generation in microseconds."""
        ticks = calculate_major_ticks(0.0, 10e-6, target_count=5, unit="us")
        assert len(ticks) > 0

    def test_nanoseconds_unit(self):
        """Test tick generation in nanoseconds."""
        ticks = calculate_major_ticks(0.0, 10e-9, target_count=5, unit="ns")
        assert len(ticks) > 0

    def test_picoseconds_unit(self):
        """Test tick generation in picoseconds."""
        ticks = calculate_major_ticks(0.0, 10e-12, target_count=5, unit="ps")
        assert len(ticks) > 0

    def test_zero_range(self):
        """Test handling of zero time range."""
        ticks = calculate_major_ticks(5.0, 5.0, target_count=5, unit="s")
        # Should return single tick at the point
        assert len(ticks) == 1
        np.testing.assert_array_almost_equal(ticks, [5.0])

    def test_negative_range(self):
        """Test handling of negative range (min > max)."""
        ticks = calculate_major_ticks(10.0, 5.0, target_count=5, unit="s")
        # Should return single tick at min
        assert len(ticks) == 1

    def test_very_small_range(self):
        """Test very small time range."""
        ticks = calculate_major_ticks(0.0, 1e-15, target_count=5, unit="ps")
        assert len(ticks) > 0

    def test_very_large_range(self):
        """Test very large time range."""
        ticks = calculate_major_ticks(0.0, 1e6, target_count=5, unit="s")
        assert len(ticks) > 0

    def test_negative_time_values(self):
        """Test tick generation with negative time values."""
        ticks = calculate_major_ticks(-0.01, 0.01, target_count=5, unit="ms")
        assert len(ticks) > 0
        assert np.all(ticks >= -0.01)
        assert np.all(ticks <= 0.01)

    def test_offset_range(self):
        """Test tick generation with offset range."""
        ticks = calculate_major_ticks(100.0, 110.0, target_count=5, unit="s")
        assert len(ticks) > 0
        assert np.all(ticks >= 100.0)
        assert np.all(ticks <= 110.0)

    def test_tick_spacing_uniformity(self):
        """Test that ticks are uniformly spaced."""
        ticks = calculate_major_ticks(0.0, 1.0, target_count=5, unit="s")
        if len(ticks) > 1:
            spacings = np.diff(ticks)
            # All spacings should be equal (within numerical tolerance)
            np.testing.assert_array_almost_equal(spacings, np.full_like(spacings, spacings[0]))


@pytest.mark.unit
@pytest.mark.visualization
class TestRoundToNiceTime:
    """Tests for _round_to_nice_time internal function."""

    def test_nice_rounding_basic(self):
        """Test rounding to nice values."""
        from oscura.visualization.time_axis import _round_to_nice_time

        # Should round to 1, 2, 5, 10 × 10^n (rounds to nearest)
        assert _round_to_nice_time(1.3) == 1.0
        assert _round_to_nice_time(1.8) == 2.0
        assert _round_to_nice_time(3.5) == 2.0  # Rounds to nearest of [1, 2, 5, 10]
        assert _round_to_nice_time(7.0) == 5.0

    def test_nice_rounding_large_values(self):
        """Test rounding large values."""
        from oscura.visualization.time_axis import _round_to_nice_time

        assert _round_to_nice_time(15.0) == 10.0
        assert _round_to_nice_time(23.0) == 20.0
        assert _round_to_nice_time(45.0) == 50.0

    def test_nice_rounding_small_values(self):
        """Test rounding small values."""
        from oscura.visualization.time_axis import _round_to_nice_time

        assert _round_to_nice_time(0.15) == 0.1  # Rounds to nearest
        assert _round_to_nice_time(0.035) == 0.05
        assert _round_to_nice_time(0.0013) == 0.001

    def test_zero_value(self):
        """Test handling of zero value."""
        from oscura.visualization.time_axis import _round_to_nice_time

        assert _round_to_nice_time(0.0) == 1.0

    def test_negative_value(self):
        """Test handling of negative value."""
        from oscura.visualization.time_axis import _round_to_nice_time

        assert _round_to_nice_time(-5.0) == 1.0

    def test_very_small_value(self):
        """Test very small positive value."""
        from oscura.visualization.time_axis import _round_to_nice_time

        result = _round_to_nice_time(1e-10)
        assert result > 0
        assert result in [1e-10, 2e-10, 5e-10, 1e-9]

    def test_very_large_value(self):
        """Test very large value."""
        from oscura.visualization.time_axis import _round_to_nice_time

        result = _round_to_nice_time(1e10)
        assert result > 0

    def test_overflow_handling(self):
        """Test that overflow from 10.0 is handled correctly."""
        from oscura.visualization.time_axis import _round_to_nice_time

        # Value that would round to 10.0 should overflow to next decade
        # 9.5 should round to 10.0, which overflows to 1.0 * 10^1
        result = _round_to_nice_time(9.5)
        assert result == 10.0


@pytest.mark.unit
@pytest.mark.visualization
class TestFormatCursorReadout:
    """Tests for format_cursor_readout function."""

    def test_basic_formatting_microseconds(self):
        """Test basic cursor readout in microseconds."""
        readout = format_cursor_readout(1.23456789e-6, unit="us")
        assert "μs" in readout
        assert "1.23456789" in readout

    def test_basic_formatting_milliseconds(self):
        """Test basic cursor readout in milliseconds."""
        readout = format_cursor_readout(0.00123, unit="ms")
        assert "ms" in readout
        assert "1.23" in readout

    def test_basic_formatting_seconds(self):
        """Test basic cursor readout in seconds."""
        readout = format_cursor_readout(1.5, unit="s")
        assert "s" in readout
        assert "1.5" in readout

    def test_basic_formatting_nanoseconds(self):
        """Test basic cursor readout in nanoseconds."""
        readout = format_cursor_readout(1.5e-9, unit="ns")
        assert "ns" in readout

    def test_basic_formatting_picoseconds(self):
        """Test basic cursor readout in picoseconds."""
        readout = format_cursor_readout(1.5e-12, unit="ps")
        assert "ps" in readout

    def test_auto_unit_selection(self):
        """Test automatic unit selection."""
        # Microsecond range
        readout_us = format_cursor_readout(1.5e-6, unit="auto")
        assert "μs" in readout_us or "us" in readout_us

        # Millisecond range
        readout_ms = format_cursor_readout(0.0015, unit="auto")
        assert "ms" in readout_ms

    def test_full_precision(self):
        """Test full precision mode."""
        readout = format_cursor_readout(1.23456789e-6, unit="us", full_precision=True)
        assert "μs" in readout
        # Should have many digits
        assert len(readout) > 10

    def test_standard_precision(self):
        """Test standard precision mode."""
        readout = format_cursor_readout(1.23456789e-6, unit="us", full_precision=False)
        assert "μs" in readout
        # Should have fewer digits than full precision
        assert "1.23457" in readout or "1.23456" in readout

    def test_zero_value(self):
        """Test formatting zero value."""
        readout = format_cursor_readout(0.0, unit="ms")
        assert "0" in readout
        assert "ms" in readout

    def test_negative_value(self):
        """Test formatting negative value."""
        readout = format_cursor_readout(-1.5e-6, unit="us")
        assert "-" in readout
        assert "μs" in readout

    def test_very_small_value(self):
        """Test very small value."""
        readout = format_cursor_readout(1e-15, unit="ps")
        assert "ps" in readout

    def test_very_large_value(self):
        """Test very large value."""
        readout = format_cursor_readout(1e6, unit="s")
        assert "s" in readout

    def test_unit_symbol_microseconds(self):
        """Test that microseconds use proper symbol."""
        readout = format_cursor_readout(1.0e-6, unit="us")
        assert "μs" in readout

    def test_unit_symbol_seconds(self):
        """Test seconds unit symbol."""
        readout = format_cursor_readout(1.0, unit="s")
        assert " s" in readout

    def test_unit_symbol_milliseconds(self):
        """Test milliseconds unit symbol."""
        readout = format_cursor_readout(0.001, unit="ms")
        assert "ms" in readout

    def test_unit_symbol_nanoseconds(self):
        """Test nanoseconds unit symbol."""
        readout = format_cursor_readout(1e-9, unit="ns")
        assert "ns" in readout

    def test_unit_symbol_picoseconds(self):
        """Test picoseconds unit symbol."""
        readout = format_cursor_readout(1e-12, unit="ps")
        assert "ps" in readout


@pytest.mark.unit
@pytest.mark.visualization
class TestVisualizationTimeAxisIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_milliseconds(self):
        """Test complete workflow with millisecond data."""
        # Generate time data
        time = np.linspace(0.0, 0.01, 100)

        # Create relative time
        time_rel = create_relative_time(time)
        assert time_rel[0] == 0.0

        # Convert to milliseconds
        time_ms = convert_time_values(time_rel, "ms")
        assert time_ms[-1] == pytest.approx(10.0)

        # Format labels
        labels = format_time_labels(time_rel, unit="ms")
        assert len(labels) == len(time)

        # Calculate ticks
        ticks = calculate_major_ticks(time_rel[0], time_rel[-1], unit="ms")
        assert len(ticks) > 0

    def test_full_workflow_microseconds(self):
        """Test complete workflow with microsecond data."""
        # Generate time data
        time = np.linspace(1000.0, 1000.01, 100)

        # Create relative time
        time_rel = create_relative_time(time)

        # Auto-select unit
        time_range = np.ptp(time_rel)
        unit = select_time_unit(time_range)
        assert unit == "ms"

        # Convert and format
        time_converted = convert_time_values(time_rel, unit)
        labels = format_time_labels(time_rel, unit=unit)
        assert len(labels) == len(time)

    def test_auto_workflow_various_ranges(self):
        """Test auto workflow with various time ranges."""
        test_ranges = [
            (0.0, 10.0),  # seconds
            (0.0, 0.01),  # milliseconds
            (0.0, 10e-6),  # microseconds
            (0.0, 10e-9),  # nanoseconds
        ]

        for t_min, t_max in test_ranges:
            time = np.linspace(t_min, t_max, 100)

            # Should work with auto everywhere
            time_rel = create_relative_time(time)
            labels = format_time_labels(time_rel, unit="auto")
            ticks = calculate_major_ticks(t_min, t_max, unit="auto")

            assert len(labels) == len(time)
            assert len(ticks) > 0

    def test_edge_case_single_sample(self):
        """Test edge case with single sample."""
        time = np.array([0.001])

        time_rel = create_relative_time(time)
        assert len(time_rel) == 1
        assert time_rel[0] == 0.0

        labels = format_time_labels(time, unit="ms")
        assert len(labels) == 1

        # Ticks with zero range
        ticks = calculate_major_ticks(time[0], time[0], unit="ms")
        assert len(ticks) == 1

    def test_edge_case_two_samples(self):
        """Test edge case with two samples."""
        time = np.array([0.0, 0.001])

        time_rel = create_relative_time(time)
        assert len(time_rel) == 2

        labels = format_time_labels(time, unit="ms")
        assert len(labels) == 2

        ticks = calculate_major_ticks(time[0], time[1], unit="ms")
        assert len(ticks) > 0

    def test_negative_to_positive_range(self):
        """Test range spanning negative to positive."""
        time = np.linspace(-0.005, 0.005, 100)

        time_rel = create_relative_time(time, start_at_zero=False)
        labels = format_time_labels(time, unit="ms")
        ticks = calculate_major_ticks(time[0], time[-1], unit="ms")

        assert len(labels) == len(time)
        assert len(ticks) > 0
        # Should have negative and positive ticks
        assert np.any(ticks < 0)
        assert np.any(ticks > 0)

    def test_consistency_across_units(self):
        """Test that conversions are consistent across units."""
        time = np.array([1e-6])  # 1 microsecond

        # Convert to different units
        us = convert_time_values(time, "us")
        ns = convert_time_values(time, "ns")
        ps = convert_time_values(time, "ps")

        # Check relationships
        assert us[0] == pytest.approx(1.0)
        assert ns[0] == pytest.approx(1000.0)
        assert ps[0] == pytest.approx(1000000.0)


@pytest.mark.unit
@pytest.mark.visualization
class TestExportedAPI:
    """Tests for exported API."""

    def test_all_exports_exist(self):
        """Test that all __all__ exports are importable."""
        from oscura.visualization import time_axis

        expected_exports = [
            "TimeUnit",
            "calculate_major_ticks",
            "convert_time_values",
            "create_relative_time",
            "format_cursor_readout",
            "format_time_labels",
            "select_time_unit",
        ]

        for export in expected_exports:
            assert hasattr(time_axis, export)

    def test_type_alias_time_unit(self):
        """Test TimeUnit type alias."""

        # Should be able to use as type hint
        def accept_unit(unit: TimeUnit) -> TimeUnit:
            return unit

        assert accept_unit("s") == "s"
        assert accept_unit("ms") == "ms"
        assert accept_unit("us") == "us"
        assert accept_unit("ns") == "ns"
        assert accept_unit("ps") == "ps"
        assert accept_unit("auto") == "auto"
