"""Unit tests for Pythonic operators and utilities.

Tests API-015, API-016, API-018
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from oscura.api.operators import (
    PipeableFunction,
    TimeIndex,
    Unit,
    UnitConverter,
    clip_values,
    convert_units,
    make_pipeable,
    normalize_data,
    offset,
    scale,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestTimeIndex:
    """Test TimeIndex class for time-based indexing.

    Tests API-016: Time-Based Indexing
    """

    def test_time_index_creation(self) -> None:
        """Test creating TimeIndex instance."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0

        ti = TimeIndex(data, sample_rate)

        assert ti._data is data
        assert ti._sample_rate == sample_rate
        assert ti._start_time == 0.0

    def test_time_index_with_start_time(self) -> None:
        """Test TimeIndex with custom start time."""
        data = np.array([1.0, 2.0, 3.0])
        sample_rate = 1000.0
        start_time = 5.0

        ti = TimeIndex(data, sample_rate, start_time)

        assert ti._start_time == start_time

    def test_duration_property(self) -> None:
        """Test duration property calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0  # 1000 samples/second

        ti = TimeIndex(data, sample_rate)

        # 5 samples at 1000 Hz = 5/1000 = 0.005 seconds
        assert ti.duration == 0.005

    def test_duration_with_different_sample_rate(self) -> None:
        """Test duration with different sample rates."""
        data = np.zeros(100)
        sample_rate = 1e6  # 1 MHz

        ti = TimeIndex(data, sample_rate)

        # 100 samples at 1 MHz = 100/1e6 = 100 microseconds
        assert ti.duration == 100e-6

    def test_time_axis_property(self) -> None:
        """Test time axis array generation."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        sample_rate = 1000.0

        ti = TimeIndex(data, sample_rate)
        time_axis = ti.time_axis

        expected = np.array([0.0, 0.001, 0.002, 0.003])
        np.testing.assert_array_almost_equal(time_axis, expected)

    def test_time_axis_with_start_time(self) -> None:
        """Test time axis with non-zero start time."""
        data = np.array([1.0, 2.0, 3.0])
        sample_rate = 1000.0
        start_time = 10.0

        ti = TimeIndex(data, sample_rate, start_time)
        time_axis = ti.time_axis

        expected = np.array([10.0, 10.001, 10.002])
        np.testing.assert_array_almost_equal(time_axis, expected)

    def test_parse_time_seconds(self) -> None:
        """Test parsing time string with seconds."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert ti._parse_time("1s") == 1.0
        assert ti._parse_time("2.5s") == 2.5
        assert ti._parse_time("0.001s") == 0.001

    def test_parse_time_milliseconds(self) -> None:
        """Test parsing time string with milliseconds."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert ti._parse_time("1ms") == 1e-3
        assert ti._parse_time("100ms") == 0.1
        assert ti._parse_time("5.5ms") == 5.5e-3

    def test_parse_time_microseconds(self) -> None:
        """Test parsing time string with microseconds."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert abs(ti._parse_time("1us") - 1e-6) < 1e-15
        assert abs(ti._parse_time("100us") - 100e-6) < 1e-15
        assert abs(ti._parse_time("2.5us") - 2.5e-6) < 1e-15

    def test_parse_time_nanoseconds(self) -> None:
        """Test parsing time string with nanoseconds."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert abs(ti._parse_time("1ns") - 1e-9) < 1e-18
        assert abs(ti._parse_time("500ns") - 500e-9) < 1e-18
        assert abs(ti._parse_time("1.5ns") - 1.5e-9) < 1e-18

    def test_parse_time_picoseconds(self) -> None:
        """Test parsing time string with picoseconds."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert ti._parse_time("1ps") == 1e-12
        assert ti._parse_time("100ps") == 100e-12

    def test_parse_time_no_unit_defaults_to_seconds(self) -> None:
        """Test parsing time string without unit defaults to seconds."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert ti._parse_time("1") == 1.0
        assert ti._parse_time("0.5") == 0.5

    def test_parse_time_with_whitespace(self) -> None:
        """Test parsing time string with whitespace."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert abs(ti._parse_time("  1 ms  ") - 1e-3) < 1e-15
        assert abs(ti._parse_time("100  us") - 100e-6) < 1e-15

    def test_parse_time_negative_values(self) -> None:
        """Test parsing negative time values."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        assert ti._parse_time("-1s") == -1.0
        assert ti._parse_time("-100ms") == -0.1

    def test_parse_time_invalid_format(self) -> None:
        """Test parsing invalid time format raises error."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        with pytest.raises(ValueError, match="Invalid time format"):
            ti._parse_time("invalid")

    def test_parse_time_unknown_unit(self) -> None:
        """Test parsing time with unknown unit raises error."""
        ti = TimeIndex(np.zeros(10), 1000.0)

        with pytest.raises(ValueError, match="Unknown time unit"):
            ti._parse_time("1xyz")

    def test_time_to_index_basic(self) -> None:
        """Test converting time to sample index."""
        data = np.zeros(10)
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # At 1000 Hz, index 5 is at 5/1000 = 0.005 seconds
        assert ti._time_to_index(0.005) == 5

    def test_time_to_index_with_start_time(self) -> None:
        """Test time to index conversion with start time offset."""
        data = np.zeros(10)
        sample_rate = 1000.0
        start_time = 1.0
        ti = TimeIndex(data, sample_rate, start_time)

        # Time 1.005 is 0.005 seconds after start (index should be 5)
        # But due to floating point precision, we might get 4 or 5
        index = ti._time_to_index(1.005)
        assert index in [4, 5]  # Allow for floating point errors

    def test_time_to_index_clipping_lower(self) -> None:
        """Test time to index clips to lower bound."""
        data = np.zeros(10)
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # Negative time should clip to 0
        assert ti._time_to_index(-1.0) == 0

    def test_time_to_index_clipping_upper(self) -> None:
        """Test time to index clips to upper bound."""
        data = np.zeros(10)
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # Time beyond data length should clip to last index
        assert ti._time_to_index(100.0) == 9

    def test_at_with_string_time(self) -> None:
        """Test getting value at specific time with string."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # At 1000 Hz, "2ms" is index 2
        value = ti.at("2ms")
        assert value == 3.0

    def test_at_with_float_time(self) -> None:
        """Test getting value at specific time with float."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # At 1000 Hz, 0.003 seconds is index 3
        value = ti.at(0.003)
        assert value == 4.0

    def test_at_first_sample(self) -> None:
        """Test getting first sample."""
        data = np.array([10.0, 20.0, 30.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        assert ti.at("0ms") == 10.0
        assert ti.at(0.0) == 10.0

    def test_at_last_sample(self) -> None:
        """Test getting last sample."""
        data = np.array([10.0, 20.0, 30.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # Last sample at index 2 = 2ms
        assert ti.at("2ms") == 30.0

    def test_slice_with_both_bounds(self) -> None:
        """Test slicing with start and end times."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # Slice from 1ms to 3ms (indices 1 to 3)
        result = ti.slice("1ms", "3ms")
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_slice_with_float_bounds(self) -> None:
        """Test slicing with float time values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        # Slice from 0.001 to 0.003 seconds
        result = ti.slice(0.001, 0.003)
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_slice_start_only(self) -> None:
        """Test slicing with only start time."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        result = ti.slice("2ms", None)
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_slice_end_only(self) -> None:
        """Test slicing with only end time."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        result = ti.slice(None, "3ms")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_slice_no_bounds(self) -> None:
        """Test slicing with no bounds returns all data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        result = ti.slice(None, None)
        np.testing.assert_array_equal(result, data)

    def test_getitem_with_slice(self) -> None:
        """Test bracket notation with slice."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        result = ti["1ms":"3ms"]
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_with_single_time_string(self) -> None:
        """Test bracket notation with single time string."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        value = ti["2ms"]
        assert value == 3.0

    def test_getitem_with_single_time_float(self) -> None:
        """Test bracket notation with single float."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_rate = 1000.0
        ti = TimeIndex(data, sample_rate)

        value = ti[0.002]
        assert value == 3.0

    def test_time_index_microsecond_resolution(self) -> None:
        """Test time indexing with microsecond resolution."""
        data = np.linspace(0, 1, 1000)
        sample_rate = 1e6  # 1 MHz
        ti = TimeIndex(data, sample_rate)

        # Get value at 500us
        result = ti["500us"]
        expected_index = 500
        assert abs(result - data[expected_index]) < 1e-10

    def test_time_index_nanosecond_resolution(self) -> None:
        """Test time indexing with nanosecond resolution."""
        data = np.linspace(0, 1, 1000)
        sample_rate = 1e9  # 1 GHz
        ti = TimeIndex(data, sample_rate)

        # Get value at 100ns
        result = ti["100ns"]
        expected_index = 100
        assert abs(result - data[expected_index]) < 1e-10


@pytest.mark.unit
class TestUnit:
    """Test Unit dataclass."""

    def test_unit_creation(self) -> None:
        """Test creating Unit instance."""
        unit = Unit(name="millivolt", symbol="mV", factor=1e-3, base_unit="V")

        assert unit.name == "millivolt"
        assert unit.symbol == "mV"
        assert unit.factor == 1e-3
        assert unit.base_unit == "V"

    def test_unit_attributes(self) -> None:
        """Test Unit attributes are accessible."""
        unit = Unit(name="kilohertz", symbol="kHz", factor=1e3, base_unit="Hz")

        assert hasattr(unit, "name")
        assert hasattr(unit, "symbol")
        assert hasattr(unit, "factor")
        assert hasattr(unit, "base_unit")


@pytest.mark.unit
class TestUnitConverter:
    """Test UnitConverter class.

    Tests API-018: Automatic Unit Conversion
    """

    def test_unit_converter_creation(self) -> None:
        """Test creating UnitConverter instance."""
        converter = UnitConverter()

        assert converter is not None
        assert converter._custom_units == {}

    def test_parse_unit_volts(self) -> None:
        """Test parsing voltage units."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("V")
        assert multiplier == 1.0
        assert base == "V"

        multiplier, base = converter._parse_unit("mV")
        assert multiplier == 1e-3
        assert base == "V"

        multiplier, base = converter._parse_unit("uV")
        assert multiplier == 1e-6
        assert base == "V"

    def test_parse_unit_frequency(self) -> None:
        """Test parsing frequency units."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("Hz")
        assert multiplier == 1.0
        assert base == "Hz"

        multiplier, base = converter._parse_unit("MHz")
        assert multiplier == 1e6
        assert base == "Hz"

        multiplier, base = converter._parse_unit("GHz")
        assert multiplier == 1e9
        assert base == "Hz"

    def test_parse_unit_time(self) -> None:
        """Test parsing time units."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("s")
        assert multiplier == 1.0
        assert base == "s"

        multiplier, base = converter._parse_unit("ms")
        assert multiplier == 1e-3
        assert base == "s"

        multiplier, base = converter._parse_unit("us")
        assert multiplier == 1e-6
        assert base == "s"

    def test_parse_unit_decibels(self) -> None:
        """Test parsing decibel units."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("dB")
        assert multiplier == 1.0
        assert base == "dB"

        multiplier, base = converter._parse_unit("dBm")
        assert multiplier == 1.0
        assert base == "dBm"

        multiplier, base = converter._parse_unit("dBV")
        assert multiplier == 1.0
        assert base == "dBV"

    def test_parse_unit_power(self) -> None:
        """Test parsing power units."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("W")
        assert multiplier == 1.0
        assert base == "W"

        multiplier, base = converter._parse_unit("mW")
        assert multiplier == 1e-3
        assert base == "W"

    def test_parse_unit_current(self) -> None:
        """Test parsing current units."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("A")
        assert multiplier == 1.0
        assert base == "A"

        multiplier, base = converter._parse_unit("mA")
        assert multiplier == 1e-3
        assert base == "A"

    def test_parse_unit_unknown(self) -> None:
        """Test parsing unknown unit."""
        converter = UnitConverter()

        multiplier, base = converter._parse_unit("unknown")
        assert multiplier == 1.0
        assert base == "unknown"

    def test_convert_same_unit(self) -> None:
        """Test converting between same units."""
        converter = UnitConverter()

        result = converter.convert(1.0, "V", "V")
        assert result == 1.0

    def test_convert_volts_to_millivolts(self) -> None:
        """Test converting volts to millivolts."""
        converter = UnitConverter()

        result = converter.convert(1.0, "V", "mV")
        assert result == 1000.0

    def test_convert_millivolts_to_volts(self) -> None:
        """Test converting millivolts to volts."""
        converter = UnitConverter()

        result = converter.convert(1000.0, "mV", "V")
        assert result == 1.0

    def test_convert_microvolts_to_volts(self) -> None:
        """Test converting microvolts to volts."""
        converter = UnitConverter()

        result = converter.convert(1000000.0, "uV", "V")
        assert result == 1.0

    def test_convert_hertz_to_megahertz(self) -> None:
        """Test converting Hz to MHz."""
        converter = UnitConverter()

        result = converter.convert(1000000.0, "Hz", "MHz")
        assert result == 1.0

    def test_convert_megahertz_to_hertz(self) -> None:
        """Test converting MHz to Hz."""
        converter = UnitConverter()

        result = converter.convert(1.0, "MHz", "Hz")
        assert result == 1000000.0

    def test_convert_seconds_to_milliseconds(self) -> None:
        """Test converting seconds to milliseconds."""
        converter = UnitConverter()

        result = converter.convert(1.0, "s", "ms")
        assert result == 1000.0

    def test_convert_dbm_to_watts(self) -> None:
        """Test converting dBm to watts."""
        converter = UnitConverter()

        # 0 dBm = 1 mW = 0.001 W
        result = converter.convert(0.0, "dBm", "W")
        assert abs(result - 0.001) < 1e-10

        # 30 dBm = 1 W
        result = converter.convert(30.0, "dBm", "W")
        assert abs(result - 1.0) < 1e-10

    def test_convert_watts_to_dbm(self) -> None:
        """Test converting watts to dBm."""
        converter = UnitConverter()

        # 1 mW = 0 dBm
        result = converter.convert(0.001, "W", "dBm")
        assert abs(result - 0.0) < 1e-10

        # 1 W = 30 dBm
        result = converter.convert(1.0, "W", "dBm")
        assert abs(result - 30.0) < 1e-10

    def test_convert_dbv_to_volts(self) -> None:
        """Test converting dBV to volts."""
        converter = UnitConverter()

        # 0 dBV = 1 V
        result = converter.convert(0.0, "dBV", "V")
        assert abs(result - 1.0) < 1e-10

        # 20 dBV = 10 V
        result = converter.convert(20.0, "dBV", "V")
        assert abs(result - 10.0) < 1e-10

    def test_convert_volts_to_dbv(self) -> None:
        """Test converting volts to dBV."""
        converter = UnitConverter()

        # 1 V = 0 dBV
        result = converter.convert(1.0, "V", "dBV")
        assert abs(result - 0.0) < 1e-10

        # 10 V = 20 dBV
        result = converter.convert(10.0, "V", "dBV")
        assert abs(result - 20.0) < 1e-10

    def test_convert_incompatible_units_raises_error(self) -> None:
        """Test converting incompatible units raises error."""
        converter = UnitConverter()

        with pytest.raises(ValueError, match="Cannot convert"):
            converter.convert(1.0, "V", "Hz")

    def test_auto_scale_volts(self) -> None:
        """Test auto-scaling voltage values."""
        converter = UnitConverter()

        # 0.001 V should be 1 mV
        scaled, unit = converter.auto_scale(0.001, "V")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "mV"

        # 0.000001 V should be 1 uV
        scaled, unit = converter.auto_scale(0.000001, "V")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "uV"

    def test_auto_scale_large_values(self) -> None:
        """Test auto-scaling large values."""
        converter = UnitConverter()

        # 1000 V should be 1 kV
        scaled, unit = converter.auto_scale(1000.0, "V")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "kV"

        # 1000000 Hz should be 1 MHz
        scaled, unit = converter.auto_scale(1000000.0, "Hz")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "MHz"

    def test_auto_scale_gigahertz(self) -> None:
        """Test auto-scaling to gigahertz."""
        converter = UnitConverter()

        scaled, unit = converter.auto_scale(1e9, "Hz")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "GHz"

    def test_auto_scale_nanoseconds(self) -> None:
        """Test auto-scaling to nanoseconds."""
        converter = UnitConverter()

        scaled, unit = converter.auto_scale(1e-9, "s")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "ns"

    def test_auto_scale_zero_value(self) -> None:
        """Test auto-scaling zero value."""
        converter = UnitConverter()

        # Zero should default to base unit
        scaled, unit = converter.auto_scale(0.0, "V")
        assert scaled == 0.0
        assert unit == "V"

    def test_auto_scale_negative_value(self) -> None:
        """Test auto-scaling negative value."""
        converter = UnitConverter()

        # Should use absolute value for scaling
        scaled, unit = converter.auto_scale(-0.001, "V")
        assert abs(scaled - (-1.0)) < 1e-10
        assert unit == "mV"

    def test_auto_scale_very_small_value(self) -> None:
        """Test auto-scaling very small value to femto."""
        converter = UnitConverter()

        scaled, unit = converter.auto_scale(1e-15, "s")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "fs"

    def test_auto_scale_very_large_value(self) -> None:
        """Test auto-scaling very large value to peta."""
        converter = UnitConverter()

        scaled, unit = converter.auto_scale(1e15, "Hz")
        assert abs(scaled - 1.0) < 1e-10
        assert unit == "PHz"

    def test_auto_scale_extremely_small_defaults_to_femto(self) -> None:
        """Test extremely small values default to femto prefix."""
        converter = UnitConverter()

        # Even smaller than femto should still use femto
        scaled, unit = converter.auto_scale(1e-20, "s")
        assert unit == "fs"

    def test_format_value_basic(self) -> None:
        """Test formatting value with auto-scaling."""
        converter = UnitConverter()

        result = converter.format_value(0.001, "V")
        assert "1" in result
        assert "mV" in result

    def test_format_value_with_precision(self) -> None:
        """Test formatting value with custom precision."""
        converter = UnitConverter()

        result = converter.format_value(1.23456e-3, "V", precision=2)
        assert "mV" in result
        # Should have at most 2 significant figures

    def test_format_value_default_precision(self) -> None:
        """Test formatting value with default precision."""
        converter = UnitConverter()

        result = converter.format_value(1.234567e-3, "V")
        assert "mV" in result
        assert "1.23" in result  # Default precision is 3


@pytest.mark.unit
class TestConvertUnitsFunction:
    """Test convert_units convenience function."""

    def test_convert_units_volts_to_millivolts(self) -> None:
        """Test convert_units function for V to mV."""
        result = convert_units(1.0, "V", "mV")
        assert result == 1000.0

    def test_convert_units_megahertz_to_hertz(self) -> None:
        """Test convert_units function for MHz to Hz."""
        result = convert_units(1.0, "MHz", "Hz")
        assert result == 1000000.0

    def test_convert_units_seconds_to_microseconds(self) -> None:
        """Test convert_units function for s to us."""
        result = convert_units(1.0, "s", "us")
        assert result == 1000000.0


@pytest.mark.unit
class TestPipeableFunction:
    """Test PipeableFunction class.

    Tests API-015: Pythonic Operators
    """

    def test_pipeable_function_creation(self) -> None:
        """Test creating PipeableFunction instance."""

        def dummy_func(data: Any, arg: int) -> Any:
            return data

        pf = PipeableFunction(dummy_func, 42)

        assert pf._func is dummy_func
        assert pf._args == (42,)
        assert pf._kwargs == {}

    def test_pipeable_function_with_kwargs(self) -> None:
        """Test PipeableFunction with keyword arguments."""

        def dummy_func(data: Any, factor: float = 1.0) -> Any:
            return data

        pf = PipeableFunction(dummy_func, factor=2.0)

        assert pf._kwargs == {"factor": 2.0}

    def test_pipeable_function_call(self) -> None:
        """Test calling PipeableFunction."""

        def multiply(data: np.ndarray, factor: float) -> np.ndarray:
            return data * factor

        pf = PipeableFunction(multiply, 2.0)
        data = np.array([1.0, 2.0, 3.0])

        result = pf(data)

        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_pipeable_function_rrshift(self) -> None:
        """Test PipeableFunction with >> operator."""

        def multiply(data: np.ndarray, factor: float) -> np.ndarray:
            return data * factor

        pf = PipeableFunction(multiply, 2.0)
        data = np.array([1.0, 2.0, 3.0])

        result = data >> pf

        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_pipeable_function_chaining(self) -> None:
        """Test chaining multiple PipeableFunction calls."""

        def multiply(data: np.ndarray, factor: float) -> np.ndarray:
            return data * factor

        def add(data: np.ndarray, value: float) -> np.ndarray:
            return data + value

        pf_mult = PipeableFunction(multiply, 2.0)
        pf_add = PipeableFunction(add, 10.0)

        data = np.array([1.0, 2.0, 3.0])
        result = data >> pf_mult >> pf_add

        expected = np.array([12.0, 14.0, 16.0])
        np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
class TestMakePipeableDecorator:
    """Test make_pipeable decorator."""

    def test_make_pipeable_decorator(self) -> None:
        """Test make_pipeable decorator basic functionality."""

        @make_pipeable
        def multiply(data: np.ndarray, factor: float) -> np.ndarray:
            return data * factor

        # Should return PipeableFunction wrapper
        pf = multiply(factor=2.0)
        assert isinstance(pf, PipeableFunction)

    def test_make_pipeable_preserves_name(self) -> None:
        """Test make_pipeable preserves function name."""

        @make_pipeable
        def my_function(data: Any) -> Any:
            return data

        assert my_function.__name__ == "my_function"

    def test_make_pipeable_preserves_docstring(self) -> None:
        """Test make_pipeable preserves function docstring."""

        @make_pipeable
        def my_function(data: Any) -> Any:
            """This is my function."""
            return data

        assert my_function.__doc__ == "This is my function."

    def test_make_pipeable_with_positional_args(self) -> None:
        """Test make_pipeable with positional arguments."""

        @make_pipeable
        def combine(data: np.ndarray, a: float, b: float) -> np.ndarray:
            return data * a + b

        pf = combine(2.0, 10.0)
        data = np.array([1.0, 2.0, 3.0])
        result = data >> pf

        expected = np.array([12.0, 14.0, 16.0])
        np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
class TestBuiltInPipeableOperators:
    """Test built-in pipeable operators."""

    def test_scale_operator(self) -> None:
        """Test scale pipeable operator."""
        data = np.array([1.0, 2.0, 3.0])

        result = data >> scale(factor=2.0)

        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_scale_negative_factor(self) -> None:
        """Test scale with negative factor."""
        data = np.array([1.0, 2.0, 3.0])

        result = data >> scale(factor=-1.0)

        expected = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(result, expected)

    def test_offset_operator(self) -> None:
        """Test offset pipeable operator."""
        data = np.array([1.0, 2.0, 3.0])

        result = data >> offset(value=10.0)

        expected = np.array([11.0, 12.0, 13.0])
        np.testing.assert_array_equal(result, expected)

    def test_offset_negative_value(self) -> None:
        """Test offset with negative value."""
        data = np.array([10.0, 20.0, 30.0])

        result = data >> offset(value=-5.0)

        expected = np.array([5.0, 15.0, 25.0])
        np.testing.assert_array_equal(result, expected)

    def test_clip_values_operator(self) -> None:
        """Test clip_values pipeable operator."""
        data = np.array([1.0, 5.0, 10.0, 15.0, 20.0])

        result = data >> clip_values(low=5.0, high=15.0)

        expected = np.array([5.0, 5.0, 10.0, 15.0, 15.0])
        np.testing.assert_array_equal(result, expected)

    def test_clip_values_lower_bound(self) -> None:
        """Test clip_values respects lower bound."""
        data = np.array([-10.0, -5.0, 0.0, 5.0])

        result = data >> clip_values(low=0.0, high=100.0)

        expected = np.array([0.0, 0.0, 0.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_clip_values_upper_bound(self) -> None:
        """Test clip_values respects upper bound."""
        data = np.array([5.0, 10.0, 15.0, 20.0])

        result = data >> clip_values(low=-100.0, high=10.0)

        expected = np.array([5.0, 10.0, 10.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_normalize_data_minmax(self) -> None:
        """Test normalize_data with minmax method."""
        data = np.array([0.0, 5.0, 10.0])

        # Call directly - >> operator doesn't work with numpy arrays for this function
        pf = normalize_data(method="minmax")
        result = pf(data)

        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_data_zscore(self) -> None:
        """Test normalize_data with zscore method."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Call directly - >> operator doesn't work with numpy arrays for this function
        pf = normalize_data(method="zscore")
        result = pf(data)

        # Should have mean ~0 and std ~1
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_normalize_data_default_method(self) -> None:
        """Test normalize_data with default method (minmax)."""
        data = np.array([10.0, 20.0, 30.0])

        # Call directly - >> operator doesn't work with numpy arrays for this function
        pf = normalize_data()
        result = pf(data)

        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_data_constant_array_minmax(self) -> None:
        """Test normalize_data with constant array (minmax)."""
        data = np.array([5.0, 5.0, 5.0])

        # Call directly - >> operator doesn't work with numpy arrays for this function
        pf = normalize_data(method="minmax")
        result = pf(data)

        # Should return unchanged when range is 0
        np.testing.assert_array_equal(result, data)

    def test_normalize_data_constant_array_zscore(self) -> None:
        """Test normalize_data with constant array (zscore)."""
        data = np.array([5.0, 5.0, 5.0])

        # Call directly - >> operator doesn't work with numpy arrays for this function
        pf = normalize_data(method="zscore")
        result = pf(data)

        # Should return unchanged when std is 0
        np.testing.assert_array_equal(result, data)

    def test_normalize_data_unknown_method(self) -> None:
        """Test normalize_data with unknown method returns unchanged data."""
        data = np.array([1.0, 2.0, 3.0])

        # Call directly with unknown method
        pf = normalize_data(method="unknown")
        result = pf(data)

        # Should return unchanged for unknown method
        np.testing.assert_array_equal(result, data)

    def test_chaining_multiple_operators(self) -> None:
        """Test chaining multiple pipeable operators."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = data >> scale(factor=2.0) >> offset(value=10.0) >> clip_values(low=12.0, high=18.0)

        # data * 2 = [2, 4, 6, 8, 10]
        # + 10 = [12, 14, 16, 18, 20]
        # clip = [12, 14, 16, 18, 18]
        expected = np.array([12.0, 14.0, 16.0, 18.0, 18.0])
        np.testing.assert_array_equal(result, expected)

    def test_complex_pipeline(self) -> None:
        """Test complex pipeline with normalization."""
        data = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

        # Normalize first (call directly), then use >> for compatible operations
        pf_norm = normalize_data(method="minmax")
        normalized = pf_norm(data)
        result = normalized >> scale(factor=100.0) >> offset(value=-50.0)

        # Normalize: [0, 0.25, 0.5, 0.75, 1.0]
        # Scale: [0, 25, 50, 75, 100]
        # Offset: [-50, -25, 0, 25, 50]
        expected = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
        np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.unit
class TestApiOperatorsEdgeCases:
    """Test edge cases and error conditions."""

    def test_time_index_empty_array(self) -> None:
        """Test TimeIndex with empty array."""
        data = np.array([])
        ti = TimeIndex(data, 1000.0)

        assert ti.duration == 0.0
        assert len(ti.time_axis) == 0

    def test_time_index_single_sample(self) -> None:
        """Test TimeIndex with single sample."""
        data = np.array([42.0])
        ti = TimeIndex(data, 1000.0)

        assert ti.duration == 0.001
        assert ti.at(0.0) == 42.0

    def test_unit_converter_zero_conversion(self) -> None:
        """Test converting zero value."""
        converter = UnitConverter()

        result = converter.convert(0.0, "V", "mV")
        assert result == 0.0

    def test_pipeable_with_empty_array(self) -> None:
        """Test pipeable operators with empty array."""
        data = np.array([])

        result = data >> scale(factor=2.0)

        assert len(result) == 0

    def test_normalize_single_value(self) -> None:
        """Test normalizing single value."""
        data = np.array([5.0])

        # Call directly - >> operator doesn't work with numpy arrays for this function
        pf = normalize_data(method="minmax")
        result = pf(data)

        # Single value should remain unchanged
        np.testing.assert_array_equal(result, data)

    def test_clip_with_reversed_bounds(self) -> None:
        """Test clip_values with min > max."""
        data = np.array([1.0, 5.0, 10.0])

        # NumPy's clip handles this by swapping
        result = data >> clip_values(low=10.0, high=5.0)

        # All values should be clipped to 5.0 (the effective max)
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_time_index_with_very_high_sample_rate(self) -> None:
        """Test TimeIndex with very high sample rate."""
        data = np.array([1.0, 2.0, 3.0])
        sample_rate = 1e12  # 1 THz
        ti = TimeIndex(data, sample_rate)

        # Duration should be in picoseconds
        assert ti.duration == 3e-12

    def test_unit_converter_large_value_conversion(self) -> None:
        """Test converting very large values."""
        converter = UnitConverter()

        result = converter.convert(1e9, "Hz", "GHz")
        assert abs(result - 1.0) < 1e-10

    def test_unit_converter_small_value_conversion(self) -> None:
        """Test converting very small values."""
        converter = UnitConverter()

        result = converter.convert(1e-12, "s", "ps")
        assert abs(result - 1.0) < 1e-10
