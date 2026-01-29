"""Pythonic operators and utilities for signal analysis.

This module provides Pythonic operators, time-based indexing,
automatic unit conversion, and convenience utilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = [
    "TimeIndex",
    "UnitConverter",
    "convert_units",
    "make_pipeable",
]


# =============================================================================
# =============================================================================


class TimeIndex:
    """Time-based indexing for trace data.

    Allows slicing trace data using time values instead of sample indices.

    Example:
        >>> ti = TimeIndex(data, sample_rate=1e9)
        >>> # Get first 1 millisecond
        >>> segment = ti["0ms":"1ms"]
        >>> # Get from 100us to 200us
        >>> segment = ti["100us":"200us"]

    References:
        API-016: Time-Based Indexing
    """

    # Time unit multipliers to seconds
    TIME_UNITS = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
    }

    def __init__(self, data: NDArray[np.float64], sample_rate: float, start_time: float = 0.0):
        """Initialize time indexer.

        Args:
            data: Trace data array
            sample_rate: Sample rate in Hz
            start_time: Start time offset in seconds
        """
        self._data = data
        self._sample_rate = sample_rate
        self._start_time = start_time

    @property
    def duration(self) -> float:
        """Get trace duration in seconds."""
        return len(self._data) / self._sample_rate

    @property
    def time_axis(self) -> NDArray[np.float64]:
        """Get time axis array."""
        return np.arange(len(self._data)) / self._sample_rate + self._start_time

    def _parse_time(self, time_str: str) -> float:
        """Parse time string to seconds.

        Args:
            time_str: Time string (e.g., "100ms", "1.5us")

        Returns:
            Time in seconds

        Raises:
            ValueError: If time format is invalid or unit is unknown.
        """
        # Match number with optional unit
        match = re.match(r"([-+]?\d*\.?\d+)\s*([a-zA-Z]*)", time_str.strip())
        if not match:
            raise ValueError(f"Invalid time format: {time_str}")

        value = float(match.group(1))
        unit = match.group(2).lower() or "s"

        if unit not in self.TIME_UNITS:
            raise ValueError(
                f"Unknown time unit: {unit}. Valid units: {list(self.TIME_UNITS.keys())}"
            )

        return value * self.TIME_UNITS[unit]

    def _time_to_index(self, time_seconds: float) -> int:
        """Convert time to sample index.

        Args:
            time_seconds: Time in seconds

        Returns:
            Sample index
        """
        relative_time = time_seconds - self._start_time
        index = int(relative_time * self._sample_rate)
        return max(0, min(index, len(self._data) - 1))

    def at(self, time: str | float) -> float:
        """Get value at specific time.

        Args:
            time: Time as string or float (seconds)

        Returns:
            Value at that time
        """
        if isinstance(time, str):
            time = self._parse_time(time)
        index = self._time_to_index(time)
        return float(self._data[index])

    def slice(
        self, start: str | float | None = None, end: str | float | None = None
    ) -> NDArray[np.float64]:
        """Slice data by time range.

        Args:
            start: Start time
            end: End time

        Returns:
            Sliced data array
        """
        if start is not None:
            if isinstance(start, str):
                start = self._parse_time(start)
            start_idx = self._time_to_index(start)
        else:
            start_idx = 0

        if end is not None:
            if isinstance(end, str):
                end = self._parse_time(end)
            end_idx = self._time_to_index(end)
        else:
            end_idx = len(self._data)

        return self._data[start_idx:end_idx]

    def __getitem__(self, key: slice | str | float) -> NDArray[np.float64] | float:  # type: ignore[valid-type]
        """Enable bracket notation for time-based indexing.

        Args:
            key: Slice with time strings, or single time

        Returns:
            Sliced data or single value
        """
        if isinstance(key, slice):
            return self.slice(key.start, key.stop)
        else:
            return self.at(key)


# =============================================================================
# =============================================================================


@dataclass
class Unit:
    """Unit definition with conversion factor.

    Attributes:
        name: Unit name
        symbol: Unit symbol
        factor: Conversion factor to base unit
        base_unit: Base unit name
    """

    name: str
    symbol: str
    factor: float
    base_unit: str


class UnitConverter:
    """Automatic unit conversion for measurements.

    Supports common electrical and signal analysis units with
    automatic prefix handling (mV, uV, MHz, etc.).

    Example:
        >>> converter = UnitConverter()
        >>> converter.convert(1000, "mV", "V")
        1.0
        >>> converter.auto_scale(0.000001, "V")
        (1.0, "uV")

    References:
        API-018: Automatic Unit Conversion
    """

    # SI prefixes
    SI_PREFIXES = {
        "P": 1e15,  # peta
        "T": 1e12,  # tera
        "G": 1e9,  # giga
        "M": 1e6,  # mega
        "k": 1e3,  # kilo
        "": 1.0,  # base
        "m": 1e-3,  # milli
        "u": 1e-6,  # micro
        "n": 1e-9,  # nano
        "p": 1e-12,  # pico
        "f": 1e-15,  # femto
    }

    # Base units
    BASE_UNITS = {
        "V": "voltage",
        "A": "current",
        "W": "power",
        "Hz": "frequency",
        "s": "time",
        "F": "capacitance",
        "H": "inductance",
        "Ohm": "resistance",
        "dB": "decibel",
        "dBm": "power_dbm",
        "dBV": "voltage_dbv",
    }

    def __init__(self) -> None:
        """Initialize converter."""
        self._custom_units: dict[str, Unit] = {}

    def _parse_unit(self, unit_str: str) -> tuple[float, str]:
        """Parse unit string into prefix multiplier and base unit.

        Args:
            unit_str: Unit string (e.g., "mV", "MHz")

        Returns:
            Tuple of (multiplier, base_unit)
        """
        # Check for dB-based units first
        for db_unit in ("dBm", "dBV", "dB"):
            if unit_str.endswith(db_unit):
                prefix = unit_str[: -len(db_unit)]
                multiplier = self.SI_PREFIXES.get(prefix, 1.0)
                return multiplier, db_unit

        # Check for other base units
        for base in sorted(self.BASE_UNITS.keys(), key=len, reverse=True):
            if unit_str.endswith(base):
                prefix = unit_str[: -len(base)]
                multiplier = self.SI_PREFIXES.get(prefix, 1.0)
                return multiplier, base

        # No recognized base unit
        return 1.0, unit_str

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units are incompatible
        """
        from_mult, from_base = self._parse_unit(from_unit)
        to_mult, to_base = self._parse_unit(to_unit)

        # Check compatibility
        if from_base != to_base:
            # Special handling for dB conversions
            if from_base == "dBm" and to_base == "W":
                return 10 ** ((value * from_mult - 30) / 10) / to_mult
            elif from_base == "W" and to_base == "dBm":
                return (10 * np.log10(value * from_mult) + 30) / to_mult  # type: ignore[no-any-return]
            elif from_base == "dBV" and to_base == "V":
                return 10 ** ((value * from_mult) / 20) / to_mult
            elif from_base == "V" and to_base == "dBV":
                return 20 * np.log10(value * from_mult) / to_mult  # type: ignore[no-any-return]
            else:
                raise ValueError(f"Cannot convert between {from_base} and {to_base}")

        # Simple conversion
        return value * from_mult / to_mult

    def auto_scale(self, value: float, base_unit: str) -> tuple[float, str]:
        """Automatically scale value to appropriate prefix.

        Args:
            value: Value in base units
            base_unit: Base unit string

        Returns:
            Tuple of (scaled_value, unit_string)

        Example:
            >>> converter.auto_scale(0.000001, "V")
            (1.0, "uV")
        """
        abs_value = abs(value) if value != 0 else 1

        # Find appropriate prefix
        prefixes_ordered = [
            ("P", 1e15),
            ("T", 1e12),
            ("G", 1e9),
            ("M", 1e6),
            ("k", 1e3),
            ("", 1.0),
            ("m", 1e-3),
            ("u", 1e-6),
            ("n", 1e-9),
            ("p", 1e-12),
            ("f", 1e-15),
        ]

        for prefix, factor in prefixes_ordered:
            scaled = abs_value / factor
            if 1.0 <= scaled < 1000.0 or prefix == "f":
                return value / factor, f"{prefix}{base_unit}"

        # Default to base unit
        return value, base_unit

    def format_value(self, value: float, base_unit: str, precision: int = 3) -> str:
        """Format value with automatic scaling.

        Args:
            value: Value in base units
            base_unit: Base unit string
            precision: Decimal precision

        Returns:
            Formatted string
        """
        scaled, unit = self.auto_scale(value, base_unit)
        return f"{scaled:.{precision}g} {unit}"


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert value between units.

    Convenience function for unit conversion.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value

    Example:
        >>> convert_units(1000, "mV", "V")
        1.0
        >>> convert_units(1, "MHz", "Hz")
        1000000.0

    References:
        API-018: Automatic Unit Conversion
    """
    return UnitConverter().convert(value, from_unit, to_unit)


# =============================================================================
# =============================================================================


class PipeableFunction:
    """Wrapper for making functions pipeable with >> operator.

    Example:
        >>> @make_pipeable
        >>> def lowpass(data, cutoff):
        ...     return filtered_data
        >>> result = data >> lowpass(cutoff=1e6) >> normalize()

    References:
        API-015: Pythonic Operators
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any):  # type: ignore[type-arg]
        """Initialize pipeable function.

        Args:
            func: Function to wrap
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        """Call function with data as first argument.

        Args:
            data: Input data

        Returns:
            Function result
        """
        return self._func(data, *self._args, **self._kwargs)

    def __rrshift__(self, other: Any) -> Any:
        """Enable data >> func() syntax.

        Args:
            other: Left operand (data)

        Returns:
            Function result
        """
        return self(other)


def make_pipeable(func: Callable) -> Callable:  # type: ignore[type-arg]
    """Decorator to make function pipeable with >> operator.

    Args:
        func: Function to wrap

    Returns:
        Wrapper that returns PipeableFunction

    Example:
        >>> @make_pipeable
        >>> def scale(data, factor):
        ...     return data * factor
        >>> result = data >> scale(factor=2)

    References:
        API-015: Pythonic Operators
    """

    def wrapper(*args: Any, **kwargs: Any) -> PipeableFunction:
        return PipeableFunction(func, *args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# Create pipeable versions of common operations
@make_pipeable
def scale(data: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    """Scale data by factor."""
    return data * factor


@make_pipeable
def offset(data: NDArray[np.float64], value: float) -> NDArray[np.float64]:
    """Add offset to data."""
    return data + value


@make_pipeable
def clip_values(data: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    """Clip data to range."""
    return np.clip(data, low, high)


@make_pipeable
def normalize_data(data: NDArray[np.float64], method: str = "minmax") -> NDArray[np.float64]:
    """Normalize data."""
    if method == "minmax":
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 0:
            result: NDArray[np.float64] = (data - dmin) / (dmax - dmin)
            return result
    elif method == "zscore":
        std = data.std()
        if std > 0:
            result_z: NDArray[np.float64] = (data - data.mean()) / std
            return result_z
    return data
