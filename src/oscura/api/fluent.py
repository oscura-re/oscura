"""Fluent interface for signal analysis.

This module provides a fluent (method chaining) interface for
expressing signal analysis operations in a readable, intuitive way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

T = TypeVar("T")

__all__ = [
    "FluentResult",
    "FluentTrace",
    "trace",
]


@dataclass
class FluentResult(Generic[T]):
    """Result container with fluent interface.

    Provides method chaining for result processing.

    Attributes:
        value: The wrapped value
        metadata: Associated metadata

    Example:
        >>> result = FluentResult(42.5)
        >>> result.format("The value is {:.2f}").print()
        The value is 42.50

    References:
        API-019: Fluent Interface
    """

    value: T
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self) -> T:
        """Get the raw value.

        Returns:
            The wrapped value
        """
        return self.value

    def map(self, func: Callable[[T], Any]) -> FluentResult:  # type: ignore[type-arg]
        """Apply function to value.

        Args:
            func: Function to apply

        Returns:
            New FluentResult with mapped value
        """
        return FluentResult(func(self.value), self.metadata.copy())

    def filter(self, predicate: Callable[[T], bool]) -> FluentResult | None:  # type: ignore[type-arg]
        """Filter based on predicate.

        Args:
            predicate: Filter function

        Returns:
            Self if predicate passes, None otherwise
        """
        if predicate(self.value):
            return self
        return None

    def format(self, fmt: str) -> FluentResult[str]:
        """Format value as string.

        Args:
            fmt: Format string

        Returns:
            New FluentResult with formatted string
        """
        return FluentResult(fmt.format(self.value), self.metadata.copy())

    def print(self, prefix: str = "") -> FluentResult[T]:
        """Print value and return self.

        Args:
            prefix: Optional prefix

        Returns:
            Self (for chaining)
        """
        print(f"{prefix}{self.value}")
        return self

    def with_metadata(self, **kwargs: Any) -> FluentResult[T]:
        """Add metadata.

        Args:
            **kwargs: Metadata key-value pairs

        Returns:
            Self (for chaining)
        """
        self.metadata.update(kwargs)
        return self

    def __repr__(self) -> str:
        return f"FluentResult({self.value!r})"


class FluentTrace:
    """Fluent interface wrapper for trace data.

    Provides method chaining for signal processing operations.

    Example:
        >>> result = (FluentTrace(data, sample_rate=1e9)
        ...     .lowpass(cutoff=1e6)
        ...     .normalize()
        ...     .fft(nfft=8192)
        ...     .magnitude()
        ...     .get())

    References:
        API-019: Fluent Interface
    """

    def __init__(self, data: NDArray[np.float64], sample_rate: float = 1.0, **metadata: Any):
        """Initialize fluent trace.

        Args:
            data: Trace data array
            sample_rate: Sample rate in Hz
            **metadata: Additional metadata
        """
        self._data = data
        self._sample_rate = sample_rate
        self._metadata = metadata
        self._history: list[str] = []

    @property
    def data(self) -> NDArray[np.float64]:
        """Get current data."""
        return self._data

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    def get(self) -> NDArray[np.float64]:
        """Get raw data array.

        Returns:
            Data array
        """
        return self._data

    def copy(self) -> FluentTrace:
        """Create copy of trace.

        Returns:
            New FluentTrace with copied data
        """
        return FluentTrace(self._data.copy(), self._sample_rate, **self._metadata.copy())

    # =========================================================================
    # Filtering Methods
    # =========================================================================

    def lowpass(self, cutoff: float, order: int = 4) -> FluentTrace:
        """Apply low-pass filter.

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Self (for chaining)
        """
        from scipy import signal

        nyq = self._sample_rate / 2
        normalized_cutoff = min(cutoff / nyq, 0.99)
        b, a = signal.butter(order, normalized_cutoff, btype="low")
        self._data = signal.filtfilt(b, a, self._data)
        self._history.append(f"lowpass(cutoff={cutoff})")
        return self

    def highpass(self, cutoff: float, order: int = 4) -> FluentTrace:
        """Apply high-pass filter.

        Args:
            cutoff: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Self (for chaining)
        """
        from scipy import signal

        nyq = self._sample_rate / 2
        normalized_cutoff = max(cutoff / nyq, 0.01)
        b, a = signal.butter(order, normalized_cutoff, btype="high")
        self._data = signal.filtfilt(b, a, self._data)
        self._history.append(f"highpass(cutoff={cutoff})")
        return self

    def bandpass(self, low: float, high: float, order: int = 4) -> FluentTrace:
        """Apply band-pass filter.

        Args:
            low: Low cutoff frequency
            high: High cutoff frequency
            order: Filter order

        Returns:
            Self (for chaining)
        """
        from scipy import signal

        nyq = self._sample_rate / 2
        b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
        self._data = signal.filtfilt(b, a, self._data)
        self._history.append(f"bandpass(low={low}, high={high})")
        return self

    def notch(self, freq: float, Q: float = 30.0) -> FluentTrace:
        """Apply notch filter.

        Args:
            freq: Notch frequency
            Q: Quality factor

        Returns:
            Self (for chaining)
        """
        from scipy import signal

        nyq = self._sample_rate / 2
        b, a = signal.iirnotch(freq / nyq, Q)
        self._data = signal.filtfilt(b, a, self._data)
        self._history.append(f"notch(freq={freq})")
        return self

    # =========================================================================
    # Transform Methods
    # =========================================================================

    def normalize(self, method: str = "minmax") -> FluentTrace:
        """Normalize data.

        Args:
            method: Normalization method (minmax, zscore, peak)

        Returns:
            Self (for chaining)
        """
        if method == "minmax":
            data_min = np.min(self._data)
            data_max = np.max(self._data)
            if data_max - data_min > 0:
                self._data = (self._data - data_min) / (data_max - data_min)
        elif method == "zscore":
            std = np.std(self._data)
            if std > 0:
                self._data = (self._data - np.mean(self._data)) / std
        elif method == "peak":
            peak = np.max(np.abs(self._data))
            if peak > 0:
                self._data = self._data / peak

        self._history.append(f"normalize(method={method})")
        return self

    def scale(self, factor: float) -> FluentTrace:
        """Scale data by factor.

        Args:
            factor: Scale factor

        Returns:
            Self (for chaining)
        """
        self._data = self._data * factor
        self._history.append(f"scale(factor={factor})")
        return self

    def offset(self, value: float) -> FluentTrace:
        """Add offset to data.

        Args:
            value: Offset value

        Returns:
            Self (for chaining)
        """
        self._data = self._data + value
        self._history.append(f"offset(value={value})")
        return self

    def clip(self, low: float, high: float) -> FluentTrace:
        """Clip data to range.

        Args:
            low: Low limit
            high: High limit

        Returns:
            Self (for chaining)
        """
        self._data = np.clip(self._data, low, high)
        self._history.append(f"clip(low={low}, high={high})")
        return self

    def abs(self) -> FluentTrace:
        """Take absolute value.

        Returns:
            Self (for chaining)
        """
        self._data = np.abs(self._data)
        self._history.append("abs()")
        return self

    def diff(self) -> FluentTrace:
        """Differentiate data.

        Returns:
            Self (for chaining)
        """
        self._data = np.diff(self._data, prepend=self._data[0])
        self._history.append("diff()")
        return self

    def integrate(self) -> FluentTrace:
        """Integrate data.

        Returns:
            Self (for chaining)
        """
        dt = 1.0 / self._sample_rate
        self._data = np.cumsum(self._data) * dt
        self._history.append("integrate()")
        return self

    # =========================================================================
    # Resampling Methods
    # =========================================================================

    def resample(self, new_length: int) -> FluentTrace:
        """Resample to new length.

        Args:
            new_length: New number of samples

        Returns:
            Self (for chaining)
        """
        from scipy import signal

        self._data = signal.resample(self._data, new_length)
        self._sample_rate = self._sample_rate * new_length / len(self._data)
        self._history.append(f"resample(new_length={new_length})")
        return self

    def decimate(self, factor: int) -> FluentTrace:
        """Decimate by factor.

        Args:
            factor: Decimation factor

        Returns:
            Self (for chaining)
        """
        from scipy import signal

        self._data = signal.decimate(self._data, factor)
        self._sample_rate = self._sample_rate / factor
        self._history.append(f"decimate(factor={factor})")
        return self

    def slice(self, start: int = 0, end: int | None = None) -> FluentTrace:
        """Slice data.

        Args:
            start: Start index
            end: End index (None for end of data)

        Returns:
            Self (for chaining)
        """
        self._data = self._data[start:end]
        self._history.append(f"slice(start={start}, end={end})")
        return self

    # =========================================================================
    # Spectral Methods
    # =========================================================================

    def fft(self, nfft: int | None = None) -> FluentTrace:
        """Compute FFT.

        Args:
            nfft: FFT size

        Returns:
            Self (for chaining, data is now complex)
        """
        self._data = np.fft.fft(self._data, n=nfft)  # type: ignore[assignment]
        self._history.append(f"fft(nfft={nfft})")
        return self

    def magnitude(self) -> FluentTrace:
        """Compute magnitude of complex data.

        Returns:
            Self (for chaining)
        """
        self._data = np.abs(self._data)
        self._history.append("magnitude()")
        return self

    def phase(self) -> FluentTrace:
        """Compute phase of complex data.

        Returns:
            Self (for chaining)
        """
        self._data = np.angle(self._data)
        self._history.append("phase()")
        return self

    def psd(self, nperseg: int = 256) -> FluentResult[tuple]:  # type: ignore[type-arg]
        """Compute power spectral density.

        Args:
            nperseg: Segment size

        Returns:
            FluentResult with (frequencies, psd) tuple
        """
        from scipy import signal

        f, psd = signal.welch(self._data, self._sample_rate, nperseg=nperseg)
        return FluentResult((f, psd))

    # =========================================================================
    # Measurement Methods
    # =========================================================================

    def mean(self) -> FluentResult[float]:
        """Compute mean.

        Returns:
            FluentResult with mean value
        """
        return FluentResult(float(np.mean(self._data)))

    def std(self) -> FluentResult[float]:
        """Compute standard deviation.

        Returns:
            FluentResult with std value
        """
        return FluentResult(float(np.std(self._data)))

    def rms(self) -> FluentResult[float]:
        """Compute RMS value.

        Returns:
            FluentResult with RMS value
        """
        return FluentResult(float(np.sqrt(np.mean(self._data**2))))

    def peak_to_peak(self) -> FluentResult[float]:
        """Compute peak-to-peak value.

        Returns:
            FluentResult with peak-to-peak value
        """
        return FluentResult(float(np.ptp(self._data)))

    def min(self) -> FluentResult[float]:
        """Get minimum value.

        Returns:
            FluentResult with min value
        """
        return FluentResult(float(np.min(self._data)))

    def max(self) -> FluentResult[float]:
        """Get maximum value.

        Returns:
            FluentResult with max value
        """
        return FluentResult(float(np.max(self._data)))

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_history(self) -> FluentTrace:
        """Print operation history.

        Returns:
            Self (for chaining)
        """
        print("Operation history:")
        for op in self._history:
            print(f"  - {op}")
        return self

    def with_metadata(self, **kwargs: Any) -> FluentTrace:
        """Add metadata.

        Args:
            **kwargs: Metadata key-value pairs

        Returns:
            Self (for chaining)
        """
        self._metadata.update(kwargs)
        return self

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"FluentTrace(samples={len(self._data)}, "
            f"sample_rate={self._sample_rate}, "
            f"operations={len(self._history)})"
        )


def trace(data: NDArray[np.float64], sample_rate: float = 1.0, **metadata: Any) -> FluentTrace:
    """Create fluent trace wrapper.

    Factory function for creating FluentTrace instances.

    Args:
        data: Trace data array
        sample_rate: Sample rate in Hz
        **metadata: Additional metadata

    Returns:
        FluentTrace instance

    Example:
        >>> result = (trace(data, sample_rate=1e9)
        ...     .lowpass(cutoff=1e6)
        ...     .normalize()
        ...     .mean()
        ...     .get())

    References:
        API-019: Fluent Interface
    """
    return FluentTrace(data, sample_rate, **metadata)
