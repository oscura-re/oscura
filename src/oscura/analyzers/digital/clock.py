"""Advanced clock recovery for digital signals.

This module provides comprehensive clock recovery and analysis tools for digital
signals, including frequency detection, clock reconstruction, baud rate detection,
and jitter measurement.


Example:
    >>> from oscura.analyzers.digital.clock import detect_clock_frequency, recover_clock
    >>> freq = detect_clock_frequency(data_trace, sample_rate=1e9)
    >>> print(f"Detected clock: {freq/1e6:.2f} MHz")
    >>> clock = recover_clock(data_trace, sample_rate=1e9, method='edge')
    >>> metrics = measure_clock_jitter(clock, sample_rate=1e9)

References:
    Gardner, F.M.: "Phaselock Techniques" (3rd Ed), Wiley, 2005
    Lee, E.A. & Messerschmitt, D.G.: "Digital Communication" (2nd Ed), 1994
    IEEE 1241-2010: Standard for Terminology and Test Methods for ADCs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from oscura.core.exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ClockMetrics:
    """Clock signal quality metrics.



    Attributes:
        frequency: Detected frequency in Hz.
        period_samples: Period in samples.
        period_seconds: Period in seconds.
        jitter_rms: RMS jitter in seconds.
        jitter_pp: Peak-to-peak jitter in seconds.
        duty_cycle: Duty cycle (0.0 to 1.0).
        stability: Stability score (0.0 to 1.0).
        confidence: Detection confidence (0.0 to 1.0).
    """

    frequency: float
    period_samples: float
    period_seconds: float
    jitter_rms: float
    jitter_pp: float
    duty_cycle: float
    stability: float
    confidence: float


@dataclass
class BaudRateResult:
    """Result of baud rate detection.



    Attributes:
        baud_rate: Detected baud rate in bits per second.
        bit_period_samples: Bit period in samples.
        confidence: Detection confidence (0.0 to 1.0).
        method: Method used for detection.
    """

    baud_rate: int
    bit_period_samples: float
    confidence: float
    method: str


class ClockRecovery:
    """Recover clock signal from data.



    This class provides multiple methods for clock recovery including edge-based,
    FFT-based, and autocorrelation-based detection, as well as PLL tracking and
    baud rate detection for asynchronous protocols.

    Can be initialized with or without sample_rate:
    - With sample_rate: ClockRecovery(sample_rate=1e9)
    - Without: ClockRecovery() - sample_rate extracted from trace metadata
    """

    # Standard baud rates for async protocols
    STANDARD_BAUD_RATES: ClassVar[list[int]] = [
        300,
        600,
        1200,
        2400,
        4800,
        9600,
        14400,
        19200,
        28800,
        38400,
        57600,
        115200,
        230400,
        460800,
        921600,
        1000000,
        2000000,
    ]

    def __init__(self, sample_rate: float | None = None):
        """Initialize with optional sample rate.

        Args:
            sample_rate: Sample rate in Hz. If None, will be extracted from trace metadata.

        Raises:
            ValidationError: If sample rate is provided and invalid.
        """
        if sample_rate is not None and sample_rate <= 0:
            raise ValidationError(f"Sample rate must be positive, got {sample_rate}")

        self.sample_rate: float | None = float(sample_rate) if sample_rate is not None else None

    def _get_sample_rate(self, trace: Any) -> float:
        """Extract sample rate from trace or use stored value.

        Args:
            trace: A DigitalTrace/WaveformTrace with metadata, or a numpy array.

        Returns:
            Sample rate in Hz.

        Raises:
            ValidationError: If sample rate cannot be determined.
        """
        if self.sample_rate is not None:
            return self.sample_rate

        # Try to extract from trace metadata
        if hasattr(trace, "metadata") and hasattr(trace.metadata, "sample_rate"):
            return float(trace.metadata.sample_rate)

        raise ValidationError(
            "Sample rate not set and cannot be extracted from trace. "
            "Either provide sample_rate to constructor or use a trace with metadata."
        )

    def _get_trace_data(self, trace: Any) -> NDArray[np.float64]:
        """Extract numpy array from trace object.

        Args:
            trace: A DigitalTrace/WaveformTrace or numpy array.

        Returns:
            Numpy array of signal data.
        """
        if hasattr(trace, "data"):
            return np.asarray(trace.data, dtype=np.float64)
        return np.asarray(trace, dtype=np.float64)

    def detect_frequency(
        self, trace: Any, method: Literal["edge", "fft", "autocorr"] = "edge"
    ) -> float:
        """Detect clock frequency from signal (supports DigitalTrace).



        This method supports both raw numpy arrays and DigitalTrace objects.
        Sample rate is extracted from trace metadata if not set in constructor.

        Args:
            trace: Signal trace data (DigitalTrace or numpy array).
            method: Detection method to use.

        Returns:
            Detected frequency in Hz.

        Example:
            >>> recovery = ClockRecovery()
            >>> freq = recovery.detect_frequency(digital_trace)
        """
        sample_rate = self._get_sample_rate(trace)
        data = self._get_trace_data(trace)

        # Temporarily set sample rate for internal methods
        old_rate = self.sample_rate
        self.sample_rate = sample_rate

        try:
            return self.detect_clock_frequency(data, method)
        finally:
            self.sample_rate = old_rate

    def detect_clock_frequency(
        self, trace: NDArray[np.float64], method: Literal["edge", "fft", "autocorr"] = "edge"
    ) -> float:
        """Detect clock frequency from signal.



        Detects the dominant clock frequency using the specified method.
        Each method has different strengths:
        - edge: Best for clean digital signals with clear transitions
        - fft: Best for noisy signals or periodic analog waveforms
        - autocorr: Best for periodic patterns with timing jitter

        Args:
            trace: Signal trace data.
            method: Detection method to use.

        Returns:
            Detected frequency in Hz.

        Raises:
            InsufficientDataError: If trace is too short.
            ValidationError: If method is invalid or detection fails.
        """
        if len(trace) < 10:
            raise InsufficientDataError("Trace must have at least 10 samples")

        if self.sample_rate is None:
            raise ValidationError(
                "Sample rate not set. Use detect_frequency() with trace metadata."
            )

        if method == "edge":
            return self._detect_frequency_edge(trace)
        elif method == "fft":
            return self._detect_frequency_fft(trace)
        elif method == "autocorr":
            return self._detect_frequency_autocorr(trace)
        else:
            raise ValidationError(f"Unknown method: {method}")

    def recover_clock(
        self, data_trace: NDArray[np.float64], method: Literal["edge", "pll", "fft"] = "edge"
    ) -> NDArray[np.float64]:
        """Recover clock signal from data.



        Reconstructs a clock signal from the data trace. The recovered clock
        is a square wave aligned to the detected clock transitions.

        Args:
            data_trace: Data signal trace.
            method: Recovery method to use.

        Returns:
            Recovered clock trace (same length as input).

        Raises:
            InsufficientDataError: If trace is too short.
            ValidationError: If method is invalid or recovery fails.
        """
        if len(data_trace) < 10:
            raise InsufficientDataError("Trace must have at least 10 samples")

        if self.sample_rate is None:
            raise ValidationError("Sample rate not set")

        # Detect clock frequency first
        freq = self.detect_clock_frequency(data_trace, method=method if method != "pll" else "edge")

        if freq <= 0:
            raise ValidationError("Failed to detect valid clock frequency")

        if method == "pll":
            # Use PLL tracking for robust recovery
            return self._pll_track(data_trace, freq)
        else:
            # Lazy import to avoid loading scipy at module import time
            from scipy import signal

            # Generate ideal square wave at detected frequency
            _period_samples = self.sample_rate / freq
            n_samples = len(data_trace)
            t = np.arange(n_samples)

            # Generate square wave (50% duty cycle)
            clock_raw = signal.square(2 * np.pi * freq * t / self.sample_rate)

            # Normalize to 0-1 range
            clock = (clock_raw + 1.0) / 2.0

            return np.asarray(clock, dtype=np.float64)

    def detect_baud_rate(
        self, trace: NDArray[np.float64], candidates: list[int] | None = None
    ) -> BaudRateResult:
        """Auto-detect baud rate for async protocols.



        Detects the baud rate by analyzing bit timing. Works best with traces
        containing start bits or transitions between different bit values.

        Args:
            trace: Signal trace data.
            candidates: List of candidate baud rates to test. If None, uses
                       standard rates.

        Returns:
            BaudRateResult with detected baud rate and confidence.

        Raises:
            InsufficientDataError: If trace is too short or not enough edges found.
            ValidationError: If sample rate is not set.
        """
        if len(trace) < 100:
            raise InsufficientDataError("Need at least 100 samples for baud rate detection")

        if self.sample_rate is None:
            raise ValidationError("Sample rate not set")

        if candidates is None:
            candidates = self.STANDARD_BAUD_RATES

        # Detect edges to find bit transitions
        edges = self._detect_edges_simple(trace)

        if len(edges) < 3:
            raise InsufficientDataError("Not enough edges to detect baud rate")

        # Calculate inter-edge intervals
        intervals = np.diff(edges)

        # The minimum interval should be close to one bit period
        # (assuming we have at least some single-bit pulses)
        # Use histogram to find most common interval
        hist, bin_edges = np.histogram(intervals, bins=50)
        most_common_interval = bin_edges[np.argmax(hist)]

        # Convert to frequency
        detected_freq = self.sample_rate / most_common_interval

        # Find closest standard baud rate
        candidates_array = np.array(candidates)
        errors = np.abs(candidates_array - detected_freq)
        best_idx = np.argmin(errors)
        best_baud = candidates_array[best_idx]

        # Calculate confidence based on how close we are to standard rate
        relative_error = errors[best_idx] / best_baud
        confidence = max(0.0, 1.0 - relative_error * 10)

        bit_period_samples = self.sample_rate / best_baud

        return BaudRateResult(
            baud_rate=int(best_baud),
            bit_period_samples=float(bit_period_samples),
            confidence=float(confidence),
            method="edge_histogram",
        )

    def measure_clock_jitter(self, clock_trace: NDArray[np.float64]) -> ClockMetrics:
        """Measure clock jitter and quality metrics.



        Analyzes a clock signal to measure jitter, duty cycle, and stability.
        Works best with recovered or measured clock signals.

        Args:
            clock_trace: Clock signal trace.

        Returns:
            ClockMetrics with comprehensive quality measurements.

        Raises:
            InsufficientDataError: If trace is too short or has too few edges.
            ValidationError: If sample rate is not set.
        """
        if len(clock_trace) < 10:
            raise InsufficientDataError("Trace must have at least 10 samples")

        if self.sample_rate is None:
            raise ValidationError("Sample rate not set")

        # Detect rising and falling edges
        rising_edges = self._detect_edges_by_type(clock_trace, "rising")
        falling_edges = self._detect_edges_by_type(clock_trace, "falling")

        if len(rising_edges) < 3:
            raise InsufficientDataError("Need at least 3 rising edges for jitter measurement")

        # Calculate periods from rising edge to rising edge
        periods = np.diff(rising_edges)

        if len(periods) == 0:
            raise InsufficientDataError("Cannot calculate period from single edge")

        # Mean period
        mean_period_samples = np.mean(periods)
        mean_period_seconds = mean_period_samples / self.sample_rate
        frequency = 1.0 / mean_period_seconds

        # RMS jitter (standard deviation of periods)
        jitter_rms_samples = np.std(periods)
        jitter_rms = jitter_rms_samples / self.sample_rate

        # Peak-to-peak jitter
        jitter_pp_samples = np.ptp(periods)
        jitter_pp = jitter_pp_samples / self.sample_rate

        # Duty cycle (high time / period)
        if len(falling_edges) >= len(rising_edges):
            # Can measure duty cycle
            high_times = []
            for _i, rise in enumerate(rising_edges):
                # Find next falling edge
                fall_idx = np.searchsorted(falling_edges, rise)
                if fall_idx < len(falling_edges):
                    high_time = falling_edges[fall_idx] - rise
                    high_times.append(high_time)

            if high_times:
                mean_high_time = np.mean(high_times)
                duty_cycle = mean_high_time / mean_period_samples
            else:
                duty_cycle = 0.5  # Assume 50% if cannot measure
        else:
            duty_cycle = 0.5

        # Stability score (inverse of relative jitter)
        relative_jitter = (
            jitter_rms_samples / mean_period_samples if mean_period_samples > 0 else 1.0
        )
        stability = max(0.0, 1.0 - relative_jitter * 10)

        # Confidence based on number of periods and stability
        confidence = min(1.0, len(periods) / 100.0) * stability

        return ClockMetrics(
            frequency=float(frequency),
            period_samples=float(mean_period_samples),
            period_seconds=float(mean_period_seconds),
            jitter_rms=float(jitter_rms),
            jitter_pp=float(jitter_pp),
            duty_cycle=float(np.clip(duty_cycle, 0.0, 1.0)),
            stability=float(stability),
            confidence=float(confidence),
        )

    def _detect_frequency_edge(self, trace: NDArray[np.float64]) -> float:
        """Detect frequency using edge timing histogram.



        Args:
            trace: Signal trace.

        Returns:
            Detected frequency in Hz.

        Raises:
            ValidationError: If not enough edges found to detect frequency.
        """
        edges = self._detect_edges_simple(trace)

        if len(edges) < 3:
            raise ValidationError("Not enough edges to detect frequency")

        # Calculate inter-edge intervals
        intervals = np.diff(edges)

        # Build histogram of intervals
        # The peak should correspond to half the period (edge to edge)
        hist, bin_edges = np.histogram(intervals, bins=50)
        _peak_interval = bin_edges[np.argmax(hist)]

        # Frequency is sample_rate / (2 * interval) for edge-to-edge
        # But we need to check if these are half-periods or full periods
        # Use median interval as robust estimator
        median_interval = np.median(intervals)

        # Assume median represents half-period (rising to falling or vice versa)
        # So full period is 2x median interval
        period_samples = 2 * median_interval
        frequency = self.sample_rate / period_samples

        return float(frequency)

    def _detect_frequency_fft(self, trace: NDArray[np.float64]) -> float:
        """Detect frequency using FFT spectral analysis.



        Args:
            trace: Signal trace.

        Returns:
            Detected frequency in Hz.

        Raises:
            ValidationError: If sample rate is not set.
        """
        # Lazy import to avoid loading scipy at module import time
        from scipy import signal

        # Remove DC component
        trace_ac = trace - np.mean(trace)

        # Apply window to reduce spectral leakage
        window = signal.windows.hann(len(trace_ac))
        trace_windowed = trace_ac * window

        # Compute FFT
        fft = np.fft.rfft(trace_windowed)
        if self.sample_rate is None:
            raise ValidationError("Sample rate not set")
        freqs = np.fft.rfftfreq(len(trace_windowed), 1.0 / self.sample_rate)

        # Find peak in magnitude spectrum
        magnitude = np.abs(fft)

        # Ignore DC and very low frequencies (below 10 Hz)
        min_freq_hz = 10.0
        min_freq_idx = np.searchsorted(freqs, min_freq_hz)
        if min_freq_idx >= len(magnitude):
            min_freq_idx = np.intp(1)

        peak_idx = min_freq_idx + np.argmax(magnitude[min_freq_idx:])
        frequency = freqs[peak_idx]

        return float(frequency)

    def _detect_frequency_autocorr(self, trace: NDArray[np.float64]) -> float:
        """Detect frequency using autocorrelation.



        Args:
            trace: Signal trace.

        Returns:
            Detected frequency in Hz.

        Raises:
            ValidationError: If no periodic pattern detected or sample rate not set.
        """
        # Lazy import to avoid loading scipy at module import time
        from scipy import signal

        # Remove mean
        trace_centered = trace - np.mean(trace)

        # Compute autocorrelation
        autocorr = signal.correlate(trace_centered, trace_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags

        # Normalize
        autocorr = autocorr / autocorr[0]

        # Find first peak after lag 0
        # Look for peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=0.3)

        if len(peaks) == 0:
            raise ValidationError("No periodic pattern detected in autocorrelation")

        # First peak corresponds to period
        period_samples = peaks[0]
        if self.sample_rate is None:
            raise ValidationError("Sample rate not set")
        frequency = self.sample_rate / period_samples

        return float(frequency)

    def _pll_track(
        self, trace: NDArray[np.float64], initial_freq: float, bandwidth: float = 0.01
    ) -> NDArray[np.float64]:
        """Software PLL for phase tracking.



        Implements a simple digital PLL for tracking phase and frequency
        variations in the input signal.

        Args:
            trace: Input data trace.
            initial_freq: Initial frequency estimate in Hz.
            bandwidth: Loop bandwidth (0.0 to 1.0), lower = more filtering.

        Returns:
            Recovered clock signal.

        Raises:
            ValidationError: If sample rate is not set.
        """
        n_samples = len(trace)
        clock = np.zeros(n_samples)

        # PLL state
        phase = 0.0
        freq = initial_freq
        if self.sample_rate is None:
            raise ValidationError("Sample rate not set")
        omega = 2 * np.pi * freq / self.sample_rate

        # Loop filter gains (proportional + integral)
        kp = 2 * bandwidth  # Proportional gain
        ki = bandwidth**2  # Integral gain

        # Detect edges for phase error calculation
        threshold = (np.max(trace) + np.min(trace)) / 2.0
        prev_sample = trace[0]

        for i in range(n_samples):
            # Generate clock output
            clock[i] = 1.0 if np.cos(phase) > 0 else 0.0

            # Detect phase error at edges
            current_sample = trace[i]
            phase_error = 0.0

            # Simple phase detector: check if edge coincides with clock transition
            if (prev_sample < threshold <= current_sample) or (
                prev_sample > threshold >= current_sample
            ):
                # Edge detected
                clock_value = np.cos(phase)
                # Phase error is sign of clock at edge
                phase_error = np.sign(clock_value) * 0.1

            # Update frequency and phase with loop filter
            _freq_adjust = kp * phase_error
            omega += ki * phase_error

            # Update phase
            phase += omega
            phase = phase % (2 * np.pi)

            prev_sample = current_sample

        return clock

    def _detect_edges_simple(self, trace: NDArray[np.float64]) -> NDArray[np.intp]:
        """Detect all edges in trace (both rising and falling).

        Args:
            trace: Signal trace.

        Returns:
            Array of edge indices.
        """
        threshold = (np.max(trace) + np.min(trace)) / 2.0
        rising = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]
        falling = np.where((trace[:-1] > threshold) & (trace[1:] <= threshold))[0]

        # Combine and sort
        all_edges = np.concatenate([rising, falling])
        all_edges.sort()

        return all_edges

    def _detect_edges_by_type(
        self, trace: NDArray[np.float64], edge_type: Literal["rising", "falling"]
    ) -> NDArray[np.intp]:
        """Detect edges of specific type.

        Args:
            trace: Signal trace.
            edge_type: Type of edge to detect.

        Returns:
            Array of edge indices.
        """
        threshold = (np.max(trace) + np.min(trace)) / 2.0

        if edge_type == "rising":
            edges = np.where((trace[:-1] < threshold) & (trace[1:] >= threshold))[0]
        else:  # falling
            edges = np.where((trace[:-1] > threshold) & (trace[1:] <= threshold))[0]

        return edges + 1  # Return index after crossing


# Convenience functions


def detect_clock_frequency(
    trace: NDArray[np.float64],
    sample_rate: float,
    method: Literal["edge", "fft", "autocorr"] = "edge",
) -> float:
    """Detect clock frequency from signal.



    Convenience function for detecting clock frequency without creating
    a ClockRecovery instance.

    Args:
        trace: Signal trace data.
        sample_rate: Sample rate in Hz.
        method: Detection method ('edge', 'fft', or 'autocorr').

    Returns:
        Detected frequency in Hz.

    Example:
        >>> freq = detect_clock_frequency(data, sample_rate=1e9, method='edge')
        >>> print(f"Clock: {freq/1e6:.2f} MHz")
    """
    recovery = ClockRecovery(sample_rate)
    return recovery.detect_clock_frequency(trace, method)


def recover_clock(
    data_trace: NDArray[np.float64],
    sample_rate: float,
    method: Literal["edge", "pll", "fft"] = "edge",
) -> NDArray[np.float64]:
    """Recover clock signal from data.



    Convenience function for recovering clock signal without creating
    a ClockRecovery instance.

    Args:
        data_trace: Data signal trace.
        sample_rate: Sample rate in Hz.
        method: Recovery method ('edge', 'pll', or 'fft').

    Returns:
        Recovered clock trace.

    Example:
        >>> clock = recover_clock(data, sample_rate=1e9, method='pll')
    """
    recovery = ClockRecovery(sample_rate)
    return recovery.recover_clock(data_trace, method)


def detect_baud_rate(
    trace: Any, sample_rate: float | None = None, candidates: list[int] | None = None
) -> int | BaudRateResult:
    """Auto-detect baud rate.



    Convenience function for baud rate detection. Supports both DigitalTrace
    objects (with metadata) and raw numpy arrays (requiring sample_rate).

    Args:
        trace: Signal trace data (DigitalTrace or numpy array).
        sample_rate: Sample rate in Hz (optional if trace has metadata).
        candidates: List of candidate baud rates. If None, uses standard rates.

    Returns:
        Detected baud rate as int (for DigitalTrace) or BaudRateResult.

    Raises:
        ValidationError: If sample_rate is required but not provided.

    Example:
        >>> baud = detect_baud_rate(digital_trace)  # Uses metadata
        >>> result = detect_baud_rate(data_array, sample_rate=1e6)  # Explicit rate
    """
    # Check if trace is a DigitalTrace with metadata
    if hasattr(trace, "metadata") and hasattr(trace.metadata, "sample_rate"):
        rate = trace.metadata.sample_rate
        data = np.asarray(trace.data, dtype=np.float64)
        recovery = ClockRecovery(rate)
        result = recovery.detect_baud_rate(data, candidates)
        return result.baud_rate  # Return just the baud rate for DigitalTrace
    elif sample_rate is not None:
        data = np.asarray(trace, dtype=np.float64)
        recovery = ClockRecovery(sample_rate)
        return recovery.detect_baud_rate(data, candidates)
    else:
        raise ValidationError("sample_rate required when trace is not a DigitalTrace with metadata")


def measure_clock_jitter(clock_trace: NDArray[np.float64], sample_rate: float) -> ClockMetrics:
    """Measure clock jitter.



    Convenience function for jitter measurement without creating
    a ClockRecovery instance.

    Args:
        clock_trace: Clock signal trace.
        sample_rate: Sample rate in Hz.

    Returns:
        ClockMetrics with jitter and quality measurements.

    Example:
        >>> metrics = measure_clock_jitter(clock, sample_rate=1e9)
        >>> print(f"RMS jitter: {metrics.jitter_rms*1e12:.2f} ps")
    """
    recovery = ClockRecovery(sample_rate)
    return recovery.measure_clock_jitter(clock_trace)


__all__ = [
    "BaudRateResult",
    "ClockMetrics",
    "ClockRecovery",
    "detect_baud_rate",
    "detect_clock_frequency",
    "measure_clock_jitter",
    "recover_clock",
]
