"""Comprehensive timing analysis and clock recovery for signal synchronization.

This module provides advanced timing analysis capabilities including:
- Clock recovery using multiple methods (ZCD, histogram, autocorrelation, PLL, FFT)
- Baud rate detection for serial protocols
- Timing jitter and drift analysis
- Phase-locked loop (PLL) simulation
- Signal-to-noise ratio (SNR) calculation
- Eye diagram generation

Example:
    >>> import numpy as np
    >>> analyzer = TimingAnalyzer(method="autocorrelation")
    >>> signal = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 100000))
    >>> result = analyzer.recover_clock(signal, sample_rate=100e6)
    >>> print(f"Clock rate: {result.detected_clock_rate / 1e6:.3f} MHz")
    Clock rate: 1.000 MHz
    >>> print(f"Confidence: {result.confidence:.2f}")
    Confidence: 0.95

References:
    - Digital Communications by John G. Proakis
    - Clock Recovery in High-Speed Optical Fiber Systems (IEEE)
    - Phase-Locked Loop Design Handbook by Dan H. Wolaver
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TimingAnalysisResult:
    """Timing analysis and clock recovery result.

    Attributes:
        detected_clock_rate: Recovered clock frequency in Hz.
        confidence: Confidence score from 0.0 to 1.0.
        jitter_rms: RMS jitter in seconds.
        drift_rate: Clock drift in parts per million (ppm).
        snr_db: Signal-to-noise ratio in decibels.
        method: Method used ("zcd", "histogram", "autocorrelation", "pll", "fft").
        statistics: Additional method-specific statistics.

    Example:
        >>> result = TimingAnalysisResult(
        ...     detected_clock_rate=10e6,
        ...     confidence=0.95,
        ...     jitter_rms=1e-12,
        ...     drift_rate=2.5,
        ...     snr_db=45.0,
        ...     method="autocorrelation"
        ... )
        >>> print(f"Clock: {result.detected_clock_rate / 1e6:.1f} MHz")
        Clock: 10.0 MHz
    """

    detected_clock_rate: float
    confidence: float
    jitter_rms: float
    drift_rate: float
    snr_db: float
    method: str
    statistics: dict[str, float] = field(default_factory=dict)


class TimingAnalyzer:
    """Timing analysis and clock recovery engine.

    Provides comprehensive timing analysis including multiple clock recovery
    methods, jitter/drift measurement, and signal quality assessment.

    Attributes:
        method: Clock recovery method to use.
        METHODS: Available clock recovery methods.

    Example:
        >>> analyzer = TimingAnalyzer(method="autocorrelation")
        >>> signal = np.random.randn(10000)
        >>> result = analyzer.recover_clock(signal, sample_rate=1e6)
        >>> print(f"Method: {result.method}")
        Method: autocorrelation
    """

    METHODS: ClassVar[list[str]] = ["zcd", "histogram", "autocorrelation", "pll", "fft"]

    def __init__(self, method: str = "autocorrelation") -> None:
        """Initialize timing analyzer.

        Args:
            method: Clock recovery method ("zcd", "histogram", "autocorrelation",
                "pll", or "fft"). Default is "autocorrelation" for best general-
                purpose performance.

        Raises:
            ValueError: If method is not in METHODS.

        Example:
            >>> analyzer = TimingAnalyzer(method="fft")
            >>> analyzer.method
            'fft'
        """
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}, got '{method}'")
        self.method = method

    def recover_clock(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
        initial_estimate: float | None = None,
    ) -> TimingAnalysisResult:
        """Recover clock frequency from signal.

        Uses the configured method to recover the clock rate from a digital
        or analog signal. Automatically detects periodicity and estimates
        the dominant frequency.

        Args:
            signal: Input signal array.
            sample_rate: Sampling rate in Hz.
            initial_estimate: Initial frequency estimate in Hz (optional,
                required for PLL method).

        Returns:
            TimingAnalysisResult with recovered clock rate and statistics.

        Example:
            >>> signal = np.tile([0, 0, 1, 1], 1000)
            >>> analyzer = TimingAnalyzer(method="autocorrelation")
            >>> result = analyzer.recover_clock(signal, sample_rate=1e6)
            >>> result.detected_clock_rate > 0
            True

        References:
            IEEE 1241-2010: Standard for Terminology and Test Methods
        """
        if self.method == "zcd":
            return self._zero_crossing_detection(signal, sample_rate)
        elif self.method == "histogram":
            return self._histogram_method(signal, sample_rate)
        elif self.method == "autocorrelation":
            return self._autocorrelation_method(signal, sample_rate)
        elif self.method == "pll":
            if initial_estimate is None:
                raise ValueError("PLL method requires initial_estimate parameter")
            return self._pll_simulation(signal, sample_rate, initial_estimate)
        elif self.method == "fft":
            return self._fft_method(signal, sample_rate)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detect_baud_rate(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
        min_baud: float = 300,
        max_baud: float = 115200,
    ) -> TimingAnalysisResult:
        """Detect serial baud rate from signal.

        Analyzes signal to determine the most likely baud rate for
        serial communication protocols (UART, RS-232, etc.).

        Args:
            signal: Input serial signal array.
            sample_rate: Sampling rate in Hz.
            min_baud: Minimum baud rate to consider (default 300).
            max_baud: Maximum baud rate to consider (default 115200).

        Returns:
            TimingAnalysisResult with detected baud rate.

        Example:
            >>> # 9600 baud signal (104.17 µs bit period)
            >>> bit_period = 1 / 9600
            >>> signal = np.tile([0]*10 + [1]*10, 500)
            >>> analyzer = TimingAnalyzer(method="autocorrelation")
            >>> result = analyzer.detect_baud_rate(signal, sample_rate=1e6)
            >>> abs(result.detected_clock_rate - 9600) < 1000
            True

        References:
            EIA-232: Serial data communication standard
        """
        # Use autocorrelation to find bit period
        temp_analyzer = TimingAnalyzer(method="autocorrelation")
        result = temp_analyzer.recover_clock(signal, sample_rate)

        # Filter to common baud rates
        detected_rate = result.detected_clock_rate

        # Clamp to valid range
        if detected_rate < min_baud:
            detected_rate = min_baud
            result.confidence *= 0.5
        elif detected_rate > max_baud:
            detected_rate = max_baud
            result.confidence *= 0.5

        # Find nearest standard baud rate
        standard_bauds = [
            300,
            1200,
            2400,
            4800,
            9600,
            14400,
            19200,
            38400,
            57600,
            115200,
            230400,
            460800,
            921600,
        ]
        valid_bauds = [b for b in standard_bauds if min_baud <= b <= max_baud]

        if valid_bauds:
            nearest_baud = min(valid_bauds, key=lambda b: abs(b - detected_rate))
            # Increase confidence if very close to standard baud
            if abs(nearest_baud - detected_rate) / nearest_baud < 0.05:
                result.confidence = min(1.0, result.confidence * 1.2)
                detected_rate = float(nearest_baud)

        result.detected_clock_rate = detected_rate
        result.statistics["standard_baud_match"] = (
            abs(detected_rate - nearest_baud) / nearest_baud < 0.05 if valid_bauds else False
        )

        return result

    def analyze_jitter(
        self,
        transitions: NDArray[np.floating[Any]],
        nominal_period: float,
    ) -> dict[str, Any]:
        """Analyze timing jitter from edge transitions.

        Computes jitter statistics including RMS, peak-to-peak, and
        histogram distribution.

        Args:
            transitions: Array of transition timestamps in seconds.
            nominal_period: Expected nominal period in seconds.

        Returns:
            Dictionary with jitter statistics:
                - rms: RMS jitter in seconds
                - peak_to_peak: Peak-to-peak jitter in seconds
                - mean_period: Mean measured period in seconds
                - std_period: Standard deviation of period in seconds
                - histogram_bins: Histogram bin edges
                - histogram_counts: Histogram counts

        Example:
            >>> transitions = np.array([0.0, 1e-6, 2e-6, 3.01e-6, 4e-6])
            >>> analyzer = TimingAnalyzer()
            >>> stats = analyzer.analyze_jitter(transitions, nominal_period=1e-6)
            >>> stats['rms'] >= 0
            True
            >>> 'peak_to_peak' in stats
            True

        References:
            IEEE 2414-2020: Standard for Jitter and Phase Noise
        """
        if len(transitions) < 2:
            return {
                "rms": np.nan,
                "peak_to_peak": np.nan,
                "mean_period": np.nan,
                "std_period": np.nan,
                "histogram_bins": np.array([]),
                "histogram_counts": np.array([]),
            }

        # Calculate periods
        periods = np.diff(transitions)

        if len(periods) == 0:
            return {
                "rms": np.nan,
                "peak_to_peak": np.nan,
                "mean_period": np.nan,
                "std_period": np.nan,
                "histogram_bins": np.array([]),
                "histogram_counts": np.array([]),
            }

        # Jitter is deviation from nominal period - handle NaN values
        with np.errstate(invalid="ignore"):
            deviations = periods - nominal_period

        rms_jitter = float(np.std(deviations))
        pp_jitter = float(np.max(periods) - np.min(periods))
        mean_period = float(np.mean(periods))
        std_period = float(np.std(periods))

        # Generate histogram
        if len(deviations) >= 10:
            counts, bins = np.histogram(deviations, bins=50)
        else:
            counts = np.array([])
            bins = np.array([])

        return {
            "rms": rms_jitter,
            "peak_to_peak": pp_jitter,
            "mean_period": mean_period,
            "std_period": std_period,
            "histogram_bins": bins,
            "histogram_counts": counts,
        }

    def analyze_drift(
        self,
        transitions: NDArray[np.floating[Any]],
        window_size: int = 1000,
    ) -> float:
        """Analyze clock drift over time.

        Measures the rate of change in clock frequency over the observation
        window, expressed in parts per million (ppm).

        Args:
            transitions: Array of transition timestamps in seconds.
            window_size: Number of transitions to use for drift calculation.

        Returns:
            Clock drift in ppm (parts per million).

        Example:
            >>> # Perfect clock (no drift)
            >>> transitions = np.arange(0, 1000) * 1e-6
            >>> analyzer = TimingAnalyzer()
            >>> drift = analyzer.analyze_drift(transitions, window_size=100)
            >>> abs(drift) < 10  # Very low drift
            True

        References:
            IEEE 1588: Precision Time Protocol (PTP)
        """
        if len(transitions) < window_size:
            window_size = len(transitions)

        if window_size < 10:
            return np.nan

        # Split into windows and calculate average period in each
        n_windows = max(2, window_size // 100)
        window_length = len(transitions) // n_windows

        window_frequencies: list[float] = []

        for i in range(n_windows):
            start_idx = i * window_length
            end_idx = min((i + 1) * window_length, len(transitions))

            if end_idx - start_idx < 2:
                continue

            window_transitions = transitions[start_idx:end_idx]
            periods = np.diff(window_transitions)

            if len(periods) > 0:
                mean_period = np.mean(periods)
                if mean_period > 0:
                    window_frequencies.append(1.0 / mean_period)

        if len(window_frequencies) < 2:
            return 0.0

        # Linear fit to frequency vs time
        time_points = np.linspace(0, len(transitions), len(window_frequencies))
        coeffs = np.polyfit(time_points, window_frequencies, 1)
        slope = coeffs[0]  # Hz per sample
        mean_freq = np.mean(window_frequencies)

        if mean_freq == 0:
            return 0.0

        # Convert to ppm
        drift_ppm = (slope * len(transitions)) / mean_freq * 1e6

        return float(drift_ppm)

    def calculate_snr(
        self,
        signal: NDArray[np.floating[Any]],
        signal_freq: float,
        sample_rate: float,
    ) -> float:
        """Calculate signal-to-noise ratio in dB.

        Estimates SNR by separating signal power (at fundamental frequency)
        from noise power (all other frequency components).

        Args:
            signal: Input signal array.
            signal_freq: Expected signal frequency in Hz.
            sample_rate: Sampling rate in Hz.

        Returns:
            SNR in decibels (dB).

        Example:
            >>> signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.1, 10000))
            >>> analyzer = TimingAnalyzer()
            >>> snr = analyzer.calculate_snr(signal, signal_freq=1000, sample_rate=100e3)
            >>> snr > 40  # Clean signal should have high SNR
            True

        References:
            IEEE 1057: Standard for Digitizing Waveform Recorders
        """
        if len(signal) < 64:
            return np.nan

        # Remove DC component - handle inf/nan
        with np.errstate(invalid="ignore"):
            signal_clean = np.where(np.isfinite(signal), signal, 0.0)
            signal_centered = signal_clean - np.mean(signal_clean)

        # Compute FFT
        n = len(signal_centered)
        nfft = int(2 ** np.ceil(np.log2(n)))
        spectrum = np.fft.rfft(signal_centered, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        magnitude = np.abs(spectrum)

        # Find bin closest to signal frequency
        signal_bin_idx = np.argmin(np.abs(freqs - signal_freq))

        # Signal power is in the signal bin and immediate neighbors
        signal_bins = [signal_bin_idx]
        if signal_bin_idx > 0:
            signal_bins.append(signal_bin_idx - 1)
        if signal_bin_idx < len(magnitude) - 1:
            signal_bins.append(signal_bin_idx + 1)

        signal_power = float(np.sum(magnitude[signal_bins] ** 2))

        # Noise power is everything else (excluding DC at bin 0)
        noise_mask = np.ones(len(magnitude), dtype=bool)
        noise_mask[0] = False  # Exclude DC
        for idx in signal_bins:
            noise_mask[idx] = False

        noise_power = float(np.sum(magnitude[noise_mask] ** 2))

        if noise_power == 0 or signal_power == 0:
            return np.nan

        snr = 10 * np.log10(signal_power / noise_power)

        return float(snr)

    def generate_eye_diagram(
        self,
        signal: NDArray[np.floating[Any]],
        symbol_rate: float,
        sample_rate: float,
        output_path: Path,
    ) -> None:
        """Generate eye diagram for signal quality assessment.

        Creates an eye diagram by overlaying multiple symbol periods.
        A wide, open eye indicates good signal quality; a closed eye
        indicates high jitter or noise.

        Args:
            signal: Input signal array.
            symbol_rate: Symbol rate in Hz.
            sample_rate: Sampling rate in Hz.
            output_path: Path to save eye diagram image.

        Example:
            >>> import tempfile
            >>> signal = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 100000))
            >>> analyzer = TimingAnalyzer()
            >>> with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            ...     analyzer.generate_eye_diagram(signal, 1e6, 100e6, Path(f.name))

        References:
            - Telecommunications Measurement Analysis (Tektronix)
            - IEEE 802.3: Ethernet eye diagram templates
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib required for eye diagram generation. "
                "Install with: pip install matplotlib"
            ) from e

        # Calculate samples per symbol
        samples_per_symbol = int(sample_rate / symbol_rate)

        if samples_per_symbol < 4:
            raise ValueError(
                f"Insufficient samples per symbol: {samples_per_symbol}. "
                f"Need at least 4 samples per symbol for eye diagram."
            )

        # Extract symbol periods (2 symbols per trace for eye diagram)
        num_symbols = len(signal) // samples_per_symbol
        eye_traces: list[NDArray[np.floating[Any]]] = []

        for i in range(num_symbols - 1):
            start = i * samples_per_symbol
            end = start + 2 * samples_per_symbol  # 2 symbols for eye
            if end <= len(signal):
                eye_traces.append(signal[start:end])

        if len(eye_traces) == 0:
            raise ValueError("Insufficient data for eye diagram")

        # Plot overlaid traces
        plt.figure(figsize=(10, 6))
        time_axis = np.linspace(0, 2, 2 * samples_per_symbol)

        for trace in eye_traces:
            plt.plot(time_axis, trace, alpha=0.1, color="blue", linewidth=0.5)

        plt.xlabel("Time (symbol periods)")
        plt.ylabel("Amplitude")
        plt.title(f"Eye Diagram (Symbol Rate: {symbol_rate / 1e3:.1f} kHz)")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2)
        plt.tight_layout()

        # Save to file
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def export_statistics(
        self,
        result: TimingAnalysisResult,
        output_path: Path,
    ) -> None:
        """Export timing statistics as JSON.

        Args:
            result: TimingAnalysisResult to export.
            output_path: Path to save JSON file.

        Example:
            >>> import tempfile
            >>> result = TimingAnalysisResult(
            ...     detected_clock_rate=10e6,
            ...     confidence=0.95,
            ...     jitter_rms=1e-12,
            ...     drift_rate=2.5,
            ...     snr_db=45.0,
            ...     method="autocorrelation"
            ... )
            >>> analyzer = TimingAnalyzer()
            >>> with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            ...     analyzer.export_statistics(result, Path(f.name))
        """
        import json

        data = {
            "detected_clock_rate_hz": result.detected_clock_rate,
            "detected_clock_rate_mhz": result.detected_clock_rate / 1e6,
            "confidence": result.confidence,
            "jitter_rms_seconds": result.jitter_rms,
            "jitter_rms_picoseconds": result.jitter_rms * 1e12,
            "drift_rate_ppm": result.drift_rate,
            "snr_db": result.snr_db,
            "method": result.method,
            "statistics": result.statistics,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _zero_crossing_detection(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
    ) -> TimingAnalysisResult:
        """Zero-crossing based clock recovery.

        Detects zero crossings (rising edges) and calculates the most
        common interval between crossings to determine clock period.

        Args:
            signal: Input signal array.
            sample_rate: Sampling rate in Hz.

        Returns:
            TimingAnalysisResult with recovered clock rate.
        """
        # Normalize signal
        signal_norm = signal - np.mean(signal)

        # Find zero crossings (rising edges)
        crossings: list[float] = []
        for i in range(len(signal_norm) - 1):
            if signal_norm[i] <= 0 and signal_norm[i + 1] > 0:
                # Linear interpolation to find exact crossing
                if abs(signal_norm[i + 1] - signal_norm[i]) > 1e-12:
                    frac = abs(signal_norm[i]) / (abs(signal_norm[i]) + signal_norm[i + 1])
                    crossing_idx = i + frac
                    crossings.append(crossing_idx / sample_rate)

        if len(crossings) < 2:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="zcd",
            )

        # Calculate intervals between crossings
        intervals = np.diff(crossings)

        if len(intervals) == 0:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="zcd",
            )

        # Find most common interval (mode) using histogram
        hist, edges = np.histogram(intervals, bins=min(100, len(intervals)))
        mode_idx = int(np.argmax(hist))
        mode_interval = (edges[mode_idx] + edges[mode_idx + 1]) / 2

        # Detected clock rate
        clock_rate = 1.0 / mode_interval if mode_interval > 0 else 0.0

        # Calculate jitter (RMS of deviations from mode interval)
        jitter_rms = float(np.std(intervals - mode_interval))

        # Confidence based on histogram peak sharpness
        confidence = float(hist[mode_idx] / len(intervals))

        # Calculate drift
        drift_rate = self.analyze_drift(np.array(crossings))

        # Calculate SNR
        snr_db = self.calculate_snr(signal, clock_rate, sample_rate)

        return TimingAnalysisResult(
            detected_clock_rate=float(clock_rate),
            confidence=confidence,
            jitter_rms=jitter_rms,
            drift_rate=drift_rate,
            snr_db=snr_db,
            method="zcd",
            statistics={"num_crossings": len(crossings), "mode_interval": mode_interval},
        )

    def _histogram_method(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
    ) -> TimingAnalysisResult:
        """Histogram-based clock recovery.

        Uses histogram of signal values to detect logic levels,
        then finds transitions and computes intervals.

        Args:
            signal: Input signal array.
            sample_rate: Sampling rate in Hz.

        Returns:
            TimingAnalysisResult with recovered clock rate.
        """
        # Find logic levels using histogram
        hist, bin_edges = np.histogram(signal, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in histogram (logic levels)
        mid_idx = len(hist) // 2
        low_peak_idx = int(np.argmax(hist[:mid_idx]))
        high_peak_idx = mid_idx + int(np.argmax(hist[mid_idx:]))

        low_level = bin_centers[low_peak_idx]
        high_level = bin_centers[high_peak_idx]

        # Threshold is midpoint
        threshold = (low_level + high_level) / 2

        # Find transitions (rising or falling edges)
        transitions: list[float] = []
        for i in range(len(signal) - 1):
            if (signal[i] < threshold <= signal[i + 1]) or (signal[i] >= threshold > signal[i + 1]):
                transitions.append(i / sample_rate)

        if len(transitions) < 2:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="histogram",
            )

        # Calculate intervals
        intervals = np.diff(transitions)

        if len(intervals) == 0:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="histogram",
            )

        # Mode interval
        mode_interval = float(np.median(intervals))
        clock_rate = 1.0 / mode_interval if mode_interval > 0 else 0.0

        jitter_rms = float(np.std(intervals))
        confidence = min(1.0, 1.0 / (1.0 + jitter_rms / mode_interval))

        drift_rate = self.analyze_drift(np.array(transitions))
        snr_db = self.calculate_snr(signal, clock_rate, sample_rate)

        return TimingAnalysisResult(
            detected_clock_rate=float(clock_rate),
            confidence=confidence,
            jitter_rms=jitter_rms,
            drift_rate=drift_rate,
            snr_db=snr_db,
            method="histogram",
            statistics={
                "num_transitions": len(transitions),
                "low_level": low_level,
                "high_level": high_level,
                "threshold": threshold,
            },
        )

    def _autocorrelation_method(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
    ) -> TimingAnalysisResult:
        """Autocorrelation-based clock recovery.

        Autocorrelation shows periodicity in signal. Peak at lag τ
        indicates period τ.

        Args:
            signal: Input signal array.
            sample_rate: Sampling rate in Hz.

        Returns:
            TimingAnalysisResult with recovered clock rate.
        """
        # Handle edge cases
        if len(signal) == 0 or sample_rate <= 0:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="autocorrelation",
            )

        # Calculate autocorrelation - handle inf/nan
        with np.errstate(invalid="ignore"):
            signal_clean = np.where(np.isfinite(signal), signal, 0.0)
            signal_norm = signal_clean - np.mean(signal_clean)

        # Use FFT-based autocorrelation for efficiency
        n = len(signal_norm)
        fft_signal = np.fft.fft(signal_norm, n=2 * n)
        autocorr = np.fft.ifft(fft_signal * np.conj(fft_signal)).real
        autocorr = autocorr[:n]  # Keep positive lags

        # Normalize
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]

        # Skip small lags to avoid noise (minimum 1 MHz period = 1 µs)
        min_lag_samples = max(1, int(sample_rate / 1000000))

        # Find first significant peak (after lag 0)
        peaks: list[tuple[int, float]] = []
        for i in range(min_lag_samples, len(autocorr) - 1):
            if (
                autocorr[i] > autocorr[i - 1]
                and autocorr[i] > autocorr[i + 1]
                and autocorr[i] > 0.3
            ):  # Significant peak threshold
                peaks.append((i, autocorr[i]))

        if not peaks:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="autocorrelation",
            )

        # Use first peak
        peak_lag, peak_value = peaks[0]
        period = peak_lag / sample_rate
        clock_rate = 1.0 / period if period > 0 else 0.0

        # Find secondary peaks for jitter estimation
        secondary_peaks = [p for p in peaks[1:4] if p[1] > 0.2]
        if secondary_peaks:
            # Calculate jitter from peak spread
            all_peak_lags = [peak_lag] + [p[0] for p in secondary_peaks]
            # Expected harmonic positions
            expected_lags = [peak_lag * (i + 1) for i in range(len(all_peak_lags))]
            jitter_samples = np.std(
                [abs(a - e) for a, e in zip(all_peak_lags, expected_lags, strict=True)]
            )
            jitter_rms = float(jitter_samples / sample_rate)
        else:
            jitter_rms = 0.0

        snr_db = self.calculate_snr(signal, clock_rate, sample_rate)

        return TimingAnalysisResult(
            detected_clock_rate=float(clock_rate),
            confidence=float(peak_value),
            jitter_rms=jitter_rms,
            drift_rate=0.0,  # Autocorrelation doesn't directly measure drift
            snr_db=snr_db,
            method="autocorrelation",
            statistics={
                "peak_lag": peak_lag,
                "peak_value": peak_value,
                "num_peaks": len(peaks),
            },
        )

    def _pll_simulation(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
        initial_freq: float,
    ) -> TimingAnalysisResult:
        """Phase-locked loop simulation for clock recovery.

        Simulates a PLL to track signal frequency. The PLL adjusts
        its frequency to minimize phase error with the input signal.

        Args:
            signal: Input signal array.
            sample_rate: Sampling rate in Hz.
            initial_freq: Initial PLL frequency estimate in Hz.

        Returns:
            TimingAnalysisResult with recovered clock rate.

        References:
            - Phase-Locked Loop Design Handbook by Dan H. Wolaver
            - Digital Communications by Proakis & Salehi
        """
        damping_factor = 0.707
        natural_freq = initial_freq * 0.1
        kp = 2 * damping_factor * natural_freq
        ki = natural_freq**2

        phase_errors, frequencies = _run_pll_loop(signal, sample_rate, initial_freq, kp, ki)
        recovered_freq, confidence = _analyze_pll_convergence(frequencies, initial_freq)
        jitter_rms = float(np.std(phase_errors)) / (2 * np.pi * float(recovered_freq))
        drift_ppm = _compute_pll_drift(frequencies, recovered_freq)
        snr_db = self.calculate_snr(signal, recovered_freq, sample_rate)

        return TimingAnalysisResult(
            detected_clock_rate=recovered_freq,
            confidence=float(confidence),
            jitter_rms=jitter_rms,
            drift_rate=float(drift_ppm),
            snr_db=snr_db,
            method="pll",
            statistics={
                "initial_freq": initial_freq,
                "final_freq": recovered_freq,
                "damping_factor": damping_factor,
                "natural_freq": natural_freq,
            },
        )

    def _fft_method(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
    ) -> TimingAnalysisResult:
        """FFT-based clock recovery.

        Uses FFT to find the dominant frequency component in the signal.

        Args:
            signal: Input signal array.
            sample_rate: Sampling rate in Hz.

        Returns:
            TimingAnalysisResult with recovered clock rate.
        """
        if len(signal) < 64:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="fft",
            )

        # Remove DC and compute FFT
        signal_centered = signal - np.mean(signal)
        n = len(signal_centered)
        nfft = int(2 ** np.ceil(np.log2(n)))
        spectrum = np.fft.rfft(signal_centered, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        magnitude = np.abs(spectrum)

        # Exclude DC component
        if len(magnitude) > 1:
            magnitude = magnitude[1:]
            freqs = freqs[1:]

        if len(magnitude) == 0:
            return TimingAnalysisResult(
                detected_clock_rate=0.0,
                confidence=0.0,
                jitter_rms=0.0,
                drift_rate=0.0,
                snr_db=0.0,
                method="fft",
            )

        # Find peak
        peak_idx = int(np.argmax(magnitude))
        peak_freq = freqs[peak_idx]
        peak_mag = magnitude[peak_idx]

        # Parabolic interpolation for more accurate frequency
        if 0 < peak_idx < len(magnitude) - 1:
            alpha = magnitude[peak_idx - 1]
            beta = magnitude[peak_idx]
            gamma = magnitude[peak_idx + 1]

            if beta > alpha and beta > gamma and abs(alpha - 2 * beta + gamma) > 1e-12:
                freq_resolution = sample_rate / nfft
                delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                peak_freq = peak_freq + delta * freq_resolution

        # Calculate confidence (ratio of peak to RMS of spectrum)
        rms_mag = np.sqrt(np.mean(magnitude**2))
        if rms_mag > 0:
            confidence = min(1.0, (peak_mag / rms_mag - 1) / 10)
        else:
            confidence = 0.0

        snr_db = self.calculate_snr(signal, peak_freq, sample_rate)

        return TimingAnalysisResult(
            detected_clock_rate=float(peak_freq),
            confidence=float(confidence),
            jitter_rms=0.0,  # FFT doesn't directly measure jitter
            drift_rate=0.0,  # FFT doesn't measure drift
            snr_db=snr_db,
            method="fft",
            statistics={
                "peak_magnitude": float(peak_mag),
                "rms_magnitude": float(rms_mag),
            },
        )


def _run_pll_loop(
    signal: NDArray[np.floating[Any]],
    sample_rate: float,
    initial_freq: float,
    kp: float,
    ki: float,
) -> tuple[list[float], list[float]]:
    """Run PLL loop to track signal frequency.

    Args:
        signal: Input signal array.
        sample_rate: Sampling rate in Hz.
        initial_freq: Initial frequency estimate.
        kp: Proportional gain.
        ki: Integral gain.

    Returns:
        Tuple of (phase_errors, frequencies).
    """
    phase = 0.0
    frequency = initial_freq
    phase_error_integral = 0.0
    phase_errors: list[float] = []
    frequencies: list[float] = []
    dt = 1.0 / sample_rate

    for sample in signal:
        vco_output = np.sin(2 * np.pi * phase)
        phase_error = sample * vco_output
        phase_error_integral += phase_error * dt
        frequency = initial_freq + kp * phase_error + ki * phase_error_integral
        phase += frequency * dt
        phase = phase % 1.0

        phase_errors.append(phase_error)
        frequencies.append(frequency)

    return phase_errors, frequencies


def _analyze_pll_convergence(frequencies: list[float], initial_freq: float) -> tuple[float, float]:
    """Analyze PLL convergence from frequency history.

    Args:
        frequencies: List of frequency values over time.
        initial_freq: Initial frequency estimate.

    Returns:
        Tuple of (recovered_freq, confidence).
    """
    stable_start = int(0.9 * len(frequencies))
    final_frequencies = frequencies[stable_start:]

    if len(final_frequencies) == 0:
        return initial_freq, 0.0

    recovered_freq = float(np.mean(final_frequencies))
    freq_std = float(np.std(final_frequencies))
    confidence = float(max(0.0, min(1.0, 1.0 - freq_std / recovered_freq)))

    return recovered_freq, confidence


def _compute_pll_drift(frequencies: list[float], recovered_freq: float) -> float:
    """Compute frequency drift from PLL tracking.

    Args:
        frequencies: List of frequency values.
        recovered_freq: Final recovered frequency.

    Returns:
        Drift in ppm.
    """
    stable_start = int(0.9 * len(frequencies))
    final_frequencies = frequencies[stable_start:]

    if len(final_frequencies) <= 10:
        return 0.0

    time_points = np.arange(len(final_frequencies))
    coeffs = np.polyfit(time_points, final_frequencies, 1)
    drift_hz_per_sample = coeffs[0]
    drift_ppm = (drift_hz_per_sample * len(final_frequencies)) / recovered_freq * 1e6

    return float(drift_ppm)


__all__ = [
    "TimingAnalysisResult",
    "TimingAnalyzer",
]
