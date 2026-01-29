"""Time-varying and multi-level threshold support for digital signal analysis.

    - RE-THR-001: Time-Varying Threshold Support
    - RE-THR-002: Multi-Level Logic Support

This module provides adaptive thresholding for signals with varying DC offset
or amplitude, and support for multi-level logic standards beyond simple
high/low states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ThresholdConfig:
    """Configuration for threshold detection.

    Implements RE-THR-001, RE-THR-002: Threshold configuration.

    Attributes:
        threshold_type: Type of thresholding ('fixed', 'adaptive', 'multi_level').
        fixed_threshold: Fixed threshold value (for 'fixed' type).
        window_size: Window size for adaptive thresholding.
        percentile: Percentile for adaptive threshold calculation.
        levels: Voltage levels for multi-level logic.
        hysteresis: Hysteresis margin to prevent oscillation.
    """

    threshold_type: Literal["fixed", "adaptive", "multi_level"] = "fixed"
    fixed_threshold: float = 0.5
    window_size: int = 1024
    percentile: float = 50.0
    levels: list[float] = field(default_factory=lambda: [0.0, 1.0])
    hysteresis: float = 0.05


@dataclass
class AdaptiveThresholdResult:
    """Result of adaptive threshold calculation.

    Implements RE-THR-001: Time-varying threshold result.

    Attributes:
        thresholds: Array of threshold values at each sample.
        binary_output: Digitized signal.
        crossings: Indices of threshold crossings.
        dc_offset: Estimated DC offset over time.
        amplitude: Estimated signal amplitude over time.
    """

    thresholds: NDArray[np.float64]
    binary_output: NDArray[np.uint8]
    crossings: list[int]
    dc_offset: NDArray[np.float64]
    amplitude: NDArray[np.float64]


@dataclass
class MultiLevelResult:
    """Result of multi-level logic detection.

    Implements RE-THR-002: Multi-level detection result.

    Attributes:
        levels: Detected logic levels at each sample.
        level_values: Voltage levels used.
        transitions: List of (index, from_level, to_level) transitions.
        level_histogram: Count of samples at each level.
        eye_heights: Eye height for each transition.
    """

    levels: NDArray[np.int32]
    level_values: list[float]
    transitions: list[tuple[int, int, int]]
    level_histogram: dict[int, int]
    eye_heights: list[float]


class AdaptiveThresholder:
    """Apply time-varying thresholds to signals.

    Implements RE-THR-001: Time-Varying Threshold Support.

    Tracks DC offset and amplitude changes to maintain accurate
    thresholding despite signal drift.

    Example:
        >>> thresholder = AdaptiveThresholder(window_size=1000)
        >>> result = thresholder.apply(analog_signal)
        >>> digital = result.binary_output
    """

    def __init__(
        self,
        window_size: int = 1024,
        percentile: float = 50.0,
        method: Literal["median", "mean", "envelope", "otsu"] = "median",
        hysteresis: float = 0.05,
    ) -> None:
        """Initialize adaptive thresholder.

        Args:
            window_size: Size of sliding window for adaptation.
            percentile: Percentile for threshold calculation.
            method: Thresholding method.
            hysteresis: Hysteresis margin. This is used as an absolute value
                when amplitude-relative calculation would be too small. For
                signals with small amplitude variations (e.g., oscillating
                around a threshold), this value is applied directly.
        """
        self.window_size = window_size
        self.percentile = percentile
        self.method = method
        self.hysteresis = hysteresis

    def apply(self, signal: NDArray[np.float64]) -> AdaptiveThresholdResult:
        """Apply adaptive thresholding to signal.

        Implements RE-THR-001: Adaptive threshold application.

        Args:
            signal: Input analog signal.

        Returns:
            AdaptiveThresholdResult with thresholds and digitized output.

        Example:
            >>> result = thresholder.apply(analog_waveform)
            >>> plt.plot(result.binary_output)
        """
        n_samples = len(signal)

        # Estimate DC offset and amplitude over time
        dc_offset = np.zeros(n_samples)
        amplitude = np.zeros(n_samples)
        thresholds = np.zeros(n_samples)

        half_window = self.window_size // 2

        for i in range(n_samples):
            # Window bounds
            start = max(0, i - half_window)
            end = min(n_samples, i + half_window)
            window = signal[start:end]

            if self.method == "median":
                dc_offset[i] = np.median(window)
                amplitude[i] = np.percentile(window, 95) - np.percentile(window, 5)
                thresholds[i] = dc_offset[i]

            elif self.method == "mean":
                dc_offset[i] = np.mean(window)
                amplitude[i] = np.std(window) * 4  # Approximate peak-to-peak
                thresholds[i] = dc_offset[i]

            elif self.method == "envelope":
                # Use min/max envelope
                high = np.max(window)
                low = np.min(window)
                dc_offset[i] = (high + low) / 2
                amplitude[i] = high - low
                thresholds[i] = dc_offset[i]

            elif self.method == "otsu":
                # Simplified Otsu's method
                threshold = self._otsu_threshold(window)
                thresholds[i] = threshold
                dc_offset[i] = threshold
                amplitude[i] = np.max(window) - np.min(window)

        # Apply hysteresis
        binary_output, crossings = self._apply_with_hysteresis(signal, thresholds, amplitude)

        return AdaptiveThresholdResult(
            thresholds=thresholds,
            binary_output=binary_output,
            crossings=crossings,
            dc_offset=dc_offset,
            amplitude=amplitude,
        )

    def calculate_threshold_profile(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate threshold values without applying.

        Implements RE-THR-001: Threshold profile calculation.

        Args:
            signal: Input signal.

        Returns:
            Array of threshold values.
        """
        result = self.apply(signal)
        return result.thresholds

    def _apply_with_hysteresis(
        self,
        signal: NDArray[np.float64],
        thresholds: NDArray[np.float64],
        amplitude: NDArray[np.float64],
    ) -> tuple[NDArray[np.uint8], list[int]]:
        """Apply thresholding with hysteresis.

        The hysteresis prevents rapid oscillation when the signal hovers near
        the threshold. The margin is calculated as:
        - If amplitude is significant: hyst_margin = amplitude * hysteresis
        - If amplitude is small: hyst_margin = hysteresis (used as absolute value)

        This ensures hysteresis remains effective even for signals with very
        small amplitude variations.

        Args:
            signal: Input signal.
            thresholds: Threshold values.
            amplitude: Signal amplitude at each point.

        Returns:
            Tuple of (binary_output, crossings).
        """
        n_samples = len(signal)
        binary = np.zeros(n_samples, dtype=np.uint8)
        crossings = []

        # Initial state
        current_state = 1 if signal[0] > thresholds[0] else 0
        binary[0] = current_state

        for i in range(1, n_samples):
            threshold = thresholds[i]
            amp = amplitude[i]

            # Calculate hysteresis margin:
            # - Use amplitude-relative margin for signals with significant amplitude
            # - Use absolute hysteresis value when amplitude is small
            # This prevents oscillation for signals hovering around the threshold
            amplitude_relative_margin = amp * self.hysteresis
            absolute_margin = self.hysteresis

            # Use the larger of the two to ensure effective hysteresis
            # When amplitude is large (e.g., > 1.0), amplitude-relative dominates
            # When amplitude is small (e.g., 0.02), absolute hysteresis dominates
            hyst_margin = max(amplitude_relative_margin, absolute_margin)

            if current_state == 0:
                # Currently low, need signal above threshold + hysteresis to go high
                if signal[i] > threshold + hyst_margin:
                    current_state = 1
                    crossings.append(i)
            else:
                # Currently high, need signal below threshold - hysteresis to go low
                if signal[i] < threshold - hyst_margin:
                    current_state = 0
                    crossings.append(i)

            binary[i] = current_state

        return binary, crossings

    def _otsu_threshold(self, data: NDArray[np.float64]) -> float:
        """Calculate Otsu's threshold.

        Args:
            data: Data window.

        Returns:
            Optimal threshold value.
        """
        # Simplified Otsu's method
        hist, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total = hist.sum()
        if total == 0:
            return float(np.mean(data))

        current_max = 0
        threshold = bin_centers[0]

        sum_total = np.sum(bin_centers * hist)
        sum_background = 0
        weight_background = 0

        for i in range(len(hist)):
            weight_background += hist[i]
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += bin_centers[i] * hist[i]

            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance_between = (
                weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            )

            if variance_between > current_max:
                current_max = variance_between
                threshold = bin_centers[i]

        return float(threshold)


class MultiLevelDetector:
    """Detect multi-level logic signals.

    Implements RE-THR-002: Multi-Level Logic Support.

    Supports PAM-2, PAM-4, PAM-8, and custom multi-level signaling
    where signals encode multiple bits per symbol.

    Example:
        >>> detector = MultiLevelDetector(levels=4)  # PAM-4
        >>> result = detector.detect(signal)
        >>> symbols = result.levels
    """

    def __init__(
        self,
        levels: int | list[float] = 2,
        auto_detect_levels: bool = True,
        hysteresis: float = 0.1,
    ) -> None:
        """Initialize multi-level detector.

        Args:
            levels: Number of levels or explicit voltage levels.
            auto_detect_levels: Automatically detect level voltages.
            hysteresis: Hysteresis fraction between levels.
        """
        if isinstance(levels, int):
            self.n_levels = levels
            self.level_values = None
        else:
            self.n_levels = len(levels)
            self.level_values = sorted(levels)

        self.auto_detect_levels = auto_detect_levels
        self.hysteresis = hysteresis

    def detect(self, signal: NDArray[np.float64]) -> MultiLevelResult:
        """Detect multi-level logic in signal.

        Implements RE-THR-002: Multi-level detection workflow.

        Args:
            signal: Input analog signal.

        Returns:
            MultiLevelResult with detected levels.

        Example:
            >>> result = detector.detect(pam4_signal)
            >>> print(f"Detected {len(result.level_values)} levels")
        """
        # Auto-detect level values if needed
        if self.level_values is None or self.auto_detect_levels:
            level_values = self._detect_levels(signal)
        else:
            level_values = self.level_values

        # Calculate decision thresholds between levels
        thresholds = [
            (level_values[i] + level_values[i + 1]) / 2 for i in range(len(level_values) - 1)
        ]

        # Apply hysteresis-aware level detection
        levels, transitions = self._detect_with_hysteresis(signal, level_values, thresholds)

        # Calculate level histogram
        level_histogram = {}
        for level in range(len(level_values)):
            level_histogram[level] = int(np.sum(levels == level))

        # Calculate eye heights
        eye_heights = self._calculate_eye_heights(signal, level_values)

        return MultiLevelResult(
            levels=levels,
            level_values=level_values,
            transitions=transitions,
            level_histogram=level_histogram,
            eye_heights=eye_heights,
        )

    def detect_levels_from_histogram(
        self, signal: NDArray[np.float64], n_levels: int | None = None
    ) -> list[float]:
        """Detect logic levels from signal histogram.

        Implements RE-THR-002: Level detection.

        Args:
            signal: Input signal.
            n_levels: Expected number of levels (auto-detect if None).

        Returns:
            List of detected voltage levels.
        """
        if n_levels is None:
            n_levels = self.n_levels

        return self._detect_levels(signal, n_levels)

    def calculate_eye_diagram(
        self,
        signal: NDArray[np.float64],
        samples_per_symbol: int,
        n_symbols: int = 100,
    ) -> NDArray[np.float64]:
        """Calculate eye diagram data for multi-level signal.

        Implements RE-THR-002: Eye diagram support.

        Args:
            signal: Input signal.
            samples_per_symbol: Samples per symbol period.
            n_symbols: Number of symbols to overlay.

        Returns:
            2D array of overlaid symbol waveforms.
        """
        n_available = len(signal) // samples_per_symbol
        n_symbols = min(n_symbols, n_available)

        # Create 2D array with overlaid symbols
        eye_data = np.zeros((n_symbols, samples_per_symbol * 2))

        for i in range(n_symbols):
            start = i * samples_per_symbol
            end = start + samples_per_symbol * 2

            if end <= len(signal):
                eye_data[i] = signal[start:end]

        return eye_data

    def _detect_levels(
        self, signal: NDArray[np.float64], n_levels: int | None = None
    ) -> list[float]:
        """Detect voltage levels using clustering.

        Args:
            signal: Input signal.
            n_levels: Expected number of levels.

        Returns:
            List of level voltage values.
        """
        if n_levels is None:
            n_levels = self.n_levels

        # Use histogram-based clustering
        hist, bin_edges = np.histogram(signal, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in histogram
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((hist[i], bin_centers[i]))

        # Sort by frequency and take top n_levels
        peaks.sort(reverse=True)
        level_values = sorted([p[1] for p in peaks[:n_levels]])

        # If not enough peaks found, use evenly spaced levels
        if len(level_values) < n_levels:
            min_val = np.min(signal)
            max_val = np.max(signal)
            level_values = list(np.linspace(min_val, max_val, n_levels))

        return level_values

    def _detect_with_hysteresis(
        self,
        signal: NDArray[np.float64],
        level_values: list[float],
        thresholds: list[float],
    ) -> tuple[NDArray[np.int32], list[tuple[int, int, int]]]:
        """Detect levels with hysteresis.

        Args:
            signal: Input signal.
            level_values: Voltage levels.
            thresholds: Decision thresholds.

        Returns:
            Tuple of (level_array, transitions).
        """
        n_samples = len(signal)
        levels = np.zeros(n_samples, dtype=np.int32)
        transitions = []

        # Calculate hysteresis margins
        margins = []
        for i in range(len(level_values) - 1):
            margin = (level_values[i + 1] - level_values[i]) * self.hysteresis
            margins.append(margin)

        # Initial level
        current_level = self._find_closest_level(signal[0], level_values)
        levels[0] = current_level

        for i in range(1, n_samples):
            new_level = current_level

            # Check for transitions
            if current_level < len(level_values) - 1:
                # Can go up
                upper_threshold = thresholds[current_level] + margins[current_level]
                if signal[i] > upper_threshold:
                    new_level = current_level + 1

            if current_level > 0:
                # Can go down
                lower_threshold = thresholds[current_level - 1] - margins[current_level - 1]
                if signal[i] < lower_threshold:
                    new_level = current_level - 1

            if new_level != current_level:
                transitions.append((i, current_level, new_level))
                current_level = new_level

            levels[i] = current_level

        return levels, transitions

    def _find_closest_level(self, value: float, level_values: list[float]) -> int:
        """Find closest level to value.

        Args:
            value: Sample value.
            level_values: Level voltages.

        Returns:
            Level index.
        """
        distances = [abs(value - lv) for lv in level_values]
        return int(np.argmin(distances))

    def _calculate_eye_heights(
        self, signal: NDArray[np.float64], level_values: list[float]
    ) -> list[float]:
        """Calculate eye heights between levels.

        Args:
            signal: Input signal.
            level_values: Level voltages.

        Returns:
            List of eye heights for each level transition.
        """
        eye_heights = []

        for i in range(len(level_values) - 1):
            lower = level_values[i]
            upper = level_values[i + 1]

            # Find samples near each level
            lower_samples = signal[np.abs(signal - lower) < (upper - lower) * 0.2]
            upper_samples = signal[np.abs(signal - upper) < (upper - lower) * 0.2]

            if len(lower_samples) > 0 and len(upper_samples) > 0:
                # Eye height is gap between worst cases
                worst_low = np.max(lower_samples)
                worst_high = np.min(upper_samples)
                eye_height = worst_high - worst_low
            else:
                eye_height = upper - lower

            eye_heights.append(max(0, eye_height))

        return eye_heights


# =============================================================================
# Convenience functions
# =============================================================================


def apply_adaptive_threshold(
    signal: NDArray[np.float64],
    window_size: int = 1024,
    method: Literal["median", "mean", "envelope", "otsu"] = "median",
    hysteresis: float = 0.05,
) -> AdaptiveThresholdResult:
    """Apply adaptive thresholding to a signal.

    Implements RE-THR-001: Time-Varying Threshold Support.

    Args:
        signal: Input analog signal.
        window_size: Adaptive window size.
        method: Thresholding method.
        hysteresis: Hysteresis margin.

    Returns:
        AdaptiveThresholdResult with thresholds and binary output.

    Example:
        >>> result = apply_adaptive_threshold(noisy_signal)
        >>> digital = result.binary_output
    """
    thresholder = AdaptiveThresholder(
        window_size=window_size,
        method=method,
        hysteresis=hysteresis,
    )
    return thresholder.apply(signal)


def detect_multi_level(
    signal: NDArray[np.float64],
    n_levels: int = 4,
    auto_detect: bool = True,
    hysteresis: float = 0.1,
) -> MultiLevelResult:
    """Detect multi-level logic in signal.

    Implements RE-THR-002: Multi-Level Logic Support.

    Args:
        signal: Input analog signal.
        n_levels: Expected number of levels.
        auto_detect: Automatically detect level voltages.
        hysteresis: Hysteresis between levels.

    Returns:
        MultiLevelResult with detected levels.

    Example:
        >>> result = detect_multi_level(pam4_signal, n_levels=4)
        >>> symbols = result.levels
    """
    detector = MultiLevelDetector(
        levels=n_levels,
        auto_detect_levels=auto_detect,
        hysteresis=hysteresis,
    )
    return detector.detect(signal)


def calculate_threshold_snr(
    signal: NDArray[np.float64],
    threshold: float | NDArray[np.float64],
) -> float:
    """Calculate signal-to-noise ratio at threshold.

    Implements RE-THR-001: Threshold quality metric.

    Args:
        signal: Input signal.
        threshold: Threshold value(s).

    Returns:
        Estimated SNR in dB.
    """
    if isinstance(threshold, np.ndarray):
        threshold = float(np.mean(threshold))

    # Separate high and low samples
    high_samples = signal[signal > threshold]
    low_samples = signal[signal <= threshold]

    if len(high_samples) == 0 or len(low_samples) == 0:
        return 0.0

    # Calculate signal power (difference between means)
    signal_power = (np.mean(high_samples) - np.mean(low_samples)) ** 2

    # Calculate noise power (variance around means)
    noise_power = (np.var(high_samples) + np.var(low_samples)) / 2

    if noise_power == 0:
        return 100.0  # Very high SNR

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return float(snr_db)


__all__ = [
    "AdaptiveThresholdResult",
    # Classes
    "AdaptiveThresholder",
    "MultiLevelDetector",
    "MultiLevelResult",
    # Data classes
    "ThresholdConfig",
    # Functions
    "apply_adaptive_threshold",
    "calculate_threshold_snr",
    "detect_multi_level",
]
