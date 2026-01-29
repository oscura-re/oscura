"""Signal quality and integrity analysis.

This module provides comprehensive signal integrity analysis for digital signals,
including noise margin measurements, transition characterization, overshoot/
undershoot detection, and ringing analysis.


Example:
    >>> import numpy as np
    >>> from oscura.analyzers.digital.signal_quality import SignalQualityAnalyzer
    >>> # Generate test signal
    >>> signal = np.concatenate([np.zeros(100), np.ones(100)])
    >>> analyzer = SignalQualityAnalyzer(sample_rate=100e6, logic_family='TTL')
    >>> report = analyzer.analyze(signal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Logic family thresholds (from existing extraction.py)
LOGIC_THRESHOLDS = {
    "ttl": {"VIL": 0.8, "VIH": 2.0, "VOL": 0.4, "VOH": 2.4, "VCC": 5.0},
    "cmos": {"VIL": 1.5, "VIH": 3.5, "VOL": 0.1, "VOH": 4.9, "VCC": 5.0},
    "lvttl": {"VIL": 0.8, "VIH": 1.5, "VOL": 0.4, "VOH": 2.4, "VCC": 3.3},
    "lvcmos": {"VIL": 0.99, "VIH": 2.31, "VOL": 0.1, "VOH": 3.2, "VCC": 3.3},
}


@dataclass
class NoiseMargins:
    """Noise margins for digital signal.

    Attributes:
        high_margin: Distance from threshold to logic high level (V).
        low_margin: Distance from threshold to logic low level (V).
        high_mean: Mean high level voltage.
        low_mean: Mean low level voltage.
        high_std: Standard deviation of high level (noise).
        low_std: Standard deviation of low level (noise).
        threshold: Detection threshold used.
    """

    high_margin: float  # Distance from threshold to logic high
    low_margin: float  # Distance from threshold to logic low
    high_mean: float  # Mean high level
    low_mean: float  # Mean low level
    high_std: float  # High level noise
    low_std: float  # Low level noise
    threshold: float  # Detection threshold


@dataclass
class TransitionMetrics:
    """Metrics for signal transitions.

    Attributes:
        rise_time: 10-90% rise time in seconds.
        fall_time: 90-10% fall time in seconds.
        slew_rate_rising: Rising edge slew rate (V/s).
        slew_rate_falling: Falling edge slew rate (V/s).
        overshoot: Overshoot as percentage of signal swing.
        undershoot: Undershoot as percentage of signal swing.
        ringing_frequency: Ringing frequency in Hz (None if no ringing).
        ringing_amplitude: Ringing amplitude in volts (None if no ringing).
        settling_time: Time to settle within tolerance (None if not measured).
    """

    rise_time: float  # 10-90% rise time
    fall_time: float  # 90-10% fall time
    slew_rate_rising: float
    slew_rate_falling: float
    overshoot: float  # Percentage overshoot
    undershoot: float  # Percentage undershoot
    ringing_frequency: float | None = None
    ringing_amplitude: float | None = None
    settling_time: float | None = None


@dataclass
class SignalIntegrityReport:
    """Complete signal integrity report.

    Attributes:
        noise_margins: Noise margin measurements.
        transitions: Transition quality metrics.
        snr_db: Signal-to-noise ratio in dB.
        signal_quality: Overall quality assessment.
        issues: List of detected issues.
        recommendations: List of recommendations for improvement.
    """

    noise_margins: NoiseMargins
    transitions: TransitionMetrics
    snr_db: float
    signal_quality: Literal["excellent", "good", "fair", "poor"]
    issues: list[str]
    recommendations: list[str]


@dataclass
class SimpleQualityMetrics:
    """Simplified quality metrics for test compatibility.

    Provides a flat interface with direct attribute access for common metrics.

    Attributes:
        noise_margin_low: Low-side noise margin in volts.
        noise_margin_high: High-side noise margin in volts.
        rise_time: Rise time in samples (or seconds depending on context).
        fall_time: Fall time in samples (or seconds depending on context).
        has_overshoot: Whether overshoot was detected.
        max_overshoot: Maximum overshoot value in volts.
        duty_cycle: Signal duty cycle (0.0 to 1.0).
    """

    noise_margin_low: float
    noise_margin_high: float
    rise_time: float
    fall_time: float
    has_overshoot: bool
    max_overshoot: float
    duty_cycle: float


class SignalQualityAnalyzer:
    """Analyze digital signal quality and integrity.

    Provides comprehensive signal integrity analysis including noise margins,
    transition metrics, overshoot/undershoot, and ringing detection.

    Supports two initialization modes:
    1. Full mode: SignalQualityAnalyzer(sample_rate=1e9, logic_family='TTL')
    2. Simple mode: SignalQualityAnalyzer(v_il=0.8, v_ih=2.0) - for test compatibility

    Attributes:
        sample_rate: Sample rate of input signals in Hz.
        logic_family: Logic family for threshold determination.
        v_il: Input low threshold voltage.
        v_ih: Input high threshold voltage.
        vdd: Supply voltage for overshoot reference.

    Example:
        >>> analyzer = SignalQualityAnalyzer(sample_rate=1e9, logic_family='TTL')
        >>> report = analyzer.analyze(signal_trace)
    """

    def __init__(
        self,
        sample_rate: float | None = None,
        logic_family: str = "auto",
        v_il: float | None = None,
        v_ih: float | None = None,
        vdd: float | None = None,
    ):
        """Initialize analyzer.

        Args:
            sample_rate: Sample rate in Hz (optional for simple mode).
            logic_family: Logic family ('TTL', 'CMOS', 'LVTTL', 'LVCMOS', 'auto').
            v_il: Input low threshold voltage (for simple mode).
            v_ih: Input high threshold voltage (for simple mode).
            vdd: Supply voltage for overshoot reference (for simple mode).

        Raises:
            ValueError: If sample rate is invalid (when provided).
        """
        # Simple mode: thresholds provided directly
        self.v_il = v_il
        self.v_ih = v_ih
        self.vdd = vdd

        # Full mode: sample rate and logic family
        if sample_rate is not None:
            if sample_rate <= 0:
                raise ValueError(f"Sample rate must be positive, got {sample_rate}")
            self.sample_rate = sample_rate
            self._time_base = 1.0 / sample_rate
        else:
            # Default sample rate for simple mode (samples per second = 1)
            self.sample_rate = 1.0
            self._time_base = 1.0

        self.logic_family = logic_family.lower() if logic_family else "auto"

        # If thresholds provided, use them to determine logic family settings
        self._threshold: float | None
        if v_il is not None and v_ih is not None:
            self._threshold = (v_il + v_ih) / 2.0
        else:
            self._threshold = None

    def analyze(
        self, trace: NDArray[np.float64], clock_trace: NDArray[np.float64] | None = None
    ) -> Any:
        """Perform complete signal integrity analysis.

        Returns SimpleQualityMetrics in simple mode (when v_il/v_ih provided),
        or SignalIntegrityReport in full mode.

        Args:
            trace: Input signal trace (analog voltage values).
            clock_trace: Optional clock signal for synchronized analysis.

        Returns:
            SimpleQualityMetrics or SignalIntegrityReport with analysis results.

        Example:
            >>> report = analyzer.analyze(signal_trace)
            >>> print(f"Signal quality: {report.signal_quality}")
        """
        trace = np.asarray(trace, dtype=np.float64)

        # Simple mode: return SimpleQualityMetrics
        if self.v_il is not None or self.v_ih is not None or self.vdd is not None:
            return self._analyze_simple(trace)

        # Full mode: return SignalIntegrityReport
        return self._analyze_full(trace, clock_trace)

    def _analyze_simple(self, trace: NDArray[np.float64]) -> SimpleQualityMetrics:
        """Simple analysis mode returning flat metrics.

        Args:
            trace: Input signal trace.

        Returns:
            SimpleQualityMetrics with measured values.
        """
        # Determine threshold
        threshold: float
        if self._threshold is not None:
            threshold = self._threshold
        else:
            threshold = float((np.max(trace) + np.min(trace)) / 2.0)

        # Separate high and low samples
        high_samples = trace[trace > threshold]
        low_samples = trace[trace <= threshold]

        # Calculate noise margins
        if len(high_samples) > 0:
            high_mean = np.mean(high_samples)
            if self.v_ih is not None:
                noise_margin_high = high_mean - self.v_ih
            else:
                noise_margin_high = high_mean - threshold
        else:
            noise_margin_high = 0.0

        if len(low_samples) > 0:
            low_mean = np.mean(low_samples)
            if self.v_il is not None:
                noise_margin_low = self.v_il - low_mean
            else:
                noise_margin_low = threshold - low_mean
        else:
            noise_margin_low = 0.0

        # Measure rise/fall times in samples
        rise_time, fall_time = self._measure_rise_fall_samples(trace, threshold)

        # Detect overshoot
        has_overshoot, max_overshoot = self._detect_overshoot_simple(trace)

        # Calculate duty cycle
        duty_cycle = self._calculate_duty_cycle(trace, threshold)

        return SimpleQualityMetrics(
            noise_margin_low=float(noise_margin_low),
            noise_margin_high=float(noise_margin_high),
            rise_time=float(rise_time),
            fall_time=float(fall_time),
            has_overshoot=has_overshoot,
            max_overshoot=float(max_overshoot),
            duty_cycle=float(duty_cycle),
        )

    def _measure_rise_fall_samples(
        self, trace: NDArray[np.float64], threshold: float
    ) -> tuple[float, float]:
        """Measure rise and fall times in samples.

        Args:
            trace: Input signal trace.
            threshold: Detection threshold.

        Returns:
            Tuple of (rise_time_samples, fall_time_samples).
        """
        # Detect edges
        crossings = np.diff((trace > threshold).astype(int))
        rising_edges = np.where(crossings > 0)[0]
        falling_edges = np.where(crossings < 0)[0]

        # Measure rise times
        rise_times = []
        for edge_idx in rising_edges:
            window_size = min(10, edge_idx, len(trace) - edge_idx - 1)
            if window_size < 2:
                continue

            window = trace[edge_idx - window_size : edge_idx + window_size + 1]
            v_min = np.min(window)
            v_max = np.max(window)

            if v_max - v_min < 1e-6:
                continue

            # Find 10% and 90% points
            v_10 = v_min + 0.1 * (v_max - v_min)
            v_90 = v_min + 0.9 * (v_max - v_min)

            idx_10 = np.where(window >= v_10)[0]
            idx_90 = np.where(window >= v_90)[0]

            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = idx_90[0] - idx_10[0]
                if rise_time > 0:
                    rise_times.append(rise_time)

        # Measure fall times
        fall_times = []
        for edge_idx in falling_edges:
            window_size = min(10, edge_idx, len(trace) - edge_idx - 1)
            if window_size < 2:
                continue

            window = trace[edge_idx - window_size : edge_idx + window_size + 1]
            v_min = np.min(window)
            v_max = np.max(window)

            if v_max - v_min < 1e-6:
                continue

            v_90 = v_min + 0.9 * (v_max - v_min)
            v_10 = v_min + 0.1 * (v_max - v_min)

            idx_90 = np.where(window <= v_90)[0]
            idx_10 = np.where(window <= v_10)[0]

            if len(idx_90) > 0 and len(idx_10) > 0:
                fall_time = idx_10[-1] - idx_90[0]
                if fall_time > 0:
                    fall_times.append(fall_time)

        rise_time = np.mean(rise_times) if rise_times else 0.0
        fall_time = np.mean(fall_times) if fall_times else 0.0

        return rise_time, fall_time

    def _detect_overshoot_simple(self, trace: NDArray[np.float64]) -> tuple[bool, float]:
        """Detect overshoot in simple mode.

        Args:
            trace: Input signal trace.

        Returns:
            Tuple of (has_overshoot, max_overshoot_value).
        """
        threshold = self._threshold or (np.max(trace) + np.min(trace)) / 2.0
        high_samples = trace[trace > threshold]

        if len(high_samples) == 0:
            return False, 0.0

        high_median = np.median(high_samples)
        max_val = np.max(trace)

        # Check if max exceeds expected high level
        if self.vdd is not None:
            # Check against VDD
            overshoot = float(max_val - self.vdd)
            has_overshoot = overshoot > 0.05  # 50mV threshold
        else:
            # Check against median high level
            overshoot = float(max_val - high_median)
            # Only count as overshoot if significantly above stable level
            _high_level = high_median
            signal_swing = high_median - np.min(trace)
            has_overshoot = overshoot > float(signal_swing * 0.05)  # 5% threshold

        return bool(has_overshoot), max(0.0, overshoot)

    def _calculate_duty_cycle(self, trace: NDArray[np.float64], threshold: float) -> float:
        """Calculate signal duty cycle.

        Args:
            trace: Input signal trace.
            threshold: Detection threshold.

        Returns:
            Duty cycle as ratio (0.0 to 1.0).
        """
        if len(trace) == 0:
            return 0.0

        # Handle boolean trace
        if trace.dtype == np.bool_:
            high_count = np.sum(trace)
        else:
            high_count = np.sum(trace > threshold)

        return float(high_count) / float(len(trace))

    def _analyze_full(
        self, trace: NDArray[np.float64], clock_trace: NDArray[np.float64] | None = None
    ) -> SignalIntegrityReport:
        """Full analysis mode returning comprehensive report.

        Args:
            trace: Input signal trace.
            clock_trace: Optional clock signal.

        Returns:
            SignalIntegrityReport with complete analysis.
        """
        # Measure noise margins
        logic_fam: Literal["ttl", "cmos", "lvttl", "lvcmos", "auto"]
        if self.logic_family in ("ttl", "cmos", "lvttl", "lvcmos", "auto"):
            logic_fam = self.logic_family  # type: ignore[assignment]
        else:
            logic_fam = "auto"
        noise_margins = self.measure_noise_margins(trace, logic_fam)

        # Measure transitions
        transitions = self.measure_transitions(trace)

        # Calculate SNR
        snr_db = self.calculate_snr(trace)

        # Assess overall quality and identify issues
        issues = []
        recommendations = []

        # Check noise margins
        if noise_margins.high_margin < 0.4:
            issues.append("Insufficient high-level noise margin")
            recommendations.append("Increase signal high level or reduce noise")

        if noise_margins.low_margin < 0.4:
            issues.append("Insufficient low-level noise margin")
            recommendations.append("Decrease signal low level or reduce noise")

        # Check transitions
        if transitions.overshoot > 20:
            issues.append(f"Excessive overshoot: {transitions.overshoot:.1f}%")
            recommendations.append("Add series termination or reduce capacitance")

        if transitions.undershoot > 20:
            issues.append(f"Excessive undershoot: {transitions.undershoot:.1f}%")
            recommendations.append("Check ground connections and reduce inductance")

        if transitions.ringing_amplitude and transitions.ringing_amplitude > 0.2:
            issues.append("Significant ringing detected")
            recommendations.append("Add damping resistor or improve impedance matching")

        # Check SNR
        if snr_db < 20:
            issues.append(f"Low SNR: {snr_db:.1f} dB")
            recommendations.append("Reduce noise sources or improve shielding")

        # Determine overall quality
        quality: Literal["excellent", "good", "fair", "poor"]
        if len(issues) == 0 and snr_db > 40:
            quality = "excellent"
        elif len(issues) <= 1 and snr_db > 30:
            quality = "good"
        elif len(issues) <= 2 and snr_db > 20:
            quality = "fair"
        else:
            quality = "poor"

        return SignalIntegrityReport(
            noise_margins=noise_margins,
            transitions=transitions,
            snr_db=snr_db,
            signal_quality=quality,
            issues=issues,
            recommendations=recommendations,
        )

    def measure_noise_margins(
        self,
        trace: NDArray[np.float64],
        logic_family: Literal["ttl", "cmos", "lvttl", "lvcmos", "auto"] = "auto",
    ) -> NoiseMargins:
        """Measure noise margins for high and low states.

        Args:
            trace: Input signal trace (analog voltage values).
            logic_family: Logic family for threshold determination.

        Returns:
            NoiseMargins object with measured margins.

        Example:
            >>> margins = analyzer.measure_noise_margins(trace, logic_family='TTL')
        """
        trace = np.asarray(trace)

        # Determine threshold
        if logic_family == "auto":
            # Auto-detect based on signal range
            signal_range = np.max(trace) - np.min(trace)
            if signal_range > 4.0:
                logic_family = "ttl"  # 5V logic
            elif signal_range > 2.5:
                logic_family = "lvttl"  # 3.3V logic
            else:
                logic_family = "lvcmos"  # Low voltage

        # Get thresholds for logic family
        thresholds = LOGIC_THRESHOLDS.get(logic_family, LOGIC_THRESHOLDS["ttl"])
        threshold = (thresholds["VIL"] + thresholds["VIH"]) / 2.0

        # Separate high and low samples
        high_samples = trace[trace > threshold]
        low_samples = trace[trace <= threshold]

        # Calculate statistics
        if len(high_samples) > 0:
            high_mean = np.mean(high_samples)
            high_std = np.std(high_samples)
            high_margin = high_mean - threshold
        else:
            high_mean = 0.0
            high_std = 0.0
            high_margin = 0.0

        if len(low_samples) > 0:
            low_mean = np.mean(low_samples)
            low_std = np.std(low_samples)
            low_margin = threshold - low_mean
        else:
            low_mean = 0.0
            low_std = 0.0
            low_margin = 0.0

        return NoiseMargins(
            high_margin=float(high_margin),
            low_margin=float(low_margin),
            high_mean=float(high_mean),
            low_mean=float(low_mean),
            high_std=float(high_std),
            low_std=float(low_std),
            threshold=float(threshold),
        )

    def measure_transitions(self, trace: NDArray[np.float64]) -> TransitionMetrics:
        """Measure transition characteristics.

        Analyzes rising and falling edges to measure rise/fall times,
        slew rates, overshoot, undershoot, and ringing.

        Args:
            trace: Input signal trace (analog voltage values).

        Returns:
            TransitionMetrics object with transition measurements.

        Example:
            >>> metrics = analyzer.measure_transitions(trace)
        """
        trace = np.asarray(trace)

        # Find threshold crossings
        threshold = (np.max(trace) + np.min(trace)) / 2.0
        signal_range = np.max(trace) - np.min(trace)

        # Detect edges (simple threshold crossing)
        crossings = np.diff((trace > threshold).astype(int))
        rising_edges = np.where(crossings > 0)[0]
        falling_edges = np.where(crossings < 0)[0]

        # Measure rise time (10-90%)
        rise_times = []
        for edge_idx in rising_edges:
            if edge_idx > 10 and edge_idx < len(trace) - 10:
                # Get window around edge
                window = trace[edge_idx - 10 : edge_idx + 10]
                v_min = np.min(window)
                v_max = np.max(window)

                # Find 10% and 90% points
                v_10 = v_min + 0.1 * (v_max - v_min)
                v_90 = v_min + 0.9 * (v_max - v_min)

                # Find sample indices
                idx_10 = np.where(window >= v_10)[0]
                idx_90 = np.where(window >= v_90)[0]

                if len(idx_10) > 0 and len(idx_90) > 0:
                    rise_time = (idx_90[0] - idx_10[0]) * self._time_base
                    rise_times.append(rise_time)

        # Measure fall time (90-10%)
        fall_times = []
        for edge_idx in falling_edges:
            if edge_idx > 10 and edge_idx < len(trace) - 10:
                window = trace[edge_idx - 10 : edge_idx + 10]
                v_min = np.min(window)
                v_max = np.max(window)

                v_90 = v_min + 0.9 * (v_max - v_min)
                v_10 = v_min + 0.1 * (v_max - v_min)

                idx_90 = np.where(window <= v_90)[0]
                idx_10 = np.where(window <= v_10)[0]

                if len(idx_90) > 0 and len(idx_10) > 0:
                    fall_time = (idx_10[-1] - idx_90[0]) * self._time_base
                    fall_times.append(fall_time)

        # Calculate average times
        rise_time = np.mean(rise_times) if rise_times else 0.0
        fall_time = np.mean(fall_times) if fall_times else 0.0

        # Calculate slew rates
        slew_rate_rising = (0.8 * signal_range / rise_time) if rise_time > 0 else 0.0
        slew_rate_falling = (0.8 * signal_range / fall_time) if fall_time > 0 else 0.0

        # Detect overshoot and undershoot
        overshoot_pct, undershoot_pct = self.detect_overshoot(trace)

        # Detect ringing
        ringing = self.detect_ringing(trace)
        if ringing:
            ringing_freq, ringing_amp = ringing
        else:
            ringing_freq, ringing_amp = None, None

        return TransitionMetrics(
            rise_time=float(rise_time),
            fall_time=float(fall_time),
            slew_rate_rising=float(slew_rate_rising),
            slew_rate_falling=float(slew_rate_falling),
            overshoot=float(overshoot_pct),
            undershoot=float(undershoot_pct),
            ringing_frequency=ringing_freq,
            ringing_amplitude=ringing_amp,
        )

    def detect_overshoot(
        self, trace: NDArray[np.float64], edges: list[Any] | None = None
    ) -> tuple[float, float]:
        """Detect and measure overshoot and undershoot.

        Args:
            trace: Input signal trace.
            edges: Optional list of edge objects (not used in this implementation).

        Returns:
            Tuple of (overshoot_percent, undershoot_percent).

        Example:
            >>> overshoot, undershoot = analyzer.detect_overshoot(trace)
        """
        trace = np.asarray(trace)

        # Determine signal levels
        threshold = (np.max(trace) + np.min(trace)) / 2.0
        high_samples = trace[trace > threshold]
        low_samples = trace[trace <= threshold]

        if len(high_samples) == 0 or len(low_samples) == 0:
            return 0.0, 0.0

        # Expected levels (mean of stable regions)
        high_level = np.median(high_samples)
        low_level = np.median(low_samples)
        signal_swing = high_level - low_level

        if signal_swing < 1e-6:
            return 0.0, 0.0

        # Overshoot: how much signal exceeds high level
        max_val = np.max(trace)
        overshoot = max_val - high_level
        overshoot_pct = (overshoot / signal_swing) * 100.0

        # Undershoot: how much signal goes below low level
        min_val = np.min(trace)
        undershoot = low_level - min_val
        undershoot_pct = (undershoot / signal_swing) * 100.0

        return max(0.0, overshoot_pct), max(0.0, undershoot_pct)

    def detect_ringing(self, trace: NDArray[np.float64]) -> tuple[float, float] | None:
        """Detect and characterize ringing (frequency, amplitude).

        Uses FFT analysis to detect oscillations after edges that indicate ringing.

        Args:
            trace: Input signal trace.

        Returns:
            Tuple of (frequency_hz, amplitude_volts) if ringing detected, None otherwise.

        Example:
            >>> ringing = analyzer.detect_ringing(trace)
            >>> if ringing:
            ...     freq, amp = ringing
        """
        trace = np.asarray(trace)

        if len(trace) < 32:
            return None

        # Detrend to remove DC offset
        detrended = trace - np.mean(trace)

        # Apply FFT to detect high-frequency oscillations
        fft = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(len(trace), self._time_base)
        power = np.abs(fft) ** 2

        # Look for peaks in high-frequency range (above 1 MHz or 1% of sample rate)
        min_freq = max(1e6, self.sample_rate * 0.01)
        max_freq = self.sample_rate / 4.0  # Below Nyquist/2 for safety

        freq_mask = (freqs > min_freq) & (freqs < max_freq)

        if not np.any(freq_mask):
            return None

        # Find dominant frequency in ringing range
        masked_power = power.copy()
        masked_power[~freq_mask] = 0

        if np.max(masked_power) < np.max(power) * 0.1:
            # No significant high-frequency content
            return None

        peak_idx = np.argmax(masked_power)
        ringing_freq = freqs[peak_idx]

        # Estimate amplitude of ringing (very simplified)
        # Band-pass filter around detected frequency
        try:
            # Design bandpass filter
            bandwidth = ringing_freq * 0.2  # 20% bandwidth
            low = max(ringing_freq - bandwidth, 1.0)
            high = min(ringing_freq + bandwidth, self.sample_rate / 2.0 - 1.0)

            if high > low:
                sos = scipy_signal.butter(4, [low, high], "band", fs=self.sample_rate, output="sos")
                filtered = scipy_signal.sosfilt(sos, detrended)
                ringing_amp = np.std(filtered) * 2.0  # Peak-to-peak estimate
            else:
                ringing_amp = 0.0
        except Exception:
            # If filtering fails, use simple estimate
            ringing_amp = np.std(detrended) * 0.5

        # Only report if amplitude is significant
        if ringing_amp < np.std(trace) * 0.1:
            return None

        return float(ringing_freq), float(ringing_amp)

    def calculate_snr(self, trace: NDArray[np.float64]) -> float:
        """Calculate signal-to-noise ratio.

        Computes SNR by separating signal from noise in stable regions.

        Args:
            trace: Input signal trace.

        Returns:
            SNR in decibels.

        Example:
            >>> snr = analyzer.calculate_snr(trace)
        """
        trace = np.asarray(trace)

        # Separate into high and low regions
        threshold = (np.max(trace) + np.min(trace)) / 2.0
        high_samples = trace[trace > threshold]
        low_samples = trace[trace <= threshold]

        if len(high_samples) == 0 or len(low_samples) == 0:
            return 0.0

        # Signal power: difference between high and low levels
        signal_level = abs(np.mean(high_samples) - np.mean(low_samples))

        # Noise power: standard deviation in stable regions
        noise_high = np.std(high_samples)
        noise_low = np.std(low_samples)
        noise_level = (noise_high + noise_low) / 2.0

        if noise_level < 1e-10:
            return 100.0  # Very high SNR

        # SNR in dB
        snr = 20 * np.log10(signal_level / noise_level)

        return float(snr)


# Convenience functions


def measure_noise_margins(trace: NDArray[np.float64], logic_family: str = "auto") -> NoiseMargins:
    """Measure noise margins.

    Convenience function for quick noise margin measurement.

    Args:
        trace: Input signal trace.
        logic_family: Logic family ('TTL', 'CMOS', 'LVTTL', 'LVCMOS', 'auto').

    Returns:
        NoiseMargins object.

    Example:
        >>> margins = measure_noise_margins(trace, 'TTL')
    """
    # Use a default sample rate for convenience
    sample_rate = 1e9  # 1 GHz default
    analyzer = SignalQualityAnalyzer(sample_rate, logic_family)
    logic_fam: Literal["ttl", "cmos", "lvttl", "lvcmos", "auto"]
    logic_family_lower = logic_family.lower()
    if logic_family_lower in ("ttl", "cmos", "lvttl", "lvcmos", "auto"):
        logic_fam = logic_family_lower  # type: ignore[assignment]
    else:
        logic_fam = "auto"
    return analyzer.measure_noise_margins(trace, logic_fam)


def analyze_signal_integrity(
    trace: NDArray[np.float64],
    sample_rate: float,
    clock_trace: NDArray[np.float64] | None = None,
) -> SignalIntegrityReport:
    """Complete signal integrity analysis.

    Convenience function for complete signal integrity analysis.

    Args:
        trace: Input signal trace.
        sample_rate: Sample rate in Hz.
        clock_trace: Optional clock signal.

    Returns:
        SignalIntegrityReport with complete analysis.

    Example:
        >>> report = analyze_signal_integrity(trace, 100e6)
    """
    analyzer = SignalQualityAnalyzer(sample_rate, logic_family="auto")
    result = analyzer.analyze(trace, clock_trace)
    # In full mode (no v_il/v_ih/vdd), this always returns SignalIntegrityReport
    assert isinstance(result, SignalIntegrityReport)
    return result


__all__ = [
    "LOGIC_THRESHOLDS",
    "NoiseMargins",
    "SignalIntegrityReport",
    "SignalQualityAnalyzer",
    "SimpleQualityMetrics",
    "TransitionMetrics",
    "analyze_signal_integrity",
    "measure_noise_margins",
]
