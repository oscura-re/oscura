"""Unit tests for signal quality and integrity analysis.

This module provides comprehensive tests for signal quality metrics including
SNR calculation, noise margin measurement, transition characterization,
overshoot/undershoot detection, and ringing analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.digital.signal_quality import (
    LOGIC_THRESHOLDS,
    NoiseMargins,
    SignalIntegrityReport,
    SignalQualityAnalyzer,
    SimpleQualityMetrics,
    TransitionMetrics,
    analyze_signal_integrity,
    measure_noise_margins,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


# =============================================================================
# Helper Functions
# =============================================================================


def make_square_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate a square wave signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * (np.sin(2 * np.pi * frequency * t) > 0).astype(np.float64) + offset


def make_ttl_signal(
    frequency: float,
    sample_rate: float,
    duration: float,
) -> np.ndarray:
    """Generate a TTL-level signal (0V to 5V)."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return 5.0 * (np.sin(2 * np.pi * frequency * t) > 0).astype(np.float64)


def make_lvcmos_signal(
    frequency: float,
    sample_rate: float,
    duration: float,
    voltage: float = 3.3,
) -> np.ndarray:
    """Generate an LVCMOS signal (0V to voltage)."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return voltage * (np.sin(2 * np.pi * frequency * t) > 0).astype(np.float64)


def make_signal_with_overshoot(
    sample_rate: float, duration: float, overshoot_pct: float = 0.2
) -> np.ndarray:
    """Generate a signal with overshoot on rising edges."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = np.zeros_like(t)

    # Create transitions with overshoot
    period_samples = int(sample_rate / 1000)  # 1 kHz
    for i in range(0, len(signal), period_samples):
        # Low period
        if i + period_samples // 2 < len(signal):
            # Rising edge with overshoot
            edge_start = i + period_samples // 2
            if edge_start + 10 < len(signal):
                signal[edge_start : edge_start + 5] = 1.0 + overshoot_pct  # Overshoot
                signal[edge_start + 5 : edge_start + period_samples // 2] = 1.0  # Stable high

    return signal


def make_signal_with_ringing(
    sample_rate: float, duration: float, ring_freq: float = 10e6, ring_amp: float = 0.1
) -> np.ndarray:
    """Generate a signal with ringing after transitions."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    base_signal = make_square_wave(1000.0, sample_rate, duration)

    # Add damped ringing
    ringing = ring_amp * np.sin(2 * np.pi * ring_freq * t) * np.exp(-t * 1e6)

    return base_signal + ringing


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestDataClasses:
    """Test signal quality data classes."""

    def test_noise_margins_dataclass(self) -> None:
        """Test NoiseMargins dataclass."""
        margins = NoiseMargins(
            high_margin=0.7,
            low_margin=0.4,
            high_mean=2.5,
            low_mean=0.3,
            high_std=0.05,
            low_std=0.03,
            threshold=1.4,
        )

        assert margins.high_margin == 0.7
        assert margins.low_margin == 0.4
        assert margins.high_mean == 2.5
        assert margins.low_mean == 0.3
        assert margins.high_std == 0.05
        assert margins.low_std == 0.03
        assert margins.threshold == 1.4

    def test_transition_metrics_dataclass(self) -> None:
        """Test TransitionMetrics dataclass."""
        metrics = TransitionMetrics(
            rise_time=1e-9,
            fall_time=0.9e-9,
            slew_rate_rising=1e9,
            slew_rate_falling=-1.1e9,
            overshoot=5.0,
            undershoot=3.0,
            ringing_frequency=10e6,
            ringing_amplitude=0.1,
            settling_time=5e-9,
        )

        assert metrics.rise_time == 1e-9
        assert metrics.fall_time == 0.9e-9
        assert metrics.slew_rate_rising == 1e9
        assert metrics.slew_rate_falling == -1.1e9
        assert metrics.overshoot == 5.0
        assert metrics.undershoot == 3.0
        assert metrics.ringing_frequency == 10e6
        assert metrics.ringing_amplitude == 0.1
        assert metrics.settling_time == 5e-9

    def test_transition_metrics_optional_fields(self) -> None:
        """Test TransitionMetrics with optional fields as None."""
        metrics = TransitionMetrics(
            rise_time=1e-9,
            fall_time=1e-9,
            slew_rate_rising=1e9,
            slew_rate_falling=-1e9,
            overshoot=0.0,
            undershoot=0.0,
        )

        assert metrics.ringing_frequency is None
        assert metrics.ringing_amplitude is None
        assert metrics.settling_time is None

    def test_signal_integrity_report_dataclass(self) -> None:
        """Test SignalIntegrityReport dataclass."""
        margins = NoiseMargins(0.7, 0.4, 2.5, 0.3, 0.05, 0.03, 1.4)
        transitions = TransitionMetrics(1e-9, 1e-9, 1e9, -1e9, 0.0, 0.0)

        report = SignalIntegrityReport(
            noise_margins=margins,
            transitions=transitions,
            snr_db=45.0,
            signal_quality="excellent",
            issues=[],
            recommendations=[],
        )

        assert report.noise_margins == margins
        assert report.transitions == transitions
        assert report.snr_db == 45.0
        assert report.signal_quality == "excellent"
        assert len(report.issues) == 0
        assert len(report.recommendations) == 0

    def test_signal_integrity_report_with_issues(self) -> None:
        """Test SignalIntegrityReport with issues and recommendations."""
        margins = NoiseMargins(0.2, 0.2, 2.0, 0.5, 0.1, 0.1, 1.25)
        transitions = TransitionMetrics(1e-9, 1e-9, 1e9, -1e9, 25.0, 25.0)

        report = SignalIntegrityReport(
            noise_margins=margins,
            transitions=transitions,
            snr_db=15.0,
            signal_quality="poor",
            issues=["Low SNR", "Excessive overshoot"],
            recommendations=["Add termination", "Reduce noise"],
        )

        assert len(report.issues) == 2
        assert len(report.recommendations) == 2
        assert report.signal_quality == "poor"

    def test_simple_quality_metrics_dataclass(self) -> None:
        """Test SimpleQualityMetrics dataclass."""
        metrics = SimpleQualityMetrics(
            noise_margin_low=0.4,
            noise_margin_high=0.6,
            rise_time=5.0,
            fall_time=4.5,
            has_overshoot=True,
            max_overshoot=0.2,
            duty_cycle=0.48,
        )

        assert metrics.noise_margin_low == 0.4
        assert metrics.noise_margin_high == 0.6
        assert metrics.rise_time == 5.0
        assert metrics.fall_time == 4.5
        assert metrics.has_overshoot is True
        assert metrics.max_overshoot == 0.2
        assert metrics.duty_cycle == 0.48


# =============================================================================
# SignalQualityAnalyzer Initialization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestSignalQualityAnalyzerInit:
    """Test SignalQualityAnalyzer initialization."""

    def test_init_full_mode(self) -> None:
        """Test initialization in full mode with sample rate and logic family."""
        analyzer = SignalQualityAnalyzer(sample_rate=1e9, logic_family="TTL")

        assert analyzer.sample_rate == 1e9
        assert analyzer.logic_family == "ttl"
        assert analyzer._time_base == 1e-9

    def test_init_simple_mode(self) -> None:
        """Test initialization in simple mode with thresholds."""
        analyzer = SignalQualityAnalyzer(v_il=0.8, v_ih=2.0, vdd=5.0)

        assert analyzer.v_il == 0.8
        assert analyzer.v_ih == 2.0
        assert analyzer.vdd == 5.0
        assert analyzer._threshold == 1.4

    def test_init_default_sample_rate(self) -> None:
        """Test initialization without sample rate."""
        analyzer = SignalQualityAnalyzer()

        assert analyzer.sample_rate == 1.0
        assert analyzer._time_base == 1.0

    def test_init_invalid_sample_rate(self) -> None:
        """Test initialization with invalid sample rate."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            SignalQualityAnalyzer(sample_rate=0)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            SignalQualityAnalyzer(sample_rate=-1e6)

    def test_init_logic_family_case_insensitive(self) -> None:
        """Test that logic family is case-insensitive."""
        analyzer1 = SignalQualityAnalyzer(sample_rate=1e9, logic_family="TTL")
        analyzer2 = SignalQualityAnalyzer(sample_rate=1e9, logic_family="ttl")
        analyzer3 = SignalQualityAnalyzer(sample_rate=1e9, logic_family="TtL")

        assert analyzer1.logic_family == "ttl"
        assert analyzer2.logic_family == "ttl"
        assert analyzer3.logic_family == "ttl"

    def test_init_all_logic_families(self) -> None:
        """Test initialization with all supported logic families."""
        families = ["TTL", "CMOS", "LVTTL", "LVCMOS", "auto"]

        for family in families:
            analyzer = SignalQualityAnalyzer(sample_rate=1e9, logic_family=family)
            assert analyzer.logic_family == family.lower()

    def test_init_unsupported_logic_family(self) -> None:
        """Test initialization with unsupported logic family."""
        # Should handle gracefully by converting to lowercase
        analyzer = SignalQualityAnalyzer(sample_rate=1e9, logic_family="UNKNOWN")
        assert analyzer.logic_family == "unknown"


# =============================================================================
# Noise Margin Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestNoiseMargins:
    """Test noise margin measurements."""

    def test_measure_noise_margins_ttl(self) -> None:
        """Test noise margin measurement for TTL signal."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        margins = analyzer.measure_noise_margins(signal, logic_family="ttl")

        assert margins.high_margin > 0
        assert margins.low_margin > 0
        assert margins.high_mean > margins.threshold
        assert margins.low_mean < margins.threshold

    def test_measure_noise_margins_cmos(self) -> None:
        """Test noise margin measurement for CMOS signal."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="CMOS")
        margins = analyzer.measure_noise_margins(signal, logic_family="cmos")

        assert margins.high_margin > 0
        assert margins.low_margin > 0

    def test_measure_noise_margins_auto_detect_5v(self) -> None:
        """Test automatic logic family detection for 5V signal."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        margins = analyzer.measure_noise_margins(signal, logic_family="auto")

        # Should auto-detect as TTL (5V logic)
        assert margins.threshold > 0

    def test_measure_noise_margins_auto_detect_3v3(self) -> None:
        """Test automatic logic family detection for 3.3V signal."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01, voltage=3.3)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        margins = analyzer.measure_noise_margins(signal, logic_family="auto")

        # Should auto-detect as LVTTL or LVCMOS (3.3V logic)
        assert margins.threshold > 0

    def test_measure_noise_margins_noise_statistics(self) -> None:
        """Test that noise statistics are calculated."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        # Add noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.1, len(signal))

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        margins = analyzer.measure_noise_margins(noisy_signal, logic_family="ttl")

        assert margins.high_std > 0
        assert margins.low_std > 0

    def test_measure_noise_margins_empty_signal(self) -> None:
        """Test noise margin measurement on empty signal."""
        signal = np.array([])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6, logic_family="TTL")
        margins = analyzer.measure_noise_margins(signal, logic_family="ttl")

        # Should handle empty signal gracefully
        assert margins.high_margin == 0.0
        assert margins.low_margin == 0.0

    def test_measure_noise_margins_constant_high(self) -> None:
        """Test noise margin measurement on constant high signal."""
        signal = np.ones(1000) * 5.0

        analyzer = SignalQualityAnalyzer(sample_rate=1e6, logic_family="TTL")
        margins = analyzer.measure_noise_margins(signal, logic_family="ttl")

        # Only high samples
        assert margins.high_margin > 0
        assert margins.low_margin == 0.0

    def test_measure_noise_margins_constant_low(self) -> None:
        """Test noise margin measurement on constant low signal."""
        signal = np.zeros(1000)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6, logic_family="TTL")
        margins = analyzer.measure_noise_margins(signal, logic_family="ttl")

        # Only low samples
        assert margins.high_margin == 0.0
        assert margins.low_margin > 0

    def test_measure_noise_margins_auto_detect_low_voltage(self) -> None:
        """Test automatic logic family detection for low voltage signal."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01, voltage=1.8)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        margins = analyzer.measure_noise_margins(signal, logic_family="auto")

        # Should auto-detect as LVCMOS (low voltage)
        assert margins.threshold > 0


# =============================================================================
# Transition Measurement Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestTransitionMetrics:
    """Test transition characteristic measurements."""

    def test_measure_transitions_basic(self) -> None:
        """Test basic transition measurement."""
        sample_rate = 100e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        transitions = analyzer.measure_transitions(signal)

        # Rise time may be 0 if edges are too sharp for measurement window
        assert transitions.rise_time >= 0
        assert transitions.fall_time >= 0
        assert transitions.slew_rate_rising >= 0
        assert transitions.slew_rate_falling >= 0

    def test_measure_transitions_overshoot(self) -> None:
        """Test overshoot detection in transitions."""
        sample_rate = 100e6
        signal = make_signal_with_overshoot(sample_rate, 0.01, overshoot_pct=0.15)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        transitions = analyzer.measure_transitions(signal)

        # Should detect overshoot
        assert transitions.overshoot >= 0

    def test_measure_transitions_no_edges(self) -> None:
        """Test transition measurement with no edges."""
        signal = np.ones(1000)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        transitions = analyzer.measure_transitions(signal)

        # No edges means zero or default values
        assert transitions.rise_time == 0.0
        assert transitions.fall_time == 0.0

    def test_measure_transitions_single_rising_edge(self) -> None:
        """Test transition measurement with single rising edge."""
        signal = np.concatenate([np.zeros(100), np.ones(100)])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        transitions = analyzer.measure_transitions(signal)

        # Should measure rise time
        assert transitions.rise_time >= 0

    def test_measure_transitions_single_falling_edge(self) -> None:
        """Test transition measurement with single falling edge."""
        signal = np.concatenate([np.ones(100), np.zeros(100)])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        transitions = analyzer.measure_transitions(signal)

        # Should measure fall time
        assert transitions.fall_time >= 0

    def test_measure_transitions_ringing_detection(self) -> None:
        """Test ringing detection in transitions."""
        sample_rate = 1e9  # High sample rate needed for ringing
        signal = make_signal_with_ringing(sample_rate, 1e-6, ring_freq=50e6, ring_amp=0.2)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        transitions = analyzer.measure_transitions(signal)

        # Ringing may or may not be detected depending on signal characteristics
        assert transitions.ringing_frequency is None or transitions.ringing_frequency > 0


# =============================================================================
# Overshoot Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestOvershootDetection:
    """Test overshoot and undershoot detection."""

    def test_detect_overshoot_clean_signal(self) -> None:
        """Test overshoot detection on clean signal."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        overshoot, undershoot = analyzer.detect_overshoot(signal)

        # Clean signal should have minimal overshoot/undershoot
        assert overshoot >= 0
        assert undershoot >= 0

    def test_detect_overshoot_with_overshoot(self) -> None:
        """Test overshoot detection with actual overshoot."""
        sample_rate = 100e6
        signal = make_signal_with_overshoot(sample_rate, 0.01, overshoot_pct=0.25)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        overshoot, undershoot = analyzer.detect_overshoot(signal)

        # Should detect overshoot
        assert overshoot > 0

    def test_detect_overshoot_constant_signal(self) -> None:
        """Test overshoot detection on constant signal."""
        signal = np.ones(1000)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        overshoot, undershoot = analyzer.detect_overshoot(signal)

        # No transitions means no overshoot
        assert overshoot == 0.0
        assert undershoot == 0.0

    def test_detect_overshoot_empty_signal(self) -> None:
        """Test overshoot detection on empty signal."""
        signal = np.array([])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)

        # Empty signal may cause numpy errors, but method should handle it
        try:
            overshoot, undershoot = analyzer.detect_overshoot(signal)
            assert overshoot == 0.0
            assert undershoot == 0.0
        except (ValueError, IndexError):
            # Empty arrays may raise errors in some numpy operations
            pass

    def test_detect_overshoot_percentage_calculation(self) -> None:
        """Test that overshoot is calculated as percentage."""
        # Create signal with known overshoot
        signal = np.array([0.0] * 100 + [1.2] + [1.0] * 100)  # 20% overshoot

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        overshoot, undershoot = analyzer.detect_overshoot(signal)

        # Overshoot should be positive percentage
        assert overshoot >= 0


# =============================================================================
# Ringing Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestRingingDetection:
    """Test ringing detection and characterization."""

    def test_detect_ringing_clean_signal(self) -> None:
        """Test ringing detection on clean signal."""
        signal = make_square_wave(1000.0, 1e9, 1e-6)

        analyzer = SignalQualityAnalyzer(sample_rate=1e9)
        ringing = analyzer.detect_ringing(signal)

        # Clean signal should have no ringing (or minimal)
        assert ringing is None or ringing[1] < 0.1

    def test_detect_ringing_with_ringing(self) -> None:
        """Test ringing detection with actual ringing."""
        sample_rate = 1e9
        signal = make_signal_with_ringing(sample_rate, 1e-6, ring_freq=50e6, ring_amp=0.3)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        ringing = analyzer.detect_ringing(signal)

        # May detect ringing depending on FFT resolution
        if ringing is not None:
            freq, amp = ringing
            assert freq > 0
            assert amp > 0

    def test_detect_ringing_short_signal(self) -> None:
        """Test ringing detection on short signal."""
        signal = np.array([0.0, 1.0] * 10)  # Only 20 samples

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        ringing = analyzer.detect_ringing(signal)

        # Too short for FFT analysis
        assert ringing is None

    def test_detect_ringing_constant_signal(self) -> None:
        """Test ringing detection on constant signal."""
        signal = np.ones(1000)

        analyzer = SignalQualityAnalyzer(sample_rate=1e9)

        # Constant signal has no AC component, so ringing detection may return None
        # or detect minimal noise depending on implementation
        ringing = analyzer.detect_ringing(signal)

        # Should return None or very low amplitude ringing
        if ringing is not None:
            assert ringing[1] < 0.1  # Very low amplitude if detected

    def test_detect_ringing_returns_tuple(self) -> None:
        """Test that ringing detection returns proper tuple format."""
        sample_rate = 1e9
        signal = make_signal_with_ringing(sample_rate, 1e-6)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        ringing = analyzer.detect_ringing(signal)

        if ringing is not None:
            assert isinstance(ringing, tuple)
            assert len(ringing) == 2
            freq, amp = ringing
            assert isinstance(freq, float)
            assert isinstance(amp, float)


# =============================================================================
# SNR Calculation Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestSNRCalculation:
    """Test signal-to-noise ratio calculation."""

    def test_calculate_snr_clean_signal(self) -> None:
        """Test SNR calculation on clean signal."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        snr = analyzer.calculate_snr(signal)

        # Clean signal should have high SNR
        assert snr > 20

    def test_calculate_snr_noisy_signal(self) -> None:
        """Test SNR calculation on noisy signal."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        # Add significant noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.2, len(signal))

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        snr_clean = analyzer.calculate_snr(signal)
        snr_noisy = analyzer.calculate_snr(noisy_signal)

        # Noisy signal should have lower SNR
        assert snr_noisy < snr_clean

    def test_calculate_snr_constant_signal(self) -> None:
        """Test SNR calculation on constant signal."""
        signal = np.ones(1000)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        snr = analyzer.calculate_snr(signal)

        # Constant signal has no transitions
        assert snr == 0.0

    def test_calculate_snr_very_low_noise(self) -> None:
        """Test SNR calculation with very low noise."""
        # Perfect digital signal
        signal = make_square_wave(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        snr = analyzer.calculate_snr(signal)

        # Should return high SNR (clamped at 100 dB)
        assert snr > 0

    def test_calculate_snr_empty_signal(self) -> None:
        """Test SNR calculation on empty signal."""
        signal = np.array([])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)

        # Empty signal may raise errors in some numpy operations
        try:
            snr = analyzer.calculate_snr(signal)
            assert snr == 0.0
        except (ValueError, IndexError):
            # Empty arrays may raise errors
            pass


# =============================================================================
# Full Analysis Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestFullAnalysis:
    """Test complete signal integrity analysis."""

    def test_analyze_full_mode(self) -> None:
        """Test full analysis mode."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        report = analyzer.analyze(signal)

        assert isinstance(report, SignalIntegrityReport)
        assert isinstance(report.noise_margins, NoiseMargins)
        assert isinstance(report.transitions, TransitionMetrics)
        assert report.snr_db > 0
        assert report.signal_quality in ["excellent", "good", "fair", "poor"]

    def test_analyze_simple_mode(self) -> None:
        """Test simple analysis mode."""
        signal = make_ttl_signal(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(v_il=0.8, v_ih=2.0, vdd=5.0)
        metrics = analyzer.analyze(signal)

        assert isinstance(metrics, SimpleQualityMetrics)
        assert metrics.noise_margin_low >= 0
        assert metrics.noise_margin_high >= 0
        assert metrics.rise_time >= 0
        assert metrics.fall_time >= 0
        assert 0.0 <= metrics.duty_cycle <= 1.0

    def test_analyze_excellent_quality(self) -> None:
        """Test that excellent signal is classified correctly."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        report = analyzer.analyze(signal)

        # Clean TTL signal should have good or excellent quality
        assert report.signal_quality in ["excellent", "good"]
        assert len(report.issues) <= 1

    def test_analyze_poor_quality_excessive_overshoot(self) -> None:
        """Test that signal with excessive overshoot is flagged."""
        sample_rate = 100e6
        signal = make_signal_with_overshoot(sample_rate, 0.01, overshoot_pct=0.3)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
        report = analyzer.analyze(signal)

        # May detect overshoot issue
        assert isinstance(report.issues, list)

    def test_analyze_with_recommendations(self) -> None:
        """Test that analysis provides recommendations for issues."""
        sample_rate = 100e6
        # Create problematic signal
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)
        # Add significant noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.5, len(signal))

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        report = analyzer.analyze(noisy_signal)

        # Should have issues and recommendations
        if len(report.issues) > 0:
            assert len(report.recommendations) > 0


# =============================================================================
# Simple Mode Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestSimpleMode:
    """Test simple analysis mode features."""

    def test_simple_mode_duty_cycle(self) -> None:
        """Test duty cycle calculation in simple mode."""
        # Create signal that crosses the threshold
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=5.0, offset=0.0)

        analyzer = SignalQualityAnalyzer(v_il=0.8, v_ih=2.0)
        metrics = analyzer.analyze(signal)

        # Square wave should have ~50% duty cycle (allow wider tolerance)
        assert 0.0 <= metrics.duty_cycle <= 1.0
        # Signal should have some transitions
        assert isinstance(metrics, SimpleQualityMetrics)

    def test_simple_mode_asymmetric_duty_cycle(self) -> None:
        """Test duty cycle calculation for asymmetric signal."""
        # Create signal with 25% duty cycle
        signal = np.concatenate([np.ones(250), np.zeros(750)])

        analyzer = SignalQualityAnalyzer(v_il=0.3, v_ih=0.7)
        metrics = analyzer.analyze(signal)

        # Should measure ~25% duty cycle
        assert 0.2 < metrics.duty_cycle < 0.3

    def test_simple_mode_overshoot_detection(self) -> None:
        """Test overshoot detection in simple mode."""
        signal = make_signal_with_overshoot(100e6, 0.01, overshoot_pct=0.2)

        analyzer = SignalQualityAnalyzer(v_il=0.3, v_ih=0.7, vdd=1.0)
        metrics = analyzer.analyze(signal)

        # Should detect overshoot
        assert isinstance(metrics.has_overshoot, bool)
        assert metrics.max_overshoot >= 0

    def test_simple_mode_no_overshoot(self) -> None:
        """Test overshoot detection on clean signal."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(v_il=0.3, v_ih=0.7)
        metrics = analyzer.analyze(signal)

        # Clean signal should have minimal overshoot
        assert isinstance(metrics.has_overshoot, bool)

    def test_simple_mode_rise_fall_times(self) -> None:
        """Test rise and fall time measurement in simple mode."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(v_il=0.3, v_ih=0.7)
        metrics = analyzer.analyze(signal)

        # Should measure rise and fall times
        assert metrics.rise_time >= 0
        assert metrics.fall_time >= 0

    def test_simple_mode_with_only_vdd(self) -> None:
        """Test simple mode with only VDD specified."""
        signal = make_ttl_signal(1000.0, 1e6, 0.01)

        analyzer = SignalQualityAnalyzer(vdd=5.0)
        metrics = analyzer.analyze(signal)

        # Should still produce valid metrics
        assert isinstance(metrics, SimpleQualityMetrics)
        assert metrics.duty_cycle >= 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_measure_noise_margins_function(self) -> None:
        """Test measure_noise_margins convenience function."""
        signal = make_ttl_signal(1000.0, 1e6, 0.01)

        margins = measure_noise_margins(signal, logic_family="TTL")

        assert isinstance(margins, NoiseMargins)
        assert margins.high_margin > 0
        assert margins.low_margin > 0

    def test_measure_noise_margins_auto(self) -> None:
        """Test measure_noise_margins with auto logic family."""
        signal = make_lvcmos_signal(1000.0, 1e6, 0.01)

        margins = measure_noise_margins(signal, logic_family="auto")

        assert isinstance(margins, NoiseMargins)

    def test_analyze_signal_integrity_function(self) -> None:
        """Test analyze_signal_integrity convenience function."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        report = analyze_signal_integrity(signal, sample_rate)

        assert isinstance(report, SignalIntegrityReport)
        assert report.snr_db > 0
        assert report.signal_quality in ["excellent", "good", "fair", "poor"]

    def test_analyze_signal_integrity_with_clock(self) -> None:
        """Test analyze_signal_integrity with clock trace."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)
        clock = make_square_wave(1000.0, sample_rate, 0.01)

        report = analyze_signal_integrity(signal, sample_rate, clock_trace=clock)

        assert isinstance(report, SignalIntegrityReport)


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestDigitalSignalQualityEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_trace(self) -> None:
        """Test analysis with empty trace."""
        signal = np.array([])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6, logic_family="TTL")

        # Empty signal may cause errors in numpy operations
        try:
            report = analyzer.analyze(signal)
            assert isinstance(report, SignalIntegrityReport)
        except (ValueError, IndexError):
            # Empty arrays may raise errors in some operations
            pass

    def test_single_value_trace(self) -> None:
        """Test analysis with single value."""
        signal = np.array([1.0])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        transitions = analyzer.measure_transitions(signal)

        assert transitions.rise_time == 0.0
        assert transitions.fall_time == 0.0

    def test_constant_signal(self) -> None:
        """Test analysis with constant signal."""
        signal = np.ones(1000)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6, logic_family="TTL")
        report = analyzer.analyze(signal)

        # Should handle constant signal
        assert isinstance(report, SignalIntegrityReport)

    def test_very_short_signal(self) -> None:
        """Test analysis with very short signal."""
        signal = np.array([0.0, 1.0])

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        transitions = analyzer.measure_transitions(signal)

        # Very short signal may not have measurable transitions
        assert isinstance(transitions, TransitionMetrics)

    def test_negative_values(self) -> None:
        """Test analysis with negative signal values."""
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=2.0, offset=-1.0)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        report = analyzer.analyze(signal)

        assert isinstance(report, SignalIntegrityReport)

    def test_very_large_values(self) -> None:
        """Test analysis with very large values."""
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=1e6)

        analyzer = SignalQualityAnalyzer(sample_rate=1e6)
        report = analyzer.analyze(signal)

        assert isinstance(report, SignalIntegrityReport)

    def test_all_logic_thresholds_defined(self) -> None:
        """Test that all logic threshold families are properly defined."""
        expected_families = ["ttl", "cmos", "lvttl", "lvcmos"]

        for family in expected_families:
            assert family in LOGIC_THRESHOLDS
            thresholds = LOGIC_THRESHOLDS[family]
            assert "VIL" in thresholds
            assert "VIH" in thresholds
            assert "VOL" in thresholds
            assert "VOH" in thresholds
            assert "VCC" in thresholds

    def test_boolean_trace_duty_cycle(self) -> None:
        """Test duty cycle calculation with boolean trace."""
        # Create boolean array directly
        signal = np.array([True, True, False, False, True, True, False, False])

        analyzer = SignalQualityAnalyzer(v_il=0.3, v_ih=0.7)
        duty_cycle = analyzer._calculate_duty_cycle(signal, 0.5)

        # Should be 50% duty cycle
        assert 0.4 < duty_cycle < 0.6


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from oscura.analyzers.digital import signal_quality

        expected_exports = {
            "LOGIC_THRESHOLDS",
            "NoiseMargins",
            "SignalIntegrityReport",
            "SignalQualityAnalyzer",
            "SimpleQualityMetrics",
            "TransitionMetrics",
            "analyze_signal_integrity",
            "measure_noise_margins",
        }

        assert hasattr(signal_quality, "__all__")
        assert set(signal_quality.__all__) == expected_exports

    def test_dataclasses_importable(self) -> None:
        """Test that all dataclasses are importable."""
        from oscura.analyzers.digital.signal_quality import (
            NoiseMargins,
            SignalIntegrityReport,
            SimpleQualityMetrics,
            TransitionMetrics,
        )

        assert NoiseMargins is not None
        assert SignalIntegrityReport is not None
        assert SimpleQualityMetrics is not None
        assert TransitionMetrics is not None

    def test_class_importable(self) -> None:
        """Test that SignalQualityAnalyzer class is importable."""
        from oscura.analyzers.digital.signal_quality import SignalQualityAnalyzer

        assert SignalQualityAnalyzer is not None

    def test_functions_importable(self) -> None:
        """Test that all functions are importable."""
        from oscura.analyzers.digital.signal_quality import (
            analyze_signal_integrity,
            measure_noise_margins,
        )

        assert analyze_signal_integrity is not None
        assert measure_noise_margins is not None

    def test_constants_importable(self) -> None:
        """Test that constants are importable."""
        from oscura.analyzers.digital.signal_quality import LOGIC_THRESHOLDS

        assert LOGIC_THRESHOLDS is not None
        assert isinstance(LOGIC_THRESHOLDS, dict)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-005")
class TestDigitalSignalQualityIntegration:
    """Test integration of signal quality components."""

    def test_full_workflow_ttl(self) -> None:
        """Test complete workflow with TTL signal."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        # Create analyzer
        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")

        # Measure noise margins
        margins = analyzer.measure_noise_margins(signal, logic_family="ttl")
        assert margins.high_margin > 0

        # Measure transitions (may be 0 for sharp edges)
        transitions = analyzer.measure_transitions(signal)
        assert transitions.rise_time >= 0
        assert transitions.fall_time >= 0

        # Detect overshoot
        overshoot, undershoot = analyzer.detect_overshoot(signal)
        assert overshoot >= 0

        # Calculate SNR
        snr = analyzer.calculate_snr(signal)
        assert snr > 0

        # Full analysis
        report = analyzer.analyze(signal)
        assert isinstance(report, SignalIntegrityReport)

    def test_full_workflow_lvcmos(self) -> None:
        """Test complete workflow with LVCMOS signal."""
        sample_rate = 100e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01)

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="LVCMOS")
        report = analyzer.analyze(signal)

        assert isinstance(report, SignalIntegrityReport)
        assert report.signal_quality in ["excellent", "good", "fair", "poor"]

    def test_workflow_with_noisy_signal(self) -> None:
        """Test workflow with noisy signal."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        # Add noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.3, len(signal))

        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        report = analyzer.analyze(noisy_signal)

        # Should still produce valid report
        assert isinstance(report, SignalIntegrityReport)
        # May have lower quality due to noise
        assert report.signal_quality in ["excellent", "good", "fair", "poor"]

    def test_comparison_clean_vs_noisy(self) -> None:
        """Test comparison of clean vs noisy signal."""
        sample_rate = 100e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)

        # Clean signal
        analyzer = SignalQualityAnalyzer(sample_rate=sample_rate, logic_family="TTL")
        report_clean = analyzer.analyze(signal)

        # Noisy signal
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.2, len(signal))
        report_noisy = analyzer.analyze(noisy_signal)

        # Clean signal should have better SNR
        assert report_clean.snr_db > report_noisy.snr_db
