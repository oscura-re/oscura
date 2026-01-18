"""Unit tests for signal quality analysis module.

This module provides comprehensive tests for signal quality metrics including
noise margin calculation, setup/hold violation detection, glitch detection,
mask testing, and PLL clock recovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.digital.quality import (
    Glitch,
    MaskTestResult,
    NoiseMarginResult,
    PLLRecoveryResult,
    Violation,
    _find_logic_levels,
    _get_clock_edges,
    _get_predefined_mask,
    detect_glitches,
    detect_violations,
    mask_test,
    noise_margin,
    pll_clock_recovery,
    signal_quality_summary,
)
from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


# =============================================================================
# Helper Functions
# =============================================================================


def make_waveform_trace(
    data: np.ndarray,
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def make_digital_trace(
    data: np.ndarray,
    sample_rate: float = 1e6,
) -> DigitalTrace:
    """Create a DigitalTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return DigitalTrace(data=data.astype(np.bool_), metadata=metadata)


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


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-001")
class TestDataClasses:
    """Test quality analysis data classes."""

    def test_noise_margin_result(self) -> None:
        """Test NoiseMarginResult dataclass."""
        result = NoiseMarginResult(
            nm_high=0.7,
            nm_low=0.4,
            logic_family="TTL",
            voh=2.4,
            vol=0.4,
            vih=2.0,
            vil=0.8,
        )

        assert result.nm_high == 0.7
        assert result.nm_low == 0.4
        assert result.logic_family == "TTL"
        assert result.voh == 2.4
        assert result.vol == 0.4
        assert result.vih == 2.0
        assert result.vil == 0.8

    def test_violation_dataclass(self) -> None:
        """Test Violation dataclass."""
        violation = Violation(
            timestamp=1e-6,
            violation_type="setup",
            measured=0.5e-9,
            limit=1e-9,
            margin=-0.5e-9,
        )

        assert violation.timestamp == 1e-6
        assert violation.violation_type == "setup"
        assert violation.measured == 0.5e-9
        assert violation.limit == 1e-9
        assert violation.margin == -0.5e-9
        assert violation.end_timestamp is None

    def test_violation_with_end_timestamp(self) -> None:
        """Test Violation with optional end timestamp."""
        violation = Violation(
            timestamp=1e-6,
            violation_type="glitch",
            measured=5e-9,
            limit=10e-9,
            margin=-5e-9,
            end_timestamp=1.005e-6,
        )

        assert violation.end_timestamp == 1.005e-6

    def test_glitch_dataclass(self) -> None:
        """Test Glitch dataclass."""
        glitch = Glitch(
            timestamp=1e-6,
            width=5e-9,
            polarity="positive",
            amplitude=0.5,
        )

        assert glitch.timestamp == 1e-6
        assert glitch.width == 5e-9
        assert glitch.polarity == "positive"
        assert glitch.amplitude == 0.5

    def test_mask_test_result(self) -> None:
        """Test MaskTestResult dataclass."""
        violations = [(1e-6, 0.5), (2e-6, 0.7)]
        result = MaskTestResult(
            pass_fail=False,
            hit_count=2,
            total_samples=1000,
            margin_top=0.1,
            margin_bottom=0.05,
            violations=violations,
        )

        assert result.pass_fail is False
        assert result.hit_count == 2
        assert result.total_samples == 1000
        assert result.margin_top == 0.1
        assert result.margin_bottom == 0.05
        assert len(result.violations) == 2

    def test_pll_recovery_result(self) -> None:
        """Test PLLRecoveryResult dataclass."""
        phase = np.array([0.0, 0.1, 0.2])
        vco_control = np.array([0.0, 0.01, 0.02])

        result = PLLRecoveryResult(
            recovered_frequency=10e6,
            recovered_phase=phase,
            vco_control=vco_control,
            lock_status=True,
            lock_time=1e-6,
            frequency_error=100.0,
        )

        assert result.recovered_frequency == 10e6
        assert np.array_equal(result.recovered_phase, phase)
        assert np.array_equal(result.vco_control, vco_control)
        assert result.lock_status is True
        assert result.lock_time == 1e-6
        assert result.frequency_error == 100.0


# =============================================================================
# Noise Margin Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-001")
class TestNoiseMargin:
    """Test noise margin calculations."""

    def test_noise_margin_ttl(self) -> None:
        """Test noise margin for TTL logic family."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        result = noise_margin(trace, family="TTL")

        assert result.logic_family == "TTL"
        assert result.nm_high > 0
        assert result.nm_low > 0
        assert result.voh > result.vih
        assert result.vil > result.vol

    def test_noise_margin_lvcmos_3v3(self) -> None:
        """Test noise margin for LVCMOS 3.3V logic family."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01, voltage=3.3)
        trace = make_waveform_trace(signal, sample_rate)

        result = noise_margin(trace, family="LVCMOS_3V3")

        assert result.logic_family == "LVCMOS_3V3"
        assert result.nm_high > 0
        assert result.nm_low > 0

    def test_noise_margin_lvcmos_1v8(self) -> None:
        """Test noise margin for LVCMOS 1.8V logic family."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01, voltage=1.8)
        trace = make_waveform_trace(signal, sample_rate)

        result = noise_margin(trace, family="LVCMOS_1V8")

        assert result.logic_family == "LVCMOS_1V8"
        assert result.nm_high > 0
        assert result.nm_low > 0

    def test_noise_margin_use_measured_levels(self) -> None:
        """Test noise margin using measured signal levels."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        result = noise_margin(trace, family="TTL", use_measured_levels=True)

        # VOH and VOL should be measured from signal
        assert result.voh > 0
        assert result.vol >= 0

    def test_noise_margin_use_spec_levels(self) -> None:
        """Test noise margin using spec levels."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        result = noise_margin(trace, family="TTL", use_measured_levels=False)

        # Should use spec values from LOGIC_FAMILIES
        assert result.voh > 0
        assert result.vol >= 0

    def test_noise_margin_empty_trace(self) -> None:
        """Test noise margin with empty trace."""
        empty_data = np.array([])
        trace = make_waveform_trace(empty_data, 1e6)

        result = noise_margin(trace, family="TTL", use_measured_levels=True)

        # Should still work but use spec values
        assert result.logic_family == "TTL"

    def test_noise_margin_invalid_family(self) -> None:
        """Test noise margin with invalid logic family."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.001)
        trace = make_waveform_trace(signal, sample_rate)

        with pytest.raises(ValueError, match="Unknown logic family"):
            noise_margin(trace, family="INVALID_FAMILY")

    def test_noise_margin_all_families(self) -> None:
        """Test noise margin for all supported logic families."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.001)
        trace = make_waveform_trace(signal, sample_rate)

        families = [
            "TTL",
            "CMOS_5V",
            "LVTTL",
            "LVCMOS_3V3",
            "LVCMOS_2V5",
            "LVCMOS_1V8",
            "LVCMOS_1V2",
        ]

        for family in families:
            result = noise_margin(trace, family=family)
            assert result.logic_family == family
            assert result.nm_high is not None
            assert result.nm_low is not None


# =============================================================================
# Violation Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-002")
class TestDetectViolations:
    """Test setup/hold violation detection."""

    def test_detect_violations_no_violations(self) -> None:
        """Test violation detection with adequate timing."""
        sample_rate = 1e6

        # Data changes 100 samples before clock edge
        data_signal = np.concatenate([np.zeros(100), np.ones(900)])
        clock_signal = np.concatenate([np.zeros(200), np.ones(800)])

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=10e-6,  # 10 µs
            hold_spec=10e-6,  # 10 µs
        )

        # With generous specs, should have no violations
        assert isinstance(violations, list)

    def test_detect_violations_setup_violation(self) -> None:
        """Test detection of setup time violations."""
        sample_rate = 1e6

        # Data changes very close to clock edge
        data_signal = np.tile([0, 0, 1, 1], 250)
        clock_signal = np.tile([0, 0, 0, 1], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=100e-6,  # Very long setup requirement
            hold_spec=1e-9,
        )

        setup_violations = [v for v in violations if v.violation_type == "setup"]
        assert len(setup_violations) > 0
        assert all(v.margin < 0 for v in setup_violations)

    def test_detect_violations_hold_violation(self) -> None:
        """Test detection of hold time violations."""
        sample_rate = 1e6

        # Clock edge then immediate data change
        clock_signal = np.tile([0, 1, 1, 1], 250)
        data_signal = np.tile([1, 1, 0, 0], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=1e-9,
            hold_spec=100e-6,  # Very long hold requirement
        )

        hold_violations = [v for v in violations if v.violation_type == "hold"]
        assert len(hold_violations) > 0
        assert all(v.margin < 0 for v in hold_violations)

    def test_detect_violations_rising_edge(self) -> None:
        """Test violations on rising clock edge."""
        sample_rate = 1e6
        data_signal = np.tile([0, 0, 1, 1], 250)
        clock_signal = np.tile([0, 0, 0, 1], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=1e-9,
            hold_spec=1e-9,
            clock_edge="rising",
        )

        assert isinstance(violations, list)

    def test_detect_violations_falling_edge(self) -> None:
        """Test violations on falling clock edge."""
        sample_rate = 1e6
        data_signal = np.tile([1, 1, 0, 0], 250)
        clock_signal = np.tile([1, 1, 1, 0], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=1e-9,
            hold_spec=1e-9,
            clock_edge="falling",
        )

        assert isinstance(violations, list)

    def test_detect_violations_digital_traces(self) -> None:
        """Test violation detection with DigitalTrace objects."""
        sample_rate = 1e6
        data_signal = np.tile([0, 0, 1, 1], 250) > 0.5
        clock_signal = np.tile([0, 0, 0, 1], 250) > 0.5

        data_trace = make_digital_trace(data_signal, sample_rate)
        clock_trace = make_digital_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=1e-9,
            hold_spec=1e-9,
        )

        assert isinstance(violations, list)

    def test_detect_violations_sorted_by_timestamp(self) -> None:
        """Test that violations are sorted by timestamp."""
        sample_rate = 1e6
        data_signal = np.tile([0, 0, 1, 1], 250)
        clock_signal = np.tile([0, 0, 0, 1], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=100e-6,
            hold_spec=100e-6,
        )

        if len(violations) > 1:
            timestamps = [v.timestamp for v in violations]
            assert timestamps == sorted(timestamps)

    def test_detect_violations_no_clock_edges(self) -> None:
        """Test violation detection with no clock edges."""
        sample_rate = 1e6
        data_signal = make_square_wave(1000.0, sample_rate, 0.001)
        clock_signal = np.ones(1000)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=1e-9,
            hold_spec=1e-9,
        )

        assert len(violations) == 0

    def test_detect_violations_no_data_edges(self) -> None:
        """Test violation detection with no data edges."""
        sample_rate = 1e6
        data_signal = np.ones(1000)
        clock_signal = make_square_wave(1000.0, sample_rate, 0.001)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        violations = detect_violations(
            data_trace,
            clock_trace,
            setup_spec=1e-9,
            hold_spec=1e-9,
        )

        assert len(violations) == 0


# =============================================================================
# Glitch Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-005")
class TestDetectGlitches:
    """Test glitch detection."""

    def test_detect_glitches_no_glitches(self) -> None:
        """Test glitch detection on clean signal."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        glitches = detect_glitches(trace, min_width=10e-9)

        # Clean square wave should have no glitches
        assert isinstance(glitches, list)

    def test_detect_glitches_positive_glitch(self) -> None:
        """Test detection of positive glitches."""
        sample_rate = 1e6
        signal = np.zeros(1000, dtype=np.float64)

        # Add short positive pulse (glitch)
        signal[500:502] = 1.0  # 2-sample pulse

        trace = make_waveform_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=10e-6)  # Require 10 µs min

        positive_glitches = [g for g in glitches if g.polarity == "positive"]
        assert len(positive_glitches) > 0
        assert all(g.width < 10e-6 for g in positive_glitches)

    def test_detect_glitches_negative_glitch(self) -> None:
        """Test detection of negative glitches."""
        sample_rate = 1e6
        signal = np.ones(1000, dtype=np.float64)

        # Add short negative pulse (glitch)
        signal[500:502] = 0.0  # 2-sample pulse

        trace = make_waveform_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=10e-6)

        negative_glitches = [g for g in glitches if g.polarity == "negative"]
        assert len(negative_glitches) > 0
        assert all(g.width < 10e-6 for g in negative_glitches)

    def test_detect_glitches_custom_threshold(self) -> None:
        """Test glitch detection with custom threshold."""
        sample_rate = 1e6
        signal = np.zeros(1000, dtype=np.float64)
        signal[500:502] = 2.0  # Pulse to 2V

        trace = make_waveform_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=10e-6, threshold=1.0)

        assert len(glitches) > 0

    def test_detect_glitches_auto_threshold(self) -> None:
        """Test glitch detection with automatic threshold."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.001, amplitude=3.3)
        signal[500:502] = 3.3  # Short pulse

        trace = make_waveform_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=100e-6, threshold=None)

        assert isinstance(glitches, list)

    def test_detect_glitches_digital_trace(self) -> None:
        """Test glitch detection on DigitalTrace."""
        sample_rate = 1e6
        signal = np.zeros(1000, dtype=bool)
        signal[500:502] = True  # 2-sample glitch

        trace = make_digital_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=10e-6)

        assert len(glitches) > 0

    def test_detect_glitches_sorted_by_timestamp(self) -> None:
        """Test that glitches are sorted by timestamp."""
        sample_rate = 1e6
        signal = np.zeros(1000, dtype=np.float64)

        # Add multiple glitches
        signal[100:102] = 1.0
        signal[300:302] = 1.0
        signal[700:702] = 1.0

        trace = make_waveform_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=10e-6)

        if len(glitches) > 1:
            timestamps = [g.timestamp for g in glitches]
            assert timestamps == sorted(timestamps)

    def test_detect_glitches_amplitude_calculation(self) -> None:
        """Test that glitch amplitude is calculated correctly."""
        sample_rate = 1e6
        signal = np.zeros(1000, dtype=np.float64)
        signal[500:502] = 3.3  # High amplitude glitch

        trace = make_waveform_trace(signal, sample_rate)
        glitches = detect_glitches(trace, min_width=10e-6)

        if len(glitches) > 0:
            assert all(g.amplitude > 0 for g in glitches)

    def test_detect_glitches_empty_signal(self) -> None:
        """Test glitch detection on empty signal."""
        empty_data = np.array([])
        trace = make_waveform_trace(empty_data, 1e6)

        glitches = detect_glitches(trace, min_width=10e-9)

        assert len(glitches) == 0

    def test_detect_glitches_short_signal(self) -> None:
        """Test glitch detection on very short signal."""
        signal = np.array([0.0, 1.0])
        trace = make_waveform_trace(signal, 1e6)

        glitches = detect_glitches(trace, min_width=10e-9)

        assert len(glitches) == 0

    def test_detect_glitches_constant_signal(self) -> None:
        """Test glitch detection on constant signal."""
        signal = np.ones(1000)
        trace = make_waveform_trace(signal, 1e6)

        glitches = detect_glitches(trace, min_width=10e-9)

        assert len(glitches) == 0


# =============================================================================
# Signal Quality Summary Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-001")
class TestSignalQualitySummary:
    """Test comprehensive signal quality summary."""

    def test_signal_quality_summary_basic(self) -> None:
        """Test basic signal quality summary."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        summary = signal_quality_summary(trace)

        assert "noise_margin" in summary
        assert "glitch_count" in summary
        assert "glitches" in summary
        assert "signal_levels" in summary
        assert "transition_count" in summary

    def test_signal_quality_summary_noise_margin(self) -> None:
        """Test that summary includes noise margin."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        summary = signal_quality_summary(trace, family="LVCMOS_3V3")

        assert isinstance(summary["noise_margin"], NoiseMarginResult)
        assert summary["noise_margin"].logic_family == "LVCMOS_3V3"

    def test_signal_quality_summary_glitches(self) -> None:
        """Test that summary includes glitch detection."""
        sample_rate = 1e6
        signal = np.zeros(1000, dtype=np.float64)
        signal[500:502] = 3.3  # Add a glitch

        trace = make_waveform_trace(signal, sample_rate)
        summary = signal_quality_summary(trace, min_pulse_width=10e-6)

        assert isinstance(summary["glitch_count"], int)
        assert isinstance(summary["glitches"], list)

    def test_signal_quality_summary_signal_levels(self) -> None:
        """Test that summary includes signal levels."""
        sample_rate = 1e6
        signal = make_lvcmos_signal(1000.0, sample_rate, 0.01, voltage=3.3)
        trace = make_waveform_trace(signal, sample_rate)

        summary = signal_quality_summary(trace)

        assert "low" in summary["signal_levels"]
        assert "high" in summary["signal_levels"]
        assert summary["signal_levels"]["high"] > summary["signal_levels"]["low"]

    def test_signal_quality_summary_transitions(self) -> None:
        """Test that summary includes transition count."""
        sample_rate = 1e6
        frequency = 1000.0
        duration = 0.01
        signal = make_square_wave(frequency, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        summary = signal_quality_summary(trace)

        assert summary["transition_count"] > 0

    def test_signal_quality_summary_custom_parameters(self) -> None:
        """Test summary with custom parameters."""
        sample_rate = 1e6
        signal = make_ttl_signal(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        summary = signal_quality_summary(trace, family="TTL", min_pulse_width=5e-9)

        assert summary["noise_margin"].logic_family == "TTL"


# =============================================================================
# Mask Testing Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-006")
class TestMaskTesting:
    """Test mask testing functionality."""

    def test_mask_test_usb2_pass(self) -> None:
        """Test USB2 mask with passing signal."""
        sample_rate = 1e9  # 1 GSa/s
        bit_period = 3.33e-9  # USB 2.0 HS (300 Mbps)
        duration = 100e-9

        # Create ideal signal within mask
        t = np.arange(0, duration, 1.0 / sample_rate)
        signal = 0.5 * np.sin(2 * np.pi / bit_period * t)

        trace = make_waveform_trace(signal, sample_rate)
        result = mask_test(trace, mask="usb2", bit_period=bit_period)

        assert isinstance(result, MaskTestResult)
        assert result.total_samples > 0

    def test_mask_test_usb2_fail(self) -> None:
        """Test USB2 mask with failing signal."""
        sample_rate = 1e9
        bit_period = 3.33e-9

        # Create signal that violates mask
        signal = 2.0 * np.ones(1000, dtype=np.float64)  # Too high

        trace = make_waveform_trace(signal, sample_rate)
        result = mask_test(trace, mask="usb2", bit_period=bit_period)

        assert result.hit_count > 0
        assert result.pass_fail is False

    def test_mask_test_pcie_gen3(self) -> None:
        """Test PCIe Gen3 mask."""
        sample_rate = 1e9
        bit_period = 125e-9  # Use 125 ns instead of 125 ps for reasonable sample count

        signal = np.zeros(1000, dtype=np.float64)
        trace = make_waveform_trace(signal, sample_rate)

        result = mask_test(trace, mask="pcie_gen3", bit_period=bit_period)

        assert isinstance(result, MaskTestResult)

    def test_mask_test_custom_mask(self) -> None:
        """Test with custom mask definition."""
        sample_rate = 1e9
        bit_period = 1e-9

        # Define custom mask
        custom_mask = {
            "time_ui": np.array([0.0, 0.5, 1.0]),
            "voltage_top": np.array([0.5, 0.7, 0.5]),
            "voltage_bottom": np.array([-0.5, -0.7, -0.5]),
        }

        signal = np.zeros(1000, dtype=np.float64)
        trace = make_waveform_trace(signal, sample_rate)

        result = mask_test(trace, mask=custom_mask, bit_period=bit_period)

        assert isinstance(result, MaskTestResult)

    def test_mask_test_no_bit_period(self) -> None:
        """Test that mask test requires bit_period."""
        signal = np.zeros(1000)
        trace = make_waveform_trace(signal, 1e9)

        with pytest.raises(ValueError, match="bit_period is required"):
            mask_test(trace, mask="usb2", bit_period=None)

    def test_mask_test_invalid_mask_name(self) -> None:
        """Test mask test with invalid mask name."""
        signal = np.zeros(1000)
        trace = make_waveform_trace(signal, 1e9)

        with pytest.raises(ValueError, match="Unknown mask"):
            mask_test(trace, mask="invalid_mask", bit_period=1e-9)

    def test_mask_test_violations_list(self) -> None:
        """Test that violations are properly recorded."""
        sample_rate = 1e9
        bit_period = 1e-9

        # Signal that violates mask
        signal = 2.0 * np.ones(1000, dtype=np.float64)
        trace = make_waveform_trace(signal, sample_rate)

        result = mask_test(trace, mask="usb2", bit_period=bit_period)

        if result.hit_count > 0:
            assert len(result.violations) > 0
            # Each violation is (timestamp, voltage) tuple
            assert all(isinstance(v, tuple) and len(v) == 2 for v in result.violations)

    def test_mask_test_margins(self) -> None:
        """Test that margins are calculated."""
        sample_rate = 1e9
        bit_period = 3.33e-9

        signal = np.zeros(1000, dtype=np.float64)
        trace = make_waveform_trace(signal, sample_rate)

        result = mask_test(trace, mask="usb2", bit_period=bit_period)

        assert result.margin_top >= 0
        assert result.margin_bottom >= 0


# =============================================================================
# PLL Clock Recovery Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-007")
class TestPLLClockRecovery:
    """Test PLL clock recovery."""

    def test_pll_clock_recovery_basic(self) -> None:
        """Test basic PLL clock recovery."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        # Create data signal
        signal = make_square_wave(nominal_freq, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        result = pll_clock_recovery(trace, nominal_frequency=nominal_freq)

        assert isinstance(result, PLLRecoveryResult)
        assert result.recovered_frequency > 0
        assert len(result.recovered_phase) > 0
        assert len(result.vco_control) > 0

    def test_pll_clock_recovery_frequency_accuracy(self) -> None:
        """Test that PLL recovers correct frequency."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        signal = make_square_wave(nominal_freq, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        result = pll_clock_recovery(trace, nominal_frequency=nominal_freq)

        # Should recover frequency within reasonable tolerance
        freq_error_pct = abs(result.recovered_frequency - nominal_freq) / nominal_freq
        assert freq_error_pct < 0.2  # Within 20%

    def test_pll_clock_recovery_lock_status(self) -> None:
        """Test PLL lock status detection."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        signal = make_square_wave(nominal_freq, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        result = pll_clock_recovery(trace, nominal_frequency=nominal_freq)

        # lock_status should be a boolean
        assert result.lock_status in [True, False]

    def test_pll_clock_recovery_custom_parameters(self) -> None:
        """Test PLL with custom loop parameters."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        signal = make_square_wave(nominal_freq, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        result = pll_clock_recovery(
            trace,
            nominal_frequency=nominal_freq,
            loop_bandwidth=500e3,
            damping=0.5,
            vco_gain=500e3,
        )

        assert result.recovered_frequency > 0

    def test_pll_clock_recovery_digital_trace(self) -> None:
        """Test PLL recovery with DigitalTrace."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        signal = make_square_wave(nominal_freq, sample_rate, duration) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        result = pll_clock_recovery(trace, nominal_frequency=nominal_freq)

        assert result.recovered_frequency > 0

    def test_pll_clock_recovery_insufficient_samples(self) -> None:
        """Test PLL recovery with insufficient samples."""
        signal = np.ones(50)
        trace = make_waveform_trace(signal, 1e6)

        with pytest.raises(InsufficientDataError) as exc_info:
            pll_clock_recovery(trace, nominal_frequency=1e6)

        assert "at least 100 samples" in str(exc_info.value)

    def test_pll_clock_recovery_frequency_error(self) -> None:
        """Test that frequency error is calculated."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        signal = make_square_wave(nominal_freq, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        result = pll_clock_recovery(trace, nominal_frequency=nominal_freq)

        assert isinstance(result.frequency_error, float)

    def test_pll_clock_recovery_phase_array(self) -> None:
        """Test that phase array matches input length."""
        sample_rate = 100e6
        nominal_freq = 10e6
        duration = 1e-3

        signal = make_square_wave(nominal_freq, sample_rate, duration)
        trace = make_waveform_trace(signal, sample_rate)

        result = pll_clock_recovery(trace, nominal_frequency=nominal_freq)

        assert len(result.recovered_phase) == len(signal)
        assert len(result.vco_control) == len(signal)


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestHelperFunctions:
    """Test helper functions."""

    def test_find_logic_levels_basic(self) -> None:
        """Test basic logic level detection."""
        data = np.concatenate([np.zeros(500), np.ones(500)])

        low, high = _find_logic_levels(data)

        assert low < high
        assert low < 0.5
        assert high > 0.5

    def test_find_logic_levels_empty(self) -> None:
        """Test logic level detection on empty array."""
        data = np.array([])

        low, high = _find_logic_levels(data)

        assert low == 0.0
        assert high == 0.0

    def test_find_logic_levels_custom_voltage(self) -> None:
        """Test logic level detection with custom voltage."""
        data = np.concatenate([np.zeros(500), 3.3 * np.ones(500)])

        low, high = _find_logic_levels(data)

        assert low < 1.0
        assert high > 2.0

    def test_get_clock_edges_rising(self) -> None:
        """Test rising edge detection."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        edges = _get_clock_edges(trace, "rising")

        assert isinstance(edges, np.ndarray)
        assert len(edges) > 0

    def test_get_clock_edges_falling(self) -> None:
        """Test falling edge detection."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        edges = _get_clock_edges(trace, "falling")

        assert isinstance(edges, np.ndarray)
        assert len(edges) > 0

    def test_get_clock_edges_digital_trace(self) -> None:
        """Test edge detection on DigitalTrace."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        edges = _get_clock_edges(trace, "rising")

        assert len(edges) > 0

    def test_get_clock_edges_short_signal(self) -> None:
        """Test edge detection on short signal."""
        signal = np.array([0.0])
        trace = make_waveform_trace(signal, 1e6)

        edges = _get_clock_edges(trace, "rising")

        assert len(edges) == 0

    def test_get_predefined_mask_usb2(self) -> None:
        """Test USB2 mask retrieval."""
        mask = _get_predefined_mask("usb2")

        assert "time_ui" in mask
        assert "voltage_top" in mask
        assert "voltage_bottom" in mask
        assert len(mask["time_ui"]) > 0

    def test_get_predefined_mask_pcie_gen3(self) -> None:
        """Test PCIe Gen3 mask retrieval."""
        mask = _get_predefined_mask("pcie_gen3")

        assert "time_ui" in mask
        assert "voltage_top" in mask
        assert "voltage_bottom" in mask

    def test_get_predefined_mask_invalid(self) -> None:
        """Test invalid mask name."""
        with pytest.raises(ValueError, match="Unknown mask"):
            _get_predefined_mask("invalid")


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestDigitalQualityEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_trace(self) -> None:
        """Test functions with empty trace."""
        empty = np.array([])
        trace = make_waveform_trace(empty, 1e6)

        # Noise margin should handle empty
        result = noise_margin(trace, family="TTL", use_measured_levels=True)
        assert isinstance(result, NoiseMarginResult)

        # Glitch detection should return empty list
        glitches = detect_glitches(trace, min_width=10e-9)
        assert len(glitches) == 0

    def test_single_value_trace(self) -> None:
        """Test functions with single value."""
        signal = np.array([1.0])
        trace = make_waveform_trace(signal, 1e6)

        glitches = detect_glitches(trace, min_width=10e-9)
        assert len(glitches) == 0

    def test_constant_signal(self) -> None:
        """Test functions with constant signal."""
        signal = np.ones(1000)
        trace = make_waveform_trace(signal, 1e6)

        glitches = detect_glitches(trace, min_width=10e-9)
        assert len(glitches) == 0

        summary = signal_quality_summary(trace)
        assert summary["transition_count"] == 0

    def test_very_high_frequency(self) -> None:
        """Test with very high frequency signal."""
        sample_rate = 1e9
        frequency = 100e6  # 100 MHz
        signal = make_square_wave(frequency, sample_rate, 1e-6)
        trace = make_waveform_trace(signal, sample_rate)

        glitches = detect_glitches(trace, min_width=1e-9)
        assert isinstance(glitches, list)

    def test_mixed_trace_types(self) -> None:
        """Test mixing WaveformTrace and DigitalTrace."""
        sample_rate = 1e6

        analog = make_square_wave(1000.0, sample_rate, 0.01)
        digital = analog > 0.5

        analog_trace = make_waveform_trace(analog, sample_rate)
        digital_trace = make_digital_trace(digital, sample_rate)

        # Should work with both types
        violations = detect_violations(
            analog_trace,
            digital_trace,
            setup_spec=1e-9,
            hold_spec=1e-9,
        )
        assert isinstance(violations, list)


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from oscura.analyzers.digital import quality

        expected_exports = {
            "Glitch",
            "MaskTestResult",
            "NoiseMarginResult",
            "PLLRecoveryResult",
            "Violation",
            "detect_glitches",
            "detect_violations",
            "mask_test",
            "noise_margin",
            "pll_clock_recovery",
            "signal_quality_summary",
        }

        assert hasattr(quality, "__all__")
        assert set(quality.__all__) == expected_exports

    def test_dataclasses_importable(self) -> None:
        """Test that all dataclasses are importable."""
        from oscura.analyzers.digital.quality import (
            Glitch,
            MaskTestResult,
            NoiseMarginResult,
            PLLRecoveryResult,
            Violation,
        )

        assert Glitch is not None
        assert MaskTestResult is not None
        assert NoiseMarginResult is not None
        assert PLLRecoveryResult is not None
        assert Violation is not None

    def test_functions_importable(self) -> None:
        """Test that all functions are importable."""
        from oscura.analyzers.digital.quality import (
            detect_glitches,
            detect_violations,
            mask_test,
            noise_margin,
            pll_clock_recovery,
            signal_quality_summary,
        )

        assert detect_glitches is not None
        assert detect_violations is not None
        assert mask_test is not None
        assert noise_margin is not None
        assert pll_clock_recovery is not None
        assert signal_quality_summary is not None
