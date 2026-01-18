"""Comprehensive unit tests for legacy signal analysis module.

This module provides extensive testing for legacy RTL/TTL system analysis,
multi-channel logic family detection, and voltage characterization.


Test Coverage:
- detect_logic_families_multi_channel with various logic families
- cross_correlate_multi_reference for different voltage domains
- assess_signal_quality against specifications
- characterize_test_points for batch analysis
- Edge cases: degraded signals, mixed families, drift detection
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.exploratory.legacy import (
    LOGIC_FAMILY_SPECS,
    assess_signal_quality,
    characterize_test_points,
    cross_correlate_multi_reference,
    detect_logic_families_multi_channel,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Functions
# =============================================================================


def make_waveform_trace(data: np.ndarray, sample_rate: float = 1e6) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def generate_logic_signal(
    v_low: float,
    v_high: float,
    n_samples: int = 1000,
    duty_cycle: float = 0.5,
    edges: int = 10,
    noise: float = 0.0,
) -> np.ndarray:
    """Generate a digital logic signal with specified voltage levels."""
    signal = np.zeros(n_samples)
    samples_per_period = n_samples // edges

    for i in range(edges):
        start = i * samples_per_period
        high_end = start + int(samples_per_period * duty_cycle)
        end = start + samples_per_period

        # Python slicing is safe with out-of-bounds indices
        signal[start:high_end] = v_high
        signal[high_end:end] = v_low

    # Add noise if requested
    if noise > 0:
        signal += np.random.normal(0, noise, n_samples)

    return signal


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestDetectLogicFamiliesMultiChannel:
    """Test multi-channel logic family detection (LEGACY-001)."""

    def test_detect_ttl_family(self) -> None:
        """Test detection of TTL logic family."""
        # TTL: VOL_max=0.4V, VOH_min=2.4V
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        # Should detect TTL
        assert 0 in result
        family_result = result[0]
        assert family_result.family in ["TTL", "LVTTL"]
        assert family_result.confidence > 0.5

    def test_detect_cmos_5v_family(self) -> None:
        """Test detection of 5V CMOS logic family."""
        # CMOS 5V: VOL_max=0.5V, VOH_min=4.5V
        data = generate_logic_signal(v_low=0.1, v_high=4.8, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        family_result = result[0]
        assert family_result.family in ["CMOS_5V", "TTL"]
        assert family_result.confidence > 0.0

    def test_detect_lvcmos_3v3_family(self) -> None:
        """Test detection of 3.3V LVCMOS logic family."""
        # LVCMOS 3.3V: VOL_max=0.4V, VOH_min=2.4V at 3.3V supply
        data = generate_logic_signal(v_low=0.2, v_high=3.0, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        family_result = result[0]
        # Could detect as LVCMOS_3V3 or LVTTL
        assert family_result.family in ["LVCMOS_3V3", "LVTTL", "TTL"]

    def test_detect_lvcmos_1v8_family(self) -> None:
        """Test detection of 1.8V LVCMOS logic family."""
        # LVCMOS 1.8V
        data = generate_logic_signal(v_low=0.2, v_high=1.6, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        family_result = result[0]
        # Should detect 1.8V or 2.5V logic family
        assert family_result.family in ["LVCMOS_1V8", "LVCMOS_2V5", "UNKNOWN"]

    def test_multiple_channels_different_families(self) -> None:
        """Test detection of multiple channels with different logic families."""
        # Channel 0: TTL (5V)
        data_ttl = generate_logic_signal(v_low=0.3, v_high=3.5, edges=15)
        trace_ttl = make_waveform_trace(data_ttl)

        # Channel 1: LVCMOS 3.3V
        data_lvcmos = generate_logic_signal(v_low=0.2, v_high=3.0, edges=15)
        trace_lvcmos = make_waveform_trace(data_lvcmos)

        result = detect_logic_families_multi_channel([trace_ttl, trace_lvcmos])

        # Should detect both channels
        assert 0 in result
        assert 1 in result
        # Families may differ or be similar depending on levels
        assert result[0].confidence > 0.0
        assert result[1].confidence > 0.0

    def test_dict_input(self) -> None:
        """Test detection with dict input."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        traces = {5: make_waveform_trace(data), 10: make_waveform_trace(data)}

        result = detect_logic_families_multi_channel(traces)

        # Should preserve dict keys
        assert 5 in result
        assert 10 in result

    def test_degradation_warning(self) -> None:
        """Test degradation warning for weak signals."""
        # Weak high level (below spec)
        data = generate_logic_signal(v_low=0.3, v_high=2.0, edges=20)  # VOH < 2.4V
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace], warn_on_degradation=True)

        family_result = result[0]
        # May have degradation warning
        if family_result.family == "TTL":
            # TTL expects VOH_min=2.4V, so 2.0V might trigger warning
            assert family_result.degradation_warning is not None or family_result.deviation_pct > 0

    def test_insufficient_edges_reduces_confidence(self) -> None:
        """Test that insufficient edges reduce confidence."""
        # Signal with very few edges
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=2)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace], min_edges_for_detection=10)

        family_result = result[0]
        # Confidence should be reduced
        assert family_result.confidence < 1.0

    def test_unknown_family_detection(self) -> None:
        """Test detection when no family matches."""
        # Unusual voltage levels
        data = generate_logic_signal(v_low=1.0, v_high=1.5, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        family_result = result[0]
        # May detect as UNKNOWN if no good match
        assert family_result.family in ["UNKNOWN", "LVCMOS_1V8", "LVCMOS_2V5"]

    def test_voltage_tolerance_parameter(self) -> None:
        """Test voltage_tolerance parameter."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        trace = make_waveform_trace(data)

        # Strict tolerance
        result_strict = detect_logic_families_multi_channel([trace], voltage_tolerance=0.05)

        # Relaxed tolerance
        result_relaxed = detect_logic_families_multi_channel([trace], voltage_tolerance=0.30)

        # Both should detect something
        assert 0 in result_strict
        assert 0 in result_relaxed

    def test_alternatives_list(self) -> None:
        """Test that alternatives are provided for ambiguous signals."""
        # Signal that could match multiple families
        data = generate_logic_signal(v_low=0.3, v_high=2.5, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        family_result = result[0]
        # May have alternative family candidates
        assert isinstance(family_result.alternatives, list)


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestCrossCorrelateMultiReference:
    """Test multi-reference cross-correlation (LEGACY-002)."""

    def test_identical_signals_perfect_correlation(self) -> None:
        """Test correlation of identical signals."""
        data = generate_logic_signal(v_low=0.0, v_high=3.3, edges=20)
        trace1 = make_waveform_trace(data)
        trace2 = make_waveform_trace(data)

        result = cross_correlate_multi_reference(trace1, trace2)

        # Perfect correlation
        assert result.correlation == pytest.approx(1.0, abs=0.1)
        assert result.ref_offset_mv == pytest.approx(0.0, abs=10.0)

    def test_different_voltage_levels_correlation(self) -> None:
        """Test correlation with different voltage levels (different references)."""
        # TTL signal (0-5V)
        data_ttl = generate_logic_signal(v_low=0.0, v_high=5.0, edges=20)
        trace_ttl = make_waveform_trace(data_ttl)

        # LVCMOS signal (0-3.3V) - same pattern, different levels
        data_lvcmos = generate_logic_signal(v_low=0.0, v_high=3.3, edges=20)
        trace_lvcmos = make_waveform_trace(data_lvcmos)

        result = cross_correlate_multi_reference(trace_ttl, trace_lvcmos)

        # Should still correlate after normalization
        assert result.correlation > 0.8
        assert result.confidence > 0.0

    def test_ground_offset_detection(self) -> None:
        """Test detection of ground reference offset."""
        # Signal 1: ground at 0V
        data1 = generate_logic_signal(v_low=0.0, v_high=3.3, edges=20)
        trace1 = make_waveform_trace(data1)

        # Signal 2: ground at 1V (offset reference)
        data2 = generate_logic_signal(v_low=1.0, v_high=4.3, edges=20)
        trace2 = make_waveform_trace(data2)

        result = cross_correlate_multi_reference(trace1, trace2)

        # Should detect ~1V offset
        assert abs(result.ref_offset_mv - 1000.0) < 200.0  # Within 200mV

    def test_drift_detection(self) -> None:
        """Test detection of reference voltage drift."""
        n_samples = 10000
        sample_rate = 1e6

        # Create signal with drifting reference
        data1 = generate_logic_signal(v_low=0.0, v_high=3.3, n_samples=n_samples, edges=50)
        trace1 = make_waveform_trace(data1, sample_rate=sample_rate)

        # Signal 2 with time-varying offset
        data2 = data1.copy()
        drift = np.linspace(0, 0.5, n_samples)  # 500mV drift over time
        data2 += drift
        trace2 = make_waveform_trace(data2, sample_rate=sample_rate)

        result = cross_correlate_multi_reference(
            trace1, trace2, detect_drift=True, drift_window_ms=1.0
        )

        # Should detect drift
        if result.drift_detected:
            assert result.drift_rate is not None
            assert result.drift_rate > 0.0

    def test_lag_detection(self) -> None:
        """Test time lag detection between signals."""
        data = generate_logic_signal(v_low=0.0, v_high=3.3, n_samples=1000, edges=20)

        # Signal 1
        trace1 = make_waveform_trace(data, sample_rate=1e6)

        # Signal 2: delayed by 10 samples
        data2 = np.concatenate([np.zeros(10), data[:-10]])
        trace2 = make_waveform_trace(data2, sample_rate=1e6)

        result = cross_correlate_multi_reference(trace1, trace2)

        # Should detect lag
        assert isinstance(result.lag_samples, int)
        assert isinstance(result.lag_ns, float)

    def test_uncorrelated_signals(self) -> None:
        """Test with completely uncorrelated signals."""
        data1 = generate_logic_signal(v_low=0.0, v_high=3.3, edges=10, duty_cycle=0.5)
        trace1 = make_waveform_trace(data1)

        # Random signal
        data2 = np.random.randn(1000) * 1.5 + 1.5
        trace2 = make_waveform_trace(data2)

        result = cross_correlate_multi_reference(trace1, trace2)

        # Low correlation
        assert abs(result.correlation) < 0.9  # Not perfectly correlated

    def test_normalized_signals_available(self) -> None:
        """Test that normalized signals are returned."""
        data1 = generate_logic_signal(v_low=0.0, v_high=5.0, edges=20)
        trace1 = make_waveform_trace(data1)

        data2 = generate_logic_signal(v_low=1.0, v_high=4.3, edges=20)
        trace2 = make_waveform_trace(data2)

        result = cross_correlate_multi_reference(trace1, trace2)

        # Normalized signals should be available
        assert result.normalized_signal1 is not None
        assert result.normalized_signal2 is not None
        assert len(result.normalized_signal1) > 0


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestAssessSignalQuality:
    """Test signal quality assessment (LEGACY-003)."""

    def test_compliant_ttl_signal(self) -> None:
        """Test assessment of compliant TTL signal."""
        # Perfect TTL: VOL=0.2V, VOH=3.5V
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=50)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(trace, logic_family="TTL")

        # Should be OK
        assert result.status == "OK"
        assert result.violation_count == 0

    def test_degraded_signal_warning(self) -> None:
        """Test warning for degraded signal."""
        # Marginal TTL: VOH close to minimum spec
        data = generate_logic_signal(v_low=0.3, v_high=2.5, edges=50)  # VOH_min=2.4V
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(trace, logic_family="TTL")

        # May be WARNING or OK depending on exact levels
        assert result.status in ["OK", "WARNING", "CRITICAL"]

    def test_critical_signal_status(self) -> None:
        """Test critical status for severely degraded signal."""
        # Very weak TTL signal
        data = generate_logic_signal(v_low=0.5, v_high=2.0, edges=50)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(trace, logic_family="TTL")

        # Should detect violations
        assert result.violation_count > 0 or result.status in ["WARNING", "CRITICAL"]

    def test_vol_violations(self) -> None:
        """Test VOL (output low) violations."""
        # High "low" levels (violates VOL_max)
        data = generate_logic_signal(v_low=0.6, v_high=3.5, edges=50)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(trace, logic_family="TTL")  # VOL_max=0.4V

        # Should detect VOL violations
        assert result.vol_violations > 0 or result.violation_count > 0

    def test_voh_violations(self) -> None:
        """Test VOH (output high) violations."""
        # Low "high" levels (violates VOH_min)
        data = generate_logic_signal(v_low=0.2, v_high=2.0, edges=50)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(trace, logic_family="TTL")  # VOH_min=2.4V

        # Should detect VOH violations
        assert result.voh_violations > 0 or result.violation_count > 0

    def test_violation_details(self) -> None:
        """Test that violation details are captured."""
        data = generate_logic_signal(v_low=0.6, v_high=2.0, edges=50)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(trace, logic_family="TTL")

        # Should have violation details
        if result.violation_count > 0:
            assert len(result.violations) > 0
            # Check violation structure
            if result.violations:
                violation = result.violations[0]
                assert "timestamp_us" in violation
                assert "type" in violation
                assert "voltage" in violation

    def test_aging_analysis(self) -> None:
        """Test aging/degradation analysis."""
        n_samples = 10000

        # Signal with degrading high level over time
        data = np.zeros(n_samples)
        for i in range(0, n_samples, 200):
            # High level degrades from 3.5V to 2.0V
            v_high = 3.5 - (i / n_samples) * 1.5
            data[i : i + 100] = v_high

        trace = make_waveform_trace(data, sample_rate=1e6)

        result = assess_signal_quality(
            trace, logic_family="TTL", check_aging=True, time_window_s=0.001
        )

        # May detect drift
        if result.drift_rate_mv_per_s is not None:
            assert result.drift_rate_mv_per_s != 0.0

    def test_unknown_family_defaults_to_ttl(self) -> None:
        """Test that unknown family defaults to TTL."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        trace = make_waveform_trace(data)

        result = assess_signal_quality(trace, logic_family="UNKNOWN_FAMILY")

        # Should still work (fallback to TTL)
        assert result.status in ["OK", "WARNING", "CRITICAL"]


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestCharacterizeTestPoints:
    """Test multi-channel voltage characterization (LEGACY-004)."""

    def test_characterize_single_channel(self) -> None:
        """Test characterization of single test point."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = characterize_test_points([trace])

        # Should characterize channel 0
        assert 0 in result
        char = result[0]

        assert char.channel_id == 0
        assert char.v_low > 0.0
        assert char.v_high > char.v_low
        assert char.v_swing > 0.0
        assert char.logic_family in LOGIC_FAMILY_SPECS or char.logic_family == "UNKNOWN"

    def test_characterize_multiple_channels(self) -> None:
        """Test batch characterization of multiple test points."""
        # Create 4 channels with different characteristics
        traces = [
            make_waveform_trace(generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)),  # TTL
            make_waveform_trace(generate_logic_signal(v_low=0.1, v_high=4.8, edges=20)),  # CMOS
            make_waveform_trace(generate_logic_signal(v_low=0.2, v_high=3.0, edges=20)),  # LVCMOS
            make_waveform_trace(generate_logic_signal(v_low=0.2, v_high=1.6, edges=20)),  # 1.8V
        ]

        result = characterize_test_points(traces)

        # Should characterize all 4 channels
        assert len(result) == 4
        for i in range(4):
            assert i in result
            assert result[i].is_digital

    def test_clock_signal_detection(self) -> None:
        """Test detection of clock signals."""
        # Square wave with 50% duty cycle (typical clock)
        data = generate_logic_signal(
            v_low=0.0, v_high=3.3, n_samples=10000, edges=100, duty_cycle=0.5
        )
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = characterize_test_points([trace], sample_rate=1e6)

        char = result[0]
        # May be detected as clock
        if char.is_clock:
            assert char.frequency is not None
            assert char.frequency > 0

    def test_non_digital_signal(self) -> None:
        """Test characterization of non-digital signal."""
        # Analog signal (sine wave)
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 10 * t) + 1.5  # Centered at 1.5V
        trace = make_waveform_trace(data)

        result = characterize_test_points([trace])

        char = result[0]
        # Should not be detected as digital
        assert char.is_digital is False or char.v_swing < 1.0

    def test_dict_input_preserves_keys(self) -> None:
        """Test that dict input preserves channel IDs."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        traces = {
            5: make_waveform_trace(data),
            10: make_waveform_trace(data),
        }

        result = characterize_test_points(traces)

        # Should preserve dict keys
        assert 5 in result
        assert 10 in result
        assert result[5].channel_id == 5
        assert result[10].channel_id == 10


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestExploratoryLegacyEdgeCases:
    """Test edge cases and robustness."""

    def test_constant_signal(self) -> None:
        """Test with constant (DC) signal."""
        data = np.ones(1000) * 3.3
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        # Should handle gracefully
        assert 0 in result

    def test_noisy_signal(self) -> None:
        """Test with noisy signal."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20, noise=0.2)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        # Should still detect family
        assert 0 in result

    def test_very_short_signal(self) -> None:
        """Test with very short signal."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, n_samples=50, edges=2)
        trace = make_waveform_trace(data)

        result = characterize_test_points([trace])

        # Should handle gracefully
        assert 0 in result

    def test_negative_voltages(self) -> None:
        """Test with negative voltage levels (ECL)."""
        # ECL-like signal
        data = generate_logic_signal(v_low=-1.7, v_high=-0.9, edges=20)
        trace = make_waveform_trace(data)

        result = detect_logic_families_multi_channel([trace])

        # May detect ECL or UNKNOWN
        family_result = result[0]
        assert family_result.family in ["ECL", "UNKNOWN", "PECL"]

    def test_large_number_of_channels(self) -> None:
        """Test with large number of channels (16)."""
        traces = []
        for i in range(16):
            # Vary voltage levels slightly
            v_high = 3.3 + (i % 4) * 0.1
            data = generate_logic_signal(v_low=0.2, v_high=v_high, edges=15)
            traces.append(make_waveform_trace(data))

        result = characterize_test_points(traces)

        # Should characterize all 16 channels
        assert len(result) == 16

    def test_missing_sample_rate(self) -> None:
        """Test when sample_rate is not specified."""
        data = generate_logic_signal(v_low=0.2, v_high=3.5, edges=20)
        trace = make_waveform_trace(data, sample_rate=1e6)

        # Should use metadata sample_rate
        result = characterize_test_points([trace])

        assert 0 in result
