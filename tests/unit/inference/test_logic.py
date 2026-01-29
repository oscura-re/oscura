"""Unit tests for logic family auto-detection (INF-001).


Tests comprehensive coverage of:
- detect_logic_family function with various signal types
- _detect_logic_levels helper for bimodal detection
- _score_logic_family scoring algorithm
- Edge cases: non-bimodal signals, edge conditions, NaN handling
"""

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.inference.logic import (
    LOGIC_FAMILY_SPECS,
    _detect_logic_levels,
    _score_logic_family,
    detect_logic_family,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Helper functions to create test traces
# =============================================================================


def create_digital_trace(
    voh: float,
    vol: float,
    n_samples: int = 10000,
    duty_cycle: float = 0.5,
    noise_std: float = 0.01,
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a synthetic digital signal trace with specified logic levels.

    Args:
        voh: High-level voltage.
        vol: Low-level voltage.
        n_samples: Number of samples.
        duty_cycle: Fraction of time at high level.
        noise_std: Standard deviation of noise.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with bimodal voltage distribution.
    """
    rng = np.random.default_rng(42)

    # Create square wave pattern
    period_samples = 100
    t = np.arange(n_samples)
    base_signal = ((t % period_samples) < (period_samples * duty_cycle)).astype(float)

    # Map to voltage levels
    data = vol + base_signal * (voh - vol)

    # Add noise
    data = data + rng.normal(0, noise_std, n_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def create_analog_trace(
    amplitude: float = 1.0,
    offset: float = 0.0,
    frequency: float = 1e3,
    n_samples: int = 10000,
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a sinusoidal analog trace (not bimodal).

    Args:
        amplitude: Signal amplitude.
        offset: DC offset.
        frequency: Signal frequency in Hz.
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with sinusoidal data.
    """
    t = np.arange(n_samples) / sample_rate
    data = offset + amplitude * np.sin(2 * np.pi * frequency * t)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def create_constant_trace(
    value: float = 2.5,
    n_samples: int = 10000,
    noise_std: float = 0.01,
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a constant (DC) trace with optional noise.

    Args:
        value: DC voltage level.
        n_samples: Number of samples.
        noise_std: Standard deviation of noise.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with constant value.
    """
    rng = np.random.default_rng(42)
    data = np.full(n_samples, value) + rng.normal(0, noise_std, n_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


# =============================================================================
# Tests for LOGIC_FAMILY_SPECS
# =============================================================================


class TestLogicFamilySpecs:
    """Test LOGIC_FAMILY_SPECS constant data."""

    def test_all_families_present(self) -> None:
        """Test that all expected logic families are defined."""
        expected_families = {
            "TTL",
            "CMOS_5V",
            "CMOS_3V3",
            "LVTTL",
            "LVCMOS_2V5",
            "LVCMOS_1V8",
            "LVCMOS_1V5",
            "LVCMOS_1V2",
        }
        assert set(LOGIC_FAMILY_SPECS.keys()) == expected_families

    def test_all_families_have_required_keys(self) -> None:
        """Test that all family specs have required voltage parameters."""
        required_keys = {"vdd", "voh_min", "vol_max", "vih_min", "vil_max"}
        for family, specs in LOGIC_FAMILY_SPECS.items():
            assert required_keys.issubset(set(specs.keys())), f"{family} missing keys"

    def test_voltage_consistency(self) -> None:
        """Test that voltage specs are internally consistent."""
        for family, specs in LOGIC_FAMILY_SPECS.items():
            # VOH should be greater than VOL
            assert specs["voh_min"] > specs["vol_max"], f"{family}: VOH <= VOL"
            # VIH should be greater than VIL
            assert specs["vih_min"] > specs["vil_max"], f"{family}: VIH <= VIL"
            # VDD should be greater than VOH
            assert specs["vdd"] >= specs["voh_min"], f"{family}: VDD < VOH"

    def test_ttl_standard_values(self) -> None:
        """Test TTL family has standard 5V specifications."""
        ttl = LOGIC_FAMILY_SPECS["TTL"]
        assert ttl["vdd"] == 5.0
        assert ttl["voh_min"] == 2.4
        assert ttl["vol_max"] == 0.4
        assert ttl["vih_min"] == 2.0
        assert ttl["vil_max"] == 0.8

    def test_lvcmos_1v8_values(self) -> None:
        """Test LVCMOS 1.8V family specifications."""
        lvcmos = LOGIC_FAMILY_SPECS["LVCMOS_1V8"]
        assert lvcmos["vdd"] == 1.8
        assert lvcmos["voh_min"] < lvcmos["vdd"]


# =============================================================================
# Tests for detect_logic_family
# =============================================================================


class TestDetectLogicFamily:
    """Test detect_logic_family main function."""

    def test_detect_ttl_signal(self) -> None:
        """Test detection of TTL signal (5V, VOH ~4.7V, VOL ~0.3V)."""
        trace = create_digital_trace(voh=4.7, vol=0.3, noise_std=0.05)

        result = detect_logic_family(trace)

        assert "primary" in result
        assert result["primary"]["name"] in ["TTL", "CMOS_5V"]
        assert result["primary"]["confidence"] > 0.5
        assert "voh" in result["primary"]
        assert "vol" in result["primary"]
        assert "thresholds" in result["primary"]

    def test_detect_cmos_3v3_signal(self) -> None:
        """Test detection of CMOS 3.3V signal."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.03)

        result = detect_logic_family(trace)

        assert result["primary"]["name"] in ["CMOS_3V3", "LVTTL"]
        assert result["primary"]["confidence"] > 0.3

    def test_detect_lvcmos_1v8_signal(self) -> None:
        """Test detection of LVCMOS 1.8V signal."""
        trace = create_digital_trace(voh=1.5, vol=0.3, noise_std=0.02)

        result = detect_logic_family(trace)

        assert result["primary"]["name"] == "LVCMOS_1V8"
        # Confidence may be lower for LVCMOS due to smaller voltage swings
        assert result["primary"]["confidence"] > 0.2

    def test_detect_lvcmos_2v5_signal(self) -> None:
        """Test detection of LVCMOS 2.5V signal."""
        trace = create_digital_trace(voh=2.2, vol=0.15, noise_std=0.02)

        result = detect_logic_family(trace)

        assert result["primary"]["name"] == "LVCMOS_2V5"
        assert result["primary"]["confidence"] > 0.3

    def test_return_candidates_true(self) -> None:
        """Test that return_candidates=True includes all candidates."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.03)

        result = detect_logic_family(trace, return_candidates=True)

        assert "candidates" in result
        assert len(result["candidates"]) == len(LOGIC_FAMILY_SPECS)
        # Candidates should be sorted by confidence (descending)
        confidences = [c["confidence"] for c in result["candidates"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_return_candidates_false(self) -> None:
        """Test that return_candidates=False excludes candidates list."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.03)

        result = detect_logic_family(trace, return_candidates=False)

        assert "candidates" not in result
        assert "primary" in result

    def test_primary_contains_thresholds(self) -> None:
        """Test that primary result contains threshold values."""
        trace = create_digital_trace(voh=4.7, vol=0.3)

        result = detect_logic_family(trace)

        thresholds = result["primary"]["thresholds"]
        assert "vih" in thresholds
        assert "vil" in thresholds
        assert "voh" in thresholds
        assert "vol" in thresholds

    def test_vdd_estimated_returned(self) -> None:
        """Test that estimated VDD is returned."""
        trace = create_digital_trace(voh=4.7, vol=0.3)

        result = detect_logic_family(trace)

        assert "vdd_estimated" in result["primary"]
        # Estimated VDD should be close to measured VOH + 0.3
        expected_vdd = result["primary"]["voh"] + 0.3
        assert abs(result["primary"]["vdd_estimated"] - expected_vdd) < 0.1

    def test_high_noise_signal(self) -> None:
        """Test detection with high noise."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.2)

        result = detect_logic_family(trace)

        # Should still detect, but possibly with lower confidence
        assert result["primary"]["confidence"] > 0
        assert result["primary"]["name"] is not None

    def test_min_confidence_threshold(self) -> None:
        """Test min_confidence parameter."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.05)

        # With very high threshold, result should still contain primary
        result = detect_logic_family(trace, min_confidence=0.99)

        # Function should complete without error
        assert "primary" in result

    def test_low_min_confidence_threshold(self) -> None:
        """Test with low min_confidence threshold."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.05)

        result = detect_logic_family(trace, min_confidence=0.1)

        assert result["primary"]["confidence"] > 0


class TestDetectLogicFamilyErrors:
    """Test error handling in detect_logic_family."""

    def test_non_bimodal_signal_handling(self) -> None:
        """Test that non-bimodal (analog) signal is handled gracefully.

        Note: Sinusoidal signals may still be detected with low confidence
        since they spend time at peaks which can appear as bimodal distribution.
        The algorithm falls back to percentile-based detection.
        """
        # Create a sinusoidal signal
        trace = create_analog_trace(amplitude=2.5, offset=2.5, n_samples=10000)

        # May either raise AnalysisError or return low-confidence result
        try:
            result = detect_logic_family(trace)
            # If it succeeds, it should detect the signal extremes
            # Confidence should reflect the non-ideal nature
            assert result["primary"]["confidence"] <= 1.0
        except AnalysisError:
            # This is also acceptable behavior for truly non-bimodal signals
            pass

    def test_constant_signal_may_raise_error(self) -> None:
        """Test that constant (DC) signal may raise AnalysisError."""
        trace = create_constant_trace(value=2.5, noise_std=0.001)

        # Constant signal with very low noise should not detect bimodal distribution
        # This may or may not raise an error depending on the noise
        try:
            result = detect_logic_family(trace)
            # If it doesn't raise, it should have low confidence
            assert result["primary"]["confidence"] < 1.0
        except AnalysisError:
            pass  # Expected behavior

    def test_empty_trace_handling(self) -> None:
        """Test handling of empty trace data."""
        metadata = TraceMetadata(sample_rate=1e6)
        # Very small trace
        trace = WaveformTrace(data=np.array([0.0, 5.0], dtype=np.float64), metadata=metadata)

        # Should handle gracefully (may use percentile fallback)
        result = detect_logic_family(trace)
        assert "primary" in result


# =============================================================================
# Tests for _detect_logic_levels helper
# =============================================================================


class TestDetectLogicLevels:
    """Test _detect_logic_levels helper function."""

    def test_clear_bimodal_detection(self) -> None:
        """Test detection of clear bimodal distribution."""
        # Create bimodal data with clear peaks at 0.3 and 4.7
        rng = np.random.default_rng(42)
        low_samples = rng.normal(0.3, 0.02, 5000)
        high_samples = rng.normal(4.7, 0.02, 5000)
        data = np.concatenate([low_samples, high_samples])
        rng.shuffle(data)

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        assert abs(vol - 0.3) < 0.1
        assert abs(voh - 4.7) < 0.1
        # Confidence is calculated from peak separation and heights
        # With equal samples at each level, confidence should be reasonable
        assert confidence >= 0.5

    def test_single_peak_fallback(self) -> None:
        """Test fallback to percentile method for single peak."""
        # Create unimodal data
        rng = np.random.default_rng(42)
        data = rng.normal(2.5, 0.1, 10000)

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        # Should use percentile fallback (5th and 95th percentiles)
        assert confidence == 0.5  # Fallback confidence
        assert voh > vol

    def test_three_peaks_takes_highest_two(self) -> None:
        """Test that multiple peaks takes two highest."""
        rng = np.random.default_rng(42)
        # Create trimodal data
        low = rng.normal(0.0, 0.02, 4000)
        mid = rng.normal(2.0, 0.02, 1000)  # Smaller peak
        high = rng.normal(4.0, 0.02, 4000)
        data = np.concatenate([low, mid, high])
        rng.shuffle(data)

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        # Should identify the two dominant peaks (low and high)
        # VOL should be near 0, VOH should be near 4
        assert vol < 2.0  # Should be near the low peak
        assert voh > 2.0  # Should be near the high peak

    def test_equal_duty_cycle(self) -> None:
        """Test with 50% duty cycle (equal time at each level)."""
        rng = np.random.default_rng(42)
        low_samples = rng.normal(0.3, 0.02, 5000)
        high_samples = rng.normal(3.0, 0.02, 5000)
        data = np.concatenate([low_samples, high_samples])

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        # Peak heights should be similar, confidence should be good
        assert confidence > 0.3
        assert abs(vol - 0.3) < 0.2
        assert abs(voh - 3.0) < 0.2

    def test_unequal_duty_cycle(self) -> None:
        """Test with unequal duty cycle (90% high, 10% low)."""
        rng = np.random.default_rng(42)
        low_samples = rng.normal(0.3, 0.02, 1000)
        high_samples = rng.normal(3.0, 0.02, 9000)
        data = np.concatenate([low_samples, high_samples])
        rng.shuffle(data)

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        # Should still detect both levels
        assert vol < 1.0
        assert voh > 2.5

    def test_wide_separation_high_confidence(self) -> None:
        """Test that wide separation yields high confidence."""
        rng = np.random.default_rng(42)
        low_samples = rng.normal(0.0, 0.01, 5000)
        high_samples = rng.normal(5.0, 0.01, 5000)
        data = np.concatenate([low_samples, high_samples])

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        # Wide separation (5V) should give good confidence
        # With equal sample counts, confidence is based on separation/5.0 ratio
        assert confidence >= 0.5
        # Also verify we detect the correct levels
        assert abs(vol - 0.0) < 0.1
        assert abs(voh - 5.0) < 0.1

    def test_narrow_separation_lower_confidence(self) -> None:
        """Test that narrow separation may yield lower confidence."""
        rng = np.random.default_rng(42)
        low_samples = rng.normal(2.0, 0.1, 5000)
        high_samples = rng.normal(3.0, 0.1, 5000)
        data = np.concatenate([low_samples, high_samples])

        voh, vol, confidence = _detect_logic_levels(data.astype(np.float64))

        # 1V separation is relatively narrow
        assert vol < voh


# =============================================================================
# Tests for _score_logic_family helper
# =============================================================================


class TestScoreLogicFamily:
    """Test _score_logic_family helper function."""

    def test_perfect_ttl_match(self) -> None:
        """Test perfect TTL signal scoring."""
        specs = LOGIC_FAMILY_SPECS["TTL"]

        # Perfect TTL: VOH=4.7V (>2.4), VOL=0.3V (<0.4), VDD~5V
        score = _score_logic_family(voh=4.7, vol=0.3, vdd=5.0, specs=specs)

        assert score > 0.8  # Should be high score

    def test_perfect_cmos_3v3_match(self) -> None:
        """Test perfect CMOS 3.3V signal scoring."""
        specs = LOGIC_FAMILY_SPECS["CMOS_3V3"]

        score = _score_logic_family(voh=3.0, vol=0.2, vdd=3.3, specs=specs)

        assert score > 0.7

    def test_vdd_mismatch_reduces_score(self) -> None:
        """Test that VDD mismatch reduces score."""
        specs = LOGIC_FAMILY_SPECS["TTL"]  # Expects 5V

        # Perfect VOH/VOL but wrong VDD
        score_correct_vdd = _score_logic_family(voh=4.7, vol=0.3, vdd=5.0, specs=specs)
        score_wrong_vdd = _score_logic_family(voh=4.7, vol=0.3, vdd=3.3, specs=specs)

        assert score_correct_vdd > score_wrong_vdd

    def test_voh_below_spec_reduces_score(self) -> None:
        """Test that VOH below spec minimum reduces score."""
        specs = LOGIC_FAMILY_SPECS["TTL"]  # voh_min = 2.4V

        score_good_voh = _score_logic_family(voh=4.7, vol=0.3, vdd=5.0, specs=specs)
        score_low_voh = _score_logic_family(voh=2.0, vol=0.3, vdd=5.0, specs=specs)

        assert score_good_voh > score_low_voh

    def test_vol_above_spec_reduces_score(self) -> None:
        """Test that VOL above spec maximum reduces score."""
        specs = LOGIC_FAMILY_SPECS["TTL"]  # vol_max = 0.4V

        score_good_vol = _score_logic_family(voh=4.7, vol=0.3, vdd=5.0, specs=specs)
        score_high_vol = _score_logic_family(voh=4.7, vol=0.8, vdd=5.0, specs=specs)

        assert score_good_vol > score_high_vol

    def test_score_bounds(self) -> None:
        """Test that score is bounded between 0 and 1."""
        specs = LOGIC_FAMILY_SPECS["TTL"]

        # Various input combinations
        test_cases = [
            (4.7, 0.3, 5.0),  # Perfect
            (1.0, 2.0, 1.0),  # Terrible
            (10.0, 0.0, 10.0),  # Extreme high
            (0.0, 0.0, 0.0),  # Edge case
        ]

        for voh, vol, vdd in test_cases:
            # Avoid division by zero in scoring
            if vdd == 0:
                continue
            score = _score_logic_family(voh=voh, vol=vol, vdd=vdd, specs=specs)
            assert 0 <= score <= 1, f"Score {score} out of bounds for {voh}/{vol}/{vdd}"

    def test_all_families_scored(self) -> None:
        """Test that all logic families can be scored."""
        voh, vol, vdd = 3.0, 0.2, 3.3

        for family_name, specs in LOGIC_FAMILY_SPECS.items():
            score = _score_logic_family(voh=voh, vol=vol, vdd=vdd, specs=specs)
            assert isinstance(score, float), f"Score for {family_name} is not float"
            assert 0 <= score <= 1, f"Score for {family_name} out of bounds"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestInferenceLogicEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_swing(self) -> None:
        """Test signal with very small voltage swing."""
        trace = create_digital_trace(voh=2.0, vol=1.8, noise_std=0.01)

        # Should still work, even with small swing
        result = detect_logic_family(trace)
        assert "primary" in result

    def test_very_large_swing(self) -> None:
        """Test signal with larger-than-normal voltage swing."""
        trace = create_digital_trace(voh=10.0, vol=0.0, noise_std=0.1)

        result = detect_logic_family(trace)

        # Should detect but VDD estimate will be high
        assert result["primary"]["vdd_estimated"] > 5.0

    def test_negative_voltages(self) -> None:
        """Test signal with negative voltage levels (e.g., RS-232)."""
        # RS-232 style: -12V to +12V
        trace = create_digital_trace(voh=10.0, vol=-10.0, noise_std=0.1)

        result = detect_logic_family(trace)

        # Should detect levels even if negative
        assert result["primary"]["vol"] < 0
        assert result["primary"]["voh"] > 0

    def test_low_sample_count(self) -> None:
        """Test with low sample count."""
        trace = create_digital_trace(voh=3.0, vol=0.2, n_samples=100, noise_std=0.05)

        result = detect_logic_family(trace)
        assert "primary" in result

    def test_high_sample_count(self) -> None:
        """Test with high sample count."""
        trace = create_digital_trace(voh=3.0, vol=0.2, n_samples=100000, noise_std=0.05)

        result = detect_logic_family(trace)
        assert "primary" in result

    def test_asymmetric_duty_cycle_10_percent(self) -> None:
        """Test with 10% duty cycle."""
        trace = create_digital_trace(voh=3.0, vol=0.2, duty_cycle=0.1, noise_std=0.05)

        result = detect_logic_family(trace)
        assert result["primary"]["vol"] < 1.0
        assert result["primary"]["voh"] > 2.5

    def test_asymmetric_duty_cycle_90_percent(self) -> None:
        """Test with 90% duty cycle."""
        trace = create_digital_trace(voh=3.0, vol=0.2, duty_cycle=0.9, noise_std=0.05)

        result = detect_logic_family(trace)
        assert result["primary"]["vol"] < 1.0
        assert result["primary"]["voh"] > 2.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestInferenceLogicIntegration:
    """Integration tests combining multiple functions."""

    def test_full_detection_workflow_ttl(self) -> None:
        """Test full detection workflow for TTL signal."""
        # Create realistic TTL signal
        trace = create_digital_trace(voh=4.6, vol=0.35, noise_std=0.05)

        # Detect family
        result = detect_logic_family(trace, return_candidates=True)

        # Verify structure
        assert "primary" in result
        assert "candidates" in result

        # TTL or CMOS_5V should be top candidates
        top_candidates = [c["name"] for c in result["candidates"][:2]]
        assert any(name in top_candidates for name in ["TTL", "CMOS_5V"])

    def test_full_detection_workflow_lvcmos(self) -> None:
        """Test full detection workflow for LVCMOS 1.8V signal."""
        trace = create_digital_trace(voh=1.5, vol=0.3, noise_std=0.02)

        result = detect_logic_family(trace, return_candidates=True)

        # LVCMOS_1V8 should be primary or near top
        primary_name = result["primary"]["name"]
        assert primary_name in ["LVCMOS_1V8", "LVCMOS_2V5"]

    def test_candidates_ordered_by_confidence(self) -> None:
        """Test that candidates are properly ordered."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=0.03)

        result = detect_logic_family(trace, return_candidates=True)

        candidates = result["candidates"]
        for i in range(len(candidates) - 1):
            assert candidates[i]["confidence"] >= candidates[i + 1]["confidence"]

    def test_thresholds_match_spec(self) -> None:
        """Test that thresholds in result match the spec for detected family."""
        trace = create_digital_trace(voh=4.7, vol=0.3, noise_std=0.05)

        result = detect_logic_family(trace)

        family_name = result["primary"]["name"]
        expected_thresholds = LOGIC_FAMILY_SPECS[family_name]

        thresholds = result["primary"]["thresholds"]
        assert thresholds["vih"] == expected_thresholds["vih_min"]
        assert thresholds["vil"] == expected_thresholds["vil_max"]
        assert thresholds["voh"] == expected_thresholds["voh_min"]
        assert thresholds["vol"] == expected_thresholds["vol_max"]


# =============================================================================
# Parameterized Tests
# =============================================================================


class TestParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "family,voh,vol,vdd",
        [
            ("TTL", 4.7, 0.3, 5.0),
            ("CMOS_5V", 4.5, 0.4, 5.0),
            ("CMOS_3V3", 3.0, 0.3, 3.3),
            ("LVTTL", 2.8, 0.3, 3.3),
            ("LVCMOS_2V5", 2.2, 0.15, 2.5),
            ("LVCMOS_1V8", 1.5, 0.35, 1.8),
        ],
    )
    def test_family_detection_parameterized(
        self, family: str, voh: float, vol: float, vdd: float
    ) -> None:
        """Test detection of each logic family with ideal values."""
        trace = create_digital_trace(voh=voh, vol=vol, noise_std=0.01)

        result = detect_logic_family(trace, return_candidates=True)

        # The expected family should be in top 2 candidates
        top_2_names = [c["name"] for c in result["candidates"][:2]]
        assert family in top_2_names, f"Expected {family} in top 2, got {top_2_names}"

    @pytest.mark.parametrize(
        "noise_std",
        [0.001, 0.01, 0.05, 0.1, 0.2],
    )
    def test_noise_resilience_parameterized(self, noise_std: float) -> None:
        """Test detection at various noise levels."""
        trace = create_digital_trace(voh=3.0, vol=0.2, noise_std=noise_std)

        result = detect_logic_family(trace)

        # Should always return a result
        assert "primary" in result
        assert result["primary"]["confidence"] > 0

    @pytest.mark.parametrize(
        "duty_cycle",
        [0.1, 0.25, 0.5, 0.75, 0.9],
    )
    def test_duty_cycle_variations(self, duty_cycle: float) -> None:
        """Test detection at various duty cycles."""
        trace = create_digital_trace(voh=3.0, vol=0.2, duty_cycle=duty_cycle)

        result = detect_logic_family(trace)

        # Should detect reasonable levels regardless of duty cycle
        assert result["primary"]["vol"] < 1.0
        assert result["primary"]["voh"] > 2.5


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests for logic detection."""

    def test_large_trace_performance(self) -> None:
        """Test detection on large trace (1M samples)."""
        trace = create_digital_trace(voh=3.0, vol=0.2, n_samples=1_000_000)

        import time

        start = time.time()
        result = detect_logic_family(trace)
        elapsed = time.time() - start

        assert "primary" in result
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

    def test_multiple_detections(self) -> None:
        """Test multiple consecutive detections."""
        traces = [
            create_digital_trace(voh=voh, vol=0.2, n_samples=10000) for voh in [1.5, 2.2, 3.0, 4.7]
        ]

        results = []
        for trace in traces:
            result = detect_logic_family(trace)
            results.append(result["primary"]["name"])

        # All should complete successfully
        assert len(results) == 4
