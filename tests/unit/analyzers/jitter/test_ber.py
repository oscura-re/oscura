"""Comprehensive tests for BER-related jitter analysis functions.

Tests cover:
- Q-factor from BER conversion
- BER from Q-factor conversion
- Total jitter at BER (dual-Dirac model)
- Bathtub curve generation
- Eye opening at BER calculations
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.analyzers.jitter.ber import (
    BathtubCurveResult,
    bathtub_curve,
    ber_from_q_factor,
    eye_opening_at_ber,
    q_factor_from_ber,
    tj_at_ber,
)

# =============================================================================
# Q-Factor and BER Conversion Tests
# =============================================================================


def test_q_factor_from_ber_standard_values() -> None:
    """Test Q-factor calculation for standard BER values."""
    # Well-known Q-factor values
    ber_values = {
        1e-3: 3.09,  # ~3.09
        1e-6: 4.75,  # ~4.75
        1e-9: 5.99,  # ~5.99
        1e-12: 7.03,  # ~7.03
    }

    for ber, expected_q in ber_values.items():
        q = q_factor_from_ber(ber)
        assert abs(q - expected_q) < 0.1, f"BER={ber}: Q={q}, expected ~{expected_q}"


def test_q_factor_from_ber_edge_cases() -> None:
    """Test Q-factor edge cases."""
    # BER = 0 is invalid
    q = q_factor_from_ber(0.0)
    assert np.isnan(q)

    # BER >= 0.5 is invalid
    q = q_factor_from_ber(0.5)
    assert np.isnan(q)

    q = q_factor_from_ber(0.6)
    assert np.isnan(q)

    # Negative BER is invalid
    q = q_factor_from_ber(-0.1)
    assert np.isnan(q)


def test_ber_from_q_factor_standard_values() -> None:
    """Test BER calculation for standard Q-factor values."""
    q_values = {
        3.09: 1e-3,
        4.75: 1e-6,
        5.99: 1e-9,
        7.03: 1e-12,
    }

    for q, expected_ber in q_values.items():
        ber = ber_from_q_factor(q)
        # Allow some tolerance due to floating point
        assert abs(np.log10(ber) - np.log10(expected_ber)) < 0.5


def test_ber_from_q_factor_edge_cases() -> None:
    """Test BER calculation edge cases."""
    # Q = 0 should give BER = 0.5 (no discrimination)
    ber = ber_from_q_factor(0.0)
    assert ber == 0.5

    # Negative Q is invalid
    ber = ber_from_q_factor(-1.0)
    assert ber == 0.5

    # Very high Q should give very low BER
    ber = ber_from_q_factor(10.0)
    assert ber < 1e-20


def test_q_ber_round_trip() -> None:
    """Test that Q-factor and BER conversions are inverses."""
    test_bers = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]

    for ber_in in test_bers:
        q = q_factor_from_ber(ber_in)
        ber_out = ber_from_q_factor(q)

        # Should recover original BER
        assert abs(np.log10(ber_out) - np.log10(ber_in)) < 0.01


# =============================================================================
# Total Jitter at BER Tests
# =============================================================================


def test_tj_at_ber_basic() -> None:
    """Test basic total jitter calculation."""
    rj_rms = 1e-12  # 1 ps RMS
    dj_pp = 10e-12  # 10 ps p-p
    ber = 1e-12

    tj = tj_at_ber(rj_rms, dj_pp, ber)

    assert isinstance(tj, float)
    assert tj > 0
    # TJ should be larger than DJ alone
    assert tj > dj_pp


def test_tj_at_ber_different_bers() -> None:
    """Test that TJ increases with lower BER (more stringent)."""
    rj_rms = 1e-12
    dj_pp = 10e-12

    tj_1e3 = tj_at_ber(rj_rms, dj_pp, ber=1e-3)
    tj_1e6 = tj_at_ber(rj_rms, dj_pp, ber=1e-6)
    tj_1e12 = tj_at_ber(rj_rms, dj_pp, ber=1e-12)

    # Lower BER requires larger jitter margin
    assert tj_1e3 < tj_1e6 < tj_1e12


def test_tj_at_ber_zero_rj() -> None:
    """Test TJ with zero random jitter (DJ only)."""
    rj_rms = 0.0
    dj_pp = 10e-12

    tj = tj_at_ber(rj_rms, dj_pp, ber=1e-12)

    # With zero RJ, TJ should equal DJ
    assert tj == dj_pp


def test_tj_at_ber_zero_dj() -> None:
    """Test TJ with zero deterministic jitter (RJ only)."""
    rj_rms = 1e-12
    dj_pp = 0.0

    tj = tj_at_ber(rj_rms, dj_pp, ber=1e-12)

    # With zero DJ, TJ = 2 * Q * RJ
    assert tj > 0
    assert tj == pytest.approx(2 * q_factor_from_ber(1e-12) * rj_rms, rel=0.01)


def test_tj_at_ber_input_validation() -> None:
    """Test input validation for tj_at_ber."""
    # Negative RJ should raise
    with pytest.raises(ValueError, match="RJ must be non-negative"):
        tj_at_ber(rj_rms=-1e-12, dj_pp=10e-12, ber=1e-12)

    # Negative DJ should raise
    with pytest.raises(ValueError, match="DJ must be non-negative"):
        tj_at_ber(rj_rms=1e-12, dj_pp=-10e-12, ber=1e-12)


def test_tj_at_ber_invalid_ber() -> None:
    """Test TJ with invalid BER values."""
    rj_rms = 1e-12
    dj_pp = 10e-12

    # Invalid BER should return nan
    tj = tj_at_ber(rj_rms, dj_pp, ber=0.0)
    assert np.isnan(tj)

    tj = tj_at_ber(rj_rms, dj_pp, ber=0.6)
    assert np.isnan(tj)


# =============================================================================
# Bathtub Curve Tests
# =============================================================================


@pytest.fixture
def clean_tie_data() -> NDArray[np.float64]:
    """Generate clean TIE data (low jitter)."""
    np.random.seed(42)
    # Simulate RJ + small DJ
    rj = np.random.normal(0, 1e-12, 10000)  # 1 ps RMS
    dj = np.random.choice([-2e-12, 2e-12], 10000)  # 2 ps DJ delta
    return rj + dj


@pytest.fixture
def noisy_tie_data() -> NDArray[np.float64]:
    """Generate noisy TIE data (high jitter)."""
    np.random.seed(43)
    # Simulate larger RJ + DJ
    rj = np.random.normal(0, 5e-12, 10000)  # 5 ps RMS
    dj = np.random.choice([-10e-12, 10e-12], 10000)  # 10 ps DJ delta
    return rj + dj


def test_bathtub_curve_basic(clean_tie_data: NDArray[np.float64]) -> None:
    """Test basic bathtub curve generation."""
    unit_interval = 100e-12  # 100 ps UI

    result = bathtub_curve(clean_tie_data, unit_interval)

    assert isinstance(result, BathtubCurveResult)
    assert len(result.positions) > 0
    assert len(result.ber_left) == len(result.positions)
    assert len(result.ber_right) == len(result.positions)
    assert len(result.ber_total) == len(result.positions)
    assert result.eye_opening >= 0
    assert result.target_ber == 1e-12  # Default


def test_bathtub_curve_positions_range(clean_tie_data: NDArray[np.float64]) -> None:
    """Test that bathtub curve positions span 0 to 1 UI."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval)

    # Positions should span 0 to 1 UI
    assert result.positions[0] == pytest.approx(0.0, abs=0.01)
    assert result.positions[-1] == pytest.approx(1.0, abs=0.01)


def test_bathtub_curve_ber_ranges(clean_tie_data: NDArray[np.float64]) -> None:
    """Test that BER values are in valid range."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval)

    # All BER values should be in [0, 0.5]
    assert np.all((result.ber_left >= 1e-18) & (result.ber_left <= 0.5))
    assert np.all((result.ber_right >= 1e-18) & (result.ber_right <= 0.5))
    assert np.all((result.ber_total >= 1e-18) & (result.ber_total <= 0.5))


def test_bathtub_curve_total_equals_sum(clean_tie_data: NDArray[np.float64]) -> None:
    """Test that total BER equals sum of left and right (before clipping)."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval)

    # Total BER should approximate left + right (within clipping tolerance)
    for i in range(len(result.positions)):
        expected = min(0.5, result.ber_left[i] + result.ber_right[i])
        assert result.ber_total[i] == pytest.approx(expected, rel=0.01, abs=1e-18)


def test_bathtub_curve_eye_opening(clean_tie_data: NDArray[np.float64]) -> None:
    """Test eye opening calculation from bathtub curve."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval, target_ber=1e-12)

    # Clean data should have non-zero eye opening
    assert result.eye_opening > 0
    assert result.eye_opening <= 1.0  # Cannot exceed 1 UI


def test_bathtub_curve_custom_ber(clean_tie_data: NDArray[np.float64]) -> None:
    """Test bathtub curve with custom target BER."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval, target_ber=1e-6)

    assert result.target_ber == 1e-6
    # Eye opening at 1e-6 should be larger than at 1e-12
    result_12 = bathtub_curve(clean_tie_data, unit_interval, target_ber=1e-12)
    assert result.eye_opening >= result_12.eye_opening


def test_bathtub_curve_custom_points(clean_tie_data: NDArray[np.float64]) -> None:
    """Test bathtub curve with custom number of points."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval, n_points=50)

    assert len(result.positions) == 50


def test_bathtub_curve_precomputed_jitter(clean_tie_data: NDArray[np.float64]) -> None:
    """Test bathtub curve with pre-computed jitter components."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval, rj_rms=1e-12, dj_delta=2e-12)

    assert isinstance(result, BathtubCurveResult)
    assert result.eye_opening >= 0


def test_bathtub_curve_noisy_data(noisy_tie_data: NDArray[np.float64]) -> None:
    """Test bathtub curve with noisy data."""
    unit_interval = 100e-12

    result = bathtub_curve(noisy_tie_data, unit_interval)

    # Noisy data should have smaller eye opening
    assert result.eye_opening >= 0
    assert result.eye_opening < 1.0


def test_bathtub_curve_unit_interval_metadata(clean_tie_data: NDArray[np.float64]) -> None:
    """Test that unit interval is stored in result."""
    unit_interval = 100e-12

    result = bathtub_curve(clean_tie_data, unit_interval)

    assert result.unit_interval == unit_interval


# =============================================================================
# Eye Opening at BER Tests
# =============================================================================


def test_eye_opening_at_ber_basic() -> None:
    """Test basic eye opening calculation."""
    rj_rms = 1e-12  # 1 ps RMS
    dj_pp = 10e-12  # 10 ps p-p
    unit_interval = 100e-12  # 100 ps UI

    opening = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-12)

    assert isinstance(opening, float)
    assert 0.0 <= opening <= 1.0  # In UI
    # Should have reasonable opening
    assert opening > 0


def test_eye_opening_at_ber_different_bers() -> None:
    """Test that eye opening decreases with lower BER."""
    rj_rms = 1e-12
    dj_pp = 10e-12
    unit_interval = 100e-12

    opening_1e3 = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-3)
    opening_1e6 = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-6)
    opening_1e12 = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-12)

    # Lower BER (more stringent) should give smaller eye opening
    assert opening_1e3 >= opening_1e6 >= opening_1e12


def test_eye_opening_at_ber_zero_jitter() -> None:
    """Test eye opening with zero jitter (perfect signal)."""
    rj_rms = 0.0
    dj_pp = 0.0
    unit_interval = 100e-12

    opening = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-12)

    # Zero jitter should give full eye opening
    assert opening == 1.0


def test_eye_opening_at_ber_closed_eye() -> None:
    """Test eye opening when jitter exceeds UI (closed eye)."""
    rj_rms = 50e-12  # Large RJ
    dj_pp = 100e-12  # Large DJ
    unit_interval = 100e-12  # 100 ps UI

    opening = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-12)

    # Closed eye should have zero opening
    assert opening == 0.0


def test_eye_opening_at_ber_high_jitter() -> None:
    """Test eye opening with high jitter."""
    rj_rms = 10e-12  # Moderate RJ
    dj_pp = 50e-12  # Moderate DJ
    unit_interval = 100e-12

    opening = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-12)

    # High jitter should reduce opening
    assert 0.0 <= opening < 0.5


def test_eye_opening_at_ber_larger_ui() -> None:
    """Test that larger UI gives more opening for same jitter."""
    rj_rms = 1e-12
    dj_pp = 10e-12

    opening_100ps = eye_opening_at_ber(rj_rms, dj_pp, 100e-12, target_ber=1e-12)
    opening_200ps = eye_opening_at_ber(rj_rms, dj_pp, 200e-12, target_ber=1e-12)

    # Larger UI should have larger opening (in UI)
    assert opening_200ps >= opening_100ps


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_bathtub_curve_empty_data() -> None:
    """Test bathtub curve with empty data."""
    tie_data = np.array([])
    unit_interval = 100e-12

    # Provide explicit RJ/DJ to avoid extraction from empty data
    result = bathtub_curve(tie_data, unit_interval, rj_rms=1e-12, dj_delta=2e-12)
    assert isinstance(result, BathtubCurveResult)


def test_bathtub_curve_single_value() -> None:
    """Test bathtub curve with single value."""
    tie_data = np.array([1e-12])
    unit_interval = 100e-12

    # Should handle minimal data
    result = bathtub_curve(tie_data, unit_interval)
    assert isinstance(result, BathtubCurveResult)


def test_bathtub_curve_nan_values() -> None:
    """Test bathtub curve with NaN values in data."""
    np.random.seed(44)
    tie_data = np.random.normal(0, 1e-12, 1000)
    tie_data[::10] = np.nan  # Insert NaNs

    unit_interval = 100e-12

    # Should filter out NaNs
    result = bathtub_curve(tie_data, unit_interval)
    assert isinstance(result, BathtubCurveResult)
    # BER values should not contain NaN
    assert not np.any(np.isnan(result.ber_total))


def test_bathtub_curve_zero_sigma() -> None:
    """Test bathtub curve with zero sigma (no random jitter)."""
    # All values identical (no RJ)
    tie_data = np.full(1000, 5e-12)
    unit_interval = 100e-12

    result = bathtub_curve(tie_data, unit_interval)

    # Should handle zero sigma (step function)
    assert isinstance(result, BathtubCurveResult)


def test_bathtub_curve_very_small_samples() -> None:
    """Test bathtub curve with very few samples."""
    tie_data = np.random.normal(0, 1e-12, 10)  # Only 10 samples
    unit_interval = 100e-12

    result = bathtub_curve(tie_data, unit_interval)

    assert isinstance(result, BathtubCurveResult)


def test_tj_at_ber_very_large_values() -> None:
    """Test TJ calculation with very large jitter values."""
    rj_rms = 1e-6  # 1 ns RMS (very large)
    dj_pp = 1e-6  # 1 ns p-p (very large)

    tj = tj_at_ber(rj_rms, dj_pp, ber=1e-12)

    assert isinstance(tj, float)
    assert tj > 0
    assert np.isfinite(tj)


def test_eye_opening_at_ber_negative_result_clamped() -> None:
    """Test that negative eye opening is clamped to 0."""
    rj_rms = 100e-12  # Very large jitter
    dj_pp = 200e-12
    unit_interval = 100e-12  # Smaller UI

    opening = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-12)

    # Should be clamped to 0, not negative
    assert opening >= 0.0
    assert opening == 0.0


def test_eye_opening_at_ber_exceeds_unity_clamped() -> None:
    """Test that eye opening > 1.0 is clamped."""
    rj_rms = 0.0
    dj_pp = 0.0
    unit_interval = 100e-12

    opening = eye_opening_at_ber(rj_rms, dj_pp, unit_interval, target_ber=1e-3)

    # Should be clamped to 1.0 UI
    assert 0.0 <= opening <= 1.0
