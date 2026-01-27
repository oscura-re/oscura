"""Comprehensive tests for eye diagram metrics and measurements.

Tests cover:
- Eye height measurements (basic and BER-extrapolated)
- Eye width measurements (basic and BER-based)
- Q-factor calculations
- Crossing percentage analysis
- Eye contour generation at multiple BER levels
- Complete eye measurements (EyeMetrics)
- Edge cases and error handling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.analyzers.eye.metrics import (
    EyeMetrics,
    crossing_percentage,
    eye_contour,
    eye_height,
    eye_width,
    measure_eye,
    q_factor,
)

# =============================================================================
# Mock EyeDiagram for Testing
# =============================================================================


@dataclass
class MockEyeDiagram:
    """Mock EyeDiagram for testing metrics functions."""

    data: NDArray[np.float64]
    time_axis: NDArray[np.float64]
    samples_per_ui: int


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_eye() -> MockEyeDiagram:
    """Create a clean eye diagram with good separation."""
    samples_per_ui = 100
    num_traces = 200
    time_axis = np.linspace(0, 1, samples_per_ui)

    # Create data with clear high and low levels
    data = np.zeros((num_traces, samples_per_ui))
    for i in range(num_traces):
        if i < num_traces // 2:
            # Low level traces (0V with small noise)
            data[i, :] = np.random.normal(0.0, 0.05, samples_per_ui)
        else:
            # High level traces (1V with small noise)
            data[i, :] = np.random.normal(1.0, 0.05, samples_per_ui)

    return MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)


@pytest.fixture
def noisy_eye() -> MockEyeDiagram:
    """Create a noisy eye diagram with reduced separation."""
    samples_per_ui = 100
    num_traces = 200
    time_axis = np.linspace(0, 1, samples_per_ui)

    # Create data with more noise
    data = np.zeros((num_traces, samples_per_ui))
    for i in range(num_traces):
        if i < num_traces // 2:
            # Low level traces with more noise
            data[i, :] = np.random.normal(0.0, 0.2, samples_per_ui)
        else:
            # High level traces with more noise
            data[i, :] = np.random.normal(1.0, 0.2, samples_per_ui)

    return MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)


@pytest.fixture
def closed_eye() -> MockEyeDiagram:
    """Create a closed eye diagram with no separation."""
    samples_per_ui = 100
    num_traces = 200
    time_axis = np.linspace(0, 1, samples_per_ui)

    # All traces overlap around 0.5V
    data = np.random.normal(0.5, 0.3, (num_traces, samples_per_ui))

    return MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)


@pytest.fixture
def asymmetric_eye() -> MockEyeDiagram:
    """Create an eye diagram with asymmetric high/low levels."""
    samples_per_ui = 100
    num_traces = 200
    time_axis = np.linspace(0, 1, samples_per_ui)

    # Create asymmetric data
    data = np.zeros((num_traces, samples_per_ui))
    for i in range(num_traces):
        if i < num_traces // 2:
            # Low level at 0.3V
            data[i, :] = np.random.normal(0.3, 0.05, samples_per_ui)
        else:
            # High level at 1.2V
            data[i, :] = np.random.normal(1.2, 0.05, samples_per_ui)

    return MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)


# =============================================================================
# Eye Height Tests
# =============================================================================


def test_eye_height_clean(clean_eye: MockEyeDiagram) -> None:
    """Test eye height measurement on clean eye diagram."""
    height = eye_height(clean_eye)

    assert isinstance(height, float)
    assert 0.7 < height < 1.0  # Should be close to 1V difference
    assert not np.isnan(height)


def test_eye_height_noisy(noisy_eye: MockEyeDiagram) -> None:
    """Test eye height measurement on noisy eye diagram."""
    height = eye_height(noisy_eye)

    assert isinstance(height, float)
    # With more noise, eye height should be smaller
    assert 0.0 < height < 1.0
    assert not np.isnan(height)


def test_eye_height_closed(closed_eye: MockEyeDiagram) -> None:
    """Test eye height measurement on closed eye (no opening)."""
    height = eye_height(closed_eye)

    # Closed eye may return 0 or nan
    assert isinstance(height, float)
    assert height >= 0 or np.isnan(height)


def test_eye_height_different_positions(clean_eye: MockEyeDiagram) -> None:
    """Test eye height at different horizontal positions."""
    positions = [0.25, 0.5, 0.75]
    heights = [eye_height(clean_eye, position=pos) for pos in positions]

    # All heights should be reasonable
    for h in heights:
        assert isinstance(h, float)
        assert 0 < h <= 1.0


def test_eye_height_with_ber(clean_eye: MockEyeDiagram) -> None:
    """Test BER-extrapolated eye height."""
    height_basic = eye_height(clean_eye)
    height_ber = eye_height(clean_eye, ber=1e-12)

    assert isinstance(height_ber, float)
    # BER-extrapolated height should be smaller (more conservative)
    assert height_ber <= height_basic


def test_eye_height_asymmetric(asymmetric_eye: MockEyeDiagram) -> None:
    """Test eye height on asymmetric levels."""
    height = eye_height(asymmetric_eye)

    assert isinstance(height, float)
    # Should measure difference between 0.3V and 1.2V (~0.9V), with some tolerance for noise
    assert 0.5 < height < 1.0


# =============================================================================
# Eye Width Tests
# =============================================================================


def test_eye_width_clean(clean_eye: MockEyeDiagram) -> None:
    """Test eye width measurement on clean eye diagram."""
    width = eye_width(clean_eye)

    assert isinstance(width, float)
    assert 0.0 <= width <= 1.0  # Width in UI
    assert not np.isnan(width)


def test_eye_width_noisy(noisy_eye: MockEyeDiagram) -> None:
    """Test eye width measurement on noisy eye diagram."""
    width = eye_width(noisy_eye)

    assert isinstance(width, float)
    # Noisy eye has narrower width
    assert 0.0 <= width <= 1.0


def test_eye_width_closed(closed_eye: MockEyeDiagram) -> None:
    """Test eye width measurement on closed eye."""
    width = eye_width(closed_eye)

    # Closed eye returns nan or reduced width
    assert isinstance(width, float)
    # Width measurement on closed eye is implementation-dependent
    assert np.isnan(width) or 0 <= width <= 1.0


def test_eye_width_with_ber(clean_eye: MockEyeDiagram) -> None:
    """Test BER-based eye width."""
    width_basic = eye_width(clean_eye)
    width_ber = eye_width(clean_eye, ber=1e-12)

    assert isinstance(width_ber, float)
    # BER-based width should be narrower
    assert width_ber <= width_basic


def test_eye_width_different_levels(clean_eye: MockEyeDiagram) -> None:
    """Test eye width at different vertical levels."""
    levels = [0.3, 0.5, 0.7]
    widths = [eye_width(clean_eye, level=lev) for lev in levels]

    # All widths should be valid
    for w in widths:
        assert isinstance(w, float)
        assert 0.0 <= w <= 1.0


# =============================================================================
# Q-Factor Tests
# =============================================================================


def test_q_factor_clean(clean_eye: MockEyeDiagram) -> None:
    """Test Q-factor on clean eye diagram."""
    q = q_factor(clean_eye)

    assert isinstance(q, float)
    # Clean eye should have high Q-factor
    assert q > 5.0
    assert np.isfinite(q)


def test_q_factor_noisy(noisy_eye: MockEyeDiagram) -> None:
    """Test Q-factor on noisy eye diagram."""
    q = q_factor(noisy_eye)

    assert isinstance(q, float)
    # Noisy eye has lower Q-factor
    assert 0.0 < q < 10.0
    assert np.isfinite(q)


def test_q_factor_closed(closed_eye: MockEyeDiagram) -> None:
    """Test Q-factor on closed eye."""
    q = q_factor(closed_eye)

    # Closed eye has very low Q or nan (may have some Q due to noise distribution)
    assert isinstance(q, float)
    assert q < 5.0 or np.isnan(q)  # Much lower than clean eye (Q > 5)


def test_q_factor_different_positions(clean_eye: MockEyeDiagram) -> None:
    """Test Q-factor at different horizontal positions."""
    positions = [0.25, 0.5, 0.75]
    q_values = [q_factor(clean_eye, position=pos) for pos in positions]

    # All Q-factors should be reasonable
    for q_val in q_values:
        assert isinstance(q_val, float)
        assert q_val > 0 or np.isnan(q_val)


def test_q_factor_perfect_separation() -> None:
    """Test Q-factor with perfect separation (zero noise)."""
    samples_per_ui = 100
    num_traces = 200
    time_axis = np.linspace(0, 1, samples_per_ui)

    # Perfect separation: all 0V or all 1V, no noise
    data = np.zeros((num_traces, samples_per_ui))
    data[num_traces // 2 :, :] = 1.0

    eye = MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)
    q = q_factor(eye)

    # With zero noise, Q-factor should be infinite
    assert np.isinf(q)


# =============================================================================
# Crossing Percentage Tests
# =============================================================================


def test_crossing_percentage_clean(clean_eye: MockEyeDiagram) -> None:
    """Test crossing percentage on clean eye diagram."""
    xing = crossing_percentage(clean_eye)

    assert isinstance(xing, float)
    # Should be around 50% for balanced eye
    assert 0.0 <= xing <= 100.0


def test_crossing_percentage_asymmetric(asymmetric_eye: MockEyeDiagram) -> None:
    """Test crossing percentage on asymmetric eye."""
    xing = crossing_percentage(asymmetric_eye)

    assert isinstance(xing, float)
    # Asymmetric levels may have non-50% crossing
    assert 0.0 <= xing <= 100.0


def test_crossing_percentage_closed(closed_eye: MockEyeDiagram) -> None:
    """Test crossing percentage on closed eye."""
    xing = crossing_percentage(closed_eye)

    # Closed eye returns 50% (fallback) or nan
    assert isinstance(xing, float)
    assert np.isnan(xing) or xing == 50.0 or (0.0 <= xing <= 100.0)


# =============================================================================
# Eye Contour Tests
# =============================================================================


def test_eye_contour_default_levels(clean_eye: MockEyeDiagram) -> None:
    """Test eye contour generation with default BER levels."""
    contours = eye_contour(clean_eye)

    assert isinstance(contours, dict)
    # Should have contours for default BER levels
    assert len(contours) > 0

    # Check each contour
    for ber, (times, voltages) in contours.items():
        assert isinstance(ber, float)
        assert isinstance(times, np.ndarray)
        assert isinstance(voltages, np.ndarray)
        assert len(times) == len(voltages)
        assert len(times) > 0


def test_eye_contour_custom_levels(clean_eye: MockEyeDiagram) -> None:
    """Test eye contour with custom BER levels."""
    custom_bers = [1e-6, 1e-9]
    contours = eye_contour(clean_eye, ber_levels=custom_bers)

    assert isinstance(contours, dict)
    # Should have contours for each requested BER
    assert len(contours) <= len(custom_bers)

    for ber in contours:
        assert ber in custom_bers


def test_eye_contour_closed_eye(closed_eye: MockEyeDiagram) -> None:
    """Test eye contour on closed eye (may return empty)."""
    contours = eye_contour(closed_eye)

    assert isinstance(contours, dict)
    # Closed eye may have no valid contours
    # But should not raise an error


def test_eye_contour_structure(clean_eye: MockEyeDiagram) -> None:
    """Test structure of returned contours."""
    contours = eye_contour(clean_eye, ber_levels=[1e-6])

    if len(contours) > 0:
        ber, (times, voltages) = next(iter(contours.items()))

        # Times should be in UI range
        assert np.all((times >= 0) & (times <= 1.0))

        # Voltages should be within reasonable range
        assert voltages.min() >= -1.0
        assert voltages.max() <= 2.0


# =============================================================================
# Complete Eye Measurement Tests
# =============================================================================


def test_measure_eye_clean(clean_eye: MockEyeDiagram) -> None:
    """Test complete eye measurements on clean eye."""
    metrics = measure_eye(clean_eye)

    assert isinstance(metrics, EyeMetrics)

    # Check all fields are present
    assert isinstance(metrics.height, float)
    assert isinstance(metrics.width, float)
    assert isinstance(metrics.q_factor, float)
    assert isinstance(metrics.crossing_percent, float)
    assert isinstance(metrics.mean_high, float)
    assert isinstance(metrics.mean_low, float)
    assert isinstance(metrics.sigma_high, float)
    assert isinstance(metrics.sigma_low, float)
    assert isinstance(metrics.snr, float)
    assert isinstance(metrics.ber_estimate, float)

    # Check reasonable values
    assert metrics.height > 0
    assert 0 <= metrics.width <= 1.0
    assert metrics.q_factor > 0
    assert metrics.mean_high > metrics.mean_low
    assert metrics.sigma_high >= 0
    assert metrics.sigma_low >= 0
    assert 0 <= metrics.ber_estimate <= 0.5


def test_measure_eye_with_ber(clean_eye: MockEyeDiagram) -> None:
    """Test complete eye measurements with custom BER."""
    metrics = measure_eye(clean_eye, ber=1e-9)

    assert isinstance(metrics, EyeMetrics)
    # BER-extrapolated values should be present
    assert metrics.height_at_ber is not None
    assert metrics.width_at_ber is not None


def test_measure_eye_noisy(noisy_eye: MockEyeDiagram) -> None:
    """Test complete eye measurements on noisy eye."""
    metrics = measure_eye(noisy_eye)

    assert isinstance(metrics, EyeMetrics)
    # Noisy eye has lower SNR and Q-factor
    assert metrics.snr < 20.0  # Lower SNR
    assert metrics.q_factor < 10.0  # Lower Q


def test_measure_eye_closed(closed_eye: MockEyeDiagram) -> None:
    """Test complete eye measurements on closed eye."""
    metrics = measure_eye(closed_eye)

    assert isinstance(metrics, EyeMetrics)
    # Closed eye has poor metrics
    assert metrics.height >= 0 or np.isnan(metrics.height)
    assert metrics.q_factor < 5.0 or np.isnan(metrics.q_factor)  # Much worse than clean (Q>5)
    assert metrics.ber_estimate > 0.01  # Relatively high BER


def test_measure_eye_snr_calculation(clean_eye: MockEyeDiagram) -> None:
    """Test SNR calculation in eye measurements."""
    metrics = measure_eye(clean_eye)

    # SNR should be positive for clean eye
    assert metrics.snr > 0
    assert np.isfinite(metrics.snr)


def test_measure_eye_ber_estimate(clean_eye: MockEyeDiagram) -> None:
    """Test BER estimate calculation."""
    metrics = measure_eye(clean_eye)

    # BER estimate should be very low for clean eye
    assert 0 <= metrics.ber_estimate <= 0.5
    assert metrics.ber_estimate < 1e-3  # Clean eye should have low BER


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_empty_eye() -> None:
    """Test handling of eye diagram with minimal traces."""
    samples_per_ui = 100
    time_axis = np.linspace(0, 1, samples_per_ui)
    # Minimal data - just 2 traces at same level
    data = np.full((2, samples_per_ui), 0.5)

    eye = MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)

    # Should handle minimal data gracefully
    height = eye_height(eye)
    assert height == 0 or np.isnan(height)


def test_single_trace_eye() -> None:
    """Test handling of eye diagram with single trace."""
    samples_per_ui = 100
    time_axis = np.linspace(0, 1, samples_per_ui)
    data = np.random.normal(0.5, 0.1, (1, samples_per_ui))

    eye = MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)

    # Should handle single trace (may return nan or zero)
    height = eye_height(eye)
    assert isinstance(height, float)


def test_constant_voltage_eye() -> None:
    """Test handling of eye diagram with constant voltage."""
    samples_per_ui = 100
    time_axis = np.linspace(0, 1, samples_per_ui)
    # All traces at same voltage
    data = np.full((100, samples_per_ui), 0.5)

    eye = MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)

    # Should return zero or nan for no separation
    height = eye_height(eye)
    assert height == 0 or np.isnan(height)

    q = q_factor(eye)
    assert q == 0 or np.isnan(q) or np.isinf(q)


def test_negative_voltage_eye() -> None:
    """Test handling of eye diagram with negative voltages."""
    samples_per_ui = 100
    num_traces = 200
    time_axis = np.linspace(0, 1, samples_per_ui)

    # Create data with negative voltages
    data = np.zeros((num_traces, samples_per_ui))
    for i in range(num_traces):
        if i < num_traces // 2:
            data[i, :] = np.random.normal(-1.0, 0.05, samples_per_ui)
        else:
            data[i, :] = np.random.normal(0.5, 0.05, samples_per_ui)

    eye = MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)

    # Should handle negative voltages
    height = eye_height(eye)
    assert isinstance(height, float)
    assert height > 0  # Should measure positive height difference


def test_very_small_samples_per_ui() -> None:
    """Test handling of eye diagram with very few samples per UI."""
    samples_per_ui = 5  # Very small
    num_traces = 100
    time_axis = np.linspace(0, 1, samples_per_ui)

    data = np.zeros((num_traces, samples_per_ui))
    for i in range(num_traces):
        if i < num_traces // 2:
            data[i, :] = 0.0
        else:
            data[i, :] = 1.0

    eye = MockEyeDiagram(data=data, time_axis=time_axis, samples_per_ui=samples_per_ui)

    # Should handle small sample count
    height = eye_height(eye)
    assert isinstance(height, float)
    assert height > 0


def test_invalid_ber_values(clean_eye: MockEyeDiagram) -> None:
    """Test handling of invalid BER values."""
    # BER <= 0 should be handled
    height = eye_height(clean_eye, ber=0.0)
    assert isinstance(height, float)

    # BER >= 0.5 should be handled
    height = eye_height(clean_eye, ber=0.6)
    assert isinstance(height, float)
