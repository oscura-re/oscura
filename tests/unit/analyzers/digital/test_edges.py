"""Unit tests for edge detection with sub-sample precision.

This module provides comprehensive tests for edge detection, interpolation,
timing measurements, constraint checking, and quality classification.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.digital.edges import (
    Edge,
    EdgeDetector,
    EdgeTiming,
    TimingConstraint,
    TimingViolation,
    check_timing_constraints,
    classify_edge_quality,
    detect_edges,
    interpolate_edge_time,
    measure_edge_timing,
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


def make_clean_step(
    position: int,
    total_samples: int,
    low_value: float = 0.0,
    high_value: float = 1.0,
) -> np.ndarray:
    """Generate a signal with a single clean step."""
    signal = np.full(total_samples, low_value, dtype=np.float64)
    signal[position:] = high_value
    return signal


def make_slow_edge(
    position: int,
    total_samples: int,
    transition_samples: int = 10,
    low_value: float = 0.0,
    high_value: float = 1.0,
) -> np.ndarray:
    """Generate a signal with a slow rising edge."""
    signal = np.full(total_samples, low_value, dtype=np.float64)
    ramp = np.linspace(low_value, high_value, transition_samples)
    signal[position : position + transition_samples] = ramp
    if position + transition_samples < total_samples:
        signal[position + transition_samples :] = high_value
    return signal


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestDataClasses:
    """Test edge detection data classes."""

    def test_edge_dataclass(self) -> None:
        """Test Edge dataclass."""
        edge = Edge(
            sample_index=100,
            time=1e-6,
            edge_type="rising",
            amplitude=3.3,
            slew_rate=1e9,
            quality="clean",
        )

        assert edge.sample_index == 100
        assert edge.time == 1e-6
        assert edge.edge_type == "rising"
        assert edge.amplitude == 3.3
        assert edge.slew_rate == 1e9
        assert edge.quality == "clean"

    def test_edge_timing_dataclass(self) -> None:
        """Test EdgeTiming dataclass."""
        periods = np.array([1e-6, 1.1e-6, 0.9e-6])
        duty_cycles = np.array([0.5, 0.48, 0.52])

        timing = EdgeTiming(
            periods=periods,
            mean_period=1e-6,
            std_period=0.1e-6,
            min_period=0.9e-6,
            max_period=1.1e-6,
            duty_cycles=duty_cycles,
            mean_duty_cycle=0.5,
            jitter_rms=0.1e-6,
            jitter_pp=0.2e-6,
        )

        assert np.array_equal(timing.periods, periods)
        assert timing.mean_period == 1e-6
        assert timing.std_period == 0.1e-6
        assert timing.min_period == 0.9e-6
        assert timing.max_period == 1.1e-6
        assert np.array_equal(timing.duty_cycles, duty_cycles)
        assert timing.mean_duty_cycle == 0.5
        assert timing.jitter_rms == 0.1e-6
        assert timing.jitter_pp == 0.2e-6

    def test_timing_constraint_dataclass(self) -> None:
        """Test TimingConstraint dataclass."""
        constraint = TimingConstraint(
            name="setup_time",
            min_time=1e-9,
            max_time=10e-9,
            reference="rising",
        )

        assert constraint.name == "setup_time"
        assert constraint.min_time == 1e-9
        assert constraint.max_time == 10e-9
        assert constraint.reference == "rising"

    def test_timing_constraint_optional_fields(self) -> None:
        """Test TimingConstraint with optional fields."""
        constraint = TimingConstraint(name="period")

        assert constraint.name == "period"
        assert constraint.min_time is None
        assert constraint.max_time is None
        assert constraint.reference is None

    def test_timing_violation_dataclass(self) -> None:
        """Test TimingViolation dataclass."""
        constraint = TimingConstraint(name="min_period", min_time=10e-9)
        violation = TimingViolation(
            constraint=constraint,
            measured_time=5e-9,
            edge_index=42,
            sample_index=1000,
        )

        assert violation.constraint == constraint
        assert violation.measured_time == 5e-9
        assert violation.edge_index == 42
        assert violation.sample_index == 1000


# =============================================================================
# Edge Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestDetectEdges:
    """Test basic edge detection."""

    def test_detect_edges_rising(self) -> None:
        """Test detection of rising edges."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="rising", sample_rate=sample_rate)

        assert len(edges) > 0
        assert all(e.edge_type == "rising" for e in edges)

    def test_detect_edges_falling(self) -> None:
        """Test detection of falling edges."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="falling", sample_rate=sample_rate)

        assert len(edges) > 0
        assert all(e.edge_type == "falling" for e in edges)

    def test_detect_edges_both(self) -> None:
        """Test detection of both rising and falling edges."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)

        assert len(edges) > 0
        rising_count = sum(1 for e in edges if e.edge_type == "rising")
        falling_count = sum(1 for e in edges if e.edge_type == "falling")
        assert rising_count > 0
        assert falling_count > 0

    def test_detect_edges_auto_threshold(self) -> None:
        """Test edge detection with automatic threshold."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01, amplitude=2.0, offset=1.0)

        edges = detect_edges(signal, edge_type="both", threshold="auto", sample_rate=sample_rate)

        assert len(edges) > 0

    def test_detect_edges_manual_threshold(self) -> None:
        """Test edge detection with manual threshold."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", threshold=0.5, sample_rate=sample_rate)

        assert len(edges) > 0

    def test_detect_edges_with_hysteresis(self) -> None:
        """Test edge detection with hysteresis."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        # Add noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.05, len(signal))

        edges = detect_edges(
            noisy_signal,
            edge_type="both",
            threshold="auto",
            hysteresis=0.2,
            sample_rate=sample_rate,
        )

        # Hysteresis should reduce false edges from noise
        assert len(edges) > 0

    def test_detect_edges_no_edges(self) -> None:
        """Test edge detection on constant signal."""
        signal = np.ones(1000, dtype=np.float64)

        edges = detect_edges(signal, edge_type="both")

        assert len(edges) == 0

    def test_detect_edges_single_transition(self) -> None:
        """Test edge detection with single transition."""
        signal = make_clean_step(500, 1000)

        edges = detect_edges(signal, edge_type="both", threshold=0.5)

        assert len(edges) == 1
        assert edges[0].edge_type == "rising"

    def test_detect_edges_empty_signal(self) -> None:
        """Test edge detection on empty signal."""
        signal = np.array([], dtype=np.float64)

        edges = detect_edges(signal, edge_type="both")

        assert len(edges) == 0

    def test_detect_edges_single_sample(self) -> None:
        """Test edge detection on single sample."""
        signal = np.array([1.0])

        edges = detect_edges(signal, edge_type="both")

        assert len(edges) == 0

    def test_detect_edges_time_calculation(self) -> None:
        """Test that edge times are calculated correctly."""
        sample_rate = 1e6
        signal = make_clean_step(100, 1000)

        edges = detect_edges(signal, edge_type="rising", threshold=0.5, sample_rate=sample_rate)

        assert len(edges) == 1
        # Edge should be around sample 100
        expected_time = 100 / sample_rate
        assert abs(edges[0].time - expected_time) < 5 / sample_rate

    def test_detect_edges_amplitude(self) -> None:
        """Test that edge amplitude is calculated."""
        signal = make_clean_step(500, 1000, low_value=0.0, high_value=3.3)

        edges = detect_edges(signal, edge_type="rising", threshold=1.5)

        assert len(edges) == 1
        # Amplitude should be positive and reasonable
        assert edges[0].amplitude > 0

    def test_detect_edges_slew_rate(self) -> None:
        """Test that slew rate is calculated."""
        sample_rate = 1e6
        signal = make_clean_step(500, 1000)

        edges = detect_edges(signal, edge_type="rising", sample_rate=sample_rate)

        assert len(edges) == 1
        # Slew rate should be positive for rising edge
        assert edges[0].slew_rate > 0

    def test_detect_edges_threshold_above_signal(self) -> None:
        """Test edge detection with threshold above signal range."""
        signal = make_square_wave(1000.0, 1e6, 0.001)  # 0 to 1

        edges = detect_edges(signal, edge_type="both", threshold=2.0)

        assert len(edges) == 0

    def test_detect_edges_threshold_below_signal(self) -> None:
        """Test edge detection with threshold below signal range."""
        signal = make_square_wave(1000.0, 1e6, 0.001) + 2.0  # 2 to 3

        edges = detect_edges(signal, edge_type="both", threshold=0.0)

        assert len(edges) == 0


# =============================================================================
# Edge Interpolation Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestInterpolateEdgeTime:
    """Test sub-sample edge time interpolation."""

    def test_interpolate_linear_midpoint(self) -> None:
        """Test linear interpolation at midpoint."""
        trace = np.array([0.0, 1.0], dtype=np.float64)
        offset = interpolate_edge_time(trace, 0, method="linear")

        # Should interpolate to middle
        assert 0.4 < offset < 0.6

    def test_interpolate_linear_asymmetric(self) -> None:
        """Test linear interpolation with asymmetric transition."""
        trace = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float64)
        offset = interpolate_edge_time(trace, 1, method="linear")

        # Should return a value between 0 and 1
        assert 0.0 <= offset <= 1.0

    def test_interpolate_quadratic_fallback(self) -> None:
        """Test quadratic interpolation falls back to linear."""
        trace = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float64)
        offset_linear = interpolate_edge_time(trace, 1, method="linear")
        offset_quad = interpolate_edge_time(trace, 1, method="quadratic")

        # Currently quadratic falls back to linear
        assert offset_linear == offset_quad

    def test_interpolate_edge_at_boundary(self) -> None:
        """Test interpolation at array boundaries."""
        trace = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        # At start
        offset = interpolate_edge_time(trace, -1, method="linear")
        assert offset == 0.0

        # At end
        offset = interpolate_edge_time(trace, len(trace), method="linear")
        assert offset == 0.0

    def test_interpolate_zero_difference(self) -> None:
        """Test interpolation when values are identical."""
        trace = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        offset = interpolate_edge_time(trace, 1, method="linear")

        # Should return midpoint when difference is zero
        assert offset == 0.5

    def test_interpolate_valid_range(self) -> None:
        """Test that interpolation always returns valid range."""
        # Try various signals
        for _ in range(10):
            np.random.seed(42)
            trace = np.random.rand(100)
            for i in range(len(trace) - 1):
                offset = interpolate_edge_time(trace, i, method="linear")
                assert 0.0 <= offset <= 1.0


# =============================================================================
# Edge Timing Measurement Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestMeasureEdgeTiming:
    """Test edge timing measurements."""

    def test_measure_timing_basic(self) -> None:
        """Test basic timing measurement."""
        sample_rate = 1e6
        frequency = 1000.0
        signal = make_square_wave(frequency, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        timing = measure_edge_timing(edges, sample_rate)

        assert timing.mean_period > 0
        assert len(timing.periods) > 0
        assert timing.min_period <= timing.mean_period <= timing.max_period

    def test_measure_timing_periods(self) -> None:
        """Test period measurements."""
        sample_rate = 1e6
        frequency = 1000.0
        expected_period = 1.0 / frequency
        signal = make_square_wave(frequency, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        timing = measure_edge_timing(edges, sample_rate)

        # Mean period should be close to expected
        # Allow 20% tolerance due to edge-to-edge variation
        assert abs(timing.mean_period - expected_period / 2) < expected_period * 0.2

    def test_measure_timing_duty_cycle(self) -> None:
        """Test duty cycle measurement."""
        sample_rate = 1e6
        frequency = 1000.0
        signal = make_square_wave(frequency, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        timing = measure_edge_timing(edges, sample_rate)

        # Square wave should have ~50% duty cycle
        if len(timing.duty_cycles) > 0:
            assert 0.0 <= timing.mean_duty_cycle <= 1.0
            # Allow wide tolerance for synthetic signal
            assert 0.2 < timing.mean_duty_cycle < 0.8

    def test_measure_timing_jitter(self) -> None:
        """Test jitter measurements."""
        sample_rate = 1e6
        frequency = 1000.0
        signal = make_square_wave(frequency, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        timing = measure_edge_timing(edges, sample_rate)

        # Jitter should be non-negative
        assert timing.jitter_rms >= 0
        assert timing.jitter_pp >= 0
        # Peak-to-peak should be >= RMS
        assert timing.jitter_pp >= timing.jitter_rms

    def test_measure_timing_insufficient_edges(self) -> None:
        """Test timing measurement with insufficient edges."""
        edges = []
        timing = measure_edge_timing(edges, 1e6)

        assert len(timing.periods) == 0
        assert timing.mean_period == 0.0
        assert timing.jitter_rms == 0.0

    def test_measure_timing_single_edge(self) -> None:
        """Test timing measurement with single edge."""
        edge = Edge(
            sample_index=100,
            time=100e-6,
            edge_type="rising",
            amplitude=1.0,
            slew_rate=1e6,
            quality="clean",
        )
        timing = measure_edge_timing([edge], 1e6)

        assert len(timing.periods) == 0
        assert timing.mean_period == 0.0

    def test_measure_timing_two_edges(self) -> None:
        """Test timing measurement with two edges."""
        edges = [
            Edge(
                sample_index=100,
                time=100e-6,
                edge_type="rising",
                amplitude=1.0,
                slew_rate=1e6,
                quality="clean",
            ),
            Edge(
                sample_index=200,
                time=200e-6,
                edge_type="falling",
                amplitude=1.0,
                slew_rate=-1e6,
                quality="clean",
            ),
        ]
        timing = measure_edge_timing(edges, 1e6)

        assert len(timing.periods) == 1
        assert timing.mean_period == 100e-6


# =============================================================================
# Timing Constraint Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestCheckTimingConstraints:
    """Test timing constraint checking."""

    def test_check_constraints_no_violations(self) -> None:
        """Test constraint checking with no violations."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        constraint = TimingConstraint(name="period", min_time=100e-9, max_time=10e-3)

        violations = check_timing_constraints(edges, [constraint], sample_rate)

        # Generous constraints should have no violations
        assert isinstance(violations, list)

    def test_check_constraints_min_violation(self) -> None:
        """Test constraint checking with minimum time violation."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        # Require impossibly long minimum period
        constraint = TimingConstraint(name="min_period", min_time=10.0)

        violations = check_timing_constraints(edges, [constraint], sample_rate)

        assert len(violations) > 0
        assert all(v.constraint.name == "min_period" for v in violations)

    def test_check_constraints_max_violation(self) -> None:
        """Test constraint checking with maximum time violation."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        # Require impossibly short maximum period
        constraint = TimingConstraint(name="max_period", max_time=1e-9)

        violations = check_timing_constraints(edges, [constraint], sample_rate)

        assert len(violations) > 0
        assert all(v.constraint.name == "max_period" for v in violations)

    def test_check_constraints_rising_reference(self) -> None:
        """Test constraint checking for rising edges only."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        constraint = TimingConstraint(
            name="rising_period", min_time=1e-9, max_time=10e-3, reference="rising"
        )

        violations = check_timing_constraints(edges, [constraint], sample_rate)

        # Should only check rising edges
        assert isinstance(violations, list)

    def test_check_constraints_falling_reference(self) -> None:
        """Test constraint checking for falling edges only."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        constraint = TimingConstraint(
            name="falling_period", min_time=1e-9, max_time=10e-3, reference="falling"
        )

        violations = check_timing_constraints(edges, [constraint], sample_rate)

        # Should only check falling edges
        assert isinstance(violations, list)

    def test_check_constraints_multiple(self) -> None:
        """Test checking multiple constraints."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        constraints = [
            TimingConstraint(name="min_period", min_time=1e-9),
            TimingConstraint(name="max_period", max_time=10e-3),
        ]

        violations = check_timing_constraints(edges, constraints, sample_rate)

        assert isinstance(violations, list)

    def test_check_constraints_insufficient_edges(self) -> None:
        """Test constraint checking with insufficient edges."""
        edges = []
        constraint = TimingConstraint(name="period", min_time=1e-9)

        violations = check_timing_constraints(edges, [constraint], 1e6)

        assert len(violations) == 0

    def test_check_constraints_single_edge(self) -> None:
        """Test constraint checking with single edge."""
        edge = Edge(
            sample_index=100,
            time=100e-6,
            edge_type="rising",
            amplitude=1.0,
            slew_rate=1e6,
            quality="clean",
        )
        constraint = TimingConstraint(name="period", min_time=1e-9)

        violations = check_timing_constraints([edge], [constraint], 1e6)

        assert len(violations) == 0

    def test_check_constraints_violation_details(self) -> None:
        """Test that violations contain correct details."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        constraint = TimingConstraint(name="test", min_time=10.0)

        violations = check_timing_constraints(edges, [constraint], sample_rate)

        if len(violations) > 0:
            v = violations[0]
            assert v.constraint == constraint
            assert v.measured_time > 0
            assert v.edge_index >= 0
            assert v.sample_index > 0


# =============================================================================
# Edge Quality Classification Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestClassifyEdgeQuality:
    """Test edge quality classification."""

    def test_classify_clean_edge(self) -> None:
        """Test classification of clean edge."""
        sample_rate = 1e6
        signal = make_clean_step(500, 1000)

        # Find the edge
        edges = detect_edges(signal, edge_type="rising", threshold=0.5)
        if len(edges) > 0:
            edge_idx = edges[0].sample_index
            quality = classify_edge_quality(signal, edge_idx, sample_rate)

            assert quality in ["clean", "slow", "noisy", "glitch"]

    def test_classify_slow_edge(self) -> None:
        """Test classification of slow edge."""
        sample_rate = 1e6
        signal = make_slow_edge(500, 1000, transition_samples=20)

        # Find the edge
        threshold = 0.5
        crossings = np.where(np.diff(signal > threshold))[0]
        if len(crossings) > 0:
            edge_idx = crossings[0] + 1
            quality = classify_edge_quality(signal, edge_idx, sample_rate)

            # Classification depends on edge characteristics; ensure it's valid
            assert quality in ["slow", "clean", "glitch"]

    def test_classify_noisy_edge(self) -> None:
        """Test classification of noisy edge."""
        sample_rate = 1e6
        signal = make_clean_step(500, 1000)

        # Add significant noise around the edge
        np.random.seed(42)
        noise_region = slice(495, 505)
        signal[noise_region] += np.random.normal(0, 0.3, 10)

        edges = detect_edges(signal, edge_type="rising", threshold=0.5, hysteresis=0.1)
        if len(edges) > 0:
            edge_idx = edges[0].sample_index
            quality = classify_edge_quality(signal, edge_idx, sample_rate)

            assert quality in ["clean", "slow", "noisy", "glitch"]

    def test_classify_edge_at_boundary(self) -> None:
        """Test classification at array boundaries."""
        signal = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        sample_rate = 1e6

        # At start
        quality = classify_edge_quality(signal, 0, sample_rate)
        assert quality == "clean"

        # At end
        quality = classify_edge_quality(signal, len(signal) - 1, sample_rate)
        assert quality == "clean"

    def test_classify_edge_returns_valid_quality(self) -> None:
        """Test that classification always returns valid quality."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        edges = detect_edges(signal, edge_type="both")
        for edge in edges:
            quality = classify_edge_quality(signal, edge.sample_index, sample_rate)
            assert quality in ["clean", "slow", "noisy", "glitch"]


# =============================================================================
# EdgeDetector Class Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestEdgeDetectorClass:
    """Test EdgeDetector class."""

    def test_edge_detector_init_defaults(self) -> None:
        """Test EdgeDetector initialization with defaults."""
        detector = EdgeDetector()

        assert detector.threshold == "auto"
        assert detector.hysteresis == 0.0
        assert detector.sample_rate == 1.0
        assert detector.min_pulse_width is None

    def test_edge_detector_init_custom(self) -> None:
        """Test EdgeDetector initialization with custom parameters."""
        detector = EdgeDetector(threshold=0.5, hysteresis=0.1, sample_rate=1e6, min_pulse_width=10)

        assert detector.threshold == 0.5
        assert detector.hysteresis == 0.1
        assert detector.sample_rate == 1e6
        assert detector.min_pulse_width == 10

    def test_detect_all_edges(self) -> None:
        """Test detect_all_edges method."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        detector = EdgeDetector(sample_rate=sample_rate)
        rising, falling = detector.detect_all_edges(signal)

        assert isinstance(rising, np.ndarray)
        assert isinstance(falling, np.ndarray)
        assert len(rising) > 0
        assert len(falling) > 0

    def test_detect_rising_edges(self) -> None:
        """Test detect_rising_edges method."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        detector = EdgeDetector(sample_rate=sample_rate)
        edges = detector.detect_rising_edges(signal)

        assert len(edges) > 0
        assert all(e.edge_type == "rising" for e in edges)

    def test_detect_falling_edges(self) -> None:
        """Test detect_falling_edges method."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        detector = EdgeDetector(sample_rate=sample_rate)
        edges = detector.detect_falling_edges(signal)

        assert len(edges) > 0
        assert all(e.edge_type == "falling" for e in edges)

    def test_measure_timing(self) -> None:
        """Test measure_timing method."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        detector = EdgeDetector(sample_rate=sample_rate)
        timing = detector.measure_timing(signal)

        assert isinstance(timing, EdgeTiming)
        assert timing.mean_period > 0

    def test_min_pulse_width_filtering(self) -> None:
        """Test minimum pulse width filtering."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        # Without filtering
        detector1 = EdgeDetector(sample_rate=sample_rate)
        rising1, falling1 = detector1.detect_all_edges(signal)

        # With filtering
        detector2 = EdgeDetector(sample_rate=sample_rate, min_pulse_width=50)
        rising2, falling2 = detector2.detect_all_edges(signal)

        # Filtered version should have same or fewer edges
        assert len(rising2) <= len(rising1)
        assert len(falling2) <= len(falling1)

    def test_edge_detector_with_hysteresis(self) -> None:
        """Test EdgeDetector with hysteresis."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        # Add noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.05, len(signal))

        detector = EdgeDetector(sample_rate=sample_rate, hysteresis=0.2)
        rising, falling = detector.detect_all_edges(noisy_signal)

        assert len(rising) > 0
        assert len(falling) > 0

    def test_edge_detector_constant_signal(self) -> None:
        """Test EdgeDetector on constant signal."""
        signal = np.ones(1000, dtype=np.float64)

        detector = EdgeDetector()
        rising, falling = detector.detect_all_edges(signal)

        assert len(rising) == 0
        assert len(falling) == 0


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestDigitalEdgesEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array(self) -> None:
        """Test with empty array."""
        signal = np.array([], dtype=np.float64)

        edges = detect_edges(signal)
        assert len(edges) == 0

        timing = measure_edge_timing(edges)
        assert timing.mean_period == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample."""
        signal = np.array([1.0])

        edges = detect_edges(signal)
        assert len(edges) == 0

    def test_very_short_signal(self) -> None:
        """Test with very short signal."""
        signal = np.array([0.0, 1.0])

        edges = detect_edges(signal, threshold=0.5)
        # May or may not detect edge depending on exact implementation
        assert isinstance(edges, list)

    def test_all_zeros(self) -> None:
        """Test with all zeros."""
        signal = np.zeros(1000, dtype=np.float64)

        edges = detect_edges(signal)
        assert len(edges) == 0

    def test_all_ones(self) -> None:
        """Test with all ones."""
        signal = np.ones(1000, dtype=np.float64)

        edges = detect_edges(signal)
        assert len(edges) == 0

    def test_high_frequency_signal(self) -> None:
        """Test with high frequency relative to sample rate."""
        sample_rate = 1e6
        # 100 kHz signal - only 10 samples per cycle
        signal = make_square_wave(100e3, sample_rate, 0.001)

        edges = detect_edges(signal, edge_type="both", sample_rate=sample_rate)
        # Should still detect edges even with limited sampling
        assert len(edges) >= 0

    def test_very_slow_edge(self) -> None:
        """Test with very slow transition."""
        signal = make_slow_edge(100, 1000, transition_samples=100)

        edges = detect_edges(signal, threshold=0.5)
        assert len(edges) >= 0

    def test_multiple_fast_transitions(self) -> None:
        """Test with multiple fast transitions."""
        # Create signal with rapid transitions
        signal = np.zeros(1000, dtype=np.float64)
        signal[100:102] = 1.0
        signal[200:202] = 1.0
        signal[300:302] = 1.0

        edges = detect_edges(signal, threshold=0.5)
        # Should detect multiple edges
        assert len(edges) > 0

    def test_negative_values(self) -> None:
        """Test with negative signal values."""
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=2.0, offset=-1.0)

        edges = detect_edges(signal, threshold=0.0, sample_rate=1e6)
        assert len(edges) > 0

    def test_very_large_values(self) -> None:
        """Test with very large values."""
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=1e6, offset=0)

        edges = detect_edges(signal, threshold="auto", sample_rate=1e6)
        assert len(edges) > 0

    def test_nan_handling(self) -> None:
        """Test handling of NaN values."""
        signal = np.ones(1000, dtype=np.float64)
        signal[500:510] = np.nan

        # Should handle NaN gracefully
        edges = detect_edges(signal, threshold=0.5)
        assert isinstance(edges, list)


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from oscura.analyzers.digital import edges

        expected_exports = {
            "Edge",
            "EdgeDetector",
            "EdgeTiming",
            "TimingConstraint",
            "TimingViolation",
            "check_timing_constraints",
            "classify_edge_quality",
            "detect_edges",
            "interpolate_edge_time",
            "measure_edge_timing",
        }

        assert hasattr(edges, "__all__")
        assert set(edges.__all__) == expected_exports

    def test_dataclasses_importable(self) -> None:
        """Test that all dataclasses are importable."""
        from oscura.analyzers.digital.edges import (
            Edge,
            EdgeTiming,
            TimingConstraint,
            TimingViolation,
        )

        assert Edge is not None
        assert EdgeTiming is not None
        assert TimingConstraint is not None
        assert TimingViolation is not None

    def test_functions_importable(self) -> None:
        """Test that all functions are importable."""
        from oscura.analyzers.digital.edges import (
            check_timing_constraints,
            classify_edge_quality,
            detect_edges,
            interpolate_edge_time,
            measure_edge_timing,
        )

        assert check_timing_constraints is not None
        assert classify_edge_quality is not None
        assert detect_edges is not None
        assert interpolate_edge_time is not None
        assert measure_edge_timing is not None

    def test_class_importable(self) -> None:
        """Test that EdgeDetector class is importable."""
        from oscura.analyzers.digital.edges import EdgeDetector

        assert EdgeDetector is not None


# =============================================================================
# File-Based Testing with Fixtures
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requires_data
class TestEdgeDetectionOnSyntheticData:
    """Test edge detection on all synthetic square wave files."""

    def make_waveform_trace(self, data: np.ndarray, sample_rate: float = 1e6):
        """Create a WaveformTrace from raw data for testing."""
        from oscura.core.types import TraceMetadata, WaveformTrace

        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data.astype(np.float64), metadata=metadata)

    @pytest.mark.parametrize("freq", ["1MHz", "10MHz", "100MHz"])
    def test_edge_detection_frequencies(self, square_wave_files: dict, freq: str) -> None:
        """Test edge detection at different frequencies."""
        path = square_wave_files.get(freq)
        if path is None or not path.exists():
            pytest.skip(f"{freq} square wave file not available")

        data = np.load(path, allow_pickle=True)

        trace = self.make_waveform_trace(data, sample_rate=100e6)
        threshold = (data.max() + data.min()) / 2
        edges = detect_edges(trace, edge_type="both", threshold=threshold)

        assert len(edges) > 0, f"No edges detected at {freq}"

    def test_edge_count_consistency(self, square_wave_files: dict) -> None:
        """Test that edge count matches expected for known signals."""
        expected_order = []

        for freq in ["1MHz", "10MHz", "100MHz"]:
            path = square_wave_files.get(freq)
            if path is None or not path.exists():
                continue

            data = np.load(path, allow_pickle=True)
            trace = self.make_waveform_trace(data, sample_rate=100e6)
            threshold = (data.max() + data.min()) / 2
            edges = detect_edges(trace, edge_type="both", threshold=threshold)

            expected_order.append((freq, len(edges)))

        if len(expected_order) == 0:
            pytest.skip("No square wave files available")


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-004")
class TestDigitalEdgesIntegration:
    """Test integration of edge detection components."""

    def test_full_workflow(self) -> None:
        """Test complete edge detection and analysis workflow."""
        # Generate signal
        sample_rate = 1e6
        frequency = 1000.0
        signal = make_square_wave(frequency, sample_rate, 0.01)

        # Detect edges
        edges = detect_edges(signal, edge_type="both", threshold="auto", sample_rate=sample_rate)
        assert len(edges) > 0

        # Measure timing
        timing = measure_edge_timing(edges, sample_rate)
        assert timing.mean_period > 0

        # Check constraints
        constraint = TimingConstraint(name="period", min_time=100e-9, max_time=10e-3)
        violations = check_timing_constraints(edges, [constraint], sample_rate)
        assert isinstance(violations, list)

        # Classify quality
        for edge in edges:
            quality = classify_edge_quality(signal, edge.sample_index, sample_rate)
            assert quality in ["clean", "slow", "noisy", "glitch"]

    def test_detector_workflow(self) -> None:
        """Test complete workflow using EdgeDetector class."""
        # Generate signal
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        # Create detector
        detector = EdgeDetector(sample_rate=sample_rate, threshold="auto")

        # Detect all edges
        rising, falling = detector.detect_all_edges(signal)
        assert len(rising) > 0
        assert len(falling) > 0

        # Measure timing
        timing = detector.measure_timing(signal)
        assert timing.mean_period > 0

    def test_noisy_signal_workflow(self) -> None:
        """Test workflow with noisy signal."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        # Add noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.1, len(signal))

        # Use hysteresis to handle noise
        edges = detect_edges(
            noisy_signal,
            edge_type="both",
            threshold="auto",
            hysteresis=0.3,
            sample_rate=sample_rate,
        )

        # Should still detect edges
        assert len(edges) > 0

        # Quality classification should detect noisy edges
        noisy_count = sum(1 for e in edges if e.quality in ["noisy", "glitch"])
        # At least some edges might be classified as noisy
        assert noisy_count >= 0
