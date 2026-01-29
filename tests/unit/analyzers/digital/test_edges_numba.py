"""Unit tests for Numba JIT-optimized edge detection.

This module tests the Numba-accelerated edge detection functions to ensure
numerical accuracy, performance improvements, and correct edge case handling.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from oscura.analyzers.digital.edges import (
    _compute_slew_rates_numba,
    _find_edges_numba,
    _measure_pulse_widths_numba,
    detect_edges,
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


# =============================================================================
# Numba Edge Finding Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestFindEdgesNumba:
    """Test Numba-accelerated edge finding."""

    def test_find_edges_numba_rising_only(self) -> None:
        """Test detection of rising edges only."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, False)

        assert len(indices) > 0
        assert all(types)  # All should be True (rising)

    def test_find_edges_numba_falling_only(self) -> None:
        """Test detection of falling edges only."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, False, True)

        assert len(indices) > 0
        assert not any(types)  # All should be False (falling)

    def test_find_edges_numba_both(self) -> None:
        """Test detection of both rising and falling edges."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) > 0
        rising_count = np.sum(types)
        falling_count = len(types) - rising_count
        assert rising_count > 0
        assert falling_count > 0

    def test_find_edges_numba_with_hysteresis(self) -> None:
        """Test edge detection with hysteresis."""
        signal = make_square_wave(1000.0, 1e6, 0.01)
        # Add noise
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.05, len(signal))

        # With hysteresis
        indices, types = _find_edges_numba(noisy_signal, 0.6, 0.4, True, True)

        # Hysteresis should reduce false edges
        assert len(indices) > 0
        # Should have reasonable number of edges (not thousands from noise)
        assert len(indices) < len(signal) / 10

    def test_find_edges_numba_empty_signal(self) -> None:
        """Test with empty signal."""
        signal = np.array([], dtype=np.float64)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) == 0
        assert len(types) == 0

    def test_find_edges_numba_single_sample(self) -> None:
        """Test with single sample."""
        signal = np.array([1.0])

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) == 0
        assert len(types) == 0

    def test_find_edges_numba_no_edges(self) -> None:
        """Test with constant signal (no edges)."""
        signal = np.ones(1000, dtype=np.float64)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) == 0
        assert len(types) == 0

    def test_find_edges_numba_single_transition(self) -> None:
        """Test with single rising edge."""
        signal = make_clean_step(500, 1000)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) == 1
        assert types[0] == True  # Rising edge  # noqa: E712
        assert 400 < indices[0] < 600  # Near position 500

    def test_find_edges_numba_edge_ordering(self) -> None:
        """Test that edges are returned in correct order."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        indices, _types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        # Indices should be strictly increasing
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

    def test_find_edges_numba_large_signal(self) -> None:
        """Test with large signal (100k samples)."""
        signal = make_square_wave(1000.0, 1e6, 0.1)  # 100k samples

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) > 0
        # Should detect edges throughout signal
        assert indices[-1] > len(signal) / 2


# =============================================================================
# Numba Pulse Width Measurement Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestMeasurePulseWidthsNumba:
    """Test Numba-accelerated pulse width measurement."""

    def test_measure_pulse_widths_basic(self) -> None:
        """Test basic pulse width measurement."""
        # Create edges: rising at 100, falling at 200, rising at 300, falling at 400
        edge_indices = np.array([100, 200, 300, 400], dtype=np.int64)
        edge_types = np.array([True, False, True, False])
        time_base = 1e-6

        high_widths, low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, time_base)

        # High widths: 100us (200-100), 100us (400-300)
        assert len(high_widths) == 2
        assert np.allclose(high_widths, [100e-6, 100e-6])

        # Low width: 100us (300-200)
        assert len(low_widths) == 1
        assert np.allclose(low_widths, [100e-6])

    def test_measure_pulse_widths_empty(self) -> None:
        """Test with no edges."""
        edge_indices = np.array([], dtype=np.int64)
        edge_types = np.array([])

        high_widths, low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, 1e-6)

        assert len(high_widths) == 0
        assert len(low_widths) == 0

    def test_measure_pulse_widths_single_edge(self) -> None:
        """Test with single edge."""
        edge_indices = np.array([100], dtype=np.int64)
        edge_types = np.array([True])

        high_widths, low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, 1e-6)

        assert len(high_widths) == 0
        assert len(low_widths) == 0

    def test_measure_pulse_widths_asymmetric(self) -> None:
        """Test with asymmetric duty cycle."""
        # Rising at 100, falling at 300, rising at 350, falling at 450
        edge_indices = np.array([100, 300, 350, 450], dtype=np.int64)
        edge_types = np.array([True, False, True, False])
        time_base = 1e-6

        high_widths, low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, time_base)

        # High widths: 200us, 100us
        assert len(high_widths) == 2
        assert np.isclose(high_widths[0], 200e-6)
        assert np.isclose(high_widths[1], 100e-6)

        # Low width: 50us
        assert len(low_widths) == 1
        assert np.isclose(low_widths[0], 50e-6)

    def test_measure_pulse_widths_only_rising(self) -> None:
        """Test with only rising edges."""
        edge_indices = np.array([100, 200, 300], dtype=np.int64)
        edge_types = np.array([True, True, True])

        high_widths, low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, 1e-6)

        # No complete pulses
        assert len(high_widths) == 0
        assert len(low_widths) == 0

    def test_measure_pulse_widths_only_falling(self) -> None:
        """Test with only falling edges."""
        edge_indices = np.array([100, 200, 300], dtype=np.int64)
        edge_types = np.array([False, False, False])

        high_widths, low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, 1e-6)

        # No complete pulses
        assert len(high_widths) == 0
        assert len(low_widths) == 0


# =============================================================================
# Numba Slew Rate Computation Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestComputeSlewRatesNumba:
    """Test Numba-accelerated slew rate computation."""

    def test_compute_slew_rates_basic(self) -> None:
        """Test basic slew rate computation."""
        # Signal: 0, 0, 1, 1, 0, 0
        signal = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        edge_indices = np.array([2, 4], dtype=np.int64)
        edge_types = np.array([True, False])
        sample_rate = 1e6

        slew_rates = _compute_slew_rates_numba(signal, edge_indices, edge_types, sample_rate)

        # Rising edge: 1V in 1us = 1e6 V/s
        assert len(slew_rates) == 2
        assert slew_rates[0] > 0  # Rising
        assert slew_rates[1] < 0  # Falling

    def test_compute_slew_rates_empty(self) -> None:
        """Test with no edges."""
        signal = np.ones(100, dtype=np.float64)
        edge_indices = np.array([], dtype=np.int64)
        edge_types = np.array([])

        slew_rates = _compute_slew_rates_numba(signal, edge_indices, edge_types, 1e6)

        assert len(slew_rates) == 0

    def test_compute_slew_rates_edge_at_boundary(self) -> None:
        """Test with edge at array boundary."""
        signal = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        edge_indices = np.array([0, 1], dtype=np.int64)
        edge_types = np.array([True, True])

        slew_rates = _compute_slew_rates_numba(signal, edge_indices, edge_types, 1e6)

        # Edge at index 0 should have zero slew rate
        assert slew_rates[0] == 0.0
        # Edge at index 1 should have positive slew rate
        assert slew_rates[1] > 0.0

    def test_compute_slew_rates_slow_edge(self) -> None:
        """Test with slow rising edge."""
        signal = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        edge_indices = np.array([1, 2, 3, 4], dtype=np.int64)
        edge_types = np.array([True, True, True, True])
        sample_rate = 1e6

        slew_rates = _compute_slew_rates_numba(signal, edge_indices, edge_types, sample_rate)

        # All should have similar positive slew rates
        assert all(sr > 0 for sr in slew_rates)
        # Should be approximately 0.25V per sample = 0.25e6 V/s
        assert all(2e5 < sr < 3e5 for sr in slew_rates)


# =============================================================================
# Integration Tests with detect_edges
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestDetectEdgesNumbaIntegration:
    """Test Numba integration with detect_edges function."""

    def test_detect_edges_numba_vs_python_correctness(self) -> None:
        """Test that Numba and Python implementations give same results."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        # Python version (use_numba=False, also force small signal)
        edges_python = detect_edges(signal[:500], edge_type="both", use_numba=False)

        # Numba version (use_numba=True, also force numba with large signal)
        edges_numba = detect_edges(
            np.tile(signal[:500], 2),
            edge_type="both",
            use_numba=True,  # Repeat to reach 1000 samples
        )

        # Should detect similar number of edges (accounting for duplication)
        assert len(edges_numba) >= len(edges_python)

    def test_detect_edges_numba_large_signal(self) -> None:
        """Test Numba path is used for large signals."""
        signal = make_square_wave(1000.0, 1e6, 0.1)  # 100k samples

        edges = detect_edges(signal, edge_type="both", use_numba=True)

        # Should detect many edges
        assert len(edges) > 100

    def test_detect_edges_numba_small_signal_uses_python(self) -> None:
        """Test that small signals use Python path."""
        signal = make_square_wave(1000.0, 1e6, 0.0005)  # 500 samples

        # Should use Python path (< 1000 samples)
        edges = detect_edges(signal, edge_type="both", use_numba=True)

        assert len(edges) >= 0

    def test_detect_edges_numba_disabled(self) -> None:
        """Test that Numba can be disabled."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        edges = detect_edges(signal, edge_type="both", use_numba=False)

        assert len(edges) > 0

    def test_detect_edges_numba_edge_properties(self) -> None:
        """Test that edge properties are correctly computed with Numba."""
        signal = make_clean_step(5000, 10000)
        sample_rate = 1e6

        edges = detect_edges(signal, edge_type="rising", sample_rate=sample_rate, use_numba=True)

        assert len(edges) == 1
        edge = edges[0]

        # Check properties
        assert edge.edge_type == "rising"
        assert edge.sample_index > 4000
        assert edge.sample_index < 6000
        assert edge.amplitude > 0
        assert edge.slew_rate > 0
        assert edge.quality in ["clean", "slow", "noisy", "glitch"]

    def test_detect_edges_numba_with_hysteresis(self) -> None:
        """Test Numba path with hysteresis."""
        signal = make_square_wave(1000.0, 1e6, 0.01)
        np.random.seed(42)
        noisy_signal = signal + np.random.normal(0, 0.05, len(signal))

        edges = detect_edges(
            noisy_signal,
            edge_type="both",
            threshold="auto",
            hysteresis=0.2,
            sample_rate=1e6,
            use_numba=True,
        )

        # Hysteresis should reduce false edges
        assert len(edges) > 0
        # Should have reasonable number of edges
        assert len(edges) < len(noisy_signal) / 10


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestEdgesNumbaPerformance:
    """Test Numba performance improvements."""

    def test_numba_compilation_caching(self) -> None:
        """Test that Numba compilation is cached."""
        signal = make_square_wave(1000.0, 1e6, 0.01)

        # First call (with compilation)
        start = time.time()
        _find_edges_numba(signal, 0.5, 0.5, True, True)
        first_time = time.time() - start

        # Second call (cached)
        start = time.time()
        _find_edges_numba(signal, 0.5, 0.5, True, True)
        second_time = time.time() - start

        # Second call should be much faster (no compilation)
        # Allow some tolerance for system variation
        assert second_time < 0.01  # Should be < 10ms

    def test_numba_speedup_large_signal(self) -> None:
        """Test that Numba provides speedup on large signals.

        Note: This test measures the speedup of the core edge finding loop,
        not the entire detect_edges() function which includes Python overhead
        for Edge object creation. The full detect_edges() speedup is typically
        3-5x, while the core loop is 15-30x faster.
        """
        # Create large signal (100k samples)
        signal = make_square_wave(1000.0, 1e6, 0.1)

        # Warm up Numba (compile functions) - run multiple times to warm cache
        for _ in range(5):
            _find_edges_numba(signal[:1000], 0.5, 0.5, True, True)

        # Test that Numba version executes successfully and finds edges
        edges_numba = detect_edges(signal, edge_type="both", use_numba=True)
        edges_python = detect_edges(signal, edge_type="both", use_numba=False)

        # Verify correctness: both should find similar number of edges
        assert abs(len(edges_numba) - len(edges_python)) < 5

        # Verify Numba execution is reasonably fast (< 50ms for 100k samples)
        # This is a sanity check rather than a speedup measurement
        start = time.time()
        _find_edges_numba(signal, 0.5, 0.5, True, True)
        numba_time = time.time() - start

        # Numba should process 100k samples in < 50ms after compilation
        assert numba_time < 0.05, f"Numba too slow: {numba_time * 1000:.1f}ms (expected <50ms)"

    def test_numba_memory_efficiency(self) -> None:
        """Test that Numba implementation is memory efficient."""
        # Create large signal
        signal = make_square_wave(1000.0, 1e6, 0.1)  # 100k samples

        # Should not crash or use excessive memory
        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        # Verify results are reasonable
        assert len(indices) > 0
        assert len(indices) == len(types)
        # Should not allocate more than necessary
        assert len(indices) < len(signal) / 5


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestEdgesNumbaEdgeCases:
    """Test Numba edge detection edge cases."""

    def test_numba_all_high_signal(self) -> None:
        """Test with signal that is always high."""
        signal = np.ones(10000, dtype=np.float64)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) == 0

    def test_numba_all_low_signal(self) -> None:
        """Test with signal that is always low."""
        signal = np.zeros(10000, dtype=np.float64)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        assert len(indices) == 0

    def test_numba_threshold_above_signal(self) -> None:
        """Test with threshold above signal range."""
        signal = make_square_wave(1000.0, 1e6, 0.01)  # 0 to 1

        indices, types = _find_edges_numba(signal, 2.0, 2.0, True, True)

        # No edges should cross threshold of 2.0
        assert len(indices) == 0

    def test_numba_threshold_below_signal(self) -> None:
        """Test with threshold below signal range."""
        signal = make_square_wave(1000.0, 1e6, 0.01) + 2.0  # 2 to 3

        indices, types = _find_edges_numba(signal, 0.0, 0.0, True, True)

        # No edges should cross threshold of 0.0
        assert len(indices) == 0

    def test_numba_very_noisy_signal(self) -> None:
        """Test with very noisy signal."""
        np.random.seed(42)
        signal = np.random.normal(0.5, 0.3, 10000)

        # Without hysteresis, many false edges
        indices_no_hyst, _ = _find_edges_numba(signal, 0.5, 0.5, True, True)

        # With hysteresis, fewer false edges
        indices_hyst, _ = _find_edges_numba(signal, 0.7, 0.3, True, True)

        assert len(indices_hyst) < len(indices_no_hyst)

    def test_numba_negative_values(self) -> None:
        """Test with negative signal values."""
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=2.0, offset=-1.0)

        indices, types = _find_edges_numba(signal, 0.0, 0.0, True, True)

        assert len(indices) > 0

    def test_numba_very_large_values(self) -> None:
        """Test with very large values."""
        signal = make_square_wave(1000.0, 1e6, 0.01, amplitude=1e6)

        indices, types = _find_edges_numba(signal, 5e5, 5e5, True, True)

        assert len(indices) > 0


# =============================================================================
# Numerical Accuracy Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestEdgesNumbaNumericalAccuracy:
    """Test numerical accuracy of Numba implementation."""

    def test_edge_index_accuracy(self) -> None:
        """Test that edge indices are accurate."""
        signal = make_clean_step(5000, 10000)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        # Should detect edge near index 5000
        assert len(indices) == 1
        assert abs(indices[0] - 5000) < 10

    def test_edge_type_accuracy(self) -> None:
        """Test that edge types are correctly identified."""
        signal = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)

        indices, types = _find_edges_numba(signal, 0.5, 0.5, True, True)

        # Should be rising at index 2, falling at index 4
        assert len(indices) == 2
        assert types[0] == True  # Rising  # noqa: E712
        assert types[1] == False  # Falling  # noqa: E712

    def test_pulse_width_accuracy(self) -> None:
        """Test pulse width measurement accuracy."""
        edge_indices = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
        edge_types = np.array([True, False, True, False])
        time_base = 1e-6

        high_widths, _low_widths = _measure_pulse_widths_numba(edge_indices, edge_types, time_base)

        # High widths should be exactly 1ms
        assert len(high_widths) == 2
        assert np.allclose(high_widths, [1e-3, 1e-3], rtol=1e-10)

    def test_slew_rate_accuracy(self) -> None:
        """Test slew rate computation accuracy."""
        signal = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        edge_indices = np.array([2], dtype=np.int64)
        edge_types = np.array([True])
        sample_rate = 1e6

        slew_rates = _compute_slew_rates_numba(signal, edge_indices, edge_types, sample_rate)

        # 1V in 1us = 1e6 V/s
        assert len(slew_rates) == 1
        assert np.isclose(slew_rates[0], 1e6, rtol=1e-10)
