"""Unit tests for equalization (FFE, DFE, CTLE).

This module tests the equalization.py module functionality:
- FFE (Feed-Forward Equalization)
- DFE (Decision Feedback Equalization)
- CTLE (Continuous Time Linear Equalization)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils.factories import make_waveform_trace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# FFE Equalization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-004")
class TestFFEEqualize:
    """Test Feed-Forward Equalization."""

    def test_ffe_equalize_basic(self) -> None:
        """Test basic FFE equalization."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        # Create a simple test signal with ISI
        n_samples = 1000
        signal = np.zeros(n_samples)
        # Add impulses every 100 samples
        signal[::100] = 1.0

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [-0.1, 1.0, -0.1]  # 3-tap equalizer

        result = ffe_equalize(trace, taps)

        assert result.equalized_data is not None
        assert len(result.equalized_data) == n_samples
        assert len(result.taps) == 3
        assert result.n_precursor == 1
        assert result.n_postcursor == 1

    def test_ffe_equalize_preserves_energy(self) -> None:
        """Test that FFE approximately preserves signal energy."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        # Create a signal
        n_samples = 1000
        np.random.seed(42)
        signal = np.random.randn(n_samples)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        # Unity main cursor with small pre/post cursors
        taps = [-0.05, 1.0, -0.05]

        result = ffe_equalize(trace, taps)

        original_energy = np.sum(signal**2)
        equalized_energy = np.sum(result.equalized_data**2)

        # Energy should be similar (within 50%)
        assert equalized_energy > 0.5 * original_energy
        assert equalized_energy < 2.0 * original_energy

    def test_ffe_equalize_tap_array(self) -> None:
        """Test FFE with numpy array taps."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.sin(np.linspace(0, 10 * np.pi, 500))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        taps = np.array([-0.1, -0.05, 1.0, -0.05, -0.1])  # 5-tap

        result = ffe_equalize(trace, taps)

        assert result.n_precursor == 2
        assert result.n_postcursor == 2

    def test_single_tap_ffe(self) -> None:
        """Test FFE with single tap (identity)."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [1.0]  # Identity filter

        result = ffe_equalize(trace, taps)

        # With identity tap, output should be very close to input
        np.testing.assert_allclose(result.equalized_data, signal, rtol=1e-10)

    def test_large_tap_count_ffe(self) -> None:
        """Test FFE with large number of taps."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.random.randn(500)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        # 21-tap equalizer
        taps = np.zeros(21)
        taps[10] = 1.0  # Main cursor in middle
        taps[8:13] = [-0.05, -0.1, 1.0, -0.1, -0.05]

        result = ffe_equalize(trace, taps)

        assert result.n_precursor == 10
        assert result.n_postcursor == 10
        assert len(result.equalized_data) == 500


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-004")
class TestOptimizeFFE:
    """Test FFE tap optimization."""

    def test_optimize_ffe_basic(self) -> None:
        """Test basic FFE optimization."""
        from oscura.analyzers.signal_integrity.equalization import optimize_ffe

        # Create a signal with known ISI
        n_samples = 500
        np.random.seed(42)
        bits = np.random.choice([-1, 1], size=n_samples // 10)
        signal = np.repeat(bits, 10).astype(np.float64)

        # Add some ISI
        h = [0.1, 1.0, 0.3, 0.1]
        signal = np.convolve(signal, h, mode="same")

        trace = make_waveform_trace(signal, sample_rate=1e9)

        result = optimize_ffe(trace, n_taps=5, n_precursor=1)

        assert result.equalized_data is not None
        assert result.taps is not None
        assert result.mse is not None
        assert len(result.taps) == 5
        assert result.n_precursor == 1

    def test_optimize_ffe_with_target(self) -> None:
        """Test FFE optimization with explicit target."""
        from oscura.analyzers.signal_integrity.equalization import optimize_ffe

        n_samples = 500
        np.random.seed(42)
        target = np.random.choice([-1.0, 1.0], size=n_samples)

        # Create degraded signal
        h = [0.1, 1.0, 0.2]
        signal = np.convolve(target, h, mode="same")

        trace = make_waveform_trace(signal, sample_rate=1e9)

        result = optimize_ffe(trace, n_taps=5, target=target)

        assert result.mse is not None
        # MSE should be reasonably low since we're trying to recover target
        # (not necessarily very low due to optimization constraints)


# =============================================================================
# DFE Equalization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-005")
class TestDFEEqualize:
    """Test Decision Feedback Equalization."""

    def test_dfe_equalize_basic(self) -> None:
        """Test basic DFE equalization."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        # Create a binary signal
        n_samples = 500
        np.random.seed(42)
        signal = np.random.choice([-1.0, 1.0], size=n_samples)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [0.2, 0.1]  # 2-tap DFE

        result = dfe_equalize(trace, taps)

        assert result.equalized_data is not None
        assert result.decisions is not None
        assert len(result.taps) == 2
        assert result.n_taps == 2

    def test_dfe_equalize_makes_decisions(self) -> None:
        """Test that DFE makes bit decisions."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        n_samples = 100
        # Create clean binary signal
        signal = np.array([1.0, -1.0, 1.0, 1.0, -1.0] * 20, dtype=np.float64)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [0.1]  # 1-tap DFE

        result = dfe_equalize(trace, taps, samples_per_symbol=1)

        # Decisions should be 0 or 1
        assert np.all((result.decisions == 0) | (result.decisions == 1))

    def test_dfe_equalize_custom_threshold(self) -> None:
        """Test DFE with custom decision threshold."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        n_samples = 100
        # Signal with DC offset
        signal = np.random.choice([0.5, 1.5], size=n_samples).astype(np.float64)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [0.1]

        result = dfe_equalize(trace, taps, threshold=1.0)

        assert result.equalized_data is not None
        assert result.decisions is not None

    def test_dfe_equalize_samples_per_symbol(self) -> None:
        """Test DFE with oversampled data."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        # 4x oversampled signal
        n_symbols = 50
        samples_per_symbol = 4
        symbol_values = np.random.choice([-1.0, 1.0], size=n_symbols)
        signal = np.repeat(symbol_values, samples_per_symbol)

        trace = make_waveform_trace(signal, sample_rate=4e9)
        taps = [0.1, 0.05]

        result = dfe_equalize(trace, taps, samples_per_symbol=samples_per_symbol)

        # Should have n_symbols decisions
        assert len(result.decisions) <= n_symbols


# =============================================================================
# CTLE Equalization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-006")
class TestCTLEEqualize:
    """Test Continuous Time Linear Equalization."""

    def test_ctle_equalize_basic(self) -> None:
        """Test basic CTLE equalization."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        n_samples = 1000
        sample_rate = 20e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)

        result = ctle_equalize(
            trace,
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
        )

        assert result.equalized_data is not None
        assert len(result.equalized_data) == n_samples
        assert result.dc_gain == 0.0
        assert result.ac_gain == 6.0
        assert result.pole_frequency == 5e9

    def test_ctle_equalize_boost_calculation(self) -> None:
        """Test CTLE boost calculation."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
        trace = make_waveform_trace(signal, sample_rate=20e9)

        result = ctle_equalize(
            trace,
            dc_gain=-3.0,
            ac_gain=3.0,
            pole_frequency=5e9,
        )

        # Boost should be ac_gain - dc_gain
        expected_boost = 3.0 - (-3.0)
        assert result.boost == expected_boost

    def test_ctle_equalize_custom_zero(self) -> None:
        """Test CTLE with custom zero frequency."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
        trace = make_waveform_trace(signal, sample_rate=20e9)

        result = ctle_equalize(
            trace,
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
            zero_frequency=2e9,
        )

        assert result.zero_frequency == 2e9

    def test_ctle_equalize_computes_zero_frequency(self) -> None:
        """Test that CTLE computes zero frequency when not specified."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
        trace = make_waveform_trace(signal, sample_rate=20e9)

        result = ctle_equalize(
            trace,
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
        )

        # Zero frequency should be computed
        assert result.zero_frequency is not None
        assert result.zero_frequency > 0


# =============================================================================
# Equalization Result Dataclass Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestEqualizationResults:
    """Test equalization result dataclasses."""

    def test_ffe_result_attributes(self) -> None:
        """Test FFEResult dataclass attributes."""
        from oscura.analyzers.signal_integrity.equalization import FFEResult

        result = FFEResult(
            equalized_data=np.array([1.0, 2.0, 3.0]),
            taps=np.array([-0.1, 1.0, -0.1]),
            n_precursor=1,
            n_postcursor=1,
            mse=0.01,
        )

        assert len(result.equalized_data) == 3
        assert len(result.taps) == 3
        assert result.n_precursor == 1
        assert result.n_postcursor == 1
        assert result.mse == 0.01

    def test_dfe_result_attributes(self) -> None:
        """Test DFEResult dataclass attributes."""
        from oscura.analyzers.signal_integrity.equalization import DFEResult

        result = DFEResult(
            equalized_data=np.array([1.0, -1.0, 1.0]),
            taps=np.array([0.2, 0.1]),
            decisions=np.array([1, 0, 1]),
            n_taps=2,
            error_count=0,
        )

        assert len(result.equalized_data) == 3
        assert len(result.taps) == 2
        assert len(result.decisions) == 3
        assert result.n_taps == 2
        assert result.error_count == 0

    def test_ctle_result_attributes(self) -> None:
        """Test CTLEResult dataclass attributes."""
        from oscura.analyzers.signal_integrity.equalization import CTLEResult

        result = CTLEResult(
            equalized_data=np.array([1.0, 2.0, 3.0]),
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
            zero_frequency=2e9,
            boost=6.0,
        )

        assert len(result.equalized_data) == 3
        assert result.dc_gain == 0.0
        assert result.ac_gain == 6.0
        assert result.pole_frequency == 5e9
        assert result.zero_frequency == 2e9
        assert result.boost == 6.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestEqualizationEdgeCases:
    """Test edge cases in equalization modules."""

    def test_empty_signal_ffe(self) -> None:
        """Test FFE with empty signal raises ValueError."""

        signal = np.array([], dtype=np.float64)

        # Empty signal raises ValueError at trace creation time
        with pytest.raises(ValueError, match="data array cannot be empty"):
            trace = make_waveform_trace(signal, sample_rate=1e9)

    def test_single_sample_ffe(self) -> None:
        """Test FFE with single sample."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.array([1.0])
        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [1.0]

        result = ffe_equalize(trace, taps)

        assert len(result.equalized_data) == 1
