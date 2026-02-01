"""Unit tests for ripple analysis.


Tests cover all public functions including:
- ripple() - AC ripple measurement (peak-to-peak and RMS)
- ripple_percentage() - Ripple as percentage of DC level
- ripple_frequency() - Dominant ripple frequency detection
- ripple_harmonics() - Harmonic analysis
- ripple_statistics() - Comprehensive ripple statistics
- extract_ripple() - AC component extraction
- ripple_envelope() - Envelope detection

Edge cases, error conditions, and boundary conditions are thoroughly tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.power.ripple import (
    extract_ripple,
    ripple,
    ripple_envelope,
    ripple_frequency,
    ripple_harmonics,
    ripple_percentage,
    ripple_statistics,
)
from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.power]


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 100000.0  # 100 kHz for good frequency resolution


def create_trace(
    data: np.ndarray,
    sample_rate: float,
) -> WaveformTrace:
    """Create a waveform trace from data array."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def create_dc_with_ripple(
    dc_level: float,
    ripple_amplitude: float,
    ripple_freq: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a DC voltage with sinusoidal AC ripple."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = dc_level + ripple_amplitude * np.sin(2 * np.pi * ripple_freq * t)
    return create_trace(data, sample_rate)


def create_dc_only(
    dc_level: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a pure DC signal with no ripple."""
    num_samples = int(sample_rate * duration)
    data = np.full(num_samples, dc_level)
    return create_trace(data, sample_rate)


def create_square_wave_ripple(
    dc_level: float,
    ripple_amplitude: float,
    ripple_freq: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create DC with square wave ripple."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    square = ripple_amplitude * np.sign(np.sin(2 * np.pi * ripple_freq * t))
    data = dc_level + square
    return create_trace(data, sample_rate)


def create_multi_freq_ripple(
    dc_level: float,
    fundamental_amp: float,
    fundamental_freq: float,
    sample_rate: float,
    duration: float = 0.1,
    harmonics: dict[int, float] | None = None,
) -> WaveformTrace:
    """Create DC with multiple frequency components.

    Args:
        dc_level: DC offset
        fundamental_amp: Amplitude of fundamental frequency
        fundamental_freq: Fundamental frequency
        sample_rate: Sample rate
        duration: Signal duration
        harmonics: Dict mapping harmonic number to relative amplitude (e.g., {2: 0.3, 3: 0.1})
    """
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Fundamental
    signal = fundamental_amp * np.sin(2 * np.pi * fundamental_freq * t)

    # Add harmonics
    if harmonics:
        for h, rel_amp in harmonics.items():
            signal += (fundamental_amp * rel_amp) * np.sin(2 * np.pi * fundamental_freq * h * t)

    data = dc_level + signal
    return create_trace(data, sample_rate)


@pytest.mark.unit
@pytest.mark.power
class TestRipple:
    """Test basic ripple measurement (peak-to-peak and RMS)."""

    def test_ripple_pure_dc(self, sample_rate: float) -> None:
        """Test ripple measurement on pure DC (no ripple)."""
        trace = create_dc_only(12.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        # No ripple should give near-zero values
        assert abs(r_pp) < 1e-10
        assert abs(r_rms) < 1e-10

    def test_ripple_with_sinusoidal_ac(self, sample_rate: float) -> None:
        """Test ripple measurement with sinusoidal AC component."""
        dc_level = 12.0
        ripple_amp = 0.1  # 100 mV amplitude
        ripple_freq = 120.0  # 120 Hz switching ripple

        trace = create_dc_with_ripple(dc_level, ripple_amp, ripple_freq, sample_rate)

        r_pp, r_rms = ripple(trace)

        # Peak-to-peak should be approximately 2 * amplitude
        expected_pp = 2 * ripple_amp
        assert abs(r_pp - expected_pp) < 0.01

        # RMS of sine wave is amplitude / sqrt(2)
        expected_rms = ripple_amp / np.sqrt(2)
        assert abs(r_rms - expected_rms) < 0.01

    def test_ripple_dc_coupling_off(self, sample_rate: float) -> None:
        """Test ripple with DC coupling disabled (default)."""
        trace = create_dc_with_ripple(12.0, 0.05, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace, dc_coupling=False)

        # Should measure only AC component
        expected_pp = 2 * 0.05
        assert abs(r_pp - expected_pp) < 0.005

    def test_ripple_dc_coupling_on(self, sample_rate: float) -> None:
        """Test ripple with DC coupling enabled."""
        dc_level = 12.0
        ripple_amp = 0.05
        trace = create_dc_with_ripple(dc_level, ripple_amp, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace, dc_coupling=True)

        # With DC coupling, RMS includes DC component
        # RMS = sqrt(DC^2 + AC_rms^2)
        ac_rms = ripple_amp / np.sqrt(2)
        expected_rms = np.sqrt(dc_level**2 + ac_rms**2)
        assert abs(r_rms - expected_rms) < 0.1

    def test_ripple_square_wave(self, sample_rate: float) -> None:
        """Test ripple measurement with square wave ripple."""
        dc_level = 5.0
        ripple_amp = 0.1

        trace = create_square_wave_ripple(dc_level, ripple_amp, 100.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        # Square wave: peak-to-peak is 2 * amplitude
        expected_pp = 2 * ripple_amp
        assert abs(r_pp - expected_pp) < 0.02

        # RMS of square wave is equal to amplitude
        expected_rms = ripple_amp
        assert abs(r_rms - expected_rms) < 0.02

    def test_ripple_multiple_frequencies(self, sample_rate: float) -> None:
        """Test ripple with multiple frequency components."""
        trace = create_multi_freq_ripple(
            dc_level=12.0,
            fundamental_amp=0.1,
            fundamental_freq=120.0,
            sample_rate=sample_rate,
            harmonics={2: 0.3, 3: 0.1},  # 2nd and 3rd harmonics
        )

        r_pp, r_rms = ripple(trace)

        # Should measure total ripple including all harmonics
        assert r_pp > 0.2  # More than just fundamental
        assert r_rms > 0

    def test_ripple_return_types(self, sample_rate: float) -> None:
        """Test that ripple returns float values."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        assert isinstance(r_pp, float)
        assert isinstance(r_rms, float)

    def test_ripple_zero_dc_level(self, sample_rate: float) -> None:
        """Test ripple measurement with zero DC level."""
        # AC signal centered at 0V
        trace = create_dc_with_ripple(0.0, 0.1, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        # Should still measure ripple correctly
        expected_pp = 2 * 0.1
        assert abs(r_pp - expected_pp) < 0.01

    def test_ripple_negative_dc_level(self, sample_rate: float) -> None:
        """Test ripple measurement with negative DC level."""
        trace = create_dc_with_ripple(-12.0, 0.05, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        # Ripple measurement should be independent of DC level
        expected_pp = 2 * 0.05
        assert abs(r_pp - expected_pp) < 0.005

    def test_ripple_very_small_ripple(self, sample_rate: float) -> None:
        """Test ripple measurement with very small ripple."""
        # 1 mV ripple on 12V supply
        trace = create_dc_with_ripple(12.0, 0.001, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        assert r_pp < 0.01
        assert r_rms < 0.01

    def test_ripple_large_ripple(self, sample_rate: float) -> None:
        """Test ripple measurement with large ripple."""
        # 1V ripple on 12V supply (poor regulation)
        trace = create_dc_with_ripple(12.0, 1.0, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)

        assert r_pp > 1.5
        assert r_rms > 0.5


@pytest.mark.unit
@pytest.mark.power
class TestRipplePercentage:
    """Test ripple percentage calculation."""

    def test_ripple_percentage_basic(self, sample_rate: float) -> None:
        """Test ripple percentage calculation."""
        dc_level = 12.0
        ripple_amp = 0.12  # 120 mV

        trace = create_dc_with_ripple(dc_level, ripple_amp, 120.0, sample_rate)

        pp_pct, rms_pct = ripple_percentage(trace)

        # Peak-to-peak: 240 mV / 12V * 100 = 2%
        expected_pp_pct = (2 * ripple_amp) / dc_level * 100
        assert abs(pp_pct - expected_pp_pct) < 0.1

        # RMS percentage
        expected_rms = (ripple_amp / np.sqrt(2)) / dc_level * 100
        assert abs(rms_pct - expected_rms) < 0.1

    def test_ripple_percentage_zero_dc(self, sample_rate: float) -> None:
        """Test ripple percentage with zero DC level."""
        trace = create_dc_with_ripple(0.0, 0.1, 120.0, sample_rate)

        pp_pct, rms_pct = ripple_percentage(trace)

        # When DC level is very close to zero, percentage is very large or NaN
        # Accept either NaN or very large values
        assert np.isnan(pp_pct) or abs(pp_pct) > 1e10
        assert np.isnan(rms_pct) or abs(rms_pct) > 1e10

    def test_ripple_percentage_negative_dc(self, sample_rate: float) -> None:
        """Test ripple percentage with negative DC level."""
        dc_level = -12.0
        ripple_amp = 0.12

        trace = create_dc_with_ripple(dc_level, ripple_amp, 120.0, sample_rate)

        pp_pct, rms_pct = ripple_percentage(trace)

        # Should still calculate percentage (negative DC is valid)
        expected_pp_pct = (2 * ripple_amp) / abs(dc_level) * 100
        assert abs(abs(pp_pct) - expected_pp_pct) < 0.1

    def test_ripple_percentage_return_types(self, sample_rate: float) -> None:
        """Test that ripple_percentage returns float values."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        pp_pct, rms_pct = ripple_percentage(trace)

        assert isinstance(pp_pct, float)
        assert isinstance(rms_pct, float)

    def test_ripple_percentage_low_ripple(self, sample_rate: float) -> None:
        """Test ripple percentage with very low ripple (good power supply)."""
        # 10 mV ripple on 12V = 0.083%
        trace = create_dc_with_ripple(12.0, 0.01, 120.0, sample_rate)

        pp_pct, rms_pct = ripple_percentage(trace)

        assert pp_pct < 0.5  # Less than 0.5%
        assert rms_pct < 0.5

    def test_ripple_percentage_high_ripple(self, sample_rate: float) -> None:
        """Test ripple percentage with high ripple (poor regulation)."""
        # 1V ripple on 12V = 8.3%
        trace = create_dc_with_ripple(12.0, 1.0, 120.0, sample_rate)

        pp_pct, rms_pct = ripple_percentage(trace)

        assert pp_pct > 10  # Greater than 10%


@pytest.mark.unit
@pytest.mark.power
class TestRippleFrequency:
    """Test dominant ripple frequency detection."""

    def test_ripple_frequency_single_tone(self, sample_rate: float) -> None:
        """Test frequency detection with single frequency ripple."""
        ripple_freq = 120.0
        trace = create_dc_with_ripple(12.0, 0.1, ripple_freq, sample_rate)

        detected_freq = ripple_frequency(trace)

        # Should detect the 120 Hz ripple
        assert abs(detected_freq - ripple_freq) < 5.0

    def test_ripple_frequency_different_frequencies(self, sample_rate: float) -> None:
        """Test frequency detection with different ripple frequencies."""
        frequencies = [60.0, 120.0, 240.0, 1000.0]

        for freq in frequencies:
            trace = create_dc_with_ripple(12.0, 0.1, freq, sample_rate)
            detected = ripple_frequency(trace)
            assert abs(detected - freq) < freq * 0.1  # Within 10%

    def test_ripple_frequency_pure_dc(self, sample_rate: float) -> None:
        """Test frequency detection with pure DC (no ripple)."""
        trace = create_dc_only(12.0, sample_rate)

        detected_freq = ripple_frequency(trace)

        # Should return 0 or very low frequency
        assert detected_freq < 10.0

    def test_ripple_frequency_with_min_max(self, sample_rate: float) -> None:
        """Test frequency detection with frequency limits."""
        # Create signal with 100 Hz and 500 Hz components
        trace = create_multi_freq_ripple(
            dc_level=12.0,
            fundamental_amp=0.1,
            fundamental_freq=100.0,
            sample_rate=sample_rate,
            harmonics={5: 0.8},  # 500 Hz component larger
        )

        # Limit search to 400-600 Hz
        detected = ripple_frequency(trace, min_frequency=400.0, max_frequency=600.0)

        # Should find the 500 Hz component
        assert abs(detected - 500.0) < 50.0

    def test_ripple_frequency_min_only(self, sample_rate: float) -> None:
        """Test frequency detection with only minimum frequency."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        detected = ripple_frequency(trace, min_frequency=50.0)

        assert abs(detected - 120.0) < 10.0

    def test_ripple_frequency_max_only(self, sample_rate: float) -> None:
        """Test frequency detection with only maximum frequency."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        detected = ripple_frequency(trace, max_frequency=200.0)

        assert abs(detected - 120.0) < 10.0

    def test_ripple_frequency_no_valid_range(self, sample_rate: float) -> None:
        """Test frequency detection when no frequencies in valid range."""
        # 120 Hz ripple, but search in 1-10 Hz range
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        detected = ripple_frequency(trace, min_frequency=1.0, max_frequency=10.0)

        # Should return 0 or very low frequency when no valid frequencies found
        assert detected < 15.0

    def test_ripple_frequency_return_type(self, sample_rate: float) -> None:
        """Test that ripple_frequency returns float."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        freq = ripple_frequency(trace)

        assert isinstance(freq, float)

    def test_ripple_frequency_harmonics(self, sample_rate: float) -> None:
        """Test frequency detection with harmonics."""
        # Fundamental at 100 Hz with strong 3rd harmonic
        trace = create_multi_freq_ripple(
            dc_level=12.0,
            fundamental_amp=0.05,
            fundamental_freq=100.0,
            sample_rate=sample_rate,
            harmonics={3: 1.5},  # 300 Hz component larger
        )

        detected = ripple_frequency(trace)

        # Should find the dominant 300 Hz component
        assert abs(detected - 300.0) < 30.0


@pytest.mark.unit
@pytest.mark.power
class TestRippleHarmonics:
    """Test ripple harmonic analysis."""

    def test_ripple_harmonics_basic(self, sample_rate: float) -> None:
        """Test basic harmonic analysis."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        harmonics = ripple_harmonics(trace, fundamental_freq=120.0, n_harmonics=5)

        # Should return dictionary with harmonic numbers
        assert isinstance(harmonics, dict)
        assert 1 in harmonics
        assert 2 in harmonics
        assert len(harmonics) == 5

    def test_ripple_harmonics_auto_detect(self, sample_rate: float) -> None:
        """Test harmonic analysis with auto-detected fundamental."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        harmonics = ripple_harmonics(trace)  # Auto-detect fundamental

        assert len(harmonics) > 0
        assert 1 in harmonics

    def test_ripple_harmonics_with_harmonics(self, sample_rate: float) -> None:
        """Test harmonic analysis with actual harmonics present."""
        trace = create_multi_freq_ripple(
            dc_level=12.0,
            fundamental_amp=0.1,
            fundamental_freq=100.0,
            sample_rate=sample_rate,
            harmonics={2: 0.5, 3: 0.3},
        )

        harmonics = ripple_harmonics(trace, fundamental_freq=100.0, n_harmonics=5)

        # Fundamental should be strongest
        assert harmonics[1] > harmonics[4]
        # 2nd harmonic should be present
        assert harmonics[2] > harmonics[5]

    def test_ripple_harmonics_zero_frequency(self, sample_rate: float) -> None:
        """Test harmonic analysis with zero fundamental frequency."""
        trace = create_dc_only(12.0, sample_rate)

        harmonics = ripple_harmonics(trace, fundamental_freq=0.0)

        # Should return empty dict
        assert len(harmonics) == 0

    def test_ripple_harmonics_pure_dc(self, sample_rate: float) -> None:
        """Test harmonic analysis on pure DC."""
        trace = create_dc_only(12.0, sample_rate)

        # Auto-detect will find no fundamental
        harmonics = ripple_harmonics(trace)

        # Should return empty dict
        assert len(harmonics) == 0

    def test_ripple_harmonics_custom_n_harmonics(self, sample_rate: float) -> None:
        """Test harmonic analysis with custom number of harmonics."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        harmonics_3 = ripple_harmonics(trace, fundamental_freq=120.0, n_harmonics=3)
        harmonics_10 = ripple_harmonics(trace, fundamental_freq=120.0, n_harmonics=10)

        assert len(harmonics_3) == 3
        assert len(harmonics_10) == 10

    def test_ripple_harmonics_return_types(self, sample_rate: float) -> None:
        """Test that harmonic amplitudes are floats."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        harmonics = ripple_harmonics(trace, fundamental_freq=120.0)

        for h, amp in harmonics.items():
            assert isinstance(h, int)
            assert isinstance(amp, float)

    def test_ripple_harmonics_negative_frequency(self, sample_rate: float) -> None:
        """Test harmonic analysis with negative fundamental frequency."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        harmonics = ripple_harmonics(trace, fundamental_freq=-120.0)

        # Should return empty dict for negative frequency
        assert len(harmonics) == 0


@pytest.mark.unit
@pytest.mark.power
class TestRippleStatistics:
    """Test comprehensive ripple statistics."""

    def test_ripple_statistics_basic(self, sample_rate: float) -> None:
        """Test basic ripple statistics calculation."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        # Check all required keys are present
        assert "dc_level" in stats
        assert "ripple_pp" in stats
        assert "ripple_rms" in stats
        assert "ripple_pp_percent" in stats
        assert "ripple_rms_percent" in stats
        assert "ripple_frequency" in stats
        assert "crest_factor" in stats

    def test_ripple_statistics_dc_level(self, sample_rate: float) -> None:
        """Test DC level in statistics."""
        dc_level = 12.0
        trace = create_dc_with_ripple(dc_level, 0.1, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        assert abs(stats["dc_level"] - dc_level) < 0.01

    def test_ripple_statistics_ripple_values(self, sample_rate: float) -> None:
        """Test ripple measurements in statistics."""
        ripple_amp = 0.1
        trace = create_dc_with_ripple(12.0, ripple_amp, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        expected_pp = 2 * ripple_amp
        expected_rms = ripple_amp / np.sqrt(2)

        assert abs(stats["ripple_pp"] - expected_pp) < 0.01
        assert abs(stats["ripple_rms"] - expected_rms) < 0.01

    def test_ripple_statistics_percentages(self, sample_rate: float) -> None:
        """Test ripple percentages in statistics."""
        dc_level = 12.0
        ripple_amp = 0.12
        trace = create_dc_with_ripple(dc_level, ripple_amp, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        expected_pp_pct = (2 * ripple_amp) / dc_level * 100
        assert abs(stats["ripple_pp_percent"] - expected_pp_pct) < 0.1

    def test_ripple_statistics_frequency(self, sample_rate: float) -> None:
        """Test frequency detection in statistics."""
        ripple_freq = 120.0
        trace = create_dc_with_ripple(12.0, 0.1, ripple_freq, sample_rate)

        stats = ripple_statistics(trace)

        assert abs(stats["ripple_frequency"] - ripple_freq) < 10.0

    def test_ripple_statistics_crest_factor(self, sample_rate: float) -> None:
        """Test crest factor calculation."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        # Crest factor for sine wave is sqrt(2) ≈ 1.414
        expected_cf = np.sqrt(2)
        assert abs(stats["crest_factor"] - expected_cf) < 0.1

    def test_ripple_statistics_zero_dc(self, sample_rate: float) -> None:
        """Test statistics with zero DC level."""
        trace = create_dc_with_ripple(0.0, 0.1, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        # DC level should be very close to zero
        assert abs(stats["dc_level"]) < 0.01
        # Percentages are either NaN or very large when DC is near zero
        assert np.isnan(stats["ripple_pp_percent"]) or abs(stats["ripple_pp_percent"]) > 1e10
        assert np.isnan(stats["ripple_rms_percent"]) or abs(stats["ripple_rms_percent"]) > 1e10

    def test_ripple_statistics_pure_dc(self, sample_rate: float) -> None:
        """Test statistics with pure DC (no ripple)."""
        trace = create_dc_only(12.0, sample_rate)

        stats = ripple_statistics(trace)

        assert abs(stats["dc_level"] - 12.0) < 0.01
        assert abs(stats["ripple_pp"]) < 1e-10
        assert abs(stats["ripple_rms"]) < 1e-10

    def test_ripple_statistics_return_types(self, sample_rate: float) -> None:
        """Test that all statistics are numeric."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        stats = ripple_statistics(trace)

        for key, value in stats.items():
            assert isinstance(value, int | float) or np.isnan(value), f"{key} is not numeric"

    def test_ripple_statistics_square_wave(self, sample_rate: float) -> None:
        """Test statistics with square wave ripple."""
        trace = create_square_wave_ripple(12.0, 0.1, 100.0, sample_rate)

        stats = ripple_statistics(trace)

        # Crest factor for square wave is 1.0
        assert abs(stats["crest_factor"] - 1.0) < 0.1


@pytest.mark.unit
@pytest.mark.power
class TestExtractRipple:
    """Test AC ripple extraction."""

    def test_extract_ripple_basic(self, sample_rate: float) -> None:
        """Test basic ripple extraction."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        ac_trace = extract_ripple(trace)

        assert isinstance(ac_trace, WaveformTrace)
        assert len(ac_trace.data) == len(trace.data)
        assert ac_trace.metadata.sample_rate == sample_rate

    def test_extract_ripple_removes_dc(self, sample_rate: float) -> None:
        """Test that DC is removed from extracted ripple."""
        dc_level = 12.0
        trace = create_dc_with_ripple(dc_level, 0.1, 120.0, sample_rate)

        ac_trace = extract_ripple(trace)

        # Mean should be approximately zero (DC removed)
        assert abs(np.mean(ac_trace.data)) < 0.01

    def test_extract_ripple_preserves_ac(self, sample_rate: float) -> None:
        """Test that AC component is preserved."""
        ripple_amp = 0.1
        trace = create_dc_with_ripple(12.0, ripple_amp, 120.0, sample_rate)

        ac_trace = extract_ripple(trace)

        # Peak-to-peak should match original ripple
        pp = np.max(ac_trace.data) - np.min(ac_trace.data)
        expected_pp = 2 * ripple_amp
        assert abs(pp - expected_pp) < 0.01

    def test_extract_ripple_with_high_pass_filter(self, sample_rate: float) -> None:
        """Test ripple extraction with high-pass filter."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        ac_trace = extract_ripple(trace, high_pass_freq=10.0)

        assert isinstance(ac_trace, WaveformTrace)
        # Mean should still be near zero
        assert abs(np.mean(ac_trace.data)) < 0.05

    def test_extract_ripple_high_pass_too_high(self, sample_rate: float) -> None:
        """Test error when high-pass frequency exceeds Nyquist."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        nyquist = sample_rate / 2
        with pytest.raises(AnalysisError, match="must be less than"):
            extract_ripple(trace, high_pass_freq=nyquist + 100.0)

    def test_extract_ripple_preserves_metadata(self, sample_rate: float) -> None:
        """Test that metadata is preserved."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        ac_trace = extract_ripple(trace)

        assert ac_trace.metadata.sample_rate == trace.metadata.sample_rate

    def test_extract_ripple_dtype(self, sample_rate: float) -> None:
        """Test that extracted ripple is float64."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        ac_trace = extract_ripple(trace)

        assert ac_trace.data.dtype == np.float64

    def test_extract_ripple_pure_dc(self, sample_rate: float) -> None:
        """Test ripple extraction from pure DC."""
        trace = create_dc_only(12.0, sample_rate)

        ac_trace = extract_ripple(trace)

        # Should be near-zero everywhere
        assert np.all(np.abs(ac_trace.data) < 1e-10)

    def test_extract_ripple_no_dc(self, sample_rate: float) -> None:
        """Test ripple extraction from signal already at zero DC."""
        trace = create_dc_with_ripple(0.0, 0.1, 120.0, sample_rate)

        ac_trace = extract_ripple(trace)

        # Should preserve the AC component
        pp = np.max(ac_trace.data) - np.min(ac_trace.data)
        assert pp > 0.1


@pytest.mark.unit
@pytest.mark.power
class TestRippleEnvelope:
    """Test ripple envelope detection."""

    def test_ripple_envelope_hilbert(self, sample_rate: float) -> None:
        """Test envelope detection using Hilbert transform."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        envelope = ripple_envelope(trace, method="hilbert")

        assert isinstance(envelope, WaveformTrace)
        assert len(envelope.data) == len(trace.data)

    def test_ripple_envelope_peak(self, sample_rate: float) -> None:
        """Test envelope detection using peak method."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        envelope = ripple_envelope(trace, method="peak")

        assert isinstance(envelope, WaveformTrace)
        assert len(envelope.data) == len(trace.data)

    def test_ripple_envelope_invalid_method(self, sample_rate: float) -> None:
        """Test error with invalid envelope method."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        with pytest.raises(AnalysisError, match="Unknown envelope method"):
            ripple_envelope(trace, method="invalid")

    def test_ripple_envelope_preserves_metadata(self, sample_rate: float) -> None:
        """Test that metadata is preserved."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        envelope = ripple_envelope(trace)

        assert envelope.metadata.sample_rate == trace.metadata.sample_rate

    def test_ripple_envelope_dtype(self, sample_rate: float) -> None:
        """Test that envelope is float64."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        envelope = ripple_envelope(trace)

        assert envelope.data.dtype == np.float64

    def test_ripple_envelope_amplitude(self, sample_rate: float) -> None:
        """Test that envelope tracks amplitude."""
        ripple_amp = 0.1
        trace = create_dc_with_ripple(12.0, ripple_amp, 120.0, sample_rate)

        envelope = ripple_envelope(trace)

        # Envelope should be approximately constant at ripple_amp
        mean_envelope = np.mean(envelope.data)
        assert abs(mean_envelope - ripple_amp) < 0.05

    def test_ripple_envelope_modulated_signal(self, sample_rate: float) -> None:
        """Test envelope with amplitude-modulated ripple."""
        # Create AM signal: carrier at 1000 Hz, modulation at 50 Hz
        num_samples = int(sample_rate * 0.1)
        t = np.arange(num_samples) / sample_rate

        carrier_freq = 1000.0
        mod_freq = 50.0
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        data = 12.0 + 0.1 * modulation * carrier

        trace = create_trace(data, sample_rate)
        envelope = ripple_envelope(trace)

        # Envelope should vary (not constant)
        assert np.std(envelope.data) > 0.01

    def test_ripple_envelope_constant_amplitude(self, sample_rate: float) -> None:
        """Test envelope with constant amplitude signal."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate)

        envelope = ripple_envelope(trace)

        # Should be relatively constant (low variation)
        variation = np.std(envelope.data) / np.mean(envelope.data)
        assert variation < 0.5  # Coefficient of variation


@pytest.mark.unit
@pytest.mark.power
class TestPowerRippleEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_trace(self, sample_rate: float) -> None:
        """Test that empty trace raises ValueError."""
        empty_data = np.array([])

        # Empty trace creation raises ValueError in v0.9.0+
        with pytest.raises(ValueError, match="data array cannot be empty"):
            create_trace(empty_data, sample_rate)

    def test_single_sample(self, sample_rate: float) -> None:
        """Test functions with single sample."""
        data = np.array([12.0])
        trace = create_trace(data, sample_rate)

        r_pp, r_rms = ripple(trace)
        # Single sample has no ripple
        assert abs(r_pp) < 1e-10

    def test_very_short_trace(self, sample_rate: float) -> None:
        """Test functions with very short trace."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate, duration=0.001)

        stats = ripple_statistics(trace)
        assert stats is not None

    def test_very_long_trace(self, sample_rate: float) -> None:
        """Test functions with long trace."""
        trace = create_dc_with_ripple(12.0, 0.1, 120.0, sample_rate, duration=1.0)

        r_pp, r_rms = ripple(trace)
        assert r_pp > 0

    def test_very_high_frequency(self, sample_rate: float) -> None:
        """Test with high frequency ripple near Nyquist."""
        nyquist = sample_rate / 2
        ripple_freq = nyquist * 0.8  # 80% of Nyquist

        trace = create_dc_with_ripple(12.0, 0.1, ripple_freq, sample_rate)

        detected = ripple_frequency(trace)
        # May not detect perfectly, but should be in range
        assert detected > 0

    def test_very_low_frequency(self, sample_rate: float) -> None:
        """Test with very low frequency ripple."""
        ripple_freq = 1.0  # 1 Hz

        trace = create_dc_with_ripple(12.0, 0.1, ripple_freq, sample_rate, duration=2.0)

        detected = ripple_frequency(trace)
        assert abs(detected - ripple_freq) < 5.0

    def test_nan_in_data(self, sample_rate: float) -> None:
        """Test handling of NaN values."""
        data = np.array([12.0, 12.1, np.nan, 11.9, 12.0])
        trace = create_trace(data, sample_rate)

        r_pp, r_rms = ripple(trace)
        # Results will contain NaN
        assert np.isnan(r_pp) or np.isnan(r_rms)

    def test_inf_in_data(self, sample_rate: float) -> None:
        """Test handling of infinite values."""
        data = np.array([12.0, 12.1, np.inf, 11.9, 12.0])
        trace = create_trace(data, sample_rate)

        # Suppress runtime warnings from np.inf operations
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_pp, r_rms = ripple(trace)
            # Results will be inf or nan
            assert np.isinf(r_pp) or np.isnan(r_pp)
            assert np.isinf(r_rms) or np.isnan(r_rms)

    def test_all_zeros(self, sample_rate: float) -> None:
        """Test with all-zero data."""
        data = np.zeros(1000)
        trace = create_trace(data, sample_rate)

        r_pp, r_rms = ripple(trace)
        assert abs(r_pp) < 1e-10
        assert abs(r_rms) < 1e-10

    def test_all_same_value(self, sample_rate: float) -> None:
        """Test with constant value (no variation)."""
        trace = create_dc_only(12.0, sample_rate)

        stats = ripple_statistics(trace)
        assert abs(stats["ripple_pp"]) < 1e-10
        assert abs(stats["ripple_rms"]) < 1e-10

    def test_alternating_values(self, sample_rate: float) -> None:
        """Test with alternating high/low values (highest frequency)."""
        # Maximum frequency ripple (Nyquist)
        data = np.array([11.9, 12.1] * 500)
        trace = create_trace(data, sample_rate)

        r_pp, r_rms = ripple(trace)
        assert r_pp > 0

    def test_very_small_amplitude(self, sample_rate: float) -> None:
        """Test with very small ripple amplitude."""
        trace = create_dc_with_ripple(12.0, 1e-6, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)
        assert r_pp < 1e-5

    def test_very_large_amplitude(self, sample_rate: float) -> None:
        """Test with very large ripple amplitude."""
        trace = create_dc_with_ripple(12.0, 10.0, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)
        assert r_pp > 10.0

    def test_negative_values(self, sample_rate: float) -> None:
        """Test with negative DC level."""
        trace = create_dc_with_ripple(-12.0, 0.1, 120.0, sample_rate)

        stats = ripple_statistics(trace)
        assert stats["dc_level"] < 0
        assert stats["ripple_pp"] > 0

    def test_bipolar_signal(self, sample_rate: float) -> None:
        """Test with signal crossing zero."""
        # -5V to +5V swing
        trace = create_dc_with_ripple(0.0, 5.0, 120.0, sample_rate)

        r_pp, r_rms = ripple(trace)
        assert r_pp > 5.0

    def test_complex_waveform(self, sample_rate: float) -> None:
        """Test with complex multi-frequency waveform."""
        trace = create_multi_freq_ripple(
            dc_level=12.0,
            fundamental_amp=0.1,
            fundamental_freq=100.0,
            sample_rate=sample_rate,
            harmonics={2: 0.3, 3: 0.2, 5: 0.1},
        )

        stats = ripple_statistics(trace)
        # Should handle complex waveform
        assert stats["ripple_pp"] > 0
        assert stats["ripple_frequency"] > 0

    def test_noise_floor(self, sample_rate: float) -> None:
        """Test with very low amplitude noise."""
        # Add tiny random noise
        num_samples = int(sample_rate * 0.1)
        noise = np.random.normal(0, 1e-6, num_samples)
        data = 12.0 + noise
        trace = create_trace(data, sample_rate)

        r_pp, r_rms = ripple(trace)
        # Should measure the noise
        assert r_rms > 0

    def test_dc_offset_independence(self, sample_rate: float) -> None:
        """Test that ripple measurement is independent of DC offset."""
        ripple_amp = 0.1
        ripple_freq = 120.0

        trace1 = create_dc_with_ripple(5.0, ripple_amp, ripple_freq, sample_rate)
        trace2 = create_dc_with_ripple(12.0, ripple_amp, ripple_freq, sample_rate)
        trace3 = create_dc_with_ripple(24.0, ripple_amp, ripple_freq, sample_rate)

        r_pp1, r_rms1 = ripple(trace1)
        r_pp2, r_rms2 = ripple(trace2)
        r_pp3, r_rms3 = ripple(trace3)

        # All should have same ripple
        assert abs(r_pp1 - r_pp2) < 0.01
        assert abs(r_pp2 - r_pp3) < 0.01
        assert abs(r_rms1 - r_rms2) < 0.01
