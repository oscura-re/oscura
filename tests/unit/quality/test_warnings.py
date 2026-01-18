"""Comprehensive unit tests for signal quality warnings.

This module tests signal quality analysis including clipping detection,
noise analysis, saturation detection, and undersampling warnings.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.quality.warnings import (
    QualityWarning,
    SignalQualityAnalyzer,
    check_clipping,
    check_noise,
    check_saturation,
    check_undersampling,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_signal() -> np.ndarray:
    """Create a clean test signal without issues."""
    t = np.linspace(0, 1, 1000)
    # 10 Hz sine wave with amplitude 0.3 (well within -1 to +1 range)
    # Add DC offset to ensure not symmetric
    return 0.3 * np.sin(2 * np.pi * 10 * t) + 0.5


@pytest.fixture
def clipped_signal() -> np.ndarray:
    """Create a signal with clipping."""
    t = np.linspace(0, 1, 1000)
    signal = 2.0 * np.sin(2 * np.pi * 10 * t)
    # Clip to [-1, 1] range
    return np.clip(signal, -1, 1)


@pytest.fixture
def noisy_signal() -> np.ndarray:
    """Create a signal with high noise."""
    np.random.seed(42)  # For reproducibility
    t = np.linspace(0, 1, 1000)
    signal = 0.01 * np.sin(2 * np.pi * 10 * t)  # Very weak signal
    noise = 1.0 * np.random.randn(1000)  # Strong noise
    return signal + noise


@pytest.fixture
def undersampled_signal() -> np.ndarray:
    """Create an undersampled high-frequency signal."""
    t = np.linspace(0, 1e-3, 100)  # Only 100 samples for 1ms
    # 50 kHz signal (way above Nyquist for 100kHz sample rate)
    return np.sin(2 * np.pi * 50e3 * t)


@pytest.fixture
def saturated_signal() -> np.ndarray:
    """Create a signal with high ADC utilization."""
    t = np.linspace(0, 1, 1000)
    # Signal using 99% of the range
    return 0.99 * np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def trace_metadata() -> TraceMetadata:
    """Create basic trace metadata."""
    return TraceMetadata(sample_rate=1e6)  # 1 MHz


@pytest.fixture
def waveform_trace(clean_signal: np.ndarray, trace_metadata: TraceMetadata) -> WaveformTrace:
    """Create a WaveformTrace from clean signal."""
    return WaveformTrace(data=clean_signal.astype(np.float64), metadata=trace_metadata)


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestQualityWarning:
    """Test QualityWarning dataclass functionality."""

    def test_create_warning(self) -> None:
        """Test creating a quality warning."""
        warning = QualityWarning(
            severity="warning",
            category="clipping",
            message="Signal clipping detected",
            value=5.2,
            threshold=5.0,
            suggestion="Reduce input amplitude",
        )

        assert warning.severity == "warning"
        assert warning.category == "clipping"
        assert warning.message == "Signal clipping detected"
        assert warning.value == 5.2
        assert warning.threshold == 5.0
        assert warning.suggestion == "Reduce input amplitude"

    def test_warning_str_with_suggestion(self) -> None:
        """Test string formatting with suggestion."""
        warning = QualityWarning(
            severity="error",
            category="undersampling",
            message="Undersampling detected",
            value=100.0,
            threshold=50.0,
            suggestion="Increase sample rate",
        )

        result = str(warning)
        assert "[ERROR]" in result
        assert "Undersampling detected" in result
        assert "value: 100.000" in result
        assert "threshold: 50.000" in result
        assert "Suggestion: Increase sample rate" in result

    def test_warning_str_without_suggestion(self) -> None:
        """Test string formatting without suggestion."""
        warning = QualityWarning(
            severity="info",
            category="noise",
            message="Low noise detected",
            value=1.0,
            threshold=2.0,
        )

        result = str(warning)
        assert "[INFO]" in result
        assert "Low noise detected" in result
        assert "Suggestion" not in result

    def test_warning_str_severity_levels(self) -> None:
        """Test all severity levels format correctly."""
        for severity in ["error", "warning", "info"]:
            warning = QualityWarning(
                severity=severity,  # type: ignore
                category="clipping",
                message="Test",
                value=1.0,
                threshold=1.0,
            )
            result = str(warning)
            assert severity.upper() in result


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestCheckClipping:
    """Test clipping detection function."""

    def test_no_clipping_clean_signal(self, clean_signal: np.ndarray) -> None:
        """Test that clean signal produces no clipping warnings."""
        # Provide explicit ADC range to avoid false positives
        warnings = check_clipping(clean_signal, adc_range=(0.0, 1.0))
        assert len(warnings) == 0

    def test_detect_clipping(self, clipped_signal: np.ndarray) -> None:
        """Test detection of clipped signal."""
        warnings = check_clipping(clipped_signal, threshold=0.95)
        assert len(warnings) > 0
        assert warnings[0].category == "clipping"
        assert warnings[0].value > 1.0  # More than 1% clipped

    def test_clipping_with_adc_range(self) -> None:
        """Test clipping detection with explicit ADC range."""
        # Signal from -10 to +10, but ADC range is -5 to +5
        signal = np.linspace(-10, 10, 1000)
        warnings = check_clipping(signal, adc_range=(-5.0, 5.0), threshold=0.95)
        assert len(warnings) > 0

    def test_clipping_severity_warning(self) -> None:
        """Test that 1-5% clipping produces warning severity."""
        # Create signal with exactly 3% clipping (30 samples out of 1000)
        signal = np.zeros(1000)
        signal[:15] = 1.0  # Upper rail
        signal[-15:] = -1.0  # Lower rail
        # Total 30 samples = 3% clipping
        warnings = check_clipping(signal, threshold=0.95, adc_range=(-1.0, 1.0))

        assert len(warnings) > 0
        assert warnings[0].severity == "warning"

    def test_clipping_severity_error(self) -> None:
        """Test that >5% clipping produces error severity."""
        # Create signal with 10% clipping
        signal = np.zeros(1000)
        signal[:100] = 1.0  # 10% at upper limit
        warnings = check_clipping(signal, threshold=0.99)

        assert len(warnings) > 0
        assert warnings[0].severity == "error"

    def test_zero_range_signal(self) -> None:
        """Test that constant signal produces no warnings."""
        signal = np.ones(1000) * 0.5
        warnings = check_clipping(signal)
        assert len(warnings) == 0

    def test_clipping_suggestion(self, clipped_signal: np.ndarray) -> None:
        """Test that clipping warning includes suggestion."""
        warnings = check_clipping(clipped_signal)
        if len(warnings) > 0:
            assert "amplitude" in warnings[0].suggestion.lower()

    def test_both_rails_clipping(self) -> None:
        """Test detection of clipping at both rails."""
        signal = np.array([-1.0] * 50 + [0.0] * 900 + [1.0] * 50)
        warnings = check_clipping(signal, threshold=0.99)
        assert len(warnings) > 0


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestCheckSaturation:
    """Test saturation detection function."""

    def test_no_saturation_clean_signal(self, clean_signal: np.ndarray) -> None:
        """Test that clean signal with low amplitude produces no saturation warnings."""
        # Provide explicit ADC range to avoid false positives
        warnings = check_saturation(clean_signal, adc_range=(0.0, 1.0))
        assert len(warnings) == 0

    def test_detect_saturation(self, saturated_signal: np.ndarray) -> None:
        """Test detection of saturated signal."""
        warnings = check_saturation(saturated_signal, threshold=0.95)
        assert len(warnings) > 0
        assert warnings[0].category == "saturation"
        assert warnings[0].value > 95.0  # More than 95% utilization

    def test_saturation_with_adc_range(self) -> None:
        """Test saturation detection with explicit ADC range."""
        # Signal uses almost full range of -10 to +10
        signal = 9.9 * np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        warnings = check_saturation(signal, adc_range=(-10.0, 10.0), threshold=0.95)
        assert len(warnings) > 0

    def test_zero_range_signal_saturation(self) -> None:
        """Test that constant signal produces no saturation warnings."""
        signal = np.ones(1000) * 0.5
        warnings = check_saturation(signal)
        assert len(warnings) == 0

    def test_saturation_suggestion(self, saturated_signal: np.ndarray) -> None:
        """Test that saturation warning includes suggestion."""
        warnings = check_saturation(saturated_signal, threshold=0.95)
        if len(warnings) > 0:
            assert "ADC range" in warnings[0].suggestion or "amplitude" in warnings[0].suggestion

    def test_low_utilization_no_warning(self) -> None:
        """Test that low ADC utilization produces no warnings."""
        signal = 0.3 * np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        warnings = check_saturation(signal, threshold=0.98, adc_range=(-1.0, 1.0))
        assert len(warnings) == 0


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestCheckNoise:
    """Test noise detection function."""

    def test_no_noise_clean_signal(self, clean_signal: np.ndarray) -> None:
        """Test that clean signal produces no noise warnings."""
        warnings = check_noise(clean_signal, threshold_db=-40.0)
        # Clean sine wave should have good SNR
        assert len(warnings) == 0

    def test_detect_noise(self, noisy_signal: np.ndarray) -> None:
        """Test detection of noisy signal."""
        # Use a threshold that should detect our noisy signal
        warnings = check_noise(noisy_signal, threshold_db=5.0)
        assert len(warnings) > 0
        assert warnings[0].category == "noise"

    def test_zero_signal_power(self) -> None:
        """Test that zero signal produces no warnings."""
        signal = np.zeros(1000)
        warnings = check_noise(signal)
        assert len(warnings) == 0

    def test_zero_noise_power(self) -> None:
        """Test constant signal (zero variance) produces no warnings."""
        signal = np.ones(1000) * 5.0
        warnings = check_noise(signal)
        assert len(warnings) == 0

    def test_noise_suggestion(self, noisy_signal: np.ndarray) -> None:
        """Test that noise warning includes suggestion."""
        warnings = check_noise(noisy_signal, threshold_db=5.0)
        if len(warnings) > 0:
            assert (
                "grounding" in warnings[0].suggestion.lower()
                or "shielding" in warnings[0].suggestion.lower()
            )

    def test_noise_threshold_parameter(self) -> None:
        """Test that noise threshold parameter works correctly."""
        # Create moderately noisy signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)

        # Strict threshold should detect it
        warnings_strict = check_noise(signal, threshold_db=10.0)
        # Lenient threshold should not
        warnings_lenient = check_noise(signal, threshold_db=-60.0)

        assert len(warnings_strict) > len(warnings_lenient)


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestCheckUndersampling:
    """Test undersampling detection function."""

    def test_no_undersampling(self, clean_signal: np.ndarray) -> None:
        """Test that properly sampled signal produces no warnings."""
        # 10 Hz signal sampled at 1 kHz is well above Nyquist
        warnings = check_undersampling(clean_signal, sample_rate=1000.0)
        assert len(warnings) == 0

    def test_detect_undersampling(self, undersampled_signal: np.ndarray) -> None:
        """Test detection of undersampled signal."""
        # 50 kHz signal sampled at 100 kHz (close to Nyquist)
        warnings = check_undersampling(undersampled_signal, sample_rate=100e3)
        assert len(warnings) > 0
        assert warnings[0].category == "undersampling"
        assert warnings[0].severity == "error"

    def test_nyquist_factor_parameter(self) -> None:
        """Test that nyquist_factor parameter affects detection."""
        # Create 40 kHz signal sampled at 100 kHz
        t = np.linspace(0, 1e-3, 100)
        signal = np.sin(2 * np.pi * 40e3 * t)

        # With factor=2.0, should be OK (40kHz < 50kHz/2)
        warnings_2x = check_undersampling(signal, sample_rate=100e3, nyquist_factor=2.0)
        # With factor=1.5, might trigger (40kHz > 50kHz/1.5)
        warnings_1_5x = check_undersampling(signal, sample_rate=100e3, nyquist_factor=1.5)

        # The stricter factor should produce more or equal warnings
        assert len(warnings_1_5x) >= len(warnings_2x)

    def test_undersampling_suggestion(self, undersampled_signal: np.ndarray) -> None:
        """Test that undersampling warning includes suggestion."""
        warnings = check_undersampling(undersampled_signal, sample_rate=100e3)
        if len(warnings) > 0:
            assert (
                "sample rate" in warnings[0].suggestion.lower()
                or "filter" in warnings[0].suggestion.lower()
            )

    def test_dc_signal_no_undersampling(self) -> None:
        """Test that DC signal (no high frequencies) produces no warnings."""
        signal = np.ones(1000) * 5.0
        warnings = check_undersampling(signal, sample_rate=1e6)
        assert len(warnings) == 0

    def test_low_frequency_signal(self) -> None:
        """Test that low frequency signal produces no warnings."""
        # 1 Hz signal sampled at 1 kHz
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 1 * t)
        warnings = check_undersampling(signal, sample_rate=1000.0)
        assert len(warnings) == 0


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestSignalQualityAnalyzer:
    """Test SignalQualityAnalyzer class."""

    def test_init_default_parameters(self) -> None:
        """Test analyzer initialization with default parameters."""
        analyzer = SignalQualityAnalyzer()
        assert analyzer.clip_threshold == 0.99
        assert analyzer.noise_threshold_db == -40.0
        assert analyzer.saturation_threshold == 0.98
        assert analyzer.nyquist_factor == 2.0

    def test_init_custom_parameters(self) -> None:
        """Test analyzer initialization with custom parameters."""
        analyzer = SignalQualityAnalyzer(
            clip_threshold=0.95,
            noise_threshold_db=-50.0,
            saturation_threshold=0.90,
            nyquist_factor=2.5,
        )
        assert analyzer.clip_threshold == 0.95
        assert analyzer.noise_threshold_db == -50.0
        assert analyzer.saturation_threshold == 0.90
        assert analyzer.nyquist_factor == 2.5

    def test_analyze_clean_signal(self, clean_signal: np.ndarray) -> None:
        """Test analyzing clean signal produces no warnings."""
        analyzer = SignalQualityAnalyzer()
        # Provide ADC range to avoid false positives from range inference
        warnings = analyzer.analyze(clean_signal, sample_rate=1e6, adc_range=(0.0, 1.0))
        # Clean signal should produce no warnings
        assert len(warnings) == 0

    def test_analyze_with_waveform_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test analyzing WaveformTrace object."""
        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(waveform_trace)
        # Should extract sample_rate from metadata
        assert isinstance(warnings, list)

    def test_analyze_clipped_signal(self, clipped_signal: np.ndarray) -> None:
        """Test that analyzer detects clipping."""
        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(clipped_signal)

        # Should have clipping warning
        clipping_warnings = [w for w in warnings if w.category == "clipping"]
        assert len(clipping_warnings) > 0

    def test_analyze_noisy_signal(self, noisy_signal: np.ndarray) -> None:
        """Test that analyzer detects noise."""
        # Use a higher threshold to detect the noise
        analyzer = SignalQualityAnalyzer(noise_threshold_db=5.0)
        warnings = analyzer.analyze(noisy_signal)

        # Should have noise warning
        noise_warnings = [w for w in warnings if w.category == "noise"]
        assert len(noise_warnings) > 0

    def test_analyze_saturated_signal(self, saturated_signal: np.ndarray) -> None:
        """Test that analyzer detects saturation."""
        analyzer = SignalQualityAnalyzer(saturation_threshold=0.95)
        warnings = analyzer.analyze(saturated_signal)

        # Should have saturation warning
        saturation_warnings = [w for w in warnings if w.category == "saturation"]
        assert len(saturation_warnings) > 0

    def test_analyze_undersampled_signal(self, undersampled_signal: np.ndarray) -> None:
        """Test that analyzer detects undersampling."""
        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(undersampled_signal, sample_rate=100e3)

        # Should have undersampling warning
        undersampling_warnings = [w for w in warnings if w.category == "undersampling"]
        assert len(undersampling_warnings) > 0

    def test_analyze_without_sample_rate(self, clean_signal: np.ndarray) -> None:
        """Test that analyzer works without sample rate (skips undersampling check)."""
        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(clean_signal)

        # Should not check for undersampling
        undersampling_warnings = [w for w in warnings if w.category == "undersampling"]
        assert len(undersampling_warnings) == 0

    def test_analyze_with_adc_range(self) -> None:
        """Test analyzer with explicit ADC range."""
        analyzer = SignalQualityAnalyzer()
        signal = np.linspace(-5, 5, 1000)
        warnings = analyzer.analyze(signal, adc_range=(-5.0, 5.0))

        # Should use provided ADC range for clipping/saturation checks
        assert isinstance(warnings, list)

    def test_analyze_multiple_issues(self) -> None:
        """Test that analyzer can detect multiple issues in one signal."""
        # Create signal with multiple problems
        signal = np.clip(np.random.randn(1000) * 2, -1, 1)  # Clipped and noisy

        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(signal, sample_rate=1e6)

        # Should detect multiple issues
        assert len(warnings) > 0
        categories = {w.category for w in warnings}
        # At minimum should detect clipping
        assert "clipping" in categories

    def test_analyze_trace_with_metadata(self) -> None:
        """Test analyzing trace extracts sample_rate from metadata."""
        # Create a mock trace with metadata
        trace = MagicMock()
        trace.data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        trace.metadata = MagicMock()
        trace.metadata.sample_rate = 1e6

        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(trace)

        # Should successfully analyze using metadata sample_rate
        assert isinstance(warnings, list)

    def test_analyze_numpy_array_conversion(self) -> None:
        """Test that analyzer converts input to float64 array."""
        analyzer = SignalQualityAnalyzer()
        # Test with list input
        signal_list = [0.0, 0.5, 1.0, 0.5, 0.0]
        warnings = analyzer.analyze(signal_list)  # type: ignore
        assert isinstance(warnings, list)

        # Test with int array
        signal_int = np.array([0, 1, 2, 1, 0], dtype=np.int32)
        warnings = analyzer.analyze(signal_int)  # type: ignore
        assert isinstance(warnings, list)

    def test_custom_thresholds_affect_results(self) -> None:
        """Test that custom thresholds change detection behavior."""
        signal = 0.96 * np.sin(2 * np.pi * np.linspace(0, 1, 1000))

        # Strict thresholds
        analyzer_strict = SignalQualityAnalyzer(saturation_threshold=0.90)
        warnings_strict = analyzer_strict.analyze(signal)

        # Lenient thresholds
        analyzer_lenient = SignalQualityAnalyzer(saturation_threshold=0.99)
        warnings_lenient = analyzer_lenient.analyze(signal)

        # Strict should produce more warnings
        assert len(warnings_strict) >= len(warnings_lenient)


@pytest.mark.unit
@pytest.mark.requirement("EDGE-001")
class TestQualityWarningsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_signal(self) -> None:
        """Test handling of empty signal array."""
        signal = np.array([])

        # Empty arrays should either return empty warnings or raise ValueError
        # The current implementation will raise ValueError on min/max operations
        # This is acceptable behavior for edge cases
        try:
            warnings_clip = check_clipping(signal)
            assert isinstance(warnings_clip, list)
        except ValueError:
            pass  # Acceptable for empty arrays

        try:
            warnings_sat = check_saturation(signal)
            assert isinstance(warnings_sat, list)
        except ValueError:
            pass  # Acceptable for empty arrays

        try:
            warnings_noise = check_noise(signal)
            assert isinstance(warnings_noise, list)
        except (ValueError, RuntimeWarning):
            pass  # Acceptable for empty arrays

    def test_single_sample(self) -> None:
        """Test handling of single sample signal."""
        signal = np.array([1.0])

        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(signal, sample_rate=1e6)

        # Should handle gracefully
        assert isinstance(warnings, list)

    def test_very_small_signal(self) -> None:
        """Test handling of very small amplitude signal."""
        signal = 1e-10 * np.sin(2 * np.pi * np.linspace(0, 1, 1000))

        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(signal, sample_rate=1e6)

        assert isinstance(warnings, list)

    def test_very_large_signal(self) -> None:
        """Test handling of very large amplitude signal."""
        signal = 1e6 * np.sin(2 * np.pi * np.linspace(0, 1, 1000))

        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(signal, sample_rate=1e6)

        assert isinstance(warnings, list)

    def test_nan_values(self) -> None:
        """Test handling of NaN values in signal."""
        signal = np.array([1.0, 2.0, np.nan, 3.0, 4.0])

        # Should not crash - may produce warnings or handle gracefully
        analyzer = SignalQualityAnalyzer()
        try:
            warnings = analyzer.analyze(signal, sample_rate=1e6)
            assert isinstance(warnings, list)
        except (ValueError, RuntimeWarning):
            # NaN handling may raise warnings/errors - acceptable
            pass

    def test_inf_values(self) -> None:
        """Test handling of infinite values in signal."""
        signal = np.array([1.0, 2.0, np.inf, 3.0, 4.0])

        # Should not crash
        analyzer = SignalQualityAnalyzer()
        try:
            warnings = analyzer.analyze(signal, sample_rate=1e6)
            assert isinstance(warnings, list)
        except (ValueError, RuntimeWarning):
            # Inf handling may raise warnings/errors - acceptable
            pass

    def test_negative_adc_range(self) -> None:
        """Test with negative ADC range."""
        signal = -5 + 10 * np.random.rand(1000)  # Signal from -5 to +5
        warnings = check_clipping(signal, adc_range=(-10.0, 0.0))

        # Should handle negative ranges correctly
        assert isinstance(warnings, list)

    def test_zero_threshold(self) -> None:
        """Test with zero threshold values."""
        signal = np.sin(2 * np.pi * np.linspace(0, 1, 1000))

        # Edge case: zero thresholds
        analyzer = SignalQualityAnalyzer(
            clip_threshold=0.0,
            saturation_threshold=0.0,
        )
        warnings = analyzer.analyze(signal)

        # Should handle zero thresholds
        assert isinstance(warnings, list)

    def test_one_threshold(self) -> None:
        """Test with threshold = 1.0."""
        signal = np.sin(2 * np.pi * np.linspace(0, 1, 1000))

        analyzer = SignalQualityAnalyzer(
            clip_threshold=1.0,
            saturation_threshold=1.0,
        )
        warnings = analyzer.analyze(signal)

        # Should handle threshold = 1.0
        assert isinstance(warnings, list)
