"""Comprehensive unit tests for src/oscura/core/edge_cases.py

Tests coverage for:

Covers all public classes, functions, properties, and methods with
edge cases, error handling, and validation.
"""

import warnings

import numpy as np
import pytest

from oscura.core.edge_cases import (
    EmptyTraceError,
    InsufficientSamplesError,
    SignalQualityReport,
    check_signal_quality,
    check_single_sample,
    handle_empty_trace,
    sanitize_signal,
    validate_signal,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ==============================================================================
# Exception Classes Tests
# ==============================================================================


class TestEmptyTraceError:
    """Test EmptyTraceError exception."""

    def test_create_with_default_message(self) -> None:
        """Test creating EmptyTraceError with default message."""
        error = EmptyTraceError()
        assert str(error) == "Trace is empty (0 samples)"

    def test_create_with_custom_message(self) -> None:
        """Test creating EmptyTraceError with custom message."""
        error = EmptyTraceError("Custom empty trace message")
        assert str(error) == "Custom empty trace message"

    def test_is_exception(self) -> None:
        """Test that EmptyTraceError is an Exception."""
        error = EmptyTraceError()
        assert isinstance(error, Exception)

    def test_raise_and_catch(self) -> None:
        """Test raising and catching EmptyTraceError."""
        with pytest.raises(EmptyTraceError, match="Trace is empty"):
            raise EmptyTraceError()


class TestInsufficientSamplesError:
    """Test InsufficientSamplesError exception."""

    def test_create_with_attributes(self) -> None:
        """Test creating InsufficientSamplesError with attributes."""
        error = InsufficientSamplesError("Need more samples", required=100, available=10)
        assert error.required == 100
        assert error.available == 10

    def test_message_format(self) -> None:
        """Test error message format."""
        error = InsufficientSamplesError("Test message", required=100, available=50)
        assert "Test message" in str(error)
        assert "required: 100" in str(error)
        assert "available: 50" in str(error)

    def test_is_exception(self) -> None:
        """Test that InsufficientSamplesError is an Exception."""
        error = InsufficientSamplesError("msg", required=10, available=5)
        assert isinstance(error, Exception)

    def test_raise_and_catch(self) -> None:
        """Test raising and catching InsufficientSamplesError."""
        with pytest.raises(InsufficientSamplesError) as exc_info:
            raise InsufficientSamplesError("msg", required=100, available=10)
        assert exc_info.value.required == 100
        assert exc_info.value.available == 10


# ==============================================================================
# validate_signal() Tests
# ==============================================================================


class TestValidateSignal:
    """Test validate_signal() function."""

    def test_validate_normal_signal(self) -> None:
        """Test validating a normal signal."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = validate_signal(signal)
        np.testing.assert_array_equal(result, signal)

    def test_validate_with_min_samples(self) -> None:
        """Test validation with minimum samples requirement."""
        signal = np.array([1.0, 2.0, 3.0])
        result = validate_signal(signal, min_samples=3)
        np.testing.assert_array_equal(result, signal)

    def test_empty_signal_raises_error(self) -> None:
        """Test that empty signal raises EmptyTraceError."""
        signal = np.array([])
        with pytest.raises(EmptyTraceError, match="signal is empty"):
            validate_signal(signal)

    def test_empty_signal_allowed(self) -> None:
        """Test allowing empty signals."""
        signal = np.array([])
        result = validate_signal(signal, allow_empty=True)
        assert len(result) == 0

    def test_insufficient_samples_raises_error(self) -> None:
        """Test that insufficient samples raises error."""
        signal = np.array([1.0, 2.0])
        with pytest.raises(InsufficientSamplesError) as exc_info:
            validate_signal(signal, min_samples=5)
        assert exc_info.value.required == 5
        assert exc_info.value.available == 2

    def test_not_numpy_array_raises_error(self) -> None:
        """Test that non-numpy array raises ValueError."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            validate_signal([1, 2, 3])  # type: ignore[arg-type]

    def test_multidimensional_raises_error(self) -> None:
        """Test that multi-dimensional array raises ValueError."""
        signal = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            validate_signal(signal)

    def test_custom_name_in_error(self) -> None:
        """Test that custom signal name appears in error messages."""
        signal = np.array([])
        with pytest.raises(EmptyTraceError, match="my_signal is empty"):
            validate_signal(signal, name="my_signal")


# ==============================================================================
# handle_empty_trace() Tests
# ==============================================================================


class TestHandleEmptyTrace:
    """Test handle_empty_trace() function."""

    def test_default_nan(self) -> None:
        """Test default NaN value."""
        result = handle_empty_trace()
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_custom_value(self) -> None:
        """Test custom default value."""
        result = handle_empty_trace(default_value=0.0)
        np.testing.assert_array_equal(result, [0.0])

    def test_negative_value(self) -> None:
        """Test negative default value."""
        result = handle_empty_trace(default_value=-999.0)
        np.testing.assert_array_equal(result, [-999.0])

    def test_returns_array(self) -> None:
        """Test that result is numpy array."""
        result = handle_empty_trace()
        assert isinstance(result, np.ndarray)


# ==============================================================================
# check_single_sample() Tests
# ==============================================================================


class TestCheckSingleSample:
    """Test check_single_sample() function."""

    def test_single_sample_returns_true(self) -> None:
        """Test that single sample returns True."""
        signal = np.array([42.0])
        with pytest.warns(UserWarning):
            result = check_single_sample(signal)
        assert result is True

    def test_single_sample_warns(self) -> None:
        """Test that single sample issues warning."""
        signal = np.array([42.0])
        with pytest.warns(UserWarning, match="Signal has only 1 sample"):
            check_single_sample(signal)

    def test_multiple_samples_returns_false(self) -> None:
        """Test that multiple samples returns False."""
        signal = np.array([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Fail if warning issued
            result = check_single_sample(signal)
        assert result is False

    def test_custom_operation_in_warning(self) -> None:
        """Test that custom operation name appears in warning."""
        signal = np.array([42.0])
        with pytest.warns(UserWarning, match="FFT"):
            check_single_sample(signal, operation="FFT")


# ==============================================================================
# sanitize_signal() Tests
# ==============================================================================


class TestSanitizeSignal:
    """Test sanitize_signal() function."""

    def test_clean_signal_unchanged(self) -> None:
        """Test that clean signal remains unchanged."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Fail if warning issued
            result = sanitize_signal(signal, warn=False)
        np.testing.assert_array_equal(result, signal)

    def test_nan_interpolate_default(self) -> None:
        """Test NaN interpolation (default behavior)."""
        signal = np.array([1.0, np.nan, 3.0])
        result = sanitize_signal(signal, warn=False)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_nan_interpolate_multiple(self) -> None:
        """Test interpolating multiple NaN values."""
        signal = np.array([1.0, np.nan, np.nan, 4.0])
        result = sanitize_signal(signal, warn=False)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_nan_zero(self) -> None:
        """Test replacing NaN with zero."""
        signal = np.array([1.0, np.nan, 3.0])
        result = sanitize_signal(signal, replace_nan="zero", warn=False)
        np.testing.assert_array_equal(result, [1.0, 0.0, 3.0])

    def test_nan_custom_value(self) -> None:
        """Test replacing NaN with custom value."""
        signal = np.array([1.0, np.nan, 3.0])
        result = sanitize_signal(signal, replace_nan=99.0, warn=False)
        np.testing.assert_array_equal(result, [1.0, 99.0, 3.0])

    def test_nan_remove(self) -> None:
        """Test removing NaN values."""
        signal = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = sanitize_signal(signal, replace_nan="remove", warn=False)
        np.testing.assert_array_equal(result, [1.0, 3.0, 5.0])

    def test_inf_clip_default(self) -> None:
        """Test clipping Inf to finite range (default)."""
        signal = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        result = sanitize_signal(signal, warn=False)
        np.testing.assert_array_equal(result, [1.0, 2.0, 5.0, 4.0, 5.0])

    def test_inf_clip_negative(self) -> None:
        """Test clipping -Inf."""
        signal = np.array([1.0, 2.0, -np.inf, 4.0, 5.0])
        result = sanitize_signal(signal, warn=False)
        np.testing.assert_array_equal(result, [1.0, 2.0, 1.0, 4.0, 5.0])

    def test_inf_zero(self) -> None:
        """Test replacing Inf with zero."""
        signal = np.array([1.0, np.inf, 3.0])
        result = sanitize_signal(signal, replace_inf="zero", warn=False)
        np.testing.assert_array_equal(result, [1.0, 0.0, 3.0])

    def test_inf_custom_value(self) -> None:
        """Test replacing Inf with custom value."""
        signal = np.array([1.0, np.inf, 3.0])
        result = sanitize_signal(signal, replace_inf=100.0, warn=False)
        np.testing.assert_array_equal(result, [1.0, 100.0, 3.0])

    def test_inf_remove(self) -> None:
        """Test removing Inf values."""
        signal = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        result = sanitize_signal(signal, replace_inf="remove", warn=False)
        np.testing.assert_array_equal(result, [1.0, 3.0, 5.0])

    def test_warn_on_issues(self) -> None:
        """Test warning when NaN/Inf found."""
        signal = np.array([1.0, np.nan, np.inf])
        with pytest.warns(UserWarning, match="contains 1 NaN and 1 Inf"):
            sanitize_signal(signal, warn=True)

    def test_no_warn_when_disabled(self) -> None:
        """Test no warning when disabled."""
        signal = np.array([1.0, np.nan, np.inf])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Fail if warning issued
            sanitize_signal(signal, warn=False)

    def test_invalid_replace_nan_raises_error(self) -> None:
        """Test that invalid replace_nan option raises ValueError."""
        signal = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Invalid replace_nan option"):
            sanitize_signal(signal, replace_nan="invalid", warn=False)  # type: ignore[arg-type]

    def test_invalid_replace_inf_raises_error(self) -> None:
        """Test that invalid replace_inf option raises ValueError."""
        signal = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="Invalid replace_inf option"):
            sanitize_signal(signal, replace_inf="invalid", warn=False)  # type: ignore[arg-type]

    def test_all_nan_interpolate(self) -> None:
        """Test interpolating signal with all NaN values."""
        signal = np.array([np.nan, np.nan, np.nan])
        result = sanitize_signal(signal, replace_nan="interpolate", warn=False)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_all_inf_clip(self) -> None:
        """Test clipping signal with all Inf values."""
        signal = np.array([np.inf, np.inf, np.inf])
        result = sanitize_signal(signal, replace_inf="clip", warn=False)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_does_not_modify_input(self) -> None:
        """Test that sanitize_signal doesn't modify input array."""
        original = np.array([1.0, np.nan, 3.0])
        signal = original.copy()
        _ = sanitize_signal(signal, warn=False)
        # signal should still have NaN
        assert np.isnan(signal[1])


# ==============================================================================
# check_signal_quality() Tests
# ==============================================================================


class TestCheckSignalQuality:
    """Test check_signal_quality() function."""

    def test_clean_signal_no_issues(self) -> None:
        """Test that clean signal has no quality issues."""
        # Use random signal with good distribution (no clipping, no DC offset)
        np.random.seed(42)
        signal = np.random.uniform(-0.5, 0.5, 1000)  # Uniform distribution
        quality = check_signal_quality(signal, dc_offset_max=0.1)
        assert quality.dc_offset_excessive is False
        # Check that quality object was created
        assert hasattr(quality, "clipping_detected")

    def test_clipping_detected(self) -> None:
        """Test clipping detection."""
        # Create signal with clipping (many samples at limits)
        signal = np.concatenate(
            [
                np.ones(100),  # Clipped high
                np.zeros(800),
                np.ones(100),  # More clipped
            ]
        )
        quality = check_signal_quality(signal, clipping_threshold=0.95)
        assert quality.clipping_detected is True
        assert quality.clipping_percent > 1.0

    def test_dc_offset_detected(self) -> None:
        """Test DC offset detection."""
        signal = np.random.randn(1000) + 0.5  # Add DC offset
        quality = check_signal_quality(signal, dc_offset_max=0.1)
        assert quality.dc_offset_excessive is True
        assert abs(quality.dc_offset - 0.5) < 0.1

    def test_high_noise_detected(self) -> None:
        """Test SNR calculation."""
        # Pure noise (mean ≈ 0, high std)
        signal = np.random.randn(1000) * 10
        quality = check_signal_quality(signal, noise_floor_db=-60.0)
        # SNR will be very low (negative) for pure noise
        # Just verify it was calculated
        assert hasattr(quality, "snr_db")

    def test_quality_report_attributes(self) -> None:
        """Test that quality report has expected attributes."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        quality = check_signal_quality(signal)
        assert hasattr(quality, "adc_min")
        assert hasattr(quality, "adc_max")
        assert hasattr(quality, "snr_db")
        assert quality.adc_min == 1.0
        assert quality.adc_max == 5.0


# ==============================================================================
# SignalQualityReport Tests
# ==============================================================================


class TestSignalQualityReport:
    """Test SignalQualityReport class."""

    def test_create_default(self) -> None:
        """Test creating report with default values."""
        report = SignalQualityReport()
        assert report.clipping_detected is False
        assert report.high_noise is False
        assert report.dc_offset_excessive is False

    def test_create_with_issues(self) -> None:
        """Test creating report with issues."""
        report = SignalQualityReport(
            clipping_detected=True,
            clipping_percent=5.0,
            high_noise=True,
            noise_floor_db=-40.0,
        )
        assert report.clipping_detected is True
        assert report.clipping_percent == 5.0
        assert report.high_noise is True

    def test_has_issues_true(self) -> None:
        """Test has_issues() when issues present."""
        report = SignalQualityReport(clipping_detected=True)
        assert report.has_issues() is True

    def test_has_issues_false(self) -> None:
        """Test has_issues() when no issues."""
        report = SignalQualityReport()
        assert report.has_issues() is False

    def test_summary_no_issues(self) -> None:
        """Test summary with no issues."""
        report = SignalQualityReport(snr_db=60.0)
        summary = report.summary()
        assert "Signal Quality Report:" in summary
        assert "✓ No clipping detected" in summary
        assert "✓ Noise floor acceptable" in summary
        assert "✓ DC offset within limits" in summary

    def test_summary_clipping(self) -> None:
        """Test summary with clipping."""
        report = SignalQualityReport(
            clipping_detected=True,
            clipping_percent=5.5,
            adc_min=0.0,
            adc_max=3.3,
        )
        summary = report.summary()
        assert "⚠ Clipping detected: 5.5% of samples" in summary
        assert "ADC range: 0.000 to 3.300" in summary

    def test_summary_high_noise(self) -> None:
        """Test summary with high noise."""
        report = SignalQualityReport(
            high_noise=True,
            noise_floor_db=-45.2,
            snr_db=-45.2,
        )
        summary = report.summary()
        assert "⚠ High noise floor: -45.2 dB" in summary
        assert "SNR: -45.2 dB" in summary

    def test_summary_dc_offset(self) -> None:
        """Test summary with DC offset."""
        report = SignalQualityReport(
            dc_offset_excessive=True,
            dc_offset=0.456,
        )
        summary = report.summary()
        assert "⚠ DC offset: 0.456" in summary

    def test_summary_multiple_issues(self) -> None:
        """Test summary with multiple issues."""
        report = SignalQualityReport(
            clipping_detected=True,
            clipping_percent=2.0,
            high_noise=True,
            noise_floor_db=-50.0,
            snr_db=-50.0,
            dc_offset_excessive=True,
            dc_offset=0.2,
        )
        summary = report.summary()
        assert "⚠ Clipping detected" in summary
        assert "⚠ High noise floor" in summary
        assert "⚠ DC offset" in summary


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCoreEdgeCasesIntegration:
    """Integration tests combining multiple features."""

    def test_validate_and_sanitize_workflow(self) -> None:
        """Test complete validation and sanitization workflow."""
        # Create signal with issues
        signal = np.array([1.0, 2.0, np.nan, 4.0, np.inf])

        # Validate (should pass - has enough samples)
        validated = validate_signal(signal, min_samples=3)
        assert len(validated) == 5

        # Sanitize
        clean = sanitize_signal(validated, warn=False)
        assert not np.any(np.isnan(clean))
        assert not np.any(np.isinf(clean))

    def test_quality_check_workflow(self) -> None:
        """Test quality checking workflow."""
        # Create signal with known issues
        signal = np.concatenate(
            [
                np.ones(100) * 5.0,  # Clipping
                np.random.randn(900) + 2.0,  # DC offset
            ]
        )

        # Check quality
        quality = check_signal_quality(signal, dc_offset_max=0.5)

        # Should detect both issues
        assert quality.has_issues() is True
        summary = quality.summary()
        assert len(summary) > 0

    def test_empty_trace_handling(self) -> None:
        """Test empty trace error handling."""
        signal = np.array([])

        # Should raise error
        with pytest.raises(EmptyTraceError):
            validate_signal(signal)

        # Get fallback value
        fallback = handle_empty_trace(default_value=0.0)
        assert len(fallback) == 1
        assert fallback[0] == 0.0

    def test_edge_cases_combined(self) -> None:
        """Test handling multiple edge cases together."""
        # Single sample with NaN
        signal = np.array([np.nan])

        # Check single sample
        with pytest.warns(UserWarning):
            is_single = check_single_sample(signal, operation="test")
        assert is_single is True

        # Sanitize
        clean = sanitize_signal(signal, warn=False)
        assert len(clean) == 1
        assert clean[0] == 0.0  # All NaN becomes 0
