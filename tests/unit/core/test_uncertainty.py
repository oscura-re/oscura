"""Tests for measurement uncertainty estimation.

Validates uncertainty propagation and GUM compliance.

References:
    JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement
"""

import numpy as np
import pytest

from oscura.core.uncertainty import MeasurementWithUncertainty, UncertaintyEstimator

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestMeasurementResult:
    """Test MeasurementResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic measurement result."""
        result = MeasurementWithUncertainty(value=1.234, uncertainty=0.005)
        assert result.value == 1.234
        assert result.uncertainty == 0.005
        assert result.unit is None
        assert result.coverage_factor == 1.0

    def test_with_unit(self):
        """Test measurement with unit."""
        result = MeasurementWithUncertainty(value=10.0, uncertainty=0.1, unit="V")
        assert result.unit == "V"
        assert "V" in str(result)

    def test_expanded_uncertainty(self):
        """Test expanded uncertainty calculation (k=2)."""
        result = MeasurementWithUncertainty(
            value=100.0, uncertainty=1.0, coverage_factor=2.0, confidence_level=0.9545
        )
        assert result.expanded_uncertainty == 2.0
        assert result.uncertainty == 1.0  # Standard uncertainty unchanged

    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        result = MeasurementWithUncertainty(value=100.0, uncertainty=2.0)
        assert result.relative_uncertainty == 0.02  # 2%

    def test_relative_uncertainty_zero_value(self):
        """Test relative uncertainty when value is zero."""
        result = MeasurementWithUncertainty(value=0.0, uncertainty=0.001)
        assert result.relative_uncertainty == np.inf

    def test_bounds(self):
        """Test uncertainty interval bounds."""
        result = MeasurementWithUncertainty(value=10.0, uncertainty=0.5)
        assert result.lower_bound == 9.5
        assert result.upper_bound == 10.5

    def test_bounds_with_coverage_factor(self):
        """Test bounds with expanded uncertainty (k=2)."""
        result = MeasurementWithUncertainty(value=10.0, uncertainty=0.5, coverage_factor=2.0)
        assert result.lower_bound == 9.0  # 10.0 - 2*0.5
        assert result.upper_bound == 11.0  # 10.0 + 2*0.5

    def test_string_representation(self):
        """Test string formatting."""
        result = MeasurementWithUncertainty(value=1.234567, uncertainty=0.005, unit="V")
        s = str(result)
        assert "1.234" in s or "1.235" in s  # Value
        assert "0.005" in s  # Uncertainty
        assert "V" in s  # Unit

    def test_validation_negative_uncertainty(self):
        """Test validation rejects negative uncertainty."""
        with pytest.raises(ValueError, match="uncertainty must be non-negative"):
            MeasurementWithUncertainty(value=1.0, uncertainty=-0.1)

    def test_validation_invalid_coverage_factor(self):
        """Test validation rejects invalid coverage factor."""
        with pytest.raises(ValueError, match="coverage_factor must be positive"):
            MeasurementWithUncertainty(value=1.0, uncertainty=0.1, coverage_factor=0.0)

    def test_validation_invalid_confidence(self):
        """Test validation rejects invalid confidence level."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            MeasurementWithUncertainty(value=1.0, uncertainty=0.1, confidence_level=1.5)

    def test_nan_value(self):
        """Test handling of NaN values."""
        result = MeasurementWithUncertainty(value=np.nan, uncertainty=0.1)
        assert np.isnan(result.value)
        assert result.uncertainty == 0.1


class TestUncertaintyEstimator:
    """Test UncertaintyEstimator utility methods."""

    def test_type_a_standard_deviation(self):
        """Test Type A uncertainty from repeated measurements."""
        data = np.array([10.0, 10.1, 9.9, 10.05, 9.95])
        u = UncertaintyEstimator.type_a_standard_deviation(data)
        expected_std = np.std(data, ddof=1)
        assert np.isclose(u, expected_std)

    def test_type_a_standard_error(self):
        """Test Type A uncertainty (standard error of mean)."""
        data = np.array([10.0, 10.1, 9.9, 10.05, 9.95])
        u = UncertaintyEstimator.type_a_standard_error(data)
        expected_se = np.std(data, ddof=1) / np.sqrt(len(data))
        assert np.isclose(u, expected_se)

    def test_type_a_insufficient_data(self):
        """Test Type A with insufficient data returns NaN."""
        data = np.array([10.0])
        u = UncertaintyEstimator.type_a_standard_deviation(data)
        assert np.isnan(u)

    def test_combined_uncertainty_uncorrelated(self):
        """Test combining uncorrelated uncertainties."""
        u1, u2, u3 = 0.01, 0.02, 0.005
        u_combined = UncertaintyEstimator.combined_uncertainty([u1, u2, u3])
        expected = np.sqrt(u1**2 + u2**2 + u3**2)
        assert np.isclose(u_combined, expected)

    def test_combined_uncertainty_correlated(self):
        """Test combining correlated uncertainties."""
        u = [0.01, 0.02]
        # 50% correlation
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        u_combined = UncertaintyEstimator.combined_uncertainty(u, R)

        # Manual calculation: u_c² = u^T R u
        u_array = np.array(u)
        expected = np.sqrt(u_array @ R @ u_array)
        assert np.isclose(u_combined, expected)

    def test_combined_uncertainty_perfect_correlation(self):
        """Test combining perfectly correlated uncertainties."""
        u = [0.01, 0.01]
        R = np.array([[1.0, 1.0], [1.0, 1.0]])  # Perfect correlation
        u_combined = UncertaintyEstimator.combined_uncertainty(u, R)
        # Perfect correlation: u_c = u1 + u2
        expected = 0.02
        assert np.isclose(u_combined, expected)

    def test_type_b_rectangular(self):
        """Test Type B uncertainty from rectangular distribution."""
        half_width = 0.5e-3  # ±0.5 mV
        u = UncertaintyEstimator.type_b_rectangular(half_width)
        expected = half_width / np.sqrt(3)
        assert np.isclose(u, expected)

    def test_type_b_triangular(self):
        """Test Type B uncertainty from triangular distribution."""
        half_width = 1.0e-3  # ±1 mV
        u = UncertaintyEstimator.type_b_triangular(half_width)
        expected = half_width / np.sqrt(6)
        assert np.isclose(u, expected)

    def test_type_b_from_tolerance_95_percent(self):
        """Test Type B from tolerance spec (95% confidence)."""
        tolerance = 0.02  # ±2%
        u = UncertaintyEstimator.type_b_from_tolerance(tolerance, confidence=0.95)
        # For Gaussian, 95% → k=1.96
        expected = tolerance / 1.96
        assert np.isclose(u, expected, rtol=0.01)

    def test_type_b_from_tolerance_99_percent(self):
        """Test Type B from tolerance spec (99% confidence)."""
        tolerance = 0.03  # ±3%
        u = UncertaintyEstimator.type_b_from_tolerance(tolerance, confidence=0.99)
        # For Gaussian, 99% → k=2.58
        expected = tolerance / 2.58
        assert np.isclose(u, expected, rtol=0.01)

    def test_time_base_uncertainty(self):
        """Test time base uncertainty calculation."""
        sample_rate = 1e9  # 1 GSa/s
        timebase_ppm = 25.0
        u_t = UncertaintyEstimator.time_base_uncertainty(sample_rate, timebase_ppm)

        # Expected: (1/1e9) * 25e-6 = 25 ps
        time_per_sample = 1e-9
        expected = time_per_sample * 25e-6
        assert np.isclose(u_t, expected)

    def test_time_base_uncertainty_units(self):
        """Test time base uncertainty has correct units."""
        sample_rate = 100e6  # 100 MSa/s
        timebase_ppm = 50.0
        u_t = UncertaintyEstimator.time_base_uncertainty(sample_rate, timebase_ppm)

        # Should be in seconds
        assert u_t > 0
        assert u_t < 1e-6  # Less than 1 µs for this sample rate

    def test_vertical_uncertainty_basic(self):
        """Test vertical uncertainty calculation."""
        reading = 1.0  # 1 V
        accuracy_pct = 2.0  # ±2%
        offset_error = 0.001  # 1 mV
        u_v = UncertaintyEstimator.vertical_uncertainty(reading, accuracy_pct, offset_error)

        # Gain error: 0.02 V, Offset: 0.001 V
        # Combined: sqrt(0.02^2 + 0.001^2) ≈ 0.02002 V
        expected = np.sqrt((0.02) ** 2 + (0.001) ** 2)
        assert np.isclose(u_v, expected, rtol=0.01)

    def test_vertical_uncertainty_zero_reading(self):
        """Test vertical uncertainty at zero reading (offset only)."""
        reading = 0.0
        accuracy_pct = 2.0
        offset_error = 0.001  # 1 mV
        u_v = UncertaintyEstimator.vertical_uncertainty(reading, accuracy_pct, offset_error)

        # Only offset contributes
        assert np.isclose(u_v, offset_error)

    def test_vertical_uncertainty_negative_reading(self):
        """Test vertical uncertainty with negative reading."""
        reading = -2.0  # -2 V
        accuracy_pct = 3.0  # ±3%
        offset_error = 0.0
        u_v = UncertaintyEstimator.vertical_uncertainty(reading, accuracy_pct, offset_error)

        # Gain error on absolute value
        expected = abs(reading) * 0.03
        assert np.isclose(u_v, expected)


class TestUncertaintyPropagation:
    """Test uncertainty propagation through calculations."""

    def test_sum_propagation(self):
        """Test uncertainty propagation for addition.

        For z = x + y (uncorrelated): u(z) = sqrt(u_x^2 + u_y^2)
        """
        u_x, u_y = 0.1, 0.2
        u_z = UncertaintyEstimator.combined_uncertainty([u_x, u_y])
        expected = np.sqrt(u_x**2 + u_y**2)
        assert np.isclose(u_z, expected)

    def test_product_propagation(self):
        """Test uncertainty propagation for multiplication.

        For z = x * y: u_r(z) = sqrt(u_r(x)^2 + u_r(y)^2)
        where u_r = relative uncertainty
        """
        x, u_x = 10.0, 0.1  # 1% relative
        y, u_y = 5.0, 0.05  # 1% relative

        # Relative uncertainties
        u_r_x = u_x / x
        u_r_y = u_y / y

        # Combined relative uncertainty
        u_r_z = np.sqrt(u_r_x**2 + u_r_y**2)

        # Absolute uncertainty
        z = x * y
        u_z = z * u_r_z

        # Verify using combined uncertainty
        # For product: convert to relative, combine, convert back
        u_z_calculated = z * np.sqrt((u_x / x) ** 2 + (u_y / y) ** 2)
        assert np.isclose(u_z, u_z_calculated)

    def test_division_propagation(self):
        """Test uncertainty propagation for division.

        For z = x / y: u_r(z) = sqrt(u_r(x)^2 + u_r(y)^2)
        """
        x, u_x = 100.0, 1.0  # 1% relative
        y, u_y = 10.0, 0.1  # 1% relative

        z = x / y  # = 10.0
        u_r_z = np.sqrt((u_x / x) ** 2 + (u_y / y) ** 2)
        u_z = z * u_r_z

        # Should be ~1.414% relative, ~0.1414 absolute
        assert np.isclose(u_r_z, np.sqrt(0.01**2 + 0.01**2))
        assert np.isclose(u_z, z * u_r_z)


class TestRealWorldExamples:
    """Test realistic measurement uncertainty scenarios."""

    def test_oscilloscope_voltage_measurement(self):
        """Test uncertainty for oscilloscope voltage measurement.

        Scenario: Tektronix TDS3000 series
        - Vertical accuracy: ±3% of reading
        - Offset accuracy: ±0.05 div * scale
        - 8-bit ADC
        """
        reading = 1.5  # 1.5 V measured
        scale = 0.5  # 500 mV/div

        # Gain uncertainty (±3%)
        u_gain = reading * 0.03

        # Offset uncertainty (±0.05 div × 0.5 V/div = ±25 mV)
        u_offset = 0.05 * scale

        # ADC quantization (8-bit, full scale ~5 V)
        full_scale = scale * 10  # 10 divisions
        lsb = full_scale / 256
        u_quant = UncertaintyEstimator.type_b_rectangular(0.5 * lsb)

        # Combine
        u_total = UncertaintyEstimator.combined_uncertainty([u_gain, u_offset, u_quant])

        # Check order of magnitude
        assert 0.04 < u_total < 0.06  # Should be ~40-60 mV

    def test_frequency_counter_measurement(self):
        """Test uncertainty for frequency counter.

        Scenario: 10 MHz clock, 25 ppm timebase
        """
        frequency = 10e6  # 10 MHz
        timebase_ppm = 25.0

        # Timebase uncertainty
        u_f_timebase = frequency * (timebase_ppm * 1e-6)

        # Quantization (1 Hz resolution typical)
        u_f_quant = UncertaintyEstimator.type_b_rectangular(0.5)

        # Combine
        u_f = UncertaintyEstimator.combined_uncertainty([u_f_timebase, u_f_quant])

        # Check: should be dominated by timebase (~250 Hz)
        assert 200 < u_f < 300

    def test_rise_time_measurement_uncertainty(self):
        """Test uncertainty for rise time measurement.

        Scenario:
        - 1 GSa/s sampling (1 ns per sample)
        - 25 ppm timebase
        - 100 mV amplitude, 2 ns rise time
        - 1 mV RMS noise
        """
        sample_rate = 1e9
        timebase_ppm = 25.0
        rise_time = 2e-9  # 2 ns
        amplitude = 0.1  # 100 mV
        noise_rms = 0.001  # 1 mV

        # Timebase uncertainty (2 edges)
        u_timebase = UncertaintyEstimator.time_base_uncertainty(sample_rate, timebase_ppm)
        u_timebase_rise = u_timebase * np.sqrt(2)

        # Interpolation uncertainty
        sample_period = 1 / sample_rate
        u_interp = UncertaintyEstimator.type_b_rectangular(0.5 * sample_period)

        # Noise-induced uncertainty
        slew_rate = amplitude / rise_time  # V/s
        u_noise = noise_rms / slew_rate

        # Combine
        u_rise = UncertaintyEstimator.combined_uncertainty([u_timebase_rise, u_interp, u_noise])

        # Check order of magnitude (should be tens to hundreds of picoseconds)
        assert 1e-12 < u_rise < 1e-9  # 1 ps to 1 ns (realistic range)
