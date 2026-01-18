"""Comprehensive unit tests for oscura.compliance.testing module.

Tests EMC compliance testing implementation including DetectorType,
ComplianceViolation, ComplianceResult, and check_compliance function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.compliance.testing import (
    ComplianceResult,
    ComplianceViolation,
    DetectorType,
    check_compliance,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


# =============================================================================
# Mock LimitMask for Testing
# =============================================================================


@dataclass
class MockLimitMask:
    """Mock limit mask for testing."""

    name: str = "TestMask"
    frequency: NDArray[np.float64] = None
    limit: NDArray[np.float64] = None
    description: str = "Test limit mask"
    unit: str = "dBuV"
    standard: str = "TEST"
    distance: float = 3.0  # meters
    detector: str = "peak"
    regulatory_body: str = "TEST"
    document: str = "TEST-DOC"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.frequency is None:
            self.frequency = np.array([1e6, 10e6, 100e6, 1e9])
        if self.limit is None:
            self.limit = np.array([60.0, 50.0, 40.0, 30.0])

    def get_limit_at_frequency(self, freq: float) -> float:
        """Interpolate limit at given frequency."""
        return float(np.interp(freq, self.frequency, self.limit))

    def interpolate(self, frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate limit values at given frequencies."""
        return np.interp(frequencies, self.frequency, self.limit)

    @property
    def frequency_range(self) -> tuple[float, float]:
        """Return (min, max) frequency range."""
        return (float(self.frequency.min()), float(self.frequency.max()))


# =============================================================================
# DetectorType Tests
# =============================================================================


class TestDetectorType:
    """Tests for DetectorType enum."""

    def test_peak_detector(self):
        """Test PEAK detector type."""
        assert DetectorType.PEAK.value == "peak"

    def test_quasi_peak_detector(self):
        """Test QUASI_PEAK detector type."""
        assert DetectorType.QUASI_PEAK.value == "quasi-peak"

    def test_average_detector(self):
        """Test AVERAGE detector type."""
        assert DetectorType.AVERAGE.value == "average"

    def test_rms_detector(self):
        """Test RMS detector type."""
        assert DetectorType.RMS.value == "rms"

    def test_all_detector_types_defined(self):
        """Test all expected detector types are defined."""
        expected = {"PEAK", "QUASI_PEAK", "AVERAGE", "RMS"}
        actual = {d.name for d in DetectorType}
        assert expected == actual


# =============================================================================
# ComplianceViolation Tests
# =============================================================================


class TestComplianceViolation:
    """Tests for ComplianceViolation dataclass."""

    def test_basic_creation(self):
        """Test basic violation creation."""
        violation = ComplianceViolation(
            frequency=100e6,
            measured_level=55.0,
            limit_level=50.0,
            excess_db=5.0,
        )

        assert violation.frequency == 100e6
        assert violation.measured_level == 55.0
        assert violation.limit_level == 50.0
        assert violation.excess_db == 5.0

    def test_default_values(self):
        """Test default values are applied."""
        violation = ComplianceViolation(
            frequency=50e6,
            measured_level=60.0,
            limit_level=55.0,
            excess_db=5.0,
        )

        assert violation.detector == "peak"
        assert violation.severity == "FAIL"

    def test_custom_detector_and_severity(self):
        """Test custom detector and severity."""
        violation = ComplianceViolation(
            frequency=1e6,
            measured_level=40.0,
            limit_level=35.0,
            excess_db=5.0,
            detector="quasi-peak",
            severity="WARNING",
        )

        assert violation.detector == "quasi-peak"
        assert violation.severity == "WARNING"

    def test_str_representation(self):
        """Test string representation."""
        violation = ComplianceViolation(
            frequency=100e6,
            measured_level=55.0,
            limit_level=50.0,
            excess_db=5.0,
        )

        text = str(violation)

        assert "100.000 MHz" in text
        assert "55.0 dB" in text
        assert "50.0 dB" in text
        assert "5.0 dB" in text

    def test_str_low_frequency(self):
        """Test string representation at low frequency."""
        violation = ComplianceViolation(
            frequency=1e6,
            measured_level=70.0,
            limit_level=60.0,
            excess_db=10.0,
        )

        text = str(violation)
        assert "1.000 MHz" in text

    def test_str_high_frequency(self):
        """Test string representation at high frequency."""
        violation = ComplianceViolation(
            frequency=1e9,
            measured_level=25.0,
            limit_level=20.0,
            excess_db=5.0,
        )

        text = str(violation)
        assert "1000.000 MHz" in text


# =============================================================================
# ComplianceResult Tests
# =============================================================================


class TestComplianceResult:
    """Tests for ComplianceResult dataclass."""

    @pytest.fixture
    def passing_result(self) -> ComplianceResult:
        """Create a passing compliance result."""
        freq = np.array([1e6, 10e6, 100e6])
        level = np.array([40.0, 35.0, 30.0])
        limit = np.array([60.0, 50.0, 40.0])

        return ComplianceResult(
            status="PASS",
            mask_name="TestMask",
            violations=[],
            margin_to_limit=10.0,
            worst_frequency=100e6,
            worst_margin=10.0,
            spectrum_freq=freq,
            spectrum_level=level,
            limit_level=limit,
        )

    @pytest.fixture
    def failing_result(self) -> ComplianceResult:
        """Create a failing compliance result."""
        freq = np.array([1e6, 10e6, 100e6])
        level = np.array([65.0, 55.0, 45.0])
        limit = np.array([60.0, 50.0, 40.0])

        violations = [
            ComplianceViolation(
                frequency=1e6,
                measured_level=65.0,
                limit_level=60.0,
                excess_db=5.0,
            ),
            ComplianceViolation(
                frequency=10e6,
                measured_level=55.0,
                limit_level=50.0,
                excess_db=5.0,
            ),
            ComplianceViolation(
                frequency=100e6,
                measured_level=45.0,
                limit_level=40.0,
                excess_db=5.0,
            ),
        ]

        return ComplianceResult(
            status="FAIL",
            mask_name="TestMask",
            violations=violations,
            margin_to_limit=-5.0,
            worst_frequency=1e6,
            worst_margin=-5.0,
            spectrum_freq=freq,
            spectrum_level=level,
            limit_level=limit,
        )

    def test_passing_result_passed_property(self, passing_result):
        """Test passed property for passing result."""
        assert passing_result.passed is True

    def test_failing_result_passed_property(self, failing_result):
        """Test passed property for failing result."""
        assert failing_result.passed is False

    def test_violation_count_zero(self, passing_result):
        """Test violation_count for passing result."""
        assert passing_result.violation_count == 0

    def test_violation_count_multiple(self, failing_result):
        """Test violation_count for failing result."""
        assert failing_result.violation_count == 3

    def test_summary_passing(self, passing_result):
        """Test summary for passing result."""
        summary = passing_result.summary()

        assert "PASS" in summary
        assert "TestMask" in summary
        assert "10.0 dB" in summary

    def test_summary_failing(self, failing_result):
        """Test summary for failing result."""
        summary = failing_result.summary()

        assert "FAIL" in summary
        assert "Violations (3)" in summary
        assert "-5.0 dB" in summary

    def test_summary_truncates_violations(self):
        """Test summary truncates long violation lists."""
        freq = np.array([1e6])
        level = np.array([70.0])
        limit = np.array([60.0])

        # Create 15 violations
        violations = [
            ComplianceViolation(
                frequency=float(i) * 1e6,
                measured_level=70.0,
                limit_level=60.0,
                excess_db=10.0,
            )
            for i in range(1, 16)
        ]

        result = ComplianceResult(
            status="FAIL",
            mask_name="TestMask",
            violations=violations,
            margin_to_limit=-10.0,
            worst_frequency=1e6,
            worst_margin=-10.0,
            spectrum_freq=freq,
            spectrum_level=level,
            limit_level=limit,
        )

        summary = result.summary()

        # Should show first 10 and "and X more"
        assert "... and 5 more" in summary

    def test_default_detector(self, passing_result):
        """Test default detector value."""
        assert passing_result.detector == "peak"

    def test_custom_detector(self):
        """Test custom detector value."""
        freq = np.array([1e6])
        level = np.array([40.0])
        limit = np.array([60.0])

        result = ComplianceResult(
            status="PASS",
            mask_name="Test",
            violations=[],
            margin_to_limit=20.0,
            worst_frequency=1e6,
            worst_margin=20.0,
            spectrum_freq=freq,
            spectrum_level=level,
            limit_level=limit,
            detector="quasi-peak",
        )

        assert result.detector == "quasi-peak"

    def test_metadata_default(self, passing_result):
        """Test metadata default is empty dict."""
        assert passing_result.metadata == {}

    def test_metadata_custom(self):
        """Test custom metadata."""
        freq = np.array([1e6])
        level = np.array([40.0])
        limit = np.array([60.0])

        result = ComplianceResult(
            status="PASS",
            mask_name="Test",
            violations=[],
            margin_to_limit=20.0,
            worst_frequency=1e6,
            worst_margin=20.0,
            spectrum_freq=freq,
            spectrum_level=level,
            limit_level=limit,
            metadata={"test_date": "2025-01-01", "sample_id": "A001"},
        )

        assert result.metadata["test_date"] == "2025-01-01"
        assert result.metadata["sample_id"] == "A001"


# =============================================================================
# check_compliance Tests
# =============================================================================


class TestCheckCompliance:
    """Tests for check_compliance function."""

    @pytest.fixture
    def sample_trace(self) -> WaveformTrace:
        """Create sample trace for testing."""
        # Generate a signal with some frequency content
        sample_rate = 1e9  # 1 GHz
        duration = 1e-3  # 1 ms
        n_samples = int(sample_rate * duration)
        t = np.arange(n_samples) / sample_rate

        # Signal with 10 MHz and 50 MHz components
        signal = np.sin(2 * np.pi * 10e6 * t) + 0.5 * np.sin(2 * np.pi * 50e6 * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="CH1",
        )

        return WaveformTrace(data=signal, metadata=metadata)

    @pytest.fixture
    def sample_spectrum(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create sample spectrum for testing."""
        # Pre-computed spectrum
        freq = np.logspace(6, 9, 100)  # 1 MHz to 1 GHz
        level = 50 - 10 * np.log10(freq / 1e6)  # Decreasing with frequency

        return freq, level

    @pytest.fixture
    def test_mask(self) -> MockLimitMask:
        """Create test limit mask."""
        return MockLimitMask()

    def test_check_compliance_with_spectrum(self, sample_spectrum, test_mask):
        """Test check_compliance with pre-computed spectrum."""
        freq, level = sample_spectrum

        result = check_compliance(sample_spectrum, test_mask)

        assert isinstance(result, ComplianceResult)
        assert result.mask_name == test_mask.name

    def test_check_compliance_with_trace(self, sample_trace, test_mask):
        """Test check_compliance with WaveformTrace."""
        result = check_compliance(sample_trace, test_mask)

        assert isinstance(result, ComplianceResult)

    def test_check_compliance_peak_detector(self, sample_spectrum, test_mask):
        """Test check_compliance with peak detector."""
        result = check_compliance(
            sample_spectrum,
            test_mask,
            detector=DetectorType.PEAK,
        )

        assert result.detector == "peak"

    def test_check_compliance_average_detector(self, sample_spectrum, test_mask):
        """Test check_compliance with average detector."""
        result = check_compliance(
            sample_spectrum,
            test_mask,
            detector=DetectorType.AVERAGE,
        )

        assert result.detector == "average"

    def test_check_compliance_string_detector(self, sample_spectrum, test_mask):
        """Test check_compliance with string detector type."""
        result = check_compliance(
            sample_spectrum,
            test_mask,
            detector="rms",
        )

        assert result.detector == "rms"

    def test_check_compliance_frequency_range(self, sample_spectrum, test_mask):
        """Test check_compliance with frequency range filter."""
        result = check_compliance(
            sample_spectrum,
            test_mask,
            frequency_range=(10e6, 100e6),
        )

        assert isinstance(result, ComplianceResult)

    def test_check_compliance_passing(self, test_mask):
        """Test check_compliance with passing signal."""
        # Low level signal that passes
        freq = np.array([1e6, 10e6, 100e6])
        level = np.array([40.0, 30.0, 20.0])  # Well below limits

        result = check_compliance((freq, level), test_mask)

        assert result.status == "PASS"
        assert result.margin_to_limit > 0

    def test_check_compliance_failing(self, test_mask):
        """Test check_compliance with failing signal."""
        # High level signal that fails
        freq = np.array([1e6, 10e6, 100e6])
        level = np.array([70.0, 60.0, 50.0])  # Above limits

        result = check_compliance((freq, level), test_mask)

        assert result.status == "FAIL"
        assert len(result.violations) > 0
        assert result.margin_to_limit < 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestComplianceTestingIntegration:
    """Integration tests for compliance testing."""

    def test_full_compliance_workflow(self):
        """Test complete compliance testing workflow."""
        # Create test signal
        sample_rate = 1e9
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate

        signal = 0.1 * np.sin(2 * np.pi * 30e6 * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="DUT",
        )

        trace = WaveformTrace(data=signal, metadata=metadata)

        # Create mask
        mask = MockLimitMask(
            name="FCC_ClassB",
            frequency=np.array([30e6, 88e6, 216e6, 1e9]),
            limit=np.array([40.0, 43.5, 46.0, 54.0]),
        )

        # Run compliance check
        result = check_compliance(trace, mask)

        # Verify result structure
        assert isinstance(result, ComplianceResult)
        assert result.mask_name == "FCC_ClassB"
        assert len(result.spectrum_freq) > 0
        assert len(result.spectrum_level) > 0
        assert len(result.limit_level) > 0

    def test_multiple_detectors_comparison(self):
        """Test comparing different detector types."""
        freq = np.array([1e6, 10e6, 100e6])
        level = np.array([55.0, 45.0, 35.0])
        mask = MockLimitMask()

        results = {}
        for detector in DetectorType:
            result = check_compliance((freq, level), mask, detector=detector)
            results[detector.value] = result

        # All results should be valid
        for detector_name, result in results.items():
            assert isinstance(result, ComplianceResult)
            assert result.detector == detector_name


# =============================================================================
# Edge Cases
# =============================================================================


class TestComplianceTestingEdgeCases:
    """Tests for edge cases."""

    def test_single_frequency_point(self):
        """Test with single frequency point."""
        freq = np.array([100e6])
        level = np.array([45.0])
        mask = MockLimitMask()

        result = check_compliance((freq, level), mask)

        assert isinstance(result, ComplianceResult)

    def test_exact_limit_match(self):
        """Test when measured equals limit exactly."""
        freq = np.array([100e6])
        # At 100 MHz, interpolated limit is around 40 dB
        level = np.array([40.0])
        mask = MockLimitMask()

        result = check_compliance((freq, level), mask)

        assert isinstance(result, ComplianceResult)
        # Exactly at limit - could be pass or fail depending on implementation

    def test_very_low_level(self):
        """Test with very low signal level."""
        freq = np.array([1e6, 100e6, 1e9])
        level = np.array([-10.0, -20.0, -30.0])  # Very low levels
        mask = MockLimitMask()

        result = check_compliance((freq, level), mask)

        assert result.status == "PASS"
        assert result.margin_to_limit > 30

    def test_very_high_level(self):
        """Test with very high signal level."""
        freq = np.array([1e6, 100e6, 1e9])
        level = np.array([100.0, 90.0, 80.0])  # Very high levels
        mask = MockLimitMask()

        result = check_compliance((freq, level), mask)

        assert result.status == "FAIL"
        assert len(result.violations) > 0

    def test_empty_frequency_range(self):
        """Test with frequency range that excludes all points."""
        freq = np.array([1e6, 10e6, 100e6])
        level = np.array([50.0, 40.0, 30.0])
        mask = MockLimitMask()

        result = check_compliance(
            (freq, level),
            mask,
            frequency_range=(500e6, 600e6),  # No data in this range
        )

        assert isinstance(result, ComplianceResult)

    def test_wide_frequency_span(self):
        """Test with very wide frequency span."""
        freq = np.logspace(3, 12, 1000)  # 1 kHz to 1 THz
        level = 60 - 10 * np.log10(freq / 1e6)
        mask = MockLimitMask(
            frequency=np.array([1e6, 1e9, 1e12]),
            limit=np.array([60.0, 30.0, 0.0]),
        )

        result = check_compliance((freq, level), mask)

        assert isinstance(result, ComplianceResult)
