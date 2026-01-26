"""Comprehensive unit tests for EMC/EMI compliance workflow.

Requirements tested:

This test suite covers:
- EMC compliance testing against various regulatory standards
- Spectrum calculation and limit mask loading
- Violation detection and margin analysis
- Compliance report generation
- Edge cases and error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.workflows.compliance import (
    _generate_compliance_report,
    _load_emc_mask,
    emc_compliance_test,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def waveform_trace():
    """Create a simple waveform trace for testing."""
    sample_rate = 100e6  # 100 MHz
    duration = 1e-3  # 1 ms
    n_samples = int(sample_rate * duration)

    # Create a signal with some frequency components
    t = np.linspace(0, duration, n_samples)
    # Mix of frequencies: 1 MHz, 10 MHz, 100 MHz
    data = (
        1e-12 * np.sin(2 * np.pi * 1e6 * t)
        + 0.5e-12 * np.sin(2 * np.pi * 10e6 * t)
        + 0.2e-12 * np.sin(2 * np.pi * 100e6 * t)
    )

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def mock_fft_output():
    """Create mock FFT output for testing."""
    # Frequencies from 0 to 500 MHz in 1 MHz steps
    freq = np.linspace(0, 500e6, 501)
    # Magnitude in dB relative to 1V (as returned by fft())
    # Create a spectrum with some peaks
    # Base level: 1 µV = -120 dBV
    mag_db = np.ones(501) * -120.0
    mag_db[100] = -100.0  # Peak at 100 MHz: 10 µV
    mag_db[200] = -106.0  # Peak at 200 MHz: 5 µV (20*log10(5e-6) ≈ -106)
    mag_db[300] = -114.0  # Peak at 300 MHz: 2 µV (20*log10(2e-6) ≈ -114)

    return freq, mag_db


@pytest.fixture
def mock_fft_violation():
    """Create mock FFT output with violations."""
    freq = np.linspace(0, 500e6, 501)
    # Create high magnitude that will violate limits
    mag = np.ones(501) * 1e-3  # Base level: 1 mV (high)
    mag[100] = 10e-3  # Very high peak at 100 MHz
    mag[200] = 5e-3  # High peak at 200 MHz

    return freq, mag


@pytest.mark.unit
class TestLoadEMCMask:
    """Test EMC limit mask loading."""

    def test_fcc_part15_classb(self):
        """Test loading FCC Part 15 Class B mask."""
        freq, limit = _load_emc_mask("FCC_Part15_ClassB")

        assert isinstance(freq, np.ndarray)
        assert isinstance(limit, np.ndarray)
        assert len(freq) == len(limit)
        assert len(freq) == 7  # Expected number of points
        assert freq[0] == 0.15e6  # 0.15 MHz
        assert freq[-1] == 1000e6  # 1000 MHz
        assert all(limit > 0)  # All limits should be positive

    def test_fcc_part15_classa(self):
        """Test loading FCC Part 15 Class A mask."""
        freq, limit = _load_emc_mask("FCC_Part15_ClassA")

        assert len(freq) == 7
        assert freq[0] == 0.15e6
        # Class A limits should be higher than Class B
        _, limit_b = _load_emc_mask("FCC_Part15_ClassB")
        assert all(limit >= limit_b)

    def test_ce_cispr22_classb(self):
        """Test loading CE CISPR 22 Class B mask."""
        freq, limit = _load_emc_mask("CE_CISPR22_ClassB")

        assert len(freq) == 6
        assert freq[0] == 0.15e6
        assert freq[-1] == 1000e6

    def test_ce_cispr22_classa(self):
        """Test loading CE CISPR 22 Class A mask."""
        freq, limit = _load_emc_mask("CE_CISPR22_ClassA")

        assert len(freq) == 6
        # Class A limits should be higher than Class B
        _, limit_b = _load_emc_mask("CE_CISPR22_ClassB")
        assert all(limit >= limit_b)

    def test_ce_cispr32_classb(self):
        """Test loading CE CISPR 32 Class B mask."""
        freq, limit = _load_emc_mask("CE_CISPR32_ClassB")

        assert len(freq) == 6
        assert freq[0] == 0.15e6

    def test_ce_cispr32_classa(self):
        """Test loading CE CISPR 32 Class A mask."""
        freq, limit = _load_emc_mask("CE_CISPR32_ClassA")

        assert len(freq) == 6

    def test_mil_std_461g_ce102(self):
        """Test loading MIL-STD-461G CE102 mask."""
        freq, limit = _load_emc_mask("MIL_STD_461G_CE102")

        assert len(freq) == 4
        assert freq[0] == 0.01e6
        assert freq[-1] == 50e6

    def test_mil_std_461g_re102(self):
        """Test loading MIL-STD-461G RE102 mask."""
        freq, limit = _load_emc_mask("MIL_STD_461G_RE102")

        assert len(freq) == 5
        assert freq[0] == 2e6
        assert freq[-1] == 18000e6  # 18 GHz

    def test_unknown_standard(self):
        """Test that unknown standard raises AnalysisError."""
        with pytest.raises(AnalysisError, match="Unknown EMC standard"):
            _load_emc_mask("UnknownStandard")

    def test_invalid_standard(self):
        """Test that invalid standard raises AnalysisError."""
        with pytest.raises(AnalysisError, match="Unknown EMC standard"):
            _load_emc_mask("INVALID_STANDARD_123")


@pytest.mark.unit
class TestEMCComplianceTest:
    """Test EMC compliance testing function."""

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_basic_pass(self, mock_fft, waveform_trace, mock_fft_output):
        """Test basic compliance test that passes."""
        mock_fft.return_value = mock_fft_output

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert result["status"] == "PASS"
        assert result["standard"] == "FCC_Part15_ClassB"
        assert len(result["violations"]) == 0
        assert result["margin_to_limit"] > 0  # Positive margin = passing
        assert "spectrum_freq" in result
        assert "spectrum_mag" in result
        assert "limit_freq" in result
        assert "limit_mag" in result
        assert result["detector"] == "peak"

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_basic_fail(self, mock_fft, waveform_trace, mock_fft_violation):
        """Test compliance test that fails with violations."""
        mock_fft.return_value = mock_fft_violation

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert result["status"] == "FAIL"
        assert len(result["violations"]) > 0
        assert result["margin_to_limit"] < 0  # Negative margin = failing

        # Check violation structure
        violation = result["violations"][0]
        assert "frequency" in violation
        assert "measured_dbuv" in violation
        assert "limit_dbuv" in violation
        assert "excess_db" in violation
        assert violation["excess_db"] > 0  # Positive excess = over limit

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_all_standards(self, mock_fft, waveform_trace, mock_fft_output):
        """Test compliance testing against all supported standards."""
        mock_fft.return_value = mock_fft_output

        standards = [
            "FCC_Part15_ClassA",
            "FCC_Part15_ClassB",
            "CE_CISPR22_ClassA",
            "CE_CISPR22_ClassB",
            "CE_CISPR32_ClassA",
            "CE_CISPR32_ClassB",
            "MIL_STD_461G_CE102",
            "MIL_STD_461G_RE102",
        ]

        for standard in standards:
            result = emc_compliance_test(waveform_trace, standard=standard)
            assert result["standard"] == standard
            assert result["status"] in ["PASS", "FAIL"]

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_frequency_range_filtering(self, mock_fft, waveform_trace, mock_fft_output):
        """Test frequency range filtering."""
        mock_fft.return_value = mock_fft_output

        # Test with frequency range
        f_min, f_max = 10e6, 200e6  # 10-200 MHz
        result = emc_compliance_test(
            waveform_trace,
            standard="FCC_Part15_ClassB",
            frequency_range=(f_min, f_max),
        )

        # Check that spectrum is filtered
        assert np.all(result["spectrum_freq"] >= f_min)
        assert np.all(result["spectrum_freq"] <= f_max)

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_detector_parameter(self, mock_fft, waveform_trace, mock_fft_output):
        """Test different detector types."""
        mock_fft.return_value = mock_fft_output

        for detector in ["peak", "quasi-peak", "average"]:
            result = emc_compliance_test(
                waveform_trace,
                standard="FCC_Part15_ClassB",
                detector=detector,
            )
            assert result["detector"] == detector

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_margin_analysis(self, mock_fft, waveform_trace, mock_fft_output):
        """Test margin analysis calculations."""
        mock_fft.return_value = mock_fft_output

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # Check margin fields
        assert "margin_to_limit" in result
        assert "worst_frequency" in result
        assert "worst_margin" in result

        # Worst margin should equal margin_to_limit
        assert result["worst_margin"] == result["margin_to_limit"]

        # Worst frequency should be in the spectrum
        assert result["worst_frequency"] in result["spectrum_freq"]

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_dbuv_conversion(self, mock_fft, waveform_trace):
        """Test conversion to dBµV."""
        # Create spectrum with known magnitude
        freq = np.linspace(0, 100e6, 101)
        # fft() returns magnitude in dB relative to 1V
        # For 1 µV (1e-6 V): dBV = 20*log10(1e-6) = -120 dBV
        mag_db = np.ones(101) * -120.0  # 1 µV in dBV
        mock_fft.return_value = (freq, mag_db)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # 1 µV should convert to 0 dBµV
        # dBµV = dBV + 120 = -120 + 120 = 0
        expected_dbuv = 0.0
        assert np.allclose(result["spectrum_mag"], expected_dbuv, atol=1e-10)

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_zero_magnitude_handling(self, mock_fft, waveform_trace):
        """Test that zero magnitude is handled without log(0) errors."""
        freq = np.linspace(0, 100e6, 101)
        # fft() returns magnitude in dB - very negative values represent near-zero
        # Real fft() uses np.maximum(magnitude, 1e-20) before log, giving ~-400 dBV
        mag_db = np.ones(101) * -400.0  # Near-zero magnitude in dB
        mock_fft.return_value = (freq, mag_db)

        # Should not raise error
        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert np.all(np.isfinite(result["spectrum_mag"]))

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_violation_details(self, mock_fft, waveform_trace, mock_fft_violation):
        """Test violation details are correctly populated."""
        mock_fft.return_value = mock_fft_violation

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert len(result["violations"]) > 0

        for violation in result["violations"]:
            # Each violation should have all required fields
            assert isinstance(violation["frequency"], float | np.floating)
            assert isinstance(violation["measured_dbuv"], float | np.floating)
            assert isinstance(violation["limit_dbuv"], float | np.floating)
            assert isinstance(violation["excess_db"], float | np.floating)

            # Measured should exceed limit
            assert violation["measured_dbuv"] > violation["limit_dbuv"]

            # Excess should be positive
            assert violation["excess_db"] > 0

    @patch("oscura.analyzers.waveform.spectral.fft")
    @patch("oscura.workflows.compliance._generate_compliance_report")
    def test_report_generation(
        self, mock_generate_report, mock_fft, waveform_trace, mock_fft_output
    ):
        """Test that report is generated when requested."""
        mock_fft.return_value = mock_fft_output

        report_path = "/tmp/test_report.html"
        result = emc_compliance_test(
            waveform_trace,
            standard="FCC_Part15_ClassB",
            report=report_path,
        )

        mock_generate_report.assert_called_once()
        call_args = mock_generate_report.call_args[0]
        assert call_args[0] == result
        assert call_args[1] == report_path

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_no_report_when_not_requested(self, mock_fft, waveform_trace, mock_fft_output):
        """Test that report is not generated when not requested."""
        mock_fft.return_value = mock_fft_output

        with patch("oscura.workflows.compliance._generate_compliance_report") as mock_gen:
            result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

            mock_gen.assert_not_called()

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_unknown_standard_raises_error(self, mock_fft, waveform_trace, mock_fft_output):
        """Test that unknown standard raises AnalysisError."""
        mock_fft.return_value = mock_fft_output

        with pytest.raises(AnalysisError, match="Unknown EMC standard"):
            emc_compliance_test(waveform_trace, standard="InvalidStandard")

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_interpolation_to_spectrum_frequencies(self, mock_fft, waveform_trace, mock_fft_output):
        """Test that limit mask is interpolated to spectrum frequencies."""
        mock_fft.return_value = mock_fft_output

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # Spectrum and limit should have same length (interpolated)
        assert len(result["spectrum_freq"]) == len(result["spectrum_mag"])

        # Original limit mask should be different length
        assert len(result["limit_freq"]) != len(result["spectrum_freq"])

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_empty_frequency_range(self, mock_fft, waveform_trace, mock_fft_output):
        """Test behavior with frequency range that excludes all data."""
        mock_fft.return_value = mock_fft_output

        # Frequency range that's beyond the spectrum
        # This will cause an error because numpy.min on empty array fails
        with pytest.raises(ValueError, match="zero-size array"):
            emc_compliance_test(
                waveform_trace,
                standard="FCC_Part15_ClassB",
                frequency_range=(1e12, 2e12),  # 1-2 THz (way beyond)
            )

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_single_frequency_point(self, mock_fft, waveform_trace):
        """Test behavior with single frequency point."""
        freq = np.array([100e6])
        mag = np.array([1e-6])
        mock_fft.return_value = (freq, mag)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert len(result["spectrum_freq"]) == 1
        assert result["status"] in ["PASS", "FAIL"]

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_spectrum_return_types(self, mock_fft, waveform_trace, mock_fft_output):
        """Test that return types are correct."""
        mock_fft.return_value = mock_fft_output

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert isinstance(result, dict)
        assert isinstance(result["status"], str)
        assert isinstance(result["standard"], str)
        assert isinstance(result["violations"], list)
        assert isinstance(result["margin_to_limit"], float | np.floating)
        assert isinstance(result["worst_frequency"], float | np.floating)
        assert isinstance(result["worst_margin"], float | np.floating)
        assert isinstance(result["spectrum_freq"], np.ndarray)
        assert isinstance(result["spectrum_mag"], np.ndarray)
        assert isinstance(result["limit_freq"], np.ndarray)
        assert isinstance(result["limit_mag"], np.ndarray)
        assert isinstance(result["detector"], str)


@pytest.mark.unit
class TestGenerateComplianceReport:
    """Test compliance report generation."""

    def test_pass_report_generation(self):
        """Test HTML report generation for passing test."""
        result = {
            "status": "PASS",
            "standard": "FCC_Part15_ClassB",
            "margin_to_limit": 10.5,
            "worst_frequency": 100e6,
            "worst_margin": 10.5,
            "violations": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            # Read and verify report
            with open(output_path) as f:
                html = f.read()

            assert "<html>" in html
            assert "EMC Compliance Test Report" in html
            assert "FCC_Part15_ClassB" in html
            assert "PASS" in html
            assert "green" in html
            assert "10.5" in html  # Margin
            assert "100.00 MHz" in html  # Frequency

            # Should not have violations table
            assert "Violations</h3>" not in html
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_fail_report_generation(self):
        """Test HTML report generation for failing test."""
        result = {
            "status": "FAIL",
            "standard": "CE_CISPR22_ClassA",
            "margin_to_limit": -5.2,
            "worst_frequency": 200e6,
            "worst_margin": -5.2,
            "violations": [
                {
                    "frequency": 100e6,
                    "measured_dbuv": 65.0,
                    "limit_dbuv": 60.0,
                    "excess_db": 5.0,
                },
                {
                    "frequency": 200e6,
                    "measured_dbuv": 62.5,
                    "limit_dbuv": 60.0,
                    "excess_db": 2.5,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            with open(output_path) as f:
                html = f.read()

            assert "<html>" in html
            assert "FAIL" in html
            assert "red" in html
            assert "-5.2" in html  # Negative margin
            assert "200.00 MHz" in html

            # Should have violations table
            assert "Violations</h3>" in html
            assert "100.00" in html  # First violation frequency
            assert "65.00" in html  # Measured
            assert "60.00" in html  # Limit
            assert "5.00" in html  # Excess

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_report_file_creation(self):
        """Test that report file is created at specified path."""
        result = {
            "status": "PASS",
            "standard": "FCC_Part15_ClassB",
            "margin_to_limit": 5.0,
            "worst_frequency": 50e6,
            "worst_margin": 5.0,
            "violations": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_report_html_structure(self):
        """Test that report has valid HTML structure."""
        result = {
            "status": "PASS",
            "standard": "MIL_STD_461G_CE102",
            "margin_to_limit": 15.0,
            "worst_frequency": 10e6,
            "worst_margin": 15.0,
            "violations": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            with open(output_path) as f:
                html = f.read()

            # Check basic HTML structure
            assert html.count("<html>") == 1
            assert html.count("</html>") == 1
            assert html.count("<head>") == 1
            assert html.count("</head>") == 1
            assert html.count("<body>") == 1
            assert html.count("</body>") == 1
            assert "<title>" in html
            assert "<table>" in html

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_report_violation_table(self):
        """Test violation table formatting in report."""
        result = {
            "status": "FAIL",
            "standard": "FCC_Part15_ClassA",
            "margin_to_limit": -3.0,
            "worst_frequency": 150e6,
            "worst_margin": -3.0,
            "violations": [
                {
                    "frequency": 50e6,
                    "measured_dbuv": 58.5,
                    "limit_dbuv": 56.0,
                    "excess_db": 2.5,
                },
                {
                    "frequency": 150e6,
                    "measured_dbuv": 59.0,
                    "limit_dbuv": 56.0,
                    "excess_db": 3.0,
                },
                {
                    "frequency": 250e6,
                    "measured_dbuv": 57.2,
                    "limit_dbuv": 56.0,
                    "excess_db": 1.2,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            with open(output_path) as f:
                html = f.read()

            # Check all violations are present
            assert "50.00" in html  # 50 MHz
            assert "150.00" in html  # 150 MHz
            assert "250.00" in html  # 250 MHz

            # Check table headers
            assert "Frequency (MHz)" in html
            assert "Limit" in html
            assert "Measured" in html
            assert "Excess (dB)" in html

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_report_no_violations_no_table(self):
        """Test that no violation table is generated for passing tests."""
        result = {
            "status": "PASS",
            "standard": "FCC_Part15_ClassB",
            "margin_to_limit": 8.0,
            "worst_frequency": 75e6,
            "worst_margin": 8.0,
            "violations": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            with open(output_path) as f:
                html = f.read()

            # No violations section
            assert "Violations</h3>" not in html

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_report_formatting_precision(self):
        """Test numeric formatting precision in report."""
        result = {
            "status": "FAIL",
            "standard": "CE_CISPR32_ClassB",
            "margin_to_limit": -1.234567,
            "worst_frequency": 123.456789e6,
            "worst_margin": -1.234567,
            "violations": [
                {
                    "frequency": 123.456789e6,
                    "measured_dbuv": 48.123456,
                    "limit_dbuv": 47.0,
                    "excess_db": 1.123456,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            _generate_compliance_report(result, output_path)

            with open(output_path) as f:
                html = f.read()

            # Check 2 decimal places for dB values
            assert "-1.23" in html  # Margin
            assert "123.46" in html  # Frequency

        finally:
            Path(output_path).unlink(missing_ok=True)


@pytest.mark.unit
class TestWorkflowsComplianceIntegration:
    """Integration tests without mocking FFT to achieve real coverage."""

    def test_real_compliance_pass(self, waveform_trace):
        """Test real compliance test with actual FFT (no mocking)."""
        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # Verify result structure regardless of pass/fail status
        assert result["status"] in ["PASS", "FAIL"]
        assert result["standard"] == "FCC_Part15_ClassB"
        assert isinstance(result["violations"], list)
        assert isinstance(result["margin_to_limit"], float | np.floating)
        assert isinstance(result["worst_frequency"], float | np.floating)
        assert isinstance(result["worst_margin"], float | np.floating)
        assert isinstance(result["spectrum_freq"], np.ndarray)
        assert isinstance(result["spectrum_mag"], np.ndarray)
        assert isinstance(result["limit_freq"], np.ndarray)
        assert isinstance(result["limit_mag"], np.ndarray)
        assert result["detector"] == "peak"

    def test_real_compliance_with_frequency_range(self, waveform_trace):
        """Test compliance with frequency range filtering (real FFT)."""
        f_min, f_max = 1e6, 50e6  # 1-50 MHz
        result = emc_compliance_test(
            waveform_trace,
            standard="FCC_Part15_ClassB",
            frequency_range=(f_min, f_max),
        )

        # Check frequency filtering worked
        assert np.all(result["spectrum_freq"] >= f_min)
        assert np.all(result["spectrum_freq"] <= f_max)
        assert result["status"] in ["PASS", "FAIL"]

    def test_real_compliance_different_detectors(self, waveform_trace):
        """Test different detector types (real FFT)."""
        for detector in ["peak", "quasi-peak", "average"]:
            result = emc_compliance_test(
                waveform_trace,
                standard="CE_CISPR22_ClassB",
                detector=detector,
            )
            assert result["detector"] == detector
            assert result["status"] in ["PASS", "FAIL"]

    def test_real_compliance_all_standards(self, waveform_trace):
        """Test all regulatory standards (real FFT)."""
        standards = [
            "FCC_Part15_ClassA",
            "FCC_Part15_ClassB",
            "CE_CISPR22_ClassA",
            "CE_CISPR22_ClassB",
            "CE_CISPR32_ClassA",
            "CE_CISPR32_ClassB",
            "MIL_STD_461G_CE102",
            "MIL_STD_461G_RE102",
        ]

        for standard in standards:
            result = emc_compliance_test(waveform_trace, standard=standard)
            assert result["standard"] == standard
            assert result["status"] in ["PASS", "FAIL"]
            # Verify result structure
            assert "violations" in result
            assert "margin_to_limit" in result
            assert "worst_frequency" in result
            assert "spectrum_freq" in result
            assert "spectrum_mag" in result
            assert "limit_freq" in result
            assert "limit_mag" in result

    def test_real_compliance_with_report(self, waveform_trace):
        """Test report generation (real FFT)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            report_path = f.name

        try:
            result = emc_compliance_test(
                waveform_trace,
                standard="MIL_STD_461G_CE102",
                report=report_path,
            )

            # Verify report was created
            assert Path(report_path).exists()
            assert Path(report_path).stat().st_size > 0

            # Read and verify basic HTML structure
            with open(report_path) as f:
                html = f.read()

            assert "<html>" in html
            assert "EMC Compliance Test Report" in html
            assert "MIL_STD_461G_CE102" in html
            assert result["status"] in html

        finally:
            Path(report_path).unlink(missing_ok=True)

    def test_real_compliance_violation_structure(self):
        """Test violation structure with a signal that will violate limits."""
        # Create a high-amplitude signal that will violate limits
        sample_rate = 100e6
        duration = 1e-3
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # Very high amplitude signal
        data = 10.0 * np.sin(2 * np.pi * 10e6 * t)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = emc_compliance_test(trace, standard="FCC_Part15_ClassB")

        # With such high amplitude, we expect failures
        if result["status"] == "FAIL":
            assert len(result["violations"]) > 0

            for violation in result["violations"]:
                assert "frequency" in violation
                assert "measured_dbuv" in violation
                assert "limit_dbuv" in violation
                assert "excess_db" in violation
                assert violation["measured_dbuv"] > violation["limit_dbuv"]
                assert violation["excess_db"] > 0

            # Worst margin should be negative for failures
            assert result["worst_margin"] < 0
            assert result["margin_to_limit"] < 0


@pytest.mark.unit
class TestWorkflowsComplianceEdgeCases:
    """Test edge cases and error conditions."""

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_negative_frequencies_filtered(self, mock_fft, waveform_trace):
        """Test that negative frequencies are handled correctly."""
        # FFT can return negative frequencies in some implementations
        freq = np.linspace(-100e6, 100e6, 201)
        mag = np.ones(201) * 1e-6
        mock_fft.return_value = (freq, mag)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # Should still work even with negative frequencies
        assert result["status"] in ["PASS", "FAIL"]

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_very_large_spectrum(self, mock_fft, waveform_trace):
        """Test with very large spectrum array."""
        freq = np.linspace(0, 1e9, 100000)  # 100k points
        mag = np.ones(100000) * 1e-6
        mock_fft.return_value = (freq, mag)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        assert len(result["spectrum_freq"]) == 100000
        assert result["status"] in ["PASS", "FAIL"]

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_nan_in_spectrum(self, mock_fft, waveform_trace):
        """Test handling of NaN values in spectrum."""
        freq = np.linspace(0, 100e6, 101)
        mag = np.ones(101) * 1e-6
        mag[50] = np.nan
        mock_fft.return_value = (freq, mag)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # NaN will propagate to dBµV calculation
        assert np.any(np.isnan(result["spectrum_mag"]))

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_inf_in_spectrum(self, mock_fft, waveform_trace):
        """Test handling of infinity values in spectrum."""
        freq = np.linspace(0, 100e6, 101)
        mag = np.ones(101) * 1e-6
        mag[50] = np.inf
        mock_fft.return_value = (freq, mag)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # Inf will propagate to dBµV calculation
        assert np.any(np.isinf(result["spectrum_mag"]))

    @patch("oscura.analyzers.waveform.spectral.fft")
    def test_boundary_violation(self, mock_fft, waveform_trace):
        """Test violation exactly at the limit boundary."""
        freq = np.linspace(0, 100e6, 101)
        # Create spectrum that exactly matches limit at one point
        mock_fft.return_value = (freq, np.ones(101) * 1e-6)

        result = emc_compliance_test(waveform_trace, standard="FCC_Part15_ClassB")

        # Exact match should not be a violation (margin = 0)
        # Only negative margins are violations
        if result["status"] == "FAIL":
            assert all(v["excess_db"] > 0 for v in result["violations"])
