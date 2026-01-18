"""Tests for EMC compliance testing module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.compliance import (
    AVAILABLE_MASKS,
    ComplianceReportFormat,
    ComplianceResult,
    ComplianceViolation,
    DetectorType,
    LimitMask,
    check_compliance,
    create_custom_mask,
    generate_compliance_report,
    load_limit_mask,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


class TestLimitMasks:
    """Tests for EMC limit masks (EMC-001)."""

    def test_available_masks(self):
        """Test built-in masks are available."""
        assert len(AVAILABLE_MASKS) > 0
        assert "FCC_Part15_ClassB" in AVAILABLE_MASKS
        assert "CE_CISPR22_ClassB" in AVAILABLE_MASKS
        assert "MIL_STD_461G_RE102" in AVAILABLE_MASKS

    def test_load_fcc_class_b(self):
        """Test loading FCC Part 15 Class B mask."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        assert mask.name == "FCC_Part15_ClassB"
        assert mask.regulatory_body == "FCC"
        assert mask.unit in ("dBuV", "dBuV/m")
        assert mask.distance == 3.0
        assert len(mask.frequency) > 0
        assert len(mask.limit) == len(mask.frequency)

    def test_load_cispr_class_b(self):
        """Test loading CISPR 22 Class B mask."""
        mask = load_limit_mask("CE_CISPR22_ClassB")

        assert mask.regulatory_body == "CE"
        assert "CISPR 22" in mask.document

    def test_load_mil_std(self):
        """Test loading MIL-STD-461G mask."""
        mask = load_limit_mask("MIL_STD_461G_RE102")

        assert mask.regulatory_body == "MIL"
        assert "MIL-STD-461G" in mask.document

    def test_frequency_range(self):
        """Test frequency range property."""
        mask = load_limit_mask("FCC_Part15_ClassB")
        f_min, f_max = mask.frequency_range

        assert f_min < f_max
        assert f_min > 0

    def test_interpolate(self):
        """Test limit interpolation."""
        mask = load_limit_mask("FCC_Part15_ClassB")
        f_min, f_max = mask.frequency_range

        # Test frequencies within range
        test_freqs = np.array([f_min, (f_min + f_max) / 2, f_max])
        limits = mask.interpolate(test_freqs)

        assert len(limits) == len(test_freqs)
        assert not np.any(np.isnan(limits))

    def test_create_custom_mask(self):
        """Test creating custom limit mask."""
        mask = create_custom_mask(
            name="CustomLimit",
            frequencies=[30e6, 100e6, 1000e6],
            limits=[40, 35, 30],
            unit="dBuV/m",
            description="Custom test limit",
        )

        assert mask.name == "CustomLimit"
        assert len(mask.frequency) == 3
        assert mask.unit == "dBuV/m"

    def test_mask_to_dict(self):
        """Test mask serialization."""
        mask = load_limit_mask("FCC_Part15_ClassB")
        data = mask.to_dict()

        assert "name" in data
        assert "frequency" in data
        assert "limit" in data
        assert isinstance(data["frequency"], list)

    def test_mask_from_dict(self):
        """Test mask deserialization."""
        data = {
            "name": "TestMask",
            "description": "Test",
            "frequency": [30e6, 100e6],
            "limit": [40, 35],
            "unit": "dBuV",
        }
        mask = LimitMask.from_dict(data)

        assert mask.name == "TestMask"
        assert len(mask.frequency) == 2

    def test_unknown_mask_error(self):
        """Test error on unknown mask name."""
        with pytest.raises(ValueError):
            load_limit_mask("NonexistentMask")


class TestComplianceTesting:
    """Tests for compliance testing (EMC-002)."""

    @pytest.fixture
    def sample_trace(self):
        """Create sample trace for testing."""
        # 1 second at 10 MHz sample rate
        sample_rate = 10e6
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Multi-tone signal
        signal = 0.001 * np.sin(2 * np.pi * 100e6 * t)  # 100 MHz
        signal += 0.0005 * np.sin(2 * np.pi * 200e6 * t)  # 200 MHz

        return WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

    def test_check_compliance_pass(self, sample_trace):
        """Test compliance testing with passing signal."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        # Use pre-computed spectrum with known values
        freq = np.array([30e6, 100e6, 200e6, 500e6])
        level = np.array([20, 25, 20, 15])  # Well below limits

        result = check_compliance((freq, level), mask)

        assert isinstance(result, ComplianceResult)
        assert result.status in ("PASS", "FAIL")
        assert result.mask_name == "FCC_Part15_ClassB"
        assert isinstance(result.margin_to_limit, float)

    def test_check_compliance_fail(self):
        """Test compliance testing with failing signal."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        # Spectrum with violations
        freq = np.array([30e6, 100e6, 200e6])
        level = np.array([50, 60, 55])  # Above typical limits

        result = check_compliance((freq, level), mask)

        assert result.status == "FAIL"
        assert len(result.violations) > 0

    def test_violation_details(self):
        """Test violation details are captured."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        # Create violation
        freq = np.array([100e6])
        level = np.array([60])  # High level

        result = check_compliance((freq, level), mask)

        if result.violations:
            v = result.violations[0]
            assert isinstance(v, ComplianceViolation)
            assert v.frequency == 100e6
            assert v.excess_db > 0

    def test_detector_types(self):
        """Test different detector types."""
        mask = load_limit_mask("FCC_Part15_ClassB")
        freq = np.array([100e6])
        level = np.array([30])

        for detector in DetectorType:
            result = check_compliance((freq, level), mask, detector=detector)
            assert result.detector == detector.value

    def test_frequency_range_filter(self):
        """Test frequency range filtering."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        freq = np.array([30e6, 100e6, 200e6, 500e6])
        level = np.array([30, 35, 30, 25])

        result = check_compliance(
            (freq, level),
            mask,
            frequency_range=(50e6, 300e6),
        )

        # Should only test frequencies in range
        assert result.spectrum_freq.min() >= 50e6
        assert result.spectrum_freq.max() <= 300e6

    def test_result_summary(self):
        """Test result summary generation."""
        mask = load_limit_mask("FCC_Part15_ClassB")
        freq = np.array([100e6])
        level = np.array([30])

        result = check_compliance((freq, level), mask)
        summary = result.summary()

        assert "EMC Compliance Test" in summary
        assert "Status:" in summary
        assert "Margin" in summary


class TestComplianceReporting:
    """Tests for compliance report generation (EMC-003)."""

    @pytest.fixture
    def sample_result(self):
        """Create sample compliance result."""
        mask = load_limit_mask("FCC_Part15_ClassB")
        freq = np.array([30e6, 100e6, 200e6])
        level = np.array([35, 40, 35])

        return check_compliance((freq, level), mask)

    def test_generate_html_report(self, sample_result):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            result_path = generate_compliance_report(
                sample_result,
                output_path,
                format="html",
            )

            assert result_path.exists()
            content = result_path.read_text()
            assert "<html>" in content
            assert sample_result.mask_name in content

    def test_generate_markdown_report(self, sample_result):
        """Test Markdown report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            result_path = generate_compliance_report(
                sample_result,
                output_path,
                format="markdown",
            )

            assert result_path.exists()
            content = result_path.read_text()
            assert "# " in content
            assert sample_result.mask_name in content

    def test_generate_json_report(self, sample_result):
        """Test JSON report generation."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            result_path = generate_compliance_report(
                sample_result,
                output_path,
                format="json",
            )

            assert result_path.exists()
            with open(result_path) as f:
                data = json.load(f)

            assert data["status"] == sample_result.status
            assert data["mask_name"] == sample_result.mask_name

    def test_report_with_dut_info(self, sample_result):
        """Test report with DUT information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            dut_info = {
                "Model": "XYZ-100",
                "Serial": "12345",
                "Firmware": "v1.2.3",
            }

            result_path = generate_compliance_report(
                sample_result,
                output_path,
                dut_info=dut_info,
            )

            content = result_path.read_text()
            assert "XYZ-100" in content
            assert "12345" in content

    def test_report_format_enum(self):
        """Test report format enumeration."""
        assert ComplianceReportFormat.HTML.value == "html"
        assert ComplianceReportFormat.MARKDOWN.value == "markdown"
        assert ComplianceReportFormat.JSON.value == "json"
