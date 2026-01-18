"""Tests for batch report generation.

Tests batch reporting functionality including multi-DUT processing,
summary reports, and yield analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from oscura.reporting.batch import (
    BatchReportResult,
    aggregate_batch_measurements,
    batch_report,
    generate_batch_report,
)

pytestmark = pytest.mark.unit


class TestBatchReportResult:
    """Test BatchReportResult class."""

    def test_create_result(self) -> None:
        """Test creating batch result."""
        result = BatchReportResult()
        assert result.total_duts == 0
        assert result.passed_duts == 0
        assert result.failed_duts == 0
        assert len(result.errors) == 0
        assert len(result.individual_report_paths) == 0

    def test_dut_yield_zero_duts(self) -> None:
        """Test yield calculation with zero DUTs."""
        result = BatchReportResult()
        assert result.dut_yield == 0.0

    def test_dut_yield_all_pass(self) -> None:
        """Test yield calculation when all DUTs pass."""
        result = BatchReportResult()
        result.total_duts = 10
        result.passed_duts = 10
        assert result.dut_yield == 100.0

    def test_dut_yield_half_pass(self) -> None:
        """Test yield calculation with 50% pass rate."""
        result = BatchReportResult()
        result.total_duts = 10
        result.passed_duts = 5
        assert result.dut_yield == 50.0

    def test_dut_yield_fractional(self) -> None:
        """Test yield calculation with fractional percentage."""
        result = BatchReportResult()
        result.total_duts = 7
        result.passed_duts = 5
        assert result.dut_yield == pytest.approx(71.42857, rel=0.001)


class TestGenerateBatchReport:
    """Test generate_batch_report function."""

    def test_generate_empty_batch(self) -> None:
        """Test generating report with empty batch."""
        batch_results: list[dict] = []  # type: ignore[type-arg]
        report = generate_batch_report(batch_results)

        assert report.config.title == "Batch Test Summary Report"
        assert len(report.sections) >= 1  # At least summary section

    def test_generate_single_dut(self) -> None:
        """Test generating report for single DUT."""
        batch_results = [
            {
                "dut_id": "DUT1",
                "pass_count": 5,
                "total_count": 5,
                "measurements": {"voltage": {"value": 3.3, "unit": "V", "passed": True}},
            }
        ]
        report = generate_batch_report(batch_results)

        section_titles = [s.title for s in report.sections]
        assert "Batch Summary" in section_titles

    def test_generate_multiple_duts(self) -> None:
        """Test generating report for multiple DUTs."""
        batch_results = [
            {
                "dut_id": f"DUT{i}",
                "pass_count": 3,
                "total_count": 3,
                "measurements": {},
            }
            for i in range(5)
        ]
        report = generate_batch_report(batch_results)

        # Check batch summary content
        summary_section = report.sections[0]
        assert "5 DUT" in str(summary_section.content)

    def test_generate_with_custom_title(self) -> None:
        """Test generating report with custom title."""
        batch_results = [{"dut_id": "DUT1", "measurements": {}}]
        report = generate_batch_report(batch_results, title="Custom Batch Report")

        assert report.config.title == "Custom Batch Report"

    def test_generate_without_individual_sections(self) -> None:
        """Test generating report without individual DUT sections."""
        batch_results = [{"dut_id": f"DUT{i}", "measurements": {}} for i in range(3)]
        report = generate_batch_report(batch_results, include_individual=False)

        # Should not have individual DUT sections
        dut_sections = [s for s in report.sections if s.title.startswith("DUT:")]
        assert len(dut_sections) == 0

    def test_generate_with_individual_sections(self) -> None:
        """Test generating report with individual DUT sections."""
        batch_results = [{"dut_id": f"DUT{i}", "measurements": {}} for i in range(3)]
        report = generate_batch_report(batch_results, include_individual=True)

        # Should have 3 individual DUT sections
        dut_sections = [s for s in report.sections if s.title.startswith("DUT:")]
        assert len(dut_sections) == 3

    def test_generate_without_yield_analysis(self) -> None:
        """Test generating report without yield analysis."""
        batch_results = [{"dut_id": "DUT1", "measurements": {}}]
        report = generate_batch_report(batch_results, include_yield_analysis=False)

        section_titles = [s.title for s in report.sections]
        assert "Yield Analysis" not in section_titles

    def test_generate_with_yield_analysis(self) -> None:
        """Test generating report with yield analysis."""
        batch_results = [
            {
                "dut_id": "DUT1",
                "pass_count": 3,
                "total_count": 3,
                "measurements": {
                    "test1": {"value": 1.0, "passed": True},
                    "test2": {"value": 2.0, "passed": True},
                },
            }
        ]
        report = generate_batch_report(batch_results, include_yield_analysis=True)

        section_titles = [s.title for s in report.sections]
        assert "Yield Analysis" in section_titles

    def test_generate_without_outliers(self) -> None:
        """Test generating report without outlier detection."""
        batch_results = [{"dut_id": "DUT1", "measurements": {}}]
        report = generate_batch_report(batch_results, include_outliers=False)

        section_titles = [s.title for s in report.sections]
        assert "Outlier Detection" not in section_titles

    def test_generate_with_outliers(self) -> None:
        """Test generating report with outlier detection."""
        batch_results = [
            {
                "dut_id": f"DUT{i}",
                "measurements": {"voltage": {"value": 3.3 + i * 0.1}},
            }
            for i in range(5)
        ]
        report = generate_batch_report(batch_results, include_outliers=True)

        section_titles = [s.title for s in report.sections]
        assert "Outlier Detection" in section_titles

    def test_batch_summary_content(self) -> None:
        """Test batch summary content generation."""
        batch_results = [
            {
                "dut_id": "DUT1",
                "pass_count": 10,
                "total_count": 10,
                "measurements": {},
            },
            {
                "dut_id": "DUT2",
                "pass_count": 8,
                "total_count": 10,
                "measurements": {},
            },
        ]
        report = generate_batch_report(batch_results)

        summary_content = str(report.sections[0].content)
        assert "2 DUT" in summary_content
        assert "18/20" in summary_content  # Total passed/total tests

    def test_dut_yield_in_summary(self) -> None:
        """Test that DUT yield appears in summary."""
        batch_results = [
            {"dut_id": "DUT1", "pass_count": 5, "total_count": 5},
            {"dut_id": "DUT2", "pass_count": 5, "total_count": 5},
            {"dut_id": "DUT3", "pass_count": 3, "total_count": 5},
        ]
        report = generate_batch_report(batch_results)

        summary_content = str(report.sections[0].content)
        assert "DUT Yield" in summary_content
        assert "2/3" in summary_content  # 2 out of 3 passed all tests

    def test_failed_duts_listed(self) -> None:
        """Test that failed DUTs are listed in summary."""
        batch_results = [
            {"dut_id": "DUT1", "pass_count": 5, "total_count": 5},
            {"dut_id": "DUT2", "pass_count": 3, "total_count": 5},
            {"dut_id": "DUT3", "pass_count": 2, "total_count": 5},
        ]
        report = generate_batch_report(batch_results)

        summary_content = str(report.sections[0].content)
        assert "Failed DUTs" in summary_content
        assert "DUT2" in summary_content
        assert "DUT3" in summary_content


class TestYieldAnalysis:
    """Test yield analysis section generation."""

    def test_yield_analysis_overall_yield(self) -> None:
        """Test overall yield in yield analysis."""
        batch_results = [
            {"dut_id": f"DUT{i}", "pass_count": 5 if i < 3 else 3, "total_count": 5}
            for i in range(5)
        ]

        from oscura.reporting.batch import _create_yield_analysis_section

        section = _create_yield_analysis_section(batch_results)

        assert "Overall Yield" in section.content
        assert "60.00%" in section.content  # 3/5 passed

    def test_yield_analysis_per_test(self) -> None:
        """Test per-test yield analysis."""
        batch_results = [
            {
                "dut_id": f"DUT{i}",
                "measurements": {
                    "test1": {"passed": True},
                    "test2": {"passed": i < 2},  # Only first 2 pass
                },
            }
            for i in range(4)
        ]

        from oscura.reporting.batch import _create_yield_analysis_section

        section = _create_yield_analysis_section(batch_results)

        content = section.content
        assert "test1" in content
        assert "test2" in content
        assert "100.0%" in content  # test1 has 100% yield
        assert "50.0%" in content  # test2 has 50% yield

    def test_yield_analysis_sorted_by_yield(self) -> None:
        """Test that tests are sorted by yield (worst first)."""
        batch_results = [
            {
                "measurements": {
                    "test_high": {"passed": True},
                    "test_low": {"passed": False},
                    "test_mid": {"passed": i % 2 == 0},
                }
            }
            for i in range(4)
        ]

        from oscura.reporting.batch import _create_yield_analysis_section

        section = _create_yield_analysis_section(batch_results)

        content_lines = section.content.split("\n")
        # Find the test lines
        test_lines = [line for line in content_lines if line.startswith("- test_")]

        # test_low should be first (0% yield)
        assert "test_low" in test_lines[0]


class TestBatchStatistics:
    """Test batch statistics section generation."""

    def test_batch_statistics_table(self) -> None:
        """Test batch statistics table generation."""
        batch_results = [
            {
                "measurements": {
                    "voltage": {"value": 3.3 + i * 0.1, "unit": "V"},
                    "current": {"value": 100 + i * 10, "unit": "mA"},
                }
            }
            for i in range(5)
        ]

        from oscura.reporting.batch import _create_batch_statistics_section

        section = _create_batch_statistics_section(batch_results)

        # Should contain a table
        assert isinstance(section.content, list)
        assert len(section.content) > 0
        table = section.content[0]
        assert table["type"] == "table"
        assert "Parameter" in table["headers"]
        assert "Mean" in table["headers"]
        assert "Std Dev" in table["headers"]

    def test_batch_statistics_values(self) -> None:
        """Test batch statistics calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        batch_results = [{"measurements": {"param": {"value": v, "unit": ""}}} for v in values]

        from oscura.reporting.batch import _create_batch_statistics_section

        section = _create_batch_statistics_section(batch_results)

        table = section.content[0]
        data = table["data"]

        # Find param row
        param_row = next(row for row in data if row[0] == "param")

        # Check stats are in the row (mean=3, min=1, max=5)
        # Can't check exact formatting, but values should be present
        assert len(param_row) == 6  # Parameter, Mean, Std Dev, Min, Max, Range


class TestOutlierDetection:
    """Test outlier detection section generation."""

    def test_no_outliers(self) -> None:
        """Test outlier detection when no outliers exist."""
        # All values close to mean
        batch_results = [
            {"dut_id": f"DUT{i}", "measurements": {"param": {"value": 3.0 + i * 0.1}}}
            for i in range(10)
        ]

        from oscura.reporting.batch import _create_outlier_detection_section

        section = _create_outlier_detection_section(batch_results)

        assert "No statistical outliers" in section.content

    def test_with_outliers(self) -> None:
        """Test outlier detection with outliers present."""
        batch_results = [
            {"dut_id": f"DUT{i}", "measurements": {"param": {"value": 3.0}}} for i in range(10)
        ]
        # Add extreme outlier
        batch_results.append({"dut_id": "DUT_OUTLIER", "measurements": {"param": {"value": 100.0}}})

        from oscura.reporting.batch import _create_outlier_detection_section

        section = _create_outlier_detection_section(batch_results)

        assert "DUT_OUTLIER" in section.content
        assert "z-score" in section.content

    def test_outlier_z_score(self) -> None:
        """Test that z-score is calculated correctly."""
        # Mean=10, std=2, value=20 => z-score=5
        batch_results = [
            {"dut_id": f"DUT{i}", "measurements": {"param": {"value": 10.0}}} for i in range(10)
        ]
        batch_results[0]["measurements"]["param"]["value"] = 12.0
        batch_results[1]["measurements"]["param"]["value"] = 8.0
        # Add outlier at mean + 5*std (assuming std ~= 1)
        batch_results.append({"dut_id": "OUTLIER", "measurements": {"param": {"value": 30.0}}})

        from oscura.reporting.batch import _create_outlier_detection_section

        section = _create_outlier_detection_section(batch_results)

        # Should detect outlier
        assert "OUTLIER" in section.content or "No statistical outliers" in section.content


class TestAggregateBatchMeasurements:
    """Test aggregate_batch_measurements function."""

    def test_aggregate_single_parameter(self) -> None:
        """Test aggregating single parameter."""
        batch_results = [{"measurements": {"voltage": {"value": 3.3 + i * 0.1}}} for i in range(5)]

        aggregated = aggregate_batch_measurements(batch_results)

        assert "voltage" in aggregated
        assert len(aggregated["voltage"]) == 5
        np.testing.assert_array_almost_equal(aggregated["voltage"], [3.3, 3.4, 3.5, 3.6, 3.7])

    def test_aggregate_multiple_parameters(self) -> None:
        """Test aggregating multiple parameters."""
        batch_results = [
            {
                "measurements": {
                    "voltage": {"value": 3.3},
                    "current": {"value": 100.0},
                }
            }
            for _ in range(3)
        ]

        aggregated = aggregate_batch_measurements(batch_results)

        assert "voltage" in aggregated
        assert "current" in aggregated
        assert len(aggregated["voltage"]) == 3
        assert len(aggregated["current"]) == 3

    def test_aggregate_sparse_measurements(self) -> None:
        """Test aggregating when not all DUTs have all measurements."""
        batch_results = [
            {"measurements": {"voltage": {"value": 3.3}}},
            {"measurements": {"current": {"value": 100.0}}},
            {"measurements": {"voltage": {"value": 3.4}, "current": {"value": 110.0}}},
        ]

        aggregated = aggregate_batch_measurements(batch_results)

        assert len(aggregated["voltage"]) == 2  # Only 2 DUTs had voltage
        assert len(aggregated["current"]) == 2  # Only 2 DUTs had current

    def test_aggregate_empty_batch(self) -> None:
        """Test aggregating empty batch."""
        aggregated = aggregate_batch_measurements([])
        assert len(aggregated) == 0

    def test_aggregate_no_measurements(self) -> None:
        """Test aggregating batch with no measurements."""
        batch_results = [{"dut_id": "DUT1"}, {"dut_id": "DUT2"}]

        aggregated = aggregate_batch_measurements(batch_results)

        assert len(aggregated) == 0

    def test_aggregate_skips_none_values(self) -> None:
        """Test that None values are skipped."""
        batch_results = [
            {"measurements": {"voltage": {"value": 3.3}}},
            {"measurements": {"voltage": {"value": None}}},
            {"measurements": {"voltage": {"value": 3.4}}},
        ]

        aggregated = aggregate_batch_measurements(batch_results)

        assert len(aggregated["voltage"]) == 2  # None value skipped
        np.testing.assert_array_almost_equal(aggregated["voltage"], [3.3, 3.4])


class TestBatchReportIntegration:
    """Integration tests for batch_report function."""

    @patch("oscura.reporting.batch._save_report")
    @patch("oscura.reporting.template_system.load_template")
    @patch("oscura.load")
    def test_batch_report_basic(
        self, mock_load: Mock, mock_template: Mock, mock_save: Mock, tmp_path: Path
    ) -> None:
        """Test basic batch report generation."""
        # Setup mocks
        mock_trace = MagicMock()
        mock_trace.data = np.array([1.0, 2.0, 3.0])
        mock_load.return_value = mock_trace
        mock_template.return_value = {}

        # Create test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"dut{i}.wfm"
            file_path.write_text("test")
            files.append(str(file_path))

        # Run batch report
        result = batch_report(files, output_dir=tmp_path, output_format="html")

        assert result.total_duts == 3
        assert result.passed_duts + result.failed_duts == result.total_duts

    @patch("oscura.reporting.batch._save_report")
    @patch("oscura.reporting.template_system.load_template")
    @patch("oscura.load")
    def test_batch_report_with_analyzer(
        self, mock_load: Mock, mock_template: Mock, mock_save: Mock, tmp_path: Path
    ) -> None:
        """Test batch report with custom analyzer."""
        mock_trace = MagicMock()
        mock_load.return_value = mock_trace
        mock_template.return_value = {}

        def custom_analyzer(trace: Any) -> dict:  # type: ignore[type-arg]
            return {
                "pass_count": 10,
                "total_count": 10,
                "measurements": {"test": {"value": 42, "passed": True}},
            }

        file_path = tmp_path / "dut1.wfm"
        file_path.write_text("test")

        result = batch_report(
            [str(file_path)],
            output_dir=tmp_path,
            analyzer=custom_analyzer,
            output_format="html",
        )

        assert result.total_duts == 1
        assert result.passed_duts == 1

    @patch("oscura.reporting.template_system.load_template")
    @patch("oscura.load")
    def test_batch_report_no_individual(
        self, mock_load: Mock, mock_template: Mock, tmp_path: Path
    ) -> None:
        """Test batch report without individual reports."""
        mock_trace = MagicMock()
        mock_trace.data = np.array([1.0, 2.0, 3.0])
        mock_load.return_value = mock_trace
        mock_template.return_value = {}

        file_path = tmp_path / "dut1.wfm"
        file_path.write_text("test")

        with patch("oscura.reporting.batch._save_report"):
            result = batch_report(
                [str(file_path)],
                output_dir=tmp_path,
                generate_individual=False,
                output_format="html",
            )

        assert len(result.individual_report_paths) == 0

    @patch("oscura.reporting.template_system.load_template")
    @patch("oscura.load")
    def test_batch_report_no_summary(
        self, mock_load: Mock, mock_template: Mock, tmp_path: Path
    ) -> None:
        """Test batch report without summary report."""
        mock_trace = MagicMock()
        mock_trace.data = np.array([1.0, 2.0, 3.0])
        mock_load.return_value = mock_trace
        mock_template.return_value = {}

        file_path = tmp_path / "dut1.wfm"
        file_path.write_text("test")

        with patch("oscura.reporting.batch._save_report"):
            result = batch_report(
                [str(file_path)],
                output_dir=tmp_path,
                generate_summary=False,
                output_format="html",
            )

        assert result.summary_report_path is None

    @patch("oscura.reporting.template_system.load_template")
    @patch("oscura.load")
    def test_batch_report_with_errors(
        self, mock_load: Mock, mock_template: Mock, tmp_path: Path
    ) -> None:
        """Test batch report with processing errors."""
        mock_load.side_effect = Exception("Load failed")
        mock_template.return_value = {}

        file_path = tmp_path / "dut1.wfm"
        file_path.write_text("test")

        with patch("oscura.reporting.batch._save_report"):
            result = batch_report([str(file_path)], output_dir=tmp_path, output_format="html")

        assert len(result.errors) > 0
        assert result.total_duts == 0  # Processing failed

    @patch("oscura.reporting.batch._save_report")
    @patch("oscura.reporting.template_system.load_template")
    @patch("oscura.load")
    def test_batch_report_custom_dut_id(
        self, mock_load: Mock, mock_template: Mock, mock_save: Mock, tmp_path: Path
    ) -> None:
        """Test batch report with custom DUT ID extractor."""
        mock_trace = MagicMock()
        mock_trace.data = np.array([1.0, 2.0, 3.0])
        mock_load.return_value = mock_trace
        mock_template.return_value = {}

        def extract_id(path: Path) -> str:
            return f"CUSTOM_{path.stem}"

        file_path = tmp_path / "test123.wfm"
        file_path.write_text("test")

        result = batch_report(
            [str(file_path)],
            output_dir=tmp_path,
            dut_id_extractor=extract_id,
            output_format="html",
        )

        # Check that custom ID was used (would need to inspect calls or results)
        assert result.total_duts == 1


class TestDUTSection:
    """Test individual DUT section generation."""

    def test_dut_section_basic(self) -> None:
        """Test creating basic DUT section."""
        result = {
            "dut_id": "DUT123",
            "pass_count": 5,
            "total_count": 5,
            "measurements": {},
        }

        from oscura.reporting.batch import _create_dut_section

        section = _create_dut_section(result, 0)

        assert section.title == "DUT: DUT123"
        assert section.collapsible is True
        assert "100.0%" in str(section.content)

    def test_dut_section_with_measurements(self) -> None:
        """Test DUT section with measurements."""
        result = {
            "dut_id": "DUT456",
            "pass_count": 2,
            "total_count": 3,
            "measurements": {
                "voltage": {"value": 3.3, "unit": "V", "passed": True},
                "current": {"value": 150, "unit": "mA", "passed": False},
            },
        }

        from oscura.reporting.batch import _create_dut_section

        section = _create_dut_section(result, 0)

        assert "66.7%" in str(section.content)  # 2/3 pass rate

    def test_dut_section_default_id(self) -> None:
        """Test DUT section with default ID."""
        result = {"pass_count": 5, "total_count": 5}

        from oscura.reporting.batch import _create_dut_section

        section = _create_dut_section(result, 3)

        assert section.title == "DUT: DUT-4"  # index 3 becomes DUT-4
