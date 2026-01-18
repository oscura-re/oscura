import pytest

"""Tests for advanced reporting features.

Tests for:
"""

import numpy as np

from oscura.reporting import (
    PPTXPresentation,
    PPTXSlide,
    auto_select_chart,
    export_pptx,
    format_with_locale,
    generate_presentation_from_report,
    get_axis_scaling,
    recommend_chart_with_reasoning,
)

pytestmark = pytest.mark.unit


class TestLocaleFormatting:
    """Test REPORT-026: Locale-aware Formatting."""

    def test_format_with_locale_en_us(self):
        """Test US locale formatting (comma thousands, period decimal)."""
        result = format_with_locale(1234.56, locale="en_US")
        assert result == "1,234.56"

    def test_format_with_locale_de_de(self):
        """Test German locale formatting (period thousands, comma decimal)."""
        result = format_with_locale(1234.56, locale="de_DE")
        assert result == "1.234,56"

    def test_format_with_locale_fr_fr(self):
        """Test French locale formatting (space thousands, comma decimal)."""
        result = format_with_locale(1234.56, locale="fr_FR")
        assert result == "1 234,56"

    def test_format_with_locale_default(self):
        """Test default locale formatting (uses system locale)."""
        result = format_with_locale(1234.56)
        # Should not raise error and return formatted string
        assert isinstance(result, str)
        # Should contain the number in some format
        assert "1" in result and "234" in result and "56" in result

    def test_format_with_locale_date_en_us(self):
        """Test US date formatting (MM/DD/YYYY)."""
        # January 15, 2024 -> 01/15/2024
        timestamp = 1705334400.0  # Unix timestamp
        result = format_with_locale(date_value=timestamp, locale="en_US")
        assert "/" in result
        # Format should be MM/DD/YYYY

    def test_format_with_locale_date_de_de(self):
        """Test German date formatting (DD.MM.YYYY)."""
        timestamp = 1705334400.0
        result = format_with_locale(date_value=timestamp, locale="de_DE")
        assert "." in result
        # Format should be DD.MM.YYYY

    def test_format_with_locale_edge_cases(self):
        """Test edge cases for locale formatting."""
        # Zero
        assert format_with_locale(0.0, locale="en_US") == "0.00"

        # Negative
        result = format_with_locale(-1234.56, locale="en_US")
        assert "-" in result

        # Very large number
        result = format_with_locale(1234567.89, locale="en_US")
        assert "1,234,567.89" in result


class TestChartAutoSelection:
    """Test REPORT-028: Automated Chart Type Selection."""

    def test_time_series_selection(self):
        """Test time series data selects line chart."""
        chart_type = auto_select_chart("time_series", (1000, 2))
        assert chart_type == "line"

    def test_frequency_selection(self):
        """Test frequency data selects spectrum chart."""
        chart_type = auto_select_chart("frequency", (512,))
        assert chart_type == "spectrum"

    def test_distribution_selection(self):
        """Test distribution data selects histogram."""
        chart_type = auto_select_chart("distribution", (500,))
        assert chart_type == "histogram"

    def test_categorical_selection(self):
        """Test categorical data selects bar chart."""
        chart_type = auto_select_chart("categorical", (5,))
        assert chart_type == "bar"

    def test_comparison_selection(self):
        """Test comparison data selects scatter or bar."""
        # Small comparison -> scatter
        chart_type = auto_select_chart("comparison", (100, 2))
        assert chart_type == "scatter"

    def test_correlation_selection(self):
        """Test correlation data selects scatter chart."""
        chart_type = auto_select_chart("correlation", (100, 2))
        assert chart_type == "scatter"

    def test_matrix_selection(self):
        """Test 2D matrix selects heatmap."""
        chart_type = auto_select_chart("matrix", (100, 100))
        assert chart_type == "heatmap"

    def test_parts_selection(self):
        """Test parts-to-whole selects pie chart."""
        chart_type = auto_select_chart("parts", (5,))
        assert chart_type == "pie"

    def test_categorical_pie_selection(self):
        """Test categorical data with positive values may select pie."""
        data = np.array([25.0, 30.0, 20.0, 15.0, 10.0])
        chart_type = auto_select_chart("categorical", (5,), data=data)
        # Should select pie for small positive parts
        assert chart_type in ("pie", "bar")

    def test_default_1d_selection(self):
        """Test default selection for 1D data."""
        # Small 1D -> bar
        chart_type = auto_select_chart("unknown", (10,))
        assert chart_type == "bar"

        # Large 1D -> histogram
        chart_type = auto_select_chart("unknown", (1000,))
        assert chart_type == "histogram"

    def test_default_2d_selection(self):
        """Test default selection for 2D data."""
        # Large 2D -> heatmap
        chart_type = auto_select_chart("unknown", (100, 100))
        assert chart_type == "heatmap"

        # Small 2D -> scatter
        chart_type = auto_select_chart("unknown", (10, 2))
        assert chart_type == "scatter"

    def test_recommend_with_reasoning(self):
        """Test chart recommendation with reasoning."""
        result = recommend_chart_with_reasoning("time_series", (1000, 2))

        assert "chart_type" in result
        assert "reasoning" in result
        assert result["chart_type"] == "line"
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0

    def test_axis_scaling_frequency(self):
        """Test axis scaling for frequency data (log scale)."""
        scaling = get_axis_scaling("frequency")

        assert scaling["x_scale"] == "log"
        assert scaling["y_scale"] == "log"

    def test_axis_scaling_time_series(self):
        """Test axis scaling for time series (linear)."""
        scaling = get_axis_scaling("time_series")

        assert scaling["x_scale"] == "linear"
        assert scaling["y_scale"] == "linear"

    def test_axis_scaling_with_data(self):
        """Test axis scaling based on data range."""
        # Data spanning > 3 orders of magnitude -> log scale
        data = np.array([1e-3, 1e0, 1e3])
        scaling = get_axis_scaling("unknown", data=data)

        assert scaling["y_scale"] == "log"


class TestPPTXExport:
    """Test REPORT-023: PowerPoint/PPTX Export."""

    def test_pptx_presentation_creation(self):
        """Test PPTXPresentation object creation."""
        pres = PPTXPresentation(
            title="Test Report",
            subtitle="Analysis Results",
            author="Test User",
        )

        assert pres.title == "Test Report"
        assert pres.subtitle == "Analysis Results"
        assert pres.author == "Test User"
        assert len(pres.slides) == 0

    def test_pptx_add_slide(self):
        """Test adding slides to presentation."""
        pres = PPTXPresentation(title="Test")

        slide = pres.add_slide(
            title="Summary",
            content="Test content",
            layout="title_content",
            notes="Speaker notes",
        )

        assert isinstance(slide, PPTXSlide)
        assert slide.title == "Summary"
        assert slide.content == "Test content"
        assert slide.layout == "title_content"
        assert slide.notes == "Speaker notes"
        assert len(pres.slides) == 1

    def test_export_pptx_basic(self, tmp_path):
        """Test basic PPTX export (creates stub file)."""
        output_path = tmp_path / "report.pptx"

        report_data = {
            "summary": "All tests passed",
            "findings": ["Rise time: 2.3ns", "Fall time: 2.1ns"],
            "measurements": [{"name": "Voltage", "value": "3.3", "unit": "V", "status": "PASS"}],
            "plots": [],
        }

        result_path = export_pptx(
            report_data,
            output_path,
            title="Signal Analysis",
            subtitle="Test Report",
            author="Test Engineer",
        )

        assert result_path == output_path
        assert output_path.exists()
        # File created (either real PPTX or stub)

    def test_export_pptx_with_plots(self, tmp_path):
        """Test PPTX export with plot images."""
        output_path = tmp_path / "report.pptx"

        # Create dummy plot file
        plot_path = tmp_path / "plot.png"
        # Create a valid PNG image (1x1 red pixel)
        import struct
        import zlib

        def create_minimal_png() -> bytes:
            """Create a minimal valid 1x1 red PNG."""
            # PNG signature
            signature = b"\x89PNG\r\n\x1a\n"
            # IHDR chunk (width=1, height=1, bit_depth=8, color_type=2 RGB)
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            # IDAT chunk (compressed image data: filter byte + RGB)
            raw_data = b"\x00\xff\x00\x00"  # filter=0, R=255, G=0, B=0
            compressed = zlib.compress(raw_data)
            idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
            idat = (
                struct.pack(">I", len(compressed))
                + b"IDAT"
                + compressed
                + struct.pack(">I", idat_crc)
            )
            # IEND chunk
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return signature + ihdr + idat + iend

        plot_path.write_bytes(create_minimal_png())

        report_data = {
            "summary": "Test summary",
            "findings": ["Finding 1", "Finding 2"],
            "measurements": [],
            "plots": [plot_path],
        }

        result_path = export_pptx(report_data, output_path, title="Test")

        assert result_path.exists()

    def test_generate_presentation_from_report(self, tmp_path):
        """Test generating presentation from report structure."""
        output_path = tmp_path / "presentation.pptx"

        report = {
            "title": "Analysis Report",
            "subtitle": "Q4 2024",
            "author": "Engineering Team",
            "executive_summary": "All systems nominal",
            "key_findings": ["No violations", "Good margins"],
            "measurements": [{"name": "Test1", "value": "1.0", "unit": "V", "status": "PASS"}],
            "plot_paths": [],
        }

        result_path = generate_presentation_from_report(report, output_path)

        assert result_path.exists()

    def test_pptx_presentation_config(self, tmp_path):
        """Test PPTX generation with custom presentation config."""
        output_path = tmp_path / "custom.pptx"

        config = PPTXPresentation(
            title="Custom Title",
            subtitle="Custom Subtitle",
            author="Custom Author",
        )

        # Add custom slides
        config.add_slide("Slide 1", "Content 1")
        config.add_slide("Slide 2", ["Bullet 1", "Bullet 2"])

        report = {
            "executive_summary": "Summary",
            "key_findings": [],
            "measurements": [],
            "plot_paths": [],
        }

        result_path = generate_presentation_from_report(
            report, output_path, presentation_config=config
        )

        assert result_path.exists()


class TestReportingAdvancedFeaturesIntegration:
    """Integration tests for advanced reporting features."""

    def test_format_and_chart_selection(self):
        """Test combining locale formatting with chart selection."""
        # Format measurement in German locale
        value = 2345.67
        formatted = format_with_locale(value, locale="de_DE")
        assert "2.345,67" in formatted

        # Select appropriate chart for the data
        data = np.random.randn(1000)
        chart_type = auto_select_chart("distribution", data.shape, data=data)
        assert chart_type == "histogram"

    def test_complete_report_workflow(self, tmp_path):
        """Test complete workflow: format, select chart, export PPTX."""
        # 1. Format measurements with locale
        measurements = [
            {"name": "Voltage", "value": 3.345, "unit": "V"},
            {"name": "Current", "value": 1234.56, "unit": "mA"},
        ]

        formatted_measurements = []
        for meas in measurements:
            formatted_value = format_with_locale(meas["value"], locale="en_US")
            formatted_measurements.append(
                {
                    **meas,
                    "formatted_value": formatted_value,
                    "status": "PASS",
                }
            )

        # 2. Select chart types for data
        time_data = np.random.randn(1000)
        chart_type = auto_select_chart("time_series", time_data.shape)
        assert chart_type == "line"

        # 3. Export to PPTX
        output_path = tmp_path / "complete_report.pptx"
        report_data = {
            "summary": "All measurements within specification",
            "findings": [
                f"{m['name']}: {m['formatted_value']} {m['unit']}" for m in formatted_measurements
            ],
            "measurements": formatted_measurements,
            "plots": [],
        }

        result_path = export_pptx(report_data, output_path, title="Complete Report")
        assert result_path.exists()
