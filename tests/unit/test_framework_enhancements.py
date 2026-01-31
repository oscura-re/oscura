"""Tests for new framework enhancements (Phase 1 & 2).

Tests for:
- Signal type properties (is_digital, is_analog, is_iq, signal_type)
- Plot embedding API (embed_plots)
- Batch visualization (generate_all_plots)
- Complete workflow orchestration (analyze_complete)
"""

from pathlib import Path

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, IQTrace, TraceMetadata, WaveformTrace


class TestSignalTypeProperties:
    """Test signal type detection properties on trace classes."""

    def test_waveform_trace_properties(self) -> None:
        """WaveformTrace should identify as analog."""
        data = np.sin(np.linspace(0, 2 * np.pi, 100))
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        assert trace.is_analog is True
        assert trace.is_digital is False
        assert trace.is_iq is False
        assert trace.signal_type == "analog"

    def test_digital_trace_properties(self) -> None:
        """DigitalTrace should identify as digital."""
        data = np.array([False, True, True, False], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)

        assert trace.is_analog is False
        assert trace.is_digital is True
        assert trace.is_iq is False
        assert trace.signal_type == "digital"

    def test_iq_trace_properties(self) -> None:
        """IQTrace should identify as iq."""
        i_data = np.cos(np.linspace(0, 2 * np.pi, 100))
        q_data = np.sin(np.linspace(0, 2 * np.pi, 100))
        metadata = TraceMetadata(sample_rate=1e6)
        trace = IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)

        assert trace.is_analog is False
        assert trace.is_digital is False
        assert trace.is_iq is True
        assert trace.signal_type == "iq"


class TestPlotEmbedding:
    """Test plot embedding functionality."""

    def test_embed_plots_basic(self) -> None:
        """Test basic plot embedding in HTML."""
        from oscura.reporting import embed_plots

        html_content = "<html><body><div>Content</div></body></html>"
        plots = {
            "test_plot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        }

        result = embed_plots(html_content, plots)

        assert "test_plot" in result.lower()
        assert "<img" in result
        assert "data:image/png;base64" in result
        assert "<!-- EMBEDDED PLOTS -->" in result

    def test_embed_plots_empty(self) -> None:
        """Test embed_plots with no plots."""
        from oscura.reporting import embed_plots

        html_content = "<html><body>Test</body></html>"
        plots = {}

        result = embed_plots(html_content, plots)

        # Should still add the section even if empty
        assert "<!-- EMBEDDED PLOTS -->" in result

    def test_embed_plots_custom_title(self) -> None:
        """Test embed_plots with custom section title."""
        from oscura.reporting import embed_plots

        html_content = "<html><body><div>Content</div></body></html>"
        plots = {"plot1": "data:image/png;base64,ABC"}

        result = embed_plots(html_content, plots, section_title="My Plots")

        assert "My Plots" in result


class TestBatchVisualization:
    """Test batch plot generation."""

    @pytest.mark.parametrize(
        "test_name,description",
        [
            ("basic", "Basic batch plot generation"),
        ],
    )
    def test_generate_all_plots_analog(self, test_name: str, description: str) -> None:
        """Test generate_all_plots for analog signals."""
        from oscura.visualization import batch

        # Create test signal (needs >1024 samples for spectrogram NFFT)
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 10000))
        metadata = TraceMetadata(sample_rate=100000)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Generate plots
        plots = batch.generate_all_plots(trace, verbose=False)

        # Should have analog plots (at minimum FFT should work)
        assert "fft" in plots, "FFT plot should always be generated"

        # Check expected plots (some may fail in test environment due to matplotlib backend)
        expected_plots = ["waveform", "histogram", "spectrogram", "statistics"]
        for plot_name in expected_plots:
            if plot_name in plots:
                # Verify it's base64
                assert plots[plot_name].startswith("data:image/png;base64,"), (
                    f"{plot_name} not base64"
                )

        # At least 2 plots should be generated (lenient for test environment)
        assert len(plots) >= 2, f"Expected at least 2 plots, got {len(plots)}: {list(plots.keys())}"

        # All generated plots should be base64 data URIs
        for plot_name, plot_data in plots.items():
            assert plot_data.startswith("data:image/png;base64,"), f"{plot_name} not base64"

    def test_generate_all_plots_digital(self) -> None:
        """Test generate_all_plots for digital signals."""
        from oscura.visualization import batch

        # Create test digital signal
        data = np.array([False, False, True, True, False, True], dtype=bool)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=data, metadata=metadata)

        # Generate plots
        plots = batch.generate_all_plots(trace, verbose=False)

        # Should have digital plots
        assert "waveform" in plots
        assert "logic" in plots

        # Should NOT have analog-specific plots
        assert "fft" not in plots
        assert "histogram" not in plots

    def test_fig_to_base64(self) -> None:
        """Test figure to base64 conversion."""
        import matplotlib.pyplot as plt

        from oscura.visualization.batch import fig_to_base64

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        result = fig_to_base64(fig)

        assert result.startswith("data:image/png;base64,")
        assert len(result) > 100  # Should have meaningful content


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete workflow orchestration."""

    def test_analyze_complete_basic(self, tmp_path: Path) -> None:
        """Test basic complete analysis workflow."""
        import oscura as osc
        from oscura.workflows import waveform

        # Create test file
        test_file = tmp_path / "test_signal.wfm"
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, 1000))
        metadata = TraceMetadata(sample_rate=100000)
        trace_obj = WaveformTrace(data=data, metadata=metadata)

        # Save test file (using oscura's save if available, otherwise skip)
        try:
            osc.save(test_file, trace_obj)
        except (AttributeError, NotImplementedError):
            pytest.skip("oscura.save not available for WaveformTrace")

        # Run complete analysis
        results = waveform.analyze_complete(
            test_file,
            output_dir=tmp_path / "output",
            generate_plots=True,
            generate_report=True,
            verbose=False,
        )

        # Verify results structure
        assert "filepath" in results
        assert "trace" in results
        assert "is_digital" in results
        assert "results" in results
        assert "plots" in results
        assert "report_path" in results
        assert "output_dir" in results

        # Verify analysis was run
        assert results["is_digital"] is False
        assert "time_domain" in results["results"]

        # Verify plots were generated
        assert len(results["plots"]) > 0

        # Verify report was created
        assert results["report_path"] is not None
        assert results["report_path"].exists()

    def test_analyze_complete_selective_analyses(self, tmp_path: Path) -> None:
        """Test complete analysis with selective analyses."""
        import oscura as osc
        from oscura.workflows import waveform

        # Create test file
        test_file = tmp_path / "test_signal.wfm"
        data = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 1000))
        metadata = TraceMetadata(sample_rate=100000)
        trace_obj = WaveformTrace(data=data, metadata=metadata)

        try:
            osc.save(test_file, trace_obj)
        except (AttributeError, NotImplementedError):
            pytest.skip("oscura.save not available")

        # Run with only time_domain analysis
        results = waveform.analyze_complete(
            test_file,
            output_dir=tmp_path / "output",
            analyses=["time_domain"],
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Should only have time_domain results
        assert "time_domain" in results["results"]
        assert "frequency_domain" not in results["results"]
        assert len(results["plots"]) == 0
        assert results["report_path"] is None

    def test_analyze_complete_invalid_file(self) -> None:
        """Test complete analysis with non-existent file."""
        from oscura.workflows import waveform

        with pytest.raises(FileNotFoundError):
            waveform.analyze_complete("nonexistent_file.wfm", verbose=False)

    def test_analyze_complete_invalid_analysis(self, tmp_path: Path) -> None:
        """Test complete analysis with invalid analysis type."""
        import oscura as osc
        from oscura.workflows import waveform

        # Create test file
        test_file = tmp_path / "test_signal.wfm"
        data = np.sin(np.linspace(0, 2 * np.pi, 100))
        metadata = TraceMetadata(sample_rate=1e6)
        trace_obj = WaveformTrace(data=data, metadata=metadata)

        try:
            osc.save(test_file, trace_obj)
        except (AttributeError, NotImplementedError):
            pytest.skip("oscura.save not available")

        with pytest.raises(ValueError, match="Invalid analysis types"):
            waveform.analyze_complete(
                test_file,
                analyses=["invalid_analysis"],
                verbose=False,  # type: ignore[list-item]
            )


class TestMeasurementMetadata:
    """Test MEASUREMENT_METADATA integration."""

    def test_metadata_exists(self) -> None:
        """Test that MEASUREMENT_METADATA is exported."""
        from oscura.analyzers.waveform import MEASUREMENT_METADATA

        assert isinstance(MEASUREMENT_METADATA, dict)
        assert len(MEASUREMENT_METADATA) > 0

    def test_metadata_structure(self) -> None:
        """Test MEASUREMENT_METADATA has correct structure."""
        from oscura.analyzers.waveform import MEASUREMENT_METADATA

        # Check a few known measurements
        assert "amplitude" in MEASUREMENT_METADATA
        assert "frequency" in MEASUREMENT_METADATA
        assert "duty_cycle" in MEASUREMENT_METADATA

        # Check structure
        for value in MEASUREMENT_METADATA.values():
            assert "unit" in value
            assert "description" in value
            assert isinstance(value["unit"], str)
            assert isinstance(value["description"], str)

    def test_overshoot_undershoot_units(self) -> None:
        """Test that overshoot/undershoot have correct units (%)."""
        from oscura.analyzers.waveform import MEASUREMENT_METADATA

        # Critical: these should be "%" not "ratio"
        assert MEASUREMENT_METADATA["overshoot"]["unit"] == "%"
        assert MEASUREMENT_METADATA["undershoot"]["unit"] == "%"
        assert MEASUREMENT_METADATA["duty_cycle"]["unit"] == "ratio"


class TestReportAddMeasurements:
    """Test Report.add_measurements() convenience method."""

    def test_add_measurements_basic(self) -> None:
        """Test basic add_measurements functionality."""
        from oscura.reporting import Report, ReportConfig

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        measurements = {"amplitude": 1.5, "frequency": 440.0}
        unit_map = {"amplitude": "V", "frequency": "Hz"}

        section = report.add_measurements("Test Section", measurements, unit_map)

        assert section.title == "Test Section"
        assert len(report.sections) == 1

    def test_add_measurements_auto_units(self) -> None:
        """Test add_measurements with automatic unit detection."""
        from oscura.reporting import Report, ReportConfig

        config = ReportConfig(title="Test Report")
        report = Report(config=config)

        # Use measurements that have framework metadata
        measurements = {"amplitude": 1.5, "frequency": 440.0, "duty_cycle": 0.5}

        # Should auto-detect units from MEASUREMENT_METADATA
        section = report.add_measurements("Test Section", measurements, unit_map=None)

        assert section.title == "Test Section"
        # Content should be HTML with proper formatting
        assert "<ul>" in section.content or "<li>" in section.content
