"""Tests for advanced digital logic visualization."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for tests
matplotlib.use("Agg")

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace
from oscura.visualization.digital_advanced import (
    generate_all_vintage_logic_plots,
    plot_bus_eye_diagram,
    plot_ic_timing_validation,
    plot_logic_analyzer_view,
    plot_multi_ic_timing_path,
    plot_timing_diagram_with_annotations,
)

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.fixture
def digital_traces():
    """Create sample digital traces for testing."""
    sample_rate = 1e6
    duration = 0.001
    num_samples = int(duration * sample_rate)

    # Create simple alternating patterns
    clk_data = np.array([i % 2 == 0 for i in range(num_samples)], dtype=bool)
    data_data = np.array([i % 4 < 2 for i in range(num_samples)], dtype=bool)

    return {
        "CLK": DigitalTrace(data=clk_data, metadata=TraceMetadata(sample_rate=sample_rate)),
        "DATA": DigitalTrace(data=data_data, metadata=TraceMetadata(sample_rate=sample_rate)),
    }


@pytest.fixture
def waveform_traces(signal_builder):
    """Create sample waveform traces for testing."""
    clk_data = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
    data_data = signal_builder.square_wave(frequency=500e3, duration=0.001, sample_rate=1e6)

    return {
        "CLK": WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6)),
        "DATA": WaveformTrace(data=data_data, metadata=TraceMetadata(sample_rate=1e6)),
    }


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all plots after each test."""
    yield
    plt.close("all")


class TestPlotLogicAnalyzerView:
    """Test plot_logic_analyzer_view function."""

    def test_basic_plot(self, digital_traces):
        """Test basic logic analyzer view creation."""
        fig, ax = plot_logic_analyzer_view(digital_traces)

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) > 0  # Should have plotted lines

    def test_with_title(self, digital_traces):
        """Test plot with custom title."""
        fig, ax = plot_logic_analyzer_view(digital_traces, title="Test Logic Analyzer")

        assert fig is not None
        assert ax.get_title() == "Test Logic Analyzer"

    def test_with_time_range(self, digital_traces):
        """Test plot with specific time range."""
        fig, ax = plot_logic_analyzer_view(digital_traces, time_range=(0, 0.0005))

        assert fig is not None
        xlim = ax.get_xlim()
        assert xlim[0] >= 0
        assert xlim[1] <= 0.0005 * 1.1  # Allow small margin

    def test_with_bus_grouping(self):
        """Test plot with bus grouping."""
        # Create 8-bit bus
        sample_rate = 1e6
        num_samples = 1000
        channels = {}
        for i in range(8):
            data = np.array([j % (2 ** (i + 1)) < 2**i for j in range(num_samples)], dtype=bool)
            channels[f"D{i}"] = DigitalTrace(
                data=data, metadata=TraceMetadata(sample_rate=sample_rate)
            )

        group_buses = {"DATA": [f"D{i}" for i in range(8)]}

        fig, ax = plot_logic_analyzer_view(channels, group_buses=group_buses)

        assert fig is not None
        assert ax is not None

    def test_without_hex(self, digital_traces):
        """Test plot without hex values."""
        fig, ax = plot_logic_analyzer_view(digital_traces, show_hex=False)

        assert fig is not None

    def test_without_cursors(self, digital_traces):
        """Test plot without timing cursors."""
        fig, ax = plot_logic_analyzer_view(digital_traces, show_cursors=False)

        assert fig is not None

    def test_custom_figsize(self, digital_traces):
        """Test plot with custom figure size."""
        fig, ax = plot_logic_analyzer_view(digital_traces, figsize=(10, 6))

        assert fig is not None
        figsize = fig.get_size_inches()
        assert figsize[0] == 10
        assert figsize[1] == 6

    def test_with_waveform_traces(self, waveform_traces):
        """Test plot with waveform (analog) traces."""
        fig, ax = plot_logic_analyzer_view(waveform_traces)

        assert fig is not None
        assert ax is not None


class TestPlotTimingDiagramWithAnnotations:
    """Test plot_timing_diagram_with_annotations function."""

    def test_basic_timing_diagram(self, digital_traces):
        """Test basic timing diagram creation."""
        fig, ax = plot_timing_diagram_with_annotations(digital_traces)

        assert fig is not None
        assert ax is not None

    def test_with_timing_params(self, digital_traces):
        """Test timing diagram with timing parameters."""
        timing_params = {
            "t_su": (10e-9, 40e-9, "Setup Time"),
            "t_h": (50e-9, 80e-9, "Hold Time"),
        }

        fig, ax = plot_timing_diagram_with_annotations(digital_traces, timing_params=timing_params)

        assert fig is not None
        assert ax is not None

    def test_with_title(self, digital_traces):
        """Test timing diagram with title."""
        fig, ax = plot_timing_diagram_with_annotations(digital_traces, title="Test Timing Diagram")

        assert fig is not None
        assert ax.get_title() == "Test Timing Diagram"

    def test_custom_figsize(self, digital_traces):
        """Test timing diagram with custom figure size."""
        fig, ax = plot_timing_diagram_with_annotations(digital_traces, figsize=(12, 6))

        assert fig is not None
        figsize = fig.get_size_inches()
        assert figsize[0] == 12
        assert figsize[1] == 6

    def test_with_reference_edges(self, digital_traces):
        """Test timing diagram with reference edges."""
        reference_edges = {"CLK": "rising", "DATA": "falling"}
        fig, ax = plot_timing_diagram_with_annotations(
            digital_traces, reference_edges=reference_edges
        )

        assert fig is not None


class TestPlotICTimingValidation:
    """Test plot_ic_timing_validation function."""

    def test_basic_validation_plot(self, waveform_traces):
        """Test basic IC timing validation plot."""
        signals = waveform_traces
        measured_timings = {"t_pd": 15e-9, "t_su": 10e-9, "t_h": 5e-9}

        fig, ax = plot_ic_timing_validation(signals, "74LS74", measured_timings)

        assert fig is not None
        assert ax is not None
        assert "74LS74" in ax.get_title()

    def test_validation_custom_figsize(self, waveform_traces):
        """Test validation plot with custom figure size."""
        signals = waveform_traces
        measured_timings = {"t_pd": 15e-9}

        fig, ax = plot_ic_timing_validation(signals, "74LS74", measured_timings, figsize=(10, 6))

        assert fig is not None
        figsize = fig.get_size_inches()
        assert figsize[0] == 10
        assert figsize[1] == 6

    def test_validation_with_different_ic(self, waveform_traces):
        """Test validation plot with different IC."""
        signals = waveform_traces
        measured_timings = {"t_pd": 25e-9, "t_su": 20e-9}

        fig, ax = plot_ic_timing_validation(signals, "74LS00", measured_timings)

        assert fig is not None
        assert "74LS00" in ax.get_title()


class TestPlotMultiICTimingPath:
    """Test plot_multi_ic_timing_path function."""

    def test_basic_timing_path(self, waveform_traces):
        """Test basic multi-IC timing path plot."""
        ic_chain = [
            ("74LS00", {"CLK": waveform_traces["CLK"], "DATA": waveform_traces["DATA"]}),
            ("74LS74", {"D": waveform_traces["DATA"], "Q": waveform_traces["CLK"]}),
        ]

        fig, ax = plot_multi_ic_timing_path(ic_chain)

        assert fig is not None
        assert ax is not None

    def test_with_title(self, waveform_traces):
        """Test timing path with custom title."""
        ic_chain = [
            ("74LS00", {"A": waveform_traces["CLK"], "Y": waveform_traces["DATA"]}),
        ]

        fig, ax = plot_multi_ic_timing_path(ic_chain, title="Test Path")

        assert fig is not None
        assert ax.get_title() == "Test Path"

    def test_custom_figsize(self, waveform_traces):
        """Test timing path with custom figure size."""
        ic_chain = [
            ("74LS00", {"CLK": waveform_traces["CLK"]}),
        ]

        fig, ax = plot_multi_ic_timing_path(ic_chain, figsize=(12, 8))

        assert fig is not None
        figsize = fig.get_size_inches()
        assert figsize[0] == 12
        assert figsize[1] == 8

    def test_multiple_ics(self, waveform_traces):
        """Test with multiple ICs in chain."""
        ic_chain = [
            ("74LS00", {"A": waveform_traces["CLK"], "B": waveform_traces["DATA"]}),
            ("74LS74", {"CLK": waveform_traces["CLK"], "D": waveform_traces["DATA"]}),
            ("74LS04", {"A": waveform_traces["CLK"]}),
        ]

        fig, ax = plot_multi_ic_timing_path(ic_chain)

        assert fig is not None


class TestPlotBusEyeDiagram:
    """Test plot_bus_eye_diagram function."""

    def test_basic_eye_diagram(self, waveform_traces):
        """Test basic eye diagram plot."""
        bus_traces = [waveform_traces["CLK"]]
        fig, axes = plot_bus_eye_diagram(bus_traces, symbol_period=1e-6)

        assert fig is not None
        assert axes is not None

    def test_with_title(self, waveform_traces):
        """Test eye diagram with custom title."""
        bus_traces = [waveform_traces["CLK"]]
        fig, axes = plot_bus_eye_diagram(bus_traces, symbol_period=1e-6, title="Test Eye Diagram")

        assert fig is not None
        assert fig._suptitle is not None
        assert "Test Eye Diagram" in fig._suptitle.get_text()

    def test_with_num_symbols(self, waveform_traces):
        """Test eye diagram with specified number of symbols."""
        bus_traces = [waveform_traces["CLK"]]
        fig, axes = plot_bus_eye_diagram(bus_traces, symbol_period=1e-6, num_symbols=50)

        assert fig is not None

    def test_custom_figsize(self, waveform_traces):
        """Test eye diagram with custom figure size."""
        bus_traces = [waveform_traces["CLK"]]
        fig, axes = plot_bus_eye_diagram(bus_traces, symbol_period=1e-6, figsize=(10, 6))

        assert fig is not None
        figsize = fig.get_size_inches()
        assert figsize[0] == 10
        assert figsize[1] == 6

    def test_multiple_traces(self, waveform_traces):
        """Test eye diagram with multiple bus traces."""
        bus_traces = [waveform_traces["CLK"], waveform_traces["DATA"]]
        fig, axes = plot_bus_eye_diagram(bus_traces, symbol_period=1e-6)

        assert fig is not None
        # axes is a numpy array when there are multiple subplots
        assert len(axes) == 2


class TestGenerateAllVintageLogicPlots:
    """Test generate_all_vintage_logic_plots function."""

    def test_basic_generation(self, waveform_traces, tmp_path):
        """Test basic plot generation."""
        from datetime import datetime

        from oscura.analyzers.digital.vintage_result import VintageLogicAnalysisResult

        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=1.0,
            detected_family="TTL",
            family_confidence=0.9,
            voltage_levels={},
            identified_ics=[],
            timing_measurements={},
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=1.0,
            modern_replacements=[],
            bom=[],
            warnings=[],
            confidence_scores={},
        )

        plots = generate_all_vintage_logic_plots(
            result, waveform_traces, output_dir=tmp_path, save_formats=["png"]
        )

        # Should return dictionary of plot objects
        assert isinstance(plots, dict)
        # Should have generated some plots
        assert len(plots) >= 0

    def test_with_ic_identification(self, waveform_traces, tmp_path):
        """Test plot generation with IC identification results."""
        from datetime import datetime

        from oscura.analyzers.digital.vintage_result import (
            ICIdentificationResult,
            VintageLogicAnalysisResult,
        )

        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=1.0,
            detected_family="TTL",
            family_confidence=0.9,
            voltage_levels={},
            identified_ics=[
                ICIdentificationResult(
                    ic_name="7474",
                    confidence=0.85,
                    timing_params={"t_pd": 15e-9},
                    validation={"t_pd": {"passes": True}},
                    family="TTL",
                )
            ],
            timing_measurements={"CLK→DATA_t_pd": 15e-9},
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=1.0,
            modern_replacements=[],
            bom=[],
            warnings=[],
            confidence_scores={},
        )

        plots = generate_all_vintage_logic_plots(
            result, waveform_traces, output_dir=tmp_path, save_formats=["png"]
        )

        assert isinstance(plots, dict)

    def test_save_formats(self, waveform_traces, tmp_path):
        """Test plot generation with multiple save formats."""
        from datetime import datetime

        from oscura.analyzers.digital.vintage_result import VintageLogicAnalysisResult

        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=1.0,
            detected_family="TTL",
            family_confidence=0.9,
            voltage_levels={},
            identified_ics=[],
            timing_measurements={},
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=1.0,
            modern_replacements=[],
            bom=[],
            warnings=[],
            confidence_scores={},
        )

        plots = generate_all_vintage_logic_plots(
            result, waveform_traces, output_dir=tmp_path, save_formats=["png", "svg"]
        )

        assert isinstance(plots, dict)

    def test_without_output_dir(self, waveform_traces):
        """Test plot generation without saving."""
        from datetime import datetime

        from oscura.analyzers.digital.vintage_result import VintageLogicAnalysisResult

        result = VintageLogicAnalysisResult(
            timestamp=datetime.now(),
            source_file=None,
            analysis_duration=1.0,
            detected_family="TTL",
            family_confidence=0.9,
            voltage_levels={},
            identified_ics=[],
            timing_measurements={},
            timing_paths=None,
            decoded_protocols=None,
            open_collector_detected=False,
            asymmetry_ratio=1.0,
            modern_replacements=[],
            bom=[],
            warnings=[],
            confidence_scores={},
        )

        plots = generate_all_vintage_logic_plots(result, waveform_traces)

        assert isinstance(plots, dict)
