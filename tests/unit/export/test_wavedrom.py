"""Tests for WaveDrom timing diagram generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace
from oscura.export.wavedrom import (
    WaveDromBuilder,
    WaveDromEdge,
    WaveDromSignal,
    export_wavedrom,
    from_digital_trace,
)

pytestmark = [pytest.mark.unit, pytest.mark.exporter]


class TestWaveDromSignal:
    """Test WaveDromSignal dataclass."""

    def test_basic_signal(self):
        """Test basic signal creation."""
        signal = WaveDromSignal(name="CLK", wave="p......")
        assert signal.name == "CLK"
        assert signal.wave == "p......"
        assert signal.data is None
        assert signal.node is None

    def test_signal_with_data(self):
        """Test signal with data labels."""
        signal = WaveDromSignal(name="DATA", wave="=.=.=", data=["A", "B", "C"])
        assert signal.name == "DATA"
        assert signal.data == ["A", "B", "C"]

    def test_signal_with_node(self):
        """Test signal with node markers."""
        signal = WaveDromSignal(name="D", wave="01..", node="A.B")
        assert signal.node == "A.B"


class TestWaveDromEdge:
    """Test WaveDromEdge dataclass."""

    def test_basic_edge(self):
        """Test basic edge creation."""
        edge = WaveDromEdge(from_node="A", to_node="B", label="t_su = 10ns")
        assert edge.from_node == "A"
        assert edge.to_node == "B"
        assert edge.label == "t_su = 10ns"
        assert edge.style == ""

    def test_edge_with_style(self):
        """Test edge with custom style."""
        edge = WaveDromEdge(from_node="A", to_node="B", label="delay", style="-~")
        assert edge.style == "-~"


class TestWaveDromBuilder:
    """Test WaveDromBuilder class."""

    def test_initialization_default(self):
        """Test builder initialization with defaults."""
        builder = WaveDromBuilder()
        assert builder.title is None
        assert builder.time_scale == 1e-9
        assert len(builder.signals) == 0
        assert len(builder.edges) == 0
        assert "hscale" in builder.config

    def test_initialization_with_title(self):
        """Test builder initialization with title."""
        builder = WaveDromBuilder(title="Test Diagram")
        assert builder.title == "Test Diagram"

    def test_initialization_with_time_scale(self):
        """Test builder with custom time scale."""
        builder = WaveDromBuilder(time_scale=1e-6)  # Microseconds
        assert builder.time_scale == 1e-6

    def test_add_clock_basic(self):
        """Test adding a basic clock signal."""
        builder = WaveDromBuilder()
        builder.add_clock("CLK", period=100e-9)

        assert len(builder.signals) == 1
        assert builder.signals[0].name == "CLK"
        assert "p" in builder.signals[0].wave  # Has pulsing pattern

    def test_add_clock_with_initial_high(self):
        """Test clock with initial high state."""
        builder = WaveDromBuilder()
        builder.add_clock("CLK", period=100e-9, initial_state="high")

        assert "n" in builder.signals[0].wave  # Inverted pulsing

    def test_add_clock_with_start_time(self):
        """Test clock with delayed start."""
        builder = WaveDromBuilder()
        builder.add_clock("CLK", period=100e-9, start_time=50e-9)

        wave = builder.signals[0].wave
        assert wave.startswith(".")  # Leading dots for delay

    def test_add_clock_with_duration(self):
        """Test clock with specified duration."""
        builder = WaveDromBuilder()
        builder.add_clock("CLK", period=100e-9, duration=500e-9)

        # Should have ~5 periods
        wave = builder.signals[0].wave
        assert wave.count("p") == 5

    def test_add_signal_with_edges(self):
        """Test adding signal with edge timestamps."""
        builder = WaveDromBuilder()
        builder.add_signal("DATA", edges=[10e-9, 50e-9, 150e-9])

        assert len(builder.signals) == 1
        assert builder.signals[0].name == "DATA"
        assert "1" in builder.signals[0].wave  # Has rising edge
        assert "0" in builder.signals[0].wave  # Has falling edge

    def test_add_signal_with_wave_string(self):
        """Test adding signal with direct wave string."""
        builder = WaveDromBuilder()
        builder.add_signal("CS", wave_string="1.0.1.0")

        assert builder.signals[0].wave == "1.0.1.0"

    def test_add_signal_with_data(self):
        """Test adding signal with data labels."""
        builder = WaveDromBuilder()
        builder.add_signal("ADDR", wave_string="=.=.=", data=["0x00", "0x01", "0x02"])

        assert builder.signals[0].data == ["0x00", "0x01", "0x02"]

    def test_add_signal_with_nodes(self):
        """Test adding signal with node markers."""
        builder = WaveDromBuilder()
        builder.add_signal("D", wave_string="01..", nodes="A.B")

        assert builder.signals[0].node == "A.B"

    def test_add_signal_without_edges_or_wave(self):
        """Test that signal requires edges or wave_string."""
        builder = WaveDromBuilder()

        with pytest.raises(ValueError, match="Must provide either edges or wave_string"):
            builder.add_signal("DATA")

    def test_add_data_bus(self):
        """Test adding a data bus with transitions."""
        builder = WaveDromBuilder()
        transitions = [
            (10e-9, "0xA5"),
            (50e-9, "0x3C"),
            (100e-9, "0xFF"),
        ]
        builder.add_data_bus("BUS", transitions=transitions)

        signal = builder.signals[0]
        assert signal.name == "BUS"
        assert signal.data == ["0xA5", "0x3C", "0xFF"]
        assert "=" in signal.wave

    def test_add_data_bus_with_initial_value(self):
        """Test data bus with custom initial value."""
        builder = WaveDromBuilder()
        builder.add_data_bus("BUS", transitions=[(10e-9, "0x00")], initial_value="z")

        signal = builder.signals[0]
        assert signal.data == ["0x00"]

    def test_add_data_bus_empty_transitions(self):
        """Test that data bus requires transitions."""
        builder = WaveDromBuilder()

        with pytest.raises(ValueError, match="Must provide at least one transition"):
            builder.add_data_bus("BUS", transitions=[])

    def test_add_data_bus_sorting(self):
        """Test that data bus sorts transitions by time."""
        builder = WaveDromBuilder()
        transitions = [
            (100e-9, "C"),
            (10e-9, "A"),
            (50e-9, "B"),
        ]
        builder.add_data_bus("BUS", transitions=transitions)

        # Data should be in time order
        assert builder.signals[0].data == ["A", "B", "C"]

    def test_add_arrow(self):
        """Test adding arrow annotation."""
        builder = WaveDromBuilder()
        builder.add_arrow(10e-9, 40e-9, "t_su = 30ns")

        assert len(builder.edges) == 1
        assert builder.edges[0].label == "t_su = 30ns"

    def test_add_multiple_arrows(self):
        """Test adding multiple arrows."""
        builder = WaveDromBuilder()
        builder.add_arrow(10e-9, 40e-9, "setup")
        builder.add_arrow(50e-9, 80e-9, "hold")

        assert len(builder.edges) == 2
        assert builder.edges[0].label == "setup"
        assert builder.edges[1].label == "hold"

    def test_edges_to_wave_empty(self):
        """Test edge conversion with no edges."""
        builder = WaveDromBuilder()
        wave = builder._edges_to_wave([])

        assert wave == "0"

    def test_edges_to_wave_single_rising(self):
        """Test edge conversion with single rising edge."""
        builder = WaveDromBuilder()
        wave = builder._edges_to_wave([10e-9])

        assert "1" in wave  # Has rising edge

    def test_edges_to_wave_alternating(self):
        """Test edge conversion with alternating edges."""
        builder = WaveDromBuilder()
        wave = builder._edges_to_wave([10e-9, 50e-9, 100e-9])

        assert "1" in wave  # Rising
        assert "0" in wave  # Falling

    def test_to_dict_empty(self):
        """Test dictionary export with no signals."""
        builder = WaveDromBuilder()
        result = builder.to_dict()

        assert "signal" in result
        assert result["signal"] == []

    def test_to_dict_with_title(self):
        """Test dictionary export with title."""
        builder = WaveDromBuilder(title="Test")
        result = builder.to_dict()

        assert "head" in result
        assert result["head"]["text"] == "Test"

    def test_to_dict_with_signals(self):
        """Test dictionary export with signals."""
        builder = WaveDromBuilder()
        builder.add_clock("CLK", period=100e-9)
        builder.add_signal("DATA", wave_string="0.1.0")

        result = builder.to_dict()
        assert len(result["signal"]) == 2
        assert result["signal"][0]["name"] == "CLK"
        assert result["signal"][1]["name"] == "DATA"

    def test_to_dict_with_edges(self):
        """Test dictionary export with edges."""
        builder = WaveDromBuilder()
        builder.add_arrow(10e-9, 40e-9, "delay")

        result = builder.to_dict()
        assert "edge" in result
        assert len(result["edge"]) == 1
        assert "delay" in result["edge"][0]

    def test_to_dict_with_config(self):
        """Test dictionary export includes config."""
        builder = WaveDromBuilder()
        result = builder.to_dict()

        assert "config" in result
        assert result["config"]["hscale"] == 2

    def test_to_json(self):
        """Test JSON string export."""
        builder = WaveDromBuilder(title="Test")
        builder.add_signal("CLK", wave_string="p.....")

        json_str = builder.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["head"]["text"] == "Test"
        assert len(data["signal"]) == 1

    def test_to_json_custom_indent(self):
        """Test JSON export with custom indentation."""
        builder = WaveDromBuilder()
        builder.add_signal("CLK", wave_string="p")

        json_str = builder.to_json(indent=4)

        # Check indentation (4 spaces)
        assert "    " in json_str

    def test_save(self, tmp_path):
        """Test saving to file."""
        builder = WaveDromBuilder(title="Test")
        builder.add_clock("CLK", period=100e-9)

        output_file = tmp_path / "test.json"
        builder.save(output_file)

        # Verify file exists and is valid JSON
        assert output_file.exists()
        with output_file.open() as f:
            data = json.load(f)
        assert data["head"]["text"] == "Test"


class TestFromDigitalTrace:
    """Test from_digital_trace function."""

    def test_basic_conversion(self):
        """Test converting digital trace to WaveDrom signal."""
        # Create simple digital trace
        data = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e9))

        signal = from_digital_trace(trace, name="TEST")

        assert signal.name == "TEST"
        assert "1" in signal.wave  # Has transitions

    def test_conversion_with_start_time(self):
        """Test conversion with start time offset."""
        data = np.array([0, 1, 0, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e9))

        signal = from_digital_trace(trace, name="TEST", start_time=10e-9)

        assert signal.name == "TEST"

    def test_conversion_with_duration_limit(self):
        """Test conversion with duration limit."""
        data = np.array([0, 1, 0, 1, 0, 1] * 10, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e9))

        signal = from_digital_trace(trace, name="TEST", duration=10e-9)

        # Should only include transitions within duration
        assert signal.name == "TEST"

    def test_conversion_no_transitions(self):
        """Test conversion of constant signal."""
        data = np.array([1, 1, 1, 1], dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e9))

        signal = from_digital_trace(trace, name="CONST")

        assert signal.name == "CONST"


class TestExportWavedrom:
    """Test export_wavedrom function."""

    def test_export_basic(self, tmp_path, signal_builder):
        """Test basic export of signals."""
        from oscura.core.types import TraceMetadata

        # Create test signals
        clk_data = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
        clk_trace = WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6))

        data_data = signal_builder.square_wave(frequency=500e3, duration=0.001, sample_rate=1e6)
        data_trace = WaveformTrace(data=data_data, metadata=TraceMetadata(sample_rate=1e6))

        signals = {
            "CLK": clk_trace,
            "DATA": data_trace,
        }

        output_file = tmp_path / "export.json"
        export_wavedrom(signals, output_file)

        # Verify file created and is valid JSON
        assert output_file.exists()
        with output_file.open() as f:
            data = json.load(f)
        assert "signal" in data
        assert len(data["signal"]) == 2

    def test_export_with_title(self, tmp_path, signal_builder):
        """Test export with diagram title."""
        from oscura.core.types import TraceMetadata

        clk_data = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
        clk_trace = WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6))

        signals = {"CLK": clk_trace}

        output_file = tmp_path / "titled.json"
        export_wavedrom(signals, output_file, title="Test Timing")

        with output_file.open() as f:
            data = json.load(f)
        assert data["head"]["text"] == "Test Timing"

    def test_export_with_custom_time_scale(self, tmp_path, signal_builder):
        """Test export with custom time scale."""
        from oscura.core.types import TraceMetadata

        clk_data = signal_builder.square_wave(frequency=1e3, duration=0.001, sample_rate=1e6)
        clk_trace = WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6))

        signals = {"CLK": clk_trace}

        output_file = tmp_path / "scaled.json"
        export_wavedrom(signals, output_file, time_scale=1e-6)  # Microseconds

        assert output_file.exists()

    def test_export_with_annotations(self, tmp_path, signal_builder):
        """Test export with timing annotations."""
        from oscura.core.types import TraceMetadata

        clk_data = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
        clk_trace = WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6))

        signals = {"CLK": clk_trace}
        annotations = [
            (10e-9, 40e-9, "t_su = 30ns"),
            (50e-9, 80e-9, "t_h = 30ns"),
        ]

        output_file = tmp_path / "annotated.json"
        export_wavedrom(signals, output_file, annotations=annotations)

        with output_file.open() as f:
            data = json.load(f)
        assert "edge" in data
        assert len(data["edge"]) == 2

    def test_export_digital_trace(self, tmp_path):
        """Test export with DigitalTrace."""
        # Create digital trace
        data = np.array([0, 0, 1, 1, 0, 0, 1, 1] * 100, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e9))

        signals = {"DIGI": trace}

        output_file = tmp_path / "digital.json"
        export_wavedrom(signals, output_file)

        # Should handle digital trace without error
        assert output_file.exists()
        with output_file.open() as f:
            data = json.load(f)
        assert len(data["signal"]) == 1

    def test_export_multiple_signals(self, tmp_path, signal_builder):
        """Test export with multiple signal types."""
        from oscura.core.types import TraceMetadata

        # Create various signals
        clk = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
        data1 = signal_builder.square_wave(frequency=500e3, duration=0.001, sample_rate=1e6)
        data2 = signal_builder.square_wave(frequency=250e3, duration=0.001, sample_rate=1e6)

        signals = {
            "CLK": WaveformTrace(data=clk, metadata=TraceMetadata(sample_rate=1e6)),
            "DATA1": WaveformTrace(data=data1, metadata=TraceMetadata(sample_rate=1e6)),
            "DATA2": WaveformTrace(data=data2, metadata=TraceMetadata(sample_rate=1e6)),
        }

        output_file = tmp_path / "multiple.json"
        export_wavedrom(signals, output_file, title="Multi-Signal")

        with output_file.open() as f:
            data = json.load(f)
        assert len(data["signal"]) == 3
        assert data["head"]["text"] == "Multi-Signal"

    def test_export_path_string(self, tmp_path, signal_builder):
        """Test export with path as string."""
        from oscura.core.types import TraceMetadata

        clk_data = signal_builder.square_wave(frequency=1e6, duration=0.001, sample_rate=1e6)
        clk_trace = WaveformTrace(data=clk_data, metadata=TraceMetadata(sample_rate=1e6))

        signals = {"CLK": clk_trace}

        output_file = str(tmp_path / "string_path.json")
        export_wavedrom(signals, output_file)

        assert Path(output_file).exists()
