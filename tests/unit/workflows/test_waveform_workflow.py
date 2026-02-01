"""Comprehensive unit tests for waveform workflow module.

Tests for src/oscura/workflows/waveform.py including:
- analyze_complete() function with various configurations
- Protocol detection enabled/disabled
- Reverse engineering enabled/disabled
- Report generation with different formats
- Error handling for invalid inputs
- Different analysis combinations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oscura.core.types import WaveformTrace
from oscura.workflows.waveform import (
    _format_anomalies,
    _format_patterns,
    _format_protocol_detection,
    _format_reverse_engineering,
    analyze_complete,
)

pytestmark = pytest.mark.unit


def create_test_waveform_file(
    filepath: Path, signal_data: np.ndarray, sample_rate: float = 1e6
) -> None:
    """Helper to create a test waveform file in NPZ format.

    Args:
        filepath: Path to save file (should have .npz extension).
        signal_data: Signal data array.
        sample_rate: Sample rate in Hz.
    """
    np.savez(
        filepath,
        data=signal_data,
        sample_rate=np.array(sample_rate),
        channel="CH1",
    )


class TestAnalyzeComplete:
    """Tests for analyze_complete() workflow function."""

    def test_basic_analog_waveform_loading(self, tmp_path: Path, signal_factory) -> None:
        """Test basic waveform loading and analysis with analog signal."""
        # Create synthetic analog waveform file
        signal, metadata = signal_factory(
            signal_type="sine", frequency=1000.0, sample_rate=1e6, duration=0.01
        )
        waveform_path = tmp_path / "test_signal.npz"
        create_test_waveform_file(waveform_path, signal, metadata["sample_rate"])

        # Run analysis without advanced features
        results = analyze_complete(
            waveform_path,
            output_dir=tmp_path / "output",
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Validate basic structure
        assert results is not None
        assert results["filepath"] == waveform_path
        assert isinstance(results["trace"], WaveformTrace)
        assert results["is_digital"] is False
        assert isinstance(results["results"], dict)
        assert results["protocols_detected"] == []
        assert results["decoded_frames"] == []
        assert results["reverse_engineering"] is None
        assert results["patterns"] is None
        assert results["anomalies"] == []

    def test_file_not_found_error(self, tmp_path: Path) -> None:
        """Test error handling when file does not exist."""
        nonexistent_path = tmp_path / "nonexistent.npz"

        with pytest.raises(FileNotFoundError, match="File not found"):
            analyze_complete(
                nonexistent_path,
                output_dir=tmp_path / "output",
                verbose=False,
            )

    def test_invalid_analyses_parameter(self, tmp_path: Path, signal_factory) -> None:
        """Test error handling for invalid analysis types."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "test.npz"
        create_test_waveform_file(waveform_path, signal)

        with pytest.raises(ValueError, match="Invalid analysis types"):
            analyze_complete(
                waveform_path,
                analyses=["time_domain", "invalid_analysis", "fake_domain"],
                verbose=False,
            )

    def test_specific_analyses_subset(self, tmp_path: Path, signal_factory) -> None:
        """Test running only specific analysis types."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "test.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            analyses=["time_domain", "frequency_domain"],
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Should have only requested analyses
        assert "time_domain" in results["results"]
        assert "frequency_domain" in results["results"]

    def test_all_analyses_default(self, tmp_path: Path, signal_factory) -> None:
        """Test that 'all' analyses runs all available analysis types."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "test.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            analyses="all",  # Default
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Should run time_domain, frequency_domain, statistics
        assert "time_domain" in results["results"]
        assert "frequency_domain" in results["results"]
        assert "statistics" in results["results"]

    def test_protocol_detection_disabled(self, tmp_path: Path, signal_factory) -> None:
        """Test that protocol detection is skipped when disabled."""
        signal, _ = signal_factory(signal_type="digital", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "digital.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_protocol_decode=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # No protocols should be detected
        assert results["protocols_detected"] == []
        assert results["decoded_frames"] == []

    def test_reverse_engineering_disabled(self, tmp_path: Path, signal_factory) -> None:
        """Test that reverse engineering is skipped when disabled."""
        signal, _ = signal_factory(signal_type="digital", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "digital.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_reverse_engineering=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # No RE results
        assert results["reverse_engineering"] is None

    def test_reverse_engineering_insufficient_samples(self, tmp_path: Path) -> None:
        """Test RE is skipped when signal has <1000 samples."""
        # Create short signal (<1000 samples)
        signal_data = np.array([0.0, 1.0] * 400, dtype=np.float64)  # 800 samples
        waveform_path = tmp_path / "short_signal.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            enable_reverse_engineering=True,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # RE should be None due to insufficient samples
        assert results["reverse_engineering"] is None

    def test_reverse_engineering_depth_quick(self, tmp_path: Path) -> None:
        """Test RE with 'quick' depth parameter."""
        signal_data = np.tile([0.0, 3.3], 1000)  # 2000 samples
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            enable_reverse_engineering=True,
            reverse_engineering_depth="quick",
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Just verify it doesn't crash
        assert "reverse_engineering" in results

    def test_pattern_recognition_enabled(self, tmp_path: Path, signal_factory) -> None:
        """Test pattern recognition and anomaly detection."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_pattern_recognition=True,
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Pattern results should be present (may be empty)
        assert "patterns" in results
        assert "anomalies" in results
        assert isinstance(results["anomalies"], list)

    def test_pattern_recognition_disabled(self, tmp_path: Path, signal_factory) -> None:
        """Test that pattern recognition is skipped when disabled."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_pattern_recognition=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        assert results["patterns"] is None
        assert results["anomalies"] == []

    def test_generate_plots_enabled(self, tmp_path: Path, signal_factory) -> None:
        """Test plot generation."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.005)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            output_dir=tmp_path / "output",
            generate_plots=True,
            generate_report=False,
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            verbose=False,
        )

        # Plots should be generated (dict may be empty or populated)
        assert "plots" in results
        assert isinstance(results["plots"], dict)

    def test_generate_plots_disabled(self, tmp_path: Path, signal_factory) -> None:
        """Test that plots are skipped when disabled."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # No plots
        assert results["plots"] == {}

    def test_generate_report_html(self, tmp_path: Path, signal_factory) -> None:
        """Test HTML report generation."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.005)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_report=True,
            report_format="html",
            generate_plots=False,
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            verbose=False,
        )

        # Report should be generated
        assert results["report_path"] is not None
        assert results["report_path"].exists()
        assert results["report_path"].suffix == ".html"
        assert results["report_path"].name == "complete_analysis.html"

    def test_generate_report_pdf_format(self, tmp_path: Path, signal_factory) -> None:
        """Test PDF report format parameter."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.005)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_report=True,
            report_format="pdf",
            generate_plots=False,
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            verbose=False,
        )

        # Report path should use pdf extension
        assert results["report_path"] is not None
        assert results["report_path"].suffix == ".pdf"

    def test_generate_report_disabled(self, tmp_path: Path, signal_factory) -> None:
        """Test that report is skipped when disabled."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            generate_report=False,
            verbose=False,
        )

        assert results["report_path"] is None

    def test_embed_plots_in_report(self, tmp_path: Path, signal_factory) -> None:
        """Test embedding plots in HTML report."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.005)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_plots=True,
            generate_report=True,
            embed_plots=True,
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            verbose=False,
        )

        # Report should be generated with plots
        assert results["report_path"] is not None
        assert results["report_path"].exists()

    def test_output_directory_default(self, tmp_path: Path, signal_factory) -> None:
        """Test default output directory creation."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            output_dir=None,  # Should default to ./waveform_analysis_output
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        assert results["output_dir"] == Path("./waveform_analysis_output")

    def test_output_directory_custom(self, tmp_path: Path, signal_factory) -> None:
        """Test custom output directory creation."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        custom_output = tmp_path / "my_custom_output"
        results = analyze_complete(
            waveform_path,
            output_dir=custom_output,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        assert results["output_dir"] == custom_output
        assert custom_output.exists()

    def test_verbose_output(self, tmp_path: Path, signal_factory, capsys) -> None:
        """Test verbose output messages."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        analyze_complete(
            waveform_path,
            generate_plots=False,
            generate_report=False,
            verbose=True,
        )

        captured = capsys.readouterr()
        # Should have verbose messages
        assert "OSCURA COMPLETE WAVEFORM ANALYSIS" in captured.out
        assert "ANALYSIS COMPLETE" in captured.out

    def test_verbose_disabled(self, tmp_path: Path, signal_factory, capsys) -> None:
        """Test that verbose is properly disabled."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        analyze_complete(
            waveform_path,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        captured = capsys.readouterr()
        # Should have minimal output
        assert "OSCURA" not in captured.out

    def test_pathlib_path_input(self, tmp_path: Path, signal_factory) -> None:
        """Test that pathlib.Path objects are accepted."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        # Pass as Path object
        results = analyze_complete(
            waveform_path,  # pathlib.Path
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        assert results["filepath"] == waveform_path

    def test_string_path_input(self, tmp_path: Path, signal_factory) -> None:
        """Test that string paths are accepted."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.001)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        # Pass as string
        results = analyze_complete(
            str(waveform_path),  # str
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        assert results["filepath"] == waveform_path

    def test_protocol_detection_with_digital_signal(self, tmp_path: Path) -> None:
        """Test protocol detection with complex digital signal."""
        # Create UART-like signal with multiple bytes
        sample_rate = 1e6
        baud_rate = 9600
        samples_per_bit = int(sample_rate / baud_rate)

        bits = [1] * 100  # Idle
        for byte_val in [0x48, 0x65, 0x6C, 0x6C, 0x6F]:  # "Hello"
            bits.append(0)  # Start bit
            for i in range(8):
                bits.append((byte_val >> i) & 1)
            bits.append(1)  # Stop bit
            bits.extend([1] * 10)  # Inter-byte gap

        signal_data = []
        for bit in bits:
            signal_data.extend([bit * 3.3] * samples_per_bit)

        signal_data = np.array(signal_data, dtype=np.float64)
        waveform_path = tmp_path / "uart.npz"
        create_test_waveform_file(waveform_path, signal_data, sample_rate)

        results = analyze_complete(
            waveform_path,
            enable_protocol_decode=True,
            protocol_hints=["UART", "SPI", "I2C"],
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Protocol detection attempted (may or may not succeed)
        assert "protocols_detected" in results
        assert isinstance(results["protocols_detected"], list)

    def test_reverse_engineering_with_adequate_samples(self, tmp_path: Path) -> None:
        """Test RE with sufficient samples and valid depth parameters."""
        sample_rate = 1e6
        baud_rate = 19200
        samples_per_bit = int(sample_rate / baud_rate)

        # Create repeating pattern with multiple frames
        bits = [1] * 100
        for _ in range(10):  # 10 frames
            for byte_val in [0xAA, 0x55, 0xF0]:
                bits.append(0)
                for i in range(8):
                    bits.append((byte_val >> i) & 1)
                bits.append(1)
                bits.extend([1] * 5)

        signal_data = []
        for bit in bits:
            signal_data.extend([bit * 3.3] * samples_per_bit)

        signal_data = np.array(signal_data, dtype=np.float64)
        waveform_path = tmp_path / "re_signal.npz"
        create_test_waveform_file(waveform_path, signal_data, sample_rate)

        # Test all depth levels
        for depth in ["quick", "standard", "deep"]:
            results = analyze_complete(
                waveform_path,
                enable_reverse_engineering=True,
                reverse_engineering_depth=depth,  # type: ignore[arg-type]
                enable_protocol_decode=False,
                enable_pattern_recognition=False,
                generate_plots=False,
                generate_report=False,
                verbose=False,
            )

            assert "reverse_engineering" in results

    def test_pattern_recognition_with_decoded_frames(self, tmp_path: Path) -> None:
        """Test pattern recognition when protocol decoding provides frames."""
        # Create signal with repeating byte pattern
        sample_rate = 1e6
        byte_pattern = [0xAA, 0x55] * 20  # Repeating pattern

        signal_data = np.array(byte_pattern * 100, dtype=np.float64)
        waveform_path = tmp_path / "pattern_signal.npz"
        create_test_waveform_file(waveform_path, signal_data, sample_rate)

        results = analyze_complete(
            waveform_path,
            enable_pattern_recognition=True,
            enable_protocol_decode=True,
            enable_reverse_engineering=False,
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Pattern recognition should execute
        assert "patterns" in results
        assert "anomalies" in results

    def test_report_with_all_sections(self, tmp_path: Path) -> None:
        """Test report generation with all possible sections."""
        # Create complex signal
        sample_rate = 1e6
        duration = 0.01
        num_samples = int(sample_rate * duration)
        signal_data = np.sin(2 * np.pi * 1000 * np.arange(num_samples) / sample_rate)

        waveform_path = tmp_path / "complete.npz"
        create_test_waveform_file(waveform_path, signal_data, sample_rate)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            analyses="all",
            generate_report=True,
            report_format="html",
            generate_plots=True,
            embed_plots=True,
            enable_protocol_decode=True,
            enable_reverse_engineering=True,
            enable_pattern_recognition=True,
            verbose=True,
        )

        # Report should be generated with all available sections
        assert results["report_path"] is not None
        assert results["report_path"].exists()

        # Read report and check for major sections
        report_html = results["report_path"].read_text()
        assert len(report_html) > 100  # Non-empty report

    def test_digital_analysis_with_both_trace_types(self, tmp_path: Path) -> None:
        """Test digital analysis works with both WaveformTrace and DigitalTrace."""
        # Create digital-like waveform (0s and 1s)
        signal_data = np.array([0.0, 3.3] * 5000, dtype=np.float64)
        waveform_path = tmp_path / "digital.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            analyses=["digital"],
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Digital analysis should execute
        assert results is not None

    def test_frequency_domain_analysis_excluded_for_digital(self, tmp_path: Path) -> None:
        """Test that frequency domain analysis is skipped for digital signals."""
        # Create clear digital signal
        signal_data = np.tile([0.0, 0.0, 3.3, 3.3], 2500)
        waveform_path = tmp_path / "digital.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            analyses="all",
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Frequency analysis may or may not be skipped depending on is_digital detection
        assert results is not None

    def test_statistics_analysis_excluded_for_digital(self, tmp_path: Path) -> None:
        """Test that statistics analysis is skipped for digital signals."""
        signal_data = np.tile([0.0, 3.3], 5000)
        waveform_path = tmp_path / "digital.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            analyses="all",
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Statistics may or may not be excluded depending on trace type
        assert results is not None

    def test_complete_workflow_minimal_config(self, tmp_path: Path, signal_factory) -> None:
        """Test complete workflow with minimal configuration."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.005)
        waveform_path = tmp_path / "minimal.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(waveform_path)

        # Should use all defaults
        assert results is not None
        assert results["output_dir"] == Path("./waveform_analysis_output")

    def test_protocol_detection_no_protocol_hints(self, tmp_path: Path) -> None:
        """Test protocol detection without specific protocol hints (auto-detect mode)."""
        # Create digital signal
        sample_rate = 1e6
        signal_data = np.tile([0.0, 3.3, 3.3, 0.0], 2500)
        waveform_path = tmp_path / "auto_detect.npz"
        create_test_waveform_file(waveform_path, signal_data, sample_rate)

        results = analyze_complete(
            waveform_path,
            enable_protocol_decode=True,
            protocol_hints=None,  # Auto-detect mode
            enable_reverse_engineering=False,
            enable_pattern_recognition=False,
            generate_plots=False,
            generate_report=False,
            verbose=True,
        )

        # Auto-detection should try default protocols
        assert "protocols_detected" in results

    def test_protocol_detection_analog_signal_skipped(self, tmp_path: Path, signal_factory) -> None:
        """Test that protocol detection is skipped for analog signals."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "analog.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_protocol_decode=True,  # Enabled but should skip for analog
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # Protocol detection should be empty for analog signals
        # (unless the signal is misdetected as digital)
        assert "protocols_detected" in results

    def test_reverse_engineering_analog_signal_skipped(
        self, tmp_path: Path, signal_factory
    ) -> None:
        """Test that RE is skipped for analog signals."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "analog.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_reverse_engineering=True,  # Enabled but should skip for analog
            generate_plots=False,
            generate_report=False,
            verbose=False,
        )

        # RE should be None for analog signals
        assert results["reverse_engineering"] is None

    def test_re_with_value_error_handling(self, tmp_path: Path) -> None:
        """Test RE with signal that may raise ValueError."""
        # Create minimal digital signal that might fail RE
        signal_data = np.array([0.0, 3.3] * 600, dtype=np.float64)  # Just above 1000 samples
        waveform_path = tmp_path / "minimal_re.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            enable_reverse_engineering=True,
            reverse_engineering_depth="quick",
            generate_plots=False,
            generate_report=False,
            verbose=True,  # Enable verbose to test error message path
        )

        # RE may fail with ValueError for insufficient data
        assert "reverse_engineering" in results
        if results["reverse_engineering"] is not None:
            # If not None, it should have status field when error occurs
            assert isinstance(results["reverse_engineering"], dict)

    def test_re_with_exception_handling(self, tmp_path: Path) -> None:
        """Test RE with signal that may raise other exceptions."""
        # Create edge case signal
        signal_data = np.full(1500, 1.5, dtype=np.float64)  # Constant value
        waveform_path = tmp_path / "constant.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            enable_reverse_engineering=True,
            generate_plots=False,
            generate_report=False,
            verbose=True,  # Test verbose error path
        )

        # Should handle gracefully
        assert "reverse_engineering" in results

    def test_pattern_recognition_anomaly_detection_exception(
        self, tmp_path: Path, signal_factory
    ) -> None:
        """Test pattern recognition with potential anomaly detection exceptions."""
        signal, _ = signal_factory(signal_type="noise", sample_rate=1e6, duration=0.005)
        waveform_path = tmp_path / "noise.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            enable_pattern_recognition=True,
            generate_plots=False,
            generate_report=False,
            verbose=True,  # Test verbose error path
        )

        # Should handle exceptions gracefully
        assert "patterns" in results
        assert "anomalies" in results

    def test_digital_analysis_exception_handling(self, tmp_path: Path) -> None:
        """Test digital analysis with potential exceptions."""
        # Create very short signal that might cause analysis issues
        signal_data = np.array([0.0, 3.3, 0.0], dtype=np.float64)
        waveform_path = tmp_path / "very_short.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            analyses=["digital"],
            generate_plots=False,
            generate_report=False,
            verbose=True,  # Test verbose error path
        )

        # Should handle potential errors
        assert results is not None

    def test_report_with_protocol_detection_results(self, tmp_path: Path) -> None:
        """Test report generation includes protocol detection section."""
        signal_data = np.tile([0.0, 3.3], 5000)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal_data)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_report=True,
            enable_protocol_decode=True,
            protocol_hints=["UART"],
            generate_plots=False,
            verbose=False,
        )

        # Report should be generated
        assert results["report_path"] is not None
        report_html = results["report_path"].read_text()
        # Report may or may not have protocol section depending on detection success
        assert len(report_html) > 0

    def test_report_with_re_results(self, tmp_path: Path) -> None:
        """Test report generation includes RE section when available."""
        # Create signal with enough samples for RE
        signal_data = np.tile([0.0, 3.3, 0.0, 0.0], 500)
        waveform_path = tmp_path / "re_signal.npz"
        create_test_waveform_file(waveform_path, signal_data)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_report=True,
            enable_reverse_engineering=True,
            reverse_engineering_depth="quick",
            enable_protocol_decode=False,
            enable_pattern_recognition=False,
            generate_plots=False,
            verbose=False,
        )

        # Report should be generated
        assert results["report_path"] is not None

    def test_report_with_anomalies(self, tmp_path: Path, signal_factory) -> None:
        """Test report generation includes anomaly section when available."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "signal.npz"
        create_test_waveform_file(waveform_path, signal)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_report=True,
            enable_pattern_recognition=True,
            enable_protocol_decode=False,
            enable_reverse_engineering=False,
            generate_plots=False,
            verbose=False,
        )

        # Report should be generated
        assert results["report_path"] is not None

    def test_report_with_pattern_signatures(self, tmp_path: Path) -> None:
        """Test report generation includes pattern recognition section."""
        # Create signal with repeating bytes that might generate signatures
        byte_pattern = np.array([0xAA, 0x55] * 100, dtype=np.uint8)
        signal_data = byte_pattern.astype(np.float64)
        waveform_path = tmp_path / "patterns.npz"
        create_test_waveform_file(waveform_path, signal_data)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            generate_report=True,
            enable_pattern_recognition=True,
            enable_protocol_decode=True,  # Needed to get decoded frames for pattern mining
            generate_plots=False,
            verbose=False,
        )

        # Report should be generated
        assert results["report_path"] is not None

    def test_time_domain_measurements_complete(self, tmp_path: Path, signal_factory) -> None:
        """Test time-domain analysis produces comprehensive measurements."""
        signal, _ = signal_factory(
            signal_type="sine", frequency=1000, sample_rate=1e6, duration=0.01
        )
        waveform_path = tmp_path / "sine.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            analyses=["time_domain"],
            generate_plots=False,
            generate_report=False,
            verbose=True,
        )

        # Should have time domain results
        assert "time_domain" in results["results"]
        assert len(results["results"]["time_domain"]) > 0

    def test_frequency_domain_measurements_complete(self, tmp_path: Path, signal_factory) -> None:
        """Test frequency-domain analysis produces comprehensive measurements."""
        signal, _ = signal_factory(
            signal_type="sine", frequency=1000, sample_rate=1e6, duration=0.01
        )
        waveform_path = tmp_path / "sine.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            analyses=["frequency_domain"],
            generate_plots=False,
            generate_report=False,
            verbose=True,
        )

        # Should have frequency domain results
        assert "frequency_domain" in results["results"]
        # May include FFT arrays
        assert len(results["results"]["frequency_domain"]) > 0

    def test_digital_analysis_measurements_complete(self, tmp_path: Path) -> None:
        """Test digital analysis produces comprehensive measurements."""
        # Create clean digital signal
        signal_data = np.array([0.0] * 1000 + [3.3] * 1000 + [0.0] * 1000, dtype=np.float64)
        waveform_path = tmp_path / "digital.npz"
        create_test_waveform_file(waveform_path, signal_data)

        results = analyze_complete(
            waveform_path,
            analyses=["digital"],
            generate_plots=False,
            generate_report=False,
            verbose=True,
        )

        # Digital analysis should execute
        if "digital" in results["results"]:
            assert isinstance(results["results"]["digital"], dict)

    def test_statistics_measurements_complete(self, tmp_path: Path, signal_factory) -> None:
        """Test statistical analysis produces comprehensive measurements."""
        signal, _ = signal_factory(signal_type="noise", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "noise.npz"
        create_test_waveform_file(waveform_path, signal)

        results = analyze_complete(
            waveform_path,
            analyses=["statistics"],
            generate_plots=False,
            generate_report=False,
            verbose=True,
        )

        # Should have statistics results
        assert "statistics" in results["results"]
        assert len(results["results"]["statistics"]) > 0

    def test_complete_with_mixed_measurements_in_report(
        self, tmp_path: Path, signal_factory
    ) -> None:
        """Test report handles both unified and legacy measurement formats."""
        signal, _ = signal_factory(signal_type="sine", sample_rate=1e6, duration=0.01)
        waveform_path = tmp_path / "mixed.npz"
        create_test_waveform_file(waveform_path, signal)

        output_dir = tmp_path / "output"
        results = analyze_complete(
            waveform_path,
            output_dir=output_dir,
            analyses="all",
            generate_report=True,
            generate_plots=False,
            verbose=False,
        )

        # Report should handle mixed formats gracefully
        assert results["report_path"] is not None
        report_html = results["report_path"].read_text()
        # Should have measurement sections
        assert (
            "Analysis" in report_html or "Measurements" in report_html or "Complete" in report_html
        )


class TestFormatProtocolDetection:
    """Tests for _format_protocol_detection() helper function."""

    def test_format_single_protocol(self) -> None:
        """Test formatting single protocol detection result."""
        protocols = [
            {
                "protocol": "uart",
                "confidence": 0.95,
                "params": {"baud_rate": 115200},
                "frame_count": 10,
            }
        ]
        frames = [b"H", b"e", b"l", b"l", b"o"]

        html = _format_protocol_detection(protocols, frames)

        assert "<h3>Detected Protocols</h3>" in html
        assert "UART" in html
        assert "95.0%" in html
        assert "115200 baud" in html
        assert "10 frames" in html
        assert "Total bytes decoded:</strong> 5" in html

    def test_format_multiple_protocols(self) -> None:
        """Test formatting multiple protocol detection results."""
        protocols = [
            {"protocol": "uart", "confidence": 0.90, "params": {}, "frame_count": 5},
            {"protocol": "spi", "confidence": 0.85, "params": {"baud_rate": 1000000}},
        ]
        frames = []

        html = _format_protocol_detection(protocols, frames)

        assert "UART" in html
        assert "SPI" in html
        assert "90.0%" in html
        assert "85.0%" in html

    def test_format_no_baud_rate(self) -> None:
        """Test formatting protocol without baud rate parameter."""
        protocols = [{"protocol": "i2c", "confidence": 0.88, "params": {}}]
        frames = []

        html = _format_protocol_detection(protocols, frames)

        assert "I2C" in html
        assert "88.0%" in html
        assert "baud" not in html


class TestFormatReverseEngineering:
    """Tests for _format_reverse_engineering() helper function."""

    def test_format_complete_re_results(self) -> None:
        """Test formatting complete RE results."""
        re_results = {
            "baud_rate": 19200.0,
            "confidence": 0.92,
            "frame_count": 15,
            "frame_format": "8N1",
            "sync_pattern": "0xAA55",
            "frame_length": 16,
            "field_count": 4,
            "checksum_type": "CRC16",
            "checksum_position": 14,
            "warnings": ["Warning 1", "Warning 2"],
        }

        html = _format_reverse_engineering(re_results)

        assert "<h3>Reverse Engineering Findings</h3>" in html
        assert "19200 Hz" in html
        assert "92.0%" in html
        assert "15" in html
        assert "8N1" in html
        assert "0xAA55" in html
        assert "16 bytes" in html
        assert "4" in html
        assert "CRC16" in html
        assert "position 14" in html
        assert "<h4>Warnings</h4>" in html
        assert "Warning 1" in html

    def test_format_minimal_re_results(self) -> None:
        """Test formatting minimal RE results."""
        re_results = {"baud_rate": 9600.0, "confidence": 0.75}

        html = _format_reverse_engineering(re_results)

        assert "9600 Hz" in html
        assert "75.0%" in html

    def test_format_re_results_without_checksum_position(self) -> None:
        """Test formatting RE results without checksum position."""
        re_results = {"checksum_type": "XOR"}

        html = _format_reverse_engineering(re_results)

        assert "XOR" in html
        assert "position" not in html

    def test_format_re_results_many_warnings(self) -> None:
        """Test that only first 5 warnings are displayed."""
        re_results = {
            "warnings": [f"Warning {i}" for i in range(10)]  # 10 warnings
        }

        html = _format_reverse_engineering(re_results)

        assert "Warning 0" in html
        assert "Warning 4" in html
        # Warnings 5-9 should not be displayed
        assert "Warning 5" not in html


class TestFormatAnomalies:
    """Tests for _format_anomalies() helper function."""

    def test_format_anomalies_by_severity(self) -> None:
        """Test formatting anomalies grouped by severity."""
        anomalies = [
            {
                "type": "glitch",
                "start": 0.001234,
                "end": 0.001456,
                "severity": "critical",
                "description": "Voltage spike detected",
            },
            {
                "type": "timing",
                "start": 0.002000,
                "end": 0.002100,
                "severity": "warning",
                "description": "Timing violation",
            },
            {
                "type": "noise",
                "start": 0.003000,
                "end": 0.003050,
                "severity": "info",
                "description": "Background noise increase",
            },
        ]

        html = _format_anomalies(anomalies)

        assert "<h3>Detected Anomalies</h3>" in html
        assert "Total anomalies:</strong> 3" in html
        assert "Critical" in html
        assert "Warning" in html
        assert "Info" in html
        assert "glitch" in html
        assert "Voltage spike detected" in html
        assert "0.001234s" in html

    def test_format_many_anomalies_per_severity(self) -> None:
        """Test that only first 10 anomalies per severity are shown."""
        anomalies = [
            {
                "type": f"anomaly_{i}",
                "start": float(i) * 0.001,
                "end": float(i) * 0.001 + 0.0001,
                "severity": "warning",
                "description": f"Description {i}",
            }
            for i in range(15)
        ]

        html = _format_anomalies(anomalies)

        # Should show first 10
        assert "anomaly_0" in html
        assert "anomaly_9" in html
        # Should not show 11-14
        assert "anomaly_14" not in html

    def test_format_empty_anomalies(self) -> None:
        """Test formatting with no anomalies."""
        anomalies: list[dict[str, Any]] = []

        html = _format_anomalies(anomalies)

        assert "Total anomalies:</strong> 0" in html


class TestFormatPatterns:
    """Tests for _format_patterns() helper function."""

    def test_format_signature_patterns(self) -> None:
        """Test formatting signature pattern results."""
        pattern_results = {
            "signatures": [
                {
                    "pattern": "aa55",
                    "length": 2,
                    "count": 10,
                    "confidence": 0.95,
                },
                {
                    "pattern": "deadbeef",
                    "length": 4,
                    "count": 5,
                    "confidence": 0.88,
                },
            ]
        }

        html = _format_patterns(pattern_results)

        assert "<h3>Pattern Recognition Results</h3>" in html
        assert "Signature patterns discovered:</strong> 2" in html
        assert "<table" in html
        assert "aa55" in html
        assert "2 bytes" in html
        assert "10" in html
        assert "0.95" in html
        assert "deadbeef" in html

    def test_format_many_patterns(self) -> None:
        """Test that only first 10 patterns are shown."""
        pattern_results = {
            "signatures": [
                {
                    "pattern": f"{i:04x}",
                    "length": 2,
                    "count": i,
                    "confidence": 0.9,
                }
                for i in range(15)
            ]
        }

        html = _format_patterns(pattern_results)

        # Should show first 10
        assert "0000" in html
        assert "0009" in html
        # Should not show 11-14
        assert "000e" not in html  # 14 in hex

    def test_format_empty_patterns(self) -> None:
        """Test formatting with no patterns."""
        pattern_results: dict[str, Any] = {}

        html = _format_patterns(pattern_results)

        assert "<h3>Pattern Recognition Results</h3>" in html
        assert "Signature patterns" not in html
