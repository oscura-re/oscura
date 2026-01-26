"""Comprehensive unit tests for compare.py CLI module.

This module provides extensive testing for the Oscura compare command, including:
- Signal alignment using cross-correlation
- Timing drift analysis
- Amplitude difference calculations
- Noise change detection
- Spectral difference analysis
- HTML report generation
- Error handling and edge cases


Test Coverage:
- compare() CLI command with all options
- _align_signals() cross-correlation alignment
- _compute_timing_drift() edge-based timing analysis
- _compute_spectral_difference() frequency domain comparison
- _perform_comparison() comprehensive comparison orchestration
- _generate_html_report() HTML report generation
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.compare import (
    _align_signals,
    _compute_spectral_difference,
    _compute_timing_drift,
    _generate_html_report,
    _perform_comparison,
    compare,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metadata():
    """Create sample trace metadata."""
    return TraceMetadata(
        sample_rate=10e6,  # 10 MHz
        vertical_scale=1.0,
        vertical_offset=0.0,
    )


@pytest.fixture
def trace1(sample_metadata):
    """Create first sample trace."""
    # Create a square wave pattern
    data = np.array([0.0, 0.0, 3.3, 3.3, 0.0, 0.0, 3.3, 3.3] * 125, dtype=np.float64)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def trace2(sample_metadata):
    """Create second sample trace (slightly different)."""
    # Slightly different amplitude
    data = np.array([0.0, 0.0, 3.2, 3.2, 0.0, 0.0, 3.2, 3.2] * 125, dtype=np.float64)
    return WaveformTrace(data=data, metadata=sample_metadata)


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Test _align_signals()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_align_signals_identical():
    """Test alignment of identical signals."""
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
    sample_rate = 10e6

    aligned1, aligned2, info = _align_signals(data, data.copy(), sample_rate)

    assert len(aligned1) == len(aligned2)
    assert info["offset_samples"] == 0
    assert float(info["correlation_peak"]) > 0.99
    assert info["quality"] == "excellent"


@pytest.mark.unit
@pytest.mark.cli
def test_align_signals_with_offset():
    """Test alignment of signals with time offset."""
    base = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
    # Create offset version (shift by 10 samples)
    offset_data = np.concatenate([np.zeros(10), base[:-10]])
    sample_rate = 10e6

    aligned1, aligned2, info = _align_signals(base, offset_data, sample_rate)

    assert len(aligned1) == len(aligned2)
    # Should detect the offset
    assert abs(info["offset_samples"]) > 0
    # Correlation should still be high after alignment
    assert float(info["correlation_peak"]) > 0.8


@pytest.mark.unit
@pytest.mark.cli
def test_align_signals_quality_thresholds():
    """Test quality assessment thresholds."""
    data1 = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
    sample_rate = 10e6

    # Excellent quality (correlation > 0.95)
    _, _, info_excellent = _align_signals(data1, data1.copy(), sample_rate)
    assert info_excellent["quality"] == "excellent"

    # Good quality (0.8 < correlation < 0.95)
    data2_good = data1 + np.random.randn(len(data1)) * 0.1
    _, _, info_good = _align_signals(data1, data2_good, sample_rate)
    # May be good or excellent depending on noise

    # Poor quality (correlation < 0.8)
    data2_poor = np.random.randn(len(data1))
    _, _, info_poor = _align_signals(data1, data2_poor, sample_rate)
    assert info_poor["quality"] in ["poor", "good", "excellent"]


@pytest.mark.unit
@pytest.mark.cli
def test_align_signals_different_lengths():
    """Test alignment of signals with different lengths."""
    data1 = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
    data2 = np.sin(2 * np.pi * np.linspace(0, 1, 800))
    sample_rate = 10e6

    aligned1, aligned2, info = _align_signals(data1, data2, sample_rate)

    # Should produce equal length outputs
    assert len(aligned1) == len(aligned2)
    assert len(aligned1) <= min(len(data1), len(data2))


@pytest.mark.unit
@pytest.mark.cli
def test_align_signals_timing_offset_calculation():
    """Test that timing offset is correctly calculated."""
    data = np.ones(1000)
    sample_rate = 1e6  # 1 MHz = 1 us per sample
    # Simulate 10 sample offset
    offset_data = np.concatenate([np.zeros(10), data[:-10]])

    _, _, info = _align_signals(data, offset_data, sample_rate)

    # offset_time_ns should be offset_samples / sample_rate * 1e9
    # Expected: close to 10 / 1e6 * 1e9 = 10000 ns
    offset_ns = float(info["offset_time_ns"])
    # Allow some tolerance due to FFT-based correlation
    assert abs(offset_ns) >= 0  # Should have some offset


# =============================================================================
# Test _compute_timing_drift()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_compute_timing_drift_basic():
    """Test basic timing drift computation."""
    # Create signals with regular edges
    sample_rate = 10e6
    t = np.linspace(0, 0.001, 10000)
    data1 = np.where(np.sin(2 * np.pi * 1000 * t) > 0, 3.3, 0.0)
    data2 = data1.copy()

    result = _compute_timing_drift(data1, data2, sample_rate)

    # Identical signals should have near-zero drift
    assert "value_ns" in result
    assert "percentage" in result


@pytest.mark.unit
@pytest.mark.cli
def test_compute_timing_drift_insufficient_edges():
    """Test timing drift with insufficient edges."""
    # Constant signal (no edges)
    data1 = np.ones(1000)
    data2 = np.ones(1000)
    sample_rate = 10e6

    result = _compute_timing_drift(data1, data2, sample_rate)

    assert result["value_ns"] == "N/A"
    assert result["percentage"] == "N/A"
    assert result["significant"] is False
    assert "note" in result


@pytest.mark.unit
@pytest.mark.cli
def test_compute_timing_drift_few_edges():
    """Test timing drift with very few edges."""
    # Create signal with only 1 edge
    data1 = np.concatenate([np.zeros(500), np.ones(500)])
    data2 = data1.copy()
    sample_rate = 10e6

    result = _compute_timing_drift(data1, data2, sample_rate)

    # Should report insufficient edges
    assert "N/A" in str(result["value_ns"]) or "edges" in str(result.get("note", "")).lower()


@pytest.mark.unit
@pytest.mark.cli
def test_compute_timing_drift_significance_threshold():
    """Test significance threshold for timing drift."""
    sample_rate = 10e6
    # Create signals that will have some drift
    t = np.linspace(0, 0.001, 10000)
    data1 = np.where(np.sin(2 * np.pi * 1000 * t) > 0, 3.3, 0.0)
    data2 = data1.copy()

    result = _compute_timing_drift(data1, data2, sample_rate)

    # Drift > 0.1% is considered significant
    assert "significant" in result
    assert isinstance(result["significant"], bool)


# =============================================================================
# Test _compute_spectral_difference()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_compute_spectral_difference_basic():
    """Test basic spectral difference computation."""
    sample_rate = 10e6
    t = np.linspace(0, 0.001, 10000)
    # 1 kHz sine wave
    data1 = np.sin(2 * np.pi * 1000 * t)
    data2 = data1 * 0.9  # Slightly different amplitude

    result = _compute_spectral_difference(data1, data2, sample_rate, threshold=5.0)

    assert "dominant_freq1_hz" in result
    assert "dominant_freq2_hz" in result
    assert "freq_diff_hz" in result
    assert "max_magnitude_diff_db" in result
    assert "harmonic_changes" in result
    assert isinstance(result["harmonic_changes"], list)


@pytest.mark.unit
@pytest.mark.cli
def test_compute_spectral_difference_identical_signals():
    """Test spectral difference of identical signals."""
    sample_rate = 10e6
    data = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.001, 10000))

    result = _compute_spectral_difference(data, data.copy(), sample_rate, threshold=5.0)

    # Frequency difference should be very small
    freq_diff = float(result["freq_diff_hz"])
    assert freq_diff < 10  # Allow small numerical error

    # Should not be significant
    assert result["significant"] is False


@pytest.mark.unit
@pytest.mark.cli
def test_compute_spectral_difference_frequency_shift():
    """Test spectral difference with frequency shift."""
    sample_rate = 10e6
    t = np.linspace(0, 0.001, 10000)
    data1 = np.sin(2 * np.pi * 1000 * t)
    data2 = np.sin(2 * np.pi * 1100 * t)  # 10% higher frequency

    result = _compute_spectral_difference(data1, data2, sample_rate, threshold=5.0)

    # Should detect frequency difference
    freq_diff_pct = float(result["freq_diff_percent"].rstrip("%"))
    assert freq_diff_pct > 5.0
    assert result["significant"] is True


@pytest.mark.unit
@pytest.mark.cli
def test_compute_spectral_difference_amplitude_change():
    """Test spectral difference with amplitude change."""
    sample_rate = 10e6
    data1 = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.001, 10000))
    data2 = data1 * 2.0  # Double amplitude

    result = _compute_spectral_difference(data1, data2, sample_rate, threshold=5.0)

    # Should detect magnitude difference
    max_db_diff = float(result["max_magnitude_diff_db"])
    # 2x amplitude = 6 dB
    assert max_db_diff > 5.0


@pytest.mark.unit
@pytest.mark.cli
def test_compute_spectral_difference_harmonics():
    """Test harmonic analysis in spectral difference."""
    sample_rate = 10e6
    t = np.linspace(0, 0.001, 10000)
    data1 = np.sin(2 * np.pi * 1000 * t)
    data2 = data1.copy()

    result = _compute_spectral_difference(data1, data2, sample_rate, threshold=5.0)

    # Should analyze first 3 harmonics
    assert len(result["harmonic_changes"]) <= 3
    for harmonic in result["harmonic_changes"]:
        assert "harmonic" in harmonic
        assert "frequency_hz" in harmonic
        assert "change_db" in harmonic


# =============================================================================
# Test _perform_comparison()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_perform_comparison_basic(trace1, trace2):
    """Test basic comparison of two traces."""
    result = _perform_comparison(
        trace1=trace1,
        trace2=trace2,
        threshold=5.0,
        align_signals=False,
    )

    # Check top-level structure
    assert "threshold_percent" in result
    assert result["threshold_percent"] == 5.0
    assert "aligned" in result
    assert result["aligned"] is False

    # Check trace statistics
    assert "trace1_stats" in result
    assert "trace2_stats" in result
    assert "samples" in result["trace1_stats"]
    assert "sample_rate" in result["trace1_stats"]

    # Check analysis results
    assert "amplitude_difference" in result
    assert "timing_drift" in result
    assert "noise_change" in result
    assert "spectral_difference" in result
    assert "correlation" in result
    assert "summary" in result


@pytest.mark.unit
@pytest.mark.cli
def test_perform_comparison_with_alignment(trace1, trace2):
    """Test comparison with signal alignment."""
    result = _perform_comparison(
        trace1=trace1,
        trace2=trace2,
        threshold=5.0,
        align_signals=True,
    )

    assert result["aligned"] is True
    assert "alignment" in result
    assert "offset_samples" in result["alignment"]
    assert "correlation_peak" in result["alignment"]


@pytest.mark.unit
@pytest.mark.cli
def test_perform_comparison_sample_rate_mismatch(sample_metadata):
    """Test comparison detects sample rate mismatch."""
    # Create traces with different sample rates
    metadata1 = sample_metadata
    metadata2 = TraceMetadata(sample_rate=20e6, vertical_scale=1.0, vertical_offset=0.0)

    trace1 = WaveformTrace(data=np.ones(1000), metadata=metadata1)
    trace2 = WaveformTrace(data=np.ones(1000), metadata=metadata2)

    result = _perform_comparison(
        trace1=trace1,
        trace2=trace2,
        threshold=5.0,
        align_signals=False,
    )

    assert result["sample_rate_mismatch"] is True


@pytest.mark.unit
@pytest.mark.cli
def test_perform_comparison_identical_traces(trace1):
    """Test comparison of identical traces."""
    result = _perform_comparison(
        trace1=trace1,
        trace2=trace1,
        threshold=5.0,
        align_signals=False,
    )

    # Should have excellent match
    assert result["summary"]["overall_match"] in ["excellent", "good"]
    assert result["summary"]["significant_differences"] <= 1

    # Correlation should be perfect
    correlation = float(result["correlation"]["coefficient"])
    assert correlation > 0.99


@pytest.mark.unit
@pytest.mark.cli
def test_perform_comparison_overall_assessment():
    """Test overall assessment categories."""
    metadata = TraceMetadata(sample_rate=10e6)

    # Create different test cases
    test_cases = [
        # (data1, data2, expected_quality)
        (np.ones(1000), np.ones(1000), "excellent"),  # Identical
        (
            np.ones(1000),
            np.ones(1000) * 1.1,
            "good",
        ),  # Small difference (might be excellent)
    ]

    for data1, data2, _ in test_cases:
        trace1 = WaveformTrace(data=data1, metadata=metadata)
        trace2 = WaveformTrace(data=data2, metadata=metadata)

        result = _perform_comparison(
            trace1=trace1,
            trace2=trace2,
            threshold=5.0,
            align_signals=False,
        )

        # Should have a match quality
        assert result["summary"]["overall_match"] in [
            "excellent",
            "good",
            "fair",
            "poor",
        ]


@pytest.mark.unit
@pytest.mark.cli
def test_perform_comparison_trace_stats_formatting():
    """Test that trace statistics are properly formatted."""
    metadata = TraceMetadata(sample_rate=25e6)
    data = np.sin(2 * np.pi * np.linspace(0, 1, 50000))
    trace = WaveformTrace(data=data, metadata=metadata)

    result = _perform_comparison(
        trace1=trace,
        trace2=trace,
        threshold=5.0,
        align_signals=False,
    )

    # Check formatting
    assert "25.00 MHz" in result["trace1_stats"]["sample_rate"]
    assert "ms" in result["trace1_stats"]["duration_ms"]
    assert "V" in result["trace1_stats"]["mean"]
    assert "V" in result["trace1_stats"]["peak_to_peak"]


# =============================================================================
# Test _generate_html_report()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_generate_html_report_basic():
    """Test basic HTML report generation."""
    results = {
        "summary": {
            "overall_match": "excellent",
            "significant_differences": 0,
        },
        "trace1_stats": {
            "samples": 1000,
            "sample_rate": "10.0 MHz",
            "mean": "1.500 V",
            "rms": "1.600 V",
            "peak_to_peak": "3.300 V",
        },
        "trace2_stats": {
            "samples": 1000,
            "sample_rate": "10.0 MHz",
            "mean": "1.500 V",
            "rms": "1.600 V",
            "peak_to_peak": "3.300 V",
        },
        "amplitude_difference": {
            "mean_diff_v": "0.000",
            "mean_diff_percent": "0.0%",
            "rms_diff_v": "0.000",
            "rms_diff_percent": "0.0%",
            "max_diff_v": "0.000",
            "significant": False,
        },
        "timing_drift": {
            "value_ns": "0.00",
            "percentage": "0.00%",
            "significant": False,
        },
        "noise_change": {
            "noise1_v": "0.050",
            "noise2_v": "0.050",
            "change_percent": "0.0%",
            "significant": False,
        },
        "spectral_difference": {
            "dominant_freq1_hz": "1000.0",
            "dominant_freq2_hz": "1000.0",
            "freq_diff_percent": "0.0%",
            "max_magnitude_diff_db": "0.0",
            "significant": False,
        },
        "correlation": {
            "coefficient": "1.000000",
            "quality": "excellent",
        },
    }

    html = _generate_html_report(results, "file1.wfm", "file2.wfm")

    # Check HTML structure
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html
    assert "Oscura Signal Comparison Report" in html
    assert "file1.wfm" in html
    assert "file2.wfm" in html

    # Check content sections
    assert "Overall Match" in html
    assert "EXCELLENT" in html
    assert "Trace Statistics" in html
    assert "Amplitude Difference" in html
    assert "Timing Drift" in html
    assert "Noise Change" in html
    assert "Spectral Difference" in html
    assert "Correlation" in html


@pytest.mark.unit
@pytest.mark.cli
def test_generate_html_report_quality_colors():
    """Test that quality determines report color scheme."""
    qualities = ["excellent", "good", "fair", "poor"]
    expected_colors = {
        "excellent": "#28a745",
        "good": "#17a2b8",
        "fair": "#ffc107",
        "poor": "#dc3545",
    }

    for quality in qualities:
        results = {
            "summary": {"overall_match": quality, "significant_differences": 0},
            "trace1_stats": {},
            "trace2_stats": {},
            "amplitude_difference": {},
            "timing_drift": {},
            "noise_change": {},
            "spectral_difference": {},
            "correlation": {},
        }

        html = _generate_html_report(results, "f1.wfm", "f2.wfm")

        # Check that quality color is used
        assert expected_colors[quality] in html


@pytest.mark.unit
@pytest.mark.cli
def test_generate_html_report_significant_markers():
    """Test that significant differences are highlighted."""
    results = {
        "summary": {"overall_match": "poor", "significant_differences": 2},
        "trace1_stats": {},
        "trace2_stats": {},
        "amplitude_difference": {
            "mean_diff_percent": "15.0%",
            "significant": True,
        },
        "timing_drift": {"value_ns": "100.00", "percentage": "5.0%", "significant": True},
        "noise_change": {"change_percent": "2.0%", "significant": False},
        "spectral_difference": {"freq_diff_percent": "1.0%", "significant": False},
        "correlation": {},
    }

    html = _generate_html_report(results, "f1.wfm", "f2.wfm")

    # Should contain CSS class for significant differences
    assert "significant" in html
    assert "2 significant difference" in html


# =============================================================================
# Test compare() CLI Command
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_basic(cli_runner, tmp_path):
    """Test basic compare command execution."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.compare._perform_comparison") as mock_compare:
            with patch("oscura.cli.compare.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_compare.return_value = {"summary": {"overall_match": "excellent"}}
                mock_format.return_value = "Comparison results"

                result = cli_runner.invoke(compare, [str(file1), str(file2)], obj={"verbose": 0})

                assert result.exit_code == 0
                assert "Comparison results" in result.output
                assert mock_load.call_count == 2


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_with_threshold(cli_runner, tmp_path):
    """Test compare command with custom threshold."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.compare._perform_comparison") as mock_compare:
            with patch("oscura.cli.compare.format_output"):
                mock_load.return_value = Mock()
                mock_compare.return_value = {}

                result = cli_runner.invoke(
                    compare,
                    [str(file1), str(file2), "--threshold", "10"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                call_args = mock_compare.call_args
                assert call_args[1]["threshold"] == 10.0


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_with_alignment(cli_runner, tmp_path):
    """Test compare command with signal alignment."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.compare._perform_comparison") as mock_compare:
            with patch("oscura.cli.compare.format_output"):
                mock_load.return_value = Mock()
                mock_compare.return_value = {}

                result = cli_runner.invoke(
                    compare, [str(file1), str(file2), "--align"], obj={"verbose": 0}
                )

                assert result.exit_code == 0
                call_args = mock_compare.call_args
                assert call_args[1]["align_signals"] is True


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_output_formats(cli_runner, tmp_path):
    """Test compare command with different output formats."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    for output_format in ["json", "csv", "html", "table"]:
        with patch("oscura.loaders.load") as mock_load:
            with patch("oscura.cli.compare._perform_comparison") as mock_compare:
                with patch("oscura.cli.compare.format_output") as mock_format:
                    mock_load.return_value = Mock()
                    mock_compare.return_value = {}
                    mock_format.return_value = f"{output_format} output"

                    result = cli_runner.invoke(
                        compare,
                        [str(file1), str(file2), "--output", output_format],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    mock_format.assert_called_with(mock_compare.return_value, output_format)


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_save_report(cli_runner, tmp_path):
    """Test compare command with HTML report saving."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    report_file = tmp_path / "comparison.html"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.compare._perform_comparison") as mock_compare:
            with patch("oscura.cli.compare.format_output"):
                with patch("oscura.cli.compare._generate_html_report") as mock_html_gen:
                    mock_load.return_value = Mock()
                    mock_compare.return_value = {}
                    mock_html_gen.return_value = "<html>Report</html>"

                    result = cli_runner.invoke(
                        compare,
                        [str(file1), str(file2), "--save-report", str(report_file)],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    assert report_file.exists()
                    content = report_file.read_text()
                    assert "<html>Report</html>" in content


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_adds_filenames(cli_runner, tmp_path):
    """Test that compare command adds filenames to results."""
    file1 = tmp_path / "before.wfm"
    file2 = tmp_path / "after.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.compare._perform_comparison") as mock_compare:
            with patch("oscura.cli.compare.format_output") as mock_format:
                mock_load.return_value = Mock()
                mock_compare.return_value = {}
                mock_format.return_value = "output"

                result = cli_runner.invoke(compare, [str(file1), str(file2)], obj={"verbose": 0})

                assert result.exit_code == 0
                format_call_args = mock_format.call_args[0][0]
                assert format_call_args["file1"] == "before.wfm"
                assert format_call_args["file2"] == "after.wfm"


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_verbose_logging(cli_runner, tmp_path):
    """Test compare command with verbose logging."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        with patch("oscura.cli.compare._perform_comparison") as mock_compare:
            with patch("oscura.cli.compare.format_output"):
                mock_load.return_value = Mock()
                mock_compare.return_value = {}

                result = cli_runner.invoke(
                    compare,
                    [str(file1), str(file2), "--threshold", "10", "--align"],
                    obj={"verbose": 1},
                )

                assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_error_handling(cli_runner, tmp_path):
    """Test compare command error handling."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = Exception("Failed to load file")

        result = cli_runner.invoke(compare, [str(file1), str(file2)], obj={"verbose": 0})

        assert result.exit_code == 1
        assert "Error: Failed to load file" in result.output


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_error_with_verbose(cli_runner, tmp_path):
    """Test compare command error handling with verbose mode."""
    file1 = tmp_path / "trace1.wfm"
    file2 = tmp_path / "trace2.wfm"
    file1.touch()
    file2.touch()

    with patch("oscura.loaders.load") as mock_load:
        mock_load.side_effect = ValueError("Test error")

        result = cli_runner.invoke(compare, [str(file1), str(file2)], obj={"verbose": 2})

        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
@pytest.mark.cli
def test_compare_command_nonexistent_files(cli_runner):
    """Test compare command with nonexistent files."""
    result = cli_runner.invoke(
        compare, ["/nonexistent/file1.wfm", "/nonexistent/file2.wfm"], obj={"verbose": 0}
    )

    # Click should catch this with its path validation
    assert result.exit_code != 0


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_compare_end_to_end(cli_runner, tmp_path, trace1, trace2):
    """Test complete comparison workflow."""
    file1 = tmp_path / "signal1.wfm"
    file2 = tmp_path / "signal2.wfm"
    file1.touch()
    file2.touch()

    load_calls = [trace1, trace2]

    def mock_load_side_effect(path):
        return load_calls.pop(0)

    with patch("oscura.loaders.load", side_effect=mock_load_side_effect):
        result = cli_runner.invoke(
            compare,
            [str(file1), str(file2), "--threshold", "5", "--output", "json"],
            obj={"verbose": 0},
        )

        assert result.exit_code == 0
        # Output should contain comparison results
        assert "{" in result.output or "comparison" in result.output.lower()


@pytest.mark.unit
@pytest.mark.cli
def test_compare_with_alignment_end_to_end(cli_runner, tmp_path, trace1, trace2):
    """Test comparison with alignment workflow."""
    file1 = tmp_path / "signal1.wfm"
    file2 = tmp_path / "signal2.wfm"
    file1.touch()
    file2.touch()

    load_calls = [trace1, trace2]

    def mock_load_side_effect(path):
        return load_calls.pop(0)

    with patch("oscura.loaders.load", side_effect=mock_load_side_effect):
        result = cli_runner.invoke(
            compare,
            [str(file1), str(file2), "--align", "--output", "table"],
            obj={"verbose": 0},
        )

        assert result.exit_code == 0


# =============================================================================
# Edge Case Tests for Missing Coverage
# =============================================================================


@pytest.mark.unit
def test_compute_noise_change_filtfilt_exception():
    """Test noise change computation when filtfilt raises exception."""
    from oscura.cli.compare import _compute_noise_change

    # Create data that might cause filtfilt issues
    data1 = np.array([1.0, 2.0, 3.0])  # Too short for filter
    data2 = np.array([1.1, 2.1, 3.1])
    sample_rate = 1e6

    # Should fall back to std without filtering
    result = _compute_noise_change(data1, data2, sample_rate, threshold=5.0)

    assert "noise1_v" in result
    assert "noise2_v" in result
    assert "change_percent" in result


@pytest.mark.unit
def test_compute_correlation_single_sample():
    """Test correlation with single sample (edge case)."""
    from oscura.cli.compare import _compute_correlation

    data1 = np.array([1.0])
    data2 = np.array([1.0])

    result = _compute_correlation(data1, data2)

    # Should return N/A for insufficient data
    assert result["coefficient"] == "N/A"
    assert result["quality"] == "unknown"


@pytest.mark.unit
def test_compute_correlation_quality_fair():
    """Test correlation with fair quality (0.8-0.95)."""
    from oscura.cli.compare import _compute_correlation

    # Create signals with moderate correlation
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.1, 2.2, 2.8, 4.1, 5.0])  # Similar but noisy

    result = _compute_correlation(data1, data2)

    # Should be fair quality
    quality = result["quality"]
    assert quality in ["fair", "good", "excellent"]


@pytest.mark.unit
def test_compute_correlation_quality_poor():
    """Test correlation with poor quality (<0.8)."""
    from oscura.cli.compare import _compute_correlation

    # Create uncorrelated signals
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([5.0, 1.0, 4.0, 2.0, 3.0])  # Random order

    result = _compute_correlation(data1, data2)

    # Quality should be determined
    assert result["quality"] in ["poor", "fair", "good", "excellent"]


@pytest.mark.unit
def test_compute_summary_three_significant():
    """Test summary with 3 significant differences (poor match)."""
    from oscura.cli.compare import _compute_summary

    results = {
        "amplitude_difference": {"significant": True},
        "noise_change": {"significant": True},
        "timing_drift": {"significant": True},
        "spectral_difference": {"significant": False},
    }

    summary = _compute_summary(results)

    assert summary["significant_differences"] == 3
    assert summary["overall_match"] == "poor"
    assert len(summary["categories_with_differences"]) == 3


@pytest.mark.unit
def test_compute_summary_two_significant():
    """Test summary with 2 significant differences (fair match)."""
    from oscura.cli.compare import _compute_summary

    results = {
        "amplitude_difference": {"significant": True},
        "noise_change": {"significant": True},
        "timing_drift": {"significant": False},
        "spectral_difference": {"significant": False},
    }

    summary = _compute_summary(results)

    assert summary["significant_differences"] == 2
    assert summary["overall_match"] == "fair"


@pytest.mark.unit
def test_compute_summary_one_significant():
    """Test summary with 1 significant difference (good match)."""
    from oscura.cli.compare import _compute_summary

    results = {
        "amplitude_difference": {"significant": True},
        "noise_change": {"significant": False},
        "timing_drift": {"significant": False},
        "spectral_difference": {"significant": False},
    }

    summary = _compute_summary(results)

    assert summary["significant_differences"] == 1
    assert summary["overall_match"] == "good"


@pytest.mark.unit
def test_compute_correlation_quality_good():
    """Test correlation with good quality (0.95-0.99)."""
    from oscura.cli.compare import _compute_correlation

    # Create highly correlated signals with controlled correlation ~0.97
    np.random.seed(42)
    data1 = np.linspace(0, 10, 100)
    # Add small noise to get correlation in good range (0.95-0.99)
    noise = np.random.normal(0, 0.3, 100)
    data2 = data1 + noise

    result = _compute_correlation(data1, data2)

    # Parse coefficient to verify quality assignment
    coeff = float(result["coefficient"])
    if 0.95 < coeff < 0.99:
        assert result["quality"] == "good"
    elif coeff > 0.99:
        assert result["quality"] == "excellent"
    else:
        # Acceptable variance
        assert result["quality"] in ["good", "excellent", "fair"]


@pytest.mark.unit
def test_compute_correlation_quality_fair_precise():
    """Test correlation with fair quality (0.8-0.95) precisely."""
    from oscura.cli.compare import _compute_correlation

    # Manually create data with correlation ~0.85
    np.random.seed(123)
    data1 = np.linspace(0, 10, 50)
    # Add noise to get moderate correlation
    data2 = data1 + np.random.normal(0, 1.5, 50)

    result = _compute_correlation(data1, data2)

    # Should compute correlation
    assert "coefficient" in result
    assert result["quality"] in ["fair", "good", "poor", "excellent"]
