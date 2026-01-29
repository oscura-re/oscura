"""Comprehensive unit tests for batch.py CLI module.

This module provides extensive testing for the Oscura batch command, including:
- Batch processing with glob patterns
- Parallel vs sequential execution
- Different analysis types (characterize, decode, spectrum)
- Error handling and continue-on-error behavior
- Output formatting and summary generation
- CSV export functionality


Test Coverage:
- batch() command with all options
- _perform_batch_analysis() for parallel and sequential modes
- _generate_summary() for aggregating results
- _save_summary() for CSV export
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from oscura.cli.batch import (
    _generate_summary,
    _perform_batch_analysis,
    _save_summary,
    batch,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.usefixtures("reset_logging_state"),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_trace() -> MagicMock:
    """Create a mock WaveformTrace object."""
    trace = MagicMock()
    trace.data = np.random.randn(1000)
    trace.metadata.sample_rate = 1e6
    return trace


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_files(temp_dir: Path) -> list[str]:
    """Create sample test files."""
    files = []
    for i in range(3):
        file_path = temp_dir / f"test_{i}.wfm"
        file_path.write_text(f"sample data {i}")
        files.append(str(file_path))
    return files


# =============================================================================
# Test batch() command
# =============================================================================


@pytest.mark.unit
def test_batch_no_files_matched(runner: CliRunner) -> None:
    """Test batch command when no files match the pattern."""
    with patch("oscura.cli.batch.glob.glob", return_value=[]):
        result = runner.invoke(
            batch,
            ["nonexistent_*.wfm", "--analysis", "characterize"],
            obj={"verbose": 0},
        )

        assert result.exit_code == 1
        assert "No files matched pattern" in result.output


@pytest.mark.unit
def test_batch_characterize_success(
    runner: CliRunner, sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test batch command with characterize analysis."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
            mock_analyze.return_value = [
                {
                    "file": "test_0.wfm",
                    "status": "success",
                    "analysis_type": "characterize",
                    "samples": 1000,
                    "sample_rate": "1.0 MHz",
                    "rise_time": "10.50 ns",
                    "fall_time": "12.30 ns",
                }
            ]

            with patch("oscura.cli.batch.format_output") as mock_format:
                mock_format.return_value = "Formatted output"

                result = runner.invoke(
                    batch,
                    ["*.wfm", "--analysis", "characterize"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                assert "Formatted output" in result.output
                mock_analyze.assert_called_once()


@pytest.mark.unit
def test_batch_decode_analysis(
    runner: CliRunner, sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test batch command with decode analysis."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
            mock_analyze.return_value = [
                {
                    "file": "test_0.wfm",
                    "status": "success",
                    "analysis_type": "decode",
                    "samples": 1000,
                    "sample_rate": "1.0 MHz",
                    "protocol": "UART",
                    "confidence": "85%",
                }
            ]

            with patch("oscura.cli.batch.format_output") as mock_format:
                mock_format.return_value = "Decoded results"

                result = runner.invoke(
                    batch,
                    ["*.wfm", "--analysis", "decode"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                assert "Decoded results" in result.output


@pytest.mark.unit
def test_batch_spectrum_analysis(
    runner: CliRunner, sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test batch command with spectrum analysis."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
            mock_analyze.return_value = [
                {
                    "file": "test_0.wfm",
                    "status": "success",
                    "analysis_type": "spectrum",
                    "samples": 1000,
                    "sample_rate": "1.0 MHz",
                    "peak_frequency": "0.250 MHz",
                    "thd": "-40.2 dB",
                }
            ]

            with patch("oscura.cli.batch.format_output") as mock_format:
                mock_format.return_value = "Spectrum results"

                result = runner.invoke(
                    batch,
                    ["*.wfm", "--analysis", "spectrum"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                assert "Spectrum results" in result.output


@pytest.mark.unit
def test_batch_parallel_processing(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command with parallel processing."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
            mock_analyze.return_value = [
                {"file": f"test_{i}.wfm", "status": "success"} for i in range(len(sample_files))
            ]

            with patch("oscura.cli.batch.format_output", return_value="Results"):
                result = runner.invoke(
                    batch,
                    ["*.wfm", "--analysis", "characterize", "--parallel", "4"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                args = mock_analyze.call_args
                assert args[1]["parallel"] == 4


@pytest.mark.unit
def test_batch_save_summary(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command with save-summary option."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        summary_path = f.name

    try:
        with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
            with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
                mock_analyze.return_value = [{"file": "test_0.wfm", "status": "success"}]

                with patch("oscura.cli.batch._save_summary") as mock_save:
                    with patch("oscura.cli.batch.format_output", return_value="Results"):
                        result = runner.invoke(
                            batch,
                            [
                                "*.wfm",
                                "--analysis",
                                "characterize",
                                "--save-summary",
                                summary_path,
                            ],
                            obj={"verbose": 0},
                        )

                        assert result.exit_code == 0
                        mock_save.assert_called_once()
    finally:
        Path(summary_path).unlink(missing_ok=True)


@pytest.mark.unit
def test_batch_output_formats(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command with different output formats."""
    formats = ["json", "csv", "html", "table"]

    for fmt in formats:
        with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
            with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
                mock_analyze.return_value = [{"file": "test.wfm", "status": "success"}]

                with patch("oscura.cli.batch.format_output") as mock_format:
                    mock_format.return_value = f"Output in {fmt}"

                    result = runner.invoke(
                        batch,
                        ["*.wfm", "--analysis", "characterize", "--output", fmt],
                        obj={"verbose": 0},
                    )

                    assert result.exit_code == 0
                    mock_format.assert_called_once()
                    args = mock_format.call_args
                    assert args[0][1] == fmt


@pytest.mark.unit
def test_batch_verbose_logging(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command with verbose logging."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
            mock_analyze.return_value = [{"file": "test.wfm", "status": "success"}]

            with patch("oscura.cli.batch.format_output", return_value="Results"):
                with patch("oscura.cli.batch.logger") as mock_logger:
                    result = runner.invoke(
                        batch,
                        ["*.wfm", "--analysis", "characterize"],
                        obj={"verbose": 1},
                    )

                    assert result.exit_code == 0
                    assert mock_logger.info.called


@pytest.mark.unit
def test_batch_error_handling(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command error handling."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch(
            "oscura.cli.batch._perform_batch_analysis",
            side_effect=Exception("Test error"),
        ):
            result = runner.invoke(
                batch,
                ["*.wfm", "--analysis", "characterize"],
                obj={"verbose": 0},
            )

            assert result.exit_code == 1
            assert "Error:" in result.output


@pytest.mark.unit
def test_batch_error_handling_verbose(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command error handling with verbose mode."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch(
            "oscura.cli.batch._perform_batch_analysis",
            side_effect=ValueError("Test error"),
        ):
            result = runner.invoke(
                batch,
                ["*.wfm", "--analysis", "characterize"],
                obj={"verbose": 2},
            )

            assert result.exit_code == 1
            assert result.exception is not None


@pytest.mark.unit
def test_batch_continue_on_error(runner: CliRunner, sample_files: list[str]) -> None:
    """Test batch command with continue-on-error flag."""
    with patch("oscura.cli.batch.glob.glob", return_value=sample_files):
        with patch("oscura.cli.batch._perform_batch_analysis") as mock_analyze:
            mock_analyze.return_value = [
                {"file": "test_0.wfm", "status": "success"},
                {"file": "test_1.wfm", "status": "error", "error": "Load failed"},
            ]

            with patch("oscura.cli.batch.format_output", return_value="Results"):
                result = runner.invoke(
                    batch,
                    [
                        "*.wfm",
                        "--analysis",
                        "characterize",
                        "--continue-on-error",
                    ],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                args = mock_analyze.call_args
                assert args[1]["continue_on_error"] is True


# =============================================================================
# Test _perform_batch_analysis() - Sequential Processing
# =============================================================================


@pytest.mark.unit
def test_perform_batch_analysis_sequential_characterize(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test sequential batch analysis with characterize."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                results = _perform_batch_analysis(
                    files=sample_files[:1],
                    analysis_type="characterize",
                    parallel=1,
                    continue_on_error=False,
                    verbose=0,
                )

                assert len(results) == 1
                assert results[0]["status"] == "success"
                assert results[0]["analysis_type"] == "characterize"
                assert "rise_time" in results[0]
                assert "fall_time" in results[0]


@pytest.mark.unit
def test_perform_batch_analysis_sequential_decode(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test sequential batch analysis with decode."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch(
            "oscura.inference.detect_protocol",
            return_value={"protocol": "UART", "confidence": 0.85},
        ):
            results = _perform_batch_analysis(
                files=sample_files[:1],
                analysis_type="decode",
                parallel=1,
                continue_on_error=False,
                verbose=0,
            )

            assert len(results) == 1
            assert results[0]["status"] == "success"
            assert results[0]["analysis_type"] == "decode"
            assert results[0]["protocol"] == "UART"
            assert results[0]["confidence"] == "85%"


@pytest.mark.unit
def test_perform_batch_analysis_sequential_spectrum(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test sequential batch analysis with spectrum."""
    freqs = np.linspace(0, 500e3, 100)
    mags = np.random.randn(100)

    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.spectral.fft", return_value=(freqs, mags)):
            with patch("oscura.analyzers.waveform.spectral.thd", return_value=-40.2):
                results = _perform_batch_analysis(
                    files=sample_files[:1],
                    analysis_type="spectrum",
                    parallel=1,
                    continue_on_error=False,
                    verbose=0,
                )

                assert len(results) == 1
                assert results[0]["status"] == "success"
                assert results[0]["analysis_type"] == "spectrum"
                assert "peak_frequency" in results[0]
                assert "thd" in results[0]


@pytest.mark.unit
def test_perform_batch_analysis_sequential_error_no_continue(
    sample_files: list[str],
) -> None:
    """Test sequential batch analysis stops on error when continue_on_error=False."""
    with patch("oscura.loaders.load", side_effect=Exception("Load failed")):
        with pytest.raises(Exception, match="Load failed"):
            _perform_batch_analysis(
                files=sample_files[:1],
                analysis_type="characterize",
                parallel=1,
                continue_on_error=False,
                verbose=0,
            )


@pytest.mark.unit
def test_perform_batch_analysis_sequential_error_continue(
    sample_files: list[str],
) -> None:
    """Test sequential batch analysis continues on error when flag is set."""
    with patch("oscura.loaders.load", side_effect=Exception("Load failed")):
        results = _perform_batch_analysis(
            files=sample_files[:1],
            analysis_type="characterize",
            parallel=1,
            continue_on_error=True,
            verbose=0,
        )

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "error" in results[0]
        assert "Load failed" in results[0]["error"]


@pytest.mark.unit
def test_perform_batch_analysis_sequential_verbose(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test sequential batch analysis with verbose logging."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                with patch("oscura.cli.batch.logger") as mock_logger:
                    results = _perform_batch_analysis(
                        files=sample_files[:1],
                        analysis_type="characterize",
                        parallel=1,
                        continue_on_error=False,
                        verbose=1,
                    )

                    assert len(results) == 1
                    assert mock_logger.info.called


# =============================================================================
# Test _perform_batch_analysis() - Parallel Processing
# =============================================================================


@pytest.mark.unit
def test_perform_batch_analysis_parallel_characterize(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test parallel batch analysis with characterize."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                results = _perform_batch_analysis(
                    files=sample_files,
                    analysis_type="characterize",
                    parallel=2,
                    continue_on_error=False,
                    verbose=0,
                )

                assert len(results) == len(sample_files)
                for result in results:
                    assert result["status"] == "success"
                    assert result["analysis_type"] == "characterize"


@pytest.mark.unit
def test_perform_batch_analysis_parallel_error_no_continue(
    sample_files: list[str],
) -> None:
    """Test parallel batch analysis stops on error when continue_on_error=False."""
    with patch("oscura.loaders.load", side_effect=Exception("Load failed")):
        with pytest.raises(Exception, match="Load failed"):
            _perform_batch_analysis(
                files=sample_files,
                analysis_type="characterize",
                parallel=2,
                continue_on_error=False,
                verbose=0,
            )


@pytest.mark.unit
def test_perform_batch_analysis_parallel_error_continue(
    sample_files: list[str],
) -> None:
    """Test parallel batch analysis continues on error when flag is set."""
    with patch("oscura.loaders.load", side_effect=Exception("Load failed")):
        results = _perform_batch_analysis(
            files=sample_files,
            analysis_type="characterize",
            parallel=2,
            continue_on_error=True,
            verbose=0,
        )

        assert len(results) == len(sample_files)
        for result in results:
            assert result["status"] == "error"
            assert "error" in result


@pytest.mark.unit
def test_perform_batch_analysis_parallel_verbose(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test parallel batch analysis with verbose logging."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                with patch("oscura.cli.batch.logger") as mock_logger:
                    results = _perform_batch_analysis(
                        files=sample_files,
                        analysis_type="characterize",
                        parallel=2,
                        continue_on_error=False,
                        verbose=1,
                    )

                    assert len(results) == len(sample_files)
                    assert mock_logger.info.called


@pytest.mark.unit
def test_perform_batch_analysis_parallel_mixed_results(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test parallel batch analysis with mixed success/error results."""
    load_calls = 0

    def mock_load(path: str) -> MagicMock:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 2:
            raise ValueError("Simulated error")
        return mock_trace

    with patch("oscura.loaders.load", side_effect=mock_load):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                results = _perform_batch_analysis(
                    files=sample_files,
                    analysis_type="characterize",
                    parallel=2,
                    continue_on_error=True,
                    verbose=0,
                )

                assert len(results) == len(sample_files)
                success_count = sum(1 for r in results if r["status"] == "success")
                error_count = sum(1 for r in results if r["status"] == "error")
                assert success_count > 0
                assert error_count > 0


# =============================================================================
# Test _perform_batch_analysis() - Edge Cases
# =============================================================================


@pytest.mark.unit
def test_perform_batch_analysis_nan_values(sample_files: list[str], mock_trace: MagicMock) -> None:
    """Test batch analysis handles NaN values correctly."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=np.nan):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=np.nan):
                results = _perform_batch_analysis(
                    files=sample_files[:1],
                    analysis_type="characterize",
                    parallel=1,
                    continue_on_error=False,
                    verbose=0,
                )

                assert len(results) == 1
                assert results[0]["rise_time"] == "N/A"
                assert results[0]["fall_time"] == "N/A"


@pytest.mark.unit
def test_perform_batch_analysis_spectrum_nan_thd(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test batch analysis handles NaN THD values correctly."""
    freqs = np.linspace(0, 500e3, 100)
    mags = np.random.randn(100)

    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.spectral.fft", return_value=(freqs, mags)):
            with patch("oscura.analyzers.waveform.spectral.thd", return_value=np.nan):
                results = _perform_batch_analysis(
                    files=sample_files[:1],
                    analysis_type="spectrum",
                    parallel=1,
                    continue_on_error=False,
                    verbose=0,
                )

                assert len(results) == 1
                assert results[0]["thd"] == "N/A"


@pytest.mark.unit
def test_perform_batch_analysis_empty_fft(sample_files: list[str], mock_trace: MagicMock) -> None:
    """Test batch analysis handles empty FFT results."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch(
            "oscura.analyzers.waveform.spectral.fft", return_value=(np.array([]), np.array([]))
        ):
            with patch("oscura.analyzers.waveform.spectral.thd", return_value=-40.2):
                results = _perform_batch_analysis(
                    files=sample_files[:1],
                    analysis_type="spectrum",
                    parallel=1,
                    continue_on_error=False,
                    verbose=0,
                )

                assert len(results) == 1
                assert "peak_frequency" in results[0]


@pytest.mark.unit
def test_perform_batch_analysis_protocol_unknown(
    sample_files: list[str], mock_trace: MagicMock
) -> None:
    """Test batch analysis handles unknown protocol detection."""
    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.inference.detect_protocol", return_value={}):
            results = _perform_batch_analysis(
                files=sample_files[:1],
                analysis_type="decode",
                parallel=1,
                continue_on_error=False,
                verbose=0,
            )

            assert len(results) == 1
            assert results[0]["protocol"] == "unknown"
            assert results[0]["confidence"] == "0%"


# =============================================================================
# Test _generate_summary()
# =============================================================================


@pytest.mark.unit
def test_generate_summary_all_success() -> None:
    """Test summary generation with all successful results."""
    results = [
        {"file": "test_0.wfm", "status": "success"},
        {"file": "test_1.wfm", "status": "success"},
        {"file": "test_2.wfm", "status": "success"},
    ]

    summary = _generate_summary(results)

    assert summary["total_files"] == 3
    assert summary["successful"] == 3
    assert summary["failed"] == 0
    assert summary["success_rate"] == "100.0%"
    assert "note" in summary


@pytest.mark.unit
def test_generate_summary_all_failed() -> None:
    """Test summary generation with all failed results."""
    results = [
        {"file": "test_0.wfm", "status": "error"},
        {"file": "test_1.wfm", "status": "error"},
    ]

    summary = _generate_summary(results)

    assert summary["total_files"] == 2
    assert summary["successful"] == 0
    assert summary["failed"] == 2
    assert summary["success_rate"] == "0.0%"
    assert "note" not in summary


@pytest.mark.unit
def test_generate_summary_mixed_results() -> None:
    """Test summary generation with mixed success/error results."""
    results = [
        {"file": "test_0.wfm", "status": "success"},
        {"file": "test_1.wfm", "status": "error"},
        {"file": "test_2.wfm", "status": "success"},
        {"file": "test_3.wfm", "status": "error"},
    ]

    summary = _generate_summary(results)

    assert summary["total_files"] == 4
    assert summary["successful"] == 2
    assert summary["failed"] == 2
    assert summary["success_rate"] == "50.0%"
    assert "note" in summary


@pytest.mark.unit
def test_generate_summary_empty_results() -> None:
    """Test summary generation with empty results."""
    results: list[dict[str, Any]] = []

    summary = _generate_summary(results)

    assert summary["total_files"] == 0
    assert summary["successful"] == 0
    assert summary["failed"] == 0
    assert summary["success_rate"] == "N/A"


@pytest.mark.unit
def test_generate_summary_single_result() -> None:
    """Test summary generation with single result."""
    results = [{"file": "test.wfm", "status": "success"}]

    summary = _generate_summary(results)

    assert summary["total_files"] == 1
    assert summary["successful"] == 1
    assert summary["failed"] == 0
    assert summary["success_rate"] == "100.0%"


# =============================================================================
# Test _save_summary()
# =============================================================================


@pytest.mark.unit
def test_save_summary_basic(temp_dir: Path) -> None:
    """Test saving summary to CSV file."""
    results = [
        {
            "file": "test_0.wfm",
            "status": "success",
            "samples": 1000,
            "sample_rate": "1.0 MHz",
        },
        {
            "file": "test_1.wfm",
            "status": "success",
            "samples": 2000,
            "sample_rate": "2.0 MHz",
        },
    ]

    output_path = temp_dir / "summary.csv"
    _save_summary(results, str(output_path))

    assert output_path.exists()

    # Read and verify CSV content
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["file"] == "test_0.wfm"
        assert rows[0]["status"] == "success"
        assert rows[1]["file"] == "test_1.wfm"


@pytest.mark.unit
def test_save_summary_different_keys(temp_dir: Path) -> None:
    """Test saving summary with results having different keys."""
    results = [
        {"file": "test_0.wfm", "status": "success", "rise_time": "10 ns"},
        {"file": "test_1.wfm", "status": "success", "protocol": "UART"},
    ]

    output_path = temp_dir / "summary.csv"
    _save_summary(results, str(output_path))

    assert output_path.exists()

    # Read and verify CSV has all columns
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

        assert "file" in fieldnames
        assert "status" in fieldnames
        assert "rise_time" in fieldnames
        assert "protocol" in fieldnames
        assert len(rows) == 2


@pytest.mark.unit
def test_save_summary_empty_results(temp_dir: Path) -> None:
    """Test saving empty results doesn't create file."""
    results: list[dict[str, Any]] = []

    output_path = temp_dir / "summary.csv"
    _save_summary(results, str(output_path))

    # File should not be created for empty results
    assert not output_path.exists()


@pytest.mark.unit
def test_save_summary_with_error_results(temp_dir: Path) -> None:
    """Test saving summary with error results."""
    results = [
        {"file": "test_0.wfm", "status": "success", "samples": 1000},
        {"file": "test_1.wfm", "status": "error", "error": "Load failed"},
    ]

    output_path = temp_dir / "summary.csv"
    _save_summary(results, str(output_path))

    assert output_path.exists()

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) == 2
        assert rows[1]["status"] == "error"
        assert rows[1]["error"] == "Load failed"


@pytest.mark.unit
def test_save_summary_sorted_fieldnames(temp_dir: Path) -> None:
    """Test that CSV fieldnames are sorted for consistency."""
    results = [
        {
            "zebra": "last",
            "apple": "first",
            "middle": "mid",
        }
    ]

    output_path = temp_dir / "summary.csv"
    _save_summary(results, str(output_path))

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Fieldnames should be sorted alphabetically
        assert fieldnames == sorted(fieldnames)
        assert fieldnames[0] == "apple"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_batch_end_to_end_characterize(
    runner: CliRunner, temp_dir: Path, mock_trace: MagicMock
) -> None:
    """Test complete batch workflow for characterize analysis."""
    # Create test files
    test_files = []
    for i in range(2):
        file_path = temp_dir / f"signal_{i}.wfm"
        file_path.write_text(f"data {i}")
        test_files.append(str(file_path))

    pattern = str(temp_dir / "signal_*.wfm")

    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                result = runner.invoke(
                    batch,
                    [pattern, "--analysis", "characterize", "--output", "json"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                assert "total_files" in result.output or "successful" in result.output


@pytest.mark.unit
def test_batch_end_to_end_with_csv_export(
    runner: CliRunner, temp_dir: Path, mock_trace: MagicMock
) -> None:
    """Test complete batch workflow with CSV export."""
    # Create test file
    file_path = temp_dir / "signal.wfm"
    file_path.write_text("data")

    summary_path = temp_dir / "results.csv"
    pattern = str(file_path)

    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                result = runner.invoke(
                    batch,
                    [
                        pattern,
                        "--analysis",
                        "characterize",
                        "--save-summary",
                        str(summary_path),
                    ],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
                assert summary_path.exists()


@pytest.mark.unit
def test_batch_recursive_glob_pattern(
    runner: CliRunner, temp_dir: Path, mock_trace: MagicMock
) -> None:
    """Test batch command with recursive glob pattern."""
    # Create nested directory structure
    subdir = temp_dir / "subdir"
    subdir.mkdir()

    file1 = temp_dir / "test_1.wfm"
    file2 = subdir / "test_2.wfm"
    file1.write_text("data1")
    file2.write_text("data2")

    pattern = str(temp_dir / "**" / "test_*.wfm")

    with patch("oscura.loaders.load", return_value=mock_trace):
        with patch("oscura.analyzers.waveform.measurements.rise_time", return_value=10.5e-9):
            with patch("oscura.analyzers.waveform.measurements.fall_time", return_value=12.3e-9):
                result = runner.invoke(
                    batch,
                    [pattern, "--analysis", "characterize"],
                    obj={"verbose": 0},
                )

                assert result.exit_code == 0
