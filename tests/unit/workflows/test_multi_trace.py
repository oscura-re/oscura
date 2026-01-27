"""Comprehensive tests for multi-trace workflow module.

Tests multi-trace processing workflows including alignment, measurement
aggregation, and parallel processing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.core.exceptions import OscuraError
from oscura.workflows.multi_trace import (
    AlignmentMethod,
    MultiTraceResults,
    MultiTraceWorkflow,
    TraceStatistics,
    load_all,
)

# =============================================================================
# Test Data Structures
# =============================================================================


@dataclass
class MockTrace:
    """Mock trace object for testing."""

    data: np.ndarray[Any, Any]
    sample_rate: float = 1e9
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_traces() -> list[MockTrace]:
    """Create simple mock traces for testing."""
    return [MockTrace(data=np.array([0, 1, 2, 3, 4, 5]) + i * 0.1) for i in range(3)]


@pytest.fixture
def triggered_traces() -> list[MockTrace]:
    """Create traces with trigger events at different positions."""
    traces = []
    for offset in [10, 15, 20]:
        # Create signal with rising edge at different positions
        data = np.zeros(100)
        data[offset:] = 1.0
        # Add some noise
        data += np.random.default_rng(42).normal(0, 0.01, 100)
        traces.append(MockTrace(data=data))
    return traces


@pytest.fixture
def temp_csv_files(tmp_path: Path) -> list[Path]:
    """Create temporary CSV files for testing file loading."""
    files = []
    for i in range(3):
        filepath = tmp_path / f"trace_{i}.csv"
        # Simple CSV format: time,voltage
        data = np.column_stack(
            [np.arange(100) * 1e-9, np.sin(2 * np.pi * 1e6 * np.arange(100) * 1e-9) + i * 0.1]
        )
        np.savetxt(filepath, data, delimiter=",", header="time,voltage", comments="")
        files.append(filepath)
    return files


# =============================================================================
# AlignmentMethod Tests
# =============================================================================


def test_alignment_method_constants() -> None:
    """Test alignment method constant definitions."""
    assert AlignmentMethod.TRIGGER == "trigger"
    assert AlignmentMethod.TIME_SYNC == "time"
    assert AlignmentMethod.CROSS_CORRELATION == "correlation"
    assert AlignmentMethod.MANUAL == "manual"


# =============================================================================
# TraceStatistics Tests
# =============================================================================


def test_trace_statistics_creation() -> None:
    """Test TraceStatistics dataclass creation."""
    stats = TraceStatistics(mean=1.5, std=0.5, min=1.0, max=2.0, median=1.5, count=10)

    assert stats.mean == 1.5
    assert stats.std == 0.5
    assert stats.min == 1.0
    assert stats.max == 2.0
    assert stats.median == 1.5
    assert stats.count == 10


def test_trace_statistics_fields() -> None:
    """Test all TraceStatistics fields are present."""
    stats = TraceStatistics(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0, count=0)

    # Check all required fields exist
    assert hasattr(stats, "mean")
    assert hasattr(stats, "std")
    assert hasattr(stats, "min")
    assert hasattr(stats, "max")
    assert hasattr(stats, "median")
    assert hasattr(stats, "count")


# =============================================================================
# MultiTraceResults Tests
# =============================================================================


def test_multi_trace_results_defaults() -> None:
    """Test MultiTraceResults default initialization."""
    results = MultiTraceResults()

    assert results.trace_ids == []
    assert results.measurements == {}
    assert results.statistics == {}
    assert results.metadata == {}


def test_multi_trace_results_with_data() -> None:
    """Test MultiTraceResults with actual data."""
    results = MultiTraceResults(
        trace_ids=["trace1", "trace2"],
        measurements={"trace1": {"rise_time": 1.5e-9}, "trace2": {"rise_time": 1.6e-9}},
        statistics={
            "rise_time": TraceStatistics(
                mean=1.55e-9, std=0.05e-9, min=1.5e-9, max=1.6e-9, median=1.55e-9, count=2
            )
        },
        metadata={"alignment": "trigger"},
    )

    assert len(results.trace_ids) == 2
    assert "trace1" in results.measurements
    assert "rise_time" in results.statistics
    assert results.metadata["alignment"] == "trigger"


# =============================================================================
# MultiTraceWorkflow Initialization Tests
# =============================================================================


def test_workflow_init_with_traces(simple_traces: list[MockTrace]) -> None:
    """Test workflow initialization with pre-loaded traces."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    assert workflow._traces == simple_traces
    assert workflow.pattern is None
    assert workflow._lazy is False
    assert not workflow._aligned


def test_workflow_init_with_pattern(temp_csv_files: list[Path]) -> None:
    """Test workflow initialization with file pattern."""
    pattern = str(temp_csv_files[0].parent / "trace_*.csv")
    workflow = MultiTraceWorkflow(pattern=pattern)

    assert workflow.pattern == pattern
    assert len(workflow._file_paths) == 3
    assert len(workflow.results.trace_ids) == 3


def test_workflow_init_no_inputs() -> None:
    """Test workflow initialization fails without inputs."""
    with pytest.raises(OscuraError, match="Must provide either pattern or traces"):
        MultiTraceWorkflow()


def test_workflow_init_lazy_mode(temp_csv_files: list[Path]) -> None:
    """Test workflow initialization in lazy loading mode."""
    pattern = str(temp_csv_files[0].parent / "trace_*.csv")
    workflow = MultiTraceWorkflow(pattern=pattern, lazy=True)

    assert workflow._lazy is True
    assert len(workflow._file_paths) == 3


def test_workflow_init_no_matching_files(tmp_path: Path) -> None:
    """Test workflow fails when no files match pattern."""
    pattern = str(tmp_path / "nonexistent_*.csv")

    with pytest.raises(OscuraError, match="No files match pattern"):
        MultiTraceWorkflow(pattern=pattern)


# =============================================================================
# File Discovery Tests
# =============================================================================


def test_discover_files(temp_csv_files: list[Path]) -> None:
    """Test file discovery from pattern."""
    pattern = str(temp_csv_files[0].parent / "*.csv")
    workflow = MultiTraceWorkflow(pattern=pattern)

    assert len(workflow._file_paths) == 3
    # Files should be sorted by name
    assert workflow._file_paths[0].name == "trace_0.csv"
    assert workflow._file_paths[1].name == "trace_1.csv"
    assert workflow._file_paths[2].name == "trace_2.csv"


def test_discover_files_updates_trace_ids(temp_csv_files: list[Path]) -> None:
    """Test that file discovery updates trace IDs."""
    pattern = str(temp_csv_files[0].parent / "*.csv")
    workflow = MultiTraceWorkflow(pattern=pattern)

    assert workflow.results.trace_ids == ["trace_0.csv", "trace_1.csv", "trace_2.csv"]


# =============================================================================
# Trace Iteration Tests
# =============================================================================


def test_iter_traces_with_preloaded(simple_traces: list[MockTrace]) -> None:
    """Test iteration over pre-loaded traces."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    traces_list = list(workflow._iter_traces())

    assert len(traces_list) == 3
    assert traces_list[0][0] == "t1"
    assert traces_list[0][1] is simple_traces[0]


def test_iter_traces_generates_ids(simple_traces: list[MockTrace]) -> None:
    """Test that iteration generates IDs when not provided."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    traces_list = list(workflow._iter_traces())

    assert traces_list[0][0] == "trace_0"
    assert traces_list[1][0] == "trace_1"
    assert traces_list[2][0] == "trace_2"


def test_iter_traces_lazy_loading(temp_csv_files: list[Path]) -> None:
    """Test lazy loading during iteration."""
    pattern = str(temp_csv_files[0].parent / "*.csv")
    workflow = MultiTraceWorkflow(pattern=pattern, lazy=True)

    # Mock the loader to verify it's called during iteration
    with patch.object(workflow, "_load_trace") as mock_load:
        mock_load.return_value = MockTrace(data=np.array([1, 2, 3]))

        list(workflow._iter_traces(lazy=True))

        # Should load each file
        assert mock_load.call_count == 3


# =============================================================================
# Alignment Tests - Trigger Method
# =============================================================================


def test_align_by_trigger_basic(triggered_traces: list[MockTrace]) -> None:
    """Test trigger-based alignment."""
    workflow = MultiTraceWorkflow(traces=triggered_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    workflow.align(method=AlignmentMethod.TRIGGER, threshold=0.5)

    assert workflow._aligned
    assert len(workflow._alignment_offset) == 3
    # Offsets should correspond to trigger positions
    assert workflow._alignment_offset["t1"] > 0
    assert workflow._alignment_offset["t2"] > 0
    assert workflow._alignment_offset["t3"] > 0


def test_align_by_trigger_auto_threshold(triggered_traces: list[MockTrace]) -> None:
    """Test trigger alignment with automatic threshold."""
    workflow = MultiTraceWorkflow(traces=triggered_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    # No threshold provided - should auto-compute
    workflow.align(method=AlignmentMethod.TRIGGER)

    assert workflow._aligned
    assert len(workflow._alignment_offset) == 3


def test_align_by_trigger_no_edge() -> None:
    """Test trigger alignment when no edge is found."""
    # Flat signal with no edges
    flat_trace = MockTrace(data=np.ones(100) * 0.5)
    workflow = MultiTraceWorkflow(traces=[flat_trace])
    workflow.results.trace_ids = ["flat"]

    workflow.align(method=AlignmentMethod.TRIGGER, threshold=0.7)

    # Should default to offset 0 when no edge found
    assert workflow._alignment_offset["flat"] == 0


def test_align_by_trigger_trace_without_data() -> None:
    """Test trigger alignment with trace lacking data attribute."""
    # Mock trace without data attribute
    mock_trace = MagicMock(spec=[])  # No 'data' attribute
    workflow = MultiTraceWorkflow(traces=[mock_trace])
    workflow.results.trace_ids = ["no_data"]

    workflow.align(method=AlignmentMethod.TRIGGER, threshold=0.5)

    # Should default to offset 0
    assert workflow._alignment_offset["no_data"] == 0


# =============================================================================
# Alignment Tests - Other Methods
# =============================================================================


def test_align_by_time(simple_traces: list[MockTrace]) -> None:
    """Test time-based alignment."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    workflow.align(method=AlignmentMethod.TIME_SYNC)

    assert workflow._aligned
    assert len(workflow._alignment_offset) == 3


def test_align_by_correlation(simple_traces: list[MockTrace]) -> None:
    """Test cross-correlation alignment."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    workflow.align(method=AlignmentMethod.CROSS_CORRELATION, channel=0)

    assert workflow._aligned
    assert len(workflow._alignment_offset) == 3


def test_align_manual_with_offsets(simple_traces: list[MockTrace]) -> None:
    """Test manual alignment with provided offsets."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    offsets = {"t1": 0, "t2": 10, "t3": 20}
    workflow.align(method=AlignmentMethod.MANUAL, offsets=offsets)

    assert workflow._aligned
    assert workflow._alignment_offset == offsets


def test_align_manual_missing_offsets(simple_traces: list[MockTrace]) -> None:
    """Test manual alignment fails without offsets parameter."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    with pytest.raises(OscuraError, match="Manual alignment requires 'offsets'"):
        workflow.align(method=AlignmentMethod.MANUAL)


def test_align_unknown_method(simple_traces: list[MockTrace]) -> None:
    """Test alignment fails with unknown method."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    with pytest.raises(OscuraError, match="Unknown alignment method"):
        workflow.align(method="invalid_method")


# =============================================================================
# Measurement Tests
# =============================================================================


def test_measure_no_measurements(simple_traces: list[MockTrace]) -> None:
    """Test measure fails when no measurements specified."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    with pytest.raises(OscuraError, match="At least one measurement required"):
        workflow.measure()


def test_measure_sequential(simple_traces: list[MockTrace]) -> None:
    """Test sequential measurement processing."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    # Mock the measurement function to return test values
    with patch.object(workflow, "_perform_measurement") as mock_measure:
        with patch("oscura.workflows.multi_trace.create_progress_tracker"):
            mock_measure.side_effect = lambda t, m: 1.5e-9 if m == "rise_time" else 2.0e-9

            workflow.measure("rise_time", "fall_time", parallel=False)

            assert len(workflow.results.measurements) == 3
            assert "rise_time" in workflow.results.measurements["t1"]
            assert "fall_time" in workflow.results.measurements["t1"]
            assert workflow.results.measurements["t1"]["rise_time"] == 1.5e-9


def test_measure_sequential_with_failure(simple_traces: list[MockTrace]) -> None:
    """Test sequential measurement handles failures gracefully."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    # Mock measurement to fail on second trace
    def mock_measure(trace: Any, meas: str) -> float:
        if trace is simple_traces[1]:
            raise ValueError("Measurement failed")
        return 1.5e-9

    with patch.object(workflow, "_perform_measurement", side_effect=mock_measure):
        with patch("oscura.workflows.multi_trace.create_progress_tracker"):
            # Should not raise - failures are captured
            workflow.measure("rise_time", parallel=False)

            # Failed measurement should be None
            assert workflow.results.measurements["t2"]["rise_time"] is None


def test_measure_parallel(simple_traces: list[MockTrace]) -> None:
    """Test parallel measurement processing."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    with patch.object(workflow, "_measure_trace") as mock_measure:
        mock_measure.return_value = {"rise_time": 1.5e-9, "fall_time": 2.0e-9}

        workflow.measure("rise_time", "fall_time", parallel=True, max_workers=2)

        assert len(workflow.results.measurements) == 3
        assert mock_measure.call_count == 3


def test_measure_trace_helper(simple_traces: list[MockTrace]) -> None:
    """Test _measure_trace helper method."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    with patch.object(workflow, "_perform_measurement") as mock_measure:
        mock_measure.side_effect = lambda t, m: 1.5e-9

        result = workflow._measure_trace(simple_traces[0], ("rise_time", "fall_time"))

        assert result["rise_time"] == 1.5e-9
        assert result["fall_time"] == 1.5e-9


def test_measure_trace_handles_exception(simple_traces: list[MockTrace]) -> None:
    """Test _measure_trace handles measurement exceptions."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    with patch.object(workflow, "_perform_measurement") as mock_measure:
        mock_measure.side_effect = ValueError("Failed")

        result = workflow._measure_trace(simple_traces[0], ("rise_time",))

        assert result["rise_time"] is None


def test_perform_measurement_not_implemented(simple_traces: list[MockTrace]) -> None:
    """Test _perform_measurement placeholder raises error."""
    workflow = MultiTraceWorkflow(traces=simple_traces)

    with pytest.raises(OscuraError, match="not yet implemented"):
        workflow._perform_measurement(simple_traces[0], "rise_time")


# =============================================================================
# Aggregation Tests
# =============================================================================


def test_aggregate_basic() -> None:
    """Test basic aggregation of measurement statistics."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1, 2, 3]))])
    workflow.results.trace_ids = ["t1", "t2", "t3"]
    workflow.results.measurements = {
        "t1": {"rise_time": 1.0},
        "t2": {"rise_time": 2.0},
        "t3": {"rise_time": 3.0},
    }

    results = workflow.aggregate()

    assert "rise_time" in results.statistics
    stats = results.statistics["rise_time"]
    assert stats.mean == 2.0
    assert stats.min == 1.0
    assert stats.max == 3.0
    assert stats.median == 2.0
    assert stats.count == 3


def test_aggregate_multiple_measurements() -> None:
    """Test aggregation with multiple measurement types."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    workflow.results.measurements = {
        "t1": {"rise_time": 1.0, "fall_time": 5.0},
        "t2": {"rise_time": 2.0, "fall_time": 6.0},
        "t3": {"rise_time": 3.0, "fall_time": 7.0},
    }

    results = workflow.aggregate()

    assert "rise_time" in results.statistics
    assert "fall_time" in results.statistics
    assert results.statistics["rise_time"].mean == 2.0
    assert results.statistics["fall_time"].mean == 6.0


def test_aggregate_skips_none_values() -> None:
    """Test aggregation skips None values from failed measurements."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    workflow.results.measurements = {
        "t1": {"rise_time": 1.0},
        "t2": {"rise_time": None},  # Failed measurement
        "t3": {"rise_time": 3.0},
    }

    results = workflow.aggregate()

    stats = results.statistics["rise_time"]
    assert stats.count == 2  # Only counted valid values
    assert stats.mean == 2.0


def test_aggregate_skips_nan_values() -> None:
    """Test aggregation skips NaN values."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    workflow.results.measurements = {
        "t1": {"rise_time": 1.0},
        "t2": {"rise_time": np.nan},
        "t3": {"rise_time": 3.0},
    }

    results = workflow.aggregate()

    stats = results.statistics["rise_time"]
    assert stats.count == 2
    assert stats.mean == 2.0


def test_aggregate_computes_std() -> None:
    """Test aggregation computes standard deviation correctly."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    workflow.results.measurements = {
        "t1": {"rise_time": 1.0},
        "t2": {"rise_time": 2.0},
        "t3": {"rise_time": 3.0},
        "t4": {"rise_time": 4.0},
    }

    results = workflow.aggregate()

    stats = results.statistics["rise_time"]
    # Std of [1,2,3,4] is approximately 1.118
    assert 1.0 < stats.std < 1.2


def test_aggregate_no_measurements() -> None:
    """Test aggregate fails when no measurements available."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])

    with pytest.raises(OscuraError, match="No measurements available"):
        workflow.aggregate()


def test_aggregate_empty_after_filtering() -> None:
    """Test aggregation when all values are None/NaN."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    workflow.results.measurements = {"t1": {"rise_time": None}, "t2": {"rise_time": np.nan}}

    results = workflow.aggregate()

    # Should not create statistics for measurements with no valid values
    assert "rise_time" not in results.statistics


# =============================================================================
# Export Tests
# =============================================================================


def test_export_json(tmp_path: Path) -> None:
    """Test JSON export functionality."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    workflow.results.trace_ids = ["t1"]
    workflow.results.measurements = {"t1": {"rise_time": 1.5e-9}}
    workflow.results.statistics = {
        "rise_time": TraceStatistics(
            mean=1.5e-9, std=0.1e-9, min=1.4e-9, max=1.6e-9, median=1.5e-9, count=1
        )
    }
    workflow.results.metadata = {"alignment": "trigger"}

    output_file = tmp_path / "results.json"
    workflow.export_report(str(output_file), format="json")

    # Verify file was created and contains expected data
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)

    assert data["trace_ids"] == ["t1"]
    assert "t1" in data["measurements"]
    assert "rise_time" in data["statistics"]
    assert data["statistics"]["rise_time"]["mean"] == 1.5e-9
    assert data["metadata"]["alignment"] == "trigger"


def test_export_pdf_not_implemented(tmp_path: Path) -> None:
    """Test PDF export raises not implemented error."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    output_file = tmp_path / "results.pdf"

    with pytest.raises(OscuraError, match="PDF export not yet implemented"):
        workflow.export_report(str(output_file), format="pdf")


def test_export_html_not_implemented(tmp_path: Path) -> None:
    """Test HTML export raises not implemented error."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    output_file = tmp_path / "results.html"

    with pytest.raises(OscuraError, match="HTML export not yet implemented"):
        workflow.export_report(str(output_file), format="html")


def test_export_unsupported_format(tmp_path: Path) -> None:
    """Test export fails with unsupported format."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])
    output_file = tmp_path / "results.xyz"

    with pytest.raises(OscuraError, match="Unsupported report format"):
        workflow.export_report(str(output_file), format="xyz")


# =============================================================================
# load_all Function Tests
# =============================================================================


def test_load_all_basic(temp_csv_files: list[Path]) -> None:
    """Test load_all function with basic pattern."""
    pattern = str(temp_csv_files[0].parent / "*.csv")

    traces = load_all(pattern)

    assert len(traces) == 3
    assert all(isinstance(t, Path) for t in traces)


def test_load_all_lazy_mode(temp_csv_files: list[Path]) -> None:
    """Test load_all with lazy mode enabled."""
    pattern = str(temp_csv_files[0].parent / "*.csv")

    traces = load_all(pattern, lazy=True)

    assert len(traces) == 3


def test_load_all_no_matches(tmp_path: Path) -> None:
    """Test load_all fails when no files match."""
    pattern = str(tmp_path / "nonexistent_*.csv")

    with pytest.raises(OscuraError, match="No files match pattern"):
        load_all(pattern)


def test_load_all_specific_pattern(temp_csv_files: list[Path]) -> None:
    """Test load_all with specific file pattern."""
    # Only match trace_0.csv
    pattern = str(temp_csv_files[0].parent / "trace_0.csv")

    traces = load_all(pattern)

    assert len(traces) == 1


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_workflow(triggered_traces: list[MockTrace]) -> None:
    """Test complete end-to-end workflow."""
    # Create workflow with traces
    workflow = MultiTraceWorkflow(traces=triggered_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    # Align traces
    workflow.align(method=AlignmentMethod.TRIGGER)
    assert workflow._aligned

    # Measure (with mocked measurements)
    with patch.object(workflow, "_perform_measurement") as mock_measure:
        with patch("oscura.workflows.multi_trace.create_progress_tracker"):
            mock_measure.return_value = 1.5e-9
            workflow.measure("rise_time", parallel=False)

    # Aggregate statistics
    results = workflow.aggregate()

    assert len(results.trace_ids) == 3
    assert "rise_time" in results.statistics
    assert results.statistics["rise_time"].count == 3


def test_workflow_with_file_pattern_integration(temp_csv_files: list[Path], tmp_path: Path) -> None:
    """Test workflow integration with file pattern loading."""
    pattern = str(temp_csv_files[0].parent / "*.csv")

    # Create workflow from pattern
    workflow = MultiTraceWorkflow(pattern=pattern)

    # Should discover files
    assert len(workflow.results.trace_ids) == 3

    # Should be able to export results
    output_file = tmp_path / "results.json"
    workflow.results.metadata["test"] = "value"
    workflow.export_report(str(output_file), format="json")

    assert output_file.exists()


def test_parallel_measurement_integration(simple_traces: list[MockTrace]) -> None:
    """Test parallel measurement processing integration."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["t1", "t2", "t3"]

    with patch.object(workflow, "_perform_measurement") as mock_measure:
        mock_measure.return_value = 1.5e-9

        # Process in parallel
        workflow.measure("rise_time", parallel=True, max_workers=2)

        # Aggregate results
        results = workflow.aggregate()

        assert results.statistics["rise_time"].count == 3


# =============================================================================
# Edge Cases
# =============================================================================


def test_workflow_single_trace() -> None:
    """Test workflow with single trace."""
    trace = MockTrace(data=np.array([1, 2, 3]))
    workflow = MultiTraceWorkflow(traces=[trace])
    workflow.results.trace_ids = ["single"]

    workflow.align(method=AlignmentMethod.TRIGGER)

    assert workflow._aligned
    assert len(workflow._alignment_offset) == 1


def test_workflow_empty_results() -> None:
    """Test workflow with empty results structure."""
    workflow = MultiTraceWorkflow(traces=[MockTrace(data=np.array([1]))])

    # Results should be empty initially
    assert len(workflow.results.trace_ids) == 0
    assert len(workflow.results.measurements) == 0
    assert len(workflow.results.statistics) == 0


def test_alignment_preserves_trace_ids(simple_traces: list[MockTrace]) -> None:
    """Test that alignment uses correct trace IDs."""
    workflow = MultiTraceWorkflow(traces=simple_traces)
    workflow.results.trace_ids = ["custom1", "custom2", "custom3"]

    workflow.align(method=AlignmentMethod.TIME_SYNC)

    # Should have offsets for all trace IDs
    assert set(workflow._alignment_offset.keys()) == {"custom1", "custom2", "custom3"}
