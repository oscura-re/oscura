"""Multi-Trace Workflow Support.

Provides workflows for processing and analyzing multiple traces together.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Iterator
from dataclasses import dataclass, field
from glob import glob as glob_func
from pathlib import Path
from typing import Any

import numpy as np

from oscura.core.exceptions import OscuraError
from oscura.core.progress import create_progress_tracker


class AlignmentMethod:
    """Alignment method constants."""

    TRIGGER = "trigger"
    TIME_SYNC = "time"
    CROSS_CORRELATION = "correlation"
    MANUAL = "manual"


@dataclass
class TraceStatistics:
    """Statistics for a measurement across traces.

    Attributes:
        mean: Mean value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        median: Median value
        count: Number of traces
    """

    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int


@dataclass
class MultiTraceResults:
    """Results from multi-trace workflow.

    Attributes:
        trace_ids: List of trace identifiers
        measurements: Dict mapping trace_id -> measurement results
        statistics: Dict mapping measurement_name -> TraceStatistics
        metadata: Additional workflow metadata
    """

    trace_ids: list[str] = field(default_factory=list)
    measurements: dict[str, dict[str, Any]] = field(default_factory=dict)
    statistics: dict[str, TraceStatistics] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiTraceWorkflow:
    """Workflow manager for multi-trace processing.

    Provides methods to load, align, process, and analyze multiple traces
    with memory-efficient streaming and optional parallelization.
    """

    def __init__(
        self,
        pattern: str | None = None,
        traces: list[Any] | None = None,
        lazy: bool = False,
    ):
        """Initialize multi-trace workflow.

        Args:
            pattern: Glob pattern for trace files (e.g., "*.csv")
            traces: Pre-loaded trace objects
            lazy: If True, load traces on demand

        Raises:
            OscuraError: If neither pattern nor traces provided
        """
        self.pattern = pattern
        self._traces = traces or []
        self._lazy = lazy
        self._file_paths: list[Path] = []
        self._aligned = False
        self._alignment_offset: dict[str, int] = {}
        self.results = MultiTraceResults()

        # Discover files if pattern provided
        if pattern:
            self._discover_files()
        elif not traces:
            raise OscuraError("Must provide either pattern or traces")

    def _discover_files(self) -> None:
        """Discover trace files matching pattern."""
        if not self.pattern:
            return

        paths = glob_func(self.pattern)
        if not paths:
            raise OscuraError(f"No files match pattern: {self.pattern}")

        self._file_paths = [Path(p) for p in sorted(paths)]
        self.results.trace_ids = [p.name for p in self._file_paths]

    def _load_trace(self, path: Path) -> Any:
        """Load a single trace file.

        Args:
            path: Path to trace file

        Returns:
            Loaded trace object

        Raises:
            OscuraError: If trace cannot be loaded
        """
        # Determine loader based on extension
        ext = path.suffix.lower()

        try:
            if ext == ".csv":
                from oscura.loaders.csv import (
                    load_csv,
                )

                return load_csv(str(path))
            elif ext == ".bin":
                from oscura.loaders.binary import (
                    load_binary,
                )

                return load_binary(str(path))
            elif ext in (".h5", ".hdf5"):
                from oscura.loaders.hdf5 import (
                    load_hdf5,
                )

                return load_hdf5(str(path))
            else:
                raise OscuraError(f"Unsupported format: {ext}")

        except ImportError as e:
            raise OscuraError(f"Loader not available for {ext}: {e}")

    def _iter_traces(self, lazy: bool = False) -> Iterator[tuple[str, Any]]:
        """Iterate over traces.

        Args:
            lazy: If True, load on demand; if False, load all first

        Yields:
            Tuple of (trace_id, trace)
        """
        # Use pre-loaded traces if available
        if self._traces:
            for i, trace in enumerate(self._traces):
                trace_id = (
                    self.results.trace_ids[i] if i < len(self.results.trace_ids) else f"trace_{i}"
                )
                yield trace_id, trace
            return

        # Load from files
        for path in self._file_paths:
            trace_id = path.name
            if lazy or self._lazy:
                # Load on demand
                trace = self._load_trace(path)
            else:
                # Would load all at once (not implemented here)
                trace = self._load_trace(path)
            yield trace_id, trace

    def align(
        self,
        method: str = AlignmentMethod.TRIGGER,
        channel: int = 0,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Align traces using specified method.

        Args:
            method: Alignment method ('trigger', 'time', 'correlation', 'manual')
            channel: Channel to use for alignment (for multi-channel traces)
            threshold: Trigger threshold (for trigger alignment)
            **kwargs: Additional method-specific parameters

        Raises:
            OscuraError: If alignment fails
        """
        if method == AlignmentMethod.TRIGGER:
            self._align_by_trigger(channel, threshold, **kwargs)
        elif method == AlignmentMethod.TIME_SYNC:
            self._align_by_time(**kwargs)
        elif method == AlignmentMethod.CROSS_CORRELATION:
            self._align_by_correlation(channel, **kwargs)
        elif method == AlignmentMethod.MANUAL:
            self._align_manual(**kwargs)
        else:
            raise OscuraError(f"Unknown alignment method: {method}")

        self._aligned = True

    def _align_by_trigger(
        self,
        channel: int,
        threshold: float | None,
        **kwargs: Any,
    ) -> None:
        """Align traces by trigger point.

        Args:
            channel: Channel index
            threshold: Trigger threshold
            **kwargs: Additional parameters
        """
        # Find trigger point in each trace
        for trace_id, trace in self._iter_traces(lazy=True):
            # Find first crossing of threshold
            if hasattr(trace, "data"):
                data = trace.data
                if threshold is None:
                    # Auto threshold: 50% of max
                    threshold = 0.5 * (np.max(data) + np.min(data))

                # Find first rising edge
                above = data > threshold
                edges = np.diff(above.astype(int))
                rising = np.where(edges > 0)[0]

                if len(rising) > 0:
                    self._alignment_offset[trace_id] = int(rising[0])
                else:
                    self._alignment_offset[trace_id] = 0
            else:
                self._alignment_offset[trace_id] = 0

    def _align_by_time(self, **kwargs: Any) -> None:
        """Align traces by timestamp.

        Args:
            **kwargs: Additional parameters
        """
        # Align based on trace timestamps
        # Placeholder implementation
        for trace_id, _trace in self._iter_traces(lazy=True):
            self._alignment_offset[trace_id] = 0

    def _align_by_correlation(self, channel: int, **kwargs: Any) -> None:
        """Align traces by cross-correlation.

        Args:
            channel: Channel index
            **kwargs: Additional parameters
        """
        # Use cross-correlation to find alignment
        # Placeholder implementation
        for trace_id, _trace in self._iter_traces(lazy=True):
            self._alignment_offset[trace_id] = 0

    def _align_manual(self, **kwargs: Any) -> None:
        """Manual alignment with specified offsets.

        Args:
            **kwargs: Must include 'offsets' dict mapping trace_id -> offset

        Raises:
            OscuraError: If offsets parameter not provided.
        """
        offsets = kwargs.get("offsets", {})
        if not offsets:
            raise OscuraError("Manual alignment requires 'offsets' parameter")

        self._alignment_offset.update(offsets)

    def measure(
        self, *measurements: str, parallel: bool = False, max_workers: int | None = None
    ) -> None:
        """Measure properties across all traces.

        Args:
            *measurements: Measurement names (rise_time, fall_time, etc.)
            parallel: If True, process traces in parallel
            max_workers: Maximum parallel workers (None = CPU count)

        Raises:
            OscuraError: If measurement fails
        """
        if not measurements:
            raise OscuraError("At least one measurement required")

        if parallel:
            self._measure_parallel(measurements, max_workers)
        else:
            self._measure_sequential(measurements)

    def _measure_sequential(self, measurements: tuple[str, ...]) -> None:
        """Measure sequentially."""
        # Progress tracking
        progress = create_progress_tracker(  # type: ignore[call-arg]
            total=len(self.results.trace_ids),
            description="Measuring traces",
        )

        for trace_id, trace in self._iter_traces(lazy=True):
            results = {}
            for meas_name in measurements:
                try:
                    # Call measurement function
                    # Placeholder - would call actual measurement
                    results[meas_name] = self._perform_measurement(trace, meas_name)
                except Exception as e:
                    results[meas_name] = None
                    print(f"Warning: {meas_name} failed for {trace_id}: {e}")

            self.results.measurements[trace_id] = results
            progress.update(1)

    def _measure_parallel(self, measurements: tuple[str, ...], max_workers: int | None) -> None:
        """Measure in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for trace_id, trace in self._iter_traces(lazy=False):
                future = executor.submit(self._measure_trace, trace, measurements)
                futures[future] = trace_id

            for future in concurrent.futures.as_completed(futures):
                trace_id = futures[future]
                try:
                    results = future.result()
                    self.results.measurements[trace_id] = results
                except Exception as e:
                    print(f"Error measuring {trace_id}: {e}")

    def _measure_trace(self, trace: Any, measurements: tuple[str, ...]) -> dict[str, Any]:
        """Measure a single trace.

        Args:
            trace: Trace object
            measurements: Measurement names

        Returns:
            Dict mapping measurement_name -> value
        """
        results = {}
        for meas_name in measurements:
            try:
                results[meas_name] = self._perform_measurement(trace, meas_name)
            except Exception:
                results[meas_name] = None
        return results

    def _perform_measurement(self, trace: Any, measurement: str) -> Any:
        """Perform a single measurement.

        Args:
            trace: Trace object
            measurement: Measurement name

        Raises:
            OscuraError: If measurement not available
        """
        # Placeholder - would call actual measurement functions
        # from oscura.analyzers.measurements
        raise OscuraError(
            f"Measurement '{measurement}' not yet implemented in multi-trace workflow"
        )

    def aggregate(self) -> MultiTraceResults:
        """Compute aggregate statistics across traces.

        Returns:
            Results with statistics

        Raises:
            OscuraError: If no measurements available
        """
        if not self.results.measurements:
            raise OscuraError("No measurements available. Call measure() first.")

        # Compute statistics for each measurement type
        all_measurements = set()  # type: ignore[var-annotated]
        for trace_results in self.results.measurements.values():
            all_measurements.update(trace_results.keys())

        for meas_name in all_measurements:
            values = []
            for trace_results in self.results.measurements.values():
                val = trace_results.get(meas_name)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(float(val))

            if values:
                self.results.statistics[meas_name] = TraceStatistics(
                    mean=float(np.mean(values)),
                    std=float(np.std(values)),
                    min=float(np.min(values)),
                    max=float(np.max(values)),
                    median=float(np.median(values)),
                    count=len(values),
                )

        return self.results

    def export_report(self, filename: str, format: str = "pdf") -> None:
        """Export combined report.

        Args:
            filename: Output filename
            format: Report format ('pdf', 'html', 'json')

        Raises:
            OscuraError: If export fails
        """
        if format == "json":
            self._export_json(filename)
        elif format == "pdf":
            self._export_pdf(filename)
        elif format == "html":
            self._export_html(filename)
        else:
            raise OscuraError(f"Unsupported report format: {format}")

    def _export_json(self, filename: str) -> None:
        """Export results to JSON."""
        import json

        data = {
            "trace_ids": self.results.trace_ids,
            "measurements": self.results.measurements,
            "statistics": {
                name: {
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "median": stats.median,
                    "count": stats.count,
                }
                for name, stats in self.results.statistics.items()
            },
            "metadata": self.results.metadata,
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def _export_pdf(self, filename: str) -> None:
        """Export results to PDF.

        Args:
            filename: Output filename

        Raises:
            OscuraError: PDF export not yet implemented
        """
        raise OscuraError("PDF export not yet implemented")

    def _export_html(self, filename: str) -> None:
        """Export results to HTML.

        Args:
            filename: Output filename

        Raises:
            OscuraError: HTML export not yet implemented
        """
        raise OscuraError("HTML export not yet implemented")


def load_all(pattern: str, lazy: bool = True) -> list[Any]:
    """Load all traces matching pattern.

    Args:
        pattern: Glob pattern
        lazy: If True, return lazy-loading proxy objects

    Returns:
        List of trace objects

    Raises:
        OscuraError: If no traces found
    """
    paths = glob_func(pattern)
    if not paths:
        raise OscuraError(f"No files match pattern: {pattern}")

    # For now, just return file paths
    # Would implement lazy loading proxy
    return [Path(p) for p in paths]
