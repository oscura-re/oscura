"""Generic analysis session implementation.

This module provides GenericSession - a concrete implementation of AnalysisSession
for general-purpose signal analysis. It wraps the existing session.Session class
to provide the unified AnalysisSession interface while maintaining backward
compatibility.

Example:
    >>> from oscura.sessions import GenericSession
    >>> from oscura.hardware.acquisition import FileSource
    >>>
    >>> # Create generic session
    >>> session = GenericSession(name="Debug Session")
    >>> session.add_recording("capture1", FileSource("signal1.wfm"))
    >>> session.add_recording("capture2", FileSource("signal2.wfm"))
    >>>
    >>> # Compare recordings
    >>> diff = session.compare("capture1", "capture2")
    >>> print(f"Similarity: {diff.similarity_score:.2%}")
    >>>
    >>> # Analyze
    >>> results = session.analyze()  # Generic waveform analysis

Pattern:
    GenericSession is the default session type for non-domain-specific
    analysis. Use domain-specific sessions (CANSession, SerialSession, etc.)
    when working within a specific protocol or analysis domain.

Migration:
    The existing oscura.session.Session class continues to work unchanged.
    GenericSession provides the new AnalysisSession interface while
    delegating to the existing Session implementation internally.

References:
    Architecture Plan Phase 0.3: AnalysisSession Generic Implementation
    src/oscura/session/session.py: Legacy Session class
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oscura.sessions.base import AnalysisSession


class GenericSession(AnalysisSession):
    """Generic analysis session for general-purpose signal analysis.

    Provides the unified AnalysisSession interface for non-domain-specific
    analysis. This is the default session type when you don't need CAN,
    Serial, or other specialized functionality.

    Example:
        >>> from oscura.sessions import GenericSession
        >>> from oscura.hardware.acquisition import FileSource
        >>>
        >>> session = GenericSession()
        >>> session.add_recording("test1", FileSource("capture1.wfm"))
        >>> session.add_recording("test2", FileSource("capture2.wfm"))
        >>>
        >>> # Compare traces
        >>> diff = session.compare("test1", "test2")
        >>> print(f"Changed: {diff.changed_bytes} samples")
        >>>
        >>> # Generic analysis
        >>> results = session.analyze()
        >>> print(results["num_recordings"])
    """

    def analyze(self) -> dict[str, Any]:
        """Perform generic waveform analysis on all recordings.

        Analyzes all loaded recordings and provides summary statistics,
        basic waveform measurements, and comparison results.

        Returns:
            Dictionary with analysis results:
                - num_recordings: Number of recordings
                - recordings: List of recording names
                - summary: Summary statistics for each recording

        Example:
            >>> session = GenericSession()
            >>> session.add_recording("sig", FileSource("capture.wfm"))
            >>> results = session.analyze()
            >>> print(results["summary"]["sig"]["mean"])
        """
        import numpy as np

        results: dict[str, Any] = {
            "num_recordings": len(self.recordings),
            "recordings": self.list_recordings(),
            "summary": {},
        }

        # Analyze each recording
        from oscura.core.types import IQTrace

        for name in self.list_recordings():
            trace = self.get_recording(name)

            # Handle IQTrace separately
            if isinstance(trace, IQTrace):
                raise TypeError("IQTrace analysis not yet supported in GenericSession")

            # Basic statistics
            summary = {
                "num_samples": len(trace.data),
                "sample_rate": trace.metadata.sample_rate,
                "duration": len(trace.data) / trace.metadata.sample_rate,
                "mean": float(np.mean(trace.data)),
                "std": float(np.std(trace.data)),
                "min": float(np.min(trace.data)),
                "max": float(np.max(trace.data)),
                "rms": float(np.sqrt(np.mean(trace.data**2))),
            }

            results["summary"][name] = summary

        # If multiple recordings, add comparisons
        if len(self.recordings) >= 2:
            results["comparisons"] = {}
            names = self.list_recordings()
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    comparison_key = f"{names[i]}_vs_{names[j]}"
                    comp_result = self.compare(names[i], names[j])
                    results["comparisons"][comparison_key] = {
                        "similarity": comp_result.similarity_score,
                        "changed_samples": comp_result.changed_bytes,
                    }

        return results

    def export_results(self, format: str, path: str | Path) -> None:
        """Export analysis results to file.

        Extends base export with additional formats for generic analysis.

        Args:
            format: Export format ("report", "json", "csv").
            path: Output file path.

        Raises:
            ValueError: If format not supported.

        Example:
            >>> session.export_results("report", "analysis.txt")
            >>> session.export_results("json", "results.json")
        """
        path = Path(path)

        if format == "json":
            # Export as JSON
            import json

            results = self.analyze()

            with open(path, "w") as f:
                json.dump(results, f, indent=2)

        elif format == "csv":
            # Export summary as CSV
            import csv

            results = self.analyze()

            with open(path, "w", newline="") as f:
                if not results["summary"]:
                    return  # No data to export

                # Get fieldnames from first recording
                first_rec = next(iter(results["summary"].values()))
                fieldnames = ["recording"] + list(first_rec.keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for name, summary in results["summary"].items():
                    row = {"recording": name, **summary}
                    writer.writerow(row)

        else:
            # Fall back to base implementation (text report)
            super().export_results(format, path)


__all__ = ["GenericSession"]
