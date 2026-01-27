"""Base class for analysis sessions.

This module defines AnalysisSession - the abstract base class for all
interactive analysis sessions in Oscura. It provides a unified pattern
for domain-specific sessions (CAN, Serial, BlackBox, etc.).

Example:
    >>> from oscura.sessions import AnalysisSession
    >>> from oscura.hardware.acquisition import FileSource, HardwareSource
    >>>
    >>> # Domain-specific sessions extend AnalysisSession
    >>> class CANSession(AnalysisSession):
    ...     def analyze(self):
    ...         # CAN-specific analysis
    ...         return self.discover_signals()
    ...
    ...     def discover_signals(self):
    ...         # Extract CAN signals from recordings
    ...         pass
    >>>
    >>> # Unified interface across all sessions
    >>> session = CANSession()
    >>> session.add_recording("baseline", FileSource("idle.blf"))
    >>> session.add_recording("active", FileSource("running.blf"))
    >>> diff = session.compare("baseline", "active")

Pattern:
    All domain-specific sessions (CAN, Serial, BlackBox, etc.) inherit
    from AnalysisSession and provide:
    - Recording management (add_recording, list_recordings)
    - Comparison and differential analysis
    - Result export (reports, specs, dissectors)
    - Domain-specific analysis methods (abstract)

Benefits:
    - Consistent API across all analysis domains
    - Polymorphic session handling
    - Shared infrastructure (comparison, export, history)
    - Domain-specific specialization via inheritance

References:
    Architecture Plan Phase 0.3: AnalysisSession Base Class
    docs/architecture/api-patterns.md: When to use Sessions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.core.types import Trace
    from oscura.hardware.acquisition import Source


@dataclass
class ComparisonResult:
    """Result of comparing two recordings.

    Attributes:
        recording1: Name of first recording.
        recording2: Name of second recording.
        changed_bytes: Number of bytes that differ.
        changed_regions: List of (start, end, description) tuples.
        similarity_score: Similarity metric (0.0 to 1.0).
        details: Additional comparison details.

    Example:
        >>> result = session.compare("baseline", "stimulus")
        >>> print(f"Changed bytes: {result.changed_bytes}")
        >>> print(f"Similarity: {result.similarity_score:.2%}")
    """

    recording1: str
    recording2: str
    changed_bytes: int
    changed_regions: list[tuple[int, int, str]] = field(default_factory=list)
    similarity_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class AnalysisSession(ABC):
    """Abstract base class for all analysis sessions.

    Provides unified interface for interactive signal analysis across
    different domains (CAN, Serial, RF, BlackBox, etc.). All domain-specific
    sessions extend this class.

    Subclasses must implement:
        - analyze(): Domain-specific analysis method

    Subclasses may override:
        - export_results(): Custom export formats
        - compare(): Domain-specific comparison logic

    Attributes:
        name: Session name.
        recordings: Dictionary mapping names to (source, trace) tuples.
        metadata: Session metadata dictionary.
        created_at: Session creation timestamp.
        modified_at: Last modification timestamp.

    Example:
        >>> # Subclass for domain-specific analysis
        >>> class SerialSession(AnalysisSession):
        ...     def analyze(self):
        ...         # Detect baud rate, decode UART
        ...         baud = self.detect_baud_rate()
        ...         frames = self.decode_uart(baud)
        ...         return {"baud_rate": baud, "frames": frames}
        ...
        ...     def detect_baud_rate(self):
        ...         # Domain-specific logic
        ...         pass
        >>>
        >>> session = SerialSession()
        >>> session.add_recording("capture", FileSource("uart.wfm"))
        >>> results = session.analyze()
    """

    def __init__(self, name: str = "Untitled Session") -> None:
        """Initialize analysis session.

        Args:
            name: Session name (default: "Untitled Session").

        Example:
            >>> session = CANSession(name="Vehicle Debug Session")
        """
        self.name = name
        self.recordings: dict[str, tuple[Source, Trace | None]] = {}
        self.metadata: dict[str, Any] = {}
        self.created_at = datetime.now()
        self.modified_at = datetime.now()

    def add_recording(
        self,
        name: str,
        source: Source,
        *,
        load_immediately: bool = True,
    ) -> None:
        """Add a recording to the session.

        Args:
            name: Name for this recording (e.g., "baseline", "stimulus1").
            source: Source to acquire data from (FileSource, HardwareSource, etc.).
            load_immediately: If True, load trace now. If False, defer loading.

        Raises:
            ValueError: If name already exists.

        Example:
            >>> from oscura.hardware.acquisition import FileSource
            >>> session.add_recording("baseline", FileSource("idle.blf"))
            >>> session.add_recording("active", FileSource("running.blf"))
        """
        if name in self.recordings:
            raise ValueError(f"Recording '{name}' already exists in session")

        # Load trace if requested
        trace = source.read() if load_immediately else None

        self.recordings[name] = (source, trace)
        self.modified_at = datetime.now()

    def get_recording(self, name: str) -> Trace:
        """Get a recording by name, loading if necessary.

        Args:
            name: Recording name.

        Returns:
            Loaded trace.

        Raises:
            KeyError: If recording not found.

        Example:
            >>> trace = session.get_recording("baseline")
            >>> print(f"Loaded {len(trace.data)} samples")
        """
        if name not in self.recordings:
            available = list(self.recordings.keys())
            raise KeyError(f"Recording '{name}' not found. Available: {available}")

        source, trace = self.recordings[name]

        # Load if not already loaded
        if trace is None:
            trace = source.read()
            self.recordings[name] = (source, trace)

        return trace

    def list_recordings(self) -> list[str]:
        """List all recording names in the session.

        Returns:
            List of recording names.

        Example:
            >>> session.list_recordings()
            ['baseline', 'stimulus1', 'stimulus2']
        """
        return list(self.recordings.keys())

    def compare(self, name1: str, name2: str) -> ComparisonResult:
        """Compare two recordings (differential analysis).

        Default implementation provides basic byte-level comparison.
        Subclasses can override for domain-specific comparison logic.

        Args:
            name1: First recording name.
            name2: Second recording name.

        Returns:
            ComparisonResult with differences.

        Raises:
            KeyError: If recordings not found.

        Example:
            >>> result = session.compare("baseline", "stimulus")
            >>> print(f"Changed: {result.changed_bytes} bytes")
            >>> print(f"Similarity: {result.similarity_score:.2%}")
        """
        trace1 = self.get_recording(name1)
        trace2 = self.get_recording(name2)

        # Basic comparison - count differing samples
        import numpy as np

        from oscura.core.types import IQTrace

        # Handle IQTrace separately
        if isinstance(trace1, IQTrace) or isinstance(trace2, IQTrace):
            raise TypeError("IQTrace comparison not yet supported in base session")

        min_len = min(len(trace1.data), len(trace2.data))
        data1 = trace1.data[:min_len]
        data2 = trace2.data[:min_len]

        # For analog traces, use threshold comparison
        threshold = 0.01  # 1% tolerance
        changed = np.abs(data1 - data2) > threshold
        changed_count = int(np.sum(changed))

        # Similarity score
        similarity = 1.0 - (changed_count / min_len)

        return ComparisonResult(
            recording1=name1,
            recording2=name2,
            changed_bytes=changed_count,
            similarity_score=similarity,
            details={
                "trace1_length": len(trace1.data),
                "trace2_length": len(trace2.data),
                "compared_length": min_len,
            },
        )

    def export_results(self, format: str, path: str | Path) -> None:
        """Export analysis results to file.

        Default implementation provides basic export. Subclasses should
        override to support domain-specific formats (DBC, Wireshark, etc.).

        Args:
            format: Export format (e.g., "report", "json", "csv").
            path: Output file path.

        Raises:
            ValueError: If format not supported.

        Example:
            >>> session.export_results("report", "analysis.txt")
            >>> session.export_results("json", "results.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "report":
            # Basic text report
            with open(path, "w") as f:
                f.write(f"Analysis Session: {self.name}\n")
                f.write(f"Created: {self.created_at}\n")
                f.write(f"Modified: {self.modified_at}\n\n")
                f.write(f"Recordings: {len(self.recordings)}\n")
                for name in self.list_recordings():
                    f.write(f"  - {name}\n")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    @abstractmethod
    def analyze(self) -> Any:
        """Perform domain-specific analysis.

        Subclasses must implement this method to provide domain-specific
        analysis functionality (CAN signal discovery, UART decoding,
        protocol reverse engineering, etc.).

        Returns:
            Analysis results (format depends on domain).

        Example:
            >>> class CANSession(AnalysisSession):
            ...     def analyze(self):
            ...         signals = self.discover_signals()
            ...         return {"signals": signals}
        """

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r}, recordings={len(self.recordings)})"


__all__ = ["AnalysisSession", "ComparisonResult"]
