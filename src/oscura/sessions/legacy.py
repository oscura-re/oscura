"""Legacy session management for backward compatibility.

This module provides backward compatibility with the old Session API,
which has been superseded by the AnalysisSession hierarchy.

For new code, use:
- GenericSession for general waveform analysis
- BlackBoxSession for protocol reverse engineering
- Or extend AnalysisSession for custom workflows

This module exists only to support existing code and tests.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from oscura.core.exceptions import SecurityError

# Session file format constants
_SESSION_MAGIC = b"OSC1"  # Magic bytes for new format with signature
_SESSION_SIGNATURE_SIZE = 32  # SHA256 hash size in bytes
_SECURITY_KEY = hashlib.sha256(b"oscura-session-v1").digest()


class AnnotationType(Enum):
    """Types of annotations."""

    POINT = "point"  # Single time point
    RANGE = "range"  # Time range
    VERTICAL = "vertical"  # Vertical line
    HORIZONTAL = "horizontal"  # Horizontal line
    REGION = "region"  # 2D region (time + amplitude)
    TEXT = "text"  # Free-floating text


@dataclass
class Annotation:
    """Single annotation on a trace.

    Attributes:
        text: Annotation text/label
        time: Time point (for point annotations)
        time_range: (start, end) time range
        amplitude: Amplitude value (for horizontal lines)
        amplitude_range: (min, max) amplitude range
        annotation_type: Type of annotation
        color: Display color (hex or name)
        style: Line style ('solid', 'dashed', 'dotted')
        visible: Whether annotation is visible
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    text: str
    time: float | None = None
    time_range: tuple[float, float] | None = None
    amplitude: float | None = None
    amplitude_range: tuple[float, float] | None = None
    annotation_type: AnnotationType = AnnotationType.POINT
    color: str = "#FF6B6B"
    style: str = "solid"
    visible: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Infer annotation type from provided parameters."""
        if self.annotation_type == AnnotationType.POINT:
            if self.amplitude_range is not None and self.time_range is not None:
                self.annotation_type = AnnotationType.REGION
            elif self.time_range is not None:
                self.annotation_type = AnnotationType.RANGE
            elif self.amplitude is not None and self.time is None:
                self.annotation_type = AnnotationType.HORIZONTAL

    @property
    def start_time(self) -> float | None:
        """Get start time for range annotations."""
        if self.time_range:
            return self.time_range[0]
        return self.time

    @property
    def end_time(self) -> float | None:
        """Get end time for range annotations."""
        if self.time_range:
            return self.time_range[1]
        return self.time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "time": self.time,
            "time_range": self.time_range,
            "amplitude": self.amplitude,
            "amplitude_range": self.amplitude_range,
            "annotation_type": self.annotation_type.value,
            "color": self.color,
            "style": self.style,
            "visible": self.visible,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create from dictionary."""
        data = data.copy()
        data["annotation_type"] = AnnotationType(data.get("annotation_type", "point"))
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class AnnotationLayer:
    """Collection of related annotations.

    Attributes:
        name: Layer name
        annotations: List of annotations
        visible: Whether layer is visible
        locked: Whether layer is locked (read-only)
        color: Default color for new annotations
        description: Layer description
    """

    name: str
    annotations: list[Annotation] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    color: str = "#FF6B6B"
    description: str = ""

    def add(
        self,
        annotation: Annotation | None = None,
        *,
        text: str = "",
        time: float | None = None,
        time_range: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Annotation:
        """Add annotation to layer.

        Args:
            annotation: Pre-built Annotation object.
            text: Annotation text (if not using pre-built).
            time: Time point.
            time_range: Time range.
            **kwargs: Additional Annotation parameters.

        Returns:
            Added annotation.

        Raises:
            ValueError: If layer is locked.
        """
        if self.locked:
            raise ValueError(f"Layer '{self.name}' is locked")

        if annotation is None:
            annotation = Annotation(
                text=text,
                time=time,
                time_range=time_range,
                color=kwargs.pop("color", self.color),
                **kwargs,
            )

        self.annotations.append(annotation)
        return annotation

    def remove(self, annotation: Annotation) -> bool:
        """Remove annotation from layer.

        Args:
            annotation: Annotation to remove.

        Returns:
            True if removed, False if not found.

        Raises:
            ValueError: If layer is locked.
        """
        if self.locked:
            raise ValueError(f"Layer '{self.name}' is locked")

        try:
            self.annotations.remove(annotation)
            return True
        except ValueError:
            return False

    def find_at_time(
        self,
        time: float,
        tolerance: float = 0.0,
    ) -> list[Annotation]:
        """Find annotations at or near a specific time.

        Args:
            time: Time to search.
            tolerance: Time tolerance for matching.

        Returns:
            List of matching annotations.
        """
        matches = []
        for ann in self.annotations:
            if ann.time is not None:
                if abs(ann.time - time) <= tolerance:
                    matches.append(ann)
            elif ann.time_range is not None and (
                ann.time_range[0] - tolerance <= time <= ann.time_range[1] + tolerance
            ):
                matches.append(ann)
        return matches

    def find_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[Annotation]:
        """Find annotations within a time range.

        Args:
            start_time: Range start.
            end_time: Range end.

        Returns:
            List of annotations within range.
        """
        matches = []
        for ann in self.annotations:
            ann_start = ann.start_time
            ann_end = ann.end_time

            if ann_start is not None and (
                start_time <= ann_start <= end_time
                or (ann_end is not None and ann_start <= end_time and ann_end >= start_time)
            ):
                matches.append(ann)

        return matches

    def clear(self) -> int:
        """Remove all annotations.

        Returns:
            Number of annotations removed.

        Raises:
            ValueError: If layer is locked.
        """
        if self.locked:
            raise ValueError(f"Layer '{self.name}' is locked")

        count = len(self.annotations)
        self.annotations.clear()
        return count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "annotations": [ann.to_dict() for ann in self.annotations],
            "visible": self.visible,
            "locked": self.locked,
            "color": self.color,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationLayer:
        """Create from dictionary."""
        data = data.copy()
        annotations_data = data.pop("annotations", [])
        layer = cls(**data)
        layer.annotations = [Annotation.from_dict(ann) for ann in annotations_data]
        return layer


@dataclass
class HistoryEntry:
    """Single history entry recording an operation.

    Attributes:
        operation: Operation name (function/method called)
        parameters: Input parameters
        result: Operation result (summary)
        timestamp: When operation was performed
        duration_ms: Operation duration in milliseconds
        success: Whether operation succeeded
        error_message: Error message if failed
        metadata: Additional metadata
    """

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "parameters": self.parameters,
            "result": self._serialize_result(self.result),
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @staticmethod
    def _serialize_result(result: Any) -> Any:
        """Serialize result for JSON storage."""
        if result is None:
            return None
        if isinstance(result, str | int | float | bool):
            return result
        if isinstance(result, dict):
            return {k: HistoryEntry._serialize_result(v) for k, v in result.items()}
        if isinstance(result, list | tuple):
            return [HistoryEntry._serialize_result(v) for v in result]
        # For complex objects, store string representation
        return str(result)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        """Create from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def to_code(self) -> str:
        """Generate Python code to replay this operation.

        Returns:
            Python code string.
        """
        # Format parameters
        params = []
        for k, v in self.parameters.items():
            if isinstance(v, str):
                params.append(f'{k}="{v}"')
            else:
                params.append(f"{k}={v!r}")

        param_str = ", ".join(params)
        return f"osc.{self.operation}({param_str})"


@dataclass
class OperationHistory:
    """History of analysis operations.

    Supports recording, replaying, and exporting operation history.

    Attributes:
        entries: List of history entries
        max_entries: Maximum entries to keep (0 = unlimited)
        auto_record: Whether to automatically record operations
    """

    entries: list[HistoryEntry] = field(default_factory=list)
    max_entries: int = 0
    auto_record: bool = True
    _current_session_start: datetime = field(default_factory=datetime.now)

    def record(
        self,
        operation: str,
        parameters: dict[str, Any] | None = None,
        result: Any = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str | None = None,
        **metadata: Any,
    ) -> HistoryEntry:
        """Record an operation.

        Args:
            operation: Operation name.
            parameters: Input parameters.
            result: Operation result.
            duration_ms: Duration in milliseconds.
            success: Whether operation succeeded.
            error_message: Error message if failed.
            **metadata: Additional metadata.

        Returns:
            Created history entry.
        """
        entry = HistoryEntry(
            operation=operation,
            parameters=parameters or {},
            result=result,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )

        self.entries.append(entry)

        # Trim if exceeded max entries
        if self.max_entries > 0 and len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

        return entry

    def clear(self) -> None:
        """Clear all history entries."""
        self.entries.clear()

    def to_script(self, include_imports: bool = True) -> str:
        """Generate Python script to replay operations.

        Args:
            include_imports: Whether to include import statement.

        Returns:
            Python script as string.
        """
        lines = []

        if include_imports:
            lines.append("import oscura as osc")
            lines.append("")

        for entry in self.entries:
            if entry.success:
                lines.append(entry.to_code())

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "max_entries": self.max_entries,
            "auto_record": self.auto_record,
            "session_start": self._current_session_start.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationHistory:
        """Create from dictionary."""
        data = data.copy()
        entries_data = data.pop("entries", [])
        if "session_start" in data:
            data["_current_session_start"] = datetime.fromisoformat(data.pop("session_start"))
        history = cls(**data)
        history.entries = [HistoryEntry.from_dict(entry) for entry in entries_data]
        return history


@dataclass
class Session:
    """Analysis session container (legacy API).

    NOTE: This is the legacy Session API for backward compatibility.
    For new code, use:
    - GenericSession for general waveform analysis
    - BlackBoxSession for protocol reverse engineering
    - Or extend AnalysisSession for custom workflows

    Manages traces, annotations, measurements, and history for a complete
    analysis session. Sessions can be saved and restored.

    Attributes:
        name: Session name
        traces: Dictionary of loaded traces (name -> trace)
        annotation_layers: Annotation layers
        measurements: Recorded measurements
        history: Operation history
        metadata: Session metadata
        created_at: Creation timestamp
        modified_at: Last modification timestamp
    """

    name: str = "Untitled Session"
    traces: dict[str, Any] = field(default_factory=dict)
    annotation_layers: dict[str, AnnotationLayer] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    history: OperationHistory = field(default_factory=OperationHistory)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    _file_path: Path | None = None

    def __post_init__(self) -> None:
        """Initialize default annotation layer."""
        if "default" not in self.annotation_layers:
            self.annotation_layers["default"] = AnnotationLayer("Default")

    def load_trace(
        self,
        path: str | Path,
        name: str | None = None,
        **load_kwargs: Any,
    ) -> Any:
        """Load a trace into the session.

        Args:
            path: Path to trace file.
            name: Name for trace in session (default: filename).
            **load_kwargs: Additional arguments for load().

        Returns:
            Loaded trace.
        """
        from oscura.loaders import load

        path = Path(path)
        trace = load(str(path), **load_kwargs)

        if name is None:
            name = path.stem

        self.traces[name] = trace
        self._mark_modified()

        self.history.record(
            "load_trace",
            {"path": str(path), "name": name},
            result=f"Loaded {name}",
        )

        return trace

    def add_trace(
        self,
        name: str,
        trace: Any,
    ) -> None:
        """Add an in-memory trace to the session.

        Args:
            name: Name for the trace in the session.
            trace: Trace object (WaveformTrace, DigitalTrace, etc.).

        Raises:
            ValueError: If name is empty or already exists.
            TypeError: If trace doesn't have expected attributes.
        """
        if not name:
            raise ValueError("Trace name cannot be empty")

        if not hasattr(trace, "data"):
            raise TypeError("Trace must have a 'data' attribute")

        self.traces[name] = trace
        self._mark_modified()

        self.history.record(
            "add_trace",
            {"name": name, "type": type(trace).__name__},
            result=f"Added {name}",
        )

    def remove_trace(self, name: str) -> None:
        """Remove a trace from the session.

        Args:
            name: Name of the trace to remove.

        Raises:
            KeyError: If trace not found.
        """
        if name not in self.traces:
            raise KeyError(f"Trace '{name}' not found in session")

        del self.traces[name]
        self._mark_modified()

        self.history.record(
            "remove_trace",
            {"name": name},
            result=f"Removed {name}",
        )

    def get_trace(self, name: str) -> Any:
        """Get trace by name.

        Args:
            name: Trace name.

        Returns:
            Trace object.
        """
        return self.traces[name]

    def list_traces(self) -> list[str]:
        """List all trace names."""
        return list(self.traces.keys())

    def annotate(
        self,
        text: str,
        *,
        time: float | None = None,
        time_range: tuple[float, float] | None = None,
        layer: str = "default",
        **kwargs: Any,
    ) -> None:
        """Add annotation to session.

        Args:
            text: Annotation text.
            time: Time point for annotation.
            time_range: Time range for annotation.
            layer: Annotation layer name.
            **kwargs: Additional annotation parameters.
        """
        if layer not in self.annotation_layers:
            self.annotation_layers[layer] = AnnotationLayer(layer)

        self.annotation_layers[layer].add(
            text=text,
            time=time,
            time_range=time_range,
            **kwargs,
        )

        self._mark_modified()

    def save(self, path: str | Path, *, compress: bool = True) -> None:
        """Save session to file.

        Args:
            path: Output file path (.tks extension).
            compress: Whether to compress with gzip.

        Raises:
            SecurityError: If session verification fails.
        """
        path = Path(path)
        self._file_path = path

        # Prepare session data
        session_data = {
            "name": self.name,
            "traces": self.traces,
            "annotation_layers": {
                name: layer.to_dict() for name, layer in self.annotation_layers.items()
            },
            "measurements": self.measurements,
            "history": self.history.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": datetime.now().isoformat(),
        }

        # Serialize
        pickled = pickle.dumps(session_data)

        # Sign the data
        signature = hmac.new(_SECURITY_KEY, pickled, hashlib.sha256).digest()

        # Combine magic + signature + data
        full_data = _SESSION_MAGIC + signature + pickled

        # Write
        if compress:
            with gzip.open(path, "wb") as f:
                f.write(full_data)
        else:
            path.write_bytes(full_data)

    def _mark_modified(self) -> None:
        """Mark session as modified."""
        self.modified_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for export."""
        return {
            "name": self.name,
            "traces": {name: str(type(trace).__name__) for name, trace in self.traces.items()},
            "annotation_layers": {
                name: layer.to_dict() for name, layer in self.annotation_layers.items()
            },
            "measurements": self.measurements,
            "history": self.history.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }


def load_session(path: str | Path) -> Session:
    """Load session from file.

    Args:
        path: Path to session file (.tks).

    Returns:
        Loaded Session object.

    Raises:
        SecurityError: If session verification fails or file format is invalid.
    """
    path = Path(path)

    # Read file as raw bytes first
    raw_data = path.read_bytes()

    # Detect and decompress gzip files (magic bytes: 0x1f 0x8b)
    if raw_data[:2] == b"\x1f\x8b":
        import io

        with gzip.open(io.BytesIO(raw_data), "rb") as f:
            full_data = f.read()
    else:
        full_data = raw_data

    # Check magic - missing magic bytes is a security issue (file lacks HMAC signature)
    if not full_data.startswith(_SESSION_MAGIC):
        raise SecurityError(
            "Invalid session file format: missing HMAC signature (magic bytes not found)"
        )

    # Extract signature and data
    signature = full_data[len(_SESSION_MAGIC) : len(_SESSION_MAGIC) + _SESSION_SIGNATURE_SIZE]
    pickled = full_data[len(_SESSION_MAGIC) + _SESSION_SIGNATURE_SIZE :]

    # Verify signature
    expected_signature = hmac.new(_SECURITY_KEY, pickled, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise SecurityError("Session file signature verification failed (data may be tampered)")

    # Deserialize
    session_data = pickle.loads(pickled)

    # Reconstruct session
    session = Session(
        name=session_data["name"],
        traces=session_data.get("traces", {}),
        measurements=session_data.get("measurements", {}),
        metadata=session_data.get("metadata", {}),
        created_at=datetime.fromisoformat(session_data["created_at"]),
        modified_at=datetime.fromisoformat(session_data["modified_at"]),
    )

    # Restore annotation layers
    for name, layer_data in session_data.get("annotation_layers", {}).items():
        session.annotation_layers[name] = AnnotationLayer.from_dict(layer_data)

    # Restore history
    session.history = OperationHistory.from_dict(session_data["history"])

    session._file_path = path

    return session


__all__ = [
    "Annotation",
    "AnnotationLayer",
    "AnnotationType",
    "HistoryEntry",
    "OperationHistory",
    "Session",
    "load_session",
]
