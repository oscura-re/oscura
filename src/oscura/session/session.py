"""Analysis session management.

This module provides session save/restore functionality for Oscura.


Example:
    >>> session = Session()
    >>> session.load_trace('capture.wfm')
    >>> session.save('debug_session.tks')
    >>>
    >>> # Later...
    >>> session = load_session('debug_session.tks')
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from oscura.core.exceptions import SecurityError
from oscura.session.annotations import AnnotationLayer
from oscura.session.history import OperationHistory

# Session file format constants
_SESSION_MAGIC = b"OSC1"  # Magic bytes for new format with signature
_SESSION_SIGNATURE_SIZE = 32  # SHA256 hash size in bytes
_SECURITY_KEY = hashlib.sha256(b"oscura-session-v1").digest()


@dataclass
class Session:
    """Analysis session container.

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

        This method allows adding traces that were created programmatically
        or loaded separately, rather than loading from a file.

        Args:
            name: Name for the trace in the session.
            trace: Trace object (WaveformTrace, DigitalTrace, etc.).

        Raises:
            ValueError: If name is empty or already exists.
            TypeError: If trace doesn't have expected attributes.

        Example:
            >>> session = Session()
            >>> data = np.sin(np.linspace(0, 2*np.pi, 1000))
            >>> trace = osc.WaveformTrace(data=data, metadata=osc.TraceMetadata(sample_rate=1e6))
            >>> session.add_trace("my_trace", trace)
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

        self.history.record(
            "annotate",
            {"text": text, "time": time, "layer": layer},
        )

    def get_annotations(
        self,
        layer: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> list[Any]:
        """Get annotations.

        Args:
            layer: Filter by layer name (None for all layers).
            time_range: Filter by time range.

        Returns:
            List of annotations.
        """
        annotations = []

        layers = [self.annotation_layers[layer]] if layer else self.annotation_layers.values()

        for ann_layer in layers:
            if time_range:
                annotations.extend(ann_layer.find_in_range(time_range[0], time_range[1]))
            else:
                annotations.extend(ann_layer.annotations)

        return annotations

    def record_measurement(
        self,
        name: str,
        value: Any,
        unit: str = "",
        trace_name: str | None = None,
        **metadata: Any,
    ) -> None:
        """Record a measurement result.

        Args:
            name: Measurement name (e.g., 'rise_time').
            value: Measurement value.
            unit: Unit of measurement.
            trace_name: Associated trace name.
            **metadata: Additional metadata.
        """
        self.measurements[name] = {
            "value": value,
            "unit": unit,
            "trace": trace_name,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }
        self._mark_modified()

        self.history.record(
            f"measure_{name}",
            {"trace": trace_name},
            result=f"{value} {unit}".strip(),
        )

    def get_measurements(self) -> dict[str, Any]:
        """Get all recorded measurements."""
        return self.measurements.copy()

    def save(
        self,
        path: str | Path | None = None,
        *,
        include_traces: bool = True,
        compress: bool = True,
    ) -> Path:
        """Save session to file with HMAC signature for integrity verification.

        Args:
            path: Output path (default: use existing or generate).
            include_traces: Include trace data in session file.
            compress: Compress session file with gzip.

        Returns:
            Path to saved file.

        Example:
            >>> session.save('analysis.tks')

        Security Note:
            Session files now include HMAC signatures for integrity verification.
            Files are still pickle-based - only load from trusted sources.
            For secure data exchange with untrusted parties, use JSON or HDF5
            export formats instead.
        """
        if path is None:
            path = self._file_path or Path(f"{self.name.replace(' ', '_')}.tks")
        else:
            path = Path(path)

        self._file_path = path
        self._mark_modified()

        # Build session data
        data = self._to_dict(include_traces=include_traces)

        # Serialize with pickle
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        # Compute HMAC signature
        signature = hmac.new(_SECURITY_KEY, serialized, hashlib.sha256).digest()

        # Write: magic bytes + signature + pickled data
        if compress:
            with gzip.open(path, "wb") as f:
                f.write(_SESSION_MAGIC)
                f.write(signature)
                f.write(serialized)
        else:
            with open(path, "wb") as f:
                f.write(_SESSION_MAGIC)
                f.write(signature)
                f.write(serialized)

        self.history.record("save", {"path": str(path)})

        return path

    def _to_dict(self, include_traces: bool = True) -> dict[str, Any]:
        """Convert session to dictionary."""
        data: dict[str, Any] = {
            "version": "1.0",
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "annotation_layers": {
                name: layer.to_dict() for name, layer in self.annotation_layers.items()
            },
            "measurements": self.measurements,
            "history": self.history.to_dict(),
            "metadata": self.metadata,
        }

        if include_traces:
            # Store traces with their data
            data["traces"] = {}
            for name, trace in self.traces.items():
                trace_data = {
                    "type": type(trace).__name__,
                    "data": trace.data.tolist() if hasattr(trace, "data") else None,
                    "sample_rate": (
                        trace.metadata.sample_rate if hasattr(trace, "metadata") else None
                    ),
                }
                data["traces"][name] = trace_data
        else:
            data["traces"] = {}

        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Session:
        """Create session from dictionary."""
        session = cls(
            name=data.get("name", "Untitled Session"),
            metadata=data.get("metadata", {}),
        )

        if "created_at" in data:
            session.created_at = datetime.fromisoformat(data["created_at"])
        if "modified_at" in data:
            session.modified_at = datetime.fromisoformat(data["modified_at"])

        # Restore annotation layers
        for name, layer_data in data.get("annotation_layers", {}).items():
            session.annotation_layers[name] = AnnotationLayer.from_dict(layer_data)

        # Restore measurements
        session.measurements = data.get("measurements", {})

        # Restore history
        if "history" in data:
            session.history = OperationHistory.from_dict(data["history"])

        # Restore traces (if included)
        if "traces" in data:
            from oscura.core.types import WaveformTrace

            for name, trace_data in data["traces"].items():
                if trace_data.get("data") is not None:
                    session.traces[name] = WaveformTrace(  # type: ignore[call-arg]
                        data=np.array(trace_data["data"]),
                        sample_rate=trace_data.get("sample_rate", 1.0),
                    )

        return session

    def _mark_modified(self) -> None:
        """Update modification timestamp."""
        self.modified_at = datetime.now()

    def summary(self) -> str:
        """Get session summary."""
        lines = [
            f"Session: {self.name}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Modified: {self.modified_at.strftime('%Y-%m-%d %H:%M')}",
            f"Traces: {len(self.traces)}",
            f"Annotations: {sum(len(l.annotations) for l in self.annotation_layers.values())}",  # noqa: E741
            f"Measurements: {len(self.measurements)}",
            f"History entries: {len(self.history.entries)}",
        ]
        return "\n".join(lines)


def load_session(path: str | Path) -> Session:
    """Load session from file with HMAC signature verification.

    Session files must be in the current OSC1 format with HMAC signature.
    Legacy session files without signatures are not supported.

    Args:
        path: Path to session file (.tks).

    Returns:
        Loaded Session object.

    Raises:
        SecurityError: If signature verification fails or file is not in OSC1 format.
        gzip.BadGzipFile: If file is neither valid gzip nor uncompressed session.

    Example:
        >>> session = load_session('debug_session.tks')
        >>> print(session.list_traces())

    Security Warning:
        Session files use pickle serialization. Only load session files from
        trusted sources. Loading a malicious .tks file could execute arbitrary
        code. Never load session files from untrusted or unknown sources.

        All session files must include HMAC signatures for integrity verification.
        For secure data exchange with untrusted parties, consider exporting to
        JSON or HDF5 formats instead of using pickle-based session files.
    """
    path = Path(path)

    def _load_with_verification(f: Any) -> dict[str, Any]:
        """Load and verify session file with HMAC signature.

        Args:
            f: File object (gzip or regular).

        Returns:
            Deserialized session dictionary.

        Raises:
            SecurityError: If magic bytes or signature verification fails.
        """
        # Read magic bytes
        magic = f.read(len(_SESSION_MAGIC))

        if magic != _SESSION_MAGIC:
            raise SecurityError(
                "This is a legacy session file. Please re-save with current version.",
                file_path=str(path),
                check_type="Session format",
                details="Expected OSC1 format with HMAC signature",
            )

        # Read signature and payload
        signature = f.read(_SESSION_SIGNATURE_SIZE)
        serialized = f.read()

        if not signature or not serialized:
            raise SecurityError(
                "This is a legacy session file. Please re-save with current version.",
                file_path=str(path),
                check_type="Session format",
                details="File is incomplete or corrupted",
            )

        # Verify HMAC signature
        expected = hmac.new(_SECURITY_KEY, serialized, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise SecurityError(
                "Session file signature verification failed",
                file_path=str(path),
                check_type="HMAC signature",
                details="File may be corrupted or tampered with",
            )

        # Deserialize verified data
        data = cast("dict[str, Any]", pickle.loads(serialized))
        return data

    # Try loading (compressed first, then uncompressed)
    try:
        with gzip.open(path, "rb") as f:
            data = _load_with_verification(f)
    except gzip.BadGzipFile:
        with open(path, "rb") as f:  # type: ignore[assignment]
            data = _load_with_verification(f)

    session = Session._from_dict(data)
    session._file_path = path

    return session


__all__ = [
    "Session",
    "load_session",
]
