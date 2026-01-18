"""Operation history tracking.

This module provides operation history tracking for analysis sessions.


Example:
    >>> history = OperationHistory()
    >>> history.record('load', {'file': 'capture.wfm'})
    >>> history.record('measure_rise_time', {'result': 1.5e-9})
    >>> print(history.to_script())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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

    def undo(self) -> HistoryEntry | None:
        """Remove and return the last entry.

        Returns:
            Removed entry, or None if empty.
        """
        if self.entries:
            return self.entries.pop()
        return None

    def clear(self) -> int:
        """Clear all history.

        Returns:
            Number of entries cleared.
        """
        count = len(self.entries)
        self.entries.clear()
        return count

    def find(
        self,
        operation: str | None = None,
        success_only: bool = False,
        since: datetime | None = None,
    ) -> list[HistoryEntry]:
        """Find entries matching criteria.

        Args:
            operation: Filter by operation name.
            success_only: Only return successful operations.
            since: Only return entries after this time.

        Returns:
            Matching entries.
        """
        results = []
        for entry in self.entries:
            if operation and entry.operation != operation:
                continue
            if success_only and not entry.success:
                continue
            if since and entry.timestamp < since:
                continue
            results.append(entry)
        return results

    def to_script(
        self,
        include_imports: bool = True,
        include_comments: bool = True,
    ) -> str:
        """Export history as Python script.

        Args:
            include_imports: Include import statements.
            include_comments: Include timestamp comments.

        Returns:
            Python script string.

        Example:
            >>> script = history.to_script()
            >>> print(script)
            # Generated by Oscura
            import oscura as osc
            osc.load("capture.wfm")
            result = osc.measure_rise_time()
        """
        lines = []

        if include_imports:
            lines.extend(
                [
                    "#!/usr/bin/env python3",
                    '"""Oscura analysis script.',
                    "",
                    f"Generated: {datetime.now().isoformat()}",
                    '"""',
                    "",
                    "import oscura as osc",
                    "",
                ]
            )

        for entry in self.entries:
            if not entry.success:
                continue

            if include_comments:
                lines.append(f"# {entry.timestamp.strftime('%H:%M:%S')} - {entry.operation}")

            lines.append(entry.to_code())
            lines.append("")

        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Get history summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.entries:
            return {
                "total_operations": 0,
                "successful": 0,
                "failed": 0,
                "total_duration_ms": 0,
                "unique_operations": 0,
            }

        successful = sum(1 for e in self.entries if e.success)
        failed = len(self.entries) - successful
        total_duration = sum(e.duration_ms for e in self.entries)
        unique_ops = len({e.operation for e in self.entries})

        # Operation frequency
        op_counts: dict[str, int] = {}
        for entry in self.entries:
            op_counts[entry.operation] = op_counts.get(entry.operation, 0) + 1

        return {
            "total_operations": len(self.entries),
            "successful": successful,
            "failed": failed,
            "total_duration_ms": total_duration,
            "unique_operations": unique_ops,
            "operation_counts": op_counts,
            "session_start": self._current_session_start.isoformat(),
            "last_operation": self.entries[-1].timestamp.isoformat() if self.entries else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "max_entries": self.max_entries,
            "session_start": self._current_session_start.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationHistory:
        """Create from dictionary."""
        entries = [HistoryEntry.from_dict(e) for e in data.get("entries", [])]
        history = cls(
            entries=entries,
            max_entries=data.get("max_entries", 0),
        )
        if "session_start" in data:
            history._current_session_start = datetime.fromisoformat(data["session_start"])
        return history


__all__ = [
    "HistoryEntry",
    "OperationHistory",
]
