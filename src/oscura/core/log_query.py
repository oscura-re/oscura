"""Log query and export functionality.

This module provides searchable log querying with filtering and export
capabilities for analysis and reporting.


Example:
    >>> from oscura.core.log_query import LogQuery
    >>> query = LogQuery()
    >>> # Query last hour of ERROR logs
    >>> from datetime import datetime, UTC, timedelta
    >>> results = query.query_logs(
    ...     start_time=datetime.now(UTC) - timedelta(hours=1),
    ...     level="ERROR"
    ... )
    >>> # Export to JSON
    >>> query.export_logs(results, "errors.json", format="json")

References:
    LOG-010: Searchable Log Query and Export
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from oscura.core.logging import format_timestamp

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class LogRecord:
    """Structured log record for querying.

    Attributes:
        timestamp: ISO 8601 timestamp of the log entry.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        module: Logger name/module.
        message: Log message.
        correlation_id: Optional correlation ID for tracing.
        metadata: Additional metadata fields.

    References:
        LOG-010: Searchable Log Query
    """

    timestamp: str
    level: str
    module: str
    message: str
    correlation_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert log record to dictionary.

        Returns:
            Dictionary representation of the log record.
        """
        result = asdict(self)
        if result["metadata"] is None:
            result["metadata"] = {}
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogRecord:
        """Create log record from dictionary.

        Args:
            data: Dictionary containing log record data.

        Returns:
            LogRecord instance.
        """
        return cls(
            timestamp=data["timestamp"],
            level=data["level"],
            module=data["module"],
            message=data["message"],
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata"),
        )


class LogQuery:
    """Query and filter log records from various sources.

    Provides structured querying of logs with filtering by timestamp,
    level, module, correlation ID, and message patterns. Supports
    pagination and multiple export formats.

    Example:
        >>> query = LogQuery()
        >>> # Load logs from file
        >>> query.load_from_file("oscura.log")
        >>> # Query with filters
        >>> results = query.query_logs(
        ...     level="ERROR",
        ...     module_pattern="oscura.loaders.*"
        ... )
        >>> # Export filtered results
        >>> query.export_logs(results, "filtered.csv", format="csv")

    References:
        LOG-010: Searchable Log Query and Export
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        """Initialize log query engine."""
        self._records: list[LogRecord] = []

    def load_from_file(self, path: str, format: Literal["json", "text"] = "text") -> int:
        """Load log records from file.

        Args:
            path: Path to log file.
            format: File format (json for JSON lines, text for plain text).

        Returns:
            Number of records loaded.

        Raises:
            FileNotFoundError: If log file does not exist.
            ValueError: If format is not supported.

        Example:
            >>> query = LogQuery()
            >>> count = query.load_from_file("logs.json", format="json")
            >>> print(f"Loaded {count} records")

        References:
            LOG-010: Searchable Log Query
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        if format == "json":
            return self._load_json_lines(path_obj)
        elif format == "text":
            return self._load_text(path_obj)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def add_record(self, record: LogRecord) -> None:
        """Add a log record to the query index.

        Args:
            record: LogRecord to add.

        Example:
            >>> query = LogQuery()
            >>> record = LogRecord(
            ...     timestamp="2025-12-21T10:00:00.000000Z",
            ...     level="INFO",
            ...     module="oscura.test",
            ...     message="Test message"
            ... )
            >>> query.add_record(record)
        """
        self._records.append(record)

    def query_logs(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        level: str | None = None,
        module: str | None = None,
        module_pattern: str | None = None,
        correlation_id: str | None = None,
        message_pattern: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[LogRecord]:
        """Query log records with filtering.

        Args:
            start_time: Return only logs after this time (UTC).
            end_time: Return only logs before this time (UTC).
            level: Filter by exact log level.
            module: Filter by exact module name.
            module_pattern: Filter by module name pattern (glob-style, e.g., "oscura.loaders.*").
            correlation_id: Filter by correlation ID.
            message_pattern: Filter by message regex pattern.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).

        Returns:
            List of matching LogRecord objects.

        Example:
            >>> from datetime import datetime, UTC, timedelta
            >>> query = LogQuery()
            >>> # Last hour of errors
            >>> results = query.query_logs(
            ...     start_time=datetime.now(UTC) - timedelta(hours=1),
            ...     level="ERROR"
            ... )
            >>> # Specific module with pattern
            >>> results = query.query_logs(
            ...     module_pattern="oscura.analyzers.*",
            ...     message_pattern="FFT.*failed"
            ... )
            >>> # Paginated results
            >>> page_1 = query.query_logs(limit=100, offset=0)
            >>> page_2 = query.query_logs(limit=100, offset=100)

        References:
            LOG-010: Searchable Log Query and Export
        """
        results = self._records.copy()

        # Apply all filters
        results = self._filter_by_time(results, start_time, end_time)
        results = self._filter_by_level(results, level)
        results = self._filter_by_module(results, module, module_pattern)
        results = self._filter_by_correlation(results, correlation_id)
        results = self._filter_by_message(results, message_pattern)

        # Apply pagination
        results = self._apply_pagination(results, offset, limit)

        return results

    def _filter_by_time(
        self,
        results: list[LogRecord],
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> list[LogRecord]:
        """Filter records by timestamp range.

        Args:
            results: Input records.
            start_time: Start time filter.
            end_time: End time filter.

        Returns:
            Filtered records.
        """
        if start_time is not None:
            start_str = format_timestamp(start_time, format="iso8601")
            results = [r for r in results if r.timestamp >= start_str]

        if end_time is not None:
            end_str = format_timestamp(end_time, format="iso8601")
            results = [r for r in results if r.timestamp <= end_str]

        return results

    def _filter_by_level(self, results: list[LogRecord], level: str | None) -> list[LogRecord]:
        """Filter records by log level.

        Args:
            results: Input records.
            level: Level filter.

        Returns:
            Filtered records.
        """
        if level is not None:
            return [r for r in results if r.level == level.upper()]
        return results

    def _filter_by_module(
        self,
        results: list[LogRecord],
        module: str | None,
        module_pattern: str | None,
    ) -> list[LogRecord]:
        """Filter records by module name.

        Args:
            results: Input records.
            module: Exact module filter.
            module_pattern: Module pattern filter.

        Returns:
            Filtered records.
        """
        if module is not None:
            results = [r for r in results if r.module == module]

        if module_pattern is not None:
            # Convert glob pattern to regex
            pattern = module_pattern.replace(".", r"\.").replace("*", ".*")
            regex = re.compile(f"^{pattern}$")
            results = [r for r in results if regex.match(r.module)]

        return results

    def _filter_by_correlation(
        self, results: list[LogRecord], correlation_id: str | None
    ) -> list[LogRecord]:
        """Filter records by correlation ID.

        Args:
            results: Input records.
            correlation_id: Correlation ID filter.

        Returns:
            Filtered records.
        """
        if correlation_id is not None:
            return [r for r in results if r.correlation_id == correlation_id]
        return results

    def _filter_by_message(
        self, results: list[LogRecord], message_pattern: str | None
    ) -> list[LogRecord]:
        """Filter records by message pattern.

        Args:
            results: Input records.
            message_pattern: Message pattern filter.

        Returns:
            Filtered records.
        """
        if message_pattern is not None:
            regex = re.compile(message_pattern)
            return [r for r in results if regex.search(r.message)]
        return results

    def _apply_pagination(
        self, results: list[LogRecord], offset: int, limit: int | None
    ) -> list[LogRecord]:
        """Apply pagination to results.

        Args:
            results: Input records.
            offset: Number to skip.
            limit: Maximum to return.

        Returns:
            Paginated records.
        """
        if offset > 0:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def export_logs(
        self,
        records: list[LogRecord],
        path: str,
        format: Literal["json", "csv", "text"] = "json",
    ) -> None:
        """Export log records to file.

        Args:
            records: List of LogRecord objects to export.
            path: Output file path.
            format: Export format (json, csv, or text).

        Raises:
            ValueError: If format is not supported.

        Example:
            >>> query = LogQuery()
            >>> results = query.query_logs(level="ERROR")
            >>> query.export_logs(results, "errors.json", format="json")
            >>> query.export_logs(results, "errors.csv", format="csv")
            >>> query.export_logs(results, "errors.txt", format="text")

        References:
            LOG-010: Searchable Log Query and Export
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._export_json(records, path_obj)
        elif format == "csv":
            self._export_csv(records, path_obj)
        elif format == "text":
            self._export_text(records, path_obj)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded log records.

        Returns:
            Dictionary with statistics:
            - total: Total number of records
            - by_level: Count by log level
            - by_module: Count by module
            - time_range: Earliest and latest timestamps

        Example:
            >>> query = LogQuery()
            >>> query.load_from_file("logs.json")
            >>> stats = query.get_statistics()
            >>> print(f"Total logs: {stats['total']}")
            >>> print(f"Errors: {stats['by_level'].get('ERROR', 0)}")

        References:
            LOG-010: Log Query
        """
        if not self._records:
            return {
                "total": 0,
                "by_level": {},
                "by_module": {},
                "time_range": None,
            }

        from collections import Counter

        level_counts = Counter(r.level for r in self._records)
        module_counts = Counter(r.module for r in self._records)

        timestamps = sorted(r.timestamp for r in self._records)
        time_range = {
            "earliest": timestamps[0],
            "latest": timestamps[-1],
        }

        return {
            "total": len(self._records),
            "by_level": dict(level_counts),
            "by_module": dict(module_counts.most_common(20)),
            "time_range": time_range,
        }

    def clear(self) -> None:
        """Clear all loaded log records.

        Example:
            >>> query = LogQuery()
            >>> query.clear()
        """
        self._records.clear()

    def _load_json_lines(self, path: Path) -> int:
        """Load JSON lines format logs.

        Args:
            path: Path to JSON lines file.

        Returns:
            Number of records loaded.
        """
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = LogRecord(
                        timestamp=data.get("timestamp", ""),
                        level=data.get("level", "INFO"),
                        module=data.get("module", "unknown"),
                        message=data.get("message", ""),
                        correlation_id=data.get("correlation_id"),
                        metadata={
                            k: v
                            for k, v in data.items()
                            if k
                            not in ("timestamp", "level", "module", "message", "correlation_id")
                        },
                    )
                    self._records.append(record)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed lines
                    continue
        return count

    def _load_text(self, path: Path) -> int:
        """Load plain text format logs.

        Attempts to parse common log formats.

        Args:
            path: Path to text log file.

        Returns:
            Number of records loaded.
        """
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Try to parse common format: TIMESTAMP [LEVEL] MODULE: MESSAGE
                match = re.match(
                    r"^(\S+)\s+\[(\w+)\]\s+(\S+):\s+(.*)$",
                    line,
                )
                if match:
                    timestamp, level, module, message = match.groups()
                    record = LogRecord(
                        timestamp=timestamp,
                        level=level,
                        module=module,
                        message=message,
                    )
                    self._records.append(record)
                    count += 1

        return count

    def _export_json(self, records: list[LogRecord], path: Path) -> None:
        """Export records as JSON.

        Args:
            records: Records to export.
            path: Output path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in records],
                f,
                indent=2,
                default=str,
            )

    def _export_csv(self, records: list[LogRecord], path: Path) -> None:
        """Export records as CSV.

        Args:
            records: Records to export.
            path: Output path.
        """
        if not records:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "level", "module", "message", "correlation_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                writer.writerow(
                    {
                        "timestamp": record.timestamp,
                        "level": record.level,
                        "module": record.module,
                        "message": record.message,
                        "correlation_id": record.correlation_id or "",
                    }
                )

    def _export_text(self, records: list[LogRecord], path: Path) -> None:
        """Export records as plain text.

        Args:
            records: Records to export.
            path: Output path.
        """
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                line = f"{record.timestamp} [{record.level}] {record.module}: {record.message}"
                if record.correlation_id:
                    line += f" [corr_id={record.correlation_id}]"
                f.write(line + "\n")


def query_logs(
    log_file: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    level: str | None = None,
    module: str | None = None,
    correlation_id: str | None = None,
    message_pattern: str | None = None,
    limit: int | None = None,
) -> list[LogRecord]:
    """Convenience function to query logs from a file.

    Args:
        log_file: Path to log file.
        start_time: Filter by start time (UTC).
        end_time: Filter by end time (UTC).
        level: Filter by log level.
        module: Filter by module name.
        correlation_id: Filter by correlation ID.
        message_pattern: Filter by message regex pattern.
        limit: Maximum number of results.

    Returns:
        List of matching LogRecord objects.

    Example:
        >>> from datetime import datetime, UTC, timedelta
        >>> from oscura.core.log_query import query_logs
        >>> # Query last hour of errors
        >>> results = query_logs(
        ...     "oscura.log",
        ...     start_time=datetime.now(UTC) - timedelta(hours=1),
        ...     level="ERROR"
        ... )

    References:
        LOG-010: Searchable Log Query and Export
    """
    query = LogQuery()  # type: ignore[no-untyped-call]
    query.load_from_file(log_file, format="json" if log_file.endswith(".json") else "text")
    return query.query_logs(
        start_time=start_time,
        end_time=end_time,
        level=level,
        module=module,
        correlation_id=correlation_id,
        message_pattern=message_pattern,
        limit=limit,
    )


__all__ = [
    "LogQuery",
    "LogRecord",
    "query_logs",
]
