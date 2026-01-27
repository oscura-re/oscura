"""Comprehensive tests for log_query module.

Tests log querying, filtering, pagination, export formats,
and statistics generation.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from oscura.core.log_query import LogQuery, LogRecord, query_logs


@pytest.fixture
def sample_log_records() -> list[LogRecord]:
    """Create sample log records for testing."""
    base_time = datetime(2025, 1, 25, 12, 0, 0, tzinfo=UTC)

    return [
        LogRecord(
            timestamp=base_time.isoformat(),
            level="INFO",
            module="oscura.loaders",
            message="Loading file test.dat",
            correlation_id="req-123",
        ),
        LogRecord(
            timestamp=(base_time + timedelta(seconds=1)).isoformat(),
            level="DEBUG",
            module="oscura.loaders.csv",
            message="Parsing CSV header",
            correlation_id="req-123",
        ),
        LogRecord(
            timestamp=(base_time + timedelta(seconds=2)).isoformat(),
            level="ERROR",
            module="oscura.analyzers.fft",
            message="FFT failed: insufficient data",
            correlation_id="req-124",
        ),
        LogRecord(
            timestamp=(base_time + timedelta(seconds=3)).isoformat(),
            level="WARNING",
            module="oscura.analyzers",
            message="Memory usage high",
            correlation_id="req-124",
        ),
        LogRecord(
            timestamp=(base_time + timedelta(seconds=4)).isoformat(),
            level="INFO",
            module="oscura.export",
            message="Exported results to output.json",
            correlation_id=None,
        ),
    ]


@pytest.mark.unit
@pytest.mark.core
class TestLogRecord:
    """Tests for LogRecord dataclass."""

    def test_creation_basic(self) -> None:
        """Should create log record with required fields."""
        record = LogRecord(
            timestamp="2025-01-25T12:00:00Z",
            level="INFO",
            module="oscura.test",
            message="Test message",
        )

        assert record.timestamp == "2025-01-25T12:00:00Z"
        assert record.level == "INFO"
        assert record.module == "oscura.test"
        assert record.message == "Test message"
        assert record.correlation_id is None
        assert record.metadata is None

    def test_creation_with_optional_fields(self) -> None:
        """Should create log record with optional fields."""
        metadata = {"user": "admin", "duration": 1.5}
        record = LogRecord(
            timestamp="2025-01-25T12:00:00Z",
            level="ERROR",
            module="oscura.test",
            message="Error occurred",
            correlation_id="req-456",
            metadata=metadata,
        )

        assert record.correlation_id == "req-456"
        assert record.metadata == metadata

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        record = LogRecord(
            timestamp="2025-01-25T12:00:00Z",
            level="INFO",
            module="oscura.test",
            message="Test",
        )

        result = record.to_dict()

        assert result["timestamp"] == "2025-01-25T12:00:00Z"
        assert result["level"] == "INFO"
        assert result["module"] == "oscura.test"
        assert result["message"] == "Test"
        assert result["metadata"] == {}

    def test_from_dict(self) -> None:
        """Should create from dictionary."""
        data = {
            "timestamp": "2025-01-25T12:00:00Z",
            "level": "WARNING",
            "module": "oscura.test",
            "message": "Warning message",
            "correlation_id": "req-789",
            "metadata": {"key": "value"},
        }

        record = LogRecord.from_dict(data)

        assert record.timestamp == data["timestamp"]
        assert record.level == data["level"]
        assert record.correlation_id == "req-789"
        assert record.metadata == {"key": "value"}


@pytest.mark.unit
@pytest.mark.core
class TestLogQueryBasic:
    """Tests for basic LogQuery operations."""

    def test_initialization(self) -> None:
        """Should initialize empty query."""
        query = LogQuery()

        assert len(query._records) == 0

    def test_add_record(self, sample_log_records: list[LogRecord]) -> None:
        """Should add log record."""
        query = LogQuery()

        query.add_record(sample_log_records[0])

        assert len(query._records) == 1
        assert query._records[0] == sample_log_records[0]

    def test_clear(self, sample_log_records: list[LogRecord]) -> None:
        """Should clear all records."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        query.clear()

        assert len(query._records) == 0


@pytest.mark.unit
@pytest.mark.core
class TestLogQueryFiltering:
    """Tests for log filtering operations."""

    def test_query_all(self, sample_log_records: list[LogRecord]) -> None:
        """Should return all records with no filters."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs()

        assert len(results) == len(sample_log_records)

    def test_filter_by_level(self, sample_log_records: list[LogRecord]) -> None:
        """Should filter by log level."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(level="ERROR")

        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_filter_by_level_case_insensitive(self, sample_log_records: list[LogRecord]) -> None:
        """Should handle lowercase level filter."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(level="error")

        assert len(results) == 1

    def test_filter_by_module_exact(self, sample_log_records: list[LogRecord]) -> None:
        """Should filter by exact module name."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(module="oscura.loaders")

        assert len(results) == 1
        assert results[0].module == "oscura.loaders"

    def test_filter_by_module_pattern(self, sample_log_records: list[LogRecord]) -> None:
        """Should filter by module pattern."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(module_pattern="oscura.loaders.*")

        # Pattern matches oscura.loaders and oscura.loaders.csv
        assert len(results) >= 1
        assert all(r.module.startswith("oscura.loaders") for r in results)

    def test_filter_by_correlation_id(self, sample_log_records: list[LogRecord]) -> None:
        """Should filter by correlation ID."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(correlation_id="req-123")

        assert len(results) == 2
        assert all(r.correlation_id == "req-123" for r in results)

    def test_filter_by_message_pattern(self, sample_log_records: list[LogRecord]) -> None:
        """Should filter by message regex."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(message_pattern="failed")

        assert len(results) == 1
        assert "failed" in results[0].message.lower()

    def test_filter_by_time_range(self, sample_log_records: list[LogRecord]) -> None:
        """Should filter by time range."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        start = datetime(2025, 1, 25, 12, 0, 1, tzinfo=UTC)
        end = datetime(2025, 1, 25, 12, 0, 3, tzinfo=UTC)

        results = query.query_logs(start_time=start, end_time=end)

        assert len(results) == 2

    def test_combined_filters(self, sample_log_records: list[LogRecord]) -> None:
        """Should apply multiple filters together."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(
            level="INFO",
            module_pattern="oscura.*",
        )

        assert len(results) == 2
        assert all(r.level == "INFO" for r in results)


@pytest.mark.unit
@pytest.mark.core
class TestLogQueryPagination:
    """Tests for pagination."""

    def test_limit(self, sample_log_records: list[LogRecord]) -> None:
        """Should limit number of results."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(limit=2)

        assert len(results) == 2

    def test_offset(self, sample_log_records: list[LogRecord]) -> None:
        """Should skip first N results."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        results = query.query_logs(offset=2)

        assert len(results) == 3

    def test_offset_and_limit(self, sample_log_records: list[LogRecord]) -> None:
        """Should support pagination with offset and limit."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        # Get records 2-3 (indices 1-2)
        results = query.query_logs(offset=1, limit=2)

        assert len(results) == 2
        assert results[0] == sample_log_records[1]
        assert results[1] == sample_log_records[2]


@pytest.mark.unit
@pytest.mark.core
class TestLogQueryLoad:
    """Tests for loading logs from files."""

    def test_load_json_lines(self, tmp_path: Path) -> None:
        """Should load JSON lines format."""
        log_file = tmp_path / "test.json"
        logs = [
            {
                "timestamp": "2025-01-25T12:00:00Z",
                "level": "INFO",
                "module": "test",
                "message": "msg1",
            },
            {
                "timestamp": "2025-01-25T12:00:01Z",
                "level": "ERROR",
                "module": "test",
                "message": "msg2",
            },
        ]

        with open(log_file, "w") as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="json")

        assert count == 2
        assert len(query._records) == 2

    def test_load_text_format(self, tmp_path: Path) -> None:
        """Should load text format logs."""
        log_file = tmp_path / "test.log"
        content = """2025-01-25T12:00:00Z [INFO] oscura.test: Test message 1
2025-01-25T12:00:01Z [ERROR] oscura.test: Test message 2
"""
        log_file.write_text(content)

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="text")

        assert count == 2

    def test_load_file_not_found(self) -> None:
        """Should raise FileNotFoundError for missing file."""
        query = LogQuery()

        with pytest.raises(FileNotFoundError):
            query.load_from_file("nonexistent.log")

    def test_load_invalid_format(self, tmp_path: Path) -> None:
        """Should raise ValueError for unsupported format."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")

        query = LogQuery()

        with pytest.raises(ValueError):
            query.load_from_file(str(log_file), format="xml")  # type: ignore[arg-type]

    def test_load_malformed_json(self, tmp_path: Path) -> None:
        """Should skip malformed JSON lines."""
        log_file = tmp_path / "test.json"
        content = """{"valid": "json"}
{broken json}
{"another": "valid"}
"""
        log_file.write_text(content)

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="json")

        assert count == 2  # Only valid lines


@pytest.mark.unit
@pytest.mark.core
class TestLogQueryExport:
    """Tests for exporting logs."""

    def test_export_json(self, tmp_path: Path, sample_log_records: list[LogRecord]) -> None:
        """Should export to JSON format."""
        query = LogQuery()
        for record in sample_log_records[:2]:
            query.add_record(record)

        output_file = tmp_path / "export.json"
        query.export_logs(query._records, str(output_file), format="json")

        # Verify file created and valid JSON
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 2

    def test_export_csv(self, tmp_path: Path, sample_log_records: list[LogRecord]) -> None:
        """Should export to CSV format."""
        query = LogQuery()
        for record in sample_log_records[:2]:
            query.add_record(record)

        output_file = tmp_path / "export.csv"
        query.export_logs(query._records, str(output_file), format="csv")

        # Verify file created and valid CSV
        assert output_file.exists()
        with open(output_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2

    def test_export_text(self, tmp_path: Path, sample_log_records: list[LogRecord]) -> None:
        """Should export to text format."""
        query = LogQuery()
        for record in sample_log_records[:2]:
            query.add_record(record)

        output_file = tmp_path / "export.txt"
        query.export_logs(query._records, str(output_file), format="text")

        # Verify file created
        assert output_file.exists()
        content = output_file.read_text()
        assert "INFO" in content
        assert "oscura.loaders" in content

    def test_export_creates_directories(
        self, tmp_path: Path, sample_log_records: list[LogRecord]
    ) -> None:
        """Should create parent directories if needed."""
        query = LogQuery()
        query.add_record(sample_log_records[0])

        output_file = tmp_path / "subdir" / "nested" / "export.json"
        query.export_logs(query._records, str(output_file), format="json")

        assert output_file.exists()

    def test_export_invalid_format(
        self, tmp_path: Path, sample_log_records: list[LogRecord]
    ) -> None:
        """Should raise ValueError for invalid format."""
        query = LogQuery()
        query.add_record(sample_log_records[0])

        output_file = tmp_path / "export.xml"

        with pytest.raises(ValueError):
            query.export_logs(query._records, str(output_file), format="xml")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.core
class TestLogQueryStatistics:
    """Tests for log statistics."""

    def test_get_statistics(self, sample_log_records: list[LogRecord]) -> None:
        """Should generate statistics."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        stats = query.get_statistics()

        assert stats["total"] == 5
        assert "by_level" in stats
        assert stats["by_level"]["INFO"] == 2
        assert stats["by_level"]["ERROR"] == 1

    def test_statistics_by_module(self, sample_log_records: list[LogRecord]) -> None:
        """Should count by module."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        stats = query.get_statistics()

        assert "by_module" in stats
        assert len(stats["by_module"]) > 0

    def test_statistics_time_range(self, sample_log_records: list[LogRecord]) -> None:
        """Should include time range."""
        query = LogQuery()
        for record in sample_log_records:
            query.add_record(record)

        stats = query.get_statistics()

        assert "time_range" in stats
        assert "earliest" in stats["time_range"]
        assert "latest" in stats["time_range"]

    def test_statistics_empty(self) -> None:
        """Should handle empty logs."""
        query = LogQuery()

        stats = query.get_statistics()

        assert stats["total"] == 0
        assert stats["by_level"] == {}
        assert stats["time_range"] is None


@pytest.mark.unit
@pytest.mark.core
class TestQueryLogsConvenience:
    """Tests for convenience query_logs function."""

    def test_query_logs_json_file(self, tmp_path: Path) -> None:
        """Should query from JSON file."""
        log_file = tmp_path / "test.json"
        logs = [
            {
                "timestamp": "2025-01-25T12:00:00Z",
                "level": "INFO",
                "module": "test",
                "message": "msg1",
            },
            {
                "timestamp": "2025-01-25T12:00:01Z",
                "level": "ERROR",
                "module": "test",
                "message": "msg2",
            },
        ]

        with open(log_file, "w") as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")

        results = query_logs(str(log_file), level="ERROR")

        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_query_logs_text_file(self, tmp_path: Path) -> None:
        """Should query from text file."""
        log_file = tmp_path / "test.log"
        content = """2025-01-25T12:00:00Z [INFO] test: Test message
2025-01-25T12:00:01Z [ERROR] test: Error message
"""
        log_file.write_text(content)

        results = query_logs(str(log_file))

        assert len(results) == 2

    def test_query_logs_with_filters(self, tmp_path: Path) -> None:
        """Should apply filters."""
        log_file = tmp_path / "test.json"
        logs = [
            {
                "timestamp": "2025-01-25T12:00:00Z",
                "level": "INFO",
                "module": "test",
                "message": "msg",
            },
            {
                "timestamp": "2025-01-25T12:00:01Z",
                "level": "ERROR",
                "module": "test",
                "message": "msg",
            },
            {
                "timestamp": "2025-01-25T12:00:02Z",
                "level": "INFO",
                "module": "test",
                "message": "msg",
            },
        ]

        with open(log_file, "w") as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")

        results = query_logs(str(log_file), level="INFO", limit=1)

        assert len(results) == 1
