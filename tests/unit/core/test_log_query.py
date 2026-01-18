"""Comprehensive unit tests for src/oscura/core/log_query.py

Tests coverage for:

Covers all public classes, functions, properties, and methods with
edge cases, error handling, and validation.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from oscura.core.log_query import LogQuery, LogRecord, query_logs

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ==============================================================================
# LogRecord Tests
# ==============================================================================


class TestLogRecord:
    """Test LogRecord dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating LogRecord with minimal fields."""
        record = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="INFO",
            module="test.module",
            message="Test message",
        )
        assert record.timestamp == "2025-12-21T10:00:00Z"
        assert record.level == "INFO"
        assert record.module == "test.module"
        assert record.message == "Test message"
        assert record.correlation_id is None
        assert record.metadata is None

    def test_create_with_correlation_id(self) -> None:
        """Test creating LogRecord with correlation ID."""
        record = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="ERROR",
            module="test",
            message="msg",
            correlation_id="abc-123",
        )
        assert record.correlation_id == "abc-123"

    def test_create_with_metadata(self) -> None:
        """Test creating LogRecord with metadata."""
        metadata = {"user": "alice", "request_id": "xyz"}
        record = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="INFO",
            module="test",
            message="msg",
            metadata=metadata,
        )
        assert record.metadata == metadata

    def test_to_dict_minimal(self) -> None:
        """Test to_dict() with minimal record."""
        record = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="INFO",
            module="test",
            message="msg",
        )
        result = record.to_dict()
        assert result["timestamp"] == "2025-12-21T10:00:00Z"
        assert result["level"] == "INFO"
        assert result["module"] == "test"
        assert result["message"] == "msg"
        assert result["correlation_id"] is None
        assert result["metadata"] == {}  # None becomes empty dict

    def test_to_dict_full(self) -> None:
        """Test to_dict() with all fields."""
        record = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="ERROR",
            module="test",
            message="msg",
            correlation_id="abc",
            metadata={"key": "value"},
        )
        result = record.to_dict()
        assert result["correlation_id"] == "abc"
        assert result["metadata"] == {"key": "value"}

    def test_from_dict_minimal(self) -> None:
        """Test from_dict() with minimal data."""
        data = {
            "timestamp": "2025-12-21T10:00:00Z",
            "level": "INFO",
            "module": "test",
            "message": "msg",
        }
        record = LogRecord.from_dict(data)
        assert record.timestamp == "2025-12-21T10:00:00Z"
        assert record.level == "INFO"
        assert record.module == "test"
        assert record.message == "msg"

    def test_from_dict_full(self) -> None:
        """Test from_dict() with all fields."""
        data = {
            "timestamp": "2025-12-21T10:00:00Z",
            "level": "ERROR",
            "module": "test",
            "message": "msg",
            "correlation_id": "abc",
            "metadata": {"key": "value"},
        }
        record = LogRecord.from_dict(data)
        assert record.correlation_id == "abc"
        assert record.metadata == {"key": "value"}

    def test_round_trip(self) -> None:
        """Test to_dict() -> from_dict() round trip."""
        original = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="ERROR",
            module="test.module",
            message="Test error",
            correlation_id="xyz",
            metadata={"source": "unit_test"},
        )
        data = original.to_dict()
        restored = LogRecord.from_dict(data)
        assert restored.timestamp == original.timestamp
        assert restored.level == original.level
        assert restored.module == original.module
        assert restored.message == original.message
        assert restored.correlation_id == original.correlation_id


# ==============================================================================
# LogQuery Initialization Tests
# ==============================================================================


class TestLogQueryInit:
    """Test LogQuery initialization."""

    def test_create_empty(self) -> None:
        """Test creating empty LogQuery."""
        query = LogQuery()
        assert query._records == []


# ==============================================================================
# LogQuery add_record() Tests
# ==============================================================================


class TestLogQueryAddRecord:
    """Test LogQuery.add_record() method."""

    def test_add_single_record(self) -> None:
        """Test adding a single record."""
        query = LogQuery()
        record = LogRecord(
            timestamp="2025-12-21T10:00:00Z",
            level="INFO",
            module="test",
            message="msg",
        )
        query.add_record(record)
        assert len(query._records) == 1
        assert query._records[0] == record

    def test_add_multiple_records(self) -> None:
        """Test adding multiple records."""
        query = LogQuery()
        for i in range(5):
            record = LogRecord(
                timestamp=f"2025-12-21T10:00:0{i}Z",
                level="INFO",
                module="test",
                message=f"msg{i}",
            )
            query.add_record(record)
        assert len(query._records) == 5


# ==============================================================================
# LogQuery query_logs() Tests
# ==============================================================================


class TestLogQueryQueryLogs:
    """Test LogQuery.query_logs() method."""

    @pytest.fixture
    def populated_query(self) -> LogQuery:
        """Create a LogQuery with test data."""
        query = LogQuery()
        # Add various test records
        times = [
            "2025-12-21T10:00:00Z",
            "2025-12-21T11:00:00Z",
            "2025-12-21T12:00:00Z",
        ]
        levels = ["INFO", "WARNING", "ERROR"]
        modules = ["module.a", "module.b", "module.c"]

        for t, lvl, mod in zip(times, levels, modules, strict=False):
            query.add_record(
                LogRecord(timestamp=t, level=lvl, module=mod, message=f"{lvl} from {mod}")
            )
        return query

    def test_query_all(self, populated_query: LogQuery) -> None:
        """Test querying without filters returns all."""
        results = populated_query.query_logs()
        assert len(results) == 3

    def test_query_by_level(self, populated_query: LogQuery) -> None:
        """Test filtering by log level."""
        results = populated_query.query_logs(level="ERROR")
        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_query_by_level_lowercase(self, populated_query: LogQuery) -> None:
        """Test filtering by level (lowercase input)."""
        results = populated_query.query_logs(level="error")
        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_query_by_module(self, populated_query: LogQuery) -> None:
        """Test filtering by exact module name."""
        results = populated_query.query_logs(module="module.a")
        assert len(results) == 1
        assert results[0].module == "module.a"

    def test_query_by_module_pattern(self, populated_query: LogQuery) -> None:
        """Test filtering by module pattern."""
        results = populated_query.query_logs(module_pattern="module.*")
        assert len(results) == 3

    def test_query_by_module_pattern_specific(self, populated_query: LogQuery) -> None:
        """Test filtering by specific module pattern."""
        results = populated_query.query_logs(module_pattern="module.[ab]")
        assert len(results) == 2

    def test_query_by_start_time(self, populated_query: LogQuery) -> None:
        """Test filtering by start time."""
        start = datetime(2025, 12, 21, 10, 30, tzinfo=UTC)
        results = populated_query.query_logs(start_time=start)
        assert len(results) == 2  # 11:00 and 12:00

    def test_query_by_end_time(self, populated_query: LogQuery) -> None:
        """Test filtering by end time."""
        end = datetime(2025, 12, 21, 10, 30, tzinfo=UTC)
        results = populated_query.query_logs(end_time=end)
        assert len(results) == 1  # Only 10:00

    def test_query_by_time_range(self, populated_query: LogQuery) -> None:
        """Test filtering by time range."""
        start = datetime(2025, 12, 21, 10, 30, tzinfo=UTC)
        end = datetime(2025, 12, 21, 11, 30, tzinfo=UTC)
        results = populated_query.query_logs(start_time=start, end_time=end)
        assert len(results) == 1  # Only 11:00

    def test_query_by_correlation_id(self) -> None:
        """Test filtering by correlation ID."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00Z",
                level="INFO",
                module="test",
                message="msg1",
                correlation_id="abc",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:01Z",
                level="INFO",
                module="test",
                message="msg2",
                correlation_id="xyz",
            )
        )
        results = query.query_logs(correlation_id="abc")
        assert len(results) == 1
        assert results[0].message == "msg1"

    def test_query_by_message_pattern(self, populated_query: LogQuery) -> None:
        """Test filtering by message regex pattern."""
        results = populated_query.query_logs(message_pattern="ERROR.*")
        assert len(results) == 1
        assert "ERROR" in results[0].message

    def test_query_with_limit(self, populated_query: LogQuery) -> None:
        """Test limiting results."""
        results = populated_query.query_logs(limit=2)
        assert len(results) == 2

    def test_query_with_offset(self, populated_query: LogQuery) -> None:
        """Test offsetting results."""
        results = populated_query.query_logs(offset=1)
        assert len(results) == 2

    def test_query_with_limit_and_offset(self, populated_query: LogQuery) -> None:
        """Test pagination with limit and offset."""
        results = populated_query.query_logs(limit=1, offset=1)
        assert len(results) == 1

    def test_query_multiple_filters(self, populated_query: LogQuery) -> None:
        """Test combining multiple filters."""
        results = populated_query.query_logs(
            level="INFO",
            module_pattern="module.*",
            message_pattern=".*module.*",
        )
        assert len(results) == 1


# ==============================================================================
# LogQuery load_from_file() Tests
# ==============================================================================


class TestLogQueryLoadFromFile:
    """Test LogQuery.load_from_file() method."""

    def test_load_json_lines(self, tmp_path: Path) -> None:
        """Test loading JSON lines format."""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:00Z",
                        "level": "INFO",
                        "module": "test",
                        "message": "Test message",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:01Z",
                        "level": "ERROR",
                        "module": "test2",
                        "message": "Error message",
                    }
                )
                + "\n"
            )

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="json")
        assert count == 2
        assert len(query._records) == 2

    def test_load_json_with_metadata(self, tmp_path: Path) -> None:
        """Test loading JSON with extra metadata fields."""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:00Z",
                        "level": "INFO",
                        "module": "test",
                        "message": "Test",
                        "custom_field": "value",
                        "another": 123,
                    }
                )
                + "\n"
            )

        query = LogQuery()
        query.load_from_file(str(log_file), format="json")
        assert query._records[0].metadata == {"custom_field": "value", "another": 123}

    def test_load_json_skip_malformed(self, tmp_path: Path) -> None:
        """Test that malformed JSON lines are skipped."""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            f.write("not valid json\n")
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:00Z",
                        "level": "INFO",
                        "module": "test",
                        "message": "Valid",
                    }
                )
                + "\n"
            )

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="json")
        assert count == 1  # Only valid line loaded

    def test_load_text_format(self, tmp_path: Path) -> None:
        """Test loading plain text format."""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            f.write("2025-12-21T10:00:00Z [INFO] test.module: Test message\n")
            f.write("2025-12-21T10:00:01Z [ERROR] test.other: Error message\n")

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="text")
        assert count == 2
        assert query._records[0].level == "INFO"
        assert query._records[1].level == "ERROR"

    def test_load_text_skip_unparseable(self, tmp_path: Path) -> None:
        """Test that unparseable text lines are skipped."""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            f.write("Random text that doesn't match\n")
            f.write("2025-12-21T10:00:00Z [INFO] test: Valid\n")

        query = LogQuery()
        count = query.load_from_file(str(log_file), format="text")
        assert count == 1

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        query = LogQuery()
        with pytest.raises(FileNotFoundError, match="Log file not found"):
            query.load_from_file("/nonexistent/file.log")

    def test_load_unsupported_format_raises_error(self, tmp_path: Path) -> None:
        """Test that unsupported format raises ValueError."""
        log_file = tmp_path / "test.log"
        log_file.touch()

        query = LogQuery()
        with pytest.raises(ValueError, match="Unsupported format"):
            query.load_from_file(str(log_file), format="xml")  # type: ignore[arg-type]


# ==============================================================================
# LogQuery export_logs() Tests
# ==============================================================================


class TestLogQueryExportLogs:
    """Test LogQuery.export_logs() method."""

    @pytest.fixture
    def sample_records(self) -> list[LogRecord]:
        """Create sample records for export tests."""
        return [
            LogRecord(
                timestamp="2025-12-21T10:00:00Z",
                level="INFO",
                module="test",
                message="Message 1",
            ),
            LogRecord(
                timestamp="2025-12-21T10:00:01Z",
                level="ERROR",
                module="test2",
                message="Message 2",
                correlation_id="abc-123",
            ),
        ]

    def test_export_json(self, tmp_path: Path, sample_records: list[LogRecord]) -> None:
        """Test exporting to JSON format."""
        output_file = tmp_path / "output.json"

        query = LogQuery()
        query.export_logs(sample_records, str(output_file), format="json")

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["level"] == "INFO"
        assert data[1]["correlation_id"] == "abc-123"

    def test_export_csv(self, tmp_path: Path, sample_records: list[LogRecord]) -> None:
        """Test exporting to CSV format."""
        output_file = tmp_path / "output.csv"

        query = LogQuery()
        query.export_logs(sample_records, str(output_file), format="csv")

        assert output_file.exists()
        content = output_file.read_text()
        assert "timestamp,level,module,message,correlation_id" in content
        assert "2025-12-21T10:00:00Z" in content
        assert "ERROR" in content

    def test_export_csv_empty_records(self, tmp_path: Path) -> None:
        """Test exporting empty records to CSV."""
        output_file = tmp_path / "output.csv"

        query = LogQuery()
        query.export_logs([], str(output_file), format="csv")

        # Should create file but be empty (no records to write)
        assert not output_file.exists() or output_file.stat().st_size == 0

    def test_export_text(self, tmp_path: Path, sample_records: list[LogRecord]) -> None:
        """Test exporting to text format."""
        output_file = tmp_path / "output.txt"

        query = LogQuery()
        query.export_logs(sample_records, str(output_file), format="text")

        assert output_file.exists()
        content = output_file.read_text()
        assert "[INFO]" in content
        assert "[ERROR]" in content
        assert "[corr_id=abc-123]" in content

    def test_export_creates_parent_dirs(
        self, tmp_path: Path, sample_records: list[LogRecord]
    ) -> None:
        """Test that export creates parent directories."""
        output_file = tmp_path / "subdir" / "output.json"

        query = LogQuery()
        query.export_logs(sample_records, str(output_file), format="json")

        assert output_file.exists()

    def test_export_unsupported_format_raises_error(
        self, tmp_path: Path, sample_records: list[LogRecord]
    ) -> None:
        """Test that unsupported export format raises ValueError."""
        output_file = tmp_path / "output.xml"

        query = LogQuery()
        with pytest.raises(ValueError, match="Unsupported export format"):
            query.export_logs(sample_records, str(output_file), format="xml")  # type: ignore[arg-type]


# ==============================================================================
# LogQuery get_statistics() Tests
# ==============================================================================


class TestLogQueryStatistics:
    """Test LogQuery.get_statistics() method."""

    def test_statistics_empty(self) -> None:
        """Test statistics for empty query."""
        query = LogQuery()
        stats = query.get_statistics()
        assert stats["total"] == 0
        assert stats["by_level"] == {}
        assert stats["by_module"] == {}
        assert stats["time_range"] is None

    def test_statistics_with_records(self) -> None:
        """Test statistics with records."""
        query = LogQuery()
        query.add_record(
            LogRecord(timestamp="2025-12-21T10:00:00Z", level="INFO", module="mod1", message="msg1")
        )
        query.add_record(
            LogRecord(timestamp="2025-12-21T10:00:01Z", level="INFO", module="mod1", message="msg2")
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:02Z", level="ERROR", module="mod2", message="msg3"
            )
        )

        stats = query.get_statistics()
        assert stats["total"] == 3
        assert stats["by_level"]["INFO"] == 2
        assert stats["by_level"]["ERROR"] == 1
        assert stats["by_module"]["mod1"] == 2
        assert stats["by_module"]["mod2"] == 1
        assert stats["time_range"]["earliest"] == "2025-12-21T10:00:00Z"
        assert stats["time_range"]["latest"] == "2025-12-21T10:00:02Z"


# ==============================================================================
# LogQuery clear() Tests
# ==============================================================================


class TestLogQueryClear:
    """Test LogQuery.clear() method."""

    def test_clear_empty(self) -> None:
        """Test clearing empty query."""
        query = LogQuery()
        query.clear()
        assert len(query._records) == 0

    def test_clear_with_records(self) -> None:
        """Test clearing query with records."""
        query = LogQuery()
        query.add_record(
            LogRecord(timestamp="2025-12-21T10:00:00Z", level="INFO", module="test", message="msg")
        )
        assert len(query._records) == 1

        query.clear()
        assert len(query._records) == 0


# ==============================================================================
# query_logs() Function Tests
# ==============================================================================


class TestQueryLogsFunction:
    """Test query_logs() convenience function."""

    def test_query_json_file(self, tmp_path: Path) -> None:
        """Test querying JSON log file."""
        log_file = tmp_path / "test.json"
        with open(log_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:00Z",
                        "level": "ERROR",
                        "module": "test",
                        "message": "Error message",
                    }
                )
                + "\n"
            )

        results = query_logs(str(log_file), level="ERROR")
        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_query_text_file(self, tmp_path: Path) -> None:
        """Test querying text log file."""
        log_file = tmp_path / "test.log"
        with open(log_file, "w") as f:
            f.write("2025-12-21T10:00:00Z [INFO] test: Message\n")

        results = query_logs(str(log_file), level="INFO")
        assert len(results) == 1

    def test_query_with_filters(self, tmp_path: Path) -> None:
        """Test querying with multiple filters."""
        log_file = tmp_path / "test.json"
        with open(log_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:00Z",
                        "level": "ERROR",
                        "module": "test.module",
                        "message": "Error happened",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "timestamp": "2025-12-21T10:00:01Z",
                        "level": "INFO",
                        "module": "test.other",
                        "message": "Info message",
                    }
                )
                + "\n"
            )

        results = query_logs(str(log_file), level="ERROR", module="test.module")
        assert len(results) == 1
        assert results[0].level == "ERROR"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCoreLogQueryIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: load, query, export."""
        # Create log file
        log_file = tmp_path / "input.json"
        with open(log_file, "w") as f:
            for i in range(10):
                f.write(
                    json.dumps(
                        {
                            "timestamp": f"2025-12-21T10:00:0{i}Z",
                            "level": "ERROR" if i % 2 == 0 else "INFO",
                            "module": f"module.{i % 3}",
                            "message": f"Message {i}",
                        }
                    )
                    + "\n"
                )

        # Load and query
        query = LogQuery()
        query.load_from_file(str(log_file), format="json")

        # Filter errors
        errors = query.query_logs(level="ERROR")
        assert len(errors) == 5

        # Export filtered results
        output_file = tmp_path / "errors.json"
        query.export_logs(errors, str(output_file), format="json")

        # Verify export
        assert output_file.exists()
        with open(output_file) as f:
            exported = json.load(f)
        assert len(exported) == 5

    def test_pagination_workflow(self, tmp_path: Path) -> None:
        """Test pagination workflow."""
        log_file = tmp_path / "logs.json"
        with open(log_file, "w") as f:
            for i in range(100):
                f.write(
                    json.dumps(
                        {
                            "timestamp": f"2025-12-21T10:00:{i:02d}Z",
                            "level": "INFO",
                            "module": "test",
                            "message": f"Message {i}",
                        }
                    )
                    + "\n"
                )

        query = LogQuery()
        query.load_from_file(str(log_file), format="json")

        # Get pages
        page1 = query.query_logs(limit=25, offset=0)
        page2 = query.query_logs(limit=25, offset=25)
        page3 = query.query_logs(limit=25, offset=50)

        assert len(page1) == 25
        assert len(page2) == 25
        assert len(page3) == 25
