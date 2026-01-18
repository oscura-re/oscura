"""Tests for log query module.

Tests requirements:
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from oscura.core.log_query import LogQuery, LogRecord, query_logs

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestLogRecord:
    """Tests for LogRecord data class."""

    def test_create_log_record(self):
        """Test creating a log record."""
        record = LogRecord(
            timestamp="2025-12-21T10:00:00.000000Z",
            level="INFO",
            module="oscura.test",
            message="Test message",
        )
        assert record.timestamp == "2025-12-21T10:00:00.000000Z"
        assert record.level == "INFO"
        assert record.module == "oscura.test"
        assert record.message == "Test message"
        assert record.correlation_id is None

    def test_log_record_to_dict(self):
        """Test converting log record to dictionary."""
        record = LogRecord(
            timestamp="2025-12-21T10:00:00.000000Z",
            level="ERROR",
            module="oscura.test",
            message="Error occurred",
            correlation_id="123-456",
        )
        data = record.to_dict()
        assert data["timestamp"] == "2025-12-21T10:00:00.000000Z"
        assert data["level"] == "ERROR"
        assert data["correlation_id"] == "123-456"

    def test_log_record_from_dict(self):
        """Test creating log record from dictionary."""
        data = {
            "timestamp": "2025-12-21T10:00:00.000000Z",
            "level": "WARNING",
            "module": "oscura.test",
            "message": "Warning message",
            "correlation_id": "abc-def",
        }
        record = LogRecord.from_dict(data)
        assert record.level == "WARNING"
        assert record.correlation_id == "abc-def"


class TestLogQuery:
    """Tests for LogQuery class."""

    def test_init_log_query(self):
        """Test initializing log query."""
        query = LogQuery()
        assert query is not None

    def test_add_record(self):
        """Test adding a log record."""
        query = LogQuery()
        record = LogRecord(
            timestamp="2025-12-21T10:00:00.000000Z",
            level="INFO",
            module="test",
            message="Test",
        )
        query.add_record(record)
        results = query.query_logs()
        assert len(results) == 1
        assert results[0].message == "Test"

    def test_query_by_level(self):
        """Test querying by log level."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="test",
                message="Info message",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:01:00.000000Z",
                level="ERROR",
                module="test",
                message="Error message",
            )
        )

        # Query for ERROR logs
        errors = query.query_logs(level="ERROR")
        assert len(errors) == 1
        assert errors[0].level == "ERROR"

        # Query for INFO logs
        infos = query.query_logs(level="INFO")
        assert len(infos) == 1
        assert infos[0].level == "INFO"

    def test_query_by_module(self):
        """Test querying by module name."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="oscura.loaders",
                message="Loader message",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:01:00.000000Z",
                level="INFO",
                module="oscura.analyzers",
                message="Analyzer message",
            )
        )

        # Query specific module
        results = query.query_logs(module="oscura.loaders")
        assert len(results) == 1
        assert results[0].module == "oscura.loaders"

    def test_query_by_module_pattern(self):
        """Test querying by module pattern."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="oscura.loaders.binary",
                message="Binary loader",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:01:00.000000Z",
                level="INFO",
                module="oscura.loaders.csv",
                message="CSV loader",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:02:00.000000Z",
                level="INFO",
                module="oscura.analyzers.spectral",
                message="Spectral analyzer",
            )
        )

        # Query with pattern
        results = query.query_logs(module_pattern="oscura.loaders.*")
        assert len(results) == 2
        assert all("loaders" in r.module for r in results)

    def test_query_by_correlation_id(self):
        """Test querying by correlation ID."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="test",
                message="Message 1",
                correlation_id="corr-123",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:01:00.000000Z",
                level="INFO",
                module="test",
                message="Message 2",
                correlation_id="corr-456",
            )
        )

        results = query.query_logs(correlation_id="corr-123")
        assert len(results) == 1
        assert results[0].correlation_id == "corr-123"

    def test_query_by_message_pattern(self):
        """Test querying by message regex pattern."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="test",
                message="Processing file data.bin",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:01:00.000000Z",
                level="INFO",
                module="test",
                message="Completed analysis",
            )
        )

        results = query.query_logs(message_pattern=r"Processing.*\.bin")
        assert len(results) == 1
        assert "data.bin" in results[0].message

    def test_query_by_time_range(self):
        """Test querying by time range."""
        query = LogQuery()
        base_time = datetime(2025, 12, 21, 10, 0, 0, tzinfo=UTC)

        query.add_record(
            LogRecord(
                timestamp=base_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                level="INFO",
                module="test",
                message="First",
            )
        )
        query.add_record(
            LogRecord(
                timestamp=(base_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                level="INFO",
                module="test",
                message="Second",
            )
        )
        query.add_record(
            LogRecord(
                timestamp=(base_time + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                level="INFO",
                module="test",
                message="Third",
            )
        )

        # Query within time range
        results = query.query_logs(
            start_time=base_time + timedelta(minutes=30),
            end_time=base_time + timedelta(hours=1, minutes=30),
        )
        assert len(results) == 1
        assert results[0].message == "Second"

    def test_query_pagination(self):
        """Test query pagination with limit and offset."""
        query = LogQuery()
        for i in range(10):
            query.add_record(
                LogRecord(
                    timestamp=f"2025-12-21T10:{i:02d}:00.000000Z",
                    level="INFO",
                    module="test",
                    message=f"Message {i}",
                )
            )

        # First page
        page1 = query.query_logs(limit=3, offset=0)
        assert len(page1) == 3
        assert page1[0].message == "Message 0"

        # Second page
        page2 = query.query_logs(limit=3, offset=3)
        assert len(page2) == 3
        assert page2[0].message == "Message 3"

        # Last page (partial)
        page4 = query.query_logs(limit=3, offset=9)
        assert len(page4) == 1
        assert page4[0].message == "Message 9"

    def test_clear_records(self):
        """Test clearing log records."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="test",
                message="Test",
            )
        )
        assert len(query.query_logs()) == 1

        query.clear()
        assert len(query.query_logs()) == 0

    def test_get_statistics(self):
        """Test getting log statistics."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="oscura.test",
                message="Info 1",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:01:00.000000Z",
                level="ERROR",
                module="oscura.test",
                message="Error 1",
            )
        )
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:02:00.000000Z",
                level="INFO",
                module="oscura.other",
                message="Info 2",
            )
        )

        stats = query.get_statistics()
        assert stats["total"] == 3
        assert stats["by_level"]["INFO"] == 2
        assert stats["by_level"]["ERROR"] == 1
        assert "oscura.test" in stats["by_module"]
        assert stats["time_range"]["earliest"] == "2025-12-21T10:00:00.000000Z"


class TestLogExport:
    """Tests for log export functionality."""

    def test_export_json(self):
        """Test exporting logs as JSON."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="test",
                message="Test message",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"
            query.export_logs(query.query_logs(), str(output_path), format="json")

            # Verify file was created and contains data
            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["message"] == "Test message"

    def test_export_csv(self):
        """Test exporting logs as CSV."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="ERROR",
                module="test",
                message="Error occurred",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.csv"
            query.export_logs(query.query_logs(), str(output_path), format="csv")

            # Verify file was created
            assert output_path.exists()
            content = output_path.read_text()
            assert "timestamp,level,module,message" in content
            assert "ERROR" in content

    def test_export_text(self):
        """Test exporting logs as plain text."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="WARNING",
                module="test",
                message="Warning message",
                correlation_id="corr-123",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            query.export_logs(query.query_logs(), str(output_path), format="text")

            # Verify file was created
            assert output_path.exists()
            content = output_path.read_text()
            assert "WARNING" in content
            assert "Warning message" in content
            assert "corr-123" in content


class TestLogLoad:
    """Tests for loading logs from files."""

    def test_load_json_lines(self):
        """Test loading JSON lines format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs.json"

            # Create test log file
            with open(log_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": "2025-12-21T10:00:00.000000Z",
                            "level": "INFO",
                            "module": "test",
                            "message": "Test message 1",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "timestamp": "2025-12-21T10:01:00.000000Z",
                            "level": "ERROR",
                            "module": "test",
                            "message": "Test message 2",
                        }
                    )
                    + "\n"
                )

            # Load logs
            query = LogQuery()
            count = query.load_from_file(str(log_file), format="json")
            assert count == 2

            results = query.query_logs()
            assert len(results) == 2
            assert results[0].level == "INFO"
            assert results[1].level == "ERROR"

    def test_load_text_format(self):
        """Test loading plain text format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs.txt"

            # Create test log file
            with open(log_file, "w") as f:
                f.write("2025-12-21T10:00:00.000000Z [INFO] test: First message\n")
                f.write("2025-12-21T10:01:00.000000Z [ERROR] test: Second message\n")

            # Load logs
            query = LogQuery()
            count = query.load_from_file(str(log_file), format="text")
            assert count == 2

            results = query.query_logs()
            assert len(results) == 2

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        query = LogQuery()
        with pytest.raises(FileNotFoundError):
            query.load_from_file("/nonexistent/path.log")

    def test_load_unsupported_format(self):
        """Test loading with unsupported format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs.txt"
            log_file.write_text("test")

            query = LogQuery()
            with pytest.raises(ValueError, match="Unsupported format"):
                query.load_from_file(str(log_file), format="xml")  # type: ignore


class TestQueryLogsConvenience:
    """Tests for query_logs convenience function."""

    def test_query_logs_function(self):
        """Test convenience function for querying logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs.json"

            # Create test log file
            with open(log_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": "2025-12-21T10:00:00.000000Z",
                            "level": "ERROR",
                            "module": "test",
                            "message": "Error message",
                        }
                    )
                    + "\n"
                )

            # Query using convenience function
            results = query_logs(str(log_file), level="ERROR")
            assert len(results) == 1
            assert results[0].level == "ERROR"


class TestComplexQueries:
    """Tests for complex query combinations."""

    def test_combined_filters(self):
        """Test combining multiple query filters."""
        query = LogQuery()
        base_time = datetime(2025, 12, 21, 10, 0, 0, tzinfo=UTC)

        # Add various log records
        for i in range(20):
            query.add_record(
                LogRecord(
                    timestamp=(base_time + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    level="ERROR" if i % 3 == 0 else "INFO",
                    module=f"oscura.module{i % 5}",
                    message=f"Message {i} with data",
                    correlation_id="corr-abc" if i < 10 else None,
                )
            )

        # Complex query: errors within time range with correlation ID
        results = query.query_logs(
            start_time=base_time,
            end_time=base_time + timedelta(minutes=10),
            level="ERROR",
            correlation_id="corr-abc",
        )

        assert len(results) > 0
        assert all(r.level == "ERROR" for r in results)
        assert all(r.correlation_id == "corr-abc" for r in results)

    def test_empty_result_set(self):
        """Test query with no matching results."""
        query = LogQuery()
        query.add_record(
            LogRecord(
                timestamp="2025-12-21T10:00:00.000000Z",
                level="INFO",
                module="test",
                message="Test",
            )
        )

        # Query for non-existent data
        results = query.query_logs(level="CRITICAL")
        assert len(results) == 0
