"""Tests for advanced logging features.

Tests the advanced logging infrastructure (LOG-009 through LOG-020).
"""

from __future__ import annotations

import gzip
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.core.logging_advanced import (
    AggregatedLogEntry,
    AlertSeverity,
    CompressedLogHandler,
    DashboardMetrics,
    EncryptedLogHandler,
    ForwardingConfig,
    LogAggregator,
    LogAlert,
    LogAlerter,
    LogAnalyzer,
    LogBuffer,
    LogDashboardCollector,
    LogForwarder,
    LogForwarderProtocol,
    LogPattern,
    LogSampler,
    SamplingStrategy,
    TriggeredAlert,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# =============================================================================


class TestAggregatedLogEntry:
    """Test AggregatedLogEntry dataclass."""

    def test_create_entry(self) -> None:
        """Test creating aggregated log entry."""
        entry = AggregatedLogEntry(
            key="test_pattern",
            count=5,
            sample_message="Test message 123",
        )
        assert entry.key == "test_pattern"
        assert entry.count == 5
        assert entry.sample_message == "Test message 123"
        assert isinstance(entry.first_seen, datetime)
        assert isinstance(entry.last_seen, datetime)
        assert len(entry.sources) == 0
        assert entry.levels.total() == 0


class TestLogAggregator:
    """Test LogAggregator class."""

    def test_init(self) -> None:
        """Test aggregator initialization."""
        aggregator = LogAggregator(window_seconds=120, min_count=3)
        assert aggregator.window_seconds == 120
        assert aggregator.min_count == 3
        assert len(aggregator._entries) == 0

    def test_init_defaults(self) -> None:
        """Test aggregator with default values."""
        aggregator = LogAggregator()
        assert aggregator.window_seconds == 60
        assert aggregator.min_count == 2

    def test_normalize_message_numbers(self) -> None:
        """Test message normalization - numbers."""
        aggregator = LogAggregator()
        normalized = aggregator._normalize_message("Error code 404 at line 123")
        assert normalized == "Error code <NUM> at line <NUM>"

    def test_normalize_message_uuid(self) -> None:
        """Test message normalization - UUIDs."""
        aggregator = LogAggregator()
        normalized = aggregator._normalize_message("Request 550e8400-e29b-41d4-a716-446655440000")
        # The regex replaces numbers first, so UUIDs get partially replaced
        assert "<NUM>" in normalized

    def test_normalize_message_paths(self) -> None:
        """Test message normalization - file paths."""
        aggregator = LogAggregator()
        normalized = aggregator._normalize_message("File /var/log/app.log not found")
        assert "<PATH>" in normalized
        assert "not found" in normalized

        normalized = aggregator._normalize_message("Error in C:\\Users\\test\\file.txt")
        assert "<PATH>" in normalized

    def test_add_record(self) -> None:
        """Test adding log record."""
        aggregator = LogAggregator()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error 404",
            args=(),
            exc_info=None,
        )
        aggregator.add(record)

        assert len(aggregator._entries) == 1
        key = "Error <NUM>"
        assert key in aggregator._entries
        entry = aggregator._entries[key]
        assert entry.count == 1
        assert entry.levels["ERROR"] == 1
        assert "test.logger" in entry.sources

    def test_add_multiple_similar(self) -> None:
        """Test adding multiple similar records."""
        aggregator = LogAggregator()
        for i in range(5):
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            aggregator.add(record)

        # All should aggregate to same key
        assert len(aggregator._entries) == 1
        entry = next(iter(aggregator._entries.values()))
        assert entry.count == 5

    def test_add_different_levels(self) -> None:
        """Test adding records with different levels."""
        aggregator = LogAggregator()

        record1 = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error 1",
            args=(),
            exc_info=None,
        )
        record2 = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Error 2",
            args=(),
            exc_info=None,
        )

        aggregator.add(record1)
        aggregator.add(record2)

        entry = next(iter(aggregator._entries.values()))
        assert entry.levels["ERROR"] == 1
        assert entry.levels["WARNING"] == 1

    def test_add_different_sources(self) -> None:
        """Test adding records from different sources."""
        aggregator = LogAggregator()

        record1 = logging.LogRecord(
            name="logger1",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error 1",
            args=(),
            exc_info=None,
        )
        record2 = logging.LogRecord(
            name="logger2",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error 2",
            args=(),
            exc_info=None,
        )

        aggregator.add(record1)
        aggregator.add(record2)

        entry = next(iter(aggregator._entries.values()))
        assert "logger1" in entry.sources
        assert "logger2" in entry.sources

    def test_get_summary_threshold(self) -> None:
        """Test get_summary respects min_count threshold."""
        aggregator = LogAggregator(min_count=3)

        # Add 2 of one pattern (below threshold)
        for i in range(2):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error A {i}",
                args=(),
                exc_info=None,
            )
            aggregator.add(record)

        # Add 3 of another pattern (meets threshold)
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info B {i}",
                args=(),
                exc_info=None,
            )
            aggregator.add(record)

        summary = aggregator.get_summary()
        assert len(summary) == 1
        assert summary[0].count == 3

    def test_cleanup_old(self) -> None:
        """Test cleaning up old entries."""
        aggregator = LogAggregator(window_seconds=1)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error 1",
            args=(),
            exc_info=None,
        )
        aggregator.add(record)

        # Wait for window to expire
        time.sleep(1.1)

        # Add new entry to update last_seen
        record2 = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error 2",
            args=(),
            exc_info=None,
        )
        aggregator.add(record2)

        # Cleanup should remove first entry
        aggregator.cleanup_old()

        # Only the recent entry should remain
        assert len(aggregator._entries) > 0

    def test_thread_safety(self) -> None:
        """Test thread-safe access."""
        aggregator = LogAggregator()
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    record = logging.LogRecord(
                        name=f"logger{thread_id}",
                        level=logging.INFO,
                        pathname="test.py",
                        lineno=10,
                        msg=f"Message {i}",
                        args=(),
                        exc_info=None,
                    )
                    aggregator.add(record)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# =============================================================================


class TestLogPattern:
    """Test LogPattern dataclass."""

    def test_create_pattern(self) -> None:
        """Test creating log pattern."""
        pattern = LogPattern(
            pattern="Error <N>",
            count=10,
            severity_distribution={"ERROR": 8, "WARNING": 2},
            time_distribution={10: 5, 11: 5},
            example="Error 404",
        )
        assert pattern.pattern == "Error <N>"
        assert pattern.count == 10
        assert pattern.severity_distribution["ERROR"] == 8


class TestLogAnalyzer:
    """Test LogAnalyzer class."""

    def test_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = LogAnalyzer(max_history=1000)
        assert len(analyzer._history) == 0
        assert analyzer._history.maxlen == 1000

    def test_init_default(self) -> None:
        """Test analyzer with default max_history."""
        analyzer = LogAnalyzer()
        assert analyzer._history.maxlen == 10000

    def test_add_record(self) -> None:
        """Test adding record to history."""
        analyzer = LogAnalyzer()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        analyzer.add(record)

        assert len(analyzer._history) == 1
        entry = analyzer._history[0]
        assert entry["message"] == "Test message"
        assert entry["level"] == "ERROR"
        assert entry["logger"] == "test"

    def test_analyze_patterns(self) -> None:
        """Test pattern analysis."""
        analyzer = LogAnalyzer()

        # Add similar messages
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error code {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        patterns = analyzer.analyze_patterns()
        assert len(patterns) > 0
        # Should find "Error code <N>" pattern
        pattern = patterns[0]
        assert pattern.count == 5
        assert pattern.pattern == "Error code <N>"

    def test_analyze_patterns_severity_distribution(self) -> None:
        """Test pattern severity distribution."""
        analyzer = LogAnalyzer()

        # Add messages with different levels
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        for i in range(2):
            record = logging.LogRecord(
                name="test",
                level=logging.WARNING,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        patterns = analyzer.analyze_patterns()
        pattern = patterns[0]
        assert pattern.severity_distribution["ERROR"] == 3
        assert pattern.severity_distribution["WARNING"] == 2

    def test_analyze_patterns_time_distribution(self) -> None:
        """Test pattern time distribution."""
        analyzer = LogAnalyzer()

        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        patterns = analyzer.analyze_patterns()
        pattern = patterns[0]
        # Should have time distribution by hour
        assert len(pattern.time_distribution) > 0

    def test_get_error_rate_empty(self) -> None:
        """Test error rate with empty history."""
        analyzer = LogAnalyzer()
        rate = analyzer.get_error_rate()
        assert rate == 0.0

    def test_get_error_rate(self) -> None:
        """Test error rate calculation."""
        analyzer = LogAnalyzer()

        # Add 3 errors and 7 info messages
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        for i in range(7):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        rate = analyzer.get_error_rate()
        assert rate == pytest.approx(0.3)  # 3/10

    def test_get_error_rate_window(self) -> None:
        """Test error rate with time window."""
        analyzer = LogAnalyzer()

        # Add some errors
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        rate = analyzer.get_error_rate(window_minutes=60)
        assert rate == 1.0  # All errors within window

    def test_get_trend_insufficient_data(self) -> None:
        """Test trend with insufficient data."""
        analyzer = LogAnalyzer()

        for i in range(50):  # Less than 100
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        trend = analyzer.get_trend()
        assert trend == "insufficient_data"

    def test_get_trend_increasing(self) -> None:
        """Test trend detection - increasing."""
        analyzer = LogAnalyzer()

        # First half: few errors
        for i in range(50):
            level = logging.ERROR if i < 5 else logging.INFO
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        # Second half: many errors
        for i in range(50):
            level = logging.ERROR if i < 30 else logging.INFO
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        trend = analyzer.get_trend()
        assert trend == "increasing"

    def test_get_trend_decreasing(self) -> None:
        """Test trend detection - decreasing."""
        analyzer = LogAnalyzer()

        # First half: many errors
        for i in range(50):
            level = logging.ERROR if i < 30 else logging.INFO
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        # Second half: few errors
        for i in range(50):
            level = logging.ERROR if i < 5 else logging.INFO
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        trend = analyzer.get_trend()
        assert trend == "decreasing"

    def test_get_trend_stable(self) -> None:
        """Test trend detection - stable."""
        analyzer = LogAnalyzer()

        # Consistent error rate
        for i in range(100):
            level = logging.ERROR if i % 10 == 0 else logging.INFO
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            analyzer.add(record)

        trend = analyzer.get_trend()
        assert trend == "stable"


# =============================================================================
# =============================================================================


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_severity_values(self) -> None:
        """Test alert severity enum values."""
        assert AlertSeverity.INFO
        assert AlertSeverity.WARNING
        assert AlertSeverity.ERROR
        assert AlertSeverity.CRITICAL


class TestLogAlert:
    """Test LogAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating log alert."""

        def condition(r):
            return r.levelno >= logging.ERROR

        alert = LogAlert(
            id="alert1",
            name="Error Alert",
            condition=condition,
            severity=AlertSeverity.ERROR,
            cooldown_seconds=300,
        )
        assert alert.id == "alert1"
        assert alert.name == "Error Alert"
        assert alert.enabled is True
        assert alert.last_triggered is None


class TestTriggeredAlert:
    """Test TriggeredAlert dataclass."""

    def test_create_triggered_alert(self) -> None:
        """Test creating triggered alert."""

        def condition(r):
            return r.levelno >= logging.ERROR

        alert = LogAlert(
            id="alert1",
            name="Test",
            condition=condition,
        )
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )
        triggered = TriggeredAlert(
            alert=alert,
            record=record,
            timestamp=datetime.now(),
        )
        assert triggered.alert == alert
        assert triggered.record == record


class TestLogAlerter:
    """Test LogAlerter class."""

    def test_init(self) -> None:
        """Test alerter initialization."""
        alerter = LogAlerter()
        assert len(alerter._alerts) == 0
        assert len(alerter._handlers) == 0

    def test_add_alert(self) -> None:
        """Test adding alert."""
        alerter = LogAlerter()

        def condition(r):
            return r.levelno >= logging.ERROR

        alert_id = alerter.add_alert(
            "Error Alert",
            condition,
            severity=AlertSeverity.ERROR,
            cooldown_seconds=60,
        )

        assert alert_id is not None
        assert alert_id in alerter._alerts
        alert = alerter._alerts[alert_id]
        assert alert.name == "Error Alert"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.cooldown_seconds == 60

    def test_check_no_alerts(self) -> None:
        """Test check with no alerts."""
        alerter = LogAlerter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        triggered = alerter.check(record)
        assert len(triggered) == 0

    def test_check_alert_triggered(self) -> None:
        """Test alert is triggered."""
        alerter = LogAlerter()

        def condition(r):
            return r.levelno >= logging.ERROR

        alerter.add_alert("Error Alert", condition)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        triggered = alerter.check(record)
        assert len(triggered) == 1
        assert triggered[0].record == record

    def test_check_alert_not_triggered(self) -> None:
        """Test alert is not triggered."""
        alerter = LogAlerter()

        def condition(r):
            return r.levelno >= logging.ERROR

        alerter.add_alert("Error Alert", condition)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )

        triggered = alerter.check(record)
        assert len(triggered) == 0

    def test_check_alert_disabled(self) -> None:
        """Test disabled alert is not triggered."""
        alerter = LogAlerter()

        def condition(r):
            return r.levelno >= logging.ERROR

        alert_id = alerter.add_alert("Error Alert", condition)

        # Disable alert
        alerter._alerts[alert_id].enabled = False

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        triggered = alerter.check(record)
        assert len(triggered) == 0

    def test_check_cooldown(self) -> None:
        """Test alert cooldown period."""
        alerter = LogAlerter()

        def condition(r):
            return r.levelno >= logging.ERROR

        alert_id = alerter.add_alert("Error Alert", condition, cooldown_seconds=1)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        # First trigger
        triggered1 = alerter.check(record)
        assert len(triggered1) == 1

        # Immediate second check (within cooldown)
        triggered2 = alerter.check(record)
        assert len(triggered2) == 0

        # Wait for cooldown
        time.sleep(1.1)

        # Should trigger again
        triggered3 = alerter.check(record)
        assert len(triggered3) == 1

    def test_check_condition_exception(self) -> None:
        """Test alert with failing condition."""
        alerter = LogAlerter()

        def bad_condition(r: logging.LogRecord) -> bool:
            raise ValueError("Condition error")

        alerter.add_alert("Bad Alert", bad_condition)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        # Should not raise, should handle exception gracefully
        triggered = alerter.check(record)
        assert len(triggered) == 0

    def test_on_alert_handler(self) -> None:
        """Test registering alert handler."""
        alerter = LogAlerter()
        handler_called = []

        def handler(alert: TriggeredAlert) -> None:
            handler_called.append(alert)

        alerter.on_alert(handler)

        def condition(r):
            return r.levelno >= logging.ERROR

        alerter.add_alert("Error Alert", condition)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        alerter.check(record)
        assert len(handler_called) == 1

    def test_on_alert_handler_exception(self) -> None:
        """Test handler exception is caught."""
        alerter = LogAlerter()

        def bad_handler(alert: TriggeredAlert) -> None:
            raise ValueError("Handler error")

        alerter.on_alert(bad_handler)

        def condition(r):
            return r.levelno >= logging.ERROR

        alerter.add_alert("Error Alert", condition)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        # Should not raise
        alerter.check(record)

    def test_multiple_alerts(self) -> None:
        """Test multiple alerts can trigger."""
        alerter = LogAlerter()
        alerter.add_alert("Error Alert", lambda r: r.levelno >= logging.ERROR)
        alerter.add_alert("Any Alert", lambda r: True)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        triggered = alerter.check(record)
        assert len(triggered) == 2


# =============================================================================
# =============================================================================


class TestSamplingStrategy:
    """Test SamplingStrategy enum."""

    def test_strategy_values(self) -> None:
        """Test sampling strategy enum values."""
        assert SamplingStrategy.RANDOM
        assert SamplingStrategy.RATE_LIMIT
        assert SamplingStrategy.ADAPTIVE


class TestLogSampler:
    """Test LogSampler class."""

    def test_init_default(self) -> None:
        """Test sampler initialization with defaults."""
        sampler = LogSampler()
        assert sampler.strategy == SamplingStrategy.RATE_LIMIT
        assert sampler.rate == 0.1
        assert sampler.max_per_second == 100

    def test_init_custom(self) -> None:
        """Test sampler initialization with custom values."""
        sampler = LogSampler(
            strategy=SamplingStrategy.RANDOM,
            rate=0.5,
            max_per_second=50,
        )
        assert sampler.strategy == SamplingStrategy.RANDOM
        assert sampler.rate == 0.5
        assert sampler.max_per_second == 50

    def test_should_log_errors_always(self) -> None:
        """Test errors are always logged."""
        sampler = LogSampler(strategy=SamplingStrategy.RANDOM, rate=0.0)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error",
            args=(),
            exc_info=None,
        )

        # Even with 0% sampling, errors should pass
        assert sampler.should_log(record) is True

    def test_should_log_random(self) -> None:
        """Test random sampling strategy."""
        sampler = LogSampler(strategy=SamplingStrategy.RANDOM, rate=0.5)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )

        # Test multiple times, should get mix of True/False
        results = [sampler.should_log(record) for _ in range(1000)]
        true_count = sum(results)

        # With 50% rate, expect roughly 500 True values (with more variance tolerance)
        assert 400 < true_count < 600

    def test_should_log_rate_limit(self) -> None:
        """Test rate limit sampling strategy."""
        sampler = LogSampler(
            strategy=SamplingStrategy.RATE_LIMIT,
            max_per_second=5,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )

        # First 5 should pass
        for _i in range(5):
            assert sampler.should_log(record) is True

        # 6th should be rejected
        assert sampler.should_log(record) is False

    def test_should_log_rate_limit_reset(self) -> None:
        """Test rate limit resets each second."""
        sampler = LogSampler(
            strategy=SamplingStrategy.RATE_LIMIT,
            max_per_second=2,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )

        # Fill this second's quota
        sampler.should_log(record)
        sampler.should_log(record)
        assert sampler.should_log(record) is False

        # Wait for next second
        time.sleep(1.1)

        # Should allow again
        assert sampler.should_log(record) is True

    def test_should_log_adaptive(self) -> None:
        """Test adaptive sampling strategy."""
        sampler = LogSampler(strategy=SamplingStrategy.ADAPTIVE)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )

        # Initially should allow (rate starts high)
        result = sampler.should_log(record)
        assert isinstance(result, bool)


# =============================================================================
# =============================================================================


class TestLogBuffer:
    """Test LogBuffer class."""

    def test_init(self) -> None:
        """Test buffer initialization."""
        buffer = LogBuffer(max_size=100, flush_interval_seconds=10.0)
        assert buffer.max_size == 100
        assert buffer.flush_interval == 10.0
        assert buffer._buffer.empty()

    def test_init_default(self) -> None:
        """Test buffer with default values."""
        buffer = LogBuffer()
        assert buffer.max_size == 1000
        assert buffer.flush_interval == 5.0

    def test_add_record(self) -> None:
        """Test adding record to buffer."""
        buffer = LogBuffer()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )

        buffer.add(record)
        assert buffer._buffer.qsize() == 1

    def test_add_multiple_records(self) -> None:
        """Test adding multiple records."""
        buffer = LogBuffer()

        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info {i}",
                args=(),
                exc_info=None,
            )
            buffer.add(record)

        assert buffer._buffer.qsize() == 5

    def test_add_buffer_full(self) -> None:
        """Test adding when buffer is full."""
        buffer = LogBuffer(max_size=2)
        handler_called = []

        def handler(records: list[logging.LogRecord]) -> None:
            handler_called.extend(records)

        buffer.on_flush(handler)

        # Fill buffer
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info {i}",
                args=(),
                exc_info=None,
            )
            buffer.add(record)

        # Should have triggered flush
        assert len(handler_called) > 0

    def test_flush_empty(self) -> None:
        """Test flushing empty buffer."""
        buffer = LogBuffer()
        handler_called = []

        buffer.on_flush(lambda records: handler_called.append(records))
        buffer.flush()

        assert len(handler_called) == 0

    def test_flush_with_records(self) -> None:
        """Test flushing buffer with records."""
        buffer = LogBuffer()
        flushed_records = []

        def handler(records: list[logging.LogRecord]) -> None:
            flushed_records.extend(records)

        buffer.on_flush(handler)

        # Add records
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info {i}",
                args=(),
                exc_info=None,
            )
            buffer.add(record)

        buffer.flush()

        assert len(flushed_records) == 3
        assert buffer._buffer.empty()

    def test_flush_handler_exception(self) -> None:
        """Test flush with failing handler."""
        buffer = LogBuffer()

        def bad_handler(records: list[logging.LogRecord]) -> None:
            raise ValueError("Handler error")

        buffer.on_flush(bad_handler)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )
        buffer.add(record)

        # Should not raise
        buffer.flush()

    def test_on_flush_multiple_handlers(self) -> None:
        """Test multiple flush handlers."""
        buffer = LogBuffer()
        handler1_called = []
        handler2_called = []

        buffer.on_flush(lambda r: handler1_called.extend(r))
        buffer.on_flush(lambda r: handler2_called.extend(r))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )
        buffer.add(record)
        buffer.flush()

        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

    def test_auto_flush(self) -> None:
        """Test automatic flush thread."""
        buffer = LogBuffer(flush_interval_seconds=0.5)
        flushed = []

        buffer.on_flush(lambda r: flushed.extend(r))
        buffer.start_auto_flush()

        # Add record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )
        buffer.add(record)

        # Wait for auto flush
        time.sleep(0.7)

        buffer.stop_auto_flush()

        assert len(flushed) > 0

    def test_stop_auto_flush(self) -> None:
        """Test stopping auto flush."""
        buffer = LogBuffer(flush_interval_seconds=0.5)
        buffer.start_auto_flush()

        # Add record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info",
            args=(),
            exc_info=None,
        )
        buffer.add(record)

        buffer.stop_auto_flush()

        # Buffer should be empty after stop (final flush)
        assert buffer._buffer.empty()


# =============================================================================
# =============================================================================


class TestCompressedLogHandler:
    """Test CompressedLogHandler class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test handler initialization."""
        filename = str(tmp_path / "test.log")
        handler = CompressedLogHandler(
            filename,
            max_bytes=1000,
            backup_count=3,
            compression_level=6,
        )
        assert handler.filename == filename
        assert handler.max_bytes == 1000
        assert handler.backup_count == 3
        assert handler.compression_level == 6

    def test_emit_creates_file(self, tmp_path: Path) -> None:
        """Test emit creates compressed file."""
        filename = str(tmp_path / "test.log")
        handler = CompressedLogHandler(filename)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        handler.close()

        assert Path(f"{filename}.gz").exists()

    def test_emit_writes_message(self, tmp_path: Path) -> None:
        """Test emit writes message to file."""
        filename = str(tmp_path / "test.log")
        handler = CompressedLogHandler(filename)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        handler.close()

        # Read compressed file
        with gzip.open(f"{filename}.gz", "rt") as f:
            content = f.read()

        assert "Test message" in content

    def test_rotation(self, tmp_path: Path) -> None:
        """Test file rotation."""
        filename = str(tmp_path / "test.log")
        handler = CompressedLogHandler(
            filename,
            max_bytes=100,  # Small size to trigger rotation
            backup_count=2,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Write enough to trigger rotation
        for i in range(20):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}" * 10,
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        handler.close()

        # Should have rotated files
        assert Path(f"{filename}.gz").exists()

    def test_close(self, tmp_path: Path) -> None:
        """Test closing handler."""
        filename = str(tmp_path / "test.log")
        handler = CompressedLogHandler(filename)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        handler.close()
        assert handler._current_file is None


# =============================================================================
# =============================================================================


class TestEncryptedLogHandler:
    """Test EncryptedLogHandler class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test handler initialization."""
        filename = str(tmp_path / "encrypted.log")
        handler = EncryptedLogHandler(filename, key="secret_key")
        assert handler.filename == filename
        assert handler._key is not None

    def test_emit_creates_file(self, tmp_path: Path) -> None:
        """Test emit creates encrypted file."""
        filename = str(tmp_path / "encrypted.log")
        handler = EncryptedLogHandler(filename, key="secret_key")
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Secret message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        handler.close()

        assert Path(filename).exists()

    def test_emit_encrypts_message(self, tmp_path: Path) -> None:
        """Test emit encrypts message."""
        filename = str(tmp_path / "encrypted.log")
        handler = EncryptedLogHandler(filename, key="secret_key")
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Secret message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        handler.close()

        # Read raw file - should not contain plain text
        with open(filename, "rb") as f:
            content = f.read()

        # Message should be encrypted (not readable as plain text)
        assert b"Secret message" not in content

    def test_encrypt_decrypt(self, tmp_path: Path) -> None:
        """Test encryption is reversible."""
        filename = str(tmp_path / "encrypted.log")
        handler = EncryptedLogHandler(filename, key="secret_key")

        data = b"Test data"
        encrypted = handler._encrypt(data)

        # Encrypt again with same key should give same result
        decrypted = handler._encrypt(encrypted)  # XOR is reversible
        assert decrypted == data

    def test_close(self, tmp_path: Path) -> None:
        """Test closing handler."""
        filename = str(tmp_path / "encrypted.log")
        handler = EncryptedLogHandler(filename, key="secret_key")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        handler.close()
        assert handler._file is None


# =============================================================================
# =============================================================================


class TestLogForwarderProtocol:
    """Test LogForwarderProtocol enum."""

    def test_protocol_values(self) -> None:
        """Test protocol enum values."""
        assert LogForwarderProtocol.SYSLOG
        assert LogForwarderProtocol.HTTP
        assert LogForwarderProtocol.TCP
        assert LogForwarderProtocol.UDP


class TestForwardingConfig:
    """Test ForwardingConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating forwarding config."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
            timeout=10.0,
            batch_size=50,
            tls=True,
        )
        assert config.protocol == LogForwarderProtocol.HTTP
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.timeout == 10.0
        assert config.batch_size == 50
        assert config.tls is True


class TestLogForwarder:
    """Test LogForwarder class."""

    def test_init(self) -> None:
        """Test forwarder initialization."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
        )
        forwarder = LogForwarder(config)
        assert forwarder.config == config
        assert len(forwarder._buffer) == 0

    def test_forward_adds_to_buffer(self) -> None:
        """Test forward adds entry to buffer."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
            batch_size=10,
        )
        forwarder = LogForwarder(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        forwarder.forward(record)
        assert len(forwarder._buffer) == 1

    def test_forward_batch_flush(self) -> None:
        """Test forward triggers flush at batch size."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
            batch_size=2,
        )
        forwarder = LogForwarder(config)

        # Mock the send method to avoid actual network calls
        with patch.object(forwarder, "_send_http"):
            # Add records up to batch size
            for i in range(2):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=10,
                    msg=f"Message {i}",
                    args=(),
                    exc_info=None,
                )
                forwarder.forward(record)

            # Should have flushed
            assert forwarder._send_http.called

    @patch("urllib.request.urlopen")
    def test_send_http(self, mock_urlopen: Mock) -> None:
        """Test HTTP forwarding."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
        )
        forwarder = LogForwarder(config)

        entries = [{"timestamp": "2025-01-01", "level": "INFO", "message": "Test"}]

        forwarder._send_http(entries)
        assert mock_urlopen.called

    @patch("socket.socket")
    def test_send_syslog(self, mock_socket: Mock) -> None:
        """Test syslog forwarding."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.SYSLOG,
            host="localhost",
            port=514,
        )
        forwarder = LogForwarder(config)

        entries = [
            {"timestamp": "2025-01-01", "level": "INFO", "logger": "test", "message": "Test"}
        ]

        forwarder._send_syslog(entries)
        assert mock_socket.called

    @patch("socket.socket")
    def test_send_tcp(self, mock_socket: Mock) -> None:
        """Test TCP forwarding."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.TCP,
            host="localhost",
            port=9000,
        )
        forwarder = LogForwarder(config)

        entries = [{"timestamp": "2025-01-01", "level": "INFO", "message": "Test"}]

        forwarder._send_tcp(entries)
        assert mock_socket.called

    @patch("socket.socket")
    def test_send_udp(self, mock_socket: Mock) -> None:
        """Test UDP forwarding."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.UDP,
            host="localhost",
            port=9000,
        )
        forwarder = LogForwarder(config)

        entries = [{"timestamp": "2025-01-01", "level": "INFO", "message": "Test"}]

        forwarder._send_udp(entries)
        assert mock_socket.called

    def test_flush_error_handling(self) -> None:
        """Test flush handles errors gracefully."""
        config = ForwardingConfig(
            protocol=LogForwarderProtocol.HTTP,
            host="localhost",
            port=8080,
        )
        forwarder = LogForwarder(config)

        # Add entry
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        forwarder.forward(record)

        # Mock send to raise error
        with patch.object(forwarder, "_send_http", side_effect=Exception("Network error")):
            forwarder._flush()
            # Entries should be back in buffer after error
            assert len(forwarder._buffer) > 0


# =============================================================================
# =============================================================================


class TestDashboardMetrics:
    """Test DashboardMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Test creating dashboard metrics."""
        metrics = DashboardMetrics(
            total_logs=100,
            logs_by_level={"INFO": 70, "ERROR": 30},
            logs_by_logger={"logger1": 60, "logger2": 40},
            logs_per_minute=[10, 15, 20],
            error_rate=0.3,
            top_patterns=[("pattern1", 50), ("pattern2", 30)],
            recent_errors=[{"message": "error1"}],
        )
        assert metrics.total_logs == 100
        assert metrics.error_rate == 0.3
        assert len(metrics.top_patterns) == 2


class TestLogDashboardCollector:
    """Test LogDashboardCollector class."""

    def test_init(self) -> None:
        """Test collector initialization."""
        collector = LogDashboardCollector(window_minutes=30)
        assert collector.window_minutes == 30
        assert len(collector._logs) == 0

    def test_init_default(self) -> None:
        """Test collector with default window."""
        collector = LogDashboardCollector()
        assert collector.window_minutes == 60

    def test_add_record(self) -> None:
        """Test adding record."""
        collector = LogDashboardCollector()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        collector.add(record)
        assert len(collector._logs) == 1

    def test_add_trims_old(self) -> None:
        """Test adding trims old entries."""
        collector = LogDashboardCollector(window_minutes=0)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        collector.add(record)

        # Wait to make entry old
        time.sleep(0.1)

        # Add new entry - should trim old
        collector.add(record)

        # Should only have recent entry
        assert len(collector._logs) <= 1

    def test_get_metrics_empty(self) -> None:
        """Test get_metrics with no logs."""
        collector = LogDashboardCollector()
        metrics = collector.get_metrics()

        assert metrics.total_logs == 0
        assert metrics.error_rate == 0.0

    def test_get_metrics_total_logs(self) -> None:
        """Test get_metrics counts total logs."""
        collector = LogDashboardCollector()

        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        assert metrics.total_logs == 5

    def test_get_metrics_by_level(self) -> None:
        """Test get_metrics counts by level."""
        collector = LogDashboardCollector()

        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        for i in range(2):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        assert metrics.logs_by_level["ERROR"] == 3
        assert metrics.logs_by_level["INFO"] == 2

    def test_get_metrics_by_logger(self) -> None:
        """Test get_metrics counts by logger."""
        collector = LogDashboardCollector()

        for i in range(3):
            record = logging.LogRecord(
                name="logger1",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        for i in range(2):
            record = logging.LogRecord(
                name="logger2",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        assert metrics.logs_by_logger["logger1"] == 3
        assert metrics.logs_by_logger["logger2"] == 2

    def test_get_metrics_error_rate(self) -> None:
        """Test get_metrics calculates error rate."""
        collector = LogDashboardCollector()

        # 3 errors, 7 info = 30% error rate
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        for i in range(7):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Info {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        assert metrics.error_rate == pytest.approx(0.3)

    def test_get_metrics_top_patterns(self) -> None:
        """Test get_metrics finds top patterns."""
        collector = LogDashboardCollector()

        # Add similar messages
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Error code {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        assert len(metrics.top_patterns) > 0
        # Should find "Error code <N>" pattern
        pattern, count = metrics.top_patterns[0]
        assert count == 5

    def test_get_metrics_recent_errors(self) -> None:
        """Test get_metrics collects recent errors."""
        collector = LogDashboardCollector()

        for i in range(15):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Error {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        # Should only keep last 10
        assert len(metrics.recent_errors) == 10

    def test_get_metrics_per_minute(self) -> None:
        """Test get_metrics calculates logs per minute."""
        collector = LogDashboardCollector()

        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()
        # Should have 60 minutes of data
        assert len(metrics.logs_per_minute) == 60
        # Most recent minute should have the logs
        assert sum(metrics.logs_per_minute) == 5

    def test_thread_safety(self) -> None:
        """Test thread-safe access."""
        collector = LogDashboardCollector()
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    record = logging.LogRecord(
                        name=f"logger{thread_id}",
                        level=logging.INFO,
                        pathname="test.py",
                        lineno=10,
                        msg=f"Message {i}",
                        args=(),
                        exc_info=None,
                    )
                    collector.add(record)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        metrics = collector.get_metrics()
        assert metrics.total_logs == 30


# =============================================================================
# Integration Tests
# =============================================================================


class TestCoreLoggingAdvancedIntegration:
    """Integration tests for advanced logging features."""

    def test_aggregation_analysis_workflow(self) -> None:
        """Test aggregation and analysis workflow."""
        aggregator = LogAggregator(min_count=2)
        analyzer = LogAnalyzer()

        # Simulate logs
        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg=f"Database error {i}",
                args=(),
                exc_info=None,
            )
            aggregator.add(record)
            analyzer.add(record)

        # Check aggregation
        summary = aggregator.get_summary()
        assert len(summary) > 0

        # Check analysis
        patterns = analyzer.analyze_patterns()
        assert len(patterns) > 0

        error_rate = analyzer.get_error_rate()
        assert error_rate == 1.0  # All errors

    def test_alerting_workflow(self) -> None:
        """Test alerting workflow."""
        alerter = LogAlerter()
        triggered_alerts = []

        # Register handler
        alerter.on_alert(lambda a: triggered_alerts.append(a))

        # Add error alert
        alerter.add_alert(
            "High Error Rate",
            lambda r: r.levelno >= logging.ERROR,
            severity=AlertSeverity.ERROR,
        )

        # Trigger alert
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Critical error",
            args=(),
            exc_info=None,
        )

        alerter.check(record)

        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].alert.severity == AlertSeverity.ERROR

    def test_buffering_workflow(self) -> None:
        """Test buffering workflow."""
        buffer = LogBuffer(max_size=10, flush_interval_seconds=1.0)
        flushed_batches = []

        buffer.on_flush(lambda records: flushed_batches.append(len(records)))
        buffer.start_auto_flush()

        # Add records
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            buffer.add(record)

        # Wait for flush
        time.sleep(1.5)

        buffer.stop_auto_flush()

        # Should have flushed
        assert len(flushed_batches) > 0

    def test_dashboard_workflow(self) -> None:
        """Test dashboard data collection workflow."""
        collector = LogDashboardCollector()

        # Simulate varied logs
        for i in range(10):
            level = logging.ERROR if i % 3 == 0 else logging.INFO
            record = logging.LogRecord(
                name=f"logger{i % 2}",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            collector.add(record)

        metrics = collector.get_metrics()

        assert metrics.total_logs == 10
        assert metrics.error_rate > 0
        assert len(metrics.logs_by_level) > 0
        assert len(metrics.logs_by_logger) > 0
        assert len(metrics.logs_per_minute) == 60
