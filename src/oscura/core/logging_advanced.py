"""Advanced logging features for Oscura.

This module provides advanced logging capabilities including log aggregation,
analysis, alerting, sampling, and external system integration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================


@dataclass
class AggregatedLogEntry:
    """Aggregated log entry.

    Attributes:
        key: Aggregation key
        count: Number of occurrences
        first_seen: First occurrence timestamp
        last_seen: Last occurrence timestamp
        sample_message: Sample message
        levels: Counter of log levels
        sources: Set of source loggers

    References:
        LOG-009: Log Aggregation
    """

    key: str
    count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    sample_message: str = ""
    levels: Counter = field(default_factory=Counter)  # type: ignore[type-arg]
    sources: set[str] = field(default_factory=set)


class LogAggregator:
    """Aggregates log messages by pattern.

    Groups similar log messages together to reduce noise and
    identify patterns.

    Example:
        >>> aggregator = LogAggregator()
        >>> aggregator.add(record)
        >>> summary = aggregator.get_summary()

    References:
        LOG-009: Log Aggregation
    """

    def __init__(self, window_seconds: int = 60, min_count: int = 2):
        """Initialize aggregator.

        Args:
            window_seconds: Aggregation window size
            min_count: Minimum occurrences to report
        """
        self.window_seconds = window_seconds
        self.min_count = min_count
        self._entries: dict[str, AggregatedLogEntry] = {}
        self._lock = threading.Lock()

    def _normalize_message(self, message: str) -> str:
        """Normalize message for grouping.

        Replaces variable parts with placeholders.

        Args:
            message: Log message to normalize.

        Returns:
            Normalized message with placeholders.
        """
        # Replace numbers
        normalized = re.sub(r"\d+", "<NUM>", message)
        # Replace UUIDs
        normalized = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "<UUID>",
            normalized,
            flags=re.IGNORECASE,
        )
        # Replace file paths
        normalized = re.sub(r"[/\\][\w./\\-]+", "<PATH>", normalized)
        return normalized

    def add(self, record: logging.LogRecord) -> None:
        """Add log record to aggregator.

        Args:
            record: Log record
        """
        key = self._normalize_message(record.getMessage())
        now = datetime.now()

        with self._lock:
            if key not in self._entries:
                self._entries[key] = AggregatedLogEntry(
                    key=key, first_seen=now, sample_message=record.getMessage()
                )

            entry = self._entries[key]
            entry.count += 1
            entry.last_seen = now
            entry.levels[record.levelname] += 1
            entry.sources.add(record.name)

    def get_summary(self) -> list[AggregatedLogEntry]:
        """Get aggregation summary.

        Returns:
            List of aggregated entries meeting threshold
        """
        with self._lock:
            return [entry for entry in self._entries.values() if entry.count >= self.min_count]

    def cleanup_old(self) -> None:
        """Remove entries outside window."""
        cutoff = datetime.now() - timedelta(seconds=self.window_seconds)
        with self._lock:
            self._entries = {k: v for k, v in self._entries.items() if v.last_seen >= cutoff}


# =============================================================================
# =============================================================================


@dataclass
class LogPattern:
    """Detected log pattern.

    References:
        LOG-010: Log Analysis and Patterns
    """

    pattern: str
    count: int
    severity_distribution: dict[str, int]
    time_distribution: dict[int, int]  # Hour -> count
    example: str


class LogAnalyzer:
    """Analyzes log patterns and trends.

    References:
        LOG-010: Log Analysis and Patterns
    """

    def __init__(self, max_history: int = 10000):
        self._history: deque = deque(maxlen=max_history)  # type: ignore[type-arg]
        self._patterns: dict[str, LogPattern] = {}

    def add(self, record: logging.LogRecord) -> None:
        """Add record to analysis history."""
        self._history.append(
            {
                "message": record.getMessage(),
                "level": record.levelname,
                "time": datetime.now(),
                "logger": record.name,
            }
        )

    def analyze_patterns(self) -> list[LogPattern]:
        """Analyze log patterns.

        Returns:
            List of detected patterns
        """
        pattern_counts: Counter = Counter()  # type: ignore[type-arg]
        pattern_levels: dict[str, Counter] = {}  # type: ignore[type-arg]
        pattern_hours: dict[str, Counter] = {}  # type: ignore[type-arg]
        pattern_examples: dict[str, str] = {}

        for entry in self._history:
            # Normalize message
            normalized = re.sub(r"\d+", "<N>", entry["message"])
            pattern_counts[normalized] += 1

            if normalized not in pattern_levels:
                pattern_levels[normalized] = Counter()
                pattern_hours[normalized] = Counter()
                pattern_examples[normalized] = entry["message"]

            pattern_levels[normalized][entry["level"]] += 1
            pattern_hours[normalized][entry["time"].hour] += 1

        return [
            LogPattern(
                pattern=pattern,
                count=count,
                severity_distribution=dict(pattern_levels.get(pattern, {})),
                time_distribution=dict(pattern_hours.get(pattern, {})),
                example=pattern_examples.get(pattern, ""),
            )
            for pattern, count in pattern_counts.most_common(20)
        ]

    def get_error_rate(self, window_minutes: int = 60) -> float:
        """Get error rate over window.

        Args:
            window_minutes: Window size in minutes

        Returns:
            Error rate (0.0 to 1.0)
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [e for e in self._history if e["time"] >= cutoff]

        if not recent:
            return 0.0

        errors = sum(1 for e in recent if e["level"] in ("ERROR", "CRITICAL"))
        return errors / len(recent)

    def get_trend(self) -> str:
        """Get error trend (increasing, stable, decreasing)."""
        if len(self._history) < 100:
            return "insufficient_data"

        # Compare first half to second half
        mid = len(self._history) // 2
        first_half = list(self._history)[:mid]
        second_half = list(self._history)[mid:]

        first_errors = sum(1 for e in first_half if e["level"] in ("ERROR", "CRITICAL"))
        second_errors = sum(1 for e in second_half if e["level"] in ("ERROR", "CRITICAL"))

        first_rate = first_errors / len(first_half)
        second_rate = second_errors / len(second_half)

        if second_rate > first_rate * 1.2:
            return "increasing"
        elif second_rate < first_rate * 0.8:
            return "decreasing"
        return "stable"


# =============================================================================
# =============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class LogAlert:
    """Log alert definition.

    References:
        LOG-012: Log Alerting
    """

    id: str
    name: str
    condition: Callable[[logging.LogRecord], bool]
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: int = 300
    last_triggered: datetime | None = None
    enabled: bool = True


@dataclass
class TriggeredAlert:
    """Triggered alert instance."""

    alert: LogAlert
    record: logging.LogRecord
    timestamp: datetime


class LogAlerter:
    """Log alerting system.

    Monitors logs and triggers alerts based on conditions.

    Example:
        >>> alerter = LogAlerter()
        >>> alerter.add_alert("error_burst", lambda r: r.levelno >= logging.ERROR)
        >>> alerter.on_alert(lambda a: send_notification(a))

    References:
        LOG-012: Log Alerting
    """

    def __init__(self) -> None:
        self._alerts: dict[str, LogAlert] = {}
        self._handlers: list[Callable[[TriggeredAlert], None]] = []
        self._lock = threading.Lock()

    def add_alert(
        self,
        name: str,
        condition: Callable[[logging.LogRecord], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown_seconds: int = 300,
    ) -> str:
        """Add alert definition.

        Args:
            name: Alert name
            condition: Condition function
            severity: Alert severity
            cooldown_seconds: Minimum time between triggers

        Returns:
            Alert ID
        """
        import uuid

        alert_id = str(uuid.uuid4())

        alert = LogAlert(
            id=alert_id,
            name=name,
            condition=condition,
            severity=severity,
            cooldown_seconds=cooldown_seconds,
        )
        self._alerts[alert_id] = alert
        return alert_id

    def check(self, record: logging.LogRecord) -> list[TriggeredAlert]:
        """Check record against all alerts.

        Args:
            record: Log record to check

        Returns:
            List of triggered alerts
        """
        triggered = []
        now = datetime.now()

        with self._lock:
            for alert in self._alerts.values():
                if not alert.enabled:
                    continue

                # Check cooldown
                if alert.last_triggered:
                    elapsed = (now - alert.last_triggered).total_seconds()
                    if elapsed < alert.cooldown_seconds:
                        continue

                # Check condition
                try:
                    if alert.condition(record):
                        alert.last_triggered = now
                        triggered_alert = TriggeredAlert(alert=alert, record=record, timestamp=now)
                        triggered.append(triggered_alert)
                        self._notify(triggered_alert)
                except Exception as e:
                    logger.warning(f"Alert condition check failed: {e}")

        return triggered

    def on_alert(self, handler: Callable[[TriggeredAlert], None]) -> None:
        """Register alert handler."""
        self._handlers.append(handler)

    def _notify(self, alert: TriggeredAlert) -> None:
        """Notify handlers of triggered alert."""
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.warning(f"Alert handler failed: {e}")


# =============================================================================
# =============================================================================


class SamplingStrategy(Enum):
    """Sampling strategy."""

    RANDOM = auto()
    RATE_LIMIT = auto()
    ADAPTIVE = auto()


class LogSampler:
    """Samples log messages for high-volume scenarios.

    References:
        LOG-015: Log Sampling for High-Volume
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.RATE_LIMIT,
        rate: float = 0.1,  # 10% for random
        max_per_second: int = 100,  # for rate limit
    ):
        self.strategy = strategy
        self.rate = rate
        self.max_per_second = max_per_second
        self._count_this_second = 0
        self._last_second = 0
        self._lock = threading.Lock()

    def should_log(self, record: logging.LogRecord) -> bool:
        """Determine if record should be logged.

        Args:
            record: Log record

        Returns:
            True if should log
        """
        # Always log errors and above
        if record.levelno >= logging.ERROR:
            return True

        if self.strategy == SamplingStrategy.RANDOM:
            import random

            return random.random() < self.rate

        elif self.strategy == SamplingStrategy.RATE_LIMIT:
            with self._lock:
                current_second = int(time.time())
                if current_second != self._last_second:
                    self._last_second = current_second
                    self._count_this_second = 0

                if self._count_this_second < self.max_per_second:
                    self._count_this_second += 1
                    return True
                return False

        elif self.strategy == SamplingStrategy.ADAPTIVE:
            # Reduce sampling as volume increases
            with self._lock:
                current_second = int(time.time())
                if current_second != self._last_second:
                    volume = self._count_this_second
                    self._last_second = current_second
                    self._count_this_second = 0

                    # Adjust rate based on volume
                    if volume > 1000:
                        self.rate = 0.01
                    elif volume > 100:
                        self.rate = 0.1
                    else:
                        self.rate = 1.0

                self._count_this_second += 1
                import random

                return random.random() < self.rate

        return True  # type: ignore[unreachable]


# =============================================================================
# =============================================================================


class LogBuffer:
    """Buffers log messages for batch writing.

    References:
        LOG-016: Log Buffer for Batch Writing
    """

    def __init__(self, max_size: int = 1000, flush_interval_seconds: float = 5.0):
        self.max_size = max_size
        self.flush_interval = flush_interval_seconds
        self._buffer: queue.Queue = queue.Queue(maxsize=max_size)  # type: ignore[type-arg]
        self._handlers: list[Callable[[list[logging.LogRecord]], None]] = []
        self._flush_thread: threading.Thread | None = None
        self._running = False

    def add(self, record: logging.LogRecord) -> None:
        """Add record to buffer."""
        try:
            self._buffer.put_nowait(record)
        except queue.Full:
            # Buffer full, force flush
            self.flush()
            self._buffer.put_nowait(record)

    def flush(self) -> None:
        """Flush buffer to handlers."""
        records = []
        while not self._buffer.empty():
            try:
                records.append(self._buffer.get_nowait())
            except queue.Empty:
                break

        if records:
            for handler in self._handlers:
                try:
                    handler(records)
                except Exception as e:
                    logger.warning(f"Buffer flush handler failed: {e}")

    def on_flush(self, handler: Callable[[list[logging.LogRecord]], None]) -> None:
        """Register flush handler."""
        self._handlers.append(handler)

    def start_auto_flush(self) -> None:
        """Start automatic flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop_auto_flush(self) -> None:
        """Stop automatic flush thread."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=2)
        self.flush()

    def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            time.sleep(self.flush_interval)
            self.flush()


# =============================================================================
# =============================================================================


class CompressedLogHandler(logging.Handler):
    """Handler that writes compressed logs.

    References:
        LOG-017: Log Compression
    """

    def __init__(
        self,
        filename: str,
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        compression_level: int = 9,
    ):
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compression_level = compression_level
        self._current_file: Any = None
        self._current_size = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record."""
        try:
            msg = self.format(record) + "\n"
            msg_bytes = msg.encode("utf-8")

            with self._lock:
                if self._current_file is None:
                    self._open_file()

                if self._current_size + len(msg_bytes) > self.max_bytes:
                    self._rotate()

                self._current_file.write(msg_bytes)
                self._current_size += len(msg_bytes)

        except Exception:
            self.handleError(record)

    def _open_file(self) -> None:
        """Open current log file."""
        self._current_file = gzip.open(
            f"{self.filename}.gz", "ab", compresslevel=self.compression_level
        )
        try:
            self._current_size = os.path.getsize(f"{self.filename}.gz")
        except OSError:
            self._current_size = 0

    def _rotate(self) -> None:
        """Rotate log files."""
        if self._current_file:
            self._current_file.close()

        # Shift existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.filename}.{i}.gz"
            dst = f"{self.filename}.{i + 1}.gz"
            if os.path.exists(src):
                os.rename(src, dst)

        # Move current to .1
        if os.path.exists(f"{self.filename}.gz"):
            os.rename(f"{self.filename}.gz", f"{self.filename}.1.gz")

        self._open_file()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None
        super().close()


# =============================================================================
# =============================================================================


class EncryptedLogHandler(logging.Handler):
    """Handler that writes encrypted logs.

    Uses simple XOR encryption for demonstration.
    In production, use proper encryption (AES, etc.).

    References:
        LOG-018: Log Encryption
    """

    def __init__(self, filename: str, key: str):
        super().__init__()
        self.filename = filename
        self._key = hashlib.sha256(key.encode()).digest()
        self._file: Any = None
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit encrypted log record."""
        try:
            msg = self.format(record) + "\n"
            encrypted = self._encrypt(msg.encode("utf-8"))

            with self._lock:
                if self._file is None:
                    self._file = open(self.filename, "ab")

                # Write length-prefixed encrypted message
                length = len(encrypted).to_bytes(4, "big")
                self._file.write(length + encrypted)
                self._file.flush()

        except Exception:
            self.handleError(record)

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data with XOR."""
        encrypted = bytearray()
        for i, byte in enumerate(data):
            encrypted.append(byte ^ self._key[i % len(self._key)])
        return bytes(encrypted)

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
        super().close()


# =============================================================================
# =============================================================================


class LogForwarderProtocol(Enum):
    """Log forwarding protocols."""

    SYSLOG = auto()
    HTTP = auto()
    TCP = auto()
    UDP = auto()


@dataclass
class ForwardingConfig:
    """Log forwarding configuration.

    References:
        LOG-019: Log Forwarding
    """

    protocol: LogForwarderProtocol
    host: str
    port: int
    timeout: float = 5.0
    batch_size: int = 100
    tls: bool = False


class LogForwarder:
    """Forwards logs to external systems.

    References:
        LOG-019: Log Forwarding
    """

    def __init__(self, config: ForwardingConfig):
        self.config = config
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def forward(self, record: logging.LogRecord) -> None:
        """Forward log record.

        Args:
            record: Log record
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        }

        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.config.batch_size:
                self._flush()

    def _flush(self) -> None:
        """Flush buffer to destination."""
        if not self._buffer:
            return

        entries = self._buffer.copy()
        self._buffer.clear()

        try:
            if self.config.protocol == LogForwarderProtocol.HTTP:
                self._send_http(entries)
            elif self.config.protocol == LogForwarderProtocol.SYSLOG:
                self._send_syslog(entries)
            elif self.config.protocol == LogForwarderProtocol.TCP:
                self._send_tcp(entries)
            elif self.config.protocol == LogForwarderProtocol.UDP:
                self._send_udp(entries)
        except Exception as e:
            logger.warning(f"Log forwarding failed: {e}")
            # Put entries back in buffer
            self._buffer.extend(entries)

    def _send_http(self, entries: list[dict[str, Any]]) -> None:
        """Send via HTTP."""
        import urllib.request

        data = json.dumps(entries).encode("utf-8")
        req = urllib.request.Request(
            f"{'https' if self.config.tls else 'http'}://{self.config.host}:{self.config.port}/logs",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=self.config.timeout)

    def _send_syslog(self, entries: list[dict[str, Any]]) -> None:
        """Send via syslog."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for entry in entries:
            msg = f"<14>{entry['timestamp']} {entry['logger']}: {entry['message']}"
            sock.sendto(msg.encode(), (self.config.host, self.config.port))
        sock.close()

    def _send_tcp(self, entries: list[dict[str, Any]]) -> None:
        """Send via TCP."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.config.timeout)
        sock.connect((self.config.host, self.config.port))
        for entry in entries:
            msg = json.dumps(entry) + "\n"
            sock.send(msg.encode())
        sock.close()

    def _send_udp(self, entries: list[dict[str, Any]]) -> None:
        """Send via UDP."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for entry in entries:
            msg = json.dumps(entry)
            sock.sendto(msg.encode(), (self.config.host, self.config.port))
        sock.close()


# =============================================================================
# =============================================================================


@dataclass
class DashboardMetrics:
    """Metrics for log visualization dashboard.

    References:
        LOG-020: Log Visualization Dashboard Data
    """

    total_logs: int = 0
    logs_by_level: dict[str, int] = field(default_factory=dict)
    logs_by_logger: dict[str, int] = field(default_factory=dict)
    logs_per_minute: list[int] = field(default_factory=list)
    error_rate: float = 0.0
    top_patterns: list[tuple[str, int]] = field(default_factory=list)
    recent_errors: list[dict[str, Any]] = field(default_factory=list)


class LogDashboardCollector:
    """Collects metrics for log visualization.

    References:
        LOG-020: Log Visualization Dashboard Data
    """

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self._logs: deque = deque()  # type: ignore[type-arg]
        self._lock = threading.Lock()

    def add(self, record: logging.LogRecord) -> None:
        """Add log record to metrics."""
        entry = {
            "timestamp": datetime.now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        with self._lock:
            self._logs.append(entry)
            # Trim old entries
            cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
            while self._logs and self._logs[0]["timestamp"] < cutoff:
                self._logs.popleft()

    def get_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics.

        Returns:
            Dashboard metrics
        """
        with self._lock:
            logs = list(self._logs)

        if not logs:
            return DashboardMetrics()

        # Count by level
        level_counts = Counter(log["level"] for log in logs)

        # Count by logger
        logger_counts = Counter(log["logger"] for log in logs)

        # Logs per minute
        now = datetime.now()
        per_minute = []
        for i in range(60):
            minute_start = now - timedelta(minutes=i + 1)
            minute_end = now - timedelta(minutes=i)
            count = sum(1 for log in logs if minute_start <= log["timestamp"] < minute_end)
            per_minute.append(count)
        per_minute.reverse()

        # Error rate
        error_count = sum(1 for log in logs if log["level"] in ("ERROR", "CRITICAL"))
        error_rate = error_count / len(logs) if logs else 0.0

        # Top patterns
        patterns: Counter[str] = Counter()
        for log in logs:
            normalized = re.sub(r"\d+", "<N>", log["message"])
            patterns[normalized] += 1

        # Recent errors
        recent_errors = [log for log in logs if log["level"] in ("ERROR", "CRITICAL")][-10:]

        return DashboardMetrics(
            total_logs=len(logs),
            logs_by_level=dict(level_counts),
            logs_by_logger=dict(logger_counts.most_common(10)),
            logs_per_minute=per_minute,
            error_rate=error_rate,
            top_patterns=patterns.most_common(10),
            recent_errors=recent_errors,
        )


__all__ = [
    # Aggregation (LOG-009)
    "AggregatedLogEntry",
    # Alerting (LOG-012)
    "AlertSeverity",
    # Compression (LOG-017)
    "CompressedLogHandler",
    # Dashboard (LOG-020)
    "DashboardMetrics",
    # Encryption (LOG-018)
    "EncryptedLogHandler",
    # Forwarding (LOG-019)
    "ForwardingConfig",
    "LogAggregator",
    "LogAlert",
    "LogAlerter",
    # Analysis (LOG-010)
    "LogAnalyzer",
    # Buffer (LOG-016)
    "LogBuffer",
    "LogDashboardCollector",
    "LogForwarder",
    "LogForwarderProtocol",
    "LogPattern",
    # Sampling (LOG-015)
    "LogSampler",
    "SamplingStrategy",
    "TriggeredAlert",
]
