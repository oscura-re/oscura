"""Tests for memory monitoring and OOM prevention.

Tests the memory monitoring infrastructure (MEM-015, MEM-024).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.core.memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    ProgressMonitor,
    ProgressWithMemory,
    monitor_memory,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        """Test creating memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            available=4 * 1024**3,  # 4 GB
            process_rss=512 * 1024**2,  # 512 MB
            process_vms=1024 * 1024**2,  # 1 GB
            pressure=0.3,
        )
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.available == 4 * 1024**3
        assert snapshot.process_rss == 512 * 1024**2
        assert snapshot.process_vms == 1024 * 1024**2
        assert snapshot.pressure == 0.3


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    @patch("oscura.core.memory_monitor.get_max_memory")
    def test_init_no_max_memory(self, mock_get_max: MagicMock) -> None:
        """Test initialization without max_memory (uses global config)."""
        mock_get_max.return_value = 8 * 1024**3  # 8 GB
        monitor = MemoryMonitor("test_op")
        assert monitor.operation == "test_op"
        assert monitor.check_interval == 100
        assert monitor.abort_on_critical is True
        assert monitor.max_memory == 8 * 1024**3
        mock_get_max.assert_called_once()

    @patch("oscura.config.memory._parse_memory_string")
    def test_init_string_max_memory(self, mock_parse: MagicMock) -> None:
        """Test initialization with string max_memory."""
        mock_parse.return_value = 4 * 1024**3  # 4 GB
        monitor = MemoryMonitor("test_op", max_memory="4GB")
        assert monitor.max_memory == 4 * 1024**3
        mock_parse.assert_called_once_with("4GB")

    def test_init_int_max_memory(self) -> None:
        """Test initialization with integer max_memory."""
        monitor = MemoryMonitor("test_op", max_memory=2 * 1024**3)
        assert monitor.max_memory == 2 * 1024**3

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        monitor = MemoryMonitor(
            "custom_op",
            max_memory=1024**3,
            check_interval=50,
            abort_on_critical=False,
        )
        assert monitor.operation == "custom_op"
        assert monitor.check_interval == 50
        assert monitor.abort_on_critical is False

    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_context_enter(self, mock_snapshot: MagicMock, mock_memory: MagicMock) -> None:
        """Test entering context manager."""
        mock_memory.return_value = 100 * 1024**2  # 100 MB
        monitor = MemoryMonitor("test_op", max_memory=1024**3)

        with monitor as mon:
            assert mon is monitor
            assert monitor.start_memory == 100 * 1024**2
            assert monitor.peak_memory == 100 * 1024**2
            assert monitor.current_memory == 100 * 1024**2
            assert monitor._start_time > 0
            mock_snapshot.assert_called_once()

    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_context_exit(self, mock_snapshot: MagicMock, mock_memory: MagicMock) -> None:
        """Test exiting context manager."""
        mock_memory.return_value = 100 * 1024**2
        monitor = MemoryMonitor("test_op", max_memory=1024**3)

        with monitor:
            pass

        # Should have called _take_snapshot twice (enter + exit)
        assert mock_snapshot.call_count == 2

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.get_memory_config")
    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    def test_check_below_interval(
        self,
        mock_process_mem: MagicMock,
        mock_config: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test check skips when iteration not at interval."""
        monitor = MemoryMonitor("test_op", max_memory=1024**3, check_interval=100)
        monitor.check(iteration=50)  # Not at interval boundary
        # Should not call memory functions
        assert not mock_process_mem.called

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.get_memory_config")
    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_check_at_interval(
        self,
        mock_snapshot: MagicMock,
        mock_process_mem: MagicMock,
        mock_config: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test check runs when iteration at interval."""
        mock_process_mem.return_value = 200 * 1024**2  # 200 MB
        mock_available.return_value = 4 * 1024**3  # 4 GB
        mock_config.return_value = Mock(critical_threshold=0.95)

        monitor = MemoryMonitor("test_op", max_memory=8 * 1024**3, check_interval=100)
        monitor.check(iteration=100)  # At interval boundary

        mock_process_mem.assert_called_once()
        assert monitor.current_memory == 200 * 1024**2
        assert monitor.peak_memory == 200 * 1024**2

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.get_memory_config")
    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    def test_check_critical_threshold(
        self,
        mock_process_mem: MagicMock,
        mock_config: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test check raises MemoryError when critical threshold exceeded."""
        mock_process_mem.return_value = 200 * 1024**2
        mock_available.return_value = 100 * 1024**2  # Very low available memory
        mock_config.return_value = Mock(critical_threshold=0.9)

        monitor = MemoryMonitor("test_op", max_memory=2 * 1024**3, abort_on_critical=True)
        monitor.check_interval = 1  # Check every time

        with pytest.raises(MemoryError, match="Critical memory pressure"):
            monitor.check(iteration=1)

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.get_memory_config")
    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    def test_check_no_abort_on_critical(
        self,
        mock_process_mem: MagicMock,
        mock_config: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test check doesn't raise when abort_on_critical=False."""
        mock_process_mem.return_value = 200 * 1024**2
        mock_available.return_value = 100 * 1024**2
        mock_config.return_value = Mock(critical_threshold=0.9)

        monitor = MemoryMonitor("test_op", max_memory=2 * 1024**3, abort_on_critical=False)
        monitor.check_interval = 1

        # Should not raise
        monitor.check(iteration=1)

    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_get_stats(self, mock_snapshot: MagicMock, mock_memory: MagicMock) -> None:
        """Test get_stats returns correct statistics."""
        mock_memory.return_value = 100 * 1024**2
        monitor = MemoryMonitor("test_op", max_memory=1024**3)

        with monitor:
            monitor.peak_memory = 150 * 1024**2
            monitor.current_memory = 120 * 1024**2
            time.sleep(0.01)  # Ensure duration > 0
            stats = monitor.get_stats()

        assert stats["start"] == 100 * 1024**2
        assert stats["current"] == 120 * 1024**2
        assert stats["peak"] == 150 * 1024**2
        assert stats["delta"] == 50 * 1024**2
        assert stats["duration"] > 0

    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_get_snapshots(self, mock_snapshot: MagicMock, mock_memory: MagicMock) -> None:
        """Test get_snapshots returns copy of snapshots list."""
        mock_memory.return_value = 100 * 1024**2
        monitor = MemoryMonitor("test_op", max_memory=1024**3)

        snapshot1 = MemorySnapshot(
            timestamp=time.time(),
            available=4 * 1024**3,
            process_rss=100 * 1024**2,
            process_vms=200 * 1024**2,
            pressure=0.2,
        )
        snapshot2 = MemorySnapshot(
            timestamp=time.time(),
            available=3 * 1024**3,
            process_rss=150 * 1024**2,
            process_vms=250 * 1024**2,
            pressure=0.3,
        )
        monitor._snapshots = [snapshot1, snapshot2]

        snapshots = monitor.get_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0] is snapshot1
        assert snapshots[1] is snapshot2
        # Verify it's a copy
        assert snapshots is not monitor._snapshots

    def test_take_snapshot_no_psutil(self) -> None:
        """Test _take_snapshot skips when psutil not available."""
        monitor = MemoryMonitor("test_op", max_memory=1024**3)

        with patch.dict("sys.modules", {"psutil": None}):
            monitor._take_snapshot()

        # Should have no snapshots (skipped due to ImportError)
        assert len(monitor._snapshots) == 0


class TestMonitorMemory:
    """Test monitor_memory context manager."""

    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_monitor_memory_context(self, mock_snapshot: MagicMock, mock_memory: MagicMock) -> None:
        """Test monitor_memory context manager wraps MemoryMonitor."""
        mock_memory.return_value = 100 * 1024**2

        with monitor_memory("test_op", max_memory="2GB", check_interval=50) as mon:
            assert isinstance(mon, MemoryMonitor)
            assert mon.operation == "test_op"
            assert mon.check_interval == 50


class TestProgressWithMemory:
    """Test ProgressWithMemory dataclass."""

    def test_create_progress(self) -> None:
        """Test creating progress with memory."""
        progress = ProgressWithMemory(
            current=50,
            total=100,
            eta_seconds=30.0,
            memory_used=1 * 1024**3,
            memory_peak=1.5 * 1024**3,
            memory_available=4 * 1024**3,
            operation="test_op",
        )
        assert progress.current == 50
        assert progress.total == 100
        assert progress.eta_seconds == 30.0
        assert progress.memory_used == 1 * 1024**3
        assert progress.operation == "test_op"

    def test_percent_property(self) -> None:
        """Test percent property calculation."""
        progress = ProgressWithMemory(
            current=25,
            total=100,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            memory_available=0,
            operation="op",
        )
        assert progress.percent == 25.0

    def test_percent_zero_total(self) -> None:
        """Test percent property with zero total."""
        progress = ProgressWithMemory(
            current=0,
            total=0,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            memory_available=0,
            operation="op",
        )
        assert progress.percent == 100.0

    @patch("oscura.utils.memory.get_memory_pressure")
    def test_memory_pressure_property(self, mock_pressure: MagicMock) -> None:
        """Test memory_pressure property."""
        mock_pressure.return_value = 0.45
        progress = ProgressWithMemory(
            current=50,
            total=100,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            memory_available=0,
            operation="op",
        )
        assert progress.memory_pressure == 0.45
        mock_pressure.assert_called_once()

    def test_format_progress(self) -> None:
        """Test format_progress output."""
        progress = ProgressWithMemory(
            current=42,
            total=100,
            eta_seconds=5.0,
            memory_used=1.2 * 1024**3,
            memory_peak=2.1 * 1024**3,
            memory_available=6 * 1024**3,
            operation="fft",
        )
        formatted = progress.format_progress()
        assert "42.0%" in formatted
        assert (
            "GB used" in formatted
        )  # Check for presence, not exact value due to binary/decimal conversion
        assert "GB peak" in formatted
        assert "GB avail" in formatted
        assert "ETA 5s" in formatted


class TestProgressMonitor:
    """Test ProgressMonitor class."""

    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_init(self, mock_monitor_class: MagicMock) -> None:
        """Test ProgressMonitor initialization."""
        callback = Mock()
        monitor = ProgressMonitor("test_op", total=1000, callback=callback, update_interval=10)

        assert monitor.operation == "test_op"
        assert monitor.total == 1000
        assert monitor.callback is callback
        assert monitor.update_interval == 10
        assert monitor.current == 0

    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_update_with_value(self, mock_monitor_class: MagicMock) -> None:
        """Test update with explicit value."""
        mock_mem_monitor = Mock()
        mock_monitor_class.return_value = mock_mem_monitor

        monitor = ProgressMonitor("test_op", total=100)
        monitor.update(current=50)

        assert monitor.current == 50
        mock_mem_monitor.check.assert_called()

    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_update_increment(self, mock_monitor_class: MagicMock) -> None:
        """Test update with no value (increment by 1)."""
        mock_mem_monitor = Mock()
        mock_monitor_class.return_value = mock_mem_monitor

        monitor = ProgressMonitor("test_op", total=100)
        monitor.update()
        monitor.update()

        assert monitor.current == 2

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_update_calls_callback(
        self, mock_monitor_class: MagicMock, mock_available: MagicMock
    ) -> None:
        """Test update calls callback at interval."""
        mock_mem_monitor = Mock()
        mock_mem_monitor.get_stats.return_value = {"current": 100 * 1024**2, "peak": 150 * 1024**2}
        mock_monitor_class.return_value = mock_mem_monitor
        mock_available.return_value = 4 * 1024**3

        callback = Mock()
        monitor = ProgressMonitor("test_op", total=100, callback=callback, update_interval=10)

        # Update 10 times
        for _i in range(10):
            monitor.update()

        # Callback should be called once (at update_count=10)
        assert callback.call_count == 1

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_get_progress(self, mock_monitor_class: MagicMock, mock_available: MagicMock) -> None:
        """Test get_progress returns ProgressWithMemory."""
        mock_mem_monitor = Mock()
        mock_mem_monitor.get_stats.return_value = {
            "current": 1 * 1024**3,
            "peak": 1.5 * 1024**3,
        }
        mock_monitor_class.return_value = mock_mem_monitor
        mock_available.return_value = 4 * 1024**3

        monitor = ProgressMonitor("test_op", total=100)
        monitor.current = 50
        time.sleep(0.01)  # Ensure elapsed > 0

        progress = monitor.get_progress()

        assert isinstance(progress, ProgressWithMemory)
        assert progress.current == 50
        assert progress.total == 100
        assert progress.memory_used == 1 * 1024**3
        assert progress.memory_peak == 1.5 * 1024**3
        assert progress.eta_seconds > 0  # Should calculate ETA

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_get_progress_zero_current(
        self, mock_monitor_class: MagicMock, mock_available: MagicMock
    ) -> None:
        """Test get_progress with zero current (ETA=0)."""
        mock_mem_monitor = Mock()
        mock_mem_monitor.get_stats.return_value = {"current": 0, "peak": 0}
        mock_monitor_class.return_value = mock_mem_monitor
        mock_available.return_value = 4 * 1024**3

        monitor = ProgressMonitor("test_op", total=100)
        progress = monitor.get_progress()

        assert progress.eta_seconds == 0.0  # Can't calculate ETA with current=0


class TestCoreMemoryMonitorIntegration:
    """Integration tests for memory monitoring."""

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.get_memory_config")
    @patch("oscura.core.memory_monitor.MemoryMonitor._get_process_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor._take_snapshot")
    def test_full_workflow(
        self,
        mock_snapshot: MagicMock,
        mock_process_mem: MagicMock,
        mock_config: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test complete memory monitoring workflow."""
        mock_process_mem.return_value = 100 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_config.return_value = Mock(critical_threshold=0.95)

        with MemoryMonitor("integration_test", max_memory="8GB") as monitor:
            # Simulate work with periodic checks
            for i in range(300):
                if i % 100 == 0:
                    monitor.check(i)

            stats = monitor.get_stats()

        assert stats["start"] == 100 * 1024**2
        assert stats["duration"] > 0

    @patch("oscura.core.memory_monitor.get_available_memory")
    @patch("oscura.core.memory_monitor.MemoryMonitor")
    def test_progress_monitor_workflow(
        self, mock_monitor_class: MagicMock, mock_available: MagicMock
    ) -> None:
        """Test complete progress monitoring workflow."""
        mock_mem_monitor = Mock()
        mock_mem_monitor.get_stats.return_value = {"current": 500 * 1024**2, "peak": 600 * 1024**2}
        mock_monitor_class.return_value = mock_mem_monitor
        mock_available.return_value = 4 * 1024**3

        results = []

        def callback(progress: ProgressWithMemory) -> None:
            results.append(progress.percent)

        monitor = ProgressMonitor("workflow_test", total=100, callback=callback, update_interval=25)

        for _i in range(100):
            monitor.update()

        # Callback should be called 4 times (at 25, 50, 75, 100)
        assert len(results) == 4
        assert results == [25.0, 50.0, 75.0, 100.0]
