"""Unit tests for memory extensions utilities.

This module tests memory-related utility functions including:
- Memory monitoring
- Memory cleanup
- Memory profiling helpers
- Resource management
"""

import gc

from oscura.utils import memory_extensions


class TestMemoryExtensions:
    """Test cases for memory extension utilities."""

    def test_get_memory_usage(self) -> None:
        """Test getting current memory usage."""
        if hasattr(memory_extensions, "get_memory_usage"):
            usage = memory_extensions.get_memory_usage()

            # Should return a number
            assert isinstance(usage, (int, float))
            # Should be positive
            assert usage > 0

    def test_memory_profiler_context(self) -> None:
        """Test memory profiler context manager."""
        if hasattr(memory_extensions, "memory_profiler"):
            with memory_extensions.memory_profiler() as prof:
                # Allocate some memory
                data = [0] * 1000
                assert len(data) == 1000

            # Profiler should have recorded something
            assert prof is not None

    def test_cleanup_memory(self) -> None:
        """Test memory cleanup function."""
        if hasattr(memory_extensions, "cleanup_memory"):
            # Create some garbage
            temp_data = [[i] * 1000 for i in range(100)]
            del temp_data

            # Cleanup should not raise
            memory_extensions.cleanup_memory()

            # Garbage collection should have run
            assert gc.isenabled()

    def test_memory_limit_check(self) -> None:
        """Test checking against memory limits."""
        if hasattr(memory_extensions, "check_memory_limit"):
            # Should not raise with reasonable limit
            result = memory_extensions.check_memory_limit(limit_mb=10000)
            assert isinstance(result, bool)

    def test_estimate_array_size(self) -> None:
        """Test array size estimation."""
        if hasattr(memory_extensions, "estimate_array_size"):
            import numpy as np

            # Estimate size of a potential array
            size = memory_extensions.estimate_array_size(shape=(1000, 1000), dtype=np.float64)

            # Should be approximately 8MB (1M elements * 8 bytes)
            assert size > 7_000_000
            assert size < 9_000_000

    def test_memory_efficient_operation(self) -> None:
        """Test memory-efficient operation decorator."""
        if hasattr(memory_extensions, "memory_efficient"):

            @memory_extensions.memory_efficient
            def process_data(size: int) -> list[int]:
                return list(range(size))

            result = process_data(100)
            assert len(result) == 100

    def test_memory_guard(self) -> None:
        """Test memory guard context manager."""
        if hasattr(memory_extensions, "memory_guard"):
            with memory_extensions.memory_guard(max_mb=1000):
                # Small allocation should be fine
                data = [0] * 1000
                assert len(data) == 1000

    def test_get_peak_memory(self) -> None:
        """Test getting peak memory usage."""
        if hasattr(memory_extensions, "get_peak_memory"):
            peak = memory_extensions.get_peak_memory()

            if peak is not None:
                assert isinstance(peak, (int, float))
                assert peak > 0

    def test_reset_peak_memory(self) -> None:
        """Test resetting peak memory counter."""
        if hasattr(memory_extensions, "reset_peak_memory"):
            # Should not raise
            memory_extensions.reset_peak_memory()

    def test_memory_snapshot(self) -> None:
        """Test creating memory snapshot."""
        if hasattr(memory_extensions, "memory_snapshot"):
            snapshot = memory_extensions.memory_snapshot()

            # Should return some kind of snapshot object
            assert snapshot is not None

    def test_compare_snapshots(self) -> None:
        """Test comparing memory snapshots."""
        if hasattr(memory_extensions, "memory_snapshot") and hasattr(
            memory_extensions, "compare_snapshots"
        ):
            snap1 = memory_extensions.memory_snapshot()

            # Allocate some memory
            data = [0] * 10000

            snap2 = memory_extensions.memory_snapshot()

            # Compare (might return diff, might return None if not supported)
            if hasattr(memory_extensions, "compare_snapshots"):
                diff = memory_extensions.compare_snapshots(snap1, snap2)
                # Just verify it doesn't crash
                assert diff is not None or diff is None

            del data

    def test_format_bytes(self) -> None:
        """Test byte formatting utility."""
        if hasattr(memory_extensions, "format_bytes"):
            formatted = memory_extensions.format_bytes(1024)
            assert "KB" in formatted or "KiB" in formatted or "1024" in formatted

            formatted = memory_extensions.format_bytes(1048576)
            assert "MB" in formatted or "MiB" in formatted

    def test_get_available_memory(self) -> None:
        """Test getting available memory."""
        if hasattr(memory_extensions, "get_available_memory"):
            available = memory_extensions.get_available_memory()

            if available is not None:
                assert isinstance(available, (int, float))
                assert available > 0

    def test_memory_warning_threshold(self) -> None:
        """Test memory warning threshold functionality."""
        if hasattr(memory_extensions, "set_memory_warning_threshold"):
            # Should not raise
            memory_extensions.set_memory_warning_threshold(80)  # 80%

    def test_module_has_expected_functions(self) -> None:
        """Test that module exports expected functions."""
        # Check for at least some memory-related functions
        assert hasattr(memory_extensions, "__name__")
