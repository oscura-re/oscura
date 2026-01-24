"""Comprehensive tests for plugin isolation system.

Requirements tested:

This module provides comprehensive tests for the plugin sandboxing
and isolation system, including resource limits, permissions, and
performance isolation.

NOTE: These tests apply actual resource limits which can interfere with
pytest-xdist worker processes. When running with xdist (parallel mode),
these tests are automatically skipped to prevent worker crashes.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from oscura.plugins.isolation import (
    IsolationManager,
    Permission,
    PermissionSet,
    PluginSandbox,
    ResourceExceededError,
    ResourceLimits,
    get_isolation_manager,
)

# Skip all tests in this module when running with pytest-xdist to avoid
# resource limit conflicts with worker processes
# Also skip in CI due to GitHub Actions runner resource limitations:
# - Memory: 2GB limit causes OOM on concurrent tests
# - Timeout: Tests require 60+ minutes, exceeding CI limits
# - Threads: Python 3.13 hits thread creation limits
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        "PYTEST_XDIST_WORKER" in os.environ,
        reason="Resource limit tests interfere with pytest-xdist workers",
    ),
    pytest.mark.skipif(
        "CI" in os.environ,
        reason="CI runner limitations: 2GB memory, 60min timeout, thread limits (tests pass locally)",
    ),
]


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self) -> None:
        """Test that all expected permissions exist."""
        assert Permission.READ_CONFIG is not None
        assert Permission.WRITE_CONFIG is not None
        assert Permission.READ_DATA is not None
        assert Permission.WRITE_DATA is not None
        assert Permission.NETWORK_ACCESS is not None
        assert Permission.SUBPROCESS is not None
        assert Permission.NATIVE_CODE is not None
        assert Permission.SYSTEM_INFO is not None

    def test_permission_uniqueness(self) -> None:
        """Test that all permissions have unique values."""
        values = [p.value for p in Permission]
        assert len(values) == len(set(values))


class TestPermissionSet:
    """Tests for PermissionSet class."""

    def test_empty_permission_set(self) -> None:
        """Test empty permission set denies all."""
        perm_set = PermissionSet()

        assert not perm_set.has_permission(Permission.READ_CONFIG)
        assert not perm_set.has_permission(Permission.WRITE_DATA)

    def test_grant_permission(self) -> None:
        """Test granting a permission."""
        perm_set = PermissionSet()

        perm_set.grant(Permission.READ_CONFIG)

        assert perm_set.has_permission(Permission.READ_CONFIG)
        assert Permission.READ_CONFIG in perm_set.allowed

    def test_deny_permission(self) -> None:
        """Test denying a permission."""
        perm_set = PermissionSet()
        perm_set.grant(Permission.READ_CONFIG)

        perm_set.deny(Permission.READ_CONFIG)

        assert not perm_set.has_permission(Permission.READ_CONFIG)
        assert Permission.READ_CONFIG in perm_set.denied
        assert Permission.READ_CONFIG not in perm_set.allowed

    def test_check_raises_on_denied(self) -> None:
        """Test that check() raises PermissionError."""
        perm_set = PermissionSet()

        with pytest.raises(PermissionError, match="READ_CONFIG"):
            perm_set.check(Permission.READ_CONFIG)

    def test_check_passes_on_granted(self) -> None:
        """Test that check() passes for granted permission."""
        perm_set = PermissionSet()
        perm_set.grant(Permission.READ_CONFIG)

        # Should not raise
        perm_set.check(Permission.READ_CONFIG)

    def test_deny_overrides_grant(self) -> None:
        """Test that explicit deny overrides grant."""
        perm_set = PermissionSet()

        perm_set.grant(Permission.NETWORK_ACCESS)
        perm_set.deny(Permission.NETWORK_ACCESS)

        assert not perm_set.has_permission(Permission.NETWORK_ACCESS)

    def test_grant_after_deny(self) -> None:
        """Test granting after denying restores permission."""
        perm_set = PermissionSet()

        perm_set.deny(Permission.SUBPROCESS)
        perm_set.grant(Permission.SUBPROCESS)

        assert perm_set.has_permission(Permission.SUBPROCESS)
        assert Permission.SUBPROCESS not in perm_set.denied

    def test_multiple_permissions(self) -> None:
        """Test managing multiple permissions."""
        perm_set = PermissionSet()

        perm_set.grant(Permission.READ_CONFIG)
        perm_set.grant(Permission.READ_DATA)
        perm_set.deny(Permission.WRITE_CONFIG)

        assert perm_set.has_permission(Permission.READ_CONFIG)
        assert perm_set.has_permission(Permission.READ_DATA)
        assert not perm_set.has_permission(Permission.WRITE_CONFIG)
        assert not perm_set.has_permission(Permission.NETWORK_ACCESS)


# =============================================================================
# Resource Limits Tests
# =============================================================================


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_limits(self) -> None:
        """Test default limits are unlimited."""
        limits = ResourceLimits()

        assert limits.max_memory_mb is None
        assert limits.max_cpu_time_sec is None
        assert limits.max_wall_time_sec is None
        assert limits.max_file_size_mb is None
        assert limits.max_open_files is None

    def test_custom_limits(self) -> None:
        """Test setting custom limits."""
        limits = ResourceLimits(
            max_memory_mb=512,
            max_cpu_time_sec=30.0,
            max_wall_time_sec=60.0,
            max_file_size_mb=100,
            max_open_files=128,
        )

        assert limits.max_memory_mb == 512
        assert limits.max_cpu_time_sec == 30.0
        assert limits.max_wall_time_sec == 60.0
        assert limits.max_file_size_mb == 100
        assert limits.max_open_files == 128

    def test_to_rlimit_dict_empty(self) -> None:
        """Test rlimit conversion with no limits."""
        limits = ResourceLimits()

        rlimits = limits.to_rlimit_dict()

        assert rlimits == {}

    def test_to_rlimit_dict_memory(self) -> None:
        """Test rlimit conversion for memory limit."""
        limits = ResourceLimits(max_memory_mb=256)

        rlimits = limits.to_rlimit_dict()

        import resource

        assert resource.RLIMIT_AS in rlimits
        expected = 256 * 1024 * 1024
        assert rlimits[resource.RLIMIT_AS] == (expected, expected)

    def test_to_rlimit_dict_cpu(self) -> None:
        """Test rlimit conversion for CPU time limit."""
        limits = ResourceLimits(max_cpu_time_sec=10.0)

        rlimits = limits.to_rlimit_dict()

        import resource

        assert resource.RLIMIT_CPU in rlimits
        assert rlimits[resource.RLIMIT_CPU] == (10, 10)

    def test_to_rlimit_dict_file_size(self) -> None:
        """Test rlimit conversion for file size limit."""
        limits = ResourceLimits(max_file_size_mb=50)

        rlimits = limits.to_rlimit_dict()

        import resource

        assert resource.RLIMIT_FSIZE in rlimits
        expected = 50 * 1024 * 1024
        assert rlimits[resource.RLIMIT_FSIZE] == (expected, expected)

    def test_to_rlimit_dict_open_files(self) -> None:
        """Test rlimit conversion for open files limit."""
        limits = ResourceLimits(max_open_files=64)

        rlimits = limits.to_rlimit_dict()

        import resource

        assert resource.RLIMIT_NOFILE in rlimits
        assert rlimits[resource.RLIMIT_NOFILE] == (64, 64)


# =============================================================================
# Plugin Sandbox Tests
# =============================================================================


class TestPluginSandbox:
    """Tests for PluginSandbox class."""

    def test_default_sandbox(self) -> None:
        """Test creating sandbox with defaults."""
        sandbox = PluginSandbox()

        assert sandbox.permissions is not None
        assert sandbox.limits is not None

    def test_sandbox_with_permissions(self) -> None:
        """Test creating sandbox with permissions."""
        perms = PermissionSet()
        perms.grant(Permission.READ_DATA)

        sandbox = PluginSandbox(permissions=perms)

        assert sandbox.permissions.has_permission(Permission.READ_DATA)

    def test_sandbox_with_limits(self) -> None:
        """Test creating sandbox with resource limits."""
        limits = ResourceLimits(max_memory_mb=128)

        sandbox = PluginSandbox(limits=limits)

        assert sandbox.limits.max_memory_mb == 128

    def test_execute_context_manager(self) -> None:
        """Test execute context manager."""
        sandbox = PluginSandbox()

        with sandbox.execute():
            # Code runs in sandbox
            result = 1 + 1

        assert result == 2

    def test_execute_with_timeout(self) -> None:
        """Test execute with timeout."""
        sandbox = PluginSandbox()

        with sandbox.execute(timeout=5.0):
            # Fast operation
            time.sleep(0.01)

    def test_check_permission_allowed(self) -> None:
        """Test checking allowed permission."""
        perms = PermissionSet()
        perms.grant(Permission.READ_CONFIG)
        sandbox = PluginSandbox(permissions=perms)

        # Should not raise
        sandbox.check_permission(Permission.READ_CONFIG)

    def test_check_permission_denied(self) -> None:
        """Test checking denied permission."""
        sandbox = PluginSandbox()

        with pytest.raises(PermissionError):
            sandbox.check_permission(Permission.NETWORK_ACCESS)


# =============================================================================
# Isolation Manager Tests
# =============================================================================


class TestIsolationManager:
    """Tests for IsolationManager class."""

    def test_create_sandbox(self) -> None:
        """Test creating sandbox for plugin."""
        manager = IsolationManager()

        sandbox = manager.create_sandbox("test_plugin")

        assert sandbox is not None
        assert manager.get_sandbox("test_plugin") is sandbox

    def test_create_sandbox_with_permissions(self) -> None:
        """Test creating sandbox with custom permissions."""
        manager = IsolationManager()
        perms = PermissionSet()
        perms.grant(Permission.READ_DATA)

        sandbox = manager.create_sandbox("reader_plugin", permissions=perms)

        assert sandbox.permissions.has_permission(Permission.READ_DATA)

    def test_create_sandbox_with_limits(self) -> None:
        """Test creating sandbox with custom limits."""
        manager = IsolationManager()
        limits = ResourceLimits(max_memory_mb=64)

        sandbox = manager.create_sandbox("limited_plugin", limits=limits)

        assert sandbox.limits.max_memory_mb == 64

    def test_get_nonexistent_sandbox(self) -> None:
        """Test getting sandbox for non-existent plugin."""
        manager = IsolationManager()

        sandbox = manager.get_sandbox("nonexistent")

        assert sandbox is None

    def test_remove_sandbox(self) -> None:
        """Test removing sandbox."""
        manager = IsolationManager()
        manager.create_sandbox("temp_plugin")

        manager.remove_sandbox("temp_plugin")

        assert manager.get_sandbox("temp_plugin") is None

    def test_remove_nonexistent_sandbox(self) -> None:
        """Test removing non-existent sandbox (should not raise)."""
        manager = IsolationManager()

        # Should not raise
        manager.remove_sandbox("nonexistent")

    def test_default_limits(self) -> None:
        """Test default resource limits."""
        manager = IsolationManager()

        sandbox = manager.create_sandbox("default_plugin")

        # Default limits should be applied
        assert sandbox.limits.max_memory_mb == 512
        assert sandbox.limits.max_cpu_time_sec == 30.0


class TestGlobalIsolationManager:
    """Tests for global isolation manager."""

    def test_get_isolation_manager_singleton(self) -> None:
        """Test that get_isolation_manager returns same instance."""
        manager1 = get_isolation_manager()
        manager2 = get_isolation_manager()

        assert manager1 is manager2


# =============================================================================
# Performance Isolation Tests (PLUG-007)
# =============================================================================


class TestPerformanceIsolation:
    """Tests for plugin performance isolation."""

    def test_cpu_isolation_between_plugins(self) -> None:
        """Test that plugins don't interfere with each other's CPU."""
        manager = IsolationManager()

        # Create two sandboxes with CPU limits
        sandbox1 = manager.create_sandbox(
            "cpu_plugin_1",
            limits=ResourceLimits(max_cpu_time_sec=10.0),
        )
        sandbox2 = manager.create_sandbox(
            "cpu_plugin_2",
            limits=ResourceLimits(max_cpu_time_sec=10.0),
        )

        # Both should have independent limits
        assert sandbox1.limits.max_cpu_time_sec == 10.0
        assert sandbox2.limits.max_cpu_time_sec == 10.0

    def test_memory_isolation_between_plugins(self) -> None:
        """Test that plugins don't interfere with each other's memory."""
        manager = IsolationManager()

        sandbox1 = manager.create_sandbox(
            "mem_plugin_1",
            limits=ResourceLimits(max_memory_mb=128),
        )
        sandbox2 = manager.create_sandbox(
            "mem_plugin_2",
            limits=ResourceLimits(max_memory_mb=256),
        )

        assert sandbox1.limits.max_memory_mb == 128
        assert sandbox2.limits.max_memory_mb == 256

    def test_concurrent_sandbox_execution(self) -> None:
        """Test concurrent execution of multiple sandboxes."""
        manager = IsolationManager()
        results: dict[str, bool] = {}
        errors: list[Exception] = []

        def run_in_sandbox(name: str) -> None:
            try:
                sandbox = manager.create_sandbox(
                    name,
                    limits=ResourceLimits(max_cpu_time_sec=5.0),
                )
                with sandbox.execute(timeout=2.0):
                    # Simulate work
                    time.sleep(0.1)
                    _ = sum(range(10000))
                results[name] = True
            except Exception as e:
                errors.append(e)
                results[name] = False

        threads = []
        for i in range(5):
            t = threading.Thread(target=run_in_sandbox, args=(f"concurrent_{i}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete successfully
        assert all(results.values()), f"Errors: {errors}"

    def test_sandbox_resource_cleanup(self) -> None:
        """Test that sandbox cleans up resources after execution."""
        sandbox = PluginSandbox(
            limits=ResourceLimits(max_memory_mb=128),
        )

        with sandbox.execute():
            # Create some data
            data = np.zeros(1000)
            del data

        # Original limits should be restored
        assert sandbox._original_limits == {}

    def test_sandbox_exception_cleanup(self) -> None:
        """Test that sandbox cleans up even on exception."""
        sandbox = PluginSandbox(
            limits=ResourceLimits(max_memory_mb=128),
        )

        try:
            with sandbox.execute():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Resources should still be cleaned up
        assert sandbox._original_limits == {}

    def test_plugin_performance_metrics(self) -> None:
        """Test measuring plugin performance within sandbox."""
        sandbox = PluginSandbox()

        start_time = time.time()

        with sandbox.execute(timeout=5.0):
            # Simulate work
            _ = [i**2 for i in range(10000)]

        elapsed = time.time() - start_time

        # Should complete within sandbox timeout (5s)
        assert elapsed < 5.0

    def test_independent_permission_sets(self) -> None:
        """Test that plugins have independent permission sets."""
        manager = IsolationManager()

        perms1 = PermissionSet()
        perms1.grant(Permission.READ_DATA)

        perms2 = PermissionSet()
        perms2.grant(Permission.NETWORK_ACCESS)

        sandbox1 = manager.create_sandbox("perm_plugin_1", permissions=perms1)
        sandbox2 = manager.create_sandbox("perm_plugin_2", permissions=perms2)

        assert sandbox1.permissions.has_permission(Permission.READ_DATA)
        assert not sandbox1.permissions.has_permission(Permission.NETWORK_ACCESS)

        assert sandbox2.permissions.has_permission(Permission.NETWORK_ACCESS)
        assert not sandbox2.permissions.has_permission(Permission.READ_DATA)


# =============================================================================
# Resource Limit Enforcement Tests
# =============================================================================


class TestResourceEnforcement:
    """Tests for resource limit enforcement."""

    def test_memory_limit_enforcement(self) -> None:
        """Test that memory limits are enforced."""
        sandbox = PluginSandbox(
            limits=ResourceLimits(max_memory_mb=1),  # Very small limit
        )

        with pytest.raises(ResourceExceededError), sandbox.execute():
            # Try to allocate more than limit
            _ = np.zeros(10 * 1024 * 1024)  # 80 MB

    def test_no_limit_allows_allocation(self) -> None:
        """Test that no limit allows memory allocation."""
        sandbox = PluginSandbox()

        with sandbox.execute():
            # Should succeed without limits
            data = np.zeros(1000)
            assert len(data) == 1000


# =============================================================================
# Permission Scenarios Tests
# =============================================================================


class TestPermissionScenarios:
    """Tests for realistic permission scenarios."""

    def test_read_only_plugin(self) -> None:
        """Test read-only plugin permissions."""
        perms = PermissionSet()
        perms.grant(Permission.READ_CONFIG)
        perms.grant(Permission.READ_DATA)
        perms.deny(Permission.WRITE_CONFIG)
        perms.deny(Permission.WRITE_DATA)

        sandbox = PluginSandbox(permissions=perms)

        # Should pass
        sandbox.check_permission(Permission.READ_CONFIG)
        sandbox.check_permission(Permission.READ_DATA)

        # Should fail
        with pytest.raises(PermissionError):
            sandbox.check_permission(Permission.WRITE_CONFIG)
        with pytest.raises(PermissionError):
            sandbox.check_permission(Permission.WRITE_DATA)

    def test_network_plugin(self) -> None:
        """Test network-enabled plugin permissions."""
        perms = PermissionSet()
        perms.grant(Permission.NETWORK_ACCESS)
        perms.grant(Permission.READ_DATA)

        sandbox = PluginSandbox(permissions=perms)

        sandbox.check_permission(Permission.NETWORK_ACCESS)
        sandbox.check_permission(Permission.READ_DATA)

        with pytest.raises(PermissionError):
            sandbox.check_permission(Permission.SUBPROCESS)

    def test_native_plugin(self) -> None:
        """Test native code plugin permissions."""
        perms = PermissionSet()
        perms.grant(Permission.NATIVE_CODE)
        perms.grant(Permission.SYSTEM_INFO)
        perms.grant(Permission.READ_DATA)
        perms.grant(Permission.WRITE_DATA)

        sandbox = PluginSandbox(permissions=perms)

        # All granted should pass
        sandbox.check_permission(Permission.NATIVE_CODE)
        sandbox.check_permission(Permission.SYSTEM_INFO)
        sandbox.check_permission(Permission.READ_DATA)
        sandbox.check_permission(Permission.WRITE_DATA)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestPluginsIsolationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_timeout(self) -> None:
        """Test sandbox with zero timeout."""
        sandbox = PluginSandbox()

        # Zero timeout should not cause issues
        with sandbox.execute(timeout=0.0):
            pass

    def test_negative_timeout(self) -> None:
        """Test sandbox with negative timeout."""
        sandbox = PluginSandbox()

        # Negative timeout treated as no timeout
        with sandbox.execute(timeout=-1.0):
            pass

    def test_none_timeout(self) -> None:
        """Test sandbox with None timeout."""
        sandbox = PluginSandbox()

        with sandbox.execute(timeout=None):
            pass

    def test_sandbox_reuse(self) -> None:
        """Test reusing sandbox multiple times."""
        sandbox = PluginSandbox()

        for i in range(5):
            with sandbox.execute():
                _ = i**2

    def test_nested_sandbox_execution(self) -> None:
        """Test nested sandbox execution."""
        sandbox = PluginSandbox()

        with sandbox.execute(), sandbox.execute():
            result = 42

        assert result == 42

    def test_very_small_memory_limit(self) -> None:
        """Test very small memory limit handling."""
        limits = ResourceLimits(max_memory_mb=1)  # 1 MB

        # Should create successfully even with small limit
        sandbox = PluginSandbox(limits=limits)
        assert sandbox.limits.max_memory_mb == 1

    def test_very_short_cpu_limit(self) -> None:
        """Test very short CPU time limit handling."""
        limits = ResourceLimits(max_cpu_time_sec=0.001)  # 1 ms

        sandbox = PluginSandbox(limits=limits)
        assert sandbox.limits.max_cpu_time_sec == 0.001

    def test_permission_set_iteration(self) -> None:
        """Test iterating over permission sets."""
        perms = PermissionSet()
        perms.grant(Permission.READ_CONFIG)
        perms.grant(Permission.WRITE_CONFIG)

        # Should be able to check membership
        assert Permission.READ_CONFIG in perms.allowed
        assert Permission.WRITE_CONFIG in perms.allowed


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of isolation components."""

    def test_concurrent_permission_checks(self) -> None:
        """Test concurrent permission checking."""
        perms = PermissionSet()
        perms.grant(Permission.READ_DATA)
        sandbox = PluginSandbox(permissions=perms)

        errors: list[Exception] = []

        def check_permissions() -> None:
            try:
                for _ in range(100):
                    sandbox.check_permission(Permission.READ_DATA)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_permissions) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_sandbox_creation(self) -> None:
        """Test concurrent sandbox creation."""
        manager = IsolationManager()
        sandboxes: list[PluginSandbox] = []
        lock = threading.Lock()

        def create_sandbox(idx: int) -> None:
            sandbox = manager.create_sandbox(f"thread_plugin_{idx}")
            with lock:
                sandboxes.append(sandbox)

        threads = [threading.Thread(target=create_sandbox, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(sandboxes) == 10

    def test_concurrent_grant_deny(self) -> None:
        """Test concurrent grant/deny operations."""
        perms = PermissionSet()
        errors: list[Exception] = []

        def grant_permissions() -> None:
            try:
                for _ in range(100):
                    perms.grant(Permission.READ_DATA)
            except Exception as e:
                errors.append(e)

        def deny_permissions() -> None:
            try:
                for _ in range(100):
                    perms.deny(Permission.WRITE_DATA)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=grant_permissions)
        t2 = threading.Thread(target=deny_permissions)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should not have any errors
        assert len(errors) == 0
