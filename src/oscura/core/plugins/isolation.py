"""Plugin isolation and sandboxing.

This module provides resource limits, permission models, and sandboxing
for plugins to prevent interference and ensure security.
"""

from __future__ import annotations

import logging
import resource
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Plugin permission types.

    References:
        PLUG-004: Plugin Isolation - permission model
    """

    READ_CONFIG = auto()
    """Read configuration files"""
    WRITE_CONFIG = auto()
    """Write configuration files"""
    READ_DATA = auto()
    """Read data files"""
    WRITE_DATA = auto()
    """Write data files"""
    NETWORK_ACCESS = auto()
    """Access network"""
    SUBPROCESS = auto()
    """Spawn subprocesses"""
    NATIVE_CODE = auto()
    """Execute native code"""
    SYSTEM_INFO = auto()
    """Access system information"""


@dataclass
class ResourceLimits:
    """Resource limits for plugin execution.

    Attributes:
        max_memory_mb: Maximum memory in MB (None for unlimited)
        max_cpu_time_sec: Maximum CPU time in seconds (None for unlimited)
        max_wall_time_sec: Maximum wall time in seconds (None for unlimited)
        max_file_size_mb: Maximum file size in MB (None for unlimited)
        max_open_files: Maximum open file descriptors (None for unlimited)

    References:
        PLUG-004: Plugin Isolation - resource limits
    """

    max_memory_mb: int | None = None
    max_cpu_time_sec: float | None = None
    max_wall_time_sec: float | None = None
    max_file_size_mb: int | None = None
    max_open_files: int | None = None

    def to_rlimit_dict(self) -> dict[int, tuple[int, int]]:
        """Convert to resource.setrlimit compatible dict.

        Returns:
            Dictionary of resource limits

        References:
            PLUG-004: Plugin Isolation - resource limits
        """
        limits = {}

        if self.max_memory_mb is not None:
            # RLIMIT_AS - address space limit
            limit_bytes = self.max_memory_mb * 1024 * 1024
            limits[resource.RLIMIT_AS] = (limit_bytes, limit_bytes)

        if self.max_cpu_time_sec is not None:
            # RLIMIT_CPU - CPU time limit
            limit_sec = int(self.max_cpu_time_sec)
            limits[resource.RLIMIT_CPU] = (limit_sec, limit_sec)

        if self.max_file_size_mb is not None:
            # RLIMIT_FSIZE - file size limit
            limit_bytes = self.max_file_size_mb * 1024 * 1024
            limits[resource.RLIMIT_FSIZE] = (limit_bytes, limit_bytes)

        if self.max_open_files is not None:
            # RLIMIT_NOFILE - number of open files
            limits[resource.RLIMIT_NOFILE] = (
                self.max_open_files,
                self.max_open_files,
            )

        return limits


@dataclass
class PermissionSet:
    """Set of permissions granted to a plugin.

    Attributes:
        allowed: Set of allowed permissions
        denied: Set of explicitly denied permissions

    References:
        PLUG-004: Plugin Isolation - permission model
    """

    allowed: set[Permission] = field(default_factory=set)
    denied: set[Permission] = field(default_factory=set)

    def grant(self, permission: Permission) -> None:
        """Grant a permission.

        Args:
            permission: Permission to grant

        References:
            PLUG-004: Plugin Isolation - permission model
        """
        self.allowed.add(permission)
        if permission in self.denied:
            self.denied.remove(permission)

    def deny(self, permission: Permission) -> None:
        """Deny a permission.

        Args:
            permission: Permission to deny

        References:
            PLUG-004: Plugin Isolation - permission model
        """
        self.denied.add(permission)
        if permission in self.allowed:
            self.allowed.remove(permission)

    def has_permission(self, permission: Permission) -> bool:
        """Check if permission is granted.

        Args:
            permission: Permission to check

        Returns:
            True if permission is granted

        References:
            PLUG-004: Plugin Isolation - permission model
        """
        if permission in self.denied:
            return False
        return permission in self.allowed

    def check(self, permission: Permission) -> None:
        """Check permission and raise if not granted.

        Args:
            permission: Permission to check

        Raises:
            PermissionError: If permission not granted

        References:
            PLUG-004: Plugin Isolation - permission model
        """
        if not self.has_permission(permission):
            raise PermissionError(f"Plugin does not have {permission.name} permission")


class TimeoutError(Exception):
    """Exception raised when execution timeout is exceeded."""


class ResourceExceededError(Exception):
    """Exception raised when resource limit is exceeded."""


class PluginSandbox:
    """Sandbox for isolated plugin execution.

    Provides resource limits, timeout enforcement, and permission checking.

    References:
        PLUG-004: Plugin Isolation - resource limits, permission model
    """

    def __init__(
        self,
        permissions: PermissionSet | None = None,
        limits: ResourceLimits | None = None,
    ) -> None:
        """Initialize sandbox.

        Args:
            permissions: Permission set for plugin
            limits: Resource limits for plugin

        References:
            PLUG-004: Plugin Isolation
        """
        self.permissions = permissions or PermissionSet()
        self.limits = limits or ResourceLimits()
        self._original_limits: dict[int, tuple[int, int]] = {}

    @contextmanager
    def execute(
        self,
        timeout: float | None = None,
    ) -> Generator[None, None, None]:
        """Execute code in sandbox with resource limits.

        Args:
            timeout: Execution timeout in seconds

        Yields:
            None

        Raises:
            ResourceExceededError: If resource limit exceeded

        Note:
            Timeout handling is logged but does not raise TimeoutError.
            Future versions will add proper timeout interruption.

        References:
            PLUG-004: Plugin Isolation - resource limits

        Example:
            >>> sandbox = PluginSandbox(limits=ResourceLimits(max_memory_mb=100))
            >>> with sandbox.execute(timeout=5.0):
            ...     # Plugin code runs here with limits
            ...     result = plugin.process_data(data)
        """
        # Apply resource limits
        self._apply_limits()

        # Setup timeout if specified
        timer = None
        if timeout is not None:
            try:
                timer = threading.Timer(timeout, self._timeout_handler)
                timer.start()
            except RuntimeError:
                # Thread limit reached, skip timeout
                logger.warning("Could not create timeout timer (thread limit reached)")
                timer = None

        try:
            yield

        except MemoryError as e:
            raise ResourceExceededError("Memory limit exceeded") from e

        finally:
            # Cancel timeout timer
            if timer is not None:
                with suppress(Exception):
                    timer.cancel()

            # Restore original limits
            self._restore_limits()

    def _apply_limits(self) -> None:
        """Apply resource limits.

        References:
            PLUG-004: Plugin Isolation - resource limits
        """
        rlimits = self.limits.to_rlimit_dict()

        for resource_type, limit in rlimits.items():
            try:
                # Save original limit
                self._original_limits[resource_type] = resource.getrlimit(resource_type)
                # Apply new limit
                resource.setrlimit(resource_type, limit)
                logger.debug(f"Applied resource limit: {resource_type} = {limit}")

            except (OSError, ValueError) as e:
                logger.warning(f"Failed to set resource limit {resource_type}: {e}")

    def _restore_limits(self) -> None:
        """Restore original resource limits.

        References:
            PLUG-004: Plugin Isolation - resource limits
        """
        for resource_type, limit in self._original_limits.items():
            try:
                resource.setrlimit(resource_type, limit)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to restore resource limit {resource_type}: {e}")

        self._original_limits.clear()

    def _timeout_handler(self) -> None:
        """Handle execution timeout.

        References:
            PLUG-004: Plugin Isolation - CPU time limit
        """
        logger.error("Plugin execution timeout exceeded")
        # In a real implementation, this would interrupt the thread
        # For now, we just log the error

    def check_permission(self, permission: Permission) -> None:
        """Check if plugin has permission.

        Args:
            permission: Permission to check

        References:
            PLUG-004: Plugin Isolation - permission model
        """
        self.permissions.check(permission)


class IsolationManager:
    """Manager for plugin isolation and sandboxing.

    Tracks sandboxes for all plugins and enforces isolation policies.

    References:
        PLUG-004: Plugin Isolation
    """

    def __init__(self) -> None:
        """Initialize isolation manager."""
        self._sandboxes: dict[str, PluginSandbox] = {}
        self._default_limits = ResourceLimits(
            max_memory_mb=512,  # 512 MB default
            max_cpu_time_sec=30.0,  # 30 seconds default
        )

    def create_sandbox(
        self,
        plugin_name: str,
        permissions: PermissionSet | None = None,
        limits: ResourceLimits | None = None,
    ) -> PluginSandbox:
        """Create sandbox for a plugin.

        Args:
            plugin_name: Plugin name
            permissions: Permission set (None for default)
            limits: Resource limits (None for default)

        Returns:
            Plugin sandbox

        References:
            PLUG-004: Plugin Isolation
        """
        if limits is None:
            limits = self._default_limits

        sandbox = PluginSandbox(permissions=permissions, limits=limits)
        self._sandboxes[plugin_name] = sandbox

        logger.info(f"Created sandbox for plugin '{plugin_name}'")
        return sandbox

    def get_sandbox(self, plugin_name: str) -> PluginSandbox | None:
        """Get sandbox for a plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin sandbox or None
        """
        return self._sandboxes.get(plugin_name)

    def remove_sandbox(self, plugin_name: str) -> None:
        """Remove sandbox for a plugin.

        Args:
            plugin_name: Plugin name
        """
        if plugin_name in self._sandboxes:
            del self._sandboxes[plugin_name]
            logger.info(f"Removed sandbox for plugin '{plugin_name}'")


# Global isolation manager
_isolation_manager: IsolationManager | None = None


def get_isolation_manager() -> IsolationManager:
    """Get global isolation manager.

    Returns:
        Global IsolationManager instance
    """
    global _isolation_manager
    if _isolation_manager is None:
        _isolation_manager = IsolationManager()
    return _isolation_manager


__all__ = [
    "IsolationManager",
    "Permission",
    "PermissionSet",
    "PluginSandbox",
    "ResourceExceededError",
    "ResourceLimits",
    "TimeoutError",
    "get_isolation_manager",
]
