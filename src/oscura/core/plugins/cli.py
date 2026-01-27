"""Plugin management CLI.

This module provides command-line interface for plugin management including
list, info, enable/disable, install, and validate operations.
"""

import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from oscura.core.plugins.discovery import discover_plugins, get_plugin_paths
from oscura.core.plugins.lifecycle import get_lifecycle_manager
from oscura.core.plugins.registry import get_plugin_registry

logger = logging.getLogger(__name__)


class PluginInstaller:
    """Plugin installer with integrity validation.

    Supports installing plugins from:
    - Repository URLs (git)
    - Archive files (.tar.gz, .zip)
    - Local directories

    References:
        PLUG-007: Plugin CLI - install from repository URL, integrity validation
    """

    def __init__(self, install_dir: Path | None = None) -> None:
        """Initialize installer.

        Args:
            install_dir: Directory to install plugins (defaults to user plugins)
        """
        if install_dir is None:
            paths = get_plugin_paths()
            # Use first user-writable path
            install_dir = paths[0] if paths else Path.home() / ".oscura" / "plugins"

        self.install_dir = install_dir
        self.install_dir.mkdir(parents=True, exist_ok=True)

    def install_from_url(
        self,
        url: str,
        *,
        checksum: str | None = None,
        checksum_algo: str = "sha256",
    ) -> Path:
        """Install plugin from URL.

        Args:
            url: Plugin repository URL or archive URL
            checksum: Expected checksum for integrity validation
            checksum_algo: Hash algorithm (sha256, sha512, md5)

        Returns:
            Path to installed plugin

        Raises:
            ValueError: If checksum verification fails or unsupported URL type.

        References:
            PLUG-007: Plugin CLI - install from repository URL, integrity validation

        Example:
            >>> installer = PluginInstaller()
            >>> path = installer.install_from_url(
            ...     "https://github.com/user/plugin.git",
            ...     checksum="abc123...",
            ... )
        """
        logger.info(f"Installing plugin from URL: {url}")

        parsed = urlparse(url)

        # Determine source type
        if parsed.path.endswith(".git") or "github.com" in parsed.netloc:
            return self._install_from_git(url, checksum, checksum_algo)
        elif parsed.path.endswith((".tar.gz", ".zip")):
            return self._install_from_archive(url, checksum, checksum_algo)
        else:
            raise ValueError(f"Unsupported URL type: {url}")

    def _install_from_git(
        self,
        url: str,
        checksum: str | None,
        checksum_algo: str,
    ) -> Path:
        """Install plugin from git repository.

        Args:
            url: Git repository URL
            checksum: Expected checksum
            checksum_algo: Hash algorithm

        Returns:
            Path to installed plugin

        Raises:
            RuntimeError: If git clone fails.
            ValueError: If checksum verification fails.

        References:
            PLUG-007: Plugin CLI - install from repository URL
        """
        # Extract plugin name from URL
        plugin_name = Path(urlparse(url).path).stem
        if plugin_name.endswith(".git"):
            plugin_name = plugin_name[:-4]

        target_dir = self.install_dir / plugin_name

        # Clone repository to temp directory first
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / plugin_name

            try:
                subprocess.run(
                    ["git", "clone", url, str(temp_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Git clone failed: {e.stderr}") from e

            # Validate checksum if provided
            if checksum:
                actual = self._compute_directory_checksum(temp_path, checksum_algo)
                if actual != checksum:
                    raise ValueError(f"Checksum mismatch: expected {checksum}, got {actual}")

            # Move to final location
            if target_dir.exists():
                logger.warning(f"Removing existing plugin at {target_dir}")
                shutil.rmtree(target_dir)

            shutil.copytree(temp_path, target_dir)

        logger.info(f"Installed plugin '{plugin_name}' to {target_dir}")
        return target_dir

    def _install_from_archive(
        self,
        url: str,
        checksum: str | None,
        checksum_algo: str,
    ) -> Path:
        """Install plugin from archive file.

        Args:
            url: Archive URL
            checksum: Expected checksum
            checksum_algo: Hash algorithm

        Returns:
            Path to installed plugin

        Raises:
            RuntimeError: If download fails.
            ValueError: If checksum verification fails or archive format is invalid.

        References:
            PLUG-007: Plugin CLI - integrity validation
        """
        import urllib.request

        # Download archive
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "plugin.archive"

            try:
                urllib.request.urlretrieve(url, archive_path)
            except Exception as e:
                raise RuntimeError(f"Download failed: {e}") from e

            # Validate checksum
            if checksum:
                actual = self._compute_file_checksum(archive_path, checksum_algo)
                if actual != checksum:
                    raise ValueError(f"Checksum mismatch: expected {checksum}, got {actual}")

            # Extract archive
            extract_dir = Path(temp_dir) / "extracted"
            shutil.unpack_archive(archive_path, extract_dir)

            # Find plugin directory (should be single top-level dir)
            plugin_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if len(plugin_dirs) != 1:
                raise ValueError(
                    f"Archive should contain single plugin directory, found {len(plugin_dirs)}"
                )

            plugin_dir = plugin_dirs[0]
            plugin_name = plugin_dir.name

            target_dir = self.install_dir / plugin_name

            # Move to final location
            if target_dir.exists():
                logger.warning(f"Removing existing plugin at {target_dir}")
                shutil.rmtree(target_dir)

            shutil.copytree(plugin_dir, target_dir)

        logger.info(f"Installed plugin '{plugin_name}' to {target_dir}")
        return target_dir

    def _compute_file_checksum(self, path: Path, algo: str) -> str:
        """Compute checksum of a file.

        Args:
            path: File path
            algo: Hash algorithm

        Returns:
            Hexadecimal checksum

        References:
            PLUG-007: Plugin CLI - integrity validation (checksum verification)
        """
        hasher = hashlib.new(algo)

        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _compute_directory_checksum(self, path: Path, algo: str) -> str:
        """Compute checksum of a directory (all files).

        Args:
            path: Directory path
            algo: Hash algorithm

        Returns:
            Hexadecimal checksum

        References:
            PLUG-007: Plugin CLI - integrity validation (checksum verification)
        """
        hasher = hashlib.new(algo)

        # Sort files for consistent ordering
        files = sorted(path.rglob("*"))

        for file_path in files:
            if file_path.is_file():
                # Include relative path in hash
                rel_path = file_path.relative_to(path)
                hasher.update(str(rel_path).encode())

                # Include file content
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def validate_integrity(
        self,
        plugin_path: Path,
        expected_checksum: str,
        algo: str = "sha256",
    ) -> bool:
        """Validate plugin integrity.

        Args:
            plugin_path: Path to plugin
            expected_checksum: Expected checksum
            algo: Hash algorithm

        Returns:
            True if checksum matches

        References:
            PLUG-007: Plugin CLI - integrity validation (checksum verification)
        """
        if plugin_path.is_file():
            actual = self._compute_file_checksum(plugin_path, algo)
        else:
            actual = self._compute_directory_checksum(plugin_path, algo)

        return actual == expected_checksum


def cli_list_plugins() -> None:
    """List all available plugins (CLI command).

    References:
        PLUG-007: Plugin CLI
    """
    plugins = discover_plugins(compatible_only=False)

    if not plugins:
        print("No plugins found")
        return

    print(f"Found {len(plugins)} plugins:\n")

    for plugin in plugins:
        status = "enabled" if plugin.metadata.enabled else "disabled"
        compat = "compatible" if plugin.compatible else "incompatible"

        print(f"  {plugin.metadata.name} v{plugin.metadata.version} [{status}]")
        if plugin.path:
            print(f"    Path: {plugin.path}")
        print(f"    API: {plugin.metadata.api_version} ({compat})")

        if plugin.metadata.provides:
            provides: list[str] = []
            for key, values in plugin.metadata.provides.items():
                provides.extend(f"{key}:{v}" for v in values)
            print(f"    Provides: {', '.join(provides)}")

        if plugin.load_error:
            print(f"    Error: {plugin.load_error}")

        print()


def cli_plugin_info(name: str) -> None:
    """Show detailed plugin information (CLI command).

    Args:
        name: Plugin name

    References:
        PLUG-007: Plugin CLI
    """
    registry = get_plugin_registry()
    metadata = registry.get_metadata(name)

    if metadata is None:
        print(f"Plugin '{name}' not found")
        sys.exit(1)

    print(f"Name: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"API Version: {metadata.api_version}")

    if metadata.author:
        print(f"Author: {metadata.author}")

    if metadata.description:
        print(f"Description: {metadata.description}")

    if metadata.homepage:
        print(f"Homepage: {metadata.homepage}")

    if metadata.license:
        print(f"License: {metadata.license}")

    if metadata.path:
        print(f"Path: {metadata.path}")

    print(f"Status: {'enabled' if metadata.enabled else 'disabled'}")

    if metadata.dependencies:
        print("\nDependencies:")
        for dep, version in metadata.dependencies.items():
            print(f"  - {dep} {version}")

    if metadata.provides:
        print("\nProvides:")
        for key, values in metadata.provides.items():
            for value in values:
                print(f"  - {key}: {value}")


def cli_enable_plugin(name: str) -> None:
    """Enable a plugin (CLI command).

    Args:
        name: Plugin name

    References:
        PLUG-007: Plugin CLI
    """
    manager = get_lifecycle_manager()

    try:
        manager.enable_plugin(name)
        print(f"Plugin '{name}' enabled")
    except Exception as e:
        print(f"Failed to enable plugin: {e}")
        sys.exit(1)


def cli_disable_plugin(name: str) -> None:
    """Disable a plugin (CLI command).

    Args:
        name: Plugin name

    References:
        PLUG-007: Plugin CLI
    """
    manager = get_lifecycle_manager()

    try:
        manager.disable_plugin(name)
        print(f"Plugin '{name}' disabled")
    except Exception as e:
        print(f"Failed to disable plugin: {e}")
        sys.exit(1)


def cli_validate_plugin(name: str) -> None:
    """Validate a plugin (CLI command).

    Args:
        name: Plugin name

    References:
        PLUG-007: Plugin CLI - integrity validation
    """
    plugins = discover_plugins(compatible_only=False)
    plugin = next((p for p in plugins if p.metadata.name == name), None)

    if plugin is None:
        print(f"Plugin '{name}' not found")
        sys.exit(1)

    print(f"Validating {name}...")

    # Check metadata
    if not plugin.metadata.name:
        print("  ✗ Missing name")
        sys.exit(1)
    print("  ✓ Metadata valid")

    # Check dependencies
    if plugin.metadata.dependencies:
        print(f"  ✓ Dependencies declared: {len(plugin.metadata.dependencies)}")
    else:
        print("  ✓ No dependencies")

    # Check API compatibility
    if plugin.compatible:
        print("  ✓ API version compatible")
    else:
        print(f"  ✗ API version incompatible: {plugin.metadata.api_version}")
        sys.exit(1)

    # Check for errors
    if plugin.load_error:
        print(f"  ✗ Load error: {plugin.load_error}")
        sys.exit(1)

    print("\nPlugin is valid")


def cli_install_plugin(
    url: str,
    *,
    checksum: str | None = None,
) -> None:
    """Install a plugin from URL (CLI command).

    Args:
        url: Plugin repository or archive URL
        checksum: Expected checksum for validation

    References:
        PLUG-007: Plugin CLI - install from repository URL, integrity validation
    """
    installer = PluginInstaller()

    try:
        path = installer.install_from_url(url, checksum=checksum)
        print(f"Successfully installed plugin to {path}")
    except Exception as e:
        print(f"Installation failed: {e}")
        sys.exit(1)


__all__ = [
    "PluginInstaller",
    "cli_disable_plugin",
    "cli_enable_plugin",
    "cli_install_plugin",
    "cli_list_plugins",
    "cli_plugin_info",
    "cli_validate_plugin",
]
