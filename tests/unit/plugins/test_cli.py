"""Comprehensive unit tests for plugin CLI.

This module provides comprehensive tests for plugin CLI functionality including
plugin installation, validation, and command-line interface operations.

Requirements tested:
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.plugins.base import PluginBase, PluginCapability
from oscura.plugins.cli import (
    PluginInstaller,
    cli_disable_plugin,
    cli_enable_plugin,
    cli_install_plugin,
    cli_list_plugins,
    cli_plugin_info,
    cli_validate_plugin,
)
from oscura.plugins.discovery import DiscoveredPlugin

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@contextmanager
def mock_tempdir_context(tmpdir):
    """Create a context manager for mocking TemporaryDirectory."""
    yield tmpdir


class MockPlugin(PluginBase):
    """Mock plugin for testing."""

    name = "mock_plugin"
    version = "1.0.0"
    api_version = "1.0.0"
    author = "Test Author"
    description = "Mock plugin for testing"
    capabilities = [PluginCapability.PROTOCOL_DECODER]


@pytest.fixture
def temp_install_dir():
    """Create temporary installation directory."""
    with tempfile.TemporaryDirectory() as real_tmpdir:
        yield Path(real_tmpdir)


@pytest.fixture
def installer(temp_install_dir):
    """Create plugin installer with temp directory."""
    return PluginInstaller(install_dir=temp_install_dir)


@pytest.fixture
def mock_git_repo(temp_install_dir):
    """Create a mock git repository structure."""
    repo_dir = temp_install_dir / "test_plugin"
    repo_dir.mkdir()
    (repo_dir / "__init__.py").write_text("# Plugin code")
    return repo_dir


# =============================================================================
# PluginInstaller Tests
# =============================================================================


def test_plugin_installer_creation_with_dir(temp_install_dir):
    """Test PluginInstaller creation with custom directory."""
    installer = PluginInstaller(install_dir=temp_install_dir)
    assert installer.install_dir == temp_install_dir
    assert temp_install_dir.exists()


def test_plugin_installer_creation_default():
    """Test PluginInstaller creation with default directory."""
    with patch("oscura.plugins.cli.get_plugin_paths", return_value=[Path("/tmp/plugins")]):
        installer = PluginInstaller()
        assert installer.install_dir == Path("/tmp/plugins")


def test_plugin_installer_creates_directory(temp_install_dir):
    """Test that PluginInstaller creates install directory."""
    install_dir = temp_install_dir / "new_dir"
    installer = PluginInstaller(install_dir=install_dir)
    assert install_dir.exists()


# =============================================================================
# Install from URL Tests
# =============================================================================


def test_install_from_url_git(installer):
    """Test installing from git URL."""
    url = "https://github.com/user/plugin.git"

    with patch.object(installer, "_install_from_git", return_value=Path("/test")) as mock_git:
        result = installer.install_from_url(url)
        mock_git.assert_called_once_with(url, None, "sha256")
        assert result == Path("/test")


def test_install_from_url_archive_tar_gz(installer):
    """Test installing from .tar.gz URL."""
    url = "https://example.com/plugin.tar.gz"

    with patch.object(
        installer, "_install_from_archive", return_value=Path("/test")
    ) as mock_archive:
        result = installer.install_from_url(url)
        mock_archive.assert_called_once_with(url, None, "sha256")
        assert result == Path("/test")


def test_install_from_url_archive_zip(installer):
    """Test installing from .zip URL."""
    url = "https://example.com/plugin.zip"

    with patch.object(
        installer, "_install_from_archive", return_value=Path("/test")
    ) as mock_archive:
        result = installer.install_from_url(url)
        mock_archive.assert_called_once()


def test_install_from_url_with_checksum(installer):
    """Test installing with checksum validation."""
    url = "https://github.com/user/plugin.git"
    checksum = "abc123"

    with patch.object(installer, "_install_from_git", return_value=Path("/test")) as mock_git:
        installer.install_from_url(url, checksum=checksum, checksum_algo="sha512")
        mock_git.assert_called_once_with(url, "abc123", "sha512")


def test_install_from_url_unsupported_type(installer):
    """Test installing from unsupported URL type."""
    url = "https://example.com/plugin.exe"

    with pytest.raises(ValueError, match="Unsupported URL type"):
        installer.install_from_url(url)


# =============================================================================
# Install from Git Tests
# =============================================================================


def test_install_from_git_success(installer, temp_install_dir):
    """Test successful git installation."""
    url = "https://github.com/user/test_plugin.git"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        with tempfile.TemporaryDirectory() as real_tmpdir:
            # Create fake plugin directory
            plugin_dir = Path(real_tmpdir) / "test_plugin"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "__init__.py").write_text("# Plugin")

            # Mock TemporaryDirectory to return our pre-created directory
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                result = installer._install_from_git(url, None, "sha256")

                assert result == temp_install_dir / "test_plugin"


def test_install_from_git_clone_failure(installer):
    """Test git clone failure handling."""
    url = "https://github.com/user/plugin.git"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr="error")

        with pytest.raises(RuntimeError, match="Git clone failed"):
            installer._install_from_git(url, None, "sha256")


def test_install_from_git_with_checksum(installer, temp_install_dir):
    """Test git installation with checksum validation."""
    url = "https://github.com/user/test_plugin.git"
    checksum = "abc123"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        with tempfile.TemporaryDirectory() as real_tmpdir:
            plugin_dir = Path(real_tmpdir) / "test_plugin"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "__init__.py").write_text("# Plugin")

            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                with patch.object(installer, "_compute_directory_checksum", return_value=checksum):
                    result = installer._install_from_git(url, checksum, "sha256")
                    assert result.exists()


def test_install_from_git_checksum_mismatch(installer):
    """Test git installation with checksum mismatch."""
    url = "https://github.com/user/test_plugin.git"
    expected_checksum = "abc123"
    actual_checksum = "def456"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        with tempfile.TemporaryDirectory() as real_tmpdir:
            plugin_dir = Path(real_tmpdir) / "test_plugin"
            plugin_dir.mkdir(parents=True)

            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                with patch.object(
                    installer, "_compute_directory_checksum", return_value=actual_checksum
                ):
                    with pytest.raises(ValueError, match="Checksum mismatch"):
                        installer._install_from_git(url, expected_checksum, "sha256")


def test_install_from_git_removes_existing(installer, temp_install_dir):
    """Test git installation removes existing plugin."""
    url = "https://github.com/user/test_plugin.git"

    # Create existing plugin directory
    existing_dir = temp_install_dir / "test_plugin"
    existing_dir.mkdir()
    (existing_dir / "old.txt").write_text("old")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        with tempfile.TemporaryDirectory() as real_tmpdir:
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                plugin_dir = Path(real_tmpdir) / "test_plugin"
                plugin_dir.mkdir(parents=True)
                (plugin_dir / "__init__.py").write_text("# Plugin")

                result = installer._install_from_git(url, None, "sha256")

                # Old file should be gone
                assert not (result / "old.txt").exists()


def test_install_from_git_name_extraction(installer):
    """Test plugin name extraction from git URL."""
    test_cases = [
        ("https://github.com/user/plugin.git", "plugin"),
        ("https://github.com/user/my_plugin.git", "my_plugin"),
        ("git@github.com:user/awesome.git", "awesome"),
    ]

    for url, expected_name in test_cases:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with tempfile.TemporaryDirectory() as real_tmpdir:
                with patch(
                    "oscura.plugins.cli.tempfile.TemporaryDirectory",
                    lambda: mock_tempdir_context(real_tmpdir),
                ):
                    plugin_dir = Path(real_tmpdir) / expected_name
                    plugin_dir.mkdir(parents=True)
                    (plugin_dir / "__init__.py").write_text("# Plugin")

                    result = installer._install_from_git(url, None, "sha256")
                    assert result.name == expected_name


# =============================================================================
# Install from Archive Tests
# =============================================================================


def test_install_from_archive_success(installer, temp_install_dir):
    """Test successful archive installation."""
    url = "https://example.com/plugin.tar.gz"

    with patch("urllib.request.urlretrieve") as mock_download:
        with tempfile.TemporaryDirectory() as real_tmpdir:
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                # Create fake archive structure
                extract_dir = Path(real_tmpdir) / "extracted"
                extract_dir.mkdir()
                plugin_dir = extract_dir / "my_plugin"
                plugin_dir.mkdir()
                (plugin_dir / "__init__.py").write_text("# Plugin")

                with patch("shutil.unpack_archive"):
                    result = installer._install_from_archive(url, None, "sha256")

                    assert result == temp_install_dir / "my_plugin"


def test_install_from_archive_download_failure(installer):
    """Test archive download failure."""
    url = "https://example.com/plugin.tar.gz"

    with patch("urllib.request.urlretrieve", side_effect=RuntimeError("Download failed")):
        with pytest.raises(RuntimeError, match="Download failed"):
            installer._install_from_archive(url, None, "sha256")


def test_install_from_archive_with_checksum(installer, temp_install_dir):
    """Test archive installation with checksum validation."""
    url = "https://example.com/plugin.tar.gz"
    checksum = "abc123"

    with patch("urllib.request.urlretrieve"):
        with tempfile.TemporaryDirectory() as real_tmpdir:
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                extract_dir = Path(real_tmpdir) / "extracted"
                extract_dir.mkdir()
                plugin_dir = extract_dir / "my_plugin"
                plugin_dir.mkdir()

                with patch("shutil.unpack_archive"):
                    with patch.object(installer, "_compute_file_checksum", return_value=checksum):
                        result = installer._install_from_archive(url, checksum, "sha256")
                        assert result.exists()


def test_install_from_archive_checksum_mismatch(installer):
    """Test archive installation with checksum mismatch."""
    url = "https://example.com/plugin.tar.gz"

    with patch("urllib.request.urlretrieve"):
        with tempfile.TemporaryDirectory() as real_tmpdir:
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                with patch.object(installer, "_compute_file_checksum", return_value="wrong"):
                    with pytest.raises(ValueError, match="Checksum mismatch"):
                        installer._install_from_archive(url, "expected", "sha256")


def test_install_from_archive_multiple_directories(installer):
    """Test archive with multiple top-level directories."""
    url = "https://example.com/plugin.tar.gz"

    with patch("urllib.request.urlretrieve"):
        with tempfile.TemporaryDirectory() as real_tmpdir:
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                extract_dir = Path(real_tmpdir) / "extracted"
                extract_dir.mkdir()
                (extract_dir / "dir1").mkdir()
                (extract_dir / "dir2").mkdir()

                with patch("shutil.unpack_archive"):
                    with pytest.raises(ValueError, match="should contain single plugin directory"):
                        installer._install_from_archive(url, None, "sha256")


def test_install_from_archive_removes_existing(installer, temp_install_dir):
    """Test archive installation removes existing plugin."""
    url = "https://example.com/plugin.tar.gz"

    # Create existing plugin
    existing_dir = temp_install_dir / "my_plugin"
    existing_dir.mkdir()
    (existing_dir / "old.txt").write_text("old")

    with patch("urllib.request.urlretrieve"):
        with tempfile.TemporaryDirectory() as real_tmpdir:
            with patch(
                "oscura.plugins.cli.tempfile.TemporaryDirectory",
                lambda: mock_tempdir_context(real_tmpdir),
            ):
                extract_dir = Path(real_tmpdir) / "extracted"
                extract_dir.mkdir()
                plugin_dir = extract_dir / "my_plugin"
                plugin_dir.mkdir()
                (plugin_dir / "__init__.py").write_text("# Plugin")

                with patch("shutil.unpack_archive"):
                    result = installer._install_from_archive(url, None, "sha256")

                    # Old file should be gone
                    assert not (result / "old.txt").exists()


# =============================================================================
# Checksum Computation Tests
# =============================================================================


def test_compute_file_checksum_sha256(installer, temp_install_dir):
    """Test computing SHA256 checksum of file."""
    test_file = temp_install_dir / "test.txt"
    test_file.write_text("test content")

    checksum = installer._compute_file_checksum(test_file, "sha256")

    # Verify it's a valid hex string
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum)


def test_compute_file_checksum_md5(installer, temp_install_dir):
    """Test computing MD5 checksum of file."""
    test_file = temp_install_dir / "test.txt"
    test_file.write_text("test content")

    checksum = installer._compute_file_checksum(test_file, "md5")

    # MD5 is 32 hex characters
    assert len(checksum) == 32


def test_compute_file_checksum_sha512(installer, temp_install_dir):
    """Test computing SHA512 checksum of file."""
    test_file = temp_install_dir / "test.txt"
    test_file.write_text("test content")

    checksum = installer._compute_file_checksum(test_file, "sha512")

    # SHA512 is 128 hex characters
    assert len(checksum) == 128


def test_compute_file_checksum_large_file(installer, temp_install_dir):
    """Test computing checksum of large file (chunked reading)."""
    test_file = temp_install_dir / "large.bin"
    # Create file larger than chunk size (8192)
    test_file.write_bytes(b"x" * 100000)

    checksum = installer._compute_file_checksum(test_file, "sha256")

    assert len(checksum) == 64


def test_compute_directory_checksum(installer, temp_install_dir):
    """Test computing checksum of directory."""
    plugin_dir = temp_install_dir / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "file1.py").write_text("content1")
    (plugin_dir / "file2.py").write_text("content2")
    subdir = plugin_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.py").write_text("content3")

    checksum = installer._compute_directory_checksum(plugin_dir, "sha256")

    assert len(checksum) == 64


def test_compute_directory_checksum_consistent(installer, temp_install_dir):
    """Test directory checksum is consistent."""
    plugin_dir = temp_install_dir / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "file1.py").write_text("content1")
    (plugin_dir / "file2.py").write_text("content2")

    checksum1 = installer._compute_directory_checksum(plugin_dir, "sha256")
    checksum2 = installer._compute_directory_checksum(plugin_dir, "sha256")

    assert checksum1 == checksum2


def test_compute_directory_checksum_order_independent(installer, temp_install_dir):
    """Test directory checksum handles file order correctly."""
    # Create two identical directories with files created in different order
    dir1 = temp_install_dir / "dir1"
    dir1.mkdir()
    (dir1 / "a.py").write_text("a")
    (dir1 / "b.py").write_text("b")

    dir2 = temp_install_dir / "dir2"
    dir2.mkdir()
    (dir2 / "b.py").write_text("b")
    (dir2 / "a.py").write_text("a")

    checksum1 = installer._compute_directory_checksum(dir1, "sha256")
    checksum2 = installer._compute_directory_checksum(dir2, "sha256")

    # Should be the same since files are sorted
    assert checksum1 == checksum2


def test_compute_directory_checksum_includes_paths(installer, temp_install_dir):
    """Test directory checksum includes file paths."""
    dir1 = temp_install_dir / "dir1"
    dir1.mkdir()
    (dir1 / "a.py").write_text("same content")

    dir2 = temp_install_dir / "dir2"
    dir2.mkdir()
    (dir2 / "b.py").write_text("same content")

    checksum1 = installer._compute_directory_checksum(dir1, "sha256")
    checksum2 = installer._compute_directory_checksum(dir2, "sha256")

    # Should be different due to different file names
    assert checksum1 != checksum2


# =============================================================================
# Validate Integrity Tests
# =============================================================================


def test_validate_integrity_file_success(installer, temp_install_dir):
    """Test validating file integrity successfully."""
    test_file = temp_install_dir / "test.txt"
    test_file.write_text("test content")

    expected = installer._compute_file_checksum(test_file, "sha256")
    result = installer.validate_integrity(test_file, expected, "sha256")

    assert result is True


def test_validate_integrity_file_failure(installer, temp_install_dir):
    """Test validating file integrity with mismatch."""
    test_file = temp_install_dir / "test.txt"
    test_file.write_text("test content")

    result = installer.validate_integrity(test_file, "wrong_checksum", "sha256")

    assert result is False


def test_validate_integrity_directory_success(installer, temp_install_dir):
    """Test validating directory integrity successfully."""
    plugin_dir = temp_install_dir / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "file1.py").write_text("content")

    expected = installer._compute_directory_checksum(plugin_dir, "sha256")
    result = installer.validate_integrity(plugin_dir, expected, "sha256")

    assert result is True


def test_validate_integrity_directory_failure(installer, temp_install_dir):
    """Test validating directory integrity with mismatch."""
    plugin_dir = temp_install_dir / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "file1.py").write_text("content")

    result = installer.validate_integrity(plugin_dir, "wrong_checksum", "sha256")

    assert result is False


# =============================================================================
# CLI List Plugins Tests
# =============================================================================


def test_cli_list_plugins_empty():
    """Test listing plugins when none are found."""
    with patch("oscura.plugins.cli.discover_plugins", return_value=[]):
        with patch("builtins.print") as mock_print:
            cli_list_plugins()
            mock_print.assert_called_once_with("No plugins found")


def test_cli_list_plugins_single():
    """Test listing single plugin."""
    plugin = MockPlugin()
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        path=Path("/test"),
        compatible=True,
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_list_plugins()

            # Check that plugin info was printed
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("mock_plugin" in str(c) for c in calls)


def test_cli_list_plugins_multiple():
    """Test listing multiple plugins."""
    plugin1 = MockPlugin()
    plugin1.name = "plugin1"

    plugin2 = MockPlugin()
    plugin2.name = "plugin2"

    discovered = [
        DiscoveredPlugin(metadata=plugin1.metadata, compatible=True),
        DiscoveredPlugin(metadata=plugin2.metadata, compatible=True),
    ]

    with patch("oscura.plugins.cli.discover_plugins", return_value=discovered):
        with patch("builtins.print") as mock_print:
            cli_list_plugins()

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("plugin1" in str(c) for c in calls)
            assert any("plugin2" in str(c) for c in calls)


def test_cli_list_plugins_with_provides():
    """Test listing plugins with provides information."""
    plugin = MockPlugin()
    plugin.metadata.provides = {"protocols": ["uart", "spi"]}

    discovered = DiscoveredPlugin(metadata=plugin.metadata, compatible=True)

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_list_plugins()

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("protocols" in str(c) for c in calls)


def test_cli_list_plugins_with_error():
    """Test listing plugin with load error."""
    plugin = MockPlugin()
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=True,
        load_error="Failed to load",
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_list_plugins()

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Error" in str(c) for c in calls)


def test_cli_list_plugins_incompatible():
    """Test listing incompatible plugin."""
    plugin = MockPlugin()
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=False,
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_list_plugins()

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("incompatible" in str(c) for c in calls)


# =============================================================================
# CLI Plugin Info Tests
# =============================================================================


def test_cli_plugin_info_success():
    """Test showing plugin info successfully."""
    plugin = MockPlugin()
    plugin.metadata.author = "Test Author"
    plugin.metadata.homepage = "https://example.com"
    plugin.metadata.license = "MIT"

    with patch("oscura.plugins.cli.get_plugin_registry") as mock_registry:
        mock_registry.return_value.get_metadata.return_value = plugin.metadata

        with patch("builtins.print") as mock_print:
            cli_plugin_info("mock_plugin")

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("mock_plugin" in str(c) for c in calls)
            assert any("Test Author" in str(c) for c in calls)


def test_cli_plugin_info_not_found():
    """Test showing info for non-existent plugin."""
    with patch("oscura.plugins.cli.get_plugin_registry") as mock_registry:
        mock_registry.return_value.get_metadata.return_value = None

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_plugin_info("nonexistent")

            assert exc_info.value.code == 1
            mock_print.assert_called_with("Plugin 'nonexistent' not found")


def test_cli_plugin_info_with_dependencies():
    """Test showing plugin info with dependencies."""
    plugin = MockPlugin()
    plugin.metadata.dependencies = {"dep1": ">=1.0.0", "dep2": ">=2.0.0"}

    with patch("oscura.plugins.cli.get_plugin_registry") as mock_registry:
        mock_registry.return_value.get_metadata.return_value = plugin.metadata

        with patch("builtins.print") as mock_print:
            cli_plugin_info("mock_plugin")

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Dependencies" in str(c) for c in calls)


def test_cli_plugin_info_with_provides():
    """Test showing plugin info with provides."""
    plugin = MockPlugin()
    plugin.metadata.provides = {"protocols": ["uart"], "algorithms": ["edge_detect"]}

    with patch("oscura.plugins.cli.get_plugin_registry") as mock_registry:
        mock_registry.return_value.get_metadata.return_value = plugin.metadata

        with patch("builtins.print") as mock_print:
            cli_plugin_info("mock_plugin")

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Provides" in str(c) for c in calls)


# =============================================================================
# CLI Enable Plugin Tests
# =============================================================================


def test_cli_enable_plugin_success():
    """Test enabling plugin via CLI."""
    with patch("oscura.plugins.cli.get_lifecycle_manager") as mock_manager:
        with patch("builtins.print") as mock_print:
            cli_enable_plugin("test_plugin")

            mock_manager.return_value.enable_plugin.assert_called_once_with("test_plugin")
            mock_print.assert_called_with("Plugin 'test_plugin' enabled")


def test_cli_enable_plugin_failure():
    """Test enabling plugin with error."""
    with patch("oscura.plugins.cli.get_lifecycle_manager") as mock_manager:
        mock_manager.return_value.enable_plugin.side_effect = ValueError("Error")

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_enable_plugin("test_plugin")

            assert exc_info.value.code == 1
            assert any("Failed to enable" in str(call) for call in mock_print.call_args_list)


# =============================================================================
# CLI Disable Plugin Tests
# =============================================================================


def test_cli_disable_plugin_success():
    """Test disabling plugin via CLI."""
    with patch("oscura.plugins.cli.get_lifecycle_manager") as mock_manager:
        with patch("builtins.print") as mock_print:
            cli_disable_plugin("test_plugin")

            mock_manager.return_value.disable_plugin.assert_called_once_with("test_plugin")
            mock_print.assert_called_with("Plugin 'test_plugin' disabled")


def test_cli_disable_plugin_failure():
    """Test disabling plugin with error."""
    with patch("oscura.plugins.cli.get_lifecycle_manager") as mock_manager:
        mock_manager.return_value.disable_plugin.side_effect = ValueError("Error")

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_disable_plugin("test_plugin")

            assert exc_info.value.code == 1


# =============================================================================
# CLI Validate Plugin Tests
# =============================================================================


def test_cli_validate_plugin_success():
    """Test validating plugin successfully."""
    plugin = MockPlugin()
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=True,
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_validate_plugin("mock_plugin")

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("valid" in str(c).lower() for c in calls)


def test_cli_validate_plugin_not_found():
    """Test validating non-existent plugin."""
    with patch("oscura.plugins.cli.discover_plugins", return_value=[]):
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_validate_plugin("nonexistent")

            assert exc_info.value.code == 1
            mock_print.assert_called_with("Plugin 'nonexistent' not found")


def test_cli_validate_plugin_incompatible():
    """Test validating incompatible plugin."""
    plugin = MockPlugin()
    plugin.api_version = "2.0.0"
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=False,
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_validate_plugin("mock_plugin")

            assert exc_info.value.code == 1
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("incompatible" in str(c).lower() for c in calls)


def test_cli_validate_plugin_with_load_error():
    """Test validating plugin with load error."""
    plugin = MockPlugin()
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=True,
        load_error="Load error",
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_validate_plugin("mock_plugin")

            assert exc_info.value.code == 1
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Load error" in str(c) for c in calls)


def test_cli_validate_plugin_with_dependencies():
    """Test validating plugin with dependencies."""
    plugin = MockPlugin()
    plugin.metadata.dependencies = {"dep1": ">=1.0.0"}
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=True,
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_validate_plugin("mock_plugin")

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Dependencies declared" in str(c) for c in calls)


def test_cli_validate_plugin_no_dependencies():
    """Test validating plugin with no dependencies."""
    plugin = MockPlugin()
    discovered = DiscoveredPlugin(
        metadata=plugin.metadata,
        compatible=True,
    )

    with patch("oscura.plugins.cli.discover_plugins", return_value=[discovered]):
        with patch("builtins.print") as mock_print:
            cli_validate_plugin("mock_plugin")

            calls = [str(call) for call in mock_print.call_args_list]
            assert any("No dependencies" in str(c) for c in calls)


# =============================================================================
# CLI Install Plugin Tests
# =============================================================================


def test_cli_install_plugin_success():
    """Test installing plugin via CLI."""
    url = "https://github.com/user/plugin.git"

    with patch("oscura.plugins.cli.PluginInstaller") as mock_installer_class:
        mock_installer = Mock()
        mock_installer.install_from_url.return_value = Path("/test/plugin")
        mock_installer_class.return_value = mock_installer

        with patch("builtins.print") as mock_print:
            cli_install_plugin(url)

            mock_installer.install_from_url.assert_called_once_with(url, checksum=None)
            mock_print.assert_called_with("Successfully installed plugin to /test/plugin")


def test_cli_install_plugin_with_checksum():
    """Test installing plugin with checksum."""
    url = "https://github.com/user/plugin.git"
    checksum = "abc123"

    with patch("oscura.plugins.cli.PluginInstaller") as mock_installer_class:
        mock_installer = Mock()
        mock_installer.install_from_url.return_value = Path("/test/plugin")
        mock_installer_class.return_value = mock_installer

        with patch("builtins.print"):
            cli_install_plugin(url, checksum=checksum)

            mock_installer.install_from_url.assert_called_once_with(url, checksum=checksum)


def test_cli_install_plugin_failure():
    """Test installing plugin with error."""
    url = "https://github.com/user/plugin.git"

    with patch("oscura.plugins.cli.PluginInstaller") as mock_installer_class:
        mock_installer = Mock()
        mock_installer.install_from_url.side_effect = RuntimeError("Install failed")
        mock_installer_class.return_value = mock_installer

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                cli_install_plugin(url)

            assert exc_info.value.code == 1
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("Installation failed" in str(c) for c in calls)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


def test_installer_integration_workflow(installer, temp_install_dir):
    """Test complete installation workflow."""
    # Create a simple archive
    plugin_dir = temp_install_dir / "source" / "test_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("# Plugin")
    (plugin_dir / "main.py").write_text("# Main code")

    # Create archive
    archive_path = temp_install_dir / "plugin.tar.gz"
    shutil.make_archive(
        str(temp_install_dir / "plugin"),
        "gztar",
        str(temp_install_dir / "source"),
    )

    # Compute expected checksum
    expected_checksum = installer._compute_file_checksum(archive_path, "sha256")

    # Install - mock urlretrieve to copy with .tar.gz extension
    def mock_download(url, path):
        # Copy to path with original extension preserved
        target = Path(str(path).replace(".archive", ".tar.gz"))
        shutil.copy(archive_path, target)
        shutil.copy(target, path)

    with patch("urllib.request.urlretrieve", side_effect=mock_download):
        with patch("shutil.unpack_archive") as mock_unpack:
            # Mock the unpack to create the expected directory
            def do_unpack(archive, extract_to):
                plugin_out = Path(extract_to) / "test_plugin"
                plugin_out.mkdir(parents=True)
                (plugin_out / "__init__.py").write_text("# Plugin")

            mock_unpack.side_effect = do_unpack

            result = installer.install_from_url(
                "https://example.com/plugin.tar.gz",
                checksum=expected_checksum,
            )

            assert result.exists()
            assert (result / "__init__.py").exists()


def test_installer_empty_plugin_directory(installer, temp_install_dir):
    """Test handling empty plugin directory."""
    plugin_dir = temp_install_dir / "empty_plugin"
    plugin_dir.mkdir()

    checksum = installer._compute_directory_checksum(plugin_dir, "sha256")
    assert len(checksum) == 64


def test_cli_functions_error_handling():
    """Test CLI functions handle various error conditions."""
    # Test with None metadata
    with patch("oscura.plugins.cli.get_plugin_registry") as mock_registry:
        mock_registry.return_value.get_metadata.return_value = None

        with patch("builtins.print"):
            with pytest.raises(SystemExit):
                cli_plugin_info("test")


def test_checksum_algorithms_supported(installer, temp_install_dir):
    """Test that all common hash algorithms are supported."""
    test_file = temp_install_dir / "test.txt"
    test_file.write_text("test")

    for algo in ["md5", "sha1", "sha256", "sha512"]:
        checksum = installer._compute_file_checksum(test_file, algo)
        assert len(checksum) > 0
        assert all(c in "0123456789abcdef" for c in checksum)
