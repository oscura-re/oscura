#!/usr/bin/env python3
"""
Edge Case Stress Tests

Tests handling of unicode in filenames, special characters,
symlinks, permissions issues, clock skew, and timezone edge cases.
"""

import json
import stat
import subprocess
import sys
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Mark all tests in this module as stress tests
pytestmark = pytest.mark.stress

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Create temporary directory for edge case tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Unicode and Special Characters Tests
# =============================================================================


class TestUnicodeHandling:
    """Test handling of unicode in filenames and content."""

    def test_unicode_filename(self, temp_test_dir: Path) -> None:
        """Test file with unicode name."""
        unicode_file = temp_test_dir / "测试文件.yaml"
        unicode_file.write_text("key: value\n")

        assert unicode_file.exists()
        assert unicode_file.read_text() == "key: value\n"

    def test_emoji_filename(self, temp_test_dir: Path) -> None:
        """Test file with emoji in name."""
        emoji_file = temp_test_dir / "test_🚀_file.json"
        emoji_file.write_text('{"rocket": true}')

        assert emoji_file.exists()
        content = json.loads(emoji_file.read_text())
        assert content["rocket"] is True

    def test_unicode_in_config_values(self, temp_test_dir: Path) -> None:
        """Test unicode values in configuration."""
        config = {
            "project": {
                "name": "测试项目",
                "author": "José García",
                "emoji": "🎉",
            }
        }

        config_file = temp_test_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)

        with open(config_file, encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["project"]["name"] == "测试项目"
        assert loaded["project"]["emoji"] == "🎉"

    def test_special_characters_in_paths(self, temp_test_dir: Path) -> None:
        """Test paths with special characters."""
        special_chars = ["space file.txt", "file[1].txt", "file(1).txt", "file'quote'.txt"]

        for name in special_chars:
            try:
                file_path = temp_test_dir / name
                file_path.write_text(f"content of {name}")
                assert file_path.exists(), f"Failed to create: {name}"
                assert file_path.read_text() == f"content of {name}"
            except OSError as e:
                # Some filesystems may reject certain characters
                pytest.skip(f"Filesystem doesn't support: {name} - {e}")


# =============================================================================
# Symlink Tests
# =============================================================================


class TestSymlinkHandling:
    """Test handling of symlinks."""

    def test_symlink_file(self, temp_test_dir: Path) -> None:
        """Test symlink to file."""
        real_file = temp_test_dir / "real.txt"
        real_file.write_text("real content")

        link_file = temp_test_dir / "link.txt"
        try:
            link_file.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        # Read through symlink
        assert link_file.read_text() == "real content"
        assert link_file.is_symlink()
        assert link_file.resolve() == real_file.resolve()

    def test_symlink_directory(self, temp_test_dir: Path) -> None:
        """Test symlink to directory."""
        real_dir = temp_test_dir / "real_dir"
        real_dir.mkdir()
        (real_dir / "file.txt").write_text("content")

        link_dir = temp_test_dir / "link_dir"
        try:
            link_dir.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        # Access through symlink
        assert (link_dir / "file.txt").read_text() == "content"

    def test_broken_symlink(self, temp_test_dir: Path) -> None:
        """Test handling of broken symlinks."""
        real_file = temp_test_dir / "will_delete.txt"
        real_file.write_text("temporary")

        link_file = temp_test_dir / "broken_link.txt"
        try:
            link_file.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        # Delete the target
        real_file.unlink()

        # Symlink exists but target doesn't
        assert link_file.is_symlink()
        assert not link_file.exists()

        # Reading broken symlink should fail
        with pytest.raises(FileNotFoundError):
            link_file.read_text()

    def test_symlink_escape_detection(self, temp_test_dir: Path) -> None:
        """Test detection of symlinks escaping project root."""
        project_root = temp_test_dir / "project"
        project_root.mkdir()

        external_file = temp_test_dir / "external.txt"
        external_file.write_text("external content")

        escape_link = project_root / "escape.txt"
        try:
            escape_link.symlink_to(external_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        # Detection logic
        def check_symlink_escape(path: Path, project_root: Path) -> bool:
            if path.is_symlink():
                target = path.resolve()
                try:
                    target.relative_to(project_root)
                    return False  # Within project
                except ValueError:
                    return True  # Escapes project
            return False

        assert check_symlink_escape(escape_link, project_root)


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermissionHandling:
    """Test handling of permission issues."""

    def test_read_only_file(self, temp_test_dir: Path) -> None:
        """Test handling read-only files."""
        readonly_file = temp_test_dir / "readonly.txt"
        readonly_file.write_text("read only content")

        # Make read-only
        readonly_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # Read should work
        assert readonly_file.read_text() == "read only content"

        # Write should fail
        with pytest.raises(PermissionError):
            readonly_file.write_text("new content")

        # Cleanup - restore permissions
        readonly_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_no_execute_hook(self, temp_test_dir: Path) -> None:
        """Test hook without execute permission."""
        hook = temp_test_dir / "hook.sh"
        hook.write_text("#!/bin/bash\necho OK\n")

        # No execute permission
        hook.chmod(stat.S_IRUSR | stat.S_IWUSR)

        # Try to execute - should raise PermissionError
        with pytest.raises(PermissionError):
            subprocess.run([str(hook)], capture_output=True, text=True, check=True)

    def test_read_only_directory(self, temp_test_dir: Path) -> None:
        """Test handling read-only directories."""
        readonly_dir = temp_test_dir / "readonly_dir"
        readonly_dir.mkdir()
        (readonly_dir / "existing.txt").write_text("exists")

        # Make read-only
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        try:
            # Reading should work
            assert (readonly_dir / "existing.txt").read_text() == "exists"

            # Creating new file should fail
            with pytest.raises(PermissionError):
                (readonly_dir / "new.txt").write_text("new")
        finally:
            # Cleanup - restore permissions
            readonly_dir.chmod(stat.S_IRWXU)


# =============================================================================
# Clock and Timezone Tests
# =============================================================================


class TestTimezoneHandling:
    """Test timezone edge cases."""

    def test_utc_timestamps(self) -> None:
        """Test UTC timestamp handling."""
        now_utc = datetime.now(UTC)
        iso_string = now_utc.isoformat()

        # Parse back
        parsed = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))

        assert parsed.tzinfo is not None
        assert parsed.tzinfo == UTC

    def test_naive_datetime_comparison(self) -> None:
        """Test comparing naive and aware datetimes."""
        naive = datetime.now()
        aware = datetime.now(UTC)

        # Direct comparison raises TypeError
        with pytest.raises(TypeError):
            _ = naive < aware

        # Correct approach: make both aware or both naive
        naive_utc = naive.replace(tzinfo=UTC)
        # Now comparison works
        _ = naive_utc < aware  # No exception

    def test_iso_format_variations(self) -> None:
        """Test parsing various ISO format variations."""
        formats = [
            "2025-12-25T10:30:00Z",
            "2025-12-25T10:30:00+00:00",
            "2025-12-25T10:30:00.123456Z",
            "2025-12-25T10:30:00.123456+00:00",
            "2025-12-25T05:30:00-05:00",
        ]

        for fmt in formats:
            # Replace Z with +00:00 for fromisoformat compatibility
            normalized = fmt.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
                assert parsed is not None
            except ValueError as e:
                pytest.fail(f"Failed to parse {fmt}: {e}")

    def test_timestamp_age_calculation(self) -> None:
        """Test age calculation with timezones."""
        now = datetime.now(UTC)

        # Create timestamp 25 hours ago
        old_time = now - timedelta(hours=25)
        old_str = old_time.isoformat()

        # Parse and calculate age
        parsed = datetime.fromisoformat(old_str.replace("Z", "+00:00"))
        age = now - parsed

        assert age >= timedelta(hours=25)
        assert age < timedelta(hours=26)

    def test_dst_transition(self) -> None:
        """Test handling around DST transitions."""
        # UTC doesn't have DST, so this tests the principle
        utc_time = datetime(2025, 3, 9, 7, 0, 0, tzinfo=UTC)  # Around US DST

        # Convert to string and back
        iso_str = utc_time.isoformat()
        parsed = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))

        assert parsed == utc_time


# =============================================================================
# Clock Skew Tests
# =============================================================================


class TestClockSkew:
    """Test handling of clock skew scenarios."""

    def test_future_timestamp(self) -> None:
        """Test handling timestamp in the future."""
        now = datetime.now(UTC)
        future = now + timedelta(hours=1)

        # Check if timestamp is in future
        is_future = future > now
        assert is_future

        # Age would be negative
        age = now - future
        assert age < timedelta(0)

    def test_very_old_timestamp(self) -> None:
        """Test handling very old timestamps."""
        now = datetime.now(UTC)
        very_old = datetime(2000, 1, 1, tzinfo=UTC)

        age = now - very_old
        age_days = age.days

        assert age_days > 9000  # More than 25 years

    def test_epoch_timestamp(self) -> None:
        """Test handling Unix epoch."""
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        now = datetime.now(UTC)

        age = now - epoch
        assert age.total_seconds() > 0


# =============================================================================
# File System Edge Cases
# =============================================================================


class TestFileSystemEdgeCases:
    """Test file system edge cases."""

    def test_empty_file(self, temp_test_dir: Path) -> None:
        """Test handling empty files."""
        empty_file = temp_test_dir / "empty.json"
        empty_file.touch()

        assert empty_file.exists()
        assert empty_file.stat().st_size == 0

        # JSON parsing should fail
        with pytest.raises(json.JSONDecodeError), open(empty_file) as f:
            json.load(f)

    def test_very_long_filename(self, temp_test_dir: Path) -> None:
        """Test handling very long filenames."""
        # Most filesystems limit to 255 bytes
        long_name = "a" * 200 + ".txt"

        try:
            long_file = temp_test_dir / long_name
            long_file.write_text("content")
            assert long_file.exists()
        except OSError:
            pytest.skip("Filesystem doesn't support long filenames")

    def test_deeply_nested_path(self, temp_test_dir: Path) -> None:
        """Test deeply nested directory paths."""
        # Create 50-level deep path
        deep_path = temp_test_dir
        for i in range(50):
            deep_path = deep_path / f"level{i}"

        try:
            deep_path.mkdir(parents=True)
            deep_file = deep_path / "deep.txt"
            deep_file.write_text("deep content")
            assert deep_file.read_text() == "deep content"
        except OSError as e:
            if "name too long" in str(e).lower():
                pytest.skip("Path too long for filesystem")
            raise

    def test_concurrent_file_creation(self, temp_test_dir: Path) -> None:
        """Test concurrent file creation."""
        import threading

        target_file = temp_test_dir / "concurrent.txt"
        errors: list[str] = []
        success_count = [0]  # Use list for mutable counter

        def create_file(content: str) -> None:
            try:
                target_file.write_text(content)
                success_count[0] += 1
            except Exception as e:
                errors.append(str(e))

        # 10 threads trying to write simultaneously
        threads = [threading.Thread(target=create_file, args=(f"content-{i}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some writes may succeed, some may fail
        # The important thing is no crashes
        assert target_file.exists()


# =============================================================================
# Main
# =============================================================================
