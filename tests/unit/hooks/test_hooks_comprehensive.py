"""Comprehensive tests for Claude Code hooks consolidation.

Tests all newly migrated and consolidated hooks:
- cleanup_stale_agents.py
- health_check.py
- session_end_cleanup.py
- validate_stop.py (both main and subagent modes)
- generate_settings.py
- shared/datetime_utils.py

Test categories:
1. Config loading and validation
2. Shared utility functions
3. Staleness detection logic
4. File cleanup operations
5. Hook-specific behavior
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Add test helpers to path
TEST_HELPERS_PATH = Path(__file__).parent
if str(TEST_HELPERS_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_HELPERS_PATH))


def run_hook(project_dir: Path, hook_name: str, extra_args: list[str] | None = None) -> dict:
    """Run a hook script and return parsed result.

    Args:
        project_dir: Temporary project directory
        hook_name: Name of hook script (e.g., 'cleanup_stale_agents.py')
        extra_args: Additional command-line arguments

    Returns:
        dict with 'returncode', 'stdout', 'stderr'
    """
    hook_path = Path(__file__).parent.parent.parent.parent / ".claude" / "hooks" / hook_name
    cmd = ["python3", str(hook_path)]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# =============================================================================
# cleanup_stale_agents.py Tests
# =============================================================================


class TestCleanupStaleAgents:
    """Tests for cleanup_stale_agents.py hook."""

    def test_no_agents_to_clean(self, tmp_path: Path) -> None:
        """Should handle empty registry gracefully."""
        registry = {"agents": {}, "metadata": {"agents_running": 0, "agents_failed": 0}}
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, "cleanup_stale_agents.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True
        assert output["stale_marked_failed"] == 0
        assert output["old_removed"] == 0

    def test_marks_stale_agents_as_failed(self, tmp_path: Path) -> None:
        """Should mark agents without recent activity as failed."""
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        registry = {
            "agents": {
                "agent-1": {
                    "status": "running",
                    "launched_at": old_time,
                }
            },
            "metadata": {"agents_running": 1, "agents_failed": 0},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, "cleanup_stale_agents.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["stale_marked_failed"] == 1
        assert "agent-1" in output["stale_agents"]

    def test_preserves_active_agents(self, tmp_path: Path) -> None:
        """Should preserve agents with recent activity."""
        recent_time = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
        registry = {
            "agents": {
                "agent-1": {
                    "status": "running",
                    "launched_at": recent_time,
                }
            },
            "metadata": {"agents_running": 1, "agents_failed": 0},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        # Create recent output file to show activity
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        output_file = outputs_dir / "agent-1-output.json"
        output_file.write_text('{"status": "running"}')

        result = run_hook(tmp_path, "cleanup_stale_agents.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["active_preserved"] == 1
        assert "agent-1" in output["preserved_agents"]

    def test_removes_old_completed_agents(self, tmp_path: Path) -> None:
        """Should remove completed agents older than max_age_days."""
        old_time = (datetime.now(UTC) - timedelta(days=35)).isoformat()
        registry = {
            "agents": {
                "agent-1": {
                    "status": "completed",
                    "completed_at": old_time,
                }
            },
            "metadata": {"agents_running": 0, "agents_failed": 0},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, "cleanup_stale_agents.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["old_removed"] == 1
        assert "agent-1" in output["old_agents"]

    def test_dry_run_mode(self, tmp_path: Path) -> None:
        """Should report changes without modifying registry."""
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        registry = {
            "agents": {
                "agent-1": {
                    "status": "running",
                    "launched_at": old_time,
                }
            },
            "metadata": {"agents_running": 1, "agents_failed": 0},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, "cleanup_stale_agents.py", ["--dry-run"])

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["dry_run"] is True
        assert output["stale_marked_failed"] == 1

        # Verify registry was NOT modified
        updated_registry = json.loads(registry_file.read_text())
        assert updated_registry["agents"]["agent-1"]["status"] == "running"


# =============================================================================
# health_check.py Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check.py hook."""

    def test_healthy_system(self, tmp_path: Path) -> None:
        """Should report healthy when all checks pass."""
        # Create required directories
        for subdir in [
            ".claude",
            ".claude/agents",
            ".claude/hooks",
            ".claude/agent-outputs",
            ".coordination",
            ".coordination/checkpoints",
        ]:
            (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

        result = run_hook(tmp_path, "health_check.py")

        # Debug output if test fails
        if result["returncode"] != 0:
            print(f"STDOUT: {result['stdout']}")
            print(f"STDERR: {result['stderr']}")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["status"] == "healthy"
        assert all(check["passed"] for check in output["checks"].values())

    def test_missing_directories(self, tmp_path: Path) -> None:
        """Should report degraded when directories are missing."""
        # Don't create any directories
        result = run_hook(tmp_path, "health_check.py")

        output = json.loads(result["stdout"])
        assert output["status"] == "degraded"
        assert not output["checks"]["directories"]["passed"]

    def test_corrupted_registry(self, tmp_path: Path) -> None:
        """Should detect corrupted agent registry."""
        (tmp_path / ".claude").mkdir(parents=True)
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.write_text("invalid json {{{")

        result = run_hook(tmp_path, "health_check.py")

        output = json.loads(result["stdout"])
        assert output["status"] == "degraded"
        assert not output["checks"]["agent_registry"]["passed"]

    def test_stale_running_agents(self, tmp_path: Path) -> None:
        """Should warn about stale running agents."""
        registry = {
            "agents": {
                "agent-1": {"status": "running"},
                "agent-2": {"status": "completed"},
            },
            "metadata": {},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, "health_check.py")

        output = json.loads(result["stdout"])
        assert output["status"] == "degraded"
        assert not output["checks"]["agent_registry"]["passed"]


# =============================================================================
# session_end_cleanup.py Tests
# =============================================================================


class TestSessionEndCleanup:
    """Tests for session_end_cleanup.py hook."""

    def test_removes_temp_files(self, tmp_path: Path) -> None:
        """Should remove temporary files matching patterns."""
        coord_dir = tmp_path / ".coordination"
        coord_dir.mkdir(parents=True)

        # Create temp files
        (coord_dir / "test.tmp").write_text("temp")
        (coord_dir / "backup.bak").write_text("backup")
        (coord_dir / "file.txt").write_text("keep")

        result = run_hook(tmp_path, "session_end_cleanup.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["removed_temp_files"] == 2
        assert not (coord_dir / "test.tmp").exists()
        assert not (coord_dir / "backup.bak").exists()
        assert (coord_dir / "file.txt").exists()

    def test_removes_expired_locks(self, tmp_path: Path) -> None:
        """Should remove expired lock files."""
        locks_dir = tmp_path / ".coordination" / "locks"
        locks_dir.mkdir(parents=True)

        # Create expired lock
        expired_lock = locks_dir / "expired.json"
        expired_time = (datetime.now() - timedelta(hours=2)).isoformat()
        expired_lock.write_text(json.dumps({"expires_at": expired_time}))

        # Create valid lock
        valid_lock = locks_dir / "valid.json"
        future_time = (datetime.now() + timedelta(hours=1)).isoformat()
        valid_lock.write_text(json.dumps({"expires_at": future_time}))

        result = run_hook(tmp_path, "session_end_cleanup.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["removed_locks"] >= 1
        assert not expired_lock.exists()
        assert valid_lock.exists()

    def test_cleans_orphaned_chunks(self, tmp_path: Path) -> None:
        """Should remove orphaned translation chunks."""
        translation_dir = tmp_path / ".coordination" / "translation" / "test-doc"
        translation_dir.mkdir(parents=True)

        # Create orphaned chunk (old, no translated version)
        orphan_chunk = translation_dir / "chunk-001.md"
        orphan_chunk.write_text("chunk content")
        # Make it old
        old_time = (datetime.now() - timedelta(hours=48)).timestamp()
        os.utime(orphan_chunk, (old_time, old_time))

        # Create chunk with translation (should be kept)
        valid_chunk = translation_dir / "chunk-002.md"
        valid_chunk.write_text("chunk content")
        translated_chunk = translation_dir / "chunk-002-translated.md"
        translated_chunk.write_text("translated content")

        result = run_hook(tmp_path, "session_end_cleanup.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["removed_orphan_chunks"] == 1
        assert not orphan_chunk.exists()
        assert valid_chunk.exists()

    def test_archives_workflow_progress(self, tmp_path: Path) -> None:
        """Should archive workflow-progress.json."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(parents=True)

        workflow_file = claude_dir / "workflow-progress.json"
        workflow_file.write_text(json.dumps({"phase": 1, "status": "complete"}))

        result = run_hook(tmp_path, "session_end_cleanup.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["archived_workflows"] == 1
        assert not workflow_file.exists()
        # Check archived file exists
        archived_files = list(claude_dir.glob("workflow-progress-archived-*.json"))
        assert len(archived_files) == 1


# =============================================================================
# validate_stop.py Tests
# =============================================================================


class TestValidateStopMainAgent:
    """Tests for validate_stop.py in main agent mode."""

    def test_allows_stop_when_no_active_work(self, tmp_path: Path) -> None:
        """Should allow stop when no active work exists."""
        result = run_hook(tmp_path, "validate_stop.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_blocks_stop_with_fresh_active_work(self, tmp_path: Path) -> None:
        """Should block stop when active work is fresh."""
        coord_dir = tmp_path / ".coordination"
        coord_dir.mkdir(parents=True)

        active_work = coord_dir / "active_work.json"
        recent_time = (datetime.now() - timedelta(minutes=30)).isoformat()
        active_work.write_text(
            json.dumps(
                {
                    "task_id": "test-task",
                    "last_update": recent_time,
                }
            )
        )

        result = run_hook(tmp_path, "validate_stop.py")

        assert result["returncode"] == 2  # Blocked
        output = json.loads(result["stdout"])
        assert output["ok"] is False
        assert "test-task" in output["reason"]

    def test_allows_stop_with_stale_active_work(self, tmp_path: Path) -> None:
        """Should allow stop when active work is stale."""
        coord_dir = tmp_path / ".coordination"
        coord_dir.mkdir(parents=True)

        active_work = coord_dir / "active_work.json"
        old_time = (datetime.now() - timedelta(hours=3)).isoformat()
        active_work.write_text(
            json.dumps(
                {
                    "task_id": "test-task",
                    "last_update": old_time,
                }
            )
        )

        result = run_hook(tmp_path, "validate_stop.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True


class TestValidateStopSubagent:
    """Tests for validate_stop.py in subagent mode."""

    def test_allows_stop_with_completion_report(self, tmp_path: Path) -> None:
        """Should allow stop when valid completion report exists."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report_file = outputs_dir / "test-agent-complete.json"
        report_file.write_text(
            json.dumps(
                {
                    "agent_id": "test-agent",
                    "status": "complete",
                    "summary": "Task completed successfully",
                }
            )
        )

        # Create recent report (modify time)
        recent_time = datetime.now().timestamp()
        os.utime(report_file, (recent_time, recent_time))

        result = run_hook(tmp_path, "validate_stop.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_blocks_stop_with_blocked_status(self, tmp_path: Path) -> None:
        """Should block stop when report shows blocked status."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report_file = outputs_dir / "test-agent-complete.json"
        report_file.write_text(
            json.dumps(
                {
                    "agent_id": "test-agent",
                    "status": "blocked",
                    "blocked_by": "Missing dependency",
                }
            )
        )

        # Make report recent
        recent_time = datetime.now().timestamp()
        os.utime(report_file, (recent_time, recent_time))

        result = run_hook(tmp_path, "validate_stop.py")

        assert result["returncode"] == 2  # Blocked
        output = json.loads(result["stdout"])
        assert output["ok"] is False
        assert "blocked" in output["reason"].lower()

    def test_auto_summarizes_large_outputs(self, tmp_path: Path) -> None:
        """Should auto-summarize outputs larger than threshold."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        summaries_dir = tmp_path / ".claude" / "summaries"
        outputs_dir.mkdir(parents=True)
        summaries_dir.mkdir(parents=True)

        # Create large report (>200KB)
        report_file = outputs_dir / "test-agent-complete.json"
        large_content = {"agent_id": "test-agent", "status": "complete", "data": "x" * 300000}
        report_file.write_text(json.dumps(large_content))

        # Make report recent
        recent_time = datetime.now().timestamp()
        os.utime(report_file, (recent_time, recent_time))

        result = run_hook(tmp_path, "validate_stop.py")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output.get("auto_summarized") is True

        # Verify summary was created
        summary_files = list(summaries_dir.glob("*-test-agent-summary.md"))
        assert len(summary_files) == 1


# =============================================================================
# generate_settings.py Tests
# =============================================================================


class TestGenerateSettings:
    """Tests for generate_settings.py hook."""

    def test_generates_settings_from_standards(self, tmp_path: Path) -> None:
        """Should generate settings.json from coding-standards.yaml."""
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Create minimal coding standards
        standards_file = claude_dir / "coding-standards.yaml"
        standards_file.write_text(
            """
cleanup:
  retention:
    checkpoint_archives: 30
"""
        )

        result = run_hook(tmp_path, "generate_settings.py")

        # Debug output if test fails
        if result["returncode"] != 0:
            print(f"STDOUT: {result['stdout']}")
            print(f"STDERR: {result['stderr']}")

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

        # Verify settings.json was created
        settings_file = claude_dir / "settings.json"
        assert settings_file.exists()

        settings = json.loads(settings_file.read_text())
        assert settings["cleanupPeriodDays"] == 30
        assert "_generated" in settings

    def test_dry_run_mode(self, tmp_path: Path) -> None:
        """Should show what would be generated without writing."""
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        standards_file = claude_dir / "coding-standards.yaml"
        standards_file.write_text("cleanup:\n  retention:\n    checkpoint_archives: 30")

        result = run_hook(tmp_path, "generate_settings.py", ["--dry-run"])

        assert result["returncode"] == 0

        # Verify settings.json was NOT created
        settings_file = claude_dir / "settings.json"
        assert not settings_file.exists()

    def test_preserves_existing_settings(self, tmp_path: Path) -> None:
        """Should preserve non-generated fields from existing settings."""
        claude_dir = tmp_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        hooks_dir.mkdir(parents=True)

        # Create existing settings with custom fields
        settings_file = claude_dir / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "model": "opus",
                    "alwaysThinkingEnabled": False,
                    "custom_field": "preserve_me",
                }
            )
        )

        standards_file = claude_dir / "coding-standards.yaml"
        standards_file.write_text("cleanup:\n  retention:\n    checkpoint_archives: 30")

        result = run_hook(tmp_path, "generate_settings.py")

        assert result["returncode"] == 0

        # Verify custom field was preserved
        updated_settings = json.loads(settings_file.read_text())
        assert updated_settings["model"] == "opus"
        assert updated_settings["alwaysThinkingEnabled"] is False
        assert updated_settings["custom_field"] == "preserve_me"


# =============================================================================
# shared/datetime_utils.py Tests
# =============================================================================


class TestDatetimeUtils:
    """Tests for shared/datetime_utils.py module."""

    def test_parse_timestamp_with_z_suffix(self) -> None:
        """Should parse ISO timestamp with Z suffix."""
        import datetime_utils_for_test

        timestamp = "2026-01-19T10:30:00Z"
        result = datetime_utils_for_test.parse_timestamp(timestamp)

        assert result is not None
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 19

    def test_parse_timestamp_invalid(self) -> None:
        """Should return None for invalid timestamps."""
        import datetime_utils_for_test

        result = datetime_utils_for_test.parse_timestamp("invalid")
        assert result is None

    def test_age_in_hours(self) -> None:
        """Should calculate age in hours correctly."""
        import datetime_utils_for_test

        old_time = datetime.now(UTC) - timedelta(hours=3)
        age = datetime_utils_for_test.age_in_hours(old_time.isoformat())

        assert age is not None
        assert 2.9 < age < 3.1  # Allow small float variance

    def test_is_stale_true(self) -> None:
        """Should detect stale timestamps."""
        import datetime_utils_for_test

        old_time = datetime.now(UTC) - timedelta(hours=5)
        result = datetime_utils_for_test.is_stale(old_time.isoformat(), 2)

        assert result is True

    def test_is_stale_false(self) -> None:
        """Should detect non-stale timestamps."""
        import datetime_utils_for_test

        recent_time = datetime.now(UTC) - timedelta(hours=1)
        result = datetime_utils_for_test.is_stale(recent_time.isoformat(), 2)

        assert result is False

    def test_is_file_stale(self, tmp_path: Path) -> None:
        """Should detect stale files by modification time."""
        import datetime_utils_for_test

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Make file old
        old_time = (datetime.now() - timedelta(hours=3)).timestamp()
        os.utime(test_file, (old_time, old_time))

        result = datetime_utils_for_test.is_file_stale(test_file, 2)
        assert result is True

    def test_format_age(self) -> None:
        """Should format age as human-readable string."""
        import datetime_utils_for_test

        # Test hours
        hours_ago = datetime.now(UTC) - timedelta(hours=2)
        assert datetime_utils_for_test.format_age(hours_ago.isoformat()) == "2h ago"

        # Test days
        days_ago = datetime.now(UTC) - timedelta(days=3)
        assert datetime_utils_for_test.format_age(days_ago.isoformat()) == "3d ago"

        # Test minutes
        minutes_ago = datetime.now(UTC) - timedelta(minutes=30)
        assert "30m ago" in datetime_utils_for_test.format_age(minutes_ago.isoformat())


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestGetConfig:
    """Tests for get_config.py utility."""

    def test_get_nested_value_simple(self) -> None:
        """Should get simple nested values."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from get_config import get_nested_value

        data = {"hooks": {"health_check": {"disk_space": 10}}}
        assert get_nested_value(data, "hooks.health_check.disk_space") == 10

    def test_get_nested_value_default(self) -> None:
        """Should return default when key not found."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from get_config import get_nested_value

        data = {"hooks": {}}
        assert get_nested_value(data, "hooks.missing.key", "default") == "default"


class TestSecurityPatterns:
    """Tests for shared/security.py pattern matching."""

    def test_matches_pattern_simple(self) -> None:
        """Should match simple patterns."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.security import matches_pattern

        assert matches_pattern(Path(".env"), ".env")
        assert matches_pattern(Path("config/.env.local"), ".env.*")

    def test_is_blocked_path_credentials(self) -> None:
        """Should block credential files."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.security import is_blocked_path

        blocked, msg = is_blocked_path(Path(".env"), Path("/project"))
        assert blocked is True
        assert "security" in msg.lower()

    def test_is_blocked_path_git_internals(self) -> None:
        """Should block .git internals."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.security import is_blocked_path

        blocked, msg = is_blocked_path(Path(".git/config"), Path("/project"))
        assert blocked is True

    def test_is_warned_path(self) -> None:
        """Should warn for critical config files."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.security import is_warned_path

        warned, msg = is_warned_path(Path(".claude/settings.json"), Path("/project"))
        assert warned is True


class TestConfigValidation:
    """Tests for shared/config.py validation functions."""

    def test_validate_config_schema_valid(self) -> None:
        """Should pass for valid config."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.config import validate_config_schema

        config = {
            "orchestration": {"agents": {"max_concurrent": 2}},
            "retention": {"reports": 30},
            "hooks": {},
            "security": {},
        }
        errors = validate_config_schema(config)
        assert len(errors) == 0

    def test_validate_config_schema_invalid_range(self) -> None:
        """Should detect out-of-range values."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.config import validate_config_schema

        config = {
            "orchestration": {"agents": {"max_concurrent": -1}},
            "retention": {},
            "hooks": {},
            "security": {},
        }
        errors = validate_config_schema(config)
        assert len(errors) > 0


class TestLoggingUtils:
    """Tests for shared/logging_utils.py."""

    def test_get_hook_logger(self, tmp_path: Path) -> None:
        """Should create logger with correct configuration."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.logging_utils import get_hook_logger

        logger = get_hook_logger("test_hook")
        assert logger.name == "test_hook"

    def test_log_hook_start_end(self, tmp_path: Path) -> None:
        """Should log hook start and end."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"))
        from shared.logging_utils import get_hook_logger, log_hook_end, log_hook_start

        logger = get_hook_logger("test_hook")
        log_hook_start(logger, "Test Hook", {"test": "data"})
        log_hook_end(logger, "Test Hook", success=True)
        # Just verify no exceptions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
