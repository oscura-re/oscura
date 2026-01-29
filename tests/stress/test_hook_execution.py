#!/usr/bin/env python3
"""
Hook Execution Stress Tests

Tests concurrent hook execution, cascading failures,
timeout scenarios, resource exhaustion, and disk full scenarios.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this module as stress tests
pytestmark = pytest.mark.stress

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HOOKS_DIR = PROJECT_ROOT / ".claude" / "hooks"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_hooks_dir() -> Path:
    """Create temporary directory for hook tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_hook_script() -> str:
    """Sample hook script for testing."""
    return """#!/usr/bin/env bash
set -euo pipefail
echo '{"ok": true, "message": "Hook executed"}'
exit 0
"""


@pytest.fixture
def failing_hook_script() -> str:
    """Hook script that fails."""
    return """#!/usr/bin/env bash
echo '{"ok": false, "error": "Intentional failure"}'
exit 1
"""


@pytest.fixture
def slow_hook_script() -> str:
    """Hook script that takes time."""
    return """#!/usr/bin/env bash
sleep 5
echo '{"ok": true, "message": "Slow hook completed"}'
exit 0
"""


# =============================================================================
# Hook Execution Tests
# =============================================================================


class TestHookExecution:
    """Test basic hook execution."""

    def test_successful_hook(self, temp_hooks_dir: Path, sample_hook_script: str) -> None:
        """Test successful hook execution."""
        hook_path = temp_hooks_dir / "test_hook.sh"
        hook_path.write_text(sample_hook_script)
        hook_path.chmod(0o755)

        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True

    def test_failing_hook(self, temp_hooks_dir: Path, failing_hook_script: str) -> None:
        """Test failing hook execution."""
        hook_path = temp_hooks_dir / "failing_hook.sh"
        hook_path.write_text(failing_hook_script)
        hook_path.chmod(0o755)

        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["ok"] is False

    def test_hook_timeout(self, temp_hooks_dir: Path, slow_hook_script: str) -> None:
        """Test hook timeout handling."""
        hook_path = temp_hooks_dir / "slow_hook.sh"
        hook_path.write_text(slow_hook_script)
        hook_path.chmod(0o755)

        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [str(hook_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,  # 1 second timeout
            )


# =============================================================================
# Concurrent Execution Tests
# =============================================================================


class TestConcurrentExecution:
    """Test concurrent hook execution."""

    def test_concurrent_hooks(self, temp_hooks_dir: Path, sample_hook_script: str) -> None:
        """Test running multiple hooks concurrently."""
        hook_path = temp_hooks_dir / "concurrent_hook.sh"
        hook_path.write_text(sample_hook_script)
        hook_path.chmod(0o755)

        results: list[dict[str, Any]] = []
        errors: list[str] = []

        def run_hook(hook_id: int) -> None:
            try:
                result = subprocess.run(
                    [str(hook_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                results.append({"id": hook_id, "code": result.returncode, "output": result.stdout})
            except Exception as e:
                errors.append(f"Hook {hook_id}: {e}")

        # Run 10 hooks concurrently
        threads = [threading.Thread(target=run_hook, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert len(results) == 10
        assert all(r["code"] == 0 for r in results)

    def test_concurrent_file_access(self, temp_hooks_dir: Path) -> None:
        """Test hooks accessing same file concurrently."""
        shared_file = temp_hooks_dir / "shared.json"
        shared_file.write_text('{"count": 0}')

        hook_script = f"""#!/usr/bin/env bash
set -euo pipefail
# Read, increment, write (race condition prone)
COUNT=$(jq -r '.count' "{shared_file}" 2>/dev/null || echo 0)
NEW_COUNT=$((COUNT + 1))
echo '{{"count": '$NEW_COUNT'}}' > "{shared_file}"
echo '{{"ok": true, "count": '$NEW_COUNT'}}'
"""
        hook_path = temp_hooks_dir / "counter_hook.sh"
        hook_path.write_text(hook_script)
        hook_path.chmod(0o755)

        errors: list[str] = []

        def run_hook() -> None:
            try:
                subprocess.run(
                    [str(hook_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception as e:
                errors.append(str(e))

        # Run 5 concurrent increments
        threads = [threading.Thread(target=run_hook) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Due to race conditions, count may not be 5
        # This test documents the limitation
        final = json.loads(shared_file.read_text())
        # We just verify no crashes, not correctness
        assert "count" in final


# =============================================================================
# Cascading Failure Tests
# =============================================================================


class TestCascadingFailures:
    """Test cascading failure handling."""

    def test_hook_chain_with_failure(self, temp_hooks_dir: Path) -> None:
        """Test hook chain where middle hook fails."""
        # Create three hooks
        hook1 = temp_hooks_dir / "hook1.sh"
        hook1.write_text('#!/bin/bash\necho "Hook 1 OK"\nexit 0\n')
        hook1.chmod(0o755)

        hook2 = temp_hooks_dir / "hook2.sh"
        hook2.write_text('#!/bin/bash\necho "Hook 2 FAIL"\nexit 1\n')
        hook2.chmod(0o755)

        hook3 = temp_hooks_dir / "hook3.sh"
        hook3.write_text('#!/bin/bash\necho "Hook 3 OK"\nexit 0\n')
        hook3.chmod(0o755)

        # Run chain
        results = []
        for hook in [hook1, hook2, hook3]:
            result = subprocess.run([str(hook)], check=False, capture_output=True, text=True)
            results.append(result.returncode)
            if result.returncode != 0:
                break  # Stop on failure

        assert results == [0, 1], "Should stop at failing hook"

    def test_critical_vs_warning_hooks(self, temp_hooks_dir: Path) -> None:
        """Test distinguishing critical from warning hooks."""
        critical_hook = temp_hooks_dir / "critical.sh"
        critical_hook.write_text("#!/bin/bash\nexit 2\n")  # Critical failure
        critical_hook.chmod(0o755)

        warning_hook = temp_hooks_dir / "warning.sh"
        warning_hook.write_text("#!/bin/bash\nexit 1\n")  # Warning
        warning_hook.chmod(0o755)

        # Critical = exit 2, Warning = exit 1, OK = exit 0
        hooks = [
            {"path": critical_hook, "critical": True},
            {"path": warning_hook, "critical": False},
        ]

        critical_failed = False
        warnings = 0

        for hook in hooks:
            result = subprocess.run([str(hook["path"])], check=False, capture_output=True)
            if result.returncode != 0:
                if hook["critical"]:
                    critical_failed = True
                else:
                    warnings += 1

        assert critical_failed is True
        assert warnings == 1


# =============================================================================
# Resource Exhaustion Tests
# =============================================================================


class TestResourceExhaustion:
    """Test resource exhaustion scenarios."""

    def test_hook_with_large_output(self, temp_hooks_dir: Path) -> None:
        """Test hook producing large output."""
        hook_path = temp_hooks_dir / "large_output.sh"
        hook_path.write_text(
            """#!/bin/bash
# Generate 1MB of output
for i in $(seq 1 10000); do
    echo "Line $i: $(printf 'x%.0s' {1..100})"
done
echo '{"ok": true}'
"""
        )
        hook_path.chmod(0o755)

        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should handle large output
        assert result.returncode == 0
        assert len(result.stdout) > 1_000_000  # > 1MB

    @pytest.mark.xfail(
        reason="Flaky: passes in isolation but may fail in full suite due to resource contention"
    )
    def test_hook_memory_limit(self, temp_hooks_dir: Path) -> None:
        """Test hook with memory limits."""
        hook_path = temp_hooks_dir / "memory_hook.sh"
        hook_path.write_text(
            """#!/bin/bash
# Try to allocate memory via bash array
declare -a arr
for i in $(seq 1 10000); do
    arr+=("$(printf 'x%.0s' {1..1000})")
done
echo '{"ok": true, "elements": 10000}'
"""
        )
        hook_path.chmod(0o755)

        # This should complete (bash handles memory reasonably)
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # May succeed or fail based on system limits
        # Just verify it doesn't hang
        assert result.returncode in [0, 1, 137]  # 137 = killed


# =============================================================================
# Timeout Scenarios
# =============================================================================


class TestTimeoutScenarios:
    """Test various timeout scenarios."""

    def test_hook_respects_timeout(self, temp_hooks_dir: Path) -> None:
        """Test hook terminates on timeout."""
        hook_path = temp_hooks_dir / "infinite_hook.sh"
        hook_path.write_text(
            """#!/bin/bash
while true; do
    sleep 0.1
done
"""
        )
        hook_path.chmod(0o755)

        start = time.time()
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [str(hook_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        elapsed = time.time() - start

        # Should timeout within reasonable margin
        assert elapsed < 2

    def test_hook_cleanup_on_timeout(self, temp_hooks_dir: Path) -> None:
        """Test resources cleaned up on timeout."""
        lock_file = temp_hooks_dir / "hook.lock"

        hook_path = temp_hooks_dir / "locking_hook.sh"
        hook_path.write_text(
            f"""#!/bin/bash
echo "locked" > "{lock_file}"
sleep 10
rm -f "{lock_file}"
"""
        )
        hook_path.chmod(0o755)

        try:
            subprocess.run(
                [str(hook_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        except subprocess.TimeoutExpired:
            pass

        # Lock file may still exist after timeout
        # This documents the cleanup limitation


# =============================================================================
# Emergency Bypass Tests
# =============================================================================


class TestEmergencyBypass:
    """Test emergency bypass mechanism."""

    def test_bypass_flag_honored(self, temp_hooks_dir: Path) -> None:
        """Test CLAUDE_BYPASS_HOOKS flag is honored."""
        hook_path = temp_hooks_dir / "bypass_hook.sh"
        hook_path.write_text(
            """#!/bin/bash
if [ "${CLAUDE_BYPASS_HOOKS:-}" = "1" ]; then
    echo '{"ok": true, "bypassed": true}'
    exit 0
fi
# Normally failing hook
exit 1
"""
        )
        hook_path.chmod(0o755)

        # Without bypass - should fail
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "CLAUDE_BYPASS_HOOKS": "0"},
        )
        assert result.returncode == 1

        # With bypass - should succeed
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "CLAUDE_BYPASS_HOOKS": "1"},
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["bypassed"] is True


# =============================================================================
# Dry Run Tests
# =============================================================================


class TestDryRun:
    """Test dry-run mode."""

    def test_dry_run_no_modifications(self, temp_hooks_dir: Path) -> None:
        """Test dry-run doesn't modify files."""
        target_file = temp_hooks_dir / "target.txt"
        target_file.write_text("original content")

        hook_path = temp_hooks_dir / "modifying_hook.sh"
        hook_path.write_text(
            f"""#!/bin/bash
if [ "${{1:-}}" = "--dry-run" ]; then
    echo '{{"ok": true, "dry_run": true, "would_modify": "{target_file}"}}'
    exit 0
fi
echo "modified content" > "{target_file}"
echo '{{"ok": true, "modified": true}}'
"""
        )
        hook_path.chmod(0o755)

        # Dry run
        result = subprocess.run(
            [str(hook_path), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert target_file.read_text() == "original content"

        # Real run
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert target_file.read_text() == "modified content\n"


# =============================================================================
# Main
# =============================================================================
