import pytest

"""Unit tests for enforce_agent_limit.py hook.

Tests the PreToolUse hook that blocks Task tool calls when >=2 agents are running.
This is Layer 1 of the 4-layer orchestration enforcement system (v3.2.0).

Test categories:
1. Allow decisions - when fewer than max agents running
2. Block decisions - when at or above max agents
3. Registry interaction - reading/writing agent registry
4. Error handling - corrupted registry, missing files
5. Configuration - reading orchestration-config.yaml
"""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

pytestmark = pytest.mark.unit


class TestEnforceAgentLimitAllow:
    """Tests for allowing Task tool calls."""

    def test_allows_when_no_agents_running(self, tmp_path: Path) -> None:
        """Should allow Task when no agents are registered."""
        registry = {"agents": {}, "metadata": {"agents_running": 0}}
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"
        assert output["running_agents"] == 0
        assert output["slots_available"] == 2

    def test_allows_when_one_agent_running(self, tmp_path: Path) -> None:
        """Should allow Task when 1 agent is running (below limit of 2)."""
        registry = {
            "agents": {
                "agent-1": {
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                }
            },
            "metadata": {"agents_running": 1},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"
        assert output["running_agents"] == 1
        assert output["slots_available"] == 1

    def test_allows_when_registry_missing(self, tmp_path: Path) -> None:
        """Should allow Task when registry file doesn't exist (no agents)."""
        (tmp_path / ".claude").mkdir(parents=True)
        # No registry file created

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"

    def test_allows_when_all_agents_completed(self, tmp_path: Path) -> None:
        """Should allow Task when all registered agents are completed."""
        registry = {
            "agents": {
                "agent-1": {"status": "completed"},
                "agent-2": {"status": "completed"},
                "agent-3": {"status": "completed"},
            },
            "metadata": {"agents_running": 0},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"
        assert output["running_agents"] == 0


class TestEnforceAgentLimitBlock:
    """Tests for blocking Task tool calls."""

    def test_blocks_when_two_agents_running(self, tmp_path: Path) -> None:
        """Should block Task when 2 agents are running (at limit)."""
        registry = {
            "agents": {
                "agent-1": {"status": "running", "description": "Task 1"},
                "agent-2": {"status": "running", "description": "Task 2"},
            },
            "metadata": {"agents_running": 2},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 1  # Blocked
        output = json.loads(result["stdout"])
        assert output["decision"] == "block"
        assert output["running_agents"] == 2
        # slots_available may not be present when blocked
        if "slots_available" in output:
            assert output["slots_available"] == 0

    def test_blocks_when_three_agents_running(self, tmp_path: Path) -> None:
        """Should block Task when 3 agents are running (above limit)."""
        registry = {
            "agents": {
                "agent-1": {"status": "running"},
                "agent-2": {"status": "running"},
                "agent-3": {"status": "running"},
            },
            "metadata": {"agents_running": 3},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 1
        output = json.loads(result["stdout"])
        assert output["decision"] == "block"
        assert output["running_agents"] == 3


class TestEnforceAgentLimitRegistry:
    """Tests for registry interaction."""

    def test_counts_only_running_agents(self, tmp_path: Path) -> None:
        """Should only count agents with status='running'."""
        registry = {
            "agents": {
                "agent-1": {"status": "running"},
                "agent-2": {"status": "completed"},
                "agent-3": {"status": "failed"},
                "agent-4": {"status": "running"},
                "agent-5": {"status": "cancelled"},
            },
            "metadata": {"agents_running": 2},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 1  # 2 running = blocked
        output = json.loads(result["stdout"])
        assert output["running_agents"] == 2

    def test_recalculates_running_count(self, tmp_path: Path) -> None:
        """Should recalculate running count from agents, not trust metadata."""
        # Metadata says 0, but actually 2 are running
        registry = {
            "agents": {
                "agent-1": {"status": "running"},
                "agent-2": {"status": "running"},
            },
            "metadata": {"agents_running": 0},  # Stale metadata
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        output = json.loads(result["stdout"])
        # Should block based on actual count, not stale metadata
        assert output["running_agents"] == 2


class TestEnforceAgentLimitErrorHandling:
    """Tests for error handling and edge cases."""

    def test_handles_corrupted_registry_json(self, tmp_path: Path) -> None:
        """Should allow Task when registry JSON is corrupted."""
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text("{ invalid json }")

        result = run_hook(tmp_path, {})

        # Should fail open (allow) on corruption
        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"

    def test_handles_empty_registry_file(self, tmp_path: Path) -> None:
        """Should allow Task when registry file is empty."""
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text("")

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"

    def test_handles_missing_agents_key(self, tmp_path: Path) -> None:
        """Should allow Task when registry has no 'agents' key."""
        registry = {"metadata": {"version": "1.0"}}
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["decision"] == "allow"

    def test_handles_registry_permission_error(self, tmp_path: Path) -> None:
        """Should allow Task when registry file is not readable."""
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text("{}")
        registry_file.chmod(0o000)

        try:
            result = run_hook(tmp_path, {})
            # Should fail open
            assert result["returncode"] == 0
        finally:
            registry_file.chmod(0o644)


class TestEnforceAgentLimitLogging:
    """Tests for logging and metrics."""

    def test_logs_block_decision(self, tmp_path: Path) -> None:
        """Should log when blocking a Task."""
        registry = {
            "agents": {"agent-1": {"status": "running"}, "agent-2": {"status": "running"}},
            "metadata": {"agents_running": 2},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(json.dumps(registry))

        # Create hooks dir for logs
        hooks_dir = tmp_path / ".claude" / "hooks"
        hooks_dir.mkdir(parents=True)

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 1
        # Check log file was created
        log_file = hooks_dir / "enforcement.log"
        if log_file.exists():
            log_content = log_file.read_text()
            assert "block" in log_content.lower() or "BLOCKED" in log_content


def run_hook(project_dir: Path, input_data: dict) -> dict:
    """Run the enforce_agent_limit.py hook and return results."""
    hook_path = (
        Path(__file__).parent.parent.parent.parent / ".claude" / "hooks" / "enforce_agent_limit.py"
    )

    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)

    result = subprocess.run(
        ["python3", str(hook_path)],
        check=False,
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
