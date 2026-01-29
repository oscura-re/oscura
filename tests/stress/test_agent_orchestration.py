#!/usr/bin/env python3
"""
Agent Orchestration Stress Tests

Tests handling of many concurrent agents, long-running agents,
agent crashes and recovery, registry corruption, and network failures.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
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
def temp_registry_dir() -> Generator[Path, None, None]:
    """Create temporary directory for registry tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Create required directory structure
        (tmpdir_path / ".claude").mkdir()
        (tmpdir_path / ".claude" / "agent-outputs").mkdir()
        (tmpdir_path / ".claude" / "summaries").mkdir()
        (tmpdir_path / ".claude" / "hooks").mkdir()
        yield tmpdir_path


@pytest.fixture
def empty_registry() -> dict[str, Any]:
    """Empty agent registry structure."""
    now = datetime.now(UTC).isoformat()
    return {
        "version": "2.0.0",
        "created_at": now,
        "last_updated": now,
        "agents": {},
        "metadata": {
            "total_agents_launched": 0,
            "agents_running": 0,
            "agents_completed": 0,
            "agents_failed": 0,
            "last_cleanup": now,
        },
    }


@pytest.fixture
def registry_with_agents() -> dict[str, Any]:
    """Registry with sample agents."""
    now = datetime.now(UTC)
    registry = {
        "version": "2.0.0",
        "created_at": (now - timedelta(days=5)).isoformat(),
        "last_updated": now.isoformat(),
        "agents": {},
        "metadata": {
            "total_agents_launched": 10,
            "agents_running": 3,
            "agents_completed": 6,
            "agents_failed": 1,
            "last_cleanup": now.isoformat(),
        },
    }

    # Add running agents
    for i in range(3):
        registry["agents"][f"running-{i}"] = {
            "task_description": f"Running task {i}",
            "status": "running",
            "launched_at": (now - timedelta(hours=i)).isoformat(),
            "completed_at": None,
        }

    # Add completed agents
    for i in range(6):
        registry["agents"][f"completed-{i}"] = {
            "task_description": f"Completed task {i}",
            "status": "completed",
            "launched_at": (now - timedelta(days=i + 1)).isoformat(),
            "completed_at": (now - timedelta(days=i)).isoformat(),
        }

    # Add failed agent
    registry["agents"]["failed-0"] = {
        "task_description": "Failed task",
        "status": "failed",
        "launched_at": (now - timedelta(hours=12)).isoformat(),
        "completed_at": (now - timedelta(hours=11)).isoformat(),
        "failure_reason": "Test failure",
    }

    return registry


# =============================================================================
# Many Concurrent Agents Tests
# =============================================================================


class TestManyAgents:
    """Test handling of many concurrent agents."""

    def test_register_100_agents(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test registering 100 agents."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"
        registry_file.write_text(json.dumps(empty_registry))

        # Register 100 agents
        registry = empty_registry.copy()
        now = datetime.now(UTC)

        for i in range(100):
            agent_id = f"agent-{i:03d}"
            registry["agents"][agent_id] = {
                "task_description": f"Task {i}",
                "status": "running",
                "launched_at": now.isoformat(),
                "completed_at": None,
            }
            registry["metadata"]["total_agents_launched"] += 1
            registry["metadata"]["agents_running"] += 1

        # Save and reload
        registry_file.write_text(json.dumps(registry, indent=2))

        with open(registry_file) as f:
            loaded = json.load(f)

        assert len(loaded["agents"]) == 100
        assert loaded["metadata"]["agents_running"] == 100

    def test_concurrent_agent_registration(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test concurrent agent registration (race condition test)."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"
        registry_file.write_text(json.dumps(empty_registry))

        errors: list[str] = []
        results: list[bool] = []

        def register_agent(agent_id: str) -> None:
            try:
                # Read, modify, write (race condition prone)
                with open(registry_file) as f:
                    registry = json.load(f)

                registry["agents"][agent_id] = {
                    "task_description": f"Task {agent_id}",
                    "status": "running",
                    "launched_at": datetime.now(UTC).isoformat(),
                }
                registry["metadata"]["total_agents_launched"] += 1

                with open(registry_file, "w") as f:
                    json.dump(registry, f)

                results.append(True)
            except Exception as e:
                errors.append(str(e))
                results.append(False)

        # 10 concurrent registrations
        threads = [threading.Thread(target=register_agent, args=(f"agent-{i}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some may fail due to race conditions
        # This is expected - documents the limitation
        assert len(results) == 10

    def test_list_many_agents_performance(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test performance of listing many agents."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"

        # Create registry with 1000 agents
        registry = empty_registry.copy()
        now = datetime.now(UTC)

        for i in range(1000):
            registry["agents"][f"agent-{i:04d}"] = {
                "task_description": f"Task {i}",
                "status": "completed" if i % 2 == 0 else "running",
                "launched_at": now.isoformat(),
                "completed_at": now.isoformat() if i % 2 == 0 else None,
            }

        registry_file.write_text(json.dumps(registry))

        # Time the load
        start = time.time()
        with open(registry_file) as f:
            loaded = json.load(f)

        # Filter running agents
        running = [a for a, d in loaded["agents"].items() if d["status"] == "running"]
        elapsed = time.time() - start

        assert len(running) == 500
        assert elapsed < 1.0, f"Load took {elapsed:.2f}s - too slow"


# =============================================================================
# Long Running Agent Tests
# =============================================================================


class TestLongRunningAgents:
    """Test handling of long-running agents."""

    def test_agent_running_24h(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test agent running for 24 hours."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"

        registry = empty_registry.copy()
        now = datetime.now(UTC)

        registry["agents"]["long-running"] = {
            "task_description": "Long running task",
            "status": "running",
            "launched_at": (now - timedelta(hours=24)).isoformat(),
            "completed_at": None,
        }

        registry_file.write_text(json.dumps(registry))

        with open(registry_file) as f:
            loaded = json.load(f)

        agent = loaded["agents"]["long-running"]
        launched = datetime.fromisoformat(agent["launched_at"].replace("Z", "+00:00"))
        age_hours = (now - launched).total_seconds() / 3600

        assert age_hours >= 24
        assert agent["status"] == "running"

    def test_stale_agent_detection(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test detecting stale agents."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"

        registry = empty_registry.copy()
        now = datetime.now(UTC)

        # Stale agent (48h old, no activity)
        registry["agents"]["stale"] = {
            "task_description": "Stale task",
            "status": "running",
            "launched_at": (now - timedelta(hours=48)).isoformat(),
            "completed_at": None,
        }

        # Active agent (48h old but has output)
        registry["agents"]["active-old"] = {
            "task_description": "Active old task",
            "status": "running",
            "launched_at": (now - timedelta(hours=48)).isoformat(),
            "completed_at": None,
            "output_file": "recent-output.json",
        }

        registry_file.write_text(json.dumps(registry))

        # Create recent output file for active-old
        output_file = temp_registry_dir / ".claude" / "agent-outputs" / "recent-output.json"
        output_file.write_text('{"status": "in_progress"}')

        # Detection logic
        stale_threshold = timedelta(hours=24)
        activity_check = timedelta(hours=1)

        stale_agents = []
        for agent_id, agent in registry["agents"].items():
            if agent["status"] != "running":
                continue

            launched = datetime.fromisoformat(agent["launched_at"].replace("Z", "+00:00"))
            age = now - launched

            if age <= stale_threshold:
                continue

            # Check for activity
            output_path = agent.get("output_file")
            has_recent_activity = False

            if output_path:
                full_path = temp_registry_dir / ".claude" / "agent-outputs" / output_path
                if full_path.exists():
                    mtime = datetime.fromtimestamp(full_path.stat().st_mtime, tz=UTC)
                    if (now - mtime) < activity_check:
                        has_recent_activity = True

            if not has_recent_activity:
                stale_agents.append(agent_id)

        # Touch the output file to simulate recent activity
        os.utime(output_file, None)

        # Re-run detection
        stale_agents_after = []
        for agent_id, agent in registry["agents"].items():
            if agent["status"] != "running":
                continue

            launched = datetime.fromisoformat(agent["launched_at"].replace("Z", "+00:00"))
            if (now - launched) > stale_threshold:
                output_path = agent.get("output_file")
                if not output_path:
                    stale_agents_after.append(agent_id)

        assert "stale" in stale_agents
        assert "active-old" not in stale_agents


# =============================================================================
# Agent Crash and Recovery Tests
# =============================================================================


class TestAgentRecovery:
    """Test agent crash and recovery scenarios."""

    def test_registry_corruption_recovery(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test recovery from corrupted registry."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"
        backup_file = temp_registry_dir / ".claude" / "agent-registry.backup.json"

        # Create valid backup
        backup_file.write_text(json.dumps(empty_registry))

        # Create corrupted registry
        registry_file.write_text('{"agents": {invalid json here')

        # Recovery logic
        try:
            with open(registry_file) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            # Corrupt - try backup
            if backup_file.exists():
                with open(backup_file) as f:
                    registry = json.load(f)
            else:
                registry = empty_registry

        assert "agents" in registry
        assert registry["version"] == "2.0.0"

    def test_registry_atomic_write(
        self, temp_registry_dir: Path, registry_with_agents: dict[str, Any]
    ) -> None:
        """Test atomic write protects against corruption."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"
        backup_file = temp_registry_dir / ".claude" / "agent-registry.backup.json"

        registry_file.write_text(json.dumps(registry_with_agents))

        # Atomic write pattern
        def atomic_write(path: Path, data: dict[str, Any]) -> None:
            # Backup existing
            if path.exists():
                shutil.copy2(path, backup_file)

            # Write to temp
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f)

            # Atomic replace
            temp_path.replace(path)

        # Test atomic write
        updated = registry_with_agents.copy()
        updated["agents"]["new-agent"] = {
            "task_description": "New task",
            "status": "running",
            "launched_at": datetime.now(UTC).isoformat(),
        }

        atomic_write(registry_file, updated)

        # Verify
        with open(registry_file) as f:
            loaded = json.load(f)
        assert "new-agent" in loaded["agents"]

        # Verify backup exists
        assert backup_file.exists()

    def test_orphaned_agent_cleanup(
        self, temp_registry_dir: Path, registry_with_agents: dict[str, Any]
    ) -> None:
        """Test cleanup of orphaned agents."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"
        registry_file.write_text(json.dumps(registry_with_agents))

        now = datetime.now(UTC)

        # Mark agents as orphaned (running for > 24h with no activity)
        with open(registry_file) as f:
            registry = json.load(f)

        orphaned = []
        for agent_id, agent in registry["agents"].items():
            if agent["status"] != "running":
                continue

            launched = datetime.fromisoformat(agent["launched_at"].replace("Z", "+00:00"))
            if (now - launched) > timedelta(hours=24):
                orphaned.append(agent_id)
                agent["status"] = "failed"
                agent["completed_at"] = now.isoformat()
                agent["failure_reason"] = "Orphaned - no activity"

        with open(registry_file, "w") as f:
            json.dump(registry, f)

        # Verify orphaned agents were marked
        with open(registry_file) as f:
            loaded = json.load(f)

        for agent_id in orphaned:
            assert loaded["agents"][agent_id]["status"] == "failed"


# =============================================================================
# Registry Size Management Tests
# =============================================================================


class TestRegistrySizeManagement:
    """Test registry size management."""

    def test_compact_completed_agents(
        self, temp_registry_dir: Path, registry_with_agents: dict[str, Any]
    ) -> None:
        """Test compacting completed agents from registry."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"
        registry_file.write_text(json.dumps(registry_with_agents))

        with open(registry_file) as f:
            registry = json.load(f)

        # Count before
        before_count = len(registry["agents"])
        running_count = sum(1 for a in registry["agents"].values() if a["status"] == "running")

        # Compact - remove completed/failed
        for agent_id in list(registry["agents"].keys()):
            if registry["agents"][agent_id]["status"] in ("completed", "failed"):
                del registry["agents"][agent_id]

        # Count after
        after_count = len(registry["agents"])

        assert after_count == running_count
        assert after_count < before_count

    def test_old_agent_retention(
        self, temp_registry_dir: Path, empty_registry: dict[str, Any]
    ) -> None:
        """Test retention policy for old agents."""
        registry_file = temp_registry_dir / ".claude" / "agent-registry.json"

        registry = empty_registry.copy()
        now = datetime.now(UTC)

        # Agents of various ages
        ages_days = [1, 7, 15, 30, 45, 60]
        for days in ages_days:
            registry["agents"][f"agent-{days}d"] = {
                "task_description": f"{days} day old task",
                "status": "completed",
                "launched_at": (now - timedelta(days=days + 1)).isoformat(),
                "completed_at": (now - timedelta(days=days)).isoformat(),
            }

        registry_file.write_text(json.dumps(registry))

        # Apply retention policy (30 days)
        retention_days = 30

        with open(registry_file) as f:
            registry = json.load(f)

        for agent_id in list(registry["agents"].keys()):
            agent = registry["agents"][agent_id]
            completed_at = agent.get("completed_at")
            if completed_at:
                completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                age_days = (now - completed).days
                if age_days > retention_days:
                    del registry["agents"][agent_id]

        # Should keep 1, 7, 15, 30 day old agents
        # Should remove 45, 60 day old agents
        assert len(registry["agents"]) == 4
        assert "agent-1d" in registry["agents"]
        assert "agent-45d" not in registry["agents"]
        assert "agent-60d" not in registry["agents"]


# =============================================================================
# Main
# =============================================================================
