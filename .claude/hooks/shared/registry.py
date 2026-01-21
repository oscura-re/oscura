#!/usr/bin/env python3
"""
Shared agent registry utilities for Claude hooks.

Consolidates registry operations that were duplicated across 4+ hooks.
Provides atomic operations and consistent error handling.

Version: 1.0.0
Created: 2026-01-19
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_registry_path(project_dir: Path | None = None) -> Path:
    """Get path to agent registry file.

    Args:
        project_dir: Project root directory

    Returns:
        Path to agent-registry.json
    """
    if project_dir is None:
        project_dir = Path(os.getenv("CLAUDE_PROJECT_DIR", "."))

    return project_dir / ".claude" / "agent-registry.json"


def load_registry(project_dir: Path | None = None) -> dict[str, Any]:
    """Load agent registry with error handling.

    Args:
        project_dir: Project root directory

    Returns:
        Registry dict with structure:
        {
            "agents": {agent_id: agent_data, ...},
            "metadata": {"agents_running": int, "last_updated": str}
        }

    If registry doesn't exist or is corrupted, returns empty structure.
    """
    registry_file = get_registry_path(project_dir)

    # Return empty if doesn't exist
    if not registry_file.exists():
        return {
            "agents": {},
            "metadata": {
                "agents_running": 0,
                "last_updated": datetime.now(UTC).isoformat(),
            },
        }

    try:
        with open(registry_file) as f:
            registry = json.load(f)

        # Validate structure
        if not isinstance(registry, dict):
            raise ValueError("Registry must be a dict")

        if "agents" not in registry:
            registry["agents"] = {}

        if "metadata" not in registry:
            registry["metadata"] = {
                "agents_running": 0,
                "last_updated": datetime.now(UTC).isoformat(),
            }

        return registry

    except (json.JSONDecodeError, ValueError, OSError) as e:
        # Try to load backup
        backup_file = registry_file.with_suffix(".json.bak")
        if backup_file.exists():
            try:
                with open(backup_file) as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception:
                pass

        # Return empty structure if both fail
        return {
            "agents": {},
            "metadata": {
                "agents_running": 0,
                "last_updated": datetime.now(UTC).isoformat(),
                "error": f"Failed to load registry: {e}",
            },
        }


def save_registry(registry: dict[str, Any], project_dir: Path | None = None) -> bool:
    """Save agent registry atomically with backup.

    Args:
        registry: Registry dict to save
        project_dir: Project root directory

    Returns:
        True if saved successfully, False otherwise
    """
    registry_file = get_registry_path(project_dir)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    # Update metadata
    if "metadata" not in registry:
        registry["metadata"] = {}

    registry["metadata"]["last_updated"] = datetime.now(UTC).isoformat()

    try:
        # Backup existing registry
        if registry_file.exists():
            backup_file = registry_file.with_suffix(".json.bak")
            shutil.copy2(registry_file, backup_file)

        # Atomic write using temp file + rename
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=registry_file.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(registry, tmp, indent=2)
            tmp_path = Path(tmp.name)

        # Atomic replace
        tmp_path.replace(registry_file)
        return True

    except (OSError, TypeError) as e:
        # Log error but don't crash
        print(f"Warning: Failed to save registry: {e}", file=sys.stderr)
        return False


def count_running_agents(registry: dict[str, Any]) -> int:
    """Count agents currently in 'running' status.

    Args:
        registry: Registry dict

    Returns:
        Number of running agents

    Note:
        Verifies metadata count matches actual count and logs mismatch.
    """
    # Count actual running agents
    actual_count = sum(
        1 for agent in registry.get("agents", {}).values() if agent.get("status") == "running"
    )

    # Check metadata consistency
    metadata_count = registry.get("metadata", {}).get("agents_running", 0)

    if metadata_count != actual_count:
        # Update metadata to match reality
        if "metadata" not in registry:
            registry["metadata"] = {}
        registry["metadata"]["agents_running"] = actual_count

    return actual_count


def update_agent_status(
    agent_id: str,
    status: str,
    project_dir: Path | None = None,
    **extra_fields: Any,
) -> bool:
    """Update agent status in registry.

    Args:
        agent_id: Agent ID
        status: New status (running, completed, failed, stale)
        project_dir: Project root directory
        **extra_fields: Additional fields to update

    Returns:
        True if updated successfully
    """
    registry = load_registry(project_dir)

    # Find agent in registry
    if agent_id not in registry["agents"]:
        return False

    # Update status
    registry["agents"][agent_id]["status"] = status
    registry["agents"][agent_id]["last_updated"] = datetime.now(UTC).isoformat()

    # Update extra fields
    for key, value in extra_fields.items():
        registry["agents"][agent_id][key] = value

    # Update running count if status changed to/from running
    old_status = registry["agents"][agent_id].get("status")
    if status == "running" and old_status != "running":
        count_running_agents(registry)  # Recalculates and updates metadata
    elif status != "running" and old_status == "running":
        count_running_agents(registry)

    return save_registry(registry, project_dir)


def register_agent(
    agent_id: str,
    task_description: str,
    project_dir: Path | None = None,
    **metadata: Any,
) -> bool:
    """Register new agent in registry.

    Args:
        agent_id: Unique agent ID
        task_description: Description of task
        project_dir: Project root directory
        **metadata: Additional metadata fields

    Returns:
        True if registered successfully
    """
    registry = load_registry(project_dir)

    # Create agent entry
    agent_data = {
        "id": agent_id,
        "task_description": task_description,
        "status": "running",
        "launched_at": datetime.now(UTC).isoformat(),
        "last_updated": datetime.now(UTC).isoformat(),
        **metadata,
    }

    registry["agents"][agent_id] = agent_data

    # Update running count
    count_running_agents(registry)

    return save_registry(registry, project_dir)


def remove_agent(agent_id: str, project_dir: Path | None = None) -> bool:
    """Remove agent from registry.

    Args:
        agent_id: Agent ID to remove
        project_dir: Project root directory

    Returns:
        True if removed successfully
    """
    registry = load_registry(project_dir)

    if agent_id not in registry["agents"]:
        return False

    del registry["agents"][agent_id]

    # Update running count
    count_running_agents(registry)

    return save_registry(registry, project_dir)


def get_stale_agents(
    registry: dict[str, Any],
    stale_hours: int = 24,
) -> list[tuple[str, dict[str, Any]]]:
    """Find agents that haven't been updated recently.

    Args:
        registry: Registry dict
        stale_hours: Hours of inactivity to consider stale

    Returns:
        List of (agent_id, agent_data) tuples for stale agents
    """
    from datetime import timedelta

    stale_threshold = datetime.now(UTC) - timedelta(hours=stale_hours)
    stale_agents = []

    for agent_id, agent_data in registry.get("agents", {}).items():
        last_updated_str = agent_data.get("last_updated", agent_data.get("launched_at"))

        if not last_updated_str:
            continue

        try:
            last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))

            # Make timezone-aware if naive
            if last_updated.tzinfo is None:
                import zoneinfo

                last_updated = last_updated.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

            if last_updated < stale_threshold:
                stale_agents.append((agent_id, agent_data))

        except (ValueError, AttributeError):
            # Can't parse timestamp - consider it stale
            stale_agents.append((agent_id, agent_data))

    return stale_agents


def cleanup_old_agents(registry: dict[str, Any], max_age_days: int = 30) -> int:
    """Remove completed agents older than max_age_days.

    Args:
        registry: Registry dict
        max_age_days: Age in days for removal

    Returns:
        Number of agents removed
    """
    from datetime import timedelta

    threshold = datetime.now(UTC) - timedelta(days=max_age_days)
    removed = 0

    agents_to_remove = []

    for agent_id, agent_data in registry.get("agents", {}).items():
        # Only remove completed/failed agents
        status = agent_data.get("status")
        if status not in ("completed", "failed"):
            continue

        # Check age
        launched_str = agent_data.get("launched_at")
        if not launched_str:
            continue

        try:
            launched = datetime.fromisoformat(launched_str.replace("Z", "+00:00"))

            # Make timezone-aware if naive
            if launched.tzinfo is None:
                import zoneinfo

                launched = launched.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))

            if launched < threshold:
                agents_to_remove.append(agent_id)

        except (ValueError, AttributeError):
            pass  # Skip agents with malformed timestamps - they'll be cleaned eventually

    # Remove old agents
    for agent_id in agents_to_remove:
        del registry["agents"][agent_id]
        removed += 1

    return removed
