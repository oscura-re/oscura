#!/usr/bin/env python3
"""
Stale Agent Cleanup Hook

Safely cleans up stale agents from the registry, with proper handling for:
- Active agents with recent activity (preserved)
- Long-running agents (checked for activity, not just age)
- Corrupted registry (recovery from backup)
- Race conditions (checks output file modification times)

Version: 2.0.0
Created: 2025-12-25
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(REPO_ROOT)))

REGISTRY_FILE = PROJECT_DIR / ".claude" / "agent-registry.json"
BACKUP_FILE = PROJECT_DIR / ".claude" / "agent-registry.backup.json"
CORRUPTED_FILE = PROJECT_DIR / ".claude" / "agent-registry.corrupted.json"
AGENT_OUTPUTS_DIR = PROJECT_DIR / ".claude" / "agent-outputs"
SUMMARIES_DIR = PROJECT_DIR / ".claude" / "summaries"
LOG_FILE = PROJECT_DIR / ".claude" / "hooks" / "hook.log"

# Thresholds
STALE_THRESHOLD_HOURS = 24  # Consider stale after 24 hours
ACTIVITY_CHECK_HOURS = 1  # Check for activity within last hour
MAX_AGE_DAYS = 30  # Remove completed agents after 30 days

# =============================================================================
# Logging Setup
# =============================================================================

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger("cleanup_stale_agents")


# =============================================================================
# Registry Operations
# =============================================================================


def load_registry() -> dict[str, Any]:
    """Load agent registry with corruption recovery."""
    if not REGISTRY_FILE.exists():
        logger.info("Registry file does not exist, creating empty registry")
        return create_empty_registry()

    try:
        with open(REGISTRY_FILE) as f:
            registry: dict[str, Any] = json.load(f)

        # Validate structure
        if not isinstance(registry.get("agents"), dict):
            raise ValueError("Invalid registry structure: missing 'agents' dict")

        return registry

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Corrupted registry: {e}")

        # Save corrupted file for diagnosis
        if REGISTRY_FILE.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            corrupted_backup = CORRUPTED_FILE.with_suffix(f".{timestamp}.json")
            shutil.copy2(REGISTRY_FILE, corrupted_backup)
            logger.info(f"Saved corrupted registry to {corrupted_backup}")

        # Try to recover from backup
        if BACKUP_FILE.exists():
            logger.info("Attempting recovery from backup")
            try:
                with open(BACKUP_FILE) as f:
                    registry_backup: dict[str, Any] = json.load(f)
                if isinstance(registry_backup.get("agents"), dict):
                    logger.info("Successfully recovered from backup")
                    # Save the recovered registry
                    save_registry(registry_backup)
                    return registry_backup
            except (json.JSONDecodeError, ValueError):
                logger.error("Backup is also corrupted")

        # Create fresh registry
        logger.warning("Creating fresh registry")
        return create_empty_registry()


def create_empty_registry() -> dict[str, Any]:
    """Create a new empty registry structure."""
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


def save_registry(registry: dict[str, Any]) -> None:
    """Save registry with backup."""
    # Backup current registry
    if REGISTRY_FILE.exists():
        shutil.copy2(REGISTRY_FILE, BACKUP_FILE)

    # Update timestamp
    registry["last_updated"] = datetime.now(UTC).isoformat()

    # Write atomically
    temp_file = REGISTRY_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump(registry, f, indent=2)
        temp_file.replace(REGISTRY_FILE)
    except OSError:
        temp_file.unlink(missing_ok=True)
        raise


# =============================================================================
# Activity Detection
# =============================================================================


def get_agent_activity_time(agent_id: str) -> datetime | None:
    """Get the most recent activity time for an agent.

    Checks:
    1. Output file modification time
    2. Summary file modification time
    """
    activity_times: list[datetime] = []

    # Check output files
    for output_file in AGENT_OUTPUTS_DIR.glob("*.json"):
        if agent_id in output_file.name or agent_id[:7] in output_file.name:
            try:
                mtime = datetime.fromtimestamp(output_file.stat().st_mtime, tz=UTC)
                activity_times.append(mtime)
            except OSError:
                continue

    # Check summary files
    summary_file = SUMMARIES_DIR / f"{agent_id}.md"
    if summary_file.exists():
        try:
            mtime = datetime.fromtimestamp(summary_file.stat().st_mtime, tz=UTC)
            activity_times.append(mtime)
        except OSError:
            pass

    return max(activity_times) if activity_times else None


def is_agent_active(agent: dict[str, Any], agent_id: str) -> bool:
    """Check if an agent shows signs of recent activity.

    An agent is considered active if:
    1. It has been updated within the activity check window
    2. Its output files have been modified recently
    """
    now = datetime.now(UTC)
    activity_threshold = now - timedelta(hours=ACTIVITY_CHECK_HOURS)

    # Check output file activity
    activity_time = get_agent_activity_time(agent_id)
    if activity_time and activity_time > activity_threshold:
        logger.debug(f"Agent {agent_id} has recent activity at {activity_time}")
        return True

    # Check if launched recently (might not have output yet)
    launched_at = agent.get("launched_at")
    if launched_at:
        try:
            launched = datetime.fromisoformat(launched_at.replace("Z", "+00:00"))
            # If launched within activity window, consider active
            if launched > activity_threshold:
                logger.debug(f"Agent {agent_id} was recently launched at {launched}")
                return True
        except (ValueError, TypeError):
            pass

    return False


def is_agent_stale(agent: dict[str, Any], agent_id: str) -> bool:
    """Check if an agent is stale (no activity for STALE_THRESHOLD_HOURS).

    An agent is stale if:
    1. It's been running for longer than the stale threshold
    2. AND it shows no recent activity
    """
    if agent.get("status") != "running":
        return False

    now = datetime.now(UTC)
    launched_at = agent.get("launched_at")

    if not launched_at:
        # No launch time = stale
        return True

    try:
        launched = datetime.fromisoformat(launched_at.replace("Z", "+00:00"))
        age = now - launched

        if age < timedelta(hours=STALE_THRESHOLD_HOURS):
            # Not old enough to be stale
            return False

        # Old enough - check for recent activity
        if is_agent_active(agent, agent_id):
            logger.info(f"Agent {agent_id} is old but still active")
            return False

        return True

    except (ValueError, TypeError):
        # Invalid timestamp = stale
        return True


# =============================================================================
# Cleanup Logic
# =============================================================================


def cleanup_stale_agents(dry_run: bool = False) -> dict[str, Any]:
    """Clean up stale agents from registry.

    Args:
        dry_run: If True, report what would be cleaned without modifying

    Returns:
        Summary of cleanup actions
    """
    registry = load_registry()
    now = datetime.now(UTC)

    stale_agents: list[str] = []
    old_completed: list[str] = []
    active_preserved: list[str] = []
    errors: list[str] = []

    for agent_id, agent in list(registry["agents"].items()):
        status = agent.get("status", "unknown")

        try:
            # Check for stale running agents
            if status == "running" and is_agent_stale(agent, agent_id):
                stale_agents.append(agent_id)

            # Check for old completed/failed agents
            elif status in ("completed", "failed"):
                completed_at = agent.get("completed_at")
                if completed_at:
                    try:
                        completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                        age_days = (now - completed).days
                        if age_days > MAX_AGE_DAYS:
                            old_completed.append(agent_id)
                    except (ValueError, TypeError):
                        # Invalid date - mark for cleanup
                        old_completed.append(agent_id)

            # Track active agents being preserved
            elif status == "running" and is_agent_active(agent, agent_id):
                active_preserved.append(agent_id)

        except Exception as e:
            errors.append(f"{agent_id}: {e}")
            logger.error(f"Error processing agent {agent_id}: {e}")

    # Perform cleanup
    if not dry_run:
        # Mark stale agents as failed
        for agent_id in stale_agents:
            registry["agents"][agent_id]["status"] = "failed"
            registry["agents"][agent_id]["completed_at"] = now.isoformat()
            registry["agents"][agent_id]["failure_reason"] = "Marked stale by cleanup hook"
            if registry["metadata"]["agents_running"] > 0:
                registry["metadata"]["agents_running"] -= 1
            registry["metadata"]["agents_failed"] += 1
            logger.info(f"Marked agent {agent_id} as failed (stale)")

        # Remove old completed agents
        for agent_id in old_completed:
            del registry["agents"][agent_id]
            logger.info(f"Removed old agent {agent_id}")

        # Update cleanup timestamp
        registry["metadata"]["last_cleanup"] = now.isoformat()

        # Save
        if stale_agents or old_completed:
            save_registry(registry)

    result = {
        "ok": True,
        "dry_run": dry_run,
        "stale_marked_failed": len(stale_agents),
        "old_removed": len(old_completed),
        "active_preserved": len(active_preserved),
        "errors": len(errors),
        "stale_agents": stale_agents,
        "old_agents": old_completed,
        "preserved_agents": active_preserved,
        "error_details": errors[:5] if errors else [],  # Limit error details
        "timestamp": now.isoformat(),
    }

    logger.info(
        f"Cleanup complete: {len(stale_agents)} stale, {len(old_completed)} old, {len(active_preserved)} preserved"
    )

    return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean up stale agents from registry")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be cleaned without modifying"
    )
    args = parser.parse_args()

    try:
        result = cleanup_stale_agents(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    except Exception as e:
        logger.exception("Cleanup failed")
        print(json.dumps({"ok": False, "error": str(e)}))
        # Exit 0 to not block session start
        sys.exit(0)


if __name__ == "__main__":
    main()
