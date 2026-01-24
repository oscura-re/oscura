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
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import get_hook_logger, load_config, load_registry, save_registry

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(REPO_ROOT)))

AGENT_OUTPUTS_DIR = PROJECT_DIR / ".claude" / "agent-outputs"
SUMMARIES_DIR = PROJECT_DIR / ".claude" / "summaries"

# Load configuration from config.yaml
config = load_config(PROJECT_DIR)
hook_config = config.get("hooks", {}).get("cleanup_stale_agents", {})

# Thresholds from config.yaml (with fallbacks)
STALE_THRESHOLD_HOURS = hook_config.get("stale_threshold_hours", 24)
ACTIVITY_CHECK_HOURS = hook_config.get("activity_check_hours", 1)
MAX_AGE_DAYS = hook_config.get("max_age_days", 30)

# =============================================================================
# Logging Setup
# =============================================================================

logger = get_hook_logger(__name__)


# =============================================================================
# Registry Operations (imported from shared utilities)
# =============================================================================
# load_registry(), save_registry() are now imported from shared.registry


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
    registry = load_registry(PROJECT_DIR)
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
            save_registry(registry, PROJECT_DIR)

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
