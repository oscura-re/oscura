#!/usr/bin/env python3
"""
Stop Hook Verification
Verifies task completion before allowing agent to stop.
Returns JSON response for Claude Code hook system.

Version: 3.0.0
Updated: 2026-01-19 - Migrated to shared utilities
"""

import contextlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import get_hook_logger, load_config

# Load configuration
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
config = load_config(PROJECT_DIR)
hook_config = config.get("hooks", {}).get("check_stop", {})
MAX_STALE_HOURS = hook_config.get("max_stale_hours", 2)

# Logger
logger = get_hook_logger(__name__)


def is_stale(work: dict[str, Any], file_path: Path, max_age_hours: int | None = None) -> bool:
    """
    Check if active work is stale (no update in max_age_hours).

    Args:
        work: Active work dictionary from active_work.json
        file_path: Path to the active_work.json file (for checking mtime)
        max_age_hours: Maximum age in hours before considering stale (default: from config.yaml)

    Returns:
        True if work is stale, False otherwise
    """
    if max_age_hours is None:
        max_age_hours = MAX_STALE_HOURS
    last_update = work.get("last_update")

    # If no last_update field, check file modification time
    if not last_update:
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            now = datetime.now()
            age_hours = (now - mtime).total_seconds() / 3600
            return age_hours > max_age_hours
        except (OSError, ValueError):
            return False  # Can't determine age, assume not stale, block stop

    try:
        # Parse the timestamp
        update_time = datetime.fromisoformat(last_update.replace("Z", "+00:00"))

        # Use naive datetime for consistency if no timezone, aware otherwise
        now = datetime.now() if update_time.tzinfo is None else datetime.now(UTC)

        age_hours = (now - update_time).total_seconds() / 3600
        return age_hours > max_age_hours
    except (ValueError, TypeError):
        return False  # Invalid timestamp = not stale, should block stop


def check_completion() -> dict[str, bool | str]:
    """
    Check if agent completed its task properly.

    Returns:
        dict: {"ok": True} or {"ok": False, "reason": "explanation"}
    """
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))

    # Check for active work that shouldn't be abandoned
    active_work = project_dir / ".coordination" / "active_work.json"
    if active_work.exists():
        try:
            with active_work.open() as f:
                work = json.load(f)

            # Check if there's any active work (any non-empty dict indicates active work)
            if work:
                # Check if work is stale (no update in 2+ hours)
                if is_stale(work, active_work):
                    task_id = work.get("task_id") or work.get("current_task", "unknown")
                    logger.error(
                        f"Stale active work detected for task '{task_id}'. "
                        "No update in 2+ hours. Allowing stop."
                    )
                    # Allow stop for stale work - likely crashed agent
                else:
                    # Fresh work still in progress - block stop
                    task_id = work.get("task_id") or work.get("current_task", "unknown")
                    return {
                        "ok": False,
                        "reason": f"Active task '{task_id}' still in progress. Complete or hand off before stopping.",
                    }
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read active_work.json: {e}")

    # All checks passed
    return {"ok": True}


def main() -> None:
    """Main entry point."""
    try:
        # Read stdin for hook context (may include stop_hook_active flag)
        input_data = {}
        if not sys.stdin.isatty():
            with contextlib.suppress(json.JSONDecodeError):
                input_data = json.load(sys.stdin)

        # CRITICAL: Check stop_hook_active FIRST to prevent infinite loops
        if input_data.get("stop_hook_active"):
            print(json.dumps({"ok": True}))
            sys.exit(0)

        result = check_completion()
        print(json.dumps(result))
        sys.exit(0 if result["ok"] else 2)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Fail safe - allow stop on error
        print(json.dumps({"ok": True}))
        sys.exit(0)


if __name__ == "__main__":
    main()
