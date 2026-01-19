#!/usr/bin/env python3
"""
Health Check Hook for SessionStart
Validates orchestration system health at session start.

Version: 1.0.0
Created: 2025-01-09

Checks:
1. Required directories exist
2. Agent registry is valid
3. No stale agents from previous session
4. Checkpoint system functional
5. Disk space adequate

Returns: JSON with health status and recommendations
"""

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared.config import load_config
from shared.logging_utils import get_hook_logger
from shared.paths import PATHS, get_absolute_path

# Load configuration
PROJECT_DIR = Path(os.getenv("CLAUDE_PROJECT_DIR", "."))
CONFIG = load_config(PROJECT_DIR)
HOOK_CONFIG = CONFIG.get("hooks", {}).get("health_check", {})

# Get config values with fallbacks
DISK_SPACE_CRITICAL_PERCENT = HOOK_CONFIG.get("disk_space_critical_percent", 5)
DISK_SPACE_WARNING_PERCENT = HOOK_CONFIG.get("disk_space_warning_percent", 10)

# Initialize logger
logger = get_hook_logger(__name__)


def check_directories() -> tuple[bool, str]:
    """Check that required directories exist."""
    required_dirs = [
        get_absolute_path("claude.root", PROJECT_DIR),
        get_absolute_path("claude.agents", PROJECT_DIR),
        get_absolute_path("claude.hooks", PROJECT_DIR),
        get_absolute_path("claude.outputs.root", PROJECT_DIR),
        get_absolute_path("claude.coordination.root", PROJECT_DIR),
        get_absolute_path("claude.coordination.checkpoints", PROJECT_DIR),
    ]

    missing = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing.append(str(dir_path))

    if len(missing) == 0:
        return True, "All required directories exist"
    else:
        return False, f"Missing directories: {', '.join(missing)}"


def check_agent_registry() -> tuple[bool, str]:
    """Check agent registry validity."""
    registry_file = PROJECT_DIR / ".claude" / "agent-registry.json"

    if not registry_file.exists():
        return True, "No existing registry (fresh start)"

    try:
        with open(registry_file) as f:
            registry = json.load(f)

        # Validate structure
        if not isinstance(registry, dict):
            return False, "Registry is not a valid JSON object"

        required_keys = ["agents", "metadata"]
        missing_keys = [k for k in required_keys if k not in registry]
        if missing_keys:
            return False, f"Registry missing keys: {missing_keys}"

        # Check for stale running agents
        running = sum(
            1 for agent in registry.get("agents", {}).values() if agent.get("status") == "running"
        )

        if running > 0:
            return (
                False,
                f"{running} agents marked as running from previous session (will be cleaned up)",
            )

        return True, f"Registry valid, {len(registry.get('agents', {}))} agents recorded"

    except json.JSONDecodeError:
        return False, "Registry file is corrupted (invalid JSON)"
    except Exception as e:
        return False, f"Error reading registry: {e}"


def check_checkpoints() -> tuple[bool, str]:
    """Check checkpoint system."""
    checkpoint_dir = get_absolute_path("claude.coordination.checkpoints", PROJECT_DIR)

    if not checkpoint_dir.exists():
        return True, "No checkpoint directory (will be created as needed)"

    # Count checkpoints
    checkpoints = list(checkpoint_dir.glob("*"))
    checkpoint_count = len([c for c in checkpoints if c.is_dir()])

    return True, f"{checkpoint_count} checkpoint(s) available"


def check_disk_space() -> tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil

        stats = shutil.disk_usage(PROJECT_DIR)
        free_gb = stats.free / (1024**3)
        percent_free = (stats.free / stats.total) * 100

        if percent_free < DISK_SPACE_CRITICAL_PERCENT:
            return False, f"Critical: Only {free_gb:.1f}GB ({percent_free:.1f}%) free"
        elif percent_free < DISK_SPACE_WARNING_PERCENT:
            return (
                True,
                f"Warning: Only {free_gb:.1f}GB ({percent_free:.1f}%) free (consider cleanup)",
            )
        else:
            return True, f"{free_gb:.1f}GB ({percent_free:.1f}%) available"

    except Exception as e:
        return True, f"Could not check disk space: {e}"


def check_old_outputs() -> tuple[bool, str]:
    """Check for old agent outputs that should be archived."""
    outputs_dir = get_absolute_path("claude.outputs.root", PROJECT_DIR)

    if not outputs_dir.exists():
        return True, "No outputs directory"

    try:
        cutoff = datetime.now(UTC) - timedelta(
            days=PATHS.get("retention", {}).get("agent_outputs", 7)
        )
        old_files = []

        for file in outputs_dir.glob("*.json"):
            if file.stat().st_mtime < cutoff.timestamp():
                old_files.append(file.name)

        if len(old_files) > 10:
            return (
                True,
                f"{len(old_files)} old outputs should be archived (run /cleanup)",
            )
        else:
            return True, f"{len(old_files)} old output(s) to archive"

    except Exception as e:
        return True, f"Could not check outputs: {e}"


def run_health_check() -> dict[str, Any]:
    """Run all health checks."""
    recommendations: list[str] = []
    checks_dict: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "healthy",
        "checks": checks_dict,
        "recommendations": recommendations,
    }

    # Run checks
    checks = {
        "directories": check_directories(),
        "agent_registry": check_agent_registry(),
        "checkpoints": check_checkpoints(),
        "disk_space": check_disk_space(),
        "old_outputs": check_old_outputs(),
    }

    for check_name, (passed, message) in checks.items():
        checks_dict[check_name] = {"passed": passed, "message": message}

        if not passed:
            results["status"] = "degraded"
            recommendations.append(f"{check_name}: {message}")

        if passed:
            logger.info(f"{check_name}: {message}")
        else:
            logger.error(f"{check_name}: {message}")

    # Add general recommendations
    if results["status"] == "healthy":
        recommendations.append("System healthy - no action required")

    return results


def main() -> None:
    """Main entry point."""
    try:
        result = run_health_check()
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if result["status"] == "healthy" else 1)

    except Exception as e:
        error_result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "error",
            "error": str(e),
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        logger.error(f"Health check failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
