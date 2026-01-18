#!/usr/bin/env python3
"""
PreToolUse hook for Task tool - enforces agent limits.

This hook is called BEFORE every Task tool invocation and can block
the launch if too many agents are already running.

Version: 1.0.0
Created: 2025-12-30

Enforcement Rules:
1. Maximum 2 agents running simultaneously (configurable)
2. Block new launches until running count drops
3. Provide clear feedback on why blocked
4. Log all enforcement actions

Integration:
- Called via PreToolUse hook in settings.json
- Reads state from agent-registry.json
- Returns JSON with decision (allow/block)
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared.paths import get_absolute_path

# Configuration
PROJECT_DIR = Path(os.getenv("CLAUDE_PROJECT_DIR", "."))
REGISTRY_FILE = PROJECT_DIR / ".claude" / "agent-registry.json"
CONFIG_FILE = get_absolute_path("claude.root", PROJECT_DIR) / "config.yaml"
LOG_FILE = get_absolute_path("claude.hooks", PROJECT_DIR) / "enforcement.log"
METRICS_FILE = get_absolute_path("claude.hooks", PROJECT_DIR) / "orchestration-metrics.json"

# Default limits (can be overridden by config)
DEFAULT_MAX_RUNNING = 2
DEFAULT_MAX_BATCH_SIZE = 2


def log_message(level: str, message: str) -> None:
    """Log enforcement actions."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")


def load_registry() -> dict[str, Any]:
    """Load agent registry."""
    if not REGISTRY_FILE.exists():
        return {"agents": {}, "metadata": {"agents_running": 0}}

    try:
        with open(REGISTRY_FILE) as f:
            registry: dict[str, Any] = json.load(f)
            return registry
    except (OSError, json.JSONDecodeError) as e:
        log_message("ERROR", f"Failed to load registry: {e}")
        return {"agents": {}, "metadata": {"agents_running": 0}}


def load_config() -> dict[str, Any]:
    """Load orchestration config for limits."""
    if not CONFIG_FILE.exists():
        return {}

    try:
        import yaml

        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: parse YAML manually for key values
        try:
            with open(CONFIG_FILE) as f:
                content = f.read()
                # Simple extraction of max_batch_size
                for line in content.split("\n"):
                    if "max_batch_size:" in line:
                        try:
                            return {"swarm": {"max_batch_size": int(line.split(":")[-1].strip())}}
                        except ValueError:
                            pass
        except OSError:
            pass
    except Exception as e:
        log_message("WARNING", f"Failed to load config: {e}")

    return {}


def count_running_agents(registry: dict[str, Any]) -> int:
    """Count agents currently in 'running' status."""
    # First try metadata (faster)
    metadata_count = registry.get("metadata", {}).get("agents_running", 0)

    # Verify by counting actual running agents
    actual_count = sum(
        1 for agent in registry.get("agents", {}).values() if agent.get("status") == "running"
    )

    # If mismatch, log warning and use actual count
    if metadata_count != actual_count:
        log_message(
            "WARNING", f"Registry count mismatch: metadata={metadata_count}, actual={actual_count}"
        )

    return actual_count


def get_running_agent_details(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Get details of currently running agents."""
    running = []
    for agent_id, agent_data in registry.get("agents", {}).items():
        if agent_data.get("status") == "running":
            running.append(
                {
                    "id": agent_id[:8],  # Shortened ID
                    "task": agent_data.get("task_description", "unknown")[:50],
                    "launched": agent_data.get("launched_at", "unknown"),
                }
            )
    return running


def get_max_running_limit(config: dict[str, Any]) -> int:
    """Get maximum running agents limit from config."""
    # Check orchestration.agents config (config.yaml v4.0.0+)
    orchestration_config = config.get("orchestration", {})
    agents_config = orchestration_config.get("agents", {})
    max_concurrent = agents_config.get("max_concurrent")
    max_batch = agents_config.get("max_batch_size")

    # Fallback: Check legacy swarm config (orchestration-config.yaml)
    if max_concurrent is None and max_batch is None:
        swarm_config = config.get("swarm", {})
        max_concurrent = swarm_config.get("max_parallel_agents")
        max_batch = swarm_config.get("max_batch_size")

    # Use the most restrictive limit
    limits = [DEFAULT_MAX_RUNNING]
    if max_concurrent is not None:
        limits.append(int(max_concurrent))
    if max_batch is not None:
        limits.append(int(max_batch))

    return min(limits)


def update_metrics(action: str, running_count: int) -> None:
    """Update enforcement metrics."""
    try:
        metrics = {}
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                metrics = json.load(f)

        if "enforcement" not in metrics:
            metrics["enforcement"] = {
                "total_checks": 0,
                "total_blocks": 0,
                "total_allows": 0,
            }

        metrics["enforcement"]["total_checks"] += 1
        if action == "block":
            metrics["enforcement"]["total_blocks"] += 1
        else:
            metrics["enforcement"]["total_allows"] += 1

        metrics["enforcement"]["last_check"] = datetime.now(UTC).isoformat()
        metrics["enforcement"]["last_running_count"] = running_count

        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        log_message("WARNING", f"Failed to update metrics: {e}")


def register_new_agent(registry: dict[str, Any], task_info: str) -> None:
    """Pre-register the agent that's about to be launched."""
    # Note: This is a placeholder - the actual agent_id isn't known yet
    # The full registration happens after Task returns the ID
    # This just updates the running count preemptively


def main() -> None:
    """Main enforcement logic."""
    # Load state
    registry = load_registry()
    config = load_config()

    # Get limits
    max_running = get_max_running_limit(config)

    # Count running agents
    running_count = count_running_agents(registry)
    running_details = get_running_agent_details(registry)

    # Make decision
    if running_count >= max_running:
        # BLOCK - too many agents running
        log_message("BLOCK", f"Blocked Task: {running_count}/{max_running} agents running")
        update_metrics("block", running_count)

        # Build informative message
        running_info = ""
        if running_details:
            running_info = "\nCurrently running:\n" + "\n".join(
                f"  - {a['id']}: {a['task']}" for a in running_details[:5]
            )

        result = {
            "decision": "block",
            "reason": (
                f"Agent limit reached: {running_count}/{max_running} agents already running. "
                f"Wait for agents to complete or retrieve their outputs first.{running_info}"
            ),
            "running_agents": running_count,
            "max_agents": max_running,
            "suggestion": "Use TaskOutput to retrieve completed agent results before launching new agents.",
        }

        print(json.dumps(result))
        sys.exit(1)

    else:
        # ALLOW - under limit
        log_message("ALLOW", f"Allowed Task: {running_count}/{max_running} agents running")
        update_metrics("allow", running_count)

        result = {
            "decision": "allow",
            "running_agents": running_count,
            "max_agents": max_running,
            "slots_available": max_running - running_count,
        }

        print(json.dumps(result))
        sys.exit(0)


if __name__ == "__main__":
    main()
