#!/usr/bin/env python3
"""
PreToolUse hook for Task tool - enforces agent limits.

This hook is called BEFORE every Task tool invocation and can block
the launch if too many agents are already running.

Version: 2.0.0
Created: 2025-12-30
Updated: 2026-01-19 - Added stdin parsing, uses shared utilities

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
from shared import (
    count_running_agents,
    get_hook_logger,
    get_orchestration_config,
    load_config,
    load_registry,
)
from shared.paths import get_absolute_path

# Configuration
PROJECT_DIR = Path(os.getenv("CLAUDE_PROJECT_DIR", "."))
LOG_FILE = get_absolute_path("claude.hooks", PROJECT_DIR) / "enforcement.log"
METRICS_FILE = get_absolute_path("claude.hooks", PROJECT_DIR) / "orchestration-metrics.json"

# Logger
logger = get_hook_logger(__name__, LOG_FILE)

# Default limits (can be overridden by config)
DEFAULT_MAX_RUNNING = 2


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
    orch_config = get_orchestration_config(config)
    agents_config = orch_config.get("agents", {})
    max_concurrent = agents_config.get("max_concurrent")
    max_batch = agents_config.get("max_batch_size")

    # Fallback: Check legacy swarm config
    if max_concurrent is None and max_batch is None:
        swarm_config = config.get("swarm", {})
        max_concurrent = swarm_config.get("max_parallel_agents")
        max_batch = swarm_config.get("max_batch_size")

    # Use configured limit, or DEFAULT if not configured
    limits = []
    if max_concurrent is not None:
        limits.append(int(max_concurrent))
    if max_batch is not None:
        limits.append(int(max_batch))
    return min(limits) if limits else DEFAULT_MAX_RUNNING


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
        logger.warning(f"Failed to update metrics: {e}")


def register_new_agent(registry: dict[str, Any], task_info: str) -> None:
    """Pre-register the agent that's about to be launched."""
    # Note: This is a placeholder - the actual agent_id isn't known yet
    # The full registration happens after Task returns the ID
    # This just updates the running count preemptively


def main() -> None:
    """Main enforcement logic for PreToolUse hook."""
    try:
        # Read from stdin (PreToolUse hook contract)
        input_data = json.load(sys.stdin)

        # Extract tool information
        tool_name = input_data.get("tool_name", "")

        # Only enforce for Task tool
        if tool_name != "Task":
            print(json.dumps({"decision": "allow", "message": "Not a Task tool"}))
            sys.exit(0)

        # Load state
        registry = load_registry(PROJECT_DIR)
        config = load_config(PROJECT_DIR)

        # Get limits
        max_running = get_max_running_limit(config)

        # Count running agents
        running_count = count_running_agents(registry)
        running_details = get_running_agent_details(registry)

        # Make decision
        if running_count >= max_running:
            # BLOCK - too many agents running
            logger.info(f"BLOCK: {running_count}/{max_running} agents running")
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
            logger.info(f"ALLOW: {running_count}/{max_running} agents running")
            update_metrics("allow", running_count)

            result = {
                "decision": "allow",
                "running_agents": running_count,
                "max_agents": max_running,
                "slots_available": max_running - running_count,
            }

            print(json.dumps(result))
            sys.exit(0)

    except json.JSONDecodeError as e:
        # FAIL CLOSED for enforcement hook - block on parse errors
        logger.error(f"Failed to parse stdin JSON: {e}")
        result = {
            "decision": "block",
            "reason": "Hook failed to parse input - blocking for safety",
            "error": str(e),
        }
        print(json.dumps(result))
        sys.exit(1)

    except Exception as e:
        # FAIL CLOSED for enforcement hook - block on errors
        logger.error(f"Hook failed: {e}", exc_info=True)
        result = {
            "decision": "block",
            "reason": "Hook encountered error - blocking for safety",
            "error": str(e),
        }
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()
