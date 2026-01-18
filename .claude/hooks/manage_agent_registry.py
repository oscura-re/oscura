#!/usr/bin/env python3
"""
Agent Registry Manager
Handles registration, status updates, and recovery for parallel agent orchestration
Version: 2.0.0 (2025-12-25)

Enhancements in 2.0.0:
- Added summary generation for agent outputs
- Added metrics tracking (batch counter, recovery actions)
- Added output validation
- Added archive manifest generation
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
BACKUP_FILE = PROJECT_DIR / ".claude" / "agent-registry.backup.json"
SUMMARIES_DIR = get_absolute_path("claude.coordination.root", PROJECT_DIR) / "summaries"
AGENT_OUTPUTS_DIR = get_absolute_path("claude.outputs.root", PROJECT_DIR)
METRICS_FILE = get_absolute_path("claude.hooks", PROJECT_DIR) / "orchestration-metrics.json"

# Ensure directories exist
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)


def load_registry() -> dict[str, Any]:
    """Load agent registry from file."""
    if not REGISTRY_FILE.exists():
        now = datetime.now(UTC).isoformat()
        return {
            "version": "1.0.0",
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

    try:
        with open(REGISTRY_FILE) as f:
            registry: dict[str, Any] = json.load(f)
            return registry
    except json.JSONDecodeError:
        print(f"Error: Corrupted registry file: {REGISTRY_FILE}", file=sys.stderr)
        if BACKUP_FILE.exists():
            print(f"Restoring from backup: {BACKUP_FILE}", file=sys.stderr)
            with open(BACKUP_FILE) as f:
                backup_registry: dict[str, Any] = json.load(f)
                return backup_registry
        raise


def save_registry(registry: dict[str, Any]) -> None:
    """Save registry to file with backup."""
    # Backup current registry
    if REGISTRY_FILE.exists():
        REGISTRY_FILE.rename(BACKUP_FILE)

    # Update timestamp
    registry["last_updated"] = datetime.now(UTC).isoformat()

    # Write new registry
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_agent(
    agent_id: str,
    task_description: str,
    config: dict[str, Any] | None = None,
) -> None:
    """Register a new agent."""
    registry = load_registry()

    registry["agents"][agent_id] = {
        "task_description": task_description,
        "status": "running",
        "launched_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "config": config or {},
        "output_file": None,
        "summary_file": None,
        "deliverables": [],
    }

    # Update metadata
    registry["metadata"]["total_agents_launched"] += 1
    registry["metadata"]["agents_running"] += 1

    save_registry(registry)
    print(f"Registered agent {agent_id}: {task_description}")


def update_agent_status(
    agent_id: str,
    status: str,
    output_file: str | None = None,
    summary_file: str | None = None,
    deliverables: list[str] | None = None,
) -> None:
    """Update agent status."""
    registry = load_registry()

    if agent_id not in registry["agents"]:
        print(f"Warning: Agent {agent_id} not in registry", file=sys.stderr)
        return

    agent = registry["agents"][agent_id]
    old_status = agent["status"]
    agent["status"] = status

    if status == "completed":
        agent["completed_at"] = datetime.now(UTC).isoformat()
        if old_status == "running":
            registry["metadata"]["agents_running"] -= 1
            registry["metadata"]["agents_completed"] += 1

    if status == "failed":
        if old_status == "running":
            registry["metadata"]["agents_running"] -= 1
            registry["metadata"]["agents_failed"] += 1

    if output_file:
        agent["output_file"] = output_file
    if summary_file:
        agent["summary_file"] = summary_file
    if deliverables:
        agent["deliverables"] = deliverables

    save_registry(registry)
    print(f"Updated agent {agent_id}: {old_status} → {status}")


def get_agent_status(agent_id: str) -> dict[str, Any] | None:
    """Get status of a specific agent."""
    registry = load_registry()
    agent_data: dict[str, Any] | None = registry["agents"].get(agent_id)
    return agent_data


def list_agents(status_filter: str | None = None) -> list[dict[str, Any]]:
    """List all agents, optionally filtered by status."""
    registry = load_registry()
    agents = []

    for agent_id, agent_data in registry["agents"].items():
        if status_filter is None or agent_data["status"] == status_filter:
            agents.append({"id": agent_id, **agent_data})

    return agents


def cleanup_old_agents(days: int = 30) -> None:
    """Remove agents older than specified days."""
    registry = load_registry()
    now = datetime.now(UTC)
    cleaned = 0

    for agent_id in list(registry["agents"].keys()):
        agent = registry["agents"][agent_id]
        completed_at = agent.get("completed_at")

        if completed_at:
            try:
                completed_date = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                if completed_date.tzinfo is None:
                    age_days = (now.replace(tzinfo=None) - completed_date).days
                else:
                    age_days = (now - completed_date).days

                if age_days > days:
                    del registry["agents"][agent_id]
                    cleaned += 1
            except (ValueError, TypeError):
                # Skip agents with invalid dates
                pass

    if cleaned > 0:
        registry["metadata"]["last_cleanup"] = datetime.now(UTC).isoformat()
        save_registry(registry)
        print(f"Cleaned up {cleaned} old agents (>{days} days)")


def compact_registry() -> dict[str, int]:
    """
    Compact the registry by removing all completed agents.
    Returns count of agents removed.
    """
    registry = load_registry()
    removed = 0
    kept_running = 0

    for agent_id in list(registry["agents"].keys()):
        agent = registry["agents"][agent_id]
        status = agent.get("status", "unknown")

        if status in ("completed", "failed"):
            del registry["agents"][agent_id]
            removed += 1
        elif status == "running":
            kept_running += 1

    if removed > 0:
        registry["metadata"]["last_cleanup"] = datetime.now(UTC).isoformat()
        save_registry(registry)

    return {"removed": removed, "kept_running": kept_running}


def complete_agent(agent_id: str, output_file: str | None = None) -> bool:
    """Mark an agent as completed. Convenience wrapper around update_agent_status."""
    registry = load_registry()
    if agent_id not in registry["agents"]:
        print(f"Warning: Agent {agent_id} not in registry", file=sys.stderr)
        return False

    update_agent_status(agent_id, "completed", output_file=output_file)
    return True


def get_summary() -> dict[str, Any]:
    """Get registry summary."""
    registry = load_registry()
    return {
        "total_agents": registry["metadata"]["total_agents_launched"],
        "running": registry["metadata"]["agents_running"],
        "completed": registry["metadata"]["agents_completed"],
        "failed": registry["metadata"]["agents_failed"],
        "last_updated": registry["last_updated"],
    }


# =============================================================================
# Summary Generation (P1 Enhancement)
# =============================================================================


def generate_summary(agent_id: str, output_file: str | None = None) -> str | None:
    """
    Generate a concise markdown summary from agent output.

    Returns the path to the generated summary file, or None if generation failed.
    """
    registry = load_registry()
    agent = registry["agents"].get(agent_id)

    if not agent:
        print(f"Warning: Agent {agent_id} not in registry", file=sys.stderr)
        return None

    # Find output file
    output_path = None
    if output_file:
        output_path = Path(output_file)
    elif agent.get("output_file"):
        output_path = Path(agent["output_file"])
    else:
        # Try to find by agent_id pattern in agent-outputs
        for f in AGENT_OUTPUTS_DIR.glob("*-complete.json"):
            if agent_id in f.name or agent_id[:7] in f.name:
                output_path = f
                break

    if not output_path or not output_path.exists():
        print(f"Warning: No output file found for agent {agent_id}", file=sys.stderr)
        return None

    # Load and parse output
    try:
        output_data = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading output file: {e}", file=sys.stderr)
        return None

    # Extract key information for summary
    summary_lines = [
        f"# Agent Summary: {agent_id[:7]}",
        "",
        f"**Task**: {agent.get('task_description', 'Unknown')}",
        f"**Status**: {agent.get('status', 'Unknown')}",
        f"**Completed**: {agent.get('completed_at', 'N/A')}",
        "",
        "## Key Information",
        "",
    ]

    # Extract deliverables/artifacts
    deliverables = agent.get("deliverables", [])
    if deliverables:
        summary_lines.append("### Deliverables")
        for d in deliverables[:10]:  # Limit to 10
            summary_lines.append(f"- {d}")
        summary_lines.append("")

    # Extract status from output if present
    if isinstance(output_data, dict):
        if "status" in output_data:
            summary_lines.append("### Output Status")
            summary_lines.append(f"- {output_data['status']}")
            summary_lines.append("")

        if "files_created" in output_data:
            summary_lines.append("### Files Created")
            for f in output_data["files_created"][:10]:
                summary_lines.append(f"- `{f}`")
            summary_lines.append("")

        if "key_decisions" in output_data:
            summary_lines.append("### Key Decisions")
            for d in output_data["key_decisions"][:5]:
                summary_lines.append(f"- {d}")
            summary_lines.append("")

    # Add output file reference
    summary_lines.extend(
        [
            "## Source",
            f"- Output file: `{output_path.name}`",
            f"- Output size: {output_path.stat().st_size} bytes",
        ]
    )

    # Write summary
    summary_path = SUMMARIES_DIR / f"{agent_id}.md"
    summary_content = "\n".join(summary_lines)

    summary_path.write_text(summary_content)

    # Update registry with summary path
    agent["summary_file"] = str(summary_path)
    save_registry(registry)

    print(f"Generated summary: {summary_path}")
    return str(summary_path)


def generate_all_summaries() -> dict[str, Any]:
    """Generate summaries for all completed agents that don't have one."""
    registry = load_registry()
    generated = 0
    skipped = 0
    failed = 0

    for agent_id, agent in registry["agents"].items():
        if agent.get("status") != "completed":
            continue

        if agent.get("summary_file") and Path(agent["summary_file"]).exists():
            skipped += 1
            continue

        result = generate_summary(agent_id)
        if result:
            generated += 1
        else:
            failed += 1

    return {
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "total_completed": registry["metadata"]["agents_completed"],
    }


# =============================================================================
# Metrics Tracking (P2 Enhancement)
# =============================================================================


def load_metrics() -> dict[str, Any]:
    """Load orchestration metrics."""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                metrics: dict[str, Any] = json.load(f)
                return metrics
        except json.JSONDecodeError:
            pass

    return {
        "version": "1.0.0",
        "session_start": None,
        "batches_executed": 0,
        "recovery_actions": 0,
        "compaction_events": 0,
        "context_usage_peak_percent": 0,
        "phase_times": {},
        "agent_times": {},
    }


def save_metrics(metrics: dict[str, Any]) -> None:
    """Save orchestration metrics."""
    metrics["last_updated"] = datetime.now(UTC).isoformat()
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def record_batch_completion(batch_id: str, agent_count: int, duration_seconds: float) -> None:
    """Record completion of a batch."""
    metrics = load_metrics()
    metrics["batches_executed"] += 1
    metrics["phase_times"][batch_id] = {
        "agents": agent_count,
        "duration_seconds": duration_seconds,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    save_metrics(metrics)
    print(f"Recorded batch {batch_id}: {agent_count} agents, {duration_seconds:.1f}s")


def record_recovery_action() -> None:
    """Increment recovery action counter."""
    metrics = load_metrics()
    metrics["recovery_actions"] += 1
    save_metrics(metrics)


def record_compaction_event() -> None:
    """Increment compaction event counter."""
    metrics = load_metrics()
    metrics["compaction_events"] += 1
    save_metrics(metrics)


def update_context_peak(usage_percent: int) -> None:
    """Update peak context usage if higher."""
    metrics = load_metrics()
    if usage_percent > metrics.get("context_usage_peak_percent", 0):
        metrics["context_usage_peak_percent"] = usage_percent
        save_metrics(metrics)


def get_metrics_summary() -> dict[str, Any]:
    """Get current metrics summary."""
    metrics = load_metrics()
    registry = load_registry()

    return {
        "batches_executed": metrics["batches_executed"],
        "recovery_actions": metrics["recovery_actions"],
        "compaction_events": metrics["compaction_events"],
        "context_usage_peak_percent": metrics["context_usage_peak_percent"],
        "agents_launched": registry["metadata"]["total_agents_launched"],
        "agents_completed": registry["metadata"]["agents_completed"],
        "agents_failed": registry["metadata"]["agents_failed"],
        "phase_times": metrics.get("phase_times", {}),
    }


# =============================================================================
# Output Validation (P3 Enhancement)
# =============================================================================


def validate_output(output_file: str) -> dict[str, Any]:
    """Validate an agent output file."""
    path = Path(output_file)
    errors: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "file": str(path),
        "size_bytes": 0,
        "errors": errors,
    }

    if not path.exists():
        errors.append("File does not exist")
        return result

    result["size_bytes"] = path.stat().st_size

    if result["size_bytes"] == 0:
        errors.append("File is empty")
        return result

    try:
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            errors.append("Output is not a JSON object")
            return result

        result["valid"] = True
        result["keys"] = list(data.keys())

    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
    except OSError as e:
        errors.append(f"Read error: {e}")

    return result


def main() -> None:
    """CLI interface."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <command> [args...]")
        print("Commands:")
        print("  register <agent_id> <description>")
        print("  update <agent_id> <status> [output_file] [summary_file]")
        print("  complete <agent_id> [output_file]  - Mark agent as completed")
        print("  status <agent_id>")
        print("  list [status_filter]")
        print("  cleanup [days]")
        print("  compact                            - Remove all completed/failed agents")
        print("  summary")
        print("")
        print("Enhancement Commands (v2.0):")
        print("  generate-summary <agent_id> [output_file]")
        print("  generate-all-summaries")
        print("  validate-output <output_file>")
        print("  metrics")
        print("  record-batch <batch_id> <agent_count> <duration_seconds>")
        print("  record-compaction")
        print("  record-recovery")
        sys.exit(1)

    command = sys.argv[1]

    if command == "register":
        if len(sys.argv) < 4:
            print("Usage: register <agent_id> <description>")
            sys.exit(1)
        register_agent(sys.argv[2], sys.argv[3])

    elif command == "update":
        if len(sys.argv) < 4:
            print("Usage: update <agent_id> <status> [output_file] [summary_file]")
            sys.exit(1)
        update_agent_status(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4] if len(sys.argv) > 4 else None,
            sys.argv[5] if len(sys.argv) > 5 else None,
        )

    elif command == "status":
        if len(sys.argv) < 3:
            print("Usage: status <agent_id>")
            sys.exit(1)
        agent = get_agent_status(sys.argv[2])
        if agent:
            print(json.dumps(agent, indent=2))
        else:
            print(f"Agent {sys.argv[2]} not found")
            sys.exit(1)

    elif command == "list":
        status_filter = sys.argv[2] if len(sys.argv) > 2 else None
        agents = list_agents(status_filter)
        print(json.dumps(agents, indent=2))

    elif command == "complete":
        if len(sys.argv) < 3:
            print("Usage: complete <agent_id> [output_file]")
            sys.exit(1)
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        if complete_agent(sys.argv[2], output_file):
            print(f"Agent {sys.argv[2]} marked as completed")
        else:
            sys.exit(1)

    elif command == "cleanup":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        cleanup_old_agents(days)

    elif command == "compact":
        result = compact_registry()
        print(json.dumps(result, indent=2))

    elif command == "summary":
        summary = get_summary()
        print(json.dumps(summary, indent=2))

    # Enhancement commands (v2.0)
    elif command == "generate-summary":
        if len(sys.argv) < 3:
            print("Usage: generate-summary <agent_id> [output_file]")
            sys.exit(1)
        output_file_arg: str | None = sys.argv[3] if len(sys.argv) > 3 else None
        summary_result = generate_summary(sys.argv[2], output_file_arg)
        if summary_result:
            print(json.dumps({"ok": True, "summary_file": summary_result}))
        else:
            print(json.dumps({"ok": False, "error": "Failed to generate summary"}))
            sys.exit(1)

    elif command == "generate-all-summaries":
        result = generate_all_summaries()
        print(json.dumps(result, indent=2))

    elif command == "validate-output":
        if len(sys.argv) < 3:
            print("Usage: validate-output <output_file>")
            sys.exit(1)
        result = validate_output(sys.argv[2])
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["valid"] else 1)

    elif command == "metrics":
        result = get_metrics_summary()
        print(json.dumps(result, indent=2))

    elif command == "record-batch":
        if len(sys.argv) < 5:
            print("Usage: record-batch <batch_id> <agent_count> <duration_seconds>")
            sys.exit(1)
        record_batch_completion(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))

    elif command == "record-compaction":
        record_compaction_event()
        print("Recorded compaction event")

    elif command == "record-recovery":
        record_recovery_action()
        print("Recorded recovery action")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
