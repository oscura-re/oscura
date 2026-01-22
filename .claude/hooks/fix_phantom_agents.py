#!/usr/bin/env python3
"""Fix phantom agents - comprehensive cleanup of stale agent registry.

This script:
1. Identifies agents marked "running" but with missing/stale output files
2. Cleans up orphaned task files in /tmp/claude/
3. Validates and repairs agent registry
4. Can run standalone or as a hook

Usage:
    python fix_phantom_agents.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by looking for .claude directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".claude").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find .claude directory")


def load_registry(registry_path: Path) -> dict:
    """Load agent registry."""
    if not registry_path.exists():
        return {"agents": {}}

    try:
        with registry_path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"WARNING: Corrupt registry at {registry_path}, resetting", file=sys.stderr)
        return {"agents": {}}


def save_registry(registry_path: Path, registry: dict, dry_run: bool = False) -> None:
    """Save agent registry."""
    if dry_run:
        print(f"[DRY RUN] Would save registry to {registry_path}")
        return

    with registry_path.open("w") as f:
        json.dump(registry, f, indent=2)


def check_output_file_exists(output_file: str | Path) -> bool:
    """Check if task output file exists and is recent."""
    path = Path(output_file)
    if not path.exists():
        return False

    # Check if file has content or was modified recently (last hour)
    if path.stat().st_size > 0:
        return True

    # Empty file - check if it's recent (within last hour)
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(hours=1)


def is_agent_stale(agent: dict, agent_id: str) -> bool:
    """Determine if agent is stale (>24h old with no recent activity)."""
    # Check if output file exists
    if "output_file" in agent:
        if not check_output_file_exists(agent["output_file"]):
            return True

    # Check age if we have a timestamp
    if "timestamp" in agent:
        try:
            agent_time = datetime.fromisoformat(agent["timestamp"].replace("Z", "+00:00"))
            age = datetime.now() - agent_time.replace(tzinfo=None)
            if age > timedelta(hours=24):
                return True
        except (ValueError, AttributeError):
            pass

    return False


def cleanup_phantom_agents(registry_path: Path, dry_run: bool = False) -> dict:
    """Clean up phantom agents from registry.

    Returns:
        Stats dict with counts of cleaned up agents
    """
    registry = load_registry(registry_path)
    stats = {
        "total_agents": len(registry.get("agents", {})),
        "phantom_agents": 0,
        "stale_running": 0,
        "missing_output": 0,
        "fixed": [],
    }

    agents = registry.get("agents", {})

    for agent_id, agent in list(agents.items()):
        status = agent.get("status", "unknown")

        # Check for agents marked "running" that are actually phantom
        if status == "running":
            is_phantom = False
            reason = []

            # Check 1: Output file missing or empty/old
            if "output_file" in agent:
                if not check_output_file_exists(agent["output_file"]):
                    is_phantom = True
                    reason.append("missing or stale output file")
                    stats["missing_output"] += 1

            # Check 2: Agent is old (>24 hours)
            if is_agent_stale(agent, agent_id):
                is_phantom = True
                reason.append("stale (>24h old)")
                stats["stale_running"] += 1

            if is_phantom:
                stats["phantom_agents"] += 1
                stats["fixed"].append(
                    {
                        "agent_id": agent_id,
                        "task": agent.get("task", "unknown"),
                        "reason": ", ".join(reason),
                    }
                )

                if dry_run:
                    print(f"[DRY RUN] Would mark agent {agent_id} as stale: {reason}")
                else:
                    # Mark as stale instead of deleting (preserves history)
                    agent["status"] = "stale"
                    agent["cleaned_at"] = datetime.now().isoformat()
                    agent["cleanup_reason"] = ", ".join(reason)

    # Save updated registry
    save_registry(registry_path, registry, dry_run)

    return stats


def cleanup_orphaned_task_files(task_dir: Path, dry_run: bool = False) -> int:
    """Clean up orphaned task output files.

    Returns:
        Number of files cleaned
    """
    if not task_dir.exists():
        return 0

    cleaned = 0
    cutoff_time = datetime.now() - timedelta(hours=24)

    for task_file in task_dir.glob("*.output"):
        # Only clean up old, empty files
        if task_file.stat().st_size == 0:
            mtime = datetime.fromtimestamp(task_file.stat().st_mtime)
            if mtime < cutoff_time:
                if dry_run:
                    print(f"[DRY RUN] Would delete old empty file: {task_file.name}")
                else:
                    task_file.unlink()
                cleaned += 1

    return cleaned


def validate_registry(registry_path: Path, dry_run: bool = False) -> dict:
    """Validate registry structure and fix issues.

    Returns:
        Validation stats
    """
    registry = load_registry(registry_path)
    stats = {"valid": True, "issues": [], "fixed": []}

    # Check required fields
    if "agents" not in registry:
        stats["valid"] = False
        stats["issues"].append("Missing 'agents' key")
        registry["agents"] = {}
        stats["fixed"].append("Added missing 'agents' key")

    # Check each agent has required fields
    for agent_id, agent in registry.get("agents", {}).items():
        if "status" not in agent:
            stats["issues"].append(f"Agent {agent_id} missing status")
            agent["status"] = "unknown"
            stats["fixed"].append(f"Added default status to {agent_id}")

    if stats["fixed"] and not dry_run:
        save_registry(registry_path, registry, dry_run)

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix phantom agents in registry")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    try:
        project_root = find_project_root()
    except FileNotFoundError:
        print("ERROR: Not in an Oscura project directory", file=sys.stderr)
        sys.exit(1)

    registry_path = project_root / ".claude" / "agent-registry.json"
    task_dir = Path(f"/tmp/claude/-{project_root.as_posix().replace('/', '-')}/tasks")

    print("=" * 70)
    print("PHANTOM AGENT CLEANUP")
    print("=" * 70)
    print(f"Project: {project_root}")
    print(f"Registry: {registry_path}")
    print(f"Task dir: {task_dir}")
    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")
    print()

    # Step 1: Validate registry structure
    print("Step 1: Validating registry structure...")
    validation = validate_registry(registry_path, args.dry_run)
    if validation["issues"]:
        print(f"  Found {len(validation['issues'])} issues:")
        for issue in validation["issues"]:
            print(f"    - {issue}")
        if validation["fixed"]:
            print(f"  Fixed {len(validation['fixed'])} issues")
    else:
        print("  Registry structure is valid")
    print()

    # Step 2: Clean up phantom agents
    print("Step 2: Cleaning phantom agents...")
    stats = cleanup_phantom_agents(registry_path, args.dry_run)
    print(f"  Total agents in registry: {stats['total_agents']}")
    print(f"  Phantom agents found: {stats['phantom_agents']}")
    if stats["phantom_agents"] > 0:
        print(f"    - Missing/stale output files: {stats['missing_output']}")
        print(f"    - Stale (>24h): {stats['stale_running']}")
        print("\n  Fixed agents:")
        for fixed in stats["fixed"]:
            print(f"    - {fixed['agent_id']}: {fixed['task']} ({fixed['reason']})")
    print()

    # Step 3: Clean up orphaned task files
    print("Step 3: Cleaning orphaned task files...")
    cleaned_files = cleanup_orphaned_task_files(task_dir, args.dry_run)
    print(f"  Cleaned {cleaned_files} old empty task files")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN - No actual changes made")
    else:
        print(f"✓ Fixed {stats['phantom_agents']} phantom agents")
        print(f"✓ Cleaned {cleaned_files} orphaned files")
        print(f"✓ Registry validated and repaired")
    print()

    if not args.dry_run and stats["phantom_agents"] > 0:
        print("Registry has been cleaned. Phantom agents marked as 'stale'.")
        print("You can now launch new agents without hitting limits.")


if __name__ == "__main__":
    main()
