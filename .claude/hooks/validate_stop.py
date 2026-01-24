#!/usr/bin/env python3
"""
Stop Hook Validation
Validates task completion before allowing agent to stop.
Consolidates functionality from check_stop.py and check_subagent_stop.py.

Detects whether it's called for main agent or subagent and applies
appropriate validation logic:

Main Agent:
- Check for active work in active_work.json
- Block stop if work is fresh (updated recently)
- Allow stop if work is stale (no update in configured hours)

Subagent:
- Check for recent completion reports
- Auto-summarize large outputs
- Update agent registry
- Block stop if status is "blocked"
- Validate artifacts exist

Version: 1.0.0
Created: 2026-01-19
"""

import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import get_hook_logger, is_stale, load_config

# Load configuration
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
CONFIG = load_config(PROJECT_DIR)

# Get configuration values for main agent
MAIN_AGENT_CONFIG = CONFIG.get("hooks", {}).get("check_stop", {})
MAX_STALE_HOURS = MAIN_AGENT_CONFIG.get("max_stale_hours", 2)

# Get configuration values for subagent
SUBAGENT_CONFIG = CONFIG.get("hooks", {}).get("check_subagent_stop", {})
OUTPUT_SIZE_THRESHOLD_BYTES = SUBAGENT_CONFIG.get("output_size_threshold_bytes", 204800)
RECENT_WINDOW_MINUTES = SUBAGENT_CONFIG.get("recent_window_minutes", 5)

# Logger
logger = get_hook_logger(__name__)


def auto_summarize_large_output(
    report_file: Path, report: dict[str, Any], project_dir: Path
) -> bool:
    """Auto-summarize large completion reports.

    Args:
        report_file: Path to the completion report file
        report: Parsed report dictionary
        project_dir: Project root directory

    Returns:
        True if summarized, False otherwise
    """
    # Check if output is large (threshold from config)
    file_size = report_file.stat().st_size
    if file_size < OUTPUT_SIZE_THRESHOLD_BYTES:
        return False

    # Create summary directory
    summaries_dir = project_dir / ".claude" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Extract key information
    summary_lines = [
        "# Agent Completion Summary",
        "",
        f"**Agent ID:** {report.get('agent_id', 'unknown')}",
        f"**Status:** {report.get('status', 'unknown')}",
        f"**Original size:** {file_size:,} bytes",
        "",
    ]

    # Extract summary field
    if "summary" in report:
        summary_lines.append("## Summary")
        summary_lines.append(f"{report['summary']}")
        summary_lines.append("")

    # Extract key findings
    if "key_findings" in report:
        summary_lines.append("## Key Findings")
        for finding in report["key_findings"]:
            summary_lines.append(f"- {finding}")
        summary_lines.append("")

    # Extract artifacts
    if "artifacts" in report:
        summary_lines.append("## Artifacts")
        for artifact in report["artifacts"]:
            summary_lines.append(f"- {artifact}")
        summary_lines.append("")

    summary_lines.append("---")
    summary_lines.append(f"*Full output auto-summarized due to size (>{file_size // 1000}KB)*")

    # Write summary
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    summary_file = summaries_dir / f"{timestamp}-{report.get('agent_id', 'agent')}-summary.md"
    summary_file.write_text("\n".join(summary_lines))

    logger.info(f"Auto-summarized large output to {summary_file.name}")
    return True


def update_agent_registry(report: dict[str, Any], project_dir: Path) -> None:
    """Update agent registry when agent completes.

    Args:
        report: Completion report dictionary
        project_dir: Project root directory
    """
    registry_file = project_dir / ".claude" / "agent-registry.json"

    if not registry_file.exists():
        return

    try:
        with registry_file.open() as f:
            registry = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read agent registry: {e}")
        return

    agent_id = report.get("agent_id")
    if not agent_id or agent_id not in registry.get("agents", {}):
        return

    # Update agent status
    registry["agents"][agent_id]["status"] = "completed"

    # Decrement running count
    if "metadata" in registry:
        running_count = registry["metadata"].get("agents_running", 0)
        if running_count > 0:
            registry["metadata"]["agents_running"] = running_count - 1

    # Write back
    try:
        with registry_file.open("w") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Updated registry for agent {agent_id}")
    except OSError as e:
        logger.warning(f"Failed to update agent registry: {e}")


def check_main_agent_completion() -> dict[str, bool | str]:
    """Check if main agent completed its task properly.

    Returns:
        dict: {"ok": True} or {"ok": False, "reason": "explanation"}
    """
    # Check for active work that shouldn't be abandoned
    active_work = PROJECT_DIR / ".coordination" / "active_work.json"
    if not active_work.exists():
        return {"ok": True}

    try:
        with active_work.open() as f:
            work = json.load(f)

        # Check if there's any active work
        if not work:
            return {"ok": True}

        # Check if work is stale (no update in configured hours)
        last_update = work.get("last_update")
        if is_stale(last_update, MAX_STALE_HOURS, fallback_path=active_work):
            task_id = work.get("task_id") or work.get("current_task", "unknown")
            logger.warning(
                f"Stale active work detected for task '{task_id}'. "
                f"No update in {MAX_STALE_HOURS}+ hours. Allowing stop."
            )
            return {"ok": True}

        # Fresh work still in progress - block stop
        task_id = work.get("task_id") or work.get("current_task", "unknown")
        return {
            "ok": False,
            "reason": f"Active task '{task_id}' still in progress. Complete or hand off before stopping.",
        }

    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read active_work.json: {e}")
        # Fail-safe: allow stop if we can't read the file
        return {"ok": True}


def check_subagent_completion() -> dict[str, bool | str]:
    """Check if subagent completed its task properly.

    Subagents should produce completion reports before stopping.
    This validates the subagent didn't abandon work.

    Returns:
        dict: {"ok": True} or {"ok": False, "reason": "explanation"}
    """
    agent_outputs = PROJECT_DIR / ".claude" / "agent-outputs"
    auto_summarized = False

    # Check for recent completion reports (within configured window)
    if not agent_outputs.exists():
        return {"ok": True}

    now = datetime.now()
    recent_reports = []
    window_seconds = RECENT_WINDOW_MINUTES * 60

    for report_file in agent_outputs.glob("*-complete.json"):
        try:
            # Check if file was modified within recent window
            mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
            if (now - mtime).total_seconds() < window_seconds:
                with report_file.open() as f:
                    report = json.load(f)
                    recent_reports.append((report_file, report))
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read {report_file}: {e}")
            continue

    # If we found recent reports, validate them
    if not recent_reports:
        return {"ok": True}

    logger.info(f"Found {len(recent_reports)} recent completion report(s)")

    for report_file, report in recent_reports:
        status = report.get("status", "unknown")

        # Auto-summarize large outputs
        if auto_summarize_large_output(report_file, report, PROJECT_DIR):
            auto_summarized = True

        # Update registry
        update_agent_registry(report, PROJECT_DIR)

        # Block if any recent report shows blocked status
        if status == "blocked":
            reason = report.get("blocked_by", "Unknown blocker")
            return {
                "ok": False,
                "reason": f"Task blocked: {reason}. Resolve before stopping.",
            }

        # Warn but allow if needs-review (orchestrator should handle)
        if status == "needs-review":
            logger.warning(
                f"Report {report_file.name} needs review - allowing stop for orchestrator to handle"
            )

        # Validate artifacts exist if specified
        artifacts = report.get("artifacts", [])
        for artifact in artifacts:
            artifact_path = PROJECT_DIR / artifact
            if not artifact_path.exists():
                logger.warning(f"Artifact missing: {artifact}")
                # Don't block on missing artifacts - may be optional

    # All checks passed
    result: dict[str, bool | str] = {"ok": True}
    if auto_summarized:
        result["auto_summarized"] = True
        result["message"] = "Large output auto-summarized to .claude/summaries/"
    return result


def detect_mode(input_data: dict[str, Any]) -> str:
    """Detect if this is a main agent or subagent stop.

    Args:
        input_data: Hook input data from stdin

    Returns:
        "main" or "subagent"
    """
    # Check for indicators in input data
    # Subagents typically have agent_id or subagent context
    if input_data.get("agent_id") or input_data.get("is_subagent"):
        return "subagent"

    # Check for agent-outputs directory with recent reports
    # This is a heuristic - if we see recent completion reports, likely subagent
    agent_outputs = PROJECT_DIR / ".claude" / "agent-outputs"
    if agent_outputs.exists():
        now = datetime.now()
        for report_file in agent_outputs.glob("*-complete.json"):
            try:
                mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
                if (now - mtime).total_seconds() < 60:  # Within last minute
                    return "subagent"
            except OSError:
                continue

    # Default to main agent
    return "main"


def main() -> None:
    """Main entry point."""
    try:
        # Read stdin for hook context
        input_data = {}
        if not sys.stdin.isatty():
            with contextlib.suppress(json.JSONDecodeError):
                input_data = json.load(sys.stdin)

        # CRITICAL: Check stop_hook_active FIRST to prevent infinite loops
        if input_data.get("stop_hook_active"):
            print(json.dumps({"ok": True}))
            sys.exit(0)

        # Detect mode and run appropriate validation
        mode = detect_mode(input_data)
        logger.info(f"Running stop validation in {mode} mode")

        if mode == "subagent":
            result = check_subagent_completion()
        else:
            result = check_main_agent_completion()

        print(json.dumps(result))
        sys.exit(0 if result["ok"] else 2)

    except Exception as e:
        logger.error(f"Unexpected error during stop validation: {e}", exc_info=True)
        # Fail-safe: allow stop on error
        print(json.dumps({"ok": True}))
        sys.exit(0)


if __name__ == "__main__":
    main()
