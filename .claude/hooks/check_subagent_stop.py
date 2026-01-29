#!/usr/bin/env python3
"""
SubagentStop Hook Verification
Verifies subagent completion before returning to orchestrator.
Returns JSON response for Claude Code hook system.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared.config import load_config
from shared.logging_utils import get_hook_logger

# Load configuration
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
CONFIG = load_config(PROJECT_DIR)
HOOK_CONFIG = CONFIG.get("hooks", {}).get("check_subagent_stop", {})

# Get config values with fallbacks
OUTPUT_SIZE_THRESHOLD_BYTES = HOOK_CONFIG.get("output_size_threshold_bytes", 204800)
RECENT_WINDOW_MINUTES = HOOK_CONFIG.get("recent_window_minutes", 5)

# Initialize logger
logger = get_hook_logger(__name__)


def auto_summarize_large_output(
    report_file: Path, report: dict[str, Any], project_dir: Path
) -> bool:
    """Auto-summarize large completion reports.

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

    return True


def update_agent_registry(report: dict[str, Any], project_dir: Path) -> None:
    """Update agent registry when agent completes."""
    registry_file = project_dir / ".claude" / "agent-registry.json"

    if not registry_file.exists():
        return

    try:
        with registry_file.open() as f:
            registry = json.load(f)
    except (OSError, json.JSONDecodeError):
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
    except OSError:
        pass


def check_subagent_completion() -> dict[str, bool | str]:
    """
    Check if subagent completed its task properly.

    Subagents should produce completion reports before stopping.
    This hook verifies the subagent didn't abandon work.

    Returns:
        dict: {"ok": True} or {"ok": False, "reason": "explanation"}
    """
    agent_outputs = PROJECT_DIR / ".claude" / "agent-outputs"
    auto_summarized = False

    # Check for recent completion reports (within configured window)
    if agent_outputs.exists():
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
        if recent_reports:
            # Log to stderr for visibility in tests
            print(
                f"Found {len(recent_reports)} recent completion report(s)",
                file=sys.stderr,
            )
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

    # All checks passed (or no recent reports found)
    result: dict[str, bool | str] = {"ok": True}
    if auto_summarized:
        result["auto_summarized"] = True
        result["message"] = "Large output auto-summarized to .claude/summaries/"
    return result


def main() -> None:
    """Main entry point."""
    try:
        # Read stdin for hook context (may include stop_hook_active flag)
        input_data = {}
        if not sys.stdin.isatty():
            import contextlib

            with contextlib.suppress(json.JSONDecodeError):
                input_data = json.load(sys.stdin)

        # CRITICAL: Check stop_hook_active FIRST to prevent infinite loops
        if input_data.get("stop_hook_active"):
            print(json.dumps({"ok": True}))
            sys.exit(0)

        result = check_subagent_completion()
        print(json.dumps(result))
        sys.exit(0 if result["ok"] else 2)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        # Fail safe - allow stop on error
        print(json.dumps({"ok": True}))
        sys.exit(0)


if __name__ == "__main__":
    main()
