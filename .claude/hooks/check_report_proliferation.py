#!/usr/bin/env python3
"""
Report Proliferation Check Hook
Warns when attempting to create reports matching forbidden patterns.

Prevents excessive one-time reports, analyses, and summaries from cluttering
the .claude/ directory. Encourages direct communication instead.

Version: 2.0.0
Created: 2025-12-25
Updated: 2026-01-19 - Fixed to read from stdin (PreToolUse contract)
"""

import json
import os
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import get_hook_logger, load_coding_standards

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(REPO_ROOT)))

# Logger
logger = get_hook_logger(__name__)


def load_forbidden_patterns() -> list[dict[str, str]]:
    """Load forbidden report patterns from coding-standards.yaml."""
    try:
        standards = load_coding_standards(PROJECT_DIR)
        patterns: list[dict[str, str]] = standards.get("report_generation", {}).get(
            "forbidden_reports", []
        )
        if patterns:
            return patterns
    except Exception as e:
        logger.warning(f"Failed to load coding standards: {e}")

    # Fallback patterns if YAML not available
    return [
        {"pattern": "*_AUDIT_*.md", "reason": "Audit results should be communicated directly"},
        {
            "pattern": "*_ANALYSIS_*.md",
            "reason": "Analysis results belong in completion reports",
        },
        {"pattern": "*_SUMMARY.md", "reason": "Use .claude/summaries/ instead"},
        {"pattern": "*_RESULTS.*", "reason": "Results belong in validation reports"},
        {"pattern": "COMPREHENSIVE_*.md", "reason": "One-time reports create clutter"},
        {
            "pattern": "ULTIMATE_*.md",
            "reason": "Superlative naming indicates temporary artifact",
        },
    ]


def check_file_path(file_path: str) -> dict[str, Any]:
    """Check if file path matches forbidden patterns."""
    if not file_path:
        return {"ok": True, "message": "No file path provided"}

    path = Path(file_path)

    # Convert to absolute path if relative
    if not path.is_absolute():
        path = PROJECT_DIR / path

    # Only check files in .claude/ directory (not src/, tests/, etc.)
    if not str(path).startswith(str(PROJECT_DIR / ".claude")):
        return {"ok": True, "message": "File not in .claude/ directory"}

    # Allow specific locations
    allowed_dirs = [
        ".claude/agent-outputs",
        ".claude/summaries",
        ".coordination/checkpoints",
    ]

    for allowed_dir in allowed_dirs:
        if str(path).startswith(str(PROJECT_DIR / allowed_dir)):
            return {"ok": True, "message": f"File in allowed directory: {allowed_dir}"}

    # Check against forbidden patterns
    filename = path.name
    forbidden_patterns = load_forbidden_patterns()

    for pattern_config in forbidden_patterns:
        pattern = pattern_config.get("pattern", "")
        reason = pattern_config.get("reason", "Unknown reason")

        if fnmatch(filename, pattern):
            logger.warning(f"Forbidden report pattern: {filename} matches {pattern}")
            return {
                "ok": False,
                "warning": True,  # Warning, not blocking error
                "pattern": pattern,
                "reason": reason,
                "filename": filename,
                "suggestion": "Communicate results directly or use completion reports instead",
            }

    return {"ok": True, "message": "File does not match forbidden patterns"}


def main() -> None:
    """Main entry point for PreToolUse hook."""
    try:
        # Check for bypass
        if os.environ.get("CLAUDE_BYPASS_HOOKS") == "1":
            print(json.dumps({"ok": True, "bypassed": True}))
            sys.exit(0)

        # Read from stdin (PreToolUse hook contract)
        input_data = json.load(sys.stdin)

        # Extract tool information
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Only check Write, Edit, NotebookEdit tools
        if tool_name not in ("Write", "Edit", "NotebookEdit"):
            print(json.dumps({"ok": True, "message": "Not a file write operation"}))
            sys.exit(0)

        # Extract file path from tool input
        file_path = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")

        if not file_path:
            print(json.dumps({"ok": True, "message": "No file path to check"}))
            sys.exit(0)

        logger.info(f"Checking file path: {file_path}")
        result = check_file_path(file_path)

        print(json.dumps(result, indent=2))

        # Warning only - don't block
        if result.get("warning"):
            logger.warning(f"Warning issued for: {file_path}")
            sys.exit(0)  # Exit 0 even on warning (non-blocking)
        else:
            logger.debug("Check passed")
            sys.exit(0)

    except json.JSONDecodeError as e:
        # Fail-open for informational hook
        logger.error(f"Failed to parse stdin JSON: {e}")
        print(json.dumps({"ok": True, "error": "Invalid JSON input"}))
        sys.exit(0)

    except Exception as e:
        # Fail-open for informational hook
        logger.error(f"Hook failed: {e}")
        print(json.dumps({"ok": True, "error": str(e)}))
        sys.exit(0)


if __name__ == "__main__":
    main()
