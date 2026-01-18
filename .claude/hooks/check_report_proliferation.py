#!/usr/bin/env python3
"""
Report Proliferation Check Hook
Warns when attempting to create reports matching forbidden patterns.

Prevents excessive one-time reports, analyses, and summaries from cluttering
the .claude/ directory. Encourages direct communication instead.

Version: 1.0.0
Created: 2025-12-25
"""

import json
import os
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(REPO_ROOT)))
LOG_FILE = PROJECT_DIR / ".claude/hooks/hook.log"

# Extract FILE_PATH from environment (set by PreToolUse hook)
FILE_PATH = os.environ.get("FILE_PATH", "")


def log(message: str) -> None:
    """Log to hook log file."""
    from datetime import datetime

    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] REPORT_CHECK: {message}\n")


def load_forbidden_patterns() -> list[dict[str, str]]:
    """Load forbidden report patterns from coding-standards.yaml."""
    try:
        import yaml

        standards_file = PROJECT_DIR / ".claude/coding-standards.yaml"
        with open(standards_file) as f:
            standards = yaml.safe_load(f)
        patterns: list[dict[str, str]] = standards.get("report_generation", {}).get(
            "forbidden_reports", []
        )
        return patterns
    except Exception:
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
            log(f"WARN: Forbidden report pattern: {filename} matches {pattern}")
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
    """Main entry point."""
    # Check for bypass
    if os.environ.get("CLAUDE_BYPASS_HOOKS") == "1":
        print(json.dumps({"ok": True, "bypassed": True}))
        sys.exit(0)

    if not FILE_PATH:
        log("No FILE_PATH in environment - skipping check")
        print(json.dumps({"ok": True, "message": "No file path to check"}))
        sys.exit(0)

    log(f"Checking file path: {FILE_PATH}")
    result = check_file_path(FILE_PATH)

    print(json.dumps(result, indent=2))

    # Warning only - don't block
    if result.get("warning"):
        log(f"Warning issued for: {FILE_PATH}")
        sys.exit(0)  # Exit 0 even on warning (non-blocking)
    else:
        log("Check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
