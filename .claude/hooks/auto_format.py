#!/usr/bin/env python3
"""Auto-format files after Claude writes/edits them.

PostToolUse hook that automatically formats files using language-specific formatters:
- Python: ruff format
- Markdown/YAML/JSON: prettier
- Shell: shfmt
- LaTeX: latexindent

Runs after Write, Edit, and NotebookEdit tool calls with minimal overhead (<1s per file).
Non-blocking: Always exits 0 to avoid disrupting workflows.

Configuration:
- Detects tools dynamically (graceful degradation if missing)
- Respects excluded directories (.venv, node_modules, __pycache__, .git)
- Validates paths are within project root (security)
- 30s timeout per formatter (prevents hangs)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Excluded directories (skip formatting)
EXCLUDED_DIRS = {".venv", "node_modules", "__pycache__", ".git", ".mypy_cache", ".ruff_cache"}

# Maximum file size to format (in bytes) - skip formatting files larger than this
MAX_FILE_SIZE_BYTES = 500_000  # 500KB - prevents slow formatting on large files

# Tool availability cache (avoid repeated lookups)
_tool_cache = {}


def has_tool(tool: str) -> bool:
    """Check if tool is available on PATH (cached)."""
    if tool not in _tool_cache:
        _tool_cache[tool] = shutil.which(tool) is not None
    return _tool_cache[tool]


def format_file(file_path: str, project_root: str) -> tuple[bool, str]:
    """Format file based on extension.

    Args:
        file_path: Path to file to format
        project_root: Project root directory (for security validation)

    Returns:
        Tuple of (formatted: bool, message: str)
    """
    path = Path(file_path)

    # Skip if file doesn't exist
    if not path.exists():
        return False, "File doesn't exist"

    # Skip excluded directories
    if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
        return False, "Excluded directory"

    # Skip large files (prevent slow formatting)
    try:
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return (
                False,
                f"File too large ({size_mb:.1f}MB, max {MAX_FILE_SIZE_BYTES / (1024 * 1024):.1f}MB)",
            )
    except OSError:
        pass  # If we can't stat, continue and let formatter handle it

    # Skip if outside project root (security check)
    try:
        path.resolve().relative_to(Path(project_root).resolve())
    except ValueError:
        return False, "Outside project root"

    formatter_timeout = 30

    try:
        # Python files: ruff format
        if path.suffix == ".py":
            if not has_tool("uv"):
                return False, "uv not available"
            result = subprocess.run(
                ["uv", "run", "ruff", "format", str(file_path)],
                cwd=project_root,
                capture_output=True,
                timeout=formatter_timeout,
                check=False,
            )
            if result.returncode == 0:
                return True, "ruff formatted"
            return False, f"ruff failed: {result.returncode}"

        # Markdown/YAML/JSON: prettier
        elif path.suffix in {".md", ".yaml", ".yml", ".json"}:
            if not has_tool("npx"):
                return False, "npx not available"
            result = subprocess.run(
                ["npx", "prettier", "--write", str(file_path)],
                cwd=project_root,
                capture_output=True,
                timeout=formatter_timeout,
                check=False,
            )
            if result.returncode == 0:
                return True, "prettier formatted"
            return False, f"prettier failed: {result.returncode}"

        # Shell scripts: shfmt
        elif path.suffix == ".sh":
            if not has_tool("shfmt"):
                return False, "shfmt not available"
            result = subprocess.run(
                ["shfmt", "-i", "2", "-w", str(file_path)],
                cwd=project_root,
                capture_output=True,
                timeout=formatter_timeout,
                check=False,
            )
            if result.returncode == 0:
                return True, "shfmt formatted"
            return False, f"shfmt failed: {result.returncode}"

        # LaTeX files: latexindent
        elif path.suffix == ".tex":
            if not has_tool("latexindent"):
                return False, "latexindent not available"
            # latexindent creates backup files, use -w to overwrite
            result = subprocess.run(
                ["latexindent", "-w", str(file_path)],
                cwd=project_root,
                capture_output=True,
                timeout=formatter_timeout,
                check=False,
            )
            if result.returncode == 0:
                return True, "latexindent formatted"
            return False, f"latexindent failed: {result.returncode}"

        return False, "No formatter configured"

    except subprocess.TimeoutExpired:
        return False, f"Formatting timeout (>{formatter_timeout}s)"
    except Exception as e:
        return False, f"Error: {e}"


def main() -> None:
    """Main entry point for PostToolUse hook."""
    try:
        # Read hook input from stdin
        input_data = json.load(sys.stdin)

        # Extract tool name and file path
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Only process Write, Edit, and NotebookEdit tools
        if tool_name not in {"Write", "Edit", "NotebookEdit"}:
            sys.exit(0)

        # Extract file path from tool input
        file_path = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")

        if not file_path:
            sys.exit(0)

        # Get project root from environment
        project_root = os.getenv("CLAUDE_PROJECT_DIR", str(Path.cwd()))

        # Format the file
        formatted, message = format_file(file_path, project_root)

        # Only print success messages to stderr (visible to user)
        if formatted:
            filename = Path(file_path).name
            print(f"✓ Auto-formatted: {filename} ({message})", file=sys.stderr)

    except json.JSONDecodeError:
        # Invalid JSON input - fail silently
        pass
    except Exception as e:
        # Log errors to stderr but don't block
        print(f"⚠ Auto-format hook error: {e}", file=sys.stderr)

    # ALWAYS exit 0 (non-blocking hook)
    sys.exit(0)


if __name__ == "__main__":
    main()
