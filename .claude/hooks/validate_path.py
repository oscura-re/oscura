#!/usr/bin/env python3
"""Validate file paths before Claude writes to them.

PreToolUse hook that prevents writing to sensitive files and validates path safety:
- Block writes to credentials (.env*, *.key, *.pem, secrets.*)
- Block writes to .git directory internals
- Warn on critical configs (pyproject.toml, package.json, .claude/settings.json)
- Prevent path traversal attacks (../../etc/passwd)
- Validate paths are within project root

Runs before Write, Edit, and NotebookEdit tool calls.
Blocking hook: Returns non-zero exit code to prevent dangerous operations.

Version: 2.0.0
Updated: 2026-01-19 - FAIL CLOSED security, fixed TOCTOU races, uses shared utilities

Configuration:
- BLOCKED_PATTERNS: Patterns that always block (security-critical)
- WARNED_PATTERNS: Patterns that warn user but allow (configs)
- Respects project root boundaries
"""

import json
import os
import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import get_hook_logger
from shared.security import is_blocked_path, is_warned_path

# Logger
logger = get_hook_logger(__name__)

# Excluded directories (skip validation - these are build/cache dirs that tools manage)
# NOTE: .git is NOT here because we explicitly block .git writes via security.py
EXCLUDED_DIRS = {".venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}


def validate_path(file_path: str, project_root: str) -> tuple[bool, str | None]:
    """Validate file path before write operation.

    Args:
        file_path: Path to validate
        project_root: Project root directory

    Returns:
        Tuple of (allowed: bool, message: Optional[str])
        - (True, None) = Allow write, no message
        - (True, "warning") = Allow with warning
        - (False, "reason") = Block write with reason
    """
    path = Path(file_path)
    project_path = Path(project_root).resolve()

    # Validate path exists and is within project root (security)
    try:
        # Resolve relative paths relative to project root, not CWD
        if not path.is_absolute():
            path = project_path / path

        # Security: Use resolve(strict=False) to avoid TOCTOU races
        # This resolves the path without following symlinks at the end
        try:
            abs_path = path.resolve(strict=False)
        except Exception as e:
            return False, f"Path resolution failed: {e}"

        # Check if within project root AFTER resolution
        try:
            abs_path.relative_to(project_path)
        except ValueError:
            return False, f"Path outside project root: {abs_path}"

        # Check for symlinks in resolved path components
        # This prevents symlink attacks while avoiding TOCTOU
        current = abs_path
        while current != project_path:
            if current.is_symlink():
                # Resolve symlink and check if it stays within project
                try:
                    link_target = current.readlink()
                    if link_target.is_absolute():
                        # Absolute symlink - check if within project
                        try:
                            link_target.relative_to(project_path)
                        except ValueError:
                            return (
                                False,
                                f"Symlink escapes project root: {current} -> {link_target}",
                            )
                except (OSError, RuntimeError):
                    return False, f"Failed to read symlink: {current}"

            current = current.parent
            if current == current.parent:  # Reached root
                break

    except (ValueError, OSError) as e:
        return False, f"Invalid path: {e}"

    # Check for path traversal attempts
    if ".." in path.parts:
        return False, "Path traversal detected (../ in path)"

    # Skip validation for excluded directories
    if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
        return True, None

    # Use security module for pattern matching
    is_blocked, block_message = is_blocked_path(abs_path, project_path)
    if is_blocked:
        return False, block_message

    # Check WARNED patterns (important configs)
    needs_warning, warn_message = is_warned_path(abs_path, project_path)
    if needs_warning:
        return True, warn_message

    # Allow all other writes
    return True, None


def main() -> None:
    """Main entry point for PreToolUse hook."""
    try:
        # Read hook input from stdin
        input_data = json.load(sys.stdin)

        # Extract tool name and file path
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Only validate Write, Edit, and NotebookEdit tools
        if tool_name not in {"Write", "Edit", "NotebookEdit"}:
            logger.debug(f"Ignoring tool: {tool_name}")
            sys.exit(0)

        # Extract file path from tool input
        file_path = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")

        if not file_path:
            logger.debug("No file path to validate")
            sys.exit(0)

        # Get project root from environment
        project_root = os.getenv("CLAUDE_PROJECT_DIR", str(Path.cwd()))

        # Validate the path
        allowed, message = validate_path(file_path, project_root)

        if message:
            # Print warning or error to stderr
            if allowed:
                logger.warning(f"{message}")
                print(f"⚠ {message}", file=sys.stderr)
            else:
                logger.error(f"BLOCKED: {message}")
                print(f"🛑 {message}", file=sys.stderr)
        else:
            logger.debug(f"Validated: {file_path}")

        # Exit code determines if operation proceeds
        # 0 = allow, non-zero = block
        sys.exit(0 if allowed else 1)

    except json.JSONDecodeError as e:
        # FAIL CLOSED for security hook - block on parse errors
        logger.error(f"Failed to parse stdin JSON: {e}")
        print("🛑 Security hook error: Invalid JSON input", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # FAIL CLOSED for security hook - block on unexpected errors
        logger.error(f"Hook failed: {e}", exc_info=True)
        print(f"🛑 Security hook error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
