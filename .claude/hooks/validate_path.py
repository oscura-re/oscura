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

Configuration:
- BLOCKED_PATTERNS: Patterns that always block (security-critical)
- WARNED_PATTERNS: Patterns that warn user but allow (configs)
- Respects project root boundaries
"""
import json
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

# Patterns that BLOCK file writes (security-critical)
BLOCKED_PATTERNS = {
    # Credentials and secrets
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    ".env.*.local",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "secrets.*",
    "credentials.*",
    "*.keystore",
    "*.jks",
    # Git internals (modifying these can corrupt repo)
    ".git/config",
    ".git/HEAD",
    ".git/index",
    ".git/objects/**",
    ".git/refs/**",
    ".git/hooks/**",
    # SSH keys
    "*.pub",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    # Cloud credentials
    "serviceaccount.json",
    "gcloud-service-key.json",
    # Database
    "*.db",
    "*.sqlite",
    "*.sqlite3",
}

# Patterns that WARN but allow (important configs)
WARNED_PATTERNS = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    ".claude/settings.json",
    "tsconfig.json",
    ".vscode/settings.json",
    "uv.lock",
    "package-lock.json",
    "Cargo.lock",
    "go.sum",
}

# Excluded directories (no validation needed - handled by auto_format)
EXCLUDED_DIRS = {".venv", "node_modules", "__pycache__", ".git", ".mypy_cache", ".ruff_cache"}


def matches_pattern(path: Path, pattern: str) -> bool:
    """Check if path matches a glob pattern.

    Supports:
    - Simple filename patterns: "*.key", ".env"
    - Directory patterns: ".git/config"
    - Recursive patterns: ".git/objects/**" (matches all files under .git/objects/)
    """
    path_str = str(path)

    # Check filename match
    if fnmatch(path.name, pattern):
        return True

    # Handle recursive patterns with /**
    if "**" in pattern:
        # Pattern like ".git/objects/**" should match ".git/objects/ab/cdef123"
        base_pattern = pattern.replace("/**", "")
        # Check if any parent directory matches the base pattern
        for parent in [path, *path.parents]:
            parent_str = str(parent)
            # Path is under the matched directory (don't match the directory itself)
            if (parent_str.endswith(base_pattern) or f"/{base_pattern}" in parent_str) and str(
                path
            ) != parent_str:
                return True
        # Also check if path starts with the pattern prefix
        if base_pattern in path_str:
            idx = path_str.find(base_pattern)
            # Verify it's a proper path component match
            if idx == 0 or path_str[idx - 1] == "/":
                remaining = path_str[idx + len(base_pattern) :]
                if remaining.startswith("/"):
                    return True

    # Check if pattern matches the full relative path
    if fnmatch(path_str, pattern):
        return True

    # Check relative path components for simple patterns
    return any(fnmatch(part, pattern) for part in path.parts)


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

        # Security: Check for symlinks BEFORE resolving (symlink attack prevention)
        # Walk up the path checking each component for symlinks
        check_path = path
        while check_path != check_path.parent:
            if check_path.is_symlink():
                # Symlink exists - verify it doesn't escape project root
                link_target = check_path.resolve()
                try:
                    link_target.relative_to(project_path)
                except ValueError:
                    return False, f"Symlink escapes project root: {check_path} -> {link_target}"
            check_path = check_path.parent

        abs_path = path.resolve()

        # Check if within project root
        try:
            abs_path.relative_to(project_path)
        except ValueError:
            return False, f"Path outside project root: {abs_path}"

    except (ValueError, OSError) as e:
        return False, f"Invalid path: {e}"

    # Check for path traversal attempts
    if ".." in path.parts:
        return False, "Path traversal detected (../ in path)"

    # Skip validation for excluded directories
    if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
        return True, None

    # Check BLOCKED patterns (security-critical)
    for pattern in BLOCKED_PATTERNS:
        if matches_pattern(path, pattern):
            return False, f"Blocked: Writing to {pattern} files is prohibited (security)"

    # Check WARNED patterns (important configs)
    for pattern in WARNED_PATTERNS:
        if matches_pattern(path, pattern):
            return True, f"Warning: Modifying {path.name} (critical config file)"

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
            sys.exit(0)

        # Extract file path from tool input
        file_path = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")

        if not file_path:
            sys.exit(0)

        # Get project root from environment
        project_root = os.getenv("CLAUDE_PROJECT_DIR", str(Path.cwd()))

        # Validate the path
        allowed, message = validate_path(file_path, project_root)

        if message:
            # Print warning or error to stderr
            if allowed:
                print(f"⚠ {message}", file=sys.stderr)
            else:
                print(f"🛑 {message}", file=sys.stderr)

        # Exit code determines if operation proceeds
        # 0 = allow, non-zero = block
        sys.exit(0 if allowed else 1)

    except json.JSONDecodeError:
        # Invalid JSON input - allow by default (fail open)
        sys.exit(0)
    except Exception as e:
        # Log errors to stderr but allow operation (fail open)
        print(f"⚠ Path validation hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
