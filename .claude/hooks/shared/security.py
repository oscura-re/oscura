"""Security utilities for Claude Code hooks.

Provides pattern matching and path validation logic extracted from validate_path.py:
- Pattern matching with glob support (including recursive patterns)
- Blocked path detection (credentials, .git internals, etc.)
- Warned path detection (critical configs that need user confirmation)

Version: 1.0.0
Created: 2026-01-19
"""

from fnmatch import fnmatch
from pathlib import Path

# Patterns that BLOCK file writes (security-critical)
BLOCKED_PATTERNS = {
    # Environment variables and secrets
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    ".env.*.local",
    "secrets.*",
    "secret.*",
    "*_secret",
    "*_secrets",
    # Private keys and certificates
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.cer",
    "*.crt",
    "*.der",
    "*.keystore",
    "*.jks",
    "*.truststore",
    # Credentials
    "credentials.*",
    "*credentials*",
    "auth_token*",
    "*_token",
    "*_token.*",
    "api_key*",
    "*_api_key",
    "password*",
    "*_password",
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
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    "id_ecdsa",
    "id_ecdsa.pub",
    "id_dsa",
    "id_dsa.pub",
    "*.ppk",  # PuTTY private key
    # Cloud provider credentials - AWS
    ".aws/credentials",
    ".aws/config",
    "aws_access_key*",
    "aws_secret_key*",
    # Cloud provider credentials - GCP
    "serviceaccount.json",
    "gcloud-service-key.json",
    ".gcloud/credentials",
    "gcp-key.json",
    # Cloud provider credentials - Azure
    ".azure/credentials",
    "azure-credentials.json",
    # Cloud provider credentials - Other
    ".config/gcloud/**",
    ".kube/config",
    "kubeconfig",
    "kubeconfig.*",
    # Database
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.db-journal",
    # Session and authentication
    "*.session",
    ".netrc",
    ".authinfo",
    ".pgpass",
    # Docker and container secrets
    ".docker/config.json",
    "docker-compose.override.yml",  # May contain secrets
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


def matches_pattern(path: Path, pattern: str) -> bool:
    """Check if path matches a glob pattern.

    Supports:
    - Simple filename patterns: "*.key", ".env"
    - Directory patterns: ".git/config"
    - Recursive patterns: ".git/objects/**" (matches all files under .git/objects/)

    Args:
        path: Path to check
        pattern: Glob pattern to match against

    Returns:
        True if path matches pattern, False otherwise

    Example:
        >>> matches_pattern(Path(".git/objects/ab/cdef"), ".git/objects/**")
        True
        >>> matches_pattern(Path("config.key"), "*.key")
        True
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


def is_blocked_path(path: Path, project_root: Path) -> tuple[bool, str | None]:
    """Check if path should be blocked (credentials, .git, etc).

    Args:
        path: Path to check (absolute or relative)
        project_root: Project root directory

    Returns:
        Tuple of (is_blocked, reason)
        - (True, "reason") if path is blocked
        - (False, None) if path is not blocked

    Example:
        >>> is_blocked_path(Path(".env"), Path("/project"))
        (True, "Blocked: Writing to .env files is prohibited (security)")
        >>> is_blocked_path(Path("src/main.py"), Path("/project"))
        (False, None)
    """
    # Resolve path relative to project root
    if not path.is_absolute():
        path = project_root / path

    try:
        # Get relative path for pattern matching
        relative_path = path.relative_to(project_root)
    except ValueError:
        # Path is outside project root - not our concern here
        # (boundary checking is done separately in validate_path)
        return False, None

    # Check BLOCKED patterns (security-critical)
    # Check both absolute and relative paths to catch all patterns
    for pattern in BLOCKED_PATTERNS:
        if matches_pattern(path, pattern) or matches_pattern(relative_path, pattern):
            return True, f"Blocked: Writing to {pattern} files is prohibited (security)"

    return False, None


def is_warned_path(path: Path, project_root: Path) -> tuple[bool, str | None]:
    """Check if path should generate warning (critical configs).

    Args:
        path: Path to check (absolute or relative)
        project_root: Project root directory

    Returns:
        Tuple of (needs_warning, message)
        - (True, "message") if path needs warning
        - (False, None) if path is OK

    Example:
        >>> is_warned_path(Path("pyproject.toml"), Path("/project"))
        (True, "Warning: Modifying pyproject.toml (critical config file)")
        >>> is_warned_path(Path("src/main.py"), Path("/project"))
        (False, None)
    """
    # Resolve path relative to project root
    if not path.is_absolute():
        path = project_root / path

    try:
        # Get relative path for pattern matching
        relative_path = path.relative_to(project_root)
    except ValueError:
        # Path is outside project root
        return False, None

    # Check WARNED patterns (important configs)
    for pattern in WARNED_PATTERNS:
        if matches_pattern(path, pattern) or matches_pattern(relative_path, pattern):
            return True, f"Warning: Modifying {relative_path.name} (critical config file)"

    return False, None


def get_security_classification(path: Path, project_root: Path) -> dict[str, str | bool | None]:
    """Get security classification for a path.

    Combines blocked and warned checks into a single classification.

    Args:
        path: Path to classify
        project_root: Project root directory

    Returns:
        dict with:
        - "blocked": bool - whether path is blocked
        - "warned": bool - whether path needs warning
        - "message": str | None - user-facing message
        - "allowed": bool - whether write is allowed

    Example:
        >>> get_security_classification(Path(".env"), Path("/project"))
        {
            "blocked": True,
            "warned": False,
            "message": "Blocked: Writing to .env files is prohibited (security)",
            "allowed": False
        }
    """
    # Check blocked first (takes precedence)
    is_blocked, block_message = is_blocked_path(path, project_root)
    if is_blocked:
        return {
            "blocked": True,
            "warned": False,
            "message": block_message,
            "allowed": False,
        }

    # Check warned
    needs_warning, warn_message = is_warned_path(path, project_root)
    if needs_warning:
        return {
            "blocked": False,
            "warned": True,
            "message": warn_message,
            "allowed": True,
        }

    # Path is OK
    return {
        "blocked": False,
        "warned": False,
        "message": None,
        "allowed": True,
    }
