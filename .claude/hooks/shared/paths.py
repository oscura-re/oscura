"""
Centralized path loading from paths.yaml.
Single source of truth for all hook paths.

Usage:
    from shared.paths import PATHS, get_path

    # Access paths via dictionary
    checkpoints = PATHS['claude']['coordination']['checkpoints']

    # Or use helper function with dot notation
    checkpoints = get_path('claude.coordination.checkpoints')
"""

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment, unused-ignore]


def load_paths() -> dict[str, Any]:
    """
    Load paths from .claude/paths.yaml.

    Returns:
        Dictionary of paths and configuration from paths.yaml
    """
    # Determine project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent  # .claude/hooks/shared -> project root
    paths_file = project_root / ".claude" / "paths.yaml"

    # Fallback if paths.yaml doesn't exist
    if not paths_file.exists():
        return _get_fallback_paths()

    # Load YAML (runtime safety check for when yaml not installed)
    if yaml is None:
        return _get_fallback_paths()

    try:
        with open(paths_file) as f:
            paths: dict[str, Any] = yaml.safe_load(f)
            return paths
    except Exception:
        return _get_fallback_paths()


def _get_fallback_paths() -> dict[str, Any]:
    """Fallback paths if paths.yaml cannot be loaded.

    NOTE: Structure must match .claude/paths.yaml exactly.
    """
    return {
        "claude": {
            "root": ".claude",
            "agents": ".claude/agents",
            "commands": ".claude/commands",
            "hooks": ".claude/hooks",
            "docs": ".claude/docs",
            "templates": ".claude/templates",
            "schemas": ".claude/schemas",
            "agent_registry": ".claude/agent-registry.json",
            "agent_outputs": ".claude/agent-outputs",
            "agent_outputs_archive": ".claude/agent-outputs/archive",
            "reports": ".claude/reports",
            "summaries": ".claude/summaries",
        },
        "coordination": {
            "root": ".coordination",
            "checkpoints": ".coordination/checkpoints",
            "archive": ".coordination/archive",
            "workflows": ".coordination/workflows",
        },
        "logs": {
            "root": ".claude/hooks",
            "enforcement": ".claude/hooks/enforcement.log",
            "orchestration": ".claude/hooks/orchestration.log",
            "context": ".claude/hooks/context.log",
            "errors": ".claude/hooks/errors.log",
            "metrics": ".claude/hooks/orchestration-metrics.json",
        },
    }


def get_path(key: str, default: Any = None) -> Any:
    """
    Get path using dot notation.

    Args:
        key: Dot-separated path key (e.g., 'claude.coordination.checkpoints')
        default: Default value if key not found

    Returns:
        Path value from PATHS dictionary

    Examples:
        >>> get_path('claude.coordination.checkpoints')
        '.coordination/checkpoints'

        >>> get_path('claude.retention.checkpoints')
        7
    """
    keys = key.split(".")
    value = PATHS

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def get_absolute_path(key: str, project_dir: Path | None = None) -> Path:
    """
    Get absolute path for a path key.

    Args:
        key: Dot-separated path key (e.g., 'claude.coordination.checkpoints')
        project_dir: Project directory (default: CLAUDE_PROJECT_DIR env or current dir)

    Returns:
        Absolute Path object

    Examples:
        >>> get_absolute_path('claude.coordination.checkpoints')
        Path('/home/user/project/.coordination/checkpoints')
    """
    if project_dir is None:
        project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()

    relative_path = get_path(key)
    if relative_path is None:
        raise ValueError(f"Path not found: {key}")

    if not isinstance(relative_path, str):
        raise ValueError(f"Path value must be a string: {key} = {relative_path}")

    return project_dir / relative_path


# Load paths once at module import
PATHS = load_paths()
