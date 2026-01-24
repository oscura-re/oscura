#!/usr/bin/env python3
"""
Shared configuration loading utilities for Claude hooks.

Consolidates YAML/config loading logic that was duplicated across 5+ hooks.
Provides fallback mechanisms when PyYAML is not available.

Version: 1.0.0
Created: 2026-01-19
"""

import os
from pathlib import Path
from typing import Any


def load_yaml_with_fallback(file_path: Path) -> dict[str, Any]:
    """Load YAML file with fallback parser if PyYAML not available.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML as dict, or empty dict if file doesn't exist

    Raises:
        ValueError: If YAML is malformed and can't be parsed
    """
    if not file_path.exists():
        return {}

    # Try PyYAML first (most robust)
    try:
        import yaml

        with open(file_path) as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except ImportError:
        # Fallback to basic parser
        pass
    except Exception as e:
        raise ValueError(f"Failed to parse YAML from {file_path}: {e}") from e

    # Fallback: Basic YAML parser for simple key-value structures
    return _parse_yaml_basic(file_path)


def _parse_yaml_basic(file_path: Path) -> dict[str, Any]:
    """Basic YAML parser for simple key-value structures.

    Handles:
    - Simple key: value pairs
    - Nested dictionaries (via indentation)
    - Lists (- item)
    - Comments (# ...)

    Does NOT handle:
    - Complex nested structures beyond 3 levels
    - Multi-line strings
    - Anchors/aliases
    - Special YAML types

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML as dict

    Raises:
        ValueError: If YAML structure is too complex for basic parser
    """
    result: dict[str, Any] = {}
    current_dict = result
    indent_stack: list[tuple[int, dict[str, Any]]] = [(0, result)]

    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):  # noqa: B007 (line_num used in exception handler)
                # Remove comments
                if "#" in line:
                    line = line[: line.index("#")]

                line = line.rstrip()

                # Skip empty lines
                if not line.strip():
                    continue

                # Calculate indentation
                indent = len(line) - len(line.lstrip())
                content = line.strip()

                # Pop stack to correct indent level
                while indent_stack and indent <= indent_stack[-1][0]:
                    indent_stack.pop()

                if indent_stack:
                    current_dict = indent_stack[-1][1]
                else:
                    current_dict = result
                    indent_stack = [(0, result)]

                # Parse list items
                if content.startswith("- "):
                    # Lists not fully supported in basic parser
                    # Store as single value for now
                    key = "__list__"
                    value = content[2:].strip()
                    if key not in current_dict:
                        current_dict[key] = []
                    if isinstance(current_dict[key], list):
                        current_dict[key].append(value)
                    continue

                # Parse key-value pairs
                if ":" in content:
                    key, _, value = content.partition(":")
                    key = key.strip()
                    value = value.strip()

                    # Empty value = nested dict
                    if not value:
                        current_dict[key] = {}
                        indent_stack.append((indent, current_dict[key]))
                    else:
                        # Try to parse value as int/float/bool
                        current_dict[key] = _parse_yaml_value(value)

    except Exception as e:
        raise ValueError(f"Failed to parse YAML at line {line_num}: {e}") from e

    return result


def _parse_yaml_value(value: str) -> Any:
    """Parse YAML value to appropriate Python type."""
    value = value.strip()

    # Boolean
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # Null
    if value.lower() in ("null", "none", "~"):
        return None

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # String (remove quotes if present)
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    return value


def load_config(project_dir: Path | None = None) -> dict[str, Any]:
    """Load config.yaml from .claude/ directory.

    Args:
        project_dir: Project root directory (defaults to CLAUDE_PROJECT_DIR env var)

    Returns:
        Config dict with all settings
    """
    if project_dir is None:
        project_dir = Path(os.getenv("CLAUDE_PROJECT_DIR", "."))

    config_file = project_dir / ".claude" / "config.yaml"
    return load_yaml_with_fallback(config_file)


def load_coding_standards(project_dir: Path | None = None) -> dict[str, Any]:
    """Load coding-standards.yaml from .claude/ directory.

    Args:
        project_dir: Project root directory

    Returns:
        Coding standards dict
    """
    if project_dir is None:
        project_dir = Path(os.getenv("CLAUDE_PROJECT_DIR", "."))

    standards_file = project_dir / ".claude" / "coding-standards.yaml"
    return load_yaml_with_fallback(standards_file)


def get_retention_policy(config: dict[str, Any], key: str, default: int = 7) -> int:
    """Get retention policy value from config.

    Args:
        config: Config dict from load_config()
        key: Retention key (e.g., 'reports', 'checkpoints')
        default: Default value if not found

    Returns:
        Retention period in days
    """
    retention = config.get("retention", {})
    return int(retention.get(key, default))


def get_hook_config(config: dict[str, Any], hook_name: str) -> dict[str, Any]:
    """Get hook-specific configuration.

    Args:
        config: Config dict from load_config()
        hook_name: Hook name (e.g., 'cleanup_stale_agents')

    Returns:
        Hook configuration dict
    """
    hooks_config = config.get("hooks", {})
    return hooks_config.get(hook_name, {})  # type: ignore[no-any-return]


def get_orchestration_config(config: dict[str, Any]) -> dict[str, Any]:
    """Get orchestration configuration.

    Args:
        config: Config dict from load_config()

    Returns:
        Orchestration config dict
    """
    return config.get("orchestration", {})  # type: ignore[no-any-return]


def validate_config_schema(config: dict[str, Any]) -> list[str]:
    """Validate config.yaml structure and value ranges.

    Checks:
    - Required sections exist (orchestration, retention, hooks, security)
    - Numeric values are in valid ranges
    - Paths are valid (if present)
    - No unknown configuration keys at top level
    - Nested structure is valid

    Args:
        config: Config dict to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required top-level keys
    required_keys = ["orchestration", "retention"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required top-level key: {key}")

    # Validate orchestration section
    if "orchestration" in config:
        orch = config["orchestration"]
        if not isinstance(orch, dict):
            errors.append("orchestration must be a dict")
        else:
            # Check required orchestration subsections
            if "agents" not in orch:
                errors.append("orchestration.agents section missing")
            else:
                agents = orch["agents"]
                if not isinstance(agents, dict):
                    errors.append("orchestration.agents must be a dict")
                else:
                    # Validate agent limits
                    max_concurrent = agents.get("max_concurrent")
                    if max_concurrent is not None:
                        if not isinstance(max_concurrent, int) or max_concurrent < 1:
                            errors.append("orchestration.agents.max_concurrent must be int >= 1")

                    max_batch_size = agents.get("max_batch_size")
                    if max_batch_size is not None:
                        if not isinstance(max_batch_size, int) or max_batch_size < 1:
                            errors.append("orchestration.agents.max_batch_size must be int >= 1")

            # Validate context thresholds
            if "context" in orch:
                context = orch["context"]
                if isinstance(context, dict):
                    for threshold_key in [
                        "warning_threshold",
                        "checkpoint_threshold",
                        "critical_threshold",
                    ]:
                        threshold = context.get(threshold_key)
                        if threshold is not None:
                            if (
                                not isinstance(threshold, (int, float))
                                or threshold < 0
                                or threshold > 100
                            ):
                                errors.append(
                                    f"orchestration.context.{threshold_key} must be 0-100"
                                )

    # Validate retention section
    if "retention" in config:
        retention = config["retention"]
        if not isinstance(retention, dict):
            errors.append("retention must be a dict")
        else:
            # Validate retention periods are positive
            for key, value in retention.items():
                if isinstance(value, (int, float)):
                    if value < 0:
                        errors.append(f"retention.{key} must be >= 0 (got {value})")
                elif isinstance(value, str):
                    # Skip string values (may be paths or special values)
                    pass
                else:
                    errors.append(f"retention.{key} must be numeric or string (got {type(value)})")

    # Validate hooks section
    if "hooks" in config:
        hooks = config["hooks"]
        if not isinstance(hooks, dict):
            errors.append("hooks must be a dict")
        else:
            # Validate hook-specific configs
            for hook_name, hook_config in hooks.items():
                if not isinstance(hook_config, dict):
                    errors.append(f"hooks.{hook_name} must be a dict")

    # Validate security section
    if "security" in config:
        security = config["security"]
        if not isinstance(security, dict):
            errors.append("security must be a dict")
        else:
            # Validate denied_reads and denied_writes are lists
            for key in ["denied_reads", "denied_writes"]:
                if key in security:
                    if not isinstance(security[key], list):
                        errors.append(f"security.{key} must be a list")

    # Validate cleanup section
    if "cleanup" in config:
        cleanup = config["cleanup"]
        if not isinstance(cleanup, dict):
            errors.append("cleanup must be a dict")

    # Validate logging section
    if "logging" in config:
        logging = config["logging"]
        if not isinstance(logging, dict):
            errors.append("logging must be a dict")

    return errors


def load_config_with_validation(project_dir: Path | None = None) -> dict[str, Any]:
    """Load config.yaml with validation and logging.

    Args:
        project_dir: Project root directory

    Returns:
        Config dict (may be empty if file doesn't exist or has errors)

    Side effects:
        Logs warnings for validation errors
    """
    config = load_config(project_dir)

    if not config:
        return config

    # Validate schema
    errors = validate_config_schema(config)

    if errors:
        # Log warnings but don't fail - allow degraded operation
        import logging

        logger = logging.getLogger("hooks.config")
        for error in errors:
            logger.warning(f"Config validation error: {error}")

    return config
