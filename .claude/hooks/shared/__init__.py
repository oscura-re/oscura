"""Shared utilities for Claude Code hooks.

This package provides common functionality used across multiple hooks:
- paths.py: Path definitions and access
- config.py: Configuration and YAML loading
- logging_utils.py: Standardized logging
- registry.py: Agent registry operations
- datetime_utils.py: Datetime and staleness utilities
- security.py: Path security and pattern matching
"""

from .config import (
    get_hook_config,
    get_orchestration_config,
    get_retention_policy,
    load_coding_standards,
    load_config,
    load_config_with_validation,
    load_yaml_with_fallback,
    validate_config_schema,
)
from .datetime_utils import (
    age_in_days,
    age_in_hours,
    format_age,
    get_file_age_days,
    get_file_age_hours,
    is_file_stale,
    is_stale,
    parse_timestamp,
    timestamp_ago,
    timestamp_now,
)
from .logging_utils import (
    HookLogger,
    get_hook_logger,
    log_hook_end,
    log_hook_start,
    log_to_file,
)
from .paths import PATHS, get_path, load_paths
from .registry import (
    cleanup_old_agents,
    count_running_agents,
    get_registry_path,
    get_stale_agents,
    load_registry,
    register_agent,
    remove_agent,
    save_registry,
    update_agent_status,
)
from .security import (
    BLOCKED_PATTERNS,
    WARNED_PATTERNS,
    get_security_classification,
    is_blocked_path,
    is_warned_path,
    matches_pattern,
)

__all__ = [  # noqa: RUF022 (organized by module for readability)
    # paths
    "PATHS",
    "get_path",
    "load_paths",
    # config
    "load_yaml_with_fallback",
    "load_config",
    "load_config_with_validation",
    "load_coding_standards",
    "get_retention_policy",
    "get_hook_config",
    "get_orchestration_config",
    "validate_config_schema",
    # logging
    "get_hook_logger",
    "log_hook_start",
    "log_hook_end",
    "log_to_file",
    "HookLogger",
    # registry
    "get_registry_path",
    "load_registry",
    "save_registry",
    "count_running_agents",
    "update_agent_status",
    "register_agent",
    "remove_agent",
    "get_stale_agents",
    "cleanup_old_agents",
    # datetime_utils
    "parse_timestamp",
    "age_in_hours",
    "age_in_days",
    "is_stale",
    "get_file_age_hours",
    "get_file_age_days",
    "is_file_stale",
    "timestamp_now",
    "timestamp_ago",
    "format_age",
    # security
    "matches_pattern",
    "is_blocked_path",
    "is_warned_path",
    "get_security_classification",
    "BLOCKED_PATTERNS",
    "WARNED_PATTERNS",
]
