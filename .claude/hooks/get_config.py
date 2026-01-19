#!/usr/bin/env python3
"""
Configuration Value Reader for Shell Scripts

Usage:
    python3 get_config.py hooks.pre_compact_cleanup.old_report_days
    python3 get_config.py retention.reports
    python3 get_config.py hooks.cleanup_stale_agents.stale_threshold_hours

Returns the value from config.yaml or exits with code 1 if not found.
"""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_config


def get_nested_value(data: dict, key_path: str, default=None):
    """Get nested value from dict using dot notation.

    Args:
        data: Dictionary to search
        key_path: Dot-separated path like "hooks.health_check.disk_space_critical_percent"
        default: Default value if key not found

    Returns:
        Value at key_path or default
    """
    keys = key_path.split(".")
    value = data

    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default

    return value


def main():
    if len(sys.argv) < 2:
        print("Usage: get_config.py <key_path>", file=sys.stderr)
        print(
            "Example: get_config.py hooks.health_check.disk_space_critical_percent", file=sys.stderr
        )
        sys.exit(1)

    key_path = sys.argv[1]
    default = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        project_dir = Path(__file__).parent.parent.parent
        config = load_config(project_dir)
        value = get_nested_value(config, key_path, default)

        if value is None:
            print(f"Key '{key_path}' not found in config.yaml", file=sys.stderr)
            sys.exit(1)

        print(value)
        sys.exit(0)

    except Exception as e:
        print(f"Error reading config: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
