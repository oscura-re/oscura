#!/usr/bin/env python3
"""
Validate config.yaml against JSON Schema.

This hook validates the config.yaml file against config-schema.json to ensure
all configuration settings are valid before they're used by the system.

Version: 1.0.0
Created: 2026-01-22

Usage:
    python3 validate_config_schema.py

Returns:
    0 if validation passes
    1 if validation fails
"""

import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
    import yaml
except ImportError:
    # If dependencies not available, skip validation
    print('{"ok": true, "warning": "jsonschema or pyyaml not available"}')
    sys.exit(0)


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Load YAML file."""
    with open(file_path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def load_json_schema(file_path: Path) -> dict[str, Any]:
    """Load JSON schema."""
    with open(file_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def validate_config() -> bool:
    """
    Validate config.yaml against schema.

    Returns:
        True if valid, False otherwise
    """
    project_dir = Path.cwd()
    schema_file = project_dir / ".claude" / "schemas" / "config-schema.json"
    config_file = project_dir / ".claude" / "config.yaml"

    # Check files exist
    if not schema_file.exists():
        print(f'{{"ok": true, "info": "Schema file not found: {schema_file}"}}')
        return True  # Don't block if schema doesn't exist

    if not config_file.exists():
        print(f'{{"ok": false, "error": "Config file not found: {config_file}"}}')
        return False

    try:
        # Load files
        schema = load_json_schema(schema_file)
        config = load_yaml(config_file)

        # Validate
        jsonschema.validate(instance=config, schema=schema)

        print('{"ok": true, "message": "Config validation passed"}')
        return True

    except jsonschema.exceptions.ValidationError as e:
        # Validation failed
        error_path = " -> ".join(str(p) for p in e.path) if e.path else "root"
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Config validation failed",
                    "path": error_path,
                    "message": e.message,
                    "validator": e.validator,
                }
            )
        )
        return False

    except Exception as e:
        # Unexpected error
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Validation error",
                    "message": str(e),
                }
            )
        )
        return False


def main() -> None:
    """Main entry point."""
    success = validate_config()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
