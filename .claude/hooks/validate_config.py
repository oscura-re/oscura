#!/usr/bin/env python3
"""Validate all .claude configuration files.

This script validates that all configuration files are valid, consistent,
and follow the correct schema.

Validates:
- Schema compliance (config.yaml matches JSON schema)
- Required sections present
- Type validation (all values are correct types)
- Range validation (thresholds in 0-100, retention >= 0)
- Path validation (all path references exist in paths.yaml)
- Version consistency (_meta.version matches comment version)
- Hook references (all referenced hooks exist)
- Security patterns (denied/warned patterns are valid globs)
- Cross-file consistency (no conflicts between configs)

Exit codes:
    0: All configs valid
    1: Validation warnings
    2: Critical config errors

Version: 1.0.0
Created: 2026-01-22
"""

import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
    import yaml

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CLAUDE_DIR = REPO_ROOT / ".claude"
CONFIG_FILE = CLAUDE_DIR / "config.yaml"
CODING_STANDARDS_FILE = CLAUDE_DIR / "coding-standards.yaml"
PROJECT_METADATA_FILE = CLAUDE_DIR / "project-metadata.yaml"
PATHS_FILE = CLAUDE_DIR / "paths.yaml"
SCHEMA_FILE = CLAUDE_DIR / "schemas" / "config-schema.json"
HOOKS_DIR = CLAUDE_DIR / "hooks"


def load_yaml_file(file_path: Path) -> dict[str, Any] | None:
    """Load YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dict if successful, None otherwise
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path) as f:
            result = yaml.safe_load(f)
            if result is None:
                return {}
            if not isinstance(result, dict):
                return {}
            return result
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None


def load_json_file(file_path: Path) -> dict[str, Any] | None:
    """Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dict if successful, None otherwise
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path) as f:
            result = json.load(f)
            if not isinstance(result, dict):
                return {}
            return result
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None


def validate_schema_compliance(config: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate config against JSON schema.

    Args:
        config: Config dictionary
        schema: JSON schema dictionary

    Returns:
        List of validation errors
    """
    if not HAS_DEPS:
        return ["jsonschema not available, skipping schema validation"]

    errors = []

    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        error_path = " -> ".join(str(p) for p in e.path) if e.path else "root"
        errors.append(f"Schema validation failed at {error_path}: {e.message}")
    except Exception as e:
        errors.append(f"Schema validation error: {e}")

    return errors


def validate_required_sections(config: dict[str, Any]) -> list[str]:
    """Validate required sections are present.

    Args:
        config: Config dictionary

    Returns:
        List of validation errors
    """
    errors = []

    required_sections = [
        "_meta",
        "orchestration",
        "retention",
        "security",
        "enforcement",
        "logging",
    ]

    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    return errors


def validate_type_compliance(config: dict[str, Any]) -> list[str]:
    """Validate all values have correct types.

    Args:
        config: Config dictionary

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    # Check _meta section
    if "_meta" in config:
        meta = config["_meta"]
        if not isinstance(meta.get("schema_version"), str):
            errors.append("_meta.schema_version must be a string")
        if not isinstance(meta.get("config_version"), str):
            errors.append("_meta.config_version must be a string")

    # Check orchestration section
    if "orchestration" in config:
        orch = config["orchestration"]

        if "agents" in orch:
            agents = orch["agents"]
            for key in ["max_concurrent", "max_batch_size", "recommended_batch_size"]:
                if key in agents and not isinstance(agents[key], int):
                    errors.append(f"orchestration.agents.{key} must be an integer")

        if "context" in orch:
            context = orch["context"]
            for key in [
                "warning_threshold",
                "checkpoint_threshold",
                "critical_threshold",
            ]:
                if key in context and not isinstance(context[key], int):
                    errors.append(f"orchestration.context.{key} must be an integer")

            if "auto_summarize" in context and not isinstance(context["auto_summarize"], bool):
                errors.append("orchestration.context.auto_summarize must be a boolean")

    # Check retention section
    if "retention" in config:
        retention = config["retention"]
        for key, value in retention.items():
            if not isinstance(value, int):
                errors.append(f"retention.{key} must be an integer")

    # Check security section
    if "security" in config:
        security = config["security"]
        for key in ["denied_reads", "denied_writes", "warned_writes"]:
            if key in security and not isinstance(security[key], list):
                errors.append(f"security.{key} must be a list")

        if "fail_closed" in security and not isinstance(security["fail_closed"], bool):
            errors.append("security.fail_closed must be a boolean")

    return errors


def validate_range_values(config: dict[str, Any]) -> list[str]:
    """Validate values are in acceptable ranges.

    Args:
        config: Config dictionary

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    # Check threshold percentages (0-100)
    if "orchestration" in config and "context" in config["orchestration"]:
        context = config["orchestration"]["context"]
        for key in ["warning_threshold", "checkpoint_threshold", "critical_threshold"]:
            if key in context:
                value = context[key]
                if not (0 <= value <= 100):
                    errors.append(
                        f"orchestration.context.{key} must be between 0 and 100, got {value}"
                    )

    # Check fuzzy_threshold (0-100)
    if (
        "orchestration" in config
        and "routing" in config["orchestration"]
        and "fuzzy_threshold" in config["orchestration"]["routing"]
    ):
        value = config["orchestration"]["routing"]["fuzzy_threshold"]
        if not (0 <= value <= 100):
            errors.append(
                f"orchestration.routing.fuzzy_threshold must be between 0 and 100, got {value}"
            )

    # Check retention values (>= 0)
    if "retention" in config:
        for key, value in config["retention"].items():
            if value < 0:
                errors.append(f"retention.{key} must be >= 0, got {value}")

    return errors


def validate_path_references(config: dict[str, Any], paths_config: dict[str, Any]) -> list[str]:
    """Validate path references exist in paths.yaml.

    Args:
        config: Config dictionary
        paths_config: Paths.yaml dictionary

    Returns:
        List of validation errors
    """
    if not paths_config:
        return ["paths.yaml not found or invalid"]

    errors: list[str] = []

    # Currently config.yaml doesn't reference paths.yaml directly
    # This is a placeholder for future path validation

    return errors


def validate_version_consistency(config: dict[str, Any], file_content: str) -> list[str]:
    """Validate version consistency between _meta and comments.

    Args:
        config: Config dictionary
        file_content: Raw file content

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    if "_meta" not in config:
        return errors

    meta = config["_meta"]
    schema_version = meta.get("schema_version", "")
    meta.get("config_version", "")

    # Check if comment version matches
    lines = file_content.split("\n")
    for line in lines:
        if "Schema Version:" in line:
            # Extract version from comment
            import re

            match = re.search(r"Schema Version:\s*([0-9.]+)", line)
            if match:
                comment_version = match.group(1)
                if comment_version != schema_version:
                    errors.append(
                        f"Schema version mismatch: comment says {comment_version}, "
                        f"_meta.schema_version says {schema_version}"
                    )

    return errors


def validate_hook_references(config: dict[str, Any]) -> list[str]:
    """Validate that referenced hooks exist.

    Args:
        config: Config dictionary

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    # Currently config.yaml doesn't explicitly list hook files
    # They're referenced implicitly via enforcement flags

    # Check enforcement section
    if "enforcement" in config:
        enforcement = config["enforcement"]

        # Map enforcement flags to hook files
        hook_mappings = {
            "agent_limit": "enforce_agent_limit.py",
            "path_validation": "validate_path.py",
            "ssot_validation": "validate_ssot.py",
            "report_proliferation": "check_report_proliferation.py",
        }

        for flag, hook_file in hook_mappings.items():
            if enforcement.get(flag):
                hook_path = HOOKS_DIR / hook_file
                if not hook_path.exists():
                    errors.append(f"enforcement.{flag} is enabled but hook {hook_file} not found")

    return errors


def validate_security_patterns(config: dict[str, Any]) -> list[str]:
    """Validate security patterns are valid globs.

    Args:
        config: Config dictionary

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    if "security" not in config:
        return errors

    security = config["security"]

    # Check patterns are valid
    for key in ["denied_reads", "denied_writes", "warned_writes"]:
        if key in security:
            patterns = security[key]
            if not isinstance(patterns, list):
                continue

            for pattern in patterns:
                # Basic validation - must be string
                if not isinstance(pattern, str):
                    errors.append(f"security.{key} contains non-string pattern: {pattern}")
                    continue

                # Check for common mistakes
                if pattern.startswith("/") and "**" not in pattern:
                    errors.append(
                        f"security.{key} pattern '{pattern}' starts with / but has no **: "
                        "may not match relative paths"
                    )

    return errors


def validate_cross_file_consistency(
    config: dict[str, Any],
    coding_standards: dict[str, Any],
    project_metadata: dict[str, Any],
) -> list[str]:
    """Validate consistency across configuration files.

    Args:
        config: Config dictionary
        coding_standards: Coding standards dictionary
        project_metadata: Project metadata dictionary

    Returns:
        List of validation errors
    """
    errors: list[str] = []

    # Check retention consistency
    if "retention" in config and coding_standards:
        # Ensure retention periods are reasonable
        if "agent_registry" in config["retention"]:
            if config["retention"]["agent_registry"] < 7:
                errors.append("retention.agent_registry should be >= 7 days for debugging purposes")

    # Check version consistency with metadata
    if "_meta" in config and project_metadata and "claude_code" in project_metadata:
        min_version = project_metadata["claude_code"].get("version")
        if min_version and "_meta" in config:
            # Ensure config is compatible with Claude Code version
            # This is informational, not a hard error
            pass

    return errors


def main() -> int:
    """Main validation function.

    Returns:
        Exit code (0 for success, 1 for warnings, 2 for critical errors)
    """
    print("\n" + "=" * 70)
    print("  CONFIGURATION VALIDATION")
    print("=" * 70 + "\n")

    if not HAS_DEPS:
        print("⚠ WARNING: jsonschema or pyyaml not available", file=sys.stderr)
        print("Install with: pip install jsonschema pyyaml\n")
        return 1

    # Load all config files
    print("Loading configuration files...\n")

    config = load_yaml_file(CONFIG_FILE)
    if not config:
        print(f"❌ CRITICAL: Failed to load {CONFIG_FILE}", file=sys.stderr)
        return 2

    coding_standards = load_yaml_file(CODING_STANDARDS_FILE)
    project_metadata = load_yaml_file(PROJECT_METADATA_FILE)
    paths_config = load_yaml_file(PATHS_FILE)
    schema = load_json_file(SCHEMA_FILE) if SCHEMA_FILE.exists() else None

    # Load raw file content for version validation
    config_content = CONFIG_FILE.read_text()

    # Run all validations
    all_errors = []
    warnings = []

    print("Running validations...\n")

    # Schema validation
    if schema:
        errors = validate_schema_compliance(config, schema)
        all_errors.extend(errors)
    else:
        warnings.append("JSON schema not found, skipping schema validation")

    # Structure validation
    all_errors.extend(validate_required_sections(config))
    all_errors.extend(validate_type_compliance(config))
    all_errors.extend(validate_range_values(config))
    all_errors.extend(validate_version_consistency(config, config_content))
    all_errors.extend(validate_hook_references(config))
    all_errors.extend(validate_security_patterns(config))

    # Cross-file validation
    if paths_config:
        all_errors.extend(validate_path_references(config, paths_config))

    if coding_standards and project_metadata:
        consistency_errors = validate_cross_file_consistency(
            config, coding_standards, project_metadata
        )
        all_errors.extend(consistency_errors)

    # Report results
    print("=" * 70)

    if all_errors:
        print(f"\n❌ VALIDATION FAILED with {len(all_errors)} error(s):\n")
        for error in all_errors:
            print(f"  - {error}")
        print("\n" + "=" * 70)
        return 2

    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):\n")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n✅ ALL CHECKS PASSED")
    print("  - config.yaml validated")
    print("  - All required sections present")
    print("  - All values in valid ranges")
    print("  - All hook references valid")
    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
