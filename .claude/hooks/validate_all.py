#!/usr/bin/env python3
"""Run all validation scripts.

This master script runs all validation scripts in sequence and reports
aggregate results.

Validators:
- validate_agents.py: Agent markdown structure and cross-references
- validate_documentation.py: Documentation links and references
- validate_portability.py: Infrastructure portability across projects
- validate_config.py: Configuration file validity
- validate_ssot.py: Single Source of Truth compliance

Exit codes:
    0: All validations passed
    1: Some validations failed

Version: 1.0.0
Created: 2026-01-22
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

# Resolve paths
HOOKS_DIR = Path(__file__).parent

# Validators to run (in order)
VALIDATORS = [
    {
        "name": "Agent validation",
        "script": "validate_agents.py",
        "description": "Validate agent markdown files",
    },
    {
        "name": "Documentation validation",
        "script": "validate_documentation.py",
        "description": "Validate documentation links",
    },
    {
        "name": "Portability validation",
        "script": "validate_portability.py",
        "description": "Check infrastructure portability",
    },
    {
        "name": "Configuration validation",
        "script": "validate_config.py",
        "description": "Validate config files",
    },
    {
        "name": "SSOT validation",
        "script": "validate_ssot.py",
        "description": "Check SSOT compliance",
    },
]


def run_validator(validator: dict[str, Any]) -> tuple[bool, str]:
    """Run a single validator script.

    Args:
        validator: Validator configuration dict

    Returns:
        Tuple of (success: bool, output: str)
    """
    script_path = HOOKS_DIR / validator["script"]

    if not script_path.exists():
        return False, f"Script not found: {validator['script']}"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout per validator
            check=False,  # Don't raise on non-zero exit, we handle it below
        )

        # Capture both stdout and stderr
        output = result.stdout + result.stderr

        # Return success if exit code is 0
        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "Validator timed out after 60 seconds"
    except Exception as e:
        return False, f"Error running validator: {e}"


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    print("\n" + "=" * 70)
    print("  MASTER VALIDATION SUITE")
    print("=" * 70 + "\n")

    print(f"Running {len(VALIDATORS)} validator(s)...\n")

    # Track results
    results = []
    failed_validators = []

    # Run each validator
    for validator in VALIDATORS:
        name = validator["name"]
        description = validator["description"]

        print(f"Running: {name}")
        print(f"  {description}")

        success, output = run_validator(validator)

        results.append(
            {
                "name": name,
                "success": success,
                "output": output,
            }
        )

        if success:
            print("  ✅ PASSED\n")
        else:
            print("  ❌ FAILED\n")
            failed_validators.append(name)

    # Print summary
    print("=" * 70)
    print("\n📊 SUMMARY")
    print(f"  Total validators: {len(VALIDATORS)}")
    print(f"  Passed: {len(results) - len(failed_validators)}")
    print(f"  Failed: {len(failed_validators)}")

    if failed_validators:
        print("\n❌ FAILED VALIDATORS:")
        for name in failed_validators:
            print(f"  - {name}")

        print("\n💡 TIP: Run individual validators for detailed output:")
        for validator in VALIDATORS:
            if validator["name"] in failed_validators:
                print(f"  python3 .claude/hooks/{validator['script']}")

    else:
        print("\n✅ ALL VALIDATIONS PASSED!")

    print("\n" + "=" * 70 + "\n")

    # Return success if all passed
    return 0 if not failed_validators else 1


if __name__ == "__main__":
    sys.exit(main())
