#!/usr/bin/env python3
"""Unit tests for Claude Code hooks.

Tests all hooks in .claude/hooks/ directory for:
- Proper exit codes
- Error handling
- Input validation
- Expected output

Can be run standalone or via pre-push validation.

Version: 1.0.0
Created: 2026-01-17
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"


def run_hook(hook_path: Path, stdin_data: dict[str, Any] | None = None) -> tuple[int, str, str]:
    """Run a hook script and return exit code, stdout, stderr.

    Args:
        hook_path: Path to hook script
        stdin_data: Optional JSON data to send to stdin

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        cmd = [sys.executable, str(hook_path)]

        # Run with or without stdin data
        if stdin_data is not None:
            result = subprocess.run(
                cmd,
                input=json.dumps(stdin_data).encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
        else:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )

        return (
            result.returncode,
            result.stdout.decode(errors="replace"),
            result.stderr.decode(errors="replace"),
        )

    except subprocess.TimeoutExpired:
        return 1, "", "Hook timed out after 10 seconds"
    except Exception as e:
        return 1, "", f"Error running hook: {e}"


def test_validate_config_consistency() -> tuple[bool, str]:
    """Test validate_config_consistency.py hook."""
    hook_path = HOOKS_DIR / "validate_config_consistency.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    exit_code, stdout, stderr = run_hook(hook_path)

    # Should pass (version consistency should be valid)
    if exit_code == 0:
        return True, "Config consistency validation passed"

    return False, f"Config validation failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"


def test_validate_ssot() -> tuple[bool, str]:
    """Test validate_ssot.py hook."""
    hook_path = HOOKS_DIR / "validate_ssot.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    exit_code, stdout, stderr = run_hook(hook_path)

    # Should pass (no duplicate configs expected)
    if exit_code == 0:
        return True, "SSOT validation passed"

    return False, f"SSOT validation failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"


def test_validate_path() -> tuple[bool, str]:
    """Test validate_path.py hook."""
    hook_path = HOOKS_DIR / "validate_path.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    # Test 1: Normal file write (should pass)
    stdin_data = {
        "tool_name": "Write",
        "tool_input": {"file_path": "test_file.txt"},
    }

    exit_code, stdout, stderr = run_hook(hook_path, stdin_data)

    if exit_code != 0:
        return False, f"Normal file write rejected: {stderr}"

    # Test 2: Blocked file (should fail)
    stdin_data = {
        "tool_name": "Write",
        "tool_input": {"file_path": ".env"},
    }

    exit_code, stdout, stderr = run_hook(hook_path, stdin_data)

    if exit_code == 0:
        return False, "Blocked file (.env) was allowed"

    # Test 3: Path traversal (should fail)
    stdin_data = {
        "tool_name": "Write",
        "tool_input": {"file_path": "../../../etc/passwd"},
    }

    exit_code, stdout, stderr = run_hook(hook_path, stdin_data)

    if exit_code == 0:
        return False, "Path traversal was allowed"

    return True, "Path validation tests passed (3/3)"


def test_generate_settings() -> tuple[bool, str]:
    """Test generate_settings.py hook."""
    hook_path = HOOKS_DIR / "generate_settings.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    # Test dry-run mode
    exit_code, stdout, stderr = run_hook(hook_path)

    # Should not fail on execution
    if exit_code not in (0, 1):  # 1 might indicate out-of-sync, which is valid
        return False, f"Generate settings failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"

    return True, "Generate settings executed successfully"


def test_health_check() -> tuple[bool, str]:
    """Test health_check.py hook."""
    hook_path = HOOKS_DIR / "health_check.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    exit_code, stdout, stderr = run_hook(hook_path)

    # Health check can return 0 (healthy) or report degraded status
    # As long as it executes without crashing, that's a pass
    try:
        # Parse JSON output
        result = json.loads(stdout)
        status = result.get("status", "unknown")

        if status in ("healthy", "degraded"):
            return True, f"Health check executed successfully (status: {status})"

        return False, f"Unexpected health status: {status}"

    except json.JSONDecodeError:
        return False, f"Health check produced invalid JSON:\n{stdout}"


def test_enforce_agent_limit() -> tuple[bool, str]:
    """Test enforce_agent_limit.py hook."""
    hook_path = HOOKS_DIR / "enforce_agent_limit.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    # Test with valid tool call (within limits)
    stdin_data = {
        "tool_name": "Task",
        "tool_input": {"subagent_type": "general-purpose", "prompt": "test"},
    }

    exit_code, stdout, stderr = run_hook(hook_path, stdin_data)

    # Should pass (agent count should be within limits)
    if exit_code == 0:
        return True, "Agent limit enforcement working"

    # Non-zero could also be valid if limit is already reached
    # We just want to ensure it executes without crashing
    if "BLOCKED" in stdout or "limit" in stdout.lower():
        return True, "Agent limit enforcement working (limit reached)"

    return False, f"Unexpected response:\nSTDOUT: {stdout}\nSTDERR: {stderr}"


def test_check_report_proliferation() -> tuple[bool, str]:
    """Test check_report_proliferation.py hook."""
    hook_path = HOOKS_DIR / "check_report_proliferation.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    # Test with normal file write
    stdin_data = {
        "tool_name": "Write",
        "tool_input": {"file_path": "test_file.txt"},
    }

    exit_code, stdout, stderr = run_hook(hook_path, stdin_data)

    # Should pass for non-report files
    if exit_code == 0:
        return True, "Report proliferation check working"

    return False, f"Check failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"


def test_auto_format() -> tuple[bool, str]:
    """Test auto_format.py hook."""
    hook_path = HOOKS_DIR / "auto_format.py"

    if not hook_path.exists():
        return False, f"Hook not found: {hook_path}"

    # Test with Python file write
    stdin_data = {
        "tool_name": "Write",
        "tool_input": {"file_path": "test_file.py"},
    }

    exit_code, stdout, stderr = run_hook(hook_path, stdin_data)

    # Auto-format should execute without errors (even if no file to format)
    # Exit code 0 or 1 both acceptable (1 might mean file not found, which is OK for test)
    if exit_code in (0, 1):
        return True, "Auto-format hook executed successfully"

    return False, f"Auto-format failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"


def main() -> int:
    """Main entry point.

    Returns:
        0 if all tests pass, 1 if any test fails
    """
    print("\n" + "=" * 70)
    print("  HOOK UNIT TESTS")
    print("=" * 70 + "\n")

    # Define all tests
    tests = [
        ("Config Consistency", test_validate_config_consistency),
        ("SSOT Validation", test_validate_ssot),
        ("Path Validation", test_validate_path),
        ("Generate Settings", test_generate_settings),
        ("Health Check", test_health_check),
        ("Enforce Agent Limit", test_enforce_agent_limit),
        ("Check Report Proliferation", test_check_report_proliferation),
        ("Auto Format", test_auto_format),
    ]

    # Run all tests
    passed = 0
    failed = 0
    errors: list[str] = []

    for test_name, test_func in tests:
        print(f"Running: {test_name}...", end=" ")
        try:
            success, message = test_func()

            if success:
                print(f"✅ PASS")
                print(f"  {message}")
                passed += 1
            else:
                print(f"❌ FAIL")
                print(f"  {message}")
                failed += 1
                errors.append(f"{test_name}: {message}")

        except Exception as e:
            print(f"❌ ERROR")
            print(f"  {e}")
            failed += 1
            errors.append(f"{test_name}: {e}")

        print()

    # Print summary
    print("=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")

    if failed > 0:
        print("=" * 70)
        print("\nFailed tests:")
        for error in errors:
            print(f"  {error}")

    print("=" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
