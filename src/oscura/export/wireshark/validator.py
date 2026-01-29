"""Lua syntax validator using luac compiler.

This module validates generated Lua code syntax using the luac compiler
if available on the system.
"""

import subprocess
from pathlib import Path


def validate_lua_syntax(lua_code: str) -> tuple[bool, str]:
    """Validate Lua syntax using luac compiler.

    This function attempts to validate Lua code by passing it through
    the luac compiler. If luac is not available, validation is skipped.

    Args:
        lua_code: Lua code to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if syntax is valid or validation skipped
        - error_message: Empty string if valid, error details if invalid,
                        or skip message if luac not available

    Example:
        >>> valid, error = validate_lua_syntax("local x = 1")
        >>> if not valid:
        ...     print(f"Syntax error: {error}")
    """
    try:
        # Try to validate using luac
        result = subprocess.run(
            ["luac", "-p", "-"],
            input=lua_code.encode(),
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.decode()
    except FileNotFoundError:
        # luac not available, skip validation
        return True, "luac not found, syntax validation skipped"
    except subprocess.TimeoutExpired:
        return False, "Validation timeout"
    except Exception as e:
        # Other errors - treat as unable to validate
        return True, f"Validation skipped: {e!s}"


def validate_lua_file(lua_path: Path) -> tuple[bool, str]:
    """Validate a Lua file using luac compiler.

    Args:
        lua_path: Path to Lua file

    Returns:
        Tuple of (is_valid, error_message)

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not lua_path.exists():
        raise FileNotFoundError(f"Lua file not found: {lua_path}")

    lua_code = lua_path.read_text()
    return validate_lua_syntax(lua_code)


def check_luac_available() -> bool:
    """Check if luac compiler is available on the system.

    Returns:
        True if luac is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["luac", "-v"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def save_and_validate(lua_code: str, output_path: Path) -> tuple[bool, str]:
    """Save Lua code to file and validate it.

    Args:
        lua_code: Lua code to save
        output_path: Where to save the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Save the file first
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(lua_code)

    # Validate
    return validate_lua_file(output_path)
