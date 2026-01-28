"""Test all examples execute successfully.

This test suite validates that all example files in the repository
can execute without errors. This ensures documentation and examples
stay synchronized with the codebase.

NOTE: This test file is EXCLUDED from regular CI (per integration-tests batch)
because it's slow (217 tests, 60s timeout each) and validates documentation
rather than core functionality. Core code is covered by 20,493 unit/integration
tests with 80%+ coverage.

This test runs in:
- Nightly CI (.github/workflows/examples-nightly.yml)
- When examples/ or demos/ directories change
- Manual workflow dispatch

Rationale: Separates fast, reliable core CI from slower documentation validation.
Example failures indicate documentation drift, not code bugs, and shouldn't
block code merges.
"""

import subprocess
from pathlib import Path

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Find all example Python files
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEMONSTRATIONS_DIR = PROJECT_ROOT / "demonstrations"
DEMOS_DIR = PROJECT_ROOT / "demos"

# Collect all example files
_all_examples: list[Path] = []
for directory in [EXAMPLES_DIR, DEMONSTRATIONS_DIR, DEMOS_DIR]:
    if directory.exists():
        _all_examples.extend(
            path
            for path in directory.rglob("*.py")
            if path.name != "__init__.py"
            and "common" not in path.parts
            and "data_generation" not in path.parts
        )

EXAMPLES = sorted(_all_examples)


def should_skip_example(example_path: Path) -> tuple[bool, str]:
    """Check if example should be skipped.

    Args:
        example_path: Path to example file

    Returns:
        Tuple of (should_skip, reason)
    """
    # Read first 50 lines to check for skip markers
    try:
        with open(example_path, encoding="utf-8") as f:
            content = "".join(f.readline() for _ in range(50))

        # Check for skip validation marker
        if "# SKIP_VALIDATION" in content:
            return True, "marked for manual testing"

        # Check for external dependencies
        if "REQUIRES:" in content:
            # Extract requirements
            for line in content.split("\n"):
                if "REQUIRES:" in line:
                    return (
                        True,
                        f"requires external dependencies: {line.split('REQUIRES:')[1].strip()}",
                    )

        return False, ""
    except Exception as e:
        return True, f"error reading file: {e}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_example_runs(example: Path) -> None:
    """Test example executes without error.

    Args:
        example: Path to example file

    Raises:
        AssertionError: If example fails to execute
    """
    # Check if should skip
    should_skip, reason = should_skip_example(example)
    if should_skip:
        pytest.skip(reason)

    # Get relative path for better error messages
    rel_path = example.relative_to(PROJECT_ROOT)

    # Run example with uv (ensures dependencies are available)
    result = subprocess.run(
        ["uv", "run", "python", str(example)],
        capture_output=True,
        text=True,
        timeout=60,  # 60 second timeout per example
        cwd=PROJECT_ROOT,
        check=False,  # We handle errors explicitly below
    )

    # Check result
    if result.returncode != 0:
        error_msg = f"Example failed: {rel_path}\n"
        error_msg += f"Exit code: {result.returncode}\n"
        error_msg += f"\nSTDOUT:\n{result.stdout}\n"
        error_msg += f"\nSTDERR:\n{result.stderr}\n"
        pytest.fail(error_msg)


def test_examples_found() -> None:
    """Verify that we found example files to test."""
    assert len(EXAMPLES) > 0, "No example files found"
    assert len(EXAMPLES) >= 50, f"Expected at least 50 examples, found {len(EXAMPLES)}"


def test_all_directories_exist() -> None:
    """Verify example directories exist."""
    assert EXAMPLES_DIR.exists(), f"Examples directory not found: {EXAMPLES_DIR}"
    assert DEMONSTRATIONS_DIR.exists(), f"Demonstrations directory not found: {DEMONSTRATIONS_DIR}"
    assert DEMOS_DIR.exists(), f"Demos directory not found: {DEMOS_DIR}"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
