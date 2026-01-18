"""Fixtures for integration reporting tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def comprehensive_test_data():
    """Generate comprehensive test data for validation tests.

    Runs the test data generation script once per test session.
    If generation fails, tests will fail (not skip).
    """
    test_data_dir = (
        Path(__file__).parent.parent.parent.parent / "test_data" / "comprehensive_validation"
    )

    # Generate data if it doesn't exist
    if not test_data_dir.exists():
        script_path = (
            Path(__file__).parent.parent.parent.parent
            / "scripts"
            / "generate_comprehensive_test_data.py"
        )

        result = subprocess.run(
            ["python", str(script_path), str(test_data_dir)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            pytest.fail(
                f"Failed to generate comprehensive test data:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

    return test_data_dir
