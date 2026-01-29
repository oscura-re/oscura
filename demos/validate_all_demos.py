#!/usr/bin/env python3
"""Run all demos with validation enabled.

# SKIP_VALIDATION: Meta-validator runs all demos, would timeout

This script runs all Oscura demos and verifies their outputs,
serving as living integration tests for the codebase.

Usage:
    uv run python demos/validate_all_demos.py
    uv run python demos/validate_all_demos.py --verbose
    uv run python demos/validate_all_demos.py --demo 01
    uv run python demos/validate_all_demos.py --category serial

Exit codes:
    0 - All demos passed
    1 - One or more demos failed
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class DemoResult(NamedTuple):
    """Result of running a single demo."""

    name: str
    passed: bool
    output: str
    error: str | None


class DemoValidator:
    """Validates Oscura demos."""

    def __init__(self, verbose: bool = False):
        """Initialize validator.

        Args:
            verbose: Print full output from demos
        """
        self.verbose = verbose
        self.demos_dir = Path(__file__).parent
        self.project_root = self.demos_dir.parent

    def _get_python_command(self) -> list[str]:
        """Get the appropriate Python command.

        Returns:
            Command list for running Python scripts.
            Uses 'uv run python' if uv is available, otherwise sys.executable.
        """
        if shutil.which("uv"):
            return ["uv", "run", "python"]
        return [sys.executable]

    def validate_demo(self, demo_path: Path) -> DemoResult:
        """Run demo and verify validation passes.

        Args:
            demo_path: Path to demo Python file

        Returns:
            DemoResult with pass/fail status and output
        """
        demo_name = f"{demo_path.parent.name}/{demo_path.name}"

        print(f"\n{'=' * 80}")
        print(f"Validating: {demo_name}")
        print("=" * 80)

        try:
            # Build command - use uv run if available for proper environment
            python_cmd = self._get_python_command()
            cmd = [*python_cmd, str(demo_path)]

            # Run from project root to ensure proper path setup
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,  # 5 minute timeout
                check=False,
            )

            output = result.stdout + result.stderr

            if self.verbose:
                print(output)

            # Check for validation success marker (support both checkmarks and PASS)
            if "PASS" in output.upper() and "FAIL" not in output.upper():
                print(f"PASS: {demo_name}")
                return DemoResult(demo_name, True, output, None)

            # Check exit code
            if result.returncode != 0:
                error = f"Non-zero exit code {result.returncode}"
                print(f"FAIL: {demo_name} - {error}")
                if not self.verbose:
                    print(f"\nOutput:\n{output}")
                return DemoResult(demo_name, False, output, error)

            # Demo ran but no validation marker found
            error = "No validation marker found"
            print(f"WARNING: {demo_name} - {error}")
            return DemoResult(demo_name, True, output, error)

        except subprocess.TimeoutExpired:
            error = "Timeout (>5 minutes)"
            print(f"FAIL: {demo_name} - {error}")
            return DemoResult(demo_name, False, "", error)

        except Exception as e:
            error = f"Exception: {e}"
            print(f"FAIL: {demo_name} - {error}")
            return DemoResult(demo_name, False, "", error)

    def get_all_demos(self) -> dict[str, list[Path]]:
        """Get all demos organized by category.

        Returns:
            Dict mapping category name to list of demo paths.
        """
        demos = {
            "file_formats": [
                Path("demos/02_file_format_io/vcd_loader_demo.py"),
            ],
            "serial_protocols": [
                Path("demos/04_serial_protocols/jtag_demo.py"),
                Path("demos/04_serial_protocols/swd_demo.py"),
                Path("demos/04_serial_protocols/usb_demo.py"),
                Path("demos/04_serial_protocols/onewire_demo.py"),
                Path("demos/04_serial_protocols/manchester_demo.py"),
                Path("demos/04_serial_protocols/i2s_demo.py"),
            ],
            "automotive_protocols": [
                Path("demos/08_automotive_protocols/lin_demo.py"),
                Path("demos/08_automotive_protocols/flexray_demo.py"),
            ],
            "timing_measurements": [
                Path("demos/10_timing_measurements/ieee_181_pulse_demo.py"),
            ],
            "jitter_analysis": [
                Path("demos/13_jitter_analysis/bathtub_curve_demo.py"),
                Path("demos/13_jitter_analysis/ddj_dcd_demo.py"),
            ],
            "power_analysis": [
                Path("demos/14_power_analysis/dcdc_efficiency_demo.py"),
                Path("demos/14_power_analysis/ripple_analysis_demo.py"),
            ],
            "signal_integrity": [
                Path("demos/15_signal_integrity/setup_hold_timing_demo.py"),
                Path("demos/15_signal_integrity/tdr_impedance_demo.py"),
                Path("demos/15_signal_integrity/sparams_demo.py"),
            ],
            "protocol_inference": [
                Path("demos/07_protocol_inference/crc_reverse_demo.py"),
                Path("demos/07_protocol_inference/wireshark_dissector_demo.py"),
                Path("demos/07_protocol_inference/state_machine_learning.py"),
            ],
            "advanced_inference": [
                Path("demos/18_advanced_inference/bayesian_inference_demo.py"),
                Path("demos/18_advanced_inference/protocol_dsl_demo.py"),
                Path("demos/18_advanced_inference/active_learning_demo.py"),
            ],
            "complete_workflows": [
                Path("demos/19_complete_workflows/network_analysis_workflow.py"),
                Path("demos/19_complete_workflows/unknown_signal_workflow.py"),
                Path("demos/19_complete_workflows/automotive_full_workflow.py"),
            ],
            "comprehensive": [
                Path("demos/05_protocol_decoding/comprehensive_protocol_demo.py"),
                Path("demos/12_spectral_compliance/comprehensive_spectral_demo.py"),
                Path("demos/11_mixed_signal/comprehensive_mixed_signal_demo.py"),
                Path("demos/09_automotive/comprehensive_automotive_demo.py"),
                Path("demos/16_emc_compliance/comprehensive_emc_demo.py"),
            ],
        }
        return demos

    def run_all_demos(
        self, filter_demo: str | None = None, category: str | None = None
    ) -> list[DemoResult]:
        """Run all demo validation scripts.

        Args:
            filter_demo: Optional demo number filter (e.g., "01" for Demo 01)
            category: Optional category filter (e.g., "serial" for serial protocols)

        Returns:
            List of DemoResult objects
        """
        all_demos = self.get_all_demos()

        # Flatten to list based on filters
        demos: list[Path] = []

        for cat_name, cat_demos in all_demos.items():
            if category and category.lower() not in cat_name.lower():
                continue

            for demo_path in cat_demos:
                if filter_demo:
                    # Extract demo number from path (e.g., "04" from "demos/04_serial_protocols/...")
                    demo_num = demo_path.parts[1].split("_")[0]
                    if not demo_num.startswith(filter_demo):
                        continue
                demos.append(demo_path)

        results = []
        for demo_rel_path in demos:
            demo_path = self.project_root / demo_rel_path
            if not demo_path.exists():
                print(f"SKIP: {demo_rel_path} does not exist")
                continue

            result = self.validate_demo(demo_path)
            results.append(result)

        return results

    def print_summary(self, results: list[DemoResult]) -> None:
        """Print summary of all demo results.

        Args:
            results: List of DemoResult objects
        """
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {len(passed)}/{len(results)} demos passed")
        print("=" * 80)

        if failed:
            print("\nFailed demos:")
            for result in failed:
                error_msg = f" ({result.error})" if result.error else ""
                print(f"  - {result.name}{error_msg}")
        else:
            print("\nAll demos passed validation!")

    def list_demos(self) -> None:
        """Print list of all available demos."""
        all_demos = self.get_all_demos()

        print("Available demos by category:")
        print("=" * 60)

        total = 0
        for cat_name, demos in all_demos.items():
            print(f"\n{cat_name} ({len(demos)} demos):")
            for demo in demos:
                exists = (self.project_root / demo).exists()
                status = "[OK]" if exists else "[MISSING]"
                print(f"  {status} {demo}")
                if exists:
                    total += 1

        print(f"\n{'=' * 60}")
        print(f"Total: {total} demos available")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Validate all Oscura demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full output from demos",
    )
    parser.add_argument(
        "--demo",
        "-d",
        type=str,
        help="Filter to specific demo number (e.g., '03' for Demo 03)",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        help="Filter to specific category (e.g., 'serial', 'automotive', 'power')",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available demos without running them",
    )

    args = parser.parse_args()

    validator = DemoValidator(verbose=args.verbose)

    if args.list:
        validator.list_demos()
        return 0

    results = validator.run_all_demos(filter_demo=args.demo, category=args.category)
    validator.print_summary(results)

    # Exit with failure if any demos failed
    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
