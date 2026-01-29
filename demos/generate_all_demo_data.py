#!/usr/bin/env python3
"""Master Demo Data Generation Script for Oscura.

This script generates OPTIMAL, REALISTIC, REPRESENTATIVE demo data for ALL Oscura demos.
Each demo gets dedicated demo_data/ subdirectory with files showcasing ALL capabilities.

Generated Data (9 demo categories with data generators):
    - demos/01_waveform_analysis/demo_data/ - Multi-channel WFM files
    - demos/03_custom_daq/demo_data/ - Binary DAQ files (large file streaming)
    - demos/05_protocol_decoding/demo_data/ - Protocol captures
    - demos/06_udp_packet_analysis/demo_data/ - PCAP files
    - demos/09_automotive/demo_data/ - CAN/LIN/OBD-II captures
    - demos/11_mixed_signal/demo_data/ - Eye diagrams, jitter test signals
    - demos/12_spectral_compliance/demo_data/ - Spectral test signals
    - demos/16_emc_compliance/demo_data/ - EMC test data with limits
    - demos/17_signal_reverse_engineering/demo_data/ - Mystery signals

Usage:
    # Generate ALL demo data (uses uv run for proper environment)
    uv run python demos/generate_all_demo_data.py

    # Generate specific demos only (by generation index 01-09)
    uv run python demos/generate_all_demo_data.py --demos 01,02,05

    # Regenerate (overwrite existing)
    uv run python demos/generate_all_demo_data.py --force

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class DemoDataGenerator:
    """Master generator for all demo data."""

    def __init__(self, force: bool = False, demos: list[str] | None = None):
        """Initialize generator.

        Args:
            force: Overwrite existing data
            demos: List of demo numbers to generate (e.g., ["01", "02"]) or None for all
        """
        self.force = force
        self.demos_root = Path(__file__).parent
        self.project_root = self.demos_root.parent
        self.demos_to_generate = demos or [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
        ]

        # Maps generation index to actual demo directory
        # These are the 9 demos that have data generators
        self.demo_info = {
            "01": {
                "name": "Waveform Analysis",
                "dir": "01_waveform_analysis",
                "generator": "generate_demo_data.py",
                "expected_files": 3,
                "expected_size_mb": 15,
            },
            "02": {
                "name": "Custom DAQ",
                "dir": "03_custom_daq",
                "generator": "generate_demo_data.py",
                "expected_files": 1,
                "expected_size_mb": 80,  # Only small file (skip large to avoid CI timeout)
            },
            "03": {
                "name": "UDP Packet Analysis",
                "dir": "06_udp_packet_analysis",
                "generator": "generate_demo_data.py",
                "expected_files": 3,
                "expected_size_mb": 10,
            },
            "04": {
                "name": "Signal Reverse Engineering",
                "dir": "17_signal_reverse_engineering",
                "generator": "generate_demo_data.py",
                "expected_files": 3,
                "expected_size_mb": 15,
            },
            "05": {
                "name": "Protocol Decoding",
                "dir": "05_protocol_decoding",
                "generator": "generate_demo_data.py",
                "expected_files": 4,
                "expected_size_mb": 11,
            },
            "06": {
                "name": "Spectral Compliance",
                "dir": "12_spectral_compliance",
                "generator": "generate_demo_data.py",
                "expected_files": 3,
                "expected_size_mb": 12,
            },
            "07": {
                "name": "Mixed-Signal",
                "dir": "11_mixed_signal",
                "generator": "generate_demo_data.py",
                "expected_files": 3,
                "expected_size_mb": 16,
            },
            "08": {
                "name": "Automotive",
                "dir": "09_automotive",
                "generator": "generate_demo_data.py",
                "expected_files": 5,
                "expected_size_mb": 24,
            },
            "09": {
                "name": "EMC Compliance",
                "dir": "16_emc_compliance",
                "generator": "generate_demo_data.py",
                "expected_files": 5,
                "expected_size_mb": 25,
            },
        }

        self.stats: dict[str, Any] = {}

    def _get_python_command(self) -> list[str]:
        """Get the appropriate Python command.

        Returns:
            Command list for running Python scripts.
            Uses 'uv run python' if uv is available, otherwise python3.
        """
        if shutil.which("uv"):
            return ["uv", "run", "python"]
        return ["python3"]

    def print_header(self, title: str) -> None:
        """Print section header."""
        print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
        print(f"{BOLD}{BLUE}{title:^80}{RESET}")
        print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")

    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"{GREEN}[OK]{RESET} {message}")

    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"{RED}[FAIL]{RESET} {message}")

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"{YELLOW}[WARN]{RESET} {message}")

    def print_info(self, message: str) -> None:
        """Print info message."""
        print(f"{BLUE}[INFO]{RESET} {message}")

    def create_directory_structure(self) -> None:
        """Create demo_data directories for all demos."""
        self.print_header("CREATING DIRECTORY STRUCTURE")

        for demo_num in self.demos_to_generate:
            info = self.demo_info[demo_num]
            demo_dir = self.demos_root / info["dir"]
            data_dir = demo_dir / "demo_data"

            if not demo_dir.exists():
                self.print_warning(f"Demo directory not found: {demo_dir}")
                continue

            if data_dir.exists() and not self.force:
                self.print_info(f"Directory exists: {data_dir.relative_to(self.demos_root)}")
            else:
                data_dir.mkdir(parents=True, exist_ok=True)
                self.print_success(f"Created: {data_dir.relative_to(self.demos_root)}")

    def generate_demo_data(self, demo_num: str) -> bool:
        """Generate data for a specific demo.

        Args:
            demo_num: Demo number (e.g., "01")

        Returns:
            True if successful, False otherwise
        """
        info = self.demo_info[demo_num]
        demo_dir = self.demos_root / info["dir"]
        generator_script = demo_dir / info["generator"]

        if not generator_script.exists():
            self.print_error(f"Generator not found: {generator_script}")
            return False

        self.print_info(f"Running: {info['dir']}/{info['generator']}")

        try:
            # Run generator script with proper environment
            python_cmd = self._get_python_command()
            cmd = [*python_cmd, str(generator_script)]
            if self.force:
                cmd.append("--force")

            # Skip large files for Custom DAQ demo (avoids CI timeouts)
            if info["dir"] == "03_custom_daq":
                cmd.append("--skip-large")

            result = subprocess.run(
                cmd,
                cwd=self.project_root,  # Run from project root for proper imports
                capture_output=True,
                text=True,
                timeout=150,  # 2.5 minute timeout per generator
                check=False,
            )

            if result.returncode == 0:
                self.print_success(f"Generated data for {info['name']}")
                # Print generator output if verbose
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        print(f"    {line}")
                return True
            else:
                self.print_error(f"Failed to generate data for {info['name']}")
                if result.stderr:
                    print(f"    Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.print_error(f"Timeout generating data for {info['name']}")
            return False
        except Exception as e:
            self.print_error(f"Exception: {e}")
            return False

    def validate_demo_data(self, demo_num: str) -> dict[str, Any]:
        """Validate generated data for a demo.

        Args:
            demo_num: Demo number

        Returns:
            Validation results dict
        """
        info = self.demo_info[demo_num]
        demo_dir = self.demos_root / info["dir"]
        data_dir = demo_dir / "demo_data"

        results = {
            "exists": data_dir.exists(),
            "file_count": 0,
            "total_size_mb": 0.0,
            "files": [],
        }

        if not data_dir.exists():
            return results

        # Count files and total size
        for file in data_dir.iterdir():
            if file.is_file() and not file.name.startswith("."):
                size_mb = file.stat().st_size / (1024 * 1024)
                results["files"].append({"name": file.name, "size_mb": size_mb})
                results["total_size_mb"] += size_mb
                results["file_count"] += 1

        return results

    def generate_all(self) -> dict[str, Any]:
        """Generate all demo data.

        Returns:
            Generation statistics
        """
        self.print_header("OSCURA DEMO DATA GENERATION")

        # Create directories
        self.create_directory_structure()

        # Generate data for each demo
        self.print_header("GENERATING DEMO DATA")

        results = {}
        total = len(self.demos_to_generate)
        for idx, demo_num in enumerate(self.demos_to_generate, 1):
            info = self.demo_info[demo_num]
            print(f"\n{BOLD}[{idx}/{total}] {info['name']} ({info['dir']}){RESET}")
            print("-" * 80)

            success = self.generate_demo_data(demo_num)
            validation = self.validate_demo_data(demo_num)

            results[demo_num] = {
                "success": success,
                "validation": validation,
            }

        # Summary report
        self.print_summary(results)

        return results

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print summary report.

        Args:
            results: Generation results
        """
        self.print_header("GENERATION SUMMARY")

        total_files = 0
        total_size_mb = 0.0
        successful_demos = 0
        failed_demos = 0

        print(f"{BOLD}{'Demo':<35} {'Dir':<25} {'Files':<8} {'Size (MB)':<12} {'Status'}{RESET}")
        print("-" * 95)

        for demo_num in sorted(results.keys()):
            info = self.demo_info[demo_num]
            result = results[demo_num]
            validation = result["validation"]

            status = f"{GREEN}OK{RESET}" if result["success"] else f"{RED}FAIL{RESET}"

            print(
                f"{info['name']:<35} "
                f"{info['dir']:<25} "
                f"{validation['file_count']:<8} "
                f"{validation['total_size_mb']:<12.2f} "
                f"{status}"
            )

            if result["success"]:
                successful_demos += 1
                total_files += validation["file_count"]
                total_size_mb += validation["total_size_mb"]
            else:
                failed_demos += 1

        print("-" * 95)
        print(f"{BOLD}{'TOTAL':<35} {'':<25} {total_files:<8} {total_size_mb:<12.2f}{RESET}")

        print(f"\n{BOLD}Results:{RESET}")
        print(f"  Successful: {GREEN}{successful_demos}/{len(results)}{RESET}")
        if failed_demos > 0:
            print(f"  Failed: {RED}{failed_demos}/{len(results)}{RESET}")

        print(f"\n{BOLD}Total Data Generated:{RESET} {total_size_mb:.2f} MB")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate all Oscura demo data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data",
    )
    parser.add_argument(
        "--demos",
        type=str,
        help="Comma-separated list of generation indices to run (e.g., '01,02,05')",
    )

    args = parser.parse_args()

    # Parse demo list if provided
    demos_to_generate = None
    if args.demos:
        demos_to_generate = [d.strip() for d in args.demos.split(",")]

    # Create generator and run
    generator = DemoDataGenerator(force=args.force, demos=demos_to_generate)
    results = generator.generate_all()

    # Check for failures
    failed = any(not r["success"] for r in results.values())

    print(f"\n{GREEN if not failed else YELLOW}{BOLD}Demo data generation complete!{RESET}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
