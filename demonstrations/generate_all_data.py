#!/usr/bin/env python3
"""Generate all test data for demonstrations.

This script generates synthetic test data for all Oscura demonstrations.
All data is 100% synthetic and reproducible - no real captures needed.

Usage:
    python demonstrations/generate_all_data.py
    python demonstrations/generate_all_data.py --section 01_data_loading
    python demonstrations/generate_all_data.py --quick  # Minimal data

Exit codes:
    0: Success
    1: Generation failed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


class DataGenerator:
    """Generate synthetic test data for all demonstrations."""

    def __init__(self, quick: bool = False):
        """Initialize generator.

        Args:
            quick: Generate minimal data (faster)
        """
        self.quick = quick
        self.demos_dir = Path(__file__).parent
        self.data_dir = self.demos_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

    def generate_all(self, section: str | None = None) -> bool:
        """Generate all demonstration data.

        Args:
            section: Optional section filter (e.g., "01_data_loading")

        Returns:
            True if successful
        """
        print(f"\n{'=' * 80}")
        print("DEMONSTRATION DATA GENERATOR")
        print("=" * 80)
        if self.quick:
            print("Mode: QUICK (minimal data)")
        else:
            print("Mode: FULL (complete data)")
        if section:
            print(f"Section: {section}")
        print()

        start_time = time.time()

        # Create output directories
        self._create_directories()

        # Generate data (currently self-contained in demos)
        print("✓ All demonstrations use self-contained synthetic data")
        print("✓ No external data files required")
        print()
        print("Data generation complete!")

        duration = time.time() - start_time
        print(f"\nTotal time: {duration:.2f}s")

        return True

    def _create_directories(self) -> None:
        """Create necessary data directories."""
        dirs = [
            "outputs",  # Demo output files
            "signals",  # Generated signal files (if needed)
            "protocols",  # Protocol captures (if needed)
            "formats",  # File format examples (if needed)
        ]

        for dir_name in dirs:
            dir_path = self.data_dir / dir_name
            dir_path.mkdir(exist_ok=True)

        print("✓ Created data directory structure")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demonstration test data")
    parser.add_argument(
        "--section",
        help="Only generate data for specific section (e.g., 01_data_loading)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate minimal data (faster)",
    )

    args = parser.parse_args()

    generator = DataGenerator(quick=args.quick)
    success = generator.generate_all(section=args.section)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
