#!/usr/bin/env python3
"""Extract DTC database from Python module to JSON.

This script extracts the DTC database from the inline Python dictionary
in src/oscura/automotive/dtc/database.py and saves it as a JSON file.

Usage:
    python scripts/data/extract_dtc_database.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

# Import DTC database
from oscura.automotive.dtc.database import DTCS


def extract_dtc_database() -> dict:
    """Extract DTC database to structured JSON format.

    Returns:
        Dictionary containing metadata and all DTC entries.
    """
    # Count categories
    categories = Counter(dtc.category for dtc in DTCS.values())

    # Build output structure
    output = {
        "version": "1.0.0",
        "format": "dtc-database",
        "metadata": {
            "total_codes": len(DTCS),
            "categories": dict(categories),
            "standards": ["SAE J2012", "ISO 14229"],
            "description": "Diagnostic Trouble Code database for automotive diagnostics",
        },
        "dtcs": {code: asdict(info) for code, info in DTCS.items()},
    }

    return output


def main() -> None:
    """Extract DTC database and save to JSON file."""
    print("Extracting DTC database...")

    # Extract database
    output = extract_dtc_database()

    # Determine output path (src/oscura/automotive/dtc/data.json)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    output_path = repo_root / "src" / "oscura" / "automotive" / "dtc" / "data.json"

    # Save to JSON
    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\nExtraction complete!")
    print(f"  Total codes: {output['metadata']['total_codes']}")
    print(f"  Categories: {dict(output['metadata']['categories'])}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
