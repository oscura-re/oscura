#!/usr/bin/env python3
"""Analyze duplicates and create optimal structure mapping.

Identifies functionality overlaps between demonstrations/ and demos/,
and designs the optimal consolidated structure.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
AUDIT_FILE = ROOT / ".claude" / "examples_audit_comprehensive.json"


def load_audit() -> list[dict[str, Any]]:
    """Load audit results."""
    with open(AUDIT_FILE) as f:
        return json.load(f)


def group_by_functionality(analyses: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group files by functionality/topic.

    Returns:
        Dictionary mapping functionality to list of files
    """
    functionality_map = defaultdict(list)

    for analysis in analyses:
        # Extract key topic from docstring and demonstrates
        topics = set()

        # From directory name
        directory = analysis["directory"]
        topics.add(directory.replace("_", " "))

        # From docstring
        if analysis["docstring"]:
            docstring = analysis["docstring"].lower()
            # Extract key terms
            if "uart" in docstring or "spi" in docstring or "i2c" in docstring:
                topics.add("serial protocols")
            if "can" in docstring or "automotive" in docstring:
                topics.add("automotive")
            if "jitter" in docstring:
                topics.add("jitter analysis")
            if "power" in docstring:
                topics.add("power analysis")
            if "emc" in docstring or "compliance" in docstring:
                topics.add("emc compliance")
            if "protocol" in docstring and "inference" in docstring:
                topics.add("protocol inference")
            if "reverse" in docstring or "unknown" in docstring:
                topics.add("reverse engineering")
            if "spectral" in docstring or "fft" in docstring:
                topics.add("spectral analysis")
            if "signal integrity" in docstring:
                topics.add("signal integrity")
            if "loader" in docstring or "loading" in docstring:
                topics.add("data loading")

        # From demonstrates
        for demo in analysis["demonstrates"]:
            demo_lower = demo.lower()
            if "uart" in demo_lower or "spi" in demo_lower:
                topics.add("serial protocols")
            if "automotive" in demo_lower or "can" in demo_lower:
                topics.add("automotive")

        # Add to all relevant topics
        for topic in topics:
            functionality_map[topic].add(analysis)

    return functionality_map


def find_duplicates(analyses: list[dict[str, Any]]) -> list[tuple[dict, dict, float]]:
    """Find duplicate/overlapping examples.

    Returns:
        List of (file1, file2, similarity_score) tuples
    """
    duplicates = []

    # Split by source directory
    demonstrations = [a for a in analyses if "demonstrations/" in a["path"]]
    demos = [a for a in analyses if "demos/" in a["path"]]

    # Compare each demos file against demonstrations
    for demo in demos:
        best_match = None
        best_score = 0.0

        for demonstration in demonstrations:
            score = calculate_similarity(demo, demonstration)
            if score > best_score:
                best_score = score
                best_match = demonstration

        if best_score > 0.3:  # Threshold for considering it a duplicate
            duplicates.append((demo, best_match, best_score))

    return duplicates


def calculate_similarity(file1: dict[str, Any], file2: dict[str, Any]) -> float:
    """Calculate similarity between two files.

    Returns:
        Similarity score from 0.0 to 1.0
    """
    score = 0.0

    # Compare directory names (normalized)
    dir1 = file1["directory"].replace("_", " ")
    dir2 = file2["directory"].replace("_", " ")

    # Direct directory match
    if dir1 == dir2:
        score += 0.4
    elif dir1 in dir2 or dir2 in dir1:
        score += 0.2

    # Compare Oscura APIs used
    apis1 = set(file1["oscura_apis"])
    apis2 = set(file2["oscura_apis"])

    if apis1 and apis2:
        api_overlap = len(apis1 & apis2) / max(len(apis1), len(apis2))
        score += api_overlap * 0.3

    # Compare demonstrates sections
    demo1 = set(d.lower() for d in file1["demonstrates"])
    demo2 = set(d.lower() for d in file2["demonstrates"])

    if demo1 and demo2:
        demo_overlap = len(demo1 & demo2) / max(len(demo1), len(demo2))
        score += demo_overlap * 0.2

    # Compare docstrings
    if file1["docstring"] and file2["docstring"]:
        doc1_words = set(file1["docstring"].lower().split())
        doc2_words = set(file2["docstring"].lower().split())
        doc_overlap = len(doc1_words & doc2_words) / max(len(doc1_words), len(doc2_words))
        score += doc_overlap * 0.1

    return min(score, 1.0)


def design_optimal_structure(analyses: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Design optimal category structure.

    Returns:
        Dictionary mapping category to list of selected file paths
    """
    optimal = {}

    # Core categories based on Oscura capabilities
    categories = {
        "00_getting_started": ["Introduction", "Core types", "Format support"],
        "01_data_loading": ["All file format loaders"],
        "02_basic_analysis": ["Measurements", "Statistics", "Spectral", "Filtering"],
        "03_protocol_decoding": ["All protocol decoders"],
        "04_advanced_analysis": ["Jitter", "Power", "Signal integrity", "Eye diagrams"],
        "05_domain_specific": ["Automotive", "EMC", "Side-channel"],
        "06_reverse_engineering": ["Unknown protocols", "CRC", "State machines"],
        "07_advanced_api": ["Pipelines", "DSL", "Optimization"],
        "08_extensibility": ["Plugins", "Custom measurements"],
        "09_batch_processing": ["Parallel processing", "Aggregation"],
        "10_sessions": ["Analysis sessions", "Persistence"],
        "11_integration": ["CLI", "Jupyter", "Hardware"],
        "12_workflows": ["Complete end-to-end workflows"],
        "13_export_visualization": ["Export formats", "Reports", "Visualization"],
        "14_standards_compliance": ["IEEE standards validation"],
    }

    # For each category, select best examples
    demonstrations = [a for a in analyses if "demonstrations/" in a["path"]]

    for category, description in categories.items():
        category_files = [
            a
            for a in demonstrations
            if a["directory"] == category or a["directory"].startswith(category[:2])
        ]

        if category_files:
            # Select files, preferring those without SKIP_VALIDATION
            selected = sorted(
                category_files, key=lambda x: (x["skip_validation"], x["complexity_score"])
            )
            optimal[category] = [f["path"] for f in selected]

    return optimal


def main() -> None:
    """Main entry point."""
    analyses = load_audit()

    print("=" * 80)
    print("DUPLICATE ANALYSIS")
    print("=" * 80)
    print()

    # Find duplicates
    duplicates = find_duplicates(analyses)

    print(f"Found {len(duplicates)} potential duplicates/overlaps")
    print()

    # Group by similarity score
    high_similarity = [d for d in duplicates if d[2] >= 0.7]
    medium_similarity = [d for d in duplicates if 0.5 <= d[2] < 0.7]
    low_similarity = [d for d in duplicates if 0.3 <= d[2] < 0.5]

    print(f"High similarity (≥70%): {len(high_similarity)}")
    print(f"Medium similarity (50-70%): {len(medium_similarity)}")
    print(f"Low similarity (30-50%): {len(low_similarity)}")
    print()

    # Show high similarity duplicates in detail
    if high_similarity:
        print("HIGH SIMILARITY DUPLICATES (≥70%):")
        print("-" * 80)
        for demo, demonstration, score in sorted(high_similarity, key=lambda x: -x[2]):
            print(f"\nSimilarity: {score:.1%}")
            print(f"  demos:          {demo['path']}")
            print(f"  demonstrations: {demonstration['path']}")
            print(
                f"  Common APIs: {len(set(demo['oscura_apis']) & set(demonstration['oscura_apis']))}"
            )

            # Skip status
            demo_skip = "[SKIP] " if demo["skip_validation"] else ""
            demonstration_skip = "[SKIP] " if demonstration["skip_validation"] else ""
            print(f"  Status: demos {demo_skip}/ demonstrations {demonstration_skip}")

            # Recommendation
            if demonstration["skip_validation"] and not demo["skip_validation"]:
                print(f"  → KEEP: demos/{demo['filename']} (demonstrations has SKIP)")
            elif demo["skip_validation"] and not demonstration["skip_validation"]:
                print(f"  → KEEP: demonstrations/{demonstration['filename']} (demos has SKIP)")
            elif demonstration["complexity_score"] < demo["complexity_score"]:
                print(f"  → KEEP: demonstrations/{demonstration['filename']} (simpler)")
            else:
                print(f"  → KEEP: demonstrations/{demonstration['filename']} (better organized)")

    print()
    print("=" * 80)
    print("CATEGORY COVERAGE")
    print("=" * 80)
    print()

    # Analyze category coverage
    demonstrations = [a for a in analyses if "demonstrations/" in a["path"]]
    demos = [a for a in analyses if "demos/" in a["path"]]

    demonstration_categories = set(a["directory"] for a in demonstrations)
    demo_categories = set(a["directory"] for a in demos)

    print(f"Unique categories in demonstrations/: {len(demonstration_categories)}")
    for cat in sorted(demonstration_categories):
        count = len([a for a in demonstrations if a["directory"] == cat])
        skip_count = len(
            [a for a in demonstrations if a["directory"] == cat and a["skip_validation"]]
        )
        skip_info = f" ({skip_count} with SKIP)" if skip_count > 0 else ""
        print(f"  {cat}: {count} files{skip_info}")

    print()
    print(f"Unique categories in demos/: {len(demo_categories)}")
    for cat in sorted(demo_categories):
        count = len([a for a in demos if a["directory"] == cat])
        skip_count = len([a for a in demos if a["directory"] == cat and a["skip_validation"]])
        skip_info = f" ({skip_count} with SKIP)" if skip_count > 0 else ""
        print(f"  {cat}: {count} files{skip_info}")

    print()

    # Categories only in demos
    demos_only = demo_categories - demonstration_categories
    if demos_only:
        print("Categories ONLY in demos/ (need to migrate):")
        for cat in sorted(demos_only):
            count = len([a for a in demos if a["directory"] == cat])
            print(f"  {cat}: {count} files")
        print()

    # Write detailed report
    report_file = ROOT / ".claude" / "duplicate_analysis_report.json"
    report = {
        "total_duplicates": len(duplicates),
        "high_similarity": [
            {
                "demos_file": d[0]["path"],
                "demonstrations_file": d[1]["path"],
                "similarity": d[2],
                "recommendation": "keep_demonstrations"
                if not d[1]["skip_validation"]
                else "keep_demos",
            }
            for d in high_similarity
        ],
        "medium_similarity": [
            {
                "demos_file": d[0]["path"],
                "demonstrations_file": d[1]["path"],
                "similarity": d[2],
            }
            for d in medium_similarity
        ],
        "categories_demonstrations": sorted(demonstration_categories),
        "categories_demos": sorted(demo_categories),
        "demos_only_categories": sorted(demos_only),
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Detailed report: {report_file}")
    print()


if __name__ == "__main__":
    main()
