#!/usr/bin/env python3
"""Validate documentation links and structure.

This script checks for broken internal links, missing files, and
structural issues in the documentation.

Usage:
    python scripts/validate_docs.py [--verbose] [--fix-suggestions]
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# Directories to skip
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coordination",
    ".claude/agent-outputs",
    ".claude/checkpoints",
    ".claude/summaries",
}

# Required documentation structure
REQUIRED_STRUCTURE = {
    "docs/index.md": "Documentation hub",
    "docs/api/expert-api.md": "Expert API reference",
    "docs/api/pipelines.md": "Pipelines API reference",
    "docs/testing/test-suite-guide.md": "Test suite guide",
    "demos/README.md": "Demos overview",
}

# Required demo categories (19 demos: 01-19)
REQUIRED_DEMO_CATEGORIES = [
    "demos/01_waveform_analysis",
    "demos/02_file_format_io",
    "demos/03_custom_daq",
    "demos/04_serial_protocols",
    "demos/05_protocol_decoding",
    "demos/06_udp_packet_analysis",
    "demos/07_protocol_inference",
    "demos/08_automotive_protocols",
    "demos/09_automotive",
    "demos/10_timing_measurements",
    "demos/11_mixed_signal",
    "demos/12_spectral_compliance",
    "demos/13_jitter_analysis",
    "demos/14_power_analysis",
    "demos/15_signal_integrity",
    "demos/16_emc_compliance",
    "demos/17_signal_reverse_engineering",
    "demos/18_advanced_inference",
    "demos/19_complete_workflows",
]


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped."""
    return any(part in SKIP_DIRS for part in path.parts)


def find_markdown_links(content: str) -> list[tuple[str, str]]:
    """Extract all markdown links from content.

    Returns list of (link_text, link_target) tuples.
    """
    pattern = r"\[([^\]]*)\]\(([^)]+)\)"
    return re.findall(pattern, content)


def is_external_link(link: str) -> bool:
    """Check if a link is external (http/https)."""
    parsed = urlparse(link)
    return parsed.scheme in ("http", "https", "mailto")


def is_anchor_link(link: str) -> bool:
    """Check if a link is an anchor-only link (#section)."""
    return link.startswith("#")


def resolve_link(source_file: Path, link: str, repo_root: Path) -> Path | None:
    """Resolve a relative link to an absolute path."""
    # Remove anchor portion
    link_path = link.split("#")[0]
    if not link_path:
        return None  # Anchor-only link

    # Handle absolute paths (from repo root)
    if link_path.startswith("/"):
        return repo_root / link_path.lstrip("/")

    # Handle relative paths
    return (source_file.parent / link_path).resolve()


def validate_file(file_path: Path, repo_root: Path, verbose: bool = False) -> list[dict]:
    """Validate a single markdown file for broken links."""
    issues = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        issues.append(
            {
                "file": str(file_path.relative_to(repo_root)),
                "type": "read_error",
                "message": f"Cannot read file: {e}",
            }
        )
        return issues

    links = find_markdown_links(content)

    for link_text, link_target in links:
        # Skip external and anchor links
        if is_external_link(link_target) or is_anchor_link(link_target):
            continue

        # Resolve the link
        resolved = resolve_link(file_path, link_target, repo_root)
        if resolved is None:
            continue

        # Check if target exists
        if not resolved.exists():
            issues.append(
                {
                    "file": str(file_path.relative_to(repo_root)),
                    "type": "broken_link",
                    "link_text": link_text,
                    "link_target": link_target,
                    "expected_path": str(resolved.relative_to(repo_root))
                    if resolved.is_relative_to(repo_root)
                    else str(resolved),
                }
            )
            if verbose:
                print(
                    f"  BROKEN: [{link_text}]({link_target}) in {file_path.relative_to(repo_root)}"
                )

    return issues


def check_structure(repo_root: Path, verbose: bool = False) -> list[dict]:
    """Check for required documentation structure."""
    issues = []

    # Check required files
    for path, description in REQUIRED_STRUCTURE.items():
        full_path = repo_root / path
        if not full_path.exists():
            issues.append(
                {
                    "type": "missing_required",
                    "path": path,
                    "description": description,
                }
            )
            if verbose:
                print(f"  MISSING: {path} ({description})")

    # Check demo categories
    for category in REQUIRED_DEMO_CATEGORIES:
        cat_path = repo_root / category
        if not cat_path.exists():
            issues.append(
                {
                    "type": "missing_category",
                    "path": category,
                }
            )
            if verbose:
                print(f"  MISSING CATEGORY: {category}")
        else:
            # Check for README in category
            readme = cat_path / "README.md"
            if not readme.exists():
                issues.append(
                    {
                        "type": "missing_readme",
                        "path": f"{category}/README.md",
                    }
                )
                if verbose:
                    print(f"  MISSING README: {category}/README.md")

    return issues


def run_validation(repo_root: Path, verbose: bool = False, fix_suggestions: bool = False) -> int:
    """Run the full validation."""
    print("=" * 60)
    print("Oscura Documentation Validation")
    print("=" * 60)

    all_issues = []

    # Check structure
    print("\n--- Checking structure ---")
    structure_issues = check_structure(repo_root, verbose)
    all_issues.extend(structure_issues)

    # Check links in all markdown files
    print("\n--- Checking links ---")
    md_files = list(repo_root.glob("**/*.md"))
    files_checked = 0

    for file_path in md_files:
        if should_skip(file_path):
            continue

        link_issues = validate_file(file_path, repo_root, verbose)
        all_issues.extend(link_issues)
        files_checked += 1

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"  Files checked: {files_checked}")

    broken_links = [i for i in all_issues if i["type"] == "broken_link"]
    missing_required = [i for i in all_issues if i["type"] == "missing_required"]
    missing_categories = [i for i in all_issues if i["type"] == "missing_category"]
    missing_readmes = [i for i in all_issues if i["type"] == "missing_readme"]

    print(f"  Broken links: {len(broken_links)}")
    print(f"  Missing required files: {len(missing_required)}")
    print(f"  Missing demo categories: {len(missing_categories)}")
    print(f"  Missing category READMEs: {len(missing_readmes)}")

    if all_issues:
        print("\n--- Issues Found ---")

        if missing_required:
            print("\nMissing Required Files:")
            for issue in missing_required:
                print(f"  - {issue['path']}: {issue['description']}")

        if missing_categories:
            print("\nMissing Demo Categories:")
            for issue in missing_categories:
                print(f"  - {issue['path']}")

        if missing_readmes:
            print("\nMissing Category READMEs:")
            for issue in missing_readmes:
                print(f"  - {issue['path']}")

        if broken_links:
            print("\nBroken Links:")
            for issue in broken_links:
                print(f"  - {issue['file']}: [{issue['link_text']}]({issue['link_target']})")
                if fix_suggestions:
                    print(f"    Expected: {issue['expected_path']}")

        return 1  # Non-zero exit code for CI

    print("\nAll validations passed!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate Oscura documentation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fix-suggestions", action="store_true", help="Show suggestions for fixing issues"
    )
    args = parser.parse_args()

    # Find repo root (parent of scripts/)
    repo_root = Path(__file__).parent.parent

    exit_code = run_validation(repo_root, args.verbose, args.fix_suggestions)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
