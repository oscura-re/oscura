#!/usr/bin/env python3
"""Verification script for synthetic test data system.

This script validates that the synthetic test data generation and loading
system is working correctly.

Usage:
    python scripts/verify_synthetic_test_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def verify_imports() -> bool:
    """Verify required imports are available."""
    print("=" * 60)
    print("1. Verifying Dependencies")
    print("=" * 60)

    try:
        import numpy

        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("✗ numpy not found - install with: pip install numpy")
        return False

    try:
        import tm_data_types

        print(f"✓ tm_data_types {tm_data_types.version}")
    except ImportError:
        print("✗ tm_data_types not found - install with: pip install tm_data_types")
        return False

    try:
        from oscura.loaders.tektronix import load_tektronix_wfm  # noqa: F401

        print("✓ Oscura WFM loader")
    except ImportError as e:
        print(f"✗ Oscura import failed: {e}")
        return False

    print()
    return True


def verify_generator() -> bool:
    """Verify generator script exists and is functional."""
    print("=" * 60)
    print("2. Verifying Generator Script")
    print("=" * 60)

    script_path = Path("scripts/generate_synthetic_wfm.py")

    if not script_path.exists():
        print(f"✗ Generator not found: {script_path}")
        return False

    print(f"✓ Generator exists: {script_path}")
    print(f"  Size: {script_path.stat().st_size:,} bytes")

    # Check if executable
    import stat

    is_executable = script_path.stat().st_mode & stat.S_IXUSR
    if is_executable:
        print("  ✓ Executable")
    else:
        print("  - Not executable (optional)")

    print()
    return True


def verify_test_data() -> bool:
    """Verify synthetic test data exists."""
    print("=" * 60)
    print("3. Verifying Test Data")
    print("=" * 60)

    synthetic_dir = Path("test_data/synthetic")

    if not synthetic_dir.exists():
        print(f"✗ Synthetic data directory not found: {synthetic_dir}")
        print("  Run: python scripts/generate_synthetic_wfm.py --generate-suite")
        return False

    print(f"✓ Synthetic data directory exists: {synthetic_dir}")

    # Count files by category
    categories = {
        "basic": 5,
        "edge_cases": 8,
        "sizes": 4,
        "frequencies": 5,
        "advanced": 7,
    }

    total_expected = 29
    total_found = 0

    for category, expected_count in categories.items():
        cat_dir = synthetic_dir / category
        if not cat_dir.exists():
            print(f"  ✗ Missing category: {category}/")
            continue

        wfm_files = list(cat_dir.glob("*.wfm"))
        found_count = len(wfm_files)
        total_found += found_count

        status = "✓" if found_count == expected_count else "✗"
        print(f"  {status} {category:15s}: {found_count:2d}/{expected_count:2d} files")

    print(f"\n  Total: {total_found}/{total_expected} files")

    if total_found != total_expected:
        print("\n  ⚠ Run: python scripts/generate_synthetic_wfm.py --generate-suite")

    print()
    return total_found == total_expected


def verify_loading() -> bool:
    """Verify WFM files load correctly."""
    print("=" * 60)
    print("4. Verifying File Loading")
    print("=" * 60)

    from oscura.loaders.tektronix import load_tektronix_wfm

    synthetic_dir = Path("test_data/synthetic")
    wfm_files = list(synthetic_dir.glob("**/*.wfm"))

    if not wfm_files:
        print("✗ No WFM files found to test")
        return False

    print(f"Testing {len(wfm_files)} WFM files...\n")

    success = 0
    failures = []

    for wfm_file in sorted(wfm_files):
        try:
            trace = load_tektronix_wfm(wfm_file)

            # Basic validation
            assert trace.data is not None, "No data"
            assert len(trace.data) > 0, "Empty data"
            assert trace.metadata is not None, "No metadata"
            assert trace.metadata.sample_rate > 0, "Invalid sample rate"

            success += 1

        except Exception as e:
            failures.append((wfm_file.name, str(e)))

    # Print summary
    if failures:
        print(f"✗ {len(failures)} files failed to load:\n")
        for name, error in failures[:5]:  # Show first 5
            print(f"  - {name}: {error}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")
    else:
        print(f"✓ All {success} files loaded successfully")

    print()
    return len(failures) == 0


def verify_documentation() -> bool:
    """Verify documentation exists."""
    print("=" * 60)
    print("5. Verifying Documentation")
    print("=" * 60)

    docs = {
        "test_data/README.md": "Test data strategy",
        "docs/guides/synthetic_test_data.md": "Usage guide",
        "docs/guides/test_data_migration.md": "Migration guide",
        "docs/guides/public_test_data_sources.md": "Data sources",
        "SYNTHETIC_TEST_DATA_SUMMARY.md": "Summary",
    }

    all_exist = True

    for doc_path, description in docs.items():
        path = Path(doc_path)
        if path.exists():
            print(f"✓ {doc_path:45s} ({description})")
        else:
            print(f"✗ {doc_path:45s} (missing)")
            all_exist = False

    print()
    return all_exist


def main() -> int:
    """Run all verifications."""
    print("\n" + "=" * 60)
    print("SYNTHETIC TEST DATA VERIFICATION")
    print("=" * 60)
    print()

    # Run verifications
    results = {
        "Dependencies": verify_imports(),
        "Generator": verify_generator(),
        "Test Data": verify_test_data(),
        "File Loading": verify_loading(),
        "Documentation": verify_documentation(),
    }

    # Print final summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {check}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ SUCCESS: All verifications passed!")
        print("\nSynthetic test data system is fully operational.")
        print("\nNext steps:")
        print("  - Use synthetic data in your tests")
        print("  - See docs/guides/synthetic_test_data.md for usage")
        print("  - Generate custom signals as needed")
        return 0
    else:
        print("\n✗ FAILED: Some verifications failed")
        print("\nPlease review the errors above and:")
        print("  1. Install missing dependencies")
        print("  2. Generate test suite if missing")
        print("  3. Check documentation exists")
        return 1


if __name__ == "__main__":
    sys.exit(main())
