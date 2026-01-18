#!/usr/bin/env python3
"""Generate architecture diagrams for Oscura documentation.

This script generates:
1. Module dependency graphs using pydeps
2. UML class diagrams using pyreverse
3. Package structure visualizations

All diagrams are saved to docs/images/architecture/
"""

import subprocess
import sys
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_IMAGES = PROJECT_ROOT / "docs" / "images" / "architecture"
SRC_ROOT = PROJECT_ROOT / "src" / "oscura"


def ensure_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {DOCS_IMAGES}")


def generate_module_dependencies() -> bool:
    """Generate module dependency graph using pydeps.

    Returns:
        True if successful, False otherwise.
    """
    print("\n=== Generating Module Dependency Graph ===")
    print("  Generating module dependencies...", end=" ", flush=True)
    output_file = DOCS_IMAGES / "module-dependencies.svg"

    try:
        subprocess.run(
            [
                "pydeps",
                str(SRC_ROOT),
                "--max-bacon=2",  # Reduced depth for performance
                "--cluster",  # Group by package
                "--no-show",  # Don't open browser
                "-o",
                str(output_file),
                "--exclude",
                "tests",
                "examples",
                "scripts",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,  # Can take a while on large codebases
        )
        print("✓ Module dependencies")
        return True
    except subprocess.TimeoutExpired:
        print("⚠ Timeout (120s, skipping)")
        return False
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[:100] if e.stderr else "unknown"
        print(f"✗ Failed: {error_msg}")
        return False
    except FileNotFoundError:
        print("\n✗ pydeps not found - install with: uv add --dev pydeps")
        return False


def generate_package_diagrams() -> bool:
    """Generate UML diagrams for major packages using pyreverse.

    Returns:
        True if successful, False otherwise.
    """
    print("\n=== Generating UML Package Diagrams ===")

    # (package_name, description, timeout_seconds)
    packages = [
        ("analyzers", "Signal Analysis Analyzers", 60),
        ("loaders", "Waveform Loaders", 30),
        ("reporting", "Report Generation", 30),
        ("inference", "Protocol Inference", 30),
    ]

    success_count = 0

    for package_name, description, timeout in packages:
        package_path = SRC_ROOT / package_name
        if not package_path.exists():
            print(f"⚠ Skipping {package_name} (not found)")
            continue

        print(f"  Generating {package_name}...", end=" ", flush=True)

        try:
            subprocess.run(
                [
                    "pyreverse",
                    "-o",
                    "svg",
                    "-p",
                    package_name,
                    "--output-directory",
                    str(DOCS_IMAGES),
                    "--colorized",
                    "--max-color-depth=2",  # Reduced for performance
                    "--only-classnames",  # Simpler diagrams
                    str(package_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # pyreverse creates files with specific naming
            class_diagram = DOCS_IMAGES / f"classes_{package_name}.svg"
            if class_diagram.exists():
                print(f"✓ {description}")
                success_count += 1
            else:
                print("⚠ No output")

        except subprocess.TimeoutExpired:
            print(f"⚠ Timeout ({timeout}s, skipping)")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr[:100] if e.stderr else "unknown"
            print(f"✗ Failed: {error_msg}")
        except FileNotFoundError:
            print("\n✗ pyreverse not found - install with: uv add --dev pylint")
            return False

    return success_count > 0


def generate_top_level_architecture() -> bool:
    """Generate top-level architecture diagram using pyreverse.

    Returns:
        True if successful, False otherwise.
    """
    print("\n=== Generating Top-Level Architecture ===")
    print("  Generating oscura overview...", end=" ", flush=True)

    try:
        subprocess.run(
            [
                "pyreverse",
                "-o",
                "svg",
                "-p",
                "oscura",
                "--output-directory",
                str(DOCS_IMAGES),
                "--colorized",
                "--max-color-depth=2",
                "--only-classnames",
                str(SRC_ROOT),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,  # Top-level can take longer
        )

        packages_diagram = DOCS_IMAGES / "packages_oscura.svg"
        classes_diagram = DOCS_IMAGES / "classes_oscura.svg"

        success = False
        if packages_diagram.exists():
            print("✓ Package architecture")
            success = True
        if classes_diagram.exists():
            print("✓ Class architecture")
            success = True

        if not success:
            print("⚠ No diagrams generated")

        return success

    except subprocess.TimeoutExpired:
        print("⚠ Timeout (90s, skipping)")
        return False
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[:100] if e.stderr else "unknown"
        print(f"✗ Failed: {error_msg}")
        return False
    except FileNotFoundError:
        print("\n✗ pyreverse not found - install with: uv add --dev pylint")
        return False


def create_diagram_index() -> None:
    """Create an index page for all generated diagrams."""
    print("\n=== Creating Diagram Index ===")

    index_file = DOCS_IMAGES / "index.md"

    diagrams = sorted(DOCS_IMAGES.glob("*.svg"))

    # Don't create index if no diagrams were generated
    if not diagrams:
        print("⚠ No diagrams generated, skipping index creation")
        # Remove index if it exists to avoid referencing missing files
        if index_file.exists():
            index_file.unlink()
        return

    content = """# Architecture Diagrams

This directory contains auto-generated architecture diagrams for Oscura.

"""

    # Only add module dependencies section if the file exists
    if (DOCS_IMAGES / "module-dependencies.svg").exists():
        content += """## Module Dependencies

![Module Dependencies](module-dependencies.svg)

Shows the dependency relationships between Oscura's Python packages.

"""

    # Only add top-level architecture if files exist
    packages_exists = (DOCS_IMAGES / "packages_oscura.svg").exists()
    classes_exists = (DOCS_IMAGES / "classes_oscura.svg").exists()

    if packages_exists or classes_exists:
        content += """## Top-Level Architecture

"""
        if packages_exists:
            content += """![Package Architecture](packages_oscura.svg)

High-level view of Oscura's package structure.

"""
        if classes_exists:
            content += """![Class Architecture](classes_oscura.svg)

Overview of major classes and their relationships.

"""

    # Add links to package-specific diagrams
    package_diagrams = [
        d for d in diagrams if d.name.startswith("classes_") and d.name != "classes_oscura.svg"
    ]

    if package_diagrams:
        content += """## Package-Specific Diagrams

"""
        for diagram in package_diagrams:
            package_name = diagram.stem.replace("classes_", "")
            content += f"### {package_name.title()} Package\n\n"
            content += f"![{package_name} Classes]({diagram.name})\n\n"

    content += """
---

*Diagrams auto-generated by `scripts/generate_diagrams.py`*
"""

    index_file.write_text(content)
    print(f"✓ Diagram index: {index_file}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("Oscura Architecture Diagram Generator")
    print("=" * 50)

    ensure_output_dir()

    results = []

    # Generate all diagram types
    results.append(("Module Dependencies", generate_module_dependencies()))
    results.append(("Top-Level Architecture", generate_top_level_architecture()))
    results.append(("Package Diagrams", generate_package_diagrams()))

    # Create index
    create_diagram_index()

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    all_success = all(success for _, success in results)
    if all_success:
        print("\n✓ All diagrams generated successfully!")
        print(f"  Output: {DOCS_IMAGES}")
        return 0
    else:
        print("\n⚠ Some diagrams failed to generate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
