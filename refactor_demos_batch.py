#!/usr/bin/env python3
"""Batch refactor demonstration files to match BaseDemo requirements.

This script automates the refactoring of 36 demo files to use:
- generate_test_data() -> dict
- run_demonstration(self, data: dict) -> dict
- validate(self, results: dict) -> bool
- super().__init__(name=..., description=..., capabilities=[...])
"""

import re
import sys
from pathlib import Path


def refactor_demo_file(file_path: Path) -> tuple[bool, str]:
    """Refactor a single demo file.

    Args:
        file_path: Path to demo file

    Returns:
        Tuple of (success, message)
    """
    try:
        content = file_path.read_text()
        original_content = content

        # Pattern 1: Move class attributes to __init__
        # Find class attributes (name, description, category)
        class_match = re.search(
            r'class\s+(\w+)\(BaseDemo\):[^\n]*\n(?:[ \t]*"""[^"]*"""[^\n]*\n)?'
            r"(?:[ \t]*\n)*"
            r'(?:[ \t]*name\s*=\s*["\']([^"\']+)["\']\s*\n)?'
            r'(?:[ \t]*description\s*=\s*["\']([^"\']+)["\']\s*\n)?'
            r'(?:[ \t]*category\s*=\s*["\']([^"\']+)["\']\s*\n)?',
            content,
            re.MULTILINE,
        )

        if class_match:
            class_name = class_match.group(1)
            name_val = class_match.group(2)
            desc_val = class_match.group(3)
            category_val = class_match.group(4)

            # Find __init__ method
            init_match = re.search(
                r"([ \t]*)def __init__\(self[^)]*\):\s*\n"
                r'(?:[ \t]*"""[^"]*"""\s*\n)?'
                r"([ \t]*)super\(\).__init__\(([^)]*)\)",
                content,
            )

            if init_match and name_val:
                indent = init_match.group(1)
                super_indent = init_match.group(2)
                existing_args = init_match.group(3)

                # Build new super().__init__() call
                new_args = []
                if name_val:
                    snake_name = name_val.lower().replace(" ", "_").replace("-", "_")
                    new_args.append(f'name="{snake_name}"')
                if desc_val:
                    new_args.append(f'description="{desc_val}"')

                # Add capabilities based on file location
                capabilities = []
                if "loader" in file_path.name or "loading" in str(file_path):
                    capabilities.append("oscura.loaders")
                if "protocol" in str(file_path):
                    capabilities.append("oscura.protocols")
                if "spectral" in file_path.name:
                    capabilities.append("oscura.spectral_analysis")
                if "jitter" in file_path.name or "eye" in file_path.name:
                    capabilities.append("oscura.jitter_analysis")
                if "power" in file_path.name:
                    capabilities.append("oscura.power_analysis")
                if "automotive" in str(file_path):
                    capabilities.append("oscura.automotive")
                if "inference" in file_path.name or "reverse" in str(file_path):
                    capabilities.append("oscura.inference")

                if capabilities:
                    cap_str = ", ".join([f'"{c}"' for c in capabilities])
                    new_args.append(f"capabilities=[{cap_str}]")

                if existing_args.strip() and "**kwargs" in existing_args:
                    new_args.append("**kwargs")

                new_super_call = f"{super_indent}super().__init__(\n"
                for arg in new_args[:-1]:
                    new_super_call += f"{super_indent}    {arg},\n"
                if new_args:
                    new_super_call += f"{super_indent}    {new_args[-1]},\n"
                new_super_call += f"{super_indent})"

                # Replace class attributes and __init__
                class_section = class_match.group(0)
                new_class_section = re.sub(
                    r'[ \t]*name\s*=\s*["\'][^"\']+["\']\s*\n'
                    r'(?:[ \t]*description\s*=\s*["\'][^"\']+["\']\s*\n)?'
                    r'(?:[ \t]*category\s*=\s*["\'][^"\']+["\']\s*\n)?',
                    "",
                    class_section,
                )
                content = content.replace(class_section, new_class_section)

                # Replace super().__init__()
                old_super = init_match.group(0)
                new_init = old_super.split("super().__init__(")[0] + new_super_call
                content = content.replace(old_super, new_init)

        # Pattern 2: Rename generate_data to generate_test_data
        content = re.sub(
            r"def generate_data\(self\)\s*->\s*None:",
            "def generate_test_data(self) -> dict:",
            content,
        )

        # Add return statement at end of generate_test_data if missing
        # This is complex, so we'll do it manually per file if needed

        # Pattern 3: Rename run_analysis to run_demonstration
        content = re.sub(
            r"def run_analysis\(self\)\s*->\s*None:",
            "def run_demonstration(self, data: dict) -> dict:",
            content,
        )

        # Pattern 4: Rename validate_results to validate
        content = re.sub(
            r"def validate_results\(self,\s*suite:\s*ValidationSuite\)\s*->\s*None:",
            "def validate(self, results: dict) -> bool:",
            content,
        )

        # Only write if content changed
        if content != original_content:
            file_path.write_text(content)
            return True, "Successfully refactored"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main entry point."""
    files_to_fix = [
        "demonstrations/01_data_loading/custom_loader_simple.py",
        "demonstrations/01_data_loading/vcd_advanced_demo.py",
        "demonstrations/02_basic_analysis/mixed_signal.py",
        "demonstrations/02_basic_analysis/spectral_compliance.py",
        "demonstrations/02_basic_analysis/waveform_comprehensive.py",
    ]

    project_root = Path(__file__).parent

    success_count = 0
    failed_count = 0

    for file_rel in files_to_fix:
        file_path = project_root / file_rel
        if not file_path.exists():
            print(f"❌ {file_rel}: File not found")
            failed_count += 1
            continue

        success, message = refactor_demo_file(file_path)
        if success:
            print(f"✓ {file_rel}: {message}")
            success_count += 1
        else:
            print(f"❌ {file_rel}: {message}")
            failed_count += 1

    print(f"\nSummary: {success_count} succeeded, {failed_count} failed")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
