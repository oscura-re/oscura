#!/usr/bin/env python3
"""Automated refactoring script for demonstration files."""

import re
from pathlib import Path

FILES_TO_FIX = [
    "demonstrations/03_protocol_decoding/i2s.py",
    "demonstrations/03_protocol_decoding/jtag.py",
    "demonstrations/03_protocol_decoding/manchester.py",
    "demonstrations/03_protocol_decoding/onewire.py",
    "demonstrations/03_protocol_decoding/protocol_comprehensive.py",
    "demonstrations/03_protocol_decoding/swd.py",
    "demonstrations/03_protocol_decoding/udp_analysis.py",
    "demonstrations/03_protocol_decoding/usb.py",
    "demonstrations/04_advanced_analysis/jitter_bathtub.py",
    "demonstrations/04_advanced_analysis/jitter_ddj_dcd.py",
    "demonstrations/04_advanced_analysis/power_dcdc.py",
    "demonstrations/04_advanced_analysis/power_ripple.py",
    "demonstrations/04_advanced_analysis/signal_integrity_sparams.py",
    "demonstrations/04_advanced_analysis/signal_integrity_tdr.py",
    "demonstrations/04_advanced_analysis/signal_integrity_timing.py",
    "demonstrations/05_domain_specific/automotive_comprehensive.py",
    "demonstrations/05_domain_specific/automotive_flexray.py",
    "demonstrations/05_domain_specific/automotive_lin.py",
    "demonstrations/05_domain_specific/emc_comprehensive.py",
    "demonstrations/05_domain_specific/timing_ieee181.py",
    "demonstrations/06_reverse_engineering/crc_reverse.py",
    "demonstrations/06_reverse_engineering/inference_active_learning.py",
    "demonstrations/06_reverse_engineering/inference_bayesian.py",
    "demonstrations/06_reverse_engineering/inference_dsl.py",
    "demonstrations/06_reverse_engineering/re_comprehensive.py",
    "demonstrations/06_reverse_engineering/state_machine_learning.py",
    "demonstrations/06_reverse_engineering/wireshark_dissector.py",
    "demonstrations/11_integration/04_configuration_files.py",
    "demonstrations/16_complete_workflows/automotive_workflow.py",
    "demonstrations/16_complete_workflows/network_workflow.py",
    "demonstrations/16_complete_workflows/unknown_signal_workflow.py",
]


def extract_class_attributes(content: str) -> tuple[str, str, str]:
    """Extract name, description, category from class attributes."""
    name_match = re.search(r'^\s+name = "([^"]+)"', content, re.MULTILINE)
    desc_match = re.search(r'^\s+description = "([^"]+)"', content, re.MULTILINE)
    cat_match = re.search(r'^\s+category = "([^"]+)"', content, re.MULTILINE)

    name = name_match.group(1) if name_match else "unknown"
    description = desc_match.group(1) if desc_match else "Unknown demonstration"
    category = cat_match.group(1) if cat_match else None

    return name, description, category


def refactor_file(filepath: Path) -> bool:
    """Refactor a single demonstration file."""
    print(f"Processing {filepath}...")

    try:
        content = filepath.read_text()
        original = content

        # Extract class attributes
        name, description, category = extract_class_attributes(content)

        # 1. Replace class attributes with __init__ parameters
        # Find the class definition
        class_match = re.search(
            r'(class \w+\(BaseDemo\):.*?""".*?""")\s+'
            r'(name = ".*?"\s+description = ".*?"\s+(?:category = ".*?"\s+)?)'
            r"(def __init__)",
            content,
            re.DOTALL,
        )

        if class_match:
            # Remove class attributes
            content = content.replace(class_match.group(2), "\n    ")

            # Update __init__ to include attributes in super().__init__()
            init_pattern = (
                r'(def __init__\(self(?:, \*\*kwargs)?\):.*?""".*?""".*?super\(\).__init__\()'
            )

            def replace_init(match):
                init_str = match.group(1)
                # Check if super().__init__() already has arguments
                if re.search(r"super\(\).__init__\([^)]+\)", init_str):
                    # Already has args, skip
                    return match.group(0)
                else:
                    # Add arguments
                    args = [
                        f'name="{name}"',
                        f'description="{description}"',
                    ]
                    if category:
                        args.append(f'category="{category}"')
                    args.append("**kwargs")
                    args_str = ",\n            ".join(args)
                    return init_str + f"\n            {args_str},\n        "

            content = re.sub(init_pattern, replace_init, content, flags=re.DOTALL)

        # 2. Rename generate_data -> generate_test_data and add return dict
        content = re.sub(
            r"def generate_data\(self\) -> None:", "def generate_test_data(self) -> dict:", content
        )

        # Add return statement before run_demonstration if missing
        if "return {" not in content and "def run_demonstration" in content:
            # Find where to add return (before run_demonstration)
            content = re.sub(r"(\n\s+)(def run_demonstration)", r"\1return {}\n\n\1\2", content)

        # 3. Rename run_analysis -> run_demonstration and add params
        content = re.sub(
            r"def run_analysis\(self\) -> None:",
            "def run_demonstration(self, data: dict) -> dict:",
            content,
        )

        # Add return self.results before validate if missing
        if "return self.results" not in content and "def validate(" in content:
            content = re.sub(r"(\n\s+)(def validate\()", r"\1return self.results\n\n\1\2", content)

        # 4. Rename validate_results -> validate
        content = re.sub(
            r"def validate_results\(self, suite: ValidationSuite\) -> None:",
            "def validate(self, results: dict) -> bool:",
            content,
        )

        # 5. Update validate to create ValidationSuite and return bool
        # Find validate method and add suite creation
        validate_pattern = r'(def validate\(self, results: dict\) -> bool:.*?""".*?""")'

        def add_suite(match):
            method = match.group(1)
            if "ValidationSuite()" not in method:
                return method + "\n        suite = ValidationSuite()\n"
            return method

        content = re.sub(validate_pattern, add_suite, content, flags=re.DOTALL)

        # Add suite.report() and return before end of validate
        # This is tricky, so we'll do it manually per file if needed

        # Save if changed
        if content != original:
            filepath.write_text(content)
            print(f"  ✓ Refactored {filepath}")
            return True
        else:
            print(f"  - No changes needed for {filepath}")
            return False

    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False


def main():
    """Main refactoring function."""
    repo_root = Path("/home/lair-click-bats/development/oscura")

    fixed_count = 0
    for file_path in FILES_TO_FIX:
        full_path = repo_root / file_path
        if full_path.exists():
            if refactor_file(full_path):
                fixed_count += 1
        else:
            print(f"  ✗ File not found: {full_path}")

    print(f"\nSummary: {fixed_count} files refactored")


if __name__ == "__main__":
    main()
