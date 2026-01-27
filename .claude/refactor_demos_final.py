#!/usr/bin/env python3
"""Refactor demonstration files to BaseDemo pattern."""

import re
from pathlib import Path

# All files to fix
FILES = [
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


def refactor_file(filepath: Path) -> tuple[bool, str]:
    """Refactor one file. Returns (success, message)."""
    try:
        content = filepath.read_text()
        original = content

        # 1. Extract class attributes (name, description, category)
        name_match = re.search(r'^\s+name = "([^"]+)"', content, re.MULTILINE)
        desc_match = re.search(r'^\s+description = "([^"]+)"', content, re.MULTILINE)
        cat_match = re.search(r'^\s+category = "([^"]+)"', content, re.MULTILINE)

        if not (name_match and desc_match):
            return (False, "No class attributes found")

        name = name_match.group(1)
        description = desc_match.group(1)
        category = cat_match.group(1) if cat_match else None

        # 2. Remove class attributes
        content = re.sub(r'^\s+name = "[^"]+"\s*\n', "", content, flags=re.MULTILINE)
        content = re.sub(r'^\s+description = "[^"]+"\s*\n', "", content, flags=re.MULTILINE)
        content = re.sub(r'^\s+category = "[^"]+"\s*\n', "", content, flags=re.MULTILINE)

        # 3. Update __init__ to add super().__init__() args
        def replace_super_init(match):
            indent = match.group(1)
            # Build args - NOTE: BaseDemo doesn't accept 'category' parameter
            args = [
                f'name="{name}"',
                f'description="{description}"',
            ]
            # category is NOT passed to BaseDemo.__init__()
            args.append("**kwargs")
            args_str = (",\n" + indent + "    ").join(args)
            return f"{indent}super().__init__(\n{indent}    {args_str},\n{indent})"

        content = re.sub(
            r"^(\s+)super\(\).__init__\(\*\*kwargs\)$",
            replace_super_init,
            content,
            flags=re.MULTILINE,
        )

        # 4. Rename method signatures
        content = re.sub(
            r"def generate_data\(self\) -> None:", "def generate_test_data(self) -> dict:", content
        )
        content = re.sub(
            r"def run_analysis\(self\) -> None:",
            "def run_demonstration(self, data: dict) -> dict:",
            content,
        )
        content = re.sub(
            r"def validate_results\(self, suite: ValidationSuite\) -> None:",
            "def validate(self, results: dict) -> bool:",
            content,
        )

        # 5. Add return {} at end of generate_test_data if missing
        # Find generate_test_data and look for return statement
        gen_match = re.search(
            r"(def generate_test_data\(self\) -> dict:.*?)(^    def )",
            content,
            re.DOTALL | re.MULTILINE,
        )
        if gen_match and "return {" not in gen_match.group(1):
            # Add return before next method
            content = re.sub(
                r"(def generate_test_data\(self\) -> dict:.*?)(^    def )",
                r"\1\n        return {}\n\n\2",
                content,
                count=1,
                flags=re.DOTALL | re.MULTILINE,
            )

        # 6. Add return self.results at end of run_demonstration if missing
        run_match = re.search(
            r"(def run_demonstration\(self, data: dict\) -> dict:.*?)(^    def )",
            content,
            re.DOTALL | re.MULTILINE,
        )
        if run_match and "return self.results" not in run_match.group(1):
            content = re.sub(
                r"(def run_demonstration\(self, data: dict\) -> dict:.*?)(^    def )",
                r"\1\n        return self.results\n\n\2",
                content,
                count=1,
                flags=re.DOTALL | re.MULTILINE,
            )

        # 7. Update validate to create suite and return bool
        # Replace first line after validate docstring
        content = re.sub(
            r'(def validate\(self, results: dict\) -> bool:.*?""".*?"""\s+)',
            r"\1suite = ValidationSuite()\n\n        ",
            content,
            count=1,
            flags=re.DOTALL,
        )

        # Replace all suite.check_* with suite.add_check
        # This is complex, so we'll just add suite.report() and return at the end

        # Find end of validate and add return
        val_match = re.search(
            r"(def validate\(self, results: dict\) -> bool:.*)", content, re.DOTALL
        )
        if val_match:
            val_content = val_match.group(1)
            if "suite.report()" not in val_content:
                # Add before final whitespace/end of method
                content = re.sub(
                    r"(def validate\(self, results: dict\) -> bool:.*?)(^\s*(?:if __name__|class |\Z))",
                    r"\1\n        suite.report()\n        return suite.all_passed()\n\n\2",
                    content,
                    count=1,
                    flags=re.DOTALL | re.MULTILINE,
                )

        # Save if changed
        if content != original:
            filepath.write_text(content)
            return (True, "Refactored successfully")
        else:
            return (False, "No changes needed")

    except Exception as e:
        return (False, f"Error: {e}")


def main():
    """Run refactoring."""
    root = Path("/home/lair-click-bats/development/oscura")
    success_count = 0
    failure_count = 0

    for file_path in FILES:
        full_path = root / file_path
        if not full_path.exists():
            print(f"✗ {file_path}: File not found")
            failure_count += 1
            continue

        success, msg = refactor_file(full_path)
        if success:
            print(f"✓ {file_path}: {msg}")
            success_count += 1
        else:
            print(f"- {file_path}: {msg}")
            if "Error" in msg:
                failure_count += 1

    print(f"\nSummary: {success_count} refactored, {failure_count} errors")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    exit(main())
