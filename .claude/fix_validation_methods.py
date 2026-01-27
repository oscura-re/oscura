#!/usr/bin/env python3
"""Fix validation methods to use suite.add_check() instead of suite.check_*()."""

import re
from pathlib import Path

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
    "demonstrations/16_complete_workflows/automotive_workflow.py",
    "demonstrations/16_complete_workflows/network_workflow.py",
    "demonstrations/16_complete_workflows/unknown_signal_workflow.py",
]


def fix_validation(filepath: Path) -> tuple[bool, str]:
    """Fix validation method. Returns (success, message)."""
    try:
        content = filepath.read_text()
        original = content

        # Replace all suite.check_* calls with simple suite.add_check(name, True/False)
        # This is a simplified conversion - proper logic would need parsing

        # Pattern: suite.check_*("name", value, threshold, ...) -> suite.add_check("name", condition)

        # For check_greater: suite.check_greater("name", val, min_val, ...) -> suite.add_check("name", val > min_val)
        content = re.sub(
            r'suite\.check_greater\(\s*"([^"]+)",\s*([^,]+),\s*([^,\)]+)(?:,\s*category="[^"]+")?\s*\)',
            r'suite.add_check("\1", \2 > \3)',
            content,
        )

        # For check_true: suite.check_true("name", condition, ...) -> suite.add_check("name", condition)
        content = re.sub(
            r'suite\.check_true\(\s*"([^"]+)",\s*([^,\)]+)(?:,\s*category="[^"]+")?\s*\)',
            r'suite.add_check("\1", \2)',
            content,
        )

        # For check_less: suite.check_less("name", val, max_val, ...) -> suite.add_check("name", val < max_val)
        content = re.sub(
            r'suite\.check_less\(\s*"([^"]+)",\s*([^,]+),\s*([^,\)]+)(?:,\s*category="[^"]+")?\s*\)',
            r'suite.add_check("\1", \2 < \3)',
            content,
        )

        # For check_equal: suite.check_equal("name", val, expected, ...) -> suite.add_check("name", val == expected)
        content = re.sub(
            r'suite\.check_equal\(\s*"([^"]+)",\s*([^,]+),\s*([^,\)]+)(?:,\s*category="[^"]+")?\s*\)',
            r'suite.add_check("\1", \2 == \3)',
            content,
        )

        # For check_in_range: suite.check_in_range("name", val, min, max, ...) -> suite.add_check("name", min <= val <= max)
        content = re.sub(
            r'suite\.check_in_range\(\s*"([^"]+)",\s*([^,]+),\s*([^,]+),\s*([^,\)]+)(?:,\s*category="[^"]+")?\s*\)',
            r'suite.add_check("\1", \3 <= \2 <= \4)',
            content,
        )

        if content != original:
            filepath.write_text(content)
            return (True, "Fixed validation methods")
        else:
            return (False, "No changes needed")

    except Exception as e:
        return (False, f"Error: {e}")


def main():
    """Run fixing."""
    root = Path("/home/lair-click-bats/development/oscura")
    success_count = 0

    for file_path in FILES:
        full_path = root / file_path
        if not full_path.exists():
            print(f"✗ {file_path}: File not found")
            continue

        success, msg = fix_validation(full_path)
        if success:
            print(f"✓ {file_path}: {msg}")
            success_count += 1
        else:
            print(f"- {file_path}: {msg}")

    print(f"\nSummary: {success_count} files fixed")
    return 0


if __name__ == "__main__":
    exit(main())
