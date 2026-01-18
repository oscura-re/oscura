#!/usr/bin/env python3
"""Validate all Oscura demos - comprehensive test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Comprehensive demo list with expected outputs
DEMOS = {
    "01_waveform_analysis": {
        "script": "all_output_formats.py",
        "timeout": 300,
        "expected_outputs": ["plots/", "exports/", "reports/", "analysis/"],
        "output_dir": "wfm_outputs_complete",
    },
    "02_file_format_io": {
        "script": "vcd_loader_demo.py",
        "timeout": 60,
        "expected_outputs": ["demo_data/demo_signals.vcd"],
    },
    "03_custom_daq": {
        "script": "simple_loader.py",
        "timeout": 60,
    },
    "04_serial_protocols": {
        "scripts": [
            "i2s_demo.py",
            "jtag_demo.py",
            "manchester_demo.py",
            "onewire_demo.py",
            "swd_demo.py",
            "usb_demo.py",
        ],
        "timeout": 60,
    },
    "05_protocol_decoding": {
        "script": "comprehensive_protocol_demo.py",
        "timeout": 120,
    },
    "06_udp_packet_analysis": {
        "script": "comprehensive_udp_analysis.py",
        "timeout": 120,
    },
    "07_protocol_inference": {
        "scripts": [
            "crc_reverse_demo.py",
            "state_machine_learning.py",
            "wireshark_dissector_demo.py",
        ],
        "timeout": 120,
    },
    "08_automotive_protocols": {
        "scripts": ["flexray_demo.py", "lin_demo.py"],
        "timeout": 120,
    },
    "09_automotive": {
        "script": "comprehensive_automotive_demo.py",
        "timeout": 120,
    },
    "10_timing_measurements": {
        "script": "ieee_181_pulse_demo.py",
        "timeout": 60,
    },
    "11_mixed_signal": {
        "script": "comprehensive_mixed_signal_demo.py",
        "timeout": 120,
    },
    "12_spectral_compliance": {
        "script": "comprehensive_spectral_demo.py",
        "timeout": 120,
    },
    "13_jitter_analysis": {
        "scripts": ["bathtub_curve_demo.py", "ddj_dcd_demo.py"],
        "timeout": 120,
    },
    "14_power_analysis": {
        "scripts": ["dcdc_efficiency_demo.py", "ripple_analysis_demo.py"],
        "timeout": 120,
    },
    "15_signal_integrity": {
        "scripts": [
            "setup_hold_timing_demo.py",
            "sparams_demo.py",
            "tdr_impedance_demo.py",
        ],
        "timeout": 120,
    },
    "16_emc_compliance": {
        "script": "comprehensive_emc_demo.py",
        "timeout": 120,
    },
    "17_signal_reverse_engineering": {
        "script": "comprehensive_re.py",
        "timeout": 120,
    },
    "18_advanced_inference": {
        "scripts": [
            "active_learning_demo.py",
            "bayesian_inference_demo.py",
            "protocol_dsl_demo.py",
        ],
        "timeout": 120,
    },
    "19_complete_workflows": {
        "scripts": [
            "automotive_full_workflow.py",
            "network_analysis_workflow.py",
            "unknown_signal_workflow.py",
        ],
        "timeout": 180,
    },
}


def run_demo(demo_dir: str, script: str, timeout: int = 120) -> dict[str, Any]:
    """Run a single demo script."""
    demo_path = Path(__file__).parent / "demos" / demo_dir
    script_path = demo_path / script

    if not script_path.exists():
        return {"status": "missing", "error": f"Script not found: {script}"}

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script_path)],
            cwd=str(demo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        # Check for validation markers
        validation_passed = "Demo validation passed!" in result.stdout
        validation_failed = "Demo validation FAILED" in result.stdout

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "validation_passed": validation_passed,
            "validation_failed": validation_failed,
            "stdout_lines": len(result.stdout.split("\n")),
            "stderr_preview": result.stderr[:500] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout": timeout}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_outputs(demo_dir: str, expected: list[str]) -> dict[str, bool]:
    """Check if expected output files/directories exist."""
    demo_path = Path(__file__).parent / "demos" / demo_dir
    results = {}

    for expected_path in expected:
        full_path = demo_path / expected_path
        results[expected_path] = full_path.exists()

    return results


def main() -> int:
    """Run and validate all demos."""
    print("=" * 80)
    print("OSCURA COMPREHENSIVE DEMO VALIDATION")
    print("=" * 80)

    all_results = {}
    total_demos = 0
    success_count = 0
    failed_count = 0
    timeout_count = 0
    missing_count = 0

    for demo_dir, config in DEMOS.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {demo_dir}")
        print("=" * 80)

        # Handle both single script and multiple scripts
        scripts = config.get("scripts", [config.get("script")])
        timeout = config.get("timeout", 120)

        demo_results = []

        for script in scripts:
            if script:
                total_demos += 1
                print(f"  Running: {script}...", end=" ", flush=True)

                result = run_demo(demo_dir, script, timeout)
                demo_results.append({"script": script, **result})

                if result["status"] == "success":
                    print("✓ SUCCESS")
                    success_count += 1
                elif result["status"] == "failed":
                    print(f"✗ FAILED (exit code: {result.get('returncode')})")
                    failed_count += 1
                elif result["status"] == "timeout":
                    print("⏱ TIMEOUT")
                    timeout_count += 1
                elif result["status"] == "missing":
                    print("⚠ MISSING")
                    missing_count += 1

        # Check expected outputs if defined
        if "expected_outputs" in config:
            output_dir = config.get("output_dir", "")
            check_dir = output_dir if output_dir else demo_dir
            outputs = check_outputs(check_dir, config["expected_outputs"])
            demo_results.append({"outputs": outputs})

            missing_outputs = [k for k, v in outputs.items() if not v]
            if missing_outputs:
                print(f"  ⚠ Missing outputs: {', '.join(missing_outputs)}")
            else:
                print("  ✓ All expected outputs present")

        all_results[demo_dir] = demo_results

    # Generate summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal demo scripts: {total_demos}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⏱ Timeout: {timeout_count}")
    print(f"  ⚠ Missing: {missing_count}")

    success_rate = (success_count / total_demos * 100) if total_demos > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")

    # Save detailed results
    results_file = Path(__file__).parent / "DEMO_VALIDATION_RESULTS.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": total_demos,
                    "success": success_count,
                    "failed": failed_count,
                    "timeout": timeout_count,
                    "missing": missing_count,
                    "success_rate": success_rate,
                },
                "details": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {results_file}")

    return 0 if (failed_count + timeout_count) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
