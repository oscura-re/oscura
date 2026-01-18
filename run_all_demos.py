#!/usr/bin/env python3
"""Run all demos and track their status."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

# Demo configuration: (directory, main_script)
DEMOS = [
    ("01_waveform_analysis", "all_output_formats.py"),
    ("02_file_format_io", "vcd_loader_demo.py"),
    ("03_custom_daq", "simple_loader.py"),
    ("04_serial_protocols", "i2s_demo.py"),
    ("04_serial_protocols", "jtag_demo.py"),
    ("04_serial_protocols", "manchester_demo.py"),
    ("04_serial_protocols", "onewire_demo.py"),
    ("04_serial_protocols", "swd_demo.py"),
    ("04_serial_protocols", "usb_demo.py"),
    ("05_protocol_decoding", "comprehensive_protocol_demo.py"),
    ("06_udp_packet_analysis", "comprehensive_udp_analysis.py"),
    ("07_protocol_inference", "crc_reverse_demo.py"),
    ("07_protocol_inference", "state_machine_learning.py"),
    ("07_protocol_inference", "wireshark_dissector_demo.py"),
    ("08_automotive_protocols", "flexray_demo.py"),
    ("08_automotive_protocols", "lin_demo.py"),
    ("09_automotive", "comprehensive_automotive_demo.py"),
    ("10_timing_measurements", "ieee_181_pulse_demo.py"),
    ("11_mixed_signal", "comprehensive_mixed_signal_demo.py"),
    ("12_spectral_compliance", "comprehensive_spectral_demo.py"),
    ("13_jitter_analysis", "bathtub_curve_demo.py"),
    ("13_jitter_analysis", "ddj_dcd_demo.py"),
    ("14_power_analysis", "dcdc_efficiency_demo.py"),
    ("14_power_analysis", "ripple_analysis_demo.py"),
    ("15_signal_integrity", "setup_hold_timing_demo.py"),
    ("15_signal_integrity", "sparams_demo.py"),
    ("15_signal_integrity", "tdr_impedance_demo.py"),
    ("16_emc_compliance", "comprehensive_emc_demo.py"),
    ("17_signal_reverse_engineering", "comprehensive_re.py"),
    ("18_advanced_inference", "active_learning_demo.py"),
    ("18_advanced_inference", "bayesian_inference_demo.py"),
    ("18_advanced_inference", "protocol_dsl_demo.py"),
    ("19_complete_workflows", "automotive_full_workflow.py"),
    ("19_complete_workflows", "network_analysis_workflow.py"),
    ("19_complete_workflows", "unknown_signal_workflow.py"),
]


def run_demo(demo_dir: str, script: str) -> dict[str, Any]:
    """Run a single demo and return its status.

    Args:
        demo_dir: Demo directory name
        script: Script name to run

    Returns:
        Dictionary with status information
    """
    demo_path = Path(__file__).parent / "demos" / demo_dir
    script_path = demo_path / script

    if not script_path.exists():
        return {
            "status": "missing",
            "error": f"Script not found: {script_path}",
        }

    print(f"\n{'=' * 80}")
    print(f"Running: {demo_dir}/{script}")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script_path)],
            cwd=str(demo_path),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per demo
            check=False,
        )

        success = result.returncode == 0

        return {
            "status": "success" if success else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "Demo exceeded 5 minute timeout",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def main() -> int:
    """Run all demos and report results."""
    results = {}

    print("=" * 80)
    print("RUNNING ALL OSCURA DEMOS")
    print("=" * 80)
    print(f"Total demos to run: {len(DEMOS)}")

    for i, (demo_dir, script) in enumerate(DEMOS, 1):
        demo_key = f"{demo_dir}/{script}"
        print(f"\n[{i}/{len(DEMOS)}] {demo_key}")

        result = run_demo(demo_dir, script)
        results[demo_key] = result

        # Print immediate status
        status = result["status"]
        if status == "success":
            print("✓ SUCCESS")
        elif status == "failed":
            print(f"✗ FAILED (exit code: {result.get('returncode', '?')})")
            if result.get("stderr"):
                print(f"  Error: {result['stderr'][:200]}")
        elif status == "timeout":
            print("⏱ TIMEOUT")
        elif status == "missing":
            print(f"⚠ MISSING: {result.get('error', '')}")
        else:
            print(f"⚠ ERROR: {result.get('error', '')}")

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    timeout_count = sum(1 for r in results.values() if r["status"] == "timeout")
    missing_count = sum(1 for r in results.values() if r["status"] == "missing")
    error_count = sum(1 for r in results.values() if r["status"] == "error")

    print(f"\nTotal demos: {len(DEMOS)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⏱ Timeout: {timeout_count}")
    print(f"  ⚠ Missing: {missing_count}")
    print(f"  ⚠ Error: {error_count}")

    if failed_count > 0:
        print("\nFailed demos:")
        for demo_key, result in results.items():
            if result["status"] == "failed":
                print(f"  - {demo_key}")
                if result.get("stderr"):
                    # Show first line of error
                    first_error = result["stderr"].split("\n")[0]
                    print(f"    {first_error[:100]}")

    if timeout_count > 0:
        print("\nTimeout demos:")
        for demo_key, result in results.items():
            if result["status"] == "timeout":
                print(f"  - {demo_key}")

    if missing_count > 0:
        print("\nMissing demos:")
        for demo_key, result in results.items():
            if result["status"] == "missing":
                print(f"  - {demo_key}: {result.get('error', '')}")

    # Write detailed results to file
    report_path = Path(__file__).parent / "DEMO_RUN_RESULTS.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED DEMO RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for demo_key, result in results.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Demo: {demo_key}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"{'=' * 80}\n")

            if result["status"] == "success":
                f.write("✓ Demo ran successfully\n")
                if result.get("stdout"):
                    f.write("\nOutput:\n")
                    f.write(result["stdout"])
            elif result["status"] == "failed":
                f.write(f"✗ Demo failed with exit code: {result.get('returncode', '?')}\n")
                if result.get("stderr"):
                    f.write("\nError output:\n")
                    f.write(result["stderr"])
                if result.get("stdout"):
                    f.write("\nStandard output:\n")
                    f.write(result["stdout"])
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")

    print(f"\nDetailed results written to: {report_path}")

    # Return non-zero if any demos failed
    return 1 if (failed_count + timeout_count + error_count) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
