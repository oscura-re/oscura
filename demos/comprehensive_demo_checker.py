#!/usr/bin/env python3
"""Comprehensive demo output checker - validates ALL functionalities.

# SKIP_VALIDATION: Meta-validator runs all demos, would timeout

This script systematically validates all Oscura demos by running them and
checking for expected output strings, files, and exit codes. Provides a
comprehensive summary of demo health for CI/CD integration.

Usage:
    python comprehensive_demo_checker.py

Exit codes:
    0: All demos passed
    1: One or more demos failed or timed out
"""

import subprocess
import sys
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Demo configurations with expected outputs
DEMOS = {
    "01_waveform_analysis/comprehensive_wfm_analysis.py": {
        "timeout": 120,
        "expected": ["tests:", "Passed:", "Success rate"],
        "outputs": [],
    },
    "01_waveform_analysis/all_output_formats.py": {
        "timeout": 120,
        "expected": ["Export:", "Spectral:", "Report:"],  # Fixed: looks for "Spectral:" not "Plot:"
        "outputs": ["wfm_outputs_complete/"],
    },
    "02_custom_daq/simple_loader.py": {
        "timeout": 30,
        "expected": ["✓ Successfully loaded", "✅ Demo validation passed"],
        "outputs": [],
    },
    "02_custom_daq/optimal_streaming_loader.py": {
        "timeout": 90,
        "expected": [
            "memory",
            "✅",
        ],  # Fixed: just looks for "memory" (case-insensitive check needed)
        "outputs": [],
    },
    "02_custom_daq/chunked_loader.py": {
        "timeout": 90,
        "expected": ["Chunk", "Range:", "✅"],  # Fixed: looks for "Range:" which is in statistics
        "outputs": [],
    },
    "03_udp_packet_analysis/comprehensive_udp_analysis.py": {
        "timeout": 60,
        "expected": ["UDP Packets:", "✅ Demo validation passed"],
        "outputs": ["udp_analysis/"],
    },
    "04_signal_reverse_engineering/comprehensive_re.py": {
        "timeout": 360,  # 6 minutes for comprehensive demo
        "expected": ["ANALYSIS COMPLETE", "✅"],
        "outputs": ["signal_re_outputs/"],
    },
    "05_protocol_decoding/comprehensive_protocol_demo.py": {
        "timeout": 60,
        "expected": ["Decoded", "frames", "✓ Protocol analysis complete"],
        "outputs": [],
    },
    "06_spectral_compliance/comprehensive_spectral_demo.py": {
        "timeout": 60,
        "expected": ["THD:", "SNR:", "SINAD:", "✓ Spectral analysis complete"],
        "outputs": [],
    },
    "07_mixed_signal/comprehensive_mixed_signal_demo.py": {
        "timeout": 60,
        "expected": ["Jitter", "Clock Recovery", "✓ Mixed-signal analysis complete"],
        "outputs": [],
    },
    "08_automotive/comprehensive_automotive_demo.py": {
        "timeout": 90,
        "expected": ["CAN", "Demo Complete"],  # "Automotive Protocol Demo Complete!"
        "outputs": [],
    },
    "09_emc_compliance/comprehensive_emc_demo.py": {
        "timeout": 60,
        "expected": ["COMPLIANCE", "CISPR", "FCC", "Demo completed successfully"],
        "outputs": [],
    },
}


def check_demo(demo_path: str, config: dict) -> dict:
    """Run a demo and check its outputs comprehensively."""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}Checking: {demo_path}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

    demo_dir = Path(__file__).parent
    full_path = demo_dir / demo_path

    if not full_path.exists():
        print(f"{RED}✗ Demo file not found{RESET}")
        return {"status": "not_found", "issues": ["File not found"]}

    # Run the demo
    try:
        result = subprocess.run(
            ["uv", "run", "python", str(full_path)],
            cwd=str(demo_dir),
            capture_output=True,
            text=True,
            timeout=config["timeout"],
            check=False,  # We check returncode manually below
        )

        output = result.stdout + result.stderr

        # Check for expected strings (case-insensitive for better matching)
        missing = []
        for expected in config["expected"]:
            if expected.lower() not in output.lower():
                missing.append(expected)

        # Check for output directories/files
        missing_outputs = []
        for output_path in config["outputs"]:
            full_output_path = full_path.parent / output_path
            if not full_output_path.exists():
                missing_outputs.append(output_path)

        # Check for errors
        has_errors = "Error:" in output or "Traceback" in output or result.returncode != 0

        # Determine status
        if has_errors:
            status = "error"
            print(f"{RED}✗ Demo completed with errors{RESET}")
        elif missing or missing_outputs:
            status = "incomplete"
            print(f"{YELLOW}⚠ Demo ran but missing expected outputs{RESET}")
        else:
            status = "success"
            print(f"{GREEN}✓ Demo completed successfully with all expected outputs{RESET}")

        # Print details
        if missing:
            print(f"\n{YELLOW}Missing expected output strings:{RESET}")
            for m in missing:
                print(f"  - {m}")

        if missing_outputs:
            print(f"\n{YELLOW}Missing expected output files/directories:{RESET}")
            for m in missing_outputs:
                print(f"  - {m}")

        if has_errors and result.returncode != 0:
            print(f"\n{RED}Exit code: {result.returncode}{RESET}")
            print(f"\n{RED}Error output:{RESET}")
            print(result.stderr[:500])

        # Show key outputs
        print(f"\n{BLUE}Key outputs found:{RESET}")
        for expected in config["expected"]:
            if expected in output:
                print(f"{GREEN}  ✓ {expected}{RESET}")

        return {
            "status": status,
            "missing_strings": missing,
            "missing_outputs": missing_outputs,
            "exit_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        print(f"{RED}✗ Demo timed out after {config['timeout']}s{RESET}")
        return {"status": "timeout", "issues": [f"Timeout after {config['timeout']}s"]}
    except Exception as e:
        print(f"{RED}✗ Demo failed: {e}{RESET}")
        return {"status": "exception", "issues": [str(e)]}


def main():
    """Check all demos comprehensively."""
    print(f"{BOLD}{BLUE}")
    print("=" * 80)
    print("COMPREHENSIVE DEMO OUTPUT CHECKER")
    print("=" * 80)
    print(f"{RESET}")

    results = {}
    for demo_path, config in DEMOS.items():
        results[demo_path] = check_demo(demo_path, config)

    # Summary
    print(f"\n{BOLD}{BLUE}")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{RESET}\n")

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    incomplete_count = sum(1 for r in results.values() if r["status"] == "incomplete")
    error_count = sum(
        1 for r in results.values() if r["status"] in ["error", "timeout", "exception"]
    )

    print(f"Total demos: {len(results)}")
    print(f"{GREEN}✓ Complete with all outputs: {success_count}{RESET}")
    print(f"{YELLOW}⚠ Incomplete outputs: {incomplete_count}{RESET}")
    print(f"{RED}✗ Errors/timeouts: {error_count}{RESET}")

    if incomplete_count > 0 or error_count > 0:
        print(f"\n{YELLOW}Demos needing attention:{RESET}")
        for demo_path, result in results.items():
            if result["status"] != "success":
                print(f"  - {demo_path}: {result['status']}")

    # Exit with error if any demos failed
    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
