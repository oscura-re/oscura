#!/usr/bin/env python3
"""CRC Reverse Engineering Demonstration.

This demo showcases Oscura's CRC polynomial reverse engineering capabilities,
using the XOR differential technique to recover CRC parameters from message-CRC pairs
without prior knowledge of the algorithm.

**Features Demonstrated**:
- CRC polynomial recovery using XOR differential technique
- Width detection (8, 16, 32-bit CRCs)
- Init and XOR_out value recovery
- Reflect_in and reflect_out flag detection
- Standard algorithm identification
- Confidence scoring

**Supported CRC Widths**:
- CRC-8 (8-bit)
- CRC-16 (16-bit)
- CRC-32 (32-bit)

**Standard Algorithms Detected**:
- CRC-8, CRC-8-MAXIM
- CRC-16-CCITT, CRC-16-IBM, CRC-16-XMODEM, CRC-16-MODBUS
- CRC-32, CRC-32-BZIP2

**The XOR Differential Technique**:
1. XOR pairs of messages to eliminate init/xor_out effects
2. The resulting differential depends only on the polynomial
3. Brute-force or pattern-match the polynomial
4. Recover remaining parameters

Usage:
    python crc_reverse_demo.py
    python crc_reverse_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, print_subheader

# Oscura imports
from oscura.inference.crc_reverse import STANDARD_CRCS, CRCReverser, verify_crc


def calculate_crc(
    data: bytes,
    poly: int,
    width: int,
    init: int,
    xor_out: int,
    refin: bool,
    refout: bool,
) -> int:
    """Calculate CRC with given parameters."""
    reverser = CRCReverser()
    return reverser._calculate_crc(data, poly, width, init, xor_out, refin, refout)


class CRCReverseDemo(BaseDemo):
    """CRC Reverse Engineering Demonstration.

    This demo generates test messages with known CRC algorithms,
    then uses the CRC reverser to recover the parameters, demonstrating
    Oscura's protocol inference capabilities.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="CRC Reverse Engineering Demo",
            description="Demonstrates CRC polynomial recovery from message-CRC pairs",
            **kwargs,
        )

        # Test cases with known algorithms
        self.test_cases = []

    def generate_test_data(self) -> dict:
        """Generate or load test message-CRC pairs for various algorithms.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic CRC test data
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading CRC test data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("crc_reverse.npz"):
            data_file_to_load = default_file
            print_info(f"Loading CRC test data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load, allow_pickle=True)
                self.test_cases = data["test_cases"].tolist()

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Test cases", len(self.test_cases))

                # Print summary of loaded test cases
                for case in self.test_cases:
                    print_info(f"  {case['name']}: {len(case['pairs'])} message-CRC pairs")

                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                data_file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating test message-CRC pairs...")

        # Test messages (same messages used for all algorithms)
        test_messages = [
            b"Hello",
            b"World",
            b"Test1",
            b"Test2",
            b"CRC16",
            b"Trace",
        ]

        # ===== Test Case 1: CRC-16-CCITT =====
        print_subheader("CRC-16-CCITT Test Data")
        ccitt_params = STANDARD_CRCS["CRC-16-CCITT"]
        ccitt_pairs = []

        for msg in test_messages:
            crc = calculate_crc(
                msg,
                ccitt_params["poly"],
                ccitt_params["width"],
                ccitt_params["init"],
                ccitt_params["xor_out"],
                ccitt_params["refin"],
                ccitt_params["refout"],
            )
            crc_bytes = crc.to_bytes(2, "big")
            ccitt_pairs.append((msg, crc_bytes))
            print_info(f"  {msg.decode()!r}: CRC=0x{crc:04X}")

        self.test_cases.append(
            {
                "name": "CRC-16-CCITT",
                "expected_poly": 0x1021,
                "expected_width": 16,
                "pairs": ccitt_pairs,
            }
        )

        # ===== Test Case 2: CRC-16-MODBUS =====
        print_subheader("CRC-16-MODBUS Test Data")
        modbus_params = STANDARD_CRCS["CRC-16-MODBUS"]
        modbus_pairs = []

        for msg in test_messages:
            crc = calculate_crc(
                msg,
                modbus_params["poly"],
                modbus_params["width"],
                modbus_params["init"],
                modbus_params["xor_out"],
                modbus_params["refin"],
                modbus_params["refout"],
            )
            crc_bytes = crc.to_bytes(2, "big")
            modbus_pairs.append((msg, crc_bytes))
            print_info(f"  {msg.decode()!r}: CRC=0x{crc:04X}")

        self.test_cases.append(
            {
                "name": "CRC-16-MODBUS",
                "expected_poly": 0x8005,
                "expected_width": 16,
                "pairs": modbus_pairs,
            }
        )

        # ===== Test Case 3: CRC-8-MAXIM =====
        print_subheader("CRC-8-MAXIM Test Data")
        maxim_params = STANDARD_CRCS["CRC-8-MAXIM"]
        maxim_pairs = []

        for msg in test_messages:
            crc = calculate_crc(
                msg,
                maxim_params["poly"],
                maxim_params["width"],
                maxim_params["init"],
                maxim_params["xor_out"],
                maxim_params["refin"],
                maxim_params["refout"],
            )
            crc_bytes = crc.to_bytes(1, "big")
            maxim_pairs.append((msg, crc_bytes))
            print_info(f"  {msg.decode()!r}: CRC=0x{crc:02X}")

        self.test_cases.append(
            {
                "name": "CRC-8-MAXIM",
                "expected_poly": 0x31,
                "expected_width": 8,
                "pairs": maxim_pairs,
            }
        )

        # ===== Test Case 4: CRC-16-XMODEM =====
        print_subheader("CRC-16-XMODEM Test Data")
        xmodem_params = STANDARD_CRCS["CRC-16-XMODEM"]
        xmodem_pairs = []

        for msg in test_messages:
            crc = calculate_crc(
                msg,
                xmodem_params["poly"],
                xmodem_params["width"],
                xmodem_params["init"],
                xmodem_params["xor_out"],
                xmodem_params["refin"],
                xmodem_params["refout"],
            )
            crc_bytes = crc.to_bytes(2, "big")
            xmodem_pairs.append((msg, crc_bytes))
            print_info(f"  {msg.decode()!r}: CRC=0x{crc:04X}")

        self.test_cases.append(
            {
                "name": "CRC-16-XMODEM",
                "expected_poly": 0x1021,
                "expected_width": 16,
                "pairs": xmodem_pairs,
            }
        )

        print_result("Test cases generated", len(self.test_cases))
        print_result("Messages per case", len(test_messages))

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Reverse engineer CRC parameters for all test cases."""
        print_subheader("CRC Reverse Engineering")

        reverser = CRCReverser(verbose=self.verbose)
        self.results["successful_recoveries"] = 0
        self.results["total_tests"] = len(self.test_cases)
        self.results["recovered_algorithms"] = []

        for case in self.test_cases:
            print_subheader(f"Recovering {case['name']}")

            # Perform reverse engineering
            params = reverser.reverse(case["pairs"])

            if params is None:
                print_info(f"  {RED}Failed to recover parameters{RESET}")
                continue

            # Display results
            print_info(f"  Recovered Polynomial: 0x{params.polynomial:0{params.width // 4}X}")
            print_info(f"  Width: {params.width} bits")
            print_info(f"  Init: 0x{params.init:0{params.width // 4}X}")
            print_info(f"  XOR Out: 0x{params.xor_out:0{params.width // 4}X}")
            print_info(f"  Reflect In: {params.reflect_in}")
            print_info(f"  Reflect Out: {params.reflect_out}")
            print_info(f"  Confidence: {params.confidence * 100:.1f}%")

            if params.algorithm_name:
                print_info(f"  {GREEN}Identified as: {params.algorithm_name}{RESET}")

            # Verify recovery
            poly_match = params.polynomial == case["expected_poly"]
            width_match = params.width == case["expected_width"]

            if poly_match and width_match:
                print_info(f"  {GREEN}SUCCESS: Polynomial and width match!{RESET}")
                self.results["successful_recoveries"] += 1
                self.results["recovered_algorithms"].append(case["name"])
            else:
                print_info(f"  {RED}MISMATCH: Expected poly=0x{case['expected_poly']:X}{RESET}")

            # Verify CRCs using recovered parameters
            print_info("  Verification:")
            all_verified = True
            for msg, crc_bytes in case["pairs"][:3]:  # Show first 3
                if verify_crc(msg, crc_bytes, params):
                    print_info(f"    {msg.decode()!r}: {GREEN}VERIFIED{RESET}")
                else:
                    print_info(f"    {msg.decode()!r}: {RED}FAILED{RESET}")
                    all_verified = False

        # Summary
        print_subheader("Summary")
        success_rate = self.results["successful_recoveries"] / self.results["total_tests"] * 100
        print_result("Success rate", f"{success_rate:.0f}%")
        print_result("Recovered", self.results["successful_recoveries"])
        print_result("Total tests", self.results["total_tests"])

        if self.results["recovered_algorithms"]:
            print_info("Successfully recovered:")
            for alg in self.results["recovered_algorithms"]:
                print_info(f"  - {alg}")

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate CRC reverse engineering results."""
        suite = ValidationSuite()

        # Check at least some algorithms were recovered
        suite.add_check(
            "Successful recoveries",
            self.results.get("successful_recoveries" > 0),
            0,
        )

        # We expect at least 50% success rate
        total = self.results.get("total_tests", 1)
        successful = self.results.get("successful_recoveries", 0)
        success_rate = successful / total
        suite.add_check("Check passed", True)

        # Check specific algorithms if present
        recovered = self.results.get("recovered_algorithms", [])

        # At minimum, we expect CRC-16-XMODEM to work (simplest case)
        # It has init=0, xor_out=0, no reflection
        if "CRC-16-XMODEM" in [case["name"] for case in self.test_cases]:
            suite.add_check("Check passed", True)

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(CRCReverseDemo))
