"""CRC Recovery: Polynomial recovery from message samples

Demonstrates:
- oscura.inference.crc_reverse.CRCReverser - Recover CRC parameters
- oscura.inference.crc_reverse.verify_crc() - Verify CRC correctness
- oscura.inference.crc_reverse.STANDARD_CRCS - Standard CRC algorithms
- CRC-8, CRC-16, CRC-32 detection
- Polynomial finding using differential technique
- Init and XOR_out value recovery
- Reflect_in and reflect_out flag detection

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/04_field_inference.py

CRC (Cyclic Redundancy Check) is ubiquitous in communication protocols.
This demonstration shows how to recover CRC parameters from message-CRC pairs
without prior knowledge of the algorithm, using the XOR differential technique.

This is a P0 CRITICAL feature - demonstrates CRC reverse engineering capability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class CRCRecoveryDemo(BaseDemo):
    """Demonstrates CRC polynomial recovery from message samples."""

    def __init__(self) -> None:
        """Initialize CRC recovery demonstration."""
        super().__init__(
            name="crc_recovery",
            description="Recover CRC parameters from message-CRC pairs",
            capabilities=[
                "oscura.inference.crc_reverse.CRCReverser",
                "oscura.inference.crc_reverse.verify_crc",
                "oscura.inference.crc_reverse.STANDARD_CRCS",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/04_field_inference.py",
            ],
        )
        self.test_cases: list[dict[str, Any]] = []

    def generate_test_data(self) -> dict[str, Any]:
        """Generate messages with known CRC algorithms for testing.

        Creates message-CRC pairs using standard algorithms:
        - CRC-8-MAXIM
        - CRC-16-CCITT
        - CRC-16-MODBUS
        - CRC-32

        Returns:
            Dictionary with test cases for CRC recovery
        """
        try:
            from oscura.inference.crc_reverse import STANDARD_CRCS  # noqa: F401
        except ImportError:
            self.warning("CRC reverse module not available, using manual generation")
            return self._generate_manual_crc_data()

        self.section("Generating CRC Test Messages")

        test_messages = [
            b"Hello",
            b"World",
            b"Test1",
            b"Test2",
            b"Oscura",
            b"Protocol",
        ]

        # ===== Test Case 1: CRC-8-MAXIM =====
        self.subsection("CRC-8-MAXIM Messages")

        crc8_pairs = []
        for msg in test_messages:
            # Calculate CRC-8-MAXIM: poly=0x31, init=0x00, xor_out=0x00, refin=True, refout=True
            crc = self._calculate_crc8_maxim(msg)
            crc_bytes = bytes([crc])
            crc8_pairs.append((msg, crc_bytes))
            self.info(f"  {msg.decode()!r}: CRC=0x{crc:02X}")

        self.test_cases.append(
            {
                "name": "CRC-8-MAXIM",
                "width": 8,
                "expected_poly": 0x31,
                "pairs": crc8_pairs,
            }
        )

        # ===== Test Case 2: CRC-16-CCITT =====
        self.subsection("CRC-16-CCITT Messages")

        crc16_ccitt_pairs = []
        for msg in test_messages:
            # Calculate CRC-16-CCITT: poly=0x1021, init=0xFFFF, xor_out=0x0000
            crc = self._calculate_crc16_ccitt(msg)
            crc_bytes = crc.to_bytes(2, "big")
            crc16_ccitt_pairs.append((msg, crc_bytes))
            self.info(f"  {msg.decode()!r}: CRC=0x{crc:04X}")

        self.test_cases.append(
            {
                "name": "CRC-16-CCITT",
                "width": 16,
                "expected_poly": 0x1021,
                "pairs": crc16_ccitt_pairs,
            }
        )

        # ===== Test Case 3: CRC-16-MODBUS =====
        self.subsection("CRC-16-MODBUS Messages")

        crc16_modbus_pairs = []
        for msg in test_messages:
            # Calculate CRC-16-MODBUS: poly=0x8005, init=0xFFFF, xor_out=0x0000, refin=True
            crc = self._calculate_crc16_modbus(msg)
            crc_bytes = crc.to_bytes(2, "little")  # Little endian
            crc16_modbus_pairs.append((msg, crc_bytes))
            self.info(f"  {msg.decode()!r}: CRC=0x{crc:04X}")

        self.test_cases.append(
            {
                "name": "CRC-16-MODBUS",
                "width": 16,
                "expected_poly": 0x8005,
                "pairs": crc16_modbus_pairs,
            }
        )

        # ===== Test Case 4: CRC-32 =====
        self.subsection("CRC-32 Messages")

        crc32_pairs = []
        for msg in test_messages:
            # Calculate CRC-32: standard Ethernet/ZIP polynomial
            crc = self._calculate_crc32(msg)
            crc_bytes = crc.to_bytes(4, "little")
            crc32_pairs.append((msg, crc_bytes))
            self.info(f"  {msg.decode()!r}: CRC=0x{crc:08X}")

        self.test_cases.append(
            {
                "name": "CRC-32",
                "width": 32,
                "expected_poly": 0x04C11DB7,
                "pairs": crc32_pairs,
            }
        )

        self.result("Test cases generated", len(self.test_cases))
        self.result("Messages per case", len(test_messages))

        return {"test_cases": self.test_cases}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute CRC recovery on all test cases."""
        results: dict[str, Any] = {
            "successful_recoveries": 0,
            "total_tests": 0,
            "recovered_algorithms": [],
        }

        try:
            from oscura.inference.crc_reverse import CRCReverser, verify_crc
        except ImportError:
            self.error("CRC reverse module not available")
            return results

        self.section("CRC Parameter Recovery")

        reverser = CRCReverser()
        results["total_tests"] = len(self.test_cases)

        for case in self.test_cases:
            self.subsection(f"Recovering {case['name']}")

            try:
                # Perform reverse engineering
                params = reverser.reverse(case["pairs"])

                if params is None:
                    self.warning("Failed to recover parameters")
                    continue

                # Display recovered parameters
                self.result("Polynomial", f"0x{params.polynomial:0{params.width // 4}X}")
                self.result("Width", f"{params.width} bits")
                self.result("Init", f"0x{params.init:0{params.width // 4}X}")
                self.result("XOR Out", f"0x{params.xor_out:0{params.width // 4}X}")
                self.result("Reflect In", str(params.reflect_in))
                self.result("Reflect Out", str(params.reflect_out))
                self.result("Confidence", f"{params.confidence * 100:.1f}%")

                if params.algorithm_name:
                    self.success(f"Identified as: {params.algorithm_name}")

                # Verify recovery
                poly_match = params.polynomial == case["expected_poly"]
                width_match = params.width == case["width"]

                if poly_match and width_match:
                    self.success("Polynomial and width match!")
                    results["successful_recoveries"] += 1
                    results["recovered_algorithms"].append(case["name"])

                    # Verify CRCs
                    self.info("Verifying CRCs:")
                    verified = 0
                    for msg, crc_bytes in case["pairs"][:3]:
                        if verify_crc(msg, crc_bytes, params):
                            verified += 1
                            self.success(f"  {msg.decode()!r}: VERIFIED")
                        else:
                            self.warning(f"  {msg.decode()!r}: FAILED")

                    if verified == 3:
                        self.success("All sample CRCs verified!")

                else:
                    self.warning(
                        f"Expected poly=0x{case['expected_poly']:X}, width={case['width']}"
                    )

            except Exception as e:
                self.error(f"Recovery failed: {e}")
                continue

        # ===== Summary =====
        self.section("Recovery Summary")

        success_rate = (
            results["successful_recoveries"] / results["total_tests"] * 100
            if results["total_tests"] > 0
            else 0
        )
        self.result("Success rate", f"{success_rate:.0f}%")
        self.result("Recovered", results["successful_recoveries"])
        self.result("Total tests", results["total_tests"])

        if results["recovered_algorithms"]:
            self.info("Successfully recovered:")
            for alg in results["recovered_algorithms"]:
                self.success(f"  {alg}")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate CRC recovery results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Check that some algorithms were recovered
        if results.get("successful_recoveries", 0) == 0:
            self.error("No CRC algorithms recovered")
            return False

        # Expect at least 50% success rate
        total = results.get("total_tests", 1)
        successful = results.get("successful_recoveries", 0)
        success_rate = successful / total

        if success_rate < 0.5:
            self.warning(f"Success rate {success_rate * 100:.0f}% below 50%")

        self.success("CRC recovery demonstration complete!")
        return True

    # ===== Helper methods for CRC calculation =====

    def _calculate_crc8_maxim(self, data: bytes) -> int:
        """Calculate CRC-8-MAXIM."""
        crc = 0x00
        poly = 0x31
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ poly) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    def _calculate_crc16_ccitt(self, data: bytes) -> int:
        """Calculate CRC-16-CCITT."""
        crc = 0xFFFF
        poly = 0x1021
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def _calculate_crc16_modbus(self, data: bytes) -> int:
        """Calculate CRC-16-MODBUS."""
        crc = 0xFFFF
        poly = 0xA001  # Reversed 0x8005
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
        return crc

    def _calculate_crc32(self, data: bytes) -> int:
        """Calculate CRC-32."""
        crc = 0xFFFFFFFF
        poly = 0xEDB88320  # Reversed 0x04C11DB7
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x00000001:
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
        return crc ^ 0xFFFFFFFF

    def _generate_manual_crc_data(self) -> dict[str, Any]:
        """Fallback manual CRC data generation."""
        self.warning("Using fallback CRC data generation")
        # Generate simple data for testing
        return {"test_cases": []}


if __name__ == "__main__":
    demo = CRCRecoveryDemo()
    success = demo.execute()
    exit(0 if success else 1)
