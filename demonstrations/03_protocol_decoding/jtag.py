#!/usr/bin/env python3
"""JTAG Protocol Decoding Demonstration.

This demo showcases Oscura's JTAG (IEEE 1149.1) protocol decoding capabilities,
including TAP state machine tracking, IR/DR shift operations, and standard
instruction identification.

**Features Demonstrated**:
- JTAG TAP state machine analysis
- Instruction Register (IR) decoding
- Data Register (DR) shift operations
- Standard JTAG instruction identification
- Boundary-scan test pattern analysis

**JTAG Fundamentals**:
- TCK (Test Clock): Synchronizes TAP state machine
- TMS (Test Mode Select): Controls state transitions
- TDI (Test Data In): Serial data input
- TDO (Test Data Out): Serial data output

**TAP States**:
- Test-Logic-Reset: Reset state (reached by 5 TMS=1 cycles)
- Run-Test/Idle: Idle state for device operation
- Shift-IR/Shift-DR: Serial data shifting
- Update-IR/Update-DR: Latch shifted data

Usage:
    python jtag_demo.py
    python jtag_demo.py --verbose

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
from demonstrations.common.formatting import print_subheader

# Oscura imports
from oscura.analyzers.protocols.jtag import (
    JTAG_INSTRUCTIONS,
    decode_jtag,
)


class JTAGDemo(BaseDemo):
    """JTAG Protocol Decoding Demonstration.

    This demo generates JTAG signals with various instructions and
    data patterns, then decodes them to demonstrate Oscura's
    JTAG analysis capabilities.
    """

    name = "JTAG Protocol Demo"
    description = "Demonstrates IEEE 1149.1 JTAG protocol decoding"
    category = "debug_protocols"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 50e6  # 50 MHz sampling
        self.tck_freq = 10e6  # 10 MHz TCK
        self.ir_length = 4  # 4-bit IR (typical for small devices)

        # Storage for signals and results
        self.tck = None
        self.tms = None
        self.tdi = None
        self.tdo = None
        self.packets = []

    def generate_data(self) -> None:
        """Generate JTAG test signals.

        Creates JTAG signals demonstrating:
        1. TAP reset sequence (5 TMS=1 cycles)
        2. IDCODE instruction and read
        3. BYPASS instruction
        4. EXTEST instruction
        5. Custom data shift
        """
        # Try loading JTAG data from file
        jtag_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            jtag_file_to_load = self.data_file
            print_info(f"Loading JTAG data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("jtag_boundary_scan.npz"):
            jtag_file_to_load = default_file
            print_info(f"Loading JTAG data from default file: {default_file.name}")

        # Load JTAG from file if found
        if jtag_file_to_load:
            try:
                data = np.load(jtag_file_to_load)
                self.tck = data["tck"]
                self.tms = data["tms"]
                self.tdi = data["tdi"]
                self.tdo = data["tdo"]

                # Load sample rate if present
                if "sample_rate" in data:
                    self.sample_rate = float(data["sample_rate"])

                # Load TCK frequency if present
                if "tck_freq" in data:
                    self.tck_freq = float(data["tck_freq"])

                print_result("JTAG loaded from file", jtag_file_to_load.name)
                print_result("Total samples", len(self.tck))
                print_result("TCK frequency", self.tck_freq / 1e6, "MHz")
                print_result("Sample rate", self.sample_rate / 1e6, "MHz")
                return
            except Exception as e:
                print_info(f"Failed to load JTAG from file: {e}, falling back to synthetic")
                jtag_file_to_load = None

        # Generate synthetic JTAG if not loaded
        print_info("Generating JTAG test signals...")

        samples_per_bit = int(self.sample_rate / self.tck_freq)
        half_period = samples_per_bit // 2

        # Build the signal sequence
        tck_bits = []
        tms_bits = []
        tdi_bits = []
        tdo_bits = []

        def add_clock_cycle(tms_val: int, tdi_val: int = 0, tdo_val: int = 1):
            """Add one TCK clock cycle."""
            # TCK low half
            tck_bits.extend([0] * half_period)
            tms_bits.extend([tms_val] * half_period)
            tdi_bits.extend([tdi_val] * half_period)
            tdo_bits.extend([tdo_val] * half_period)
            # TCK high half (data sampled on rising edge)
            tck_bits.extend([1] * half_period)
            tms_bits.extend([tms_val] * half_period)
            tdi_bits.extend([tdi_val] * half_period)
            tdo_bits.extend([tdo_val] * half_period)

        # ===== Initial idle =====
        for _ in range(2):
            add_clock_cycle(tms_val=0)

        # ===== Reset: 5 TMS=1 cycles =====
        print_info("  Adding TAP reset sequence (5 TMS=1)")
        for _ in range(5):
            add_clock_cycle(tms_val=1)

        # ===== Go to Run-Test/Idle: TMS=0 =====
        add_clock_cycle(tms_val=0)

        # ===== IDCODE instruction (0x02) =====
        print_info("  Adding IDCODE instruction (0x02)")
        # Select-DR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Select-IR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Capture-IR: TMS=0
        add_clock_cycle(tms_val=0)
        # Shift-IR: TMS=0
        add_clock_cycle(tms_val=0)

        # Shift IR: IDCODE = 0x02 = 0010 (LSB first)
        ir_bits = [0, 1, 0, 0]  # 0x02 LSB first
        for i, bit in enumerate(ir_bits):
            tms = 1 if i == len(ir_bits) - 1 else 0  # Exit on last bit
            add_clock_cycle(tms_val=tms, tdi_val=bit)

        # Update-IR: TMS=1
        add_clock_cycle(tms_val=1)
        # Run-Test/Idle: TMS=0
        add_clock_cycle(tms_val=0)

        # Now read IDCODE (32-bit DR)
        # Select-DR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Capture-DR: TMS=0
        add_clock_cycle(tms_val=0)
        # Shift-DR: TMS=0
        add_clock_cycle(tms_val=0)

        # Simulate IDCODE: 0x1CACE551 (example)
        idcode = 0x1CACE551
        print_info(f"  Adding IDCODE data (0x{idcode:08X})")
        for i in range(32):
            bit_tdi = 0  # We're reading, TDI doesn't matter
            bit_tdo = (idcode >> i) & 1
            tms = 1 if i == 31 else 0  # Exit on last bit
            add_clock_cycle(tms_val=tms, tdi_val=bit_tdi, tdo_val=bit_tdo)

        # Update-DR: TMS=1
        add_clock_cycle(tms_val=1)
        # Run-Test/Idle: TMS=0
        add_clock_cycle(tms_val=0)

        # ===== BYPASS instruction (0x0F) =====
        print_info("  Adding BYPASS instruction (0x0F)")
        # Select-DR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Select-IR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Capture-IR: TMS=0
        add_clock_cycle(tms_val=0)
        # Shift-IR: TMS=0
        add_clock_cycle(tms_val=0)

        # Shift IR: BYPASS = 0x0F = 1111 (LSB first)
        ir_bits_bypass = [1, 1, 1, 1]
        for i, bit in enumerate(ir_bits_bypass):
            tms = 1 if i == len(ir_bits_bypass) - 1 else 0
            add_clock_cycle(tms_val=tms, tdi_val=bit)

        # Update-IR: TMS=1
        add_clock_cycle(tms_val=1)
        # Run-Test/Idle: TMS=0
        add_clock_cycle(tms_val=0)

        # ===== EXTEST instruction (0x00) =====
        print_info("  Adding EXTEST instruction (0x00)")
        # Select-DR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Select-IR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Capture-IR: TMS=0
        add_clock_cycle(tms_val=0)
        # Shift-IR: TMS=0
        add_clock_cycle(tms_val=0)

        # Shift IR: EXTEST = 0x00 = 0000 (LSB first)
        ir_bits_extest = [0, 0, 0, 0]
        for i, bit in enumerate(ir_bits_extest):
            tms = 1 if i == len(ir_bits_extest) - 1 else 0
            add_clock_cycle(tms_val=tms, tdi_val=bit)

        # Update-IR: TMS=1
        add_clock_cycle(tms_val=1)
        # Run-Test/Idle: TMS=0
        add_clock_cycle(tms_val=0)

        # Shift some boundary scan data (8 bits)
        # Select-DR-Scan: TMS=1
        add_clock_cycle(tms_val=1)
        # Capture-DR: TMS=0
        add_clock_cycle(tms_val=0)
        # Shift-DR: TMS=0
        add_clock_cycle(tms_val=0)

        # Boundary scan data: 0xA5
        bscan_data = 0xA5
        print_info(f"  Adding boundary scan data (0x{bscan_data:02X})")
        for i in range(8):
            bit_tdi = (bscan_data >> i) & 1
            tms = 1 if i == 7 else 0
            add_clock_cycle(tms_val=tms, tdi_val=bit_tdi)

        # Update-DR: TMS=1
        add_clock_cycle(tms_val=1)
        # Run-Test/Idle: TMS=0
        add_clock_cycle(tms_val=0)

        # Final idle
        for _ in range(4):
            add_clock_cycle(tms_val=0)

        # Convert to numpy arrays
        self.tck = np.array(tck_bits, dtype=bool)
        self.tms = np.array(tms_bits, dtype=bool)
        self.tdi = np.array(tdi_bits, dtype=bool)
        self.tdo = np.array(tdo_bits, dtype=bool)

        print_result("Total samples", len(self.tck))
        print_result("TCK frequency", self.tck_freq / 1e6, "MHz")
        print_result("Sample rate", self.sample_rate / 1e6, "MHz")

    def run_analysis(self) -> None:
        """Decode JTAG signals and analyze transactions."""
        print_subheader("JTAG Decoding")

        # Decode using convenience function
        self.packets = decode_jtag(
            tck=self.tck,
            tms=self.tms,
            tdi=self.tdi,
            tdo=self.tdo,
            sample_rate=self.sample_rate,
        )

        print_result("Decoded packets", len(self.packets))

        # Analyze IR shifts
        print_subheader("Instruction Register (IR) Shifts")
        ir_packets = [p for p in self.packets if "ir_value" in p.annotations]
        self.results["ir_count"] = len(ir_packets)

        for i, pkt in enumerate(ir_packets):
            ir_val = pkt.annotations["ir_value"]
            ir_bits = pkt.annotations["ir_bits"]
            instr = pkt.annotations.get("instruction", "UNKNOWN")
            timestamp = pkt.timestamp * 1e6  # Convert to us

            print_info(
                f"  IR #{i + 1}: 0x{ir_val:02X} ({ir_bits} bits) = {instr} @ {timestamp:.2f} us"
            )

            if instr in JTAG_INSTRUCTIONS.values():
                print_info("    Standard IEEE 1149.1 instruction")

        # Analyze DR shifts
        print_subheader("Data Register (DR) Shifts")
        dr_packets = [p for p in self.packets if "dr_value_tdi" in p.annotations]
        self.results["dr_count"] = len(dr_packets)

        for i, pkt in enumerate(dr_packets):
            dr_val = pkt.annotations["dr_value_tdi"]
            dr_bits = pkt.annotations["dr_bits"]
            timestamp = pkt.timestamp * 1e6

            if dr_bits == 32 and "dr_value_tdo" in pkt.annotations:
                # Likely IDCODE response
                tdo_val = pkt.annotations["dr_value_tdo"]
                print_info(
                    f"  DR #{i + 1}: {dr_bits} bits TDO=0x{tdo_val:08X} @ {timestamp:.2f} us"
                )
                self.results["idcode"] = tdo_val

                # Parse IDCODE fields
                if dr_bits == 32:
                    version = (tdo_val >> 28) & 0xF
                    part_num = (tdo_val >> 12) & 0xFFFF
                    manufacturer = (tdo_val >> 1) & 0x7FF
                    marker = tdo_val & 1

                    print_info("    IDCODE Parsed:")
                    print_info(f"      Version: 0x{version:X}")
                    print_info(f"      Part Number: 0x{part_num:04X}")
                    print_info(f"      Manufacturer: 0x{manufacturer:03X}")
                    print_info(f"      Marker bit: {marker}")
            else:
                print_info(f"  DR #{i + 1}: {dr_bits} bits TDI=0x{dr_val:X} @ {timestamp:.2f} us")

        # Summary statistics
        print_subheader("Summary")
        self.results["total_packets"] = len(self.packets)
        self.results["instruction_names"] = [
            p.annotations.get("instruction", "N/A") for p in ir_packets
        ]

        print_result("Total transactions", len(self.packets))
        print_result("IR shifts", len(ir_packets))
        print_result("DR shifts", len(dr_packets))

        instructions_found = {
            p.annotations.get("instruction") for p in ir_packets if "instruction" in p.annotations
        }
        print_result("Unique instructions", len(instructions_found))
        for instr in instructions_found:
            print_info(f"  - {instr}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate JTAG decoding results."""
        # Check that packets were decoded
        suite.check_greater("Total packets", self.results.get("total_packets", 0), 0)

        # Check IR shifts
        suite.check_greater_equal("IR shifts", self.results.get("ir_count", 0), 3)

        # Check DR shifts
        suite.check_greater_equal("DR shifts", self.results.get("dr_count", 0), 2)

        # Check that we found expected instructions
        instruction_names = self.results.get("instruction_names", [])

        # We expect IDCODE, BYPASS, and EXTEST
        suite.check_true(
            "Found IDCODE instruction", "IDCODE" in instruction_names, category="instructions"
        )

        # Check IDCODE value if present
        if "idcode" in self.results:
            # Our test IDCODE is 0x1CACE551
            suite.check_equal("IDCODE value", self.results["idcode"], 0x1CACE551)

        # Verify signal integrity
        suite.check_true(
            "Signals generated", self.tck is not None and len(self.tck) > 0, category="signals"
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(JTAGDemo))
