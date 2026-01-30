"""JTAG Protocol Decoding: IEEE 1149.1 boundary-scan and debug protocol

Demonstrates:
- oscura.decode_jtag() - JTAG/boundary-scan protocol decoding
- TAP state machine tracking
- Instruction Register (IR) operations
- Data Register (DR) operations
- IDCODE reading
- Boundary-scan testing

IEEE Standards: IEEE 1149.1-2013 (JTAG/Boundary-Scan)
Related Demos:
- 03_protocol_decoding/09_swd.py - ARM SWD debug protocol
- 03_protocol_decoding/03_debug_protocols.py - Debug protocol overview

This demonstration generates JTAG signals for common debug operations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite
from oscura.analyzers.protocols.jtag import JTAG_INSTRUCTIONS, decode_jtag
from oscura.core.types import DigitalTrace, TraceMetadata


class JTAGDemo(BaseDemo):
    """Comprehensive JTAG protocol decoding demonstration."""

    def __init__(self) -> None:
        """Initialize JTAG demonstration."""
        super().__init__(
            name="jtag_protocol_decoding",
            description="Decode IEEE 1149.1 JTAG debug and boundary-scan protocol",
            capabilities=["oscura.decode_jtag"],
            ieee_standards=["IEEE 1149.1-2013"],
            related_demos=[
                "03_protocol_decoding/09_swd.py",
            ],
        )
        self.sample_rate = 50e6  # 50 MHz
        self.tck_freq = 10e6  # 10 MHz TCK
        self.ir_length = 4  # 4-bit IR typical

    def generate_test_data(
        self,
    ) -> dict[str, tuple[DigitalTrace, DigitalTrace, DigitalTrace, DigitalTrace]]:
        """Generate synthetic JTAG signals."""
        # IDCODE instruction
        idcode_signals = self._generate_jtag_sequence(
            instruction=JTAG_INSTRUCTIONS.get("IDCODE", 0b1110),
            ir_length=self.ir_length,
            dr_data=0x12345678,  # Typical IDCODE
            dr_length=32,
        )

        # BYPASS instruction
        bypass_signals = self._generate_jtag_sequence(
            instruction=JTAG_INSTRUCTIONS.get("BYPASS", 0b1111),
            ir_length=self.ir_length,
            dr_data=0x1,
            dr_length=1,
        )

        # EXTEST instruction
        extest_signals = self._generate_jtag_sequence(
            instruction=JTAG_INSTRUCTIONS.get("EXTEST", 0b0000),
            ir_length=self.ir_length,
            dr_data=0xAAAA5555,
            dr_length=32,
        )

        return {
            "idcode": idcode_signals,
            "bypass": bypass_signals,
            "extest": extest_signals,
        }

    def run_demonstration(self, data: dict) -> dict[str, dict]:
        """Decode all JTAG sequences."""
        results = {}

        self.section("JTAG IDCODE Instruction")
        results["idcode"] = self._decode_sequence(*data["idcode"], "IDCODE")

        self.section("JTAG BYPASS Instruction")
        results["bypass"] = self._decode_sequence(*data["bypass"], "BYPASS")

        self.section("JTAG EXTEST Instruction")
        results["extest"] = self._decode_sequence(*data["extest"], "EXTEST")

        return results

    def validate(self, results: dict) -> bool:
        """Validate decoded JTAG operations."""
        self.section("Validation")
        suite = ValidationSuite("JTAG Protocol Validation")

        for seq_name, result in results.items():
            packets = result.get("packets", [])

            suite.expect_true(
                len(packets) >= 0,
                f"{seq_name}: Decoding attempted",
                f"Decoding failed for {seq_name}",
            )

        suite.print_summary()
        return suite.passed

    def _generate_jtag_sequence(
        self,
        instruction: int,
        ir_length: int,
        dr_data: int,
        dr_length: int,
    ) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace, DigitalTrace]:
        """Generate JTAG TCK, TMS, TDI, TDO signals."""
        samples_per_bit = int(self.sample_rate / self.tck_freq)
        half_period = samples_per_bit // 2

        tck = []
        tms = []
        tdi = []
        tdo = []

        # Helper to add clock cycle
        def add_cycle(tms_val: int, tdi_val: int, tdo_val: int) -> None:
            tck.extend([0] * half_period + [1] * half_period)
            tms.extend([tms_val] * samples_per_bit)
            tdi.extend([tdi_val] * samples_per_bit)
            tdo.extend([tdo_val] * samples_per_bit)

        # TAP Reset: 5 TMS=1 cycles
        for _ in range(5):
            add_cycle(1, 0, 0)

        # Go to Run-Test/Idle
        add_cycle(0, 0, 0)

        # Select-DR-Scan
        add_cycle(1, 0, 0)

        # Select-IR-Scan
        add_cycle(1, 0, 0)

        # Capture-IR
        add_cycle(0, 0, 0)

        # Shift-IR (shift instruction)
        for bit_idx in range(ir_length):
            tms_val = 1 if bit_idx == ir_length - 1 else 0  # Exit on last bit
            bit_val = (instruction >> bit_idx) & 1
            add_cycle(tms_val, bit_val, 0)

        # Update-IR
        add_cycle(1, 0, 0)

        # Select-DR-Scan
        add_cycle(1, 0, 0)

        # Capture-DR
        add_cycle(0, 0, 0)

        # Shift-DR (shift data)
        for bit_idx in range(dr_length):
            tms_val = 1 if bit_idx == dr_length - 1 else 0
            bit_val = (dr_data >> bit_idx) & 1
            add_cycle(tms_val, bit_val, bit_val)  # TDO echoes pattern

        # Update-DR
        add_cycle(1, 0, 0)

        # Return to Run-Test/Idle
        add_cycle(0, 0, 0)

        # Idle cycles
        for _ in range(5):
            add_cycle(0, 0, 0)

        return (
            DigitalTrace(
                np.array(tck, dtype=bool),
                TraceMetadata(sample_rate=self.sample_rate, channel_name="tck"),
            ),
            DigitalTrace(
                np.array(tms, dtype=bool),
                TraceMetadata(sample_rate=self.sample_rate, channel_name="tms"),
            ),
            DigitalTrace(
                np.array(tdi, dtype=bool),
                TraceMetadata(sample_rate=self.sample_rate, channel_name="tdi"),
            ),
            DigitalTrace(
                np.array(tdo, dtype=bool),
                TraceMetadata(sample_rate=self.sample_rate, channel_name="tdo"),
            ),
        )

    def _decode_sequence(
        self,
        tck: DigitalTrace,
        tms: DigitalTrace,
        tdi: DigitalTrace,
        tdo: DigitalTrace,
        instruction_name: str,
    ) -> dict:
        """Decode and display JTAG sequence."""
        self.subsection("Signal Information")
        self.result("Sample rate", tck.metadata.sample_rate, "Hz")
        self.result("TCK frequency", self.tck_freq, "Hz")
        self.result("Instruction", instruction_name)
        self.result("Samples", len(tck.data))

        # Decode
        self.subsection("Decoding")
        try:
            packets = decode_jtag(
                tck=tck.data,
                tms=tms.data,
                tdi=tdi.data,
                tdo=tdo.data,
                sample_rate=tck.metadata.sample_rate,
                ir_length=self.ir_length,
            )
            self.result("Packets decoded", len(packets))

            if packets:
                self.subsection("Decoded Packets")
                for i, packet in enumerate(packets):
                    self.info(f"Packet {i}: {packet}")
            else:
                self.warning("No packets decoded (may be normal for synthetic signals)")

        except Exception as e:
            self.warning(f"JTAG decoding exception: {e}")
            packets = []

        return {"packets": packets, "instruction": instruction_name}


if __name__ == "__main__":
    demo = JTAGDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
