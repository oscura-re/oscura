#!/usr/bin/env python3
"""SWD Protocol Decoding Demonstration.

This demo showcases Oscura's ARM Serial Wire Debug (SWD) protocol decoding
capabilities, including DP/AP register access, ACK responses, and data transfers.

**Features Demonstrated**:
- SWD transaction decoding
- Debug Port (DP) and Access Port (AP) access detection
- Read/Write operation identification
- ACK response handling (OK/WAIT/FAULT)
- Data phase decoding with parity verification

**SWD Fundamentals**:
- SWCLK (Serial Wire Clock): Synchronizes data transfer
- SWDIO (Serial Wire Data I/O): Bidirectional data line

**SWD Transaction Phases**:
1. Host Request: 8 bits (Start, APnDP, RnW, A[2:3], Parity, Stop, Park)
2. Turnaround: Line direction change
3. Target ACK: 3 bits (OK=001, WAIT=010, FAULT=100)
4. Data Phase (if ACK=OK): 32 bits + parity

**Common DP Registers**:
- 0x00: DPIDR (Debug Port ID Register)
- 0x04: CTRL/STAT (Control/Status)
- 0x08: SELECT (AP Select)
- 0x0C: RDBUFF (Read Buffer)

Usage:
    python swd_demo.py
    python swd_demo.py --verbose
    python swd_demo.py --data-file path/to/swd_capture.npz

**Data File Format**:
NPZ files should contain:
- `swclk`: Boolean array of SWCLK signal samples
- `swdio`: Boolean array of SWDIO signal samples
- `sample_rate`: Float, sampling rate in Hz

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader

# Oscura imports
from oscura.analyzers.protocols.swd import decode_swd

# SWD DP register names
DP_REGISTERS = {
    0x00: "DPIDR",
    0x04: "CTRL/STAT",
    0x08: "SELECT",
    0x0C: "RDBUFF",
}


class SWDDemo(BaseDemo):
    """SWD Protocol Decoding Demonstration.

    This demo generates SWD signals with various DP/AP register accesses
    and decodes them to demonstrate Oscura's SWD analysis capabilities.
    """

    name = "SWD Protocol Demo"
    description = "Demonstrates ARM Serial Wire Debug protocol decoding"
    category = "debug_protocols"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 50e6  # 50 MHz sampling
        self.swclk_freq = 4e6  # 4 MHz SWCLK (typical SWD speed)

        # Storage for signals and results
        self.swclk = None
        self.swdio = None
        self.packets = []

    def generate_data(self) -> None:
        """Generate or load SWD test signals.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data

        Creates SWD signals demonstrating:
        1. Read DPIDR (Debug Port ID)
        2. Write CTRL/STAT (enable debug)
        3. Read AP register
        4. Write transaction
        """
        # Try loading SWD data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading SWD data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("swd_debug_session.npz"):
            file_to_load = default_file
            print_info(f"Loading SWD data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load)
                self.swclk = data["swclk"]
                self.swdio = data["swdio"]
                self.sample_rate = float(data["sample_rate"])

                print_result("SWD loaded from file", file_to_load.name)
                print_result("Total samples", len(self.swclk))
                print_result("Sample rate", self.sample_rate / 1e6, "MHz")
                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating SWD test signals...")

        samples_per_bit = int(self.sample_rate / self.swclk_freq)
        half_period = samples_per_bit // 2

        # Build the signal sequence
        swclk_bits = []
        swdio_bits = []

        def add_clock_cycle(swdio_val: int):
            """Add one SWCLK clock cycle."""
            # SWCLK low half
            swclk_bits.extend([0] * half_period)
            swdio_bits.extend([swdio_val] * half_period)
            # SWCLK high half (data sampled on rising edge)
            swclk_bits.extend([1] * half_period)
            swdio_bits.extend([swdio_val] * half_period)

        def add_swd_transaction(apndp: int, rnw: int, addr: int, data: int = 0):
            """Add complete SWD transaction.

            Args:
                apndp: 0=DP, 1=AP
                rnw: 0=Write, 1=Read
                addr: Register address (bits 2-3 only used)
                data: Data value for write, or expected read response
            """
            # Extract address bits (A[2] and A[3])
            a2 = (addr >> 2) & 1
            a3 = (addr >> 3) & 1

            # Calculate request parity (odd parity of APnDP, RnW, A[2:3])
            parity = (apndp + rnw + a2 + a3) % 2

            # Host Request Phase (8 bits)
            # Bit 0: Start (1)
            add_clock_cycle(1)
            # Bit 1: APnDP
            add_clock_cycle(apndp)
            # Bit 2: RnW
            add_clock_cycle(rnw)
            # Bit 3: A[2]
            add_clock_cycle(a2)
            # Bit 4: A[3]
            add_clock_cycle(a3)
            # Bit 5: Parity
            add_clock_cycle(parity)
            # Bit 6: Stop (0)
            add_clock_cycle(0)
            # Bit 7: Park (1)
            add_clock_cycle(1)

            # Turnaround (host releases line)
            add_clock_cycle(1)  # Line goes high due to pull-up

            # ACK Response (target drives, 3 bits)
            # ACK=OK (001, LSB first)
            add_clock_cycle(1)  # OK bit 0
            add_clock_cycle(0)  # OK bit 1
            add_clock_cycle(0)  # OK bit 2

            # Turnaround
            add_clock_cycle(1)

            # Data Phase (32 bits + parity)
            data_parity = 0
            for i in range(32):
                bit = (data >> i) & 1
                data_parity ^= bit
                add_clock_cycle(bit)

            # Data parity (odd)
            add_clock_cycle(data_parity)

            # Turnaround/idle
            add_clock_cycle(1)
            add_clock_cycle(1)

        # ===== Initial idle (line high) =====
        for _ in range(8):
            add_clock_cycle(1)

        # ===== Line reset sequence (50+ clocks with SWDIO=1) =====
        print_info("  Adding line reset sequence")
        for _ in range(52):
            add_clock_cycle(1)

        # Idle
        for _ in range(4):
            add_clock_cycle(0)

        # ===== Transaction 1: Read DPIDR (DP, Read, Addr=0x00) =====
        print_info("  Adding DP Read DPIDR (addr=0x00)")
        # DPIDR example value: 0x0BB11477 (typical ARM Cortex-M)
        add_swd_transaction(apndp=0, rnw=1, addr=0x00, data=0x0BB11477)

        # Idle
        for _ in range(4):
            add_clock_cycle(1)

        # ===== Transaction 2: Write CTRL/STAT (DP, Write, Addr=0x04) =====
        print_info("  Adding DP Write CTRL/STAT (addr=0x04)")
        # Enable debug (CSYSPWRUPREQ | CDBGPWRUPREQ)
        ctrl_value = 0x50000000
        add_swd_transaction(apndp=0, rnw=0, addr=0x04, data=ctrl_value)

        # Idle
        for _ in range(4):
            add_clock_cycle(1)

        # ===== Transaction 3: Write SELECT (DP, Write, Addr=0x08) =====
        print_info("  Adding DP Write SELECT (addr=0x08)")
        # Select AP 0, bank 0
        select_value = 0x00000000
        add_swd_transaction(apndp=0, rnw=0, addr=0x08, data=select_value)

        # Idle
        for _ in range(4):
            add_clock_cycle(1)

        # ===== Transaction 4: Read AP IDR (AP, Read, Addr=0x0C) =====
        print_info("  Adding AP Read IDR (addr=0x0C)")
        # AP IDR example value
        ap_idr = 0x24770011
        add_swd_transaction(apndp=1, rnw=1, addr=0x0C, data=ap_idr)

        # Idle
        for _ in range(4):
            add_clock_cycle(1)

        # ===== Transaction 5: Read RDBUFF (DP, Read, Addr=0x0C) =====
        print_info("  Adding DP Read RDBUFF (addr=0x0C)")
        # Read buffer contains previous AP read result
        add_swd_transaction(apndp=0, rnw=1, addr=0x0C, data=ap_idr)

        # Final idle
        for _ in range(8):
            add_clock_cycle(1)

        # Convert to numpy arrays
        self.swclk = np.array(swclk_bits, dtype=bool)
        self.swdio = np.array(swdio_bits, dtype=bool)

        print_result("Total samples", len(self.swclk))
        print_result("SWCLK frequency", self.swclk_freq / 1e6, "MHz")
        print_result("Sample rate", self.sample_rate / 1e6, "MHz")

    def run_analysis(self) -> None:
        """Decode SWD signals and analyze transactions."""
        print_subheader("SWD Decoding")

        # Decode using convenience function
        self.packets = decode_swd(
            swclk=self.swclk,
            swdio=self.swdio,
            sample_rate=self.sample_rate,
        )

        print_result("Decoded packets", len(self.packets))

        # Analyze transactions
        print_subheader("SWD Transactions")
        self.results["total_packets"] = len(self.packets)

        dp_reads = []
        dp_writes = []
        ap_reads = []
        ap_writes = []

        for i, pkt in enumerate(self.packets):
            port = pkt.annotations.get("apndp", "DP")
            is_read = pkt.annotations.get("read", False)
            reg_addr = pkt.annotations.get("register_addr", 0)
            ack = pkt.annotations.get("ack", "UNKNOWN")
            data = pkt.annotations.get("data", 0)
            timestamp = pkt.timestamp * 1e6

            # Get register name if DP
            reg_name = ""
            if port == "DP":
                reg_name = DP_REGISTERS.get(reg_addr, f"0x{reg_addr:02X}")
            else:
                reg_name = f"0x{reg_addr:02X}"

            access_type = "Read" if is_read else "Write"
            print_info(
                f"  #{i + 1}: {port} {access_type} {reg_name} = 0x{data:08X} [{ack}] @ {timestamp:.2f} us"
            )

            # Categorize
            if port == "DP":
                if is_read:
                    dp_reads.append(pkt)
                else:
                    dp_writes.append(pkt)
            else:
                if is_read:
                    ap_reads.append(pkt)
                else:
                    ap_writes.append(pkt)

            # Check for errors
            if pkt.errors:
                for err in pkt.errors:
                    print_info(f"    Error: {err}")

        self.results["dp_reads"] = len(dp_reads)
        self.results["dp_writes"] = len(dp_writes)
        self.results["ap_reads"] = len(ap_reads)
        self.results["ap_writes"] = len(ap_writes)

        # Extract DPIDR if present
        for pkt in dp_reads:
            if pkt.annotations.get("register_addr") == 0x00:
                dpidr = pkt.annotations.get("data", 0)
                self.results["dpidr"] = dpidr
                print_subheader("DPIDR Analysis")
                print_info(f"  DPIDR Value: 0x{dpidr:08X}")

                # Parse DPIDR fields (ARM Debug Interface v5+)
                revision = (dpidr >> 28) & 0xF
                partno = (dpidr >> 20) & 0xFF
                min_dp = (dpidr >> 16) & 1
                version = (dpidr >> 12) & 0xF
                designer = (dpidr >> 1) & 0x7FF
                presence = dpidr & 1

                print_info(f"    Revision: {revision}")
                print_info(f"    Part Number: 0x{partno:02X}")
                print_info(f"    Minimal DP: {min_dp}")
                print_info(f"    Version: {version}")
                print_info(f"    Designer: 0x{designer:03X}")
                print_info(f"    Presence: {presence}")
                break

        # ACK analysis
        print_subheader("ACK Statistics")
        ack_counts = {"OK": 0, "WAIT": 0, "FAULT": 0, "INVALID": 0}
        for pkt in self.packets:
            ack = pkt.annotations.get("ack", "INVALID")
            ack_counts[ack] = ack_counts.get(ack, 0) + 1

        for ack_type, count in ack_counts.items():
            if count > 0:
                print_result(f"ACK={ack_type}", count)

        self.results["ack_ok_count"] = ack_counts.get("OK", 0)

        # Summary
        print_subheader("Summary")
        print_result("Total transactions", len(self.packets))
        print_result("DP Reads", len(dp_reads))
        print_result("DP Writes", len(dp_writes))
        print_result("AP Reads", len(ap_reads))
        print_result("AP Writes", len(ap_writes))

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate SWD decoding results."""
        # Check that packets were decoded
        suite.check_greater("Total packets", self.results.get("total_packets", 0), 0)

        # Check we have expected number of transactions
        suite.check_greater_equal("Total transactions", self.results.get("total_packets", 0), 4)

        # Check for successful ACKs
        suite.check_greater("ACK=OK count", self.results.get("ack_ok_count", 0), 0)

        # Check DP operations
        suite.check_greater_equal(
            "DP reads", self.results.get("dp_reads", 0), 2, category="dp_access"
        )
        suite.check_greater_equal(
            "DP writes", self.results.get("dp_writes", 0), 2, category="dp_access"
        )

        # Check AP operations
        suite.check_greater_equal(
            "AP reads", self.results.get("ap_reads", 0), 1, category="ap_access"
        )

        # Check DPIDR if present
        if "dpidr" in self.results:
            # Our test DPIDR is 0x0BB11477
            suite.check_equal("DPIDR value", self.results["dpidr"], 0x0BB11477)

        # Verify signal integrity
        suite.check_true(
            "Signals generated", self.swclk is not None and len(self.swclk) > 0, category="signals"
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(SWDDemo))
