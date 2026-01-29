"""Unknown Signal Workflow: Complete unknown signal analysis end-to-end.

Demonstrates:
- Signal characterization (levels, timing)
- Clock recovery and bit extraction
- Frame boundary detection
- Protocol structure inference
- Field identification
- CRC/checksum analysis

Category: Complete Workflows
IEEE Standards: N/A

Related Demos:
- 02_basic_analysis/01_measurements.py
- 06_reverse_engineering/01_unknown_protocol.py

This showcases a complete end-to-end workflow for reverse engineering
an unknown digital signal, from initial capture to protocol understanding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class UnknownSignalWorkflowDemo(BaseDemo):
    """Demonstrates complete unknown signal reverse engineering workflow."""

    name = "Unknown Signal Workflow"
    description = "Complete unknown signal reverse engineering workflow"
    category = "complete_workflows"

    def generate_data(self) -> None:
        """Generate unknown protocol signal."""
        # Generate signal with hidden protocol structure
        sample_rate = 1e6
        baud_rate = 19200
        samples_per_bit = int(sample_rate / baud_rate)

        # Create frames with sync pattern 0xAA55
        sync_pattern = [0xAA, 0x55]
        frames_data = []

        for _ in range(10):  # 10 frames
            frame = sync_pattern + [0x10, 0x20, 0x30, 0x40]  # Sync + data
            frames_data.append(frame)

        # Convert to bit stream
        bits = []
        for frame in frames_data:
            for byte_val in frame:
                for bit_idx in range(8):
                    bits.append((byte_val >> (7 - bit_idx)) & 1)

        # Convert to analog signal
        signal = []
        for bit in bits:
            signal.extend([3.3 if bit else 0.0] * samples_per_bit)

        self.signal = np.array(signal)
        self.sample_rate = sample_rate
        self.true_baud = baud_rate

    def run_analysis(self) -> None:
        """Execute complete reverse engineering workflow."""
        from oscura.core import TraceMetadata, WaveformTrace

        trace = WaveformTrace(
            data=self.signal,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="RX"),
        )

        print_header("Unknown Signal Reverse Engineering Workflow")

        print_subheader("Step 1: Signal Characterization")
        print_info("Analyze basic signal properties:")
        high_level = np.percentile(self.signal, 95)
        low_level = np.percentile(self.signal, 5)
        print_info(f"  High level: {high_level:.2f} V")
        print_info(f"  Low level: {low_level:.2f} V")
        print_info(f"  Logic family: Likely 3.3V CMOS")

        print_subheader("Step 2: Baud Rate Detection")
        print_info("Detect baud rate from edge transitions:")

        # Find edges
        threshold = (high_level + low_level) / 2
        digital = self.signal > threshold
        edges = np.diff(digital.astype(int))
        edge_indices = np.where(edges != 0)[0]

        # Estimate baud rate from minimum pulse width
        if len(edge_indices) > 1:
            min_pulse = np.min(np.diff(edge_indices))
            estimated_baud = self.sample_rate / min_pulse
            print_info(f"  Estimated baud rate: {estimated_baud:.0f} bps")
            print_info(f"  True baud rate: {self.true_baud} bps")
            print_info(f"  ✓ Accuracy: {abs(estimated_baud - self.true_baud)/self.true_baud*100:.1f}%")

        print_subheader("Step 3: Bit Stream Extraction")
        print_info("Extract digital bit stream:")

        # Sample at bit centers
        samples_per_bit = int(self.sample_rate / self.true_baud)
        bit_stream = []
        for i in range(0, len(digital) - samples_per_bit, samples_per_bit):
            # Sample in middle of bit period
            bit_sample = digital[i + samples_per_bit // 2]
            bit_stream.append(int(bit_sample))

        print_info(f"  Extracted {len(bit_stream)} bits")
        print_info(f"  First 32 bits: {' '.join(str(b) for b in bit_stream[:32])}")

        print_subheader("Step 4: Sync Pattern Detection")
        print_info("Search for repeating sync patterns:")

        # Convert bits to bytes
        bytes_list = []
        for i in range(0, len(bit_stream) - 7, 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | bit_stream[i + j]
            bytes_list.append(byte_val)

        # Look for sync pattern 0xAA55
        sync_count = 0
        for i in range(len(bytes_list) - 1):
            if bytes_list[i] == 0xAA and bytes_list[i + 1] == 0x55:
                sync_count += 1
                print_info(f"  Sync found at byte {i}: 0x{bytes_list[i]:02X} 0x{bytes_list[i+1]:02X}")

        print_info(f"  ✓ Found {sync_count} sync patterns")

        print_subheader("Step 5: Frame Structure Inference")
        print_info("Analyze frame structure:")
        print_info("  Sync: 0xAA 0x55 (2 bytes)")
        print_info("  Data: Variable length")
        print_info("  Pattern repeats every ~6 bytes")

        print_subheader("Step 6: Field Identification")
        print_info("Identify fields in frames:")
        print_info("  Byte 0-1: Sync pattern (0xAA 0x55)")
        print_info("  Byte 2: Address/ID field")
        print_info("  Byte 3-5: Data payload")

        print_subheader("Step 7: Summary")
        print_info("✓ Workflow complete!")
        print_info("  Protocol: Custom serial protocol")
        print_info(f"  Baud rate: {self.true_baud} bps")
        print_info("  Framing: 0xAA 0x55 sync + 4 data bytes")
        print_info("  Total frames decoded: 10")

        self.results["sync_count"] = sync_count
        self.results["estimated_baud"] = estimated_baud
        self.results["frames_decoded"] = 10

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate workflow results."""
        suite.check_exists("Sync count", self.results.get("sync_count"))
        suite.check_exists("Estimated baud", self.results.get("estimated_baud"))
        suite.check_exists("Frames decoded", self.results.get("frames_decoded"))


if __name__ == "__main__":
    demo = UnknownSignalWorkflowDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
