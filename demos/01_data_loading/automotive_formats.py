"""Automotive Format Loading

Demonstrates loading automotive-specific file formats:
- CAN bus capture files
- LIN bus captures
- Automotive diagnostic data
- Frame extraction and decoding

Related to automotive reverse engineering workflows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    format_table,
)


class AutomotiveFormatsDemo(BaseDemo):
    """Demonstrate loading automotive-specific file formats."""

    def __init__(self) -> None:
        """Initialize automotive formats demonstration."""
        super().__init__(
            name="automotive_formats",
            description="Load and parse automotive bus capture formats",
            capabilities=[
                "Automotive format detection",
                "CAN bus frame extraction",
                "LIN bus parsing",
                "Frame timestamp handling",
            ],
            ieee_standards=[],
            related_demos=[
                "02_logic_analyzers.py",
                "08_network_formats.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic automotive capture data."""
        self.info("Creating synthetic automotive captures...")

        # CAN bus capture
        can_data = self._create_can_synthetic()
        self.info("  ✓ CAN bus capture (500 kbps, 100 frames)")

        # LIN bus capture
        lin_data = self._create_lin_synthetic()
        self.info("  ✓ LIN bus capture (19.2 kbps, 50 frames)")

        return {
            "can": can_data,
            "lin": lin_data,
        }

    def _create_can_synthetic(self) -> dict[str, Any]:
        """Create synthetic CAN bus capture data."""
        num_frames = 100
        frames = []

        # Generate realistic CAN frames
        for i in range(num_frames):
            timestamp = i * 0.001  # 1 ms spacing
            can_id = 0x100 + (i % 8)  # Rotating IDs
            data_length = 8
            data = np.random.randint(0, 256, size=data_length, dtype=np.uint8)

            frames.append(
                {
                    "timestamp": timestamp,
                    "id": can_id,
                    "dlc": data_length,
                    "data": data.tolist(),
                    "extended": False,
                }
            )

        return {
            "bus_type": "CAN",
            "bitrate": 500000,
            "frames": frames,
            "total_duration": frames[-1]["timestamp"],
        }

    def _create_lin_synthetic(self) -> dict[str, Any]:
        """Create synthetic LIN bus capture data."""
        num_frames = 50
        frames = []

        # Generate realistic LIN frames
        for i in range(num_frames):
            timestamp = i * 0.005  # 5 ms spacing
            frame_id = 0x10 + (i % 4)  # Rotating IDs
            data_length = 4
            data = np.random.randint(0, 256, size=data_length, dtype=np.uint8)
            checksum = sum(data) & 0xFF

            frames.append(
                {
                    "timestamp": timestamp,
                    "id": frame_id,
                    "data": data.tolist(),
                    "checksum": checksum,
                }
            )

        return {
            "bus_type": "LIN",
            "bitrate": 19200,
            "frames": frames,
            "total_duration": frames[-1]["timestamp"],
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the automotive formats demonstration."""
        results = {}

        self.subsection("Automotive Bus Formats")
        self.info("Common automotive bus protocols:")
        self.info("  • CAN (Controller Area Network): 125 kbps - 1 Mbps")
        self.info("  • CAN-FD (Flexible Data-rate): up to 5 Mbps data phase")
        self.info("  • LIN (Local Interconnect Network): 2.4 - 20 kbps")
        self.info("  • FlexRay: 10 Mbps dual channel")
        self.info("")

        # CAN bus analysis
        self.subsection("CAN Bus Capture Analysis")
        results["can"] = self._analyze_can_capture(data["can"])

        # LIN bus analysis
        self.subsection("LIN Bus Capture Analysis")
        results["lin"] = self._analyze_lin_capture(data["lin"])

        # Comparison
        self.subsection("Protocol Comparison")
        self._display_protocol_comparison(data)

        # Best practices
        self.subsection("Automotive Capture Best Practices")
        self._show_best_practices()

        return results

    def _analyze_can_capture(self, can_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze CAN bus capture."""
        frames = can_data["frames"]

        self.result("Bus Type", can_data["bus_type"])
        self.result("Bitrate", f"{can_data['bitrate'] / 1000:.0f}", "kbps")
        self.result("Total Frames", len(frames))
        self.result("Capture Duration", f"{can_data['total_duration']:.3f}", "s")

        # Calculate frame statistics
        frame_ids = [f["id"] for f in frames]
        unique_ids = set(frame_ids)
        self.result("Unique IDs", len(unique_ids))

        # Show sample frames
        self.info("\nSample CAN Frames:")
        sample_rows = []
        for frame in frames[:5]:
            data_hex = " ".join([f"{b:02X}" for b in frame["data"]])
            sample_rows.append(
                [
                    f"{frame['timestamp']:.3f}",
                    f"0x{frame['id']:03X}",
                    frame["dlc"],
                    data_hex,
                ]
            )

        headers = ["Time (s)", "ID", "DLC", "Data"]
        print(format_table(sample_rows, headers=headers))

        return {
            "num_frames": len(frames),
            "unique_ids": len(unique_ids),
            "bitrate": can_data["bitrate"],
            "duration": can_data["total_duration"],
        }

    def _analyze_lin_capture(self, lin_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze LIN bus capture."""
        frames = lin_data["frames"]

        self.result("Bus Type", lin_data["bus_type"])
        self.result("Bitrate", f"{lin_data['bitrate'] / 1000:.1f}", "kbps")
        self.result("Total Frames", len(frames))
        self.result("Capture Duration", f"{lin_data['total_duration']:.3f}", "s")

        # Calculate frame statistics
        frame_ids = [f["id"] for f in frames]
        unique_ids = set(frame_ids)
        self.result("Unique IDs", len(unique_ids))

        # Show sample frames
        self.info("\nSample LIN Frames:")
        sample_rows = []
        for frame in frames[:5]:
            data_hex = " ".join([f"{b:02X}" for b in frame["data"]])
            sample_rows.append(
                [
                    f"{frame['timestamp']:.3f}",
                    f"0x{frame['id']:02X}",
                    data_hex,
                    f"0x{frame['checksum']:02X}",
                ]
            )

        headers = ["Time (s)", "ID", "Data", "Checksum"]
        print(format_table(sample_rows, headers=headers))

        return {
            "num_frames": len(frames),
            "unique_ids": len(unique_ids),
            "bitrate": lin_data["bitrate"],
            "duration": lin_data["total_duration"],
        }

    def _display_protocol_comparison(self, data: dict[str, Any]) -> None:
        """Display comparison of automotive protocols."""
        comparison = [
            [
                "CAN",
                "125k-1M bps",
                "11/29-bit ID",
                "0-8 bytes",
                "Multi-master",
                "Body, powertrain",
            ],
            ["CAN-FD", "5M bps", "11/29-bit ID", "0-64 bytes", "Multi-master", "High-speed ECUs"],
            ["LIN", "2.4-20k bps", "6-bit ID", "1-8 bytes", "Master-slave", "Body, comfort"],
            [
                "FlexRay",
                "10M bps",
                "Static+dynamic",
                "0-254 bytes",
                "Time-triggered",
                "Safety-critical",
            ],
        ]

        headers = ["Protocol", "Speed", "Addressing", "Payload", "Topology", "Use Case"]
        print(format_table(comparison, headers=headers))
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for automotive captures."""
        self.info("""
Automotive Capture Best Practices:

1. TIMING ACCURACY
   - Use hardware-timestamped captures (CAN adapters with HW timestamps)
   - Verify timestamp resolution (microsecond or better)
   - Account for bus arbitration delays

2. FILTERING
   - Filter by ID ranges for specific ECUs
   - Exclude periodic heartbeat messages if not needed
   - Capture error frames for bus health analysis

3. DATA INTERPRETATION
   - Use DBC (CAN database) files for signal decoding
   - Document bit positions and scaling factors
   - Handle endianness (CAN typically big-endian)

4. CAPTURE TOOLS
   - Vector CANalyzer/CANoe: Industry standard
   - Kvaser/PEAK adapters: Common hardware
   - SocketCAN (Linux): Open-source alternative
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate automotive format loading results."""
        suite = ValidationSuite()

        # Validate CAN capture
        if "can" in results:
            can = results["can"]
            suite.check_equal(can["num_frames"], 100, "CAN frame count")
            suite.check_equal(can["bitrate"], 500000, "CAN bitrate")
            suite.check_true(can["unique_ids"] > 0, "CAN unique IDs")

        # Validate LIN capture
        if "lin" in results:
            lin = results["lin"]
            suite.check_equal(lin["num_frames"], 50, "LIN frame count")
            suite.check_equal(lin["bitrate"], 19200, "LIN bitrate")
            suite.check_true(lin["unique_ids"] > 0, "LIN unique IDs")

        if suite.all_passed():
            self.success("All automotive format validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Use DBC files for signal extraction")
            self.info("  - Explore automotive protocol decoding demos")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = AutomotiveFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
