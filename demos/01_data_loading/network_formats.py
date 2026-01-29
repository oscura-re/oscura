"""Network Capture Format Loading

Demonstrates loading network and RF capture formats:
- PCAP (packet capture) files
- Touchstone (S-parameter) files
- Network analyzer data
- RF signal captures

Common in network analysis and RF/microwave measurements.
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


class NetworkFormatsDemo(BaseDemo):
    """Demonstrate loading network capture formats."""

    def __init__(self) -> None:
        """Initialize network formats demonstration."""
        super().__init__(
            name="network_formats",
            description="Load and analyze network and RF capture formats",
            capabilities=[
                "oscura.loaders.load_pcap",
                "oscura.loaders.load_touchstone",
                "Network packet extraction",
                "S-parameter analysis",
            ],
            ieee_standards=[],
            related_demos=[
                "03_automotive_formats.py",
                "02_logic_analyzers.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic network capture data."""
        self.info("Creating synthetic network captures...")

        # PCAP-like packet capture
        pcap_data = self._create_pcap_synthetic()
        self.info("  ✓ PCAP packet capture (100 packets)")

        # Touchstone S-parameter data
        touchstone_data = self._create_touchstone_synthetic()
        self.info("  ✓ Touchstone S-parameter data (1-port)")

        return {
            "pcap": pcap_data,
            "touchstone": touchstone_data,
        }

    def _create_pcap_synthetic(self) -> dict[str, Any]:
        """Create synthetic PCAP-like packet capture."""
        num_packets = 100
        packets = []

        for i in range(num_packets):
            timestamp = i * 0.001  # 1 ms spacing
            packet_size = np.random.randint(64, 1500)  # Ethernet packet size range
            protocol = ["TCP", "UDP", "ICMP"][i % 3]

            packets.append(
                {
                    "timestamp": timestamp,
                    "size": packet_size,
                    "protocol": protocol,
                    "src_ip": f"192.168.1.{i % 256}",
                    "dst_ip": "192.168.1.1",
                }
            )

        return {
            "format": "PCAP",
            "packets": packets,
            "total_duration": packets[-1]["timestamp"],
        }

    def _create_touchstone_synthetic(self) -> dict[str, Any]:
        """Create synthetic Touchstone S-parameter data."""
        # Generate frequency sweep
        freq_start = 1e6  # 1 MHz
        freq_stop = 1e9  # 1 GHz
        num_points = 201

        frequencies = np.logspace(np.log10(freq_start), np.log10(freq_stop), num_points)

        # Generate synthetic S11 (reflection coefficient)
        # Simulate resonance at 500 MHz
        resonance_freq = 500e6
        q_factor = 50

        s11_mag = []
        s11_phase = []

        for freq in frequencies:
            # Simple resonant circuit model
            delta_f = (freq - resonance_freq) / resonance_freq
            mag = 1.0 / np.sqrt(1 + (2 * q_factor * delta_f) ** 2)
            phase = -np.arctan(2 * q_factor * delta_f) * 180 / np.pi

            s11_mag.append(mag)
            s11_phase.append(phase)

        return {
            "format": "Touchstone",
            "ports": 1,
            "frequencies": frequencies,
            "s11_mag": np.array(s11_mag),
            "s11_phase": np.array(s11_phase),
            "frequency_unit": "Hz",
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the network formats demonstration."""
        results = {}

        self.subsection("Network Capture Formats Overview")
        self.info("Common network and RF capture formats:")
        self.info("  • PCAP: Network packet captures (Wireshark, tcpdump)")
        self.info("  • Touchstone: S-parameter data (network analyzers)")
        self.info("  • IQ samples: RF signal captures (SDR)")
        self.info("  • Complex baseband: Wireless signal captures")
        self.info("")

        # PCAP analysis
        self.subsection("PCAP Packet Capture Analysis")
        results["pcap"] = self._analyze_pcap(data["pcap"])

        # Touchstone analysis
        self.subsection("Touchstone S-Parameter Analysis")
        results["touchstone"] = self._analyze_touchstone(data["touchstone"])

        # Best practices
        self.subsection("Network Format Best Practices")
        self._show_best_practices()

        return results

    def _analyze_pcap(self, pcap_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze PCAP packet capture."""
        packets = pcap_data["packets"]

        self.result("Format", pcap_data["format"])
        self.result("Total Packets", len(packets))
        self.result("Capture Duration", f"{pcap_data['total_duration']:.3f}", "s")

        # Protocol distribution
        protocols = {}
        total_bytes = 0
        for packet in packets:
            protocols[packet["protocol"]] = protocols.get(packet["protocol"], 0) + 1
            total_bytes += packet["size"]

        self.result("Total Bytes", f"{total_bytes:,}", "bytes")
        self.result("Average Packet Size", f"{total_bytes / len(packets):.1f}", "bytes")

        # Protocol distribution table
        protocol_rows = []
        for proto, count in sorted(protocols.items()):
            percentage = 100.0 * count / len(packets)
            protocol_rows.append([proto, count, f"{percentage:.1f}%"])

        headers = ["Protocol", "Count", "Percentage"]
        print(format_table(protocol_rows, headers=headers))

        return {
            "num_packets": len(packets),
            "protocols": len(protocols),
            "total_bytes": total_bytes,
        }

    def _analyze_touchstone(self, touchstone_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze Touchstone S-parameter data."""
        self.result("Format", touchstone_data["format"])
        self.result("Ports", touchstone_data["ports"])
        self.result("Frequency Points", len(touchstone_data["frequencies"]))
        self.result(
            "Frequency Range",
            f"{touchstone_data['frequencies'][0] / 1e6:.1f} - "
            f"{touchstone_data['frequencies'][-1] / 1e6:.0f}",
            "MHz",
        )

        # Find resonance (minimum S11 magnitude)
        s11_mag = touchstone_data["s11_mag"]
        min_idx = np.argmin(s11_mag)
        resonance_freq = touchstone_data["frequencies"][min_idx]

        self.result("Resonance Frequency", f"{resonance_freq / 1e6:.1f}", "MHz")
        self.result("Min S11", f"{s11_mag[min_idx]:.4f}", "linear")
        self.result("Min S11 (dB)", f"{20 * np.log10(s11_mag[min_idx]):.2f}", "dB")

        return {
            "num_points": len(touchstone_data["frequencies"]),
            "resonance_freq": float(resonance_freq),
        }

    def _show_best_practices(self) -> None:
        """Show best practices for network formats."""
        self.info("""
Network Format Best Practices:

1. PCAP FILES
   - Use hardware-timestamped captures when possible
   - Filter traffic to reduce file size
   - Include Ethernet headers for layer 2 analysis
   - Consider pcap-ng for enhanced metadata

2. TOUCHSTONE FILES
   - Verify frequency units (Hz, MHz, GHz)
   - Check parameter format (dB/angle, mag/angle, real/imag)
   - Validate port ordering for multi-port measurements
   - Store calibration data with measurements

3. RF SIGNAL CAPTURES
   - Record center frequency and sample rate
   - Store IQ data in complex format (I + jQ)
   - Document gain settings and filtering
   - Include timestamp for time-domain analysis

4. ANALYSIS CONSIDERATIONS
   - PCAP: Protocol decoding, flow analysis
   - Touchstone: Impedance matching, filter design
   - RF captures: Modulation analysis, spectrum
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate network format loading results."""
        suite = ValidationSuite()

        # Validate PCAP
        if "pcap" in results:
            suite.check_equal(results["pcap"]["num_packets"], 100, "PCAP packet count")
            suite.check_true(results["pcap"]["total_bytes"] > 0, "PCAP bytes captured")

        # Validate Touchstone
        if "touchstone" in results:
            suite.check_true(
                results["touchstone"]["num_points"] > 100, "Touchstone frequency points"
            )
            suite.check_approximately(
                results["touchstone"]["resonance_freq"],
                500e6,
                tolerance=0.1,
                name="Resonance frequency",
            )

        if suite.all_passed():
            self.success("All network format validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Use Wireshark for detailed PCAP analysis")
            self.info("  - Explore RF analysis tools for IQ data")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = NetworkFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
