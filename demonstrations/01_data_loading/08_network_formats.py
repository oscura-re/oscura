"""Network and RF File Format Loading

Demonstrates loading network-related data formats:
- Touchstone .sNp files (S-parameter data for signal integrity)
- PCAP/PCAPNG packet captures (network traffic analysis)
- Integration with protocol decoders for PCAP analysis
- S-parameter analysis (insertion loss, return loss, etc.)

IEEE Standards: IEEE 370-2020 (Electrical Characterization of PCBs)
Related Demos:
- 01_data_loading/04_scientific_formats.py
- 05_domain_specific/02_signal_integrity.py
- 03_protocol_decoding/01_serial_comprehensive.py

This demonstration shows:
1. How to generate and load Touchstone S-parameter files
2. How to perform S-parameter analysis (insertion/return loss)
3. How to generate and load PCAP packet captures
4. How to integrate PCAP data with protocol decoders
5. Frequency domain analysis for signal integrity
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    format_table,
    validate_approximately,
    validate_exists,
)
from oscura.analyzers.signal_integrity import insertion_loss, return_loss
from oscura.loaders.pcap import load_pcap
from oscura.loaders.touchstone import load_touchstone


class NetworkFormatsDemo(BaseDemo):
    """Demonstrate loading network and RF file formats with synthetic data."""

    def __init__(self) -> None:
        """Initialize network formats demonstration."""
        super().__init__(
            name="network_formats",
            description="Load and analyze Touchstone and PCAP network formats",
            capabilities=[
                "oscura.loaders.load_touchstone",
                "oscura.loaders.load_pcap",
                "oscura.analyzers.signal_integrity.insertion_loss",
                "oscura.analyzers.signal_integrity.return_loss",
                "S-parameter frequency domain analysis",
                "PCAP packet inspection and filtering",
            ],
            ieee_standards=["IEEE 370-2020"],
            related_demos=[
                "01_data_loading/04_scientific_formats.py",
                "05_domain_specific/02_signal_integrity.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic Touchstone and PCAP test data."""
        self.info("Creating synthetic network data files...")

        # Generate Touchstone .s2p file
        touchstone_path = self._generate_touchstone_file()
        self.info(f"  ✓ Generated {touchstone_path}")

        # Generate PCAP network capture
        pcap_path = self._generate_pcap_file()
        self.info(f"  ✓ Generated {pcap_path}")

        return {
            "touchstone_path": touchstone_path,
            "pcap_path": pcap_path,
        }

    def _generate_touchstone_file(self) -> Path:
        """Generate synthetic .s2p Touchstone file for a lossy cable.

        Creates a 2-port S-parameter file with realistic cable loss characteristics.

        Returns:
            Path to generated .s2p file
        """
        output_dir = self.get_output_dir()
        touchstone_path = output_dir / "cable_10cm.s2p"

        # Frequency range: 100 MHz to 10 GHz (typical for high-speed digital)
        frequencies = np.logspace(8, 10, 50)  # 100 MHz to 10 GHz, 50 points

        with open(touchstone_path, "w") as f:
            # Write Touchstone header
            f.write("! Synthetic 10cm cable S-parameters\n")
            f.write("! Generated for Oscura demonstration\n")
            f.write("! Length: 10 cm, characteristic impedance: 50 Ohm\n")
            f.write("! Loss: 0.05 dB/cm @ 1 GHz\n")
            f.write("# Hz S MA R 50\n")
            f.write("!\n")

            # Generate S-parameters for each frequency
            for freq in frequencies:
                # Realistic cable model parameters
                # Loss increases with sqrt(frequency) (skin effect)
                loss_per_cm_db = 0.05 * np.sqrt(freq / 1e9)  # dB/cm at freq
                cable_length_cm = 10.0

                # S21 (insertion loss): signal transmission
                s21_db = -loss_per_cm_db * cable_length_cm
                s21_mag = 10 ** (s21_db / 20)
                # Phase shift from electrical length
                wavelength = 3e8 / freq  # c / f
                velocity_factor = 0.66  # Typical for coax
                phase_shift = -360 * cable_length_cm * 0.01 / (wavelength * velocity_factor)

                # S11 (return loss): reflection at input
                # Better return loss at lower frequencies
                s11_db = -15 - 5 * np.log10(freq / 1e8)  # -15 dB @ 100 MHz, worse at HF
                s11_mag = 10 ** (s11_db / 20)
                s11_phase = 45.0  # Some phase variation

                # S22 (return loss at output): similar to S11
                s22_mag = s11_mag * 0.95  # Slightly different
                s22_phase = -45.0

                # S12 (reverse transmission): same as S21 for passive, reciprocal device
                s12_mag = s21_mag
                s12_phase = phase_shift

                # Write frequency and S-parameters in MA (magnitude/angle) format
                f.write(
                    f"{freq:.6e} {s11_mag:.6f} {s11_phase:.2f} "
                    f"{s21_mag:.6f} {phase_shift:.2f} "
                    f"{s12_mag:.6f} {s12_phase:.2f} "
                    f"{s22_mag:.6f} {s22_phase:.2f}\n"
                )

        return touchstone_path

    def _generate_pcap_file(self) -> Path:
        """Generate synthetic PCAP file with network packets.

        Creates a PCAP file with TCP, UDP, and ICMP packets.

        Returns:
            Path to generated .pcap file
        """
        output_dir = self.get_output_dir()
        pcap_path = output_dir / "network_capture.pcap"

        # PCAP file format constants
        PCAP_MAGIC = 0xA1B2C3D4
        VERSION_MAJOR = 2
        VERSION_MINOR = 4
        THISZONE = 0
        SIGFIGS = 0
        SNAPLEN = 65535
        NETWORK = 1  # Ethernet

        with open(pcap_path, "wb") as f:
            # Write PCAP global header (24 bytes)
            f.write(
                struct.pack(
                    "<IHHiIII",
                    PCAP_MAGIC,
                    VERSION_MAJOR,
                    VERSION_MINOR,
                    THISZONE,
                    SIGFIGS,
                    SNAPLEN,
                    NETWORK,
                )
            )

            # Generate synthetic packets
            base_time = 1700000000.0  # Timestamp base

            # Packet 1: TCP SYN (connection establishment)
            tcp_syn = self._create_tcp_packet(
                src_ip="192.168.1.100",
                dst_ip="192.168.1.1",
                src_port=54321,
                dst_port=80,
                flags=0x02,  # SYN
                payload=b"",
            )
            self._write_pcap_packet(f, base_time + 0.0, tcp_syn)

            # Packet 2: TCP SYN-ACK
            tcp_syn_ack = self._create_tcp_packet(
                src_ip="192.168.1.1",
                dst_ip="192.168.1.100",
                src_port=80,
                dst_port=54321,
                flags=0x12,  # SYN+ACK
                payload=b"",
            )
            self._write_pcap_packet(f, base_time + 0.001, tcp_syn_ack)

            # Packet 3: TCP ACK
            tcp_ack = self._create_tcp_packet(
                src_ip="192.168.1.100",
                dst_ip="192.168.1.1",
                src_port=54321,
                dst_port=80,
                flags=0x10,  # ACK
                payload=b"",
            )
            self._write_pcap_packet(f, base_time + 0.002, tcp_ack)

            # Packet 4: HTTP GET request
            http_request = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
            tcp_data = self._create_tcp_packet(
                src_ip="192.168.1.100",
                dst_ip="192.168.1.1",
                src_port=54321,
                dst_port=80,
                flags=0x18,  # PSH+ACK
                payload=http_request,
            )
            self._write_pcap_packet(f, base_time + 0.003, tcp_data)

            # Packet 5: UDP DNS query
            udp_dns = self._create_udp_packet(
                src_ip="192.168.1.100",
                dst_ip="8.8.8.8",
                src_port=53124,
                dst_port=53,
                payload=b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00",
            )
            self._write_pcap_packet(f, base_time + 0.1, udp_dns)

            # Packet 6: ICMP Echo Request (ping)
            icmp_ping = self._create_icmp_packet(
                src_ip="192.168.1.100",
                dst_ip="8.8.8.8",
                icmp_type=8,  # Echo request
                icmp_code=0,
                payload=b"Oscura test ping data",
            )
            self._write_pcap_packet(f, base_time + 0.2, icmp_ping)

        return pcap_path

    def _create_tcp_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        flags: int,
        payload: bytes,
    ) -> bytes:
        """Create a synthetic TCP packet (Ethernet + IP + TCP)."""
        # Ethernet header (14 bytes)
        eth_dst = b"\xff\xff\xff\xff\xff\xff"
        eth_src = b"\x00\x11\x22\x33\x44\x55"
        eth_type = b"\x08\x00"  # IPv4
        ethernet = eth_dst + eth_src + eth_type

        # IP header (20 bytes, simplified)
        ip_src = bytes(int(x) for x in src_ip.split("."))
        ip_dst = bytes(int(x) for x in dst_ip.split("."))
        ip_header = (
            b"\x45\x00"  # Version=4, IHL=5, ToS=0
            + struct.pack(">H", 20 + 20 + len(payload))  # Total length
            + b"\x00\x01\x00\x00"  # ID, flags, fragment offset
            + b"\x40\x06"  # TTL=64, Protocol=TCP(6)
            + b"\x00\x00"  # Checksum (0 for now)
            + ip_src
            + ip_dst
        )

        # TCP header (20 bytes, simplified)
        tcp_header = (
            struct.pack(">H", src_port)
            + struct.pack(">H", dst_port)
            + b"\x00\x00\x00\x01"  # Seq number
            + b"\x00\x00\x00\x00"  # Ack number
            + struct.pack(">B", 0x50)  # Data offset (5 * 4 = 20 bytes)
            + struct.pack(">B", flags)  # Flags
            + b"\x20\x00"  # Window size
            + b"\x00\x00"  # Checksum
            + b"\x00\x00"  # Urgent pointer
        )

        return ethernet + ip_header + tcp_header + payload

    def _create_udp_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        payload: bytes,
    ) -> bytes:
        """Create a synthetic UDP packet (Ethernet + IP + UDP)."""
        # Ethernet header
        eth_dst = b"\xff\xff\xff\xff\xff\xff"
        eth_src = b"\x00\x11\x22\x33\x44\x55"
        eth_type = b"\x08\x00"  # IPv4
        ethernet = eth_dst + eth_src + eth_type

        # IP header
        ip_src = bytes(int(x) for x in src_ip.split("."))
        ip_dst = bytes(int(x) for x in dst_ip.split("."))
        ip_header = (
            b"\x45\x00"
            + struct.pack(">H", 20 + 8 + len(payload))  # Total length
            + b"\x00\x01\x00\x00"
            + b"\x40\x11"  # TTL=64, Protocol=UDP(17)
            + b"\x00\x00"  # Checksum
            + ip_src
            + ip_dst
        )

        # UDP header (8 bytes)
        udp_header = (
            struct.pack(">H", src_port)
            + struct.pack(">H", dst_port)
            + struct.pack(">H", 8 + len(payload))  # Length
            + b"\x00\x00"  # Checksum
        )

        return ethernet + ip_header + udp_header + payload

    def _create_icmp_packet(
        self,
        src_ip: str,
        dst_ip: str,
        icmp_type: int,
        icmp_code: int,
        payload: bytes,
    ) -> bytes:
        """Create a synthetic ICMP packet (Ethernet + IP + ICMP)."""
        # Ethernet header
        eth_dst = b"\xff\xff\xff\xff\xff\xff"
        eth_src = b"\x00\x11\x22\x33\x44\x55"
        eth_type = b"\x08\x00"
        ethernet = eth_dst + eth_src + eth_type

        # IP header
        ip_src = bytes(int(x) for x in src_ip.split("."))
        ip_dst = bytes(int(x) for x in dst_ip.split("."))
        ip_header = (
            b"\x45\x00"
            + struct.pack(">H", 20 + 8 + len(payload))  # Total length
            + b"\x00\x01\x00\x00"
            + b"\x40\x01"  # TTL=64, Protocol=ICMP(1)
            + b"\x00\x00"
            + ip_src
            + ip_dst
        )

        # ICMP header (8 bytes)
        icmp_header = (
            struct.pack(">B", icmp_type)
            + struct.pack(">B", icmp_code)
            + b"\x00\x00"  # Checksum
            + b"\x00\x01\x00\x01"  # ID and sequence
        )

        return ethernet + ip_header + icmp_header + payload

    def _write_pcap_packet(self, f: io.BufferedWriter, timestamp: float, data: bytes) -> None:
        """Write a packet to PCAP file with timestamp and data."""
        ts_sec = int(timestamp)
        ts_usec = int((timestamp - ts_sec) * 1_000_000)
        incl_len = len(data)
        orig_len = len(data)

        # Packet header (16 bytes)
        f.write(struct.pack("<IIII", ts_sec, ts_usec, incl_len, orig_len))

        # Packet data
        f.write(data)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the network formats loading demonstration."""
        results: dict[str, Any] = {}

        # Part 1: Touchstone S-Parameter Loading
        self.section("Part 1: Touchstone S-Parameter Loading")
        self.info("Touchstone files store S-parameters for RF and signal integrity analysis.")
        self.info("Formats: .s1p (1-port), .s2p (2-port), up to .s8p (8-port)")
        self.info("")

        self.subsection("Loading 2-Port S-Parameters (Cable)")
        touchstone_path = data["touchstone_path"]
        s_params = load_touchstone(touchstone_path)

        self.result("File", str(touchstone_path))
        self.result("Number of Ports", s_params.n_ports)
        self.result("Frequency Points", len(s_params.frequencies))
        self.result(
            "Frequency Range", f"{s_params.frequencies[0]:.2e} to {s_params.frequencies[-1]:.2e} Hz"
        )
        self.result("Reference Impedance", f"{s_params.z0}", "Ω")
        self.result("Format", s_params.format.upper())
        self.info("")

        # Part 2: S-Parameter Analysis
        self.section("Part 2: S-Parameter Analysis")
        self.info("Analyzing signal integrity metrics from S-parameters...")
        self.info("")

        # Return loss at multiple frequencies
        self.subsection("Return Loss Analysis (S11)")
        self.info("Return loss indicates how well the input port is matched to Z0.")
        self.info("Higher return loss (dB) = better match, less reflection")
        self.info("")

        test_freqs = [1e8, 1e9, 5e9, 10e9]  # 100 MHz, 1 GHz, 5 GHz, 10 GHz
        rl_results = []

        for freq in test_freqs:
            if freq <= s_params.frequencies[-1]:
                rl = return_loss(s_params, frequency=freq, port=1)
                rl_results.append([f"{freq:.2e} Hz", f"{rl:.2f} dB"])
                self.result(f"  Return Loss @ {freq:.2e} Hz", f"{rl:.2f}", "dB")

        # Insertion loss analysis
        self.info("")
        self.subsection("Insertion Loss Analysis (S21)")
        self.info("Insertion loss shows signal attenuation through the cable.")
        self.info("Lower insertion loss (dB) = less attenuation")
        self.info("")

        il_results = []
        for freq in test_freqs:
            if freq <= s_params.frequencies[-1]:
                il = insertion_loss(s_params, frequency=freq, input_port=1, output_port=2)
                il_results.append([f"{freq:.2e} Hz", f"{il:.2f} dB"])
                self.result(f"  Insertion Loss @ {freq:.2e} Hz", f"{il:.2f}", "dB")

        # Full frequency sweep
        self.info("")
        self.subsection("Frequency Sweep Analysis")
        rl_sweep = return_loss(s_params, frequency=None, port=1)
        il_sweep = insertion_loss(s_params, frequency=None, input_port=1, output_port=2)

        # Find worst case points
        worst_rl_idx = np.argmin(rl_sweep)
        worst_il_idx = np.argmax(il_sweep)

        self.result(
            "Best Return Loss",
            f"{np.max(rl_sweep):.2f} dB @ {s_params.frequencies[np.argmax(rl_sweep)]:.2e} Hz",
        )
        self.result(
            "Worst Return Loss",
            f"{rl_sweep[worst_rl_idx]:.2f} dB @ {s_params.frequencies[worst_rl_idx]:.2e} Hz",
        )
        self.result(
            "Best Insertion Loss",
            f"{np.min(il_sweep):.2f} dB @ {s_params.frequencies[np.argmin(il_sweep)]:.2e} Hz",
        )
        self.result(
            "Worst Insertion Loss",
            f"{il_sweep[worst_il_idx]:.2f} dB @ {s_params.frequencies[worst_il_idx]:.2e} Hz",
        )
        self.info("")

        results["s_params"] = {
            "n_ports": s_params.n_ports,
            "n_frequencies": len(s_params.frequencies),
            "freq_min": float(s_params.frequencies[0]),
            "freq_max": float(s_params.frequencies[-1]),
            "return_loss_1ghz": float(return_loss(s_params, frequency=1e9, port=1)),
            "insertion_loss_1ghz": float(
                insertion_loss(s_params, frequency=1e9, input_port=1, output_port=2)
            ),
            "worst_return_loss": float(rl_sweep[worst_rl_idx]),
            "worst_insertion_loss": float(il_sweep[worst_il_idx]),
        }

        # Part 3: PCAP Network Capture Loading
        self.section("Part 3: PCAP Network Capture Loading")
        self.info("PCAP files store captured network packets with timestamps.")
        self.info("Used for network analysis, protocol debugging, and security research.")
        self.info("")

        self.subsection("Loading PCAP File")
        pcap_path = data["pcap_path"]
        packets = load_pcap(pcap_path)

        self.result("File", str(pcap_path))
        self.result("Total Packets", len(packets))
        self.result("Link Type", packets.link_type)
        self.result("Snaplen", packets.snaplen, "bytes")
        self.info("")

        # Part 4: PCAP Packet Analysis
        self.section("Part 4: PCAP Packet Analysis")
        self.info("Analyzing captured network packets...")
        self.info("")

        # Display packet summary
        self.subsection("Packet Summary")
        packet_table = []
        for i, pkt in enumerate(packets[:6], 1):  # Show first 6 packets
            protocol = pkt.protocol
            size = len(pkt.data)
            timestamp_rel = pkt.timestamp - packets[0].timestamp

            # Extract additional info from annotations
            info = ""
            if "src_ip" in pkt.annotations:
                src_ip = pkt.annotations["src_ip"]
                dst_ip = pkt.annotations["dst_ip"]
                info = f"{src_ip} → {dst_ip}"

            if "src_port" in pkt.annotations:
                src_port = pkt.annotations["src_port"]
                dst_port = pkt.annotations["dst_port"]
                info += f":{src_port} → :{dst_port}"

            packet_table.append([i, f"{timestamp_rel:.6f}", protocol, size, info])

        headers = ["#", "Time (s)", "Protocol", "Size (B)", "Info"]
        self.info(format_table(packet_table, headers))
        self.info("")

        # Protocol statistics
        self.subsection("Protocol Statistics")
        protocol_counts: dict[str, int] = {}
        total_bytes = 0

        for pkt in packets:
            protocol = pkt.protocol
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
            total_bytes += len(pkt.data)

        for protocol, count in sorted(protocol_counts.items()):
            self.result(f"  {protocol} packets", count)

        self.result("Total bytes captured", total_bytes, "bytes")
        self.info("")

        # Packet filtering
        self.subsection("Packet Filtering")
        self.info("PCAP files can be filtered by protocol, size, or other criteria.")
        self.info("")

        tcp_packets = packets.filter(protocol="TCP")
        udp_packets = packets.filter(protocol="UDP")
        large_packets = packets.filter(min_size=100)

        self.result("TCP packets", len(tcp_packets))
        self.result("UDP packets", len(udp_packets))
        self.result("Large packets (≥100 bytes)", len(large_packets))
        self.info("")

        # Part 5: PCAP Protocol Decoder Integration
        self.section("Part 5: PCAP Protocol Decoder Integration")
        self.info("PCAP packets can be integrated with Oscura protocol decoders.")
        self.info("Packet annotations provide protocol-specific information:")
        self.info("")

        self.subsection("TCP Packet Annotations")
        tcp_example = tcp_packets[0] if tcp_packets else None
        if tcp_example:
            self.result("  Protocol", tcp_example.protocol)
            self.result("  Timestamp", f"{tcp_example.timestamp:.6f}", "s")
            self.result("  Size", len(tcp_example.data), "bytes")

            if tcp_example.annotations:
                self.info("  Annotations:")
                for key, value in sorted(tcp_example.annotations.items()):
                    if key == "tcp_flags":
                        flag_str = self._decode_tcp_flags(value)
                        self.result(f"    {key}", f"0x{value:02x} ({flag_str})")
                    else:
                        self.result(f"    {key}", str(value))
        self.info("")

        results["pcap"] = {
            "total_packets": len(packets),
            "tcp_packets": len(tcp_packets),
            "udp_packets": len(udp_packets),
            "total_bytes": total_bytes,
            "protocols": list(protocol_counts.keys()),
        }

        return results

    def _decode_tcp_flags(self, flags: int) -> str:
        """Decode TCP flags to human-readable string."""
        flag_names = []
        if flags & 0x01:
            flag_names.append("FIN")
        if flags & 0x02:
            flag_names.append("SYN")
        if flags & 0x04:
            flag_names.append("RST")
        if flags & 0x08:
            flag_names.append("PSH")
        if flags & 0x10:
            flag_names.append("ACK")
        if flags & 0x20:
            flag_names.append("URG")
        return "|".join(flag_names) if flag_names else "NONE"

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate network formats loading results."""
        self.info("Validating network data loading and analysis...")

        all_valid = True

        # Validate S-parameter results
        if "s_params" in results:
            s_params = results["s_params"]

            # Check basic S-parameter properties
            if not validate_exists(s_params.get("n_ports"), "S-parameter port count"):
                all_valid = False

            if s_params.get("n_ports") != 2:
                self.error(f"Expected 2-port S-parameters, got {s_params.get('n_ports')}")
                all_valid = False
            else:
                self.success("S-parameter port count correct (2-port)")

            if not validate_exists(s_params.get("n_frequencies"), "Frequency points"):
                all_valid = False

            # Validate frequency range (100 MHz to 10 GHz)
            if not validate_approximately(
                s_params.get("freq_min", 0),
                1e8,
                tolerance=0.01,
                name="Minimum frequency",
            ):
                all_valid = False
            else:
                self.success("Minimum frequency correct (100 MHz)")

            if not validate_approximately(
                s_params.get("freq_max", 0),
                1e10,
                tolerance=0.01,
                name="Maximum frequency",
            ):
                all_valid = False
            else:
                self.success("Maximum frequency correct (10 GHz)")

            # Validate return loss (should be positive, typically 10-30 dB for good match)
            rl_1ghz = s_params.get("return_loss_1ghz", 0)
            if rl_1ghz < 5 or rl_1ghz > 50:
                self.error(f"Return loss out of expected range: {rl_1ghz:.2f} dB")
                all_valid = False
            else:
                self.success(f"Return loss @ 1 GHz reasonable: {rl_1ghz:.2f} dB")

            # Validate insertion loss (should be positive, typically 0.5-5 dB for 10cm cable)
            il_1ghz = s_params.get("insertion_loss_1ghz", 0)
            if il_1ghz < 0.1 or il_1ghz > 10:
                self.error(f"Insertion loss out of expected range: {il_1ghz:.2f} dB")
                all_valid = False
            else:
                self.success(f"Insertion loss @ 1 GHz reasonable: {il_1ghz:.2f} dB")

        # Validate PCAP results
        if "pcap" in results:
            pcap = results["pcap"]

            # Check total packets
            if not validate_exists(pcap.get("total_packets"), "Total packets"):
                all_valid = False

            if pcap.get("total_packets", 0) != 6:
                self.error(f"Expected 6 packets, got {pcap.get('total_packets')}")
                all_valid = False
            else:
                self.success("Total packet count correct (6 packets)")

            # Check protocol distribution
            tcp_count = pcap.get("tcp_packets", 0)
            udp_count = pcap.get("udp_packets", 0)

            if tcp_count != 4:
                self.error(f"Expected 4 TCP packets, got {tcp_count}")
                all_valid = False
            else:
                self.success("TCP packet count correct (4 packets)")

            if udp_count != 1:
                self.error(f"Expected 1 UDP packet, got {udp_count}")
                all_valid = False
            else:
                self.success("UDP packet count correct (1 packet)")

            # Check total bytes
            total_bytes = pcap.get("total_bytes", 0)
            if total_bytes < 300 or total_bytes > 2000:
                self.warning(f"Total bytes may be unexpected: {total_bytes}")
            else:
                self.success(f"Total bytes captured: {total_bytes} bytes")

        if all_valid:
            self.success("All network format validations passed!")
            self.info("""
Next steps for real network files:

1. LOADING TOUCHSTONE FILES
   from oscura.loaders import load_touchstone
   s_params = load_touchstone("device.s2p")  # 2-port
   s_params = load_touchstone("connector.s4p")  # 4-port

2. S-PARAMETER ANALYSIS
   from oscura.analyzers.signal_integrity import insertion_loss, return_loss

   # Analyze at specific frequency
   rl = return_loss(s_params, frequency=5e9)
   il = insertion_loss(s_params, frequency=5e9)

   # Frequency sweep (all frequencies)
   rl_sweep = return_loss(s_params)
   il_sweep = insertion_loss(s_params)

3. LOADING PCAP FILES
   from oscura.loaders import load_pcap

   packets = load_pcap("capture.pcap")

   # With filtering
   packets = load_pcap("capture.pcap", protocol_filter="TCP", max_packets=1000)

4. PCAP ANALYSIS
   # Filter packets
   tcp_packets = packets.filter(protocol="TCP")
   large_packets = packets.filter(min_size=1000)

   # Iterate over packets
   for pkt in packets:
       print(f"{pkt.timestamp:.6f}s: {pkt.protocol} {len(pkt.data)} bytes")

       # Access annotations (requires dpkt library)
       if "src_ip" in pkt.annotations:
           print(f"  {pkt.annotations['src_ip']} → {pkt.annotations['dst_ip']}")

5. SIGNAL INTEGRITY WORKFLOW
   # Load channel S-parameters
   channel = load_touchstone("pcie_lane.s2p")

   # Check if channel meets specification
   il_max = insertion_loss(channel, frequency=8e9)  # PCIe Gen3 @ 8 GHz
   if il_max > 12:  # PCIe spec limit
       print("Channel insertion loss exceeds specification!")
            """)
        else:
            self.error("Some network format validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = NetworkFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
