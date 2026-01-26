#!/usr/bin/env python3
"""Network Packet Analysis Workflow Demonstration.

This demo showcases a complete end-to-end workflow for network packet
analysis using Oscura, from capture loading to protocol inference.

**Workflow Steps**:
1. Load packet capture data
2. Extract network sessions
3. Decode protocol layers
4. Analyze timing patterns
5. Infer higher-level protocol behavior
6. Generate report

**Features Demonstrated**:
- PCAP-like data handling
- Protocol stack decoding
- Session reconstruction
- Timing analysis
- Statistical summaries
- Pattern detection

**Supported Protocols**:
- Ethernet/IP/TCP/UDP layers
- Application protocol inference
- Custom protocol detection

Usage:
    python network_analysis_workflow.py
    python network_analysis_workflow.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RESET, YELLOW, print_subheader


@dataclass
class NetworkPacket:
    """Simulated network packet."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    payload: bytes
    flags: int = 0


@dataclass
class NetworkSession:
    """A network session (flow)."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packets: list[NetworkPacket]
    start_time: float
    end_time: float
    bytes_sent: int
    bytes_received: int


class NetworkAnalysisWorkflow(BaseDemo):
    """Network Packet Analysis Workflow Demonstration.

    This demo simulates a complete network analysis workflow including
    packet parsing, session reconstruction, and protocol inference.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="Network Analysis Workflow",
            description="Complete end-to-end network packet analysis workflow",
            **kwargs,
        )

        self.packets: list[NetworkPacket] = []
        self.sessions: list[NetworkSession] = []

    def _generate_http_session(
        self, base_time: float, client_ip: str, server_ip: str
    ) -> list[NetworkPacket]:
        """Generate simulated HTTP request/response.

        Args:
            base_time: Session start time.
            client_ip: Client IP address.
            server_ip: Server IP address.

        Returns:
            List of packets for this session.
        """
        packets = []
        t = base_time

        # TCP SYN
        packets.append(
            NetworkPacket(
                timestamp=t,
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=49152,
                dst_port=80,
                protocol="TCP",
                payload=b"",
                flags=0x02,  # SYN
            )
        )
        t += 0.001

        # TCP SYN-ACK
        packets.append(
            NetworkPacket(
                timestamp=t,
                src_ip=server_ip,
                dst_ip=client_ip,
                src_port=80,
                dst_port=49152,
                protocol="TCP",
                payload=b"",
                flags=0x12,  # SYN-ACK
            )
        )
        t += 0.001

        # TCP ACK
        packets.append(
            NetworkPacket(
                timestamp=t,
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=49152,
                dst_port=80,
                protocol="TCP",
                payload=b"",
                flags=0x10,  # ACK
            )
        )
        t += 0.0005

        # HTTP GET
        http_request = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
        packets.append(
            NetworkPacket(
                timestamp=t,
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=49152,
                dst_port=80,
                protocol="TCP",
                payload=http_request,
                flags=0x18,  # PSH-ACK
            )
        )
        t += 0.050  # RTT

        # HTTP Response
        http_response = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n" + b"X" * 100
        packets.append(
            NetworkPacket(
                timestamp=t,
                src_ip=server_ip,
                dst_ip=client_ip,
                src_port=80,
                dst_port=49152,
                protocol="TCP",
                payload=http_response,
                flags=0x18,  # PSH-ACK
            )
        )
        t += 0.001

        # TCP FIN sequence
        packets.append(
            NetworkPacket(
                timestamp=t,
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=49152,
                dst_port=80,
                protocol="TCP",
                payload=b"",
                flags=0x11,  # FIN-ACK
            )
        )

        return packets

    def _generate_dns_query(
        self, base_time: float, client_ip: str, dns_ip: str
    ) -> list[NetworkPacket]:
        """Generate simulated DNS query/response.

        Args:
            base_time: Query start time.
            client_ip: Client IP address.
            dns_ip: DNS server IP address.

        Returns:
            List of packets.
        """
        packets = []

        # DNS Query
        dns_query = b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
        dns_query += b"\x07example\x03com\x00\x00\x01\x00\x01"

        packets.append(
            NetworkPacket(
                timestamp=base_time,
                src_ip=client_ip,
                dst_ip=dns_ip,
                src_port=53001,
                dst_port=53,
                protocol="UDP",
                payload=dns_query,
            )
        )

        # DNS Response
        dns_response = dns_query + b"\x00\x04\x5d\xb8\xd8\x22"  # Answer

        packets.append(
            NetworkPacket(
                timestamp=base_time + 0.020,
                src_ip=dns_ip,
                dst_ip=client_ip,
                src_port=53,
                dst_port=53001,
                protocol="UDP",
                payload=dns_response,
            )
        )

        return packets

    def generate_test_data(self) -> dict:
        """Generate or load simulated network capture.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data file if exists
        3. Generate synthetic network traffic
        """
        # Try loading from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("network_capture.npz"):
            data_file_to_load = default_file
            print_info(f"Loading data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load, allow_pickle=True)
                # Reconstruct packets from NPZ
                self.packets = list(data["packets"])

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Total packets", len(self.packets))
                if self.packets:
                    print_result("Time span", f"{self.packets[-1].timestamp:.3f} s")

                # Protocol breakdown
                protocols = {}
                for pkt in self.packets:
                    protocols[pkt.protocol] = protocols.get(pkt.protocol, 0) + 1

                print_info("Protocol breakdown:")
                for proto, count in sorted(protocols.items()):
                    print_info(f"  {proto}: {count} packets")
                return
            except Exception as e:
                print_info(f"Could not load from {data_file_to_load.name}: {e}")
                print_info("Falling back to synthetic data generation...")

        # Fallback: Generate synthetic data
        print_info("Generating simulated network capture...")

        base_time = 0.0

        # Generate multiple sessions
        # DNS queries
        for i in range(5):
            dns_packets = self._generate_dns_query(
                base_time + i * 0.1, f"192.168.1.{100 + i}", "8.8.8.8"
            )
            self.packets.extend(dns_packets)

        # HTTP sessions
        for i in range(10):
            http_packets = self._generate_http_session(
                base_time + 1.0 + i * 0.5, f"192.168.1.{100 + i}", "93.184.216.34"
            )
            self.packets.extend(http_packets)

        # Sort by timestamp
        self.packets.sort(key=lambda p: p.timestamp)

        print_result("Total packets", len(self.packets))
        print_result("Time span", f"{self.packets[-1].timestamp:.3f} s")

        # Protocol breakdown
        protocols = {}
        for pkt in self.packets:
            protocols[pkt.protocol] = protocols.get(pkt.protocol, 0) + 1

        print_info("Protocol breakdown:")
        for proto, count in sorted(protocols.items()):
            print_info(f"  {proto}: {count} packets")

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Run complete network analysis workflow."""
        # ===== Step 1: Session Reconstruction =====
        print_subheader("Step 1: Session Reconstruction")

        # Group packets into sessions (5-tuple)
        session_map: dict[tuple, list[NetworkPacket]] = {}

        for pkt in self.packets:
            # Normalize flow key (smaller IP/port first)
            if (pkt.src_ip, pkt.src_port) < (pkt.dst_ip, pkt.dst_port):
                key = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.protocol)
            else:
                key = (pkt.dst_ip, pkt.dst_port, pkt.src_ip, pkt.src_port, pkt.protocol)

            if key not in session_map:
                session_map[key] = []
            session_map[key].append(pkt)

        # Create session objects
        for key, pkts in session_map.items():
            bytes_sent = sum(len(p.payload) for p in pkts if p.src_ip == key[0])
            bytes_recv = sum(len(p.payload) for p in pkts if p.dst_ip == key[0])

            session = NetworkSession(
                src_ip=key[0],
                dst_ip=key[2],
                src_port=key[1],
                dst_port=key[3],
                protocol=key[4],
                packets=pkts,
                start_time=pkts[0].timestamp,
                end_time=pkts[-1].timestamp,
                bytes_sent=bytes_sent,
                bytes_received=bytes_recv,
            )
            self.sessions.append(session)

        print_result("Sessions identified", len(self.sessions))

        self.results["session_count"] = len(self.sessions)

        # Session breakdown
        tcp_sessions = sum(1 for s in self.sessions if s.protocol == "TCP")
        udp_sessions = sum(1 for s in self.sessions if s.protocol == "UDP")
        print_result("TCP sessions", tcp_sessions)
        print_result("UDP sessions", udp_sessions)

        # ===== Step 2: Protocol Detection =====
        print_subheader("Step 2: Protocol Detection")

        detected_protocols: dict[str, int] = {}

        for session in self.sessions:
            app_proto = "Unknown"

            # Check for HTTP
            for pkt in session.packets:
                if pkt.payload.startswith(b"GET ") or pkt.payload.startswith(b"HTTP/"):
                    app_proto = "HTTP"
                    break
                elif pkt.dst_port == 53 or pkt.src_port == 53:
                    app_proto = "DNS"
                    break

            detected_protocols[app_proto] = detected_protocols.get(app_proto, 0) + 1

        print_info("Detected application protocols:")
        for proto, count in sorted(detected_protocols.items()):
            print_info(f"  {proto}: {count} sessions")

        self.results["http_sessions"] = detected_protocols.get("HTTP", 0)
        self.results["dns_sessions"] = detected_protocols.get("DNS", 0)

        # ===== Step 3: Timing Analysis =====
        print_subheader("Step 3: Timing Analysis")

        # Inter-packet timing
        timestamps = [p.timestamp for p in self.packets]
        inter_packet = np.diff(timestamps)

        if len(inter_packet) > 0:
            print_result("Mean inter-packet time", f"{np.mean(inter_packet) * 1000:.2f} ms")
            print_result("Min inter-packet time", f"{np.min(inter_packet) * 1000:.3f} ms")
            print_result("Max inter-packet time", f"{np.max(inter_packet) * 1000:.2f} ms")

            self.results["mean_ipt_ms"] = np.mean(inter_packet) * 1000

        # Session duration analysis
        durations = [s.end_time - s.start_time for s in self.sessions]
        if durations:
            print_result("Mean session duration", f"{np.mean(durations) * 1000:.2f} ms")
            print_result("Max session duration", f"{np.max(durations) * 1000:.2f} ms")

            self.results["mean_session_duration_ms"] = np.mean(durations) * 1000

        # ===== Step 4: Traffic Statistics =====
        print_subheader("Step 4: Traffic Statistics")

        total_bytes = sum(len(p.payload) for p in self.packets)
        print_result("Total payload bytes", total_bytes)

        # Bytes per protocol
        bytes_by_proto: dict[str, int] = {}
        for pkt in self.packets:
            bytes_by_proto[pkt.protocol] = bytes_by_proto.get(pkt.protocol, 0) + len(pkt.payload)

        print_info("Bytes by transport protocol:")
        for proto, byte_count in sorted(bytes_by_proto.items()):
            pct = byte_count / max(total_bytes, 1) * 100
            print_info(f"  {proto}: {byte_count} bytes ({pct:.1f}%)")

        self.results["total_bytes"] = total_bytes

        # Top talkers
        bytes_by_ip: dict[str, int] = {}
        for pkt in self.packets:
            bytes_by_ip[pkt.src_ip] = bytes_by_ip.get(pkt.src_ip, 0) + len(pkt.payload)

        top_talkers = sorted(bytes_by_ip.items(), key=lambda x: x[1], reverse=True)[:5]
        print_info("Top talkers (by bytes sent):")
        for ip, byte_count in top_talkers:
            print_info(f"  {ip}: {byte_count} bytes")

        # ===== Step 5: Anomaly Detection =====
        print_subheader("Step 5: Anomaly Detection")

        anomalies = []

        # Check for unusual port usage
        for session in self.sessions:
            if session.protocol == "TCP" and session.dst_port not in [80, 443, 22, 25]:
                if session.dst_port > 1024:
                    anomalies.append(f"Non-standard port: {session.dst_port}")

        # Check for large DNS responses
        for session in self.sessions:
            if session.dst_port == 53 or session.src_port == 53:
                for pkt in session.packets:
                    if len(pkt.payload) > 512:
                        anomalies.append("Large DNS response (>512 bytes)")

        if anomalies:
            print_info(f"  {YELLOW}Potential anomalies detected:{RESET}")
            for anomaly in anomalies[:5]:
                print_info(f"    - {anomaly}")
        else:
            print_info(f"  {GREEN}No anomalies detected{RESET}")

        self.results["anomaly_count"] = len(anomalies)

        # ===== Step 6: Summary Report =====
        print_subheader("Step 6: Summary Report")

        print_info("=" * 50)
        print_info("NETWORK ANALYSIS SUMMARY")
        print_info("=" * 50)
        print_info(f"Capture duration: {self.packets[-1].timestamp:.3f} seconds")
        print_info(f"Total packets: {len(self.packets)}")
        print_info(f"Total sessions: {len(self.sessions)}")
        print_info(f"Total bytes: {total_bytes}")
        print_info(f"Protocols: {', '.join(detected_protocols.keys())}")

        print_info("")
        print_info("Session Types:")
        print_info(f"  HTTP: {detected_protocols.get('HTTP', 0)}")
        print_info(f"  DNS: {detected_protocols.get('DNS', 0)}")

        print_info("")
        print_info("Performance Metrics:")
        if len(inter_packet) > 0:
            print_info(
                f"  Avg packet rate: {len(self.packets) / self.packets[-1].timestamp:.1f} pps"
            )
            print_info(
                f"  Avg throughput: {total_bytes * 8 / self.packets[-1].timestamp / 1000:.1f} kbps"
            )

        print_info("=" * 50)

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate workflow results."""
        suite = ValidationSuite()

        # Check packets were generated
        packet_count = results.get("packet_count", 0)
        suite.add_check("Packets generated", packet_count > 0, f"Generated {packet_count} packets")

        # Check sessions were identified
        session_count = results.get("session_count", 0)
        suite.add_check("Sessions identified", session_count > 0, f"Found {session_count} sessions")

        # Check HTTP sessions detected
        http_sessions = results.get("http_sessions", 0)
        suite.add_check("HTTP sessions", http_sessions > 0, f"Found {http_sessions} HTTP sessions")

        # Check DNS sessions detected
        dns_sessions = results.get("dns_sessions", 0)
        suite.add_check("DNS sessions", dns_sessions > 0, f"Found {dns_sessions} DNS sessions")

        # Check timing analysis completed
        mean_ipt_ms = results.get("mean_ipt_ms", 0)
        suite.add_check("Timing analysis", mean_ipt_ms > 0, f"Mean IPT: {mean_ipt_ms:.2f} ms")

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(NetworkAnalysisWorkflow))
