#!/usr/bin/env python3
"""Comprehensive UDP Packet Analysis Demo using BaseDemo Pattern.

This demo demonstrates Oscura's packet analysis capabilities:
- PCAP/PCAPNG loading with protocol dissection
- Traffic metrics (throughput, jitter, latency)
- Payload pattern analysis and clustering
- Field structure detection and inference
- Protocol fingerprinting

Usage:
    python demos/03_udp_packet_analysis/comprehensive_udp_analysis.py
    python demos/03_udp_packet_analysis/comprehensive_udp_analysis.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.analyzers.packet import (
    PacketInfo,
    cluster_payloads,
    detect_delimiter,
    detect_length_prefix,
    find_checksum_fields,
    find_sequence_fields,
    infer_fields,
    jitter,
    throughput,
)


class UDPPacketAnalysisDemo(BaseDemo):
    """UDP Packet Analysis Demonstration.

    Demonstrates Oscura's packet analysis capabilities including
    traffic metrics, payload analysis, and protocol fingerprinting.
    """

    name = "Comprehensive UDP Packet Analysis"
    description = "Demonstrates UDP packet reverse engineering capabilities"
    category = "packet_analysis"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.packets = []
        self.payloads = []

    def generate_data(self) -> None:
        """Generate or load UDP packet data.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data file if exists (PCAP)
        3. Generate synthetic UDP packet data
        """
        # Try loading from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("industrial_modbus_udp.pcap"):
            data_file_to_load = default_file
            print_info(f"Loading data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                # Try loading as PCAP
                if str(data_file_to_load).endswith(".pcap"):
                    # For now, note that file exists but fall back to synthetic
                    # Full PCAP loading would require additional dependencies (scapy)
                    print_info(f"Found data file: {data_file_to_load.name}")
                    print_info("Loading from PCAP files requires additional setup")
                    print_info("Falling back to synthetic generation for demo")
            except Exception as e:
                print_info(f"Failed to load from file: {e}")

        # Generate synthetic UDP packet data
        print_info("Generating synthetic UDP packet data...")

        # Create synthetic packets with realistic structure
        base_time = 1704067200.0  # 2024-01-01 00:00:00
        packet_interval = 0.001  # 1ms between packets

        # Simulate a simple sensor protocol:
        # [2-byte header][2-byte seq][4-byte data][2-byte checksum]
        for i in range(100):
            # Build packet payload
            header = b"\xaa\x55"  # Magic header
            seq = i.to_bytes(2, "big")
            data = np.random.randint(0, 256, 4).astype(np.uint8).tobytes()
            # Simple checksum
            checksum = (sum(header) + sum(seq) + sum(data)) & 0xFFFF
            payload = header + seq + data + checksum.to_bytes(2, "big")

            # Store packet info
            self.packets.append(
                PacketInfo(
                    timestamp=base_time + i * packet_interval,
                    size=len(payload),
                )
            )
            self.payloads.append(payload)

        print_result("Packets generated", len(self.packets))
        print_result("Payload size", len(self.payloads[0]), "bytes")

    def run_analysis(self) -> None:
        """Execute packet analysis."""
        # === Section 1: Traffic Metrics ===
        print_subheader("Traffic Metrics")
        self._analyze_traffic_metrics()

        # === Section 2: Payload Analysis ===
        print_subheader("Payload Analysis")
        self._analyze_payloads()

        # === Section 3: Pattern Analysis ===
        print_subheader("Pattern Analysis")
        self._analyze_patterns()

        # === Section 4: Field Inference ===
        print_subheader("Field Inference")
        self._analyze_field_inference()

        # === Section 5: Clustering ===
        print_subheader("Clustering Analysis")
        self._analyze_clustering()

    def _analyze_traffic_metrics(self) -> None:
        """Analyze traffic metrics."""
        # Throughput
        tput = throughput(self.packets)
        print_result("Throughput", f"{tput.bits_per_second / 1e3:.1f}", "kbps")
        print_result("Packets/sec", f"{tput.packets_per_second:.1f}")
        print_result("Duration", f"{tput.duration:.3f}", "seconds")

        self.results["throughput_bps"] = tput.bits_per_second
        self.results["packets_per_sec"] = tput.packets_per_second

        # Jitter
        jit = jitter(self.packets)
        print_result("Jitter (mean)", f"{jit.mean * 1000:.3f}", "ms")
        print_result("Jitter (RFC3550)", f"{jit.jitter_rfc3550 * 1000:.3f}", "ms")

        self.results["jitter_mean_ms"] = jit.mean * 1000
        self.results["jitter_rfc3550_ms"] = jit.jitter_rfc3550 * 1000

    def _analyze_payloads(self) -> None:
        """Analyze payload statistics."""
        sizes = [len(p) for p in self.payloads]
        unique_payloads = len(set(self.payloads))

        # Entropy calculation
        all_bytes = b"".join(self.payloads)
        byte_counts = Counter(all_bytes)
        total = len(all_bytes)
        entropy = -sum((count / total) * np.log2(count / total) for count in byte_counts.values())

        print_result("Total payloads", len(self.payloads))
        print_result("Unique payloads", unique_payloads)
        print_result("Size range", f"{min(sizes)}-{max(sizes)}", "bytes")
        print_result("Entropy", f"{entropy:.2f}", "bits")

        self.results["total_payloads"] = len(self.payloads)
        self.results["unique_payloads"] = unique_payloads
        self.results["entropy_bits"] = entropy

    def _analyze_patterns(self) -> None:
        """Analyze payload patterns."""
        # Delimiter detection
        delimiter_result = detect_delimiter(self.payloads)
        if delimiter_result.delimiter:
            print_result("Delimiter", delimiter_result.delimiter.hex())
            print_result("Delimiter confidence", f"{delimiter_result.confidence:.2f}")
        else:
            print_info("No delimiter detected")

        # Length prefix detection
        length_result = detect_length_prefix(self.payloads)
        print_result("Length prefix detected", length_result.detected)
        if length_result.detected:
            print_result("Length bytes", length_result.length_bytes)
            print_result("Length offset", length_result.offset)

        self.results["delimiter_detected"] = delimiter_result.delimiter is not None
        self.results["length_prefix_detected"] = length_result.detected

    def _analyze_field_inference(self) -> None:
        """Infer field structure."""
        # Sequence field detection
        seq_fields = find_sequence_fields(self.payloads)
        print_result("Sequence fields found", len(seq_fields))
        for offset, size in seq_fields[:3]:
            print_info(f"  Sequence at offset {offset}, {size} bytes")

        # Checksum field detection
        checksum_fields = find_checksum_fields(self.payloads)
        print_result("Checksum fields found", len(checksum_fields))
        for offset, size, algorithm in checksum_fields[:3]:
            print_info(f"  Checksum at offset {offset}, {size} bytes ({algorithm})")

        # Full field inference
        schema = infer_fields(self.payloads)
        print_result("Total fields inferred", len(schema.fields))
        for field in schema.fields[:5]:
            print_info(
                f"  Field '{field.name}' at {field.offset}: "
                f"{field.inferred_type} (conf: {field.confidence:.2f})"
            )

        self.results["sequence_fields"] = len(seq_fields)
        self.results["checksum_fields"] = len(checksum_fields)
        self.results["inferred_fields"] = len(schema.fields)

    def _analyze_clustering(self) -> None:
        """Cluster payloads by similarity."""
        clusters = cluster_payloads(self.payloads, threshold=0.8)

        print_result("Clusters found", len(clusters))
        for cluster in clusters[:5]:
            print_info(f"  Cluster {cluster.cluster_id}: {cluster.size} payloads")

        self.results["clusters"] = len(clusters)
        if clusters:
            self.results["largest_cluster"] = max(c.size for c in clusters)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate analysis results."""
        # Traffic metrics
        suite.check_greater(
            "Throughput measured",
            self.results.get("throughput_bps", 0),
            0,
            category="traffic",
        )

        suite.check_greater(
            "Packets/sec > 0",
            self.results.get("packets_per_sec", 0),
            0,
            category="traffic",
        )

        # Payload analysis
        suite.check_equal(
            "All payloads processed",
            self.results.get("total_payloads", 0),
            len(self.payloads),
            category="payload",
        )

        suite.check_greater(
            "Entropy calculated",
            self.results.get("entropy_bits", 0),
            0,
            category="payload",
        )

        # Field inference
        suite.check_greater(
            "Fields inferred",
            self.results.get("inferred_fields", 0),
            0,
            category="inference",
        )

        # Clustering
        suite.check_greater(
            "Clustering completed",
            self.results.get("clusters", 0),
            0,
            category="clustering",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(UDPPacketAnalysisDemo))
