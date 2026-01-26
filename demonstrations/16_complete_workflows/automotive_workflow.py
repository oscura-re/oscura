#!/usr/bin/env python3
"""Automotive Full Stack Analysis Workflow Demonstration.

This demo showcases a complete end-to-end automotive protocol analysis
workflow covering CAN, LIN, and FlexRay protocols.

**Workflow Steps**:
1. Multi-protocol capture loading
2. Bus identification and characterization
3. Protocol-specific decoding
4. Message correlation across buses
5. Timing analysis and latency
6. Diagnostic session analysis
7. Security assessment
8. Report generation

**Features Demonstrated**:
- Multi-protocol analysis
- Cross-bus message correlation
- Timing budget verification
- UDS diagnostic decoding
- Gateway latency measurement
- Security vulnerability scanning

**Protocols Covered**:
- CAN 2.0A/B (Classical CAN)
- CAN FD
- LIN 2.x
- FlexRay

Usage:
    python automotive_full_workflow.py
    python automotive_full_workflow.py --verbose

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
from demonstrations.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader


@dataclass
class CANMessage:
    """CAN message representation."""

    timestamp: float
    arbitration_id: int
    data: bytes
    dlc: int
    is_extended: bool = False
    is_fd: bool = False
    bus_name: str = "CAN1"


@dataclass
class LINFrame:
    """LIN frame representation."""

    timestamp: float
    pid: int
    data: bytes
    checksum: int
    bus_name: str = "LIN1"


@dataclass
class FlexRayFrame:
    """FlexRay frame representation."""

    timestamp: float
    slot_id: int
    cycle: int
    data: bytes
    channel: str = "A"


@dataclass
class GatewayLatency:
    """Gateway routing latency measurement."""

    source_bus: str
    dest_bus: str
    source_id: int
    dest_id: int
    latency_ms: float


class AutomotiveFullWorkflow(BaseDemo):
    """Automotive Full Stack Analysis Workflow.

    This demo simulates a complete automotive network analysis including
    CAN, LIN, and FlexRay buses with cross-bus correlation.
    """

    name = "Automotive Full Stack Workflow"
    description = "Complete automotive multi-protocol analysis workflow"
    category = "complete_workflows"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)

        self.can_messages: list[CANMessage] = []
        self.lin_frames: list[LINFrame] = []
        self.flexray_frames: list[FlexRayFrame] = []
        self.gateway_latencies: list[GatewayLatency] = []

    def _generate_can_traffic(self, duration: float = 1.0) -> list[CANMessage]:
        """Generate simulated CAN bus traffic.

        Args:
            duration: Capture duration in seconds.

        Returns:
            List of CAN messages.
        """
        messages = []
        t = 0.0

        # Simulate typical automotive CAN messages
        periodic_msgs = [
            {"id": 0x100, "dlc": 8, "period": 0.010, "name": "EngineRPM"},
            {"id": 0x120, "dlc": 8, "period": 0.020, "name": "VehicleSpeed"},
            {"id": 0x140, "dlc": 8, "period": 0.050, "name": "SteeringAngle"},
            {"id": 0x200, "dlc": 8, "period": 0.100, "name": "BrakeStatus"},
            {"id": 0x300, "dlc": 4, "period": 0.200, "name": "TurnSignals"},
        ]

        # Generate periodic messages
        counters = {msg["id"]: 0.0 for msg in periodic_msgs}

        while t < duration:
            for msg_def in periodic_msgs:
                if t >= counters[msg_def["id"]]:
                    # Generate data
                    if msg_def["id"] == 0x100:  # Engine RPM
                        rpm = 2000 + int(1000 * np.sin(t * 2))
                        data = bytes([rpm >> 8, rpm & 0xFF, 0, 0, 0, 0, 0, int(t * 10) & 0xFF])
                    elif msg_def["id"] == 0x120:  # Speed
                        speed = int(60 + 20 * np.sin(t))
                        data = bytes([speed, 0, 0, 0, 0, 0, 0, int(t * 10) & 0xFF])
                    else:
                        data = bytes(np.random.randint(0, 256, msg_def["dlc"]))

                    messages.append(
                        CANMessage(
                            timestamp=t,
                            arbitration_id=msg_def["id"],
                            data=data[: msg_def["dlc"]],
                            dlc=msg_def["dlc"],
                            bus_name="CAN1",
                        )
                    )
                    counters[msg_def["id"]] = t + msg_def["period"]

            t += 0.001  # 1ms resolution

        # Add some diagnostic messages (UDS)
        messages.append(
            CANMessage(
                timestamp=0.5,
                arbitration_id=0x7DF,  # Functional address
                data=bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),  # DiagSessionControl
                dlc=8,
                bus_name="CAN1",
            )
        )
        messages.append(
            CANMessage(
                timestamp=0.505,
                arbitration_id=0x7E8,  # ECU response
                data=bytes([0x02, 0x50, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),  # Positive response
                dlc=8,
                bus_name="CAN1",
            )
        )

        messages.sort(key=lambda m: m.timestamp)
        return messages

    def _generate_lin_traffic(self, duration: float = 1.0) -> list[LINFrame]:
        """Generate simulated LIN bus traffic.

        Args:
            duration: Capture duration in seconds.

        Returns:
            List of LIN frames.
        """
        frames = []
        t = 0.0

        # LIN schedule table
        schedule = [
            {"pid": 0x10, "dlc": 4, "name": "RainSensor"},
            {"pid": 0x20, "dlc": 2, "name": "LightSensor"},
            {"pid": 0x30, "dlc": 8, "name": "SeatControl"},
            {"pid": 0x40, "dlc": 4, "name": "MirrorControl"},
        ]

        slot_time = 0.010  # 10ms per slot
        slot_idx = 0

        while t < duration:
            sched = schedule[slot_idx % len(schedule)]

            data = bytes(np.random.randint(0, 256, sched["dlc"]))
            checksum = (~sum(data)) & 0xFF

            frames.append(
                LINFrame(
                    timestamp=t,
                    pid=sched["pid"],
                    data=data,
                    checksum=checksum,
                    bus_name="LIN1",
                )
            )

            slot_idx += 1
            t += slot_time

        return frames

    def _generate_flexray_traffic(self, duration: float = 1.0) -> list[FlexRayFrame]:
        """Generate simulated FlexRay traffic.

        Args:
            duration: Capture duration.

        Returns:
            List of FlexRay frames.
        """
        frames = []

        # FlexRay cycle time
        cycle_time = 0.005  # 5ms cycle
        n_cycles = int(duration / cycle_time)

        # Static slots (deterministic)
        static_slots = [
            {"slot": 1, "name": "BrakeController", "dlc": 16},
            {"slot": 5, "name": "SteeringController", "dlc": 16},
            {"slot": 10, "name": "SuspensionController", "dlc": 8},
        ]

        for cycle in range(min(n_cycles, 64)):  # FlexRay has 64 cycle counter
            for slot_def in static_slots:
                t = cycle * cycle_time + slot_def["slot"] * 0.0001

                data = bytes(np.random.randint(0, 256, slot_def["dlc"]))

                frames.append(
                    FlexRayFrame(
                        timestamp=t,
                        slot_id=slot_def["slot"],
                        cycle=cycle % 64,
                        data=data,
                        channel="A",
                    )
                )

        return frames

    def generate_data(self) -> None:
        """Generate or load multi-protocol automotive capture.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data file if exists
        3. Generate synthetic automotive traffic
        """
        # Try loading from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("automotive_multiprotocol.npz"):
            data_file_to_load = default_file
            print_info(f"Loading data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load, allow_pickle=True)
                # Reconstruct messages from NPZ
                self.can_messages = list(data["can_messages"])
                self.lin_frames = list(data["lin_frames"])
                self.flexray_frames = list(data["flexray_frames"])

                print_result("Loaded from file", data_file_to_load.name)
                print_result("CAN messages", len(self.can_messages))
                print_result("LIN frames", len(self.lin_frames))
                print_result("FlexRay frames", len(self.flexray_frames))
                if self.can_messages:
                    duration = max(
                        self.can_messages[-1].timestamp if self.can_messages else 0,
                        self.lin_frames[-1].timestamp if self.lin_frames else 0,
                        self.flexray_frames[-1].timestamp if self.flexray_frames else 0,
                    )
                    print_result("Capture duration", f"{duration:.1f} s")
                return
            except Exception as e:
                print_info(f"Could not load from {data_file_to_load.name}: {e}")
                print_info("Falling back to synthetic data generation...")

        # Fallback: Generate synthetic data
        print_info("Generating multi-protocol automotive capture...")

        duration = 2.0  # 2 second capture

        # Generate traffic for each bus
        print_info("  Generating CAN traffic...")
        self.can_messages = self._generate_can_traffic(duration)

        print_info("  Generating LIN traffic...")
        self.lin_frames = self._generate_lin_traffic(duration)

        print_info("  Generating FlexRay traffic...")
        self.flexray_frames = self._generate_flexray_traffic(duration)

        print_result("CAN messages", len(self.can_messages))
        print_result("LIN frames", len(self.lin_frames))
        print_result("FlexRay frames", len(self.flexray_frames))
        print_result("Capture duration", f"{duration} s")

    def run_analysis(self) -> None:
        """Run complete automotive analysis workflow."""
        # ===== Step 1: Bus Characterization =====
        print_subheader("Step 1: Bus Characterization")

        # CAN analysis
        can_ids = {m.arbitration_id for m in self.can_messages}
        print_result("CAN unique IDs", len(can_ids))
        print_result("CAN message rate", f"{len(self.can_messages) / 2:.1f} msg/s")

        # LIN analysis
        lin_pids = {f.pid for f in self.lin_frames}
        print_result("LIN unique PIDs", len(lin_pids))
        print_result("LIN frame rate", f"{len(self.lin_frames) / 2:.1f} frames/s")

        # FlexRay analysis
        fr_slots = {f.slot_id for f in self.flexray_frames}
        print_result("FlexRay active slots", len(fr_slots))
        print_result("FlexRay cycles captured", max(f.cycle for f in self.flexray_frames) + 1)

        self.results["can_unique_ids"] = len(can_ids)
        self.results["lin_unique_pids"] = len(lin_pids)
        self.results["flexray_slots"] = len(fr_slots)

        # ===== Step 2: Message Classification =====
        print_subheader("Step 2: Message Classification")

        # Classify CAN messages
        periodic_can = {}
        for msg in self.can_messages:
            if msg.arbitration_id not in periodic_can:
                periodic_can[msg.arbitration_id] = []
            periodic_can[msg.arbitration_id].append(msg.timestamp)

        print_info("CAN Periodic Analysis:")
        for arb_id, timestamps in sorted(periodic_can.items())[:5]:
            if len(timestamps) > 1:
                periods = np.diff(timestamps)
                mean_period = np.mean(periods) * 1000
                jitter = np.std(periods) * 1000
                print_info(
                    f"  ID 0x{arb_id:03X}: {mean_period:.1f}ms period, {jitter:.2f}ms jitter"
                )

        # Identify diagnostic traffic
        diag_msgs = [m for m in self.can_messages if 0x700 <= m.arbitration_id <= 0x7FF]
        print_result("Diagnostic messages", len(diag_msgs))

        self.results["diagnostic_msgs"] = len(diag_msgs)

        # ===== Step 3: UDS Diagnostic Analysis =====
        print_subheader("Step 3: UDS Diagnostic Analysis")

        uds_sessions = []
        for msg in diag_msgs:
            if len(msg.data) >= 2:
                length = msg.data[0]
                sid = msg.data[1]

                # Decode UDS service
                uds_services = {
                    0x10: "DiagnosticSessionControl",
                    0x11: "ECUReset",
                    0x14: "ClearDiagnosticInformation",
                    0x19: "ReadDTCInformation",
                    0x22: "ReadDataByIdentifier",
                    0x27: "SecurityAccess",
                    0x2E: "WriteDataByIdentifier",
                    0x31: "RoutineControl",
                    0x34: "RequestDownload",
                    0x36: "TransferData",
                    0x37: "RequestTransferExit",
                    0x3E: "TesterPresent",
                }

                # Positive responses have 0x40 added to SID
                if sid >= 0x40:
                    service_name = uds_services.get(sid - 0x40, "Unknown")
                    print_info(f"  0x{msg.arbitration_id:03X}: +Response {service_name}")
                else:
                    service_name = uds_services.get(sid, "Unknown")
                    print_info(f"  0x{msg.arbitration_id:03X}: Request {service_name}")

                uds_sessions.append({"id": msg.arbitration_id, "sid": sid, "name": service_name})

        self.results["uds_services"] = len(uds_sessions)

        # ===== Step 4: Cross-Bus Correlation =====
        print_subheader("Step 4: Cross-Bus Correlation")

        # Simulate gateway latency measurement
        # Looking for correlated messages between CAN and FlexRay

        print_info("Gateway routing analysis:")
        print_info("  CAN -> FlexRay routing:")

        # Example: Brake status might be routed from FlexRay to CAN
        brake_can = [m for m in self.can_messages if m.arbitration_id == 0x200]
        brake_fr = [f for f in self.flexray_frames if f.slot_id == 1]

        if brake_can and brake_fr:
            # Measure apparent latency
            latencies = []
            for can_msg in brake_can[:10]:
                # Find nearest FlexRay frame
                for fr_frame in brake_fr:
                    if fr_frame.timestamp < can_msg.timestamp:
                        latency = (can_msg.timestamp - fr_frame.timestamp) * 1000
                        if latency < 50:  # Reasonable gateway latency
                            latencies.append(latency)
                            break

            if latencies:
                avg_latency = np.mean(latencies)
                print_result("  FlexRay->CAN latency", f"{avg_latency:.2f} ms")
                self.gateway_latencies.append(
                    GatewayLatency(
                        source_bus="FlexRay",
                        dest_bus="CAN",
                        source_id=1,
                        dest_id=0x200,
                        latency_ms=avg_latency,
                    )
                )
                self.results["gateway_latency_ms"] = avg_latency

        # ===== Step 5: Timing Analysis =====
        print_subheader("Step 5: Timing Budget Analysis")

        # Check if periodic messages meet timing requirements
        timing_violations = []

        for arb_id, timestamps in periodic_can.items():
            if len(timestamps) > 5:
                periods = np.diff(timestamps) * 1000
                expected_period = np.median(periods)

                # Check for violations (>10% deviation)
                violations = np.sum(np.abs(periods - expected_period) > expected_period * 0.1)
                if violations > 0:
                    timing_violations.append((arb_id, violations, expected_period))

        if timing_violations:
            print_info(f"  {YELLOW}Timing violations detected:{RESET}")
            for arb_id, count, period in timing_violations[:5]:
                print_info(f"    ID 0x{arb_id:03X}: {count} violations (expected {period:.1f}ms)")
        else:
            print_info(f"  {GREEN}No timing violations detected{RESET}")

        self.results["timing_violations"] = len(timing_violations)

        # ===== Step 6: Security Assessment =====
        print_subheader("Step 6: Security Assessment")

        security_findings = []

        # Check for security access attempts
        sec_access = [m for m in diag_msgs if len(m.data) >= 2 and m.data[1] == 0x27]
        if sec_access:
            security_findings.append(f"SecurityAccess requests detected: {len(sec_access)}")

        # Check for programming session
        prog_session = [
            m for m in diag_msgs if len(m.data) >= 3 and m.data[1] == 0x10 and m.data[2] == 0x02
        ]
        if prog_session:
            security_findings.append("Programming session detected")

        # Check for download requests
        download_req = [m for m in diag_msgs if len(m.data) >= 2 and m.data[1] == 0x34]
        if download_req:
            security_findings.append(f"Download requests detected: {len(download_req)}")

        if security_findings:
            print_info("Security-relevant findings:")
            for finding in security_findings:
                print_info(f"  - {finding}")
        else:
            print_info("  No security-relevant diagnostic activity detected")

        self.results["security_findings"] = len(security_findings)

        # ===== Step 7: Bus Load Analysis =====
        print_subheader("Step 7: Bus Load Analysis")

        # Calculate CAN bus load
        total_bits = sum(64 + msg.dlc * 8 + 25 for msg in self.can_messages)  # SOF + data + stuff
        bus_load_pct = total_bits / (2.0 * 500000) * 100  # 500 kbps, 2 sec

        print_result("CAN bus load", f"{bus_load_pct:.1f}%")

        if bus_load_pct > 70:
            print_info(f"  {RED}High bus load - may cause latency{RESET}")
        elif bus_load_pct > 50:
            print_info(f"  {YELLOW}Moderate bus load{RESET}")
        else:
            print_info(f"  {GREEN}Bus load acceptable{RESET}")

        self.results["can_bus_load_pct"] = bus_load_pct

        # ===== Step 8: Summary Report =====
        print_subheader("Step 8: Analysis Summary")

        print_info("=" * 60)
        print_info("AUTOMOTIVE NETWORK ANALYSIS REPORT")
        print_info("=" * 60)

        print_info("")
        print_info("Network Configuration:")
        print_info(f"  CAN Bus: {len(can_ids)} unique IDs, {bus_load_pct:.1f}% load")
        print_info(f"  LIN Bus: {len(lin_pids)} unique PIDs")
        print_info(f"  FlexRay: {len(fr_slots)} active slots")

        print_info("")
        print_info("Diagnostic Activity:")
        print_info(f"  Total diagnostic messages: {len(diag_msgs)}")
        print_info(f"  UDS services detected: {len(uds_sessions)}")

        print_info("")
        print_info("Gateway Performance:")
        if self.gateway_latencies:
            for gl in self.gateway_latencies:
                print_info(f"  {gl.source_bus}->{gl.dest_bus}: {gl.latency_ms:.2f}ms")

        print_info("")
        print_info("Timing Analysis:")
        print_info(f"  Timing violations: {len(timing_violations)}")

        print_info("")
        print_info("Security Assessment:")
        print_info(f"  Security findings: {len(security_findings)}")

        print_info("=" * 60)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate workflow results."""
        # Check traffic was generated
        suite.check_greater(
            "CAN messages",
            len(self.can_messages),
            0,
            category="capture",
        )

        suite.check_greater(
            "LIN frames",
            len(self.lin_frames),
            0,
            category="capture",
        )

        suite.check_greater(
            "FlexRay frames",
            len(self.flexray_frames),
            0,
            category="capture",
        )

        # Check analysis completed
        suite.check_greater(
            "CAN unique IDs",
            self.results.get("can_unique_ids", 0),
            0,
            category="analysis",
        )

        suite.check_greater(
            "Diagnostic messages",
            self.results.get("diagnostic_msgs", 0),
            0,
            category="diagnostics",
        )

        # Check bus load is reasonable
        suite.check_less(
            "CAN bus load",
            self.results.get("can_bus_load_pct", 100),
            100,
            category="performance",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(AutomotiveFullWorkflow))
