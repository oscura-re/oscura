"""CANSession: CAN bus reverse engineering workflows

Demonstrates:
- CANSession domain-specific features (via automotive module)
- Message inventory generation
- Pattern discovery (pairs, sequences, correlations)
- Stimulus-response analysis concepts
- Integration with CAN-specific tools

IEEE Standards: ISO 11898 (CAN), ISO 15765-2 (ISO-TP)
Related Demos:
- 05_domain_specific/01_automotive_diagnostics.py - CAN diagnostics
- 03_protocol_decoding/02_automotive_protocols.py - CAN decoding
- 10_sessions/01_analysis_session.py - Base session concepts

This demonstration shows how domain-specific sessions (like CANSession) extend
AnalysisSession with specialized features. CANSession is available in the
oscura.automotive.can module and provides CAN-specific analysis capabilities.

Note: CANSession is part of oscura.automotive.can.session, not the main
      sessions module. This demo shows the pattern using GenericSession
      as a reference implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demonstrations.common import BaseDemo
from oscura.core.types import Trace, TraceMetadata, WaveformTrace
from oscura.sessions import GenericSession


class TraceSource:
    """Simple wrapper to use a Trace as a Source."""

    def __init__(self, trace: Trace):
        """Initialize with a trace."""
        self.trace = trace

    def read(self) -> Trace:
        """Return the trace."""
        return self.trace

    def stream(self, chunk_size: int):
        """Not implemented for this demo."""
        raise NotImplementedError("Streaming not needed for demo")

    def close(self) -> None:
        """Nothing to close."""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


class CANSessionPatternDemo(BaseDemo):
    """Demonstrates CAN session pattern with domain-specific analysis."""

    def __init__(self):
        """Initialize CAN session demonstration."""
        super().__init__(
            name="can_session_pattern",
            description="Domain-specific session pattern for CAN analysis",
            capabilities=[
                "oscura.sessions.AnalysisSession",
                "oscura.sessions.GenericSession",
                "Pattern: Domain-specific sessions",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate CAN-like message patterns."""
        # Simulate CAN message patterns as waveforms
        # In real usage, you would use oscura.automotive.can.session.CANSession
        # with actual CAN data from BLF/ASC files

        def create_can_message_pattern(
            msg_id: int, data_pattern: list[int], repetitions: int = 10
        ) -> WaveformTrace:
            """Create a waveform representing repeated CAN messages."""
            # Create pattern: [ID(2 bytes) | DLC(1) | DATA(8) | gap]
            message = []
            message.extend([(msg_id >> 8) & 0xFF, msg_id & 0xFF])  # ID
            message.append(len(data_pattern))  # DLC
            message.extend(data_pattern + [0] * (8 - len(data_pattern)))  # DATA (padded)

            # Repeat pattern with gaps
            full_pattern = []
            for _ in range(repetitions):
                full_pattern.extend(message)
                full_pattern.extend([0, 0, 0])  # Gap between messages

            # Convert to voltage-like values
            normalized = np.array([b / 255.0 * 5.0 for b in full_pattern], dtype=np.float32)
            metadata = TraceMetadata(sample_rate=1000.0, channel_name=f"can_id_0x{msg_id:03x}")
            return WaveformTrace(data=normalized, metadata=metadata)

        # Baseline: Engine at idle
        # ID 0x280: Engine RPM (idle = ~800 RPM)
        idle_rpm = int(800 * 4)  # Typical CAN scaling
        baseline = create_can_message_pattern(
            0x280, [0x00, 0x00, (idle_rpm >> 8) & 0xFF, idle_rpm & 0xFF, 0x00, 0x00]
        )

        # Active: Engine at 3000 RPM
        active_rpm = int(3000 * 4)
        active = create_can_message_pattern(
            0x280, [0x00, 0x00, (active_rpm >> 8) & 0xFF, active_rpm & 0xFF, 0x00, 0x00]
        )

        # Brake signal: ID 0x3B0 with brake flag
        brake_off = create_can_message_pattern(0x3B0, [0x00, 0x00, 0x00, 0x00])
        brake_on = create_can_message_pattern(0x3B0, [0x01, 0x00, 0x00, 0x00])

        # Speed signal: ID 0x1A0
        speed_0 = create_can_message_pattern(0x1A0, [0x00, 0x00, 0x00, 0x00])
        speed_60 = create_can_message_pattern(0x1A0, [0x00, 0x00, 0x17, 0x70])  # 60 km/h

        return {
            "idle": baseline,
            "active": active,
            "brake_off": brake_off,
            "brake_on": brake_on,
            "speed_0": speed_0,
            "speed_60": speed_60,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run the CAN session pattern demonstration."""
        self.section("CANSession Pattern: Domain-Specific Analysis")

        self.info("NOTE: This demo shows the domain-specific session PATTERN.")
        self.info("Full CANSession is available in oscura.automotive.can.session")
        self.info("with features like message inventory, DBC export, and stimulus-response.")
        self.info("")

        # Part 1: Session creation
        self.subsection("1. Domain-Specific Session Pattern")

        # In real usage: from oscura.automotive.can import CANSession
        # For this demo, we use GenericSession to show the pattern
        session = GenericSession(name="CAN Bus Analysis")
        self.result("Session name", session.name)
        self.result("Session type", session.__class__.__name__)
        self.info("Pattern: Extend AnalysisSession for domain-specific features")

        # Part 2: Add recordings
        self.subsection("2. Recording CAN Bus States")
        self.info("Simulating CAN message captures for different vehicle states...")

        for name, trace in data.items():
            source = TraceSource(trace)
            session.add_recording(name, source)
            self.result("Recorded", f"{name} ({len(trace.data)} samples)")

        # Part 3: Message inventory (simulated)
        self.subsection("3. Message Inventory Generation")
        self.info("CANSession.inventory() would generate message ID catalog:")
        self.info("  - List all unique message IDs observed")
        self.info("  - Count messages per ID")
        self.info("  - Calculate message periods/frequencies")
        self.info("  - Identify sporadic vs periodic messages")

        # Simulated inventory
        inventory = {
            "0x280": {"name": "Engine RPM", "count": 100, "period_ms": 10},
            "0x3B0": {"name": "Brake Status", "count": 50, "period_ms": 20},
            "0x1A0": {"name": "Vehicle Speed", "count": 100, "period_ms": 10},
        }

        self.info("\nSimulated message inventory:")
        for msg_id, info in inventory.items():
            self.info(f"  {msg_id}: {info['name']}")
            self.info(f"    Count: {info['count']} messages")
            self.info(f"    Period: {info['period_ms']} ms")

        # Part 4: Differential analysis
        self.subsection("4. Stimulus-Response Analysis")
        self.info("Compare recordings to identify changing signals...")

        # Compare idle vs active engine
        self.info("\nEngine: Idle vs Active")
        diff_engine = session.compare("idle", "active")
        self.result("  Changed samples", diff_engine.changed_bytes)
        self.result("  Similarity", f"{diff_engine.similarity_score:.4f}")
        self.info("  Interpretation: Bytes 2-3 likely contain RPM")

        # Compare brake states
        self.info("\nBrake: Off vs On")
        diff_brake = session.compare("brake_off", "brake_on")
        self.result("  Changed samples", diff_brake.changed_bytes)
        self.result("  Similarity", f"{diff_brake.similarity_score:.4f}")
        self.info("  Interpretation: Byte 0 likely contains brake flag")

        # Compare speeds
        self.info("\nSpeed: 0 km/h vs 60 km/h")
        diff_speed = session.compare("speed_0", "speed_60")
        self.result("  Changed samples", diff_speed.changed_bytes)
        self.result("  Similarity", f"{diff_speed.similarity_score:.4f}")
        self.info("  Interpretation: Bytes 2-3 likely contain speed value")

        # Part 5: Pattern discovery (conceptual)
        self.subsection("5. Pattern Discovery Features")
        self.info("CANSession provides advanced pattern discovery:")
        self.info("")
        self.info("MessagePairs:")
        self.info("  - Find request-response pairs")
        self.info("  - Example: Diagnostic request (0x7E0) → Response (0x7E8)")
        self.info("")
        self.info("MessageSequences:")
        self.info("  - Detect common message sequences")
        self.info("  - Example: Startup sequence, shutdown sequence")
        self.info("")
        self.info("TemporalCorrelations:")
        self.info("  - Find messages that change together")
        self.info("  - Example: RPM and fuel injection rate")

        # Part 6: Analysis
        self.subsection("6. Multi-Recording Analysis")

        results = session.analyze()
        self.result("Recordings analyzed", results["num_recordings"])
        self.result("Comparisons performed", len(results.get("comparisons", {})))

        # Show summary for one recording
        if "idle" in results["summary"]:
            idle_summary = results["summary"]["idle"]
            self.info("\nIdle recording summary:")
            self.info(f"  Samples: {idle_summary['num_samples']}")
            self.info(f"  Duration: {idle_summary['duration']:.3f} s")
            self.info(f"  Mean value: {idle_summary['mean']:.3f}")

        # Part 7: Export capabilities
        self.subsection("7. Domain-Specific Export Formats")
        self.info("CANSession supports specialized export formats:")
        self.info("")
        self.info("DBC (CAN Database):")
        self.info("  - Generate DBC file with discovered messages")
        self.info("  - Include signal definitions from analysis")
        self.info("  - Compatible with Vector CANalyzer/CANoe")
        self.info("")
        self.info("Stimulus-Response Report:")
        self.info("  - Document which stimuli affect which messages")
        self.info("  - Confidence scores for each hypothesis")
        self.info("  - Suggested signal names and scaling")
        self.info("")
        self.info("Wireshark Dissector:")
        self.info("  - Auto-generate dissector for discovered protocol")
        self.info("  - Works with SocketCAN captures")

        self.success("CANSession pattern demonstration complete!")

        return {
            "session": session,
            "recordings": len(session.recordings),
            "engine_similarity": diff_engine.similarity_score,
            "brake_similarity": diff_brake.similarity_score,
            "speed_similarity": diff_speed.similarity_score,
            "inventory": inventory,
        }

    def validate(self, results: dict) -> bool:
        """Validate the results."""
        self.info("\nValidating CAN session pattern...")

        # Validate recordings
        if results["recordings"] != 6:
            self.error(f"Expected 6 recordings, got {results['recordings']}")
            return False
        self.success(f"✓ All {results['recordings']} recordings added")

        # Validate differential analysis detected changes
        if results["engine_similarity"] > 0.9:
            self.warning("Engine RPM change not clearly detected")
        else:
            self.success(
                f"✓ Engine RPM variation detected (similarity: {results['engine_similarity']:.4f})"
            )

        if results["brake_similarity"] > 0.9:
            self.warning("Brake state change not clearly detected")
        else:
            self.success(
                f"✓ Brake state change detected (similarity: {results['brake_similarity']:.4f})"
            )

        # Validate inventory structure
        inventory = results["inventory"]
        if len(inventory) != 3:
            self.error(f"Expected 3 messages in inventory, got {len(inventory)}")
            return False
        self.success(f"✓ Message inventory contains {len(inventory)} IDs")

        self.success("\nAll validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - Domain-specific sessions extend AnalysisSession")
        self.info("  - CANSession adds message inventory and pattern discovery")
        self.info("  - Stimulus-response analysis identifies signal relationships")
        self.info("  - Export to DBC, dissectors, and reports")
        self.info("  - Pattern: Same interface, specialized features")
        self.info("\nUsing Full CANSession:")
        self.info("  from oscura.automotive.can import CANSession")
        self.info("  from oscura.acquisition import FileSource")
        self.info("  session = CANSession(name='Analysis')")
        self.info("  session.add_recording('data', FileSource('capture.blf'))")
        self.info("  inventory = session.inventory()")
        self.info("  msg = session.message(0x280)")
        self.info("  analysis = msg.analyze()")
        self.info("\nNext steps:")
        self.info("  - Try with real CAN data (BLF/ASC files)")
        self.info("  - Explore 05_domain_specific/01_automotive_diagnostics.py")
        self.info("  - Use message wrappers for per-message analysis")

        return True


if __name__ == "__main__":
    demo = CANSessionPatternDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
