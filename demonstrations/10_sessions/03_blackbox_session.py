"""BlackBoxSession: Protocol reverse engineering and differential analysis

Demonstrates:
- BlackBoxSession for unknown protocol analysis
- Differential analysis workflow
- Field hypothesis generation
- State machine inference fundamentals
- Protocol specification export
- Byte-level comparison and analysis

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py - Basic protocol RE
- 06_reverse_engineering/04_field_inference.py - Field detection
- 06_reverse_engineering/03_state_machines.py - State machine inference

This demonstration shows BlackBoxSession - specialized for reverse engineering
unknown protocols through differential analysis, field inference, and automatic
protocol specification generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demonstrations.common import BaseDemo
from oscura.core.types import Trace, TraceMetadata, WaveformTrace
from oscura.sessions import BlackBoxSession


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


class BlackBoxSessionDemo(BaseDemo):
    """Demonstrates BlackBoxSession for protocol reverse engineering."""

    def __init__(self):
        """Initialize blackbox session demonstration."""
        super().__init__(
            name="blackbox_session",
            description="Protocol reverse engineering with differential analysis",
            capabilities=[
                "oscura.sessions.BlackBoxSession",
                "oscura.sessions.FieldHypothesis",
                "oscura.sessions.ProtocolSpec",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate synthetic protocol messages for analysis."""
        # Create synthetic protocol messages
        # Format: [PREAMBLE(1) | LENGTH(1) | TYPE(1) | PAYLOAD(n) | CHECKSUM(1)]

        def create_message(msg_type: int, payload: bytes) -> bytes:
            """Create a synthetic protocol message."""
            preamble = 0xAA
            length = len(payload)
            # Simple checksum: XOR of all bytes
            checksum = preamble ^ length ^ msg_type
            for b in payload:
                checksum ^= b
            return bytes([preamble, length, msg_type]) + payload + bytes([checksum])

        # Baseline: Status message with idle state
        baseline_msg = create_message(0x01, bytes([0x00, 0x00, 0x00, 0x00]))  # Idle

        # Button press: Status message with button flag
        button_msg = create_message(0x01, bytes([0x01, 0x00, 0x00, 0x00]))  # Button pressed

        # Temperature 25C: Sensor reading
        temp_25c = 25 * 10  # Temperature in 0.1°C units
        temp_25c_bytes = temp_25c.to_bytes(2, byteorder="big")
        temp_25c_msg = create_message(0x02, bytes([0x00, 0x00]) + temp_25c_bytes)

        # Temperature 30C: Sensor reading
        temp_30c = 30 * 10
        temp_30c_bytes = temp_30c.to_bytes(2, byteorder="big")
        temp_30c_msg = create_message(0x02, bytes([0x00, 0x00]) + temp_30c_bytes)

        # Counter message: Incrementing counter
        counter_msg = create_message(0x03, bytes([0x00, 0x00, 0x00, 0x42]))  # Counter = 66

        # Convert to WaveformTrace (treating bytes as voltage levels)
        def bytes_to_trace(data: bytes, name: str) -> WaveformTrace:
            """Convert byte array to WaveformTrace."""
            # Normalize to voltage-like values (0-5V range)
            normalized = np.array([b / 255.0 * 5.0 for b in data], dtype=np.float32)
            metadata = TraceMetadata(sample_rate=1000.0, channel_name=name)
            return WaveformTrace(data=normalized, metadata=metadata)

        return {
            "baseline": bytes_to_trace(baseline_msg, "baseline"),
            "button": bytes_to_trace(button_msg, "button"),
            "temp_25c": bytes_to_trace(temp_25c_msg, "temp_25c"),
            "temp_30c": bytes_to_trace(temp_30c_msg, "temp_30c"),
            "counter": bytes_to_trace(counter_msg, "counter"),
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run the blackbox session demonstration."""
        self.section("BlackBoxSession: Protocol Reverse Engineering")

        # Part 1: Create session and add recordings
        self.subsection("1. Creating BlackBox Analysis Session")

        session = BlackBoxSession(name="IoT Device Protocol Analysis")
        self.result("Session name", session.name)
        self.result("Session type", session.__class__.__name__)
        self.info("Purpose: Unknown protocol reverse engineering")

        # Add recordings
        self.subsection("2. Adding Protocol Captures")

        for name, trace in data.items():
            source = TraceSource(trace)
            session.add_recording(name, source)
            self.result("Added capture", f"{name} ({len(trace.data)} samples)")

        recordings = session.list_recordings()
        self.result("Total captures", len(recordings))

        # Part 2: Differential analysis
        self.subsection("3. Differential Analysis")
        self.info("Compare captures to identify changing bytes...")

        # Compare baseline to button press
        self.info("\nBaseline vs Button Press:")
        diff_button = session.compare("baseline", "button")
        self.result("  Changed bytes", diff_button.changed_bytes)
        self.result("  Similarity", f"{diff_button.similarity_score:.4f}")
        if diff_button.changed_regions:
            self.info("  Changed regions:")
            for start, end, desc in diff_button.changed_regions:
                self.info(f"    Bytes {start}-{end}: {desc}")

        # Compare temperature readings
        self.info("\nTemperature 25°C vs 30°C:")
        diff_temp = session.compare("temp_25c", "temp_30c")
        self.result("  Changed bytes", diff_temp.changed_bytes)
        self.result("  Similarity", f"{diff_temp.similarity_score:.4f}")
        if diff_temp.changed_regions:
            self.info("  Changed regions:")
            for start, end, desc in diff_temp.changed_regions:
                self.info(f"    Bytes {start}-{end}: {desc}")

        # Compare baseline to counter
        self.info("\nBaseline vs Counter Message:")
        diff_counter = session.compare("baseline", "counter")
        self.result("  Changed bytes", diff_counter.changed_bytes)
        self.result("  Similarity", f"{diff_counter.similarity_score:.4f}")

        # Part 3: Comprehensive analysis
        self.subsection("4. Comprehensive Protocol Analysis")

        analysis_results = session.analyze()
        self.result("Recordings analyzed", analysis_results["num_recordings"])
        self.result("Fields detected", len(analysis_results["field_hypotheses"]))

        # Show field hypotheses
        if analysis_results["field_hypotheses"]:
            self.info("\nField Hypotheses:")
            for field in analysis_results["field_hypotheses"]:
                self.info(f"  {field.name}:")
                self.info(f"    Offset: {field.offset}")
                self.info(f"    Length: {field.length} bytes")
                self.info(f"    Type: {field.field_type}")
                self.info(f"    Confidence: {field.confidence:.2f}")
        else:
            self.info("\nNo fields automatically detected (expected for minimal test data)")

        # Part 4: Generate protocol specification
        self.subsection("5. Protocol Specification Generation")

        spec = session.generate_protocol_spec()
        self.result("Protocol name", spec.name)
        self.result("Fields defined", len(spec.fields))
        self.result("Constants found", len(spec.constants))
        self.result("CRC info entries", len(spec.crc_info))

        if spec.state_machine:
            self.info(f"State machine: {type(spec.state_machine).__name__}")
        else:
            self.info("State machine: Not inferred (expected for simple data)")

        # Part 5: State machine inference
        self.subsection("6. State Machine Inference")

        state_machine = session.infer_state_machine()
        if state_machine:
            self.result("State machine inferred", "Yes")
            self.info(f"  Type: {type(state_machine).__name__}")
        else:
            self.result("State machine inferred", "No (expected for minimal test data)")
            self.info("  Note: State machines require more complex message sequences")

        # Part 6: Export results
        self.subsection("7. Exporting Protocol Documentation")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Export as Markdown report
            report_path = tmppath / "protocol_analysis.md"
            session.export_results("report", report_path)
            self.result("Exported report", report_path.name)
            report_size = report_path.stat().st_size
            self.info(f"  Size: {report_size} bytes")

            # Preview report content
            if report_size > 0:
                preview = report_path.read_text()[:200]
                self.info(f"  Preview: {preview}...")

            # Export as Wireshark dissector
            dissector_path = tmppath / "protocol.lua"
            session.export_results("dissector", dissector_path)
            self.result("Exported dissector", dissector_path.name)
            self.info(f"  Size: {dissector_path.stat().st_size} bytes")

            # Export as JSON specification
            spec_path = tmppath / "protocol_spec.json"
            session.export_results("spec", spec_path)
            self.result("Exported spec", spec_path.name)
            self.info(f"  Size: {spec_path.stat().st_size} bytes")

            # Export as CSV
            csv_path = tmppath / "fields.csv"
            session.export_results("csv", csv_path)
            self.result("Exported CSV", csv_path.name)
            self.info(f"  Size: {csv_path.stat().st_size} bytes")

        self.success("BlackBoxSession demonstration complete!")

        return {
            "session": session,
            "diff_button_changes": diff_button.changed_bytes,
            "diff_temp_changes": diff_temp.changed_bytes,
            "diff_counter_changes": diff_counter.changed_bytes,
            "field_count": len(analysis_results["field_hypotheses"]),
            "protocol_spec": spec,
        }

    def validate(self, results: dict) -> bool:
        """Validate the results."""
        self.info("\nValidating protocol analysis...")

        # Validate differential analysis detected changes
        if results["diff_button_changes"] == 0:
            self.error("Button press should show byte changes")
            return False
        self.success(f"✓ Button press detected: {results['diff_button_changes']} bytes changed")

        if results["diff_temp_changes"] == 0:
            self.error("Temperature change should show byte changes")
            return False
        self.success(
            f"✓ Temperature variation detected: {results['diff_temp_changes']} bytes changed"
        )

        # Validate protocol spec was generated
        spec = results["protocol_spec"]
        if spec is None:
            self.error("Protocol specification not generated")
            return False
        self.success(f"✓ Protocol specification generated: {spec.name}")

        # Check spec structure
        if not hasattr(spec, "fields"):
            self.error("Protocol spec missing fields attribute")
            return False
        self.success("✓ Protocol spec structure valid")

        # Field detection is optional for minimal test data
        field_count = results["field_count"]
        if field_count > 0:
            self.success(f"✓ Detected {field_count} protocol fields")
        else:
            self.info("  Note: No fields auto-detected (expected for minimal test data)")
            self.info("  Field inference requires larger datasets with patterns")

        self.success("\nAll validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - BlackBoxSession specializes in unknown protocol analysis")
        self.info("  - compare() performs byte-level differential analysis")
        self.info("  - analyze() generates field hypotheses automatically")
        self.info("  - generate_protocol_spec() creates complete documentation")
        self.info("  - infer_state_machine() discovers protocol states")
        self.info("  - Multiple export formats (report, dissector, spec, CSV)")
        self.info("\nNext steps:")
        self.info("  - Try with real unknown protocol captures")
        self.info("  - Use larger datasets for better field inference")
        self.info("  - Combine with CRC recovery for complete RE")
        self.info("  - Export dissectors to Wireshark for analysis")

        return True


if __name__ == "__main__":
    demo = BlackBoxSessionDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
