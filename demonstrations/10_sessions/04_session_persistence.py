"""Session Persistence: Save/load workflows and audit trails

Demonstrates:
- Session metadata and annotations
- Session state tracking (created_at, modified_at)
- Session history and audit trail concepts
- Metadata management
- Best practices for session documentation

IEEE Standards: N/A
Related Demos:
- 10_sessions/01_analysis_session.py - Core session concepts
- 10_sessions/05_interactive_analysis.py - Interactive workflows

This demonstration shows session persistence patterns - managing metadata,
annotations, and state tracking for reproducible analysis workflows.

Note: This demo focuses on the metadata and state management aspects that
      are available in the current AnalysisSession implementation. Full
      serialization to .tks files with HMAC integrity would be a future
      enhancement to the session system.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime
from typing import TYPE_CHECKING

from demonstrations.common import BaseDemo, generate_sine_wave
from oscura.sessions import GenericSession

if TYPE_CHECKING:
    from oscura.core.types import Trace


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


class SessionPersistenceDemo(BaseDemo):
    """Demonstrates session persistence and metadata management."""

    def __init__(self):
        """Initialize session persistence demonstration."""
        super().__init__(
            name="session_persistence",
            description="Session metadata, state tracking, and audit trails",
            capabilities=[
                "oscura.sessions.AnalysisSession",
                "Session metadata",
                "Session state tracking",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for persistence demo."""
        # Create test signals
        trace1 = generate_sine_wave(frequency=1000.0, amplitude=1.0, duration=0.01)
        trace2 = generate_sine_wave(frequency=2000.0, amplitude=1.5, duration=0.01)
        trace3 = generate_sine_wave(frequency=500.0, amplitude=0.8, duration=0.01)

        return {"signal1": trace1, "signal2": trace2, "signal3": trace3}

    def run_demonstration(self, data: dict) -> dict:
        """Run the session persistence demonstration."""
        self.section("Session Persistence: Metadata and State Tracking")

        # Part 1: Session creation and timestamps
        self.subsection("1. Session Lifecycle Tracking")

        session = GenericSession(name="Persistence Demo Session")
        creation_time = session.created_at
        initial_modified = session.modified_at

        self.result("Session name", session.name)
        self.result("Created at", creation_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.result("Modified at", initial_modified.strftime("%Y-%m-%d %H:%M:%S"))
        self.info("Note: created_at and modified_at track session lifecycle")

        # Part 2: Metadata management
        self.subsection("2. Session Metadata Management")

        # Add custom metadata
        session.metadata["analyst"] = "Research Team"
        session.metadata["project"] = "Protocol Analysis Project"
        session.metadata["device_under_test"] = "IoT Device v2.1"
        session.metadata["test_date"] = datetime.now().isoformat()
        session.metadata["environment"] = {
            "temperature": 25.0,
            "humidity": 45.0,
            "location": "Lab A",
        }
        session.metadata["notes"] = [
            "Initial capture",
            "Baseline established",
            "Normal operating conditions",
        ]

        self.result("Metadata entries", len(session.metadata))
        self.info("\nStored metadata:")
        for key, value in session.metadata.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for k, v in value.items():
                    self.info(f"    {k}: {v}")
            elif isinstance(value, list):
                self.info(f"  {key}:")
                for item in value:
                    self.info(f"    - {item}")
            else:
                self.info(f"  {key}: {value}")

        # Part 3: Recording management with state tracking
        self.subsection("3. Recording Management (State Updates)")

        # Track modified_at changes
        times = [initial_modified]

        # Add first recording
        import time

        time.sleep(0.01)  # Ensure timestamp changes
        session.add_recording("signal1", TraceSource(data["signal1"]))
        times.append(session.modified_at)
        self.result("Added recording 1", "signal1")
        self.info(f"  Modified at: {session.modified_at.strftime('%H:%M:%S.%f')[:-3]}")

        # Add second recording
        time.sleep(0.01)
        session.add_recording("signal2", TraceSource(data["signal2"]))
        times.append(session.modified_at)
        self.result("Added recording 2", "signal2")
        self.info(f"  Modified at: {session.modified_at.strftime('%H:%M:%S.%f')[:-3]}")

        # Add third recording
        time.sleep(0.01)
        session.add_recording("signal3", TraceSource(data["signal3"]))
        times.append(session.modified_at)
        self.result("Added recording 3", "signal3")
        self.info(f"  Modified at: {session.modified_at.strftime('%H:%M:%S.%f')[:-3]}")

        self.info(f"\nTotal modifications tracked: {len(times)} (including initial state)")

        # Part 4: Audit trail pattern
        self.subsection("4. Audit Trail Pattern")
        self.info("Pattern: Use metadata to track analysis history...")

        # Simulate audit trail in metadata
        if "history" not in session.metadata:
            session.metadata["history"] = []

        session.metadata["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "recordings_added",
                "details": {"count": 3, "names": list(session.recordings.keys())},
            }
        )

        # Perform analysis
        results = session.analyze()
        session.metadata["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "analysis_performed",
                "details": {"num_recordings": results["num_recordings"]},
            }
        )

        # Perform comparisons
        session.compare("signal1", "signal2")
        session.metadata["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "comparison_performed",
                "details": {"pair": ["signal1", "signal2"]},
            }
        )

        self.info("\nAudit trail (stored in metadata['history']):")
        for idx, entry in enumerate(session.metadata["history"], 1):
            timestamp = datetime.fromisoformat(entry["timestamp"])
            self.info(f"  {idx}. [{timestamp.strftime('%H:%M:%S')}] {entry['action']}")
            if "details" in entry:
                for key, value in entry["details"].items():
                    self.info(f"      {key}: {value}")

        # Part 5: Session state export
        self.subsection("5. Session State Export")
        self.info("Exporting session state to JSON format...")

        # Create serializable session state
        session_state = {
            "name": session.name,
            "created_at": session.created_at.isoformat(),
            "modified_at": session.modified_at.isoformat(),
            "recordings": {
                name: {
                    "samples": len(session.get_recording(name).data),
                    "sample_rate": session.get_recording(name).metadata.sample_rate,
                }
                for name in session.list_recordings()
            },
            "metadata": session.metadata,
        }

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "session_state.json"
            with open(state_path, "w") as f:
                json.dump(session_state, f, indent=2, default=str)

            self.result("Exported state", state_path.name)
            self.result("File size", f"{state_path.stat().st_size} bytes")

            # Show preview
            with open(state_path) as f:
                content = f.read()
                lines = content.split("\n")[:10]
                self.info("\nState file preview (first 10 lines):")
                for line in lines:
                    self.info(f"  {line}")
                if len(content.split("\n")) > 10:
                    self.info("  ...")

        # Part 6: Session annotations
        self.subsection("6. Session Annotations")
        self.info("Pattern: Use metadata for annotations and documentation...")

        # Add annotations to metadata
        session.metadata["annotations"] = {
            "signal1": {
                "description": "Baseline reference signal",
                "quality": "good",
                "confidence": 0.95,
            },
            "signal2": {
                "description": "Higher frequency test signal",
                "quality": "good",
                "confidence": 0.92,
            },
            "signal3": {
                "description": "Lower frequency comparison",
                "quality": "excellent",
                "confidence": 0.98,
            },
        }

        self.info("\nRecording annotations:")
        for name, annotation in session.metadata["annotations"].items():
            self.info(f"  {name}:")
            for key, value in annotation.items():
                self.info(f"    {key}: {value}")

        # Part 7: Session summary
        self.subsection("7. Complete Session State")

        self.result("Session name", session.name)
        self.result("Total recordings", len(session.recordings))
        self.result("Metadata keys", len(session.metadata))
        self.result("History entries", len(session.metadata.get("history", [])))
        self.result("Annotations", len(session.metadata.get("annotations", {})))

        session_duration = session.modified_at - session.created_at
        self.result("Session duration", f"{session_duration.total_seconds():.3f} seconds")

        self.success("Session persistence demonstration complete!")

        return {
            "session": session,
            "creation_time": creation_time,
            "final_modified_time": session.modified_at,
            "metadata_entries": len(session.metadata),
            "history_entries": len(session.metadata["history"]),
            "modification_count": len(times),
            "session_state": session_state,
        }

    def validate(self, results: dict) -> bool:
        """Validate the results."""
        self.info("\nValidating session persistence...")

        # Validate timestamps
        creation = results["creation_time"]
        final_modified = results["final_modified_time"]

        if final_modified < creation:
            self.error("Modified time should be >= creation time")
            return False
        self.success("✓ Timestamp tracking correct")

        # Validate metadata
        if results["metadata_entries"] < 5:
            self.error(f"Expected at least 5 metadata entries, got {results['metadata_entries']}")
            return False
        self.success(f"✓ Metadata populated: {results['metadata_entries']} entries")

        # Validate history
        if results["history_entries"] < 3:
            self.error(f"Expected at least 3 history entries, got {results['history_entries']}")
            return False
        self.success(f"✓ Audit trail recorded: {results['history_entries']} events")

        # Validate modification tracking
        if results["modification_count"] < 4:
            self.error(
                f"Expected at least 4 modification timestamps, got {results['modification_count']}"
            )
            return False
        self.success(f"✓ State updates tracked: {results['modification_count']} modifications")

        # Validate session state export
        state = results["session_state"]
        if not all(key in state for key in ["name", "created_at", "modified_at", "recordings"]):
            self.error("Session state export incomplete")
            return False
        self.success("✓ Session state export complete")

        self.success("\nAll validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - created_at and modified_at track session lifecycle")
        self.info("  - metadata dict stores arbitrary session information")
        self.info("  - Use metadata['history'] for audit trails")
        self.info("  - Use metadata['annotations'] for recording notes")
        self.info("  - Session state can be exported to JSON")
        self.info("  - Timestamp updates track session modifications")
        self.info("\nBest Practices:")
        self.info("  - Document analyst, project, and test conditions")
        self.info("  - Track all significant analysis operations")
        self.info("  - Annotate recordings with quality/confidence")
        self.info("  - Export session state for reproducibility")
        self.info("  - Use ISO 8601 timestamps for portability")
        self.info("\nFuture Enhancement:")
        self.info("  - Full serialization with pickle/JSON")
        self.info("  - HMAC integrity checking")
        self.info("  - Compressed .tks session file format")
        self.info("  - Session versioning and migration")

        return True


if __name__ == "__main__":
    demo = SessionPersistenceDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
