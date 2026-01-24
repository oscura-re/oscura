"""Interactive Analysis: Collaborative workflows and session management

Demonstrates:
- Interactive session workflows
- Annotation and note-taking patterns
- History tracking and replay concepts
- Collaborative analysis patterns
- Session comparison and merging concepts

IEEE Standards: N/A
Related Demos:
- 10_sessions/01_analysis_session.py - Core session features
- 10_sessions/04_session_persistence.py - Metadata and persistence

This demonstration shows interactive analysis patterns - managing long-running
analysis sessions, tracking hypotheses, comparing different analysis approaches,
and documenting findings for collaboration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


class InteractiveAnalysisDemo(BaseDemo):
    """Demonstrates interactive analysis workflows and collaboration patterns."""

    def __init__(self):
        """Initialize interactive analysis demonstration."""
        super().__init__(
            name="interactive_analysis",
            description="Interactive workflows and collaborative analysis patterns",
            capabilities=[
                "oscura.sessions.AnalysisSession",
                "Interactive workflows",
                "Hypothesis tracking",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for interactive analysis."""
        # Create signals with known characteristics for hypothesis testing
        baseline = generate_sine_wave(frequency=1000.0, amplitude=1.0, duration=0.01)
        variant1 = generate_sine_wave(frequency=1100.0, amplitude=1.0, duration=0.01)
        variant2 = generate_sine_wave(frequency=1000.0, amplitude=1.2, duration=0.01)
        variant3 = generate_sine_wave(frequency=1200.0, amplitude=1.5, duration=0.01)

        return {
            "baseline": baseline,
            "variant1": variant1,
            "variant2": variant2,
            "variant3": variant3,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run the interactive analysis demonstration."""
        self.section("Interactive Analysis: Workflows and Collaboration")

        # Part 1: Session initialization for interactive work
        self.subsection("1. Starting an Interactive Session")

        session = GenericSession(name="Interactive Protocol Analysis")

        # Set up session for interactive work
        session.metadata["analyst"] = "Lead Researcher"
        session.metadata["session_type"] = "interactive"
        session.metadata["objective"] = "Identify unknown signal characteristics"
        session.metadata["start_time"] = datetime.now().isoformat()

        self.result("Session name", session.name)
        self.result("Analyst", session.metadata["analyst"])
        self.result("Objective", session.metadata["objective"])

        # Part 2: Hypothesis tracking
        self.subsection("2. Hypothesis Management")
        self.info("Track analysis hypotheses and test results...")

        # Initialize hypothesis tracking in metadata
        session.metadata["hypotheses"] = []

        # Hypothesis 1: Frequency varies between recordings
        hypothesis1 = {
            "id": "H1",
            "timestamp": datetime.now().isoformat(),
            "description": "Frequency varies between recordings",
            "status": "testing",
            "confidence": None,
            "evidence": [],
        }
        session.metadata["hypotheses"].append(hypothesis1)
        self.result("Hypothesis 1", hypothesis1["description"])

        # Hypothesis 2: Amplitude encodes information
        hypothesis2 = {
            "id": "H2",
            "timestamp": datetime.now().isoformat(),
            "description": "Amplitude encodes information",
            "status": "testing",
            "confidence": None,
            "evidence": [],
        }
        session.metadata["hypotheses"].append(hypothesis2)
        self.result("Hypothesis 2", hypothesis2["description"])

        # Part 3: Iterative recording addition
        self.subsection("3. Iterative Data Collection")
        self.info("Add recordings incrementally as analysis progresses...")

        # Add baseline
        session.add_recording("baseline", TraceSource(data["baseline"]))
        session.metadata["hypotheses"][0]["evidence"].append(
            {"recording": "baseline", "note": "Reference signal established"}
        )
        self.result("Added", "baseline (reference)")

        # Add first variant
        session.add_recording("variant1", TraceSource(data["variant1"]))
        comparison1 = session.compare("baseline", "variant1")
        session.metadata["hypotheses"][0]["evidence"].append(
            {
                "recording": "variant1",
                "comparison": "baseline",
                "similarity": comparison1.similarity_score,
                "note": "Frequency differs slightly",
            }
        )
        self.result("Added", f"variant1 (similarity: {comparison1.similarity_score:.4f})")

        # Update hypothesis status
        if comparison1.similarity_score < 0.8:
            session.metadata["hypotheses"][0]["status"] = "supported"
            session.metadata["hypotheses"][0]["confidence"] = 0.85
            self.info("  → Hypothesis H1 supported by evidence")

        # Add second variant
        session.add_recording("variant2", TraceSource(data["variant2"]))
        comparison2 = session.compare("baseline", "variant2")
        session.metadata["hypotheses"][1]["evidence"].append(
            {
                "recording": "variant2",
                "comparison": "baseline",
                "similarity": comparison2.similarity_score,
                "note": "Amplitude differs",
            }
        )
        self.result("Added", f"variant2 (similarity: {comparison2.similarity_score:.4f})")

        if comparison2.similarity_score < 0.9:
            session.metadata["hypotheses"][1]["status"] = "supported"
            session.metadata["hypotheses"][1]["confidence"] = 0.75
            self.info("  → Hypothesis H2 supported by evidence")

        # Part 4: Analysis workflow tracking
        self.subsection("4. Workflow History")
        self.info("Track analysis steps for reproducibility...")

        if "workflow" not in session.metadata:
            session.metadata["workflow"] = []

        # Log workflow steps
        session.metadata["workflow"].extend(
            [
                {
                    "step": 1,
                    "timestamp": datetime.now().isoformat(),
                    "action": "load_baseline",
                    "result": "Established reference signal",
                },
                {
                    "step": 2,
                    "timestamp": datetime.now().isoformat(),
                    "action": "compare_frequency_variant",
                    "result": f"Detected frequency variation (similarity: {comparison1.similarity_score:.4f})",
                },
                {
                    "step": 3,
                    "timestamp": datetime.now().isoformat(),
                    "action": "compare_amplitude_variant",
                    "result": f"Detected amplitude variation (similarity: {comparison2.similarity_score:.4f})",
                },
            ]
        )

        self.info("\nWorkflow history:")
        for step_info in session.metadata["workflow"]:
            ts = datetime.fromisoformat(step_info["timestamp"])
            self.info(f"  Step {step_info['step']}: {step_info['action']}")
            self.info(f"    [{ts.strftime('%H:%M:%S')}] {step_info['result']}")

        # Part 5: Notes and annotations
        self.subsection("5. Session Notes and Observations")

        session.metadata["notes"] = [
            {
                "timestamp": datetime.now().isoformat(),
                "type": "observation",
                "content": "Baseline signal shows stable 1kHz frequency",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "type": "observation",
                "content": "Variant1 shows frequency shift to ~1.1kHz",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "type": "hypothesis",
                "content": "Frequency may encode temperature or state information",
            },
            {
                "timestamp": datetime.now().isoformat(),
                "type": "finding",
                "content": "Both frequency and amplitude vary independently",
            },
        ]

        self.info("\nSession notes:")
        for note in session.metadata["notes"]:
            ts = datetime.fromisoformat(note["timestamp"])
            self.info(f"  [{ts.strftime('%H:%M:%S')}] {note['type']}: {note['content']}")

        # Part 6: Session comparison pattern
        self.subsection("6. Multi-Session Comparison Pattern")
        self.info("Pattern: Compare different analysis approaches...")

        # Create second session with different approach
        session2 = GenericSession(name="Alternative Analysis Approach")
        session2.metadata["approach"] = "Spectral analysis focused"
        session2.add_recording("baseline", TraceSource(data["baseline"]))
        session2.add_recording("variant3", TraceSource(data["variant3"]))

        # Compare session outcomes
        comparison_result = {
            "session1": {
                "name": session.name,
                "recordings": len(session.recordings),
                "hypotheses": len(session.metadata["hypotheses"]),
                "supported_hypotheses": sum(
                    1 for h in session.metadata["hypotheses"] if h["status"] == "supported"
                ),
            },
            "session2": {
                "name": session2.name,
                "recordings": len(session2.recordings),
                "approach": session2.metadata["approach"],
            },
        }

        self.info("\nSession comparison:")
        for session_id, info in comparison_result.items():
            self.info(f"  {session_id}:")
            for key, value in info.items():
                self.info(f"    {key}: {value}")

        # Part 7: Collaboration metadata
        self.subsection("7. Collaborative Analysis Metadata")

        session.metadata["collaboration"] = {
            "reviewers": ["Senior Analyst", "Domain Expert"],
            "review_status": "pending",
            "shared_findings": [
                "Frequency modulation confirmed",
                "Amplitude variation independent of frequency",
            ],
            "questions": [
                "What triggers frequency changes?",
                "Is amplitude variation continuous or discrete?",
            ],
            "next_experiments": [
                "Capture during known state transitions",
                "Test with controlled temperature variation",
            ],
        }

        self.info("\nCollaboration info:")
        collab = session.metadata["collaboration"]
        self.info(f"  Reviewers: {', '.join(collab['reviewers'])}")
        self.info(f"  Review status: {collab['review_status']}")
        self.info(f"\n  Shared findings ({len(collab['shared_findings'])}):")
        for finding in collab["shared_findings"]:
            self.info(f"    - {finding}")
        self.info(f"\n  Open questions ({len(collab['questions'])}):")
        for question in collab["questions"]:
            self.info(f"    - {question}")

        # Part 8: Session summary
        self.subsection("8. Interactive Session Summary")

        _analysis_results = session.analyze()  # Available for detailed inspection

        self.result("Total recordings", len(session.recordings))
        self.result("Hypotheses tested", len(session.metadata["hypotheses"]))
        self.result(
            "Hypotheses supported",
            sum(1 for h in session.metadata["hypotheses"] if h["status"] == "supported"),
        )
        self.result("Workflow steps", len(session.metadata["workflow"]))
        self.result("Session notes", len(session.metadata["notes"]))
        self.result("Pending reviews", len(session.metadata["collaboration"]["reviewers"]))

        self.success("Interactive analysis demonstration complete!")

        return {
            "session": session,
            "session2": session2,
            "hypotheses": session.metadata["hypotheses"],
            "workflow_steps": len(session.metadata["workflow"]),
            "notes_count": len(session.metadata["notes"]),
            "comparison1_similarity": comparison1.similarity_score,
            "comparison2_similarity": comparison2.similarity_score,
        }

    def validate(self, results: dict) -> bool:
        """Validate the results."""
        self.info("\nValidating interactive analysis workflows...")

        # Validate hypothesis tracking
        hypotheses = results["hypotheses"]
        if len(hypotheses) != 2:
            self.error(f"Expected 2 hypotheses, got {len(hypotheses)}")
            return False
        self.success(f"✓ Hypothesis tracking initialized: {len(hypotheses)} hypotheses")

        supported = sum(1 for h in hypotheses if h["status"] == "supported")
        if supported < 1:
            self.error("Expected at least 1 supported hypothesis")
            return False
        self.success(f"✓ Hypotheses validated: {supported} supported")

        # Validate workflow tracking
        if results["workflow_steps"] < 3:
            self.error(f"Expected at least 3 workflow steps, got {results['workflow_steps']}")
            return False
        self.success(f"✓ Workflow documented: {results['workflow_steps']} steps")

        # Validate notes
        if results["notes_count"] < 4:
            self.error(f"Expected at least 4 notes, got {results['notes_count']}")
            return False
        self.success(f"✓ Notes captured: {results['notes_count']} entries")

        # Validate comparisons performed
        if not (0.0 <= results["comparison1_similarity"] <= 1.0):
            self.error("Invalid similarity score for comparison 1")
            return False
        if not (0.0 <= results["comparison2_similarity"] <= 1.0):
            self.error("Invalid similarity score for comparison 2")
            return False
        self.success("✓ Comparisons performed successfully")

        self.success("\nAll validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - Use metadata for hypothesis tracking")
        self.info("  - Document workflow steps for reproducibility")
        self.info("  - Capture observations and findings as you work")
        self.info("  - Track evidence for each hypothesis")
        self.info("  - Include collaboration metadata for team work")
        self.info("  - Compare different analysis approaches")
        self.info("\nInteractive Workflow Pattern:")
        self.info("  1. Initialize session with objective")
        self.info("  2. Define hypotheses to test")
        self.info("  3. Add recordings iteratively")
        self.info("  4. Update hypothesis status with evidence")
        self.info("  5. Document observations and findings")
        self.info("  6. Track workflow for reproducibility")
        self.info("  7. Include collaboration metadata")
        self.info("\nBest Practices:")
        self.info("  - Timestamp all actions and observations")
        self.info("  - Link evidence to specific hypotheses")
        self.info("  - Document 'why' not just 'what'")
        self.info("  - Include open questions for future work")
        self.info("  - Plan next experiments based on findings")
        self.info("  - Enable peer review with clear metadata")
        self.info("\nNext steps:")
        self.info("  - Apply to real unknown protocol analysis")
        self.info("  - Use with BlackBoxSession for protocol RE")
        self.info("  - Export session state for collaboration")
        self.info("  - Build custom analysis notebooks/scripts")

        return True


if __name__ == "__main__":
    demo = InteractiveAnalysisDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
