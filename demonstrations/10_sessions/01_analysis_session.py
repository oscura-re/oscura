"""AnalysisSession: Core session management and multi-recording workflows

Demonstrates:
- AnalysisSession base class usage
- GenericSession for non-domain-specific analysis
- Recording management (add_recording, get_recording, list_recordings)
- Comparison operations between recordings
- Session persistence fundamentals
- Multi-recording analysis workflow

IEEE Standards: N/A
Related Demos:
- 10_sessions/02_can_session.py - Domain-specific session (CAN)
- 10_sessions/03_blackbox_session.py - Protocol reverse engineering
- 10_sessions/04_session_persistence.py - Save/load workflows

This demonstration shows the core AnalysisSession pattern - the unified interface
for interactive analysis across all domains. Perfect for understanding multi-recording
workflows and differential analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


class AnalysisSessionDemo(BaseDemo):
    """Demonstrates AnalysisSession base class and recording management."""

    def __init__(self):
        """Initialize analysis session demonstration."""
        super().__init__(
            name="analysis_session",
            description="Core session management and multi-recording workflows",
            capabilities=[
                "oscura.sessions.AnalysisSession",
                "oscura.sessions.GenericSession",
                "oscura.acquisition.Source",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate multiple test signals with different characteristics."""
        # Create baseline signal (1kHz sine wave)
        baseline_trace = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,  # 10ms
            sample_rate=100e3,
        )

        # Create signal with frequency variation (1.2kHz)
        freq_variant_trace = generate_sine_wave(
            frequency=1200.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
        )

        # Create signal with amplitude variation (2V)
        amp_variant_trace = generate_sine_wave(
            frequency=1000.0,
            amplitude=2.0,
            duration=0.01,
            sample_rate=100e3,
        )

        # Create noisy signal
        noisy_trace = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
        )
        # Add 10% noise manually
        import numpy as np

        noise = np.random.normal(0, 0.1, len(noisy_trace.data))
        noisy_trace.data = noisy_trace.data + noise  # type: ignore[operator]

        return {
            "baseline": baseline_trace,
            "freq_variant": freq_variant_trace,
            "amp_variant": amp_variant_trace,
            "noisy": noisy_trace,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run the analysis session demonstration."""
        self.section("AnalysisSession: Multi-Recording Workflows")

        # Part 1: Create session and add recordings
        self.subsection("1. Session Creation and Recording Management")

        session = GenericSession(name="Multi-Signal Analysis")
        self.result("Session name", session.name)
        self.result("Session type", session.__class__.__name__)
        self.info(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        # Add recordings using TraceSource
        self.subsection("2. Adding Recordings")

        recordings_added = []
        for name, trace in data.items():
            source = TraceSource(trace)
            session.add_recording(name, source)
            recordings_added.append(name)
            self.result("Added recording", name)

        # List recordings
        self.subsection("3. Listing Recordings")
        recordings = session.list_recordings()
        self.result("Total recordings", len(recordings))
        for idx, name in enumerate(recordings, 1):
            self.info(f"  {idx}. {name}")

        # Part 2: Access recordings
        self.subsection("4. Accessing Individual Recordings")

        baseline = session.get_recording("baseline")
        self.result("Baseline samples", len(baseline.data))
        self.result("Baseline sample rate", baseline.metadata.sample_rate, "Hz")
        self.result("Baseline duration", len(baseline.data) / baseline.metadata.sample_rate, "s")

        # Part 3: Compare recordings (differential analysis)
        self.subsection("5. Differential Analysis (Comparing Recordings)")

        # Compare baseline to frequency variant
        self.info("\nComparing baseline vs frequency variant:")
        diff_freq = session.compare("baseline", "freq_variant")
        self.result("  Changed samples", diff_freq.changed_bytes)
        self.result("  Similarity score", f"{diff_freq.similarity_score:.4f}")
        self.result("  Trace 1 length", diff_freq.details["trace1_length"])
        self.result("  Trace 2 length", diff_freq.details["trace2_length"])

        # Compare baseline to amplitude variant
        self.info("\nComparing baseline vs amplitude variant:")
        diff_amp = session.compare("baseline", "amp_variant")
        self.result("  Changed samples", diff_amp.changed_bytes)
        self.result("  Similarity score", f"{diff_amp.similarity_score:.4f}")

        # Compare baseline to noisy signal
        self.info("\nComparing baseline vs noisy signal:")
        diff_noise = session.compare("baseline", "noisy")
        self.result("  Changed samples", diff_noise.changed_bytes)
        self.result("  Similarity score", f"{diff_noise.similarity_score:.4f}")

        # Part 4: Analyze all recordings
        self.subsection("6. Multi-Recording Analysis")

        analysis_results = session.analyze()
        self.result("Recordings analyzed", analysis_results["num_recordings"])

        self.info("\nSummary statistics for each recording:")
        for name, summary in analysis_results["summary"].items():
            self.info(f"\n  {name}:")
            self.info(f"    Samples: {summary['num_samples']}")
            self.info(f"    Duration: {summary['duration']:.6f} s")
            self.info(f"    Mean: {summary['mean']:.6f}")
            self.info(f"    RMS: {summary['rms']:.6f}")
            self.info(f"    Min: {summary['min']:.6f}")
            self.info(f"    Max: {summary['max']:.6f}")

        # Show pairwise comparisons
        if "comparisons" in analysis_results:
            self.info("\nPairwise comparisons:")
            for key, comp in analysis_results["comparisons"].items():
                self.info(f"  {key}:")
                self.info(f"    Similarity: {comp['similarity']:.4f}")
                self.info(f"    Changed: {comp['changed_samples']} samples")

        # Part 5: Export results
        self.subsection("7. Exporting Results")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "analysis_report.txt"
            json_path = Path(tmpdir) / "results.json"
            csv_path = Path(tmpdir) / "summary.csv"

            # Export as text report
            session.export_results("report", report_path)
            self.result("Exported report", report_path.name)
            self.info(f"  Size: {report_path.stat().st_size} bytes")

            # Export as JSON
            session.export_results("json", json_path)
            self.result("Exported JSON", json_path.name)
            self.info(f"  Size: {json_path.stat().st_size} bytes")

            # Export as CSV
            session.export_results("csv", csv_path)
            self.result("Exported CSV", csv_path.name)
            self.info(f"  Size: {csv_path.stat().st_size} bytes")

        # Summary
        self.subsection("8. Session State")
        self.result("Session modified", session.modified_at.strftime("%Y-%m-%d %H:%M:%S"))
        self.result("Total recordings", len(session.recordings))
        self.result("Metadata entries", len(session.metadata))

        self.success("AnalysisSession demonstration complete!")

        return {
            "session": session,
            "recordings_added": recordings_added,
            "analysis_results": analysis_results,
            "diff_freq_similarity": diff_freq.similarity_score,
            "diff_amp_similarity": diff_amp.similarity_score,
            "diff_noise_similarity": diff_noise.similarity_score,
        }

    def validate(self, results: dict) -> bool:
        """Validate the results."""
        self.info("\nValidating session operations...")

        # Validate correct number of recordings
        if len(results["recordings_added"]) != 4:
            self.error(f"Expected 4 recordings, got {len(results['recordings_added'])}")
            return False
        self.success("✓ Correct number of recordings added")

        # Validate analysis results structure
        analysis = results["analysis_results"]
        if analysis["num_recordings"] != 4:
            self.error(f"Expected 4 recordings analyzed, got {analysis['num_recordings']}")
            return False
        self.success("✓ All recordings analyzed")

        # Validate similarity scores are in valid range
        similarities = [
            results["diff_freq_similarity"],
            results["diff_amp_similarity"],
            results["diff_noise_similarity"],
        ]
        for i, sim in enumerate(similarities, 1):
            if not (0.0 <= sim <= 1.0):
                self.error(f"Similarity score {i} out of range: {sim}")
                return False
        self.success("✓ Similarity scores in valid range [0, 1]")

        # Validate frequency variant shows significant difference
        if results["diff_freq_similarity"] > 0.5:
            self.error(
                f"Expected low similarity for frequency variant, got {results['diff_freq_similarity']:.4f}"
            )
            return False
        self.success("✓ Frequency variant correctly identified as different")

        # Validate noisy signal still has reasonable similarity
        if results["diff_noise_similarity"] < 0.8:
            self.warning(
                f"Noisy signal similarity lower than expected: {results['diff_noise_similarity']:.4f}"
            )
        else:
            self.success(
                f"✓ Noisy signal similarity acceptable: {results['diff_noise_similarity']:.4f}"
            )

        self.success("\nAll validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - AnalysisSession provides unified interface for all domains")
        self.info("  - GenericSession works for general waveform analysis")
        self.info("  - add_recording() manages multiple captures")
        self.info("  - compare() performs differential analysis")
        self.info("  - analyze() processes all recordings together")
        self.info("  - export_results() supports multiple formats")
        self.info("\nNext steps:")
        self.info("  - Try 02_can_session.py for domain-specific analysis")
        self.info("  - Explore 03_blackbox_session.py for protocol RE")
        self.info("  - Learn persistence in 04_session_persistence.py")

        return True


if __name__ == "__main__":
    demo = AnalysisSessionDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
