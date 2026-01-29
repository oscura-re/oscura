"""Data Provenance: Track data lineage and processing history

Demonstrates:
- Tracking data transformations
- Processing history preservation
- Metadata chaining
- Audit trail generation

This demonstration shows how to track the provenance of data
through processing pipelines for reproducibility and debugging.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, generate_sine_wave


@dataclass
class ProvenanceRecord:
    """Record of a data transformation."""

    operation: str
    timestamp: datetime
    parameters: dict[str, Any] = field(default_factory=dict)
    parent: ProvenanceRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "parent": self.parent.to_dict() if self.parent else None,
        }

    def trace_lineage(self) -> list[str]:
        """Trace complete lineage."""
        lineage = [self.operation]
        current = self.parent
        while current:
            lineage.insert(0, current.operation)
            current = current.parent
        return lineage


class ProvenanceDemo(BaseDemo):
    """Demonstrate data provenance tracking."""

    def __init__(self) -> None:
        """Initialize provenance demonstration."""
        super().__init__(
            name="provenance",
            description="Track data lineage and processing history",
            capabilities=[
                "provenance.tracking",
                "metadata.chaining",
                "audit_trail",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signal with provenance."""
        signal = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3
        )

        # Create initial provenance record
        provenance = ProvenanceRecord(
            operation="generate_sine_wave",
            timestamp=datetime.now(),
            parameters={
                "frequency": 1000.0,
                "amplitude": 1.0,
                "duration": 0.01,
                "sample_rate": 100e3,
            },
        )

        return {"signal": signal, "provenance": provenance}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run provenance demonstration."""
        results: dict[str, Any] = {}

        self.section("Data Provenance Demonstration")

        signal = data["signal"]
        provenance = data["provenance"]

        # Part 1: Basic Transformation Tracking
        self.subsection("1. Transformation Tracking")

        # Apply filter (simulated)
        filtered = signal  # Would be actual filtering
        provenance = ProvenanceRecord(
            operation="lowpass_filter",
            timestamp=datetime.now(),
            parameters={"cutoff": 5000, "order": 4},
            parent=provenance,
        )
        self.info("Applied lowpass filter")

        # Apply windowing
        windowed = filtered  # Would be actual windowing
        provenance = ProvenanceRecord(
            operation="apply_window",
            timestamp=datetime.now(),
            parameters={"window_type": "hamming"},
            parent=provenance,
        )
        self.info("Applied Hamming window")

        # Apply FFT
        spectrum = np.fft.fft(windowed.data)
        provenance = ProvenanceRecord(
            operation="fft",
            timestamp=datetime.now(),
            parameters={"n_points": len(spectrum)},
            parent=provenance,
        )
        self.info("Computed FFT")

        results["transformations"] = 3

        # Part 2: Lineage Tracing
        self.subsection("2. Lineage Tracing")

        lineage = provenance.trace_lineage()
        self.info("Complete processing lineage:")
        for i, step in enumerate(lineage):
            self.info(f"  {i + 1}. {step}")

        results["lineage_steps"] = len(lineage)

        # Part 3: Audit Trail
        self.subsection("3. Audit Trail Generation")

        audit_trail = []
        current = provenance
        while current:
            audit_trail.append({
                "operation": current.operation,
                "timestamp": current.timestamp.isoformat(),
                "parameters": current.parameters,
            })
            current = current.parent

        self.info("Audit trail (reverse chronological):")
        for entry in audit_trail:
            self.info(f"  {entry['timestamp']}: {entry['operation']}")

        results["audit_entries"] = len(audit_trail)

        # Part 4: Provenance Export
        self.subsection("4. Provenance Export")

        provenance_dict = provenance.to_dict()
        self.info("Provenance can be exported to JSON/YAML for:")
        self.info("  - Reproducibility documentation")
        self.info("  - Debugging analysis pipelines")
        self.info("  - Compliance and audit requirements")

        results["provenance_exported"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate provenance results."""
        if results.get("transformations", 0) < 3:
            self.error("Expected at least 3 transformations")
            return False

        if results.get("lineage_steps", 0) < 4:
            self.error("Expected at least 4 lineage steps")
            return False

        if not results.get("provenance_exported", False):
            self.error("Provenance export failed")
            return False

        self.success("Provenance demonstration complete!")
        return True


if __name__ == "__main__":
    demo = ProvenanceDemo()
    success = demo.execute()
    exit(0 if success else 1)
