"""Black-box protocol analysis session.

This module provides BlackBoxSession - a specialized analysis session for
reverse engineering unknown protocols through differential analysis and
field hypothesis generation.

Example:
    >>> from oscura.sessions import BlackBoxSession
    >>> from oscura.hardware.acquisition import FileSource
    >>>
    >>> # Create session for black-box protocol analysis
    >>> session = BlackBoxSession(name="IoT Device Protocol Analysis")
    >>>
    >>> # Add recordings from different stimuli
    >>> session.add_recording("baseline", FileSource("idle.bin"))
    >>> session.add_recording("button_press", FileSource("button.bin"))
    >>> session.add_recording("temperature_25C", FileSource("temp25.bin"))
    >>> session.add_recording("temperature_30C", FileSource("temp30.bin"))
    >>>
    >>> # Compare recordings to find differences
    >>> diff = session.compare("baseline", "button_press")
    >>> print(f"Changed bytes: {diff.changed_bytes}")
    >>>
    >>> # Generate protocol specification
    >>> spec = session.generate_protocol_spec()
    >>> print(f"Inferred fields: {len(spec['fields'])}")
    >>>
    >>> # Infer state machine
    >>> sm = session.infer_state_machine()
    >>> print(f"States: {len(sm.states)}")
    >>>
    >>> # Export results
    >>> session.export_results("report", "analysis_report.md")
    >>> session.export_results("dissector", "protocol.lua")

Pattern:
    BlackBoxSession extends AnalysisSession and adds:
    - Differential analysis (byte-level comparison)
    - Field hypothesis generation
    - State machine inference
    - CRC/checksum reverse engineering
    - Protocol specification generation
    - Wireshark dissector export

Use Cases:
    - IoT device protocol reverse engineering
    - Proprietary protocol understanding
    - Security vulnerability discovery
    - Right-to-repair device replication
    - Commercial intelligence

References:
    Architecture Plan Phase 1.1: BlackBoxSession
    docs/architecture/api-patterns.md: When to use Sessions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oscura.inference.alignment import align_global
from oscura.inference.message_format import infer_format
from oscura.inference.state_machine import infer_rpni
from oscura.sessions.base import AnalysisSession, ComparisonResult

if TYPE_CHECKING:
    from oscura.core.types import Trace


@dataclass
class FieldHypothesis:
    """Hypothesis about a field in the protocol.

    Attributes:
        name: Field name (e.g., "counter", "temperature", "checksum").
        offset: Byte offset in message.
        length: Field length in bytes.
        field_type: Inferred type ("counter", "constant", "checksum", "data", "unknown").
        confidence: Confidence score (0.0 to 1.0).
        evidence: Supporting evidence for this hypothesis.

    Example:
        >>> field = FieldHypothesis(
        ...     name="message_counter",
        ...     offset=2,
        ...     length=1,
        ...     field_type="counter",
        ...     confidence=0.95,
        ...     evidence={"increments_by_1": True, "wraps_at_256": True}
        ... )
    """

    name: str
    offset: int
    length: int
    field_type: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolSpec:
    """Protocol specification generated from analysis.

    Attributes:
        name: Protocol name.
        fields: List of inferred fields.
        state_machine: State machine (if inferred).
        crc_info: CRC/checksum information (if found).
        constants: Dictionary of constant values.
        metadata: Additional protocol metadata.

    Example:
        >>> spec = ProtocolSpec(
        ...     name="IoT Device Protocol",
        ...     fields=[field1, field2, field3],
        ...     state_machine=sm,
        ...     crc_info={"polynomial": 0x1021, "location": (4, 6)}
        ... )
    """

    name: str
    fields: list[FieldHypothesis] = field(default_factory=list)
    state_machine: Any = None
    crc_info: dict[str, Any] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BlackBoxSession(AnalysisSession):
    """Session for black-box protocol reverse engineering.

    Provides differential analysis, field inference, and protocol
    specification generation for unknown protocols.

    Features:
        - Byte-level differential analysis
        - Automatic field hypothesis generation
        - State machine inference
        - CRC/checksum reverse engineering
        - Protocol specification export
        - Wireshark dissector generation

    Example:
        >>> session = BlackBoxSession(name="Device RE")
        >>> session.add_recording("idle", FileSource("idle.bin"))
        >>> session.add_recording("active", FileSource("active.bin"))
        >>> diff = session.compare("idle", "active")
        >>> spec = session.generate_protocol_spec()
        >>> session.export_results("dissector", "protocol.lua")

    References:
        Architecture Plan Phase 1.1: BlackBoxSession
    """

    def __init__(
        self,
        name: str = "Black-Box Analysis",
        auto_crc: bool = True,
        crc_min_messages: int = 10,
    ) -> None:
        """Initialize black-box analysis session.

        Args:
            name: Session name (default: "Black-Box Analysis").
            auto_crc: Enable automatic CRC recovery (default: True).
            crc_min_messages: Minimum messages required for CRC recovery (default: 10).

        Example:
            >>> session = BlackBoxSession(name="IoT Protocol RE")
            >>> session = BlackBoxSession(name="RE", auto_crc=True, crc_min_messages=5)
        """
        super().__init__(name)
        self._field_hypotheses: list[FieldHypothesis] = []
        self._protocol_spec: ProtocolSpec | None = None
        self.auto_crc = auto_crc
        self.crc_min_messages = crc_min_messages
        self._crc_params: Any = None  # CRCParameters from inference.crc_reverse

    def analyze(self) -> dict[str, Any]:
        """Perform comprehensive black-box protocol analysis.

        Analyzes all recordings to:
        - Generate field hypotheses
        - Infer state machine
        - Detect CRC/checksums
        - Build protocol specification

        Returns:
            Dictionary with analysis results:
            - num_recordings: Number of recordings analyzed
            - field_hypotheses: List of inferred fields
            - state_machine: Inferred state machine (if found)
            - crc_info: CRC/checksum information (if found)
            - protocol_spec: Complete protocol specification

        Example:
            >>> session = BlackBoxSession()
            >>> session.add_recording("r1", FileSource("data1.bin"))
            >>> session.add_recording("r2", FileSource("data2.bin"))
            >>> results = session.analyze()
            >>> print(f"Found {len(results['field_hypotheses'])} fields")

        References:
            Architecture Plan Phase 1.1: BlackBoxSession Analysis
        """
        if not self.recordings:
            return {
                "num_recordings": 0,
                "field_hypotheses": [],
                "state_machine": None,
                "crc_info": {},
                "protocol_spec": None,
            }

        # Load all recordings
        traces = []
        for name, (source, cached_trace) in self.recordings.items():
            if cached_trace is None:
                cached_trace = source.read()
                self.recordings[name] = (source, cached_trace)
            traces.append(cached_trace)

        # Generate field hypotheses
        self._field_hypotheses = self._generate_field_hypotheses(traces)

        # Automatically recover CRC if enabled
        self._auto_recover_crc(traces)

        # Infer state machine
        state_machine = self._infer_state_machine(traces)

        # Detect CRC/checksums (includes auto-recovered params)
        crc_info = self._detect_crc(traces)

        # Build protocol specification
        self._protocol_spec = ProtocolSpec(
            name=f"{self.name} Protocol",
            fields=self._field_hypotheses,
            state_machine=state_machine,
            crc_info=crc_info,
        )

        return {
            "num_recordings": len(self.recordings),
            "field_hypotheses": self._field_hypotheses,
            "state_machine": state_machine,
            "crc_info": crc_info,
            "protocol_spec": self._protocol_spec,
        }

    def compare(self, name1: str, name2: str) -> ComparisonResult:
        """Compare two recordings for differential analysis.

        Performs byte-level comparison to identify:
        - Changed bytes
        - Changed regions
        - Similarity score

        Args:
            name1: Name of first recording.
            name2: Name of second recording.

        Returns:
            ComparisonResult with detailed differences.

        Example:
            >>> session = BlackBoxSession()
            >>> session.add_recording("baseline", FileSource("idle.bin"))
            >>> session.add_recording("stimulus", FileSource("button.bin"))
            >>> diff = session.compare("baseline", "stimulus")
            >>> print(f"Changed bytes: {diff.changed_bytes}")
            >>> for start, end, desc in diff.changed_regions:
            ...     print(f"  Region {start}-{end}: {desc}")

        References:
            Architecture Plan Phase 1.1: Differential Analysis
        """
        # Get recordings
        if name1 not in self.recordings:
            raise KeyError(f"Recording '{name1}' not found")
        if name2 not in self.recordings:
            raise KeyError(f"Recording '{name2}' not found")

        # Load traces
        source1, trace1 = self.recordings[name1]
        if trace1 is None:
            trace1 = source1.read()
            self.recordings[name1] = (source1, trace1)

        source2, trace2 = self.recordings[name2]
        if trace2 is None:
            trace2 = source2.read()
            self.recordings[name2] = (source2, trace2)

        # Convert to byte arrays for comparison
        data1 = self._trace_to_bytes(trace1)
        data2 = self._trace_to_bytes(trace2)

        # Align sequences if different lengths
        if len(data1) != len(data2):
            alignment = align_global(data1.tolist(), data2.tolist())
            # AlignmentResult uses aligned_a and aligned_b, not seq1_aligned/seq2_aligned
            data1 = np.array(alignment.aligned_a, dtype=np.int32)  # -1 for gaps
            data2 = np.array(alignment.aligned_b, dtype=np.int32)  # -1 for gaps

        # Find differences
        min_len = min(len(data1), len(data2))
        diffs = data1[:min_len] != data2[:min_len]
        changed_bytes = int(np.sum(diffs))

        # Find changed regions
        changed_regions = self._find_changed_regions(diffs)

        # Calculate similarity
        total_bytes = max(len(data1), len(data2))
        similarity = 1.0 - (changed_bytes / total_bytes) if total_bytes > 0 else 1.0

        return ComparisonResult(
            recording1=name1,
            recording2=name2,
            changed_bytes=changed_bytes,
            changed_regions=changed_regions,
            similarity_score=similarity,
            details={
                "len1": len(data1),
                "len2": len(data2),
                "alignment_required": len(data1) != len(data2),
            },
        )

    def generate_protocol_spec(self) -> ProtocolSpec:
        """Generate complete protocol specification.

        Runs full analysis and returns protocol specification with:
        - Inferred fields
        - State machine
        - CRC information
        - Constants

        Returns:
            ProtocolSpec with complete protocol information.

        Example:
            >>> session = BlackBoxSession()
            >>> # ... add recordings ...
            >>> spec = session.generate_protocol_spec()
            >>> print(f"Protocol: {spec.name}")
            >>> for field in spec.fields:
            ...     print(f"  {field.name} @ offset {field.offset}")

        References:
            Architecture Plan Phase 1.1: Protocol Specification
        """
        if self._protocol_spec is None:
            self.analyze()

        return self._protocol_spec  # type: ignore[return-value]

    def infer_state_machine(self) -> Any:
        """Infer state machine from recordings.

        Analyzes message sequences to infer:
        - States
        - Transitions
        - Triggers

        Returns:
            StateMachine object with inferred states and transitions.

        Example:
            >>> session = BlackBoxSession()
            >>> # ... add recordings ...
            >>> sm = session.infer_state_machine()
            >>> print(f"States: {len(sm.states)}")
            >>> print(f"Transitions: {len(sm.transitions)}")

        References:
            Architecture Plan Phase 1.1: State Machine Inference
        """
        if not self.recordings:
            return None

        # Load all traces
        traces = []
        for name, (source, cached_trace) in self.recordings.items():
            if cached_trace is None:
                cached_trace = source.read()
                self.recordings[name] = (source, cached_trace)
            traces.append(cached_trace)

        return self._infer_state_machine(traces)

    def export_results(self, format: str, path: str | Path) -> None:
        """Export analysis results to file.

        Supported formats:
        - "report": Markdown analysis report
        - "dissector": Wireshark Lua dissector
        - "spec": Protocol specification JSON
        - "json": Complete results as JSON
        - "csv": Field hypotheses as CSV

        Args:
            format: Export format.
            path: Output file path.

        Example:
            >>> session = BlackBoxSession()
            >>> # ... perform analysis ...
            >>> session.export_results("report", "analysis.md")
            >>> session.export_results("dissector", "protocol.lua")
            >>> session.export_results("spec", "protocol.json")

        References:
            Architecture Plan Phase 1.1: Result Export
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "report":
            self._export_report(path_obj)
        elif format == "dissector":
            self._export_dissector(path_obj)
        elif format == "spec":
            self._export_spec_json(path_obj)
        elif format == "json":
            self._export_json(path_obj)
        elif format == "csv":
            self._export_csv(path_obj)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # Private helper methods

    def _trace_to_bytes(self, trace: Trace) -> NDArray[np.uint8]:
        """Convert trace to byte array."""
        from oscura.core.types import IQTrace

        # Handle IQTrace separately
        if isinstance(trace, IQTrace):
            raise TypeError("IQTrace not supported in blackbox session")

        # If trace data is already bytes, return as-is
        if trace.data.dtype == np.uint8:
            return trace.data  # type: ignore[return-value]

        # Otherwise, quantize to bytes
        data = trace.data
        if data.dtype in (np.float32, np.float64):
            # Normalize to 0-255
            data_min = np.min(data)
            data_max = np.max(data)
            if float(data_max) > float(data_min):
                normalized = (data - data_min) / (data_max - data_min) * 255  # type: ignore[operator,call-overload]
            else:
                normalized = np.zeros_like(data)
            result: NDArray[np.uint8] = normalized.astype(np.uint8)
            return result

        # For integer types, just convert
        return data.astype(np.uint8)

    def _find_changed_regions(self, diffs: NDArray[np.bool_]) -> list[tuple[int, int, str]]:
        """Find contiguous regions of changes."""
        regions = []
        in_region = False
        start = 0

        for i, changed in enumerate(diffs):
            if bool(changed) and not in_region:
                # Start of new region
                start = i
                in_region = True
            elif not changed and in_region:
                # End of region
                regions.append((start, i - 1, "Changed"))
                in_region = False

        # Handle region extending to end
        if in_region:
            regions.append((start, len(diffs) - 1, "Changed"))

        return regions

    def _generate_field_hypotheses(self, traces: list[Trace]) -> list[FieldHypothesis]:
        """Generate field hypotheses from traces."""
        if not traces:
            return []

        # Convert traces to byte arrays
        byte_arrays = [self._trace_to_bytes(t) for t in traces]

        # Early exit for small/unsuitable datasets (performance optimization)
        if len(byte_arrays) < 3 or any(len(arr) < 10 for arr in byte_arrays):
            return []

        # Use message format inference
        try:
            schema = infer_format(byte_arrays)  # type: ignore[arg-type]

            # Convert to field hypotheses
            hypotheses = []
            for field in schema.fields:
                hyp = FieldHypothesis(
                    name=field.name,
                    offset=field.offset,
                    length=field.size,  # InferredField uses 'size', not 'length'
                    field_type=field.field_type,
                    confidence=field.confidence,
                    evidence={},
                )
                hypotheses.append(hyp)

            return hypotheses
        except Exception:
            # Fallback: basic field detection
            return []

    def _infer_state_machine(self, traces: list[Trace]) -> Any:
        """Infer state machine from traces."""
        if not traces:
            return None

        # Early exit for small datasets (performance optimization)
        # State machine inference requires meaningful sequences
        if len(traces) < 3:
            return None

        try:
            # Convert traces to sequences of strings for RPNI
            # Each trace becomes a sequence
            byte_arrays = [self._trace_to_bytes(t) for t in traces]

            # Skip if sequences are too short for meaningful state machine
            if any(len(arr) < 10 for arr in byte_arrays):
                return None

            # Convert to lists of strings for RPNI input format
            # Cast: list[list[str]] is compatible with list[list[str | int]] at runtime
            sequences = [[str(b) for b in arr.tolist()] for arr in byte_arrays]
            sequences_union: list[list[str | int]] = sequences  # type: ignore[assignment]
            return infer_rpni(sequences_union)
        except Exception:
            return None

    def _auto_recover_crc(self, traces: list[Trace]) -> None:
        """Automatically recover CRC parameters if enough messages.

        Args:
            traces: List of traces to analyze for CRC.

        Note:
            This method attempts automatic CRC recovery if:
            - auto_crc is enabled
            - Number of traces >= crc_min_messages
            - CRC recovery succeeds with confidence > 0.8
        """
        if not self.auto_crc or len(traces) < self.crc_min_messages:
            return

        try:
            from oscura.inference.crc_reverse import CRCReverser

            # Convert traces to message-CRC pairs
            # For black-box analysis, we assume each trace is a complete message
            # and attempt to find CRC fields within each message
            byte_arrays = [self._trace_to_bytes(t) for t in traces]

            # Try to detect CRC location and extract message-CRC pairs
            # This is a heuristic: assume CRC is at the end (last 1-4 bytes)
            messages = []
            for data in byte_arrays:
                if len(data) >= 4:
                    # Try 2-byte CRC at end (most common)
                    messages.append((bytes(data[:-2]), bytes(data[-2:])))

            if len(messages) >= 4:
                reverser = CRCReverser(verbose=False)
                params = reverser.reverse(messages)

                if params is not None and params.confidence > 0.8:
                    self._crc_params = params
                    # Log successful recovery
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"Auto-recovered CRC: poly=0x{params.polynomial:04x}, "
                        f"width={params.width}, confidence={params.confidence:.2f}"
                    )
        except Exception:
            # CRC recovery is best-effort; don't fail if it doesn't work
            pass

    def _detect_crc(self, traces: list[Trace]) -> dict[str, Any]:
        """Detect CRC/checksums in traces.

        Args:
            traces: List of traces to analyze.

        Returns:
            Dictionary with CRC information if found.
        """
        if not traces:
            return {}

        # If CRC was auto-recovered, include in results
        if self._crc_params is not None:
            return {
                "polynomial": f"0x{self._crc_params.polynomial:04x}",
                "width": self._crc_params.width,
                "init": f"0x{self._crc_params.init:04x}",
                "xor_out": f"0x{self._crc_params.xor_out:04x}",
                "reflect_in": self._crc_params.reflect_in,
                "reflect_out": self._crc_params.reflect_out,
                "confidence": self._crc_params.confidence,
                "algorithm_name": self._crc_params.algorithm_name,
            }

        return {}

    def _export_report(self, path: Path) -> None:
        """Export Markdown analysis report."""
        if self._protocol_spec is None:
            self.analyze()

        report = f"# {self.name} - Analysis Report\n\n"
        report += f"**Generated**: {self.modified_at}\n\n"

        report += "## Recordings\n\n"
        for name in self.recordings:
            report += f"- {name}\n"
        report += "\n"

        report += "## Field Hypotheses\n\n"
        if self._field_hypotheses:
            report += "| Field | Offset | Length | Type | Confidence |\n"
            report += "|-------|--------|--------|------|------------|\n"
            for field_hyp in self._field_hypotheses:
                report += f"| {field_hyp.name} | {field_hyp.offset} | {field_hyp.length} | "
                report += f"{field_hyp.field_type} | {field_hyp.confidence:.2f} |\n"
        else:
            report += "No fields inferred.\n"
        report += "\n"

        path.write_text(report)

    def _export_dissector(self, path: Path) -> None:
        """Export Wireshark Lua dissector."""
        if self._protocol_spec is None:
            self.analyze()

        # Basic Lua dissector template
        dissector = f"-- Wireshark dissector for {self.name}\n"
        dissector += "-- Auto-generated by Oscura BlackBoxSession\n\n"
        dissector += (
            f"local proto = Proto('{self.name.lower().replace(' ', '_')}', '{self.name}')\n\n"
        )

        # Add fields
        if self._field_hypotheses:
            for field_hyp in self._field_hypotheses:
                dissector += (
                    f"-- {field_hyp.name} (offset={field_hyp.offset}, length={field_hyp.length})\n"
                )

        dissector += "\n-- TODO: Implement dissector logic (user should add Lua dissector code)\n"

        path.write_text(dissector)

    def _export_spec_json(self, path: Path) -> None:
        """Export protocol specification as JSON."""
        import json

        if self._protocol_spec is None:
            self.analyze()

        spec_dict = {
            "name": self._protocol_spec.name,  # type: ignore[union-attr]
            "fields": [
                {
                    "name": f.name,
                    "offset": f.offset,
                    "length": f.length,
                    "type": f.field_type,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                }
                for f in self._field_hypotheses
            ],
            "crc_info": self._protocol_spec.crc_info,  # type: ignore[union-attr]
            "constants": self._protocol_spec.constants,  # type: ignore[union-attr]
        }

        path.write_text(json.dumps(spec_dict, indent=2))

    def _export_json(self, path: Path) -> None:
        """Export complete results as JSON."""
        import json

        results = self.analyze()

        # Make JSON serializable
        json_results = {
            "num_recordings": results["num_recordings"],
            "field_hypotheses": [
                {
                    "name": f.name,
                    "offset": f.offset,
                    "length": f.length,
                    "type": f.field_type,
                    "confidence": f.confidence,
                }
                for f in results["field_hypotheses"]
            ],
        }

        path.write_text(json.dumps(json_results, indent=2))

    def _export_csv(self, path: Path) -> None:
        """Export field hypotheses as CSV."""
        if self._protocol_spec is None:
            self.analyze()

        csv = "field_name,offset,length,type,confidence\n"
        for field_hyp in self._field_hypotheses:
            csv += f"{field_hyp.name},{field_hyp.offset},{field_hyp.length},"
            csv += f"{field_hyp.field_type},{field_hyp.confidence:.3f}\n"

        path.write_text(csv)
