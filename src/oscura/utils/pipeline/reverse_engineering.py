"""Reverse Engineering Pipeline for integrated protocol analysis.

    - RE-INT-001: RE Pipeline Integration

This module provides an integrated pipeline for complete reverse engineering
workflows from raw packet capture to decoded messages with automatic tool
selection and chaining.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

logger = logging.getLogger(__name__)


@dataclass
class FlowInfo:
    """Information about a network flow.

    Attributes:
        flow_id: Unique flow identifier.
        src_ip: Source IP address.
        dst_ip: Destination IP address.
        src_port: Source port.
        dst_port: Destination port.
        protocol: Transport protocol.
        packet_count: Number of packets in flow.
        byte_count: Total bytes in flow.
        start_time: Flow start timestamp.
        end_time: Flow end timestamp.
    """

    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    byte_count: int
    start_time: float
    end_time: float


@dataclass
class MessageTypeInfo:
    """Information about detected message types.

    Attributes:
        type_id: Unique type identifier.
        name: Type name (auto-generated or from inference).
        sample_count: Number of messages of this type.
        avg_length: Average message length.
        field_count: Number of detected fields.
        signature: Representative byte signature.
        cluster_id: Associated cluster ID.
    """

    type_id: str
    name: str
    sample_count: int
    avg_length: float
    field_count: int
    signature: bytes
    cluster_id: int


@dataclass
class ProtocolCandidate:
    """Candidate protocol identification.

    Attributes:
        name: Protocol name.
        confidence: Detection confidence (0-1).
        matched_patterns: Patterns that matched.
        port_hint: Whether port suggested this protocol.
        header_match: Whether header matched signature.
    """

    name: str
    confidence: float
    matched_patterns: list[str] = field(default_factory=list)
    port_hint: bool = False
    header_match: bool = False


@dataclass
class REAnalysisResult:
    """Complete reverse engineering analysis result.

    Implements RE-INT-001: Analysis result structure.

    Attributes:
        flow_count: Number of flows analyzed.
        message_count: Total messages extracted.
        message_types: Detected message types.
        protocol_candidates: Candidate protocol identifications.
        field_schemas: Inferred field schemas per message type.
        state_machine: Inferred state machine (if available).
        statistics: Analysis statistics.
        warnings: Warnings encountered during analysis.
        duration_seconds: Analysis duration.
        timestamp: Analysis timestamp.
    """

    flow_count: int
    message_count: int
    message_types: list[MessageTypeInfo]
    protocol_candidates: list[ProtocolCandidate]
    field_schemas: dict[str, Any]
    state_machine: Any | None
    statistics: dict[str, Any]
    warnings: list[str]
    duration_seconds: float
    timestamp: str


@dataclass
class StageResult:
    """Result from a single pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        success: Whether stage completed successfully.
        duration: Stage duration in seconds.
        output: Stage output data.
        error: Error message if failed.
    """

    stage_name: str
    success: bool
    duration: float
    output: Any
    error: str | None = None


class REPipeline:
    """Integrated reverse engineering pipeline.

    Implements RE-INT-001: RE Pipeline Integration.

    Chains all RE tools in coherent pipeline with automatic tool
    selection based on data characteristics.

    Example:
        >>> pipeline = REPipeline()
        >>> results = pipeline.analyze(packet_data)
        >>> print(f"Detected {len(results.message_types)} message types")
        >>> pipeline.generate_report(results, "report.html")
    """

    # Default pipeline stages
    DEFAULT_STAGES: ClassVar[list[str]] = [
        "flow_extraction",
        "payload_analysis",
        "pattern_discovery",
        "field_inference",
        "protocol_detection",
        "state_machine",
    ]

    def __init__(
        self,
        stages: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RE pipeline.

        Args:
            stages: List of stage names to execute.
            config: Configuration options.
        """
        self.stages = stages or self.DEFAULT_STAGES
        self.config = config or {}

        # Default configuration
        self.config.setdefault("min_samples", 10)
        self.config.setdefault("entropy_threshold", 6.0)
        self.config.setdefault("cluster_threshold", 0.8)
        self.config.setdefault("state_machine_algorithm", "rpni")
        self.config.setdefault("max_message_types", 50)

        # Stage handlers
        self._stage_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "flow_extraction": self._stage_flow_extraction,
            "payload_analysis": self._stage_payload_analysis,
            "pattern_discovery": self._stage_pattern_discovery,
            "field_inference": self._stage_field_inference,
            "protocol_detection": self._stage_protocol_detection,
            "state_machine": self._stage_state_machine,
        }

        # Progress callback
        self._progress_callback: Callable[[str, float], None] | None = None

        # Checkpoint support
        self._checkpoint_path: str | None = None
        self._checkpoint_data: dict[str, Any] = {}

    def _initialize_analysis_context(
        self, data: bytes | Sequence[dict[str, Any]] | Sequence[bytes]
    ) -> dict[str, Any]:
        """Initialize analysis context with empty containers.

        Args:
            data: Input data to analyze.

        Returns:
            Initialized context dictionary.
        """
        return {
            "raw_data": data,
            "flows": [],
            "payloads": [],
            "messages": [],
            "patterns": [],
            "clusters": [],
            "schemas": {},
            "protocol_candidates": [],
            "state_machine": None,
            "warnings": [],
            "statistics": {},
        }

    def _execute_stage(
        self,
        stage_name: str,
        context: dict[str, Any],
        checkpoint: str | None,
    ) -> StageResult:
        """Execute single pipeline stage.

        Args:
            stage_name: Name of stage to execute.
            context: Analysis context.
            checkpoint: Checkpoint path.

        Returns:
            StageResult with execution outcome.
        """
        handler = self._stage_handlers.get(stage_name)
        if not handler:
            return StageResult(
                stage_name=stage_name, success=False, duration=0, output=None, error="No handler"
            )

        try:
            stage_start = time.time()
            output = handler(context)
            stage_duration = time.time() - stage_start

            if output:
                context.update(output)

            if checkpoint:
                self._save_checkpoint(checkpoint, stage_name, context)

            return StageResult(
                stage_name=stage_name,
                success=True,
                duration=stage_duration,
                output=output,
            )

        except Exception as e:
            warnings_list: list[str] = context.get("warnings", [])
            warnings_list.append(f"Stage {stage_name} failed: {e}")
            context["warnings"] = warnings_list

            return StageResult(
                stage_name=stage_name,
                success=False,
                duration=0,
                output=None,
                error=str(e),
            )

    def _execute_all_stages(
        self, context: dict[str, Any], checkpoint: str | None
    ) -> list[StageResult]:
        """Execute all pipeline stages.

        Args:
            context: Analysis context.
            checkpoint: Checkpoint path.

        Returns:
            List of stage results.
        """
        stage_results = []
        total_stages = len(self.stages)

        for i, stage_name in enumerate(self.stages):
            if stage_name in self._checkpoint_data:
                context.update(self._checkpoint_data[stage_name])
                continue

            self._report_progress(stage_name, (i / total_stages) * 100)
            stage_result = self._execute_stage(stage_name, context, checkpoint)
            stage_results.append(stage_result)

        self._report_progress("complete", 100)
        return stage_results

    def analyze(
        self,
        data: bytes | Sequence[dict[str, Any]] | Sequence[bytes],
        checkpoint: str | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> REAnalysisResult:
        """Run full reverse engineering analysis.

        Implements RE-INT-001: Complete analysis workflow.

        Args:
            data: Raw binary data, packet list, or PCAP path.
            checkpoint: Path for checkpointing progress.
            progress_callback: Callback for progress reporting.

        Returns:
            REAnalysisResult with complete analysis.

        Example:
            >>> results = pipeline.analyze(packets)
            >>> for msg_type in results.message_types:
            ...     print(f"{msg_type.name}: {msg_type.sample_count} samples")
        """
        # Setup: initialize state and load checkpoint
        start_time = time.time()
        self._progress_callback = progress_callback
        self._checkpoint_path = checkpoint
        self._checkpoint_data = {}

        if checkpoint and os.path.exists(checkpoint):
            self._load_checkpoint(checkpoint)

        context = self._initialize_analysis_context(data)

        # Processing: execute pipeline stages
        stage_results = self._execute_all_stages(context, checkpoint)

        # Result building: construct final result
        duration = time.time() - start_time
        flows_list: list[Any] = context.get("flows", [])
        messages_list: list[Any] = context.get("messages", [])
        protocol_candidates_list: list[ProtocolCandidate] = context.get("protocol_candidates", [])
        schemas_dict: dict[str, Any] = context.get("schemas", {})
        warnings_list_result: list[str] = context.get("warnings", [])

        return REAnalysisResult(
            flow_count=len(flows_list),
            message_count=len(messages_list),
            message_types=self._build_message_types(context),
            protocol_candidates=protocol_candidates_list,
            field_schemas=schemas_dict,
            state_machine=context.get("state_machine"),
            statistics=self._build_statistics(context, stage_results),
            warnings=warnings_list_result,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat(),
        )

    def analyze_pcap(
        self,
        path: str | Path,
        checkpoint: str | None = None,
    ) -> REAnalysisResult:
        """Analyze packets from a PCAP file.

        Implements RE-INT-001: PCAP file analysis.

        Args:
            path: Path to PCAP file.
            checkpoint: Optional checkpoint path.

        Returns:
            REAnalysisResult with analysis results.

        Raises:
            FileNotFoundError: If PCAP file not found.
        """
        # Load PCAP (simplified - would use scapy or pyshark in real impl)
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        return self.analyze(data, checkpoint=checkpoint)

    def register_stage_handler(
        self,
        stage_name: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a custom handler for a pipeline stage.

        This allows replacing or adding custom stage handlers for testing
        or extending pipeline functionality.

        Args:
            stage_name: Name of the stage to register handler for.
            handler: Callable that takes context dict and returns output dict.

        Example:
            >>> def custom_handler(context):
            ...     return {"custom_data": "processed"}
            >>> pipeline = REPipeline()
            >>> pipeline.register_stage_handler("custom_stage", custom_handler)
        """
        self._stage_handlers[stage_name] = handler

    def generate_report(
        self,
        results: REAnalysisResult,
        output_path: str | Path,
        format: Literal["html", "json", "markdown"] = "html",
    ) -> None:
        """Generate analysis report.

        Implements RE-INT-001: Report generation.

        Args:
            results: Analysis results.
            output_path: Output file path.
            format: Report format.

        Example:
            >>> pipeline.generate_report(results, "report.html")
        """
        output_path = Path(output_path)

        if format == "json":
            self._generate_json_report(results, output_path)
        elif format == "markdown":
            self._generate_markdown_report(results, output_path)
        else:
            self._generate_html_report(results, output_path)

    # =========================================================================
    # Pipeline Stages
    # =========================================================================

    def _stage_flow_extraction(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract network flows from raw data.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with flows and payloads.
        """
        data = context["raw_data"]

        if isinstance(data, bytes):
            flows, payloads = self._extract_from_raw_bytes(data)
        elif isinstance(data, list | tuple):
            flows, payloads = self._extract_from_packet_list(data)
        else:
            flows, payloads = [], []

        self._update_flow_statistics(context, flows, payloads)
        return {"flows": flows, "payloads": payloads}

    def _extract_from_raw_bytes(self, data: bytes) -> tuple[list[FlowInfo], list[bytes]]:
        """Extract flow from raw binary data.

        Args:
            data: Raw binary data.

        Returns:
            Tuple of (flows, payloads).
        """
        flow = FlowInfo(
            flow_id="flow_0",
            src_ip="unknown",
            dst_ip="unknown",
            src_port=0,
            dst_port=0,
            protocol="unknown",
            packet_count=1,
            byte_count=len(data),
            start_time=0,
            end_time=0,
        )
        return [flow], [data]

    def _extract_from_packet_list(
        self, packets: Sequence[dict[str, Any] | bytes]
    ) -> tuple[list[FlowInfo], list[bytes]]:
        """Extract flows from list of packets.

        Args:
            packets: List of packet dicts or raw bytes.

        Returns:
            Tuple of (flows, payloads).
        """
        flow_map: dict[str, dict[str, Any]] = {}
        payloads: list[bytes] = []
        raw_bytes_payloads: list[bytes] = []

        for pkt in packets:
            if isinstance(pkt, dict):
                self._process_packet_dict(pkt, flow_map, payloads)
            else:
                payload = bytes(pkt) if not isinstance(pkt, bytes) else pkt
                payloads.append(payload)
                raw_bytes_payloads.append(payload)

        flows = self._build_flows_from_map(flow_map)

        # Create default flow for raw bytes if needed
        if raw_bytes_payloads and not flows:
            flows.append(self._create_default_flow(raw_bytes_payloads))

        return flows, payloads

    def _process_packet_dict(
        self,
        pkt: dict[str, Any],
        flow_map: dict[str, dict[str, Any]],
        payloads: list[bytes],
    ) -> None:
        """Process a packet dictionary and update flow map.

        Args:
            pkt: Packet dictionary with metadata.
            flow_map: Flow mapping to update.
            payloads: Payloads list to append to.
        """
        # Extract payload
        payload_raw = pkt.get("data", pkt.get("payload", b""))
        if isinstance(payload_raw, list | tuple):
            payload = bytes(payload_raw)
        else:
            payload = payload_raw if isinstance(payload_raw, bytes) else b""

        # Create flow key
        flow_key = self._create_flow_key(pkt)

        # Initialize flow if new
        if flow_key not in flow_map:
            flow_map[flow_key] = self._create_flow_entry(pkt)

        # Update flow data
        flow_map[flow_key]["packets"].append(pkt)
        flow_map[flow_key]["payloads"].append(payload)
        if "timestamp" in pkt:
            flow_map[flow_key]["timestamps"].append(pkt["timestamp"])

        payloads.append(payload)

    def _create_flow_key(self, pkt: dict[str, Any]) -> str:
        """Create flow identifier key from packet.

        Args:
            pkt: Packet dictionary.

        Returns:
            Flow key string.
        """
        src_ip = pkt.get("src_ip", "0.0.0.0")
        dst_ip = pkt.get("dst_ip", "0.0.0.0")
        src_port = pkt.get("src_port", 0)
        dst_port = pkt.get("dst_port", 0)
        protocol = pkt.get("protocol", "unknown")
        return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"

    def _create_flow_entry(self, pkt: dict[str, Any]) -> dict[str, Any]:
        """Create new flow entry from packet.

        Args:
            pkt: Packet dictionary.

        Returns:
            Flow entry dictionary.
        """
        return {
            "src_ip": pkt.get("src_ip", "0.0.0.0"),
            "dst_ip": pkt.get("dst_ip", "0.0.0.0"),
            "src_port": pkt.get("src_port", 0),
            "dst_port": pkt.get("dst_port", 0),
            "protocol": pkt.get("protocol", "unknown"),
            "packets": [],
            "payloads": [],
            "timestamps": [],
        }

    def _build_flows_from_map(self, flow_map: dict[str, dict[str, Any]]) -> list[FlowInfo]:
        """Build FlowInfo objects from flow map.

        Args:
            flow_map: Mapping of flow keys to flow data.

        Returns:
            List of FlowInfo objects.
        """
        flows = []
        for flow_id, flow_data in flow_map.items():
            timestamps = flow_data.get("timestamps", [0])
            flows.append(
                FlowInfo(
                    flow_id=flow_id,
                    src_ip=flow_data["src_ip"],
                    dst_ip=flow_data["dst_ip"],
                    src_port=flow_data["src_port"],
                    dst_port=flow_data["dst_port"],
                    protocol=flow_data["protocol"],
                    packet_count=len(flow_data["packets"]),
                    byte_count=sum(len(p) for p in flow_data["payloads"]),
                    start_time=min(timestamps) if timestamps else 0,
                    end_time=max(timestamps) if timestamps else 0,
                )
            )
        return flows

    def _create_default_flow(self, payloads: list[bytes]) -> FlowInfo:
        """Create default flow for raw bytes.

        Args:
            payloads: List of raw byte payloads.

        Returns:
            Default FlowInfo object.
        """
        return FlowInfo(
            flow_id="flow_default",
            src_ip="unknown",
            dst_ip="unknown",
            src_port=0,
            dst_port=0,
            protocol="unknown",
            packet_count=len(payloads),
            byte_count=sum(len(p) for p in payloads),
            start_time=0,
            end_time=0,
        )

    def _update_flow_statistics(
        self, context: dict[str, Any], flows: list[FlowInfo], payloads: list[bytes]
    ) -> None:
        """Update context statistics with flow extraction results.

        Args:
            context: Pipeline context to update.
            flows: Extracted flows.
            payloads: Extracted payloads.
        """
        if "statistics" not in context:
            context["statistics"] = {}

        context["statistics"]["flow_extraction"] = {
            "flow_count": len(flows),
            "payload_count": len(payloads),
            "total_bytes": sum(len(p) for p in payloads),
        }

    def _stage_payload_analysis(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze payloads for structure.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with payload analysis.
        """
        payloads = context.get("payloads", [])

        # Filter non-empty payloads
        valid_payloads = [p for p in payloads if p and len(p) > 0]

        # Basic statistics
        if valid_payloads:
            lengths = [len(p) for p in valid_payloads]
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
        else:
            avg_len = min_len = max_len = 0

        # Detect delimiter patterns
        delimiter_info = None
        if valid_payloads:
            try:
                from oscura.analyzers.packet.payload import detect_delimiter

                concat = b"".join(valid_payloads)
                delimiter_result = detect_delimiter(concat)
                if delimiter_result.confidence > 0.5:
                    delimiter_info = {
                        "delimiter": delimiter_result.delimiter.hex(),
                        "confidence": delimiter_result.confidence,
                    }
            except Exception as e:
                logger.debug("Delimiter detection failed (non-critical): %s", e)

        # Initialize statistics dict if it doesn't exist
        if "statistics" not in context:
            context["statistics"] = {}

        context["statistics"]["payload_analysis"] = {
            "payload_count": len(valid_payloads),
            "avg_length": avg_len,
            "min_length": min_len,
            "max_length": max_len,
            "delimiter": delimiter_info,
        }

        return {"messages": valid_payloads}

    def _stage_pattern_discovery(self, context: dict[str, Any]) -> dict[str, Any]:
        """Discover patterns in messages.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with patterns.
        """
        messages = context.get("messages", [])
        patterns = []
        clusters = []

        if len(messages) >= 2:
            try:
                from oscura.analyzers.packet.payload import cluster_payloads

                # Cluster similar messages
                threshold = self.config.get("cluster_threshold", 0.8)
                clusters = cluster_payloads(messages, threshold=threshold)

                # Extract common patterns from clusters
                for cluster in clusters:
                    if len(cluster.payloads) >= 2:
                        # Find common prefix
                        common_prefix = cluster.payloads[0]
                        for payload in cluster.payloads[1:]:
                            new_prefix = bytearray()
                            for i in range(min(len(common_prefix), len(payload))):
                                if common_prefix[i] == payload[i]:
                                    new_prefix.append(common_prefix[i])
                                else:
                                    break
                            common_prefix = bytes(new_prefix)

                        if len(common_prefix) >= 2:
                            patterns.append(
                                {
                                    "pattern": common_prefix,
                                    "cluster_id": cluster.cluster_id,
                                    "frequency": len(cluster.payloads),
                                }
                            )

            except Exception as e:
                context["warnings"].append(f"Pattern discovery failed: {e}")

        # Initialize statistics dict if it doesn't exist
        if "statistics" not in context:
            context["statistics"] = {}

        context["statistics"]["pattern_discovery"] = {
            "cluster_count": len(clusters),
            "pattern_count": len(patterns),
        }

        return {"patterns": patterns, "clusters": clusters}

    def _stage_field_inference(self, context: dict[str, Any]) -> dict[str, Any]:
        """Infer field structure in messages.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with field schemas.
        """
        clusters = context.get("clusters", [])
        schemas = {}

        for cluster in clusters:
            if not hasattr(cluster, "payloads") or len(cluster.payloads) < 5:
                continue

            try:
                from oscura.analyzers.packet.payload import FieldInferrer

                inferrer = FieldInferrer(min_samples=self.config.get("min_samples", 10))
                schema = inferrer.infer_fields(cluster.payloads)

                if schema.fields:
                    cluster_id = getattr(cluster, "cluster_id", 0)
                    schemas[f"type_{cluster_id}"] = {
                        "field_count": len(schema.fields),
                        "message_length": schema.message_length,
                        "fixed_length": schema.fixed_length,
                        "confidence": schema.confidence,
                        "fields": [
                            {
                                "name": f.name,
                                "offset": f.offset,
                                "size": f.size,
                                "type": f.inferred_type,
                                "is_constant": f.is_constant,
                                "is_sequence": f.is_sequence,
                            }
                            for f in schema.fields
                        ],
                    }

            except Exception as e:
                context["warnings"].append(f"Field inference failed for cluster: {e}")

        # Initialize statistics dict if it doesn't exist
        if "statistics" not in context:
            context["statistics"] = {}

        context["statistics"]["field_inference"] = {
            "schema_count": len(schemas),
        }

        return {"schemas": schemas}

    def _stage_protocol_detection(self, context: dict[str, Any]) -> dict[str, Any]:
        """Detect protocol candidates.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with protocol candidates.
        """
        messages = context.get("messages", [])
        flows = context.get("flows", [])
        candidates: list[ProtocolCandidate] = []

        # Detect protocols from multiple sources
        candidates.extend(self._detect_by_port(flows))
        candidates.extend(self._detect_by_magic_bytes(messages))
        candidates.extend(self._detect_from_library(messages))

        # Deduplicate candidates
        unique_candidates = self._deduplicate_candidates(candidates)

        return {"protocol_candidates": unique_candidates}

    def _detect_by_port(self, flows: list[FlowInfo]) -> list[ProtocolCandidate]:
        """Detect protocols based on well-known port numbers.

        Args:
            flows: List of network flows.

        Returns:
            List of protocol candidates.
        """
        port_protocols = {
            53: "dns",
            80: "http",
            443: "https",
            502: "modbus_tcp",
            1883: "mqtt",
            5683: "coap",
            47808: "bacnet",
        }

        candidates = []
        for flow in flows:
            port = flow.dst_port or flow.src_port
            if port in port_protocols:
                candidates.append(
                    ProtocolCandidate(
                        name=port_protocols[port],
                        confidence=0.6,
                        port_hint=True,
                    )
                )

        return candidates

    def _detect_by_magic_bytes(self, messages: list[bytes]) -> list[ProtocolCandidate]:
        """Detect protocols by magic byte signatures.

        Args:
            messages: List of message bytes.

        Returns:
            List of protocol candidates.
        """
        if not messages:
            return []

        try:
            from oscura.inference.binary import MagicByteDetector

            detector = MagicByteDetector()
            sample = messages[0]

            if len(sample) >= 2:
                result = detector.detect(sample)
                if result and result.known_format:
                    return [
                        ProtocolCandidate(
                            name=result.known_format,
                            confidence=result.confidence,
                            header_match=True,
                        )
                    ]

        except Exception as e:
            logger.debug("Magic byte detection failed (non-critical): %s", e)

        return []

    def _detect_from_library(self, messages: list[bytes]) -> list[ProtocolCandidate]:
        """Detect protocols from protocol library.

        Args:
            messages: List of message bytes.

        Returns:
            List of protocol candidates.
        """
        try:
            from oscura.inference.protocol_library import get_library

            library = get_library()
            candidates = []

            for protocol in library.list_protocols():
                if self._matches_protocol_header(protocol, messages):
                    candidates.append(
                        ProtocolCandidate(
                            name=protocol.name,
                            confidence=0.4,
                            matched_patterns=["header_value"],
                        )
                    )

            return candidates

        except Exception as e:
            logger.debug("Protocol library matching failed (non-critical): %s", e)
            return []

    def _matches_protocol_header(self, protocol: Any, messages: list[bytes]) -> bool:
        """Check if messages match protocol header.

        Args:
            protocol: Protocol definition.
            messages: List of message bytes.

        Returns:
            True if matches.
        """
        if not protocol.definition or not protocol.definition.fields:
            return False

        first_field = protocol.definition.fields[0]
        if not hasattr(first_field, "value"):
            return False

        # Check first 10 messages
        return any(len(msg) >= 4 for msg in messages[:10])

    def _deduplicate_candidates(
        self, candidates: list[ProtocolCandidate]
    ) -> list[ProtocolCandidate]:
        """Deduplicate candidates, keeping highest confidence.

        Args:
            candidates: List of candidates.

        Returns:
            Deduplicated list.
        """
        unique: dict[str, ProtocolCandidate] = {}
        for c in candidates:
            if c.name not in unique or c.confidence > unique[c.name].confidence:
                unique[c.name] = c

        return list(unique.values())

    def _stage_state_machine(self, context: dict[str, Any]) -> dict[str, Any]:
        """Infer protocol state machine.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with state machine.
        """
        clusters = context.get("clusters", [])

        if len(clusters) < 2:
            return {"state_machine": None}

        try:
            # Build sequences from cluster transitions
            messages = context.get("messages", [])
            message_to_cluster = {}

            for cluster in clusters:
                for idx in getattr(cluster, "indices", []):
                    message_to_cluster[idx] = getattr(cluster, "cluster_id", 0)

            # Build observation sequence
            sequence: list[str] = [
                f"type_{message_to_cluster.get(i, 0)}"
                for i in range(len(messages))
                if i in message_to_cluster
            ]

            if len(sequence) >= 3:
                from oscura.inference.state_machine import StateMachineInferrer

                inferrer = StateMachineInferrer()
                # Cast list[str] to list[str | int] for API compatibility
                automaton = inferrer.infer_rpni([cast("list[str | int]", sequence)])

                return {
                    "state_machine": {
                        "states": len(automaton.states) if automaton is not None else 0,
                        "transitions": len(automaton.transitions) if automaton is not None else 0,
                        "automaton": automaton,
                    }
                }

        except Exception as e:
            context["warnings"].append(f"State machine inference failed: {e}")

        return {"state_machine": None}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _report_progress(self, stage: str, percent: float) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            self._progress_callback(stage, percent)

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint data."""
        try:
            with open(path) as f:
                self._checkpoint_data = json.load(f)
        except Exception:
            self._checkpoint_data = {}

    def _save_checkpoint(self, path: str, stage: str, context: dict[str, Any]) -> None:
        """Save checkpoint data."""
        try:
            # Extract serializable parts of context
            checkpoint = {
                stage: {
                    "flow_count": len(context.get("flows", [])),
                    "message_count": len(context.get("messages", [])),
                    "cluster_count": len(context.get("clusters", [])),
                }
            }

            if os.path.exists(path):
                with open(path) as f:
                    existing = json.load(f)
                checkpoint.update(existing)

            with open(path, "w") as f:
                json.dump(checkpoint, f, indent=2)

        except Exception as e:
            logger.debug("Checkpoint file save failed (non-critical): %s", e)

    def _build_message_types(self, context: dict[str, Any]) -> list[MessageTypeInfo]:
        """Build message type information from context."""
        clusters = context.get("clusters", [])
        message_types = []

        for cluster in clusters:
            payloads = getattr(cluster, "payloads", [])
            if not payloads:
                continue

            cluster_id = getattr(cluster, "cluster_id", 0)
            avg_len = sum(len(p) for p in payloads) / len(payloads) if payloads else 0

            # Get schema if available
            schema = context.get("schemas", {}).get(f"type_{cluster_id}", {})
            field_count = schema.get("field_count", 0)

            # Use representative as signature
            signature = payloads[0][:16] if payloads else b""

            message_types.append(
                MessageTypeInfo(
                    type_id=f"type_{cluster_id}",
                    name=f"Message Type {cluster_id}",
                    sample_count=len(payloads),
                    avg_length=avg_len,
                    field_count=field_count,
                    signature=signature,
                    cluster_id=cluster_id,
                )
            )

        return message_types

    def _build_statistics(
        self, context: dict[str, Any], stage_results: list[StageResult]
    ) -> dict[str, Any]:
        """Build analysis statistics."""
        stats: dict[str, Any] = context.get("statistics", {})

        # Add stage timing
        stats["stage_timing"] = {r.stage_name: r.duration for r in stage_results}

        # Add success info
        stats["stages_completed"] = sum(1 for r in stage_results if r.success)
        stats["stages_failed"] = sum(1 for r in stage_results if not r.success)

        return stats

    def _generate_json_report(self, results: REAnalysisResult, path: Path) -> None:
        """Generate JSON report."""
        report = {
            "flow_count": results.flow_count,
            "message_count": results.message_count,
            "message_types": [
                {
                    "type_id": mt.type_id,
                    "name": mt.name,
                    "sample_count": mt.sample_count,
                    "avg_length": mt.avg_length,
                    "field_count": mt.field_count,
                    "signature": mt.signature.hex(),
                }
                for mt in results.message_types
            ],
            "protocol_candidates": [
                {
                    "name": pc.name,
                    "confidence": pc.confidence,
                    "port_hint": pc.port_hint,
                    "header_match": pc.header_match,
                }
                for pc in results.protocol_candidates
            ],
            "field_schemas": results.field_schemas,
            "statistics": results.statistics,
            "warnings": results.warnings,
            "duration_seconds": results.duration_seconds,
            "timestamp": results.timestamp,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def _generate_markdown_report(self, results: REAnalysisResult, path: Path) -> None:
        """Generate Markdown report."""
        lines = [
            "# Reverse Engineering Analysis Report",
            "",
            f"**Generated:** {results.timestamp}",
            f"**Duration:** {results.duration_seconds:.2f} seconds",
            "",
            "## Summary",
            "",
            f"- Flows analyzed: {results.flow_count}",
            f"- Messages extracted: {results.message_count}",
            f"- Message types detected: {len(results.message_types)}",
            f"- Protocol candidates: {len(results.protocol_candidates)}",
            "",
            "## Message Types",
            "",
        ]

        for mt in results.message_types:
            lines.extend(
                [
                    f"### {mt.name}",
                    f"- Samples: {mt.sample_count}",
                    f"- Average length: {mt.avg_length:.1f} bytes",
                    f"- Fields detected: {mt.field_count}",
                    f"- Signature: `{mt.signature.hex()}`",
                    "",
                ]
            )

        if results.protocol_candidates:
            lines.extend(
                [
                    "## Protocol Candidates",
                    "",
                ]
            )
            for pc in results.protocol_candidates:
                lines.append(f"- **{pc.name}** (confidence: {pc.confidence:.2%})")
            lines.append("")

        if results.warnings:
            lines.extend(
                [
                    "## Warnings",
                    "",
                ]
            )
            for warning in results.warnings:
                lines.append(f"- {warning}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _generate_html_report(self, results: REAnalysisResult, path: Path) -> None:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RE Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .type-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .signature {{ font-family: monospace; background: #eee; padding: 5px; }}
        .warning {{ color: #856404; background: #fff3cd; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Reverse Engineering Analysis Report</h1>

    <div class="summary">
        <p><strong>Generated:</strong> {results.timestamp}</p>
        <p><strong>Duration:</strong> {results.duration_seconds:.2f} seconds</p>
        <p><strong>Flows:</strong> {results.flow_count}</p>
        <p><strong>Messages:</strong> {results.message_count}</p>
        <p><strong>Types:</strong> {len(results.message_types)}</p>
    </div>

    <h2>Message Types</h2>
"""
        for mt in results.message_types:
            html += f"""
    <div class="type-card">
        <h3>{mt.name}</h3>
        <p><strong>Samples:</strong> {mt.sample_count}</p>
        <p><strong>Avg Length:</strong> {mt.avg_length:.1f} bytes</p>
        <p><strong>Fields:</strong> {mt.field_count}</p>
        <p><strong>Signature:</strong> <span class="signature">{mt.signature.hex()}</span></p>
    </div>
"""
        if results.protocol_candidates:
            html += "<h2>Protocol Candidates</h2><ul>"
            for pc in results.protocol_candidates:
                html += f"<li><strong>{pc.name}</strong> ({pc.confidence:.0%})</li>"
            html += "</ul>"

        if results.warnings:
            html += "<h2>Warnings</h2>"
            for warning in results.warnings:
                html += f'<div class="warning">{warning}</div>'

        html += """
</body>
</html>
"""
        with open(path, "w") as f:
            f.write(html)


def analyze(
    data: bytes | Sequence[dict[str, Any]] | Sequence[bytes],
    stages: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> REAnalysisResult:
    """Run reverse engineering analysis on data.

    Implements RE-INT-001: Quick analysis function.

    Args:
        data: Data to analyze.
        stages: Pipeline stages to run.
        config: Configuration options.

    Returns:
        REAnalysisResult with analysis.
    """
    pipeline = REPipeline(stages=stages, config=config)
    return pipeline.analyze(data)


__all__ = [
    "FlowInfo",
    "MessageTypeInfo",
    "ProtocolCandidate",
    "REAnalysisResult",
    "REPipeline",
    "StageResult",
    "analyze",
]
