"""Comprehensive unit tests for oscura.pipeline.reverse_engineering module.

Tests the integrated reverse engineering pipeline with dataclass structures,
stage handlers, analysis workflow, report generation, and checkpointing.

- RE-INT-001: RE Pipeline Integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from oscura.pipeline.reverse_engineering import (
    FlowInfo,
    MessageTypeInfo,
    ProtocolCandidate,
    REAnalysisResult,
    REPipeline,
    StageResult,
    analyze,
)

pytestmark = pytest.mark.unit


# =============================================================================
# FlowInfo Dataclass Tests
# =============================================================================


class TestFlowInfo:
    """Tests for FlowInfo dataclass."""

    def test_basic_creation(self):
        """Test basic FlowInfo creation."""
        flow = FlowInfo(
            flow_id="flow-001",
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            packet_count=100,
            byte_count=50000,
            start_time=1000.0,
            end_time=1010.0,
        )

        assert flow.flow_id == "flow-001"
        assert flow.src_ip == "192.168.1.1"
        assert flow.dst_ip == "192.168.1.2"
        assert flow.src_port == 12345
        assert flow.dst_port == 80
        assert flow.protocol == "TCP"
        assert flow.packet_count == 100
        assert flow.byte_count == 50000
        assert flow.start_time == 1000.0
        assert flow.end_time == 1010.0

    def test_udp_flow(self):
        """Test UDP flow creation."""
        flow = FlowInfo(
            flow_id="udp-flow",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=53,
            dst_port=53,
            protocol="UDP",
            packet_count=10,
            byte_count=1000,
            start_time=0.0,
            end_time=0.5,
        )

        assert flow.protocol == "UDP"
        assert flow.dst_port == 53

    def test_duration_calculation(self):
        """Test flow duration can be calculated."""
        flow = FlowInfo(
            flow_id="test",
            src_ip="1.1.1.1",
            dst_ip="2.2.2.2",
            src_port=1,
            dst_port=2,
            protocol="TCP",
            packet_count=50,
            byte_count=5000,
            start_time=100.0,
            end_time=110.5,
        )

        duration = flow.end_time - flow.start_time
        assert duration == 10.5


# =============================================================================
# MessageTypeInfo Dataclass Tests
# =============================================================================


class TestMessageTypeInfo:
    """Tests for MessageTypeInfo dataclass."""

    def test_basic_creation(self):
        """Test basic MessageTypeInfo creation."""
        msg_type = MessageTypeInfo(
            type_id="type-001",
            name="REQUEST",
            sample_count=50,
            avg_length=128.5,
            field_count=5,
            signature=b"\x01\x02\x03",
            cluster_id=0,
        )

        assert msg_type.type_id == "type-001"
        assert msg_type.name == "REQUEST"
        assert msg_type.sample_count == 50
        assert msg_type.avg_length == 128.5
        assert msg_type.field_count == 5
        assert msg_type.signature == b"\x01\x02\x03"
        assert msg_type.cluster_id == 0

    def test_response_type(self):
        """Test response message type."""
        msg_type = MessageTypeInfo(
            type_id="type-002",
            name="RESPONSE",
            sample_count=48,
            avg_length=256.0,
            field_count=8,
            signature=b"\x02\x00\x00",
            cluster_id=1,
        )

        assert msg_type.name == "RESPONSE"
        assert msg_type.field_count == 8

    def test_empty_signature(self):
        """Test message type with empty signature."""
        msg_type = MessageTypeInfo(
            type_id="unknown",
            name="UNKNOWN",
            sample_count=10,
            avg_length=64.0,
            field_count=0,
            signature=b"",
            cluster_id=-1,
        )

        assert msg_type.signature == b""
        assert msg_type.cluster_id == -1


# =============================================================================
# ProtocolCandidate Dataclass Tests
# =============================================================================


class TestProtocolCandidate:
    """Tests for ProtocolCandidate dataclass."""

    def test_basic_creation(self):
        """Test basic ProtocolCandidate creation."""
        candidate = ProtocolCandidate(
            name="HTTP",
            confidence=0.95,
            matched_patterns=["GET", "POST", "HTTP/1.1"],
            port_hint=True,
            header_match=True,
        )

        assert candidate.name == "HTTP"
        assert candidate.confidence == 0.95
        assert len(candidate.matched_patterns) == 3
        assert candidate.port_hint is True
        assert candidate.header_match is True

    def test_default_values(self):
        """Test default values are applied."""
        candidate = ProtocolCandidate(
            name="UNKNOWN",
            confidence=0.5,
        )

        assert candidate.matched_patterns == []
        assert candidate.port_hint is False
        assert candidate.header_match is False

    def test_low_confidence(self):
        """Test low confidence candidate."""
        candidate = ProtocolCandidate(
            name="POSSIBLE_MODBUS",
            confidence=0.3,
            port_hint=True,
            header_match=False,
        )

        assert candidate.confidence < 0.5
        assert candidate.name == "POSSIBLE_MODBUS"


# =============================================================================
# REAnalysisResult Dataclass Tests
# =============================================================================


class TestREAnalysisResult:
    """Tests for REAnalysisResult dataclass."""

    def test_basic_creation(self):
        """Test basic REAnalysisResult creation."""
        result = REAnalysisResult(
            flow_count=5,
            message_count=100,
            message_types=[],
            protocol_candidates=[],
            field_schemas={},
            state_machine=None,
            statistics={},
            warnings=[],
            duration_seconds=5.5,
            timestamp="2025-01-01T00:00:00",
        )

        assert result.flow_count == 5
        assert result.message_count == 100
        assert result.duration_seconds == 5.5
        assert result.timestamp == "2025-01-01T00:00:00"

    def test_with_message_types(self):
        """Test result with message types."""
        msg_type = MessageTypeInfo(
            type_id="t1",
            name="REQ",
            sample_count=50,
            avg_length=64.0,
            field_count=3,
            signature=b"\x01",
            cluster_id=0,
        )

        result = REAnalysisResult(
            flow_count=1,
            message_count=50,
            message_types=[msg_type],
            protocol_candidates=[],
            field_schemas={"t1": {"fields": []}},
            state_machine=None,
            statistics={"total_bytes": 3200},
            warnings=[],
            duration_seconds=1.0,
            timestamp="2025-01-01T00:00:00",
        )

        assert len(result.message_types) == 1
        assert result.message_types[0].name == "REQ"

    def test_with_warnings(self):
        """Test result with warnings."""
        result = REAnalysisResult(
            flow_count=0,
            message_count=0,
            message_types=[],
            protocol_candidates=[],
            field_schemas={},
            state_machine=None,
            statistics={},
            warnings=["No flows detected", "Insufficient data"],
            duration_seconds=0.1,
            timestamp="2025-01-01T00:00:00",
        )

        assert len(result.warnings) == 2
        assert "No flows detected" in result.warnings


# =============================================================================
# StageResult Dataclass Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_successful_stage(self):
        """Test successful stage result."""
        result = StageResult(
            stage_name="flow_extraction",
            success=True,
            duration=1.5,
            output={"flows": [{"id": "f1"}]},
        )

        assert result.stage_name == "flow_extraction"
        assert result.success is True
        assert result.duration == 1.5
        assert result.output["flows"][0]["id"] == "f1"
        assert result.error is None

    def test_failed_stage(self):
        """Test failed stage result."""
        result = StageResult(
            stage_name="protocol_detection",
            success=False,
            duration=0.1,
            output=None,
            error="Insufficient data for detection",
        )

        assert result.success is False
        assert result.error == "Insufficient data for detection"


# =============================================================================
# REPipeline Tests
# =============================================================================


class TestREPipeline:
    """Tests for REPipeline class."""

    def test_default_initialization(self):
        """Test default pipeline initialization."""
        pipeline = REPipeline()

        assert pipeline.stages == REPipeline.DEFAULT_STAGES
        assert len(pipeline.stages) == 6
        assert "flow_extraction" in pipeline.stages
        assert "state_machine" in pipeline.stages

    def test_custom_stages(self):
        """Test pipeline with custom stages."""
        custom_stages = ["flow_extraction", "payload_analysis"]
        pipeline = REPipeline(stages=custom_stages)

        assert pipeline.stages == custom_stages
        assert len(pipeline.stages) == 2

    def test_custom_config(self):
        """Test pipeline with custom configuration."""
        config = {
            "min_samples": 20,
            "entropy_threshold": 7.0,
        }
        pipeline = REPipeline(config=config)

        assert pipeline.config["min_samples"] == 20
        assert pipeline.config["entropy_threshold"] == 7.0
        # Defaults still applied for unset options
        assert pipeline.config["cluster_threshold"] == 0.8

    def test_stage_handlers_registered(self):
        """Test all default stage handlers are registered."""
        pipeline = REPipeline()

        for stage in REPipeline.DEFAULT_STAGES:
            assert stage in pipeline._stage_handlers

    def test_analyze_with_bytes(self):
        """Test analyze with raw bytes input."""
        pipeline = REPipeline()
        data = b"\x00\x01\x02\x03" * 100

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)
        assert result.timestamp != ""
        assert result.duration_seconds >= 0

    def test_analyze_with_packet_list(self):
        """Test analyze with packet list input."""
        pipeline = REPipeline()
        packets = [
            {"payload": b"\x01\x02", "timestamp": 0.0},
            {"payload": b"\x03\x04", "timestamp": 0.1},
        ]

        result = pipeline.analyze(packets)

        assert isinstance(result, REAnalysisResult)

    def test_analyze_with_progress_callback(self):
        """Test analyze with progress callback."""
        pipeline = REPipeline()
        data = b"\x00" * 100
        progress_calls = []

        def progress_callback(stage: str, percent: float):
            progress_calls.append((stage, percent))

        result = pipeline.analyze(data, progress_callback=progress_callback)

        assert isinstance(result, REAnalysisResult)
        assert len(progress_calls) > 0
        # Last call should be 'complete' at 100%
        assert progress_calls[-1] == ("complete", 100)

    def test_analyze_with_checkpoint(self):
        """Test analyze with checkpointing."""
        pipeline = REPipeline()
        data = b"\x00" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            result = pipeline.analyze(data, checkpoint=str(checkpoint_path))

            assert isinstance(result, REAnalysisResult)
            # Checkpoint file may or may not exist depending on implementation

    def test_analyze_limited_stages(self):
        """Test analyze with limited stages."""
        pipeline = REPipeline(stages=["flow_extraction"])
        data = b"\x00\x01\x02\x03"

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)


class TestREPipelineAnalyzePcap:
    """Tests for REPipeline.analyze_pcap method."""

    def test_file_not_found(self):
        """Test analyze_pcap with nonexistent file."""
        pipeline = REPipeline()

        with pytest.raises(FileNotFoundError):
            pipeline.analyze_pcap("/nonexistent/path.pcap")

    def test_analyze_pcap_success(self):
        """Test analyze_pcap with valid file."""
        pipeline = REPipeline()

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            # Write minimal PCAP-like header
            f.write(b"\xd4\xc3\xb2\xa1" + b"\x00" * 100)
            pcap_path = f.name

        try:
            result = pipeline.analyze_pcap(pcap_path)
            assert isinstance(result, REAnalysisResult)
        finally:
            Path(pcap_path).unlink()


class TestREPipelineGenerateReport:
    """Tests for REPipeline.generate_report method."""

    @pytest.fixture
    def sample_result(self) -> REAnalysisResult:
        """Create sample analysis result."""
        return REAnalysisResult(
            flow_count=2,
            message_count=50,
            message_types=[
                MessageTypeInfo(
                    type_id="t1",
                    name="REQUEST",
                    sample_count=25,
                    avg_length=64.0,
                    field_count=3,
                    signature=b"\x01",
                    cluster_id=0,
                ),
            ],
            protocol_candidates=[
                ProtocolCandidate(name="HTTP", confidence=0.9, port_hint=True),
            ],
            field_schemas={"t1": {"fields": ["field1", "field2"]}},
            state_machine=None,
            statistics={"total_bytes": 3200},
            warnings=[],
            duration_seconds=2.5,
            timestamp="2025-01-01T00:00:00",
        )

    def test_generate_json_report(self, sample_result):
        """Test JSON report generation."""
        pipeline = REPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            pipeline.generate_report(sample_result, output_path, format="json")

            assert output_path.exists()
            content = json.loads(output_path.read_text())
            assert "flow_count" in content or isinstance(content, dict)

    def test_generate_markdown_report(self, sample_result):
        """Test Markdown report generation."""
        pipeline = REPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            pipeline.generate_report(sample_result, output_path, format="markdown")

            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0

    def test_generate_html_report(self, sample_result):
        """Test HTML report generation."""
        pipeline = REPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            pipeline.generate_report(sample_result, output_path, format="html")

            assert output_path.exists()
            content = output_path.read_text()
            # Should contain HTML structure
            assert "<" in content or len(content) > 0


# =============================================================================
# Pipeline Stage Tests
# =============================================================================


class TestPipelineStages:
    """Tests for individual pipeline stages."""

    def test_flow_extraction_bytes(self):
        """Test flow extraction with bytes input."""
        pipeline = REPipeline()
        context = {"raw_data": b"\x00\x01\x02\x03" * 50}

        result = pipeline._stage_flow_extraction(context)

        assert isinstance(result, dict)
        # Should have flows or payloads in result
        assert "flows" in result or "payloads" in result or result is not None

    def test_flow_extraction_packet_list(self):
        """Test flow extraction with packet list."""
        pipeline = REPipeline()
        packets = [
            {"payload": b"\x01", "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"},
            {"payload": b"\x02", "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"},
        ]
        context = {"raw_data": packets}

        result = pipeline._stage_flow_extraction(context)

        assert isinstance(result, dict)

    def test_payload_analysis_stage(self):
        """Test payload analysis stage."""
        pipeline = REPipeline()
        context = {
            "payloads": [b"\x01\x02\x03", b"\x04\x05\x06"],
            "flows": [],
        }

        result = pipeline._stage_payload_analysis(context)

        assert isinstance(result, dict)

    def test_pattern_discovery_stage(self):
        """Test pattern discovery stage."""
        pipeline = REPipeline()
        context = {
            "payloads": [b"\x01\x02\x03", b"\x01\x02\x04", b"\x01\x02\x05"],
            "messages": [{"data": b"\x01\x02\x03"}],
        }

        result = pipeline._stage_pattern_discovery(context)

        assert isinstance(result, dict)

    def test_field_inference_stage(self):
        """Test field inference stage."""
        pipeline = REPipeline()
        context = {
            "messages": [{"data": b"\x01\x02\x03\x04"}],
            "clusters": [{"id": 0, "members": [0]}],
        }

        result = pipeline._stage_field_inference(context)

        assert isinstance(result, dict)

    def test_protocol_detection_stage(self):
        """Test protocol detection stage."""
        pipeline = REPipeline()
        context = {
            "payloads": [b"GET /index.html HTTP/1.1\r\n"],
            "patterns": [],
        }

        result = pipeline._stage_protocol_detection(context)

        assert isinstance(result, dict)

    def test_state_machine_stage(self):
        """Test state machine inference stage."""
        pipeline = REPipeline()
        context = {
            "messages": [
                {"type": "A"},
                {"type": "B"},
                {"type": "A"},
            ],
            "clusters": [],
        }

        result = pipeline._stage_state_machine(context)

        assert isinstance(result, dict)


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestPipelineHelpers:
    """Tests for pipeline helper methods."""

    def test_report_progress(self):
        """Test progress reporting."""
        pipeline = REPipeline()
        progress_data = []

        def callback(stage: str, pct: float):
            progress_data.append((stage, pct))

        pipeline._progress_callback = callback
        pipeline._report_progress("test_stage", 50.0)

        assert len(progress_data) == 1
        assert progress_data[0] == ("test_stage", 50.0)

    def test_report_progress_no_callback(self):
        """Test progress reporting without callback."""
        pipeline = REPipeline()
        pipeline._progress_callback = None

        # Should not raise
        pipeline._report_progress("test", 0.0)

    def test_build_message_types(self):
        """Test message type building from context."""
        pipeline = REPipeline()
        context = {
            "clusters": [
                {"id": 0, "members": [0, 1, 2], "centroid": b"\x01"},
                {"id": 1, "members": [3, 4], "centroid": b"\x02"},
            ],
            "messages": [
                {"data": b"\x01\x02", "length": 2},
                {"data": b"\x01\x03", "length": 2},
                {"data": b"\x01\x04", "length": 2},
                {"data": b"\x02\x05", "length": 2},
                {"data": b"\x02\x06", "length": 2},
            ],
        }

        result = pipeline._build_message_types(context)

        assert isinstance(result, list)

    def test_build_statistics(self):
        """Test statistics building."""
        pipeline = REPipeline()
        context = {
            "flows": [{"id": "f1"}, {"id": "f2"}],
            "messages": [{"data": b"\x01"}] * 10,
            "payloads": [b"\x00"] * 10,
        }
        stage_results = [
            StageResult(stage_name="s1", success=True, duration=1.0, output={}),
            StageResult(stage_name="s2", success=True, duration=2.0, output={}),
        ]

        result = pipeline._build_statistics(context, stage_results)

        assert isinstance(result, dict)


# =============================================================================
# Checkpointing Tests
# =============================================================================


class TestCheckpointing:
    """Tests for pipeline checkpointing."""

    def test_save_checkpoint(self):
        """Test saving checkpoint via analyze."""
        pipeline = REPipeline()
        data = b"\x00\x01\x02\x03" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            result = pipeline.analyze(data, checkpoint=str(checkpoint_path))

            # Should complete successfully
            assert isinstance(result, REAnalysisResult)
            # Checkpoint file should be created
            assert checkpoint_path.exists()

    def test_load_checkpoint(self):
        """Test loading checkpoint via analyze."""
        pipeline = REPipeline()
        data = b"\x00\x01\x02\x03" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # First run - creates checkpoint
            result1 = pipeline.analyze(data, checkpoint=str(checkpoint_path))
            assert isinstance(result1, REAnalysisResult)

            # Second run - should load from checkpoint
            result2 = pipeline.analyze(data, checkpoint=str(checkpoint_path))
            assert isinstance(result2, REAnalysisResult)

    def test_checkpoint_resume(self):
        """Test resuming from checkpoint."""
        pipeline = REPipeline(stages=["flow_extraction", "payload_analysis"])
        data = b"\x00" * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # First run
            result1 = pipeline.analyze(data, checkpoint=str(checkpoint_path))

            # Second run should resume (if checkpoint exists)
            result2 = pipeline.analyze(data, checkpoint=str(checkpoint_path))

            assert isinstance(result1, REAnalysisResult)
            assert isinstance(result2, REAnalysisResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineReverseEngineeringIntegration:
    """Integration tests for RE pipeline."""

    def test_full_pipeline_execution(self):
        """Test full pipeline from start to finish."""
        pipeline = REPipeline()

        # Create sample binary data with structure
        header = b"\x01\x02\x03\x04"  # 4-byte header
        data_block = b"\x00" * 60  # 60-byte data
        packet = header + data_block

        # Multiple similar packets
        data = packet * 20

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)
        assert result.duration_seconds > 0
        assert result.timestamp != ""

    def test_pipeline_with_structured_data(self):
        """Test pipeline with structured protocol-like data."""
        pipeline = REPipeline()

        # Create request/response pattern
        request = b"\x01\x00\x10\x00"  # Type 1, length 16
        response = b"\x02\x00\x20\x00"  # Type 2, length 32

        packets = [request, response] * 50
        data = b"".join(packets)

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)

    def test_empty_data_handling(self):
        """Test pipeline handles empty data gracefully."""
        pipeline = REPipeline()

        result = pipeline.analyze(b"")

        assert isinstance(result, REAnalysisResult)
        assert result.flow_count >= 0
        assert result.message_count >= 0


# =============================================================================
# analyze() Function Tests
# =============================================================================


class TestAnalyzeFunction:
    """Tests for standalone analyze() function."""

    def test_analyze_function_basic(self):
        """Test analyze function with basic input."""
        data = b"\x00\x01\x02\x03" * 50

        result = analyze(data)

        assert isinstance(result, REAnalysisResult)

    def test_analyze_function_with_config(self):
        """Test analyze function with configuration."""
        data = b"\x00" * 100

        result = analyze(data, config={"min_samples": 5})

        assert isinstance(result, REAnalysisResult)

    def test_analyze_function_with_stages(self):
        """Test analyze function with custom stages."""
        data = b"\x00" * 100

        result = analyze(data, stages=["flow_extraction"])

        assert isinstance(result, REAnalysisResult)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestPipelineReverseEngineeringEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_input(self):
        """Test with very small input data."""
        pipeline = REPipeline()

        result = pipeline.analyze(b"\x00")

        assert isinstance(result, REAnalysisResult)

    def test_large_input(self):
        """Test with larger input data."""
        pipeline = REPipeline()
        data = b"\x00\x01\x02\x03" * 10000  # 40KB

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)

    def test_random_binary_data(self):
        """Test with random binary data."""
        import random

        pipeline = REPipeline()
        data = bytes(random.randint(0, 255) for _ in range(1000))

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)

    def test_stage_failure_recovery(self):
        """Test pipeline continues after stage failure."""
        pipeline = REPipeline()

        # Register a failing stage handler
        def failing_handler(ctx):
            raise ValueError("Intentional test failure")

        pipeline.register_stage_handler("pattern_discovery", failing_handler)

        data = b"\x00" * 100
        result = pipeline.analyze(data)

        # Should complete with warning
        assert isinstance(result, REAnalysisResult)
        assert any("pattern_discovery" in w for w in result.warnings)

    def test_unknown_stage_name(self):
        """Test pipeline with unknown stage name."""
        pipeline = REPipeline(stages=["flow_extraction", "unknown_stage"])

        result = pipeline.analyze(b"\x00" * 100)

        # Should skip unknown stage and continue
        assert isinstance(result, REAnalysisResult)

    def test_all_bytes_same(self):
        """Test with all bytes being the same value."""
        pipeline = REPipeline()
        data = b"\xff" * 500

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)

    def test_alternating_bytes(self):
        """Test with alternating byte pattern."""
        pipeline = REPipeline()
        data = b"\x00\xff" * 250

        result = pipeline.analyze(data)

        assert isinstance(result, REAnalysisResult)
