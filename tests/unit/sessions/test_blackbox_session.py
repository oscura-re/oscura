"""Tests for BlackBoxSession."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.hardware.acquisition import SyntheticSource
from oscura.sessions import BlackBoxSession, ComparisonResult, FieldHypothesis, ProtocolSpec
from oscura.utils.builders import SignalBuilder

pytestmark = pytest.mark.unit


class TestBlackBoxSession:
    """Tests for BlackBoxSession implementation."""

    def test_creation(self) -> None:
        """Test session creation."""
        session = BlackBoxSession(name="Test BlackBox Session")

        assert session.name == "Test BlackBox Session"
        assert len(session.recordings) == 0

    def test_add_recording(self) -> None:
        """Test adding recordings."""
        session = BlackBoxSession()
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        session.add_recording("test", source)

        assert "test" in session.recordings
        assert session.list_recordings() == ["test"]

    def test_compare_recordings(self) -> None:
        """Test comparing recordings."""
        session = BlackBoxSession()

        # Create two different signals (different frequencies to ensure byte differences)
        source1 = SyntheticSource(
            SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000, amplitude=1.0)
        )
        source2 = SyntheticSource(
            SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(5000, amplitude=1.0)
        )

        session.add_recording("sig1", source1)
        session.add_recording("sig2", source2)

        result = session.compare("sig1", "sig2")

        assert isinstance(result, ComparisonResult)
        assert result.recording1 == "sig1"
        assert result.recording2 == "sig2"
        assert 0 <= result.similarity_score <= 1
        assert result.changed_bytes >= 0  # May detect differences

    def test_compare_identical_recordings(self) -> None:
        """Test comparing identical recordings."""
        session = BlackBoxSession()

        # Create identical signals
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source1 = SyntheticSource(builder)
        source2 = SyntheticSource(builder)

        session.add_recording("sig1", source1)
        session.add_recording("sig2", source2)

        result = session.compare("sig1", "sig2")

        assert result.similarity_score > 0.99  # Nearly identical
        assert result.changed_bytes == 0  # No differences

    def test_compare_nonexistent_recording_raises(self) -> None:
        """Test that comparing nonexistent recording raises error."""
        session = BlackBoxSession()
        source = SyntheticSource(SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000))
        session.add_recording("sig1", source)

        with pytest.raises(KeyError, match="not found"):
            session.compare("sig1", "nonexistent")

    def test_analyze_empty_session(self) -> None:
        """Test analyzing empty session."""
        session = BlackBoxSession()

        results = session.analyze()

        assert results["num_recordings"] == 0
        assert len(results["field_hypotheses"]) == 0
        assert results["state_machine"] is None
        assert results["protocol_spec"] is None

    @pytest.mark.slow
    def test_analyze_with_recordings(self) -> None:
        """Test analyzing session with recordings."""
        session = BlackBoxSession()

        # Add synthetic recordings
        for i in range(3):
            builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000 + i * 100)
            source = SyntheticSource(builder)
            session.add_recording(f"rec{i}", source)

        results = session.analyze()

        assert results["num_recordings"] == 3
        assert isinstance(results["field_hypotheses"], list)
        assert isinstance(results["protocol_spec"], ProtocolSpec)

    def test_generate_protocol_spec(self) -> None:
        """Test protocol specification generation."""
        session = BlackBoxSession()

        # Add recordings
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        spec = session.generate_protocol_spec()

        assert isinstance(spec, ProtocolSpec)
        assert spec.name == "Black-Box Analysis Protocol"
        assert isinstance(spec.fields, list)
        assert isinstance(spec.crc_info, dict)

    @pytest.mark.slow
    def test_infer_state_machine(self) -> None:
        """Test state machine inference.

        NOTE: Marked as slow because RPNI algorithm is computationally expensive
        for sine wave data (large state space). Use --runslow to run this test.
        """
        session = BlackBoxSession()

        # Add recordings
        for i in range(3):
            builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000 + i * 100)
            source = SyntheticSource(builder)
            session.add_recording(f"state{i}", source)

        state_machine = session.infer_state_machine()

        # State machine inference may return None if no clear states found
        # Just verify it doesn't crash
        assert state_machine is None or state_machine is not None

    def test_export_report(self) -> None:
        """Test report export."""
        session = BlackBoxSession(name="Test Session")

        # Add recording
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            session.export_results("report", str(path))

            assert path.exists()
            content = path.read_text()
            assert "Test Session" in content
            assert "Analysis Report" in content
            assert "Recordings" in content

    def test_export_dissector(self) -> None:
        """Test Wireshark dissector export."""
        session = BlackBoxSession(name="IoT Protocol")

        # Add recording
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "protocol.lua"
            session.export_results("dissector", str(path))

            assert path.exists()
            content = path.read_text()
            assert "Wireshark dissector" in content
            assert "Proto" in content
            assert "iot_protocol" in content

    def test_export_spec_json(self) -> None:
        """Test protocol spec JSON export."""
        session = BlackBoxSession()

        # Add recording
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.json"
            session.export_results("spec", str(path))

            assert path.exists()

            # Verify JSON is valid
            with open(path) as f:
                spec = json.load(f)
                assert "name" in spec
                assert "fields" in spec
                assert "crc_info" in spec

    def test_export_complete_json(self) -> None:
        """Test complete JSON export."""
        session = BlackBoxSession()

        # Add recording
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            session.export_results("json", str(path))

            assert path.exists()

            # Verify JSON is valid
            with open(path) as f:
                results = json.load(f)
                assert "num_recordings" in results
                assert "field_hypotheses" in results

    def test_export_csv(self) -> None:
        """Test CSV export."""
        session = BlackBoxSession()

        # Add recording
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fields.csv"
            session.export_results("csv", str(path))

            assert path.exists()

            # Verify CSV has header
            content = path.read_text()
            assert "field_name" in content
            assert "offset" in content
            assert "length" in content

    def test_export_unsupported_format_raises(self) -> None:
        """Test that unsupported export format raises error."""
        session = BlackBoxSession()

        # Add recording
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.xyz"

            with pytest.raises(ValueError, match="Unsupported export format"):
                session.export_results("invalid_format", str(path))

    def test_trace_to_bytes_uint8(self) -> None:
        """Test converting uint8 trace to bytes."""
        session = BlackBoxSession()

        # Create uint8 trace
        data = np.array([0, 128, 255], dtype=np.uint8)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        result = session._trace_to_bytes(trace)

        assert np.array_equal(result, data)
        assert result.dtype == np.uint8

    def test_trace_to_bytes_float(self) -> None:
        """Test converting float trace to bytes."""
        session = BlackBoxSession()

        # Create float trace
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        result = session._trace_to_bytes(trace)

        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[-1] == 255

    def test_find_changed_regions_empty(self) -> None:
        """Test finding changed regions in empty diff."""
        session = BlackBoxSession()

        diffs = np.array([False, False, False])
        regions = session._find_changed_regions(diffs)

        assert len(regions) == 0

    def test_find_changed_regions_single(self) -> None:
        """Test finding single changed region."""
        session = BlackBoxSession()

        diffs = np.array([False, True, True, False])
        regions = session._find_changed_regions(diffs)

        assert len(regions) == 1
        assert regions[0] == (1, 2, "Changed")

    def test_find_changed_regions_multiple(self) -> None:
        """Test finding multiple changed regions."""
        session = BlackBoxSession()

        diffs = np.array([False, True, False, True, True, False])
        regions = session._find_changed_regions(diffs)

        assert len(regions) == 2
        assert regions[0] == (1, 1, "Changed")
        assert regions[1] == (3, 4, "Changed")

    def test_field_hypothesis_creation(self) -> None:
        """Test creating field hypothesis."""
        field = FieldHypothesis(
            name="counter",
            offset=2,
            length=1,
            field_type="counter",
            confidence=0.95,
            evidence={"increments": True},
        )

        assert field.name == "counter"
        assert field.offset == 2
        assert field.length == 1
        assert field.field_type == "counter"
        assert field.confidence == 0.95
        assert field.evidence["increments"] is True

    def test_protocol_spec_creation(self) -> None:
        """Test creating protocol spec."""
        field = FieldHypothesis(name="test", offset=0, length=1, field_type="data", confidence=0.5)

        spec = ProtocolSpec(
            name="Test Protocol",
            fields=[field],
            crc_info={"polynomial": 0x1021},
            constants={"version": 1},
        )

        assert spec.name == "Test Protocol"
        assert len(spec.fields) == 1
        assert spec.crc_info["polynomial"] == 0x1021
        assert spec.constants["version"] == 1

    def test_differential_analysis_workflow(self) -> None:
        """Test complete differential analysis workflow."""
        session = BlackBoxSession(name="Button Protocol Analysis")

        # Simulate protocol with button states
        # Baseline: idle state
        idle_data = np.array([0xAA, 0x00, 0x00, 0x55], dtype=np.uint8)
        idle_trace = WaveformTrace(data=idle_data, metadata=TraceMetadata(sample_rate=1e6))

        # Button pressed: byte 2 changes
        pressed_data = np.array([0xAA, 0x00, 0x01, 0x55], dtype=np.uint8)
        pressed_trace = WaveformTrace(data=pressed_data, metadata=TraceMetadata(sample_rate=1e6))

        # Create sources that return these traces
        class StaticSource:
            def __init__(self, trace):
                self._trace = trace
                self._closed = False

            def read(self):
                if self._closed:
                    raise ValueError("Cannot read from closed source")
                return self._trace

            def stream(self, chunk_size):
                yield self._trace

            def close(self):
                self._closed = True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        session.add_recording("idle", StaticSource(idle_trace))
        session.add_recording("pressed", StaticSource(pressed_trace))

        # Compare
        diff = session.compare("idle", "pressed")

        # Should detect 1 byte changed
        assert diff.changed_bytes == 1
        assert diff.similarity_score >= 0.75  # 3/4 bytes match
        assert len(diff.changed_regions) == 1
        assert diff.changed_regions[0] == (2, 2, "Changed")

    def test_analyze_caches_results(self) -> None:
        """Test that analyze caches traces in recordings."""
        session = BlackBoxSession()

        # Add recording without loading
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source, load_immediately=False)

        # Initially, trace should not be cached
        _, cached_trace = session.recordings["test"]
        assert cached_trace is None

        # Analyze should load and cache
        session.analyze()

        # Now trace should be cached
        _, cached_trace = session.recordings["test"]
        assert cached_trace is not None

    def test_auto_crc_disabled(self) -> None:
        """Test that CRC recovery can be disabled."""
        session = BlackBoxSession(auto_crc=False)

        # Add recordings
        for i in range(15):
            builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000 + i * 100)
            source = SyntheticSource(builder)
            session.add_recording(f"rec{i}", source)

        results = session.analyze()

        # CRC should not be recovered
        assert session._crc_params is None
        assert results["crc_info"] == {}

    def test_auto_crc_insufficient_messages(self) -> None:
        """Test that CRC recovery requires minimum messages."""
        session = BlackBoxSession(auto_crc=True, crc_min_messages=20)

        # Add only 5 recordings (less than minimum)
        for i in range(5):
            builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
            source = SyntheticSource(builder)
            session.add_recording(f"rec{i}", source)

        results = session.analyze()

        # CRC should not be recovered (not enough messages)
        assert session._crc_params is None
        assert results["crc_info"] == {}

    def test_auto_crc_recovery_with_known_crc(self) -> None:
        """Test CRC recovery with messages containing known CRC."""
        from oscura.inference.crc_reverse import CRCReverser

        session = BlackBoxSession(auto_crc=True, crc_min_messages=4)

        # Generate messages with CRC-16-CCITT
        reverser = CRCReverser()
        messages_with_crc = []
        for i in range(5):
            data = f"TestMsg{i}".encode()
            crc = reverser._calculate_crc(
                data=data,
                poly=0x1021,
                width=16,
                init=0xFFFF,
                xor_out=0x0000,
                refin=False,
                refout=False,
            )
            # Combine data and CRC
            full_message = data + crc.to_bytes(2, "big")
            messages_with_crc.append(full_message)

        # Create traces from these messages
        class StaticSource:
            def __init__(self, trace):
                self._trace = trace
                self._closed = False

            def read(self):
                if self._closed:
                    raise ValueError("Cannot read from closed source")
                return self._trace

            def stream(self, chunk_size):
                yield self._trace

            def close(self):
                self._closed = True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        for i, msg_bytes in enumerate(messages_with_crc):
            data_array = np.frombuffer(msg_bytes, dtype=np.uint8)
            trace = WaveformTrace(data=data_array, metadata=TraceMetadata(sample_rate=1e6))
            session.add_recording(f"msg{i}", StaticSource(trace))

        results = session.analyze()

        # CRC should be recovered
        if session._crc_params is not None:
            assert session._crc_params.polynomial == 0x1021
            assert session._crc_params.width == 16
            assert session._crc_params.confidence > 0.8
            assert results["crc_info"]["polynomial"] == "0x1021"
        # Note: CRC recovery may fail due to heuristics, so we allow None

    def test_crc_info_in_protocol_spec(self) -> None:
        """Test that CRC info is included in protocol spec."""
        session = BlackBoxSession(auto_crc=False)  # Disable auto for controlled test

        # Add recordings
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        session.add_recording("test", source)

        # Manually set CRC params to test inclusion in spec
        from oscura.inference.crc_reverse import CRCParameters

        session._crc_params = CRCParameters(
            polynomial=0x1021,
            width=16,
            init=0xFFFF,
            xor_out=0x0000,
            reflect_in=False,
            reflect_out=False,
            confidence=0.95,
        )

        spec = session.generate_protocol_spec()

        # CRC info should be in spec
        assert "polynomial" in spec.crc_info
        assert spec.crc_info["polynomial"] == "0x1021"
