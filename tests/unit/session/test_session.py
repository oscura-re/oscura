"""Tests for session management module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.session import (
    Annotation,
    AnnotationLayer,
    AnnotationType,
    HistoryEntry,
    OperationHistory,
    Session,
    load_session,
)

pytestmark = pytest.mark.unit


class TestAnnotation:
    """Tests for Annotation class (SESS-002)."""

    def test_point_annotation(self):
        """Test point annotation creation."""
        ann = Annotation(text="Glitch", time=1.5e-6)

        assert ann.text == "Glitch"
        assert ann.time == 1.5e-6
        assert ann.annotation_type == AnnotationType.POINT

    def test_range_annotation(self):
        """Test range annotation creation."""
        ann = Annotation(text="Packet", time_range=(1e-6, 2e-6))

        assert ann.time_range == (1e-6, 2e-6)
        assert ann.annotation_type == AnnotationType.RANGE
        assert ann.start_time == 1e-6
        assert ann.end_time == 2e-6

    def test_annotation_serialization(self):
        """Test annotation to/from dict."""
        ann = Annotation(
            text="Test",
            time=1.0,
            color="#FF0000",
            metadata={"key": "value"},
        )

        data = ann.to_dict()
        restored = Annotation.from_dict(data)

        assert restored.text == ann.text
        assert restored.time == ann.time
        assert restored.color == ann.color
        assert restored.metadata == ann.metadata


class TestAnnotationLayer:
    """Tests for AnnotationLayer class (SESS-002)."""

    def test_add_annotation(self):
        """Test adding annotations to layer."""
        layer = AnnotationLayer("Test Layer")

        layer.add(text="Point 1", time=1.0)
        layer.add(text="Point 2", time=2.0)

        assert len(layer.annotations) == 2

    def test_add_pre_built_annotation(self):
        """Test adding pre-built annotation."""
        layer = AnnotationLayer("Test")
        ann = Annotation(text="Pre-built", time=1.5)

        layer.add(ann)

        assert layer.annotations[0] == ann

    def test_remove_annotation(self):
        """Test removing annotations."""
        layer = AnnotationLayer("Test")
        layer.add(text="A", time=1.0)
        layer.add(text="B", time=2.0)

        ann = layer.annotations[0]
        result = layer.remove(ann)

        assert result is True
        assert len(layer.annotations) == 1

    def test_find_at_time(self):
        """Test finding annotations at specific time."""
        layer = AnnotationLayer("Test")
        layer.add(text="A", time=1.0)
        layer.add(text="B", time=2.0)
        layer.add(text="C", time_range=(1.5, 2.5))

        # Find at time 2.0
        matches = layer.find_at_time(2.0)
        assert len(matches) == 2  # Point B and range C

    def test_find_in_range(self):
        """Test finding annotations in range."""
        layer = AnnotationLayer("Test")
        layer.add(text="A", time=1.0)
        layer.add(text="B", time=5.0)
        layer.add(text="C", time=10.0)

        matches = layer.find_in_range(0.5, 6.0)
        assert len(matches) == 2  # A and B

    def test_locked_layer(self):
        """Test locked layer prevents modifications."""
        layer = AnnotationLayer("Test", locked=True)

        with pytest.raises(ValueError):
            layer.add(text="Should fail", time=1.0)

    def test_layer_serialization(self):
        """Test layer to/from dict."""
        layer = AnnotationLayer("Test Layer", description="Test description")
        layer.add(text="A", time=1.0)
        layer.add(text="B", time=2.0)

        data = layer.to_dict()
        restored = AnnotationLayer.from_dict(data)

        assert restored.name == layer.name
        assert restored.description == layer.description
        assert len(restored.annotations) == 2


class TestOperationHistory:
    """Tests for OperationHistory class (SESS-003)."""

    def test_record_operation(self):
        """Test recording operations."""
        history = OperationHistory()

        history.record("load", {"file": "test.wfm"})
        history.record("measure_rise_time", result=1.5e-9)

        assert len(history.entries) == 2

    def test_undo(self):
        """Test undo removes last entry."""
        history = OperationHistory()
        history.record("op1")
        history.record("op2")

        removed = history.undo()

        assert removed.operation == "op2"
        assert len(history.entries) == 1

    def test_find_operations(self):
        """Test finding operations by criteria."""
        history = OperationHistory()
        history.record("load", success=True)
        history.record("measure", success=True)
        history.record("export", success=False)

        # Find all successful
        successful = history.find(success_only=True)
        assert len(successful) == 2

        # Find specific operation
        load_ops = history.find(operation="load")
        assert len(load_ops) == 1

    def test_to_script(self):
        """Test script generation."""
        history = OperationHistory()
        history.record("load", {"file": "test.wfm"})
        history.record("measure_rise_time", result=1.5e-9)

        script = history.to_script()

        assert "import oscura" in script
        assert "osc.load" in script
        assert "osc.measure_rise_time" in script

    def test_summary(self):
        """Test history summary."""
        history = OperationHistory()
        history.record("load", success=True)
        history.record("measure", success=True)
        history.record("export", success=False)

        summary = history.summary()

        assert summary["total_operations"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1

    def test_max_entries(self):
        """Test max entries limit."""
        history = OperationHistory(max_entries=3)

        for i in range(5):
            history.record(f"op{i}")

        assert len(history.entries) == 3
        assert history.entries[-1].operation == "op4"

    def test_history_serialization(self):
        """Test history to/from dict."""
        history = OperationHistory()
        history.record("load", {"file": "test.wfm"})
        history.record("measure")

        data = history.to_dict()
        restored = OperationHistory.from_dict(data)

        assert len(restored.entries) == 2


class TestHistoryEntry:
    """Tests for HistoryEntry class (SESS-003)."""

    def test_to_code(self):
        """Test code generation from entry."""
        entry = HistoryEntry(
            operation="load",
            parameters={"file": "test.wfm", "channel": 1},
        )

        code = entry.to_code()

        assert "osc.load" in code
        assert '"test.wfm"' in code
        assert "channel=1" in code

    def test_serialization(self):
        """Test entry to/from dict."""
        entry = HistoryEntry(
            operation="test_op",
            parameters={"a": 1},
            result="success",
            duration_ms=100.5,
        )

        data = entry.to_dict()
        restored = HistoryEntry.from_dict(data)

        assert restored.operation == entry.operation
        assert restored.parameters == entry.parameters
        assert restored.duration_ms == entry.duration_ms


class TestSession:
    """Tests for Session class (SESS-001)."""

    def test_create_session(self):
        """Test session creation."""
        session = Session(name="Test Session")

        assert session.name == "Test Session"
        assert "default" in session.annotation_layers
        assert len(session.traces) == 0

    def test_annotate(self):
        """Test adding annotations via session."""
        session = Session()

        session.annotate("Glitch 1", time=1e-6)
        session.annotate("Glitch 2", time=2e-6, layer="debug")

        default_anns = session.get_annotations(layer="default")
        assert len(default_anns) == 1

        debug_anns = session.get_annotations(layer="debug")
        assert len(debug_anns) == 1

    def test_record_measurement(self):
        """Test recording measurements."""
        session = Session()

        session.record_measurement("rise_time", 1.5e-9, unit="s")
        session.record_measurement("frequency", 100e6, unit="Hz")

        measurements = session.get_measurements()

        assert "rise_time" in measurements
        assert measurements["rise_time"]["value"] == 1.5e-9
        assert measurements["rise_time"]["unit"] == "s"

    def test_save_and_load(self):
        """Test session save and restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate session
            session = Session(name="Test Session")
            session.annotate("Test annotation", time=1.0)
            session.record_measurement("test", 42)

            # Save
            save_path = Path(tmpdir) / "session.tks"
            session.save(save_path)

            # Load
            loaded = load_session(save_path)

            assert loaded.name == "Test Session"
            assert len(loaded.get_annotations()) == 1
            assert "test" in loaded.get_measurements()

    def test_save_without_traces(self):
        """Test saving session without trace data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session()

            # Add a trace (would fail to load without actual loader)
            session.traces["test"] = WaveformTrace(
                data=np.array([1.0, 2.0, 3.0]),
                metadata=TraceMetadata(sample_rate=1e6),
            )

            save_path = Path(tmpdir) / "session_no_traces.tks"
            session.save(save_path, include_traces=False)

            loaded = load_session(save_path)
            assert len(loaded.traces) == 0

    def test_session_summary(self):
        """Test session summary generation."""
        session = Session(name="Test")
        session.annotate("Ann 1", time=1.0)
        session.annotate("Ann 2", time=2.0)
        session.record_measurement("m1", 100)

        summary = session.summary()

        assert "Test" in summary
        assert "Annotations: 2" in summary
        assert "Measurements: 1" in summary

    def test_history_tracking(self):
        """Test operation history is tracked."""
        session = Session()

        session.annotate("Test", time=1.0)
        session.record_measurement("test", 42)

        assert len(session.history.entries) == 2

    def test_list_traces(self):
        """Test listing trace names."""
        session = Session()
        session.traces["ch1"] = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]), metadata=TraceMetadata(sample_rate=1e6)
        )
        session.traces["ch2"] = WaveformTrace(
            data=np.array([4.0, 5.0, 6.0]), metadata=TraceMetadata(sample_rate=1e6)
        )

        traces = session.list_traces()
        assert "ch1" in traces
        assert "ch2" in traces

    def test_save_with_hmac_signature(self):
        """Test session saves with HMAC signature (SECURITY-01)."""
        import gzip

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save session (uncompressed for easy verification)
            session = Session(name="Signed Session")
            session.annotate("Test", time=1.0)

            save_path = Path(tmpdir) / "signed.tks"
            session.save(save_path, compress=False)

            # Verify file contains signature (magic bytes + signature)
            with open(save_path, "rb") as f:
                magic = f.read(4)
                assert magic == b"OSC1", "Session file should start with OSC1 magic bytes"

            # Load should succeed with signature verification
            loaded = load_session(save_path)
            assert loaded.name == "Signed Session"

            # Test with compression
            compressed_path = Path(tmpdir) / "signed_compressed.tks"
            session.save(compressed_path, compress=True)

            # Verify compressed file can be loaded
            with gzip.open(compressed_path, "rb") as f:
                magic = f.read(4)
                assert magic == b"OSC1", (
                    "Compressed session should have OSC1 magic after decompression"
                )

            loaded_compressed = load_session(compressed_path)
            assert loaded_compressed.name == "Signed Session"

    def test_signature_verification_detects_tampering(self):
        """Test HMAC verification detects tampered files (SECURITY-01)."""
        from oscura.core.exceptions import SecurityError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save session (uncompressed to avoid gzip errors)
            session = Session(name="Original")
            save_path = Path(tmpdir) / "tampered.tks"
            session.save(save_path, compress=False)

            # Tamper with file (corrupt a byte in the data)
            with open(save_path, "rb") as f:
                content = f.read()

            # Flip a bit in the pickled data (after magic + signature)
            tampered = bytearray(content)
            tampered[40] ^= 0xFF  # Flip bits at position 40

            with open(save_path, "wb") as f:
                f.write(tampered)

            # Loading should fail with SecurityError
            with pytest.raises(SecurityError, match="signature verification failed"):
                load_session(save_path)

    def test_legacy_session_raises_security_error(self):
        """Test legacy session files (no signature) raise SecurityError (SECURITY-01)."""
        import pickle

        from oscura.core.exceptions import SecurityError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legacy session file (old format without signature)
            session_data = {
                "version": "1.0",
                "name": "Legacy Session",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "annotation_layers": {},
                "measurements": {},
                "history": {"entries": []},
                "metadata": {},
                "traces": {},
            }

            save_path = Path(tmpdir) / "legacy.tks"
            with open(save_path, "wb") as f:
                pickle.dump(session_data, f)

            # Load should fail with SecurityError
            with pytest.raises(SecurityError, match="legacy session file"):
                load_session(save_path)

    def test_incomplete_signature_raises_error(self):
        """Test incomplete signature (missing or truncated) raises SecurityError."""
        from oscura.core.exceptions import SecurityError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with magic bytes but incomplete signature
            save_path = Path(tmpdir) / "incomplete.tks"
            with open(save_path, "wb") as f:
                f.write(b"OSC1")  # Magic bytes
                f.write(b"short")  # Incomplete signature (not 32 bytes)

            # Load should fail with SecurityError
            with pytest.raises(SecurityError, match="legacy session file"):
                load_session(save_path)

    def test_compressed_session_with_signature(self):
        """Test compressed sessions work with signatures (SECURITY-01)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save compressed session
            session = Session(name="Compressed")
            session.annotate("Test", time=1.0)

            save_path = Path(tmpdir) / "compressed.tks"
            session.save(save_path, compress=True)

            # Load and verify
            loaded = load_session(save_path)
            assert loaded.name == "Compressed"
            assert len(loaded.get_annotations()) == 1

    def test_uncompressed_session_with_signature(self):
        """Test uncompressed sessions work with signatures (SECURITY-01)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uncompressed session
            session = Session(name="Uncompressed")
            session.annotate("Test", time=1.0)

            save_path = Path(tmpdir) / "uncompressed.tks"
            session.save(save_path, compress=False)

            # Load and verify
            loaded = load_session(save_path)
            assert loaded.name == "Uncompressed"
            assert len(loaded.get_annotations()) == 1
