"""Tests for GenericSession."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from oscura.hardware.acquisition import SyntheticSource
from oscura.sessions import ComparisonResult, GenericSession
from oscura.utils.builders import SignalBuilder

pytestmark = pytest.mark.unit


class TestGenericSession:
    """Tests for GenericSession implementation."""

    def test_creation(self) -> None:
        """Test session creation."""
        session = GenericSession(name="Test Session")

        assert session.name == "Test Session"
        assert len(session.recordings) == 0

    def test_add_recording(self) -> None:
        """Test adding recordings."""
        session = GenericSession()
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        session.add_recording("test", source)

        assert "test" in session.recordings
        assert session.list_recordings() == ["test"]

    def test_add_duplicate_recording_raises(self) -> None:
        """Test that duplicate recording names raise error."""
        session = GenericSession()
        source = SyntheticSource(SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000))

        session.add_recording("test", source)

        with pytest.raises(ValueError, match="already exists"):
            session.add_recording("test", source)

    def test_get_recording(self) -> None:
        """Test getting recordings."""
        session = GenericSession()
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        session.add_recording("test", source)
        trace = session.get_recording("test")

        assert len(trace.data) == 1000

    def test_get_nonexistent_recording_raises(self) -> None:
        """Test that getting nonexistent recording raises error."""
        session = GenericSession()

        with pytest.raises(KeyError, match="not found"):
            session.get_recording("nonexistent")

    def test_compare_recordings(self) -> None:
        """Test comparing recordings."""
        session = GenericSession()

        # Create two similar signals
        source1 = SyntheticSource(
            SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000, amplitude=1.0)
        )
        source2 = SyntheticSource(
            SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000, amplitude=1.1)
        )

        session.add_recording("sig1", source1)
        session.add_recording("sig2", source2)

        result = session.compare("sig1", "sig2")

        assert isinstance(result, ComparisonResult)
        assert result.recording1 == "sig1"
        assert result.recording2 == "sig2"
        assert 0 <= result.similarity_score <= 1

    def test_analyze(self) -> None:
        """Test generic analysis."""
        session = GenericSession()

        source = SyntheticSource(SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000))
        session.add_recording("test", source)

        results = session.analyze()

        assert results["num_recordings"] == 1
        assert "test" in results["summary"]
        assert "mean" in results["summary"]["test"]
        assert "std" in results["summary"]["test"]

    def test_export_json(self) -> None:
        """Test JSON export."""
        session = GenericSession()
        source = SyntheticSource(SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000))
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            session.export_results("json", str(path))

            assert path.exists()

            # Verify JSON is valid
            with open(path) as f:
                data = json.load(f)
                assert "num_recordings" in data
                assert "summary" in data

    def test_export_csv(self) -> None:
        """Test CSV export."""
        session = GenericSession()
        source = SyntheticSource(SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000))
        session.add_recording("test", source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            session.export_results("csv", str(path))

            assert path.exists()

            # Verify CSV has content
            content = path.read_text()
            assert "recording" in content
            assert "test" in content
