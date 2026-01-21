"""Tests for AnalysisSession base class.

Architecture: Phase 0.3 - AnalysisSession Base Class
Tests the base functionality shared across all domain-specific sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from oscura.acquisition import SyntheticSource
from oscura.builders import SignalBuilder
from oscura.core.types import IQTrace, TraceMetadata, WaveformTrace
from oscura.sessions.base import AnalysisSession, ComparisonResult

if TYPE_CHECKING:
    from oscura.acquisition import Source

pytestmark = pytest.mark.unit


class ConcreteSession(AnalysisSession):
    """Concrete implementation for testing."""

    def analyze(self) -> Any:
        """Dummy analyze implementation."""
        return {"result": "test"}


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ComparisonResult creation."""
        result = ComparisonResult(
            recording1="baseline",
            recording2="stimulus",
            changed_bytes=42,
        )

        assert result.recording1 == "baseline"
        assert result.recording2 == "stimulus"
        assert result.changed_bytes == 42
        assert result.changed_regions == []
        assert result.similarity_score == 0.0
        assert result.details == {}

    def test_full_creation(self) -> None:
        """Test ComparisonResult with all fields."""
        result = ComparisonResult(
            recording1="test1",
            recording2="test2",
            changed_bytes=10,
            changed_regions=[(0, 10, "header"), (20, 30, "footer")],
            similarity_score=0.95,
            details={"method": "differential"},
        )

        assert len(result.changed_regions) == 2
        assert result.similarity_score == 0.95
        assert result.details["method"] == "differential"


class TestAnalysisSession:
    """Tests for AnalysisSession base class."""

    @pytest.fixture
    def session(self) -> ConcreteSession:
        """Create a concrete session for testing."""
        return ConcreteSession(name="Test Session")

    @pytest.fixture
    def sample_source(self) -> Source:
        """Create a sample data source."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001)
        builder = builder.add_sine(frequency=1000, amplitude=1.0)
        return SyntheticSource(builder)

    @pytest.fixture
    def alternative_source(self) -> Source:
        """Create an alternative data source for comparison."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001)
        builder = builder.add_sine(frequency=2000, amplitude=1.5)
        return SyntheticSource(builder)

    def test_init_default_name(self) -> None:
        """Test session initialization with default name."""
        session = ConcreteSession()
        assert session.name == "Untitled Session"
        assert session.recordings == {}
        assert session.metadata == {}
        assert session.created_at is not None
        assert session.modified_at is not None

    def test_init_custom_name(self) -> None:
        """Test session initialization with custom name."""
        session = ConcreteSession(name="My Analysis")
        assert session.name == "My Analysis"

    def test_add_recording(self, session: ConcreteSession, sample_source: Source) -> None:
        """Test adding a recording to session."""
        session.add_recording("baseline", sample_source)

        assert "baseline" in session.recordings
        assert len(session.recordings) == 1

    def test_add_recording_deferred_loading(
        self, session: ConcreteSession, sample_source: Source
    ) -> None:
        """Test adding recording with deferred loading."""
        session.add_recording("baseline", sample_source, load_immediately=False)

        source, trace = session.recordings["baseline"]
        assert trace is None  # Not loaded yet

    def test_add_recording_duplicate_raises(
        self, session: ConcreteSession, sample_source: Source
    ) -> None:
        """Test that adding duplicate recording raises error."""
        session.add_recording("baseline", sample_source)

        with pytest.raises(ValueError, match="Recording 'baseline' already exists"):
            session.add_recording("baseline", sample_source)

    def test_get_recording(self, session: ConcreteSession, sample_source: Source) -> None:
        """Test retrieving a recording."""
        session.add_recording("baseline", sample_source)
        trace = session.get_recording("baseline")

        assert isinstance(trace, WaveformTrace)
        assert len(trace.data) > 0

    def test_get_recording_lazy_load(self, session: ConcreteSession, sample_source: Source) -> None:
        """Test that get_recording loads deferred recordings."""
        session.add_recording("baseline", sample_source, load_immediately=False)

        # Should be None initially
        _, initial_trace = session.recordings["baseline"]
        assert initial_trace is None

        # get_recording should load it
        trace = session.get_recording("baseline")
        assert isinstance(trace, WaveformTrace)

        # Should now be cached
        _, cached_trace = session.recordings["baseline"]
        assert cached_trace is not None

    def test_get_recording_not_found(self, session: ConcreteSession) -> None:
        """Test that getting nonexistent recording raises KeyError."""
        with pytest.raises(KeyError, match="Recording 'nonexistent' not found"):
            session.get_recording("nonexistent")

    def test_list_recordings_empty(self, session: ConcreteSession) -> None:
        """Test listing recordings when none exist."""
        assert session.list_recordings() == []

    def test_list_recordings(
        self,
        session: ConcreteSession,
        sample_source: Source,
        alternative_source: Source,
    ) -> None:
        """Test listing multiple recordings."""
        session.add_recording("baseline", sample_source)
        session.add_recording("stimulus", alternative_source)

        recordings = session.list_recordings()
        assert len(recordings) == 2
        assert "baseline" in recordings
        assert "stimulus" in recordings

    def test_compare_recordings(
        self,
        session: ConcreteSession,
        sample_source: Source,
        alternative_source: Source,
    ) -> None:
        """Test comparing two recordings."""
        session.add_recording("baseline", sample_source)
        session.add_recording("stimulus", alternative_source)

        result = session.compare("baseline", "stimulus")

        assert isinstance(result, ComparisonResult)
        assert result.recording1 == "baseline"
        assert result.recording2 == "stimulus"
        assert result.changed_bytes >= 0
        assert 0.0 <= result.similarity_score <= 1.0

    def test_compare_identical_recordings(
        self, session: ConcreteSession, sample_source: Source
    ) -> None:
        """Test comparing identical recordings."""
        session.add_recording("rec1", sample_source)
        session.add_recording("rec2", sample_source)

        result = session.compare("rec1", "rec2")

        # Should be very similar (same source)
        assert result.similarity_score > 0.99
        assert result.changed_bytes < 10

    def test_compare_nonexistent_recording(
        self, session: ConcreteSession, sample_source: Source
    ) -> None:
        """Test that comparing nonexistent recording raises error."""
        session.add_recording("baseline", sample_source)

        with pytest.raises(KeyError):
            session.compare("baseline", "nonexistent")

    def test_export_results_report(self, session: ConcreteSession, tmp_path: Path) -> None:
        """Test exporting results as report."""
        output_file = tmp_path / "report.txt"
        session.export_results("report", output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Test Session" in content
        assert "Created:" in content
        assert "Recordings: 0" in content

    def test_export_results_with_recordings(
        self, session: ConcreteSession, sample_source: Source, tmp_path: Path
    ) -> None:
        """Test exporting results with recordings."""
        session.add_recording("baseline", sample_source)
        session.add_recording("stimulus", sample_source)

        output_file = tmp_path / "report.txt"
        session.export_results("report", output_file)

        content = output_file.read_text()
        assert "Recordings: 2" in content
        assert "baseline" in content
        assert "stimulus" in content

    def test_export_results_creates_directory(
        self, session: ConcreteSession, tmp_path: Path
    ) -> None:
        """Test that export creates parent directories."""
        output_file = tmp_path / "subdir" / "nested" / "report.txt"
        session.export_results("report", output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_export_results_unsupported_format(
        self, session: ConcreteSession, tmp_path: Path
    ) -> None:
        """Test that unsupported format raises error."""
        output_file = tmp_path / "output.xyz"

        with pytest.raises(ValueError, match="Unsupported export format: xyz"):
            session.export_results("xyz", output_file)

    def test_analyze_abstract_method(self) -> None:
        """Test that analyze is abstract and must be implemented."""
        # ConcreteSession implements it, so it should work
        session = ConcreteSession()
        result = session.analyze()
        assert result == {"result": "test"}

        # Cannot instantiate AnalysisSession directly
        with pytest.raises(TypeError):
            AnalysisSession()  # type: ignore[abstract]

    def test_repr(self, session: ConcreteSession, sample_source: Source) -> None:
        """Test string representation."""
        session.add_recording("baseline", sample_source)

        repr_str = repr(session)
        assert "ConcreteSession" in repr_str
        assert "Test Session" in repr_str
        assert "recordings=1" in repr_str

    def test_modified_at_updates(self, session: ConcreteSession, sample_source: Source) -> None:
        """Test that modified_at updates when adding recordings."""
        initial_time = session.modified_at

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        session.add_recording("baseline", sample_source)

        assert session.modified_at > initial_time

    def test_metadata_dict(self, session: ConcreteSession) -> None:
        """Test that metadata dictionary is mutable."""
        session.metadata["test_key"] = "test_value"
        session.metadata["count"] = 42

        assert session.metadata["test_key"] == "test_value"
        assert session.metadata["count"] == 42

    def test_compare_with_details(
        self,
        session: ConcreteSession,
        sample_source: Source,
        alternative_source: Source,
    ) -> None:
        """Test that comparison includes details."""
        session.add_recording("rec1", sample_source)
        session.add_recording("rec2", alternative_source)

        result = session.compare("rec1", "rec2")

        assert "trace1_length" in result.details
        assert "trace2_length" in result.details
        assert "compared_length" in result.details
        assert result.details["trace1_length"] > 0

    def test_session_workflow(
        self,
        session: ConcreteSession,
        sample_source: Source,
        alternative_source: Source,
        tmp_path: Path,
    ) -> None:
        """Test complete session workflow."""
        # Add recordings
        session.add_recording("baseline", sample_source)
        session.add_recording("stimulus", alternative_source)

        # List recordings
        recordings = session.list_recordings()
        assert len(recordings) == 2

        # Get individual recording
        trace = session.get_recording("baseline")
        assert len(trace.data) > 0

        # Compare recordings
        comparison = session.compare("baseline", "stimulus")
        assert comparison.changed_bytes >= 0

        # Analyze (domain-specific)
        results = session.analyze()
        assert results == {"result": "test"}

        # Export results
        report_path = tmp_path / "final_report.txt"
        session.export_results("report", report_path)
        assert report_path.exists()

    def test_compare_iqtrace_not_supported(self, session: ConcreteSession) -> None:
        """Test that IQTrace comparison raises appropriate error."""
        # Create IQTrace manually with i_data and q_data
        i_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        q_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        iq_trace = IQTrace(
            i_data=i_data,
            q_data=q_data,
            metadata=TraceMetadata(sample_rate=1e6, source_file="test.iq", channel_name="IQ"),
        )

        # Mock the recordings with IQTrace
        from oscura.acquisition import SyntheticSource

        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        normal_source = SyntheticSource(builder)

        session.add_recording("normal", normal_source)

        # Manually inject IQTrace into recordings
        session.recordings["iq"] = (normal_source, iq_trace)

        with pytest.raises(TypeError, match="IQTrace comparison not yet supported"):
            session.compare("normal", "iq")

    def test_deferred_loading_multiple_get_calls(
        self, session: ConcreteSession, sample_source: Source
    ) -> None:
        """Test that deferred loading only loads once."""
        session.add_recording("test", sample_source, load_immediately=False)

        # First call loads
        trace1 = session.get_recording("test")

        # Second call returns cached
        trace2 = session.get_recording("test")

        # Should be the same instance (cached)
        assert trace1 is trace2
