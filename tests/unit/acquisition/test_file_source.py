"""Tests for FileSource acquisition.

Architecture: Phase 0.1 - Unified Acquisition Layer
Tests the Source protocol implementation for file-based acquisition.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from oscura.acquisition import FileSource
from oscura.builders import SignalBuilder
from oscura.core.types import DigitalTrace, IQTrace, WaveformTrace

if TYPE_CHECKING:
    from oscura.core.types import Trace

pytestmark = pytest.mark.unit


class TestFileSource:
    """Tests for FileSource class."""

    @pytest.fixture
    def sample_waveform_file(self, tmp_path: Path) -> Path:
        """Create a sample waveform file for testing."""
        # Create a simple CSV file
        file_path = tmp_path / "test_waveform.csv"
        data = np.sin(np.linspace(0, 2 * np.pi, 100))
        np.savetxt(file_path, data, delimiter=",")
        return file_path

    @pytest.fixture
    def sample_trace(self) -> WaveformTrace:
        """Create a sample trace."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001)
        builder = builder.add_sine(frequency=1000, amplitude=1.0)
        return builder.build()

    def test_init_with_string_path(self, sample_waveform_file: Path) -> None:
        """Test FileSource initialization with string path."""
        source = FileSource(str(sample_waveform_file))
        assert source.path == sample_waveform_file
        assert source.format is None
        assert source.kwargs == {}
        assert source._closed is False

    def test_init_with_path_object(self, sample_waveform_file: Path) -> None:
        """Test FileSource initialization with Path object."""
        source = FileSource(sample_waveform_file)
        assert source.path == sample_waveform_file
        assert source.format is None

    def test_init_with_format(self, sample_waveform_file: Path) -> None:
        """Test FileSource initialization with explicit format."""
        source = FileSource(sample_waveform_file, format="csv")
        assert source.format == "csv"

    def test_init_with_kwargs(self, sample_waveform_file: Path) -> None:
        """Test FileSource initialization with additional kwargs."""
        source = FileSource(sample_waveform_file, channel=1, sample_rate=1e6)
        assert source.kwargs == {"channel": 1, "sample_rate": 1e6}

    def test_read_csv(self, sample_waveform_file: Path) -> None:
        """Test reading CSV file."""
        source = FileSource(sample_waveform_file)
        trace = source.read()

        # Verify trace is valid
        assert isinstance(trace, (WaveformTrace, DigitalTrace, IQTrace))
        assert len(trace.data) > 0

    def test_read_closed_source_raises(self, sample_waveform_file: Path) -> None:
        """Test that reading from closed source raises ValueError."""
        source = FileSource(sample_waveform_file)
        source.close()

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()

    def test_stream_closed_source_raises(self, sample_waveform_file: Path) -> None:
        """Test that streaming from closed source raises ValueError."""
        source = FileSource(sample_waveform_file)
        source.close()

        with pytest.raises(ValueError, match="Cannot stream from closed source"):
            list(source.stream(chunk_size=10))

    def test_stream_chunks(self, sample_waveform_file: Path) -> None:
        """Test streaming file in chunks."""
        source = FileSource(sample_waveform_file)
        chunks = list(source.stream(chunk_size=25))

        # Should get multiple chunks
        assert len(chunks) > 1

        # Each chunk should be a Trace
        for chunk in chunks:
            assert isinstance(chunk, (WaveformTrace, DigitalTrace, IQTrace))

        # Last chunk might be smaller
        assert len(chunks[-1].data) <= 25

    def test_close(self, sample_waveform_file: Path) -> None:
        """Test closing FileSource."""
        source = FileSource(sample_waveform_file)
        source.close()
        assert source._closed is True

    def test_context_manager(self, sample_waveform_file: Path) -> None:
        """Test FileSource as context manager."""
        with FileSource(sample_waveform_file) as source:
            trace = source.read()
            assert len(trace.data) > 0

        # Should be closed after exiting context
        assert source._closed is True

    def test_context_manager_auto_close(self, sample_waveform_file: Path) -> None:
        """Test that context manager closes source automatically."""
        with FileSource(sample_waveform_file) as source:
            _ = source.read()

        # After context exit, should not be able to read
        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()

    def test_repr(self, sample_waveform_file: Path) -> None:
        """Test string representation."""
        source = FileSource(sample_waveform_file, format="csv")
        repr_str = repr(source)

        assert "FileSource" in repr_str
        assert "test_waveform.csv" in repr_str
        assert "csv" in repr_str

    def test_multiple_reads(self, sample_waveform_file: Path) -> None:
        """Test that multiple reads work correctly."""
        source = FileSource(sample_waveform_file)

        trace1 = source.read()
        trace2 = source.read()

        # Should get same data
        assert len(trace1.data) == len(trace2.data)
        assert np.allclose(trace1.data, trace2.data)

    def test_read_with_kwargs(self, tmp_path: Path) -> None:
        """Test reading with format-specific kwargs."""
        # Create a file with known data
        file_path = tmp_path / "test.csv"
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.savetxt(file_path, data)

        # Read with kwargs
        source = FileSource(file_path, sample_rate=1e6)
        trace = source.read()

        # Verify kwargs were passed (sample_rate should be in metadata)
        if hasattr(trace.metadata, "sample_rate"):
            assert trace.metadata.sample_rate == 1e6

    def test_stream_full_file_reconstruction(self, sample_waveform_file: Path) -> None:
        """Test that streaming chunks can reconstruct full file."""
        source = FileSource(sample_waveform_file)

        # Get full trace
        full_trace = source.read()

        # Stream in chunks
        source = FileSource(sample_waveform_file)  # Re-create to reset state
        chunks = list(source.stream(chunk_size=25))

        # Reconstruct
        reconstructed = np.concatenate([chunk.data for chunk in chunks])

        # Should match full trace
        assert np.allclose(full_trace.data, reconstructed)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Test that reading nonexistent file raises appropriate error."""
        nonexistent = tmp_path / "does_not_exist.csv"
        source = FileSource(nonexistent)

        with pytest.raises((FileNotFoundError, OSError)):
            source.read()

    def test_source_protocol_compliance(self, sample_waveform_file: Path) -> None:
        """Test that FileSource implements Source protocol."""
        source = FileSource(sample_waveform_file)

        # Should have required methods
        assert hasattr(source, "read")
        assert hasattr(source, "stream")
        assert hasattr(source, "close")
        assert callable(source.read)
        assert callable(source.stream)
        assert callable(source.close)

    def test_polymorphic_usage(self, sample_waveform_file: Path) -> None:
        """Test polymorphic usage with other sources."""

        def process_source(source: object) -> Trace:
            """Process any source that implements read()."""
            return source.read()  # type: ignore[attr-defined, no-any-return]

        source = FileSource(sample_waveform_file)
        trace = process_source(source)

        assert isinstance(trace, (WaveformTrace, DigitalTrace, IQTrace))

    def test_stream_empty_file(self, tmp_path: Path) -> None:
        """Test streaming empty file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        source = FileSource(empty_file)

        # Should handle empty file gracefully
        with pytest.raises((ValueError, OSError)):
            list(source.stream(chunk_size=10))

    def test_close_idempotent(self, sample_waveform_file: Path) -> None:
        """Test that calling close multiple times is safe."""
        source = FileSource(sample_waveform_file)
        source.close()
        source.close()  # Should not raise
        assert source._closed is True
