"""Tests for SyntheticSource."""

from __future__ import annotations

import pytest

from oscura.acquisition import SyntheticSource
from oscura.builders import SignalBuilder
from oscura.core.types import WaveformTrace

pytestmark = pytest.mark.unit


class TestSyntheticSource:
    """Tests for SyntheticSource implementation."""

    def test_basic_creation(self) -> None:
        """Test basic source creation."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        assert source is not None
        assert not source._closed

    def test_read_returns_waveform_trace(self) -> None:
        """Test that read() returns WaveformTrace."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        trace = source.read()

        assert isinstance(trace, WaveformTrace)
        assert len(trace.data) == 1000
        assert trace.metadata.sample_rate == 1e6

    def test_read_caching(self) -> None:
        """Test that read() caches result."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        trace1 = source.read()
        trace2 = source.read()

        # Should return same cached instance
        assert trace1 is trace2

    def test_channel_selection(self) -> None:
        """Test channel selection for multi-channel signals."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001)
        builder.add_sine(1000, channel="sig")
        builder.add_square(500, channel="clk")

        source = SyntheticSource(builder, channel="sig")
        trace = source.read()

        assert isinstance(trace, WaveformTrace)
        # Should have selected "sig" channel
        assert trace.metadata.channel_name == "sig"

    def test_streaming(self) -> None:
        """Test streaming functionality."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        chunks = list(source.stream(chunk_size=250))

        assert len(chunks) == 4  # 1000 samples / 250 per chunk
        assert all(isinstance(c, WaveformTrace) for c in chunks)
        assert all(len(c.data) == 250 for c in chunks[:3])  # First 3 full chunks
        assert len(chunks[3].data) == 250  # Last chunk

    def test_context_manager(self) -> None:
        """Test context manager support."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)

        with SyntheticSource(builder) as source:
            trace = source.read()
            assert isinstance(trace, WaveformTrace)

        # Should be closed after context
        assert source._closed

    def test_closed_source_raises(self) -> None:
        """Test that closed source raises error."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        source.close()

        with pytest.raises(ValueError, match="Cannot read from closed source"):
            source.read()

    def test_close_clears_cache(self) -> None:
        """Test that close() clears cached trace."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        _ = source.read()  # Read to populate cache
        assert source._cached_trace is not None

        source.close()
        assert source._cached_trace is None

    def test_stream_closed_raises(self) -> None:
        """Test that streaming from closed source raises error."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)
        source.close()

        with pytest.raises(ValueError, match="Cannot stream from closed source"):
            list(source.stream(chunk_size=100))

    def test_repr(self) -> None:
        """Test string representation."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder, channel="test")

        repr_str = repr(source)
        assert "SyntheticSource" in repr_str
        assert "test" in repr_str

    def test_stream_metadata_preserved(self) -> None:
        """Test that streaming preserves metadata in chunks."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        full_trace = source.read()
        source = SyntheticSource(builder)  # Re-create to test streaming
        chunks = list(source.stream(chunk_size=250))

        # All chunks should have same sample rate
        for chunk in chunks:
            assert chunk.metadata.sample_rate == full_trace.metadata.sample_rate

    def test_close_idempotent(self) -> None:
        """Test that calling close multiple times is safe."""
        builder = SignalBuilder(sample_rate=1e6, duration=0.001).add_sine(1000)
        source = SyntheticSource(builder)

        source.close()
        source.close()  # Should not raise
        assert source._closed is True
