"""Synthetic signal generation source.

This module provides SyntheticSource, which wraps SignalBuilder to implement
the unified Source protocol. SyntheticSource makes synthetic signals consistent
with all other acquisition methods (files, hardware).

Example:
    >>> from oscura.hardware.acquisition import SyntheticSource
    >>> from oscura import SignalBuilder
    >>>
    >>> # Create signal builder
    >>> builder = SignalBuilder(sample_rate=1e6, duration=0.01)
    >>> builder = builder.add_sine(frequency=1000, amplitude=1.0)
    >>> builder = builder.add_noise(snr_db=40)
    >>>
    >>> # Wrap in Source for unified interface
    >>> source = SyntheticSource(builder)
    >>> trace = source.read()
    >>>
    >>> # Or one-liner
    >>> source = SyntheticSource(
    ...     SignalBuilder().sample_rate(1e6).add_sine(1000).add_noise(snr_db=40)
    ... )
    >>> trace = source.read()

Pattern:
    SyntheticSource bridges SignalBuilder (generator pattern) with
    Source protocol (acquisition pattern). This enables:
    - Polymorphic use with FileSource and HardwareSource
    - Session management with synthetic signals
    - Pipeline composition with generated data

Note:
    SignalBuilder.build() returns WaveformTrace for single-channel signals.
    SignalBuilder.build_channels() returns dict[str, WaveformTrace] for multi-channel.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.core.types import Trace, WaveformTrace
    from oscura.utils.builders.signal_builder import SignalBuilder


class SyntheticSource:
    """Synthetic signal source implementing Source protocol.

    Wraps SignalBuilder to provide unified acquisition interface.
    Enables synthetic signals to be used anywhere a Source is expected.

    Attributes:
        builder: SignalBuilder instance to generate from.
        channel: Channel name to extract (for multi-channel signals).

    Example:
        >>> from oscura import SignalBuilder
        >>> from oscura.hardware.acquisition import SyntheticSource
        >>>
        >>> # Create builder
        >>> builder = (SignalBuilder(sample_rate=1e6, duration=0.01)
        ...     .add_sine(frequency=1000)
        ...     .add_noise(snr_db=40))
        >>>
        >>> # Wrap in source
        >>> source = SyntheticSource(builder)
        >>> trace = source.read()
        >>>
        >>> # Multi-channel
        >>> builder = SignalBuilder().sample_rate(1e6)
        >>> builder = builder.add_sine(1000, channel="sig")
        >>> builder = builder.add_square(500, channel="clk")
        >>> source = SyntheticSource(builder, channel="sig")
        >>> trace = source.read()  # Gets "sig" channel only
    """

    def __init__(
        self,
        builder: SignalBuilder,
        *,
        channel: str | None = None,
    ) -> None:
        """Initialize synthetic source.

        Args:
            builder: SignalBuilder instance to generate from.
            channel: Optional channel name for multi-channel signals.
                If None, uses first channel (or converts multi-channel to
                single-channel trace if possible).

        Example:
            >>> builder = SignalBuilder().sample_rate(1e6).add_sine(1000)
            >>> source = SyntheticSource(builder)
            >>> source = SyntheticSource(builder, channel="ch1")
        """
        self.builder = builder
        self.channel = channel
        self._closed = False
        self._cached_trace: WaveformTrace | None = None

    def read(self) -> Trace:
        """Generate and return complete trace.

        Returns:
            WaveformTrace with generated signal data.

        Raises:
            ValueError: If source is closed or builder has no signals.

        Example:
            >>> builder = SignalBuilder().sample_rate(1e6).add_sine(1000)
            >>> source = SyntheticSource(builder)
            >>> trace = source.read()
            >>> print(f"Generated {len(trace.data)} samples")

        Note:
            This method caches the generated trace to avoid regenerating
            on multiple calls. Call close() and recreate to generate new data.
        """
        if self._closed:
            raise ValueError("Cannot read from closed source")

        # Return cached trace if available
        if self._cached_trace is not None:
            return self._cached_trace

        # Build signal
        # Phase 0.2: build() now returns WaveformTrace directly
        trace = self.builder.build(channel=self.channel)

        self._cached_trace = trace
        return trace

    def stream(self, chunk_size: int) -> Iterator[Trace]:
        """Stream generated signal in chunks.

        Args:
            chunk_size: Number of samples per chunk.

        Yields:
            Trace chunks.

        Example:
            >>> builder = SignalBuilder().sample_rate(1e6).duration(1.0).add_sine(1000)
            >>> source = SyntheticSource(builder)
            >>> for chunk in source.stream(chunk_size=10000):
            ...     process_chunk(chunk)

        Note:
            Synthetic signals are generated once and sliced into chunks.
            This is different from hardware streaming where data arrives continuously.
        """
        if self._closed:
            raise ValueError("Cannot stream from closed source")

        # Generate full trace
        trace = self.read()

        # Yield chunks
        from oscura.core.types import DigitalTrace, IQTrace, TraceMetadata, WaveformTrace

        # IQTrace not supported by SyntheticSource
        if isinstance(trace, IQTrace):
            raise TypeError("IQTrace not supported by SyntheticSource")

        n_samples = len(trace.data)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk_data = trace.data[start:end]

            # Create chunk metadata (same as parent)
            chunk_metadata = TraceMetadata(
                sample_rate=trace.metadata.sample_rate,
                vertical_scale=trace.metadata.vertical_scale,
                vertical_offset=trace.metadata.vertical_offset,
                acquisition_time=trace.metadata.acquisition_time,
                trigger_info=trace.metadata.trigger_info,
                source_file=trace.metadata.source_file,
                channel_name=trace.metadata.channel_name,
                calibration_info=trace.metadata.calibration_info,
            )

            if isinstance(trace, DigitalTrace):
                yield DigitalTrace(data=chunk_data, metadata=chunk_metadata)  # type: ignore[arg-type]
            else:
                yield WaveformTrace(data=chunk_data, metadata=chunk_metadata)  # type: ignore[arg-type]

    def close(self) -> None:
        """Close the source and release resources.

        For synthetic sources, this clears the cached trace.

        Example:
            >>> source = SyntheticSource(builder)
            >>> trace = source.read()
            >>> source.close()
            >>> # source.read() would now fail
        """
        self._closed = True
        self._cached_trace = None

    def __enter__(self) -> SyntheticSource:
        """Context manager entry.

        Returns:
            Self for use in 'with' statement.

        Example:
            >>> with SyntheticSource(builder) as source:
            ...     trace = source.read()
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit.

        Automatically calls close() when exiting 'with' block.
        """
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"SyntheticSource(builder={self.builder!r}, channel={self.channel!r})"


__all__ = ["SyntheticSource"]
