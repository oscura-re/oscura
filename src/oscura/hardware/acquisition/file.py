"""File-based signal acquisition.

This module provides FileSource, which wraps existing file loaders to implement
the unified Source protocol. FileSource makes file loading consistent with all
other acquisition methods (hardware, synthetic).

Example:
    >>> from oscura.hardware.acquisition import FileSource
    >>>
    >>> # Basic usage
    >>> source = FileSource("capture.wfm")
    >>> trace = source.read()
    >>>
    >>> # Context manager (recommended)
    >>> with FileSource("capture.wfm") as source:
    ...     trace = source.read()
    ...     # Process trace
    ...  # Automatic cleanup
    >>>
    >>> # Streaming for large files
    >>> with FileSource("huge_capture.wfm") as source:
    ...     for chunk in source.stream(chunk_size=10000):
    ...         process_chunk(chunk)
    >>>
    >>> # Format override
    >>> source = FileSource("data.bin", format="tektronix")
    >>> trace = source.read()

Pattern:
    FileSource is a thin wrapper around the existing load() function.
    It provides the Source interface for consistency and composition.

Backward Compatibility:
    The existing oscura.load() function continues to work unchanged.
    FileSource is the new preferred pattern for explicit acquisition.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

    from oscura.core.types import Trace


class FileSource:
    """File-based signal source implementing Source protocol.

    Wraps existing file loaders to provide unified acquisition interface.
    Supports all file formats that oscura.load() supports:
    - Tektronix WFM
    - Rigol WFM
    - CSV, HDF5, NumPy
    - Sigrok, VCD, PCAP
    - WAV, TDMS, Touchstone

    Attributes:
        path: Path to the file.
        format: Optional format override (auto-detected if None).
        kwargs: Additional loader arguments.

    Example:
        >>> # Auto-detect format
        >>> source = FileSource("capture.wfm")
        >>> trace = source.read()
        >>>
        >>> # Override format
        >>> source = FileSource("data.bin", format="tektronix")
        >>> trace = source.read()
        >>>
        >>> # Specify channel for multi-channel files
        >>> source = FileSource("multi.wfm", channel=1)
        >>> trace = source.read()
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize file source.

        Args:
            path: Path to file to load.
            format: Optional format override (e.g., "tektronix", "rigol").
            **kwargs: Additional arguments passed to loader.

        Example:
            >>> source = FileSource("capture.wfm")
            >>> source = FileSource("data.bin", format="tektronix", channel=1)
        """
        self.path = Path(path)
        self.format = format
        self.kwargs = kwargs
        self._closed = False

    def read(self) -> Trace:
        """Read complete trace from file.

        Returns:
            Complete trace (WaveformTrace, DigitalTrace, or IQTrace).

        Raises:
            FileNotFoundError: If file doesn't exist.
            UnsupportedFormatError: If file format not recognized.
            LoaderError: If file cannot be loaded.

        Example:
            >>> source = FileSource("capture.wfm")
            >>> trace = source.read()
            >>> print(f"Loaded {len(trace.data)} samples")
        """
        if self._closed:
            raise ValueError("Cannot read from closed source")

        # Import here to avoid circular dependency
        from oscura.loaders import load

        return load(self.path, format=self.format, **self.kwargs)

    def stream(self, chunk_size: int) -> Iterator[Trace]:
        """Stream trace in chunks for large files.

        Args:
            chunk_size: Number of samples per chunk.

        Yields:
            Trace chunks.

        Raises:
            FileNotFoundError: If file doesn't exist.
            LoaderError: If file cannot be loaded.

        Example:
            >>> source = FileSource("huge_capture.wfm")
            >>> for chunk in source.stream(chunk_size=10000):
            ...     metrics = analyze(chunk)
            ...     print(f"Chunk: {metrics}")

        Note:
            Currently uses load_trace_chunks for chunked loading.
            For formats without native chunking support, loads full file
            and yields slices.
        """
        if self._closed:
            raise ValueError("Cannot stream from closed source")

        # Try lazy/chunked loading if available
        try:
            from oscura.utils.streaming import load_trace_chunks

            # load_trace_chunks expects Path-like and chunk_size
            yield from load_trace_chunks(self.path, chunk_size=chunk_size)
        except ImportError:
            # Fallback: load full trace and yield chunks
            trace = self.read()

            # Import here to avoid circular dependency
            from oscura.core.types import IQTrace

            # Get sample count based on trace type
            if isinstance(trace, IQTrace):
                n_samples = len(trace.i_data)
            else:
                n_samples = len(trace.data)

            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                # Create chunk trace with sliced data
                if isinstance(trace, IQTrace):
                    # IQTrace doesn't have .data attribute
                    chunk_data = None  # Will be handled separately below
                else:
                    chunk_data = trace.data[start:end]

                # Import here to avoid circular dependency
                from oscura.core.types import (
                    DigitalTrace,
                    TraceMetadata,
                    WaveformTrace,
                )

                # Create appropriate trace type
                chunk_metadata = TraceMetadata(
                    sample_rate=trace.metadata.sample_rate,
                    vertical_scale=trace.metadata.vertical_scale,
                    vertical_offset=trace.metadata.vertical_offset,
                    acquisition_time=trace.metadata.acquisition_time,
                    trigger_info=trace.metadata.trigger_info,
                    source_file=str(self.path),
                    channel_name=trace.metadata.channel_name,
                    calibration_info=trace.metadata.calibration_info,
                )

                if isinstance(trace, WaveformTrace):
                    yield WaveformTrace(data=chunk_data, metadata=chunk_metadata)  # type: ignore[arg-type]
                elif isinstance(trace, DigitalTrace):
                    yield DigitalTrace(
                        data=chunk_data,  # type: ignore[arg-type]
                        metadata=chunk_metadata,
                    )
                elif isinstance(trace, IQTrace):
                    # Handle I/Q separately
                    chunk_i = trace.i_data[start:end]
                    chunk_q = trace.q_data[start:end]
                    yield IQTrace(
                        i_data=chunk_i,
                        q_data=chunk_q,
                        metadata=chunk_metadata,
                    )

    def close(self) -> None:
        """Close the source and release resources.

        For file sources, this is mostly a no-op since Python handles
        file cleanup. Included for protocol compliance.

        Example:
            >>> source = FileSource("capture.wfm")
            >>> trace = source.read()
            >>> source.close()
        """
        self._closed = True

    def __enter__(self) -> FileSource:
        """Context manager entry.

        Returns:
            Self for use in 'with' statement.

        Example:
            >>> with FileSource("capture.wfm") as source:
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
        return f"FileSource({self.path!r}, format={self.format!r})"


__all__ = ["FileSource"]
