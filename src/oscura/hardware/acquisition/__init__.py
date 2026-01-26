"""Unified acquisition layer for Oscura.

This module provides the Source protocol - a unified interface for acquiring
signal data from any source (files, hardware, synthetic generation).

The Source protocol enables polymorphic data acquisition:
- FileSource: Load from oscilloscope file formats
- HardwareSource: Acquire from live hardware (SocketCAN, Saleae, PyVISA)
- SyntheticSource: Generate synthetic test signals

All sources implement the same interface, making them interchangeable:

Example:
    >>> from oscura.hardware.acquisition import FileSource, HardwareSource, SyntheticSource
    >>> from oscura import SignalBuilder
    >>>
    >>> # All sources use the same interface
    >>> file_src = FileSource("capture.wfm")
    >>> hw_src = HardwareSource.socketcan("can0", bitrate=500000)  # Future
    >>> synth_src = SyntheticSource(SignalBuilder().sine(freq=1000))
    >>>
    >>> # Polymorphic consumption
    >>> def analyze_from_source(source: Source):
    ...     trace = source.read()
    ...     return analyze(trace)
    >>>
    >>> # Works with any source
    >>> analyze_from_source(file_src)
    >>> analyze_from_source(synth_src)

Pattern Decision:
    - Use Source.read() for one-shot acquisition (complete trace)
    - Use Source.stream() for chunked/streaming acquisition (large files or hardware)
    - All sources are context managers (use 'with' for resource cleanup)

References:
    Architecture Plan Phase 0.1: Unified Acquisition Layer
    docs/architecture/api-patterns.md: When to use Source vs load()
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from oscura.core.types import Trace


@runtime_checkable
class Source(Protocol):
    """Unified acquisition interface for all data sources.

    This protocol defines the contract that all acquisition sources must implement.
    Sources can be files, hardware devices, or synthetic signal generators.

    Methods:
        read: Read complete trace (one-shot acquisition)
        stream: Stream trace in chunks (for large files or continuous acquisition)
        close: Release resources (e.g., file handles, device connections)

    Example:
        >>> class CustomSource:
        ...     def read(self) -> Trace:
        ...         # Load/acquire complete trace
        ...         return trace
        ...
        ...     def stream(self, chunk_size: int) -> Iterator[Trace]:
        ...         # Yield trace chunks
        ...         while has_data:
        ...             yield chunk
        ...
        ...     def close(self) -> None:
        ...         # Clean up resources
        ...         pass
        ...
        ...     def __enter__(self):
        ...         return self
        ...
        ...     def __exit__(self, *args):
        ...         self.close()

    Protocol Compliance:
        Any class implementing these methods can be used as a Source, even
        without explicit inheritance. This enables duck typing and flexibility.
    """

    def read(self) -> Trace:
        """Read complete trace (one-shot acquisition).

        Returns:
            Complete trace from source (WaveformTrace, DigitalTrace, or IQTrace).

        Raises:
            LoaderError: If acquisition fails.
            FileNotFoundError: If source file doesn't exist (FileSource).

        Example:
            >>> source = FileSource("capture.wfm")
            >>> trace = source.read()
            >>> print(f"Loaded {len(trace.data)} samples")
        """

    def stream(self, chunk_size: int) -> Iterator[Trace]:
        """Stream trace in chunks (for large files or continuous acquisition).

        Args:
            chunk_size: Number of samples per chunk.

        Yields:
            Trace chunks (each chunk is a complete Trace object).

        Example:
            >>> source = FileSource("huge_capture.wfm")
            >>> for chunk in source.stream(chunk_size=10000):
            ...     process_chunk(chunk)
        """

    def close(self) -> None:
        """Release resources (file handles, device connections, etc.).

        Called automatically when using source as context manager.

        Example:
            >>> with FileSource("capture.wfm") as source:
            ...     trace = source.read()
            ...  # close() called automatically
        """

    def __enter__(self) -> Source:
        """Context manager entry."""

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""


# Import concrete implementations
from oscura.hardware.acquisition.file import FileSource
from oscura.hardware.acquisition.hardware import HardwareSource
from oscura.hardware.acquisition.synthetic import SyntheticSource

__all__ = [
    "FileSource",
    "HardwareSource",
    "Source",
    "SyntheticSource",
]
