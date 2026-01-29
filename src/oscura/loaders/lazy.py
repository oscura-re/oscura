"""Lazy loading for huge waveform files.

This module provides memory-mapped file loading where metadata is loaded
immediately but data is deferred until first access. Useful for multi-GB files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import LoaderError

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from oscura.core.types import WaveformTrace


class LazyWaveformTrace:
    """Lazy-loading wrapper for WaveformTrace.

    Loads metadata immediately but defers data loading until first access.
    Uses numpy.memmap for efficient memory-mapped file access.

    Example:
        >>> from oscura.loaders.lazy import load_trace_lazy
        >>> trace = load_trace_lazy('huge_trace.npy', lazy=True)
        >>> # Metadata available immediately
        >>> print(f"Length: {trace.length}, Sample rate: {trace.sample_rate}")
        >>> # Data loaded on first access
        >>> data = trace.data  # Loads data now
        >>> subset = trace[1000:2000]  # Only loads requested slice

    References:
        API-017: Lazy Loading for Huge Files
    """

    def __init__(
        self,
        file_path: str | Path,
        sample_rate: float,
        length: int,
        *,
        dtype: DTypeLike = np.float64,
        offset: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize lazy trace.

        Args:
            file_path: Path to binary data file.
            sample_rate: Sample rate in Hz.
            length: Number of samples.
            dtype: Data type of samples.
            offset: Byte offset to start of data in file.
            metadata: Additional metadata.

        Raises:
            LoaderError: If file not found.

        Example:
            >>> trace = LazyWaveformTrace(
            ...     file_path='trace.npy',
            ...     sample_rate=1e9,
            ...     length=10_000_000
            ... )
        """
        self._file_path = Path(file_path)
        self._sample_rate = sample_rate
        self._length = length
        self._dtype = np.dtype(dtype)
        self._offset = offset
        self._metadata = metadata or {}

        # Deferred data - loaded on first access
        self._data: NDArray[np.float64] | None = None
        self._memmap: np.memmap[Any, np.dtype[Any]] | None = None

        # Verify file exists
        if not self._file_path.exists():
            raise LoaderError(f"File not found: {self._file_path}")

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def length(self) -> int:
        """Number of samples."""
        return self._length

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._length / self._sample_rate

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dictionary."""
        return self._metadata

    @property
    def data(self) -> NDArray[np.float64]:
        """Waveform data array.

        Loads data on first access. Subsequent accesses return cached data.

        Returns:
            Numpy array of waveform samples.
        """
        if self._data is None:
            self._load_data()
        return self._data  # type: ignore[return-value]

    @property
    def time_vector(self) -> NDArray[np.float64]:
        """Time vector in seconds.

        Returns:
            Array of time values corresponding to samples.
        """
        return np.arange(self._length) / self._sample_rate

    def _load_data(self) -> None:
        """Load data from file using memory mapping."""
        try:
            # Use memmap for efficient access
            self._memmap = np.memmap(
                self._file_path,
                dtype=self._dtype,
                mode="r",
                offset=self._offset,
                shape=(self._length,),
            )

            # Convert to regular array (copies data into memory)
            self._data = np.array(self._memmap, dtype=np.float64)

        except Exception as e:
            raise LoaderError(f"Failed to load data from {self._file_path}: {e}") from e

    def __getitem__(self, key: int | slice) -> LazyWaveformTrace | float:
        """Slice the trace.

        Slicing remains lazy - only loads requested portion.

        Args:
            key: Index or slice.

        Returns:
            LazyWaveformTrace for slice, float for single index.

        Raises:
            TypeError: If key is not int or slice.

        Example:
            >>> subset = trace[1000:2000]  # Lazy - doesn't load full data
            >>> sample = trace[500]  # Loads single sample
        """
        if isinstance(key, int):
            # Load single sample
            if self._memmap is None:
                self._memmap = np.memmap(
                    self._file_path,
                    dtype=self._dtype,
                    mode="r",
                    offset=self._offset,
                    shape=(self._length,),
                )
            return float(self._memmap[key])

        elif isinstance(key, slice):
            # Create new lazy trace for slice
            start, stop, step = key.indices(self._length)

            if step != 1:
                # Non-unit step requires loading data
                if self._data is None:
                    self._load_data()
                sliced_data = self._data[key]  # type: ignore[index]

                # Return eager trace
                from oscura.core.types import TraceMetadata, WaveformTrace

                metadata = TraceMetadata(
                    sample_rate=self._sample_rate,
                    **self._metadata,
                )
                return WaveformTrace(data=sliced_data, metadata=metadata)  # type: ignore[return-value]

            # Create lazy slice
            length = stop - start
            offset = self._offset + start * self._dtype.itemsize

            return LazyWaveformTrace(
                file_path=self._file_path,
                sample_rate=self._sample_rate,
                length=length,
                dtype=self._dtype,
                offset=offset,
                metadata=self._metadata.copy(),
            )

        else:
            raise TypeError(f"Indices must be int or slice, not {type(key)}")

    def to_eager(self) -> WaveformTrace:
        """Convert to regular WaveformTrace by loading all data.

        Returns:
            WaveformTrace with data loaded in memory.

        Example:
            >>> eager_trace = lazy_trace.to_eager()
        """
        from oscura.core.types import TraceMetadata, WaveformTrace

        metadata = TraceMetadata(
            sample_rate=self._sample_rate,
            **self._metadata,
        )
        return WaveformTrace(data=self.data, metadata=metadata)

    def close(self) -> None:
        """Close memory-mapped file handle.

        Should be called when done with the trace to free resources.

        Example:
            >>> trace = load_trace_lazy('huge_file.npy', lazy=True)
            >>> # ... use trace ...
            >>> trace.close()
        """
        if self._memmap is not None:
            del self._memmap
            self._memmap = None

    def __del__(self) -> None:
        """Cleanup memory map on deletion."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LazyWaveformTrace(file={self._file_path.name}, "
            f"sample_rate={self._sample_rate:.2e}, "
            f"length={self._length}, "
            f"loaded={self._data is not None})"
        )

    def __len__(self) -> int:
        """Number of samples."""
        return self._length


def load_trace_lazy(
    file_path: str | Path,
    sample_rate: float | None = None,
    *,
    lazy: bool = True,
    **kwargs: Any,
) -> LazyWaveformTrace | WaveformTrace:
    """Load trace with optional lazy loading.

    Loads metadata immediately but defers data loading until first access
    if lazy=True.

    Args:
        file_path: Path to trace file (must be .npy or raw binary).
        sample_rate: Sample rate in Hz (required for raw files).
        lazy: If True, defer data loading. If False, load immediately.
        **kwargs: Additional arguments (dtype, offset, etc.).

    Returns:
        LazyWaveformTrace if lazy=True, otherwise WaveformTrace.

    Raises:
        LoaderError: If file not found or has invalid format.

    Example:
        >>> # Lazy loading for huge files
        >>> trace = load_trace_lazy('10GB_trace.npy', lazy=True)
        >>> print(f"Duration: {trace.duration} seconds")  # No data loaded yet
        >>> data_subset = trace[1000:2000].data  # Only loads this slice
        >>>
        >>> # Eager loading
        >>> trace = load_trace_lazy('small_trace.npy', lazy=False)
        >>> data = trace.data  # Already loaded

    References:
        API-017: Lazy Loading for Huge Files
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LoaderError(f"File not found: {file_path}")

    if sample_rate is None:
        raise LoaderError("sample_rate is required")

    # Determine format and extract metadata
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        length, dtype, offset = _extract_npy_metadata(file_path)
    else:
        length, dtype, offset = _extract_raw_metadata(file_path, kwargs)

    # Return lazy or eager trace
    if lazy:
        return LazyWaveformTrace(
            file_path=file_path,
            sample_rate=sample_rate,
            length=length,
            dtype=dtype,
            offset=offset,
        )
    else:
        return _load_eager_trace(file_path, sample_rate, length, dtype, offset, suffix == ".npy")


def _extract_npy_metadata(file_path: Path) -> tuple[int, DTypeLike, int]:
    """Extract metadata from NumPy file without loading data.

    Args:
        file_path: Path to .npy file.

    Returns:
        Tuple of (length, dtype, offset).

    Raises:
        LoaderError: If file format is invalid.
    """
    with open(file_path, "rb") as f:
        import numpy.lib.format as npf

        npf.read_magic(f)  # type: ignore[no-untyped-call]
        shape, _fortran_order, dtype = npf.read_array_header_1_0(f)  # type: ignore[no-untyped-call]
        offset = f.tell()

    if not isinstance(shape, tuple) or len(shape) != 1:
        raise LoaderError(f"Expected 1D array, got shape {shape}")

    return shape[0], dtype, offset


def _extract_raw_metadata(file_path: Path, kwargs: dict[str, Any]) -> tuple[int, DTypeLike, int]:
    """Extract metadata from raw binary file.

    Args:
        file_path: Path to raw binary file.
        kwargs: Additional arguments (dtype, offset).

    Returns:
        Tuple of (length, dtype, offset).
    """
    dtype = kwargs.get("dtype", np.float64)
    offset = kwargs.get("offset", 0)

    file_size = file_path.stat().st_size - offset
    dtype_size = np.dtype(dtype).itemsize
    length = file_size // dtype_size

    return length, dtype, offset


def _load_eager_trace(
    file_path: Path,
    sample_rate: float,
    length: int,
    dtype: DTypeLike,
    offset: int,
    is_npy: bool,
) -> WaveformTrace:
    """Load trace data eagerly into memory.

    Args:
        file_path: Path to trace file.
        sample_rate: Sample rate in Hz.
        length: Number of samples.
        dtype: Data type.
        offset: Byte offset in file.
        is_npy: True if .npy format, False if raw binary.

    Returns:
        WaveformTrace with data loaded.
    """
    from oscura.core.types import TraceMetadata, WaveformTrace

    if is_npy:
        data = np.load(file_path).astype(np.float64)
    else:
        data = np.fromfile(file_path, dtype=dtype, count=length, offset=offset).astype(np.float64)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


__all__ = ["LazyWaveformTrace", "load_trace_lazy"]
