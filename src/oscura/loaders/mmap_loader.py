"""Memory-mapped file loader for huge waveform files.

This module provides efficient memory-mapped loading for GB+ files that cannot
fit in RAM. Unlike eager loading, memory-mapped arrays don't load the entire
file into memory but access it in chunks on-demand via the OS page cache.

Key features:
- Zero-copy data access via numpy.memmap
- Chunked iteration for processing huge files
- Integration with existing Oscura loader infrastructure
- Support for common binary formats (raw, NPY, structured)
- Automatic fallback to regular loading for small files

Example:
    >>> from oscura.loaders.mmap_loader import load_mmap
    >>> # Load 10 GB file without loading all data to RAM
    >>> trace = load_mmap("huge_trace.npy", sample_rate=1e9)
    >>> print(f"Length: {len(trace)} samples")
    >>>
    >>> # Process in chunks to avoid OOM
    >>> for chunk in trace.iter_chunks(chunk_size=1_000_000):
    ...     result = analyze_chunk(chunk)

References:
    Performance optimization for huge files (>1 GB)
    API-017: Lazy Loading for Huge Files
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import LoaderError
from oscura.core.types import TraceMetadata

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import DTypeLike, NDArray


# File size threshold for automatic mmap suggestion (1 GB)
MMAP_THRESHOLD = 1024 * 1024 * 1024


class MmapWaveformTrace:
    """Memory-mapped waveform trace for huge files.

    Provides lazy access to waveform data via memory mapping. Data is not
    loaded into RAM but accessed directly from disk through the OS page cache.

    This allows working with files larger than available RAM without OOM errors.

    Attributes:
        file_path: Path to the memory-mapped file.
        sample_rate: Sample rate in Hz.
        length: Number of samples in the trace.
        dtype: NumPy dtype of the samples.
        metadata: Additional trace metadata.

    Example:
        >>> trace = MmapWaveformTrace(
        ...     file_path="huge_trace.bin",
        ...     sample_rate=1e9,
        ...     length=10_000_000_000,
        ...     dtype=np.float32
        ... )
        >>> # Access subset without loading entire file
        >>> subset = trace[1000:2000]
        >>> # Process in chunks
        >>> for chunk in trace.iter_chunks(chunk_size=1_000_000):
        ...     process(chunk)
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
        mode: str = "r",
    ) -> None:
        """Initialize memory-mapped trace.

        Args:
            file_path: Path to binary data file.
            sample_rate: Sample rate in Hz.
            length: Number of samples.
            dtype: Data type of samples.
            offset: Byte offset to start of data in file.
            metadata: Additional metadata dictionary.
            mode: File access mode ('r' for read-only, 'r+' for read-write).

        Raises:
            LoaderError: If file not found or invalid parameters.

        Example:
            >>> trace = MmapWaveformTrace(
            ...     file_path="trace.f32",
            ...     sample_rate=1e9,
            ...     length=1_000_000_000,
            ...     dtype=np.float32
            ... )
        """
        self._file_path = Path(file_path)
        self._sample_rate = float(sample_rate)
        self._length = int(length)
        self._dtype = np.dtype(dtype)
        self._offset = int(offset)
        self._metadata = metadata or {}
        self._mode = mode

        # Memory-mapped array - created on first access
        self._memmap: np.memmap[Any, np.dtype[Any]] | None = None

        # Validate inputs
        if self._sample_rate <= 0:
            raise LoaderError(f"sample_rate must be positive, got {self._sample_rate}")
        if self._length < 0:
            raise LoaderError(f"length must be non-negative, got {self._length}")
        if self._offset < 0:
            raise LoaderError(f"offset must be non-negative, got {self._offset}")

        # Verify file exists
        if not self._file_path.exists():
            raise LoaderError(f"File not found: {self._file_path}")

        # Verify file size
        expected_size = self._offset + self._length * self._dtype.itemsize
        actual_size = self._file_path.stat().st_size
        if actual_size < expected_size:
            raise LoaderError(
                f"File too small for requested data. "
                f"Expected at least {expected_size} bytes, got {actual_size} bytes",
                file_path=str(self._file_path),
            )

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
    def dtype(self) -> np.dtype[Any]:
        """Data type of samples."""
        return self._dtype

    @property
    def file_path(self) -> Path:
        """Path to memory-mapped file."""
        return self._file_path

    @property
    def data(self) -> np.memmap[Any, np.dtype[Any]]:
        """Memory-mapped data array.

        Returns a numpy.memmap object that behaves like a numpy array
        but doesn't load data into memory until accessed.

        Returns:
            Memory-mapped numpy array.

        Example:
            >>> trace = load_mmap("huge.npy", sample_rate=1e9)
            >>> data = trace.data  # No data loaded yet
            >>> subset = data[1000:2000]  # Only this range loaded
        """
        if self._memmap is None:
            self._memmap = np.memmap(  # type: ignore[call-overload]
                str(self._file_path),
                dtype=self._dtype,
                mode=self._mode,
                offset=self._offset,
                shape=(self._length,),
            )
        return self._memmap

    @property
    def time_vector(self) -> NDArray[np.float64]:
        """Time vector in seconds.

        Note: For huge traces, this can consume significant memory.
        Consider using time values on-demand instead.

        Returns:
            Array of time values corresponding to samples.

        Example:
            >>> # For huge traces, avoid materializing full time vector
            >>> # Instead compute on-demand:
            >>> t_start = 0
            >>> t_end = trace.length / trace.sample_rate
        """
        return np.arange(self._length, dtype=np.float64) / self._sample_rate

    def __getitem__(self, key: int | slice) -> float | NDArray[np.float64]:
        """Slice the memory-mapped trace.

        Supports both integer indexing and slicing. Only the requested
        portion is loaded from disk.

        Args:
            key: Index or slice.

        Returns:
            Single sample (float) or array slice.

        Raises:
            TypeError: If key is not int or slice.

        Example:
            >>> sample = trace[1000]  # Load single sample
            >>> chunk = trace[1000:2000]  # Load 1000 samples
            >>> every_10th = trace[::10]  # Load decimated data
        """
        if isinstance(key, (int, slice)):
            return self.data[key]
        else:
            raise TypeError(f"Indices must be int or slice, not {type(key).__name__}")

    def __len__(self) -> int:
        """Number of samples."""
        return self._length

    def iter_chunks(
        self, chunk_size: int = 1_000_000, overlap: int = 0
    ) -> Iterator[NDArray[np.float64]]:
        """Iterate over trace in chunks.

        Yields consecutive chunks of data, optionally with overlap between
        chunks. This is efficient for processing huge files that don't fit
        in memory.

        Args:
            chunk_size: Number of samples per chunk.
            overlap: Number of samples to overlap between chunks.

        Yields:
            Numpy arrays of chunk_size (or smaller for last chunk).

        Raises:
            ValueError: If chunk_size or overlap invalid.

        Example:
            >>> # Process 10 GB file in 1M sample chunks
            >>> for chunk in trace.iter_chunks(chunk_size=1_000_000):
            ...     result = compute_fft(chunk)
            ...
            >>> # With 50% overlap for windowed processing
            >>> for chunk in trace.iter_chunks(chunk_size=2048, overlap=1024):
            ...     spectrum = analyze_spectrum(chunk)
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        data = self.data
        step = chunk_size - overlap

        for start in range(0, self._length, step):
            end = min(start + chunk_size, self._length)
            # Convert memmap slice to regular array to avoid keeping file handle open
            yield np.asarray(data[start:end], dtype=np.float64)

    def to_eager(self) -> Any:
        """Convert to eager WaveformTrace by loading all data.

        WARNING: This loads the entire file into memory. Only use this
        if you're sure the data fits in RAM.

        Returns:
            WaveformTrace with data loaded in memory.

        Example:
            >>> # Only convert to eager if file is small enough
            >>> if trace.length < 10_000_000:
            ...     eager_trace = trace.to_eager()
        """
        from oscura.core.types import WaveformTrace

        # Load all data into memory
        data = np.asarray(self.data, dtype=np.float64)

        metadata = TraceMetadata(
            sample_rate=self._sample_rate,
            source_file=str(self._file_path),
            **self._metadata,
        )

        return WaveformTrace(data=data, metadata=metadata)

    def close(self) -> None:
        """Close memory-mapped file handle.

        Should be called when done with the trace to free resources.
        The trace cannot be used after closing.

        Example:
            >>> trace = load_mmap("huge.npy", sample_rate=1e9)
            >>> # ... use trace ...
            >>> trace.close()
        """
        if self._memmap is not None:
            # Delete reference to allow garbage collection
            del self._memmap
            self._memmap = None

    def __del__(self) -> None:
        """Cleanup memory map on deletion."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        size_mb = (self._length * self._dtype.itemsize) / (1024 * 1024)
        return (
            f"MmapWaveformTrace("
            f"file={self._file_path.name}, "
            f"sample_rate={self._sample_rate:.2e} Hz, "
            f"length={self._length:,} samples, "
            f"size={size_mb:.1f} MB, "
            f"dtype={self._dtype})"
        )

    def __enter__(self) -> MmapWaveformTrace:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close the file."""
        self.close()


def load_mmap(
    file_path: str | PathLike[str],
    sample_rate: float | None = None,
    *,
    dtype: DTypeLike | None = None,
    offset: int = 0,
    length: int | None = None,
    mode: str = "r",
    **metadata: Any,
) -> MmapWaveformTrace:
    """Load waveform file with memory mapping.

    Creates a memory-mapped trace that doesn't load data into RAM.
    Supports .npy files (auto-detects format) and raw binary files.

    Args:
        file_path: Path to waveform file (.npy or raw binary).
        sample_rate: Sample rate in Hz (required for raw files, optional for .npy).
        dtype: Data type (required for raw files, auto-detected for .npy).
        offset: Byte offset to data start (auto-computed for .npy).
        length: Number of samples (auto-computed if possible).
        mode: File access mode ('r' for read-only, 'r+' for read-write).
        **metadata: Additional metadata to store.

    Returns:
        MmapWaveformTrace for memory-mapped access.

    Raises:
        LoaderError: If file not found or parameters invalid.

    Example:
        >>> # Load NumPy file (auto-detects format)
        >>> trace = load_mmap("huge_trace.npy", sample_rate=1e9)
        >>>
        >>> # Load raw binary file
        >>> trace = load_mmap(
        ...     "data.f32",
        ...     sample_rate=1e9,
        ...     dtype=np.float32,
        ...     length=1_000_000_000
        ... )
        >>>
        >>> # Use context manager
        >>> with load_mmap("huge.npy", sample_rate=1e9) as trace:
        ...     for chunk in trace.iter_chunks(chunk_size=1_000_000):
        ...         process(chunk)

    References:
        API-017: Lazy Loading for Huge Files
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise LoaderError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # Handle .npy files with automatic format detection
    if suffix == ".npy":
        return _load_npy_mmap(file_path, sample_rate, mode, metadata)

    # Handle .npz files (not directly memory-mappable, but can extract)
    elif suffix == ".npz":
        raise LoaderError(
            "NPZ files cannot be directly memory-mapped. "
            "Extract the array first using np.load() and save as .npy",
            file_path=str(file_path),
            fix_hint="Use: np.save('array.npy', np.load('file.npz')['array'])",
        )

    # Handle raw binary files
    else:
        if dtype is None:
            raise LoaderError(
                "dtype is required for raw binary files",
                file_path=str(file_path),
                fix_hint="Specify dtype, e.g., dtype=np.float32",
            )
        if sample_rate is None:
            raise LoaderError(
                "sample_rate is required for raw binary files",
                file_path=str(file_path),
            )

        # Compute length from file size if not provided
        dtype_np = np.dtype(dtype)
        if length is None:
            file_size = file_path.stat().st_size - offset
            length = file_size // dtype_np.itemsize

        return MmapWaveformTrace(
            file_path=file_path,
            sample_rate=sample_rate,
            length=length,
            dtype=dtype_np,
            offset=offset,
            metadata=metadata,
            mode=mode,
        )


def _load_npy_mmap(
    file_path: Path,
    sample_rate: float | None,
    mode: str,
    metadata: dict[str, Any],
) -> MmapWaveformTrace:
    """Load NumPy .npy file with memory mapping.

    Reads the .npy header to extract dtype, shape, and data offset,
    then creates a memory-mapped array.

    Args:
        file_path: Path to .npy file.
        sample_rate: Sample rate in Hz (required).
        mode: File access mode.
        metadata: Additional metadata.

    Returns:
        MmapWaveformTrace for the .npy file.

    Raises:
        LoaderError: If sample_rate not provided or file invalid.
    """
    if sample_rate is None:
        raise LoaderError(
            "sample_rate is required for .npy files",
            file_path=str(file_path),
            fix_hint="Specify sample_rate, e.g., sample_rate=1e9",
        )

    try:
        # Read NumPy header without loading data
        with open(file_path, "rb") as f:
            import numpy.lib.format as npf

            # Read header
            version = npf.read_magic(f)  # type: ignore[no-untyped-call]

            if version == (1, 0):
                shape, fortran_order, dtype = npf.read_array_header_1_0(f)  # type: ignore[no-untyped-call]
            elif version == (2, 0):
                shape, fortran_order, dtype = npf.read_array_header_2_0(f)  # type: ignore[no-untyped-call]
            else:
                raise LoaderError(
                    f"Unsupported NPY version: {version}",
                    file_path=str(file_path),
                )

            # Get data offset
            offset = f.tell()

        # Validate shape
        if not isinstance(shape, tuple):
            raise LoaderError(
                f"Invalid .npy shape: {shape}",
                file_path=str(file_path),
            )

        if len(shape) != 1:
            raise LoaderError(
                f"Expected 1D array, got shape {shape}",
                file_path=str(file_path),
                fix_hint="Reshape to 1D or extract specific column",
            )

        length = shape[0]

        if fortran_order:
            raise LoaderError(
                "Fortran-ordered arrays not supported for memory mapping",
                file_path=str(file_path),
                fix_hint="Resave array in C order: np.save('file.npy', arr, allow_pickle=False)",
            )

        return MmapWaveformTrace(
            file_path=file_path,
            sample_rate=sample_rate,
            length=length,
            dtype=dtype,
            offset=offset,
            metadata=metadata,
            mode=mode,
        )

    except Exception as e:
        if isinstance(e, LoaderError):
            raise
        raise LoaderError(
            f"Failed to load .npy file: {e}",
            file_path=str(file_path),
        ) from e


def should_use_mmap(file_path: str | PathLike[str], threshold: int = MMAP_THRESHOLD) -> bool:
    """Check if file should use memory mapping.

    Recommends memory mapping for files larger than threshold (default 1 GB).

    Args:
        file_path: Path to file.
        threshold: Size threshold in bytes (default: 1 GB).

    Returns:
        True if file size >= threshold, False otherwise.

    Example:
        >>> if should_use_mmap("huge_trace.npy"):
        ...     trace = load_mmap("huge_trace.npy", sample_rate=1e9)
        ... else:
        ...     trace = load("huge_trace.npy", sample_rate=1e9)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    file_size = file_path.stat().st_size
    return file_size >= threshold


__all__ = [
    "MMAP_THRESHOLD",
    "MmapWaveformTrace",
    "load_mmap",
    "should_use_mmap",
]
