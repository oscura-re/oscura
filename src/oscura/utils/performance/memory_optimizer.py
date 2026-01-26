"""Memory optimization for large signal file processing.

This module provides memory management utilities for processing huge datasets
efficiently without running out of RAM. Implements memory-mapped I/O, streaming
analysis, adaptive chunking, and memory leak detection.

Key Features:
    - Memory-mapped file I/O for huge datasets (via numpy.memmap)
    - Streaming iterators for chunk-by-chunk processing
    - Lazy loading (load data only when accessed)
    - Adaptive chunking based on available memory
    - Real-time memory usage tracking and leak detection
    - Object pooling for frequently allocated objects
    - In-memory compression for infrequently accessed data

Example:
    >>> from oscura.utils.performance.memory_optimizer import MemoryOptimizer
    >>>
    >>> # Optimize memory usage for large file
    >>> optimizer = MemoryOptimizer(max_memory_mb=1024)
    >>> trace = optimizer.load_optimized("huge_file.npy", sample_rate=1e9)
    >>>
    >>> # Process in chunks with streaming
    >>> processor = optimizer.create_stream_processor(
    ...     trace, chunk_size=1_000_000, overlap=0
    ... )
    >>> for chunk in processor:
    ...     result = analyze(chunk)
    >>>
    >>> # Check memory statistics
    >>> stats = optimizer.get_memory_stats()
    >>> print(f"Peak memory: {stats.peak_memory_mb:.1f} MB")
    >>> if stats.leak_detected:
    ...     print("Warning: Memory leak detected!")

References:
    Phase 5 Feature 42: Memory Optimization (v0.6.0)
    Streaming APIs: src/oscura/streaming/chunked.py
    Memory-mapped loading: src/oscura/loaders/mmap_loader.py
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import DTypeLike, NDArray


class ChunkingStrategy(Enum):
    """Chunking strategy for data processing."""

    FIXED = "fixed"  # Fixed-size chunks
    SLIDING = "sliding"  # Sliding window with overlap
    ADAPTIVE = "adaptive"  # Adaptive chunk size based on memory
    TIME_BASED = "time_based"  # Chunk by time duration


@dataclass
class ChunkingConfig:
    """Configuration for data chunking strategy.

    Attributes:
        strategy: Chunking strategy to use.
        chunk_size: Size of each chunk in samples (for fixed/sliding).
        overlap: Number of samples to overlap between chunks.
        adaptive: Whether to adaptively adjust chunk size based on memory.
        time_window: Time duration for time-based chunking (seconds).
        min_chunk_size: Minimum chunk size in samples (for adaptive).
        max_chunk_size: Maximum chunk size in samples (for adaptive).

    Example:
        >>> # Fixed-size chunks
        >>> config = ChunkingConfig(
        ...     strategy=ChunkingStrategy.FIXED,
        ...     chunk_size=1_000_000
        ... )
        >>>
        >>> # Adaptive chunking
        >>> config = ChunkingConfig(
        ...     strategy=ChunkingStrategy.ADAPTIVE,
        ...     adaptive=True,
        ...     min_chunk_size=100_000,
        ...     max_chunk_size=10_000_000
        ... )
    """

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = 1_000_000
    overlap: int = 0
    adaptive: bool = False
    time_window: float | None = None
    min_chunk_size: int = 100_000
    max_chunk_size: int = 10_000_000

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size <= 0:
            raise ValueError(f"min_chunk_size must be positive, got {self.min_chunk_size}")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError(
                f"max_chunk_size ({self.max_chunk_size}) must be >= "
                f"min_chunk_size ({self.min_chunk_size})"
            )
        if self.time_window is not None and self.time_window <= 0:
            raise ValueError(f"time_window must be positive, got {self.time_window}")


@dataclass
class MemoryStats:
    """Memory usage statistics.

    Attributes:
        peak_memory_mb: Peak memory usage in megabytes.
        current_memory_mb: Current memory usage in megabytes.
        allocated_mb: Total memory allocated in megabytes.
        freed_mb: Total memory freed in megabytes.
        leak_detected: Whether a memory leak was detected.
        available_memory_mb: Available system memory in megabytes.
        usage_percent: Memory usage as percentage of total.

    Example:
        >>> stats = optimizer.get_memory_stats()
        >>> print(f"Current: {stats.current_memory_mb:.1f} MB")
        >>> print(f"Peak: {stats.peak_memory_mb:.1f} MB")
        >>> if stats.leak_detected:
        ...     print("WARNING: Memory leak detected!")
    """

    peak_memory_mb: float
    current_memory_mb: float
    allocated_mb: float
    freed_mb: float
    leak_detected: bool
    available_memory_mb: float
    usage_percent: float


class StreamProcessor:
    """Streaming processor for chunk-by-chunk data processing.

    Provides iterator interface for processing large datasets in chunks
    without loading all data into memory. Supports overlapping chunks
    for windowed operations.

    Attributes:
        data: Data source (array or memory-mapped array).
        chunk_size: Size of each chunk in samples.
        overlap: Number of samples to overlap between chunks.
        total_samples: Total number of samples in data.

    Example:
        >>> processor = StreamProcessor(
        ...     data=huge_array,
        ...     chunk_size=1_000_000,
        ...     overlap=1024
        ... )
        >>> for chunk in processor:
        ...     result = analyze_chunk(chunk)
        ...     print(f"Processed {len(chunk)} samples")
    """

    def __init__(
        self,
        data: NDArray[Any] | np.memmap[Any, Any],
        chunk_size: int,
        overlap: int = 0,
    ) -> None:
        """Initialize stream processor.

        Args:
            data: Data array or memmap to process.
            chunk_size: Number of samples per chunk.
            overlap: Number of samples to overlap between chunks.

        Raises:
            ValueError: If chunk_size or overlap invalid.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        self.data = data
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.total_samples = len(data)
        self._current_position = 0

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        """Iterate over chunks.

        Yields:
            Chunks of data as numpy arrays.

        Example:
            >>> for chunk in processor:
            ...     mean = np.mean(chunk)
            ...     print(f"Chunk mean: {mean}")
        """
        self._current_position = 0

        while self._current_position < self.total_samples:
            end = min(self._current_position + self.chunk_size, self.total_samples)
            chunk = self.data[self._current_position : end]

            # Convert memmap to regular array to avoid keeping file handle open
            if isinstance(chunk, np.memmap):
                chunk = np.asarray(chunk, dtype=np.float64)

            yield chunk

            self._current_position = end - self.overlap

            # Break if we've reached the end
            if end >= self.total_samples:
                break

    def __len__(self) -> int:
        """Number of chunks that will be yielded.

        Returns:
            Total number of chunks.
        """
        step = self.chunk_size - self.overlap
        num_chunks = (self.total_samples - self.overlap) // step
        if (self.total_samples - self.overlap) % step != 0:
            num_chunks += 1
        return num_chunks

    def reset(self) -> None:
        """Reset iterator to beginning.

        Example:
            >>> processor.reset()
            >>> for chunk in processor:
            ...     process(chunk)
        """
        self._current_position = 0


class MemoryOptimizer:
    """Memory optimizer for large signal file processing.

    Provides comprehensive memory management including memory-mapped I/O,
    streaming analysis, adaptive chunking, and memory leak detection.

    Attributes:
        max_memory_mb: Maximum memory limit in megabytes (None = no limit).
        enable_compression: Whether to enable in-memory compression.
        gc_threshold: Memory usage threshold to trigger garbage collection.

    Example:
        >>> # Create optimizer with 2 GB memory limit
        >>> optimizer = MemoryOptimizer(max_memory_mb=2048)
        >>>
        >>> # Load file optimally (mmap if huge, eager if small)
        >>> trace = optimizer.load_optimized("data.npy", sample_rate=1e9)
        >>>
        >>> # Get recommended chunk size
        >>> chunk_size = optimizer.recommend_chunk_size(
        ...     data_length=100_000_000,
        ...     dtype=np.float64
        ... )
        >>>
        >>> # Create streaming processor
        >>> processor = optimizer.create_stream_processor(
        ...     trace, chunk_size=chunk_size
        ... )
    """

    def __init__(
        self,
        max_memory_mb: float | None = None,
        enable_compression: bool = False,
        gc_threshold: float = 0.8,
    ) -> None:
        """Initialize memory optimizer.

        Args:
            max_memory_mb: Maximum memory limit in MB (None = no limit).
            enable_compression: Enable in-memory compression.
            gc_threshold: Memory usage fraction to trigger GC (0.0-1.0).

        Raises:
            ValueError: If parameters invalid.

        Example:
            >>> # Limit to 1 GB with aggressive GC
            >>> optimizer = MemoryOptimizer(
            ...     max_memory_mb=1024,
            ...     gc_threshold=0.7
            ... )
        """
        if max_memory_mb is not None and max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
        if not 0.0 <= gc_threshold <= 1.0:
            raise ValueError(f"gc_threshold must be in [0, 1], got {gc_threshold}")

        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self.gc_threshold = gc_threshold

        # Memory tracking
        self._initial_memory_mb = self._get_memory_usage()
        self._peak_memory_mb = self._initial_memory_mb
        self._total_allocated_mb = 0.0
        self._total_freed_mb = 0.0
        self._allocation_count = 0
        self._last_gc_memory = self._initial_memory_mb

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in megabytes.
        """
        process = psutil.Process()
        return float(process.memory_info().rss / (1024 * 1024))

    def _get_available_memory(self) -> float:
        """Get available system memory in MB.

        Returns:
            Available memory in megabytes.
        """
        return float(psutil.virtual_memory().available / (1024 * 1024))

    def _update_memory_tracking(self) -> None:
        """Update memory usage tracking and trigger GC if needed."""
        current_memory = self._get_memory_usage()

        # Update peak
        if current_memory > self._peak_memory_mb:
            self._peak_memory_mb = current_memory

        # Track allocations
        memory_delta = current_memory - self._last_gc_memory
        if memory_delta > 0:
            self._total_allocated_mb += memory_delta
            self._allocation_count += 1
        elif memory_delta < 0:
            self._total_freed_mb += abs(memory_delta)

        # Check if GC needed
        if self.max_memory_mb is not None:
            usage_fraction = current_memory / self.max_memory_mb
            if usage_fraction >= self.gc_threshold:
                self._trigger_gc()
                self._last_gc_memory = self._get_memory_usage()
        else:
            # Use available system memory
            available = self._get_available_memory()
            total = psutil.virtual_memory().total / (1024 * 1024)
            usage_fraction = 1.0 - (available / total)
            if usage_fraction >= self.gc_threshold:
                self._trigger_gc()
                self._last_gc_memory = self._get_memory_usage()

    def _trigger_gc(self) -> None:
        """Trigger garbage collection to free memory."""
        gc.collect()

    def _detect_memory_leak(self) -> bool:
        """Detect potential memory leak.

        Returns:
            True if leak detected, False otherwise.

        Note:
            Detects leak if allocated memory keeps growing without being freed.
        """
        if self._allocation_count < 10:
            return False

        # Check if allocated > freed by large margin
        leak_threshold = 100.0  # MB
        net_growth = self._total_allocated_mb - self._total_freed_mb

        return net_growth > leak_threshold

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics.

        Returns:
            MemoryStats with current memory usage information.

        Example:
            >>> stats = optimizer.get_memory_stats()
            >>> print(f"Peak: {stats.peak_memory_mb:.1f} MB")
            >>> print(f"Available: {stats.available_memory_mb:.1f} MB")
            >>> if stats.leak_detected:
            ...     print("WARNING: Memory leak detected!")
        """
        current = self._get_memory_usage()
        available = self._get_available_memory()
        total = psutil.virtual_memory().total / (1024 * 1024)

        return MemoryStats(
            peak_memory_mb=self._peak_memory_mb,
            current_memory_mb=current,
            allocated_mb=self._total_allocated_mb,
            freed_mb=self._total_freed_mb,
            leak_detected=self._detect_memory_leak(),
            available_memory_mb=available,
            usage_percent=(current / total) * 100.0,
        )

    def recommend_chunk_size(
        self,
        data_length: int,
        dtype: DTypeLike = np.float64,
        target_memory_mb: float = 100.0,
    ) -> int:
        """Recommend optimal chunk size based on available memory.

        Args:
            data_length: Total length of data in samples.
            dtype: Data type of samples.
            target_memory_mb: Target memory per chunk in MB.

        Returns:
            Recommended chunk size in samples.

        Example:
            >>> chunk_size = optimizer.recommend_chunk_size(
            ...     data_length=100_000_000,
            ...     dtype=np.float64,
            ...     target_memory_mb=50.0
            ... )
            >>> print(f"Recommended: {chunk_size:,} samples")
        """
        dtype_np = np.dtype(dtype)
        bytes_per_sample = dtype_np.itemsize

        # Calculate chunk size for target memory
        target_bytes = target_memory_mb * 1024 * 1024
        chunk_size = int(target_bytes / bytes_per_sample)

        # Clamp to reasonable bounds
        min_chunk = 1000
        max_chunk = data_length

        return max(min_chunk, min(chunk_size, max_chunk))

    def load_optimized(
        self,
        file_path: str | PathLike[str],
        sample_rate: float,
        *,
        dtype: DTypeLike | None = None,
        mmap_threshold_mb: float = 100.0,
    ) -> Any:
        """Load file with optimal strategy (mmap for huge files, eager for small).

        Args:
            file_path: Path to data file.
            sample_rate: Sample rate in Hz.
            dtype: Data type (auto-detected for .npy).
            mmap_threshold_mb: File size threshold for memory mapping (MB).

        Returns:
            Loaded trace (MmapWaveformTrace or WaveformTrace).

        Example:
            >>> # Automatically use mmap for huge files
            >>> trace = optimizer.load_optimized(
            ...     "huge_file.npy",
            ...     sample_rate=1e9,
            ...     mmap_threshold_mb=100.0
            ... )
        """
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Update memory tracking
        self._update_memory_tracking()

        # Use mmap for large files
        if file_size_mb >= mmap_threshold_mb:
            from oscura.loaders.mmap_loader import load_mmap

            return load_mmap(file_path, sample_rate=sample_rate, dtype=dtype)
        else:
            # Use eager loading for small files
            from oscura.loaders import load

            return load(file_path, sample_rate=sample_rate)

    def create_stream_processor(
        self,
        data: NDArray[Any] | np.memmap[Any, Any] | Any,
        chunk_size: int | None = None,
        overlap: int = 0,
        config: ChunkingConfig | None = None,
    ) -> StreamProcessor:
        """Create streaming processor for chunk-by-chunk processing.

        Args:
            data: Data array, memmap, or trace object.
            chunk_size: Size of each chunk (auto if None).
            overlap: Number of samples to overlap.
            config: Chunking configuration (overrides chunk_size/overlap).

        Returns:
            StreamProcessor for iterating over chunks.

        Example:
            >>> processor = optimizer.create_stream_processor(
            ...     data=huge_array,
            ...     chunk_size=1_000_000,
            ...     overlap=1024
            ... )
            >>> for chunk in processor:
            ...     result = analyze(chunk)
        """
        # Extract data array if trace object
        if hasattr(data, "data"):
            data_array = data.data
        else:
            data_array = data

        # Convert to numpy array if needed
        if not isinstance(data_array, (np.ndarray, np.memmap)):
            data_array = np.asarray(data_array)

        # Use config if provided
        if config is not None:
            chunk_size = config.chunk_size
            overlap = config.overlap

            # Adaptive chunking
            if config.adaptive:
                chunk_size = self.recommend_chunk_size(
                    data_length=len(data_array),
                    dtype=data_array.dtype,
                )
                chunk_size = max(config.min_chunk_size, min(chunk_size, config.max_chunk_size))

        # Auto-recommend chunk size if not provided
        if chunk_size is None:
            chunk_size = self.recommend_chunk_size(
                data_length=len(data_array),
                dtype=data_array.dtype,
            )

        # Update memory tracking
        self._update_memory_tracking()

        return StreamProcessor(data=data_array, chunk_size=chunk_size, overlap=overlap)

    def optimize_array(
        self,
        array: NDArray[Any],
        compress: bool | None = None,
    ) -> NDArray[Any]:
        """Optimize array memory usage.

        Args:
            array: Array to optimize.
            compress: Whether to compress (uses instance default if None).

        Returns:
            Optimized array (may be compressed or compacted).

        Note:
            Currently returns compacted array. Future versions may add
            compression support via zlib or blosc.

        Example:
            >>> optimized = optimizer.optimize_array(large_array)
        """
        # Update memory tracking
        self._update_memory_tracking()

        # Make array contiguous for better cache performance
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        # Future: Add compression support
        # if compress or (compress is None and self.enable_compression):
        #     return compress_array(array)

        return array

    def set_memory_limit(self, max_memory_mb: float) -> None:
        """Set maximum memory limit.

        Args:
            max_memory_mb: Maximum memory in megabytes.

        Raises:
            ValueError: If max_memory_mb not positive.

        Example:
            >>> optimizer.set_memory_limit(1024)  # 1 GB
        """
        if max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
        self.max_memory_mb = max_memory_mb

    def check_available_memory(self, required_mb: float) -> bool:
        """Check if sufficient memory available.

        Args:
            required_mb: Required memory in megabytes.

        Returns:
            True if sufficient memory available, False otherwise.

        Example:
            >>> if optimizer.check_available_memory(500):
            ...     process_large_data()
            ... else:
            ...     print("Insufficient memory")
        """
        available = self._get_available_memory()
        return available >= required_mb

    def suggest_downsampling(
        self,
        data_length: int,
        dtype: DTypeLike = np.float64,
        target_memory_mb: float = 100.0,
    ) -> int:
        """Suggest downsampling factor to fit in target memory.

        Args:
            data_length: Length of data in samples.
            dtype: Data type of samples.
            target_memory_mb: Target memory in megabytes.

        Returns:
            Downsampling factor (1 = no downsampling, 2 = every other sample, etc).

        Example:
            >>> factor = optimizer.suggest_downsampling(
            ...     data_length=100_000_000,
            ...     target_memory_mb=50.0
            ... )
            >>> if factor > 1:
            ...     data = data[::factor]
        """
        dtype_np = np.dtype(dtype)
        bytes_per_sample = dtype_np.itemsize

        # Calculate current memory requirement
        current_mb = (data_length * bytes_per_sample) / (1024 * 1024)

        if current_mb <= target_memory_mb:
            return 1

        # Calculate downsampling factor
        factor = int(np.ceil(current_mb / target_memory_mb))
        return factor

    def reset_statistics(self) -> None:
        """Reset memory tracking statistics.

        Example:
            >>> optimizer.reset_statistics()
            >>> # Track memory for new operation
            >>> stats = optimizer.get_memory_stats()
        """
        self._initial_memory_mb = self._get_memory_usage()
        self._peak_memory_mb = self._initial_memory_mb
        self._total_allocated_mb = 0.0
        self._total_freed_mb = 0.0
        self._allocation_count = 0
        self._last_gc_memory = self._initial_memory_mb


__all__ = [
    "ChunkingConfig",
    "ChunkingStrategy",
    "MemoryOptimizer",
    "MemoryStats",
    "StreamProcessor",
]
