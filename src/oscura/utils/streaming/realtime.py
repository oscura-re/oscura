"""Real-time streaming APIs for live data acquisition and processing.

This module provides interfaces for real-time data capture, buffering, and
on-the-fly analysis of streaming waveforms. Supports pluggable input sources,
configurable sample buffers, and streaming statistics.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray


@dataclass
class RealtimeConfig:
    """Configuration for real-time streaming."""

    sample_rate: float
    """Sample rate in Hz."""
    buffer_size: int = 10000
    """Size of the circular buffer in samples."""
    chunk_size: int = 1000
    """Number of samples to yield per chunk."""
    timeout: float = 10.0
    """Timeout in seconds for buffer operations."""
    window_size: int | None = None
    """Window size for rolling statistics. If None, uses buffer_size."""
    enable_validation: bool = True
    """Enable input validation."""

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_size > self.buffer_size:
            raise ValueError("chunk_size cannot exceed buffer_size")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("window_size must be positive")


class RealtimeBuffer:
    """Thread-safe circular buffer for real-time streaming.

    Maintains a fixed-size buffer of the most recent samples with
    thread-safe read/write operations and overflow handling.

    Example:
        >>> buffer = RealtimeBuffer(config)
        >>> buffer.write(samples)
        >>> chunk = buffer.read(chunk_size)
    """

    def __init__(self, config: RealtimeConfig) -> None:
        """Initialize real-time buffer.

        Args:
            config: Realtime configuration.
        """
        config.validate()
        self.config = config

        self._buffer: deque[float] = deque(maxlen=config.buffer_size)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._total_samples = 0
        self._overflow_count = 0

    def write(self, data: NDArray[np.floating[Any]]) -> int:
        """Write samples to buffer.

        Args:
            data: Array of samples to write.

        Returns:
            Number of samples written.

        Raises:
            TypeError: If data is not numeric array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be numpy array")

        if data.dtype.kind not in "fc":  # float or complex
            raise TypeError("data must be float or complex array")

        with self._not_empty:
            initial_len = len(self._buffer)

            # Write samples to buffer
            for sample in data.flat:
                self._buffer.append(float(sample))

            # Track overflow
            if len(self._buffer) == self.config.buffer_size:
                written = self.config.buffer_size - initial_len
                if written < len(data):
                    self._overflow_count += len(data) - written
            else:
                written = len(data)

            self._total_samples += len(data)
            self._not_empty.notify_all()

            return written

    def read(self, n_samples: int, timeout: float | None = None) -> NDArray[np.float64]:
        """Read samples from buffer (blocking).

        Args:
            n_samples: Number of samples to read.
            timeout: Timeout in seconds (None = use config timeout).

        Returns:
            Array of samples, may be shorter if timeout occurs.

        Raises:
            ValueError: If n_samples is invalid.
            TimeoutError: If timeout occurs without sufficient data.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if timeout is None:
            timeout = self.config.timeout

        with self._not_empty:
            # Wait for data with timeout
            if not self._wait_for_data(n_samples, timeout):
                if len(self._buffer) == 0:
                    raise TimeoutError("No data available")

            # Read available samples
            n_read = min(n_samples, len(self._buffer))
            data = np.array(list(self._buffer)[:n_read], dtype=np.float64)
            return data

    def _wait_for_data(self, n_samples: int, timeout: float) -> bool:
        """Wait for minimum data in buffer.

        Args:
            n_samples: Minimum samples to wait for.
            timeout: Timeout in seconds.

        Returns:
            True if sufficient data available, False on timeout.
        """
        deadline = time.time() + timeout
        while len(self._buffer) < n_samples:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            if not self._not_empty.wait(timeout=min(remaining, 0.1)):
                continue
        return True

    def get_available(self) -> int:
        """Get number of available samples in buffer.

        Returns:
            Number of samples currently in buffer.
        """
        with self._lock:
            return len(self._buffer)

    def get_stats(self) -> dict[str, int]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats (total_samples, overflow_count, available).
        """
        with self._lock:
            return {
                "total_samples": self._total_samples,
                "overflow_count": self._overflow_count,
                "available": len(self._buffer),
            }

    def clear(self) -> None:
        """Clear buffer contents.

        Example:
            >>> buffer.clear()
        """
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0
            self._overflow_count = 0

    def close(self) -> None:
        """Close buffer and release resources.

        Example:
            >>> buffer.close()
        """
        self.clear()


class RealtimeSource:
    """Base class for real-time data sources.

    Subclass to implement custom data sources that feed the real-time
    buffer. Must implement the acquire method.

    Example:
        >>> class CustomSource(RealtimeSource):
        ...     def acquire(self) -> np.ndarray:
        ...         # Get data from hardware
        ...         return np.array([...])
    """

    def acquire(self) -> NDArray[np.floating[Any]]:
        """Acquire samples from source.

        Raises:
            NotImplementedError: Subclasses must implement.
        """
        raise NotImplementedError("Subclasses must implement acquire()")

    def start(self) -> None:
        """Start acquisition (optional).

        Default implementation does nothing.
        """

    def stop(self) -> None:
        """Stop acquisition (optional).

        Default implementation does nothing.
        """


class SimulatedSource(RealtimeSource):
    """Simulated data source for testing and examples.

    Generates synthetic waveforms (sine, square, noise) for real-time
    streaming without requiring hardware.

    Args:
        signal_type: Type of signal ("sine", "square", "noise", "mixed").
        frequency: Signal frequency in Hz (for periodic signals).
        amplitude: Signal amplitude.
        sample_rate: Sample rate in Hz.
        chunk_size: Number of samples per acquire() call.
        noise_level: Noise amplitude (0-1 relative to signal).

    Example:
        >>> source = SimulatedSource("sine", frequency=1000, sample_rate=48000)
        >>> data = source.acquire()  # Get one chunk
    """

    def __init__(
        self,
        signal_type: str = "sine",
        *,
        frequency: float = 1000.0,
        amplitude: float = 1.0,
        sample_rate: float = 48000.0,
        chunk_size: int = 1024,
        noise_level: float = 0.0,
    ) -> None:
        """Initialize simulated source."""
        self.signal_type = signal_type
        self.frequency = frequency
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_level = noise_level
        self._phase = 0.0
        self._running = False

    def acquire(self) -> NDArray[np.floating[Any]]:
        """Acquire one chunk of simulated data.

        Returns:
            Array of simulated samples.

        Raises:
            ValueError: If signal_type is not recognized.
        """
        t = np.arange(self.chunk_size) / self.sample_rate
        t += self._phase / (2 * np.pi * self.frequency)

        # Generate base signal
        if self.signal_type == "sine":
            signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        elif self.signal_type == "square":
            signal = self.amplitude * np.sign(np.sin(2 * np.pi * self.frequency * t))
        elif self.signal_type == "noise":
            signal = self.amplitude * np.random.randn(self.chunk_size)
        elif self.signal_type == "mixed":
            # Mix sine at frequency and sine at 3*frequency
            signal = self.amplitude * (
                np.sin(2 * np.pi * self.frequency * t)
                + 0.3 * np.sin(2 * np.pi * 3 * self.frequency * t)
            )
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")

        # Add noise if requested
        if self.noise_level > 0:
            noise = self.noise_level * self.amplitude * np.random.randn(self.chunk_size)
            signal = signal + noise

        # Update phase for continuity
        self._phase = (
            self._phase + 2 * np.pi * self.frequency * self.chunk_size / self.sample_rate
        ) % (2 * np.pi)

        # Cast to the expected return type
        result: NDArray[np.floating[Any]] = signal.astype(np.float64)
        return result

    def start(self) -> None:
        """Start acquisition."""
        self._running = True
        self._phase = 0.0

    def stop(self) -> None:
        """Stop acquisition."""
        self._running = False


class RealtimeAnalyzer:
    """Analyzer for real-time streaming data.

    Maintains rolling statistics and runs configurable analysis on
    incoming data chunks.

    Example:
        >>> config = RealtimeConfig(sample_rate=1e6)
        >>> analyzer = RealtimeAnalyzer(config)
        >>> analyzer.accumulate(chunk)
        >>> stats = analyzer.get_statistics()
    """

    def __init__(self, config: RealtimeConfig) -> None:
        """Initialize real-time analyzer.

        Args:
            config: Realtime configuration.
        """
        config.validate()
        self.config = config

        self._window_size = config.window_size or config.buffer_size
        self._samples: deque[float] = deque(maxlen=self._window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._update_count = 0

    def accumulate(self, data: NDArray[np.floating[Any]]) -> None:
        """Accumulate statistics from data chunk.

        Args:
            data: Array of samples to process.

        Raises:
            TypeError: If data is not numeric array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be numpy array")

        if data.dtype.kind not in "fc":
            raise TypeError("data must be float or complex array")

        for sample in data.flat:
            sample_float = float(sample)

            # Remove oldest sample from stats if window full
            if len(self._samples) == self._window_size:
                old_sample = self._samples[0]
                self._sum -= old_sample
                self._sum_sq -= old_sample**2

            # Add new sample
            self._samples.append(sample_float)
            self._sum += sample_float
            self._sum_sq += sample_float**2
            self._min = min(self._min, sample_float)
            self._max = max(self._max, sample_float)
            self._update_count += 1

    def get_statistics(self) -> dict[str, float]:
        """Get current rolling statistics.

        Returns:
            Dictionary with mean, std, min, max, peak_to_peak.

        Raises:
            ValueError: If no data accumulated.
        """
        if len(self._samples) == 0:
            raise ValueError("No data accumulated yet")

        n = len(self._samples)
        mean = self._sum / n
        variance = (self._sum_sq / n) - (mean**2)
        std = np.sqrt(max(0, variance))

        return {
            "mean": mean,
            "std": std,
            "min": self._min,
            "max": self._max,
            "peak_to_peak": self._max - self._min,
            "n_samples": n,
        }

    def reset(self) -> None:
        """Reset accumulated statistics.

        Example:
            >>> analyzer.reset()
        """
        self._samples.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")


class RealtimeStream:
    """High-level API for real-time data streaming and analysis.

    Manages a data source, circular buffer, and analyzer for streaming
    waveform processing.

    Example:
        >>> config = RealtimeConfig(sample_rate=1e6)
        >>> source = CustomSource()
        >>> stream = RealtimeStream(config, source)
        >>> stream.start()
        >>> for chunk in stream.iter_chunks(chunk_size=1000):
        ...     print(chunk.data.mean())
        >>> stream.stop()
    """

    def __init__(
        self,
        config: RealtimeConfig,
        source: RealtimeSource,
        on_chunk: Callable[[WaveformTrace], None] | None = None,
    ) -> None:
        """Initialize real-time stream.

        Args:
            config: Realtime configuration.
            source: Data source for acquisition.
            on_chunk: Optional callback for each chunk acquired.
        """
        config.validate()
        self.config = config
        self.source = source
        self._on_chunk = on_chunk

        self._buffer = RealtimeBuffer(config)
        self._analyzer = RealtimeAnalyzer(config)
        self._is_running = False
        self._acquire_thread: threading.Thread | None = None
        self._chunk_count = 0

    def start(self) -> None:
        """Start acquisition thread.

        Example:
            >>> stream.start()
        """
        if self._is_running:
            return

        self._is_running = True
        self.source.start()

        self._acquire_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._acquire_thread.start()

    def stop(self) -> None:
        """Stop acquisition thread.

        Example:
            >>> stream.stop()
        """
        if not self._is_running:
            return

        self._is_running = False
        self.source.stop()

        if self._acquire_thread is not None:
            self._acquire_thread.join(timeout=5.0)

    def iter_chunks(self) -> Generator[WaveformTrace, None, None]:
        """Iterate over data chunks as they arrive.

        Yields chunks of configured size as data becomes available.

        Yields:
            WaveformTrace chunks.

        Raises:
            RuntimeError: If stream not started.

        Example:
            >>> for chunk in stream.iter_chunks():
            ...     print(f"Chunk {chunk.metadata.start_index} has {len(chunk.data)} samples")
        """
        if not self._is_running:
            raise RuntimeError("Stream not started")

        sample_index = 0

        while self._is_running:
            try:
                data = self._buffer.read(self.config.chunk_size)

                if len(data) > 0:
                    # Accumulate statistics
                    self._analyzer.accumulate(data)

                    # Create trace chunk
                    metadata = TraceMetadata(
                        sample_rate=self.config.sample_rate,
                    )

                    chunk = WaveformTrace(data=data, metadata=metadata)
                    sample_index += len(data)
                    self._chunk_count += 1

                    # Call callback if provided
                    if self._on_chunk is not None:
                        self._on_chunk(chunk)

                    yield chunk

            except TimeoutError:
                # Check if stopped during timeout
                # Note: _is_running may change asynchronously in another thread
                continue

    def get_statistics(self) -> dict[str, float]:
        """Get current statistics.

        Returns:
            Dictionary with stream statistics.
        """
        try:
            return self._analyzer.get_statistics()
        except ValueError:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "peak_to_peak": 0.0,
                "n_samples": 0,
            }

    def get_buffer_stats(self) -> dict[str, int]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats.
        """
        return self._buffer.get_stats()

    def get_chunk_count(self) -> int:
        """Get total number of chunks acquired.

        Returns:
            Number of chunks yielded so far.
        """
        return self._chunk_count

    def _acquire_loop(self) -> None:
        """Background thread that acquires data from source."""
        while self._is_running:
            try:
                data = self.source.acquire()
                if data is not None and len(data) > 0:
                    self._buffer.write(data)
            except Exception:
                if self._is_running:
                    time.sleep(0.001)


__all__ = [
    "RealtimeAnalyzer",
    "RealtimeBuffer",
    "RealtimeConfig",
    "RealtimeSource",
    "RealtimeStream",
    "SimulatedSource",
]
