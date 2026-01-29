"""Progressive streaming analysis with incremental confidence.

Enables real-time analysis with confidence that increases as more data
is processed, allowing early stopping when confidence is sufficient.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.validation.quality.scoring import assess_data_quality

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class StreamingProgress:
    """Progress update from streaming analysis.

    Attributes:
        samples_processed: Number of samples processed so far
        total_samples: Total samples expected (None if unknown)
        confidence: Current confidence level (0-1)
        preliminary_results: Current analysis results
        is_complete: Whether analysis is complete
        can_stop_early: Whether confidence is sufficient for early stopping
        message: Human-readable status message
    """

    samples_processed: int
    total_samples: int | None
    confidence: float
    preliminary_results: dict[str, Any]
    is_complete: bool = False
    can_stop_early: bool = False
    message: str = ""

    @property
    def progress_percent(self) -> float | None:
        """Get progress as percentage if total is known.

        Returns:
            Progress percentage (0-100) or None if total unknown
        """
        if self.total_samples:
            return 100.0 * self.samples_processed / self.total_samples
        return None


@dataclass
class StreamingConfig:
    """Configuration for streaming analysis.

    Attributes:
        chunk_size: Samples per processing chunk
        overlap: Overlap ratio between chunks (0-1)
        min_samples_for_result: Minimum samples before generating results
        early_stop_confidence: Confidence threshold for early stopping
        max_buffer_size: Maximum buffer size to maintain
        update_interval_samples: Samples between progress updates
    """

    chunk_size: int = 1024
    overlap: float = 0.25
    min_samples_for_result: int = 100
    early_stop_confidence: float = 0.9
    max_buffer_size: int = 100000
    update_interval_samples: int = 512


class ProgressiveAnalyzer:
    """Streaming analyzer with progressive confidence updates.

    Processes data in chunks and provides incremental updates with
    increasing confidence as more data is analyzed. Enables early
    stopping when confidence threshold is met.

        API-004: Real-time streaming analysis
        QUAL-001: Quality scoring with progressive confidence

    Example:
        >>> analyzer = ProgressiveAnalyzer(sample_rate=1000.0)
        >>> analyzer.subscribe(lambda p: print(f"Confidence: {p.confidence:.0%}"))
        >>> progress = analyzer.process_chunk(data_chunk)
        >>> if progress.can_stop_early:
        ...     final = analyzer.finalize()
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        config: StreamingConfig | None = None,
    ):
        """Initialize progressive analyzer.

        Args:
            sample_rate: Sample rate in Hz
            config: Streaming configuration
        """
        self.sample_rate = sample_rate
        self.config = config or StreamingConfig()

        # Internal state
        self._buffer: deque[float] = deque(maxlen=self.config.max_buffer_size)
        self._samples_processed = 0
        self._current_results: dict[str, Any] = {}
        self._confidence = 0.0
        self._callbacks: list[Callable[[StreamingProgress], None]] = []

        # Running statistics (incremental calculation)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min_val = float("inf")
        self._max_val = float("-inf")

        # Frequency estimation state
        self._zero_crossings: list[int] = []
        self._last_sign = 0

    def reset(self) -> None:
        """Reset analyzer state to initial conditions."""
        self._buffer.clear()
        self._samples_processed = 0
        self._current_results = {}
        self._confidence = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min_val = float("inf")
        self._max_val = float("-inf")
        self._zero_crossings = []
        self._last_sign = 0

    def subscribe(self, callback: Callable[[StreamingProgress], None]) -> None:
        """Subscribe to progress updates.

        Args:
            callback: Function called with StreamingProgress updates
        """
        self._callbacks.append(callback)

    def process_chunk(self, chunk: NDArray[np.float64]) -> StreamingProgress:
        """Process a chunk of data.

        Args:
            chunk: Data chunk to process

        Returns:
            StreamingProgress with current state
        """
        chunk_arr = np.asarray(chunk).flatten()

        # Handle empty chunks
        if len(chunk_arr) == 0:
            return StreamingProgress(
                samples_processed=self._samples_processed,
                total_samples=None,
                confidence=self._confidence,
                preliminary_results=self._current_results.copy(),
                is_complete=False,
                can_stop_early=self._confidence >= self.config.early_stop_confidence,
                message=self._get_status_message(),
            )

        # Add to buffer
        for val in chunk_arr:
            self._buffer.append(val)

        # Update running statistics
        self._update_statistics(chunk_arr)

        # Update frequency estimation
        self._update_frequency_estimation(chunk_arr)

        self._samples_processed += len(chunk_arr)

        # Calculate current confidence
        self._update_confidence()

        # Generate preliminary results
        self._update_results()

        # Create progress update
        progress = StreamingProgress(
            samples_processed=self._samples_processed,
            total_samples=None,
            confidence=self._confidence,
            preliminary_results=self._current_results.copy(),
            is_complete=False,
            can_stop_early=self._confidence >= self.config.early_stop_confidence,
            message=self._get_status_message(),
        )

        # Notify subscribers
        if self._samples_processed % self.config.update_interval_samples < len(chunk_arr):
            self._notify(progress)

        return progress

    def finalize(self) -> StreamingProgress:
        """Finalize analysis and return final results.

        Returns:
            Final StreamingProgress with complete results
        """
        # Do final calculations with all buffered data
        if self._buffer:
            buffer_array = np.array(list(self._buffer))

            # Enhanced frequency detection with full buffer
            self._current_results["frequency_final"] = self._detect_frequency_fft(buffer_array)

            # Final quality assessment
            data_quality = assess_data_quality(buffer_array, self.sample_rate)
            self._current_results["data_quality"] = {
                "snr_db": data_quality.snr_db,
                "sample_count": data_quality.sample_count,
                "completeness": data_quality.completeness,
            }

        self._confidence = min(1.0, self._confidence * 1.1)  # Boost for finalization

        progress = StreamingProgress(
            samples_processed=self._samples_processed,
            total_samples=self._samples_processed,
            confidence=self._confidence,
            preliminary_results=self._current_results.copy(),
            is_complete=True,
            can_stop_early=False,
            message="Analysis complete",
        )

        self._notify(progress)
        return progress

    def _update_statistics(self, chunk: NDArray[np.float64]) -> None:
        """Update running statistics with new chunk."""
        self._sum += float(np.sum(chunk))
        self._sum_sq += float(np.sum(chunk**2))
        self._min_val = min(self._min_val, float(np.min(chunk)))
        self._max_val = max(self._max_val, float(np.max(chunk)))

    def _update_frequency_estimation(self, chunk: NDArray[np.float64]) -> None:
        """Update frequency estimation from zero crossings."""
        # Track zero crossings
        mean_val = self._sum / max(1, self._samples_processed)

        for i, val in enumerate(chunk):
            current_sign = 1 if val > mean_val else -1
            if self._last_sign != 0 and current_sign != self._last_sign:
                self._zero_crossings.append(self._samples_processed - len(chunk) + i)
            self._last_sign = current_sign

    def _update_confidence(self) -> None:
        """Update confidence based on data processed."""
        # Base confidence from sample count (logarithmic)
        sample_factor = min(1.0, np.log10(max(1, self._samples_processed)) / 4.0)

        # Frequency confidence from consistent zero crossings
        freq_factor = 0.5
        if len(self._zero_crossings) > 10:
            intervals = np.diff(self._zero_crossings)
            if len(intervals) > 5:
                cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                freq_factor = min(1.0, 1.0 - cv)

        # Combine factors - ensure monotonic increase by taking max with previous
        new_confidence = 0.4 * sample_factor + 0.4 * freq_factor + 0.2
        self._confidence = max(self._confidence, new_confidence)

    def _update_results(self) -> None:
        """Update preliminary results."""
        n = self._samples_processed
        if n < self.config.min_samples_for_result:
            return

        # Running mean and std
        mean = self._sum / n
        variance = (self._sum_sq / n) - (mean**2)
        std = np.sqrt(max(0, variance))

        self._current_results.update(
            {
                "mean": mean,
                "std": std,
                "min": self._min_val,
                "max": self._max_val,
                "amplitude": self._max_val - self._min_val,
                "sample_count": n,
            }
        )

        # Frequency from zero crossings
        if len(self._zero_crossings) > 4:
            intervals = np.diff(self._zero_crossings)
            avg_half_period = np.mean(intervals) / self.sample_rate
            if avg_half_period > 0:
                self._current_results["frequency_estimate"] = 0.5 / avg_half_period

    def _detect_frequency_fft(self, data: NDArray[np.float64]) -> float | None:
        """Detect frequency using FFT on full buffer.

        Args:
            data: Signal data

        Returns:
            Dominant frequency in Hz or None if detection failed
        """
        try:
            data_ac = data - np.mean(data)
            fft_result = np.fft.rfft(data_ac)
            freqs = np.fft.rfftfreq(len(data_ac), d=1.0 / self.sample_rate)
            magnitude = np.abs(fft_result[1:])

            if len(magnitude) > 0:
                peak_idx = np.argmax(magnitude)
                return float(freqs[1:][peak_idx])
        except Exception:
            pass
        return None

    def _get_status_message(self) -> str:
        """Generate status message.

        Returns:
            Human-readable status string
        """
        if self._samples_processed < self.config.min_samples_for_result:
            return f"Collecting data... ({self._samples_processed} samples)"

        if self._confidence < 0.5:
            return f"Low confidence ({self._confidence:.0%}), collecting more data..."
        elif self._confidence < 0.8:
            return f"Medium confidence ({self._confidence:.0%}), analysis in progress"
        else:
            return f"High confidence ({self._confidence:.0%}), results reliable"

    def _notify(self, progress: StreamingProgress) -> None:
        """Notify all subscribers.

        Args:
            progress: Progress update to send
        """
        for callback in self._callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.debug(f"Callback error: {e}")


def create_progressive_analyzer(
    sample_rate: float = 1.0,
    chunk_size: int = 1024,
    early_stop_confidence: float = 0.9,
) -> ProgressiveAnalyzer:
    """Create a progressive analyzer with common settings.

    Args:
        sample_rate: Sample rate in Hz
        chunk_size: Samples per chunk
        early_stop_confidence: Confidence threshold for early stopping

    Returns:
        Configured ProgressiveAnalyzer

    Example:
        >>> analyzer = create_progressive_analyzer(
        ...     sample_rate=1000.0,
        ...     chunk_size=512,
        ...     early_stop_confidence=0.85
        ... )
    """
    config = StreamingConfig(
        chunk_size=chunk_size,
        early_stop_confidence=early_stop_confidence,
    )
    return ProgressiveAnalyzer(sample_rate, config)


__all__ = [
    "ProgressiveAnalyzer",
    "StreamingConfig",
    "StreamingProgress",
    "create_progressive_analyzer",
]
