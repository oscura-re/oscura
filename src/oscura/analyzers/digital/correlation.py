"""Multi-channel time correlation for synchronized analysis.

This module provides tools for correlating and aligning multiple signal channels
that may have timing offsets, different sample rates, or require trigger-based
synchronization.


Example:
    >>> from oscura.analyzers.digital.correlation import correlate_channels, align_by_trigger
    >>> result = correlate_channels(channel_a, channel_b, sample_rate=1e9)
    >>> print(f"Time offset: {result.offset_seconds:.9f} seconds")
    >>> aligned = align_by_trigger(channels, trigger_channel='clk', edge='rising')
    >>> print(f"Aligned channels: {aligned.channel_names}")

References:
    Oppenheim & Schafer: Discrete-Time Signal Processing (3rd Ed), Chapter 2
    Press et al: Numerical Recipes (3rd Ed), Section 13.2 - Correlation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal

from oscura.core.exceptions import InsufficientDataError, ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CorrelationResult:
    """Result of cross-correlation analysis.



    Attributes:
        offset_samples: Time offset in samples (positive = channel_b leads).
        offset_seconds: Time offset in seconds.
        correlation_coefficient: Peak correlation value (-1.0 to 1.0).
        confidence: Confidence score (0.0 to 1.0) based on peak sharpness.
        quality: Quality classification.
    """

    offset_samples: int
    offset_seconds: float
    correlation_coefficient: float
    confidence: float
    quality: str  # 'excellent', 'good', 'fair', 'poor'


class CorrelatedChannels:
    """Container for time-aligned multi-channel data.



    Attributes:
        channels: Dictionary mapping channel names to aligned data arrays.
        sample_rate: Common sample rate for all channels.
        offsets: Dictionary mapping channel names to their time offsets (samples).
    """

    def __init__(
        self, channels: dict[str, NDArray[np.float64]], sample_rate: float, offsets: dict[str, int]
    ):
        """Initialize correlated channels container.

        Args:
            channels: Dictionary of channel name -> aligned data array.
            sample_rate: Sample rate in Hz (same for all channels).
            offsets: Dictionary of channel name -> offset in samples.

        Raises:
            ValidationError: If channels are empty or inconsistent.
        """
        if not channels:
            raise ValidationError("At least one channel is required")

        # Validate all channels have same length
        lengths = {name: len(data) for name, data in channels.items()}
        if len(set(lengths.values())) > 1:
            raise ValidationError(f"Channel length mismatch: {lengths}")

        if sample_rate <= 0:
            raise ValidationError(f"Sample rate must be positive, got {sample_rate}")

        self.channels = channels
        self.sample_rate = float(sample_rate)
        self.offsets = offsets

    @property
    def channel_names(self) -> list[str]:
        """Get list of channel names."""
        return list(self.channels.keys())

    def get_channel(self, name: str) -> NDArray[np.float64]:
        """Get aligned data for a specific channel.

        Args:
            name: Channel name.

        Returns:
            Aligned data array.
        """
        return self.channels[name]

    def get_time_vector(self) -> NDArray[np.float64]:
        """Get time vector for aligned data.

        Returns:
            Time array in seconds, starting from 0.
        """
        first_channel = next(iter(self.channels.values()))
        n_samples = len(first_channel)
        return np.arange(n_samples) / self.sample_rate


class ChannelCorrelator:
    """Correlate multiple signal channels in time.



    This class provides methods for aligning channels using cross-correlation,
    trigger edge detection, or resampling to a common sample rate.
    """

    def __init__(self, reference_channel: str | None = None):
        """Initialize correlator.

        Args:
            reference_channel: Name of reference channel for multi-channel alignment.
                             If None, first channel will be used as reference.
        """
        self.reference_channel = reference_channel

    def correlate(
        self,
        signal1: NDArray[np.float64],
        signal2: NDArray[np.float64],
    ) -> float:
        """Compute correlation coefficient between two signals.

        Simple correlation interface for test compatibility.

        Args:
            signal1: First signal array.
            signal2: Second signal array.

        Returns:
            Correlation coefficient (-1.0 to 1.0).

        Example:
            >>> correlator = ChannelCorrelator()
            >>> corr = correlator.correlate(signal1, signal2)
        """
        signal1 = np.asarray(signal1, dtype=np.float64)
        signal2 = np.asarray(signal2, dtype=np.float64)

        if len(signal1) != len(signal2):
            # Use shorter length
            min_len = min(len(signal1), len(signal2))
            signal1 = signal1[:min_len]
            signal2 = signal2[:min_len]

        if len(signal1) < 2:
            return 0.0

        # Compute Pearson correlation coefficient
        s1_centered = signal1 - np.mean(signal1)
        s2_centered = signal2 - np.mean(signal2)

        num = np.sum(s1_centered * s2_centered)
        denom = np.sqrt(np.sum(s1_centered**2) * np.sum(s2_centered**2))

        if denom == 0:
            return 0.0

        return float(num / denom)

    def find_lag(
        self,
        signal1: NDArray[np.float64],
        signal2: NDArray[np.float64],
    ) -> int:
        """Find the time lag between two signals using cross-correlation.

        Args:
            signal1: First signal array.
            signal2: Second signal array.

        Returns:
            Lag in samples (positive = signal2 lags signal1).

        Example:
            >>> correlator = ChannelCorrelator()
            >>> lag = correlator.find_lag(signal1, signal2)
        """
        signal1 = np.asarray(signal1, dtype=np.float64)
        signal2 = np.asarray(signal2, dtype=np.float64)

        if len(signal1) < 2 or len(signal2) < 2:
            return 0

        # Center signals
        s1_centered = signal1 - np.mean(signal1)
        s2_centered = signal2 - np.mean(signal2)

        # Compute cross-correlation
        correlation = np.correlate(s1_centered, s2_centered, mode="full")

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))

        # Convert to lag (relative to signal2)
        lag = peak_idx - (len(signal2) - 1)

        return int(lag)

    def correlation_matrix(
        self,
        channels: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Compute pairwise correlation matrix for multiple channels.

        Args:
            channels: List of signal arrays.

        Returns:
            NxN correlation matrix where N is number of channels.

        Example:
            >>> correlator = ChannelCorrelator()
            >>> matrix = correlator.correlation_matrix([ch1, ch2, ch3])
        """
        n = len(channels)
        matrix = np.ones((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.correlate(channels[i], channels[j])
                matrix[i, j] = corr
                matrix[j, i] = corr

        return matrix

    def correlate_channels(
        self,
        channel_a: NDArray[np.float64],
        channel_b: NDArray[np.float64],
        sample_rate: float = 1.0,
    ) -> CorrelationResult:
        """Find time offset between two channels using cross-correlation.



        Uses normalized cross-correlation to find the time offset that maximizes
        alignment between two channels. Handles zero-mean normalization for
        robustness against DC offsets.

        Args:
            channel_a: First channel data.
            channel_b: Second channel data.
            sample_rate: Sample rate in Hz (default 1.0 for sample-based results).

        Returns:
            CorrelationResult with offset and quality metrics.

        Raises:
            InsufficientDataError: If channels are too short.
            ValidationError: If sample rate is invalid.
        """
        if len(channel_a) < 2 or len(channel_b) < 2:
            raise InsufficientDataError("Channels must have at least 2 samples")

        if sample_rate <= 0:
            raise ValidationError(f"Sample rate must be positive, got {sample_rate}")

        # Convert to zero-mean for better correlation
        a_mean = np.mean(channel_a)
        b_mean = np.mean(channel_b)
        a_centered = channel_a - a_mean
        b_centered = channel_b - b_mean

        # Compute cross-correlation using scipy (more efficient than numpy)
        correlation = signal.correlate(a_centered, b_centered, mode="full", method="auto")

        # Normalize by signal energies for correlation coefficient
        a_energy = np.sum(a_centered**2)
        b_energy = np.sum(b_centered**2)

        if a_energy == 0 or b_energy == 0:
            # One or both signals are constant
            return CorrelationResult(
                offset_samples=0,
                offset_seconds=0.0,
                correlation_coefficient=0.0,
                confidence=0.0,
                quality="poor",
            )

        normalization = np.sqrt(a_energy * b_energy)
        correlation_normalized = correlation / normalization

        # Find peak correlation
        peak_idx = np.argmax(np.abs(correlation_normalized))
        peak_value = correlation_normalized[peak_idx]

        # Convert peak index to offset (positive = channel_b leads)
        offset_samples = peak_idx - (len(channel_b) - 1)
        offset_seconds = offset_samples / sample_rate

        # Estimate confidence from peak sharpness
        # High confidence = sharp peak, low confidence = broad/weak peak
        confidence = self._estimate_correlation_confidence(correlation_normalized, int(peak_idx))

        # Classify quality
        quality = self._classify_correlation_quality(abs(peak_value), confidence)

        return CorrelationResult(
            offset_samples=int(offset_samples),
            offset_seconds=float(offset_seconds),
            correlation_coefficient=float(peak_value),
            confidence=float(confidence),
            quality=quality,
        )

    def align_by_trigger(
        self,
        channels: dict[str, NDArray[np.float64]],
        trigger_channel: str,
        edge: Literal["rising", "falling"] = "rising",
        threshold: float = 0.5,
    ) -> CorrelatedChannels:
        """Align channels using trigger edge from one channel.



        Aligns all channels by detecting the first trigger edge in the specified
        channel and trimming all channels to start from that point.

        Args:
            channels: Dictionary of channel name -> data array.
            trigger_channel: Name of channel to use for trigger detection.
            edge: Edge type to detect ('rising' or 'falling').
            threshold: Trigger threshold (normalized 0-1 if float, or absolute value).

        Returns:
            CorrelatedChannels with aligned data.

        Raises:
            InsufficientDataError: If trigger channel is too short.
            ValidationError: If trigger channel not found or no edge detected.
        """
        if trigger_channel not in channels:
            raise ValidationError(f"Trigger channel '{trigger_channel}' not found")

        trigger_data = channels[trigger_channel]

        if len(trigger_data) < 2:
            raise InsufficientDataError("Trigger channel too short")

        # Normalize threshold if needed
        if 0.0 <= threshold <= 1.0:
            data_min = np.min(trigger_data)
            data_max = np.max(trigger_data)
            threshold_abs = float(data_min + threshold * (data_max - data_min))
        else:
            threshold_abs = float(threshold)

        # Detect first edge
        trigger_idx = self._find_first_edge(trigger_data, edge, threshold_abs)

        if trigger_idx is None:
            raise ValidationError(f"No {edge} edge found in trigger channel")

        # Align all channels by trimming to trigger point
        aligned_channels = {}
        offsets = {}

        for name, data in channels.items():
            if trigger_idx < len(data):
                aligned_channels[name] = data[trigger_idx:]
                offsets[name] = trigger_idx
            else:
                # Trigger point is beyond this channel's data
                aligned_channels[name] = np.array([])
                offsets[name] = len(data)

        # Assume all channels have same sample rate (no rate given)
        # Use default of 1.0 Hz for sample-based indexing
        return CorrelatedChannels(aligned_channels, sample_rate=1.0, offsets=offsets)

    def resample_to_common_rate(
        self,
        channels: dict[str, tuple[NDArray[np.float64], float]],
        target_rate: float | None = None,
    ) -> CorrelatedChannels:
        """Resample all channels to common sample rate.



        Resamples channels with different sample rates to a common rate using
        polyphase resampling for high quality. Uses the highest sample rate
        as target if not specified.

        Args:
            channels: Dictionary of channel name -> (data, sample_rate) tuples.
            target_rate: Target sample rate in Hz. If None, uses highest rate.

        Returns:
            CorrelatedChannels with resampled data at common rate.

        Raises:
            ValidationError: If channels are empty or rates are invalid.
        """
        if not channels:
            raise ValidationError("At least one channel is required")

        # Determine target rate
        if target_rate is None:
            rates = [rate for _, rate in channels.values()]
            target_rate = max(rates)

        if target_rate <= 0:
            raise ValidationError(f"Target rate must be positive, got {target_rate}")

        resampled_channels = {}
        offsets = {}

        for name, (data, original_rate) in channels.items():
            if original_rate <= 0:
                raise ValidationError(f"Invalid sample rate for '{name}': {original_rate}")

            if len(data) < 2:
                # Skip empty/trivial channels
                resampled_channels[name] = data
                offsets[name] = 0
                continue

            # Calculate resampling ratio
            ratio = target_rate / original_rate

            if abs(ratio - 1.0) < 1e-6:
                # Already at target rate
                resampled_channels[name] = data
            else:
                # Resample using polyphase method
                num_samples = int(np.round(len(data) * ratio))
                resampled_channels[name] = signal.resample(data, num_samples)

            offsets[name] = 0

        return CorrelatedChannels(resampled_channels, sample_rate=target_rate, offsets=offsets)

    def auto_align(
        self,
        channels: dict[str, NDArray[np.float64]],
        sample_rate: float,
        method: Literal["correlation", "trigger", "edge"] = "correlation",
    ) -> CorrelatedChannels:
        """Auto-align channels using best-guess method.



        Automatically aligns multiple channels using the specified method.
        For correlation method, aligns all channels to the reference channel.

        Args:
            channels: Dictionary of channel name -> data array.
            sample_rate: Sample rate in Hz (same for all channels).
            method: Alignment method to use.

        Returns:
            CorrelatedChannels with aligned data.

        Raises:
            ValidationError: If method is invalid or alignment fails.
        """
        if not channels:
            raise ValidationError("At least one channel is required")

        if len(channels) < 2:
            # Single channel, no alignment needed
            return CorrelatedChannels(
                channels=channels, sample_rate=sample_rate, offsets=dict.fromkeys(channels, 0)
            )

        # Determine reference channel
        if self.reference_channel and self.reference_channel in channels:
            ref_name = self.reference_channel
        else:
            ref_name = next(iter(channels))

        ref_data = channels[ref_name]

        if method == "correlation":
            # Correlate all channels to reference
            aligned_channels = {ref_name: ref_data}
            offsets = {ref_name: 0}

            for name, data in channels.items():
                if name == ref_name:
                    continue

                # Cross-correlate with reference
                result = self.correlate_channels(ref_data, data, sample_rate)

                # Apply offset to align
                offset = -result.offset_samples  # Negative because we want to shift data

                if offset > 0:
                    # Trim start of data
                    aligned_channels[name] = data[offset:]
                elif offset < 0:
                    # Pad start of data
                    pad = np.zeros(-offset)
                    aligned_channels[name] = np.concatenate([pad, data])
                else:
                    aligned_channels[name] = data

                offsets[name] = offset

            # Trim all to same length
            min_len = min(len(d) for d in aligned_channels.values())
            aligned_channels = {name: data[:min_len] for name, data in aligned_channels.items()}

            return CorrelatedChannels(aligned_channels, sample_rate, offsets)

        elif method in ("trigger", "edge"):
            # Use first channel as trigger
            return self.align_by_trigger(channels, ref_name, edge="rising")

        else:
            raise ValidationError(f"Unknown alignment method: {method}")

    def _estimate_correlation_confidence(
        self, correlation: NDArray[np.float64], peak_idx: int
    ) -> float:
        """Estimate confidence from correlation peak sharpness.

        Args:
            correlation: Normalized correlation array.
            peak_idx: Index of peak correlation.

        Returns:
            Confidence score 0.0 to 1.0.
        """
        peak_value = abs(correlation[peak_idx])

        # Calculate peak-to-sidelobe ratio
        # Higher ratio = sharper peak = higher confidence
        window_size = min(20, len(correlation) // 10)
        start = max(0, peak_idx - window_size)
        end = min(len(correlation), peak_idx + window_size + 1)

        # Exclude peak itself
        sidelobe_indices = np.concatenate(
            [np.arange(start, peak_idx), np.arange(peak_idx + 1, end)]
        )

        if len(sidelobe_indices) > 0:
            max_sidelobe = np.max(np.abs(correlation[sidelobe_indices]))
            if max_sidelobe > 0:
                ratio = peak_value / max_sidelobe
                # Map ratio to confidence (empirically tuned)
                confidence = min(1.0, ratio / 5.0)
            else:
                confidence = 1.0
        else:
            confidence = peak_value

        return float(confidence)

    def _classify_correlation_quality(self, correlation: float, confidence: float) -> str:
        """Classify correlation quality.

        Args:
            correlation: Correlation coefficient (0.0 to 1.0).
            confidence: Confidence score (0.0 to 1.0).

        Returns:
            str: Quality rating - 'excellent', 'good', 'fair', or 'poor'.
        """
        score = (correlation + confidence) / 2.0

        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _find_first_edge(
        self, data: NDArray[np.float64], edge: str, threshold: float
    ) -> int | None:
        """Find first edge in data.

        Args:
            data: Signal data.
            edge: Edge type ('rising' or 'falling').
            threshold: Threshold value.

        Returns:
            Index of first edge, or None if not found.
        """
        if edge == "rising":
            # Find first point where signal crosses above threshold
            crossings = np.where((data[:-1] < threshold) & (data[1:] >= threshold))[0]
        else:  # falling
            crossings = np.where((data[:-1] > threshold) & (data[1:] <= threshold))[0]

        if len(crossings) > 0:
            return int(crossings[0] + 1)  # Return index after crossing
        else:
            return None


# Convenience functions


def correlate_channels(
    channel_a: NDArray[np.float64], channel_b: NDArray[np.float64], sample_rate: float = 1.0
) -> CorrelationResult:
    """Find time offset between two channels.



    Convenience function for correlating two channels without creating
    a ChannelCorrelator instance.

    Args:
        channel_a: First channel data.
        channel_b: Second channel data.
        sample_rate: Sample rate in Hz (default 1.0 for sample-based results).

    Returns:
        CorrelationResult with offset and quality metrics.

    Example:
        >>> result = correlate_channels(ch1, ch2, sample_rate=1e9)
        >>> print(f"Offset: {result.offset_seconds*1e9:.2f} ns")
    """
    correlator = ChannelCorrelator()
    return correlator.correlate_channels(channel_a, channel_b, sample_rate)


def align_by_trigger(
    channels: dict[str, NDArray[np.float64]],
    trigger_channel: str,
    edge: Literal["rising", "falling"] = "rising",
    threshold: float = 0.5,
) -> CorrelatedChannels:
    """Align channels using trigger edge.



    Convenience function for trigger-based alignment without creating
    a ChannelCorrelator instance.

    Args:
        channels: Dictionary of channel name -> data array.
        trigger_channel: Name of channel to use for trigger detection.
        edge: Edge type to detect ('rising' or 'falling').
        threshold: Trigger threshold (0-1 normalized or absolute).

    Returns:
        CorrelatedChannels with aligned data.

    Example:
        >>> aligned = align_by_trigger(
        ...     {'clk': clk_data, 'data': data_signal},
        ...     trigger_channel='clk',
        ...     edge='rising'
        ... )
    """
    correlator = ChannelCorrelator()
    return correlator.align_by_trigger(channels, trigger_channel, edge, threshold)


def resample_to_common_rate(
    channels: dict[str, tuple[NDArray[np.float64], float]], target_rate: float | None = None
) -> CorrelatedChannels:
    """Resample all channels to common rate.



    Convenience function for resampling channels without creating
    a ChannelCorrelator instance.

    Args:
        channels: Dictionary of channel name -> (data, sample_rate) tuples.
        target_rate: Target sample rate in Hz. If None, uses highest rate.

    Returns:
        CorrelatedChannels with resampled data at common rate.

    Example:
        >>> resampled = resample_to_common_rate({
        ...     'ch1': (data1, 1e9),
        ...     'ch2': (data2, 2e9)
        ... })
        >>> print(f"Common rate: {resampled.sample_rate} Hz")
    """
    correlator = ChannelCorrelator()
    return correlator.resample_to_common_rate(channels, target_rate)


__all__ = [
    "ChannelCorrelator",
    "CorrelatedChannels",
    "CorrelationResult",
    "align_by_trigger",
    "correlate_channels",
    "resample_to_common_rate",
]
