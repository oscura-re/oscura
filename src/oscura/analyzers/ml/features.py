"""Feature extraction utilities for ML-based signal classification.

This module provides comprehensive feature extraction for signal classification:
- Statistical features (mean, variance, skewness, kurtosis)
- Spectral features (FFT-based frequency domain analysis)
- Temporal features (autocorrelation, zero-crossings, peaks)
- Entropy features (Shannon entropy, permutation entropy)
- Shape features (rise/fall time, duty cycle for digital signals)

Features are extracted in a standardized format suitable for ML algorithms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal as sp_signal
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract comprehensive features from signals for ML classification.

    This class provides a unified interface for extracting multiple feature types
    from time-domain signals. Features are returned as dictionaries with consistent
    naming for use in machine learning pipelines.

    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_all(signal, sample_rate=1e6)
        >>> print(f"Extracted {len(features)} features")
        >>> print(f"Dominant frequency: {features['dominant_frequency']:.1f} Hz")
    """

    def extract_all(self, data: NDArray[np.floating[Any]], sample_rate: float) -> dict[str, float]:
        """Extract all feature types from signal.

        Args:
            data: Input signal as 1D numpy array.
            sample_rate: Sampling rate in Hz.

        Returns:
            Dictionary of feature name to value mappings. Contains 40+ features
            spanning statistical, spectral, temporal, entropy, and shape categories.

        Example:
            >>> signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 10000))
            >>> features = extractor.extract_all(signal, 10000)
            >>> features['dominant_frequency']  # Should be ~1000 Hz
            1000.0
        """
        features: dict[str, float] = {}

        # Extract each feature category
        features.update(self.extract_statistical(data))
        features.update(self.extract_spectral(data, sample_rate))
        features.update(self.extract_temporal(data))
        features.update(self.extract_entropy(data))
        features.update(self.extract_shape(data, sample_rate))

        return features

    def extract_statistical(self, data: NDArray[np.floating[Any]]) -> dict[str, float]:
        """Extract statistical features from signal.

        Computes basic statistical moments and distribution properties:
        - Central tendency: mean, median
        - Dispersion: std, variance, range, IQR
        - Shape: skewness, kurtosis

        Args:
            data: Input signal as 1D numpy array.

        Returns:
            Dictionary with 9 statistical features.

        Example:
            >>> gaussian = np.random.normal(0, 1, 10000)
            >>> features = extractor.extract_statistical(gaussian)
            >>> abs(features['mean']) < 0.1  # Near zero
            True
            >>> 0.9 < features['std'] < 1.1  # Near 1
            True
        """
        import warnings

        # For constant signals, skew/kurtosis cause precision warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))

        # Handle NaN from constant signals
        if np.isnan(skewness):
            skewness = 0.0
        if np.isnan(kurtosis):
            kurtosis = 0.0

        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "variance": float(np.var(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.ptp(data)),
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def extract_spectral(
        self, data: NDArray[np.floating[Any]], sample_rate: float
    ) -> dict[str, float]:
        """Extract spectral features via FFT analysis.

        Analyzes frequency domain properties:
        - Dominant frequency (peak in spectrum)
        - Spectral centroid (center of mass of spectrum)
        - Bandwidth (frequency range containing most energy)
        - Spectral energy (total power)
        - Spectral flatness (measure of tone vs noise)
        - Spectral rolloff (95th percentile frequency)

        Args:
            data: Input signal as 1D numpy array.
            sample_rate: Sampling rate in Hz.

        Returns:
            Dictionary with 8 spectral features.

        Example:
            >>> # 1 kHz sine wave
            >>> t = np.linspace(0, 1, 10000)
            >>> signal = np.sin(2 * np.pi * 1000 * t)
            >>> features = extractor.extract_spectral(signal, 10000)
            >>> 950 < features['dominant_frequency'] < 1050
            True
        """
        # Compute FFT (use real FFT for efficiency)
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
        magnitude = np.abs(fft)
        power = magnitude**2

        # Avoid division by zero
        magnitude_safe = magnitude + 1e-10
        power_safe = power + 1e-10

        # Dominant frequency (peak in spectrum)
        if len(magnitude) > 1:
            dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
            dominant_freq = float(freqs[dominant_idx])
        else:
            dominant_freq = 0.0

        # Spectral centroid (center of mass)
        spectral_centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude_safe))

        # Bandwidth (frequencies containing >10% of max power)
        threshold = 0.1 * np.max(power)
        bandwidth_freqs = freqs[power > threshold]
        bandwidth = (
            float(bandwidth_freqs[-1] - bandwidth_freqs[0]) if len(bandwidth_freqs) > 1 else 0.0
        )

        # Spectral energy
        spectral_energy = float(np.sum(power))

        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = float(np.exp(np.mean(np.log(magnitude_safe))))
        arithmetic_mean = float(np.mean(magnitude))
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0

        # Spectral rolloff (95th percentile frequency)
        cumsum = np.cumsum(power)
        total_energy = cumsum[-1]
        rolloff_threshold = 0.95 * total_energy
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        spectral_rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0

        # Number of spectral peaks
        peaks, _ = sp_signal.find_peaks(magnitude, height=0.1 * np.max(magnitude))
        num_spectral_peaks = float(len(peaks))

        # Spectral spread (standard deviation around centroid)
        spectral_spread = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / np.sum(power_safe))
        )

        return {
            "dominant_frequency": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "bandwidth": bandwidth,
            "spectral_energy": spectral_energy,
            "spectral_flatness": spectral_flatness,
            "spectral_rolloff": spectral_rolloff,
            "num_spectral_peaks": num_spectral_peaks,
            "spectral_spread": spectral_spread,
        }

    def extract_temporal(self, data: NDArray[np.floating[Any]]) -> dict[str, float]:
        """Extract temporal domain features.

        Analyzes time-domain signal properties:
        - Zero-crossings (transitions through mean)
        - Autocorrelation (self-similarity measure)
        - Peak count (number of local maxima)
        - Peak prominence (average peak height)
        - Energy (sum of squared values)
        - RMS (root mean square)

        Args:
            data: Input signal as 1D numpy array.

        Returns:
            Dictionary with 8 temporal features.

        Example:
            >>> # Square wave has many zero crossings
            >>> square = np.sign(np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)))
            >>> features = extractor.extract_temporal(square)
            >>> features['zero_crossing_rate'] > 0.01
            True
        """
        # Zero-crossings (normalized by length)
        mean_centered = data - np.mean(data)
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        zero_crossing_rate = float(zero_crossings) / len(data)

        # Autocorrelation at lag 1
        if len(data) > 1:
            autocorr = float(np.corrcoef(data[:-1], data[1:])[0, 1])
            # Handle NaN from constant signals
            autocorr = 0.0 if np.isnan(autocorr) else autocorr
        else:
            autocorr = 0.0

        # Peak detection
        peaks, properties = sp_signal.find_peaks(data, prominence=0.1 * np.ptp(data))
        peak_count = float(len(peaks))
        peak_prominence = float(np.mean(properties["prominences"])) if len(peaks) > 0 else 0.0

        # Energy and RMS
        energy = float(np.sum(data**2))
        rms = float(np.sqrt(np.mean(data**2)))

        # Signal to noise ratio estimate (robust)
        # Use median absolute deviation for noise estimate
        mad = float(np.median(np.abs(data - np.median(data))))
        signal_power = float(np.mean(data**2))
        noise_power = (1.4826 * mad) ** 2  # Convert MAD to std estimate
        snr_estimate = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0

        # Crest factor (peak to RMS ratio)
        crest_factor = float(np.max(np.abs(data)) / rms) if rms > 0 else 0.0

        return {
            "zero_crossing_rate": zero_crossing_rate,
            "autocorrelation": autocorr,
            "peak_count": peak_count,
            "peak_prominence": peak_prominence,
            "energy": energy,
            "rms": rms,
            "snr_estimate": snr_estimate,
            "crest_factor": crest_factor,
        }

    def extract_entropy(self, data: NDArray[np.floating[Any]]) -> dict[str, float]:
        """Extract entropy-based features.

        Computes information-theoretic measures:
        - Shannon entropy (information content)
        - Approximate entropy (regularity measure)
        - Sample entropy (complexity measure, similar to ApEn)

        Args:
            data: Input signal as 1D numpy array.

        Returns:
            Dictionary with 3 entropy features.

        Example:
            >>> # Random signal has high entropy
            >>> random = np.random.randn(1000)
            >>> features = extractor.extract_entropy(random)
            >>> features['shannon_entropy'] > 5.0
            True
        """
        # Shannon entropy (discretize signal into bins)
        # Normalize to 0-255 range for byte-like entropy
        data_normalized = ((data - np.min(data)) / (np.ptp(data) + 1e-10) * 255).astype(np.uint8)
        _, counts = np.unique(data_normalized, return_counts=True)
        probabilities = counts / len(data_normalized)
        shannon_entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))

        # Approximate entropy (ApEn)
        # Measures regularity - low for regular signals, high for complex
        approx_entropy = self._approximate_entropy(data, m=2, r=0.2 * np.std(data))

        # Sample entropy (SampEn) - improved version of ApEn
        sample_entropy = self._sample_entropy(data, m=2, r=0.2 * np.std(data))

        return {
            "shannon_entropy": shannon_entropy,
            "approximate_entropy": approx_entropy,
            "sample_entropy": sample_entropy,
        }

    def _approximate_entropy(self, data: NDArray[np.floating[Any]], m: int, r: float) -> float:
        """Calculate approximate entropy (ApEn).

        Args:
            data: Input signal.
            m: Embedding dimension.
            r: Tolerance (fraction of std).

        Returns:
            Approximate entropy value.
        """
        n = len(data)

        # For very long signals, downsample to avoid O(n^2) cost
        # Use aggressive downsampling to keep performance reasonable
        # Approximate entropy is statistical and works well on smaller samples
        if n > 200:
            downsample_factor = max(n // 100, 1)  # Target ~100 samples max
            data = data[::downsample_factor]
            n = len(data)

        def _phi(m: int) -> float:
            if n - m + 1 <= 0:
                return 0.0

            patterns = np.array([data[i : i + m] for i in range(n - m + 1)])
            counts = np.zeros(len(patterns))

            for i, pattern in enumerate(patterns):
                # Count patterns within tolerance r
                distances = np.max(np.abs(patterns - pattern), axis=1)
                counts[i] = np.sum(distances <= r)

            # Avoid log(0)
            counts = np.maximum(counts, 1)
            phi_value = np.sum(np.log(counts / (n - m + 1))) / (n - m + 1)
            return float(phi_value)

        try:
            return float(_phi(m) - _phi(m + 1))
        except (ValueError, RuntimeWarning):
            return 0.0

    def _sample_entropy(self, data: NDArray[np.floating[Any]], m: int, r: float) -> float:
        """Calculate sample entropy (SampEn).

        Args:
            data: Input signal.
            m: Embedding dimension.
            r: Tolerance (fraction of std).

        Returns:
            Sample entropy value.
        """
        n = len(data)

        # For very long signals, downsample to avoid O(n^2) cost
        # Use aggressive downsampling to keep performance reasonable
        # Sample entropy is statistical and works well on smaller samples
        if n > 200:
            downsample_factor = max(n // 100, 1)  # Target ~100 samples max
            data = data[::downsample_factor]
            n = len(data)

        def _count_matches(m: int) -> tuple[int, int]:
            if n - m <= 0:
                return 0, 0

            patterns = np.array([data[i : i + m] for i in range(n - m)])
            count_a = 0
            count_b = 0

            for i in range(len(patterns)):
                # Don't count self-matches
                for j in range(i + 1, len(patterns)):
                    dist = np.max(np.abs(patterns[i] - patterns[j]))
                    if dist <= r:
                        count_b += 1
                        if m > 1:
                            # Check if extended pattern also matches
                            if i + m < n and j + m < n:
                                if abs(data[i + m] - data[j + m]) <= r:
                                    count_a += 1

            return count_a, count_b

        try:
            count_a, count_b = _count_matches(m)
            if count_a == 0 or count_b == 0:
                return 0.0
            return float(-np.log(count_a / count_b))
        except (ValueError, RuntimeWarning):
            return 0.0

    def _normalize_signal(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Normalize signal to 0-1 range.

        Args:
            data: Input signal

        Returns:
            Normalized signal
        """
        normalized: NDArray[np.floating[Any]] = (data - np.min(data)) / (np.ptp(data) + 1e-10)
        return normalized

    def _find_edges(
        self, data_normalized: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Find rising and falling edges.

        Args:
            data_normalized: Normalized signal data

        Returns:
            Tuple of (rising edges, falling edges)
        """
        rising_edges = np.where((data_normalized[:-1] < 0.5) & (data_normalized[1:] >= 0.5))[0]
        falling_edges = np.where((data_normalized[:-1] >= 0.5) & (data_normalized[1:] < 0.5))[0]
        return rising_edges, falling_edges

    def _calculate_rise_times(
        self,
        rising_edges: NDArray[np.intp],
        data_normalized: NDArray[np.floating[Any]],
        threshold_low: float,
        threshold_high: float,
    ) -> list[int]:
        """Calculate rise times for all rising edges.

        Args:
            rising_edges: Array of rising edge indices
            data_normalized: Normalized signal data
            threshold_low: Low threshold (10%)
            threshold_high: High threshold (90%)

        Returns:
            List of rise times in samples
        """
        rise_times = []
        for edge in rising_edges:
            # Look backward for 10% crossing
            start_idx = edge
            for i in range(max(0, edge - 100), edge):
                if data_normalized[i] < threshold_low:
                    start_idx = i
                    break
            # Look forward for 90% crossing
            end_idx = edge
            for i in range(edge, min(len(data_normalized), edge + 100)):
                if data_normalized[i] > threshold_high:
                    end_idx = i
                    break
            if end_idx > start_idx:
                rise_times.append(end_idx - start_idx)
        return rise_times

    def _calculate_fall_times(
        self,
        falling_edges: NDArray[np.intp],
        data_normalized: NDArray[np.floating[Any]],
        threshold_low: float,
        threshold_high: float,
    ) -> list[int]:
        """Calculate fall times for all falling edges.

        Args:
            falling_edges: Array of falling edge indices
            data_normalized: Normalized signal data
            threshold_low: Low threshold (10%)
            threshold_high: High threshold (90%)

        Returns:
            List of fall times in samples
        """
        fall_times = []
        for edge in falling_edges:
            # Look backward for 90% crossing
            start_idx = edge
            for i in range(max(0, edge - 100), edge):
                if data_normalized[i] > threshold_high:
                    start_idx = i
                    break
            # Look forward for 10% crossing
            end_idx = edge
            for i in range(edge, min(len(data_normalized), edge + 100)):
                if data_normalized[i] < threshold_low:
                    end_idx = i
                    break
            if end_idx > start_idx:
                fall_times.append(end_idx - start_idx)
        return fall_times

    def _calculate_pulse_widths(
        self, rising_edges: NDArray[np.intp], falling_edges: NDArray[np.intp]
    ) -> list[int]:
        """Calculate pulse widths from edges.

        Args:
            rising_edges: Array of rising edge indices
            falling_edges: Array of falling edge indices

        Returns:
            List of pulse widths in samples
        """
        pulse_widths = []
        for rising in rising_edges:
            # Find next falling edge
            next_falling = falling_edges[falling_edges > rising]
            if len(next_falling) > 0:
                pulse_widths.append(int(next_falling[0] - rising))
        return pulse_widths

    def _calculate_form_factor(self, data: NDArray[np.floating[Any]]) -> float:
        """Calculate form factor (RMS / mean).

        Args:
            data: Input signal

        Returns:
            Form factor
        """
        rms = float(np.sqrt(np.mean(data**2)))
        mean_abs = float(np.mean(np.abs(data)))
        return rms / mean_abs if mean_abs > 0 else 0.0

    def extract_shape(
        self, data: NDArray[np.floating[Any]], sample_rate: float
    ) -> dict[str, float]:
        """Extract shape-related features for digital signals.

        Analyzes waveform shape properties:
        - Rise time (10% to 90% transition)
        - Fall time (90% to 10% transition)
        - Duty cycle (high time / period for digital)
        - Pulse width (average high duration)
        - Form factor (RMS / mean)

        Args:
            data: Input signal as 1D numpy array.
            sample_rate: Sampling rate in Hz.

        Returns:
            Dictionary with 5 shape features.

        Example:
            >>> # 50% duty cycle square wave
            >>> square = np.tile([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 100)
            >>> features = extractor.extract_shape(square, 1000)
            >>> 0.4 < features['duty_cycle'] < 0.6
            True
        """
        # Normalize and extract basic features
        data_normalized = self._normalize_signal(data)
        duty_cycle = float(np.mean(data_normalized > 0.5))

        # Find edges
        rising_edges, falling_edges = self._find_edges(data_normalized)

        # Calculate timing features
        threshold_low = 0.1
        threshold_high = 0.9

        rise_times = self._calculate_rise_times(
            rising_edges, data_normalized, threshold_low, threshold_high
        )
        rise_time = float(np.mean(rise_times)) / sample_rate if rise_times else 0.0

        fall_times = self._calculate_fall_times(
            falling_edges, data_normalized, threshold_low, threshold_high
        )
        fall_time = float(np.mean(fall_times)) / sample_rate if fall_times else 0.0

        pulse_widths = self._calculate_pulse_widths(rising_edges, falling_edges)
        pulse_width = float(np.mean(pulse_widths)) / sample_rate if pulse_widths else 0.0

        form_factor = self._calculate_form_factor(data)

        return {
            "duty_cycle": duty_cycle,
            "rise_time": rise_time,
            "fall_time": fall_time,
            "pulse_width": pulse_width,
            "form_factor": form_factor,
        }
