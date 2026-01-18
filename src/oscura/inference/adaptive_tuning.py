"""Adaptive parameter tuning for analysis functions.

Auto-configures analysis parameters based on signal characteristics,
reducing the need for manual parameter specification.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('signal.wfm')
    >>> tuner = osc.AdaptiveParameterTuner(trace.data, trace.metadata.sample_rate)
    >>> params = tuner.get_spectral_params()
    >>> print(f"NFFT: {params.get('nfft')}")
    >>> print(f"Window: {params.get('window')}")
    >>> print(f"Reasoning: {params.reasoning}")

References:
    Harris, F. J. (1978): On the use of windows for harmonic analysis with DFT
    Oppenheim, A. V. & Schafer, R. W. (2010): Discrete-Time Signal Processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TunedParameters:
    """Container for auto-tuned parameters.

    Attributes:
        parameters: Dictionary of parameter names to values.
        confidence: Confidence in parameter tuning (0.0-1.0).
        reasoning: Dictionary mapping parameter names to reasoning strings.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    reasoning: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default.

        Args:
            key: Parameter name.
            default: Default value if parameter not found.

        Returns:
            Parameter value or default.
        """
        return self.parameters.get(key, default)


class AdaptiveParameterTuner:
    """Auto-configure analysis parameters based on signal characteristics.

    This class analyzes signal characteristics and provides intelligent
    parameter suggestions for various analysis domains (spectral, digital,
    timing, jitter, pattern recognition).

    Attributes:
        data: Input signal data array.
        sample_rate: Sample rate in Hz.
        signal_type: Optional signal type hint (digital, analog, etc.).

    Example:
        >>> tuner = AdaptiveParameterTuner(signal_data, sample_rate=1e6)
        >>> spectral_params = tuner.get_spectral_params()
        >>> print(spectral_params.parameters)
        {'nfft': 8192, 'window': 'hann', 'overlap': 0.5}
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        sample_rate: float = 1.0,
        signal_type: str | None = None,
    ):
        """Initialize tuner with signal data.

        Args:
            data: Input signal data.
            sample_rate: Sample rate in Hz.
            signal_type: Optional signal type hint (digital, analog, etc.).
        """
        self.data = data
        self.sample_rate = sample_rate
        self.signal_type = signal_type

        # Pre-compute signal characteristics
        self._characteristics = self._analyze_signal()

    def _analyze_signal(self) -> dict[str, Any]:
        """Analyze signal characteristics for parameter tuning.

        Returns:
            Dictionary of signal characteristics including statistics,
            noise estimates, frequency content, and signal type indicators.
        """
        chars: dict[str, Any] = {}

        try:
            # Basic statistics
            chars["mean"] = float(np.mean(self.data))
            chars["std"] = float(np.std(self.data))
            chars["min"] = float(np.min(self.data))
            chars["max"] = float(np.max(self.data))
            chars["range"] = chars["max"] - chars["min"]
            chars["n_samples"] = len(self.data)
            chars["duration"] = len(self.data) / self.sample_rate

            # Detect if digital
            unique_values = len(np.unique(np.round(self.data, decimals=2)))
            chars["likely_digital"] = unique_values < 10

            # Estimate dominant frequency
            chars["dominant_freq"] = self._estimate_dominant_frequency()

            # Estimate noise floor
            median = np.median(self.data)
            mad = np.median(np.abs(self.data - median)) * 1.4826
            chars["noise_floor"] = float(mad)

            # SNR estimate
            signal_power = np.var(self.data)
            noise_power = mad**2
            if noise_power > 0:
                chars["snr_db"] = float(10 * np.log10(signal_power / noise_power))
            else:
                chars["snr_db"] = 40.0

        except Exception as e:
            logger.debug(f"Error analyzing signal: {e}")

        return chars

    def _estimate_dominant_frequency(self) -> float | None:
        """Estimate dominant frequency using FFT.

        Returns:
            Dominant frequency in Hz, or None if not detectable.
        """
        try:
            data_ac = self.data - np.mean(self.data)
            fft_result = np.fft.rfft(data_ac)
            freqs = np.fft.rfftfreq(len(data_ac), d=1.0 / self.sample_rate)
            magnitude = np.abs(fft_result[1:])  # Skip DC

            if len(magnitude) > 0:
                peak_idx = np.argmax(magnitude)
                return float(freqs[1:][peak_idx])
        except Exception:
            pass
        return None

    def get_spectral_params(self) -> TunedParameters:
        """Get tuned parameters for spectral analysis.

        Selects FFT size, window function, and overlap based on signal
        characteristics and quality requirements.

        Returns:
            TunedParameters with spectral analysis configuration.

        Example:
            >>> params = tuner.get_spectral_params()
            >>> print(f"NFFT: {params.get('nfft')}")
            >>> print(f"Reasoning: {params.reasoning['nfft']}")
        """
        params = {}
        reasoning = {}
        confidence = 0.8

        n_samples = self._characteristics.get("n_samples", 1000)

        # NFFT - power of 2, balancing resolution and computation
        ideal_nfft = min(8192, max(256, 2 ** int(np.ceil(np.log2(n_samples / 4)))))
        params["nfft"] = ideal_nfft
        reasoning["nfft"] = f"Power of 2 for efficiency, ~{n_samples / ideal_nfft:.0f} averages"

        # Window selection based on signal characteristics
        snr = self._characteristics.get("snr_db", 20)
        if snr < 15:
            params["window"] = "blackman"
            reasoning["window"] = "Low SNR - using Blackman for better noise rejection"
        elif snr < 25:
            params["window"] = "hann"
            reasoning["window"] = "Moderate SNR - using Hann for balance"
        else:
            params["window"] = "hamming"
            reasoning["window"] = "Good SNR - using Hamming for resolution"

        # Overlap
        params["overlap"] = 0.5
        reasoning["overlap"] = "Standard 50% overlap for smooth averaging"

        # Frequency range based on dominant frequency
        dom_freq = self._characteristics.get("dominant_freq")
        if dom_freq and dom_freq > 0:
            params["freq_min"] = max(0, dom_freq / 10)
            params["freq_max"] = min(self.sample_rate / 2, dom_freq * 5)
            reasoning["freq_range"] = f"Based on dominant frequency {dom_freq:.1f} Hz"

        return TunedParameters(parameters=params, confidence=confidence, reasoning=reasoning)

    def get_digital_params(self) -> TunedParameters:
        """Get tuned parameters for digital signal analysis.

        Determines threshold levels, edge detection sensitivity, and
        baud rate hints based on signal characteristics.

        Returns:
            TunedParameters with digital analysis configuration.

        Example:
            >>> params = tuner.get_digital_params()
            >>> print(f"Threshold: {params.get('threshold')}")
            >>> print(f"Baud rate hint: {params.get('baud_rate_hint')}")
        """
        params = {}
        reasoning = {}
        confidence = 0.7

        chars = self._characteristics

        # Threshold based on signal levels
        if chars.get("likely_digital"):
            mid = (chars["min"] + chars["max"]) / 2
            params["threshold"] = mid
            params["threshold_low"] = chars["min"] + 0.3 * chars["range"]
            params["threshold_high"] = chars["max"] - 0.3 * chars["range"]
            reasoning["threshold"] = (
                f"Midpoint of signal range ({chars['min']:.2f} to {chars['max']:.2f})"
            )
            confidence = 0.85
        else:
            params["threshold"] = chars.get("mean", 0)
            reasoning["threshold"] = "Using mean (signal may not be digital)"
            confidence = 0.5

        # Edge detection sensitivity based on noise
        noise = chars.get("noise_floor", 0.1)
        params["min_edge_separation"] = max(2, int(noise * 10))
        reasoning["min_edge_separation"] = f"Based on noise floor {noise:.3f}"

        # Baud rate hint from dominant frequency
        dom_freq = chars.get("dominant_freq")
        if dom_freq and dom_freq > 0:
            # Common baud rates
            common_bauds = [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
            closest_baud = min(common_bauds, key=lambda b: abs(b - dom_freq * 2))
            params["baud_rate_hint"] = closest_baud
            reasoning["baud_rate"] = f"Estimated from frequency {dom_freq:.0f} Hz"

        return TunedParameters(parameters=params, confidence=confidence, reasoning=reasoning)

    def get_timing_params(self) -> TunedParameters:
        """Get tuned parameters for timing analysis.

        Configures time resolution, expected period, and edge timing
        thresholds based on sample rate and signal characteristics.

        Returns:
            TunedParameters with timing analysis configuration.

        Example:
            >>> params = tuner.get_timing_params()
            >>> print(f"Expected period: {params.get('expected_period')}")
            >>> print(f"Tolerance: {params.get('period_tolerance')}")
        """
        params = {}
        reasoning = {}
        confidence = 0.75

        chars = self._characteristics

        # Time resolution based on sample rate
        params["time_resolution"] = 1.0 / self.sample_rate
        reasoning["time_resolution"] = f"Based on sample rate {self.sample_rate:.0f} Hz"

        # Expected period from dominant frequency
        dom_freq = chars.get("dominant_freq")
        if dom_freq and dom_freq > 0:
            params["expected_period"] = 1.0 / dom_freq
            params["period_tolerance"] = 0.2 / dom_freq  # 20% tolerance
            reasoning["period"] = f"From dominant frequency {dom_freq:.1f} Hz"
            confidence = 0.85

        # Edge timing thresholds
        noise = chars.get("noise_floor", 0.1)
        params["edge_threshold"] = noise * 3  # 3-sigma
        reasoning["edge_threshold"] = f"3x noise floor ({noise:.3f})"

        return TunedParameters(parameters=params, confidence=confidence, reasoning=reasoning)

    def get_jitter_params(self) -> TunedParameters:
        """Get tuned parameters for jitter analysis.

        Determines unit interval, histogram binning, and tolerance
        parameters for jitter measurements.

        Returns:
            TunedParameters with jitter analysis configuration.

        Example:
            >>> params = tuner.get_jitter_params()
            >>> print(f"Unit interval: {params.get('unit_interval')}")
            >>> print(f"Histogram bins: {params.get('histogram_bins')}")
        """
        params = {}
        reasoning = {}
        confidence = 0.7

        chars = self._characteristics

        # Unit interval from dominant frequency
        dom_freq = chars.get("dominant_freq")
        if dom_freq and dom_freq > 0:
            ui = 1.0 / dom_freq
            params["unit_interval"] = ui
            params["ui_tolerance"] = ui * 0.1
            reasoning["unit_interval"] = f"From dominant frequency {dom_freq:.1f} Hz"
            confidence = 0.85

        # Histogram bins based on data range and noise
        snr = chars.get("snr_db", 20)
        if snr > 30:
            params["histogram_bins"] = 256
        elif snr > 20:
            params["histogram_bins"] = 128
        else:
            params["histogram_bins"] = 64
        reasoning["histogram_bins"] = f"Based on SNR {snr:.0f} dB"

        return TunedParameters(parameters=params, confidence=confidence, reasoning=reasoning)

    def get_pattern_params(self) -> TunedParameters:
        """Get tuned parameters for pattern analysis.

        Configures minimum pattern length and maximum distance for
        fuzzy matching based on signal characteristics.

        Returns:
            TunedParameters with pattern analysis configuration.

        Example:
            >>> params = tuner.get_pattern_params()
            >>> print(f"Min pattern length: {params.get('min_length')}")
            >>> print(f"Max fuzzy distance: {params.get('max_distance')}")
        """
        params = {}
        reasoning = {}
        confidence = 0.7

        chars = self._characteristics
        n_samples = chars.get("n_samples", 1000)

        # Min pattern length based on signal characteristics
        dom_freq = chars.get("dominant_freq")
        if dom_freq and dom_freq > 0:
            samples_per_period = self.sample_rate / dom_freq
            params["min_length"] = max(3, int(samples_per_period / 4))
            reasoning["min_length"] = (
                f"Quarter of estimated period ({samples_per_period:.0f} samples)"
            )
        else:
            params["min_length"] = max(3, n_samples // 100)
            reasoning["min_length"] = "1% of signal length"

        # Max distance for fuzzy matching based on noise
        noise_ratio = chars.get("noise_floor", 0.1) / max(chars.get("range", 1), 0.001)
        params["max_distance"] = max(1, int(noise_ratio * 10))
        reasoning["max_distance"] = f"Based on noise ratio {noise_ratio:.2%}"

        return TunedParameters(parameters=params, confidence=confidence, reasoning=reasoning)

    def get_params_for_domain(self, domain: str) -> TunedParameters:
        """Get tuned parameters for a specific analysis domain.

        Args:
            domain: Analysis domain name (spectral, digital, timing, jitter, pattern).

        Returns:
            TunedParameters for the specified domain.

        Example:
            >>> params = tuner.get_params_for_domain("spectral")
            >>> print(params.parameters)
            {'nfft': 8192, 'window': 'hann', 'overlap': 0.5}
        """
        domain_lower = domain.lower()

        if "spectral" in domain_lower or "fft" in domain_lower:
            return self.get_spectral_params()
        elif "digital" in domain_lower:
            return self.get_digital_params()
        elif "timing" in domain_lower:
            return self.get_timing_params()
        elif "jitter" in domain_lower:
            return self.get_jitter_params()
        elif "pattern" in domain_lower:
            return self.get_pattern_params()
        else:
            # Return basic params for unknown domains
            return TunedParameters(
                parameters={},
                confidence=0.5,
                reasoning={"note": "No domain-specific tuning available"},
            )


def get_adaptive_parameters(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    domain: str,
    signal_type: str | None = None,
) -> TunedParameters:
    """Convenience function to get adaptive parameters.

    This is a shortcut for creating an AdaptiveParameterTuner and
    getting parameters for a specific domain.

    Args:
        data: Input signal data.
        sample_rate: Sample rate in Hz.
        domain: Analysis domain (spectral, digital, timing, jitter, pattern).
        signal_type: Optional signal type hint.

    Returns:
        TunedParameters for the specified domain.

    Example:
        >>> params = get_adaptive_parameters(signal, 1e6, "spectral")
        >>> print(f"Window: {params.get('window')}")
        >>> print(f"Confidence: {params.confidence}")
    """
    tuner = AdaptiveParameterTuner(data, sample_rate, signal_type)
    return tuner.get_params_for_domain(domain)


__all__ = [
    "AdaptiveParameterTuner",
    "TunedParameters",
    "get_adaptive_parameters",
]
