"""Signal quality warnings for Oscura.

This module provides automated detection and warning of signal quality issues
including clipping, noise, saturation, and undersampling.


Example:
    >>> from oscura.validation.quality.warnings import SignalQualityAnalyzer
    >>> analyzer = SignalQualityAnalyzer()
    >>> warnings = analyzer.analyze(trace)
    >>> for warning in warnings:
    ...     print(warning)

References:
    - IEEE 1057: Standard for Digitizing Waveform Recorders
    - Nyquist sampling theorem
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class QualityWarning:
    """Signal quality warning.

    Attributes:
        severity: Warning severity (error, warning, info)
        category: Warning category (clipping, noise, saturation, undersampling)
        message: Human-readable warning message
        value: Numeric value associated with warning
        threshold: Threshold that triggered warning
        suggestion: Suggested action to fix issue

    Example:
        >>> warning = QualityWarning(
        ...     severity="warning",
        ...     category="clipping",
        ...     message="Signal clipping detected",
        ...     value=5.2,
        ...     threshold=5.0,
        ...     suggestion="Reduce input amplitude or increase ADC range"
        ... )
        >>> print(warning)

    References:
        EDGE-001: Signal Quality Warnings
    """

    severity: Literal["error", "warning", "info"]
    category: Literal["clipping", "noise", "saturation", "undersampling", "dc_offset"]
    message: str
    value: float
    threshold: float
    suggestion: str = ""

    def __str__(self) -> str:
        """Format warning as string.

        Returns:
            Formatted warning message
        """
        prefix = {"error": "ERROR", "warning": "WARNING", "info": "INFO"}[self.severity]
        msg = f"[{prefix}] {self.message}"
        if self.value is not None:
            msg += f" (value: {self.value:.3f}, threshold: {self.threshold:.3f})"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


class SignalQualityAnalyzer:
    """Analyzer for signal quality issues.

    : Detect clipping, undersampling, noise, and saturation.
    Performs comprehensive signal quality checks and generates warnings.

    Args:
        clip_threshold: Clipping detection threshold (fraction of range, default: 0.99)
        noise_threshold_db: Noise floor threshold in dB (default: -40)
        saturation_threshold: ADC saturation threshold (default: 0.98)
        nyquist_factor: Factor for Nyquist frequency check (default: 2.0)

    Example:
        >>> from oscura.validation.quality.warnings import SignalQualityAnalyzer
        >>> analyzer = SignalQualityAnalyzer(clip_threshold=0.95)
        >>> warnings = analyzer.analyze(trace)
        >>> if warnings:
        ...     for w in warnings:
        ...         print(w)

    References:
        EDGE-001: Signal Quality Warnings
    """

    def __init__(
        self,
        *,
        clip_threshold: float = 0.99,
        noise_threshold_db: float = -40.0,
        saturation_threshold: float = 0.98,
        nyquist_factor: float = 2.0,
    ) -> None:
        """Initialize signal quality analyzer.

        Args:
            clip_threshold: Clipping detection threshold
            noise_threshold_db: Noise floor threshold in dB
            saturation_threshold: ADC saturation threshold
            nyquist_factor: Nyquist frequency factor
        """
        self.clip_threshold = clip_threshold
        self.noise_threshold_db = noise_threshold_db
        self.saturation_threshold = saturation_threshold
        self.nyquist_factor = nyquist_factor

    def analyze(
        self,
        trace: WaveformTrace | NDArray[np.float64],
        *,
        sample_rate: float | None = None,
        adc_range: tuple[float, float] | None = None,
    ) -> list[QualityWarning]:
        """Analyze signal quality and generate warnings.

        : Comprehensive signal quality detection.

        Args:
            trace: Input trace or signal array
            sample_rate: Sample rate in Hz (required for undersampling check)
            adc_range: ADC range as (min, max) tuple

        Returns:
            List of QualityWarning objects

        Example:
            >>> warnings = analyzer.analyze(trace, sample_rate=1e9)
            >>> for warning in warnings:
            ...     print(warning)

        References:
            EDGE-001: Signal Quality Warnings
        """
        # Extract data and sample rate
        if hasattr(trace, "data"):
            data = trace.data
            if sample_rate is None and hasattr(trace, "metadata"):
                sample_rate = trace.metadata.sample_rate
        else:
            data = trace  # type: ignore[assignment]

        # Ensure data is NDArray[np.float64]
        data_array: NDArray[np.float64] = np.asarray(data, dtype=np.float64)

        warnings: list[QualityWarning] = []

        # Check clipping
        warnings.extend(
            check_clipping(
                data_array,
                threshold=self.clip_threshold,
                adc_range=adc_range,
            )
        )

        # Check saturation
        warnings.extend(
            check_saturation(
                data_array,
                threshold=self.saturation_threshold,
                adc_range=adc_range,
            )
        )

        # Check noise
        warnings.extend(
            check_noise(
                data_array,
                threshold_db=self.noise_threshold_db,
            )
        )

        # Check undersampling (requires sample rate)
        if sample_rate is not None:
            warnings.extend(
                check_undersampling(
                    data_array,
                    sample_rate=sample_rate,
                    nyquist_factor=self.nyquist_factor,
                )
            )

        return warnings


def check_clipping(
    signal: NDArray[np.float64],
    *,
    threshold: float = 0.99,
    adc_range: tuple[float, float] | None = None,
) -> list[QualityWarning]:
    """Detect signal clipping.

    : Detect clipping (signal hits rail).

    Args:
        signal: Input signal array
        threshold: Fraction of range for clipping detection (default: 0.99)
        adc_range: ADC range as (min, max) tuple

    Returns:
        List of clipping warnings

    Example:
        >>> import numpy as np
        >>> signal = np.clip(np.random.randn(1000), -1, 1)
        >>> warnings = check_clipping(signal)
        >>> if warnings:
        ...     print("Clipping detected!")

    References:
        EDGE-001: Detect clipping
    """
    warnings: list[QualityWarning] = []

    # Determine signal range
    if adc_range is not None:
        min_val, max_val = adc_range
    else:
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))

    signal_range = max_val - min_val
    if signal_range == 0:
        return warnings

    # Count samples near limits
    upper_limit = min_val + signal_range * threshold
    lower_limit = min_val + signal_range * (1 - threshold)

    n_upper: int = int(np.sum(signal >= upper_limit))
    n_lower: int = int(np.sum(signal <= lower_limit))
    n_clipped = n_upper + n_lower
    clip_percent = float(100.0 * n_clipped / len(signal))

    if clip_percent > 1.0:  # More than 1% clipped
        severity: Literal["error", "warning"] = "error" if clip_percent > 5.0 else "warning"
        warnings.append(
            QualityWarning(
                severity=severity,
                category="clipping",
                message=f"Signal clipping detected at {clip_percent:.1f}% of samples",
                value=clip_percent,
                threshold=1.0,
                suggestion="Reduce input amplitude or increase ADC range",
            )
        )

    return warnings


def check_saturation(
    signal: NDArray[np.float64],
    *,
    threshold: float = 0.98,
    adc_range: tuple[float, float] | None = None,
) -> list[QualityWarning]:
    """Detect ADC saturation.

    : Detect saturation (ADC range utilization).

    Args:
        signal: Input signal array
        threshold: Saturation threshold as fraction of range (default: 0.98)
        adc_range: ADC range as (min, max) tuple

    Returns:
        List of saturation warnings

    Example:
        >>> warnings = check_saturation(signal, threshold=0.95)

    References:
        EDGE-001: Detect saturation
    """
    warnings: list[QualityWarning] = []

    # Determine signal range
    if adc_range is not None:
        adc_min, adc_max = adc_range
        adc_span = adc_max - adc_min
    else:
        # Assume signal uses full observed range as ADC range
        adc_min = float(np.min(signal))
        adc_max = float(np.max(signal))
        adc_span = adc_max - adc_min

    if adc_span == 0:
        return warnings

    # Calculate range utilization
    signal_min = float(np.min(signal))
    signal_max = float(np.max(signal))
    signal_span = signal_max - signal_min
    utilization = signal_span / adc_span

    if utilization > threshold:
        warnings.append(
            QualityWarning(
                severity="warning",
                category="saturation",
                message=f"High ADC range utilization: {utilization * 100:.1f}%",
                value=utilization * 100,
                threshold=threshold * 100,
                suggestion="Consider increasing ADC range or reducing signal amplitude",
            )
        )

    return warnings


def check_noise(
    signal: NDArray[np.float64],
    *,
    threshold_db: float = -40.0,
) -> list[QualityWarning]:
    """Detect excessive noise.

    : Detect noise (SNR below threshold warning).

    Args:
        signal: Input signal array
        threshold_db: Noise threshold in dB (default: -40)

    Returns:
        List of noise warnings

    Example:
        >>> warnings = check_noise(signal, threshold_db=-50)

    References:
        EDGE-001: Detect noise
    """
    warnings: list[QualityWarning] = []

    # Estimate SNR
    signal_power = float(np.mean(signal**2))
    if signal_power == 0:
        return warnings

    # Estimate noise from high-frequency components
    # Simple approach: use standard deviation as noise estimate
    noise_power = float(np.var(signal))
    if noise_power == 0:
        return warnings

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

    if snr_db < threshold_db:
        warnings.append(
            QualityWarning(
                severity="warning",
                category="noise",
                message=f"High noise level detected: SNR = {snr_db:.1f} dB",
                value=snr_db,
                threshold=threshold_db,
                suggestion="Check signal source, grounding, and shielding",
            )
        )

    return warnings


def check_undersampling(
    signal: NDArray[np.float64],
    *,
    sample_rate: float,
    nyquist_factor: float = 2.0,
) -> list[QualityWarning]:
    """Detect undersampling (Nyquist violation).

    : Detect undersampling (Nyquist violation warning).

    Args:
        signal: Input signal array
        sample_rate: Sample rate in Hz
        nyquist_factor: Required factor above Nyquist (default: 2.0)

    Returns:
        List of undersampling warnings

    Example:
        >>> warnings = check_undersampling(signal, sample_rate=1e9)

    References:
        EDGE-001: Detect undersampling
        Nyquist-Shannon sampling theorem
    """
    warnings: list[QualityWarning] = []

    # Estimate highest frequency component using FFT
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    power = np.abs(fft) ** 2

    # Find frequency where power drops to 1% of peak
    peak_power: float = float(np.max(power))
    threshold_power = peak_power * 0.01

    # Find highest significant frequency
    significant_freqs = freqs[power > threshold_power]
    if len(significant_freqs) > 0:
        max_freq = float(np.max(significant_freqs))
        nyquist_freq = sample_rate / 2.0
        required_nyquist = max_freq * nyquist_factor

        if required_nyquist > nyquist_freq:
            warnings.append(
                QualityWarning(
                    severity="error",
                    category="undersampling",
                    message=(
                        f"Undersampling detected: signal contains "
                        f"{max_freq / 1e6:.1f} MHz, but Nyquist frequency is "
                        f"{nyquist_freq / 1e6:.1f} MHz"
                    ),
                    value=max_freq,
                    threshold=nyquist_freq / nyquist_factor,
                    suggestion=(
                        f"Increase sample rate to at least "
                        f"{required_nyquist * 2 / 1e6:.1f} MS/s or apply anti-aliasing filter"
                    ),
                )
            )

    return warnings


__all__ = [
    "QualityWarning",
    "SignalQualityAnalyzer",
    "check_clipping",
    "check_noise",
    "check_saturation",
    "check_undersampling",
]
