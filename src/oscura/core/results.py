"""Analysis result classes with intermediate data access.

This module provides rich result objects that store intermediate computation
results (FFT coefficients, filter states, etc.) for multi-step analysis
without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .types import WaveformTrace


@dataclass
class AnalysisResult:
    """Container for analysis results with intermediate data.

    Stores the final result along with intermediate computation artifacts
    like FFT coefficients, filter states, wavelet coefficients, etc.
    Enables multi-step analysis without recomputation.

    Attributes:
        value: The final computed value (measurement, trace, etc.).
        intermediates: Dictionary of intermediate computation results.
        metadata: Additional metadata about the computation.

    Example:
        >>> result = AnalysisResult(
        ...     value=42.5,
        ...     intermediates={'fft_coeffs': coeffs, 'frequencies': freqs}
        ... )
        >>> fft_data = result.get_intermediate('fft_coeffs')

    References:
        API-005: Intermediate Result Access
    """

    value: Any
    intermediates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_intermediate(self, key: str) -> Any:
        """Get intermediate result by key.

        Args:
            key: Name of the intermediate result.

        Returns:
            The intermediate data.

        Raises:
            KeyError: If key not found in intermediates.

        Example:
            >>> spectrum = result.get_intermediate('fft_spectrum')
        """
        if key not in self.intermediates:
            available = list(self.intermediates.keys())
            raise KeyError(f"Intermediate '{key}' not found. Available: {available}")
        return self.intermediates[key]

    def has_intermediate(self, key: str) -> bool:
        """Check if intermediate result exists.

        Args:
            key: Name of the intermediate result.

        Returns:
            True if key exists in intermediates.

        Example:
            >>> if result.has_intermediate('fft_coeffs'):
            ...     coeffs = result.get_intermediate('fft_coeffs')
        """
        return key in self.intermediates

    def list_intermediates(self) -> list[str]:
        """List all available intermediate result keys.

        Returns:
            List of intermediate result names.

        Example:
            >>> print(result.list_intermediates())
            ['fft_spectrum', 'fft_frequencies', 'fft_power', 'fft_phase']
        """
        return list(self.intermediates.keys())


@dataclass
class FFTResult(AnalysisResult):
    """Result object for FFT analysis with intermediate data.

    Provides convenient access to FFT spectrum, frequencies, power,
    and phase information.

    Attributes:
        spectrum: Complex FFT coefficients.
        frequencies: Frequency bins in Hz.
        power: Power spectrum (magnitude squared).
        phase: Phase spectrum in radians.
        trace: Original or transformed trace (optional).

    Example:
        >>> fft_result = osc.fft(trace, nfft=8192)
        >>> spectrum = fft_result.spectrum
        >>> frequencies = fft_result.frequencies
        >>> power = fft_result.power
        >>> phase = fft_result.phase
        >>> peak_freq = frequencies[power.argmax()]

    References:
        API-005: Intermediate Result Access
    """

    spectrum: NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    frequencies: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    power: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    phase: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    trace: WaveformTrace | None = None

    def __post_init__(self) -> None:
        """Initialize intermediate results dictionary."""
        # Store as intermediates for generic access
        self.intermediates.update(
            {
                "spectrum": self.spectrum,
                "frequencies": self.frequencies,
                "power": self.power,
                "phase": self.phase,
            }
        )
        if self.trace is not None:
            self.intermediates["trace"] = self.trace

        # Set value to spectrum by default
        if self.value is None:
            self.value = self.spectrum

    @property
    def peak_frequency(self) -> float:
        """Frequency of maximum power.

        Returns:
            Frequency in Hz where power spectrum peaks.

        Example:
            >>> print(f"Peak at {fft_result.peak_frequency:.2e} Hz")
        """
        if len(self.power) == 0:
            return 0.0
        return float(self.frequencies[self.power.argmax()])

    @property
    def magnitude(self) -> NDArray[np.float64]:
        """Magnitude spectrum (absolute value of FFT).

        Returns:
            Magnitude of complex spectrum.

        Example:
            >>> mag = fft_result.magnitude
        """
        return np.abs(self.spectrum)


@dataclass
class FilterResult(AnalysisResult):
    """Result object for filter operations with intermediate data.

    Provides access to filtered trace along with filter characteristics
    like transfer function and impulse response.

    Attributes:
        trace: Filtered WaveformTrace.
        transfer_function: Filter transfer function H(f) (optional).
        impulse_response: Filter impulse response h[n] (optional).
        frequency_response: Tuple of (frequencies, response) (optional).
        filter_coefficients: Filter coefficients (sos or ba format) (optional).

    Example:
        >>> filter_result = osc.low_pass(trace, cutoff=1e6, return_details=True)
        >>> filtered_trace = filter_result.trace
        >>> transfer_func = filter_result.transfer_function
        >>> impulse_resp = filter_result.impulse_response

    References:
        API-005: Intermediate Result Access
        API-009: Filter Introspection API
    """

    trace: WaveformTrace | None = None
    transfer_function: NDArray[np.complex128] | None = None
    impulse_response: NDArray[np.float64] | None = None
    frequency_response: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None
    filter_coefficients: Any | None = None

    def __post_init__(self) -> None:
        """Initialize intermediate results dictionary."""
        if self.trace is not None:
            self.intermediates["trace"] = self.trace
        if self.transfer_function is not None:
            self.intermediates["transfer_function"] = self.transfer_function
        if self.impulse_response is not None:
            self.intermediates["impulse_response"] = self.impulse_response
        if self.frequency_response is not None:
            self.intermediates["frequency_response"] = self.frequency_response
        if self.filter_coefficients is not None:
            self.intermediates["filter_coefficients"] = self.filter_coefficients

        # Set value to trace by default
        if self.value is None:
            self.value = self.trace


@dataclass
class WaveletResult(AnalysisResult):
    """Result object for wavelet transform with intermediate data.

    Provides access to wavelet coefficients, scales, and frequencies.

    Attributes:
        coeffs: Wavelet coefficients.
        scales: Wavelet scales.
        frequencies: Corresponding frequencies in Hz.
        trace: Original trace (optional).

    Example:
        >>> wavelet_result = osc.wavelet_transform(trace)
        >>> coeffs = wavelet_result.coeffs
        >>> scales = wavelet_result.scales
        >>> frequencies = wavelet_result.frequencies

    References:
        API-005: Intermediate Result Access
    """

    coeffs: NDArray[np.complex128] | None = None
    scales: NDArray[np.float64] | None = None
    frequencies: NDArray[np.float64] | None = None
    trace: WaveformTrace | None = None

    def __post_init__(self) -> None:
        """Initialize intermediate results dictionary."""
        if self.coeffs is not None:
            self.intermediates["coeffs"] = self.coeffs
        if self.scales is not None:
            self.intermediates["scales"] = self.scales
        if self.frequencies is not None:
            self.intermediates["frequencies"] = self.frequencies
        if self.trace is not None:
            self.intermediates["trace"] = self.trace

        # Set value to coeffs by default
        if self.value is None:
            self.value = self.coeffs


@dataclass
class MeasurementResult(AnalysisResult):
    """Result object for measurements with metadata.

    Stores a measurement value along with units, method, and parameters
    used for computation.

    Attributes:
        value: Measured value.
        units: Units of measurement (e.g., 'V', 'Hz', 's').
        method: Method or algorithm used.
        parameters: Dictionary of parameters used.
        confidence: Confidence interval or uncertainty (optional).

    Example:
        >>> result = MeasurementResult(
        ...     value=3.3,
        ...     units='V',
        ...     method='peak_to_peak',
        ...     parameters={'window': (0, 1e-3)}
        ... )

    References:
        API-005: Intermediate Result Access
        API-011: Measurement Provenance Tracking
    """

    units: str | None = None
    method: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Initialize metadata dictionary."""
        self.metadata.update(
            {
                "units": self.units,
                "method": self.method,
                "parameters": self.parameters,
                "confidence": self.confidence,
            }
        )

    def __str__(self) -> str:
        """String representation of measurement."""
        if self.units:
            return f"{self.value} {self.units}"
        return str(self.value)

    def __repr__(self) -> str:
        """Detailed representation of measurement."""
        parts = [f"value={self.value}"]
        if self.units:
            parts.append(f"units='{self.units}'")
        if self.method:
            parts.append(f"method='{self.method}'")
        return f"MeasurementResult({', '.join(parts)})"


__all__ = [
    "AnalysisResult",
    "FFTResult",
    "FilterResult",
    "MeasurementResult",
    "WaveletResult",
]
