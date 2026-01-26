"""Base filter classes for Oscura filtering module.

Provides abstract base classes for IIR and FIR filter implementations
with common interface for filter application and introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal

from oscura.core.exceptions import AnalysisError
from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FilterResult:
    """Result of filter application with optional introspection data.

    Attributes:
        trace: Filtered waveform trace.
        transfer_function: Optional frequency response H(f).
        impulse_response: Optional impulse response h[n].
        group_delay: Optional group delay in samples.
    """

    trace: WaveformTrace
    transfer_function: NDArray[np.complex128] | None = None
    impulse_response: NDArray[np.float64] | None = None
    group_delay: NDArray[np.float64] | None = None


class Filter(ABC):
    """Abstract base class for all filters.

    Defines the common interface for filter application and introspection.
    All filter implementations must inherit from this class.

    Attributes:
        sample_rate: Sample rate in Hz for digital filter design.
        is_stable: Whether the filter is stable (for IIR filters).
    """

    def __init__(self, sample_rate: float | None = None) -> None:
        """Initialize filter.

        Args:
            sample_rate: Sample rate in Hz. If None, must be provided at apply time.
        """
        self._sample_rate = sample_rate
        self._is_designed = False

    @property
    def sample_rate(self) -> float | None:
        """Sample rate in Hz."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        """Set sample rate and mark filter for redesign."""
        if value != self._sample_rate:
            self._sample_rate = value
            self._is_designed = False

    @property
    @abstractmethod
    def is_stable(self) -> bool:
        """Check if filter is stable."""
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Filter order."""
        ...

    @abstractmethod
    def apply(
        self,
        trace: WaveformTrace,
        *,
        return_details: bool = False,
    ) -> WaveformTrace | FilterResult:
        """Apply filter to a waveform trace.

        Args:
            trace: Input waveform trace.
            return_details: If True, return FilterResult with introspection data.

        Returns:
            Filtered trace, or FilterResult if return_details=True.
        """
        ...

    @abstractmethod
    def get_frequency_response(
        self,
        worN: int | NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Get frequency response of the filter.

        Args:
            worN: Frequencies at which to evaluate. If int, that many frequencies
                  from 0 to pi (Nyquist). If array, specific frequencies in rad/s.
                  If None, uses 512 points.

        Returns:
            Tuple of (frequencies, complex response H(f)).
        """
        ...

    @abstractmethod
    def get_impulse_response(
        self,
        n_samples: int = 256,
    ) -> NDArray[np.float64]:
        """Get impulse response of the filter.

        Args:
            n_samples: Number of samples in impulse response.

        Returns:
            Impulse response h[n].
        """
        ...

    @abstractmethod
    def get_step_response(
        self,
        n_samples: int = 256,
    ) -> NDArray[np.float64]:
        """Get step response of the filter.

        Args:
            n_samples: Number of samples in step response.

        Returns:
            Step response s[n].
        """
        ...

    def get_transfer_function(
        self,
        freqs: NDArray[np.float64] | None = None,
    ) -> NDArray[np.complex128]:
        """Get transfer function H(f) at specified frequencies.

        Args:
            freqs: Frequencies in Hz. If None, uses 512 points from 0 to Nyquist.

        Returns:
            Complex transfer function values.

        Raises:
            AnalysisError: If sample rate is not set.
        """
        if self._sample_rate is None:
            raise AnalysisError("Sample rate must be set to compute transfer function")

        if freqs is None:
            freqs = np.linspace(0, self._sample_rate / 2, 512)

        # Convert Hz to normalized frequency
        w = 2 * np.pi * freqs / self._sample_rate
        _, h = self.get_frequency_response(w)
        return h

    def get_group_delay(
        self,
        worN: int | NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get group delay of the filter.

        Args:
            worN: Frequencies at which to evaluate. If int, that many frequencies.
                  If None, uses 512 points.

        Returns:
            Tuple of (frequencies, group delay in samples).
        """
        if worN is None:
            worN = 512
        # Default implementation using phase derivative
        w, h = self.get_frequency_response(worN)
        phase = np.unwrap(np.angle(h))
        dw = np.diff(w)
        dphi = np.diff(phase)
        # Avoid division by zero
        gd = np.zeros_like(w)
        gd[:-1] = -dphi / dw
        gd[-1] = gd[-2] if len(gd) > 1 else 0
        return w, gd


class IIRFilter(Filter):
    """Infinite Impulse Response filter base class.

    Stores filter coefficients in Second-Order Sections (SOS) format
    for numerical stability, with optional B/A polynomial format.

    Attributes:
        sos: Second-order sections coefficients (preferred format).
        ba: Numerator/denominator polynomial coefficients (optional).
    """

    def __init__(
        self,
        sample_rate: float | None = None,
        sos: NDArray[np.float64] | None = None,
        ba: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> None:
        """Initialize IIR filter.

        Args:
            sample_rate: Sample rate in Hz.
            sos: Second-order sections array (n_sections, 6).
            ba: Tuple of (b, a) polynomial coefficients.
        """
        super().__init__(sample_rate)
        self._sos = sos
        self._ba = ba
        self._is_designed = sos is not None or ba is not None

    @property
    def sos(self) -> NDArray[np.float64] | None:
        """Second-order sections coefficients."""
        return self._sos

    @property
    def ba(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """B/A polynomial coefficients."""
        if self._ba is not None:
            return self._ba
        if self._sos is not None:
            # Convert SOS to BA
            b, a = signal.sos2tf(self._sos)
            return (b, a)
        return None

    @property
    def is_stable(self) -> bool:
        """Check if filter is stable (all poles inside unit circle)."""
        if self._sos is None and self._ba is None:
            return True  # Not designed yet

        ba = self.ba
        if ba is None:
            return True

        _, a = ba
        poles = np.roots(a)
        return bool(np.all(np.abs(poles) < 1.0))

    @property
    def order(self) -> int:
        """Filter order."""
        if self._sos is not None:
            return 2 * len(self._sos)
        if self._ba is not None:
            return len(self._ba[1]) - 1
        return 0

    @property
    def poles(self) -> NDArray[np.complex128]:
        """Filter poles in z-domain."""
        ba = self.ba
        if ba is None:
            return np.array([], dtype=np.complex128)
        _, a = ba
        return np.roots(a).astype(np.complex128)

    @property
    def zeros(self) -> NDArray[np.complex128]:
        """Filter zeros in z-domain."""
        ba = self.ba
        if ba is None:
            return np.array([], dtype=np.complex128)
        b, _ = ba
        return np.roots(b).astype(np.complex128)

    def apply(
        self,
        trace: WaveformTrace,
        *,
        return_details: bool = False,
        filtfilt: bool = True,
    ) -> WaveformTrace | FilterResult:
        """Apply IIR filter to waveform.

        Args:
            trace: Input waveform trace.
            return_details: If True, return FilterResult with introspection data.
            filtfilt: If True, use zero-phase filtering (forward-backward).
                     If False, use causal filtering.

        Returns:
            Filtered trace, or FilterResult if return_details=True.

        Raises:
            AnalysisError: If filter not designed or is unstable.
        """
        if self._sos is None and self._ba is None:
            raise AnalysisError("Filter not designed - no coefficients available")

        if not self.is_stable:
            raise AnalysisError("Cannot apply unstable filter")

        # Apply filter
        if self._sos is not None:
            if filtfilt:
                filtered_data = signal.sosfiltfilt(self._sos, trace.data)
            else:
                filtered_data = signal.sosfilt(self._sos, trace.data)
        else:
            b, a = self._ba  # type: ignore[misc]
            if filtfilt:
                filtered_data = signal.filtfilt(b, a, trace.data)
            else:
                filtered_data = signal.lfilter(b, a, trace.data)

        filtered_trace = WaveformTrace(
            data=filtered_data.astype(np.float64),
            metadata=trace.metadata,
        )

        if return_details:
            _w, h = self.get_frequency_response()
            impulse = self.get_impulse_response()
            _, gd = self.get_group_delay()
            return FilterResult(
                trace=filtered_trace,
                transfer_function=h,
                impulse_response=impulse,
                group_delay=gd,
            )

        return filtered_trace

    def get_frequency_response(
        self,
        worN: int | NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Get frequency response."""
        if worN is None:
            worN = 512

        if self._sos is not None:
            w, h = signal.sosfreqz(self._sos, worN=worN)
        elif self._ba is not None:
            w, h = signal.freqz(self._ba[0], self._ba[1], worN=worN)
        else:
            raise AnalysisError("Filter not designed")

        return w.astype(np.float64), h.astype(np.complex128)

    def get_impulse_response(
        self,
        n_samples: int = 256,
    ) -> NDArray[np.float64]:
        """Get impulse response."""
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0

        response: NDArray[np.float64]
        if self._sos is not None:
            response = signal.sosfilt(self._sos, impulse).astype(np.float64)
        elif self._ba is not None:
            response = signal.lfilter(self._ba[0], self._ba[1], impulse).astype(np.float64)
        else:
            raise AnalysisError("Filter not designed")

        return response

    def get_step_response(
        self,
        n_samples: int = 256,
    ) -> NDArray[np.float64]:
        """Get step response."""
        step = np.ones(n_samples)

        response: NDArray[np.float64]
        if self._sos is not None:
            response = signal.sosfilt(self._sos, step).astype(np.float64)
        elif self._ba is not None:
            response = signal.lfilter(self._ba[0], self._ba[1], step).astype(np.float64)
        else:
            raise AnalysisError("Filter not designed")

        return response


class FIRFilter(Filter):
    """Finite Impulse Response filter base class.

    Stores filter coefficients as a single array of tap weights.
    FIR filters are always stable and can achieve linear phase.

    Attributes:
        coeffs: Filter tap coefficients.
    """

    def __init__(
        self,
        sample_rate: float | None = None,
        coeffs: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize FIR filter.

        Args:
            sample_rate: Sample rate in Hz.
            coeffs: Filter coefficients (tap weights).
        """
        super().__init__(sample_rate)
        self._coeffs = coeffs
        self._is_designed = coeffs is not None

    @property
    def coeffs(self) -> NDArray[np.float64] | None:
        """Filter coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value: NDArray[np.float64]) -> None:
        """Set filter coefficients."""
        self._coeffs = value
        self._is_designed = True

    @property
    def is_stable(self) -> bool:
        """FIR filters are always stable."""
        return True

    @property
    def order(self) -> int:
        """Filter order (number of taps - 1)."""
        if self._coeffs is not None:
            return len(self._coeffs) - 1
        return 0

    @property
    def is_linear_phase(self) -> bool:
        """Check if filter has linear phase (symmetric or antisymmetric coefficients)."""
        if self._coeffs is None:
            return False
        len(self._coeffs)
        # Check symmetry
        symmetric = np.allclose(self._coeffs, self._coeffs[::-1])
        antisymmetric = np.allclose(self._coeffs, -self._coeffs[::-1])
        return symmetric or antisymmetric

    def apply(
        self,
        trace: WaveformTrace,
        *,
        return_details: bool = False,
        mode: Literal["full", "same", "valid"] = "same",
    ) -> WaveformTrace | FilterResult:
        """Apply FIR filter to waveform.

        Args:
            trace: Input waveform trace.
            return_details: If True, return FilterResult with introspection data.
            mode: Convolution mode - "same" preserves length.

        Returns:
            Filtered trace, or FilterResult if return_details=True.

        Raises:
            AnalysisError: If filter not designed.
        """
        if self._coeffs is None:
            raise AnalysisError("Filter not designed - no coefficients available")

        # Apply filter using convolution
        filtered_data = np.convolve(trace.data, self._coeffs, mode=mode)

        filtered_trace = WaveformTrace(
            data=filtered_data.astype(np.float64),
            metadata=trace.metadata,
        )

        if return_details:
            _w, h = self.get_frequency_response()
            impulse = self.get_impulse_response()
            _, gd = self.get_group_delay()
            return FilterResult(
                trace=filtered_trace,
                transfer_function=h,
                impulse_response=impulse,
                group_delay=gd,
            )

        return filtered_trace

    def get_frequency_response(
        self,
        worN: int | NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Get frequency response."""
        if self._coeffs is None:
            raise AnalysisError("Filter not designed")

        if worN is None:
            worN = 512

        w, h = signal.freqz(self._coeffs, 1, worN=worN)
        return w.astype(np.float64), h.astype(np.complex128)

    def get_impulse_response(
        self,
        n_samples: int = 256,
    ) -> NDArray[np.float64]:
        """Get impulse response (just the coefficients, zero-padded)."""
        if self._coeffs is None:
            raise AnalysisError("Filter not designed")

        if len(self._coeffs) >= n_samples:
            return self._coeffs[:n_samples].astype(np.float64)

        response = np.zeros(n_samples)
        response[: len(self._coeffs)] = self._coeffs
        return response.astype(np.float64)

    def get_step_response(
        self,
        n_samples: int = 256,
    ) -> NDArray[np.float64]:
        """Get step response."""
        if self._coeffs is None:
            raise AnalysisError("Filter not designed")

        step = np.ones(n_samples)
        response = np.convolve(step, self._coeffs, mode="full")[:n_samples]
        return response.astype(np.float64)

    def get_group_delay(
        self,
        worN: int | NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get group delay."""
        if self._coeffs is None:
            raise AnalysisError("Filter not designed")

        if worN is None:
            worN = 512

        w, gd = signal.group_delay((self._coeffs, 1), w=worN)
        return w.astype(np.float64), gd.astype(np.float64)


__all__ = [
    "FIRFilter",
    "Filter",
    "FilterResult",
    "IIRFilter",
]
