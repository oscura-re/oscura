"""Filter design functions for Oscura.

Provides high-level filter design API with support for Butterworth,
Chebyshev, Bessel, and Elliptic filter types. Supports automatic
order calculation from specifications.


Example:
    >>> from oscura.filtering.design import LowPassFilter, design_filter
    >>> # Simple filter creation
    >>> lpf = LowPassFilter(cutoff=1e6, sample_rate=10e6, order=4)
    >>> # Spec-based design
    >>> filt = design_filter_spec(
    ...     passband=1e6, stopband=2e6,
    ...     passband_ripple=1.0, stopband_atten=40.0,
    ...     sample_rate=10e6
    ... )
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import signal

from oscura.core.exceptions import AnalysisError
from oscura.filtering.base import IIRFilter

FilterType = Literal["butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"]
BandType = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def design_filter(
    filter_type: FilterType,
    cutoff: float | tuple[float, float],
    sample_rate: float,
    order: int,
    btype: BandType = "lowpass",
    *,
    ripple_db: float = 1.0,
    stopband_atten_db: float = 40.0,
    analog: bool = False,
    output: Literal["sos", "ba"] = "sos",
) -> IIRFilter:
    """Design an IIR filter with specified parameters.

    Args:
        filter_type: Type of filter ("butterworth", "chebyshev1", "chebyshev2",
                     "bessel", or "elliptic").
        cutoff: Cutoff frequency in Hz. For bandpass/bandstop, tuple of (low, high).
        sample_rate: Sample rate in Hz.
        order: Filter order.
        btype: Band type ("lowpass", "highpass", "bandpass", "bandstop").
        ripple_db: Passband ripple in dB (for Chebyshev and Elliptic).
        stopband_atten_db: Stopband attenuation in dB (for Chebyshev2 and Elliptic).
        analog: If True, design analog filter (s-domain). Default is digital (z-domain).
        output: Output format - "sos" for second-order sections (recommended),
                "ba" for transfer function polynomials.

    Returns:
        IIRFilter object with designed coefficients.

    Raises:
        AnalysisError: If cutoff frequency is invalid or filter design fails.

    Example:
        >>> lpf = design_filter("butterworth", 1e6, 10e6, order=4)
        >>> filtered = lpf.apply(trace)

    References:
        scipy.signal.iirfilter, butter, cheby1, cheby2, ellip, bessel
    """
    # Normalize cutoff frequency
    if isinstance(cutoff, tuple):
        Wn = cutoff if analog else (cutoff[0] / (sample_rate / 2), cutoff[1] / (sample_rate / 2))
    else:
        Wn = cutoff if analog else cutoff / (sample_rate / 2)  # type: ignore[assignment]

    # Validate normalized frequency
    if not analog:
        if isinstance(Wn, tuple):
            if not (0 < Wn[0] < 1 and 0 < Wn[1] < 1):
                raise AnalysisError(
                    f"Normalized cutoff must be in (0, 1), got {Wn}. "
                    f"Cutoff {cutoff} Hz must be less than Nyquist {sample_rate / 2} Hz."
                )
        elif not 0 < Wn < 1:  # type: ignore[unreachable]
            raise AnalysisError(
                f"Normalized cutoff must be in (0, 1), got {Wn}. "
                f"Cutoff {cutoff} Hz must be less than Nyquist {sample_rate / 2} Hz."
            )

    # Design filter
    ftype_map = {
        "butterworth": "butter",
        "chebyshev1": "cheby1",
        "chebyshev2": "cheby2",
        "bessel": "bessel",
        "elliptic": "ellip",
    }
    ftype = ftype_map.get(filter_type)
    if ftype is None:
        raise AnalysisError(f"Unknown filter type: {filter_type}")

    try:
        if filter_type == "butterworth":
            if output == "sos":
                sos = signal.butter(order, Wn, btype=btype, analog=analog, output="sos")
                return IIRFilter(sample_rate=sample_rate, sos=sos)
            else:
                b, a = signal.butter(order, Wn, btype=btype, analog=analog, output="ba")
                return IIRFilter(sample_rate=sample_rate, ba=(b, a))

        elif filter_type == "chebyshev1":
            if output == "sos":
                sos = signal.cheby1(order, ripple_db, Wn, btype=btype, analog=analog, output="sos")
                return IIRFilter(sample_rate=sample_rate, sos=sos)
            else:
                b, a = signal.cheby1(order, ripple_db, Wn, btype=btype, analog=analog, output="ba")
                return IIRFilter(sample_rate=sample_rate, ba=(b, a))

        elif filter_type == "chebyshev2":
            if output == "sos":
                sos = signal.cheby2(
                    order,
                    stopband_atten_db,
                    Wn,
                    btype=btype,
                    analog=analog,
                    output="sos",
                )
                return IIRFilter(sample_rate=sample_rate, sos=sos)
            else:
                b, a = signal.cheby2(
                    order,
                    stopband_atten_db,
                    Wn,
                    btype=btype,
                    analog=analog,
                    output="ba",
                )
                return IIRFilter(sample_rate=sample_rate, ba=(b, a))

        elif filter_type == "bessel":
            if output == "sos":
                sos = signal.bessel(
                    order, Wn, btype=btype, analog=analog, output="sos", norm="phase"
                )
                return IIRFilter(sample_rate=sample_rate, sos=sos)
            else:
                b, a = signal.bessel(
                    order, Wn, btype=btype, analog=analog, output="ba", norm="phase"
                )
                return IIRFilter(sample_rate=sample_rate, ba=(b, a))

        elif filter_type == "elliptic":
            if output == "sos":
                sos = signal.ellip(
                    order,
                    ripple_db,
                    stopband_atten_db,
                    Wn,
                    btype=btype,
                    analog=analog,
                    output="sos",
                )
                return IIRFilter(sample_rate=sample_rate, sos=sos)
            else:
                b, a = signal.ellip(
                    order,
                    ripple_db,
                    stopband_atten_db,
                    Wn,
                    btype=btype,
                    analog=analog,
                    output="ba",
                )
                return IIRFilter(sample_rate=sample_rate, ba=(b, a))

        else:
            raise AnalysisError(f"Unsupported filter type: {filter_type}")

    except Exception as e:
        raise AnalysisError(f"Filter design failed: {e}") from e


def design_filter_spec(
    passband: float | tuple[float, float],
    stopband: float | tuple[float, float],
    sample_rate: float,
    passband_ripple: float = 1.0,
    stopband_atten: float = 40.0,
    *,
    filter_type: FilterType = "elliptic",
    analog: bool = False,
) -> IIRFilter:
    """Design filter from passband/stopband specifications.

    Automatically computes the minimum filter order required to meet
    the specifications.

    Args:
        passband: Passband edge frequency in Hz. Tuple for bandpass/bandstop.
        stopband: Stopband edge frequency in Hz. Tuple for bandpass/bandstop.
        sample_rate: Sample rate in Hz.
        passband_ripple: Maximum passband ripple in dB.
        stopband_atten: Minimum stopband attenuation in dB.
        filter_type: Filter type to design.
        analog: If True, design analog filter.

    Returns:
        IIRFilter object with minimum-order design.

    Raises:
        AnalysisError: If filter order cannot be determined.

    Example:
        >>> # Design a filter with 1MHz passband, 2MHz stopband, 40dB rejection
        >>> filt = design_filter_spec(
        ...     passband=1e6, stopband=2e6,
        ...     passband_ripple=1.0, stopband_atten=40.0,
        ...     sample_rate=10e6
        ... )
    """
    # Normalize frequencies
    if isinstance(passband, tuple):
        wp = (passband[0] / (sample_rate / 2), passband[1] / (sample_rate / 2))
        ws = (stopband[0] / (sample_rate / 2), stopband[1] / (sample_rate / 2))  # type: ignore[index]
    else:
        wp = passband / (sample_rate / 2)  # type: ignore[assignment]
        ws = stopband / (sample_rate / 2)  # type: ignore[assignment, operator]

    # Determine band type
    if isinstance(passband, tuple):
        # Bandpass or bandstop
        if passband[0] < stopband[0]:  # type: ignore[index]
            btype: BandType = "bandstop"
        else:
            btype = "bandpass"
    # Lowpass or highpass
    elif passband < stopband:  # type: ignore[operator]
        btype = "lowpass"
    else:
        btype = "highpass"

    # Compute minimum order
    try:
        if filter_type == "butterworth":
            order, Wn = signal.buttord(wp, ws, passband_ripple, stopband_atten, analog=analog)
        elif filter_type == "chebyshev1":
            order, Wn = signal.cheb1ord(wp, ws, passband_ripple, stopband_atten, analog=analog)
        elif filter_type == "chebyshev2":
            order, Wn = signal.cheb2ord(wp, ws, passband_ripple, stopband_atten, analog=analog)
        elif filter_type == "elliptic":
            order, Wn = signal.ellipord(wp, ws, passband_ripple, stopband_atten, analog=analog)
        else:
            # Bessel doesn't have an ord function, estimate based on Butterworth
            order, Wn = signal.buttord(wp, ws, passband_ripple, stopband_atten, analog=analog)

    except Exception as e:
        raise AnalysisError(f"Could not determine filter order: {e}") from e

    # Design with computed order
    cutoff = (
        tuple(w * sample_rate / 2 for w in Wn)
        if isinstance(Wn, np.ndarray)
        else Wn * sample_rate / 2
    )

    return design_filter(
        filter_type=filter_type,
        cutoff=cutoff,
        sample_rate=sample_rate,
        order=int(order),
        btype=btype,
        ripple_db=passband_ripple,
        stopband_atten_db=stopband_atten,
        analog=analog,
    )


# =============================================================================
# Convenience Filter Classes
# =============================================================================


class LowPassFilter(IIRFilter):
    """Low-pass Butterworth filter.

    Convenient class for creating low-pass filters with sensible defaults.

    Example:
        >>> lpf = LowPassFilter(cutoff=1e6, sample_rate=10e6, order=4)
        >>> filtered = lpf.apply(trace)
    """

    def __init__(
        self,
        cutoff: float,
        sample_rate: float,
        order: int = 4,
        *,
        filter_type: FilterType = "butterworth",
        ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
    ) -> None:
        """Initialize low-pass filter.

        Args:
            cutoff: Cutoff frequency in Hz (-3dB point for Butterworth).
            sample_rate: Sample rate in Hz.
            order: Filter order.
            filter_type: Type of filter to use.
            ripple_db: Passband ripple for Chebyshev/Elliptic.
            stopband_atten_db: Stopband attenuation for Chebyshev2/Elliptic.
        """
        filt = design_filter(
            filter_type=filter_type,
            cutoff=cutoff,
            sample_rate=sample_rate,
            order=order,
            btype="lowpass",
            ripple_db=ripple_db,
            stopband_atten_db=stopband_atten_db,
        )
        super().__init__(sample_rate=sample_rate, sos=filt.sos)
        self._cutoff = cutoff
        self._filter_type = filter_type

    @property
    def cutoff(self) -> float:
        """Cutoff frequency in Hz."""
        return self._cutoff


class HighPassFilter(IIRFilter):
    """High-pass Butterworth filter.

    Example:
        >>> hpf = HighPassFilter(cutoff=1e3, sample_rate=100e3, order=4)
        >>> filtered = hpf.apply(trace)
    """

    def __init__(
        self,
        cutoff: float,
        sample_rate: float,
        order: int = 4,
        *,
        filter_type: FilterType = "butterworth",
        ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
    ) -> None:
        """Initialize high-pass filter."""
        filt = design_filter(
            filter_type=filter_type,
            cutoff=cutoff,
            sample_rate=sample_rate,
            order=order,
            btype="highpass",
            ripple_db=ripple_db,
            stopband_atten_db=stopband_atten_db,
        )
        super().__init__(sample_rate=sample_rate, sos=filt.sos)
        self._cutoff = cutoff

    @property
    def cutoff(self) -> float:
        """Cutoff frequency in Hz."""
        return self._cutoff


class BandPassFilter(IIRFilter):
    """Band-pass filter.

    Example:
        >>> bpf = BandPassFilter(low=1e3, high=10e3, sample_rate=100e3, order=4)
        >>> filtered = bpf.apply(trace)
    """

    def __init__(
        self,
        low: float,
        high: float,
        sample_rate: float,
        order: int = 4,
        *,
        filter_type: FilterType = "butterworth",
        ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
    ) -> None:
        """Initialize band-pass filter."""
        filt = design_filter(
            filter_type=filter_type,
            cutoff=(low, high),
            sample_rate=sample_rate,
            order=order,
            btype="bandpass",
            ripple_db=ripple_db,
            stopband_atten_db=stopband_atten_db,
        )
        super().__init__(sample_rate=sample_rate, sos=filt.sos)
        self._low = low
        self._high = high

    @property
    def passband(self) -> tuple[float, float]:
        """Passband frequencies (low, high) in Hz."""
        return (self._low, self._high)


class BandStopFilter(IIRFilter):
    """Band-stop (notch) filter.

    Example:
        >>> bsf = BandStopFilter(low=50, high=60, sample_rate=1000, order=4)
        >>> filtered = bsf.apply(trace)  # Remove 50-60 Hz interference
    """

    def __init__(
        self,
        low: float,
        high: float,
        sample_rate: float,
        order: int = 4,
        *,
        filter_type: FilterType = "butterworth",
        ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
    ) -> None:
        """Initialize band-stop filter."""
        filt = design_filter(
            filter_type=filter_type,
            cutoff=(low, high),
            sample_rate=sample_rate,
            order=order,
            btype="bandstop",
            ripple_db=ripple_db,
            stopband_atten_db=stopband_atten_db,
        )
        super().__init__(sample_rate=sample_rate, sos=filt.sos)
        self._low = low
        self._high = high

    @property
    def stopband(self) -> tuple[float, float]:
        """Stopband frequencies (low, high) in Hz."""
        return (self._low, self._high)


# =============================================================================
# Filter Type Classes
# =============================================================================


class ButterworthFilter(IIRFilter):
    """Butterworth filter with maximally flat passband.

    Example:
        >>> filt = ButterworthFilter(cutoff=1e6, sample_rate=10e6, order=4, btype="lowpass")
    """

    def __init__(
        self,
        cutoff: float | tuple[float, float],
        sample_rate: float,
        order: int = 4,
        btype: BandType = "lowpass",
    ) -> None:
        filt = design_filter("butterworth", cutoff, sample_rate, order, btype)
        super().__init__(sample_rate=sample_rate, sos=filt.sos)


class ChebyshevType1Filter(IIRFilter):
    """Chebyshev Type I filter with passband ripple.

    Example:
        >>> filt = ChebyshevType1Filter(cutoff=1e6, sample_rate=10e6, order=4, ripple_db=0.5)
    """

    def __init__(
        self,
        cutoff: float | tuple[float, float],
        sample_rate: float,
        order: int = 4,
        btype: BandType = "lowpass",
        ripple_db: float = 1.0,
    ) -> None:
        filt = design_filter("chebyshev1", cutoff, sample_rate, order, btype, ripple_db=ripple_db)
        super().__init__(sample_rate=sample_rate, sos=filt.sos)


class ChebyshevType2Filter(IIRFilter):
    """Chebyshev Type II filter with stopband ripple.

    Example:
        >>> filt = ChebyshevType2Filter(cutoff=1e6, sample_rate=10e6, order=4, stopband_atten_db=40)
    """

    def __init__(
        self,
        cutoff: float | tuple[float, float],
        sample_rate: float,
        order: int = 4,
        btype: BandType = "lowpass",
        stopband_atten_db: float = 40.0,
    ) -> None:
        filt = design_filter(
            "chebyshev2",
            cutoff,
            sample_rate,
            order,
            btype,
            stopband_atten_db=stopband_atten_db,
        )
        super().__init__(sample_rate=sample_rate, sos=filt.sos)


class BesselFilter(IIRFilter):
    """Bessel filter with maximally flat group delay.

    Best for preserving waveform shape during filtering.

    Example:
        >>> filt = BesselFilter(cutoff=1e6, sample_rate=10e6, order=4)
    """

    def __init__(
        self,
        cutoff: float | tuple[float, float],
        sample_rate: float,
        order: int = 4,
        btype: BandType = "lowpass",
    ) -> None:
        filt = design_filter("bessel", cutoff, sample_rate, order, btype)
        super().__init__(sample_rate=sample_rate, sos=filt.sos)


class EllipticFilter(IIRFilter):
    """Elliptic (Cauer) filter with equiripple passband and stopband.

    Provides the sharpest transition band for a given order.

    Example:
        >>> filt = EllipticFilter(cutoff=1e6, sample_rate=10e6, order=4,
        ...                        ripple_db=0.5, stopband_atten_db=60)
    """

    def __init__(
        self,
        cutoff: float | tuple[float, float],
        sample_rate: float,
        order: int = 4,
        btype: BandType = "lowpass",
        ripple_db: float = 1.0,
        stopband_atten_db: float = 40.0,
    ) -> None:
        filt = design_filter(
            "elliptic",
            cutoff,
            sample_rate,
            order,
            btype,
            ripple_db=ripple_db,
            stopband_atten_db=stopband_atten_db,
        )
        super().__init__(sample_rate=sample_rate, sos=filt.sos)


def suggest_filter_type(
    transition_bandwidth: float,
    passband_ripple_db: float,
    stopband_atten_db: float,
) -> FilterType:
    """Suggest best filter type based on requirements.

    Recommends filter type based on design tradeoffs:
    - Butterworth: Maximally flat passband, moderate rolloff
    - Chebyshev1: Faster rolloff, passband ripple
    - Chebyshev2: Faster rolloff, stopband ripple
    - Elliptic: Sharpest rolloff, both passband and stopband ripple
    - Bessel: Linear phase, slowest rolloff (for waveform preservation)

    Args:
        transition_bandwidth: Normalized transition bandwidth (stopband - passband) / sample_rate.
        passband_ripple_db: Acceptable passband ripple in dB.
        stopband_atten_db: Required stopband attenuation in dB.

    Returns:
        Recommended filter type.

    Example:
        >>> # Sharp transition, can tolerate some ripple
        >>> ftype = suggest_filter_type(
        ...     transition_bandwidth=0.1,
        ...     passband_ripple_db=0.5,
        ...     stopband_atten_db=60.0
        ... )
        >>> print(ftype)  # 'elliptic'

    References:
        API-020: Filter Design Auto-Order
    """
    # For very sharp transitions with ripple tolerance, use elliptic
    if transition_bandwidth < 0.15 and passband_ripple_db >= 0.1:
        return "elliptic"

    # For moderate sharpness with low passband ripple, use Chebyshev2
    if transition_bandwidth < 0.2 and passband_ripple_db < 0.1:
        return "chebyshev2"

    # For moderate sharpness with ripple tolerance, use Chebyshev1
    if transition_bandwidth < 0.3 and passband_ripple_db >= 0.1:
        return "chebyshev1"

    # For phase linearity (waveform preservation), use Bessel
    if stopband_atten_db < 40.0:
        return "bessel"

    # Default to Butterworth for balanced performance
    return "butterworth"


def auto_design_filter(
    passband: float | tuple[float, float],
    stopband: float | tuple[float, float],
    sample_rate: float,
    *,
    passband_ripple_db: float = 1.0,
    stopband_atten_db: float = 40.0,
    suggest_type: bool = True,
) -> tuple[IIRFilter, dict[str, Any]]:
    """Automatically design optimal filter from specifications.

    Automatically computes filter order and optionally suggests the best
    filter type based on transition band and ripple requirements.

    Args:
        passband: Passband edge frequency in Hz. Tuple for bandpass/bandstop.
        stopband: Stopband edge frequency in Hz. Tuple for bandpass/bandstop.
        sample_rate: Sample rate in Hz.
        passband_ripple_db: Maximum passband ripple in dB (default: 1.0).
        stopband_atten_db: Minimum stopband attenuation in dB (default: 40.0).
        suggest_type: If True, automatically suggest filter type (default: True).

    Returns:
        Tuple of (IIRFilter, design_info_dict).
        design_info_dict contains: filter_type, order, cutoff, transition_bandwidth.

    Example:
        >>> # Automatic filter design with type suggestion
        >>> filt, info = auto_design_filter(
        ...     passband=1e6,
        ...     stopband=1.5e6,
        ...     sample_rate=10e6,
        ...     stopband_atten_db=60.0
        ... )
        >>> print(f"Designed {info['filter_type']} filter with order {info['order']}")
        >>> filtered = filt.apply(trace)

    References:
        API-020: Filter Design Auto-Order
    """
    # Compute transition bandwidth
    if isinstance(passband, tuple):
        # Bandpass/bandstop - use average
        transition_bw = (
            abs(stopband[0] - passband[0]) + abs(stopband[1] - passband[1])  # type: ignore[index]
        ) / 2.0
    else:
        transition_bw = abs(stopband - passband)  # type: ignore[operator]

    normalized_transition = transition_bw / sample_rate

    # Suggest filter type if requested
    if suggest_type:
        filter_type = suggest_filter_type(
            transition_bandwidth=normalized_transition,
            passband_ripple_db=passband_ripple_db,
            stopband_atten_db=stopband_atten_db,
        )
    else:
        filter_type = "butterworth"

    # Design filter with auto-order computation
    filt = design_filter_spec(
        passband=passband,
        stopband=stopband,
        sample_rate=sample_rate,
        passband_ripple=passband_ripple_db,
        stopband_atten=stopband_atten_db,
        filter_type=filter_type,
    )

    # Extract design info
    design_info = {
        "filter_type": filter_type,
        "order": filt.sos.shape[0] * 2 if filt.sos is not None else 0,
        "cutoff": passband,
        "transition_bandwidth": transition_bw,
        "passband_ripple_db": passband_ripple_db,
        "stopband_atten_db": stopband_atten_db,
    }

    return filt, design_info


__all__ = [
    "BandPassFilter",
    "BandStopFilter",
    "BandType",
    "BesselFilter",
    "ButterworthFilter",
    "ChebyshevType1Filter",
    "ChebyshevType2Filter",
    "EllipticFilter",
    "FilterType",
    "HighPassFilter",
    "LowPassFilter",
    "auto_design_filter",
    "design_filter",
    "design_filter_spec",
    "suggest_filter_type",
]
