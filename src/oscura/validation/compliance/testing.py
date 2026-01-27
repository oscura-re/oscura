"""EMC compliance testing implementation.

This module provides compliance testing against regulatory limit masks.


Example:
    >>> from oscura.validation.compliance import load_limit_mask, test_compliance
    >>> mask = load_limit_mask('FCC_Part15_ClassB')
    >>> result = test_compliance(trace, mask)
    >>> print(f"Status: {result.status}")

References:
    CISPR 16-1-1 (Measuring Apparatus)
    ANSI C63.2 (Instrumentation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace
    from oscura.validation.compliance.masks import LimitMask


class DetectorType(Enum):
    """EMC measurement detector types."""

    PEAK = "peak"
    QUASI_PEAK = "quasi-peak"
    AVERAGE = "average"
    RMS = "rms"


@dataclass
class ComplianceViolation:
    """Single compliance violation record.

    Attributes:
        frequency: Violation frequency in Hz
        measured_level: Measured level in mask unit (dBuV, etc.)
        limit_level: Limit level at this frequency
        excess_db: Amount exceeding limit (positive = violation)
        detector: Detector type used
        severity: Severity classification
    """

    frequency: float
    measured_level: float
    limit_level: float
    excess_db: float
    detector: str = "peak"
    severity: str = "FAIL"

    def __str__(self) -> str:
        """Format violation as string."""
        freq_mhz = self.frequency / 1e6
        return (
            f"{freq_mhz:.3f} MHz: {self.measured_level:.1f} dB "
            f"(limit: {self.limit_level:.1f} dB, excess: {self.excess_db:.1f} dB)"
        )


@dataclass
class ComplianceResult:
    """Compliance test result.

    Attributes:
        status: Overall status ('PASS' or 'FAIL')
        mask_name: Name of limit mask used
        violations: List of violations
        margin_to_limit: Minimum margin in dB (negative = failing)
        worst_frequency: Frequency with worst margin
        worst_margin: Worst margin value in dB
        spectrum_freq: Tested frequency array
        spectrum_level: Measured level array
        limit_level: Limit level array (interpolated to spectrum frequencies)
        detector: Detector type used
        metadata: Additional result metadata
    """

    status: str
    mask_name: str
    violations: list[ComplianceViolation]
    margin_to_limit: float
    worst_frequency: float
    worst_margin: float
    spectrum_freq: NDArray[np.float64]
    spectrum_level: NDArray[np.float64]
    limit_level: NDArray[np.float64]
    detector: str = "peak"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True if compliance test passed."""
        return self.status == "PASS"

    @property
    def violation_count(self) -> int:
        """Return number of violations."""
        return len(self.violations)

    def summary(self) -> str:
        """Generate text summary of result."""
        lines = [
            f"EMC Compliance Test: {self.mask_name}",
            f"Status: {self.status}",
            f"Margin to limit: {self.margin_to_limit:.1f} dB",
            f"Worst frequency: {self.worst_frequency / 1e6:.3f} MHz",
            f"Worst margin: {self.worst_margin:.1f} dB",
        ]

        if self.violations:
            lines.append(f"\nViolations ({len(self.violations)}):")
            for v in self.violations[:10]:  # Limit to first 10
                lines.append(f"  - {v}")
            if len(self.violations) > 10:
                lines.append(f"  ... and {len(self.violations) - 10} more")

        return "\n".join(lines)


def _prepare_spectrum(
    trace_or_spectrum: WaveformTrace | tuple[NDArray[np.float64], NDArray[np.float64]],
    detector: DetectorType,
    unit_conversion: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepare spectrum data from trace or pre-computed spectrum.

    Args:
        trace_or_spectrum: Either WaveformTrace or (freq, mag) tuple.
        detector: Detector type.
        unit_conversion: Optional unit conversion.

    Returns:
        Tuple of (frequency, spectrum_level_dB).
    """
    from oscura.core.types import WaveformTrace

    if isinstance(trace_or_spectrum, WaveformTrace):
        freq, mag = _compute_spectrum(trace_or_spectrum, detector)
    else:
        freq, mag = trace_or_spectrum

    # Convert to dB if needed
    if unit_conversion == "V_to_dBuV":
        spectrum_level = 20 * np.log10(np.abs(mag) * 1e6 + 1e-12)
    elif unit_conversion == "W_to_dBm":
        spectrum_level = 10 * np.log10(np.abs(mag) * 1000 + 1e-12)
    elif mag.max() > 0 and mag.max() < 10:
        spectrum_level = 20 * np.log10(np.abs(mag) * 1e6 + 1e-12)
    else:
        spectrum_level = mag

    return freq, spectrum_level


def _apply_frequency_filters(
    freq: NDArray[np.float64],
    spectrum_level: NDArray[np.float64],
    mask: LimitMask,
    frequency_range: tuple[float, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply frequency range filters to spectrum.

    Args:
        freq: Frequency array.
        spectrum_level: Spectrum level array.
        mask: Limit mask.
        frequency_range: Optional user-specified frequency range.

    Returns:
        Filtered (freq, spectrum_level) tuple.
    """
    # Apply user frequency range filter
    if frequency_range is not None:
        f_min, f_max = frequency_range
        mask_filter = (freq >= f_min) & (freq <= f_max)
        freq = freq[mask_filter]
        spectrum_level = spectrum_level[mask_filter]

    # Limit to mask frequency range
    mask_f_min, mask_f_max = mask.frequency_range
    in_range = (freq >= mask_f_min) & (freq <= mask_f_max)
    freq = freq[in_range]
    spectrum_level = spectrum_level[in_range]

    return freq, spectrum_level


def _find_violations(
    freq: NDArray[np.float64],
    spectrum_level: NDArray[np.float64],
    limit_level: NDArray[np.float64],
    margin: NDArray[np.float64],
    detector: DetectorType,
) -> list[ComplianceViolation]:
    """Find compliance violations.

    Args:
        freq: Frequency array.
        spectrum_level: Measured spectrum level.
        limit_level: Limit level.
        margin: Margin to limit (positive = passing).
        detector: Detector type.

    Returns:
        List of violations.
    """
    violations: list[ComplianceViolation] = []
    violation_mask = margin < 0

    if np.any(violation_mask):
        violation_indices = np.where(violation_mask)[0]
        for idx in violation_indices:
            violations.append(
                ComplianceViolation(
                    frequency=float(freq[idx]),
                    measured_level=float(spectrum_level[idx]),
                    limit_level=float(limit_level[idx]),
                    excess_db=float(-margin[idx]),
                    detector=detector.value,
                    severity="FAIL",
                )
            )

    return violations


def check_compliance(
    trace_or_spectrum: WaveformTrace | tuple[NDArray[np.float64], NDArray[np.float64]],
    mask: LimitMask,
    *,
    detector: DetectorType | str = DetectorType.PEAK,
    frequency_range: tuple[float, float] | None = None,
    unit_conversion: str | None = None,
) -> ComplianceResult:
    """Check signal against EMC limit mask.

    Args:
        trace_or_spectrum: Either a WaveformTrace to analyze, or a tuple of
            (frequency_array, magnitude_array) if spectrum already computed.
        mask: LimitMask to test against.
        detector: Detector type to use ('peak', 'quasi-peak', 'average', 'rms').
        frequency_range: Optional (min, max) frequency range to test.
        unit_conversion: Optional unit conversion ('V_to_dBuV', 'W_to_dBm', etc.)

    Returns:
        ComplianceResult with pass/fail status and violation details.

    Example:
        >>> mask = load_limit_mask('FCC_Part15_ClassB')
        >>> result = check_compliance(trace, mask)
        >>> print(result.summary())
    """
    # Handle detector type
    if isinstance(detector, str):
        detector = DetectorType(detector.lower().replace("-", "_").replace(" ", "_"))

    # Prepare spectrum
    freq, spectrum_level = _prepare_spectrum(trace_or_spectrum, detector, unit_conversion)

    # Apply frequency filters
    freq, spectrum_level = _apply_frequency_filters(freq, spectrum_level, mask, frequency_range)

    # Handle empty frequency range
    if len(freq) == 0:
        return ComplianceResult(
            status="PASS",
            mask_name=mask.name,
            violations=[],
            margin_to_limit=np.inf,
            worst_frequency=0.0,
            worst_margin=np.inf,
            spectrum_freq=np.array([]),
            spectrum_level=np.array([]),
            limit_level=np.array([]),
            detector=detector.value,
        )

    # Interpolate limit and calculate margin
    limit_level = mask.interpolate(freq)
    margin = limit_level - spectrum_level

    # Find violations
    violations = _find_violations(freq, spectrum_level, limit_level, margin, detector)

    # Compute overall results
    status = "FAIL" if violations else "PASS"
    margin_to_limit = float(np.min(margin))
    worst_idx = int(np.argmin(margin))
    worst_frequency = float(freq[worst_idx])
    worst_margin = float(margin[worst_idx])

    return ComplianceResult(
        status=status,
        mask_name=mask.name,
        violations=violations,
        margin_to_limit=margin_to_limit,
        worst_frequency=worst_frequency,
        worst_margin=worst_margin,
        spectrum_freq=freq,
        spectrum_level=spectrum_level,
        limit_level=limit_level,
        detector=detector.value,
        metadata={
            "unit": mask.unit,
            "distance": mask.distance,
            "regulatory_body": mask.regulatory_body,
        },
    )


def _compute_spectrum(
    trace: WaveformTrace,
    detector: DetectorType,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute spectrum from trace with specified detector.

    Args:
        trace: Input waveform trace.
        detector: Detector type.

    Returns:
        (frequency, magnitude) arrays.
    """
    from oscura.analyzers.waveform.spectral import fft, psd

    if detector == DetectorType.PEAK:
        # Use FFT for peak detection
        freq, mag = fft(trace)  # type: ignore[misc]
        return freq, np.abs(mag)
    elif detector == DetectorType.AVERAGE:
        # Use Welch PSD for averaging
        freq, mag = psd(trace, method="welch")  # type: ignore[call-arg]
        return freq, np.sqrt(mag)  # Convert PSD to magnitude
    elif detector == DetectorType.QUASI_PEAK:
        # Quasi-peak requires special weighting (simplified here)
        # Real implementation would use CISPR 16 weighting network
        freq, mag = fft(trace)  # type: ignore[misc]
        # Apply simplified quasi-peak envelope
        return freq, np.abs(mag) * 0.8  # Approximate QP < peak
    else:  # RMS
        freq, mag = psd(trace, method="welch")  # type: ignore[call-arg]
        return freq, np.sqrt(mag)


__all__ = [
    "ComplianceResult",
    "ComplianceViolation",
    "DetectorType",
    "check_compliance",
]
