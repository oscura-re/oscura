"""Golden waveform comparison for Oscura.

This module provides golden reference waveform management and comparison
functions for pass/fail testing against known-good waveforms.


Example:
    >>> from oscura.utils.comparison import create_golden, compare_to_golden
    >>> golden = create_golden(reference_trace)
    >>> result = compare_to_golden(measured_trace, golden)

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.exceptions import AnalysisError, LoaderError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class GoldenReference:
    """Golden reference waveform for comparison.

    Contains a reference waveform with tolerance bounds for pass/fail
    testing of measured waveforms.

    Attributes:
        data: Reference waveform data.
        sample_rate: Sample rate in Hz.
        upper_bound: Upper tolerance bound.
        lower_bound: Lower tolerance bound.
        tolerance: Tolerance used to create bounds.
        tolerance_type: How tolerance was applied.
        name: Reference name.
        description: Optional description.
        created: Creation timestamp.
        metadata: Additional metadata.
    """

    data: NDArray[np.float64]
    sample_rate: float
    upper_bound: NDArray[np.float64]
    lower_bound: NDArray[np.float64]
    tolerance: float
    tolerance_type: Literal["absolute", "percentage", "sigma"] = "absolute"
    name: str = "golden"
    description: str = ""
    created: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Number of samples in the reference."""
        return len(self.data)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_samples / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data.tolist(),
            "sample_rate": self.sample_rate,
            "upper_bound": self.upper_bound.tolist(),
            "lower_bound": self.lower_bound.tolist(),
            "tolerance": self.tolerance,
            "tolerance_type": self.tolerance_type,
            "name": self.name,
            "description": self.description,
            "created": self.created.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenReference:
        """Create from dictionary."""
        return cls(
            data=np.array(data["data"], dtype=np.float64),
            sample_rate=data["sample_rate"],
            upper_bound=np.array(data["upper_bound"], dtype=np.float64),
            lower_bound=np.array(data["lower_bound"], dtype=np.float64),
            tolerance=data["tolerance"],
            tolerance_type=data.get("tolerance_type", "absolute"),
            name=data.get("name", "golden"),
            description=data.get("description", ""),
            created=datetime.fromisoformat(data["created"])
            if "created" in data
            else datetime.now(),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save golden reference to file.

        Args:
            path: File path (JSON format).
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> GoldenReference:
        """Load golden reference from file.

        Args:
            path: File path.

        Returns:
            GoldenReference instance.

        Raises:
            LoaderError: If golden reference file not found.
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(
                f"Golden reference file not found: {path}",
                file_path=str(path),
            )

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)


@dataclass
class GoldenComparisonResult:
    """Result of golden waveform comparison.

    Attributes:
        passed: True if measured waveform is within tolerance.
        num_violations: Number of samples outside tolerance.
        violation_rate: Fraction of samples outside tolerance.
        max_deviation: Maximum deviation from reference.
        rms_deviation: RMS deviation from reference.
        upper_violations: Indices exceeding upper bound.
        lower_violations: Indices below lower bound.
        margin: Minimum margin to tolerance bound.
        margin_percentage: Margin as percentage of tolerance.
        statistics: Additional comparison statistics.
    """

    passed: bool
    num_violations: int
    violation_rate: float
    max_deviation: float
    rms_deviation: float
    upper_violations: NDArray[np.int64] | None = None
    lower_violations: NDArray[np.int64] | None = None
    margin: float | None = None
    margin_percentage: float | None = None
    statistics: dict[str, Any] = field(default_factory=dict)


def create_golden(
    trace: WaveformTrace,
    *,
    tolerance: float | None = None,
    tolerance_pct: float | None = None,
    tolerance_sigma: float | None = None,
    name: str = "golden",
    description: str = "",
) -> GoldenReference:
    """Create a golden reference from a trace.

    Creates a golden reference waveform with tolerance bounds for
    subsequent comparison testing.

    Args:
        trace: Reference waveform trace.
        tolerance: Absolute tolerance value.
        tolerance_pct: Percentage tolerance (0-100).
        tolerance_sigma: Tolerance as multiple of standard deviation.
        name: Name for the reference.
        description: Optional description.

    Returns:
        GoldenReference for comparison testing.

    Example:
        >>> golden = create_golden(trace, tolerance_pct=5)  # 5% tolerance
        >>> golden = create_golden(trace, tolerance=0.01)  # 10mV tolerance
    """
    data = trace.data.astype(np.float64)

    # Determine tolerance and type
    if tolerance is not None:
        tol = tolerance
        tol_type: Literal["absolute", "percentage", "sigma"] = "absolute"
    elif tolerance_pct is not None:
        # Calculate absolute tolerance from percentage
        data_range = float(np.ptp(data))
        tol = data_range * tolerance_pct / 100.0
        tol_type = "percentage"
    elif tolerance_sigma is not None:
        # Calculate tolerance from standard deviation
        tol = float(np.std(data)) * tolerance_sigma
        tol_type = "sigma"
    else:
        # Default: 1% of range
        data_range = float(np.ptp(data))
        tol = data_range * 0.01
        tol_type = "percentage"

    # Create bounds
    upper_bound = data + tol
    lower_bound = data - tol

    return GoldenReference(
        data=data,
        sample_rate=trace.metadata.sample_rate,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        tolerance=tol,
        tolerance_type=tol_type,
        name=name,
        description=description,
        metadata={
            "source_file": trace.metadata.source_file,
            "channel_name": trace.metadata.channel_name,
        },
    )


def tolerance_envelope(
    trace: WaveformTrace,
    *,
    absolute: float | None = None,
    percentage: float | None = None,
    sigma: float | None = None,
    tolerance: float | None = None,  # Alias for absolute
    tolerance_pct: float | None = None,  # Alias for percentage
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create tolerance envelope around a trace.

    Generates upper and lower bounds based on the specified tolerance.

    Args:
        trace: Reference trace.
        absolute: Absolute tolerance value.
        percentage: Percentage tolerance (0-100).
        sigma: Tolerance as multiple of standard deviation.
        tolerance: Alias for absolute (deprecated, use absolute).
        tolerance_pct: Alias for percentage (deprecated, use percentage).

    Returns:
        Tuple of (upper_bound, lower_bound) arrays.

    Raises:
        ValueError: If no tolerance type specified.

    Example:
        >>> upper, lower = tolerance_envelope(trace, percentage=5)
        >>> upper, lower = tolerance_envelope(trace, tolerance=0.1)  # absolute
    """
    data = trace.data.astype(np.float64)

    # Handle legacy parameter names
    if tolerance is not None:
        absolute = tolerance
    if tolerance_pct is not None:
        percentage = tolerance_pct

    if absolute is not None:
        tol = absolute
    elif percentage is not None:
        data_range = float(np.ptp(data))
        tol = data_range * percentage / 100.0
    elif sigma is not None:
        tol = float(np.std(data)) * sigma
    else:
        raise ValueError("Must specify absolute, percentage, or sigma tolerance")

    return data + tol, data - tol


def compare_to_golden(
    trace: WaveformTrace,
    golden: GoldenReference,
    *,
    align: bool = True,
    interpolate: bool = True,
) -> GoldenComparisonResult:
    """Compare a trace to a golden reference.

    Tests if the measured trace falls within the tolerance bounds.

    Args:
        trace: Measured trace to compare.
        golden: Golden reference to compare against.
        align: Attempt to align traces by cross-correlation.
        interpolate: Interpolate if sample counts differ.

    Returns:
        GoldenComparisonResult with pass/fail status.

    Example:
        >>> result = compare_to_golden(measured, golden)
        >>> if result.passed: print("PASS")
    """
    measured, reference, upper, lower = _prepare_data(trace, golden, interpolate)

    if align and len(measured) > 10:
        measured = _align_signals(measured, reference)

    violations = _find_violations(measured, upper, lower)
    deviation = measured - reference
    margin = _compute_margin(measured, upper, lower)
    margin_pct = (margin / golden.tolerance * 100) if golden.tolerance > 0 else None

    statistics = _compute_statistics(measured, reference, deviation)

    return GoldenComparisonResult(
        passed=violations["count"] == 0,
        num_violations=violations["count"],
        violation_rate=violations["rate"],
        max_deviation=float(np.max(np.abs(deviation))),
        rms_deviation=float(np.sqrt(np.mean(deviation**2))),
        upper_violations=violations["upper"] if len(violations["upper"]) > 0 else None,
        lower_violations=violations["lower"] if len(violations["lower"]) > 0 else None,
        margin=margin,
        margin_percentage=margin_pct,
        statistics=statistics,
    )


def _prepare_data(
    trace: WaveformTrace, golden: GoldenReference, interpolate: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Prepare and align data arrays."""
    measured = trace.data.astype(np.float64)
    reference, upper, lower = (
        golden.data.copy(),
        golden.upper_bound.copy(),
        golden.lower_bound.copy(),
    )

    if len(measured) != len(reference):
        if interpolate:
            x_measured, x_reference = (
                np.linspace(0, 1, len(measured)),
                np.linspace(0, 1, len(reference)),
            )
            measured = np.interp(x_reference, x_measured, measured)
        else:
            min_len = min(len(measured), len(reference))
            measured, reference, upper, lower = (
                measured[:min_len],
                reference[:min_len],
                upper[:min_len],
                lower[:min_len],
            )

    return measured, reference, upper, lower


def _align_signals(
    measured: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Align signals using cross-correlation."""
    from scipy import signal as sp_signal

    corr = sp_signal.correlate(measured, reference, mode="same")
    shift = len(measured) // 2 - np.argmax(corr)
    return np.roll(measured, -shift) if abs(shift) < len(measured) // 4 else measured


def _find_violations(
    measured: NDArray[np.float64], upper: NDArray[np.float64], lower: NDArray[np.float64]
) -> dict[str, Any]:
    """Find tolerance violations."""
    upper_viol, lower_viol = np.where(measured > upper)[0], np.where(measured < lower)[0]
    all_violations = np.union1d(upper_viol, lower_viol)
    return {
        "upper": upper_viol,
        "lower": lower_viol,
        "count": len(all_violations),
        "rate": len(all_violations) / len(measured) if len(measured) > 0 else 0.0,
    }


def _compute_margin(
    measured: NDArray[np.float64], upper: NDArray[np.float64], lower: NDArray[np.float64]
) -> float:
    """Compute minimum margin to tolerance bounds."""
    return min(float(np.min(upper - measured)), float(np.min(measured - lower)))


def _compute_statistics(
    measured: NDArray[np.float64], reference: NDArray[np.float64], deviation: NDArray[np.float64]
) -> dict[str, float]:
    """Compute deviation and correlation statistics."""
    measured_std, reference_std = np.std(measured), np.std(reference)
    if measured_std == 0 or reference_std == 0:
        correlation = 1.0 if np.allclose(measured, reference) else float("nan")
    else:
        correlation = float(np.corrcoef(measured, reference)[0, 1])

    return {
        "mean_deviation": float(np.mean(deviation)),
        "std_deviation": float(np.std(deviation)),
        "max_positive_deviation": float(np.max(deviation)),
        "max_negative_deviation": float(np.min(deviation)),
        "correlation": correlation,
    }


def batch_compare_to_golden(
    traces: list[WaveformTrace],
    golden: GoldenReference,
    *,
    align: bool = True,
) -> list[GoldenComparisonResult]:
    """Compare multiple traces to a golden reference.

    Tests a batch of measured traces against the same golden reference.

    Args:
        traces: List of traces to compare.
        golden: Golden reference.
        align: Attempt to align traces.

    Returns:
        List of comparison results.

    Example:
        >>> results = batch_compare_to_golden(traces, golden)
        >>> pass_rate = sum(r.passed for r in results) / len(results)
    """
    return [compare_to_golden(trace, golden, align=align) for trace in traces]


def golden_from_average(
    traces: list[WaveformTrace],
    *,
    tolerance_sigma: float = 3.0,
    name: str = "averaged_golden",
) -> GoldenReference:
    """Create golden reference from averaged traces.

    Creates a golden reference from the average of multiple traces,
    with tolerance based on the standard deviation.

    Args:
        traces: List of traces to average.
        tolerance_sigma: Number of standard deviations for tolerance.
        name: Name for the reference.

    Returns:
        GoldenReference based on averaged data.

    Raises:
        AnalysisError: If no traces provided for averaging.

    Example:
        >>> golden = golden_from_average(sample_traces, tolerance_sigma=3)
    """
    if not traces:
        raise AnalysisError("No traces provided for averaging")

    # Get common length
    lengths = [len(t.data) for t in traces]
    min_len = min(lengths)

    # Stack and average
    stacked = np.array([t.data[:min_len] for t in traces], dtype=np.float64)
    avg_data = np.mean(stacked, axis=0)
    std_data = np.std(stacked, axis=0)

    # Create tolerance from standard deviation
    tolerance = std_data * tolerance_sigma

    # Use constant tolerance (max of varying tolerance)
    max_tol = float(np.max(tolerance))

    return GoldenReference(
        data=avg_data,
        sample_rate=traces[0].metadata.sample_rate,
        upper_bound=avg_data + tolerance,
        lower_bound=avg_data - tolerance,
        tolerance=max_tol,
        tolerance_type="sigma",
        name=name,
        description=f"Averaged from {len(traces)} traces, {tolerance_sigma} sigma tolerance",
        metadata={
            "num_traces_averaged": len(traces),
            "tolerance_sigma": tolerance_sigma,
        },
    )
