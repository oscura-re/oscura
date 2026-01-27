"""Limit testing for Oscura.

This module provides specification limit testing including upper/lower
bounds, pass/fail determination, and margin analysis.


Example:
    >>> from oscura.utils.comparison import check_limits, margin_analysis
    >>> result = check_limits(trace, upper=1.5, lower=-0.5)
    >>> margins = margin_analysis(trace, limits)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.exceptions import AnalysisError
from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LimitSpec:
    """Specification limit definition.

    Defines upper and lower limits for a measurement with optional
    guardbands and absolute/relative modes.

    Attributes:
        upper: Upper limit value.
        lower: Lower limit value.
        upper_guardband: Guardband below upper limit (margin).
        lower_guardband: Guardband above lower limit (margin).
        name: Name of the specification.
        unit: Unit of measurement.
        mode: Limit mode ("absolute" or "relative").
    """

    upper: float | None = None
    lower: float | None = None
    upper_guardband: float = 0.0
    lower_guardband: float = 0.0
    name: str = "spec"
    unit: str = ""
    mode: Literal["absolute", "relative"] = "absolute"

    def __post_init__(self) -> None:
        """Validate limit specification."""
        if self.upper is None and self.lower is None:
            raise ValueError("At least one of upper or lower limit must be specified")
        if self.upper is not None and self.lower is not None and self.upper < self.lower:
            raise ValueError(f"Upper limit ({self.upper}) must be >= lower limit ({self.lower})")


@dataclass
class LimitTestResult:
    """Result of a limit test.

    Attributes:
        passed: True if all samples are within limits.
        num_violations: Number of samples violating limits.
        violation_rate: Fraction of samples violating limits.
        upper_violations: Indices of samples exceeding upper limit.
        lower_violations: Indices of samples below lower limit.
        max_value: Maximum value in data.
        min_value: Minimum value in data.
        upper_margin: Margin to upper limit (positive = within, negative = exceeded).
        lower_margin: Margin to lower limit (positive = within, negative = exceeded).
        margin_percentage: Smallest margin as percentage of limit range.
        within_guardband: True if within guardband but outside tight limits.
    """

    passed: bool
    num_violations: int
    violation_rate: float
    upper_violations: NDArray[np.int64] | None = None
    lower_violations: NDArray[np.int64] | None = None
    max_value: float = 0.0
    min_value: float = 0.0
    upper_margin: float | None = None
    lower_margin: float | None = None
    margin_percentage: float | None = None
    within_guardband: bool = False


def create_limit_spec(
    *,
    upper: float | None = None,
    lower: float | None = None,
    center: float | None = None,
    tolerance: float | None = None,
    tolerance_pct: float | None = None,
    guardband_pct: float = 0.0,
    name: str = "spec",
    unit: str = "",
) -> LimitSpec:
    """Create a limit specification.

    Creates a LimitSpec from various input formats including
    center +/- tolerance notation.

    Args:
        upper: Upper limit value.
        lower: Lower limit value.
        center: Center value (used with tolerance).
        tolerance: Absolute tolerance (+/- from center).
        tolerance_pct: Percentage tolerance (+/- % of center).
        guardband_pct: Guardband as percentage of limit range.
        name: Specification name.
        unit: Unit of measurement.

    Returns:
        LimitSpec instance.

    Raises:
        ValueError: If center requires tolerance or tolerance_pct, or if no limits specified.

    Example:
        >>> spec = create_limit_spec(center=1.0, tolerance_pct=5)  # 1.0 +/- 5%
        >>> spec = create_limit_spec(upper=1.5, lower=0.5, guardband_pct=10)
    """
    if center is not None:
        if tolerance is not None:
            upper = center + tolerance
            lower = center - tolerance
        elif tolerance_pct is not None:
            abs_tol = abs(center) * tolerance_pct / 100.0
            upper = center + abs_tol
            lower = center - abs_tol
        else:
            raise ValueError("center requires tolerance or tolerance_pct")

    if upper is None and lower is None:
        raise ValueError("Must specify limits (upper/lower or center+tolerance)")

    # Calculate guardbands
    upper_gb = 0.0
    lower_gb = 0.0
    if guardband_pct > 0 and upper is not None and lower is not None:
        range_val = upper - lower
        guardband = range_val * guardband_pct / 100.0
        upper_gb = guardband
        lower_gb = guardband

    return LimitSpec(
        upper=upper,
        lower=lower,
        upper_guardband=upper_gb,
        lower_guardband=lower_gb,
        name=name,
        unit=unit,
    )


def check_limits(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    limits: LimitSpec | None = None,
    *,
    upper: float | None = None,
    lower: float | None = None,
    reference: float | None = None,
) -> LimitTestResult:
    """Check if trace data is within specification limits.

    Tests all samples against upper and lower limits and returns
    detailed violation information.

    Args:
        trace: Input trace or data array.
        limits: LimitSpec defining the limits.
        upper: Upper limit (alternative to LimitSpec).
        lower: Lower limit (alternative to LimitSpec).
        reference: Reference value for relative limits.

    Returns:
        LimitTestResult with pass/fail status and violation details.

    Raises:
        ValueError: If no limits or bounds specified.

    Example:
        >>> result = check_limits(trace, upper=1.5, lower=-0.5)
        >>> if not result.passed:
        ...     print(f"{result.num_violations} violations found")
    """
    # Extract data array
    data = _extract_data_array(trace)

    # Get or create limit specification
    limits = _get_or_create_limits(limits, upper, lower)

    # Apply relative mode adjustment if needed
    actual_upper, actual_lower = _apply_relative_limits(limits, reference)

    # Find violations in data
    upper_viol, lower_viol = _find_violations(data, actual_upper, actual_lower)

    # Compute violation statistics
    num_violations, violation_rate = _compute_violation_stats(upper_viol, lower_viol, len(data))

    # Compute data range statistics
    max_val = float(np.max(data))
    min_val = float(np.min(data))

    # Compute margins to limits
    upper_margin, lower_margin = _compute_limit_margins(
        actual_upper, actual_lower, max_val, min_val
    )

    # Compute margin percentage
    margin_pct = _compute_margin_pct(actual_upper, actual_lower, upper_margin, lower_margin)

    # Check guardband status
    within_guardband = _check_guardband_status(num_violations, limits, upper_margin, lower_margin)

    return LimitTestResult(
        passed=num_violations == 0,
        num_violations=num_violations,
        violation_rate=violation_rate,
        upper_violations=upper_viol if len(upper_viol) > 0 else None,
        lower_violations=lower_viol if len(lower_viol) > 0 else None,
        max_value=max_val,
        min_value=min_val,
        upper_margin=upper_margin,
        lower_margin=lower_margin,
        margin_percentage=margin_pct,
        within_guardband=within_guardband,
    )


@dataclass
class MarginAnalysis:
    """Margin analysis result.

    Attributes:
        upper_margin: Margin to upper limit.
        lower_margin: Margin to lower limit.
        min_margin: Smallest margin (most critical).
        margin_percentage: Margin as percentage of limit range.
        critical_limit: Which limit has the smallest margin.
        warning: True if margin is below warning threshold.
        margin_status: "pass", "warning", or "fail".
    """

    upper_margin: float | None
    lower_margin: float | None
    min_margin: float
    margin_percentage: float
    critical_limit: Literal["upper", "lower", "both", "none"]
    warning: bool
    margin_status: Literal["pass", "warning", "fail"]


def margin_analysis(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    limits: LimitSpec,
    *,
    warning_threshold_pct: float = 20.0,
) -> MarginAnalysis:
    """Analyze margins to specification limits.

    Calculates how much margin exists between the data and the
    specification limits.

    Args:
        trace: Input trace or data array.
        limits: LimitSpec defining the limits.
        warning_threshold_pct: Threshold for margin warning (percent).

    Returns:
        MarginAnalysis with margin details.

    Raises:
        AnalysisError: If no limits defined for margin analysis.

    Example:
        >>> margins = margin_analysis(trace, limits)
        >>> print(f"Margin: {margins.margin_percentage:.1f}%")
    """
    # Extract data array
    data = _extract_data_array(trace)
    max_val = float(np.max(data))
    min_val = float(np.min(data))

    # Calculate margins to limits
    upper_margin, lower_margin = _calculate_margins(limits, max_val, min_val)

    # Determine critical limit and minimum margin
    min_margin, critical_limit = _find_critical_limit(upper_margin, lower_margin)

    # Calculate margin as percentage of limit range
    margin_pct = _calculate_margin_percentage(limits, upper_margin, lower_margin, min_margin)

    # Determine pass/warning/fail status
    margin_status, warning = _determine_margin_status(min_margin, margin_pct, warning_threshold_pct)

    return MarginAnalysis(
        upper_margin=upper_margin,
        lower_margin=lower_margin,
        min_margin=min_margin,
        margin_percentage=margin_pct,
        critical_limit=critical_limit,
        warning=warning,
        margin_status=margin_status,
    )


def _get_or_create_limits(
    limits: LimitSpec | None, upper: float | None, lower: float | None
) -> LimitSpec:
    """Get existing limits or create from upper/lower bounds.

    Args:
        limits: Existing LimitSpec or None.
        upper: Upper limit value.
        lower: Lower limit value.

    Returns:
        LimitSpec instance.

    Raises:
        ValueError: If no limits specified.
    """
    if limits is None:
        if upper is None and lower is None:
            raise ValueError("Must specify limits or upper/lower bounds")
        limits = LimitSpec(upper=upper, lower=lower)
    return limits


def _apply_relative_limits(
    limits: LimitSpec, reference: float | None
) -> tuple[float | None, float | None]:
    """Apply relative mode adjustment to limits.

    Args:
        limits: Limit specification.
        reference: Reference value for relative limits.

    Returns:
        Tuple of (actual_upper, actual_lower).
    """
    actual_upper = limits.upper
    actual_lower = limits.lower

    if limits.mode == "relative" and reference is not None:
        if actual_upper is not None:
            actual_upper = reference + actual_upper
        if actual_lower is not None:
            actual_lower = reference + actual_lower

    return (actual_upper, actual_lower)


def _find_violations(
    data: NDArray[np.float64], actual_upper: float | None, actual_lower: float | None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Find samples violating upper and lower limits.

    Args:
        data: Data array to check.
        actual_upper: Upper limit (None if no upper limit).
        actual_lower: Lower limit (None if no lower limit).

    Returns:
        Tuple of (upper_violations, lower_violations) index arrays.
    """
    upper_viol = np.array([], dtype=np.int64)
    lower_viol = np.array([], dtype=np.int64)

    if actual_upper is not None:
        upper_viol = np.where(data > actual_upper)[0]
    if actual_lower is not None:
        lower_viol = np.where(data < actual_lower)[0]

    return (upper_viol, lower_viol)


def _compute_violation_stats(
    upper_viol: NDArray[np.int64], lower_viol: NDArray[np.int64], data_length: int
) -> tuple[int, float]:
    """Compute violation count and rate.

    Args:
        upper_viol: Upper limit violations.
        lower_viol: Lower limit violations.
        data_length: Total number of samples.

    Returns:
        Tuple of (num_violations, violation_rate).
    """
    all_violations = np.union1d(upper_viol, lower_viol)
    num_violations = len(all_violations)
    violation_rate = num_violations / data_length if data_length > 0 else 0.0
    return (num_violations, violation_rate)


def _compute_limit_margins(
    actual_upper: float | None, actual_lower: float | None, max_val: float, min_val: float
) -> tuple[float | None, float | None]:
    """Compute margins to upper and lower limits.

    Args:
        actual_upper: Upper limit.
        actual_lower: Lower limit.
        max_val: Maximum value in data.
        min_val: Minimum value in data.

    Returns:
        Tuple of (upper_margin, lower_margin).
    """
    upper_margin = None
    lower_margin = None

    if actual_upper is not None:
        upper_margin = float(actual_upper - max_val)
    if actual_lower is not None:
        lower_margin = float(min_val - actual_lower)

    return (upper_margin, lower_margin)


def _compute_margin_pct(
    actual_upper: float | None,
    actual_lower: float | None,
    upper_margin: float | None,
    lower_margin: float | None,
) -> float | None:
    """Compute margin as percentage of limit range.

    Args:
        actual_upper: Upper limit.
        actual_lower: Lower limit.
        upper_margin: Margin to upper limit.
        lower_margin: Margin to lower limit.

    Returns:
        Margin percentage or None if cannot compute.
    """
    if actual_upper is not None and actual_lower is not None:
        limit_range = actual_upper - actual_lower
        if limit_range > 0:
            min_margin = min(
                upper_margin if upper_margin is not None else float("inf"),
                lower_margin if lower_margin is not None else float("inf"),
            )
            return (min_margin / limit_range) * 100.0
    return None


def _check_guardband_status(
    num_violations: int,
    limits: LimitSpec,
    upper_margin: float | None,
    lower_margin: float | None,
) -> bool:
    """Check if data is within guardband region.

    Args:
        num_violations: Number of limit violations.
        limits: Limit specification with guardbands.
        upper_margin: Margin to upper limit.
        lower_margin: Margin to lower limit.

    Returns:
        True if within guardband but outside tight limits.
    """
    if num_violations > 0:
        return False

    within_guardband = False

    if limits.upper_guardband > 0 and upper_margin is not None:
        if upper_margin < limits.upper_guardband:
            within_guardband = True

    if limits.lower_guardband > 0 and lower_margin is not None:
        if lower_margin < limits.lower_guardband:
            within_guardband = True

    return within_guardband


def _extract_data_array(trace: WaveformTrace | NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Extract numpy array from trace or array input.

    Args:
        trace: WaveformTrace object or numpy array.

    Returns:
        Data as float64 numpy array.
    """
    if isinstance(trace, WaveformTrace):
        return trace.data.astype(np.float64)
    else:
        return np.asarray(trace, dtype=np.float64)


def _calculate_margins(
    limits: LimitSpec, max_val: float, min_val: float
) -> tuple[float | None, float | None]:
    """Calculate margin to upper and lower limits.

    Args:
        limits: Specification limits.
        max_val: Maximum value in data.
        min_val: Minimum value in data.

    Returns:
        Tuple of (upper_margin, lower_margin). None if limit not defined.
    """
    upper_margin = None
    lower_margin = None

    if limits.upper is not None:
        upper_margin = limits.upper - max_val
    if limits.lower is not None:
        lower_margin = min_val - limits.lower

    return (upper_margin, lower_margin)


def _find_critical_limit(
    upper_margin: float | None, lower_margin: float | None
) -> tuple[float, Literal["upper", "lower", "both", "none"]]:
    """Find minimum margin and identify critical limit.

    Args:
        upper_margin: Margin to upper limit (None if no upper limit).
        lower_margin: Margin to lower limit (None if no lower limit).

    Returns:
        Tuple of (minimum_margin, critical_limit_name).

    Raises:
        AnalysisError: If no limits defined.
    """
    margins: list[tuple[str, float]] = []
    if upper_margin is not None:
        margins.append(("upper", upper_margin))
    if lower_margin is not None:
        margins.append(("lower", lower_margin))

    if not margins:
        raise AnalysisError("No limits defined for margin analysis")

    # Find minimum margin
    min_margin_tuple = min(margins, key=lambda x: x[1])
    min_margin = min_margin_tuple[1]

    # Determine critical limit (both if equal margins)
    if len(margins) == 2 and abs(margins[0][1] - margins[1][1]) < 1e-10:
        critical_limit: Literal["upper", "lower", "both", "none"] = "both"
    else:
        critical_limit = min_margin_tuple[0]  # type: ignore[assignment]

    return (min_margin, critical_limit)


def _calculate_margin_percentage(
    limits: LimitSpec,
    upper_margin: float | None,
    lower_margin: float | None,
    min_margin: float,
) -> float:
    """Calculate margin as percentage of limit range.

    Args:
        limits: Specification limits.
        upper_margin: Margin to upper limit.
        lower_margin: Margin to lower limit.
        min_margin: Minimum of upper/lower margins.

    Returns:
        Margin percentage (0-100+).
    """
    # Prefer range-based percentage if both limits defined
    if limits.upper is not None and limits.lower is not None:
        limit_range = limits.upper - limits.lower
        if limit_range > 0:
            return (min_margin / limit_range) * 100.0

    # Single limit: use absolute value
    if limits.upper is not None and upper_margin is not None:
        return (upper_margin / abs(limits.upper)) * 100.0 if limits.upper != 0 else 0.0
    elif limits.lower is not None and lower_margin is not None:
        return (lower_margin / abs(limits.lower)) * 100.0 if limits.lower != 0 else 0.0

    return 0.0


def _determine_margin_status(
    min_margin: float, margin_pct: float, warning_threshold_pct: float
) -> tuple[Literal["pass", "warning", "fail"], bool]:
    """Determine margin status (pass/warning/fail).

    Args:
        min_margin: Minimum margin value.
        margin_pct: Margin as percentage.
        warning_threshold_pct: Warning threshold percentage.

    Returns:
        Tuple of (status, warning_flag).
    """
    if min_margin < 0:
        return ("fail", False)
    elif margin_pct < warning_threshold_pct:
        return ("warning", True)
    else:
        return ("pass", False)
