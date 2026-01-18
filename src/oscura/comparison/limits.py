"""Limit testing for Oscura.

This module provides specification limit testing including upper/lower
bounds, pass/fail determination, and margin analysis.


Example:
    >>> from oscura.comparison import check_limits, margin_analysis
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
    # Get data
    if isinstance(trace, WaveformTrace):
        data = trace.data.astype(np.float64)
    else:
        data = np.asarray(trace, dtype=np.float64)

    # Create or use limits
    if limits is None:
        if upper is None and lower is None:
            raise ValueError("Must specify limits or upper/lower bounds")
        limits = LimitSpec(upper=upper, lower=lower)

    # Handle relative limits
    actual_upper = limits.upper
    actual_lower = limits.lower
    if limits.mode == "relative" and reference is not None:
        if actual_upper is not None:
            actual_upper = reference + actual_upper
        if actual_lower is not None:
            actual_lower = reference + actual_lower

    # Find violations
    upper_viol = np.array([], dtype=np.int64)
    lower_viol = np.array([], dtype=np.int64)

    if actual_upper is not None:
        upper_viol = np.where(data > actual_upper)[0]
    if actual_lower is not None:
        lower_viol = np.where(data < actual_lower)[0]

    # Combine violations
    all_violations = np.union1d(upper_viol, lower_viol)
    num_violations = len(all_violations)
    violation_rate = num_violations / len(data) if len(data) > 0 else 0.0

    # Compute statistics
    max_val = float(np.max(data))
    min_val = float(np.min(data))

    # Compute margins
    upper_margin = None
    lower_margin = None
    if actual_upper is not None:
        upper_margin = float(actual_upper - max_val)
    if actual_lower is not None:
        lower_margin = float(min_val - actual_lower)

    # Compute margin percentage
    margin_pct = None
    if actual_upper is not None and actual_lower is not None:
        limit_range = actual_upper - actual_lower
        if limit_range > 0:
            min_margin = min(
                upper_margin if upper_margin is not None else float("inf"),
                lower_margin if lower_margin is not None else float("inf"),
            )
            margin_pct = (min_margin / limit_range) * 100.0

    # Check guardband
    within_guardband = False
    if num_violations == 0:
        # Check if within guardband
        if limits.upper_guardband > 0 and upper_margin is not None:
            if upper_margin < limits.upper_guardband:
                within_guardband = True
        if limits.lower_guardband > 0 and lower_margin is not None:
            if lower_margin < limits.lower_guardband:
                within_guardband = True

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
    # Get data
    if isinstance(trace, WaveformTrace):
        data = trace.data.astype(np.float64)
    else:
        data = np.asarray(trace, dtype=np.float64)

    max_val = float(np.max(data))
    min_val = float(np.min(data))

    # Compute margins
    upper_margin = None
    lower_margin = None

    if limits.upper is not None:
        upper_margin = limits.upper - max_val
    if limits.lower is not None:
        lower_margin = min_val - limits.lower

    # Determine minimum margin and critical limit
    margins = []
    if upper_margin is not None:
        margins.append(("upper", upper_margin))
    if lower_margin is not None:
        margins.append(("lower", lower_margin))

    if not margins:
        raise AnalysisError("No limits defined for margin analysis")

    # Find minimum margin
    min_margin_tuple = min(margins, key=lambda x: x[1])
    min_margin = min_margin_tuple[1]

    # Determine critical limit
    if len(margins) == 2 and abs(margins[0][1] - margins[1][1]) < 1e-10:
        critical_limit: Literal["upper", "lower", "both", "none"] = "both"
    else:
        critical_limit = min_margin_tuple[0]  # type: ignore[assignment]

    # Compute margin percentage
    margin_pct = 0.0
    if limits.upper is not None and limits.lower is not None:
        limit_range = limits.upper - limits.lower
        if limit_range > 0:
            margin_pct = (min_margin / limit_range) * 100.0
    elif limits.upper is not None and upper_margin is not None:
        margin_pct = (upper_margin / abs(limits.upper)) * 100.0 if limits.upper != 0 else 0
    elif limits.lower is not None and lower_margin is not None:
        margin_pct = (lower_margin / abs(limits.lower)) * 100.0 if limits.lower != 0 else 0

    # Determine status
    warning = False
    if min_margin < 0:
        margin_status: Literal["pass", "warning", "fail"] = "fail"
    elif margin_pct < warning_threshold_pct:
        margin_status = "warning"
        warning = True
    else:
        margin_status = "pass"

    return MarginAnalysis(
        upper_margin=upper_margin,
        lower_margin=lower_margin,
        min_margin=min_margin,
        margin_percentage=margin_pct,
        critical_limit=critical_limit,
        warning=warning,
        margin_status=margin_status,
    )
