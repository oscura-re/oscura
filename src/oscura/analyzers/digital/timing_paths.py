"""Multi-IC timing path analysis.

Analyzes timing through chains of ICs to identify critical paths and
validate system-level timing budgets.

Example:
    >>> from oscura.analyzers.digital.timing_paths import analyze_timing_path
    >>> path = [
    ...     ("74LS74", clk_trace, q_trace),
    ...     ("74LS00", q_trace, y_trace),
    ...     ("74LS74", y_trace, q2_trace),
    ... ]
    >>> result = analyze_timing_path(path)
    >>> print(f"Total propagation delay: {result.total_delay*1e9:.1f}ns")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


@dataclass
class ICStage:
    """A single IC stage in a timing path.

    Attributes:
        ic_name: IC part number.
        input_trace: Input signal trace.
        output_trace: Output signal trace.
        measured_delay: Measured propagation delay.
        spec_delay: Specification delay from database.
        margin: Timing margin (positive = meets spec).
    """

    ic_name: str
    input_trace: WaveformTrace
    output_trace: WaveformTrace
    measured_delay: float
    spec_delay: float | None
    margin: float | None


@dataclass
class TimingPathResult:
    """Result of timing path analysis.

    Attributes:
        stages: List of IC stages in the path.
        total_delay: Total path propagation delay.
        total_spec_delay: Total specification delay.
        critical_stage_idx: Index of stage with worst margin.
        path_margin: Overall path timing margin.
        meets_timing: Whether path meets all timing specs.
    """

    stages: list[ICStage]
    total_delay: float
    total_spec_delay: float | None
    critical_stage_idx: int | None
    path_margin: float | None
    meets_timing: bool


def analyze_timing_path(
    path: list[tuple[str, WaveformTrace, WaveformTrace]],
    *,
    target_frequency: float | None = None,
) -> TimingPathResult:
    """Analyze timing through a chain of ICs.

    Args:
        path: List of (ic_name, input_trace, output_trace) tuples.
        target_frequency: Optional target frequency for setup time validation.

    Returns:
        TimingPathResult object.

    Example:
        >>> path = [
        ...     ("74LS74", clk, q1),
        ...     ("74LS00", q1, y),
        ...     ("74LS74", y, q2),
        ... ]
        >>> result = analyze_timing_path(path, target_frequency=10e6)
        >>> if not result.meets_timing:
        ...     print(f"Timing violation in stage {result.critical_stage_idx}")
    """
    from oscura.analyzers.digital.ic_database import IC_DATABASE
    from oscura.analyzers.digital.timing import propagation_delay

    stages: list[ICStage] = []
    total_delay = 0.0
    total_spec_delay = 0.0
    worst_margin = float("inf")
    critical_stage_idx = None

    for idx, (ic_name, input_trace, output_trace) in enumerate(path):
        # Measure propagation delay
        measured_pd_raw = propagation_delay(input_trace, output_trace, edge_type="rising")
        measured_pd = (
            float(measured_pd_raw)
            if isinstance(measured_pd_raw, (int, float, np.number))
            else float(measured_pd_raw.item())
        )

        # Get spec from database if available
        spec_delay = None
        margin = None

        if ic_name in IC_DATABASE:
            ic_spec = IC_DATABASE[ic_name]
            if "t_pd" in ic_spec.timing:
                spec_delay = ic_spec.timing["t_pd"]
                margin_raw = spec_delay - measured_pd
                margin = (
                    float(margin_raw)
                    if isinstance(margin_raw, (int, float, np.number))
                    else float(margin_raw.item())
                )
                total_spec_delay += spec_delay

                # Track worst margin
                if margin < worst_margin:
                    worst_margin = margin
                    critical_stage_idx = idx

        total_delay += measured_pd

        stages.append(
            ICStage(
                ic_name=ic_name,
                input_trace=input_trace,
                output_trace=output_trace,
                measured_delay=measured_pd,
                spec_delay=spec_delay,
                margin=margin,
            )
        )

    # Determine if path meets timing
    meets_timing = True
    for stage in stages:
        if stage.margin is not None and stage.margin < 0:
            meets_timing = False
            break

    # Check against target frequency if provided
    path_margin = None
    if target_frequency is not None:
        target_period = 1.0 / target_frequency
        path_margin = target_period - total_delay
        if path_margin < 0:
            meets_timing = False

    return TimingPathResult(
        stages=stages,
        total_delay=total_delay,
        total_spec_delay=total_spec_delay if total_spec_delay > 0 else None,
        critical_stage_idx=critical_stage_idx,
        path_margin=path_margin,
        meets_timing=meets_timing,
    )


def find_critical_paths(
    ic_graph: dict[str, list[tuple[str, WaveformTrace, WaveformTrace]]],
    *,
    start_node: str,
    end_node: str,
) -> list[TimingPathResult]:
    """Find all timing paths from start to end node.

    Args:
        ic_graph: Graph of IC connections.
        start_node: Starting IC name.
        end_node: Ending IC name.

    Returns:
        List of TimingPathResult objects sorted by total delay.
    """
    # This is a placeholder for a more sophisticated graph traversal
    # Would implement DFS/BFS to find all paths
    paths: list[TimingPathResult] = []
    return paths


def calculate_timing_budget(
    path_result: TimingPathResult,
    *,
    target_frequency: float,
    margin_target: float = 0.1,  # 10% margin
) -> dict[str, float]:
    """Calculate timing budget allocation for each stage.

    Args:
        path_result: Timing path analysis result.
        target_frequency: Target operating frequency.
        margin_target: Target margin fraction (0.0-1.0).

    Returns:
        Dictionary mapping stage index to allocated delay budget.

    Example:
        >>> budget = calculate_timing_budget(result, target_frequency=10e6)
        >>> print(f"Stage 0 budget: {budget[0]*1e9:.1f}ns")
    """
    target_period = 1.0 / target_frequency

    # Calculate available time (period minus desired margin)
    available_time = target_period * (1.0 - margin_target)

    # Allocate proportionally based on spec delays
    budget: dict[str, float] = {}

    total_spec = sum(s.spec_delay for s in path_result.stages if s.spec_delay is not None)

    if total_spec > 0:
        for idx, stage in enumerate(path_result.stages):
            if stage.spec_delay is not None:
                # Proportional allocation
                budget[str(idx)] = (stage.spec_delay / total_spec) * available_time
            else:
                # Equal allocation for unspecified stages
                budget[str(idx)] = available_time / len(path_result.stages)
    else:
        # Equal allocation if no specs available
        for idx in range(len(path_result.stages)):
            budget[str(idx)] = available_time / len(path_result.stages)

    return budget


@dataclass
class SetupHoldAnalysis:
    """Setup and hold time analysis for synchronous paths.

    Attributes:
        clock_period: Clock period in seconds.
        data_path_delay: Data path delay in seconds.
        clock_path_delay: Clock path delay in seconds.
        setup_time: Required setup time in seconds.
        hold_time: Required hold time in seconds.
        setup_slack: Setup time slack (positive = passes).
        hold_slack: Hold time slack (positive = passes).
        meets_setup: Whether setup time is met.
        meets_hold: Whether hold time is met.
    """

    clock_period: float
    data_path_delay: float
    clock_path_delay: float
    setup_time: float
    hold_time: float
    setup_slack: float
    hold_slack: float
    meets_setup: bool
    meets_hold: bool


def analyze_setup_hold(
    data_path: list[tuple[str, WaveformTrace, WaveformTrace]],
    clock_path: list[tuple[str, WaveformTrace, WaveformTrace]],
    *,
    clock_period: float,
    destination_ic: str,
) -> SetupHoldAnalysis:
    """Analyze setup and hold time for a synchronous path.

    Args:
        data_path: Data path IC chain.
        clock_path: Clock path IC chain.
        clock_period: Clock period in seconds.
        destination_ic: Destination IC part number.

    Returns:
        SetupHoldAnalysis object.

    Example:
        >>> analysis = analyze_setup_hold(
        ...     data_path=[(\"74LS00\", a, y)],
        ...     clock_path=[(\"CLK\", clk_in, clk_out)],
        ...     clock_period=100e-9,
        ...     destination_ic=\"74LS74\",
        ... )
        >>> if not analysis.meets_setup:
        ...     print(f\"Setup violation: {analysis.setup_slack*1e9:.1f}ns\")
    """
    from oscura.analyzers.digital.ic_database import IC_DATABASE

    # Analyze both paths
    data_result = analyze_timing_path(data_path)
    clock_result = analyze_timing_path(clock_path)

    # Get destination IC specs
    if destination_ic not in IC_DATABASE:
        raise ValueError(f"IC '{destination_ic}' not found in database")

    ic_spec = IC_DATABASE[destination_ic]
    setup_time = ic_spec.timing.get("t_su", 0.0)
    hold_time = ic_spec.timing.get("t_h", 0.0)

    # Calculate setup slack
    # Setup: data_delay + setup_time < clock_period + clock_delay
    setup_slack = (clock_period + clock_result.total_delay) - (data_result.total_delay + setup_time)
    meets_setup = setup_slack >= 0

    # Calculate hold slack
    # Hold: data_delay - clock_delay > hold_time
    hold_slack = data_result.total_delay - clock_result.total_delay - hold_time
    meets_hold = hold_slack >= 0

    return SetupHoldAnalysis(
        clock_period=clock_period,
        data_path_delay=data_result.total_delay,
        clock_path_delay=clock_result.total_delay,
        setup_time=setup_time,
        hold_time=hold_time,
        setup_slack=setup_slack,
        hold_slack=hold_slack,
        meets_setup=meets_setup,
        meets_hold=meets_hold,
    )


__all__ = [
    "ICStage",
    "SetupHoldAnalysis",
    "TimingPathResult",
    "analyze_setup_hold",
    "analyze_timing_path",
    "calculate_timing_budget",
    "find_critical_paths",
]
