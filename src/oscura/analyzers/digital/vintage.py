"""High-level vintage logic analysis API.

This module provides a unified interface for complete vintage logic system analysis,
orchestrating all analysis steps and returning consolidated results.

Example:
    >>> from oscura.analyzers.digital.vintage import analyze_vintage_logic
    >>> result = analyze_vintage_logic(
    ...     traces={"CLK": clk_trace, "DATA": data_trace},
    ...     target_frequency=2e6
    ... )
    >>> print(f"Detected: {result.detected_family}")
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from oscura.analyzers.digital.extraction import (
    LOGIC_FAMILIES,
    detect_logic_family,
    detect_open_collector,
)
from oscura.analyzers.digital.ic_database import identify_ic, validate_ic_timing
from oscura.analyzers.digital.timing import hold_time, propagation_delay, setup_time
from oscura.analyzers.digital.timing_paths import TimingPathResult, analyze_timing_path
from oscura.analyzers.digital.vintage_result import (
    BOMEntry,
    ICIdentificationResult,
    ModernReplacementIC,
    VintageLogicAnalysisResult,
)
from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from oscura.core.types import DigitalTrace, WaveformTrace


# Modern IC replacement recommendations
REPLACEMENT_DATABASE: dict[str, ModernReplacementIC] = {
    "7400": ModernReplacementIC(
        original_ic="7400",
        replacement_ic="74HCT00",
        family="74HCTxx",
        benefits=["Lower power", "Wider voltage range", "TTL-compatible inputs"],
        notes="HCT family maintains TTL compatibility with CMOS benefits",
    ),
    "7474": ModernReplacementIC(
        original_ic="7474",
        replacement_ic="74HCT74",
        family="74HCTxx",
        benefits=["Lower power", "Wider voltage range", "TTL-compatible inputs"],
        notes="HCT family maintains TTL compatibility with CMOS benefits",
    ),
    "74LS00": ModernReplacementIC(
        original_ic="74LS00",
        replacement_ic="74HCT00",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
    "74LS74": ModernReplacementIC(
        original_ic="74LS74",
        replacement_ic="74HCT74",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
    "74LS138": ModernReplacementIC(
        original_ic="74LS138",
        replacement_ic="74HCT138",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
    "74LS244": ModernReplacementIC(
        original_ic="74LS244",
        replacement_ic="74HCT244",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
    "74LS245": ModernReplacementIC(
        original_ic="74LS245",
        replacement_ic="74HCT245",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
    "74LS273": ModernReplacementIC(
        original_ic="74LS273",
        replacement_ic="74HCT273",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
    "74LS374": ModernReplacementIC(
        original_ic="74LS374",
        replacement_ic="74HCT374",
        family="74HCTxx",
        benefits=["Lower power", "Similar speed", "Better availability"],
        notes="Direct pin-compatible replacement",
    ),
}


def analyze_vintage_logic(
    traces: dict[str, WaveformTrace | DigitalTrace],
    *,
    target_frequency: float | None = None,
    system_description: str | None = None,
    enable_protocol_decode: bool = False,
    timing_paths: list[tuple[str, WaveformTrace, WaveformTrace]] | None = None,
) -> VintageLogicAnalysisResult:
    """Complete vintage logic system analysis.

    High-level API that orchestrates all vintage logic analysis steps and returns
    a comprehensive result object suitable for reporting and export.

    Args:
        traces: Dictionary mapping channel names to traces.
        target_frequency: Target system clock frequency in Hz.
        system_description: Optional description for documentation.
        enable_protocol_decode: Enable automatic protocol decoding (GPIB, ISA, etc.).
        timing_paths: Optional list of (ic_name, input_trace, output_trace) tuples
            for multi-IC timing path analysis.

    Returns:
        VintageLogicAnalysisResult object with complete analysis data.

    Example:
        >>> result = analyze_vintage_logic(
        ...     traces={"CLK": clk_trace, "DATA": data_trace},
        ...     target_frequency=2e6,
        ...     system_description="1976 Microcomputer System"
        ... )
        >>> print(f"Detected: {result.detected_family}")
        >>> print(f"ICs: {[ic.ic_name for ic in result.identified_ics]}")
    """
    start_time = time.time()
    warnings: list[str] = []
    confidence_scores: dict[str, float] = {}

    # Use first trace for logic family detection
    first_trace_name = next(iter(traces.keys()))
    first_trace = traces[first_trace_name]

    # Ensure we have a WaveformTrace for voltage-based analysis
    if not hasattr(first_trace.data, "dtype") or first_trace.data.dtype == bool:
        warnings.append(
            f"Digital trace provided for {first_trace_name}, "
            "logic family detection may be inaccurate"
        )
        detected_family = "unknown"
        family_confidence = 0.0
        voltage_levels = {}
    else:
        # Step 1: Detect logic family
        # Type narrowing: we've verified this is a WaveformTrace above
        waveform_trace = cast("WaveformTrace", first_trace)
        detected_family, family_confidence = detect_logic_family(waveform_trace)
        confidence_scores["logic_family"] = family_confidence

        # Get voltage levels from detected family
        if detected_family in LOGIC_FAMILIES:
            voltage_levels = dict(LOGIC_FAMILIES[detected_family])
        else:
            voltage_levels = {}
            warnings.append(f"Unknown logic family: {detected_family}")

    # Step 2: Detect open-collector outputs
    open_collector_detected = False
    asymmetry_ratio = 1.0
    if hasattr(first_trace.data, "dtype") and first_trace.data.dtype != bool:
        waveform_trace = cast("WaveformTrace", first_trace)
        open_collector_detected, asymmetry_ratio = detect_open_collector(waveform_trace)
        if open_collector_detected:
            warnings.append(
                "Open-collector output detected - consider 10kΩ pull-up in modern design"
            )

    # Step 3: Measure timing parameters and identify ICs
    identified_ics: list[ICIdentificationResult] = []
    timing_measurements: dict[str, float] = {}

    # Analyze each trace pair for timing
    trace_list = list(traces.items())
    for i in range(len(trace_list) - 1):
        name1, trace1 = trace_list[i]
        name2, trace2 = trace_list[i + 1]

        # Measure propagation delay
        try:
            t_pd_raw = propagation_delay(trace1, trace2)
            t_pd = (
                float(t_pd_raw)
                if isinstance(t_pd_raw, (int, float, np.number))
                else float(t_pd_raw.item())
            )
            if t_pd > 0:
                timing_measurements[f"{name1}→{name2}_t_pd"] = t_pd
        except Exception as e:
            warnings.append(f"Failed to measure propagation delay {name1}→{name2}: {e}")

        # Measure setup and hold times if clock-like signal
        try:
            t_su_raw = setup_time(trace1, trace2)
            t_su = (
                float(t_su_raw)
                if isinstance(t_su_raw, (int, float, np.number))
                else float(t_su_raw.item())
            )
            if t_su > 0:
                timing_measurements[f"{name2}_t_su"] = t_su
        except Exception:
            pass  # Optional measurement

        try:
            t_h_raw = hold_time(trace1, trace2)
            t_h = (
                float(t_h_raw)
                if isinstance(t_h_raw, (int, float, np.number))
                else float(t_h_raw.item())
            )
            if t_h > 0:
                timing_measurements[f"{name2}_t_h"] = t_h
        except Exception:
            pass  # Optional measurement

    # Attempt IC identification from timing measurements
    if timing_measurements:
        # Extract core timing parameters
        core_params = {}
        for key, value in timing_measurements.items():
            if "_t_pd" in key:
                core_params["t_pd"] = value
            elif "_t_su" in key and "t_su" not in core_params:
                core_params["t_su"] = value
            elif "_t_h" in key and "t_h" not in core_params:
                core_params["t_h"] = value

        if core_params:
            ic_name, ic_confidence = identify_ic(core_params, tolerance=0.5)
            confidence_scores["ic_identification"] = ic_confidence

            if ic_name != "unknown":
                # Validate against database
                try:
                    validation = validate_ic_timing(ic_name, core_params, tolerance=0.3)

                    identified_ics.append(
                        ICIdentificationResult(
                            ic_name=ic_name,
                            confidence=ic_confidence,
                            timing_params=core_params,
                            validation=validation,
                            family=detected_family,
                        )
                    )

                    # Check if any parameters fail validation
                    failed_params = [k for k, v in validation.items() if v.get("passes") is False]
                    if failed_params:
                        warnings.append(
                            f"IC {ic_name} timing validation failed for: {', '.join(failed_params)}"
                        )
                except KeyError:
                    warnings.append(f"IC {ic_name} not found in database")

    # Step 4: Analyze timing paths if provided
    timing_path_results: list[TimingPathResult] | None = None
    if timing_paths:
        try:
            path_result = analyze_timing_path(timing_paths, target_frequency=target_frequency)
            timing_path_results = [path_result]

            if not path_result.meets_timing:
                warnings.append(
                    f"Timing path violation detected at stage {path_result.critical_stage_idx}"
                )
        except Exception as e:
            warnings.append(f"Timing path analysis failed: {e}")

    # Step 5: Protocol decoding (if enabled)
    decoded_protocols: dict[str, list[Any]] | None = None
    if enable_protocol_decode:
        # Protocol decoding would be implemented here
        # For now, just a placeholder
        decoded_protocols = {}

    # Step 6: Generate modern replacement recommendations
    modern_replacements: list[ModernReplacementIC] = []
    for ic_result in identified_ics:
        if ic_result.ic_name in REPLACEMENT_DATABASE:
            modern_replacements.append(REPLACEMENT_DATABASE[ic_result.ic_name])

    # Step 7: Generate BOM
    bom: list[BOMEntry] = []

    # Add identified ICs
    for ic_result in identified_ics:
        bom.append(
            BOMEntry(
                part_number=ic_result.ic_name,
                description=f"Original IC from {detected_family} family",
                quantity=1,
                category="IC",
                notes=f"Confidence: {ic_result.confidence * 100:.1f}%",
            )
        )

    # Add modern replacements
    for replacement in modern_replacements:
        bom.append(
            BOMEntry(
                part_number=replacement.replacement_ic,
                description=f"Modern replacement for {replacement.original_ic}",
                quantity=1,
                category="IC",
                notes=f"Benefits: {', '.join(replacement.benefits)}",
            )
        )

    # Add supporting components
    if open_collector_detected:
        bom.append(
            BOMEntry(
                part_number="Pull-up resistor",
                description="10kΩ resistor for open-collector pull-up",
                quantity=len(list(traces.values())),
                category="Resistor",
                notes="Use for all open-collector outputs",
            )
        )

    # Add decoupling capacitors (good practice)
    bom.append(
        BOMEntry(
            part_number="0.1µF ceramic capacitor",
            description="Decoupling capacitor",
            quantity=len(identified_ics) * 2,
            category="Capacitor",
            notes="One per IC, place close to VCC/GND pins",
        )
    )

    # Calculate analysis duration
    analysis_duration = time.time() - start_time

    return VintageLogicAnalysisResult(
        timestamp=datetime.now(),
        source_file=system_description,
        analysis_duration=analysis_duration,
        detected_family=detected_family,
        family_confidence=family_confidence,
        voltage_levels=voltage_levels,
        identified_ics=identified_ics,
        timing_measurements=timing_measurements,
        timing_paths=timing_path_results,
        decoded_protocols=decoded_protocols,
        open_collector_detected=open_collector_detected,
        asymmetry_ratio=asymmetry_ratio,
        modern_replacements=modern_replacements,
        bom=bom,
        warnings=warnings,
        confidence_scores=confidence_scores,
    )


__all__ = [
    "REPLACEMENT_DATABASE",
    "analyze_vintage_logic",
]
