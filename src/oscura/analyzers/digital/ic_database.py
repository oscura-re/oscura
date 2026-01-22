"""IC timing database for vintage and modern logic ICs.

This module provides timing specifications for common ICs to enable automatic
validation and identification.

Example:
    >>> from oscura.analyzers.digital.ic_database import IC_DATABASE, identify_ic
    >>> spec = IC_DATABASE["74LS74"]
    >>> print(f"Setup time: {spec.timing['t_su']*1e9:.1f}ns")
    >>>
    >>> # Auto-identify IC from measurements
    >>> ic_name, conf = identify_ic(measured_timings)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ICTiming:
    """Timing specification for an IC.

    Attributes:
        part_number: IC part number (e.g., "74LS74").
        description: Brief description of function.
        family: Logic family (e.g., "TTL", "LS-TTL", "HC-CMOS").
        vcc_nom: Nominal supply voltage.
        vcc_range: Supply voltage range (min, max).
        timing: Dictionary of timing parameters in seconds.
        voltage_levels: Dictionary of voltage thresholds.
    """

    part_number: str
    description: str
    family: str
    vcc_nom: float
    vcc_range: tuple[float, float]
    timing: dict[str, float] = field(default_factory=dict)
    voltage_levels: dict[str, float] = field(default_factory=dict)


# Timing parameter definitions:
# t_pd: Propagation delay
# t_su: Setup time
# t_h: Hold time
# t_w: Pulse width (minimum)
# t_r: Rise time
# t_f: Fall time
# t_co: Clock-to-output delay

# 74xx TTL Series (Standard TTL - 1970s era)
IC_7400_STD = ICTiming(
    part_number="7400",
    description="Quad 2-input NAND gate",
    family="TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 22e-9,  # Typ 22ns, max 33ns
        "t_r": 12e-9,
        "t_f": 8e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.4,
        "VOH_min": 2.4,
    },
)

IC_7474_STD = ICTiming(
    part_number="7474",
    description="Dual D-type flip-flop",
    family="TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 25e-9,  # Clock-to-output
        "t_su": 20e-9,  # Setup time
        "t_h": 5e-9,  # Hold time
        "t_w": 25e-9,  # Minimum clock pulse width
        "t_r": 12e-9,
        "t_f": 8e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.4,
        "VOH_min": 2.4,
    },
)

# 74LSxx Low-Power Schottky TTL Series (1970s-1980s)
IC_74LS00 = ICTiming(
    part_number="74LS00",
    description="Quad 2-input NAND gate",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 10e-9,  # Typ 10ns, max 15ns
        "t_r": 10e-9,
        "t_f": 7e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

IC_74LS74 = ICTiming(
    part_number="74LS74",
    description="Dual D-type flip-flop",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 25e-9,  # Clock-to-output (typ), max 40ns
        "t_su": 20e-9,  # Setup time
        "t_h": 5e-9,  # Hold time
        "t_w": 25e-9,  # Min clock pulse width
        "t_r": 10e-9,
        "t_f": 7e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

IC_74LS244 = ICTiming(
    part_number="74LS244",
    description="Octal buffer/line driver",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 12e-9,  # Typ 12ns, max 18ns
        "t_r": 7e-9,
        "t_f": 5e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

IC_74LS245 = ICTiming(
    part_number="74LS245",
    description="Octal bus transceiver",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 12e-9,  # Typ 12ns, max 18ns
        "t_r": 7e-9,
        "t_f": 5e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

IC_74LS138 = ICTiming(
    part_number="74LS138",
    description="3-to-8 line decoder/demultiplexer",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 21e-9,  # Typ 21ns, max 41ns
        "t_r": 10e-9,
        "t_f": 7e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

IC_74LS273 = ICTiming(
    part_number="74LS273",
    description="Octal D-type flip-flop with clear",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 20e-9,  # Clock-to-output
        "t_su": 20e-9,  # Setup time
        "t_h": 5e-9,  # Hold time
        "t_w": 20e-9,  # Min clock pulse width
        "t_r": 10e-9,
        "t_f": 7e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

IC_74LS374 = ICTiming(
    part_number="74LS374",
    description="Octal D-type flip-flop with 3-state outputs",
    family="LS-TTL",
    vcc_nom=5.0,
    vcc_range=(4.75, 5.25),
    timing={
        "t_pd": 20e-9,  # Clock-to-output
        "t_su": 20e-9,  # Setup time
        "t_h": 5e-9,  # Hold time
        "t_w": 20e-9,  # Min clock pulse width
        "t_pz": 18e-9,  # Output enable to output
        "t_pzh": 30e-9,  # Output disable to high-Z
        "t_r": 10e-9,
        "t_f": 7e-9,
    },
    voltage_levels={
        "VIL_max": 0.8,
        "VIH_min": 2.0,
        "VOL_max": 0.5,
        "VOH_min": 2.7,
    },
)

# 74HCxx High-Speed CMOS Series (1980s-present)
IC_74HC00 = ICTiming(
    part_number="74HC00",
    description="Quad 2-input NAND gate",
    family="HC-CMOS",
    vcc_nom=5.0,
    vcc_range=(2.0, 6.0),
    timing={
        "t_pd": 8e-9,  # At 5V, typ 8ns
        "t_r": 6e-9,
        "t_f": 6e-9,
    },
    voltage_levels={
        "VIL_max": 1.35,
        "VIH_min": 3.15,
        "VOL_max": 0.33,
        "VOH_min": 3.84,
    },
)

IC_74HC74 = ICTiming(
    part_number="74HC74",
    description="Dual D-type flip-flop",
    family="HC-CMOS",
    vcc_nom=5.0,
    vcc_range=(2.0, 6.0),
    timing={
        "t_pd": 16e-9,  # Clock-to-output at 5V
        "t_su": 14e-9,  # Setup time
        "t_h": 3e-9,  # Hold time
        "t_w": 14e-9,  # Min clock pulse width
        "t_r": 6e-9,
        "t_f": 6e-9,
    },
    voltage_levels={
        "VIL_max": 1.35,
        "VIH_min": 3.15,
        "VOL_max": 0.33,
        "VOH_min": 3.84,
    },
)

IC_74HC595 = ICTiming(
    part_number="74HC595",
    description="8-bit shift register with output latches",
    family="HC-CMOS",
    vcc_nom=5.0,
    vcc_range=(2.0, 6.0),
    timing={
        "t_pd": 16e-9,  # Clock-to-output
        "t_su": 14e-9,  # Setup time
        "t_h": 3e-9,  # Hold time
        "t_w": 14e-9,  # Min clock pulse width
        "t_r": 6e-9,
        "t_f": 6e-9,
    },
    voltage_levels={
        "VIL_max": 1.35,
        "VIH_min": 3.15,
        "VOL_max": 0.33,
        "VOH_min": 3.84,
    },
)

# 4000 Series CMOS (1970s-1980s)
IC_4011 = ICTiming(
    part_number="4011",
    description="Quad 2-input NAND gate",
    family="CMOS_5V",
    vcc_nom=5.0,
    vcc_range=(3.0, 18.0),
    timing={
        "t_pd": 90e-9,  # At 5V, typ 90ns
        "t_r": 60e-9,
        "t_f": 60e-9,
    },
    voltage_levels={
        "VIL_max": 1.5,
        "VIH_min": 3.5,
        "VOL_max": 0.05,
        "VOH_min": 4.95,
    },
)

IC_4013 = ICTiming(
    part_number="4013",
    description="Dual D-type flip-flop",
    family="CMOS_5V",
    vcc_nom=5.0,
    vcc_range=(3.0, 18.0),
    timing={
        "t_pd": 140e-9,  # Clock-to-output at 5V
        "t_su": 60e-9,  # Setup time
        "t_h": 40e-9,  # Hold time
        "t_w": 100e-9,  # Min clock pulse width
        "t_r": 60e-9,
        "t_f": 60e-9,
    },
    voltage_levels={
        "VIL_max": 1.5,
        "VIH_min": 3.5,
        "VOL_max": 0.05,
        "VOH_min": 4.95,
    },
)

# Comprehensive IC database
IC_DATABASE: dict[str, ICTiming] = {
    # Standard TTL
    "7400": IC_7400_STD,
    # LS-TTL (74LS series preferred over 74 for specificity)
    "74LS00": IC_74LS00,
    "74LS74": IC_74LS74,
    "74LS138": IC_74LS138,
    "74LS244": IC_74LS244,
    "74LS245": IC_74LS245,
    "74LS273": IC_74LS273,
    "74LS374": IC_74LS374,
    # HC-CMOS
    "74HC00": IC_74HC00,
    "74HC74": IC_74HC74,
    "74HC595": IC_74HC595,
    # 4000 series CMOS
    "4011": IC_4011,
    "4013": IC_4013,
}


def identify_ic(
    measured_timings: dict[str, float],
    *,
    tolerance: float = 0.5,
    min_confidence: float = 0.6,
) -> tuple[str, float]:
    """Identify IC from measured timing parameters.

    Args:
        measured_timings: Dictionary of measured timing values (e.g., {'t_pd': 25e-9}).
        tolerance: Allowable deviation (0.0-1.0, 0.5 = 50% tolerance).
        min_confidence: Minimum confidence score (0.0-1.0).

    Returns:
        Tuple of (ic_name, confidence_score).
        Returns ("unknown", 0.0) if no match above min_confidence.

    Example:
        >>> timings = {'t_pd': 25e-9, 't_su': 20e-9, 't_h': 5e-9}
        >>> ic, conf = identify_ic(timings)
        >>> print(f"Identified: {ic} ({conf*100:.1f}% confidence)")
    """
    scores: dict[str, float] = {}

    for ic_name, ic_spec in IC_DATABASE.items():
        # Calculate match score for this IC
        param_scores = []

        for param, measured_value in measured_timings.items():
            if param not in ic_spec.timing:
                continue

            spec_value = ic_spec.timing[param]

            # Calculate relative error
            if spec_value == 0:
                continue

            error = abs(measured_value - spec_value) / spec_value

            # Score based on error (within tolerance gets high score)
            if error <= tolerance:
                param_score = 1.0 - (error / tolerance)
            else:
                param_score = 0.0

            param_scores.append(param_score)

        # Overall score is average of parameter scores
        if param_scores:
            scores[ic_name] = sum(param_scores) / len(param_scores)

    # Find best match
    if not scores:
        return ("unknown", 0.0)

    best_ic = max(scores.items(), key=lambda x: x[1])

    if best_ic[1] < min_confidence:
        return ("unknown", best_ic[1])

    return best_ic


def validate_ic_timing(
    ic_name: str,
    measured_timings: dict[str, float],
    *,
    tolerance: float = 0.3,
) -> dict[str, dict[str, float | bool | None]]:
    """Validate measured timings against IC specification.

    Args:
        ic_name: IC part number (e.g., "74LS74").
        measured_timings: Dictionary of measured timing values.
        tolerance: Allowable deviation (0.0-1.0).

    Returns:
        Dictionary mapping parameter names to validation results:
        {'t_pd': {'measured': 25e-9, 'spec': 25e-9, 'passes': True, 'error': 0.0}}

    Raises:
        KeyError: If IC not found in database.

    Example:
        >>> results = validate_ic_timing("74LS74", {'t_pd': 30e-9})
        >>> if not results['t_pd']['passes']:
        ...     print(f"Propagation delay out of spec!")
    """
    if ic_name not in IC_DATABASE:
        raise KeyError(f"IC '{ic_name}' not found in database")

    ic_spec = IC_DATABASE[ic_name]
    results: dict[str, dict[str, float | bool | None]] = {}

    for param, measured_value in measured_timings.items():
        if param not in ic_spec.timing:
            results[param] = {
                "measured": measured_value,
                "spec": None,
                "passes": None,
                "error": None,
            }
            continue

        spec_value = ic_spec.timing[param]

        # Calculate relative error
        if spec_value == 0:
            error = 0.0
        else:
            error = abs(measured_value - spec_value) / spec_value

        # Check if within tolerance
        passes = error <= tolerance

        results[param] = {
            "measured": measured_value,
            "spec": spec_value,
            "passes": passes,
            "error": error,
        }

    return results


__all__ = [
    "IC_DATABASE",
    "ICTiming",
    "identify_ic",
    "validate_ic_timing",
]
