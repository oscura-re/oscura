"""Bit error pattern analysis and capture diagnostics.


This module characterizes bit error patterns to diagnose capture quality
issues (EMI, USB problems, clock jitter) and suggests likely causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ErrorPattern(Enum):
    """Classified error pattern types.

    Attributes:
        RANDOM: Errors uniformly distributed, no clustering (likely EMI)
        BURST: Errors clustered together (likely USB transmission issue)
        PERIODIC: Errors repeat at regular intervals (likely clock jitter)
        UNKNOWN: Pattern doesn't match known types
    """

    RANDOM = "random"
    BURST = "burst"
    PERIODIC = "periodic"
    UNKNOWN = "unknown"


@dataclass
class ErrorAnalysis:
    """Result from bit error pattern analysis.

    Attributes:
        bit_error_rate: Ratio of errors to total bits
        error_count: Total number of bit errors detected
        total_bits: Total bits examined
        pattern_type: Classified error pattern (random, burst, periodic)
        mean_error_gap: Mean number of bits between errors
        error_positions: Array of bit positions where errors occurred
        diagnosis: Suggested cause based on pattern
        severity: Error severity level (low, moderate, severe)
    """

    bit_error_rate: float
    error_count: int
    total_bits: int
    pattern_type: ErrorPattern
    mean_error_gap: float
    error_positions: NDArray[np.int64]
    diagnosis: str
    severity: str


def analyze_bit_errors(
    received: NDArray[np.uint8],
    expected: NDArray[np.uint8],
    *,
    burst_threshold: int = 100,
    periodicity_threshold: float = 0.1,
) -> ErrorAnalysis:
    """Characterize bit error patterns for capture diagnostics.

    : Analyzes bit errors to diagnose capture quality
    issues and distinguish between EMI, USB problems, and clock jitter.

    Error pattern classification (DAQ-005):
    - Random: Errors uniformly distributed, no clustering
    - Burst: Errors clustered, mean_gap < 100 bits
    - Periodic: Errors repeat at regular intervals (FFT peak in positions)

    Diagnosis suggestions (DAQ-005):
    - BER > 0.01: Severe capture issue, check connections
    - BER 0.001-0.01: Moderate errors, reduce sample rate
    - BER < 0.001: Acceptable, likely EMI
    - Burst errors: USB transmission issue
    - Periodic errors: Clock jitter or interference

    Args:
        received: Received bit array (actual capture)
        expected: Expected bit array (golden reference)
        burst_threshold: Mean gap threshold for burst classification
        periodicity_threshold: FFT peak threshold for periodic detection

    Returns:
        ErrorAnalysis with BER, pattern type, and diagnosis

    Raises:
        ValueError: If received and expected have different lengths
        ValueError: If arrays are empty

    Examples:
        >>> # Analyze random EMI errors
        >>> import numpy as np
        >>> expected = np.random.randint(0, 2, 10000, dtype=np.uint8)
        >>> received = expected.copy()
        >>> errors = np.random.choice(10000, 50, replace=False)
        >>> received[errors] = 1 - received[errors]  # Flip bits
        >>> analysis = analyze_bit_errors(received, expected)
        >>> print(f"BER: {analysis.bit_error_rate:.6f}")
        >>> print(f"Pattern: {analysis.pattern_type.value}")

        >>> # Analyze burst errors (USB issue)
        >>> received = expected.copy()
        >>> received[1000:1050] = 1 - received[1000:1050]  # 50-bit burst
        >>> analysis = analyze_bit_errors(received, expected)
        >>> print(analysis.diagnosis)
        'USB transmission issue'

    References:
        DAQ-005: Bit Error Pattern Analysis and Capture Diagnostics
    """
    if len(received) != len(expected):
        raise ValueError("Received and expected arrays must have same length")

    if len(received) == 0:
        raise ValueError("Arrays cannot be empty")

    # Find bit errors (XOR)
    errors = received != expected
    error_positions = np.where(errors)[0]
    error_count = len(error_positions)
    total_bits = len(received)

    # Calculate BER
    bit_error_rate = error_count / total_bits if total_bits > 0 else 0.0

    if error_count == 0:
        # No errors
        return ErrorAnalysis(
            bit_error_rate=0.0,
            error_count=0,
            total_bits=total_bits,
            pattern_type=ErrorPattern.RANDOM,
            mean_error_gap=float(total_bits),
            error_positions=error_positions,
            diagnosis="No errors detected - good capture quality",
            severity="low",
        )

    # Calculate error gaps
    if error_count > 1:
        error_gaps = np.diff(error_positions)
        mean_gap = float(np.mean(error_gaps))
    else:
        mean_gap = float(total_bits)

    # Classify error pattern
    pattern_type = ErrorPattern.UNKNOWN
    diagnosis = ""

    # Check for burst pattern (errors clustered)
    if error_count > 1 and mean_gap < burst_threshold:
        pattern_type = ErrorPattern.BURST
        diagnosis = "Burst errors detected - likely USB transmission issue"

    # Check for periodic pattern (FFT analysis)
    # Need at least 10 errors for reliable periodicity detection
    elif error_count >= 10:
        # Create binary error signal
        error_signal = errors.astype(float)

        # Compute FFT to detect periodicity
        fft = np.fft.rfft(error_signal)
        fft_mag = np.abs(fft[1:])  # Skip DC component

        if len(fft_mag) > 0:
            # Check if there's a strong peak relative to mean (not just max)
            mean_mag = np.mean(fft_mag)
            max_mag = np.max(fft_mag)
            peak_ratio = max_mag / (mean_mag + 1e-12)

            # Require strong peak (>10x mean) and exceeds threshold
            if peak_ratio > 10 and (max_mag / (np.max(fft_mag) + 1e-12)) > periodicity_threshold:
                pattern_type = ErrorPattern.PERIODIC
                diagnosis = "Periodic errors detected - likely clock jitter or interference"

    # If not burst or periodic, classify as random
    if pattern_type == ErrorPattern.UNKNOWN:
        # Check if errors are uniformly distributed
        if error_count > 2:
            # Use coefficient of variation of gaps
            if error_count > 1:
                gap_std = float(np.std(error_gaps))
                gap_cv = gap_std / (mean_gap + 1e-12)

                if gap_cv < 1.0:  # Relatively uniform spacing
                    pattern_type = ErrorPattern.RANDOM
                    diagnosis = "Random errors detected - likely EMI or noise"
                else:
                    diagnosis = "Mixed error pattern - multiple causes possible"
            else:
                pattern_type = ErrorPattern.RANDOM
                diagnosis = "Single error - insufficient data for classification"
        else:
            pattern_type = ErrorPattern.RANDOM
            diagnosis = "Few errors - likely random EMI or noise"

    # Determine severity based on BER
    if bit_error_rate > 0.01:
        severity = "severe"
        diagnosis += ". SEVERE: Check connections and hardware"
    elif bit_error_rate > 0.001:
        severity = "moderate"
        diagnosis += ". MODERATE: Consider reducing sample rate"
    else:
        severity = "low"
        diagnosis += ". Acceptable error rate"

    return ErrorAnalysis(
        bit_error_rate=bit_error_rate,
        error_count=error_count,
        total_bits=total_bits,
        pattern_type=pattern_type,
        mean_error_gap=mean_gap,
        error_positions=error_positions,
        diagnosis=diagnosis,
        severity=severity,
    )


def generate_error_visualization_data(
    analysis: ErrorAnalysis,
    *,
    histogram_bins: int = 50,
) -> dict[str, NDArray[np.float64]]:
    """Generate data for error distribution visualization.

    Creates histogram and timeline data suitable for plotting error patterns.

    Args:
        analysis: ErrorAnalysis result from analyze_bit_errors()
        histogram_bins: Number of bins for error position histogram

    Returns:
        Dictionary with 'histogram_counts', 'histogram_edges', and
        'timeline' arrays for visualization

    Examples:
        >>> # Generate visualization data
        >>> analysis = analyze_bit_errors(received, expected)
        >>> viz_data = generate_error_visualization_data(analysis)
        >>> # Plot with matplotlib
        >>> import matplotlib.pyplot as plt
        >>> plt.hist(analysis.error_positions, bins=viz_data['histogram_edges'])
        >>> plt.xlabel('Bit Position')
        >>> plt.ylabel('Error Count')
        >>> plt.show()

    References:
        DAQ-005: Bit Error Pattern Analysis and Capture Diagnostics
    """
    if len(analysis.error_positions) == 0:
        # No errors - return empty data
        return {
            "histogram_counts": np.array([], dtype=np.float64),
            "histogram_edges": np.array([], dtype=np.float64),
            "timeline": np.array([], dtype=np.float64),
        }

    # Generate histogram over full bit range
    counts, edges = np.histogram(
        analysis.error_positions, bins=histogram_bins, range=(0, analysis.total_bits), density=False
    )

    # Timeline: binary array with 1s at error positions
    timeline = np.zeros(analysis.total_bits, dtype=np.float64)
    timeline[analysis.error_positions] = 1.0

    return {
        "histogram_counts": counts.astype(np.float64),
        "histogram_edges": edges.astype(np.float64),
        "timeline": timeline,
    }
