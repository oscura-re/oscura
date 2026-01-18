"""Signal quality analysis for digital signals.

This module provides signal quality metrics including noise margin
calculation, setup/hold violation detection, and glitch detection.


Example:
    >>> from oscura.analyzers.digital.quality import noise_margin, detect_glitches
    >>> margins = noise_margin(trace, family="TTL")
    >>> glitches = detect_glitches(trace, min_width=10e-9)

References:
    JEDEC Standard No. 8C: High-Speed CMOS Interface
    Various IC manufacturer datasheets for logic family specifications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.analyzers.digital.extraction import LOGIC_FAMILIES
from oscura.analyzers.digital.timing import hold_time, setup_time
from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


@dataclass
class NoiseMarginResult:
    """Result of noise margin calculation.

    Attributes:
        nm_high: Noise margin high (VOH_min - VIH_min).
        nm_low: Noise margin low (VIL_max - VOL_max).
        logic_family: Logic family used for calculation.
        voh: Output high voltage (measured or spec).
        vol: Output low voltage (measured or spec).
        vih: Input high threshold (from spec).
        vil: Input low threshold (from spec).
    """

    nm_high: float
    nm_low: float
    logic_family: str
    voh: float
    vol: float
    vih: float
    vil: float


@dataclass
class Violation:
    """Represents a timing or signal violation.

    Attributes:
        timestamp: Time of violation in seconds.
        violation_type: Type of violation.
        measured: Measured value.
        limit: Specification limit.
        margin: Margin to specification (negative = violation).
        end_timestamp: End time of violation (if applicable).
    """

    timestamp: float
    violation_type: str
    measured: float
    limit: float
    margin: float
    end_timestamp: float | None = None


@dataclass
class Glitch:
    """Represents a detected glitch.

    Attributes:
        timestamp: Start time of glitch in seconds.
        width: Duration of glitch in seconds.
        polarity: "positive" (spike high) or "negative" (spike low).
        amplitude: Peak amplitude of glitch.
    """

    timestamp: float
    width: float
    polarity: Literal["positive", "negative"]
    amplitude: float


def noise_margin(
    trace: WaveformTrace,
    *,
    family: str = "LVCMOS_3V3",
    use_measured_levels: bool = True,
) -> NoiseMarginResult:
    """Calculate noise margins for a digital signal.

    Computes noise margin high (NMH) and noise margin low (NML) based on
    measured signal levels or logic family specifications.

    Args:
        trace: Input waveform trace.
        family: Logic family for threshold levels.
            Options: TTL, CMOS_5V, LVTTL, LVCMOS_3V3, LVCMOS_2V5, LVCMOS_1V8, LVCMOS_1V2
        use_measured_levels: If True, use measured VOH/VOL from signal.
            If False, use spec values from logic family.

    Returns:
        NoiseMarginResult with calculated margins.

    Raises:
        ValueError: If logic family is not recognized.

    Example:
        >>> result = noise_margin(trace, family="TTL")
        >>> print(f"NMH: {result.nm_high:.3f} V")
        >>> print(f"NML: {result.nm_low:.3f} V")

    References:
        JEDEC Standard No. 8C
    """
    if family not in LOGIC_FAMILIES:
        available = ", ".join(LOGIC_FAMILIES.keys())
        raise ValueError(f"Unknown logic family: {family}. Available: {available}")

    specs = LOGIC_FAMILIES[family]
    vih = specs["VIH_min"]
    vil = specs["VIL_max"]

    if use_measured_levels and len(trace.data) > 0:
        # Measure actual output levels from signal
        data = trace.data
        low, high = _find_logic_levels(data)
        voh = high
        vol = low
    else:
        # Use specification values
        voh = specs["VOH_min"]
        vol = specs["VOL_max"]

    # Calculate noise margins
    # NMH = VOH - VIH (margin when output is high)
    # NML = VIL - VOL (margin when output is low)
    nm_high = voh - vih
    nm_low = vil - vol

    return NoiseMarginResult(
        nm_high=nm_high,
        nm_low=nm_low,
        logic_family=family,
        voh=voh,
        vol=vol,
        vih=vih,
        vil=vil,
    )


def detect_violations(
    data_trace: WaveformTrace | DigitalTrace,
    clock_trace: WaveformTrace | DigitalTrace,
    *,
    setup_spec: float,
    hold_spec: float,
    clock_edge: Literal["rising", "falling"] = "rising",
) -> list[Violation]:
    """Detect setup and hold time violations.

    Compares measured setup and hold times to specifications and
    reports any violations with timestamps and margins.

    Args:
        data_trace: Data signal trace.
        clock_trace: Clock signal trace.
        setup_spec: Required setup time in seconds.
        hold_spec: Required hold time in seconds.
        clock_edge: Clock edge to reference ("rising" or "falling").

    Returns:
        List of Violation objects for each detected violation.

    Example:
        >>> violations = detect_violations(
        ...     data_trace, clock_trace,
        ...     setup_spec=2e-9, hold_spec=1e-9
        ... )
        >>> for v in violations:
        ...     print(f"{v.violation_type}: {v.margin*1e12:.0f} ps margin")

    References:
        JEDEC Standard No. 65B
    """
    violations: list[Violation] = []

    # Get all setup times
    setup_times = setup_time(data_trace, clock_trace, clock_edge=clock_edge, return_all=True)

    if isinstance(setup_times, np.ndarray) and len(setup_times) > 0:
        clock_edges = _get_clock_edges(clock_trace, clock_edge)

        for _i, (t_setup, clk_edge) in enumerate(
            zip(setup_times, clock_edges[: len(setup_times)], strict=False)
        ):
            margin = t_setup - setup_spec
            if margin < 0:  # Violation
                violations.append(
                    Violation(
                        timestamp=clk_edge,
                        violation_type="setup",
                        measured=t_setup,
                        limit=setup_spec,
                        margin=margin,
                    )
                )

    # Get all hold times
    hold_times = hold_time(data_trace, clock_trace, clock_edge=clock_edge, return_all=True)

    if isinstance(hold_times, np.ndarray) and len(hold_times) > 0:
        clock_edges = _get_clock_edges(clock_trace, clock_edge)

        for _i, (t_hold, clk_edge) in enumerate(
            zip(hold_times, clock_edges[: len(hold_times)], strict=False)
        ):
            margin = t_hold - hold_spec
            if margin < 0:  # Violation
                violations.append(
                    Violation(
                        timestamp=clk_edge,
                        violation_type="hold",
                        measured=t_hold,
                        limit=hold_spec,
                        margin=margin,
                    )
                )

    # Sort by timestamp
    violations.sort(key=lambda v: v.timestamp)

    return violations


def detect_glitches(
    trace: WaveformTrace | DigitalTrace,
    *,
    min_width: float,
    threshold: float | None = None,
) -> list[Glitch]:
    """Detect glitches (pulses shorter than minimum width).

    Identifies short pulses that violate minimum pulse width specifications,
    which may cause logic errors or be artifacts.

    Args:
        trace: Input trace (analog or digital).
        min_width: Minimum valid pulse width in seconds.
        threshold: Threshold for digital conversion (analog traces only).
            If None, auto-detected from signal.

    Returns:
        List of Glitch objects for each detected glitch.

    Example:
        >>> glitches = detect_glitches(trace, min_width=10e-9)
        >>> for g in glitches:
        ...     print(f"Glitch at {g.timestamp*1e6:.2f} us, width={g.width*1e9:.1f} ns")

    References:
        Application note AN-905: Understanding Glitch Detection
    """
    if isinstance(trace, DigitalTrace):
        # Already digital - use directly
        digital = trace.data
        sample_rate = trace.metadata.sample_rate
        threshold_used = 0.5  # Not used for amplitude calc on digital
        data = trace.data.astype(np.float64)
    else:
        # Analog trace - need to threshold
        data = trace.data
        sample_rate = trace.metadata.sample_rate

        if len(data) < 3:
            return []

        # Find threshold
        low, high = _find_logic_levels(data)
        threshold_used = (low + high) / 2 if threshold is None else threshold

        amplitude = high - low
        if amplitude <= 0:
            return []

        # Convert to binary
        digital = data >= threshold_used

    if len(digital) < 3:
        return []

    sample_period = 1.0 / sample_rate

    glitches: list[Glitch] = []

    # Find all pulse edges
    transitions = np.diff(digital.astype(np.int8))
    rising_edges = np.where(transitions == 1)[0]
    falling_edges = np.where(transitions == -1)[0]

    # Check positive pulses (rising to falling)
    for rising_idx in rising_edges:
        # Find next falling edge
        subsequent_falling = falling_edges[falling_edges > rising_idx]
        if len(subsequent_falling) > 0:
            falling_idx = subsequent_falling[0]
            width = (falling_idx - rising_idx) * sample_period

            if width < min_width:
                # Calculate amplitude within pulse
                pulse_data = data[rising_idx : falling_idx + 1]
                if isinstance(trace, DigitalTrace):
                    # For digital trace, amplitude is just 1.0 (logic high)
                    pulse_amplitude = 1.0
                else:
                    pulse_amplitude = float(np.max(pulse_data) - threshold_used)

                glitches.append(
                    Glitch(
                        timestamp=rising_idx * sample_period,
                        width=width,
                        polarity="positive",
                        amplitude=pulse_amplitude,
                    )
                )

    # Check negative pulses (falling to rising)
    for falling_idx in falling_edges:
        # Find next rising edge
        subsequent_rising = rising_edges[rising_edges > falling_idx]
        if len(subsequent_rising) > 0:
            rising_idx = subsequent_rising[0]
            width = (rising_idx - falling_idx) * sample_period

            if width < min_width:
                # Calculate amplitude within pulse
                pulse_data = data[falling_idx : rising_idx + 1]
                if isinstance(trace, DigitalTrace):
                    # For digital trace, amplitude is just 1.0 (logic low)
                    pulse_amplitude = 1.0
                else:
                    pulse_amplitude = float(threshold_used - np.min(pulse_data))

                glitches.append(
                    Glitch(
                        timestamp=falling_idx * sample_period,
                        width=width,
                        polarity="negative",
                        amplitude=pulse_amplitude,
                    )
                )

    # Sort by timestamp
    glitches.sort(key=lambda g: g.timestamp)

    return glitches


def signal_quality_summary(
    trace: WaveformTrace,
    *,
    family: str = "LVCMOS_3V3",
    min_pulse_width: float = 10e-9,
) -> dict:  # type: ignore[type-arg]
    """Generate comprehensive signal quality summary.

    Combines multiple quality metrics into a single report.

    Args:
        trace: Input waveform trace.
        family: Logic family for noise margin calculation.
        min_pulse_width: Minimum pulse width for glitch detection.

    Returns:
        Dictionary with quality metrics:
            - noise_margin: NoiseMarginResult
            - glitch_count: Number of detected glitches
            - glitches: List of Glitch objects
            - signal_levels: Measured low/high levels
            - transition_count: Number of transitions

    Example:
        >>> summary = signal_quality_summary(trace)
        >>> print(f"NMH: {summary['noise_margin'].nm_high:.3f} V")
        >>> print(f"Glitches: {summary['glitch_count']}")
    """
    # Noise margin analysis
    nm_result = noise_margin(trace, family=family)

    # Glitch detection
    glitches = detect_glitches(trace, min_width=min_pulse_width)

    # Signal levels
    low, high = _find_logic_levels(trace.data)

    # Transition count
    threshold = (low + high) / 2
    digital = trace.data >= threshold
    transitions = np.sum(np.abs(np.diff(digital.astype(np.int8))))

    return {
        "noise_margin": nm_result,
        "glitch_count": len(glitches),
        "glitches": glitches,
        "signal_levels": {"low": low, "high": high},
        "transition_count": int(transitions),
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _find_logic_levels(data: NDArray[np.floating[Any]]) -> tuple[float, float]:
    """Find low and high logic levels from signal data.

    Uses histogram analysis to identify stable high and low levels.

    Args:
        data: Waveform data array.

    Returns:
        Tuple of (low_level, high_level).
    """
    if len(data) == 0:
        return 0.0, 0.0

    # Use percentiles for robust level detection
    p10, p90 = np.percentile(data, [10, 90])

    try:
        # Refine using histogram peaks
        hist, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in lower and upper halves
        mid_idx = len(hist) // 2
        low_idx = np.argmax(hist[:mid_idx])
        high_idx = mid_idx + np.argmax(hist[mid_idx:])

        low = bin_centers[low_idx]
        high = bin_centers[high_idx]

        # Sanity check
        if high <= low:
            return float(p10), float(p90)

        return float(low), float(high)
    except (ValueError, IndexError):
        return float(p10), float(p90)


def _get_clock_edges(
    trace: WaveformTrace | DigitalTrace,
    edge_type: Literal["rising", "falling"],
) -> NDArray[np.float64]:
    """Get clock edge timestamps.

    Args:
        trace: Clock trace.
        edge_type: Type of edges to find.

    Returns:
        Array of edge timestamps in seconds.
    """
    data = trace.data.astype(np.float64) if isinstance(trace, DigitalTrace) else trace.data

    if len(data) < 2:
        return np.array([], dtype=np.float64)

    sample_period = trace.metadata.time_base

    # Find threshold
    low, high = _find_logic_levels(data)
    threshold = (low + high) / 2

    if edge_type == "rising":
        crossings = np.where((data[:-1] < threshold) & (data[1:] >= threshold))[0]
    else:
        crossings = np.where((data[:-1] >= threshold) & (data[1:] < threshold))[0]

    # Convert to timestamps with interpolation
    timestamps = np.zeros(len(crossings), dtype=np.float64)

    for i, idx in enumerate(crossings):
        base_time = idx * sample_period
        if idx < len(data) - 1:
            v1, v2 = data[idx], data[idx + 1]
            if abs(v2 - v1) > 1e-12:
                t_offset = (threshold - v1) / (v2 - v1) * sample_period
                t_offset = max(0, min(sample_period, t_offset))
                timestamps[i] = base_time + t_offset
            else:
                timestamps[i] = base_time + sample_period / 2
        else:
            timestamps[i] = base_time

    return timestamps


@dataclass
class MaskTestResult:
    """Result of mask testing.

    Attributes:
        pass_fail: Overall pass/fail result.
        hit_count: Number of samples violating the mask.
        total_samples: Total number of samples tested.
        margin_top: Margin to top mask boundary in volts (minimum).
        margin_bottom: Margin to bottom mask boundary in volts (minimum).
        violations: List of violation timestamps and amplitudes.
    """

    pass_fail: bool
    hit_count: int
    total_samples: int
    margin_top: float
    margin_bottom: float
    violations: list[tuple[float, float]]  # (time, voltage) pairs


@dataclass
class PLLRecoveryResult:
    """Result of PLL clock recovery.

    Attributes:
        recovered_frequency: Recovered clock frequency in Hz.
        recovered_phase: Recovered phase trajectory (radians).
        vco_control: VCO control voltage trajectory.
        lock_status: True if PLL is locked.
        lock_time: Time to achieve lock in seconds (if locked).
        frequency_error: Final frequency error in Hz.
    """

    recovered_frequency: float
    recovered_phase: NDArray[np.float64]
    vco_control: NDArray[np.float64]
    lock_status: bool
    lock_time: float | None
    frequency_error: float


def mask_test(
    trace: WaveformTrace,
    mask: dict[str, NDArray[np.float64]] | str = "usb2",
    *,
    bit_period: float | None = None,
) -> MaskTestResult:
    """Test signal against compliance mask template.

    Performs mask testing for signal integrity verification against
    predefined templates (USB, PCIe, etc.) or custom masks.

    Args:
        trace: Input waveform trace.
        mask: Either mask name ("usb2", "pcie_gen3") or custom mask dict with:
            - "time_ui": Time coordinates in UI (0.0 to 2.0 for 2-UI mask)
            - "voltage_top": Upper voltage boundary
            - "voltage_bottom": Lower voltage boundary
        bit_period: Bit period in seconds (required if mask uses UI coordinates).

    Returns:
        MaskTestResult with pass/fail and violation statistics.

    Raises:
        ValueError: If bit_period is not provided or mask name is not recognized.

    Example:
        >>> result = mask_test(signal_trace, mask="usb2", bit_period=3.33e-9)
        >>> print(f"Pass: {result.pass_fail}, Violations: {result.hit_count}")

    References:
        USB 2.0 Specification, PCIe Base Specification
    """
    # Load predefined mask if string
    mask_data = _get_predefined_mask(mask) if isinstance(mask, str) else mask

    # Extract mask boundaries
    time_ui = mask_data["time_ui"]
    v_top = mask_data["voltage_top"]
    v_bottom = mask_data["voltage_bottom"]

    # Get signal data
    data = trace.data
    n_samples = len(data)
    sample_rate = trace.metadata.sample_rate

    if bit_period is None:
        raise ValueError("bit_period is required for mask testing with UI coordinates")

    # Convert UI to sample indices
    samples_per_ui = bit_period * sample_rate
    time_samples = time_ui * samples_per_ui

    # For simplicity, test over one or two bit periods
    # Align signal to start of bit period
    n_ui = int(np.max(time_ui))  # 1 or 2 UI mask

    violations: list[tuple[float, float]] = []
    hit_count = 0

    # Test all complete bit periods in the signal
    n_periods = n_samples // int(samples_per_ui * n_ui)

    for period_idx in range(n_periods):
        period_start_sample = int(period_idx * samples_per_ui * n_ui)

        # Extract samples for this period
        for i, _t_ui in enumerate(time_ui):
            sample_idx = period_start_sample + int(time_samples[i])

            if sample_idx >= n_samples:
                break

            voltage = data[sample_idx]

            # Check if voltage violates mask
            if voltage > v_top[i] or voltage < v_bottom[i]:
                timestamp = sample_idx / sample_rate
                violations.append((timestamp, voltage))
                hit_count += 1

    # Calculate margins (minimum distance to mask boundaries)
    margin_top = float(np.inf)
    margin_bottom = float(np.inf)

    for period_idx in range(n_periods):
        period_start_sample = int(period_idx * samples_per_ui * n_ui)

        for i, _t_ui in enumerate(time_ui):
            sample_idx = period_start_sample + int(time_samples[i])

            if sample_idx >= n_samples:
                break

            voltage = data[sample_idx]

            # Margin to top
            margin_top = min(margin_top, v_top[i] - voltage)

            # Margin to bottom
            margin_bottom = min(margin_bottom, voltage - v_bottom[i])

    # Pass if no hits
    pass_fail = hit_count == 0

    return MaskTestResult(
        pass_fail=pass_fail,
        hit_count=hit_count,
        total_samples=n_periods * len(time_ui),
        margin_top=margin_top if margin_top != np.inf else 0.0,
        margin_bottom=margin_bottom if margin_bottom != np.inf else 0.0,
        violations=violations,
    )


def _get_predefined_mask(mask_name: str) -> dict[str, NDArray[np.float64]]:
    """Get predefined mask template.

    Args:
        mask_name: Name of standard mask ("usb2", "pcie_gen3", etc.).

    Returns:
        Dictionary with time_ui, voltage_top, voltage_bottom arrays.

    Raises:
        ValueError: If mask name is not recognized.
    """
    if mask_name == "usb2":
        # USB 2.0 high-speed eye mask (simplified)
        # 2-UI mask, normalized to ±1V amplitude
        time_ui = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0])
        v_top = np.array([0.6, 0.6, 0.8, 0.9, 0.8, 0.6, 0.4, 0.6, 0.8, 0.9, 0.8, 0.6, 0.6])
        v_bottom = np.array(
            [
                -0.6,
                -0.6,
                -0.8,
                -0.9,
                -0.8,
                -0.6,
                -0.4,
                -0.6,
                -0.8,
                -0.9,
                -0.8,
                -0.6,
                -0.6,
            ]
        )

    elif mask_name == "pcie_gen3":
        # PCIe Gen 3 eye mask (simplified)
        time_ui = np.array([0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0])
        v_top = np.array([0.5, 0.5, 0.7, 0.8, 0.7, 0.5, 0.5])
        v_bottom = np.array([-0.5, -0.5, -0.7, -0.8, -0.7, -0.5, -0.5])

    else:
        raise ValueError(
            f"Unknown mask: {mask_name}. Available: usb2, pcie_gen3. Or provide custom mask dict."
        )

    return {"time_ui": time_ui, "voltage_top": v_top, "voltage_bottom": v_bottom}


def pll_clock_recovery(
    trace: WaveformTrace | DigitalTrace,
    *,
    nominal_frequency: float,
    loop_bandwidth: float = 1e6,
    damping: float = 0.707,
    vco_gain: float = 1e6,
) -> PLLRecoveryResult:
    """Recover clock using PLL emulation.

    Emulates a second-order PLL to recover embedded clock from NRZ,
    NRZI, or Manchester-encoded data streams.

    Args:
        trace: Input data trace.
        nominal_frequency: Nominal clock frequency in Hz.
        loop_bandwidth: PLL loop bandwidth in Hz (default 1 MHz).
        damping: Damping factor (default 0.707 for critical damping).
        vco_gain: VCO gain in Hz/V (default 1 MHz/V).

    Returns:
        PLLRecoveryResult with recovered clock parameters.

    Raises:
        InsufficientDataError: If trace has fewer than 100 samples.

    Example:
        >>> result = pll_clock_recovery(data_trace, nominal_frequency=1e9)
        >>> print(f"Recovered: {result.recovered_frequency / 1e9:.3f} GHz")
        >>> print(f"Locked: {result.lock_status}")

    References:
        Gardner, F. M. (2005). Phaselock Techniques, 3rd ed.
    """
    data = trace.data.astype(np.float64) if isinstance(trace, DigitalTrace) else trace.data

    sample_rate = trace.metadata.sample_rate
    n_samples = len(data)

    if n_samples < 100:
        raise InsufficientDataError(
            "PLL recovery requires at least 100 samples",
            required=100,
            available=n_samples,
            analysis_type="pll_clock_recovery",
        )

    dt = 1.0 / sample_rate

    # PLL parameters
    omega_n = 2 * np.pi * loop_bandwidth  # Natural frequency
    K_vco = 2 * np.pi * vco_gain  # VCO gain in rad/s/V

    # Loop filter coefficients (2nd order)
    # Transfer function: F(s) = K1 + K2/s
    K1 = (2 * damping * omega_n) / K_vco
    K2 = (omega_n**2) / K_vco

    # Initialize PLL state
    phase = np.zeros(n_samples)
    vco_control = np.zeros(n_samples)
    integrator = 0.0
    theta = 0.0

    # Nominal phase increment per sample
    nominal_phase_inc = 2 * np.pi * nominal_frequency * dt

    # Find edges for phase detection (simplified)
    threshold = (np.max(data) + np.min(data)) / 2
    edges = np.where(np.abs(np.diff(np.sign(data - threshold))) > 0)[0]

    edge_idx = 0

    # Run PLL loop
    for i in range(n_samples):
        # Phase detector: compare VCO phase to input transitions
        if edge_idx < len(edges) and i == edges[edge_idx]:
            # Edge detected - compute phase error
            # Phase error = expected phase - actual VCO phase
            expected_phase = (edges[edge_idx] * nominal_phase_inc) % (2 * np.pi)
            phase_error = expected_phase - (theta % (2 * np.pi))

            # Wrap to [-pi, pi]
            phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi

            edge_idx += 1
        else:
            phase_error = 0.0

        # Loop filter (proportional + integral)
        integrator += K2 * phase_error * dt
        vco_input = K1 * phase_error + integrator

        # VCO: frequency = nominal + K_vco * control voltage
        vco_freq = nominal_frequency + K_vco * vco_input / (2 * np.pi)
        phase_increment = 2 * np.pi * vco_freq * dt

        # Update phase
        theta += phase_increment

        # Store results
        phase[i] = theta
        vco_control[i] = vco_input

    # Analyze lock status
    # Consider locked if VCO control voltage is stable in last 20%
    lock_threshold = 0.1  # 10% variation
    last_20_percent = vco_control[int(0.8 * n_samples) :]

    if len(last_20_percent) > 0:
        vco_std = np.std(last_20_percent)
        vco_mean = np.abs(np.mean(last_20_percent))
        lock_status = vco_std < lock_threshold * max(vco_mean, 1.0)

        # Find lock time (when variation drops below threshold)
        if lock_status:
            # Search for first point where subsequent variance is low
            window = int(0.1 * n_samples)  # 10% window
            for i in range(window, n_samples - window):
                window_std = np.std(vco_control[i : i + window])
                if window_std < lock_threshold:
                    lock_time = i * dt
                    break
            else:
                lock_time = None
        else:
            lock_time = None
    else:
        lock_status = False
        lock_time = None

    # Recovered frequency from final VCO state
    final_vco = np.mean(vco_control[-int(0.1 * n_samples) :])
    recovered_frequency = nominal_frequency + K_vco * final_vco / (2 * np.pi)

    frequency_error = recovered_frequency - nominal_frequency

    return PLLRecoveryResult(
        recovered_frequency=float(recovered_frequency),
        recovered_phase=phase,
        vco_control=vco_control,
        lock_status=lock_status,
        lock_time=lock_time,
        frequency_error=float(frequency_error),
    )


__all__ = [
    "Glitch",
    "MaskTestResult",
    "NoiseMarginResult",
    "PLLRecoveryResult",
    "Violation",
    "detect_glitches",
    "detect_violations",
    "mask_test",
    "noise_margin",
    "pll_clock_recovery",
    "signal_quality_summary",
]
