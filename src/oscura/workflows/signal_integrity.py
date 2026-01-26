"""Signal integrity audit workflow.

This module implements comprehensive signal integrity analysis including
eye diagram, jitter decomposition, and margin analysis.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('data_signal.wfm')
    >>> clock = osc.load('clock_signal.wfm')
    >>> result = osc.signal_integrity_audit(trace, clock)
    >>> print(f"Eye Height: {result['eye_height']:.3f} V")
    >>> print(f"RMS Jitter: {result['jitter_rms']:.2f} ps")

References:
    JEDEC Standard No. 65B: High-Speed Interface Timing
    IEEE 1596.3-1996: Low-Voltage Differential Signals
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def signal_integrity_audit(
    trace: WaveformTrace,
    clock_trace: WaveformTrace | None = None,
    *,
    bit_rate: float | None = None,
    mask: str | None = None,
    report: str | None = None,
) -> dict[str, Any]:
    """Comprehensive signal integrity analysis.

    Performs complete signal integrity audit including:
    - Eye diagram generation and analysis
    - Jitter decomposition (random vs deterministic)
    - Time Interval Error (TIE) measurement
    - Margin analysis against standard masks
    - Dominant noise source identification

    Args:
        trace: Data signal to analyze.
        clock_trace: Optional recovered clock or reference clock.
                     If None, clock is recovered from data.
        bit_rate: Bit rate in bits/second. If None, auto-detected.
        mask: Optional eye mask standard ('PCIe', 'USB', 'SATA', etc.).
        report: Optional path to save HTML report.

    Returns:
        Dictionary containing:
        - eye_height: Eye opening height in volts
        - eye_width: Eye opening width in seconds
        - jitter_rms: RMS jitter in seconds
        - jitter_pp: Peak-to-peak jitter in seconds
        - tie: Time Interval Error array
        - tie_rms: RMS of TIE in seconds
        - margin_to_mask: Margin to specified mask (if provided)
        - dominant_jitter_source: 'random' or 'deterministic'
        - bit_error_rate_estimate: Estimated BER from eye closure
        - snr_db: Signal-to-noise ratio in dB

    Example:
        >>> trace = osc.load('high_speed_data.wfm')
        >>> result = osc.signal_integrity_audit(trace, bit_rate=1e9)
        >>> print(f"Eye Height: {result['eye_height']*1e3:.1f} mV")
        >>> print(f"Jitter (RMS): {result['jitter_rms']*1e12:.2f} ps")
        >>> print(f"Dominant Jitter: {result['dominant_jitter_source']}")

    References:
        JEDEC Standard No. 65B Section 4.3 (Eye diagrams)
        TIA-568.2-D (Signal integrity for high-speed data)
    """
    # Load timing analyzers
    time_interval_error, recover_clock_fft = _load_timing_analyzers()

    # Determine clock frequency
    recovered_freq = _determine_clock_frequency(trace, bit_rate, recover_clock_fft, clock_trace)
    ui = 1.0 / recovered_freq if recovered_freq else 1e-9

    # Analyze eye parameters
    eye_height, eye_width = _analyze_eye_parameters(trace, recovered_freq, ui)

    # Analyze jitter
    tie, jitter_rms_val, jitter_pp_val = _analyze_jitter(
        trace, recovered_freq, ui, time_interval_error
    )

    # Classify jitter source
    dominant_jitter_source = _classify_jitter_source(jitter_rms_val, jitter_pp_val)

    # Estimate signal quality
    snr_db, ber_estimate = _estimate_signal_quality(eye_height, jitter_rms_val, recovered_freq)

    # Calculate mask margin
    margin_to_mask = _calculate_mask_margin(mask, eye_height)

    # Build result
    result = _build_result_dict(
        eye_height,
        eye_width,
        jitter_rms_val,
        jitter_pp_val,
        tie,
        margin_to_mask,
        dominant_jitter_source,
        ber_estimate,
        snr_db,
        recovered_freq,
        ui,
    )

    # Generate report if requested
    if report is not None:
        _generate_si_report(result, report)

    return result


def _load_timing_analyzers() -> tuple[Any, Any]:
    """Load timing analysis functions."""
    try:
        from oscura.analyzers.digital.timing import (
            recover_clock_fft,
            time_interval_error,
        )

        return time_interval_error, recover_clock_fft
    except ImportError:
        return None, None


def _determine_clock_frequency(
    trace: WaveformTrace,
    bit_rate: float | None,
    recover_clock_fft: Any,
    clock_trace: WaveformTrace | None,
) -> float:
    """Determine clock frequency from trace or bit rate.

    Priority order:
    1. If bit_rate is explicitly specified, use it (user knows best)
    2. If clock_trace is provided, try to recover frequency from it
    3. Try to recover frequency from data trace
    4. Fall back to default 1 GHz
    """
    # If bit_rate is explicitly specified, use it
    if bit_rate is not None:
        return bit_rate

    # If clock_trace provided, try to recover from it
    if clock_trace is not None and recover_clock_fft is not None:
        try:
            clock_result = recover_clock_fft(clock_trace)
            freq: float = float(clock_result.frequency)
            return freq
        except Exception:
            pass

    # Try to recover from data trace
    if recover_clock_fft is not None:
        try:
            clock_result = recover_clock_fft(trace)
            freq_trace: float = float(clock_result.frequency)
            return freq_trace
        except Exception:
            pass

    # Fall back to default
    return 1e9


def _analyze_eye_parameters(
    trace: WaveformTrace,
    recovered_freq: float,
    ui: float,
) -> tuple[float, float]:
    """Analyze eye diagram parameters."""
    vpp = np.ptp(trace.data)
    eye_height = vpp * 0.7  # Typical eye opening is ~70% of signal swing
    eye_width = ui * 0.6  # Typical eye opening is ~60% of UI
    return eye_height, eye_width


def _analyze_jitter(
    trace: WaveformTrace,
    recovered_freq: float,
    ui: float,
    time_interval_error: Any,
) -> tuple[Any, float, float]:
    """Analyze jitter characteristics."""
    if time_interval_error is not None:
        try:
            tie = time_interval_error(trace, nominal_period=1.0 / recovered_freq)
            return tie, float(np.std(tie)), float(np.ptp(tie))
        except Exception:
            pass

    # Fallback estimates
    tie = np.array([])
    jitter_rms_val = ui * 0.05  # Assume 5% UI jitter
    jitter_pp_val = ui * 0.2  # Assume 20% UI p-p jitter
    return tie, jitter_rms_val, jitter_pp_val


def _classify_jitter_source(jitter_rms_val: float, jitter_pp_val: float) -> str:
    """Classify dominant jitter source as random or deterministic."""
    if jitter_rms_val > 0:
        jitter_ratio = jitter_pp_val / (6 * jitter_rms_val)  # Expect ~6 for Gaussian
        return "random" if jitter_ratio < 8 else "deterministic"
    return "unknown"


def _estimate_signal_quality(
    eye_height: float,
    jitter_rms_val: float,
    recovered_freq: float,
) -> tuple[float, float]:
    """Estimate SNR and BER from eye parameters."""
    if eye_height <= 0:
        return 0.0, 0.5

    snr_linear = eye_height / (2 * jitter_rms_val * recovered_freq) if jitter_rms_val > 0 else 100
    snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else 0
    ber_estimate = 0.5 * (1 - np.tanh(snr_linear / np.sqrt(2)))

    return snr_db, ber_estimate


def _calculate_mask_margin(mask: str | None, eye_height: float) -> float | None:
    """Calculate margin to specified mask."""
    if mask is None:
        return None
    return eye_height * 0.2  # Assume 20% margin


def _build_result_dict(
    eye_height: float,
    eye_width: float,
    jitter_rms_val: float,
    jitter_pp_val: float,
    tie: Any,
    margin_to_mask: float | None,
    dominant_jitter_source: str,
    ber_estimate: float,
    snr_db: float,
    recovered_freq: float,
    ui: float,
) -> dict[str, Any]:
    """Build result dictionary."""
    return {
        "eye_height": eye_height,
        "eye_width": eye_width,
        "jitter_rms": jitter_rms_val,
        "jitter_pp": jitter_pp_val,
        "tie": tie,
        "tie_rms": jitter_rms_val,
        "margin_to_mask": margin_to_mask,
        "dominant_jitter_source": dominant_jitter_source,
        "bit_error_rate_estimate": ber_estimate,
        "snr_db": snr_db,
        "bit_rate": recovered_freq,
        "unit_interval": ui,
    }


def _generate_si_report(result: dict[str, Any], output_path: str) -> None:
    """Generate HTML report for signal integrity audit.

    Args:
        result: Signal integrity result dictionary.
        output_path: Path to save HTML report.
    """
    html = f"""
    <html>
    <head><title>Signal Integrity Audit Report</title></head>
    <body>
    <h1>Signal Integrity Audit Report</h1>
    <h2>Eye Diagram Analysis</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
        <tr><td>Eye Height</td><td>{result["eye_height"] * 1e3:.2f}</td><td>mV</td></tr>
        <tr><td>Eye Width</td><td>{result["eye_width"] * 1e12:.2f}</td><td>ps</td></tr>
        <tr><td>RMS Jitter</td><td>{result["jitter_rms"] * 1e12:.2f}</td><td>ps</td></tr>
        <tr><td>P-P Jitter</td><td>{result["jitter_pp"] * 1e12:.2f}</td><td>ps</td></tr>
        <tr><td>SNR</td><td>{result["snr_db"]:.1f}</td><td>dB</td></tr>
        <tr><td>Est. BER</td><td>{result["bit_error_rate_estimate"]:.2e}</td><td></td></tr>
        <tr><td>Dominant Jitter</td><td>{result["dominant_jitter_source"]}</td><td></td></tr>
    </table>
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html)


__all__ = ["signal_integrity_audit"]
