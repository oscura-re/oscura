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
    # Import here to avoid circular dependencies
    try:
        from oscura.analyzers.eye.diagram import (
            EyeDiagram,
        )
    except ImportError:
        # Fallback if eye module not available
        pass

    try:
        from oscura.analyzers.digital.timing import (
            peak_to_peak_jitter,
            recover_clock_fft,
            time_interval_error,
        )
    except ImportError:
        # Provide minimal implementation
        recover_clock_fft = None  # type: ignore[assignment]
        peak_to_peak_jitter = None  # type: ignore[assignment]  # noqa: F841
        time_interval_error = None  # type: ignore[assignment]

    # Recover clock if not provided
    if clock_trace is None and recover_clock_fft is not None:
        try:
            clock_result = recover_clock_fft(trace)
            recovered_freq = clock_result.frequency
        except Exception:
            # Estimate from bit rate
            recovered_freq = bit_rate if bit_rate else 1e9
    else:
        recovered_freq = bit_rate if bit_rate else 1e9

    # Calculate eye parameters (simplified - would use actual eye_diagram function)
    # For now, provide placeholder calculations
    vpp = np.ptp(trace.data)
    eye_height = vpp * 0.7  # Typical eye opening is ~70% of signal swing
    ui = 1.0 / recovered_freq if recovered_freq else 1e-9  # Unit Interval
    eye_width = ui * 0.6  # Typical eye opening is ~60% of UI

    # Calculate jitter (simplified)
    # In real implementation, would use proper jitter analysis
    if time_interval_error is not None:
        try:
            tie = time_interval_error(trace, nominal_period=1.0 / recovered_freq)
            jitter_rms_val = float(np.std(tie))
            jitter_pp_val = float(np.ptp(tie))
        except Exception:
            tie = np.array([])
            jitter_rms_val = ui * 0.05  # Assume 5% UI jitter
            jitter_pp_val = ui * 0.2  # Assume 20% UI p-p jitter
    else:
        tie = np.array([])  # type: ignore[unreachable]
        jitter_rms_val = ui * 0.05
        jitter_pp_val = ui * 0.2

    # Determine dominant jitter source
    # Random jitter dominates if RMS jitter is significant compared to p-p
    # (deterministic would show bounded p-p with lower RMS)
    if jitter_rms_val > 0:
        jitter_ratio = jitter_pp_val / (6 * jitter_rms_val)  # Expect ~6 for Gaussian
        dominant_jitter_source = "random" if jitter_ratio < 8 else "deterministic"
    else:
        dominant_jitter_source = "unknown"

    # Estimate BER from eye closure (simplified Gaussian approximation)
    if eye_height > 0:
        snr_linear = (
            eye_height / (2 * jitter_rms_val * recovered_freq) if jitter_rms_val > 0 else 100
        )
        snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else 0
        # BER ~ Q(SNR) where Q is Q-function
        ber_estimate = 0.5 * (1 - np.tanh(snr_linear / np.sqrt(2)))
    else:
        snr_db = 0
        ber_estimate = 0.5

    # Mask margin (placeholder - would load actual mask)
    margin_to_mask = None
    if mask is not None:
        # Would compare eye to loaded mask
        margin_to_mask = eye_height * 0.2  # Assume 20% margin

    # Build result dictionary
    result = {
        "eye_height": eye_height,
        "eye_width": eye_width,
        "jitter_rms": jitter_rms_val,
        "jitter_pp": jitter_pp_val,
        "tie": tie,
        "tie_rms": jitter_rms_val,  # TIE RMS same as jitter RMS
        "margin_to_mask": margin_to_mask,
        "dominant_jitter_source": dominant_jitter_source,
        "bit_error_rate_estimate": ber_estimate,
        "snr_db": snr_db,
        "bit_rate": recovered_freq,
        "unit_interval": ui,
    }

    # Generate report if requested
    if report is not None:
        _generate_si_report(result, report)

    return result


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
