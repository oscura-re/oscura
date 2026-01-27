"""EMC/EMI compliance testing workflow.

This module implements spectral compliance testing against regulatory limits.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('emissions.wfm')
    >>> result = osc.emc_compliance_test(trace, standard='FCC_Part15_ClassB')
    >>> print(f"Status: {result['status']}")
    >>> print(f"Violations: {len(result['violations'])}")

References:
    FCC Part 15: Radio Frequency Devices
    CISPR 22/32: Information Technology Equipment
    MIL-STD-461: EMI/EMC Requirements
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def emc_compliance_test(
    trace: WaveformTrace,
    *,
    standard: str = "FCC_Part15_ClassB",
    frequency_range: tuple[float, float] | None = None,
    detector: Literal["peak", "quasi-peak", "average"] = "peak",
    report: str | None = None,
) -> dict[str, Any]:
    """EMC/EMI compliance testing against regulatory limits.

    Performs spectral compliance testing:
    - Computes spectrum (FFT or welch)
    - Loads regulatory limit mask
    - Overlays limit lines on spectrum
    - Identifies violations
    - Generates compliance report

    Args:
        trace: Signal to test for emissions.
        standard: Regulatory standard to test against:
                  'FCC_Part15_ClassA', 'FCC_Part15_ClassB',
                  'CE_CISPR22_ClassA', 'CE_CISPR22_ClassB',
                  'CE_CISPR32_ClassA', 'CE_CISPR32_ClassB',
                  'MIL_STD_461G_CE102', 'MIL_STD_461G_RE102'
        frequency_range: Optional frequency range (f_min, f_max) in Hz.
        detector: Detector type ('peak', 'quasi-peak', 'average').
        report: Optional path to save HTML compliance report.

    Returns:
        Dictionary containing:
        - status: 'PASS' or 'FAIL'
        - standard: Standard tested against
        - violations: List of frequency violations
        - margin_to_limit: Minimum margin in dB (negative if failing)
        - worst_frequency: Frequency with worst margin
        - worst_margin: Worst margin value in dB
        - spectrum_freq: Frequency array for spectrum
        - spectrum_mag: Magnitude array for spectrum (dBµV or dBm)
        - limit_freq: Frequency array for limit mask
        - limit_mag: Magnitude array for limit mask

    Example:
        >>> trace = osc.load('radiated_emissions.wfm')
        >>> result = osc.emc_compliance_test(trace, standard='FCC_Part15_ClassB')
        >>> print(f"Compliance: {result['status']}")
        >>> print(f"Margin: {result['margin_to_limit']:.1f} dB")
        >>> if result['violations']:
        ...     print(f"Violations at: {[v['frequency']/1e6 for v in result['violations']]} MHz")

    References:
        FCC Part 15 Subpart B (Unintentional Radiators)
        CISPR 22/32 (Information Technology Equipment EMC)
        MIL-STD-461G (Military EMC Requirements)
    """
    from oscura.analyzers.waveform.spectral import fft

    # Calculate spectrum and convert to dBµV
    freq, mag_db = fft(trace)  # type: ignore[misc]
    spectrum_dbuv = mag_db + 120  # Convert dBV to dBµV

    # Load limit and apply frequency range
    limit_freq, limit_mag = _load_emc_mask(standard)
    freq, spectrum_dbuv = _apply_frequency_range(freq, spectrum_dbuv, frequency_range)

    # Check compliance
    limit_interp = np.interp(freq, limit_freq, limit_mag)
    margin = limit_interp - spectrum_dbuv
    violations = _build_violations_list(freq, spectrum_dbuv, limit_interp, margin)
    status = "FAIL" if violations else "PASS"

    # Margin analysis
    margin_to_limit = np.min(margin)
    worst_idx = np.argmin(margin)

    result = {
        "status": status,
        "standard": standard,
        "violations": violations,
        "margin_to_limit": margin_to_limit,
        "worst_frequency": freq[worst_idx],
        "worst_margin": margin[worst_idx],
        "spectrum_freq": freq,
        "spectrum_mag": spectrum_dbuv,
        "limit_freq": limit_freq,
        "limit_mag": limit_mag,
        "detector": detector,
    }

    if report is not None:
        _generate_compliance_report(result, report)

    return result


def _apply_frequency_range(
    freq: NDArray[Any],
    spectrum: NDArray[Any],
    frequency_range: tuple[float, float] | None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Apply frequency range filter if specified."""
    if frequency_range is not None:
        f_min, f_max = frequency_range
        mask = (freq >= f_min) & (freq <= f_max)
        return freq[mask], spectrum[mask]
    return freq, spectrum


def _build_violations_list(
    freq: NDArray[Any],
    spectrum_dbuv: NDArray[Any],
    limit_interp: NDArray[Any],
    margin: NDArray[Any],
) -> list[dict[str, float]]:
    """Build list of compliance violations."""
    violations = []
    violations_mask = margin < 0
    if np.any(violations_mask):
        for idx in np.where(violations_mask)[0]:
            violations.append(
                {
                    "frequency": freq[idx],
                    "measured_dbuv": spectrum_dbuv[idx],
                    "limit_dbuv": limit_interp[idx],
                    "excess_db": -margin[idx],
                }
            )
    return violations


def _load_emc_mask(
    standard: str,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Load EMC limit mask for a standard.

    Args:
        standard: Standard name.

    Returns:
        Tuple of (frequency array, limit array in dBµV).

    Raises:
        AnalysisError: If unknown EMC standard.
    """
    # Simplified mask data - real implementation would load from data files
    masks = {
        "FCC_Part15_ClassB": {
            # Frequencies in MHz, limits in dBµV at 3m
            "freq": np.array([0.15, 0.5, 5.0, 30.0, 88.0, 216.0, 1000.0]) * 1e6,
            "limit": np.array([60, 60, 56, 46, 46, 46, 46]),  # dBµV/m
        },
        "FCC_Part15_ClassA": {
            "freq": np.array([0.15, 0.5, 5.0, 30.0, 88.0, 216.0, 1000.0]) * 1e6,
            "limit": np.array([70, 70, 66, 56, 56, 56, 56]),
        },
        "CE_CISPR22_ClassB": {
            "freq": np.array([0.15, 0.5, 5.0, 30.0, 230.0, 1000.0]) * 1e6,
            "limit": np.array([66, 56, 56, 47, 47, 47]),
        },
        "CE_CISPR22_ClassA": {
            "freq": np.array([0.15, 0.5, 5.0, 30.0, 230.0, 1000.0]) * 1e6,
            "limit": np.array([79, 73, 73, 60, 60, 60]),
        },
        "CE_CISPR32_ClassB": {
            "freq": np.array([0.15, 0.5, 5.0, 30.0, 230.0, 1000.0]) * 1e6,
            "limit": np.array([66, 56, 56, 47, 47, 47]),
        },
        "CE_CISPR32_ClassA": {
            "freq": np.array([0.15, 0.5, 5.0, 30.0, 230.0, 1000.0]) * 1e6,
            "limit": np.array([79, 73, 73, 60, 60, 60]),
        },
        "MIL_STD_461G_CE102": {
            "freq": np.array([0.01, 0.15, 10.0, 50.0]) * 1e6,
            "limit": np.array([90, 80, 80, 80]),
        },
        "MIL_STD_461G_RE102": {
            "freq": np.array([2, 30, 200, 1000, 18000]) * 1e6,
            "limit": np.array([54, 54, 34, 34, 34]),
        },
    }

    if standard not in masks:
        raise AnalysisError(f"Unknown EMC standard: {standard}")

    mask_data = masks[standard]
    return mask_data["freq"], mask_data["limit"]


def _generate_compliance_report(result: dict[str, Any], output_path: str) -> None:
    """Generate HTML compliance report.

    Args:
        result: Compliance test result dictionary.
        output_path: Path to save HTML report.
    """
    status_color = "green" if result["status"] == "PASS" else "red"

    html = f"""
    <html>
    <head><title>EMC Compliance Report</title></head>
    <body>
    <h1>EMC Compliance Test Report</h1>
    <h2>Standard: {result["standard"]}</h2>
    <h2 style="color: {status_color}">Status: {result["status"]}</h2>

    <h3>Summary</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Margin to Limit</td><td>{result["margin_to_limit"]:.2f} dB</td></tr>
        <tr><td>Worst Frequency</td><td>{result["worst_frequency"] / 1e6:.2f} MHz</td></tr>
        <tr><td>Worst Margin</td><td>{result["worst_margin"]:.2f} dB</td></tr>
        <tr><td>Violations</td><td>{len(result["violations"])}</td></tr>
    </table>
    """
    if result["violations"]:
        html += """
        <h3>Violations</h3>
        <table>
            <tr><th>Frequency (MHz)</th><th>Measured (dBµV)</th><th>Limit (dBµV)</th><th>Excess (dB)</th></tr>
        """
        for v in result["violations"]:
            html += f"""
            <tr>
                <td>{v["frequency"] / 1e6:.2f}</td>
                <td>{v["measured_dbuv"]:.2f}</td>
                <td>{v["limit_dbuv"]:.2f}</td>
                <td>{v["excess_db"]:.2f}</td>
            </tr>
            """
        html += "</table>"

    html += """
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html)


__all__ = ["emc_compliance_test"]
