"""Protocol type auto-detection from signal characteristics.

This module analyzes edge timing, symbol rate, and idle levels to
automatically detect serial protocol types.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('serial_data.wfm')
    >>> result = osc.detect_protocol(trace)
    >>> print(f"Protocol: {result['protocol']}")
    >>> print(f"Confidence: {result['confidence']:.1%}")

References:
    UART: TIA-232-F
    I2C: NXP UM10204
    SPI: Motorola SPI Block Guide
    CAN: ISO 11898
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def detect_protocol(
    trace: WaveformTrace,
    *,
    min_confidence: float = 0.6,
    return_candidates: bool = False,
    parallel: bool = True,
) -> dict[str, Any]:
    """Auto-detect serial protocol type.

    Analyzes signal characteristics to identify protocol:
    - Edge timing and regularity
    - Symbol rate detection
    - Idle level analysis
    - Transition patterns

    Detects: UART, SPI, I2C, CAN, 1-Wire, Manchester

    Args:
        trace: Signal to analyze.
        min_confidence: Minimum confidence threshold (0-1).
        return_candidates: If True, return all candidate protocols.
        parallel: If True, run protocol scoring in parallel (default True).

    Returns:
        Dictionary containing:
        - protocol: Detected protocol name
        - confidence: Detection confidence (0-1)
        - config: Suggested decoder configuration dict
        - characteristics: Dict with detected signal characteristics
        - candidates: List of all candidates (if return_candidates=True)

    Raises:
        AnalysisError: If no protocol can be detected with sufficient confidence.

    Example:
        >>> trace = osc.load('unknown_serial.wfm')
        >>> result = osc.detect_protocol(trace, return_candidates=True)
        >>> print(f"Detected: {result['protocol']}")
        >>> print(f"Baud rate: {result['config'].get('baud_rate', 'N/A')}")
        >>> for candidate in result['candidates']:
        ...     print(f"  {candidate['protocol']}: {candidate['confidence']:.1%}")

    References:
        sigrok Protocol Decoder heuristics
        UART: Asynchronous, idle high, start bit low
        I2C: Clock + data, open-drain, pull-up
    """
    # Analyze signal characteristics
    characteristics = _analyze_signal_characteristics(trace)

    # Define protocol detectors
    protocol_detectors = _build_protocol_detectors(characteristics)

    # Score protocols
    candidates = _score_protocols(protocol_detectors, characteristics, parallel)

    # Validate and select primary
    primary = _validate_detection(candidates, min_confidence)

    # Build result
    return _build_detection_result(primary, characteristics, candidates, return_candidates)


def _build_protocol_detectors(
    characteristics: dict[str, Any],
) -> list[tuple[str, Callable[[dict[str, Any]], float], dict[str, Any]]]:
    """Build list of protocol detectors with configurations."""
    return [
        (
            "UART",
            _score_uart,
            {
                "baud_rate": characteristics.get("symbol_rate", 115200),
                "data_bits": 8,
                "parity": "none",
                "stop_bits": 1,
            },
        ),
        (
            "SPI",
            _score_spi,
            {
                "clock_polarity": 0,
                "clock_phase": 0,
                "bit_order": "MSB",
            },
        ),
        (
            "I2C",
            _score_i2c,
            {
                "clock_rate": characteristics.get("symbol_rate", 100000),
                "address_bits": 7,
            },
        ),
        (
            "CAN",
            _score_can,
            {
                "baud_rate": characteristics.get("symbol_rate", 500000),
                "sample_point": 0.75,
            },
        ),
    ]


def _score_protocols(
    protocol_detectors: list[tuple[str, Callable[[dict[str, Any]], float], dict[str, Any]]],
    characteristics: dict[str, Any],
    parallel: bool,
) -> list[dict[str, Any]]:
    """Score all protocol detectors."""
    if parallel:
        return _score_protocols_parallel(protocol_detectors, characteristics)
    return _score_protocols_sequential(protocol_detectors, characteristics)


def _score_protocols_parallel(
    protocol_detectors: list[tuple[str, Callable[[dict[str, Any]], float], dict[str, Any]]],
    characteristics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score protocols in parallel using ThreadPoolExecutor."""
    candidates = []

    with ThreadPoolExecutor(max_workers=len(protocol_detectors)) as executor:
        future_to_protocol = {
            executor.submit(scorer, characteristics): (name, config)
            for name, scorer, config in protocol_detectors
        }

        for future in as_completed(future_to_protocol):
            name, config = future_to_protocol[future]
            try:
                score = future.result()
                if score > 0:
                    candidates.append(
                        {
                            "protocol": name,
                            "confidence": score,
                            "config": config,
                        }
                    )
            except Exception:
                pass

    candidates.sort(key=lambda x: x["confidence"], reverse=True)  # type: ignore[arg-type, return-value]
    return candidates


def _score_protocols_sequential(
    protocol_detectors: list[tuple[str, Callable[[dict[str, Any]], float], dict[str, Any]]],
    characteristics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score protocols sequentially."""
    candidates = []

    for name, scorer, config in protocol_detectors:
        score = scorer(characteristics)
        if score > 0:
            candidates.append(
                {
                    "protocol": name,
                    "confidence": score,
                    "config": config,
                }
            )

    candidates.sort(key=lambda x: x["confidence"], reverse=True)  # type: ignore[arg-type, return-value]
    return candidates


def _validate_detection(
    candidates: list[dict[str, Any]],
    min_confidence: float,
) -> dict[str, Any]:
    """Validate that detection meets confidence threshold."""
    if not candidates:
        raise AnalysisError(
            "Could not detect protocol type. Signal may be analog or unsupported protocol."
        )

    primary = candidates[0]

    if float(primary["confidence"]) < min_confidence:
        raise AnalysisError(
            f"Protocol detection confidence too low: {primary['confidence']:.1%} "
            f"(minimum: {min_confidence:.1%}). Try specifying protocol manually."
        )

    return primary


def _build_detection_result(
    primary: dict[str, Any],
    characteristics: dict[str, Any],
    candidates: list[dict[str, Any]],
    return_candidates: bool,
) -> dict[str, Any]:
    """Build detection result dictionary."""
    result = {
        "protocol": primary["protocol"],
        "confidence": primary["confidence"],
        "config": primary["config"],
        "characteristics": characteristics,
    }

    if return_candidates:
        result["candidates"] = candidates

    return result


def _analyze_signal_characteristics(trace: WaveformTrace) -> dict[str, Any]:
    """Analyze signal to extract protocol-relevant characteristics.

    Args:
        trace: Signal to analyze.

    Returns:
        Dictionary with characteristics.
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Handle empty data
    if len(data) == 0:
        return {
            "regularity": 0,
            "symbol_rate": 0,
            "idle_level": "low",
            "duty_cycle": 0,
            "transition_density": 0,
            "edge_count": 0,
        }

    # Detect edges
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold
    edges = np.diff(digital.astype(int))
    edge_indices = np.where(edges != 0)[0]

    # Edge statistics
    if len(edge_indices) > 1:
        edge_times = edge_indices / sample_rate
        edge_intervals = np.diff(edge_times)

        # Detect if edges are regular (clock-like) or irregular
        if len(edge_intervals) > 10:
            interval_std = np.std(edge_intervals)
            interval_mean = np.mean(edge_intervals)
            regularity = 1.0 - min(1.0, interval_std / (interval_mean + 1e-12))
        else:
            regularity = 0.5

        # Estimate symbol rate from edge intervals
        # For clock-based: symbol_rate = 1 / (2 * edge_interval)
        # For async: symbol_rate = 1 / min_interval
        median_interval = np.median(edge_intervals)
        symbol_rate = 1.0 / median_interval if median_interval > 0 else 0
    elif len(edge_indices) == 1:
        # With exactly 1 edge, regularity is indeterminate
        regularity = 0.5
        symbol_rate = 0
    else:
        # No edges: completely DC signal, no regularity
        regularity = 0
        symbol_rate = 0

    # Idle level (high or low)
    idle_level = "high" if np.mean(data) > threshold else "low"

    # Duty cycle
    duty_cycle = np.sum(digital) / len(digital)

    # Transition density (edges per second)
    duration = len(data) / sample_rate
    transition_density = len(edge_indices) / duration if duration > 0 else 0

    return {
        "regularity": regularity,
        "symbol_rate": symbol_rate,
        "idle_level": idle_level,
        "duty_cycle": duty_cycle,
        "transition_density": transition_density,
        "edge_count": len(edge_indices),
    }


def _score_uart(characteristics: dict[str, Any]) -> float:
    """Score likelihood of UART protocol.

    UART characteristics:
    - Irregular edges (async)
    - Idle high
    - Low transition density

    Args:
        characteristics: Signal characteristics.

    Returns:
        Score from 0 to 1.
    """
    score = 0.0

    # UART is asynchronous - low regularity
    regularity = characteristics["regularity"]
    if regularity < 0.3:
        score += 0.4
    elif regularity < 0.5:
        score += 0.2

    # UART idles high
    if characteristics["idle_level"] == "high":
        score += 0.3

    # UART has moderate transition density
    density = characteristics["transition_density"]
    if 1000 < density < 1e6:  # Typical UART range
        score += 0.3

    # Cap at 0.99 to reflect inherent uncertainty
    return min(0.99, score)


def _score_spi(characteristics: dict[str, Any]) -> float:
    """Score likelihood of SPI protocol.

    SPI characteristics:
    - Regular clock edges
    - ~50% duty cycle
    - High transition density

    Args:
        characteristics: Signal characteristics.

    Returns:
        Score from 0 to 1.
    """
    score = 0.0

    # SPI has regular clock - high regularity
    regularity = characteristics["regularity"]
    if regularity > 0.8:
        score += 0.5
    elif regularity > 0.6:
        score += 0.3

    # SPI clock typically ~50% duty cycle
    duty_cycle = characteristics["duty_cycle"]
    duty_error = abs(duty_cycle - 0.5)
    if duty_error < 0.1:
        score += 0.3

    # SPI has high transition density
    density = characteristics["transition_density"]
    if density > 1e5:  # High speed
        score += 0.2

    # Cap at 0.99 to reflect inherent uncertainty
    return min(0.99, score)


def _score_i2c(characteristics: dict[str, Any]) -> float:
    """Score likelihood of I2C protocol.

    I2C characteristics:
    - Clock-like regularity
    - Idle high (pull-up)
    - Moderate transition density

    Args:
        characteristics: Signal characteristics.

    Returns:
        Score from 0 to 1.
    """
    score = 0.0

    # I2C clock has regularity
    regularity = characteristics["regularity"]
    if regularity > 0.6:
        score += 0.4

    # I2C idles high
    if characteristics["idle_level"] == "high":
        score += 0.3

    # I2C has lower transition density than SPI
    density = characteristics["transition_density"]
    if 1e3 < density < 1e6:
        score += 0.3

    # Cap at 0.99 to reflect inherent uncertainty
    return min(0.99, score)


def _score_can(characteristics: dict[str, Any]) -> float:
    """Score likelihood of CAN protocol.

    CAN characteristics:
    - Irregular edges (NRZ encoding with bit stuffing)
    - Idle high (recessive)
    - Moderate to high transition density

    Args:
        characteristics: Signal characteristics.

    Returns:
        Score from 0 to 1.
    """
    score = 0.0

    # CAN has some irregularity due to bit stuffing
    regularity = characteristics["regularity"]
    if 0.3 < regularity < 0.7:
        score += 0.4

    # CAN idles high (recessive state)
    if characteristics["idle_level"] == "high":
        score += 0.3

    # CAN has specific baud rates (typically 125k, 250k, 500k, 1M)
    symbol_rate = characteristics["symbol_rate"]
    common_rates = [125000, 250000, 500000, 1000000]
    for rate in common_rates:
        if abs(symbol_rate - rate) / rate < 0.1:  # Within 10%
            score += 0.3
            break

    # Cap at 0.99 to reflect inherent uncertainty
    return min(0.99, score)


__all__ = ["detect_protocol"]
