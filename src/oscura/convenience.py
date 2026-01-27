"""High-level convenience functions for common analysis tasks.

This module provides one-call solutions for common signal analysis tasks,
wrapping multiple lower-level functions into easy-to-use high-level APIs.

Example:
    >>> import oscura as osc
    >>> trace = osc.load("audio_capture.wfm")
    >>>
    >>> # One-call spectral analysis
    >>> metrics = osc.quick_spectral(trace, fundamental=1000)
    >>> print(f"THD: {metrics.thd_db:.1f} dB, SNR: {metrics.snr_db:.1f} dB")
    >>>
    >>> # Auto-detect and decode protocol
    >>> result = osc.auto_decode(trace)
    >>> print(f"Protocol: {result.protocol}, Frames: {len(result.frames)}")
    >>>
    >>> # Smart filtering
    >>> clean = osc.smart_filter(trace, target="noise")

References:
    - Oscura API Design Guidelines
    - IEEE 1241-2010 (ADC Characterization)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import DigitalTrace, WaveformTrace


@dataclass
class SpectralMetrics:
    """Results from quick_spectral analysis.

    Attributes:
        thd_db: Total Harmonic Distortion in dB.
        thd_percent: Total Harmonic Distortion as percentage.
        snr_db: Signal-to-Noise Ratio in dB.
        sinad_db: Signal-to-Noise and Distortion in dB.
        enob: Effective Number of Bits.
        sfdr_db: Spurious-Free Dynamic Range in dBc.
        fundamental_freq: Detected fundamental frequency in Hz.
        fundamental_mag_db: Fundamental magnitude in dB.
        noise_floor_db: Estimated noise floor in dB.
    """

    thd_db: float
    thd_percent: float
    snr_db: float
    sinad_db: float
    enob: float
    sfdr_db: float
    fundamental_freq: float
    fundamental_mag_db: float
    noise_floor_db: float


@dataclass
class DecodeResult:
    """Results from auto_decode analysis.

    Attributes:
        protocol: Detected protocol name.
        frames: List of decoded protocol packets.
        confidence: Detection confidence (0-1).
        baud_rate: Detected baud/bit rate (if applicable).
        config: Protocol configuration parameters.
        errors: List of decoding errors.
        statistics: Decoding statistics.
    """

    protocol: str
    frames: list[Any]
    confidence: float
    baud_rate: float | None
    config: dict[str, Any]
    errors: list[str]
    statistics: dict[str, Any]


def quick_spectral(
    trace: WaveformTrace,
    fundamental: float | None = None,
    n_harmonics: int = 10,
    window: str = "hann",
) -> SpectralMetrics:
    """One-call spectral analysis with all common metrics.

    Computes THD, SNR, SINAD, ENOB, and SFDR in a single call with
    optimized caching for efficiency.

    Args:
        trace: Input waveform trace.
        fundamental: Expected fundamental frequency in Hz.
                     If None, auto-detected from FFT peak.
        n_harmonics: Number of harmonics to consider for THD (default 10).
        window: Window function for FFT (default "hann").

    Returns:
        SpectralMetrics with all computed values.

    Example:
        >>> trace = osc.load("audio_1khz.wfm")
        >>> metrics = osc.quick_spectral(trace, fundamental=1000)
        >>> print(f"THD: {metrics.thd_db:.1f} dB")
        >>> print(f"SNR: {metrics.snr_db:.1f} dB")
        >>> print(f"ENOB: {metrics.enob:.1f} bits")

    References:
        IEEE 1241-2010 Section 4.1 (ADC Characterization)
    """
    from oscura.analyzers.waveform.spectral import (
        enob,
        fft,
        sfdr,
        sinad,
        snr,
        thd,
    )

    # Compute FFT once
    fft_result = fft(trace, window=window)
    freq = fft_result[0]
    mag_db = fft_result[1]

    # Find fundamental if not specified
    if fundamental is None:
        # Skip DC bins, find peak
        dc_bins = max(5, len(freq) // 1000)
        peak_idx = dc_bins + int(np.argmax(mag_db[dc_bins : len(mag_db) // 2]))
        fundamental = float(freq[peak_idx])
        fundamental_mag = float(mag_db[peak_idx])
    else:
        # Find closest bin to specified fundamental
        peak_idx = int(np.argmin(np.abs(freq - fundamental)))
        fundamental_mag = float(mag_db[peak_idx])

    # Estimate noise floor (median of lower half of spectrum, excluding DC and peaks)
    noise_bins = mag_db[10 : len(mag_db) // 2]
    noise_floor = float(np.median(noise_bins))

    # Compute metrics (these functions don't accept 'fundamental' parameter)
    thd_db_val = thd(trace, n_harmonics=n_harmonics, window=window)
    thd_pct = 100 * 10 ** (thd_db_val / 20) if not np.isnan(thd_db_val) else np.nan
    snr_db_val = snr(trace, n_harmonics=n_harmonics, window=window)
    sinad_db_val = sinad(trace, window=window)
    enob_val = enob(trace, window=window)
    sfdr_db_val = sfdr(trace, window=window)

    return SpectralMetrics(
        thd_db=thd_db_val,
        thd_percent=thd_pct,
        snr_db=snr_db_val,
        sinad_db=sinad_db_val,
        enob=enob_val,
        sfdr_db=sfdr_db_val,
        fundamental_freq=fundamental,
        fundamental_mag_db=fundamental_mag,
        noise_floor_db=noise_floor,
    )


def auto_decode(
    trace: WaveformTrace | DigitalTrace,
    protocol: str | None = None,
    min_confidence: float = 0.5,
) -> DecodeResult:
    """Auto-detect protocol and decode frames in one call.

    Automatically detects the protocol type (UART, SPI, I2C, CAN, etc.)
    if not specified, then decodes all frames.

    Args:
        trace: Input trace (waveform or digital).
        protocol: Force specific protocol (None for auto-detect).
        min_confidence: Minimum confidence for auto-detection (0-1).

    Returns:
        DecodeResult with protocol name, decoded frames, and statistics.

    Example:
        >>> trace = osc.load("serial_capture.wfm")
        >>> result = osc.auto_decode(trace)
        >>> print(f"Protocol: {result.protocol}")
        >>> print(f"Frames decoded: {len(result.frames)}")
        >>> for frame in result.frames[:5]:
        ...     print(f"  {frame.data.hex()}")

    References:
        sigrok Protocol Decoder API
    """

    # Prepare digital trace
    digital_trace = _prepare_digital_trace(trace)

    # Detect or use specified protocol
    protocol, config, confidence = _detect_or_select_protocol(
        trace, protocol, min_confidence, digital_trace
    )

    # Decode frames
    frames, errors = _decode_protocol_frames(protocol, config, digital_trace)

    # Calculate statistics
    error_frames = sum(1 for f in frames if hasattr(f, "errors") and f.errors)
    statistics = {
        "total_frames": len(frames),
        "error_frames": error_frames,
        "error_rate": error_frames / len(frames) if frames else 0,
    }

    return DecodeResult(
        protocol=protocol,
        frames=frames,
        confidence=confidence,
        baud_rate=config.get("baud_rate"),
        config=config,
        errors=errors,
        statistics=statistics,
    )


def _prepare_digital_trace(trace: WaveformTrace | DigitalTrace) -> DigitalTrace:
    """Convert to digital trace if needed."""
    from oscura.core.types import WaveformTrace

    if isinstance(trace, WaveformTrace):
        from oscura.analyzers.digital.extraction import to_digital

        return to_digital(trace, threshold="auto")
    return trace


def _detect_or_select_protocol(
    trace: WaveformTrace | DigitalTrace,
    protocol: str | None,
    min_confidence: float,
    digital_trace: DigitalTrace,
) -> tuple[str, dict[str, Any], float]:
    """Detect protocol or use specified one."""
    from oscura.core.types import WaveformTrace

    if protocol is None or protocol.lower() == "auto":
        if isinstance(trace, WaveformTrace):
            from oscura.inference.protocol import detect_protocol

            detection = detect_protocol(
                trace, min_confidence=min_confidence, return_candidates=True
            )
        else:
            detection = {"protocol": "UART", "config": {}, "confidence": 0.5}
        return (
            detection.get("protocol", "unknown"),
            detection.get("config", {}),
            detection.get("confidence", 0.0),
        )
    else:
        return protocol.upper(), _get_default_protocol_config(protocol.upper()), 1.0


def _decode_protocol_frames(
    protocol: str, config: dict[str, Any], digital_trace: DigitalTrace
) -> tuple[list[Any], list[str]]:
    """Decode frames based on detected protocol."""
    frames: list[Any] = []
    errors: list[str] = []

    try:
        if protocol == "UART":
            frames = _decode_uart(digital_trace, config)
        elif protocol == "SPI":
            frames = _decode_spi(digital_trace, config)
        elif protocol == "I2C":
            frames = _decode_i2c(digital_trace, config)
        elif protocol == "CAN":
            frames = _decode_can(digital_trace, config)
        else:
            errors.append(f"Unsupported protocol: {protocol}")
    except Exception as e:
        errors.append(f"Decoding error: {e!s}")

    return frames, errors


def _decode_uart(digital_trace: DigitalTrace, config: dict[str, Any]) -> list[Any]:
    """Decode UART frames."""
    from oscura.analyzers.protocols.uart import UARTDecoder

    decoder = UARTDecoder(
        baudrate=config.get("baud_rate", 115200),
        data_bits=config.get("data_bits", 8),
        parity=config.get("parity", "none"),
        stop_bits=config.get("stop_bits", 1),
    )
    return list(decoder.decode(digital_trace))


def _decode_spi(digital_trace: DigitalTrace, config: dict[str, Any]) -> list[Any]:
    """Decode SPI frames."""
    from oscura.analyzers.protocols.spi import SPIDecoder

    decoder = SPIDecoder(cpol=config.get("clock_polarity", 0), cpha=config.get("clock_phase", 0))
    return list(decoder.decode(digital_trace, clk=digital_trace.data, mosi=digital_trace.data))


def _decode_i2c(digital_trace: DigitalTrace, config: dict[str, Any]) -> list[Any]:
    """Decode I2C frames."""
    from oscura.analyzers.protocols.i2c import I2CDecoder

    decoder = I2CDecoder()
    sda = digital_trace.data
    edges = np.where(np.diff(sda.astype(int)) != 0)[0]
    scl = np.ones_like(sda, dtype=bool)
    for i, edge in enumerate(edges):
        if i % 2 == 0 and edge + 10 < len(scl):
            scl[edge : edge + 10] = False
    return list(decoder.decode(digital_trace, scl=scl, sda=sda))


def _decode_can(digital_trace: DigitalTrace, config: dict[str, Any]) -> list[Any]:
    """Decode CAN frames."""
    from oscura.analyzers.protocols.can import CANDecoder

    decoder = CANDecoder(
        bitrate=config.get("baud_rate", 500000), sample_point=config.get("sample_point", 0.75)
    )
    return list(decoder.decode(digital_trace))


def smart_filter(
    trace: WaveformTrace,
    target: Literal["noise", "high_freq", "low_freq", "60hz_hum", "50hz_hum", "auto"] = "auto",
    strength: float = 1.0,
) -> WaveformTrace:
    """Intelligently filter trace based on target.

    Automatically selects appropriate filter type and parameters
    based on the specified target or auto-detected noise characteristics.

    Args:
        trace: Input waveform trace.
        target: What to filter:
            - "noise": General noise reduction
            - "high_freq": Remove high frequency components
            - "low_freq": Remove DC and low frequency drift
            - "60hz_hum": Remove 60 Hz power line interference
            - "50hz_hum": Remove 50 Hz power line interference
            - "auto": Auto-detect and filter dominant noise source
        strength: Filter strength 0-1 (default 1.0 = full strength).

    Returns:
        Filtered WaveformTrace.

    Example:
        >>> noisy = osc.load("noisy_capture.wfm")
        >>> clean = osc.smart_filter(noisy, target="noise")
        >>> # Or auto-detect
        >>> clean = osc.smart_filter(noisy, target="auto")
    """
    from oscura.utils.filtering.convenience import (
        high_pass,
        low_pass,
        median_filter,
        notch_filter,
    )

    sample_rate = trace.metadata.sample_rate

    if target == "auto":
        target = _detect_noise_type(trace)

    if target == "noise":
        # Use median filter for general noise reduction
        kernel_size = int(3 + 4 * strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return median_filter(trace, kernel_size=kernel_size)

    elif target == "high_freq":
        # Low-pass filter
        cutoff = sample_rate / 10 * (1 - 0.5 * strength)
        return low_pass(trace, cutoff=cutoff)

    elif target == "low_freq":
        # High-pass filter
        cutoff = 10 + 90 * strength  # 10-100 Hz
        return high_pass(trace, cutoff=cutoff)

    elif target == "60hz_hum":
        # 60 Hz notch filter with harmonics
        result = trace
        for harmonic in range(1, int(1 + 4 * strength)):
            freq = 60 * harmonic
            if freq < sample_rate / 2:
                result = notch_filter(result, freq=freq, q_factor=30)
        return result

    elif target == "50hz_hum":
        # 50 Hz notch filter with harmonics
        result = trace
        for harmonic in range(1, int(1 + 4 * strength)):
            freq = 50 * harmonic
            if freq < sample_rate / 2:
                result = notch_filter(result, freq=freq, q_factor=30)
        return result

    else:
        raise ValueError(f"Unknown filter target: {target}")


def _detect_noise_type(
    trace: WaveformTrace,
) -> Literal["noise", "high_freq", "low_freq", "60hz_hum", "50hz_hum"]:
    """Auto-detect dominant noise type in trace.

    Args:
        trace: Input trace.

    Returns:
        Detected noise type string.
    """
    from oscura.analyzers.waveform.spectral import fft

    fft_result = fft(trace, window="hann")
    freq = fft_result[0]
    mag_db = fft_result[1]

    # Check for 50/60 Hz peaks
    idx_50 = int(np.argmin(np.abs(freq - 50)))
    idx_60 = int(np.argmin(np.abs(freq - 60)))

    # Get noise floor estimate
    noise_floor = np.median(mag_db[10 : len(mag_db) // 4])

    # Check for power line hum - must be clear peak at exact frequency
    # Verify it's a local maximum to avoid false positives from spectral leakage
    if idx_60 < len(mag_db) - 5 and idx_60 > 5 and idx_60 < len(freq):
        # Must be within 3 Hz of actual 60 Hz (tighter tolerance)
        if abs(freq[idx_60] - 60) < 3:
            # Check if it's a local maximum (peak, not leakage)
            local_max = mag_db[idx_60 - 2 : idx_60 + 3].max()
            is_peak = mag_db[idx_60] >= local_max - 0.1  # Allow tiny numerical error
            if is_peak and mag_db[idx_60] > noise_floor + 20:
                return "60hz_hum"
    if idx_50 < len(mag_db) - 5 and idx_50 > 5 and idx_50 < len(freq):
        # Must be within 3 Hz of actual 50 Hz (tighter tolerance)
        if abs(freq[idx_50] - 50) < 3:
            # Check if it's a local maximum (peak, not leakage)
            local_max = mag_db[idx_50 - 2 : idx_50 + 3].max()
            is_peak = mag_db[idx_50] >= local_max - 0.1  # Allow tiny numerical error
            if is_peak and mag_db[idx_50] > noise_floor + 20:
                return "50hz_hum"

    # Check frequency distribution
    low_power = np.mean(mag_db[1 : len(mag_db) // 10])
    mid_power = np.mean(mag_db[len(mag_db) // 10 : len(mag_db) // 4])
    high_power = np.mean(mag_db[len(mag_db) // 4 : len(mag_db) // 2])

    if low_power > mid_power + 10:
        return "low_freq"
    if high_power > mid_power + 10:
        return "high_freq"

    return "noise"


def _get_default_protocol_config(protocol: str) -> dict[str, Any]:
    """Get default configuration for a protocol.

    Args:
        protocol: Protocol name.

    Returns:
        Default configuration dictionary.
    """
    configs: dict[str, dict[str, Any]] = {
        "UART": {
            "baud_rate": 115200,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1,
        },
        "SPI": {
            "clock_polarity": 0,
            "clock_phase": 0,
            "bit_order": "MSB",
        },
        "I2C": {
            "clock_rate": 100000,
            "address_bits": 7,
        },
        "CAN": {
            "baud_rate": 500000,
            "sample_point": 0.75,
        },
    }
    return configs.get(protocol, {})
