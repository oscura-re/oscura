"""Oscura analyzers module.

Provides signal analysis functionality including:
- Waveform measurements (timing, amplitude)
- Digital signal analysis (edge detection, thresholding, timing, quality)
- Spectral analysis (FFT, PSD, quality metrics)
- Statistical analysis (outliers, correlation, trends)
- Protocol decoding (UART, SPI, I2C, CAN)
- Jitter analysis (RJ, DJ, PJ, DDJ, bathtub curves)
- Eye diagram analysis (height, width, Q-factor)
- Signal integrity (S-parameters, equalization)
- Side-channel analysis (DPA, CPA, timing attacks)
- Machine learning classification (automatic protocol detection)
"""

# Import measurements module as namespace for DSL compatibility
from oscura.analyzers import (
    digital,
    eye,
    jitter,
    measurements,
    ml,
    protocols,
    side_channel,
    signal_integrity,
    statistics,
    validation,
    waveform,
)

__all__ = [
    "digital",
    "eye",
    "jitter",
    "measurements",
    "ml",
    "protocols",
    "side_channel",
    "signal_integrity",
    "statistics",
    "validation",
    "waveform",
]
