"""Centralized data generation framework for Oscura demos.

This module provides a unified API for generating test signals, protocol
waveforms, and demo data. It serves as the Single Source of Truth for
all demo data generation, eliminating code duplication across demos.

Usage:
    from demos.data_generation import SignalBuilder

    # Generate a UART signal
    signal = (SignalBuilder()
        .sample_rate(10e6)
        .duration(0.01)
        .add_uart(baud_rate=115200, data=b"Hello Oscura!")
        .add_noise(snr_db=40)
        .build())

    # Save to file
    signal.save_npz("uart_demo.npz")

Available Builders:
    - SignalBuilder: Main fluent builder for signal generation
    - ProtocolSignalGenerator: Protocol-specific signal generation

Available Signal Types:
    Analog:
        - Sine, square, triangle, sawtooth waves
        - Noise (white, pink, gaussian)
        - Modulated signals (AM, FM, FSK)
        - Chirp/sweep signals

    Digital/Protocol:
        - UART/RS-232 frames
        - SPI transactions
        - I2C transactions
        - CAN/CAN-FD frames
        - JTAG TAP sequences
        - SWD transactions
        - USB packets

    Impairments:
        - Gaussian noise at specified SNR
        - Jitter (random, deterministic)
        - Quantization effects
        - ISI (inter-symbol interference)

Output Formats:
    - NumPy NPZ (compressed)
    - Raw arrays for direct use
"""

from demos.data_generation.core.base import BaseSignalGenerator
from demos.data_generation.core.builder import SignalBuilder

__all__ = [
    "BaseSignalGenerator",
    "SignalBuilder",
]
