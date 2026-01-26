"""Testing utilities for Oscura.

This module provides synthetic test data generation with known ground truth
for validation and testing purposes.

Signal Generators
-----------------
These functions return WaveformTrace objects ready for use:

- generate_sine_wave: Pure sine wave
- generate_square_wave: Square wave with configurable duty cycle
- generate_dc: DC (constant) signal
- generate_multi_tone: Sum of multiple sine waves
- generate_pulse: Single pulse with configurable rise/fall times

Example:
    >>> from oscura.validation.testing import generate_sine_wave, generate_square_wave
    >>> sine = generate_sine_wave(frequency=1e6, amplitude=1.0)
    >>> square = generate_square_wave(frequency=500e3, duty_cycle=0.3)
"""

from oscura.validation.testing.synthetic import (
    GroundTruth,
    SyntheticDataGenerator,
    SyntheticMessageConfig,
    SyntheticPacketConfig,
    SyntheticSignalConfig,
    generate_dc,
    generate_digital_signal,
    generate_multi_tone,
    generate_packets,
    generate_protocol_messages,
    generate_pulse,
    generate_sine_wave,
    generate_square_wave,
    generate_test_dataset,
)

__all__ = [
    "GroundTruth",
    "SyntheticDataGenerator",
    "SyntheticMessageConfig",
    "SyntheticPacketConfig",
    "SyntheticSignalConfig",
    "generate_dc",
    "generate_digital_signal",
    "generate_multi_tone",
    "generate_packets",
    "generate_protocol_messages",
    "generate_pulse",
    "generate_sine_wave",
    "generate_square_wave",
    "generate_test_dataset",
]
