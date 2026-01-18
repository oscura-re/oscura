"""Signal and protocol builders for Oscura.

This module provides fluent builders for generating test signals, protocol
transactions, and test scenarios. These builders enable composable signal
generation without manual numpy operations.

Example:
    >>> import oscura as osc
    >>> # Simple sine wave with noise
    >>> signal = (osc.SignalBuilder(sample_rate=1e6, duration=0.01)
    ...     .add_sine(frequency=1000, amplitude=1.0)
    ...     .add_noise(snr_db=40)
    ...     .build())
    >>>
    >>> # UART signal for protocol testing
    >>> uart = (osc.SignalBuilder(sample_rate=10e6)
    ...     .add_uart(baud_rate=115200, data=b"Hello Oscura!")
    ...     .add_noise(snr_db=30)
    ...     .build())
    >>>
    >>> # Multi-channel SPI transaction
    >>> spi = (osc.SignalBuilder(sample_rate=10e6)
    ...     .add_spi(clock_freq=1e6, data_mosi=b"\\x9F\\x00\\x00")
    ...     .build())

References:
    - Oscura Signal Generation Guide
    - Protocol Test Signal Specifications
"""

from oscura.builders.signal_builder import (
    GeneratedSignal,
    SignalBuilder,
    SignalMetadata,
)

__all__ = [
    "GeneratedSignal",
    "SignalBuilder",
    "SignalMetadata",
]
