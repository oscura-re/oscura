"""Signal and protocol builders for Oscura.

This module provides fluent builders for generating test signals, protocol
transactions, and test scenarios. These builders enable composable signal
generation without manual numpy operations.

Example:
    >>> import oscura as osc
    >>> # Simple sine wave with noise
    >>> trace = (osc.SignalBuilder(sample_rate=1e6, duration=0.01)
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
    >>> builder = osc.SignalBuilder(sample_rate=10e6)
    >>> builder.add_spi(clock_freq=1e6, data_mosi=b"\\x9F\\x00\\x00")
    >>> channels = builder.build_channels()  # Returns dict[str, WaveformTrace]

API:
    - SignalBuilder.build() returns WaveformTrace for single-channel signals
    - SignalBuilder.build_channels() returns dict[str, WaveformTrace] for multi-channel

References:
    - Oscura Signal Generation Guide
    - Protocol Test Signal Specifications
"""

from oscura.utils.builders.signal_builder import SignalBuilder

__all__ = [
    "SignalBuilder",
]
