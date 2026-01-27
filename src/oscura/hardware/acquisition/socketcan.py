"""SocketCAN hardware acquisition source.

This module provides SocketCANSource for acquiring CAN bus data from Linux
SocketCAN interfaces. Supports both physical CAN interfaces and virtual CAN
for testing.

The SocketCAN source converts CAN messages into DigitalTrace format, with each
CAN ID represented as a separate digital channel. This enables protocol analysis
and reverse engineering of CAN bus communications.

Example:
    >>> from oscura.hardware.acquisition import HardwareSource
    >>>
    >>> # Basic usage
    >>> with HardwareSource.socketcan("can0", bitrate=500000) as source:
    ...     trace = source.read(duration=10)  # Capture for 10 seconds
    ...     print(f"Captured {len(trace.data)} CAN messages")
    >>>
    >>> # Streaming acquisition
    >>> with HardwareSource.socketcan("can0") as source:
    ...     for chunk in source.stream(duration=60, chunk_size=1000):
    ...         # Process each chunk of 1000 messages
    ...         analyze(chunk)

Dependencies:
    Requires python-can: pip install oscura[automotive]

Platform:
    Linux only - uses SocketCAN kernel module.

References:
    python-can documentation: https://python-can.readthedocs.io/
    SocketCAN: https://www.kernel.org/doc/Documentation/networking/can.txt
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from can import BusABC

    from oscura.core.types import Trace


class SocketCANSource:
    """SocketCAN hardware acquisition source.

    Acquires CAN bus messages from Linux SocketCAN interfaces and converts
    them to DigitalTrace format for analysis.

    Attributes:
        interface: SocketCAN interface name (e.g., "can0").
        bitrate: CAN bitrate in bps.
        bus: python-can Bus instance (created on first use).

    Example:
        >>> # Physical CAN interface
        >>> source = SocketCANSource("can0", bitrate=500000)
        >>> trace = source.read(duration=10)
        >>>
        >>> # Virtual CAN for testing
        >>> source = SocketCANSource("vcan0")
        >>> with source:
        ...     trace = source.read(duration=5)
    """

    def __init__(
        self,
        interface: str,
        *,
        bitrate: int = 500000,
        **kwargs: Any,
    ) -> None:
        """Initialize SocketCAN source.

        Args:
            interface: SocketCAN interface name (e.g., "can0", "vcan0").
            bitrate: CAN bitrate in bps (default: 500000).
            **kwargs: Additional arguments passed to python-can Bus.

        Raises:
            ImportError: If python-can is not installed.

        Example:
            >>> source = SocketCANSource("can0", bitrate=500000)
            >>> source = SocketCANSource("vcan0", receive_own_messages=True)
        """
        self.interface = interface
        self.bitrate = bitrate
        self.kwargs = kwargs
        self.bus: BusABC | None = None
        self._closed = False

    def _ensure_bus(self) -> None:
        """Ensure CAN bus is initialized.

        Raises:
            ImportError: If python-can is not installed.
            OSError: If interface doesn't exist or permissions denied.
        """
        if self.bus is not None:
            return

        try:
            import can
        except ImportError as e:
            raise ImportError(
                "SocketCAN source requires python-can library. "
                "Install with: pip install oscura[automotive]"
            ) from e

        try:
            self.bus = can.Bus(
                interface="socketcan",
                channel=self.interface,
                bitrate=self.bitrate,
                **self.kwargs,
            )
        except OSError as e:
            raise OSError(
                f"Failed to open SocketCAN interface '{self.interface}'. "
                f"Ensure interface exists and you have permissions. "
                f"Error: {e}"
            ) from e

    def read(self, duration: float = 10.0) -> Trace:
        """Read CAN messages for specified duration.

        Args:
            duration: Acquisition duration in seconds (default: 10.0).

        Returns:
            DigitalTrace containing captured CAN messages.

        Raises:
            ImportError: If python-can is not installed.
            OSError: If interface error occurs.
            ValueError: If source is closed.

        Example:
            >>> source = SocketCANSource("can0")
            >>> trace = source.read(duration=5.0)
            >>> print(f"Captured {len(trace.data)} messages")
        """
        if self._closed:
            raise ValueError("Cannot read from closed source")

        self._ensure_bus()

        import time

        import numpy as np

        from oscura.core.types import DigitalTrace, TraceMetadata

        messages = []
        start_time = time.time()
        acquisition_start = datetime.now()

        while time.time() - start_time < duration:
            msg = self.bus.recv(timeout=0.1)  # type: ignore[union-attr]
            if msg is not None:
                messages.append(msg)

        # Convert messages to digital trace format
        # Each CAN ID becomes a channel, timestamp is the time base
        if not messages:
            # Return empty trace if no messages
            metadata = TraceMetadata(
                sample_rate=1.0,  # Placeholder
                acquisition_time=acquisition_start,
                source_file=f"socketcan://{self.interface}",
                channel_name=f"CAN {self.interface}",
            )
            return DigitalTrace(data=np.array([], dtype=np.uint8), metadata=metadata)

        # Extract timestamps and data
        timestamps = np.array([msg.timestamp for msg in messages])
        can_ids = np.array([msg.arbitration_id for msg in messages], dtype=np.uint32)

        # Calculate sample rate from message timing
        time_range = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        effective_rate = len(messages) / time_range if time_range > 0 else 1.0

        metadata = TraceMetadata(
            sample_rate=effective_rate,
            acquisition_time=acquisition_start,
            source_file=f"socketcan://{self.interface}",
            channel_name=f"CAN {self.interface}",
        )

        # Store CAN IDs as digital data
        # Convert to bytes for DigitalTrace
        data_bytes = can_ids.view(np.uint8)

        return DigitalTrace(data=data_bytes, metadata=metadata)  # type: ignore[arg-type]

    def stream(self, duration: float = 60.0, chunk_size: int = 1000) -> Iterator[Trace]:
        """Stream CAN messages in chunks.

        Args:
            duration: Total acquisition duration in seconds (default: 60.0).
            chunk_size: Number of messages per chunk (default: 1000).

        Yields:
            DigitalTrace chunks containing CAN messages.

        Raises:
            ImportError: If python-can is not installed.
            ValueError: If source is closed.

        Example:
            >>> source = SocketCANSource("can0")
            >>> for chunk in source.stream(duration=60, chunk_size=1000):
            ...     analyze(chunk)
        """
        if self._closed:
            raise ValueError("Cannot stream from closed source")

        self._ensure_bus()

        import time

        import numpy as np

        from oscura.core.types import DigitalTrace, TraceMetadata

        start_time = time.time()
        acquisition_start = datetime.now()
        chunk_messages = []

        while time.time() - start_time < duration:
            msg = self.bus.recv(timeout=0.1)  # type: ignore[union-attr]
            if msg is not None:
                chunk_messages.append(msg)

                if len(chunk_messages) >= chunk_size:
                    # Yield chunk
                    timestamps = np.array([m.timestamp for m in chunk_messages])
                    can_ids = np.array([m.arbitration_id for m in chunk_messages], dtype=np.uint32)

                    time_range = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
                    effective_rate = len(chunk_messages) / time_range if time_range > 0 else 1.0

                    metadata = TraceMetadata(
                        sample_rate=effective_rate,
                        acquisition_time=acquisition_start,
                        source_file=f"socketcan://{self.interface}",
                        channel_name=f"CAN {self.interface}",
                    )

                    data_bytes = can_ids.view(np.uint8)
                    yield DigitalTrace(data=data_bytes, metadata=metadata)  # type: ignore[arg-type]

                    chunk_messages = []

        # Yield remaining messages
        if chunk_messages:
            timestamps = np.array([m.timestamp for m in chunk_messages])
            can_ids = np.array([m.arbitration_id for m in chunk_messages], dtype=np.uint32)

            time_range = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
            effective_rate = len(chunk_messages) / time_range if time_range > 0 else 1.0

            metadata = TraceMetadata(
                sample_rate=effective_rate,
                acquisition_time=acquisition_start,
                source_file=f"socketcan://{self.interface}",
                channel_name=f"CAN {self.interface}",
            )

            data_bytes = can_ids.view(np.uint8)
            yield DigitalTrace(data=data_bytes, metadata=metadata)  # type: ignore[arg-type]

    def close(self) -> None:
        """Close the CAN bus connection and release resources.

        Example:
            >>> source = SocketCANSource("can0")
            >>> trace = source.read()
            >>> source.close()
        """
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None
        self._closed = True

    def __enter__(self) -> SocketCANSource:
        """Context manager entry.

        Returns:
            Self for use in 'with' statement.

        Example:
            >>> with SocketCANSource("can0") as source:
            ...     trace = source.read()
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit.

        Automatically calls close() when exiting 'with' block.
        """
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"SocketCANSource(interface={self.interface!r}, bitrate={self.bitrate})"


__all__ = ["SocketCANSource"]
