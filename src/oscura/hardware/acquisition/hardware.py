"""Hardware-based signal acquisition sources.

This module provides the base HardwareSource class and factory methods for
creating hardware-based acquisition sources. All hardware sources implement
the unified Source protocol, making them interchangeable with FileSource and
SyntheticSource.

Supported Hardware:
    - SocketCAN: Linux CAN bus interface (requires python-can)
    - Saleae Logic: Logic analyzer (requires saleae library)
    - PyVISA: Oscilloscopes and instruments (requires pyvisa)

Example:
    >>> from oscura.hardware.acquisition import HardwareSource
    >>>
    >>> # SocketCAN source
    >>> with HardwareSource.socketcan("can0", bitrate=500000) as source:
    ...     trace = source.read()
    >>>
    >>> # Saleae Logic source
    >>> with HardwareSource.saleae() as source:
    ...     source.configure(sample_rate=1e6, duration=10)
    ...     trace = source.read()
    >>>
    >>> # PyVISA oscilloscope
    >>> with HardwareSource.visa("USB0::0x0699::0x0401::INSTR") as scope:
    ...     scope.configure(channels=[1, 2], timebase=1e-6)
    ...     trace = scope.read()

Pattern:
    Each hardware type has its own module (socketcan.py, saleae.py, visa.py)
    containing implementation classes. HardwareSource provides factory methods
    for convenient creation.

Dependencies:
    Hardware sources require optional dependencies. Install with:
    - SocketCAN: pip install oscura[automotive]  (includes python-can)
    - Saleae: pip install saleae
    - PyVISA: pip install pyvisa pyvisa-py

References:
    Architecture Plan Phase 2: Hardware Integration
    docs/architecture/api-patterns.md: Source Protocol
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.hardware.acquisition.saleae import SaleaeSource
    from oscura.hardware.acquisition.socketcan import SocketCANSource
    from oscura.hardware.acquisition.visa import VISASource


class HardwareSource:
    """Factory for creating hardware acquisition sources.

    This class provides static methods for creating hardware-based acquisition
    sources. Each method returns a specific hardware source implementation that
    follows the Source protocol.

    Methods:
        socketcan: Create Linux SocketCAN interface source
        saleae: Create Saleae Logic analyzer source
        visa: Create PyVISA instrument source

    Example:
        >>> # Create SocketCAN source
        >>> can = HardwareSource.socketcan("can0", bitrate=500000)
        >>> trace = can.read()
        >>>
        >>> # Create Saleae source
        >>> logic = HardwareSource.saleae()
        >>> logic.configure(sample_rate=1e6, duration=10)
        >>> trace = logic.read()
    """

    @staticmethod
    def socketcan(interface: str, *, bitrate: int = 500000, **kwargs: Any) -> SocketCANSource:
        """Create SocketCAN hardware source for Linux CAN interfaces.

        Args:
            interface: SocketCAN interface name (e.g., "can0", "vcan0").
            bitrate: CAN bitrate in bps (default: 500000).
            **kwargs: Additional arguments passed to python-can Bus.

        Returns:
            SocketCANSource instance ready for acquisition.

        Raises:
            ImportError: If python-can is not installed.
            OSError: If interface doesn't exist or permissions denied.

        Example:
            >>> # Physical CAN interface
            >>> can = HardwareSource.socketcan("can0", bitrate=500000)
            >>> trace = can.read()
            >>>
            >>> # Virtual CAN for testing
            >>> vcan = HardwareSource.socketcan("vcan0")
            >>> with vcan:
            ...     for chunk in vcan.stream(duration=60):
            ...         process(chunk)

        Note:
            Requires python-can library: pip install oscura[automotive]
            Linux only - uses SocketCAN kernel module.
        """
        from oscura.hardware.acquisition.socketcan import SocketCANSource

        return SocketCANSource(interface=interface, bitrate=bitrate, **kwargs)

    @staticmethod
    def saleae(device_id: str | None = None, **kwargs: Any) -> SaleaeSource:
        """Create Saleae Logic analyzer source.

        Args:
            device_id: Saleae device ID (optional, auto-detects if None).
            **kwargs: Additional configuration options.

        Returns:
            SaleaeSource instance ready for acquisition.

        Raises:
            ImportError: If saleae library is not installed.
            RuntimeError: If no Saleae device found.

        Example:
            >>> # Auto-detect device
            >>> logic = HardwareSource.saleae()
            >>> logic.configure(sample_rate=1e6, duration=10)
            >>> trace = logic.read()
            >>>
            >>> # Specify device
            >>> logic = HardwareSource.saleae(device_id="ABC123")
            >>> logic.configure(digital_channels=[0, 1, 2, 3])
            >>> with logic:
            ...     trace = logic.read()

        Note:
            Requires saleae library: pip install saleae
            Supports Logic 8, Logic Pro 8, Logic Pro 16.
        """
        from oscura.hardware.acquisition.saleae import SaleaeSource

        return SaleaeSource(device_id=device_id, **kwargs)

    @staticmethod
    def visa(resource: str | None = None, **kwargs: Any) -> VISASource:
        """Create PyVISA instrument source (oscilloscopes, etc.).

        Args:
            resource: VISA resource string (optional, auto-detects if None).
                Examples: "USB0::0x0699::0x0401::INSTR", "TCPIP::192.168.1.100::INSTR"
            **kwargs: Additional PyVISA configuration options.

        Returns:
            VISASource instance ready for acquisition.

        Raises:
            ImportError: If pyvisa is not installed.
            RuntimeError: If no VISA resource found.

        Example:
            >>> # Auto-detect instrument
            >>> scope = HardwareSource.visa()
            >>> scope.configure(channels=[1, 2], timebase=1e-6)
            >>> trace = scope.read()
            >>>
            >>> # Specific instrument
            >>> scope = HardwareSource.visa("USB0::0x0699::0x0401::INSTR")
            >>> scope.configure(channels=[1], vertical_scale=0.5)
            >>> with scope:
            ...     trace = scope.read()

        Note:
            Requires pyvisa and pyvisa-py: pip install pyvisa pyvisa-py
            Supports Tektronix, Keysight, Rigol, and other SCPI oscilloscopes.
        """
        from oscura.hardware.acquisition.visa import VISASource

        return VISASource(resource=resource, **kwargs)


__all__ = ["HardwareSource"]
