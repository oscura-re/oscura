"""PyVISA oscilloscope and instrument acquisition source.

This module provides VISASource for acquiring waveforms from oscilloscopes and
other SCPI-compatible instruments via PyVISA. Supports USB, GPIB, Ethernet, and
serial connections.

The VISA source communicates with instruments using SCPI commands and acquires
waveform data directly from the oscilloscope, returning WaveformTrace format
for analysis.

Example:
    >>> from oscura.hardware.acquisition import HardwareSource
    >>>
    >>> # USB oscilloscope
    >>> with HardwareSource.visa("USB0::0x0699::0x0401::INSTR") as scope:
    ...     scope.configure(channels=[1, 2], timebase=1e-6)
    ...     trace = scope.read()
    >>>
    >>> # Ethernet oscilloscope
    >>> with HardwareSource.visa("TCPIP::192.168.1.100::INSTR") as scope:
    ...     scope.configure(channels=[1], vertical_scale=0.5)
    ...     trace = scope.read()
    >>>
    >>> # Auto-detect
    >>> with HardwareSource.visa() as scope:
    ...     trace = scope.read()

Dependencies:
    Requires pyvisa and pyvisa-py: pip install pyvisa pyvisa-py

Platform:
    Cross-platform (Windows, macOS, Linux).

Supported Instruments:
    - Tektronix oscilloscopes (DPO, MSO, MDO series)
    - Keysight oscilloscopes (InfiniiVision, MSO-X series)
    - Rigol oscilloscopes (DS1000Z, DS4000 series)
    - Other SCPI-compatible instruments

References:
    PyVISA: https://pyvisa.readthedocs.io/
    SCPI Standard: IEEE 488.2
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

# Optional dependency - import at module level for mocking in tests
try:
    import pyvisa
except ImportError:
    pyvisa = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from pyvisa import ResourceManager

    from oscura.core.types import Trace


class VisaResource(Protocol):
    """Protocol for PyVISA Resource interface."""

    def query(self, message: str) -> str:
        """Query the instrument."""
        ...

    def write(self, message: str) -> int:
        """Write to the instrument."""
        ...

    def close(self) -> None:
        """Close the resource."""
        ...


class VISASource:
    """PyVISA instrument acquisition source.

    Acquires waveforms from oscilloscopes and other SCPI-compatible instruments
    via PyVISA. Supports multiple connection types (USB, GPIB, Ethernet, serial).

    Attributes:
        resource: VISA resource string.
        instrument: PyVISA instrument instance.
        channels: Configured channel list.
        timebase: Configured timebase in seconds/division.
        vertical_scale: Configured vertical scale in volts/division.

    Example:
        >>> scope = VISASource("USB0::0x0699::0x0401::INSTR")
        >>> scope.configure(channels=[1, 2], timebase=1e-6)
        >>> trace = scope.read()
    """

    def __init__(
        self,
        resource: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize VISA source.

        Args:
            resource: VISA resource string (optional, auto-detects if None).
                Examples: "USB0::0x0699::0x0401::INSTR", "TCPIP::192.168.1.100::INSTR"
            **kwargs: Additional PyVISA configuration options.

        Raises:
            ImportError: If pyvisa is not installed.

        Example:
            >>> scope = VISASource("USB0::0x0699::0x0401::INSTR")
            >>> scope = VISASource()  # Auto-detect
        """
        self.resource = resource
        self.kwargs = kwargs
        self.rm: ResourceManager | None = None
        self.instrument: VisaResource | None = None
        self._closed = False

        # Configuration
        self.channels: list[int] = [1]
        self.timebase: float = 1e-6
        self.vertical_scale: float = 1.0
        self.record_length: int = 10000

    def _ensure_connection(self) -> None:
        """Ensure connection to VISA instrument.

        Raises:
            ImportError: If pyvisa is not installed.
            RuntimeError: If no instrument found or connection fails.
        """
        if self.instrument is not None:
            return

        if pyvisa is None:
            raise ImportError(
                "VISA source requires pyvisa library. Install with: pip install pyvisa pyvisa-py"
            )

        try:
            self.rm = pyvisa.ResourceManager()

            # Auto-detect if resource not specified
            if self.resource is None:
                resources = self.rm.list_resources()
                if not resources:
                    raise RuntimeError("No VISA instruments found")
                self.resource = resources[0]

            # Cast: pyvisa's Resource implements our VisaResource Protocol
            self.instrument = cast(
                "VisaResource", self.rm.open_resource(self.resource, **self.kwargs)
            )

            # Query instrument identity
            try:
                idn = self.instrument.query("*IDN?")
                print(f"Connected to: {idn.strip()}")
            except Exception:
                # Some instruments may not support *IDN?
                pass

        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to VISA instrument '{self.resource}'. "
                f"Ensure instrument is connected and powered on. "
                f"Error: {e}"
            ) from e

    def configure(
        self,
        *,
        channels: list[int] | None = None,
        timebase: float | None = None,
        vertical_scale: float | None = None,
        record_length: int | None = None,
    ) -> None:
        """Configure oscilloscope acquisition parameters.

        Args:
            channels: List of channel numbers to acquire (e.g., [1, 2]).
            timebase: Timebase in seconds/division (e.g., 1e-6 for 1 µs/div).
            vertical_scale: Vertical scale in volts/division (e.g., 0.5).
            record_length: Number of samples to acquire (e.g., 10000).

        Example:
            >>> scope = VISASource()
            >>> scope.configure(
            ...     channels=[1, 2],
            ...     timebase=1e-6,
            ...     vertical_scale=0.5,
            ...     record_length=10000
            ... )
        """
        if channels is not None:
            self.channels = channels
        if timebase is not None:
            self.timebase = timebase
        if vertical_scale is not None:
            self.vertical_scale = vertical_scale
        if record_length is not None:
            self.record_length = record_length

        # Apply configuration to instrument
        self._ensure_connection()

        try:
            # Set horizontal (timebase)
            self.instrument.write(f":TIMebase:SCALe {self.timebase}")  # type: ignore[union-attr]

            # Set vertical scale for each channel
            for ch in self.channels:
                self.instrument.write(f":CHANnel{ch}:SCALe {self.vertical_scale}")  # type: ignore[union-attr]
                self.instrument.write(f":CHANnel{ch}:DISPlay ON")  # type: ignore[union-attr]

            # Set record length
            self.instrument.write(f":ACQuire:POINts {self.record_length}")  # type: ignore[union-attr]

        except Exception as e:
            # SCPI commands vary by manufacturer, log but continue
            print(f"Warning: Configuration command failed: {e}")

    def read(self, channel: int | None = None) -> Trace:
        """Read waveform from oscilloscope.

        Args:
            channel: Channel to read (uses first configured channel if None).

        Returns:
            WaveformTrace containing acquired waveform.

        Raises:
            ImportError: If pyvisa is not installed.
            RuntimeError: If acquisition fails.
            ValueError: If source is closed.

        Example:
            >>> scope = VISASource()
            >>> scope.configure(channels=[1, 2])
            >>> trace = scope.read(channel=1)
        """
        if self._closed:
            raise ValueError("Cannot read from closed source")

        self._ensure_connection()

        import numpy as np

        from oscura.core.types import CalibrationInfo, TraceMetadata, WaveformTrace

        if channel is None:
            channel = self.channels[0]

        acquisition_start = datetime.now()

        try:
            # Single acquisition
            self.instrument.write(":SINGle")  # type: ignore[union-attr]

            # Wait for acquisition to complete
            import time

            time.sleep(0.1)

            # Set data source
            self.instrument.write(f":DATa:SOURce CHANnel{channel}")  # type: ignore[union-attr]

            # Get waveform preamble (metadata)
            preamble = self.instrument.query(":WAVeform:PREamble?")  # type: ignore[union-attr]
            preamble_parts = preamble.split(",")

            # Parse preamble (format varies by manufacturer)
            try:
                x_increment = float(preamble_parts[4])  # Time between samples
                sample_rate = 1.0 / x_increment
            except (IndexError, ValueError):
                sample_rate = 1e9  # Default 1 GSa/s

            # Get waveform data
            self.instrument.write(":WAVeform:FORMat WORD")  # type: ignore[union-attr]
            raw_data = self.instrument.query_binary_values(  # type: ignore[union-attr]
                ":WAVeform:DATA?", datatype="h", is_big_endian=True
            )

            # Convert to voltage
            data = np.array(raw_data, dtype=np.float64)

            # Get instrument identity for calibration info
            try:
                idn = self.instrument.query("*IDN?").strip()  # type: ignore[union-attr]
            except Exception:
                idn = "Unknown Instrument"

            calibration_info = CalibrationInfo(
                instrument=idn,
                coupling="DC",  # Default
                vertical_resolution=8,  # Typical for oscilloscopes
            )

            metadata = TraceMetadata(
                sample_rate=sample_rate,
                vertical_scale=self.vertical_scale,
                acquisition_time=acquisition_start,
                source_file=f"visa://{self.resource}",
                channel_name=f"CH{channel}",
                calibration_info=calibration_info,
            )

            return WaveformTrace(data=data, metadata=metadata)

        except Exception as e:
            raise RuntimeError(
                f"Failed to acquire waveform from channel {channel}. Error: {e}"
            ) from e

    def stream(self, duration: float = 60.0, interval: float = 1.0) -> Iterator[Trace]:
        """Stream waveforms at regular intervals.

        Args:
            duration: Total acquisition duration in seconds (default: 60.0).
            interval: Time between acquisitions in seconds (default: 1.0).

        Yields:
            WaveformTrace for each acquisition.

        Raises:
            ValueError: If source is closed.

        Example:
            >>> scope = VISASource()
            >>> scope.configure(channels=[1])
            >>> for trace in scope.stream(duration=60, interval=1):
            ...     analyze(trace)

        Note:
            Acquires repeated single-shot waveforms, not continuous streaming.
        """
        if self._closed:
            raise ValueError("Cannot stream from closed source")

        start_time = time.time()

        while time.time() - start_time < duration:
            # Acquire waveform
            trace = self.read()
            yield trace

            # Wait for next acquisition
            time.sleep(interval)

    def close(self) -> None:
        """Close connection to VISA instrument.

        Example:
            >>> scope = VISASource()
            >>> scope.configure(channels=[1])
            >>> trace = scope.read()
            >>> scope.close()
        """
        if self.instrument is not None:
            self.instrument.close()
            self.instrument = None
        if self.rm is not None:
            self.rm.close()
            self.rm = None
        self._closed = True

    def __enter__(self) -> VISASource:
        """Context manager entry.

        Returns:
            Self for use in 'with' statement.

        Example:
            >>> with VISASource() as scope:
            ...     scope.configure(channels=[1])
            ...     trace = scope.read()
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit.

        Automatically calls close() when exiting 'with' block.
        """
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"VISASource(resource={self.resource!r})"


__all__ = ["VISASource"]
